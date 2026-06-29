"""Local 1D advection-diffusion-reaction test of the deployed latent transport (no CFD).

The 0D rollout (aposteriori_rollout.py) showed the free-running latent transport is unstable. That
march omits what real CFD has: AXIAL DIFFUSION and cell-to-cell coupling. This solves the actual
deployment PDE for each latent component on the real reactor grid:

    u·dZ_k/dz − D·d²Z_k/dz² = ω_Z,k(Z, q)        (steady 1D, prescribed thermo)

via pseudo-transient continuation (explicit, CFL-limited), with the UDF manifold projection applied
each iteration. Inlet Z fixed = encode(inlet); outlet zero-gradient. We SWEEP the effective scalar
diffusivity D and ask: at what D does the steady solve stop diverging, and how does that D compare to
the PHYSICAL scalar diffusivity α = k/(ρ·cp) in the reactor? If a CFD-realistic D stabilizes it (and
recovers the true trajectory), the a-priori-excellent model deploys fine; if it needs D ≫ physical +
turbulent + grid-numerical diffusion, the latent transport is genuinely broken.

Run: .venv/bin/python scripts/pfr_1d_diffusion.py [runs/merged_best] [--n-cases 12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.training.datamodule import tripartite_case_split
from aposteriori_rollout import Surrogate, _relrmse


def solve_case(sg, z, u, T, P, Y_true, D, maxiter=8000, tol=1e-6, blow=1e3):
    """Pseudo-transient steady solve of u·Z' − D·Z'' = ω_Z on the case z-grid.
    Returns (Y_pred or None-if-blew, n_iter, converged_bool, max_envfrac)."""
    N = len(z)
    dz = np.diff(z)                                   # (N-1,)
    qn = sg.q(T, P)                                   # (N,4)
    Z0 = sg.encode(Y_true[0])[0]                      # (k,)
    Z = np.tile(Z0, (N, 1)).astype(float)            # init field = inlet everywhere
    span = np.maximum(sg.env_hi - sg.env_lo, 1e-12)
    umax = max(float(np.max(np.abs(u))), 1e-9); dzmin = max(float(np.min(dz)), 1e-12)
    dt = 0.4 * min(dzmin / umax, (dzmin ** 2 / (2 * D)) if D > 0 else np.inf)
    conv = False
    for it in range(maxiter):
        zp = sg.project(Z, qn)                        # manifold projection (UDF DEFINE_ADJUST)
        src = sg.omega_z(zp, qn)                      # (N,k) latent source
        Znew = Z.copy()
        adv = -u[1:, None] * (Z[1:] - Z[:-1]) / dz[:, None]          # upwind (u>0), i=1..N-1
        Znew[1:] = Z[1:] + dt * (adv + src[1:])
        if D > 0:
            dzc = 0.5 * (dz[:-1] + dz[1:])                            # (N-2,)
            diff = D * ((Z[2:] - Z[1:-1]) / dz[1:, None] - (Z[1:-1] - Z[:-2]) / dz[:-1, None]) / dzc[:, None]
            Znew[1:-1] += dt * diff
        Znew[0] = Z0
        if not np.all(np.isfinite(Znew)) or np.max(np.abs(Znew) / span) > blow:
            return None, it, False, float(np.max(np.abs(Znew) / span))
        res = np.linalg.norm(Znew - Z) / (np.linalg.norm(Z) + 1e-12)
        Z = Znew
        if res < tol:
            conv = True; break
    Y = sg.decode_mass(Z, qn)
    return Y, it, conv, float(np.max(np.abs(Z) / span))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default="runs/merged_best")
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--D", default="0,1e-4,1e-2,1,1e2",
                    help="comma-list of effective scalar diffusivities D [m^2/s] to sweep")
    args = ap.parse_args()
    Dvals = [float(x) for x in args.D.split(",")]

    df = load_database(args.db)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    sg = Surrogate(args.bundle)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    cid_col = sch.meta["CaseID"]
    in_cols = [f"Y_{s}" for s in sg.input]
    from scarfs.schema import MAJOR_SPECIES
    maj_idx = [sg.input.index(s) for s in MAJOR_SPECIES if s in sg.input]

    cids = dft[cid_col].drop_duplicates().to_numpy()
    pick = cids[:: max(1, len(cids) // args.n_cases)][: args.n_cases]

    cases, alphas, gridpe_phys = [], [], []
    for c in pick:
        sub = dft.loc[(dft[cid_col] == c).to_numpy()]
        if "z [m]" not in sub or "u [m/s]" not in sub:
            continue
        order = np.argsort(sub["z [m]"].to_numpy())
        z = sub["z [m]"].to_numpy()[order]; u = sub["u [m/s]"].to_numpy()[order]
        if len(z) < 5 or not np.all(np.diff(z) > 0):
            continue
        T = sub[tc].to_numpy()[order]; P = sub[pc].to_numpy()[order]
        Y_true = sub[in_cols].to_numpy(float)[order]
        # physical scalar diffusivity alpha = k/(rho*cp)  (Le~1 proxy)
        k_ = sub["k [W/m/K]"].to_numpy()[order]; rho = sub["rho [kg/m3]"].to_numpy()[order]
        cp = sub["cp_mass [J/kg/K]"].to_numpy()[order]
        alpha = float(np.median(k_ / (rho * cp)))
        cases.append((int(c), z, u, T, P, Y_true)); alphas.append(alpha)
        gridpe_phys.append(float(np.median(u)) * float(np.median(np.diff(z))) / max(alpha, 1e-30))

    alpha_med = float(np.median(alphas))
    print(f"=== 1D advection-diffusion-reaction stability sweep — {args.bundle} ===")
    print(f"cases: {len(cases)}   physical scalar diffusivity α=k/(ρcp): median={alpha_med:.2e} m²/s")
    print(f"storage-grid Péclet at physical α (u·Δz/α): median={np.median(gridpe_phys):.2e}  "
          f"(≫1 ⇒ advection-dominated ⇒ physical diffusion negligible at this grid)\n")
    print(f"  {'D [m²/s]':>10} {'D/α':>10} {'converged':>10} {'survived':>9} {'traj_relRMSE(med)':>18} {'env(med)':>10}")
    for D in Dvals:
        rls, envs, nconv, nsurv = [], [], 0, 0
        for (cid, z, u, T, P, Y_true) in cases:
            Ypred, it, conv, env = solve_case(sg, z, u, T, P, Y_true, D)
            nconv += int(conv); envs.append(env)
            if Ypred is not None:
                nsurv += 1
                rls.append(float(np.mean([_relrmse(Ypred[:, j], Y_true[:, j]) for j in maj_idx])))
        med_rl = float(np.median(rls)) if rls else float("nan")
        dα = D / alpha_med if alpha_med > 0 else float("inf")
        print(f"  {D:>10.1e} {dα:>10.1e} {nconv:>7}/{len(cases)} {nsurv:>7}/{len(cases)} "
              f"{med_rl:>18.4f} {float(np.median(envs)):>10.2e}")
    print("\n  Reading it: traj_relRMSE→O(0.1)+converged = diffusion stabilized the latent transport.")
    print("  Compare the stabilizing D to physical α (above) and to plausible turbulent/numerical")
    print("  diffusivity. D/α ~ O(1-100) achievable in CFD; D/α ~ 1e4+ means the loop is not")
    print("  realistically stabilizable by diffusion → the model/coupling needs the fix.")


if __name__ == "__main__":
    main()

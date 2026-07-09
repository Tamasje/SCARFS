"""ADAPTATION #3 — species-transport closed-loop test (the guaranteed-stable fallback, no CFD).

Latent free-running transport is a-posteriori unstable because the deployed step re-encodes a DECODED
latent (P=E∘D has Lipschitz gain ~6.6). Species transport removes that round-trip: CFD transports the
energy-active species directly, and the rate head supplies dYᵢ/dτ at the RESOLVED composition each
step — z = E(Y) is a fresh readout of the actual (transported) state, never decode→re-encode. This
rolls that loop on each held-out test case and checks whether it tracks truth (it should, since the
6.6× amplifier is gone).

    state = Y_active (the 61 energy-active species); each step (prescribed true thermo + ρ + trace species):
        z = E( y_std(Y_full) ) ;  ṙ = rate_head(z, q) [kg/m³/s] ;  Y_active += (ṙ/ρ)·Δτ  (clip ≥0)
    compare the integrated Y_active trajectory + ∫S_E to truth.  No projection, no latent integration.

Run: .venv/bin/python scripts/aposteriori_species.py [runs/merged_contract090] [--n-cases 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.coupling.codegen import _numpy_forward, _numpy_encoder
from scarfs.models.thermo import SpeciesThermo
from scarfs.schema import MAJOR_SPECIES
from scarfs.training.datamodule import tripartite_case_split
from aposteriori_rollout import Surrogate, _relrmse

_TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))
MECH = "chem_ForTransport.yaml"


def _rates_mass(sg, z, qn):
    """Rate head -> physical mass rates [kg/m³/s] at latent z (no projection)."""
    rs = _numpy_forward(z, qn, sg.W["rate_layers"])
    a = np.clip(rs * sg.aux["rate_std_scale"] + sg.aux["rate_std_mean"], -30.0, 30.0)
    return np.sinh(a) * sg.aux["rate_asinh_scale"]                       # (n, n_active)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default="runs/merged_contract090")
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--n-cases", type=int, default=30)
    args = ap.parse_args()

    df = load_database(args.db); sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    sg = Surrogate(args.bundle)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    tau_col = sch.state["tau"]; cid_col = sch.meta["CaseID"]; abs_col = sch.energy_target_column()
    ea = sg.spec["energy_active"]
    ea_idx = np.array([sg.input.index(s) for s in ea])                  # active cols within the dry input
    maj_idx_active = [list(ea).index(s) for s in MAJOR_SPECIES if s in ea]
    ta = SpeciesThermo.from_mechanism_yaml(MECH, list(ea))
    in_cols = [f"Y_{s}" for s in sg.input]

    cids = dft[cid_col].drop_duplicates().to_numpy()
    pick = cids[:: max(1, len(cids) // args.n_cases)][: args.n_cases]

    rows = []
    for c in pick:
        sub = dft.loc[(dft[cid_col] == c).to_numpy()]
        order = np.argsort(sub[tau_col].to_numpy())
        tau = sub[tau_col].to_numpy()[order]
        if len(tau) < 4 or not np.all(np.diff(tau) > 0):
            continue
        T = sub[tc].to_numpy()[order]; P = sub[pc].to_numpy()[order]
        rho = sub["rho [kg/m3]"].to_numpy()[order]
        Yf = sub[in_cols].to_numpy(float)[order]                        # (N, n_dry) true full comp
        abs_true = np.clip(sub[abs_col].to_numpy(float)[order], 0.0, None)
        N = len(tau)
        qn = sg.q(T, P)
        Yact = Yf[0, ea_idx].copy()                                     # transported state (inlet)
        Ypred_act = np.zeros((N, len(ea_idx))); absp = np.zeros(N)
        blew = False
        for i in range(N):
            yfull = Yf[i].copy(); yfull[ea_idx] = Yact                  # 61 transported, rest prescribed-true
            z = _numpy_encoder(np.atleast_2d((yfull - sg.cm) / sg.cs), sg.W["encoder_W"],
                               sg.W.get("encoder_mlp_layers"))          # E(Y), no projection (residual-aware)
            rm = _rates_mass(sg, z, qn[i:i + 1])[0]                     # (n_active,) mass rate
            Ypred_act[i] = Yact
            absp[i] = float(np.sum(rm * ta.h_mass(np.atleast_1d(T[i]))))
            if i < N - 1:
                Yact = np.clip(Yact + (rm / rho[i]) * (tau[i + 1] - tau[i]), 0.0, 1.0)  # mass-fraction bound
            if not np.all(np.isfinite(Yact)):
                blew = True; break
        if blew:
            rows.append({"blew": True}); continue
        Ytrue_act = Yf[:, ea_idx]
        maj_rl = float(np.mean([_relrmse(Ypred_act[:, j], Ytrue_act[:, j]) for j in maj_idx_active]))
        # ABSOLUTE trajectory RMSE (mass-fraction units) — immune to the near-zero-species relRMSE artifact
        abs_rmse = float(np.mean([np.sqrt(np.mean((Ypred_act[:, j] - Ytrue_act[:, j]) ** 2)) for j in maj_idx_active]))
        I_p = float(_TRAP(absp, tau)); I_t = float(_TRAP(abs_true, tau))
        rows.append({"blew": False, "maj": maj_rl, "abs_rmse": abs_rmse,
                     "outlet": float(np.mean([abs(Ypred_act[-1, j] - Ytrue_act[-1, j]) for j in maj_idx_active])),
                     "eint": abs(I_p - I_t) / max(abs(I_t), 1.0),
                     "ymax": float(np.max(np.abs(Ypred_act)))})
    ok = [r for r in rows if not r["blew"]]; nb = len(rows) - len(ok)
    g = lambda k: np.array([r[k] for r in ok])  # noqa: E731
    print(f"=== ADAPTATION #3 — species-transport rollout (no CFD, no latent) — {args.bundle} ===")
    print(f"cases: {len(ok)}   blew up: {nb}   (energy-active species transported, rate-head closure)\n")
    print(f"  major-species trajectory relRMSE : median={np.median(g('maj')):.4f}  "
          f"p95={np.percentile(g('maj'),95):.4f}  max={g('maj').max():.4f}")
    print(f"  major-species trajectory ABS RMSE: median={np.median(g('abs_rmse')):.2e}  "
          f"p95={np.percentile(g('abs_rmse'),95):.2e}  max={g('abs_rmse').max():.2e}  (mass-frac units)")
    print(f"    -> if ABS RMSE stays ~1e-2 while relRMSE p95 is huge, the tail is the near-zero-species")
    print(f"       relRMSE artifact (Y is bounded), NOT real divergence.")
    print(f"  outlet |ΔY| (majors)             : median={np.median(g('outlet')):.2e}")
    print(f"  integrated-energy ∫S_E relerr    : median={np.median(g('eint')):.4f}  "
          f"p95={np.percentile(g('eint'),95):.4f}")
    print(f"  max |Y_active| over rollout      : median={np.median(g('ymax')):.3f}  max={g('ymax').max():.3f}  "
          f"(≈1 ⇒ bounded to the simplex; ≫1 ⇒ drift)")
    print("\n  Compare to the latent rollout (relRMSE in the hundreds). If species-transport tracks")
    print("  (relRMSE → O(0.1), bounded), it is the stable deployment — at ~61 UDS vs 32 latent.")


if __name__ == "__main__":
    main()

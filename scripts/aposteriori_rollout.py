"""A-posteriori test WITHOUT CFD: closed-loop latent rollout of the deployed surrogate.

The a-priori metrics (scripts/full_test_eval.py, full_acceptance.py) score point-wise accuracy at
TRUE states. They cannot reveal whether the closed loop is STABLE — whether small per-step errors
accumulate/drift once the model is fed its own output. That is what CFD would exercise, and it is
testable here without Fluent by reproducing the deployed integration loop in numpy.

This marches the LATENT Z exactly as the generated UDF does — reusing the codegen numpy primitives
that pass the 1e-4 torch-consistency check — over each held-out TEST case's residence-time grid:

  Z_0 = encode(Y_inlet)
  each step i (prescribed true T_i, P_i):           # "prescribed-thermo" a-posteriori test:
      z_proj = E · decode(Z, q)                      #   isolates latent-transport stability from
      Y_i    = decode(z_proj)  (+ H2O closure)       #   the T-feedback loop (the standard first
      S_E,i  = Σ h_i(T)·ω̇_i   (rate path = deployed) #   a-posteriori check; coupled-T is a v2)
      Z      = clip_envelope( z_proj + ω_Z·Δτ )

then compares the ROLLED-OUT trajectory (major-species profiles, outlet composition, integrated
energy) against the ground-truth CRACKSIM case, and reports drift/stability. A "truth-integration"
control (integrate the STORED dY/dτ over the same grid) validates the τ-scheme independently.

Run: .venv/bin/python scripts/aposteriori_rollout.py [runs/merged_best] [--n-cases 40] [--db ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.coupling.codegen import (
    _build_thermo_features, _load_energy_thermo, _load_model_weights, _load_scalers,
    _load_spec, _numpy_encoder, _numpy_forward,
)
from scarfs.schema import MAJOR_SPECIES
from scarfs.training.datamodule import tripartite_case_split

_TRAP = getattr(np, "trapezoid", getattr(np, "trapz", None))


class Surrogate:
    """Deployed-forward primitives loaded from a trained bundle (UDF-faithful)."""

    def __init__(self, bundle: str):
        bundle = Path(bundle)
        spec = _load_spec(bundle)
        scalers = _load_scalers(bundle)
        self.spec = spec
        self.cm = np.asarray(spec["composition_mean"], float)
        self.cs = np.asarray(spec["composition_scale"], float)
        legacy = scalers.get("legacy_scalers")
        self.tsc = getattr(legacy, "thermo", None) if legacy is not None else scalers.get("thermo_scaler")
        self.W, _ = _load_model_weights(bundle, spec)
        self.aux = _load_energy_thermo(spec, scalers)
        st = spec["export_stats"]
        self.Tmin, self.Tmax = st["T_train_min"], st["T_train_max"]
        self.Pmin, self.Pmax = st["P_train_min"], st["P_train_max"]
        self.env_lo = np.asarray(st["latent_env_min"], float)
        self.env_hi = np.asarray(st["latent_env_max"], float)
        self.s_z = np.asarray(spec["arcsinh_latent_scale"], float)
        self.input = spec["input"]
        # stable latent ODE (direct transport): advance raw z by the field f = ω_Z − β·z, NO
        # re-projection; readout heads at raw (clamped) z. reproject = legacy E∘D path.
        self.direct = spec.get("transport_mode") == "direct"
        self.beta = float(spec.get("field_damping", 0.0))

    def field(self, z, qn):  # stable-ODE physical latent field f = sinh(ω_Z)·s_Z − β·z at raw z
        return self.omega_z(z, qn) - self.beta * z

    def q(self, T, P):
        Tn = np.clip(np.atleast_1d(T), self.Tmin, self.Tmax)
        Pn = np.clip(np.atleast_1d(P), self.Pmin, self.Pmax)
        qq = _build_thermo_features(Tn, Pn)
        if self.tsc is not None:
            qq = (qq - self.tsc.mean_) / self.tsc.scale_
        return qq

    def encode(self, Y):  # (n_dry,) -> (1,k) clipped to envelope
        ystd = (np.atleast_2d(Y) - self.cm) / self.cs
        return np.clip(_numpy_encoder(ystd, self.W["encoder_W"], self.W.get("encoder_mlp_layers")),
                       self.env_lo, self.env_hi)

    def project(self, z, qn):
        ydec = _numpy_forward(z, qn, self.W["decoder_layers"])
        return _numpy_encoder(ydec, self.W["encoder_W"], self.W.get("encoder_mlp_layers"))

    def decode_mass(self, z, qn):  # -> (1,n_dry) physical mass fractions, H2O-closed
        ydec = _numpy_forward(z, qn, self.W["decoder_layers"])
        y = np.maximum(ydec * self.cs + self.cm, 0.0)
        s = y.sum(axis=1, keepdims=True)
        y = np.where(s > 1.0, y / s, y)
        return y

    def omega_z(self, zproj, qn):
        o = _numpy_forward(zproj, qn, self.W["latent_source_layers"])
        return np.sinh(np.clip(o, -20.0, 20.0)) * self.s_z

    def absorption_rate(self, zproj, qn, T_true):  # deployed energy (rate path), +endothermic
        rs = _numpy_forward(zproj, qn, self.W["rate_layers"])
        a = np.clip(rs * self.aux["rate_std_scale"] + self.aux["rate_std_mean"], -30.0, 30.0)
        rates_mass = np.sinh(a) * self.aux["rate_asinh_scale"]
        h = self.aux["thermo"].h_mass(np.atleast_1d(T_true))
        return float(np.sum(rates_mass * h))


def _relrmse(pred, true):
    d = np.sqrt(np.mean(true ** 2))
    return float(np.sqrt(np.mean((pred - true) ** 2)) / d) if d > 0 else float("nan")


def rollout_case(sg: Surrogate, T, P, tau, Y_true, abs_true, substeps: int = 1):
    """Latent rollout under prescribed thermo. ``substeps`` subdivides each storage interval
    (re-projecting + re-querying ω_Z each substep, as a fine CFD timestep would).
    Returns (Y_pred, abs_pred, max_env_frac)."""
    n = len(tau)
    Y_pred = np.zeros_like(Y_true)
    abs_pred = np.zeros(n)
    z = sg.encode(Y_true[0])
    span = np.maximum(sg.env_hi - sg.env_lo, 1e-12)
    max_env = 0.0
    for i in range(n):
        qn = sg.q(T[i], P[i])
        zp = z if sg.direct else sg.project(z, qn)   # direct: heads read raw (clamped) z, no E∘D
        Y_pred[i] = sg.decode_mass(zp, qn)[0]
        abs_pred[i] = sg.absorption_rate(zp, qn, T[i])
        if i < n - 1:
            h = (tau[i + 1] - tau[i]) / substeps
            for ss in range(substeps):
                frac = (ss + 0.5) / substeps  # midpoint-interpolated thermo within the interval
                qm = sg.q(T[i] + (T[i + 1] - T[i]) * frac, P[i] + (P[i + 1] - P[i]) * frac)
                if sg.direct:
                    z = z + sg.field(z, qm) * h            # stable ODE: z += Δτ·(ω_Z − β·z), no reproject
                else:
                    zps = sg.project(z, qm)
                    z = zps + sg.omega_z(zps, qm) * h
                max_env = max(max_env, float(np.max(np.abs(np.clip(z, sg.env_lo, sg.env_hi) - z) / span)))
                z = np.clip(z, sg.env_lo, sg.env_hi)  # the UDF's latent envelope guard, each step
    return Y_pred, abs_pred, max_env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default="runs/merged_best")
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--n-cases", type=int, default=40)
    ap.add_argument("--substeps", default="1",
                    help="comma-list of sub-steps/interval to sweep (proxy for finer CFD timesteps), e.g. 1,20,100")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    substep_vals = [int(s) for s in str(args.substeps).split(",") if s.strip()]

    df = load_database(args.db)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=args.seed, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    sg = Surrogate(args.bundle)

    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    tau_col = sch.state["tau"]; cid_col = sch.meta["CaseID"]
    abs_col = sch.energy_target_column()
    in_cols = [f"Y_{s}" for s in sg.input]
    majors = [s for s in MAJOR_SPECIES if s in sg.input]
    maj_idx = [sg.input.index(s) for s in majors]

    cids = dft[cid_col].drop_duplicates().to_numpy()
    pick = cids[:: max(1, len(cids) // args.n_cases)][: args.n_cases]

    # cache per-case arrays once + the model-independent τ-scheme control
    cases, ctrl_rl = [], []
    for c in pick:
        sub = dft.loc[(dft[cid_col] == c).to_numpy()]
        order = np.argsort(sub[tau_col].to_numpy())
        tau = sub[tau_col].to_numpy()[order]
        if len(tau) < 4 or not np.all(np.diff(tau) > 0):
            continue
        T = sub[tc].to_numpy()[order]; P = sub[pc].to_numpy()[order]
        Y_true = sub[in_cols].to_numpy(float)[order]
        abs_true = np.clip(sub[abs_col].to_numpy(float)[order], 0.0, None)
        dydt_true = sub[[f"dYdt_{s} [1/s]" for s in sg.input]].to_numpy(float)[order]
        cases.append((int(c), T, P, tau, Y_true, abs_true))
        Y_ctrl = Y_true[0] + np.concatenate([[np.zeros(Y_true.shape[1])],
                  np.cumsum(0.5 * (dydt_true[1:] + dydt_true[:-1]) * np.diff(tau)[:, None], axis=0)])
        ctrl_rl.append(np.mean([_relrmse(Y_ctrl[:, j], Y_true[:, j]) for j in maj_idx]))

    print(f"=== A-POSTERIORI latent rollout (no CFD) — {args.bundle} ===")
    print(f"cases: {len(cases)}   substep sweep: {substep_vals}")
    print(f"τ-scheme control (integrate STORED dY/dτ): major-traj relRMSE "
          f"median={np.median(ctrl_rl):.4f}  (≈0 ⇒ integration scheme + units correct)\n")
    print(f"  {'substeps':>8} {'rolled':>7} {'blewup':>7} {'traj_relRMSE(med)':>18} {'traj_ABS_RMSE(med/p95)':>24} "
          f"{'outlet|ΔY|(med)':>16} {'∫S_E relerr(med)':>16} {'env-clamp(med/max)':>20}")
    for ns in substep_vals:
        rows = []
        for (c, T, P, tau, Y_true, abs_true) in cases:
            Y_pred, abs_pred, max_env = rollout_case(sg, T, P, tau, Y_true, abs_true, substeps=ns)
            if not np.all(np.isfinite(Y_pred)):
                rows.append({"blewup": True}); continue
            I_p = float(_TRAP(abs_pred, tau)); I_t = float(_TRAP(abs_true, tau))
            rows.append({"blewup": False,
                         "maj": float(np.mean([_relrmse(Y_pred[:, j], Y_true[:, j]) for j in maj_idx])),
                         # ABSOLUTE trajectory RMSE (mass-frac units) — immune to near-zero-species relRMSE artifact
                         "abs": float(np.mean([np.sqrt(np.mean((Y_pred[:, j] - Y_true[:, j]) ** 2)) for j in maj_idx])),
                         "term": float(np.mean([abs(Y_pred[-1, j] - Y_true[-1, j]) for j in maj_idx])),
                         "eint": abs(I_p - I_t) / max(abs(I_t), 1.0), "env": max_env})
        ok = [r for r in rows if not r["blewup"]]; nb = len(rows) - len(ok)
        g = lambda k: np.array([r[k] for r in ok])  # noqa: E731
        print(f"  {ns:>8} {len(ok):>7} {nb:>7} {np.median(g('maj')):>18.4f} "
              f"{np.median(g('abs')):>11.2e}/{np.percentile(g('abs'),95):<11.2e} {np.median(g('term')):>16.2e} "
              f"{np.median(g('eint')):>16.4f} {np.median(g('env')):>9.2e}/{g('env').max():<9.2e}")
    print("\n  Reading it: traj_relRMSE -> O(0.1) = stable & accurate; >>1 / env-clamp >>0 = closed-loop drift.")
    print("  If finer substeps collapse the error, the storage-grid explicit-Euler was the artifact, not the model.")


if __name__ == "__main__":
    main()

"""Pinpoint the a-posteriori instability: is the ω_Z latent-source head weak, or is a GOOD source
amplified by the closed loop? Evaluated at TRUE encoded states on the held-out test split.

For each test row:  z_true = clip(E·(Y-μ)/σ, env);  z_proj = E·decode(z_true, q)
  - projection residual ||z_proj - z_true|| / envelope-span     (is the manifold map tight at truth?)
  - ω_Z model = sinh(latent_source(z_proj,q))·s_z   vs   ω_Z true = (dY/dt /σ) @ Eᵀ   (source accuracy)

Compares to the rate head's energy accuracy (already R²≈0.995) to locate the weak link.

Run: .venv/bin/python scripts/diag_latent_source.py runs/merged_best
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.coupling.codegen import _numpy_forward
from scarfs.training.datamodule import tripartite_case_split
from aposteriori_rollout import Surrogate


def _r2(pred, true):
    ss = np.sum((true - true.mean(0)) ** 2, axis=0)
    return 1.0 - np.sum((pred - true) ** 2, axis=0) / np.where(ss > 0, ss, 1.0)


def main() -> None:
    bundle = sys.argv[1] if len(sys.argv) > 1 else "runs/merged_best"
    db = "Database_FINAL.parquet"
    df = load_database(db)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    sg = Surrogate(bundle)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    T = dft[tc].to_numpy(float); P = dft[pc].to_numpy(float)
    Y = dft[[f"Y_{s}" for s in sg.input]].to_numpy(float)
    dydt = dft[[f"dYdt_{s} [1/s]" for s in sg.input]].to_numpy(float)

    qn = sg.q(T, P)
    z_true = np.clip((Y - sg.cm) / sg.cs @ sg.W["encoder_W"].T, sg.env_lo, sg.env_hi)
    y_dec = _numpy_forward(z_true, qn, sg.W["decoder_layers"])
    z_proj = y_dec @ sg.W["encoder_W"].T
    span = np.maximum(sg.env_hi - sg.env_lo, 1e-12)

    proj_resid = np.linalg.norm(z_proj - z_true, axis=1) / np.linalg.norm(span)
    omega_model = np.sinh(np.clip(_numpy_forward(z_proj, qn, sg.W["latent_source_layers"]), -20, 20)) * sg.s_z
    omega_true = (dydt / sg.cs) @ sg.W["encoder_W"].T

    r2_dim = _r2(omega_model, omega_true)
    # overall scale-invariant agreement
    corr = np.corrcoef(omega_model.ravel(), omega_true.ravel())[0, 1]
    relrmse = np.sqrt(np.mean((omega_model - omega_true) ** 2)) / np.sqrt(np.mean(omega_true ** 2))

    print(f"=== latent-source (ω_Z) accuracy at TRUE states — {bundle} (n={len(dft)}) ===\n")
    print(f"  manifold projection residual ||E·D(z)-z||/||span||:  median={np.median(proj_resid):.4f}  "
          f"p95={np.percentile(proj_resid,95):.4f}   (small ⇒ map tight at truth)\n")
    print(f"  ω_Z  vs  true dZ/dτ:")
    print(f"     overall correlation = {corr:.4f}")
    print(f"     overall relRMSE     = {relrmse:.4f}")
    print(f"     per-dim R²: median={np.median(r2_dim):.4f}  min={r2_dim.min():.4f}  "
          f"max={r2_dim.max():.4f}  frac(dims R²>0.9)={np.mean(r2_dim>0.9):.2f}")
    print(f"\n  Contrast: the rate head's energy path scores R²≈0.995 (a-priori). If ω_Z R² is much")
    print(f"  lower, the latent-transport SOURCE is the weak link behind the closed-loop drift.")


if __name__ == "__main__":
    main()

"""E1 (overnight diagnosis): target-statistics audit for the latent-source loss — H1 test.

No training. Loads the EXISTING bootstrap bundle (read-only) and the stride5 val split it
was trained against, rebuilds the exact training-time quantities (standardized composition,
ż = E·(Ẏ⊘σ) with the TRAINED encoder, the bundle's saved s_Z), and answers:

  (a) How far apart are median|z| (what `freeze_latent_arcsinh_scale` froze) and median|ż|
      (what the arcsinh transform of the SOURCE target needs)?
  (b) Does Var(arcsinh(ż/s_Z_saved)) per dim ≈ the observed stuck val loss (~50)?  If the
      model's per-dim MSE ≈ that variance, the head has learned ≈ nothing beyond the mean —
      the loss floor is the target's own variance under the mis-scaled transform (H1).
  (c) Under the corrected scale s_Z* = median|ż| per dim, what do Var and the linear-region
      fraction become?

Usage:  .venv/bin/python scripts/overnight/e1_target_stats.py [--database PATH] [--rows-cap N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from scarfs.benchmark.loader import infer_schema
from scarfs.models.adapter import TorchSurrogate
from scarfs.training.datamodule import tripartite_case_split

BUNDLE = REPO / "runs" / "merged_bootstrap_stride5"   # overridden by --bundle
DEFAULT_DB = "/Users/tamasbuzogany/Documents/SCARFS/TEST_ETHANE_LOW_sobol_stride5.parquet"


def load_val_frame(db: str, rows_cap: int) -> tuple[pd.DataFrame, object, dict]:
    """Load the bundle's val split rows (columns projected), capped by whole cases."""
    spec = json.loads((BUNDLE / "spec.json").read_text(encoding="utf-8"))
    cfg = spec["config_echo"]["data"]
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(db)
    needed = [c for c in pf.schema_arrow.names
              if c.startswith(("Y_", "dYdt_")) or c == "CaseID"
              or c.split("[")[0].strip() in ("T", "P", "rho", "tau")
              or "absorption" in c.lower() or c.startswith("R_")]
    df = pd.read_parquet(db, columns=needed)
    schema = infer_schema(df)
    _, val_mask, _ = tripartite_case_split(
        df, schema, val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"],
        seed=cfg["seed"], split_by_case=cfg["split_by_case"])
    dfv = df.loc[val_mask].reset_index(drop=True)
    if len(dfv) > rows_cap:
        keep_cases = pd.unique(dfv[schema.meta["CaseID"]])[: max(1, rows_cap // 21)]
        dfv = dfv[dfv[schema.meta["CaseID"]].isin(set(keep_cases))].reset_index(drop=True)
    return dfv, schema, spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=DEFAULT_DB)
    ap.add_argument("--rows-cap", type=int, default=20000)
    ap.add_argument("--bundle", default=None,
                    help="Bundle dir to audit (default: runs/merged_bootstrap_stride5)")
    args = ap.parse_args()
    global BUNDLE
    if args.bundle:
        BUNDLE = Path(args.bundle)

    dfv, schema, spec = load_val_frame(args.database, args.rows_cap)
    print(f"data: {args.database}")
    print(f"val rows used: {len(dfv)} ({dfv[schema.meta['CaseID']].nunique()} cases)")

    surr = TorchSurrogate.from_merged_bundle(BUNDLE, schema)
    model, scalers = surr.model, surr.scalers
    input_species = list(surr.spec.input_species)
    s_z_saved = np.asarray(spec["arcsinh_latent_scale"], dtype=float)

    # exact training-time quantities
    y = dfv[schema.y_columns(input_species)].to_numpy(float)
    x_std = scalers.composition.transform(y)
    sigma = np.asarray(scalers.composition.scale_, dtype=float)
    dydt = dfv[schema.dydt_columns(input_species)].to_numpy(float)
    E = model.encoder.weight.detach().numpy()                       # (k, n_dry), trained
    z = x_std @ E.T
    z_dot = (dydt / sigma) @ E.T

    from scarfs.models.common import thermo_features
    q_std = scalers.thermo.transform(thermo_features(
        dfv[schema.state["T"]].to_numpy(float), dfv[schema.state["P"]].to_numpy(float)))
    with torch.no_grad():
        zt = torch.as_tensor(x_std, dtype=torch.float32)
        qt = torch.as_tensor(q_std, dtype=torch.float32)
        z_enc = model.encode(zt)
        z_proj = model.encoder(model.decode(z_enc, qt))
        pred_scaled = model.latent_source(z_proj, qt).numpy()       # head output (target space)

    s_z_corr = np.maximum(np.median(np.abs(z_dot), axis=0), 1e-12)
    t_saved = np.arcsinh(z_dot / s_z_saved)
    t_corr = np.arcsinh(z_dot / s_z_corr)

    k = z.shape[1]
    print("\n(a) scale audit — per latent dim")
    print(f"{'dim':>3} {'median|z|':>12} {'s_Z saved':>12} {'median|zdot|':>14} {'ratio zdot/saved':>17}")
    for i in range(k):
        mz = np.median(np.abs(z[:, i]))
        mzd = np.median(np.abs(z_dot[:, i]))
        print(f"{i:>3} {mz:12.4g} {s_z_saved[i]:12.4g} {mzd:14.4g} {mzd / s_z_saved[i]:17.4g}")

    print("\n(b) target stats under SAVED s_Z (training-time) — H1 floor test")
    mse_dim = ((pred_scaled - t_saved) ** 2).mean(axis=0)
    var_dim = t_saved.var(axis=0)
    print(f"{'dim':>3} {'Var(target)':>12} {'model MSE':>12} {'R2(target)':>11} {'frac |t|>5':>11}")
    for i in range(k):
        r2 = 1.0 - mse_dim[i] / var_dim[i] if var_dim[i] > 0 else float("nan")
        print(f"{i:>3} {var_dim[i]:12.4g} {mse_dim[i]:12.4g} {r2:11.3f} "
              f"{np.mean(np.abs(t_saved[:, i]) > 5):11.3f}")
    print(f"mean Var(target | saved s_Z) = {var_dim.mean():.3f}   "
          f"(observed stuck val latent_source ≈ 50.9)")
    print(f"mean model MSE (this val subset) = {mse_dim.mean():.3f}")

    print("\n(c) target stats under CORRECTED s_Z* = median|zdot| per dim")
    var_c = t_corr.var(axis=0)
    lin_c = (np.abs(t_corr) < 1.0).mean(axis=0)
    print(f"{'dim':>3} {'Var(target)':>12} {'frac |t|<1':>11} {'frac |t|>5':>11}")
    for i in range(k):
        print(f"{i:>3} {var_c[i]:12.4g} {lin_c[i]:11.3f} {np.mean(np.abs(t_corr[:, i]) > 5):11.3f}")
    print(f"mean Var(target | corrected s_Z*) = {var_c.mean():.3f}")


if __name__ == "__main__":
    main()

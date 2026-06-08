"""Local, dependency-light sanity check that the SCARFS pipeline executes end-to-end.

This does NOT train a model (PyTorch/Cantera live on the HPC). It proves the data path runs on the
real database fixture: enriched case sampling -> schema -> datamodule features/targets ->
a-priori benchmark harness (with a baseline standing in for a trained surrogate) -> figures.

Run:  python scripts/local_sanity_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/local_sanity_check.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from scarfs.benchmark.apriori import AprioriConfig, holdout_split, run_apriori
from scarfs.benchmark.baselines import NearestNeighborRates
from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.benchmark.metrics import relative_error
from scarfs.data.config import DataGenConfig
from scarfs.data.sampling import build_cases, coverage_summary
from scarfs.plotting import figures
from scarfs.training.config import DataConfig
from scarfs.training.datamodule import prepare_data

DB = "Database_Validation3.csv"
OUT = Path("sanity_figures")


def main() -> None:
    print("=" * 78)
    print("STEP 1 — enriched case sampling (F1/F4), no Cantera needed")
    cases = build_cases(DataGenConfig(n_body_cases=200, n_inlet_seed_cases=40, n_highT_cases=20))
    cov = coverage_summary(cases)
    print(f"  generated {cov['n_cases']} cases, regimes={cov['regimes']}")
    print(f"  T_in range [K] = {tuple(round(x,1) for x in cov['T_in_K'])}, "
          f"L range [m] = {tuple(round(x,2) for x in cov['L_m'])}, X_H2O = {cov['X_H2O']}")

    print("=" * 78)
    print(f"STEP 2 — load real database fixture: {DB}")
    df = load_database(DB)
    schema = infer_schema(df)
    print(f"  rows={len(df)}, species={len(schema.species)} "
          f"(molecular={len(schema.molecular_species())}, radical={len(schema.radical_species())})")
    print(f"  major species present: {schema.major_species()[:8]} ...")

    print("=" * 78)
    print("STEP 3 — datamodule: scaled features/targets + near-inlet weighting (F1)")
    major = list(schema.major_species())
    data_cfg = DataConfig(input_species=major, target_species=major,
                          val_fraction=0.3, split_by_case=False)
    bundle = prepare_data(data_cfg, df, schema)
    print(f"  X_train={bundle.X_train.shape}, Y_train={bundle.Y_train.shape}, "
          f"X_val={bundle.X_val.shape}; inlet up-weighted rows={(bundle.w_train > 1).sum()}")

    print("=" * 78)
    print("STEP 4 — a-priori benchmark harness on real data (baseline as stand-in surrogate)")
    cfg = AprioriConfig(focus_species=tuple(major))
    train_df, test_df, _, _ = holdout_split(df, schema, cfg)
    predictor = NearestNeighborRates(tuple(major)).fit(train_df, schema)
    report = run_apriori(predictor, train_df, test_df, schema, cfg)
    print(report.summary())

    print("=" * 78)
    print("STEP 5 — figures (user palette, 400 DPI, dual K/°C)")
    OUT.mkdir(exist_ok=True)
    pred = predictor.predict(test_df)
    true = test_df[schema.r_columns(major)].to_numpy(dtype=float)
    rel = relative_error(true, pred.rates).ravel()
    T_K = test_df[schema.state["T"]].to_numpy(dtype=float)
    figures.parity_plot(true, pred.rates, major, log_scale=True,
                        path=str(OUT / "parity_rates.png"))
    figures.relative_error_histogram(rel, path=str(OUT / "rel_error_hist.png"))
    figures.error_vs_temperature(np.repeat(T_K, true.shape[1]), np.abs(rel),
                                 path=str(OUT / "error_vs_T.png"))
    print(f"  wrote figures to {OUT.resolve()}")

    print("=" * 78)
    print("SANITY CHECK COMPLETE — data path executes end-to-end on real data.")
    print("Full training (PyTorch) and CFD coupling (Fluent) run on the HPC; see README_SCARFS_ML.md.")


if __name__ == "__main__":
    main()

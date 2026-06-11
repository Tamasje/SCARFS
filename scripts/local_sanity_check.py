"""Local sanity check that the SCARFS pipeline executes end-to-end on real data.

Two parts:

1. LEGACY path (steps 1–5, dependency-light): enriched case sampling -> schema ->
   datamodule features/targets -> a-priori benchmark harness (baseline as stand-in
   surrogate) -> figures, on the legacy CSV fixture.
2. MERGED path (steps 6–9, needs PyTorch): tiny MergedCoil training on a real parquet
   slice (defaults to the in-repo stride6 fixture; point ``--merged-db`` at a stride5
   slice for a stronger check) -> §5 energy acceptance suite on the val split ->
   diagnostics audits (energy unit/coverage, sanity, front resolution) -> Fluent UDF
   codegen incl. the compiled standalone forward test -> energy figures.

Run:  python scripts/local_sanity_check.py [--merged-db PATH] [--max-cases N] [--skip-merged]
"""

from __future__ import annotations

import argparse
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
MERGED_DB_DEFAULT = Path(__file__).resolve().parent.parent / "tests" / "data" / "stride6_sample.parquet"
MECH_YAML = Path(__file__).resolve().parent.parent / "chem_ForTransport.yaml"
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
    print("LEGACY-PATH SANITY COMPLETE — data path executes end-to-end on real data.")


def merged_sanity(merged_db: Path, max_cases: int) -> None:
    """Steps 6–9: the merged pipeline end-to-end on a real parquet slice (needs PyTorch)."""
    import pandas as pd

    from scarfs.benchmark.energy import evaluate_energy
    from scarfs.coupling.codegen import export_merged_udf
    from scarfs.diagnostics import (
        run_energy_coverage_audit,
        run_energy_unit_audit,
        run_front_resolution_audit,
        run_raw_database_sanity,
    )
    from scarfs.models.features import build_absorption_target
    from scarfs.training.config import LossConfig, ModelConfig, OptimConfig, TrainConfig
    from scarfs.training.datamodule import tripartite_case_split
    from scarfs.training.train import train

    print("=" * 78)
    print(f"STEP 6 — merged training on real parquet slice: {merged_db}")
    df = load_database(merged_db)
    schema = infer_schema(df)
    case_col = schema.meta["CaseID"]
    keep = sorted(df[case_col].unique())[:max_cases]
    df = df[df[case_col].isin(keep)].reset_index(drop=True)
    df.to_parquet(OUT / "_merged_sanity_slice.parquet", index=False)
    print(f"  rows={len(df)}, cases={len(keep)}, species={len(schema.species)}")

    seed = next(
        s for s in range(50)
        if all(m.any() for m in tripartite_case_split(
            df, schema, val_fraction=0.3, test_fraction=0.2, seed=s, split_by_case=True))
    )
    cfg = TrainConfig(
        data=DataConfig(
            database_path=str(OUT / "_merged_sanity_slice.parquet"),
            input_species="dry_all", target_species="energy_active",
            val_fraction=0.3, test_fraction=0.2, split_by_case=True,
            mech_yaml=str(MECH_YAML), tail_strata=5, seed=seed,
        ),
        model=ModelConfig(kind="merged", latent_dim=4),
        optim=OptimConfig(epochs=5, batch_size=1024, patience=5),
        loss=LossConfig(rollout_mode="lagrangian", noise_std=0.01),
        output_dir=str(OUT / "merged_sanity_run"),
    )
    metrics = train(cfg)
    print(f"  trained: epochs={metrics['epochs_run']}, best_val={metrics['best_val']:.4g}")
    print(f"  val absorption (rate-derived): {metrics['absorption_metrics_val'].get('rate_derived')}")
    print(f"  val absorption (distilled head): {metrics['absorption_metrics_val'].get('head')}")

    print("=" * 78)
    print("STEP 7 — §5 energy acceptance suite on the val split (sanity numbers, not certification)")
    from scarfs.models.adapter import TorchSurrogate
    _, val_mask, _ = tripartite_case_split(
        df, schema, val_fraction=0.3, test_fraction=0.2, seed=seed, split_by_case=True)
    df_val = df.loc[val_mask].reset_index(drop=True)
    target = build_absorption_target(df_val, schema).values
    surrogate = TorchSurrogate.from_merged_bundle(cfg.output_dir, schema)
    pred = surrogate.predict(df_val)
    # MergedCoil head path: pred.energy IS absorption (positive), per the adapter contract.
    tau = df_val[schema.state["tau"]].to_numpy(dtype=float)
    cases = df_val[case_col].to_numpy()
    report = evaluate_energy(pred.energy, target, cases, tau)
    print(report.summary())

    figures.energy_parity_figure(pred.energy, target, str(OUT / "energy_parity.png"))
    rel = np.abs(pred.energy - target) / np.maximum(np.abs(target), 1.0)
    figures.tail_rel_err_hist_figure(rel, str(OUT / "energy_tail_rel_err_hist.png"))
    print(f"  wrote energy figures to {OUT.resolve()}")

    print("=" * 78)
    print("STEP 8 — diagnostics audits on the slice")
    diag_out = OUT / "diagnostics"
    diag_out.mkdir(parents=True, exist_ok=True)
    unit = run_energy_unit_audit(df, schema, str(MECH_YAML), diag_out)
    print(f"  energy unit audit: corr={unit.corr:.6f}, relRMSE={unit.rel_rmse:.3g}, PASS={unit.passed}")
    energy_active = list(__import__('json').loads((Path(cfg.output_dir) / 'spec.json').read_text())["energy_active"])
    cov = run_energy_coverage_audit(df, schema, str(MECH_YAML), energy_active, diag_out)
    print(f"  energy coverage audit: miss_fraction={cov.miss_fraction:.3g} (n_active={len(energy_active)}), PASS={cov.passed}")
    san = run_raw_database_sanity(df, schema, diag_out)
    print(f"  raw sanity: PASS={san.passed}")
    front = run_front_resolution_audit(df, schema, diag_out)
    print(f"  front resolution: median jump={front.median_jump_frac:.1%} of case peak, "
          f"p95={front.p95_jump_frac:.1%} (motivates front-adaptive storage on regeneration)")

    print("=" * 78)
    print("STEP 9 — Fluent UDF codegen + compiled standalone forward test")
    export = export_merged_udf(Path(cfg.output_dir), OUT / "udf_export")
    print(f"  artifacts: {sorted(p.name for p in export.artifacts.values())}")
    print(f"  numpy-vs-torch consistency maxima: Y={export.consistency_max_rel_diff_y:.2e}, "
          f"omega_Z={export.consistency_max_rel_diff_omega_z:.2e}, "
          f"S_h={export.consistency_max_rel_diff_sh:.2e}")

    print("=" * 78)
    print("MERGED-PATH SANITY COMPLETE — train -> energy metrics -> audits -> UDF export all execute.")
    print("Real accuracy requires full training on the HPC (stride5/regenerated DB); see README_SCARFS_ML.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCARFS local sanity check")
    parser.add_argument("--merged-db", default=str(MERGED_DB_DEFAULT),
                        help="Parquet database for the merged-path steps (default: in-repo stride6 fixture)")
    parser.add_argument("--max-cases", type=int, default=4, help="Cases to keep for the merged slice")
    parser.add_argument("--skip-merged", action="store_true", help="Run only the legacy steps 1-5")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    main()
    if not args.skip_merged:
        merged_sanity(Path(args.merged_db), args.max_cases)

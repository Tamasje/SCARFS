"""CLI script: §5 beats-both-parents benchmark (plan §5).

Orchestrates the complete parent-comparison protocol:

1. Load the database.
2. Perform tripartite case split (same seed semantics as training).
3. Score:
   - Parent-1: colleague's q12 model (ColleagueReducedSurrogate).
   - Parent-2: our mimic baseline bundle (OurMimicBaseline).
   - Merged:   our merged bundle (OurMimicBaseline from merged bundle dir).
4. Evaluate each with ``evaluate_energy`` and per-species rate R².
5. Emit a PASS/FAIL comparison as Markdown.

Smoke-test
----------
The accompanying unit test ``tests/test_benchmark_parents_script.py`` runs
``main()`` on the stride6 fixture with stub surrogates (frozen-rate baseline)
to avoid any training in tests.

Usage example
-------------
    .venv/bin/python scripts/benchmark_parents.py \\
        --database TEST_ETHANE_LOW_sobol_stride5.parquet \\
        --parent1-dir /path/to/colleague/outputs \\
        --parent2-dir runs/mimic/k3 \\
        --merged-dir runs/merged/k12 \\
        --seed 0 \\
        --out benchmark_parents_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _score_surrogate(
    surrogate: Any,
    df: pd.DataFrame,
    schema: Any,
    label: str,
) -> dict[str, Any]:
    """Score a surrogate on *df* and return a results dict."""
    from scarfs.benchmark.energy import evaluate_energy, EnergyThresholds
    from scarfs.benchmark.metrics import r2_score

    try:
        pred = surrogate.predict(df)
    except Exception as exc:
        return {"label": label, "error": str(exc), "passed": False}

    case_col = schema.meta.get("CaseID")
    tau_col = schema.state.get("tau")

    if case_col and case_col in df.columns:
        case_ids = df[case_col].to_numpy()
    else:
        case_ids = np.zeros(len(df), dtype=int)

    if tau_col and tau_col in df.columns:
        tau = df[tau_col].to_numpy(dtype=float)
    else:
        tau = np.arange(len(df), dtype=float)

    # Get target absorption
    result: dict[str, Any] = {"label": label, "error": None}
    try:
        abs_col = schema.energy_target_column()
        target_abs = df[abs_col].to_numpy(dtype=float)
        pred_abs = pred.energy  # convention: absorption, positive = endothermic
        energy_report = evaluate_energy(
            pred_abs, target_abs, case_ids, tau, thresholds=EnergyThresholds()
        )
        result["energy_report"] = energy_report
        result["passed"] = energy_report.passed
        result["global_r2"] = energy_report.global_r2
        result["global_rel_rmse"] = energy_report.global_rel_rmse
    except Exception as exc:
        result["energy_error"] = str(exc)
        result["passed"] = False
        result["global_r2"] = np.nan
        result["global_rel_rmse"] = np.nan

    # Per major-species rate R²
    from scarfs.schema import MAJOR_SPECIES
    from scarfs.models.features import build_mass_rate_matrix

    major_r2: dict[str, float] = {}
    try:
        present_major = [s for s in MAJOR_SPECIES if s in schema.species and s in pred.species]
        if present_major:
            true_rates = build_mass_rate_matrix(df, schema, present_major, prefer_dydt=True)
            for i, sp in enumerate(present_major):
                if sp in pred.species:
                    sp_idx = list(pred.species).index(sp)
                    major_r2[sp] = r2_score(true_rates[:, i], pred.rates[:, sp_idx])
    except Exception as exc:
        result["rate_error"] = str(exc)

    result["major_species_r2"] = major_r2
    return result


def _format_markdown(results: list[dict[str, Any]]) -> str:
    """Format the benchmark comparison as Markdown."""
    lines = [
        "# SCARFS Benchmark — Beats-Both-Parents Protocol",
        "",
        "All metrics are on held-out test cases.  "
        "Results labeled *bootstrap, non-certifying* when computed on stride5/6.",
        "",
    ]

    # Summary table
    lines += [
        "## Summary",
        "",
        "| Model | Global R² | relRMSE | PASS |",
        "|-------|----------:|---------:|:----:|",
    ]
    for r in results:
        label = r["label"]
        if r.get("error"):
            lines.append(f"| {label} | ERROR | ERROR | FAIL |")
            continue
        r2 = r.get("global_r2", np.nan)
        rrmse = r.get("global_rel_rmse", np.nan)
        passed = "PASS" if r.get("passed", False) else "FAIL"
        lines.append(f"| {label} | {r2:.4f} | {rrmse:.4f} | {passed} |")

    lines.append("")

    # Per-model detail
    for r in results:
        label = r["label"]
        lines += [f"## {label}", ""]
        if r.get("error"):
            lines.append(f"**ERROR**: {r['error']}")
            lines.append("")
            continue

        er = r.get("energy_report")
        if er is not None:
            lines.append("### Energy Metrics")
            lines.append("")
            lines.append(er.summary().replace("\n", "\n"))
            lines.append("")

        major = r.get("major_species_r2", {})
        if major:
            lines += ["### Major Species Rate R²", "", "| Species | R² |", "|---------|---:|"]
            for sp, r2v in sorted(major.items()):
                lines.append(f"| {sp} | {r2v:.4f} |")
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the benchmark_parents script."""
    parser = argparse.ArgumentParser(
        description="§5 beats-both-parents benchmark protocol."
    )
    parser.add_argument("--database", type=str, required=True, help="Database path.")
    parser.add_argument("--parent1-dir", type=str, default=None,
                        help="Colleague outputs dir (model_arrays.npz + model_metadata.json).")
    parser.add_argument("--parent2-dir", type=str, default=None,
                        help="Our mimic bundle dir (model.pt + scalers.pkl + spec.json).")
    parser.add_argument("--merged-dir", type=str, default=None,
                        help="Our merged bundle dir.")
    parser.add_argument("--seed", type=int, default=0, help="Split seed.")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--out", type=str, default=None,
                        help="Optional Markdown output path.")
    parser.add_argument("--use-stub", action="store_true", default=False,
                        help="Use FrozenComposition stubs for testing (no real models).")
    args = parser.parse_args(argv)

    from scarfs.benchmark.loader import load_database, infer_schema
    from scarfs.training.datamodule import tripartite_case_split

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        return 1

    print(f"Loading database: {db_path}", flush=True)
    df = load_database(str(db_path))
    schema = infer_schema(df)

    # Tripartite split — use same seed semantics as training
    _, _, test_mask = tripartite_case_split(
        df, schema,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        split_by_case=True,
    )
    test_df = df[test_mask].reset_index(drop=True)
    if len(test_df) == 0:
        # Fall back to all data for very small databases
        test_df = df.copy()

    print(f"Test set: {len(test_df)} rows, {test_df['CaseID'].nunique() if 'CaseID' in test_df.columns else '?'} cases.", flush=True)

    results: list[dict] = []

    if args.use_stub:
        # Smoke-test mode: use FrozenComposition baseline for all models
        from scarfs.benchmark.baselines import FrozenComposition
        active = schema.active_species()
        for label in ["Parent-1 (stub)", "Parent-2 (stub)", "Merged (stub)"]:
            stub = FrozenComposition(active_species=active)
            r = _score_surrogate(stub, test_df, schema, label)
            results.append(r)
    else:
        # Parent-1: colleague reduced surrogate
        if args.parent1_dir:
            print("Loading Parent-1 (colleague) ...", flush=True)
            from scarfs.benchmark.parents import ColleagueReducedSurrogate
            try:
                p1 = ColleagueReducedSurrogate.from_outputs_dir(args.parent1_dir)
                results.append(_score_surrogate(p1, test_df, schema, "Parent-1 (colleague q12)"))
            except Exception as exc:
                results.append({"label": "Parent-1 (colleague q12)", "error": str(exc), "passed": False})
        else:
            print("Skipping Parent-1 (no --parent1-dir).", flush=True)

        # Parent-2: our mimic
        if args.parent2_dir:
            print("Loading Parent-2 (our mimic) ...", flush=True)
            from scarfs.benchmark.parents import OurMimicBaseline
            try:
                p2 = OurMimicBaseline.from_bundle_dir(args.parent2_dir)
                results.append(_score_surrogate(p2, test_df, schema, "Parent-2 (our mimic)"))
            except Exception as exc:
                results.append({"label": "Parent-2 (our mimic)", "error": str(exc), "passed": False})
        else:
            print("Skipping Parent-2 (no --parent2-dir).", flush=True)

        # Merged
        if args.merged_dir:
            print("Loading Merged model ...", flush=True)
            from scarfs.benchmark.parents import OurMimicBaseline
            try:
                merged = OurMimicBaseline.from_bundle_dir(args.merged_dir)
                results.append(_score_surrogate(merged, test_df, schema, "Merged"))
            except Exception as exc:
                results.append({"label": "Merged", "error": str(exc), "passed": False})
        else:
            print("Skipping Merged (no --merged-dir).", flush=True)

    if not results:
        print("No surrogates to evaluate.  Provide --parent1-dir, --parent2-dir, or --merged-dir.", file=sys.stderr)
        return 1

    report_text = _format_markdown(results)
    print(report_text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to {out_path}", flush=True)

    # Return code: 0 if all pass, 1 if any fail
    all_passed = all(r.get("passed", False) for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

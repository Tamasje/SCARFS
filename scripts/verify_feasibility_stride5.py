"""CLI script: run the kNN feasibility pre-gate on a database (§4 E-e).

Intended use
------------
Re-verify the critic's feasibility numbers on a re-split stride5 database:

    .venv/bin/python scripts/verify_feasibility_stride5.py \\
        --database TEST_ETHANE_LOW_sobol_stride5.parquet \\
        --rows-cap 20000 \\
        --columns-projection \\
        --out feasibility_stride5.txt

The script guards memory by loading only the columns needed for feasibility
analysis (Y_* species, T, P, CaseID, absorption).  This avoids pulling the
full ~1100-column stride5 file into RAM.

Output is printed to stdout and optionally saved to --out.

No test executes this against stride5 (environment-dependent, large file);
there is a unit test running main() against the stride6 fixture via subprocess.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Make the repo root importable when run as `python scripts/verify_feasibility_stride5.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Resolve columns to load for column projection
_NEEDED_STATE_BASES = ("T [K]", "P [Pa]", "rho [kg/m3]", "tau [s]", "z [m]")
_NEEDED_META = ("CaseID",)
_ABSORPTION_COL = "Reaction heat absorption [J/s/m3]"


def _build_column_list(df_columns: Sequence[str]) -> list[str]:
    """Select only the columns needed for feasibility analysis.

    Includes Y_*, dYdt_*, R_* (rate-family columns needed by Schema.from_columns
    to pass rate-coverage validation), as well as T, P, CaseID, and absorption.
    """
    cols = []
    for c in df_columns:
        base = c.split("[")[0].strip()
        # Y_ columns (composition)
        if c.startswith("Y_"):
            cols.append(c)
        # Rate-family columns — required by Schema.from_columns for coverage validation
        elif c.startswith("dYdt_") or c.startswith("R_"):
            cols.append(c)
        # Needed state/meta columns
        elif c in _NEEDED_STATE_BASES or base.lower() in ("t", "p", "rho", "tau", "z"):
            cols.append(c)
        elif c in _NEEDED_META or base.lower() == "caseid":
            cols.append(c)
        elif c == _ABSORPTION_COL or "absorption" in c.lower():
            cols.append(c)
    return cols


def _load_database_projected(path: Path, rows_cap: int | None, columns_projection: bool) -> "pd.DataFrame":
    """Load database with optional column projection and row cap."""
    import pandas as pd

    if path.suffix.lower() in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(str(path))
            all_cols = pf.schema_arrow.names
            if columns_projection:
                needed = _build_column_list(all_cols)
                if not needed:
                    needed = all_cols
            else:
                needed = None
            df = pf.read(columns=needed).to_pandas()
        except ImportError:
            df = pd.read_parquet(str(path))
            if columns_projection:
                needed = _build_column_list(list(df.columns))
                df = df[needed]
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(str(path))
        if columns_projection:
            needed = _build_column_list(list(df.columns))
            df = df[needed]
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    if rows_cap is not None and rows_cap > 0 and len(df) > rows_cap:
        # Sample rows deterministically (stratified by CaseID if available)
        import numpy as np
        case_col = next((c for c in df.columns if c.lower() == "caseid" or c == "CaseID"), None)
        if case_col is not None:
            # Sample proportionally from each case
            rng = np.random.default_rng(42)
            sampled_idx: list[int] = []
            for _cid, grp in df.groupby(case_col):
                n_from_case = max(1, int(rows_cap * len(grp) / len(df)))
                chosen = rng.choice(len(grp), size=min(n_from_case, len(grp)), replace=False)
                sampled_idx.extend(grp.index[chosen].tolist())
            df = df.loc[sampled_idx].reset_index(drop=True)
        else:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(df), size=rows_cap, replace=False)
            df = df.iloc[idx].reset_index(drop=True)

    return df


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI.

    Returns 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Run kNN feasibility pre-gate (§4 E-e) on a database.",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="/Users/tamasbuzogany/Documents/SCARFS/TEST_ETHANE_LOW_sobol_stride5.parquet",
        help="Path to the database file (CSV or Parquet).",
    )
    parser.add_argument(
        "--rows-cap",
        type=int,
        default=None,
        help="Cap on number of rows to load (random sample; None = all).",
    )
    parser.add_argument(
        "--columns-projection",
        action="store_true",
        default=False,
        help="Load only needed columns (Y_*, T, P, CaseID, absorption).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to write the report text.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="4,6,8,12,16",
        help="Comma-separated latent dimensions to evaluate.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.20,
        help="Fraction of rows (by |target|) defining the tail region.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for GroupKFold fold assignment.",
    )

    args = parser.parse_args(argv)
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"ERROR: database not found: {db_path}", file=sys.stderr)
        return 1

    ks = [int(k.strip()) for k in args.ks.split(",") if k.strip()]

    print(f"Loading database: {db_path}", flush=True)
    try:
        df = _load_database_projected(db_path, args.rows_cap, args.columns_projection)
    except Exception as exc:
        print(f"ERROR: failed to load database: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.", flush=True)

    from scarfs.benchmark.loader import infer_schema
    from scarfs.benchmark.feasibility import feasibility_table

    try:
        schema = infer_schema(df)
    except Exception as exc:
        print(f"ERROR: schema inference failed: {exc}", file=sys.stderr)
        return 1

    try:
        report = feasibility_table(
            df,
            schema,
            ks=ks,
            tail_fraction=args.tail_fraction,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"ERROR: feasibility_table failed: {exc}", file=sys.stderr)
        return 1

    text = report.summary()
    print(text)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\nReport saved to {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

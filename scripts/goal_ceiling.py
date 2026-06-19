"""Achievability-ceiling probe for the energy (absorption) target on the pilot DB.

The out-of-fold kNN R² of a smooth regressor on the FULL state is the best any model of this input
could reach (the information ceiling); PCA-k approximates what the k-dim linear encoder retains.
The gap between this ceiling and a trained model's R² is the real headroom — it tells us, honestly,
how large an error reduction is physically possible (R² is bounded by 1; relRMSE by the noise floor).

Subsamples by CASE to keep the brute-force kNN tractable (the full 47k×~210 run OOMs).
Run: .venv/bin/python scripts/goal_ceiling.py [n_cases]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scarfs.benchmark.feasibility import feasibility_table
from scarfs.benchmark.loader import infer_schema, load_database


def main() -> None:
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 350
    df = pd.read_parquet("pilot.parquet")
    df = df[df["sample_kind"] == "trajectory"].reset_index(drop=True)
    schema = infer_schema(df)
    case_col = schema.meta["CaseID"]
    cases = np.array(sorted(df[case_col].unique()))
    rng = np.random.default_rng(0)
    keep = set(rng.choice(cases, size=min(n_cases, len(cases)), replace=False).tolist())
    sub = df[df[case_col].isin(keep)].reset_index(drop=True)
    print(f"ceiling probe: {len(sub)} rows / {len(keep)} cases (of {len(df)} rows / {len(cases)} cases)",
          flush=True)

    rep = feasibility_table(sub, schema, ks=(8, 16, 24), n_neighbors=5, tail_fraction=0.20, seed=0)
    print(rep.summary(), flush=True)

    # headroom vs the current trained baseline (100ep all-on)
    base_R2 = 0.8550
    ceil = rep.baseline_global_r2
    pca16 = next((e for e in rep.entries if e.space == "pca" and e.k == 16), None)
    print("\n=== HEADROOM (energy, global) ===")
    print(f"  current model R²        : {base_R2:.4f}  (relRMSE 0.372)")
    print(f"  PCA-16 kNN ceiling R²   : {pca16.global_r2:.4f}" if pca16 else "  PCA-16 n/a")
    print(f"  full-state kNN ceiling  : {ceil:.4f}")
    if ceil < 1.0:
        # error-reduction factor available in (1-R²) terms if the model reached the ceiling
        cur_err = 1.0 - base_R2
        ceil_err = max(1.0 - ceil, 1e-9)
        print(f"  max (1-R²) reduction to ceiling: {cur_err/ceil_err:.2f}x  "
              f"(current 1-R²={cur_err:.3f} -> ceiling 1-R²={1-ceil:.3f})")
        print(f"  => relRMSE floor ~ sqrt(1-ceiling) ~ {np.sqrt(max(1-ceil,0)):.3f} "
              f"(10x from 0.372 would need relRMSE 0.037 -> R² 0.9986)")


if __name__ == "__main__":
    main()

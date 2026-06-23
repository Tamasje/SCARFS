"""Data-scaling (learning-curve) capstone: does TEST error fall with the number of on-manifold
training cases? If yes, the pilot's val<->test gap is case-count variance (fixable only by the
regenerated DB), and we can estimate how many cases 10x needs.

Method: the test split is hash-deterministic per CaseID, so keeping ALL test cases + a fraction of
the non-test cases leaves the test set identical while shrinking train+val. Train combo_eck on each
fraction; evaluate on the fixed pilot test split via goal_test_eval.

Run: .venv/bin/python scripts/goal_scaling.py 0.3 0.6
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from ab_pilot_physics import make_cfg
from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.training.datamodule import _case_hash
from scarfs.training.train import train

SEED, TEST_FRAC = 0, 0.15


def main() -> None:
    fracs = [float(x) for x in sys.argv[1:]] or [0.3, 0.6]
    df = load_database("pilot.parquet")
    sch = infer_schema(df)
    case_col = sch.meta["CaseID"]
    cases = np.array(sorted(df[case_col].unique()))
    h = np.array([_case_hash(c, SEED) for c in cases])
    test_cases = set(cases[h < TEST_FRAC].tolist())
    non_test = cases[h >= TEST_FRAC]
    rng = np.random.default_rng(0)
    for f in fracs:
        n = max(1, int(round(f * len(non_test))))
        keep = set(rng.choice(non_test, size=n, replace=False).tolist()) | test_cases
        sub = df[df[case_col].isin(keep)].reset_index(drop=True)
        out_db = Path(f"runs/pilot_frac{int(f*100)}.parquet")
        sub.to_parquet(out_db)
        print(f"\n=== frac={f}: {n} train+val cases (+{len(test_cases)} fixed test), {len(sub)} rows ===", flush=True)
        cfg = make_cfg(f"runs/goal_scale_{int(f*100)}", physics_on=True, epochs=800, patience=150, head_ft=50)
        cfg.loss.keq_weight = 0.0
        cfg.data.database_path = str(out_db)
        cfg.model.latent_dim = 32
        cfg.model.rate_hidden = (256, 256, 128)
        cfg.loss.energy_weight = 1.0
        cfg.data.tail_weight_alpha = 4.0
        cfg.optim.checkpoint_metric = "energy_relrmse"
        train(cfg)
        print(f"frac={f} done -> runs/goal_scale_{int(f*100)}", flush=True)
    print("\ntest-eval: .venv/bin/python scripts/goal_test_eval.py "
          + " ".join(f"runs/goal_scale_{int(f*100)}" for f in fracs) + " runs/goal_combo_eck", flush=True)


if __name__ == "__main__":
    main()

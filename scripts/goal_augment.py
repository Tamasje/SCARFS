"""Data-augmentation experiment: train the best config (combo_eck) on pilot + the 60k off-manifold
points, then evaluate on the SAME pilot TEST split (via goal_test_eval).

The pilot's large val↔test gap (val 0.038 vs test 0.167) says the model is data-limited. The
off-manifold parquet is +60k single-point (state→rate) samples around the manifold — the only
data lever available without the HPC regeneration. Pilot test cases hash identically in the
combined DB, so the test metric stays comparable.

Run: .venv/bin/python scripts/goal_augment.py [epochs]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from ab_pilot_physics import make_cfg
from scarfs.training.train import train

COMBINED = Path("runs/combined_pilot_offmanifold.parquet")


def main() -> None:
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    if not COMBINED.exists():
        p = pd.read_parquet("pilot.parquet")
        o = pd.read_parquet("offmanifold_60000.parquet")
        comb = pd.concat([p, o], ignore_index=True)
        comb.to_parquet(COMBINED)
        print(f"combined: pilot {len(p)} + offmanifold {len(o)} = {len(comb)} rows -> {COMBINED}", flush=True)

    cfg = make_cfg("runs/goal_combo_eck_aug", physics_on=True, epochs=epochs, patience=150, head_ft=50)
    cfg.loss.keq_weight = 0.0
    cfg.data.database_path = str(COMBINED)
    # combo_eck config
    cfg.model.latent_dim = 32
    cfg.model.rate_hidden = (256, 256, 128)
    cfg.loss.energy_weight = 1.0
    cfg.data.tail_weight_alpha = 4.0
    cfg.optim.checkpoint_metric = "energy_relrmse"
    train(cfg)
    print("done -> test-eval: .venv/bin/python scripts/goal_test_eval.py runs/goal_combo_eck_aug", flush=True)


if __name__ == "__main__":
    main()

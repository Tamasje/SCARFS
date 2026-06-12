"""Overnight-diagnosis figure: why latent_source was stuck, and what the fix/k change did.

Reproducible from on-disk artifacts only (bundle metrics.json files); palette per the brief.
Writes runs/overnight_fig/latent_source_diagnosis.png (400 DPI).

Usage: .venv/bin/python scripts/overnight/fig_diagnosis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = ["#3A0CA3", "#4361EE", "#7209B7", "#F72585", "#CBE0D1",
           "#FBEE43", "#00A320", "#4CC9F0"]

# (run, label, color, Var(target) under that run's own s_Z — the predict-the-mean floor,
#  measured by scripts/overnight/e1_target_stats.py --bundle <run>; commands in OVERNIGHT_LOG.md)
RUNS = [
    ("runs/merged_bootstrap_stride5", "baseline k=8 (state-s_Z bug)", PALETTE[0], 49.996),
    ("runs/overnight_e5_k8_fix", "FIX-1 k=8 (source-s_Z)", PALETTE[2], 15.795),
    ("runs/overnight_e6_k16_fix", "FIX-1 k=16", PALETTE[1], 14.007),
    ("runs/overnight_e8_k16_floor", "FIX-1+2 k=16 (σ-floor)", PALETTE[6], 5.398),
    ("runs/overnight_e9_k16_floor_long", "FIX-1+2 k=16 long", PALETTE[3], 5.398),
]


def main() -> None:
    out_dir = REPO / "runs" / "overnight_fig"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    for run, label, color, var_floor in RUNS:
        p = REPO / run / "metrics.json"
        if not p.exists():
            continue
        h = json.loads(p.read_text(encoding="utf-8"))["history"]
        ep = [r["epoch"] for r in h]
        ls = [r.get("val_parts", {}).get("latent_source", np.nan) / var_floor for r in h]
        ax1.plot(ep, ls, color=color, label=label, lw=2)
    ax1.axhline(1.0, color="k", ls=":", lw=1.2, label="predict-the-mean floor (MSE = Var)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("val latent_source MSE / Var(own target)\n(floor-relative; <1 = real skill)")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("ω_Z head: floor-bound until s_Z fix + σ-floor")
    ax1.legend(fontsize=7.5, loc="lower left")

    # single-batch overfit summary (numbers from the logged E2/E3/E2b/E3b/E4b runs;
    # regenerate with scripts/overnight/e2_single_batch.py — commands in OVERNIGHT_LOG.md)
    variants = [
        ("saved s_Z\nvia z_proj", 0.757, PALETTE[0]),
        ("corrected\nvia z_proj", 0.711, PALETTE[2]),
        ("saved s_Z\nz-direct", 0.560, PALETTE[1]),
        ("corrected\nz-direct", 0.432, PALETTE[3]),
        ("corrected\nFULL 212-d input", 0.155, PALETTE[6]),
    ]
    xs = np.arange(len(variants))
    ax2.bar(xs, [v[1] for v in variants], color=[v[2] for v in variants])
    for x, (_, v, _) in zip(xs, variants):
        ax2.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax2.set_xticks(xs, [v[0] for v in variants], fontsize=8)
    ax2.set_ylabel("single-batch final MSE / Var(target)\n(1 = mean-prediction, 0 = memorized)")
    ax2.set_title("Overfit discriminator: information ceiling at k=8,\nnot capacity/optimization")
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")

    fig.tight_layout()
    out = out_dir / "latent_source_diagnosis.png"
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

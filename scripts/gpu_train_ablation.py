"""GPU (MPS/CUDA) k-ablation of the merged model on stride5 — the overnight report's rec #1+#2.

Trains the canonical `configs/train_merged.json` (FIX-1 + FIX-2 + head fine-tune) at each
requested latent dimension, full length, on the best available device. Writes one bundle per k
under --out and a summary.json with the val metrics per k.

Usage:
  .venv/bin/python scripts/gpu_train_ablation.py \
      [--database PATH] [--ks 8,12,16] [--epochs 300] [--patience 50] [--head-finetune 40]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scarfs.training.config import TrainConfig
from scarfs.training.train import _select_device, train

DEFAULT_DB = "/Users/tamasbuzogany/Documents/SCARFS/TEST_ETHANE_LOW_sobol_stride5.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=DEFAULT_DB)
    ap.add_argument("--ks", default="16,12,8", help="latent dims, candidate first")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--head-finetune", type=int, default=40)
    ap.add_argument("--out", default="runs/gpu_ablation")
    args = ap.parse_args()

    print(f"device: {_select_device()}", flush=True)
    out_root = REPO / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    base = json.loads((REPO / "configs" / "train_merged.json").read_text(encoding="utf-8"))
    summary: dict[str, dict] = {}

    for k in [int(x) for x in args.ks.split(",") if x.strip()]:
        cfg_doc = json.loads(json.dumps(base))  # deep copy
        cfg_doc.pop("_comment", None)
        cfg_doc["data"]["database_path"] = args.database
        cfg_doc["model"]["latent_dim"] = k
        cfg_doc["optim"]["epochs"] = args.epochs
        cfg_doc["optim"]["patience"] = args.patience
        cfg_doc["optim"]["head_finetune_epochs"] = args.head_finetune
        cfg_doc["output_dir"] = str(out_root / f"k{k}")
        t0 = time.time()
        metrics = train(TrainConfig.from_mapping(cfg_doc))
        wall = time.time() - t0
        last_latent = metrics["history"][-1]["val_parts"].get("latent_source")
        summary[str(k)] = {
            "epochs_run": metrics["epochs_run"],
            "wall_s": round(wall, 1),
            "val_latent_source_last": last_latent,
            "absorption_val": metrics["absorption_metrics_val"],
            "head_finetune_best": metrics.get("head_finetune", {}).get("best_val_head_loss"),
            "bundle": cfg_doc["output_dir"],
        }
        print(f"[k={k}] epochs={metrics['epochs_run']} wall={wall/60:.1f} min "
              f"val_latent={last_latent:.3f} "
              f"absorption={ {p: round(v['r2'], 4) for p, v in metrics['absorption_metrics_val'].items()} }",
              flush=True)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"summary -> {out_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

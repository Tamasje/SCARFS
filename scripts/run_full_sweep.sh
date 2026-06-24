#!/usr/bin/env bash
# Full-DB training sweep on the regenerated DB (Database_FINAL.parquet -> full.parquet):
#   1) k=32 headline  (configs/train_merged_best.json) -> runs/merged_best
#   2) k=16 ablation  (configs/_merged_k16.json)        -> runs/merged_k16   (CFD-cost tradeoff)
#   3) honest held-out TEST-split energy relRMSE on both
#
# Run from the repo root:   bash scripts/run_full_sweep.sh
# Live progress prints per epoch (best = val energy relRMSE) and is also tee'd to runs/sweep.log.
# Ctrl-C after [k32 end] if you only want the headline run and want to skip the k=16 ablation.
#
# ~4.4 s/epoch on this Mac's GPU (MPS): each 800-epoch run is ~60 min (less if it early-stops on
# the 150-epoch patience). caffeinate keeps the machine awake so an idle-sleep can't kill it.
set -uo pipefail
cd "$(dirname "$0")/.."

export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
LOG=runs/sweep.log
CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"

{
  echo "[SWEEP START $(date)]"
  echo "[k32 start $(date)] -> runs/merged_best"
  $CAF $PY -m scarfs.training.train --config configs/train_merged_best.json
  echo "[k32 end exit=$? $(date)]"
  echo "[k16 start $(date)] -> runs/merged_k16"
  $CAF $PY -m scarfs.training.train --config configs/_merged_k16.json
  echo "[k16 end exit=$? $(date)]"
  echo "[eval start $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_best runs/merged_k16
  echo "[SWEEP DONE $(date)]"
} 2>&1 | tee "$LOG"

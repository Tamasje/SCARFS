#!/usr/bin/env bash
# PHASE 2: Stage-B (freeze-encoder + pushforward) on the k=6 residual model -> low-k stable transport.
# Run: bash scripts/run_nlenc_k6_stageB.sh   (~30-40 min, caffeinated, logs runs/nlenc_k6_stageB.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[K6 STAGEB START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/train_merged_nlenc_k6_stageB.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI (k6_stageB vs k6_stageA vs contract090) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_nlenc_k6_stageB runs/merged_nlenc_k6 runs/merged_contract090
  echo "[PROJECTION GAIN (k6_stageB) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_nlenc_k6_stageB
  echo "[A-POSTERIORI 0D (k6_stageB) — decisive $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_nlenc_k6_stageB --n-cases 30 --substeps 1,50
  echo "[K6 STAGEB DONE $(date)]"
} 2>&1 | tee runs/nlenc_k6_stageB.log

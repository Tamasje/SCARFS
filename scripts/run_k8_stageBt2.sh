#!/usr/bin/env bash
# k=8 Stage-B2 refinement (tighter contraction + longer pushforward horizon) to clean up G_F<1 + energy.
# Run: bash scripts/run_k8_stageBt2.sh   (~50-70 min, caffeinated, logs runs/k8_stageBt2.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[K8 STAGEBt2 START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/train_merged_nlenc_k8_stageBt2.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI (k8_t2 vs k8_t vs k8_stageA) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_nlenc_k8_stageBt2 runs/merged_nlenc_k8_stageBt runs/merged_nlenc_k8
  echo "[PROJECTION GAIN (k8_t2) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_nlenc_k8_stageBt2
  echo "[A-POSTERIORI 0D (k8_t2) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_nlenc_k8_stageBt2 --n-cases 30 --substeps 1,50
  echo "[K8 STAGEBt2 DONE $(date)]"
} 2>&1 | tee runs/k8_stageBt2.log

#!/usr/bin/env bash
# ADAPTATION #1 — contraction + species-space pushforward, trained JOINTLY (closed-loop tracking).
# Run from repo root:  bash scripts/run_joint_retrain.sh   (~2-2.5 h, caffeinated, logs to runs/joint.log)
# Reads configs/train_merged_joint.json (contraction gain 0.9 w 0.5 + pushforward w 0.5 K 8) -> runs/merged_joint.
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[JOINT START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/train_merged_joint.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI (joint vs contract090 vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_joint runs/merged_contract090 runs/merged_best
  echo "[PROJECTION GAIN (joint) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_joint
  echo "[A-POSTERIORI 0D (joint) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_joint --n-cases 30 --substeps 1,50
  echo "[A-POSTERIORI 1D (joint) $(date)]"
  $CAF $PY scripts/pfr_1d_diffusion.py runs/merged_joint --n-cases 12 --D 0,1e-4,1e-2,1
  echo "[JOINT DONE $(date)]"
} 2>&1 | tee runs/joint.log

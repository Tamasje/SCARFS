#!/usr/bin/env bash
# DE-RISK: staged freeze-encoder + pushforward (does staging escape the Pareto wall?).
# Warm-starts runs/merged_contract090, FREEZES the encoder, fine-tunes decoder+omega_Z with
# contraction + species-space pushforward + a-priori anchors. Run: bash scripts/run_destageB.sh
# ~30-40 min (120-epoch fine-tune + eval), caffeinated, logs to runs/destageB.log.
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[DESTAGEB START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/train_merged_destageB.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI (destageB vs contract090 vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_destageB runs/merged_contract090 runs/merged_best
  echo "[PROJECTION GAIN (destageB) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_destageB
  echo "[A-POSTERIORI 0D (destageB) — the decisive gate $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_destageB --n-cases 30 --substeps 1,50
  echo "[DESTAGEB DONE $(date)]"
} 2>&1 | tee runs/destageB.log

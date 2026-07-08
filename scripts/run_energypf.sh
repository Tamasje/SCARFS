#!/usr/bin/env bash
# ENERGY-AWARE pushforward retrain: does matching rate-derived S_E in the closed-loop rollout reduce
# the a-posteriori integrated-energy error (∫S_E ~0.25 on the stageBt2 deployment model)?
# Warm-starts runs/merged_nlenc_k8_stageBt2, FREEZES the encoder, adds pushforward_energy_weight
# (w=1.0 and w=2.0 to bracket). Only NEW variable vs the deployment model = the energy term.
# Run: bash scripts/run_energypf.sh   (~1.5 h for 2 runs + eval, caffeinated, logs to runs/energypf.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
BASE=runs/merged_nlenc_k8_stageBt2
{
  echo "[ENERGYPF START $(date)]"
  for w in w1 w2; do
    echo "[TRAIN energypf ${w} $(date)] -> runs/merged_k8_energypf_${w}"
    $CAF $PY -m scarfs.training.train --config configs/train_merged_k8_energypf_${w}.json
    echo "[TRAIN energypf ${w} END exit=$? $(date)]"
  done

  echo "[A-PRIORI compare (energypf w1/w2 vs stageBt2 deployment) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_k8_energypf_w1 runs/merged_k8_energypf_w2 $BASE

  for m in runs/merged_k8_energypf_w1 runs/merged_k8_energypf_w2 $BASE; do
    echo "[PROJECTION GAIN (${m}) $(date)]"
    $CAF $PY scripts/diag_projection_gain.py $m
    echo "[A-POSTERIORI 0D — the decisive ∫S_E gate (${m}) $(date)]"
    $CAF $PY scripts/aposteriori_rollout.py $m --n-cases 30 --substeps 1,50
  done
  echo "[ENERGYPF DONE $(date)]"
} 2>&1 | tee runs/energypf.log

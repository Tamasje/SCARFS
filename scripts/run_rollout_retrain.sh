#!/usr/bin/env bash
# Re-train k=32 with the Lagrangian rollout loss (closed-loop stability fix), then evaluate BOTH:
#   - a-priori regression: held-out TEST energy relRMSE (new vs current best) — did accuracy survive?
#   - a-posteriori stability: latent rollout drift (new model) — did the loop stabilize?
#
# Run from repo root:  bash scripts/run_rollout_retrain.sh
# ~1 h training (800 ep, ~4.4 s/ep, early-stops on patience) + ~10 min eval. caffeinated; live progress.
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
{
  echo "[ROLLOUT RETRAIN START $(date)]"
  echo "[train start $(date)] -> runs/merged_rollout"
  $CAF $PY -m scarfs.training.train --config configs/_merged_rollout.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI test eval (rollout vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_rollout runs/merged_best
  echo "[A-POSTERIORI stability (rollout model) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_rollout --n-cases 30 --substeps 1,50
  echo "[ROLLOUT RETRAIN DONE $(date)]"
} 2>&1 | tee runs/rollout_retrain.log

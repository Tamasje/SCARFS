#!/usr/bin/env bash
# Stage-1 deeper stability fix: retrain k=32 with spectral_norm=True (contractive latent map:
# decoder + heads Lipschitz-bounded), then evaluate stability + a-priori regression.
# Run from repo root:  bash scripts/run_specnorm_retrain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
{
  echo "[SPECNORM RETRAIN START $(date)]"
  echo "[train start $(date)] -> runs/merged_specnorm"
  $CAF $PY -m scarfs.training.train --config configs/_merged_specnorm.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI test eval (specnorm vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_specnorm runs/merged_best
  echo "[A-POSTERIORI 0D stability (specnorm) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_specnorm --n-cases 30 --substeps 1,50
  echo "[A-POSTERIORI 1D diffusion (specnorm) $(date)]"
  $CAF $PY scripts/pfr_1d_diffusion.py runs/merged_specnorm --n-cases 12
  echo "[SPECNORM RETRAIN DONE $(date)]"
} 2>&1 | tee runs/specnorm_retrain.log

#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
{
  echo "[CONTRACT097 START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/_merged_contract097.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI (contract097 vs contract vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_contract097 runs/merged_contract runs/merged_best
  echo "[PROJECTION GAIN $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_contract097
  echo "[A-POSTERIORI 0D $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_contract097 --n-cases 30 --substeps 1,50
  echo "[A-POSTERIORI 1D $(date)]"
  $CAF $PY scripts/pfr_1d_diffusion.py runs/merged_contract097 --n-cases 12 --D 0,1e-4,1e-2,1
  echo "[CONTRACT097 DONE $(date)]"
} 2>&1 | tee runs/contract097.log

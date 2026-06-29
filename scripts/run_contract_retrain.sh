#!/usr/bin/env bash
# Contractive-by-design retrain + full stability/accuracy eval. Run: bash scripts/run_contract_retrain.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
{
  echo "[CONTRACT RETRAIN START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/_merged_contract.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI test (contract vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_contract runs/merged_best
  echo "[PROJECTION GAIN (contract) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_contract
  echo "[A-POSTERIORI 0D (contract) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_contract --n-cases 30 --substeps 1,50
  echo "[A-POSTERIORI 1D (contract) $(date)]"
  $CAF $PY scripts/pfr_1d_diffusion.py runs/merged_contract --n-cases 12 --D 0,1e-4,1e-2,1
  echo "[CONTRACT RETRAIN DONE $(date)]"
} 2>&1 | tee runs/contract_retrain.log

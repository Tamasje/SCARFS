#!/usr/bin/env bash
# ADAPTATION #2 — slow-manifold latent: anisotropic (transverse/fast-mode) contraction + neutral
# isotropic cap, so the fast directions are slaved to the manifold (Layer-2 attraction) while the
# slow along-trajectory dynamics stay free. Geometric only — no a-priori-competing trajectory loss.
# Run from repo root:  bash scripts/run_slowmanifold_retrain.sh   (~1.5-2 h, caffeinated, logs runs/slowmanifold.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[SLOWMANIFOLD START $(date)]"
  $CAF $PY -m scarfs.training.train --config configs/train_merged_slowmanifold.json
  echo "[train end exit=$? $(date)]"
  echo "[A-PRIORI (slowmanifold vs contract090 vs best) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_slowmanifold runs/merged_contract090 runs/merged_best
  echo "[PROJECTION GAIN (slowmanifold) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/merged_slowmanifold
  echo "[A-POSTERIORI 0D (slowmanifold) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/merged_slowmanifold --n-cases 30 --substeps 1,50
  echo "[A-POSTERIORI 1D (slowmanifold) $(date)]"
  $CAF $PY scripts/pfr_1d_diffusion.py runs/merged_slowmanifold --n-cases 12 --D 0,1e-4,1e-2,1
  echo "[SLOWMANIFOLD DONE $(date)]"
} 2>&1 | tee runs/slowmanifold.log

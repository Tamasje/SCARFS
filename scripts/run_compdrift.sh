#!/usr/bin/env bash
# ATTACK COMPOSITION DRIFT — the real ∫S_E bottleneck (energy-pushforward negative result showed the
# rollout energy is capped by composition drift ~2.3e-2, not energy-head fidelity). All 3 warm-start
# the deployment model stageBt2, FREEZE the encoder, and checkpoint on rollout_composition (closed-loop
# major-species drift) so we SELECT the lowest-drift epoch — which a-priori metrics cannot see.
#   Exp1 rollckpt      : stageBt2 recipe, ONLY the new checkpoint (isolates selection effect)
#   Exp2 rollckpt_slow : + slow-manifold anisotropic contraction (cut transverse/off-manifold drift)
#   Exp3 rollckpt_pf   : + stronger/longer pushforward 1.5/K24 (harder closed-loop tracking)
# Run: bash scripts/run_compdrift.sh   (~1.5-2 h, caffeinated, logs to runs/compdrift.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
BASE=runs/merged_nlenc_k8_stageBt2
{
  echo "[COMPDRIFT START $(date)]"
  for cfg in rollckpt rollckpt_slow rollckpt_pf; do
    echo "[TRAIN ${cfg} $(date)] -> runs/merged_k8_${cfg}"
    $CAF $PY -m scarfs.training.train --config configs/train_merged_k8_${cfg}.json
    echo "[TRAIN ${cfg} END exit=$? $(date)]"
  done

  echo "[A-PRIORI compare (rollckpt variants vs stageBt2) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_k8_rollckpt runs/merged_k8_rollckpt_slow runs/merged_k8_rollckpt_pf $BASE

  for m in runs/merged_k8_rollckpt runs/merged_k8_rollckpt_slow runs/merged_k8_rollckpt_pf $BASE; do
    echo "[PROJECTION GAIN (${m}) $(date)]"
    $CAF $PY scripts/diag_projection_gain.py $m
    echo "[A-POSTERIORI 0D — composition drift + ∫S_E (${m}) $(date)]"
    $CAF $PY scripts/aposteriori_rollout.py $m --n-cases 30 --substeps 1,50
  done
  echo "[COMPDRIFT DONE $(date)]"
} 2>&1 | tee runs/compdrift.log

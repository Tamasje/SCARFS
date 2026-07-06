#!/usr/bin/env bash
# NONLINEAR-ENCODER a-priori k-sweep (Phase 1 of the redesign): does an MLP encoder let a LOW-k latent
# hold the a-priori accuracy that the linear encoder needs k=32 for? Stage-A (a-priori) training only.
# Run: bash scripts/run_nlenc_sweep.sh   (~3-4 h for 3 k's, caffeinated, logs to runs/nlenc_sweep.log)
# Ordered k=6 (target) first so its result appears early. Compares to linear k=32 (7.75x) / k=16 (5.30x).
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[NLENC SWEEP START $(date)]"
  # generate Stage-A nonlinear-encoder configs (MLP encoder [96], no pushforward/contraction)
  $PY - <<'PYEOF'
import json, copy
base = json.load(open('configs/train_merged_best.json')); base.pop('_comment', None)
for k in (6, 4, 8):
    c = copy.deepcopy(base)
    c['model']['encoder_hidden'] = [96]
    c['model']['latent_dim'] = k
    c['loss']['consistency_weight'] = 0.0   # linear-encoder-weight term; N/A for MLP encoder
    c['output_dir'] = f"runs/merged_nlenc_k{k}"
    json.dump(c, open(f"configs/_nlenc_k{k}.json", "w"), indent=2)
    print(f"  wrote configs/_nlenc_k{k}.json (encoder MLP [96], k={k}, Stage-A a-priori)")
PYEOF
  for k in 6 4 8; do
    echo "[TRAIN nlenc k=${k} $(date)] -> runs/merged_nlenc_k${k}"
    $CAF $PY -m scarfs.training.train --config configs/_nlenc_k${k}.json
    echo "[TRAIN nlenc k=${k} END exit=$? $(date)]"
  done
  echo "[A-PRIORI compare (nonlinear k6/k4/k8 vs linear k32/k16) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_nlenc_k6 runs/merged_nlenc_k4 runs/merged_nlenc_k8 \
        runs/merged_contract090 runs/merged_best
  echo "[NLENC SWEEP DONE $(date)]"
} 2>&1 | tee runs/nlenc_sweep.log

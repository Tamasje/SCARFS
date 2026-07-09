#!/usr/bin/env bash
# Tune the low-k operating point: Stage-B with STRONGER/tighter contraction (w=2.0, gain=0.8) + longer,
# for k=6 (from merged_nlenc_k6) and k=8 (from merged_nlenc_k8). Pick the cleanest G_F<1 + a-priori + tracking.
# Run: bash scripts/run_lowk_stageB_tune.sh   (~1-1.5 h, caffeinated, logs runs/lowk_stageB_tune.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[LOWK TUNE START $(date)]"
  $PY - <<'PYEOF'
import json, copy
base = json.load(open('configs/train_merged_best.json')); base.pop('_comment', None)
for k, src in [(6, 'runs/merged_nlenc_k6'), (8, 'runs/merged_nlenc_k8')]:
    c = copy.deepcopy(base)
    c['model']['encoder_hidden'] = [96]; c['model']['latent_dim'] = k
    c['loss'].update({'consistency_weight':0.0,'contraction_weight':2.0,'contraction_gain':0.8,
                      'pushforward_weight':0.5,'pushforward_steps':8})
    c['optim'].update({'init_from':src,'freeze_encoder':True,'lr':3e-4,'epochs':200,'patience':60,
                       'warmup_epochs':5,'head_finetune_epochs':30,'lr_schedule':'cosine','checkpoint_metric':'energy_relrmse'})
    c['output_dir'] = f"runs/merged_nlenc_k{k}_stageBt"
    json.dump(c, open(f"configs/_nlenc_k{k}_stageBt.json", "w"), indent=2)
    print(f"  wrote k={k} tuned Stage-B (contraction w2.0 gain0.8, 200ep) <- {src}")
PYEOF
  for k in 6 8; do
    echo "[TRAIN k${k} tuned Stage-B $(date)]"
    $CAF $PY -m scarfs.training.train --config configs/_nlenc_k${k}_stageBt.json
    echo "[TRAIN k${k} END exit=$? $(date)]"
  done
  echo "[A-PRIORI (k6t / k8t vs Stage-A k6/k8 vs contract090) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/merged_nlenc_k6_stageBt runs/merged_nlenc_k8_stageBt \
        runs/merged_nlenc_k6 runs/merged_nlenc_k8 runs/merged_contract090
  for k in 6 8; do
    echo "[PROJECTION GAIN k${k}t $(date)]"; $CAF $PY scripts/diag_projection_gain.py runs/merged_nlenc_k${k}_stageBt
    echo "[A-POSTERIORI 0D k${k}t $(date)]"; $CAF $PY scripts/aposteriori_rollout.py runs/merged_nlenc_k${k}_stageBt --n-cases 30 --substeps 1,50
  done
  echo "[LOWK TUNE DONE $(date)]"
} 2>&1 | tee runs/lowk_stageB_tune.log

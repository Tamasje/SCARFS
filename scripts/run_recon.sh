#!/usr/bin/env bash
# ATTACK THE RECONSTRUCTION FLOOR at fixed k=8 (respects low-k). Diagnosis proved the closed-loop
# drift that caps ∫S_E EQUALS the autoencoder reconstruction floor decode(encode(Y)), which is FLAT
# vs k (8→16) — so the bottleneck is the DECODER capacity + training balance, not the latent width.
# Lever: bigger decoder [128,256]->[256,512] + reconstruction-prioritised training (recon_weight 1->5,
# qoi_recon 0->2 on the major species). Still 8 UDS transported; only per-cell compute grows.
#   Stage-A: recon-prioritised a-priori (residual enc [96], k=8, big decoder)
#   Stage-B: warm-start, FREEZE encoder, pushforward(0.5/K16)+contraction(2.5/0.75), keep recon high
# Run: bash scripts/run_recon.sh   (~1.5 h, caffeinated, logs runs/recon.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
BASE=runs/merged_nlenc_k8_stageBt2
{
  echo "[RECON START $(date)]"
  $PY - <<'PYEOF'
import json, copy
A=json.load(open('configs/train_merged_best.json')); A.pop('_comment',None)
B=json.load(open('configs/train_merged_nlenc_k8_stageBt2.json')); B.pop('_comment',None)
a=copy.deepcopy(A)
a['model']['encoder_hidden']=[96]; a['model']['latent_dim']=8; a['model']['decoder_hidden']=[256,512]
a['loss']['consistency_weight']=0.0; a['loss']['recon_weight']=5.0; a['loss']['qoi_recon_weight']=2.0
a['optim']['epochs']=800; a['optim']['patience']=150
a['output_dir']='runs/recon_k8_A'
json.dump(a, open('configs/_recon_k8_A.json','w'), indent=2); print("wrote _recon_k8_A.json (big decoder + recon-prioritised)")
b=copy.deepcopy(B)
b['model']['latent_dim']=8; b['model']['encoder_hidden']=[96]; b['model']['decoder_hidden']=[256,512]
b['loss']['recon_weight']=5.0; b['loss']['qoi_recon_weight']=2.0
b['optim']['init_from']='runs/recon_k8_A'; b['optim']['freeze_encoder']=True
b['optim']['epochs']=120; b['optim']['patience']=50; b['optim']['lr']=0.0002; b['optim']['checkpoint_metric']='energy_relrmse'
b['output_dir']='runs/recon_k8_B'
json.dump(b, open('configs/_recon_k8_B.json','w'), indent=2); print("wrote _recon_k8_B.json (Stage-B, keep recon high)")
PYEOF

  echo "[TRAIN Stage-A recon $(date)] -> runs/recon_k8_A"
  $CAF $PY -m scarfs.training.train --config configs/_recon_k8_A.json
  echo "[TRAIN Stage-A recon END exit=$? $(date)]"
  echo "[TRAIN Stage-B recon $(date)] -> runs/recon_k8_B"
  $CAF $PY -m scarfs.training.train --config configs/_recon_k8_B.json
  echo "[TRAIN Stage-B recon END exit=$? $(date)]"

  echo "[RECONSTRUCTION FLOOR (recon_k8_B vs stageBt2) $(date)]"
  $CAF $PY scripts/diag_reconstruction.py runs/recon_k8_B $BASE
  echo "[A-PRIORI (recon_k8_B vs stageBt2) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/recon_k8_B $BASE
  echo "[PROJECTION GAIN (recon_k8_B) $(date)]"
  $CAF $PY scripts/diag_projection_gain.py runs/recon_k8_B
  echo "[A-POSTERIORI 0D — composition drift + ∫S_E (recon_k8_B) $(date)]"
  $CAF $PY scripts/aposteriori_rollout.py runs/recon_k8_B --n-cases 40 --substeps 1,50
  echo "[RECON DONE $(date)]"
} 2>&1 | tee runs/recon.log

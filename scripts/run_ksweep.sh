#!/usr/bin/env bash
# k-SWEEP: map the closed-loop FLOOR (composition drift, ∫S_E) and a-priori factor vs latent dim k,
# to pick the accuracy-vs-UDS-cost knee deliberately. The energy-pushforward + composition-drift
# campaigns proved ∫S_E~0.25 is a hard k=8 floor set by the k=8 MANIFOLD; this measures how the floor
# moves with k (the actual bottleneck). ONE consistent 2-stage recipe per k:
#   Stage-A (a-priori): residual encoder [96], cold-start (k=8 reuses runs/merged_nlenc_k8).
#   Stage-B (closed-loop): warm-start Stage-A, FREEZE encoder, pushforward(0.5/K16)+contraction(2.5/0.75).
# UDS cost ≈ 5 flow eqns + k  →  k=8:~2.6x  k=10:~2.9x  k=12:~3.2x  k=16:~3.8x  base CFD cost.
# Run: bash scripts/run_ksweep.sh   (~4.5 h: 3 Stage-A @800/150 + 4 Stage-B + eval; caffeinated; logs runs/ksweep.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
{
  echo "[KSWEEP START $(date)]"

  # --- generate Stage-A (k=10,12,16) and Stage-B (k=8,10,12,16) configs ---
  $PY - <<'PYEOF'
import json, copy
A = json.load(open('configs/train_merged_best.json')); A.pop('_comment', None)
B = json.load(open('configs/train_merged_nlenc_k8_stageBt2.json')); B.pop('_comment', None)
STAGEA = {8: "runs/merged_nlenc_k8"}   # reuse existing k=8 residual-encoder Stage-A
for k in (10, 12, 16):
    a = copy.deepcopy(A)
    a['model']['encoder_hidden'] = [96]
    a['model']['latent_dim'] = k
    a['loss']['consistency_weight'] = 0.0          # linear-encoder-weight term; N/A for residual encoder
    # MATCH merged_nlenc_k8's recipe EXACTLY (epochs 800 / patience 150 / cosine over 800) so the k=8
    # reuse is legitimate and all four k's converge under identical optimisation — else higher-k Stage-A
    # under-trains vs the reused k=8 and biases the floor comparison against higher k.
    a['optim']['epochs'] = 800; a['optim']['patience'] = 150
    a['output_dir'] = f"runs/sweep_k{k}_A"
    json.dump(a, open(f"configs/_sweep_k{k}_A.json", "w"), indent=2)
    STAGEA[k] = a['output_dir']
    print(f"  wrote _sweep_k{k}_A.json (Stage-A a-priori, residual enc, k={k})")
for k in (8, 10, 12, 16):
    b = copy.deepcopy(B)
    b['model']['latent_dim'] = k                   # encoder_hidden already [96]
    b['optim']['init_from'] = STAGEA[k]
    b['optim']['freeze_encoder'] = True
    b['optim']['epochs'] = 120; b['optim']['patience'] = 50; b['optim']['lr'] = 0.0002
    b['optim']['checkpoint_metric'] = "energy_relrmse"   # reliable deploy metric (rollout-comp didn't transfer)
    b['output_dir'] = f"runs/sweep_k{k}_B"
    json.dump(b, open(f"configs/_sweep_k{k}_B.json", "w"), indent=2)
    print(f"  wrote _sweep_k{k}_B.json (Stage-B closed-loop, k={k}, init_from={STAGEA[k]})")
PYEOF

  # --- Stage-A training (k=10,12,16; k=8 reused) ---
  for k in 10 12 16; do
    echo "[TRAIN Stage-A k=${k} $(date)] -> runs/sweep_k${k}_A"
    $CAF $PY -m scarfs.training.train --config configs/_sweep_k${k}_A.json
    echo "[TRAIN Stage-A k=${k} END exit=$? $(date)]"
  done

  # --- Stage-B training (all k) ---
  for k in 8 10 12 16; do
    echo "[TRAIN Stage-B k=${k} $(date)] -> runs/sweep_k${k}_B"
    $CAF $PY -m scarfs.training.train --config configs/_sweep_k${k}_B.json
    echo "[TRAIN Stage-B k=${k} END exit=$? $(date)]"
  done

  # --- evaluate the deployed floor vs k ---
  echo "[A-PRIORI compare (sweep Stage-B k=8/10/12/16) $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/sweep_k8_B runs/sweep_k10_B runs/sweep_k12_B runs/sweep_k16_B
  for k in 8 10 12 16; do
    echo "[PROJECTION GAIN (k=${k}) $(date)]"
    $CAF $PY scripts/diag_projection_gain.py runs/sweep_k${k}_B
    echo "[A-POSTERIORI 0D — composition drift + ∫S_E (k=${k}) $(date)]"
    $CAF $PY scripts/aposteriori_rollout.py runs/sweep_k${k}_B --n-cases 30 --substeps 1,50
  done
  echo "[KSWEEP DONE $(date)]"
} 2>&1 | tee runs/ksweep.log

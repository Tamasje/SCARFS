#!/usr/bin/env bash
# RESOLVE THE STABILITY-FIDELITY TENSION. recon_k8_A (big decoder + recon-priority) reaches a 5.3e-2
# reconstruction floor (−43% vs stageBt2) but Stage-B's STRONG contraction (2.5) flattens the decoder
# and reverts it to ~9e-2. Test whether WEAK contraction preserves the reconstruction gain while still
# reaching G_F<1 (closed-loop stable). Reuses recon_k8_A (exists) → Stage-B only, cheap.
#   Bc05 : contraction 0.5   Bc10 : contraction 1.0   (vs stageBt2's 2.5)
# Gate: reconstruction < 9e-2 AND G_F < 1 AND ∫S_E < stageBt2's ~0.25.
# Run: bash scripts/run_weakcontract.sh   (~1.5 h, caffeinated, logs runs/weakcontract.log)
set -uo pipefail
cd "$(dirname "$0")/.."
export SCARFS_PROGRESS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python; CAF=""; command -v caffeinate >/dev/null 2>&1 && CAF="caffeinate -i"
[ -e Database_FINAL.parquet ] || ln -sf full.parquet Database_FINAL.parquet
BASE=runs/merged_nlenc_k8_stageBt2
{
  echo "[WEAKCONTRACT START $(date)]"
  $PY - <<'PYEOF'
import json, copy
B=json.load(open('configs/_recon_k8_B.json'))   # big decoder, recon 5, qoi 2, init_from recon_k8_A, frozen enc
for tag,cw in [("c05",0.5),("c10",1.0)]:
    b=copy.deepcopy(B); b['loss']['contraction_weight']=cw
    b['output_dir']=f"runs/recon_k8_B{tag}"
    b['_comment']=f"Stage-B from recon_k8_A with WEAK contraction {cw} (vs 2.5) — preserve reconstruction while staying G_F<1."
    json.dump(b, open(f"configs/_recon_k8_B{tag}.json","w"), indent=2)
    print(f"  wrote _recon_k8_B{tag}.json (contraction={cw})")
PYEOF
  for tag in c05 c10; do
    echo "[TRAIN Stage-B ${tag} $(date)] -> runs/recon_k8_B${tag}"
    $CAF $PY -m scarfs.training.train --config configs/_recon_k8_B${tag}.json
    echo "[TRAIN Stage-B ${tag} END exit=$? $(date)]"
  done
  echo "[RECONSTRUCTION FLOOR (weak-contraction variants vs stageBt2) $(date)]"
  $CAF $PY scripts/diag_reconstruction.py runs/recon_k8_Bc05 runs/recon_k8_Bc10 $BASE
  echo "[A-PRIORI $(date)]"
  $CAF $PY scripts/full_test_eval.py runs/recon_k8_Bc05 runs/recon_k8_Bc10 $BASE
  for m in runs/recon_k8_Bc05 runs/recon_k8_Bc10; do
    echo "[PROJECTION GAIN (${m}) $(date)]"
    $CAF $PY scripts/diag_projection_gain.py $m
    echo "[A-POSTERIORI 0D — composition drift + ∫S_E (${m}) $(date)]"
    $CAF $PY scripts/aposteriori_rollout.py $m --n-cases 40 --substeps 1,50
  done
  echo "[WEAKCONTRACT DONE $(date)]"
} 2>&1 | tee runs/weakcontract.log

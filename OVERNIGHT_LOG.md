# OVERNIGHT LOG — append-only experiment ledger (overnight-diagnosis-20260612)

Format per entry: id · hypothesis · command(s) · data used · result numbers · verdict.
Training-experiment count is tracked against the 12-run cap; analyses (no training) are unnumbered-A.

---

## E0-A · verify the lead (no training)
- Command: `.venv/bin/python - <<EOF` parse of `runs/merged_bootstrap_stride5/metrics.json`
  (inline script, logged in transcript; epochs 0–7, 15, 30, 45, 62).
- Data: the existing bootstrap run's own metrics (stride5 re-split, 63 epochs, k=8).
- Result: best val total 48.755 @ epoch 2; val latent_source 53.87→50.94 (flat, weight 1.0);
  val consistency 53.0→49.3 (flat, weight 0.02); val rate 0.6697→0.0860; val energy_direct
  1.9329→0.0473; val absorption R² 0.708 (rate-derived) / 0.231 (head).
- Verdict: lead CONFIRMED as stated. The two flat terms are exactly the two consumers of
  `arcsinh_latent_scale`.

---

## E1-A · H1 target-statistics audit (no training)
- Command: `.venv/bin/python scripts/overnight/e1_target_stats.py` (defaults).
- Data: stride5 val split of the bootstrap bundle (14,388 rows / 670 cases), bundle read-only.
- Result: s_Z saved is the latent-STATE scale (≈median|z|, 0.013–2.1) while median|ż| is
  13.6–78 → mis-scale ratio 10–1075× per dim. Var(target | saved s_Z) mean = **49.996** vs
  observed stuck val latent_source ≈ 50.9 → the loss sits AT the predict-the-mean floor.
  Model per-dim R² in target space: 0.007–0.095 (≈ nothing learned). Corrected s_Z* =
  median|ż| gives Var mean 15.8, 51% of rows in the |t|<1 linear region.
- Verdict: H1 floor fingerprint CONFIRMED (units bug real; consistency term shares it).

## E2/E3 · training exp 1–2 · decisive single-batch overfit (latent_source only)
- Command: `.venv/bin/python scripts/overnight/e2_single_batch.py` (defaults: 400 steps, lr 1e-3,
  batch 2048 seeded from the train split, fresh PCA-init model, loss exactly as training:
  head reads z_proj).
- Result: E2 (saved s_Z) 65.6→42.5, final/Var = 0.757; E3 (corrected) 14.6→10.1, final/Var = 0.711.
- Verdict: NEITHER memorizes → per the brief's tree, not a pure scale fix. Two confounds
  identified: z_proj passes through an UNTRAINED decoder when recon/manifold are off
  (experiment artifact), and 400 steps may undertrain → E2b/E3b/E4b.

## E2b/E3b/E4b · training exp 3–5 · overfit with confounds removed
- Command: `.venv/bin/python scripts/overnight/e2_single_batch.py --steps 2400 --lr 2e-3`
  (adds: z-direct head input; FULL-input bypass = plain MLP[256,256] on (x_std, q)).
- Result (final/Var): saved+z-direct 0.560; corrected+z-direct **0.432**; corrected+FULL-input
  **0.155 and still descending** (2.19 absolute from 14.76).
- Verdict: (i) corrected s_Z consistently learns more at equal compute → H1 real, secondary.
  (ii) NEW DOMINANT CAUSE **H7 [CONFIRMED in-sample]**: the k=8 variance-PCA latent does not
  carry the information that determines ż — (z,q)→ż caps at ~0.55–0.6 explained variance ON
  THE TRAINING BATCH ITSELF, while the full 212-dim composition memorizes the same target.
  The ω_Z closure is input-information-limited at k=8, not optimization- or capacity-limited.

---

## FIX-1 · s_Z source-based freeze (Phase 4, applied)
- Change: `scarfs/training/train.py::freeze_latent_arcsinh_scale` now freezes
  s_Z,i = median|E·(Ẏ⊘σ)|_i (the SOURCE distribution, ArcsinhScaler convention) instead of
  median|E·x|_i (the STATE); legacy call signature falls back with a loud warning. Call site
  moved after the dYdt block. Regression test added:
  `tests/test_training_merged.py::test_latent_arcsinh_scale_uses_source_not_state`.
- Verification: targeted tests 26 passed (incl. end-to-end integration).

## E5 · training exp 6 · 14-epoch full train, fix @ k=8
- Command: `.venv/bin/python -m scarfs.training.train --config runs/overnight_e5_cfg.json`
  (= bootstrap cfg + epochs/patience 14, out runs/overnight_e5_k8_fix).
- Data: stride5 re-split (same seed 0 splits as baseline).
- Result: saved s_Z now 4.3–103 (source units ✓). Val latent_source ≈ 15.40 FLAT vs
  corrected-target Var 15.8 (E1) → still at the mean floor in the new units; val rate 0.111;
  val absorption R² 0.668 (rate-derived) / −0.05 (head) at 14 epochs.
- Verdict: fix corrects units/conditioning (floor 50→15.8 by construction; pre-registered
  caveat: the ≥25% numeric drop is a UNIT change, not learning). Val skill at k=8/14 epochs
  ≈ 0 → consistent with the H7 information ceiling; k-comparison (E6) is the discriminator.

---

## E6 · training exp 7 · 14-epoch full train, fix @ k=16 (config-only change vs E5)
- Command: `.venv/bin/python -m scarfs.training.train --config runs/overnight_e6_cfg.json`.
- Result: val latent_source 14.89 vs own-target Var 14.01 (audit cmd:
  `scripts/overnight/e1_target_stats.py --bundle runs/overnight_e6_k16_fix --rows-cap 8000`)
  → MSE/Var ≈ 0.96, still ≈ mean floor at this budget. BUT val absorption R²
  **0.808 rate-derived / 0.307 head** — best of any run, at 14 epochs (k=8 14-ep: 0.668;
  k=8 63-ep baseline: 0.708).
- Verdict: (i) latent-head skill needs optimizer budget far beyond 14 shared epochs
  (single-batch needed 2400 DEDICATED steps to reach 0.43·Var) — optimization budget
  [CONFIRMED secondary factor] stacked on the H7 ceiling; (ii) k=16 materially improves the
  CFD-relevant downstream (absorption) — consistent with the 2026-06-11 feasibility table
  (PCA-12/16 PASS). E7 (60-epoch k=16) launched as the final discriminator/confirmation.

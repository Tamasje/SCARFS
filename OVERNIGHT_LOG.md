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

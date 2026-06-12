# OVERNIGHT REPORT — why `latent_source` won't learn (merged surrogate)

Branch `overnight-diagnosis-20260612` · started 2026-06-12 07:57 CEST · driver: Claude (unattended)
Ledger: [`OVERNIGHT_LOG.md`](OVERNIGHT_LOG.md) — every number below has a logged, reproducible command.
Goal tracking: the brief's /goal criteria are tracked manually in this report (no goal-mode tooling
available in this session); caps = 8 h wall / 12 training experiments, honored.

## The lead (VERIFIED against `runs/merged_bootstrap_stride5/metrics.json`, exp E0)

- Best val total **48.755 at epoch 2** of 63; val `latent_source` is **flat 50–54 the entire run**
  (53.87 → 50.94), weight 1.0 → ≈96% of the objective.
- `consistency` is identically flat (~53 → 49.3). These are the ONLY two terms that share the frozen
  `arcsinh_latent_scale` (s_Z). Every other term trains: rate 0.67→0.086, energy_direct 1.93→0.047,
  recon/manifold < 1.
- Val absorption R²: 0.708 (rate-derived) / 0.231 (head). The CFD-transported ω_Z head is the
  non-learner.

## Working hypothesis set (Phase 2 ranking — updated as experiments land)

- **H1 [primary, under test]: s_Z units bug.** `freeze_latent_arcsinh_scale`
  (`scarfs/training/train.py`) freezes s_Z from the latent **state** distribution
  `median|E·x_std|` (O(1–10)), but s_Z scales the latent **source** target
  `arcsinh(ż/s_Z)` with `ż = E·(Ẏ⊘σ)` (O(1e2–1e8), 1/timescale units, σ-amplified).
  A 3–6 orders-of-magnitude-too-small scale pushes every target onto the log branches:
  targets become a ±8–17 sign×log-magnitude field with a ~±20 cliff at every sign change —
  the exact "discontinuous magnitude+sign" pathology our own `ArcsinhScaler` docstring warns
  about. Predicting the per-dim mean of such a field gives MSE ≈ Var(target) ≈ the observed
  stuck ~50. Both stuck terms share s_Z; no healthy term uses it.
- **H2: per-dim conditioning** — even with a correct s_Z, one stiff latent dim could dominate
  the MSE. (Localize per-dim.)
- **H3: target noise floor** — σ-amplified trace-species dYdt may be solver noise; some dims
  may be unlearnable from (Z, q). (kNN achievable-R² audit.)
- **H4: optimization** — lr/clipping inadequate for this head. (Weak prior: the rate head —
  same architecture, same trunk inputs, LayerNorm — trains fine.)
- **H5: capacity/architecture** — (128,128) head too small / LayerNorm interaction. (Weak prior,
  same reason as H4.)
- **H6: moving target** — ż target uses the live (detached) encoder; E drifts. (Weak: the curve
  is flat, not oscillating; E is PCA-anchored by recon/manifold.)

## Plan

- P0 ✓ branch, report, ledger, pytest baseline.
- P1: (E1) no-training target-statistics audit — distribution/variance of arcsinh(ż/s_Z) per dim
  under the CURRENT s_Z vs a CORRECTED s_Z (median|ż| per dim); does Var(target) ≈ 50?
  (E2/E3) decisive single-batch overfit, latent_source only: current vs corrected s_Z.
  One read-only Sonnet subagent runs the independent data-quality audit (H3: per-dim kNN
  achievable-R², dYdt noise for trace species, sign-flip coverage) in parallel.
- P2: rank with evidence tags.
- P3: one-variable experiments per surviving hypothesis (short, seeded, capped; ledger per run).
- P4: if the fix is localized + behavior-preserving + test-passing and measurably reduces the val
  latent_source term → apply with a regression test; full pytest + sanity check; commit.
  ⚠ Honesty note recorded up front: changing s_Z changes the UNITS of the latent_source loss, so
  the raw before/after value is not by itself proof of learning. The report will therefore also
  show scale-invariant skill: per-dim R² vs predict-the-mean in target space, learning curves
  (flat vs descending), and downstream val absorption / QoI metrics.
- P5: finalize (root cause, FIXED-LOCALLY vs NEEDS-HPC, ranked next steps).

## Status

- Phase 0 complete (baseline suite result in ledger E0).
- Constraint honored: campaign-memory file lives outside the repo (OneDrive) and is NOT updated
  tonight per the stay-inside-the-repo guardrail — next session should copy the outcome there.

*(Sections below are appended as phases complete.)*

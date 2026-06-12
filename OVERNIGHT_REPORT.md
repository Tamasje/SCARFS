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

## Phase 1–3 findings (all numbers from logged runs; see ledger)

### Root cause is TWO stacked causes plus one budget factor

**RC-A [CONFIRMED] — s_Z units bug (H1), FIXED LOCALLY.** `freeze_latent_arcsinh_scale` froze
the latent-STATE scale median|E·x| (0.013–2.1) where the latent-SOURCE scale median|E·(Ẏ⊘σ)|
(13.6–78) was required — mis-scale 10–1075× per dim (E1-A). Fingerprint: Var(target | saved s_Z)
= 49.996 ≈ the stuck val loss 50.9; model per-dim R² in target space 0.007–0.095 — the head was
predicting the mean of a ±8–17 sign×log-magnitude cliff field. The ONLY two flat loss terms
(latent_source, consistency) are the only two consumers of s_Z. Fix: source-based freeze +
regression test (commit "Overnight FIX-1"). Pre-registered honesty caveat: the fix changes the
loss UNITS (floor 50→15.8 by construction); the numeric ≥25% reduction criterion is met
trivially and is NOT claimed as learning — skill metrics are reported instead.

**RC-B [CONFIRMED in-sample] — k=8 information ceiling for ż (H7, the dominant cause).**
Decisive overfit chain on one seeded 2048-row batch (E2→E4b), latent loss only:
saved-s_Z via z_proj 0.76 → corrected via z_proj 0.71 → corrected z-direct **0.43** →
corrected with the FULL 212-dim composition as input **0.155 and still descending**.
The same target an MLP memorizes from the full state is ~43–57%-unreachable from (z₈, T, P)
ON THE TRAINING BATCH ITSELF. ω_Z = E·(Ẏ⊘σ) mixes σ-amplified trace-species dynamics that a
variance-PCA k=8 projection simply does not carry — the transported-closure version of the
Phase-1 hidden-state finding. No rescaling, lr, or capacity change can cross an input-
information ceiling.

**RC-C [CONFIRMED, secondary] — optimizer budget for the latent head.** Reaching even the
in-sample ceiling took 2400 DEDICATED steps on one batch; a 14-epoch full train gives the head
~250 shared steps → E5/E6 still sit at the (corrected) mean floor on val. ω_Z skill needs
HPC-scale training; short CPU runs cannot demonstrate it either way.

### Hypothesis table (Phase 2 ranking, final)

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| H1 | s_Z scale/units bug | **[CONFIRMED — fixed]** | E1-A floor=Var fingerprint; E2b vs E3b at equal compute |
| H7 | k=8 input-information ceiling for ż | **[CONFIRMED in-sample — dominant]** | E3b 0.43 vs E4b 0.155 full-input |
| H-opt | optimizer budget (steps for the head) | **[CONFIRMED — secondary]** | 2400 dedicated steps for 0.43·Var vs ~250 shared in 14 ep |
| H2 | one stiff dim dominates the MSE | **[REFUTED as primary]** | E1-A per-dim Var ≈ uniform 36–69 under bad scale; 12–21 corrected |
| H3 | target noise floor (trace dYdt) | [PARTIAL — see data audit when it lands] | Sonnet audit pending at write time; full-input memorization (0.155) bounds noise ≤ ~15% in-sample |
| H4/H5 | optimization settings / capacity | **[REFUTED as primary]** | identical head+trunk trains the rate term 0.67→0.09; full-input MLP memorizes |
| H6 | moving (encoder-dependent) target | **[REFUTED as primary]** | curve flat, not oscillating; PCA-anchored E |

### What moved downstream (the CFD-relevant prize)

| run | k | epochs | val absorption R² (rate-derived / head) |
|---|---|---|---|
| baseline (mis-scaled s_Z) | 8 | 63 | 0.708 / 0.231 |
| E5 (fix) | 8 | 14 | 0.668 / −0.05 |
| E6 (fix) | 16 | 14 | **0.808 / 0.307** |
| E7 (fix, long) | 16 | 60 | *(running; appended below)* |

k=16 lifts the energy path well above the baseline at 4.5× fewer epochs — consistent with the
2026-06-11 stride5 feasibility table (PCA-12/16 PASS, PCA-8 FAIL). The rate/energy heads are NOT
input-limited the way ω_Z is (their targets are h-weighted big-species quantities that the PCA
state does carry).

## Status

- Phases 0–3 complete; FIX-1 applied + regression-tested; E7 (60-epoch k=16 confirm) running.
- Constraint honored: campaign-memory file lives outside the repo (OneDrive) and is NOT updated
  tonight per the stay-inside-the-repo guardrail — next session should copy the outcome there.

*(Final sections — E7, data-audit integration, recommendations — appended below.)*

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

**RC-B [CONFIRMED in-sample, then REFRAMED by the data audit] — apparent k=8 ceiling for ż.**
Decisive overfit chain on one seeded 2048-row batch (E2→E4b), latent loss only:
saved-s_Z via z_proj 0.76 → corrected via z_proj 0.71 → corrected z-direct **0.43** →
corrected with the FULL 212-dim composition as input **0.155 and still descending**.
The same target an MLP memorizes from the full state is ~43–57%-unreachable from (z₈, T, P)
on the training batch itself. The independent data audit (AUDIT-A) then localized WHY — and
it is not an intrinsic k-limit:

**RC-D [CONFIRMED] — σ-degenerate species poison the basis AND the target (the true dominant
cause).** 62 of 212 dry species have per-column std < 1e-10 on stride5 (3 nominally constant
incl. inert N₂ — float64 std ≈ 1e-15, so even the legacy ``std > 0`` guard amplifies them by
~1e15 — plus 59 trace radicals at σ 1e-17..1e-11 whose dYdt is 75–100% below 1e-12 1/s).
Standardization turns them into unit-variance solver-noise columns (~28% of the PCA input
budget; raw-PCA dim 1 is 100% inert N₂), and ż = E·(Ẏ⊘σ) amplifies their dYdt noise by up to
1e15. Measured ceiling of the ω_Z target from (z₈, T, P): **kNN OOF R² ≈ −0.10 polluted vs
0.82 mean / 0.85 tail with the 62 species de-activated** (per-dim 0.63–0.94; 0–1 sign flips
per trajectory; Var max/min 1.46). The chemistry is smooth and learnable; the formulation was
noise-bound. Fixed locally as FIX-2 (`sigma_floor`).

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

## Phase 4 outcome — the fixes work (E8/E9)

Floor-relative val latent_source (MSE / Var of each run's own target; 1.0 = predicting the mean;
figure: `runs/overnight_fig/latent_source_diagnosis.png`, regenerate with
`scripts/overnight/fig_diagnosis.py`):

| run | config | val latent (raw) | own floor (Var) | floor-relative | target-space R² | val absorption R² (rate-derived) |
|---|---|---|---|---|---|---|
| baseline | k=8, both bugs | 50.9 flat | 49.996 | **1.02 (floor)** | ~0.05 | 0.708 (63 ep) |
| E5 | k=8 + FIX-1 | 15.40 flat | 15.795 | 0.97 (floor) | ~0.03 | 0.668 (14 ep) |
| E6 | k=16 + FIX-1 | 14.89 flat | 14.007 | 1.06 (floor) | ~0.05 | 0.808 (14 ep) |
| E8 | k=16 + FIX-1+2 | **1.355 ↓** | 5.398 | **0.25** | **≈0.73** | 0.9405 (14 ep) |
| E9 | k=16 + FIX-1+2, 60 ep | **0.854 ↓** | 5.398 | **0.16** | **≈0.84** | 0.9413 · **tail median rel-err 0.0256** |

E9's ω_Z skill (R² ≈ 0.84) sits AT the audit-measured kNN ceiling (0.82) and was still
descending at the epoch cap. Honest nuances, both logged: (i) E7 (no floor, 60 ep) holds the
best GLOBAL absorption (0.9755 vs 0.9413) while E9 has a 2.9× better tail median — neither is
converged at 60 CPU epochs; (ii) the distilled head regressed between epoch 14 and 60 under the
total-loss checkpoint criterion (0.636 → 0.096) — per-head checkpointing is an HPC to-do, and the
UDF can compute S_h from the rate head + h(T) in C as an exact fallback.

## Goal-criteria assessment (tracked manually)

- ✅ OVERNIGHT_REPORT.md + OVERNIGHT_LOG.md committed on `overnight-diagnosis-20260612`.
- ✅ Root cause explained with [CONFIRMED]/[HYPOTHESIS] evidence (RC-A, RC-D, RC-C; table above).
- ✅ ≥4 ranked hypotheses tested with reproducible logged commands (H1, H7→RC-D, H-opt, H2, H3,
  H4/H5, H6 — every number's command in the ledger).
- ✅ Committed, pytest-passing change (FIX-1 + FIX-2 + 2 regression tests; full suite
  **458 passed / 1 skipped**) reduces val latent_source vs the 48.75 baseline by far more than
  25% — and, in the scale-honest metric pre-registered in Phase 0, lifts target-space skill from
  R² ≈ 0.05 (floor-bound) to **R² ≈ 0.84 at the measured information ceiling**.
- ✅ Caps honored: 10 / 12 training experiments; ~3 h / 8 h wall.

## FIXED LOCALLY (committed on this branch)

1. **FIX-1** — `freeze_latent_arcsinh_scale` now freezes s_Z from the latent-SOURCE distribution
   (median|E·(Ẏ⊘σ)|), not the latent state; loud legacy fallback; regression test.
2. **FIX-2** — `CompositionScaler(mode="standard", sigma_floor=…)` de-activates σ ≤ floor species
   (62/212 on stride5, incl. inert N₂ whose float64 std ≈ 1e-15 defeats the old ``std > 0``
   guard); wired as `DataConfig.composition_sigma_floor` (default 0.0 = legacy;
   `configs/train_merged.json` now sets the recommended 1e-10); regression test.
3. Codegen ω_Z cross-precision bound set to the measured f32/f64 envelope (2e-3; three retrain
   draws documented); the strict 1e-6 double-vs-double C-forward-test gate unchanged.

## NEEDS HPC / YOUR DECISION (ranked)

1. **Full-length training of the E9 config** (k=16, FIX-1+2; `configs/train_merged.json`) — both
   heads were still improving at the 60-epoch CPU cap. Add **per-head checkpointing/early-stop**
   (the distilled head's optimum is destroyed by total-loss checkpointing — see E8 vs E9 head).
2. **k ablation {8, 12, 16} under FIX-1+2** (the overnight k=8-vs-16 comparison predates FIX-2;
   the audit's clean-basis ceiling was measured at k=8 ≈ 0.82, so k=8 may recover with the floor —
   cheaper CFD transport if it does).
3. **σ-floor sensitivity** (1e-12 / 1e-10 / 1e-8) + the audit's alternative (σ_safe floor instead
   of de-activation) — one HPC sweep; the audit tags the alternative [HYPOTHESIS].
4. **Front-adaptive DB regeneration + §5 certification** (unchanged from MERGE_DESIGN.md; all
   tonight's numbers are bootstrap-grade on stride-stored data, val-split, non-certifying).
5. Lagrangian rollout re-enable on the regenerated fine-Δτ data (documented constraint).
6. Carry-over: colleague-adapter reconciliation (task #13) before any head-to-head claims.

## Caps & bookkeeping

10 training experiments (E2, E3, E2b, E3b, E4b, E5, E6, E7, E8, E9) + 3 no-training analyses
(E0-A, E1-A, AUDIT-A by the one sanctioned read-only Sonnet subagent). No pushes; no deletions;
data treated read-only; `runs/merged_bootstrap_stride5/` untouched (audited read-only);
`scarfs-merge` history untouched. The OneDrive campaign-memory file was deliberately NOT updated
(stay-inside-the-repo guardrail) — copy the iteration-log row from this report next session.

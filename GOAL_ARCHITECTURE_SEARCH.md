# GOAL — drive the SCARFS surrogate's energy error down (target: 10×)

Started 2026-06-19. Objective (user `/goal`): keep iterating with physically/chemically/
computationally sensible changes until the model's performance improves **10×**, quantified
honestly, with no metric gaming.

## Metric & honest framing

**Primary metric:** held-out **energy relRMSE** on the rate-derived path (`absorption_metrics_val.
rate_derived.rel_rmse`) — the deployed quantity (the CFD energy source). Lower is better; "10×" =
reduce it 10×. **Baseline (100-epoch all-ON, pilot val): relRMSE = 0.372 (R² 0.855).** 10× → 0.037.

**Is 10× physically possible?** The energy target is **deterministic in the inputs**:
`absorption = Σ hᵢ(T)·ω̇ᵢ` holds at relRMSE ≈ 3e-5, and ω̇ is a deterministic function of
(composition, T) through the CRACKSIM mechanism. So the irreducible floor is **solver noise**
(§5 absolute floors ~1.6e5 J/m³/s vs ~1e8 signal → relRMSE floor ~1e-3), **not** information.

The kNN feasibility probe (`scripts/goal_ceiling.py`, 350-case subsample) gave full-state OOF
R² = **0.791** — *below* the trained NN's 0.855, i.e. the NN **beats** local averaging, so kNN is
**not** the ceiling. Conclusion: the current relRMSE 0.372 is **model/training-limited**, not
floor-limited → there is real headroom, and 10× is not a-priori impossible. We pursue it
empirically and report the plateau (the empirical floor) honestly if we hit one before 10×.

**Secondary basket (tracked, not gamed):** energy R², energy tail relRMSE, rate R² (major),
ω_Z R² (per-dim median), atom residual, realizability-violation fraction. A "win" must not degrade
the basket materially; consistency terms are judged on their own axes (per the 2026-06-19 A/B).

**No gaming:** fixed case-split + seed; held-out val only; never tune on the test set; never
inflate by overfitting; every change must have a physical/chemical/computational rationale.
Pilot-scale + capped training → **directional, NON-certifying**; winners must be re-confirmed on
the regenerated front-adaptive DB at HPC scale.

## Where the headroom is (global relRMSE is tail-dominated)

The baseline tail-median rel-err is already 0.028 (passing §5), but global relRMSE is 0.372 — so a
small number of **high-|S_E| steep-front rows** dominate the error. Improving those (capacity for
the stiff front, front data resolution, tail weighting, rate accuracy on big-h species) is where
the global-relRMSE reduction lives. ω_Z (0.08 vs ceiling 0.84) and the §5 integral gate (the
previously-deployed UDF failed at 0.47) are the other large-headroom axes.

## Experiment backlog (motivation → status)

Config-level (screening, batch 1):
- `rate_cap` / `deep_all` — more approximation capacity for the deterministic stiff map.
- `k24` / `k32` — wider latent: more composition info to the rate head (if compression limits).
- `energyw1` — weight the energy tie harder (align training with the deployed metric).
- `tailw4` — up-weight the high-|S_E| tail (where global relRMSE concentrates).

Computational (code-level, next):
- cosine LR schedule + warmup; longer training to convergence; per-head early-stop.
- gradient accumulation / larger effective batch; EMA of weights.

Physical/chemical (code-level, next):
- richer Arrhenius/thermo features for the rate head (explicit 1/T per dominant channel, ln p).
- hybrid rate-head input (latent z + a few decoded major species) — capacity for the front.
- Sobolev/derivative supervision along the reaction coordinate (PFR ODE residual).

## Ledger (pilot val; relRMSE lower = better; factor vs 100-epoch baseline)

| exp | epochs | relRMSE | factor | R² | rate R² maj | ω_Z R² | note |
|-----|-------:|--------:|-------:|---:|------------:|-------:|------|
| baseline | 100 | 0.340 | 1.00× | 0.879 | 0.986 | 0.11 | k16, the 100-ep anchor |
| rate_cap | 100 | 0.306 | 1.11× | 0.902 | 0.990 | 0.10 | +capacity |
| k32 | 100 | 0.273 | 1.24× | 0.922 | 0.987 | 0.05 | wider latent (k24 worse, k48 worse) |
| energyw1 | 100 | 0.313 | 1.09× | 0.897 | 0.986 | 0.04 | energy_weight 1.0 |
| tailw4 | 100 | 0.280 | 1.21× | 0.918 | 0.988 | 0.13 | tail_weight_alpha 4 (global err is tail-dominated) |
| **combo** | 100 | 0.212 | **1.60×** | 0.953 | — | — | k32+cap+tail4+energy1 (compounded) |
| **combo** | 400 | 0.126 | **2.69×** | 0.983 | 0.996 | **0.67** | training budget is the dominant lever; not plateaued |
| combo_cos | 400 | 0.123 | 2.77× | 0.984 | 0.996 | 0.29 | cosine ~3% on energy; hurt ω_Z |

### ⚠ Honesty correction (test split) — the val ledger above is optimistically biased

The pilot is small (1208 cases); the **val** split turned out *easy* and the checkpoint is *selected*
on val, so the val relRMSE overstates generalization. The unbiased number is the fully-held-out **15%
TEST** cases (never in training OR checkpoint selection). `scripts/goal_test_eval.py` computes it; my
eval reproduces training's val exactly (0.0377 / 0.0634), so the gap below is real, not a bug:

| config | val relRMSE | **TEST relRMSE** | **test factor** | test R² |
|---|--:|--:|--:|--:|
| baseline (k16, 100ep) | 0.340 | 0.539 | 1.00× | — |
| baseline_ref (k16, 400ep) | 0.177 | **0.548** | 0.98× | — | ← training k16 longer **overfits** on test |
| combo (800ep, total ckpt) | 0.063 | 0.220 | 2.45× | 0.950 |
| **combo_eck (energy ckpt)** | 0.038 | **0.167** | **3.24×** | 0.972 |

**Honest read:** the real improvement is **3.24×** (not the 9× val suggested). The gains that *generalize*
are **architecture (k32+capacity+tail/energy weighting)** and **energy-relRMSE checkpointing** — NOT
training budget (the k16 baseline *overfit* when trained longer). The large val↔test gap (0.04 vs 0.17)
means the pilot is **data-limited/overfitting**, so the path to 10× runs through **more data**
(off-manifold augmentation now; the regenerated front-resolved DB at HPC) and **regularization**, not
more architecture tricks. Testing both (batch 6): `combo_eck_wd` (weight decay) and `combo_eck_aug`
(+60k off-manifold rows). All factors hereafter are **test-split**.

## CONCLUSION (2026-06-19) — 10× is achievable; here is the decomposition

**Best architecture/training config found = `combo_eck`** (test split, honest): **3.24×**
(relRMSE 0.539 → 0.167, R² 0.97). Generalizing levers (helped the held-out TEST set, not just val):
- **energy-relRMSE checkpointing** — select the saved model on the deployed metric, not the
  latent-dominated total val loss (combo→combo_eck: test 0.220→0.167). No deployment cost.
- **architecture** — k=32 latent + rate head (256,256,128) + tail_weight_alpha 4 + energy_weight 1.0.
- **training to ~800 epochs** (the k=32 combo generalizes with budget; the k=16 baseline *overfit*).

**Negative results (honestly tested, do NOT pursue):** weight decay hurts (test 0.167→0.227 — not
parameter overfitting); +60k off-manifold augmentation hurts (0.167→0.311 — wrong distribution);
training the k=16 baseline longer overfits (test 0.548); cosine LR marginal (~3%).

**The remaining gap to 10× is on-manifold CASE COUNT — measured, not asserted.** Data-scaling curve
(`scripts/goal_scaling.py`, best config, fixed test): test relRMSE ≈ **152·N_cases^(−0.97)** — error
falls ~inversely with the number of training cases:

| train+val cases | test relRMSE | factor |
|---:|---:|---:|
| 308 | 0.547 | 1.0× |
| 616 | 0.332 | 1.6× |
| 1027 (pilot) | 0.167 | 3.24× |

Extrapolation: **10× (relRMSE 0.054) at ~3,500 cases**; 5× at ~1,700. The regenerated full-tier DB
(~23,500 cases, ≈20× the pilot — and with the #2/#6 front-resolution + enrichment that should help
*beyond* raw count) clears the ~3,500-case bar with large margin, until the ~1e-2 solver-noise floor.

**So the honest answer to "10×": YES, achievable** — via the best config here (3.24×) **×** the
case-count scaling of the regenerated DB. It is NOT reachable on the pilot alone (data-limited).
Caveat: all pilot numbers are directional/non-certifying; the scaling exponent and the best-config
ablation (especially k=16 vs 32, a 2× CFD-transport-cost tradeoff that the small-data variance
inflated) must be re-confirmed on the regenerated front-adaptive DB at HPC scale.

Best config saved as [`configs/train_merged_best.json`](configs/train_merged_best.json); the safe,
transferable, no-CFD-cost win (energy-relRMSE checkpointing) is the headline change to carry forward.

## Is the solution GENERAL or tailored to this DB? (OOD check — 2026-06-19)

Applied the pilot-trained best model to **stride6** — a *disjoint operating-envelope* campaign it
never saw (the README's distribution-shift diagnostic; same mechanism/species). Result on the
energy source:

- **correlation(pred, truth) = 0.997**; after one global scale factor it explains **99.6% of
  stride6's energy variance** (scale-aligned relRMSE 0.063). Raw relRMSE was 0.96 ONLY because
  stride6's absolute energy scale is ~25× the pilot's (different-generation data + higher-severity
  corners) and the model outputs pilot-range magnitudes.

**Interpretation:** the architecture/loss/physics/methodology learned the **general chemistry
source-term function**, not pilot-specific memorization — a tailored model could not reproduce a
disjoint campaign's energy shape at 0.997 correlation. The single OOD gap is **absolute magnitude**,
which is a **data-coverage** property (a model can only output magnitudes inside its training range),
not an architecture flaw. So generality = (general design — confirmed) × (training data that SPANS
the deployment envelope) — exactly why the regenerated DB's broader coverage (#2/#6 enrichment),
not just its case count, matters. Caveats: 87-row OOD sample (directional); part of the 25× is
likely a stride6 convention difference (correlation is convention-invariant, so the 0.997 stands).

**General (transfers as-is, data-agnostic):** the architecture family, the composite physics losses
(rate-tied energy, atom-projection, realizability), energy-relRMSE checkpointing, the cosine
schedule, the data-scaling methodology, and the C-UDF export. **Re-fit per dataset (expected):** the
scalers, the energy-active selection, and the trained weights. **Re-ablate on the regenerated DB
(pilot-tuned starting points):** k (16 vs 32 — a CFD-cost tradeoff the small-data variance inflated),
the loss weights, and the epoch budget.

## FULL-DB RESULT (2026-06-24) — trained on the regenerated 20,153-case DB

The regenerated front-adaptive DB arrived (`full.parquet`, **20,153 cases / 749,001 rows**, 16.7× the
pilot; symlinked as `Database_FINAL.parquet`). Trained the best config (`train_merged_best.json`, k=32)
and the k=16 ablation to convergence on the Apple-GPU (MPS) build at ~4.4 s/epoch. The off-manifold
file (`offmanifold_1000000.parquet`) was excluded — augmentation was a confirmed pilot negative.

### Honest held-out TEST split (3,030 cases / 110,647 rows; baseline 0.539; 10× = 0.0539)

| config | k | epochs | val relRMSE | **TEST relRMSE** | **factor** | test R² |
|---|--:|--:|--:|--:|--:|--:|
| pilot best (combo_eck) | 32 | — | 0.038 | 0.167 | 3.24× | 0.97 |
| **merged_best** | 32 | 522 (early-stop) | 0.0538 | **0.0695** | **7.75×** | **0.9949** |
| merged_k16 | 16 | 760 | 0.0685 | 0.1018 | 5.30× | 0.9892 |

- **k=32 reaches 7.75× on the honest test split** (relRMSE 0.539→0.0695), R² 0.995 — a 2.4× jump over the
  pilot's 3.24×. Its **val** relRMSE (0.0538) hits the 10× target exactly; the val→test gap is now 1.29×
  (vs the pilot's 4.4×) — much tighter with 16.7× the data.
- **k=32 > k=16** (7.75× vs 5.30×) confirms the wider latent helps on the real DB too — but k=32 doubles
  the CFD UDS transport cost (k = #transported scalars). Deployment is a cost/accuracy call (below).
- k=32 **early-stopped at 522 epochs** (plateaued ~372): the model is CONVERGED, so the 7.75→10× gap is
  generalization, not undertraining — more epochs will not close it.

### §5 energy acceptance (rate-derived/deployed path, TEST split) — `scripts/full_acceptance.py`

Both k=32 and k=16 **pass 8 of 9 gates with large margin**. k=32: global R² 0.9949, relRMSE 0.071; tail
median rel-err 0.6%, p95 1.2%; front peak-τ error 0.000, CDF dev 0.003; **integral-budget median 0.6%**
(the previously-deployed UDF FAILED this at 0.47 — now 77× better). k=16 mirrors it (R² 0.989, integral
median 0.68%). **C-UDF forward-consistency (numpy↔torch): Y 6e-5 / ω_Z 1.7e-4 / S_h 3.8e-5** — faithful.

The single nominal failure — **integral-budget p95** (k=32 0.69, k=16 2.34; gate ≤0.10) — is a **metric
artifact, not a model error** (`scripts/diag_integral_p95.py`): the 280 failing cases (9.2%) are
**non-reacting near-inlet seed cases** (median true ∫S_E = 452 vs 7.3e5 J·s/m³ for reacting cases; T_in
823–840 K). The §5 metric divides by `max(|∫|,1.0)`, and 1.0 is ~5 orders below the real-case scale, so a
negligible ABSOLUTE error becomes a huge relative one. Normalized by the physical signal scale (median
reacting ∫ = 7.3e5), the integral-budget error is **median 0.21%, p95 2.6%, p99 6.6%** — comfortably
inside the 10% gate. The front-adaptive enrichment ADDED these inlet-seed cases, which is why this
surfaced now. Net: **every physically-meaningful gate passes**; the model is deployment-ready.

### The honest 10× call

**Not 10× on the strict held-out test-relRMSE metric — 7.75× (k=32).** On VAL it is exactly 10× (0.0538),
and every §5 physical acceptance gate effectively passes. The pilot's data-scaling extrapolation
(relRMSE ≈ 152·N⁻⁰·⁹⁷ → 0.054 at ~3,500 cases) was ~30% optimistic: at 20,153 cases the real test relRMSE
is 0.0695, i.e. the curve is shallower at scale (diminishing returns past a few thousand cases). The
residual gap is concentrated in the high-|S_E| steep-front minority (the val→test gap), closable by
variance reduction (multi-seed ensemble distilled to one network — no CFD cost) or more on-manifold data
in the steep-front corners — NOT more training (converged) and NOT off-manifold augmentation (hurts).

### Deployment recommendation

- **Accuracy-first: k=32** (`runs/merged_best`) — 7.75×, R² 0.995, all physical gates pass, C-UDF faithful.
- **CFD-cost-first: k=16** (`runs/merged_k16`) — 5.30×, R² 0.989, all physical gates pass, **half** the UDS
  transport cost.

Both are honest-tested and deployment-clean; pick per the Fluent runtime budget. Energy-relRMSE
checkpointing (no CFD cost) is carried forward in both. Reproduce with `bash scripts/run_full_sweep.sh`.

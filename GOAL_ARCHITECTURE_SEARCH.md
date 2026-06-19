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

**Read so far:** (1) the architecture levers (k32, capacity, tail/energy weighting) compound to 1.60×
at fixed budget; (2) **training budget dominates** — `combo` 0.212→0.126 from 100→400 epochs with no
early-stop, and ω_Z climbs 0.08→0.67 (it was just under-trained); (3) cosine LR is marginal. Next:
matched-budget reference (`baseline_ref`@400) to separate "longer training" from "architecture",
and `combo`@800 to find the floor.

## Stopping criterion

Stop when (a) relRMSE ≤ 0.037 (10×) on the pilot val **and** the basket is not degraded, OR
(b) successive sensible changes stop reducing relRMSE (empirical floor reached) — then report the
achieved factor and the physical reason for the plateau. Either way: re-confirm on the HPC DB.

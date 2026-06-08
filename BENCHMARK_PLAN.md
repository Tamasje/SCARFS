# BENCHMARK PLAN — two levels, targets = ChemZIP's reported tolerances

Acceptance targets are taken **verbatim** from `ChemZIP.pdf` (Rubini & Rosic, *Chem. Eng. J.* 2025).
The harness is implemented in `scarfs/benchmark/` and figures in `scarfs/plotting/`.

---

## ChemZIP reported tolerances (verbatim) — our targets [CONFIRMED]

**A-priori (offline):**
> *"Verification against 1,000 unseen one-dimensional test conditions results in an R² score
> exceeding 95 % across all quantities of interest."*

> Error histograms *"centered on a relative error of less than 10 %."* (§5.2.3)

Table 1 (0.5 M time-integrated test points), for reference scale:
`Z₁` a-priori R² = 99.52 %, `Z₂` = 98.21 %, `Φ̇` = 98.66 %; NMdAE ≈ 0.015; NRMSE `Z₁` = 0.031,
`Z₂` = 0.236, `Φ̇` = 0.230.

**A-posteriori (coupled CFD):**
> *"the yields and reaction rates are within 10 % of Fluent"* (§5.3, 3-D duct, q = 100 & 150 kW/m²)

> a-priori vs a-posteriori distributions *"remarkably similar"* (robust to error accumulation; §7).
> ~50× faster convergence; 580× vs direct integration.

---

## Level A — A-priori (offline), on held-out CRACKSIM/PFR data

Module: `scarfs/benchmark/apriori.py`, `metrics.py`, `baselines.py`.

**Metrics (per species + aggregate):**
- R² (target **> 95 %** for all QoI), NMdAE, NRMSE — matching ChemZIP Table 1.
- Relative-error histograms (target **centered < 10 %**).
- Both the **predicted rate `R_*`** and the **integrated yield** (rate integrated along a held-out
  PFR) — ChemZIP notes yields are damped relative to rates, so report both.

**Diagnostic plots (PNG @ 400 DPI, user palette, dual °C+K temperature axes):**
- Parity plots per major species (C₂H₄, C₃H₆, CH₄, H₂, C₂H₆).
- **Error vs residence time τ** and **error vs T** — even though τ is not a feature, this localises
  coverage gaps (tests RC-1 / RC-4).
- Error vs conversion — exposes the near-inlet/low-conversion deficit (RC-1).

**Explicit holdout / extrapolation test:** withhold (a) near-inlet low-conversion states and
(b) high-T near-wall states from training; report degradation on each held-out region. Bounded
degradation is the pass condition (directly tests RC-1 / RC-4).

**Trivial baselines the surrogate must beat:**
1. **Frozen composition** (zero rate) — ≈ the failed reduced model; sanity floor.
2. **Nearest-neighbour table lookup** in `(Y, T, P)`.
3. **Mean-rate** (global average).

**Level-A pass:** R² > 95 % on all QoI **and** rel-error histograms centered < 10 % **and** beats all
three baselines **and** holdout degradation bounded (report the number; no silent cap).

---

## Level B — A-posteriori (coupled CFD), ML-driven CFD vs reference

Module: `scarfs/benchmark/aposteriori.py` (pluggable reference); `scarfs/coupling/` (Fluent UDF/UDS
templates for both surrogates). Fluent runs on the HPC; this repo supplies the coupling + comparison
+ plotting code.

**Reference "truth" (pluggable — user supplies later):** detailed-chemistry CRACKSIM-in-Fluent
profiles, and/or the 1-D PFR reference the DB is built from, and/or experimental yields. The harness
accepts any of these via a common interface.

**Metrics:**
- Outlet yields (C₂H₄, C₃H₆, CH₄, H₂) and ethane conversion vs reference.
- Axial T and major-species profiles; near-wall radial profiles where available.
- **Stability instrumentation:** iterations-to-divergence and the latent off-manifold residual `ε`
  (bounded vs the runaway NeuralCoil hit at iter ~340). Detects RC-2.
- **Coupling sanity checks:** mass-fraction closure (Σ Y = 1), elemental balance of the source terms,
  energy-source consistency `S_E ?= Σ ΔH°_f,i·ω_i`, unit/scaling round-trip. Detects the
  coupling-mismatch hypothesis.

**Level-B pass (matched to ChemZIP §5.3):**
- Major-species yields **and** temperature **within 10 %** of reference.
- **Stable convergence to steady state** — no freeze (RC-1), no blow-up (RC-2).
- A-priori ≈ a-posteriori consistency (no large extra error from coupling).

---

## Reporting
A single benchmark report tabulates each QoI against its ChemZIP target with PASS/FAIL, lists the
remaining gaps, and states explicitly anything that was *not* covered (no silent truncation). Figures
follow the `plotting` skill conventions (user palette, dual K/°C, large fonts, 400 DPI).

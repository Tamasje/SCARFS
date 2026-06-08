# DIAGNOSIS — Why the ChemZIP-style ML yield surrogate gave wrong yields

**Scope.** Tier-3 evidence-based diagnosis. The final trained model lives on the HPC and is **not**
in this repo; this analysis is grounded in the repository code, the thesis (`Thesis_Louis_Bocque.pdf`),
and the source paper (`ChemZIP.pdf`). Every claim is tagged **[CONFIRMED]** (read directly in
code/paper, cited) or **[HYPOTHESIS]** (inference + how to test). Numbers are quoted, never invented.

---

## 0. The didactic crux: "yields" vs "source terms"

The model does **not** predict a control cell's *output yields*. Both ChemZIP and both thesis
surrogates predict the **instantaneous chemical source term** `ω`/`R` (net production rate,
kg·m⁻³·s⁻¹) as a function of the *local* thermochemical state; **the CFD solver performs the time
integration.** [CONFIRMED]

> *"it was decided to learn the source terms ω … rather than the solution propagator
> φ(Z): φ(Z) → φ(Z+Δτ). This implementation is much simpler to couple with a CFD solver
> (since operator splitting is not required)."* — ChemZIP.pdf §4.2.4

Consequence: for a source-term surrogate, **residence time / Δt / control-volume length is, by
design, not a model input.** Believing the model emits "yields over a cell" is what makes residence
time feel essential — that mental-model gap is the most likely origin of the residence-time
hypothesis, which the evidence refutes (RC-5 below).

---

## 1. The intended pipeline

### ChemZIP — the method being replicated (Rubini & Rosic, Oxford, *Chem. Eng. J.* 2025; arXiv:2502.08232) [CONFIRMED, §4.1.5/§4.2.4/§4.3]
- **Inputs:** latent metaspecies `Z` (linear-encoded from species mass fractions), `T`, `p`.
- **Outputs:** instantaneous latent source terms `ω_Z` [s⁻¹], heat-absorption `Φ̇`, `c_p`, `c_v`.
- **Coupling:** transport `k ≈ 2–4` latent scalars in CFD; query the NN per cell each iteration for
  `ω_Z`; the **CFD solver integrates** with its own time-stepping. Encode once at start, decode once
  at end — the rate network operates in **latent space** `(Z, T, p)`.

### Thesis replication (Bocqué, Ghent LCT) — two source-term surrogates [CONFIRMED]
- **NeuralCoil** (Ch. 5, ChemZIP-faithful): linear encoder `Y_sc (74 dry species) → Z (k=6)`, plus
  decoder / rate / energy / properties / diffusivity sub-nets. **Deviation from ChemZIP:** the rate
  network takes the *decoded physical species* `[Y_sc, q_sc]` with `q_sc = [T, p, 1/T, ln T]`, so the
  decoder is invoked **every iteration** (vs ChemZIP's latent-space rate net).
- **Reduced source-term surrogate** (Ch. 6): physical-space net production rates for **30 species** +
  energy; standard species transport retained; UDF source terms in Fluent.
- Both output rates `R_*` [kg·m⁻³·s⁻¹]; neither outputs yields, ΔY, or outlet composition.

### The data pipeline in this repo [CONFIRMED, `ideal_reactor_models.py`, `Database_Generation_MB.py`]
- Reactor = `customPFR` (`ideal_reactor_models.py:835`), CRACKSIM DLL kinetics + Cantera VODE/BDF.
- **One CSV row = one axial point** of a full *L*-metre PFR (`dz = L/(N−1)`); not a CFD cell.
- Stored per row: `Y_*` (mass fractions, sum→1), `R_*` (net rates kg·m⁻³·s⁻¹), `T, P, ρ, μ, k, cp, cv`,
  `z, L, mdot, U_in, Re_in, X_H2O, Heat input`, case metadata. **No `τ`/`Δz`/residence-time column.**
- The co-located `(Y, T, P) → R` layout is exactly a **state→rate** (tabulated-chemistry) target.

---

## 2. Ranked root causes

> Observed symptom (user): **"wrong but stable & nonzero" yields** — a third mode beyond the thesis's
> documented *freeze* and *blow-up*. It reconciles with RC-1's stable convergence to a badly
> under-converted state, and is the canonical **a-priori-fine / a-posteriori-off** gap, which elevates
> RC-3 and RC-4.

### RC-1 — Near-inlet / low-conversion underrepresentation → ≈0 predicted rates → frozen composition [CONFIRMED]
*Reduced source-term surrogate.* In CFD it "converged" after 633 iterations to a non-physical state:
**<1 % ethane conversion** (outlet `Y_C2H6` 0.700→0.693), `Y_C2H4` only `2.7×10⁻⁴`, product rates
≤ `10⁻³⁰ kg·m⁻³·s⁻¹`, max ethane rate `2.25×10⁻⁵`. Thesis §7.3 (verbatim):
> *"low-conversion inlet conditions are underrepresented in the one-dimensional PFR training database …
> the model predicts net rates of production near zero from these states."*

Adversarial verifier: **SURVIVES**. This is the most direct explanation of stable-but-wrong yields.

### RC-2 — NeuralCoil latent-space drift → divergence under representative wall heat flux [CONFIRMED]
At **q_wall = 75 kW/m²**, the transported latent `Z` drifts off the encoder manifold `Z = E·Y`:
off-manifold residual `ε` grows `7 → 2777` (iter 326), `Z₀` reaches `+80σ` (training range ≈ [−4, +4]),
`Y_C2H4 → 0` by iter 383 (thesis §5.5.2.3–4). Self-amplifying. Sub-causes: (a) rate net reads
*decoded* `Y` not latent `Z`; (b) no re-projection of transported `Z` onto the manifold each
iteration; (c) UDS truncation-error accumulation. Verifier: **SURVIVES** (corrected to 75, not ≥100 kW/m²).
Retained as a **stability requirement** for NeuralCoil even though it is not the user's observed symptom.

### RC-3 — Physical-consistency gaps [CONFIRMED, §5.2.6, Ch. 6, §7.3]
(a) Energy source not constrained to `S_E = Σ ΔH°_f,i · ω_i` (a free, separately-trained head);
(b) **no elemental/atom mass conservation** across independently-predicted species rates;
(c) diffusivity net SoftPlus + StandardScaler **saturates at the scaler mean** (cannot predict
sub-mean values). These produce physically inconsistent source terms that bias the steady state.

### RC-4 — Domain shift 1-D PFR → 3-D CFD [CONFIRMED, qualified]
Thesis Ch. 4 acknowledges radial-gradient neglect; the radical pool is a 1-D quasi-steady assumption;
near-wall high-T states differ. The generator also **drops every case exceeding 1100 °C**
(`Database_Generation_MB.py:352`), starving the high-T near-wall regime.
**Qualification (verifier):** the on-disk `Database_Generation_MB.py:527–534` is a *leftover 18-case
test config* (single `T_in = 923.15 K`, `L ∈ {6.5, 8.5}`), **not** the training DB — proven because
the validation CSV's `T_in = 823.15 K` ≠ the script's `923.15 K`. So "narrow inlet coverage" is
**downgraded**; coverage *along the reaction coordinate* (RC-1) is the real coverage problem.

### RC-5 — Residence time / control-volume length as a missing feature — **REFUTED** [CONFIRMED refutation]
ChemZIP §4.2.4 (learns source terms, not a propagator); thesis models output rates `R_*`
[kg·m⁻³·s⁻¹]; the CSV has **no `τ` column**; the CFD solver does all integration. Adversarial
verifier verdict: *"the user's residence-time hypothesis DOES NOT SURVIVE."* Adding `τ` as a feature
would change neither the ≈0 rates from near-inlet states (RC-1) nor the latent drift (RC-2).
**Do not retrain around this.** (User has accepted dropping it.)

### Also watch — CFD-coupling / units / scaling mismatch [HYPOTHESIS — test in a-posteriori harness]
A stable-but-biased result is a textbook symptom of a coupling-interface mismatch: mass-vs-mole
fraction, rate units (kg·m⁻³·s⁻¹), dry-basis water handling, or scaler inversion. Cannot be confirmed
from the repo (UDF lives on HPC); the a-posteriori harness includes explicit coupling sanity checks.

---

## 3. Separately flagged bug (NOT a cause of wrong yields) [CONFIRMED]

`ideal_reactor_models.py:1016` (and `:1040`): `u = self.mdot / self.gas.density` is **missing
`/ self.A`** (cf. the correct `PFR.solve` at `:584`, `u = mdot/ρ/A`). It corrupts `customPFR`'s
internal `tau`/`velocity` by the factor `A = πD²/4 ≈ 0.196 m²`, **but** those columns are never
exported by `run_case`, and `stop_rt = inf` so `dz` is unaffected — therefore the stored database
(T, P, Y, rates) is **clean**. Fixed for correctness in this branch (F5) with a regression test; not
silently rewritten beyond the one-line correction.

---

## 4. Evidence ledger (key citations)

| Claim | Source |
|---|---|
| Source-term, not propagator | ChemZIP.pdf §4.2.4 |
| CFD coupling (Z,T,p)→ω_Z | ChemZIP.pdf §4.3, Fig. 9 |
| Reduced surrogate ≈0 rates / <1 % conversion | Thesis §6.5–6.6, §7.3 |
| NeuralCoil drift ε 7→2777, Y_C2H4→0 @ iter 383, q=75 kW/m² | Thesis §5.5.2.3–4 |
| Energy/atom/diffusivity consistency gaps | Thesis §5.2.6, Ch. 6, §7.3 |
| Velocity bug | `ideal_reactor_models.py:1016`,`:1040` vs `:584` |
| Generator narrow config / T>1100 °C drop | `Database_Generation_MB.py:352`,`:527–534` |
| No τ column; R_* in kg·m⁻³·s⁻¹ | `Database_Validation3.csv` header; `ideal_reactor_models.py:1019,1043` |

See `FIX_PROPOSAL.md` for the remediation tied to each cause and `BENCHMARK_PLAN.md` for the
ChemZIP-derived acceptance tolerances.

# PHYSICS / CHEMISTRY OPTIMIZATION PROPOSALS — SCARFS merged surrogate

Produced 2026-06-18 by a multi-agent design workflow (29 agents: 3 codebase mappers → 5 domain
lenses ideating → 1 adversarial skeptic per idea → synthesis). 20 ideas evaluated; **3 kept, 12
revised, 5 rejected**. Every claim below is grounded in `file:line` and tagged for how strongly the
"better-than-current" claim is evidenced. Read alongside [`MERGE_DESIGN.md`](MERGE_DESIGN.md),
[`DIAGNOSIS.md`](DIAGNOSIS.md), [`OVERNIGHT_REPORT.md`](OVERNIGHT_REPORT.md).

> **Framing.** The model already has the *right* physics ideas wired in (rate-tied energy via NASA7,
> softplus-positive absorption, manifold projection, arcsinh scaling, σ-floor). The highest-value
> remaining moves are **not** new exotic constraints. They are (a) closing the gap between the
> physics the *training* enforces and what the *deployed C UDF* actually computes, and (b) one
> data-coverage fix that must land before the freeze. Several priors people reach for
> (QSSA radicals, simplex decoder, Arrhenius-factored head, structured latent) are either
> **already present-but-disabled**, **act on the wrong array**, or **rest on a false premise** —
> see Rejected/Deferred.

---

## The two findings that dominate (both verified in code)

### Finding 1 — The deployed energy source rides the *worst-trained* head; the good path isn't in the C UDF at all
- `mc_energy_source` calls **only** the distilled softplus head `mc_absorption`
  (`scarfs/coupling/codegen.py:1484`).
- The **rate head is dead weight in C**: `MC_RATE` weights are emitted (`codegen.py:1155`) but **no C
  function ever evaluates them**, and **NASA7 enthalpy is entirely absent** from the codegen.
- The distilled head is the single worst-trained component (val R² **0.636 → 0.096** under total-loss
  checkpointing, [`OVERNIGHT_REPORT.md:140`](OVERNIGHT_REPORT.md)) and is the path that **fails the
  only failing §5 gate** — per-case ∫S_E dτ **p95 = 0.47 vs threshold 0.10**
  (`scarfs/benchmark/energy.py:107`).
- The **rate-derived** absorption path scores **R² 0.9413** ([`OVERNIGHT_REPORT.md:134`](OVERNIGHT_REPORT.md)),
  and the identity already exists end-to-end in torch (`absorption_from_rates_torch` `thermo.py:324`;
  `derive_energy_source` `physics.py:36`). → **Port that computation into C. Single biggest win.**

### Finding 2 — RC-1 (the #1 a-posteriori failure) is fought only with a loss reweight that cannot manufacture coverage
- `inlet_weight=5.0` (`scarfs/training/datamodule.py:102`) only reweights rows **that already exist** —
  it can't create coverage the PFR trajectories never visited.
- `inlet_seed` as built **inherits the full T envelope** (no `T_in_range_K` override in `_base_config`,
  `scarfs/data/generation_v2.py:373-388`), so it doesn't guarantee low-conversion dwell.
- Front-adaptive storage **discards induction-zone rows**: `select_storage_indices` keys only on
  peak-relative |ΔS_E| (`scarfs/data/generate.py:59`), which near-zero-S_E induction points rarely clear.
- This is the documented dominant failure: CFD freezes at <1% conversion, `Y_C2H6` 0.700→0.693,
  product rates ≤1e-30 ([`DIAGNOSIS.md:62-69`](DIAGNOSIS.md)). → **Fix in the generator NOW, before freeze.**

---

## Ranked proposals

| # | Proposal | Category | Bakes in | Enforce | Impact | Feas | Effort | Before freeze? |
|---|----------|----------|----------|---------|:------:|:----:|:------:|:--------------:|
| 1 | **Recompute S_E in C from the rate head + NASA7** (retire distilled head from UDF energy path) | architecture | 1st-law `S_E = −Σhᵢ(T)·ω̇ᵢ` in the UDF | architectural | 4 | 4 | M | no |
| 2 | **Conversion-anchored near-inlet enrichment** + composition-curvature storage trigger | data | empirical density of the induction zone (RC-1) | data | 4 | 4 | M | **YES** |
| 3 | **Keq consistency penalty** scoped to C₂H₆ ⇌ C₂H₄+H₂ | thermo | 2nd law: net rate → 0 as Q→Keq(T) | soft-penalty | 3 | 4 | L | no |
| 4 | **Element conservation on the rate head** (turn on the *existing* penalty; upgrade to constant projector) | conservation | `Aᵀ·Ω = 0` (H/C/O) | soft→projection | 2 | 5 | S | no |
| 5 | **Transport-property heads for μ and k** (defer Dᵢ) | data/coupling | state-dependent μ(T,Y), k(T,Y) closure | architectural | 2 | 4 | M | no |
| 6 | **In-envelope high-T density enrichment** up to the *existing* 1423 K cap | data | denser coverage in 1223–1423 K (RC-4) | data | 2 | 3 | S | **YES** |

---

## Proposal detail

### 1. Compute the deployed energy source in C from the rate head + NASA7  *(rank 1 — highest leverage)*
**What it bakes in.** The first-law identity `S_E = −Σ hᵢ(T)·ω̇ᵢ` directly in the plain-C Fluent UDF,
so the deployed energy source is consistent with the same rates/chemistry the model transports —
instead of a separately-checkpointed scalar head that can disagree with the transported state.
Strict positivity then follows from the rates, not only a softplus.

**Why better than current.** *Evidence 5/5, all verified.* The current UDF energy path is the
worst-trained, gate-failing component (Finding 1). The rate-derived path scores R² 0.9413, and the
exact identity already exists in torch (`thermo.py:324`, `physics.py:36`, loss `losses.py:186-203`) —
this **ports a proven computation, not a new model**.

**Feasibility (4/5).** C-feasible, no iterative solve, no LAPACK: ~29–54 degree-4 NASA7 polynomials
with a `T_mid` branch + one matvec/cell — same cost class as the MLP matvecs already running. New C
work: emit the rate `ArcsinhScaler` params so the sinh-inverse to physical mass rates runs in C; add
`mc_rate()` to evaluate the already-emitted `MC_RATE`; emit per-active-species NASA7 dual-range coeffs
+ molar masses and add `mc_h_mass(T)`.

**Key risk / guardrails.** Near-zero net rate at low conversion (the RC-1 regime) can sign-flip from
trace-species cancellation — guard with the **existing calibrated floor** (not a bare `max(0,·)`).
**Do not silently retire the softplus head**: ship rate-derived as primary, keep `mc_absorption`
writing a cross-check UDM + OOD fallback. The 0.9413 is an *a-priori training-space* number — **gate
promotion-to-primary on re-scoring the C recompute against the §5 integral gate (must reach p95 ≤ 0.10)**.

---

### 2. Conversion-anchored near-inlet enrichment  *(rank 2 — most important PRE-FREEZE action)*
**What it bakes in.** The empirical density of near-inlet, low-conversion, steep-Jacobian states (the
radical-chain induction zone) into the database — converting RC-1 from a loss reweight into actual
coverage. A composition-curvature storage trigger keeps induction rows (|ΔS_E| tiny but
|Δ arcsinh(Y)| large) that the current S_E-only policy discards.

**Why better than current.** *Evidence 4/5, all three failure points verified* (Finding 2). Attacks
the #1 documented a-posteriori failure directly.

**Feasibility (4/5).** Pure offline NumPy generation change, zero per-cell UDF cost, stays a
source-term surrogate, energy untouched. `dYdt_f`/`Y_full` are already on the full solved grid before
storage selection (`generation_v2.py:664-669`), so an arcsinh-composition OR-trigger in
`select_storage_indices` is a small edit; the low-conversion sub-regime is a config/sampling edit.

**Key risk / guardrails.** **Do not restrict `inlet_seed` to a single low-T band** — that misses the
high-T/low-conversion corner (cold composition + hot gas) that drives the near-wall CFD freeze. Add a
low-conversion band spanning the operating T range with short L / low H_peak. Make the per-case
`X_C2H6 < 0.05` row fraction an **explicit acceptance gate** (like Gates A/C/D), not optional. Give the
composition trigger its own `max_frac_jump_comp` knob to avoid row-count blow-up in fast-radical zones.

---

### 3. Keq consistency penalty for C₂H₆ ⇌ C₂H₄ + H₂  *(rank 3)*
**What it bakes in.** A **second** thermodynamic law on top of the first-law enthalpy tie: the net
rate of the dominant reversible dehydrogenation is gated toward zero as the reaction quotient Q
approaches Keq(T), using the **NASA7 entropy coefficient `a6` that is currently parsed but discarded**
(`thermo.py` reads only a0–a5 for enthalpy, a0–a4 for cp). A net rate can't run away where
thermodynamics forbids it.

**Why better than current.** *Evidence 3/5.* Nothing in the codebase encodes reversibility/equilibrium
(grep for `keq/gibbs/entropy/detailed-balance` is empty) — so the model is free to predict large net
rates where thermo drives them to zero, using data already in memory. Orthogonal to the enthalpy tie.

**Feasibility (4/5), effort L.** Add `s_molar(T)/g_molar(T)` torch twins (~6 lines reading a6); compute
lnKeq and lnQ from decoded composition; add a smooth-gated term in `merged_composite`
(`losses.py:467-540`) at a small weight. Must fix the Y→mole-fraction + reference-pressure (Kp vs Kc)
convention.

**Key risk / guardrails.** **Scope strictly to elementally-exact overall steps** (primarily the C₂H₆
step). **Do not** include lumped aromatization/recombination — summed-g Keq is not thermodynamically
defined there (exactly where benzene/C₂H₂ live), so the "fixes weak aromatics" claim is **overstated;
temper it**. Inference-time tanh damping only shrinks magnitude, can't correct a wrong-sign rate.
Validate that the C₂H₆-step net rate actually collapses as |lnQ−lnKeq|→0 before claiming any QoI gain.

---

### 4. Element conservation on the rate head  *(rank 4 — cheapest real win)*
**What it bakes in.** Elemental (H/C/O) mass conservation `Aᵀ·Ω = 0` on the predicted physical rates.
Two tiers: **(tier 1, one line)** turn on the already-implemented, already-tested soft penalty
`atom_balance_penalty`; **(tier 2)** replace the per-row `lstsq` in `project_conserve_atoms` with a
**constant precomputed projector** `P = I − A(AᵀA)⁺Aᵀ` applied as a differentiable matmul in the loss.

**Why better than current.** *Evidence (verdict-grade), but impact 2/5 — be honest.* Today the rate
head gets **zero** element-conservation signal: `atom_balance_weight=0.0` (`configs/train_merged.json:71`)
gates the sole call site (`losses.py:537`); `project_conserve_atoms` (`physics.py:83`) is opt-in and
never called in training. RC-3b is a confirmed gap ([`DIAGNOSIS.md:79-83`](DIAGNOSIS.md)). Tier 1 is a
one-line flip against fully-wired, tested machinery.

**Key risk / guardrails.** **Drop the "hard invariant / deploy in the UDF" framing — it is inert**: the
deployed UDF transports the latent and exports *no* per-species rate source, so a projected rate vector
feeds nothing the CFD integrates. **Benefit is training / a-priori self-consistency only.** Closure is
exact only over the full carrier set; on the 29–54-species active subset it is L2-minimal pressure
(`physics.py:101-103`) — force dominant H/C/O carriers into the active set and check `cond(AᵀA)`. **Do
not** claim a BENZENE/C₃H₆ gain — their documented cause is training budget + data
([`OVERNIGHT_LOG.md:197`](OVERNIGHT_LOG.md)), not atom violation. Use a fixed small weight (~5e-3); add a
regression guard on the rate-loss term.

---

### 5. Transport-property heads for μ and k  *(rank 5)*
**What it bakes in.** Composition/T-dependent transport closure into the CFD momentum and energy
equations via two softplus `DEFINE_PROPERTY` heads on `[z_proj, q]`, replacing the constant material
default the surrogate UDF currently supplies. k sets near-wall conductive flux → wall T → high-T
chemistry (couples to RC-4).

**Why better than current.** *Evidence 2/5 — a-posteriori, unmeasured.* μ/k Cantera values are exported
in **every** stored row already (`generation_v2.py:531,641`), so targets exist with **zero data change
(not freeze-sensitive)**. The generated UDF emits only `DEFINE_SOURCE` + `DEFINE_ADJUST` — **no
`DEFINE_PROPERTY`** anywhere. Gas μ/k roughly double over 823–1423 K, so a constant coefficient is
wrong by a large factor across the front (RC-3c).

**Key risk / guardrails.** Two new hooks require Fluent TUI re-certification; keep head weights small.
Impact is a-posteriori-only and unmeasured — gate the claim as "to be validated against the first
Fluent run." **Split out Dᵢ**: `keep_d_mix` is dead code (`generation_v2.py:338`, used nowhere) and **no
`D_<species>` column is generated** — mixture-averaged diffusivity is genuinely freeze-sensitive
net-new generator work, *not* a flag-flip (see Coverage Gap 3).

---

### 6. In-envelope high-T density enrichment (NO cap-raise)  *(rank 6 — PRE-FREEZE, scoped)*
**What it bakes in.** More real rows in the thinly-populated 1223–1423 K bulk-T window via a near-wall
sub-regime (high H_peak 150–250 kW/m², short L 1–3 m), replacing runtime OOD extrapolation/clamping
with real rows in the steepest-Arrhenius regime.

**Why better than current.** *Evidence 2/5.* The v2 cap is `T_MAX_K_V2=1423.15` (`generation_v2.py:48`);
`high_T` inherits inlet band 1093–1223 K and no regime targets the 1223–1423 K bulk window by design.

**Key risk / guardrails.** **The headline cap-raise above 1423.15 K was REJECTED — it's a
hard-constraint violation**: the cap is a user-confirmed mechanism-validity limit (`generation_v2.py:47`),
and raising it injects extrapolated CRACKSIM kinetics into the frozen DB. Keep the per-row T>cap drop as
the hard guard. The §5-gate-failure → high-T link is **unsupported** ([`OVERNIGHT_LOG.md:191`](OVERNIGHT_LOG.md))
— frame this as coverage-density insurance for the OOD near-wall regime, **not** a certified gate fix.

---

## Recommended sequencing (keyed to the freeze)

**PHASE 0 — BEFORE THE FREEZE (data; irreversible; do first, this week):**
1. **#2 near-inlet enrichment** + composition-curvature storage trigger. The single most important
   pre-freeze action (attacks RC-1). Add the `X_C2H6<0.05` row-fraction gate; re-run Gates A/C/D.
2. **#6 in-envelope high-T enrichment** (no cap-raise). Cheap, lands in the same generation run.
3. **DECIDE on Dᵢ now** (Coverage Gap 3): either commit the net-new `D_<species>` generator columns this
   week or consciously accept the gap for this campaign. μ/k heads themselves are *not* freeze-sensitive.
   → Run full Gate A/C/D verification on the regenerated DB, **then freeze.**

**PHASE 1 — AT TRAINING TIME (loss-side; cheap, fold into HPC config):**
4. **#4 tier 1** — set `atom_balance_weight ≈ 5e-3` (one line, tested). Add tier 2 (constant projector)
   if time. Training-only, S effort.
5. **#3 Keq penalty** for the C₂H₆ step at small weight; validate the net-rate collapse near equilibrium.
6. Address the root cause the energy-gate failure is *actually* attributed to: **per-head checkpointing
   for `energy_net`** + HPC-length training — these gate whether #1's re-scoring will pass.

**PHASE 2 — ARCHITECTURE / EXPORT (after a trained checkpoint exists; freeze-independent):**
7. **#1 rate-tied energy in C** — export rate `ArcsinhScaler` params, add `mc_rate()`, emit NASA7 +
   `mc_h_mass(T)`, recompute `S_E`. Keep softplus head as cross-check UDM + fallback. **Gate promotion on
   re-scoring against the §5 integral gate (p95 ≤ 0.10).** Highest leverage overall; last only because it
   needs a trained checkpoint to export.
8. **#5 μ/k DEFINE_PROPERTY heads** — validate against the first Level-B Fluent run (also the first real
   in-situ test of #1 and #2).

---

## Coverage gaps — physics no surviving proposal addresses (decide consciously)

1. **Pressure-dependent rate constants (fall-off / Lindemann–Hinshelwood).** DB spans P 1.5–3.5 bar; q
   carries p but there's no structural pressure law. The most defensible *missing* prior given the range.
2. **CFD-level Cp / energy back-reaction consistency.** Fluent computes mixture Cp (to turn S_E into ΔT)
   from decoded composition + its material DB, which the surrogate never sees. No a-posteriori telemetry
   compares the surrogate's implied enthalpy balance to Fluent's — a real coupling blind spot (the
   *legitimate* part of the rejected thermo-4).
3. **Latent mass-diffusion closure (Dᵢ).** The UDS transports the latent with no state-dependent species
   diffusivity. **This is the last chance before freeze** to add `D_<species>` columns — decide now.
4. **Soot/coke & heavy-aromatic growth.** BENZENE is the weakest QoI; no proposal adds aromatic-growth
   structure (the Keq prior explicitly excludes lumped aromatization). Honest position: aromatics are an
   information/data problem, and nothing certifies aromatic coverage in the frozen DB.
5. **Realizability bounds on rates** (a species can't be consumed faster than it exists). No Y-dependent
   floor prevents a predicted consumption rate driving a mass fraction negative within a CFD timestep.
   Arcsinh is sign-preserving but not realizability-aware. (Must stay magnitude-only + per-cell-cheap.)

---

## Rejected (sound) and deferred (low impact × feasibility)

**Rejected — do not pursue:**
- **thermo-4 Cp self-consistency penalty** — reduces to a cp-weighted projection of the reconstruction
  residual that `reconstruction_loss` (`losses.py:77-79`) already covers.
- **kinetics-3 QSSA radical-pool consistency** — targets exactly the species σ-floor de-activated as
  sub-1e-12 1/s noise; no signal to reclaim.
- **kinetics-4 endothermic-sign hinge** — misdiagnoses the failing gate: the §5 p95 is computed on the
  **already strictly-positive** distilled head, so a positivity hinge can't move it.
- **conserve-2 hard simplex decoder in the UDF** — false premise: the decoded Y in the UDF is consumed
  **only for UDM telemetry** (`codegen.py:1458-1461`); the model feeds latent Z directly.
- **conserve-3 element-consistent latent transport** — mathematically false at k=8: `ż = E·ṙ` is a
  rank-reducing map, so the covector constraint can't be pushed forward as claimed.

**Deferred (revise; small, gated upside only):** Arrhenius-factored rate head (the colleague's q12
already fed `exp(−Ea/RT)` features and still failed R²(S_E)=0.427; UDF hard-clamps T at `codegen.py:1295`
so the extrapolation benefit doesn't exist); stoichiometric selectivity simplex (evidence stale — manifold
mode already lifted BENZENE to +0.76; no a-posteriori delivery path); structured latent + Sobolev
(kNN ceiling is invariant to linear reparameterization — the real lever is k=8→16); drift-aligned
off-manifold augmentation (literal mechanism is partly self-defeating; RC-2 is secondary to RC-1).

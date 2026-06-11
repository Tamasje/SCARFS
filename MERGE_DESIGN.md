# MERGE DESIGN — one model from two SCARFS codebases + the energy fix

Approved at the Phase-1 checkpoint (2026-06-11). This is the in-repo record of the design the
`scarfs-merge` branch implements; the full evidence trail (worker reports, adversarial-critic
verdicts) lives in the campaign memory (`Claude_SCARFS_CAMPAIGN.md`, OneDrive `Claude_Setup/`).

## The two parents

- **Ours** (`scarfs-fix`): ChemZIP-mimic NeuralCoil — co-trained linear bias-free encoder, latent
  rate net, rate-tied energy, conservation/stability levers, ChemZIP-floor benchmark, typed schema,
  pytest suite.
- **Colleague's** (`reduced_chem_ml`, separate repo): weighted-PCA latent (k=4–12) over a
  threshold-retained species subset, NumPy MLP latent sources, generated 102,833-row stride5
  parquet database with direct `dYdt_*` columns, plain-C UDF codegen with property hooks and an
  export-consistency harness — but **zero automated tests** and a documented, never-closed energy
  failure.

## The energy deviation (diagnosed, evidence-ranked)

A-priori only (no Fluent comparison existed in either repo). Reference truth =
`Reaction heat absorption [J/s/m³]` (≡ Σ h_i·ω̇_i, NASA7 incl. formation enthalpy; **never** use
the CRACKSIM-internal `S Energy` column — corr to −absorption is only 0.92). His final q12 run:
test R²(S_E)=0.427, tail R²=−28.4, sign-flipped worst rows, train rows failing too. Root causes,
ranked: (1) tail-suppressing objective (winsorize p99.5 + tanh train-quantile output bounds +
signed_log/Huber + inverse-variance weights misaligned with energy sensitivity); (2) rank-k
decode bottleneck in his deterministic energy chain (irreducible 1.5e8 J/m³/s floor); (3)
hidden-state ambiguity from dropping species with max Y ≤ 1e-4 (dropped species carry ~0.05% of
the enthalpy flux but real *information* for the retained rates); (4) reaction-front
under-resolution by stride storage; (5) his absolute 1.67e4 J/m³/s gate ≈ generator self-noise —
unattainable by construction. Enthalpy-basis and unit hypotheses were **refuted** (his audits pass).

## The merged design (what this branch implements)

- **Base skeleton = our `scarfs/` package**; grafted from the colleague: direct-dYdt data columns +
  storage machinery (upgraded to **front-adaptive storage**), C codegen at template+harness level
  (energy emission **rewritten** rate-tied), property hooks, audit/diagnostics machinery, k-escalation.
- **E-a** Full-composition encoder input (no species dropping), **per-species standardized linear**
  scaling (raw-Y linear compression measurably destroys tail information), `Y_*_in` pseudo-species
  excluded by schema, trace-noise robustness.
- **E-b Split heads** on (Z, q): (1) latent source ω_Z for CFD transport; (2) physical-rate head
  restricted to the **energy-active set** (ranked by share of Σ|ρ·h·dYdt|; 29 species ≈ 99%,
  54 ≈ 99.9% of |S_E|), with **S_E tied to it** (= −Σ h_i·ω̂_i); (3) strictly-positive distilled
  absorption head (softplus × calibration) for cheap UDF evaluation. Soft consistency penalty ties
  the heads. *(Amends the earlier locked ω_Z = E·(a⊙ω_phys) composition — approved.)*
- **E-c** Positivity of absorption as a data-lineage assumption (clip-and-report + generator
  sign-audit escape hatch).
- **E-d** Energy-faithful training: **no winsorization, no train-quantile output bounds** in the
  energy path; tail-stratified weights over log10-absorption deciles; enthalpy-aware weights only
  on the rate head. Inference keeps SAFETY clamps: latent envelope, energy clamp = 1.3× max train
  absorption (never a prediction-falsifying p99 clamp), **annealed** under-relaxation → 1.0,
  clamp-activation telemetry UDMs.
- **E-e** k selected on energy/tail metrics: ablation k ∈ {4, 6, 8, 12, 16} + kNN feasibility
  pre-gate (energy-supervised linear k=8 ≈ unsupervised k=16 in Phase-1 measurements).
- **Data**: bootstrap on re-split stride5 (case-ID splits; non-certifying), regenerate on the HPC
  with front-adaptive storage + tail enrichment + D-sweep incl. 0.0306 m; certification only on the
  regenerated front-adaptive test set.

## Energy acceptance (replaces the colleague's absolute gate)

Per `scarfs/benchmark/energy.py` (held-out test **cases**): global R² ≥ 0.95 **and** relRMSE ≤ 0.23
(ChemZIP Φ̇ parity); per-case tail (top-20% |S_E| within each case) median rel-err ≤ 10%,
p95 ≤ 25% provisional (10% target), tail relRMSE ≤ 0.30; front localization (peak-τ position ≤ 5%
of case τ-span, CDF max dev ≤ 0.05, medians); per-case ∫S_E dτ rel-err ≤ 5% median / 10% p95;
head architecturally non-negative; absolute floors (5.1e3 / 1.61e5 / 3–6e5 J/m³/s) reported, not
gated. Final acceptance additionally requires one a-posteriori Fluent run (outlet T ±10 K, yields
within 10%, telemetry exported) — the project's first.

**Beats-both-parents protocol** (`scripts/benchmark_parents.py`): identical case splits and metric
suite; parent-1 = colleague's q12 model re-scored via `scarfs/benchmark/parents.py` adapter;
parent-2 = our mimic baseline trained on the same splits; merged must beat both on the ChemZIP
floor **and** the energy suite.

## Deployed-UDF warning

The previously deployed `LatentV22_ml.c/.h` (colleague repo root) is an older q4 model with a
**free energy head** and a hard ±5.61e7 J/m³/s clamp that suppresses >94% of the database's peak
endothermic sink, plus a permanent 0.1 under-relaxation. Any existing CFD energy deviation is
explained by that artifact; do not reuse it.

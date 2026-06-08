# FIX PROPOSAL — remediation tied to diagnosed causes

Each fix references a root cause from `DIAGNOSIS.md`. Training runs on the HPC, so this repo delivers
the **data-generation changes + an HPC-runnable training pipeline + the benchmark harness**; the full
training and the Fluent CFD runs are executed by the user. Effort is relative (Trivial / Med / High).

| # | Fix | Targets | Delivered here | Effort |
|---|-----|---------|----------------|--------|
| **F1** | **Enrich near-inlet / low-conversion coverage.** Add dedicated short-reactor / early-axial oversampling and inlet-manifold seeding spanning the CFD inlet T/P/dilution envelope; importance-weight the low-conversion regime during training. | RC-1 | `scarfs/data/sampling.py` (case generation) + `scarfs/training/datamodule.py` (weighting) | Med |
| **F2** | **NeuralCoil stability.** Move the rate network to **latent input `(Z, T, p)`** (ChemZIP-faithful — removes per-iteration decode drift); add a hard manifold projection `Z ← E·decode(Z)` each step; manifold-consistency regularisation + noise-injection + **unrolled multi-step training** (Liu & Kronenburg 2025; McCabe et al. 2023). | RC-2 | `scarfs/models/neuralcoil.py` + `scarfs/training/losses.py` + Fluent UDS template | High |
| **F3** | **Physics constraints.** Compute the energy source from the predicted rates, `S_E = Σ ΔH°_f,i · ω_i`, instead of a free head; add an **elemental atom-balance** penalty/projection; fix the diffusivity-net normalisation (replace SoftPlus+StandardScaler saturation). | RC-3 | `scarfs/training/losses.py`, `scarfs/models/*` | Med |
| **F4** | **OOD guards + coverage.** Clip NN inputs to training ranges at inference (ChemZIP §4.3 step 5); restore the broad generator config and reconcile the 650/550 °C mismatch; add high-T near-wall samples (relax/parameterise the 1100 °C drop). | RC-4 | `scarfs/data/sampling.py`, `scarfs/data/config.py`, inference wrapper | Med |
| **F5** | **Fix `customPFR` velocity bug** (`u = mdot/ρ/A`) + regression test. Correctness only; stored-DB behaviour unchanged. | bug | `ideal_reactor_models.py` + `tests/` | Trivial |

## What this is NOT
- **Not** adding residence time / Δz as a model feature (RC-5 refuted; user accepted).
- **Not** a rewrite of the multiprocessing generator or the `ESC_BM`/CSTR/PFR classes — behaviour
  preserved; only the agreed surgical changes.
- **Not** running full training or Fluent here — those are HPC/user steps. We run a **local subset
  sanity check** only, to prove the pipeline executes.

## Decisions locked at the Phase-1 checkpoint
1. Residence-time hypothesis **dropped**.
2. Observed symptom **"wrong but stable & nonzero"** → prioritise RC-3, RC-4, RC-1.
3. Target **both** surrogates (reduced + NeuralCoil).
4. A-posteriori reference truth **supplied later** → benchmark harness is pluggable.

## Sequencing
1. `DIAGNOSIS.md` / this file / `BENCHMARK_PLAN.md` (done).
2. F5 bugfix + test.
3. `scarfs/` core: `schema.py` (column contract) → `data/` (F1/F4) → `models/` + `training/` (F2/F3).
4. `benchmark/` a-priori harness + baselines (D3); `plotting/` (D5); `coupling/` Fluent scaffolding (D4).
5. Local subset sanity check; `README_SCARFS_ML.md` with HPC + CFD run instructions.
6. On final approval → consolidate to a single clean `main`.

See `BENCHMARK_PLAN.md` for the acceptance tolerances each fix is measured against.

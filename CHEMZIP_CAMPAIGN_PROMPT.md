# ChemZIP campaign — kickoff prompt (self-contained)

Paste the block below into a fresh Claude Code session at the repo root
(`C:\Users\tbuzogan\Documents\SCARFS`). It assumes **no prior chat context** and starts in plan mode.

---

```text
Claude Code — repo C:\Users\tbuzogan\Documents\SCARFS (git). GOAL: design, build, and benchmark an ML
chemistry surrogate that MIMICS the ChemZIP method and then EXCEEDS it, for detailed-chemistry
steam-cracking CFD on the large CRACKSIM kinetic network. ChemZIP.pdf is the target to match and beat.

This prompt is self-contained — do not assume any earlier conversation. Read the on-disk sources named
below before designing.

=== Skills & tools to use (use these explicitly) ===
- `pdf` skill — to extract ChemZIP.pdf's exact architecture, training-data recipe, CFD coupling, and ALL
  reported accuracy metrics/tolerances verbatim. Also mine Thesis_Louis_Bocque.pdf for the prior attempt.
- `deep-research` skill (or, if the literature is broad, an `ultracode` Tier-4 workflow) — for the SOTA
  "how to exceed ChemZIP" survey in Phase 1 worker B. Use the wider tool only for that one slice.
- `python-refactor` skill — all Phase-2 code changes (type hints, purposeful docstrings, pytest one-per-
  function arrange/act/assert). The existing package was built to this standard.
- `plotting` skill — all benchmark figures: the user's palette, dual °C + K temperature axes (°C primary),
  PNG @ 400 DPI. (Already wired into scarfs/plotting/.)
- Standing-context doc: maintain a campaign memory file across sessions. If one does not already exist, ASK
  me for the filename before creating it (convention: `Claude_<TOPIC>.md`, e.g.
  `Claude_CHEMZIP_CAMPAIGN.md`, in the OneDrive `Claude_Setup` folder). Record: design decisions, chosen k,
  data-spec version, and benchmark margins per iteration.
- REUSE the existing `scarfs/` package and READ `DIAGNOSIS.md` + `BENCHMARK_PLAN.md` + `README_SCARFS_ML.md`
  before proposing anything — do not rebuild what exists.

=== What ChemZIP is (standing context — do not re-derive) ===
ChemZIP = Rubini & Rosic, Oxford, Chem. Eng. J. 2025 (arXiv:2502.08232). It learns INSTANTANEOUS chemical
source terms ω in a low-dimensional LATENT space: inputs (latent Z, T, p) → latent source terms ω_Z +
energy. It is NOT a state propagator, so residence time / Δt is NOT a model input — the CFD solver does the
time integration. It transports k≈3 latent scalars in CFD and reports R²>95% a-priori, within 10% of Fluent
a-posteriori, ~580× speedup. These are the FLOOR we must match, then beat. (A prior thesis replication —
"NeuralCoil" and a "reduced source-term" surrogate — FAILED; see DIAGNOSIS.md. Do NOT build on the thesis
approach; use it only as documented failure modes to avoid: near-inlet coverage gaps, latent drift, and
energy/atom inconsistency.)

=== Why we can beat it / the central constraint ===
We want to keep as much of the ~200+ species CRACKSIM kinetic detail as possible INSIDE CFD while keeping
calc time low. The COST KNOB is k = number of transported latent scalars. So the central engineering
question is: maximise recovered kinetic detail (especially the yield / QoI species) per unit k.

=== Repository state (already on disk) ===
- `scarfs/` Python package (REUSE):
  - `schema.py` (CRACKSIM/PFR database column contract), `data/` (Sobol case sampling + enrichment +
    Cantera flow finalisation), `models/` (scalers + Surrogate contract in common.py; features.py;
    physics.py = energy-from-rates + atom balance; nets.py; reduced.py; neuralcoil.py = latent encoder +
    latent-Z rate net + manifold projection; adapter.py), `training/` (config, datamodule with near-inlet
    weighting, losses, train entry point), `benchmark/` (loader, metrics R²/NMdAE/NRMSE, baselines,
    yields, apriori harness with ChemZIP-tolerance PASS/FAIL), `coupling/` (Fluent UDF/UDS templates +
    weight export + sanity checks), `plotting/` (figures, user palette, dual °C/K, 400 DPI).
  - `tests/` (numpy/pandas core fully unit-tested), `scripts/` (generate_database.py, local_sanity_check.py),
    `configs/` (train_reduced.json, train_neuralcoil.json), `requirements.txt`.
- Docs: `DIAGNOSIS.md`, `FIX_PROPOSAL.md`, `BENCHMARK_PLAN.md`, `README_SCARFS_ML.md`.
- CRACKSIM 1-D PFR data-gen: `Database_Generation_MB.py`, `ideal_reactor_models.py` (customPFR), plus the
  CRACKSIM DLLs, `chem.inp`, `networkfiles/`, and a sample `Database_Validation3.csv`.
- Git: branch `scarfs-fix` holds all the above; `main` holds only a baseline commit. Note the
  latent-source-term code in scarfs/models is close to ChemZIP and is a reasonable starting point — the
  design phase decides what to keep, replace, or extend; validate against the PAPER, not the thesis.

=== Environment & execution settings ===
- This machine has only numpy/pandas/matplotlib. torch + Cantera + Ansys Fluent run on the HPC. The
  convergence loop (train → CFD-benchmark → iterate) runs OUTSIDE this session — nothing here runs to
  completion, so do NOT use /goal.
- Orchestrator (this session) = Opus 4.8: decomposition, architecture judgement, accuracy-vs-cost
  tradeoffs, synthesis, integration.
- All research/design + adversarial-critic + Phase-2 builder subagents = Sonnet 4.6. No Haiku — every role
  is reasoning- or correctness-critical. Subagents cannot spawn subagents — you (orchestrator) do all
  dispatch, sequencing, and integration.
- START IN PLAN MODE. Phase 1 is read-only and produces design docs only (no model/data code modified).
  Present the design for approval; only after I approve do you exit plan mode and build.

=== Phase 1 — Research → Design → Propose (plan mode; fan out read-only subagents in parallel) ===
Dispatch concurrently; collect evidence-tagged findings ([CONFIRMED] = read in code/paper with cite;
[HYPOTHESIS] = inference + how to test); then synthesise:

A · ChemZIP faithful spec (use the `pdf` skill on ChemZIP.pdf): the EXACT architecture (encoder type &
   latent k, decoder, source-term network I/O), training-data recipe, CFD coupling, and every reported
   metric/tolerance VERBATIM with page refs → the reference "mimic" baseline.
B · "Exceed ChemZIP" SOTA (use `deep-research`/WebSearch): survey and RANK concrete levers to beat ChemZIP
   for a LARGE network at small k — each with citation, expected gain, cost/risk. Evaluate (do not adopt
   blindly): (1) nonlinear autoencoder vs ChemZIP's near-linear/PCA encoder for more detail per k;
   (2) conservation-by-construction (elemental balance + Σmass=1 in architecture/loss); (3) autoregressive
   stability — latent-Z rate net + per-iteration manifold projection Z←E·D(Z) + unrolled/rollout training +
   Lipschitz/spectral control + noise injection; (4) QoI-weighted / hierarchical decoding so light olefins,
   BTX, and coke precursors stay accurate at low k; (5) active-learning / error-driven resampling of CFD-
   visited states ChemZIP didn't cover; (6) neural-ODE / continuous-time vs source-term (note, but do NOT
   default away from source terms — they are the proven, residence-time-free choice).
C · CRACKSIM data specification (the concrete "what to generate" deliverable): reactor model (1-D PFR via
   customPFR + Cantera + CRACKSIM, already in repo); sampling design (Sobol/LHS over inlet T, P, steam
   dilution, feed, reactor length / residence-time spread, and wall heat-flux shapes); how to bound the
   sampled thermal envelope to what the CFD will actually visit (a NON-REACTING CFD precursor of the real
   geometry); what to STORE per axial point (all ~200+ species mass fractions, T, P, net production rates
   ω_i [kg/m³/s] as targets, energy source, cp/cv, transport props); mandatory COVERAGE (near-inlet/low-
   conversion, high-T near-wall, full radical pool, full severity range, explicit holdouts); target sample
   count; and how it maps onto Database_Generation_MB.py + scarfs.data so it is generation-ready.
D · Repo reuse audit: what in scarfs/ is reusable as-is vs needs change for the mimic+exceed design, so
   Phase 2 maximises reuse.

Synthesise into a DESIGN proposal: (1) the faithful ChemZIP baseline; (2) the improved architecture with
the chosen beat-it levers; (3) the CRACKSIM DATA SPEC; (4) a benchmark plan with ChemZIP tolerances as the
FLOOR and explicit EXCEED targets (per-species accuracy, accuracy-vs-k curve, a-posteriori stability,
speedup); (5) an accuracy-vs-k / detail-vs-cost ABLATION plan to find the smallest k meeting the detail
target.

Adversarial check: dispatch one critic subagent (Sonnet) to REFUTE the top 1–2 beat-it levers and the data
spec against the code + papers before the checkpoint; re-rank if a lever doesn't survive.

➡ STOP. Present the faithful spec, the improved-design proposal, the CRACKSIM data spec, the benchmark plan
(with the critic's verdict), and the open questions below. Ask for my feedback. Change no code until I
approve.

Open questions to resolve at the checkpoint (assume a default + flag it if I don't answer):
- CFD solver/geometry + whether a non-reacting precursor exists for envelope sampling (assume Fluent + a
  precursor must be defined).
- Feedstock(s) and the priority QoI species / acceptance yields (assume ethane; light olefins + BTX +
  conversion).
- The a-posteriori reference truth (assume detailed-chem CRACKSIM-in-Fluent and/or experiment, supplied
  later).
- Cost budget: max acceptable k and target speedup (assume "smallest k clearing the detail target",
  speedup ≳ ChemZIP's).
- Branch: continue on `scarfs-fix` or start fresh (assume continue on scarfs-fix).

=== Phase 2 — Build (after approval; exit plan mode) ===
On the agreed branch, implement the approved design REUSING scarfs/ (only the agreed changes — no
abstractions beyond the design): the faithful baseline + the improved variant(s), the training pipeline,
the a-priori harness, and the generation-ready CRACKSIM data spec. Fan out builders (Sonnet) on non-
overlapping scarfs/ modules; integrate serially yourself. Deliver an HPC-runnable training + data-gen
pipeline (entry points, configs, run instructions). Run only a small LOCAL subset sanity check proving the
pipeline executes; full data-gen (CRACKSIM) and training (torch) run on the HPC.

=== Phase 3 — Benchmark + active-learning loop (scaffolding; runs on HPC) ===
Wire the a-priori benchmark vs ChemZIP tolerances + the exceed targets; prepare the a-posteriori CFD
coupling; document the closed active-learning loop (train → CFD-benchmark → find under-covered states →
regenerate CRACKSIM data → retrain) as the external HPC convergence loop. Report: what changed, the
accuracy-vs-k story, the a-priori sanity results, exactly how to run data-gen + training + the CFD
benchmark on the HPC, and the remaining gaps vs / margin over ChemZIP.

=== Guardrails ===
- Ground every claim in code or the papers; cite source. Tag [CONFIRMED] vs [HYPOTHESIS]. Invent no
  numbers, architectures, or results.
- Read-only until I approve the design. Preserve behaviour elsewhere; flag suspected bugs separately, do
  not silently rewrite. Cap research/build subagents per phase. The HPC/torch/Cantera/Fluent steps are mine.
```

---

*Saved so it survives a context clear. The latent-source-term code already in `scarfs/models/` is close to
ChemZIP and a reasonable starting point; the design phase decides what to keep vs replace, validating
against the paper, not the failed thesis.*

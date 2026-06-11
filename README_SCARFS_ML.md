# SCARFS ML surrogate — merged pipeline (branch `scarfs-merge`)

One model merged from two ChemZIP-style source-term surrogates, with the energy-prediction
deviation resolved by design. **Read first:** [`MERGE_DESIGN.md`](MERGE_DESIGN.md) (the approved
merge + energy-fix design and acceptance criteria), then the Phase-1 docs
[`DIAGNOSIS.md`](DIAGNOSIS.md) / [`FIX_PROPOSAL.md`](FIX_PROPOSAL.md) /
[`BENCHMARK_PLAN.md`](BENCHMARK_PLAN.md) (still valid for the parent design).

Headline: the energy deviation was **not** an enthalpy-basis or unit problem (audits pass at
relRMSE ≈ 3e-5). It came from tail-suppressing training, a rank-k decode bottleneck, species-drop
hidden-state loss, and front under-resolution — all removed in the merged model (split heads,
full-composition standardized encoder, rate-tied energy, tail-faithful losses, front-adaptive data).

## Package layout

```
scarfs/
  schema.py            # column contract: CSV + parquet (dYdt/wdot families, pseudo-species guard,
                       #   `Reaction heat absorption` = the ONLY energy target)
  data/                # broadened + enriched case sampling, front-adaptive storage, D-sweep, sign audit
  models/              # scalers (incl. standard/linear composition), thermo (cantera-free NASA7),
                       #   features, physics (+ exact atom projection), nets, reduced, neuralcoil
                       #   (+ split-head MergedCoil), adapter (TorchSurrogate.from_merged_bundle)
  training/            # config, datamodule (case GroupKFold, tail strata, enthalpy weights),
                       #   losses (merged composite incl. rate-tied energy + Lagrangian rollout), train
  benchmark/           # apriori (ChemZIP floor) + energy (§5 acceptance suite) + parents (beats-both
                       #   protocol) + feasibility (kNN pre-gate) + ablation (k sweep) + baselines
  diagnostics/         # energy unit/coverage audits, state-ambiguity, conservation, sanity,
                       #   front-resolution — markdown+CSV reports
  coupling/            # legacy templates + codegen.py: full merged-UDF C generation (UDS latent
                       #   transport, rate-tied energy head, property hooks, OOD clamps + telemetry,
                       #   TUI, folded BCs, compiled standalone forward test, consistency report)
  plotting/            # palette, dual °C+K, 400-DPI figures (+ energy parity/tail/front/k figures)
configs/               # train_merged.json (k=8 exceed), train_merged_mimic.json (parent-2 baseline),
                       #   legacy reduced/neuralcoil configs
scripts/               # generate_database.py, local_sanity_check.py (legacy + merged end-to-end),
                       #   verify_feasibility_stride5.py, benchmark_parents.py
tests/                 # pytest suite (456 tests; incl. real-data fixture tests/data/stride6_sample.parquet)
chem_ForTransport.yaml # mechanism thermo source (NASA7) — validated vs DB absorption at relRMSE 2.9e-5
```

## Environments

| Task | Where | Needs |
|---|---|---|
| Tests, sanity, audits, feasibility, UDF codegen + C forward test | local or HPC | `.venv` (numpy, pandas, pyarrow, scipy, sklearn, matplotlib, torch CPU, pytest; `cc` for the C check) |
| Database generation (CRACKSIM) | HPC | cantera + `SA_CRACKSIM.dll` + `chem.inp`/`transport_chemkin.dat` |
| Full training + k ablation | HPC (GPU optional) | torch |
| A-posteriori CFD | HPC | Ansys Fluent (compile the generated UDF) |

Local setup: `uv venv .venv --python 3.12 && uv pip install -p .venv/bin/python numpy pandas
matplotlib scipy pyarrow torch pyyaml pytest scikit-learn`.

## 1. Verify locally

```bash
.venv/bin/python -m pytest -q                  # 456 passed, 1 skipped (cantera-gated)
.venv/bin/python scripts/local_sanity_check.py # legacy path + MERGED path end-to-end:
                                               #   tiny train -> §5 energy gates -> audits ->
                                               #   UDF codegen + compiled forward test -> figures
```

The merged sanity trains on the in-repo stride6 fixture by default; point `--merged-db` at a
stride5 slice for a stronger check. A toy model FAILS the §5 gates by design — the point is that
the machinery runs and gates honestly.

## 2. Bootstrap data now, regenerate for certification (HPC)

- **Bootstrap (now)**: train/benchmark on the colleague's `TEST_ETHANE_LOW_sobol_stride5.parquet`
  (102,833 rows) with case-ID GroupKFold splits. Results are labeled *bootstrap, non-certifying*
  (front under-resolved; see `scarfs/diagnostics/front_resolution.py`).
- **Regenerate (HPC)**: `scripts/generate_database.py` with the merged generator — front-adaptive
  storage (`storage.mode="front_adaptive"`), D-sweep incl. 0.0306 m, tail enrichment, all-species
  `Y_*`/`dYdt_*` + `Reaction heat absorption` + tau/z, generator sign-audit. Regenerate a
  front-adaptive **test** DB too — §5 certification runs only on that.

## 3. Pre-gate k, then train (HPC)

```bash
# kNN feasibility pre-gate (also re-verifies the Phase-1 numbers on re-split stride5):
.venv/bin/python scripts/verify_feasibility_stride5.py --database <stride5> --columns-projection --out feas.md
# train the merged model / the parent-2 mimic baseline:
python -m scarfs.training.train --config configs/train_merged.json
python -m scarfs.training.train --config configs/train_merged_mimic.json
# k ablation (energy/tail-metric selection, plan §4 E-e):
python -c "from scarfs.benchmark.ablation import run_k_ablation; ..."   # ks=[4,6,8,12,16]
```

Outputs per run: `model.pt`, `scalers.pkl`, `spec.json` (incl. energy calibration + export safety
stats), `metrics.json` (incl. val absorption metrics for BOTH energy paths). No winsorization, no
output bounds anywhere in the energy path; tail-stratified + enthalpy-aware weighting are on by
config.

## 4. Benchmark — must beat BOTH parents

```bash
.venv/bin/python scripts/benchmark_parents.py --database <db> \
    --colleague-outputs /path/to/reduced_chem_ml/outputs \
    --merged-bundle runs/merged --mimic-bundle runs/mimic
```

Identical case splits + identical metrics: ChemZIP floor (R²>95% all QoI, rel-err <10%, beats
frozen/NN/mean baselines — `scarfs/benchmark/apriori.py`) **and** the §5 energy suite
(`scarfs/benchmark/energy.py`: per-case tail gates, front localization, ∫S_E dτ budget; absolute
floors reported, never gated). Run the diagnostics audits alongside
(`scarfs.diagnostics.run_energy_unit_audit` / `run_energy_coverage_audit` / `run_ambiguity_collapse`).

## 5. Fluent UDF (HPC)

```python
from scarfs.coupling.codegen import export_merged_udf
export_merged_udf("runs/merged", "udf_out")   # merged_coil_udf.c/.h + TUI + folded BCs +
                                              # forward test + export-consistency report
```

Compile `merged_coil_udf.c` in Fluent; hook the k UDS sources, the adjust (manifold projection +
species reconstruction), and the energy source (S_h = −absorption head; clamp = 1.3× max train
absorption; under-relaxation ANNEALS to 1.0; clamp/OOD telemetry in UDMs). Run the standalone
forward test on the HPC login node first (`cc merged_coil_forward_test.c -lm && ./a.out`).
A-posteriori pass: outlet T ±10 K, major yields within 10% vs the detailed-chemistry reference,
stable convergence, telemetry showing clamps inactive in-envelope.

## Known limitations

- Accuracy numbers require the full HPC training run (local runs are pipeline proofs).
- 20 of 213 species lack thermo blocks in `chem_ForTransport.yaml` (cannot be energy-active;
  their direct enthalpy-flux share was measured negligible — see `metrics.json`
  `thermo_missing_species`).
- Transport properties in the UDF use decoded-composition NASA cp but database-median fallbacks
  for μ/k (tracked gap, inherited from the colleague's design).
- stride6 is a *different campaign* than stride5 (disjoint envelope corners) — use it only as a
  distribution-shift diagnostic, never as held-out-of-stride5.

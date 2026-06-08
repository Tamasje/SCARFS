# SCARFS ML surrogate — fix pipeline (Phase 2)

This is the `scarfs-fix` branch deliverable for the Tier-3 diagnosis + fix of the ChemZIP-style
source-term surrogate. **Read first:** [`DIAGNOSIS.md`](DIAGNOSIS.md) (ranked root causes),
[`FIX_PROPOSAL.md`](FIX_PROPOSAL.md) (fixes F1–F5 tied to causes), [`BENCHMARK_PLAN.md`](BENCHMARK_PLAN.md)
(ChemZIP-derived acceptance tolerances).

Headline: the failure was **not** a missing residence-time feature (both ChemZIP and the thesis
models are *source-term* surrogates — the CFD solver integrates). The fixes target the real causes:
near-inlet coverage (RC-1), NeuralCoil latent drift (RC-2), energy/atom consistency (RC-3), and
domain shift / OOD (RC-4).

## Package layout

```
scarfs/
  schema.py            # canonical column contract for the CRACKSIM/PFR database
  data/                # F1/F4: broadened + near-inlet/high-T enriched case sampling
  models/              # scalers + contract (common), features, physics, nets, reduced, neuralcoil, adapter
  training/            # config, datamodule (F1 weighting), losses (F2/F3), train (entry point)
  benchmark/           # a-priori harness, metrics, baselines, yields, (a-posteriori comparison)
  coupling/            # Fluent UDF/UDS templates + weight export + sanity checks
  plotting/            # user palette, dual °C/K, 400-DPI figures
configs/               # train_reduced.json, train_neuralcoil.json
scripts/               # generate_database.py, local_sanity_check.py
tests/                 # pytest suite (numpy core fully tested locally)
```

## Environments

| Task | Where | Needs |
|---|---|---|
| Tests, a-priori harness, plotting, sanity | local or HPC | numpy, pandas, matplotlib (+ scipy/pyarrow optional) |
| Database generation (CRACKSIM) | HPC | cantera + `SA_CRACKSIM.dll` + `chem.inp`/`transport_chemkin.dat` |
| Training (both surrogates) | HPC | torch |
| A-posteriori CFD | HPC | Ansys Fluent (compile the UDF templates) |

`pip install -r requirements.txt` for the full set. The local box here runs only the core (no torch
/ cantera), which is why training + CFD are HPC steps.

## 1. Verify locally

```bash
python -m pytest -q                 # 166 pass, 3 skip (parquet/cantera env-gated)
python scripts/local_sanity_check.py   # data -> features -> benchmark -> figures, on real data
```

## 2. Regenerate the database (HPC) — fixes F1/F4

```bash
# coverage check anywhere (NumPy only):
python scripts/generate_database.py --no-flow
# finalise inlet flow with Cantera, then feed the manifest to the CRACKSIM harness:
python scripts/generate_database.py --mech chem.yaml --out case_manifest.json
```

The manifest (`scarfs.data.build_cases`) replaces the leftover narrow 18-case config in
`Database_Generation_MB.py` with ~2160 quasi-random cases plus **near-inlet/low-conversion** and
**high-T near-wall** enrichment. Plug the cases into the existing multiprocessing harness (its
`run_case` consumes the same dict schema) to produce `Database_FINAL.parquet`.
Tune ranges/enrichment in `scarfs/data/config.py` (`DataGenConfig`).

## 3. Train (HPC) — fixes F2/F3

```bash
python -m scarfs.training.train --config configs/train_reduced.json
python -m scarfs.training.train --config configs/train_neuralcoil.json
```

Outputs under `output_dir`: `model.pt`, `scalers.pkl`, `spec.json`, `metrics.json`. The datamodule
up-weights near-inlet states (F1, `inlet_weight`) and the key products (`species_weights`). The
reduced model derives the energy source from predicted rates (F3, no free head); NeuralCoil uses a
latent-space rate net + manifold-consistency + noise injection (F2).

## 4. A-priori benchmark (offline)

```python
from scarfs.benchmark.loader import load_database, infer_schema
from scarfs.benchmark.apriori import AprioriConfig, holdout_split, run_apriori
from scarfs.models.adapter import TorchSurrogate   # wraps a trained model as a Surrogate
# build TorchSurrogate(model, scalers, spec, schema); then:
report = run_apriori(surrogate, train_df, test_df, schema, AprioriConfig())
print(report.summary())     # PASS requires R2>0.95 all QoI, median rel-err<10%, beats baselines
```

Targets (verbatim ChemZIP): R² > 95 % on all QoI; relative-error histograms centered < 10 %.

## 5. A-posteriori benchmark (coupled CFD, HPC)

```python
from scarfs.coupling.export import export_bundle   # serialise weights+scalers to text the UDF reads
```

1. Export the trained model bundle. 2. Compile `scarfs/coupling/fluent_reduced_source.c` (reduced) or
`fluent_neuralcoil_uds.c` (NeuralCoil, with the **latent-manifold projection** that fixes RC-2) in
Fluent. 3. Run vs your reference (detailed-chem CRACKSIM-in-Fluent / 1-D PFR / experiment — pluggable).
Pass = yields & T within 10 % of reference **and** stable convergence (no freeze, no blow-up). See
`scarfs/coupling/README.md`.

## Fix → cause map

| Fix | Cause | Where |
|---|---|---|
| F1 near-inlet enrichment + weighting | RC-1 | `data/`, `training/datamodule.py` |
| F2 latent-Z rate net + manifold projection + noise | RC-2 | `models/neuralcoil.py`, `training/losses.py`, UDS template |
| F3 energy-from-rates + atom-balance penalty | RC-3 | `models/physics.py`, `training/losses.py` |
| F4 broad envelope + OOD input clipping | RC-4 | `data/config.py`, coupling templates |
| F5 `customPFR` velocity `/A` | flagged bug | `ideal_reactor_models.py` |

(Also fixed: `np.infty` → `np.inf` in `ideal_reactor_models.py` for NumPy ≥ 2 import compatibility.)

## Known gaps vs ChemZIP tolerances
- Accuracy numbers require the HPC training run; not yet measured (no local torch/data).
- Atom balance on the reduced active set is a soft pressure, not exact closure (needs full species).
- The exact 30-species active set is configurable; defaults to the molecular set — set explicitly to
  match the thesis if required.
- A-posteriori reference is pluggable and **awaiting your reference data**.

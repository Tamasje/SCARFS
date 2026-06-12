# RUNBOOK — v2 database generation on the Windows machine

Target: the ideal-DB spec approved 2026-06-12 (front-adaptive trajectories + off-manifold
state cloud + verification gates), generated where `SA_CRACKSIM.dll` loads. T cap = 1423.15 K
(confirmed). Tiers let you **prove the pipeline on a small subset first**: smoke ⊂ pilot ⊂ full
are prefixes of the same Sobol streams, so nothing run early is wasted.

## 0. Prerequisites (once)

1. Pull this branch on the Windows checkout (`C:\Users\tbuzogan\Documents\SCARFS` or a fresh
   clone): branch `overnight-diagnosis-20260612` (or wherever this lands after merge).
2. Repo root must contain: `SA_CRACKSIM.dll` (untracked — copy from your existing checkout),
   `chem.inp`, `transport_chemkin.DAT` (both tracked). `chem.yaml` is auto-built by the script
   via `ck2yaml` on first run.
3. Python env with: `numpy pandas pyarrow scipy cantera` (the env that ran the stride5
   generation works as-is). No torch needed for generation.

## 1. Smoke (minutes) — pipeline shakedown + gates

```bat
python scripts\generate_database_v2.py --tier smoke --n-cpu 4
```

10 cases (2 per regime), solves on 400-point grids, writes `out_v2\smoke.parquet`, then
**auto-runs the gates**:

- **GATE A — DLL/dYdt consistency**: re-evaluates the DLL at stored states; must reproduce
  the stored `dYdt_*` columns to max rel diff < 1e-6. Reported TWICE: in-worker (post-solve
  context, printed by worker 0) and in a fresh process. The diagnostics distinguish the
  failure modes: `zero-recompute fraction ≈ 1.0` ⇒ NetRates_C returns zeros outside a solve
  context (Fortran statefulness — in-worker PASS + fresh-process FAIL confirms it, and then
  off-manifold generation simply runs one warm-up solve per worker); `double-call max |diff|
  > 0` ⇒ the DLL is stateful even between identical calls; uniform large rel diffs with
  nonzero recompute ⇒ ordering/units (hard stop — send me the output).
- **GATE C — front resolution**: pass/fail applies to POLICY-chosen jumps only (storage
  skipping decisions). Single-solver-step jumps are grid-limited — no storage policy can
  beat the solve grid — and are reported as a `--n-points` multiplier recommendation
  (e.g. "factor ~3x" ⇒ rerun with `--n-points 1200` for the production tiers).
- Sign audit (E-c positivity lineage) is reported per tier; `material_negative=True`
  (any case min < −1e6 J/m³/s) is informative, not fatal — it triggers the documented
  shifted-softplus escape hatch on the training side.

Optional standalone gate run against the OLD database for comparison:
```bat
python scripts\generate_database_v2.py --gates --reference TEST_ETHANE_LOW_sobol_stride5.parquet
```
(Expect GATE A to PASS — same pipeline lineage — and GATE C to FAIL loudly: that failure is
the documented reason v2 exists.)

## 2. Pilot (~1,230 cases) — the prove-it-works subset

```bat
python scripts\generate_database_v2.py --tier pilot --n-cpu 10 --skip-existing
```

~6% of every regime stream (body 720, inlet 150, high-T 150, tail 120, deep-conversion 90).
Rough duration: stride5 averaged tens of seconds per case at 100 grid points; at 400 points
expect ~1–4 min/case → **roughly 4–8 h on 10 workers** (the smoke run prints `cases/h` — use
it to extrapolate before committing). Output `out_v2\pilot.parquet` (~100–150k rows expected;
front-adaptive density varies per case). Dropped-case stats land in `out_v2\pilot_audits.json`
— hot-corner cases hitting the 1423 K cap are *expected* drops, the envelope samples up to it.

**Add a pilot off-manifold cloud** (cheap — single-point calls, no integration):
```bat
python scripts\generate_database_v2.py --off-manifold 60000 --source out_v2\pilot.parquet --n-cpu 10
```

### Train on the pilot (proof step)

Copy `out_v2\pilot.parquet` (and the off-manifold file) to the Mac — or train on Windows if
torch is installed there — and run the standard pipeline:

```bash
.venv/bin/python scripts/gpu_train_ablation.py --database <path>/pilot.parquet \
    --ks 12 --epochs 300 --head-finetune 40 --out runs/pilot_v2
.venv/bin/python scripts/concluding_figures.py --ablation-dir runs/pilot_v2 --database <path>/pilot.parquet
```

Success criteria for the proof: training runs end-to-end on v2 data, GATE-C-grade front
resolution shows up as a *much* lower per-case ∫S_E dτ p95 than the stride5 bootstrap (0.47),
and absorption metrics land in the same band as the stride5 run or better. Then proceed.

## 3. Full + test + off-manifold (the production run)

```bat
python scripts\generate_database_v2.py --tier full --n-cpu 12 --skip-existing
python scripts\generate_database_v2.py --tier test --n-cpu 12
python scripts\generate_database_v2.py --off-manifold 1000000 --source out_v2\full.parquet --n-cpu 12
```

- `--skip-existing` resumes after interruptions AND reuses every pilot case (same stream).
- full ≈ 20,500 cases → extrapolate from the smoke/pilot `cases/h` figure; days-scale on one
  workstation is normal. Per-case parquets commit atomically — you can stop/restart freely.
- `test` (~3,000 cases) uses an **independent Sobol stream** and CaseIDs ≥ 1,000,000 — this is
  the §5 certification set; never train on it.

## 4. What comes back to me

`out_v2/full.parquet`, `out_v2/test.parquet`, `offmanifold_*.parquet`, plus the three
`*_manifest.json` / `*_audits.json` files. Then: full training + k-ablation + §5 certification
on the front-adaptive test set, and the UDF export for the Fluent run.

## Troubleshooting

- **Worker init "CRACKSIM initialise failed"** — DLL not in repo root, or 32/64-bit Python
  mismatch with the DLL (use the same Python that ran stride5 generation).
- **`ck2yaml` not found** — `pip install cantera` provides it; or pre-build `chem.yaml` once
  with your existing command.
- **Many dropped cases in `tail`/`high_T` regimes** — expected (cap at 1423.15 K); the
  audits file lists per-case reasons. If body-regime drops exceed ~10%, send me the audits.
- **A case hangs** — solver struggling at a hard corner; per-case wall prints identify it,
  and killing/restarting with `--skip-existing` skips completed work. The 400-point grid can
  be lowered per run with `--n-points 250` at some front-resolution cost (gates will tell).

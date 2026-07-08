# SCARFS stable-latent-ODE surrogate — Fluent deployment (k=8)

ML surrogate for ethane steam-cracking chemistry (ChemZIP-style). Fluent transports **8 latent
scalars** (UDS) and this UDF supplies their source terms, the decoded species field, the energy
source, and optional transport properties. Self-contained: the model weights are baked into the
`.h` — you do **not** need Python, the training repo, or the database to run it.

## Files

| file | required? | what it is |
|---|---|---|
| `merged_coil_udf.c` | **yes** | the Fluent UDF — compile this in Fluent |
| `merged_coil_udf.h` | **yes** | model weights + macros (`#include`d by the `.c`); 9.6 MB |
| `fluent_merged_setup.tui` | **yes** | TUI script that hooks the 8 UDS, source terms, and property functions |
| `inlet_bc.txt` / `inlet_bc.csv` | **yes** | inlet boundary condition: the 8 latent-z values (+ decoded composition for reference) |
| `merged_coil_forward_test.c` | optional | standalone self-check (no Fluent) — compile + run to confirm the C reproduces the model |
| `export_consistency_report.txt` | reference | numpy↔torch parity at export (Y 2.9e-4 / field 2.2e-5 / S_h 4.5e-7) |

## Deploy in Fluent

1. Put `merged_coil_udf.c` and `merged_coil_udf.h` in your Fluent working dir.
2. Compile the UDF (Fluent: User-Defined → Functions → Compiled → add `merged_coil_udf.c` → Build → Load).
3. Allocate **8 UDS** and enough UDMs (see `MC_TOTAL_UDM` in the `.h`), then run
   `fluent_merged_setup.tui` (File → Read → Journal) — it hooks:
   - `DEFINE_SOURCE mc_latent_uds_0..7_source` → the 8 UDS equations (`S_i = ρ·f_i(z)`),
   - `DEFINE_ADJUST mc_manifold_project` → per-iteration latent clamp + decoded species/UDM update,
   - `DEFINE_SOURCE mc_energy_source` → energy equation source `S_h`,
   - `DEFINE_PROPERTY mc_viscosity`, `mc_thermal_conductivity` (optional).
4. Set the inlet UDS values from `inlet_bc.txt`.
5. Solve. Monitor the telemetry UDMs (OOD flag, latent-/energy-clamp counts, last `S_h`).

**Transport model:** this build is the **stable latent ODE** (`#define MC_DIRECT_TRANSPORT`): the latent
is advanced directly by `dz/dτ = sinh(ω_Z)·s_Z − β·z` (β=`MC_BETA`≈0.265), with **no E∘D re-projection**
(`mc_project` is identity; the DEFINE_ADJUST only clamps + decodes for readout). It is a genuine ODE, so it
**converges with finer timesteps** — no special under-relaxation of the latent needed.

## Validate before trusting it (recommended)

```
cc -O0 -lm merged_coil_forward_test.c -o mc_fwd_test && ./mc_fwd_test
```
Expect: `PASS: all 6 states within rel_tol=1.00e-06`. This confirms the C on your compiler reproduces
the trained model bit-for-bit before you wire it into Fluent.

## Performance (0D/1D prescribed-thermo proxy vs the previous reproject model)

| metric | previous (stageBt2) | this (stable ODE) |
|---|---|---|
| species-composition reconstruction | 2.3e-2 | **6.6e-4** (35× better) |
| closed-loop composition drift | 2.5e-2 | **7.0e-3** (3.5×) |
| integrated energy ∫S_E error | 0.33 | **0.10** (3.3×) |
| latent stability (envelope clamp) | frequent | **barely clamps** |
| point-wise energy a-priori factor | 13× | 5.9× |

**Honest caveats:** (1) point-wise energy accuracy is lower than the previous model (5.9× vs 13×) — the
integrated/closed-loop energy is nonetheless better; watch local temperature hot-spots. (2) All numbers
above are a 0D/1D prescribed-thermo proxy — **this Fluent run is the real validation** of ∫S_E under
coupled T + diffusion + the steady solve.

## Provenance
Generated from `runs/stableode_k8_bal` by `scarfs.coupling.codegen.export_merged_udf`. Method + full
results: `STABLE_LATENT_ODE.md` in the repo root. Reproduce: `bash scripts/run_recon.sh` →
train `configs/train_stableode_k8_bal.json` → export.

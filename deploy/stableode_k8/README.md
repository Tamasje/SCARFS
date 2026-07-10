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
   - `DEFINE_ADJUST mc_manifold_project` → per-iteration latent clamp + decoded species/UDM update (also stores mean MW),
   - `DEFINE_SOURCE mc_energy_source` → energy equation source `S_h`.
4. Hook the **material properties** (Materials → your mixture material → user-defined):
   - Density → `mc_density` — **default: ideal gas** `ρ = P·M/(R·T)`, with `M` the mean molecular
     weight computed from the decoded composition (stored in `UDM_WMEAN` by the adjust hook).
   - Specific heat (Cp) → `mc_specific_heat` (`DEFINE_SPECIFIC_HEAT`) — **composition-dependent**:
     `cp = Σ yᵢ·cpᵢ(T)`, `h = Σ yᵢ·(hᵢ(T)−hᵢ(Tref))` from NASA7, over the Fluent species `yᵢ`
     (requires the species-mixture setup below — see `fluent_species_order.txt`). Set
     `MC_CP_COMPOSITION_DEPENDENT 0` in the `.h` to fall back to the T-only table instead.
   - Viscosity → `mc_viscosity`; Thermal conductivity → `mc_thermal_conductivity` (transport head).
   - Speed of sound → `mc_speed_of_sound` (ideal-gas; only needed for a density-based/compressible solver).
5. Set the inlet UDS values from `inlet_bc.txt`.
6. Solve. Monitor the telemetry UDMs (OOD flag, latent-/energy-clamp counts, last `S_h`, mean MW).

**Density switch:** the default is MW→ideal-gas. A `#define MC_DIRECT_DENSITY` placeholder exists in
the `.h` for direct density prediction — it is **off** (no density head is trained); flip it and wire a
UDM only once a density head exists.

## Composition-dependent Cp — required Fluent setup (`MC_CP_COMPOSITION_DEPENDENT 1`, default)

Fluent's `DEFINE_SPECIFIC_HEAT` gets no cell — the *only* way it can vary Cp per cell is through the
species mass fractions `yᵢ`. So Cp rides on a **species mixture** whose composition we drive from the
latent (no species PDEs are solved). One-time case setup:

1. **Species model on**, fluid = a **mixture material** whose species are defined in **exactly** the
   order in `fluent_species_order.txt` (61 energy-active species, then **H2O last as the bulk**). The
   order must match `MC_EA_TO_INPUT` / `MC_NASA_*` in the `.h`.
2. **Turn the species transport equations OFF** (Solution Controls → Equations → uncheck the species).
   The 8 latent UDS carry the chemistry; the species field is only a property carrier.
3. `mc_manifold_project` (the adjust hook) writes the decoded composition into `C_YI` each iteration;
   `mc_specific_heat` then returns the true composition-weighted Cp + sensible enthalpy.

This carries a per-cell species-mass-fraction field (memory, **not** extra PDEs). If you'd rather avoid
the species setup, set `MC_CP_COMPOSITION_DEPENDENT 0` in the `.h` → the T-only mixture table is used
(composition-independent, the LatentV22 compromise). Validated: the C `yᵢ`-weighted NASA7 Cp/enthalpy
matches the Python reference to machine precision (rel-diff 0) at 900/1100/1300 K.

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

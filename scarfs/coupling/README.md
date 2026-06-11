# SCARFS Coupling — Fluent UDF scaffolding

## Overview

This directory contains the scaffolding to couple a trained SCARFS surrogate into
Ansys Fluent as user-defined source terms (UDFs/UDS).

**Fluent cannot be run in this repository.** The C UDF templates are compiled and
loaded by the user on the HPC cluster.  The Python export utilities (`export.py`)
and sanity checks (`sanity.py`) run locally.

---

## Units convention

| Quantity | Symbol | Unit | Notes |
|---|---|---|---|
| Species net production rate | R_i | kg m⁻³ s⁻¹ | Fluent species source term |
| Volumetric energy source | S_E | J m⁻³ s⁻¹ = W m⁻³ | Fluent energy source term |
| Mass fractions | Y_i | — (dimensionless) | Input to surrogate |
| Temperature | T | K | Input to surrogate |
| Pressure | P | Pa | Input to surrogate |

Water (H2O) is a **fixed diluent** (thesis Ch. 5.6).  It is excluded from the
surrogate's active/encoder species set.  The CFD solver handles the Nth-species
constraint; no source term is registered for H2O.

---

## Step 1 — Export the trained surrogate (Python, local or HPC login node)

```python
from scarfs.coupling.export import (
    ModelBundle, export_bundle, export_mlp_weights, export_scalers, export_active_species
)

# Build a bundle from your trained model objects:
bundle = ModelBundle(
    layers   = my_mlp.to_layer_list(),   # list of (W, b, activation) tuples
    scalers  = {
        "composition": comp_scaler,      # CompositionScaler fitted on training Y
        "thermo":      thermo_scaler,    # StandardScaler fitted on [T, P, 1/T, lnT]
        "rates":       rate_scaler,      # ArcsinhScaler fitted on R_i columns
    },
    active_species = list(schema.active_species()),  # ordered, NO H2O
    name           = "surrogate",
)

paths = export_bundle(bundle, out_dir="hpc_export/")
# Creates:
#   hpc_export/surrogate_weights.txt
#   hpc_export/surrogate_scalers.txt
#   hpc_export/surrogate_species.txt
```

For NeuralCoil export four separate weight files (encoder, decoder, rate net,
energy net) using `export_mlp_weights()` individually, and the shared scalers
with `export_scalers()`.

### Verify the round-trip before copying to HPC

```python
from scarfs.coupling.export import load_bundle
b2 = load_bundle("hpc_export/", name="surrogate")
# All arrays should match to machine precision.
```

---

## Step 2 — Copy export files to HPC

Copy the `hpc_export/` directory alongside the compiled UDF shared library.
The C templates read the files by **relative path from the Fluent working directory**.
Update the `*_FILE` defines at the top of each `.c` file to absolute paths if the
working directory differs from where the files are stored.

---

## Step 3 — Edit the C template (resolve all TODO markers)

Open `fluent_reduced_source.c` (reduced surrogate) or `fluent_neuralcoil_uds.c`
(NeuralCoil) and resolve every `/* TODO */` comment:

1. Set `N_ACTIVE` to match the number of lines in `surrogate_species.txt`.
2. Wire `C_YI(c, t, i)` to the correct Fluent species index for each active species.
   The Fluent species index is determined by the order in your mixture material, NOT
   by the species.txt order.  Build a mapping array `int fluent_idx[N_ACTIVE]`.
3. Confirm `C_P(c, t)` returns absolute pressure (operating + gauge) in Pa.
4. For NeuralCoil: set `K_LATENT` and `N_SC` to match your encoder.
5. Duplicate the per-species / per-latent `DEFINE_SOURCE` stubs for all N_ACTIVE or
   K_LATENT functions.

---

## Step 4 — Compile and load in Fluent (on HPC)

1. Open your Fluent case file.
2. Go to **User Defined → Functions → Compiled UDFs**.
3. In the *Source Files* box, add `fluent_reduced_source.c` (or `fluent_neuralcoil_uds.c`).
4. Click **Build** (Fluent invokes the C compiler; check the console for errors).
5. Click **Load** to register the UDF functions.

---

## Step 5 — Fluent panel wiring (reduced surrogate)

For each active species in the Fluent Species panel:
- Set the **Source Terms** → *Chemical Species Source* to the corresponding
  `scarfs_src_species_<i>` function.

For the energy equation:
- Set the **Source Terms** → *Energy Source* to `scarfs_src_energy`.

Hook the initialisation:
- **User Defined → Initialization** → select `scarfs_reduced_init`.

---

## Step 6 — NeuralCoil UDS setup

1. Go to **User Defined → Scalars** and create **k** User-Defined Scalars
   (one per latent dimension, k = 6 in the thesis).
2. Name them `Z_0` through `Z_{k-1}`.
3. Set the diffusivity of each UDS to the NeuralCoil diffusivity sub-net output
   (or to molecular diffusivity as a first approximation).
4. Hook **User Defined → Adjust** → `scarfs_nc_adjust`  (manifold projection runs here).
5. For each UDS transport equation, set the Source Term to `scarfs_nc_uds_source_<j>`.
6. For the energy equation, set the Source Term to `scarfs_nc_energy_source`.
7. Hook **User Defined → Initialization** → `scarfs_nc_init`.
8. Set initial UDS values: encode the inlet composition Y_inlet through E to get Z_inlet.

---

## RC-2 guard: manifold projection (NeuralCoil only)

DIAGNOSIS.md RC-2 documents that the transported latent Z drifts off the encoder
manifold, causing off-manifold residuals to grow from 7 to 2777 in 326 iterations
and driving Y_C2H4 to zero (thesis §5.5.2.3-4).

**The fix (F2)** is implemented in `DEFINE_ADJUST`:

```
Z_hat = D(Z)          # decode to physical species
Z_proj = E · Z_hat    # re-encode (linear encoder)
write Z_proj back to UDS
```

This re-anchors Z to the manifold each iteration before the rate net is queried.

---

## RC-1 / RC-4 guard: input clipping

Both C templates implement **input clipping** immediately after reading cell state
(labelled `STEP 2: INPUT CLIPPING`).  This is ChemZIP §4.3 step 5:

- Mass fractions are clipped to `[floor, ∞)` before the log transform.
- Temperature and pressure are clipped to `mean ± 5·std` of the training distribution.

This does NOT fix the RC-1 root cause (under-represented near-inlet states in
training data — fixed by F1/data enrichment), but it prevents the surrogate from
evaluating states that are completely outside its training distribution, which would
produce arbitrarily wrong rates and potentially destabilise the CFD solver.

---

## Sanity checks (Python)

`scarfs.coupling.sanity` provides four checks callable in the a-posteriori harness:

```python
from scarfs.coupling.sanity import (
    mass_fraction_closure,
    source_term_mass_balance,
    energy_consistency,
    check_scaler_roundtrip,
)

res, ok = mass_fraction_closure(Y)           # |sum Y - 1| <= 1e-6
res, ok = source_term_mass_balance(rates)    # |sum R_i| / max|R_i| <= 1e-4
res, ok = energy_consistency(rates, h_f, W, S_E)   # relative S_E error <= 5 %
res, ok = check_scaler_roundtrip(scaler, x)  # max abs reconstruction error <= 1e-10
```

These detect the *stable-but-biased* failure mode documented in DIAGNOSIS.md:
a coupling-interface mismatch (wrong units, mole vs mass fractions, scaler inversion)
will produce large residuals here even when the CFD solver converges.

---

## MergedCoil UDF code generator (kind="merged" bundles)

For bundles produced by the merged training path (`kind="merged"` in `spec.json`),
use `scarfs.coupling.codegen.export_merged_udf` instead of the handwritten templates
above.  It generates all artefacts from the bundle in one call:

```python
from scarfs.coupling.codegen import export_merged_udf, InletSpec

result = export_merged_udf(
    bundle_dir="runs/exp",          # directory with model.pt / scalers.pkl / spec.json
    out_dir="hpc_export_merged/",
    n_reference_states=6,           # C forward-test reference vectors
    inlet=InletSpec(                # optional: custom inlet composition for folded-BC
        composition={"C2H6": 0.7, "H2O": 0.3},
        T=923.0, P=2.0e5,
    ),
)
# result.artifacts keys: header, udf_source, tui_setup, forward_test,
#                        inlet_bc_txt, inlet_bc_csv, consistency_report
# result.consistency_max_rel_diff_sh   — numpy-mirror vs torch max rel diff on S_h
# result.spectral_norm_detected        — True if bundle had spectral-norm parametrizations
```

Generated files:

| File | Purpose |
|---|---|
| `merged_coil_udf.h` | All static const weight/bias arrays; `#define` macros |
| `merged_coil_udf.c` | `DEFINE_ADJUST` (projection) + `DEFINE_SOURCE` (energy + UDS) |
| `fluent_merged_setup.tui` | TUI helper: UDM allocation + hook wiring notes |
| `merged_coil_forward_test.c` | Standalone compile-and-run parity test (no `udf.h`) |
| `inlet_bc.txt` / `inlet_bc.csv` | Encoded inlet z values for Fluent initialisation |
| `export_consistency_report.txt` | NumPy-mirror vs torch parity table |

Compile the forward test locally to verify before HPC deployment:

```bash
cc -O0 -lm merged_coil_forward_test.c -o mc_fwd_test && ./mc_fwd_test
```

The C UDF hooks follow the MergedCoil transport contract:
- k UDS scalars transport z = E·((Y_dry − μ)⊘σ).
- `DEFINE_ADJUST mc_manifold_project`: per-iteration manifold projection with annealed URF.
- `DEFINE_SOURCE mc_energy_source`: S_h = −absorption_head(z, q) [J/m³/s].
- Energy clamp `MC_ENERGY_CLAMP` ≈ 1.3× train-max (safety only; not prediction-falsifying).
- Telemetry UDMs: OOD flag, latent-clamp count, energy-clamp count, last S_h.

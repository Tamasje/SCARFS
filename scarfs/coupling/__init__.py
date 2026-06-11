"""Fluent UDF/UDS coupling scaffolding for SCARFS surrogates.

Public Python API
-----------------
Export a trained surrogate to portable text files that a Fluent C UDF can read:

    from scarfs.coupling import (
        ModelBundle,
        export_bundle,
        load_bundle,
        export_mlp_weights,
        load_mlp_weights,
        export_scalers,
        load_scalers,
        export_active_species,
        load_active_species,
    )

Coupling sanity checks (no Cantera required):

    from scarfs.coupling import (
        mass_fraction_closure,
        source_term_mass_balance,
        energy_consistency,
        check_scaler_roundtrip,
    )

MergedCoil UDF code generator (kind="merged" bundles):

    from scarfs.coupling import export_merged_udf, ExportResult, InletSpec

    result = export_merged_udf(bundle_dir, out_dir, n_reference_states=6)

The C UDF templates (``fluent_reduced_source.c``, ``fluent_neuralcoil_uds.c``) are
deliverable **templates** that the user compiles on the HPC.  Ansys Fluent cannot be
run in this repository.  See ``README.md`` for compilation and wiring instructions.
For MergedCoil bundles, see ``codegen.py`` and the ``merged_coil_udf.*`` artefacts it
generates.
"""

from __future__ import annotations

from scarfs.coupling.export import (
    ModelBundle,
    export_active_species,
    export_bundle,
    export_mlp_weights,
    export_scalers,
    load_active_species,
    load_bundle,
    load_mlp_weights,
    load_scalers,
)
from scarfs.coupling.sanity import (
    check_scaler_roundtrip,
    energy_consistency,
    mass_fraction_closure,
    source_term_mass_balance,
)
from scarfs.coupling.codegen import ExportResult, InletSpec, export_merged_udf

__all__ = [
    # export (legacy / reduced / neuralcoil paths)
    "ModelBundle",
    "export_bundle",
    "export_mlp_weights",
    "export_scalers",
    "export_active_species",
    "load_bundle",
    "load_mlp_weights",
    "load_scalers",
    "load_active_species",
    # sanity
    "mass_fraction_closure",
    "source_term_mass_balance",
    "energy_consistency",
    "check_scaler_roundtrip",
    # merged-coil code generator
    "export_merged_udf",
    "ExportResult",
    "InletSpec",
]

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

The C UDF templates (``fluent_reduced_source.c``, ``fluent_neuralcoil_uds.c``) are
deliverable **templates** that the user compiles on the HPC.  Ansys Fluent cannot be
run in this repository.  See ``README.md`` for compilation and wiring instructions.
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

__all__ = [
    # export
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
]

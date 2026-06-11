"""SCARFS diagnostics package — standing audit writers for the merged pipeline.

Each writer runs a specific diagnostic on the training database and emits BOTH a
markdown report and a CSV next to it in a caller-supplied ``out_dir``.  All
functions also return a small dataclass so tests can assert on values rather
than file text.

Context (plan §2 / §3):
    These audits were designed to address the five root causes of the colleague's
    energy-prediction failure (tail-suppressing training, rank-k decode
    bottleneck, hidden-state ambiguity from species dropping, front
    under-resolution, miscalibrated gate) and are shipped as standing
    pre-export checks with RECALIBRATED thresholds.

Public API
----------
run_energy_unit_audit
    Verify DB absorption = Σ ρ·h_mass·dYdt; gate corr/relRMSE/median-ratio.
run_energy_coverage_audit
    Compare energy-active subset vs full recomputed; gate on miss fraction.
run_state_ambiguity
    NN-based ambiguity statistics for retained or full state.
run_ambiguity_collapse
    Phase-1 collapse test (both states + dropped-subspace conditional check).
run_conservation_audit
    Element conservation and mass-rate closure audit.
run_raw_database_sanity
    Non-finite / negative-Y / row-sum / absorption positivity checks.
run_front_resolution_audit
    Consecutive |ΔS_E| / peak statistics (storage-stride quality check).
"""

from .energy_audit import (
    EnergyCoverageAudit,
    EnergyUnitAudit,
    run_energy_coverage_audit,
    run_energy_unit_audit,
)
from .ambiguity import (
    AmbiguityCollapseReport,
    AmbiguityReport,
    run_ambiguity_collapse,
    run_state_ambiguity,
)
from .conservation_report import ConservationAudit, run_conservation_audit
from .sanity import SanityReport, run_raw_database_sanity
from .front_resolution import FrontResolutionReport, run_front_resolution_audit

__all__ = [
    # energy
    "EnergyUnitAudit",
    "EnergyCoverageAudit",
    "run_energy_unit_audit",
    "run_energy_coverage_audit",
    # ambiguity
    "AmbiguityReport",
    "AmbiguityCollapseReport",
    "run_state_ambiguity",
    "run_ambiguity_collapse",
    # conservation
    "ConservationAudit",
    "run_conservation_audit",
    # sanity
    "SanityReport",
    "run_raw_database_sanity",
    # front resolution
    "FrontResolutionReport",
    "run_front_resolution_audit",
]

"""Energy unit and coverage audit for the SCARFS diagnostics package.

Context (plan §2 / energy-deviation diagnosis):
    The colleague's energy failure was traced to five root causes; cause #2
    (rank-k decode bottleneck, 1.5e8 J/m³/s irreducible floor) and cause #5
    (miscalibrated absolute gate ≈ generator self-noise 1.38e4 J/m³/s,
    species-truncation floor 1.61e5 J/m³/s) motivate two standing audits:

    1. **Unit audit** — recompute absorption = Σ ρ·h_mass(T)·dYdt over ALL
       species with thermo (missing_ok) and compare to the database column.
       Validates that the DB column is internally consistent (his audits passed,
       verified; corr ~1.0, relRMSE ~3e-5 on stride5/6).

    2. **Coverage audit** — recompute from the energy-active SUBSET and compare
       to the full recomputation.  The miss fraction |1 − Σ|retained|/Σ|full||
       quantifies how much absorption information the model's retained species
       carry.  His gate (≤ 0.05) is kept — it was validated on stride5.

Thresholds (his validated gates, kept):
    - corr ≥ 0.98, relRMSE ≤ 0.08, median ratio ∈ [0.90, 1.10] for unit audit.
    - miss fraction ≤ 0.05 for coverage audit.

Absolute floors are REPORTED only, not gated (plan §5 item 6):
    - p95 |diff| (unit audit) ≈ generator self-noise floor ~1.38e4 J/m³/s.
    - p95 |diff| (coverage audit) ≈ species-truncation floor ~1.61e5 J/m³/s.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..models.thermo import SpeciesThermo, select_energy_active_species
from ..schema import Schema
from .report import md_header, md_kv_block, md_table, write_report_pair


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EnergyUnitAudit:
    """Scalar summary of the energy unit audit.

    Attributes
    ----------
    corr
        Pearson correlation between recomputed and DB absorption.
    rel_rmse
        Relative RMSE: RMSE / (mean |DB absorption|), using max(|val|, 1.0)
        as denominator to avoid division by zero on near-zero rows.
    median_ratio
        Median of recomputed / DB absorption (row-wise).
    p95_abs_diff
        95th percentile of |recomputed − DB| [J/m³/s].  This is the
        'generator self-noise' floor; reported, not gated.
    passed
        ``True`` when corr ≥ 0.98, rel_rmse ≤ 0.08, and median_ratio ∈ [0.90, 1.10].
    n_thermo_species
        Number of species with thermo data (after missing_ok filtering).
    n_missing
        Number of species requested but absent from the mechanism YAML.
    """

    corr: float
    rel_rmse: float
    median_ratio: float
    p95_abs_diff: float
    passed: bool
    n_thermo_species: int
    n_missing: int


@dataclass
class EnergyCoverageAudit:
    """Scalar summary of the energy coverage audit.

    Attributes
    ----------
    corr
        Correlation between active-subset recomputed and DB absorption.
    rel_rmse
        Relative RMSE of active-subset vs DB absorption.
    p95_abs_diff
        95th percentile of |active-subset − DB| [J/m³/s].  This is the
        'species-truncation floor'; reported, not gated.
    miss_fraction
        |1 − Σ|active recomputed| / Σ|full recomputed||.  Measures how much
        absolute absorption the retained species fail to capture.
    passed
        ``True`` when miss_fraction ≤ 0.05.
    n_active
        Number of energy-active species selected.
    coverage_target
        The coverage fraction passed to ``select_energy_active_species``.
    """

    corr: float
    rel_rmse: float
    p95_abs_diff: float
    miss_fraction: float
    passed: bool
    n_active: int
    coverage_target: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _relative_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Relative RMSE using max(|ref|, 1.0) as row-wise denominator."""
    denom = np.maximum(np.abs(ref), 1.0)
    return float(np.sqrt(np.mean(((pred - ref) / denom) ** 2)))


def _load_dydt_and_state(
    df: pd.DataFrame,
    schema: Schema,
    species: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (dydt, rho, T, db_absorption) arrays for the given species."""
    dydt_cols = schema.dydt_columns(list(species))
    dydt = df[dydt_cols].to_numpy(dtype=float)
    rho = df[schema.require_state("rho")[0]].to_numpy(dtype=float)
    T = df[schema.require_state("T")[0]].to_numpy(dtype=float)
    db_abs = df[schema.energy_target_column()].to_numpy(dtype=float)
    return dydt, rho, T, db_abs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_energy_unit_audit(
    df: pd.DataFrame,
    schema: Schema,
    mech_yaml: str | os.PathLike,
    out_dir: str | os.PathLike,
) -> EnergyUnitAudit:
    """Verify that the DB absorption column matches Σ ρ·h_mass·dYdt.

    Recomputes absorption from ALL species with thermo data (missing_ok=True)
    and compares to ``schema.energy_target_column()``.  Writes a markdown +
    CSV report pair into *out_dir*.

    Parameters
    ----------
    df
        Database rows (must contain dYdt_*, T, P, rho, and the absorption column).
    schema
        Column contract resolved from *df*.
    mech_yaml
        Path to the Cantera-format mechanism YAML.
    out_dir
        Directory to write ``energy_unit_audit.md`` and ``energy_unit_audit.csv``.

    Returns
    -------
    :class:`EnergyUnitAudit` with the key scalar metrics.

    Notes
    -----
    PASS gates (validated by colleague on stride5; kept unchanged):
        - corr ≥ 0.98
        - relRMSE ≤ 0.08
        - median_ratio ∈ [0.90, 1.10]

    The p95 |diff| is the 'generator self-noise' floor (~1.38e4 J/m³/s on
    stride5).  It is reported but not gated.
    """
    thermo = SpeciesThermo.from_mechanism_yaml(mech_yaml, schema.species, missing_ok=True)
    dydt, rho, T, db_abs = _load_dydt_and_state(df, schema, thermo.species)
    recomp = thermo.absorption_from_dydt(dydt, rho, T)

    corr = float(np.corrcoef(recomp, db_abs)[0, 1])
    rel_rmse = _relative_rmse(recomp, db_abs)
    safe_ref = np.where(np.abs(db_abs) < 1.0, 1.0, db_abs)
    ratio = recomp / safe_ref
    median_ratio = float(np.median(ratio))
    p95_abs_diff = float(np.percentile(np.abs(recomp - db_abs), 95))

    passed = (corr >= 0.98) and (rel_rmse <= 0.08) and (0.90 <= median_ratio <= 1.10)

    result = EnergyUnitAudit(
        corr=corr,
        rel_rmse=rel_rmse,
        median_ratio=median_ratio,
        p95_abs_diff=p95_abs_diff,
        passed=passed,
        n_thermo_species=len(thermo.species),
        n_missing=len(thermo.missing),
    )

    # --- write report ---
    kv: dict[str, object] = {
        "passed": passed,
        "corr": f"{corr:.8f}",
        "rel_rmse": f"{rel_rmse:.2e}",
        "median_ratio": f"{median_ratio:.6f}",
        "p95_abs_diff_J_m3_s (self-noise floor)": f"{p95_abs_diff:.4e}",
        "n_thermo_species": len(thermo.species),
        "n_missing_in_yaml": len(thermo.missing),
        "n_db_species": len(schema.species),
        "gate_corr": ">=0.98",
        "gate_rel_rmse": "<=0.08",
        "gate_median_ratio": "[0.90, 1.10]",
    }
    csv_rows = [
        {
            "quantity": k,
            "value": str(v),
        }
        for k, v in sorted(kv.items())
    ]
    write_report_pair(
        out_dir,
        "energy_unit_audit",
        md_sections=[
            (md_header("Energy Unit Audit"), md_kv_block(kv)),
            (md_header("Row Table", level=2), md_table(csv_rows)),
        ],
        csv_rows=csv_rows,
    )
    return result


def run_energy_coverage_audit(
    df: pd.DataFrame,
    schema: Schema,
    mech_yaml: str | os.PathLike,
    energy_active: Sequence[str],
    out_dir: str | os.PathLike,
) -> EnergyCoverageAudit:
    """Compare energy-active-subset absorption to the full-species recomputed value.

    The miss fraction |1 − Σ|active| / Σ|full|| measures how much absolute
    absorption the retained species set fails to account for.  Writes a
    markdown + CSV report pair into *out_dir*.

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract resolved from *df*.
    mech_yaml
        Path to the Cantera-format mechanism YAML.
    energy_active
        Energy-active species names (e.g. from ``select_energy_active_species``).
    out_dir
        Directory to write ``energy_coverage_audit.md`` / ``energy_coverage_audit.csv``.

    Returns
    -------
    :class:`EnergyCoverageAudit`.

    Notes
    -----
    PASS gate: miss fraction ≤ 0.05 (colleague's validated gate, kept).
    The p95 |diff| vs DB is the 'species-truncation floor' (~1.61e5 J/m³/s on
    stride5).  It is reported but not gated.
    """
    # full recompute (all-thermo species)
    thermo_full = SpeciesThermo.from_mechanism_yaml(mech_yaml, schema.species, missing_ok=True)
    dydt_full, rho, T, db_abs = _load_dydt_and_state(df, schema, thermo_full.species)
    recomp_full = thermo_full.absorption_from_dydt(dydt_full, rho, T)

    # active-subset recompute
    active_in_thermo = [s for s in energy_active if s in set(thermo_full.species)]
    thermo_active = SpeciesThermo.from_mechanism_yaml(mech_yaml, active_in_thermo, missing_ok=True)
    dydt_active, _, _, _ = _load_dydt_and_state(df, schema, thermo_active.species)
    recomp_active = thermo_active.absorption_from_dydt(dydt_active, rho, T)

    # metrics vs DB
    corr = float(np.corrcoef(recomp_active, db_abs)[0, 1])
    rel_rmse = _relative_rmse(recomp_active, db_abs)
    p95_abs_diff = float(np.percentile(np.abs(recomp_active - db_abs), 95))

    # miss fraction vs full recomputed
    sum_full = float(np.sum(np.abs(recomp_full)))
    sum_active = float(np.sum(np.abs(recomp_active)))
    miss_fraction = abs(1.0 - sum_active / max(sum_full, 1e-100))
    coverage_target = float(sum_active / max(sum_full, 1e-100))

    passed = miss_fraction <= 0.05

    result = EnergyCoverageAudit(
        corr=corr,
        rel_rmse=rel_rmse,
        p95_abs_diff=p95_abs_diff,
        miss_fraction=miss_fraction,
        passed=passed,
        n_active=len(thermo_active.species),
        coverage_target=coverage_target,
    )

    kv: dict[str, object] = {
        "passed": passed,
        "miss_fraction": f"{miss_fraction:.6e}",
        "coverage_fraction": f"{coverage_target:.8f}",
        "corr_active_vs_db": f"{corr:.8f}",
        "rel_rmse_active_vs_db": f"{rel_rmse:.4e}",
        "p95_abs_diff_J_m3_s (truncation floor)": f"{p95_abs_diff:.4e}",
        "n_active_species": len(thermo_active.species),
        "n_full_thermo_species": len(thermo_full.species),
        "gate_miss_fraction": "<=0.05",
    }
    csv_rows = [{"quantity": k, "value": str(v)} for k, v in sorted(kv.items())]
    write_report_pair(
        out_dir,
        "energy_coverage_audit",
        md_sections=[
            (md_header("Energy Coverage Audit"), md_kv_block(kv)),
            (md_header("Row Table", level=2), md_table(csv_rows)),
        ],
        csv_rows=csv_rows,
    )
    return result

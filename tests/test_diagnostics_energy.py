"""Tests for scarfs.diagnostics.energy_audit — AAA pattern, one function per test.

Runs on the stride6_sample.parquet real fixture (87 rows, 4 cases) with the
chem_ForTransport.yaml mechanism.  Phase-1 measured values are noted in
comments so regressions are immediately visible.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_PARQUET = Path(__file__).parent / "data" / "stride6_sample.parquet"
MECH_YAML = Path(__file__).parent.parent / "chem_ForTransport.yaml"


@pytest.fixture(scope="module")
def stride6_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE_PARQUET)


@pytest.fixture(scope="module")
def stride6_schema(stride6_df: pd.DataFrame):
    from scarfs.schema import Schema
    return Schema.from_columns(list(stride6_df.columns))


# ---------------------------------------------------------------------------
# Energy unit audit
# ---------------------------------------------------------------------------


def test_energy_unit_audit_passes_on_fixture(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Unit audit PASSES on stride6 with corr > 0.99 and relRMSE < 1e-3."""
    # Arrange
    from scarfs.diagnostics import run_energy_unit_audit

    # Act
    result = run_energy_unit_audit(stride6_df, stride6_schema, MECH_YAML, tmp_path)

    # Assert — Phase-1 measured: corr ~1.0, relRMSE ~3e-5
    assert result.passed, f"Unit audit FAIL: {result}"
    assert result.corr > 0.99, f"corr too low: {result.corr}"
    assert result.rel_rmse < 1e-3, f"relRMSE too high: {result.rel_rmse}"
    assert 0.90 <= result.median_ratio <= 1.10, f"median_ratio out of range: {result.median_ratio}"


def test_energy_unit_audit_p95_floor_finite(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """p95 |diff| (self-noise floor) is finite and positive."""
    # Arrange
    from scarfs.diagnostics import run_energy_unit_audit

    # Act
    result = run_energy_unit_audit(stride6_df, stride6_schema, MECH_YAML, tmp_path)

    # Assert
    assert np.isfinite(result.p95_abs_diff)
    assert result.p95_abs_diff >= 0.0


def test_energy_unit_audit_thermo_species_count(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Thermo is loaded for 193 species (20 missing in YAML) with missing_ok."""
    # Arrange
    from scarfs.diagnostics import run_energy_unit_audit

    # Act
    result = run_energy_unit_audit(stride6_df, stride6_schema, MECH_YAML, tmp_path)

    # Assert — mechanism has 193 of 213 species with thermo data
    assert result.n_thermo_species == 193, f"Expected 193, got {result.n_thermo_species}"
    assert result.n_missing == 20, f"Expected 20 missing, got {result.n_missing}"


def test_energy_unit_audit_writes_report_files(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Unit audit writes both .md and .csv files."""
    # Arrange
    from scarfs.diagnostics import run_energy_unit_audit

    # Act
    run_energy_unit_audit(stride6_df, stride6_schema, MECH_YAML, tmp_path)

    # Assert
    assert (tmp_path / "energy_unit_audit.md").exists()
    assert (tmp_path / "energy_unit_audit.csv").exists()


# ---------------------------------------------------------------------------
# Energy coverage audit
# ---------------------------------------------------------------------------


def test_energy_coverage_audit_passes_at_coverage_0999(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Coverage audit PASSES with energy_active from select_energy_active_species(0.999)."""
    # Arrange
    from scarfs.diagnostics import run_energy_coverage_audit
    from scarfs.models.thermo import SpeciesThermo, select_energy_active_species

    thermo = SpeciesThermo.from_mechanism_yaml(MECH_YAML, stride6_schema.species, missing_ok=True)
    T = stride6_df[stride6_schema.require_state("T")[0]].to_numpy(dtype=float)
    rho = stride6_df[stride6_schema.require_state("rho")[0]].to_numpy(dtype=float)
    dydt_cols = stride6_schema.dydt_columns(list(thermo.species))
    dydt = stride6_df[dydt_cols].to_numpy(dtype=float)
    energy_active = select_energy_active_species(dydt, rho, T, thermo.species, thermo, coverage=0.999)

    # Act
    result = run_energy_coverage_audit(
        stride6_df, stride6_schema, MECH_YAML, energy_active, tmp_path
    )

    # Assert — miss fraction should be << 0.001 (Phase-1: ~7e-4 on stride6 sample)
    assert result.passed, f"Coverage audit FAIL: miss_fraction={result.miss_fraction:.4e}"
    assert result.miss_fraction <= 0.001, f"miss_fraction too high: {result.miss_fraction:.4e}"


def test_energy_coverage_audit_miss_fraction_finite(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Coverage audit miss fraction is finite and non-negative."""
    # Arrange
    from scarfs.diagnostics import run_energy_coverage_audit
    from scarfs.models.thermo import SpeciesThermo, select_energy_active_species

    thermo = SpeciesThermo.from_mechanism_yaml(MECH_YAML, stride6_schema.species, missing_ok=True)
    T = stride6_df[stride6_schema.require_state("T")[0]].to_numpy(dtype=float)
    rho = stride6_df[stride6_schema.require_state("rho")[0]].to_numpy(dtype=float)
    dydt_cols = stride6_schema.dydt_columns(list(thermo.species))
    dydt = stride6_df[dydt_cols].to_numpy(dtype=float)
    energy_active = select_energy_active_species(dydt, rho, T, thermo.species, thermo, coverage=0.999)

    # Act
    result = run_energy_coverage_audit(
        stride6_df, stride6_schema, MECH_YAML, energy_active, tmp_path
    )

    # Assert
    assert np.isfinite(result.miss_fraction)
    assert result.miss_fraction >= 0.0


def test_energy_coverage_audit_writes_report_files(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Coverage audit writes both .md and .csv files."""
    # Arrange
    from scarfs.diagnostics import run_energy_coverage_audit
    from scarfs.models.thermo import SpeciesThermo, select_energy_active_species

    thermo = SpeciesThermo.from_mechanism_yaml(MECH_YAML, stride6_schema.species, missing_ok=True)
    T = stride6_df[stride6_schema.require_state("T")[0]].to_numpy(dtype=float)
    rho = stride6_df[stride6_schema.require_state("rho")[0]].to_numpy(dtype=float)
    dydt_cols = stride6_schema.dydt_columns(list(thermo.species))
    dydt = stride6_df[dydt_cols].to_numpy(dtype=float)
    energy_active = select_energy_active_species(dydt, rho, T, thermo.species, thermo, coverage=0.999)

    # Act
    run_energy_coverage_audit(
        stride6_df, stride6_schema, MECH_YAML, energy_active, tmp_path
    )

    # Assert
    assert (tmp_path / "energy_coverage_audit.md").exists()
    assert (tmp_path / "energy_coverage_audit.csv").exists()

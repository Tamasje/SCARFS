"""Tests for scarfs.diagnostics.sanity — AAA pattern, one function per test.

Tests the PASS case on the clean stride6 fixture and the FAIL case on a
corrupted frame (injected Y=2.0, which violates the Y > 1 check).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE_PARQUET = Path(__file__).parent / "data" / "stride6_sample.parquet"


@pytest.fixture(scope="module")
def stride6_df() -> pd.DataFrame:
    return pd.read_parquet(FIXTURE_PARQUET)


@pytest.fixture(scope="module")
def stride6_schema(stride6_df: pd.DataFrame):
    from scarfs.schema import Schema
    return Schema.from_columns(list(stride6_df.columns))


# ---------------------------------------------------------------------------
# PASS case — clean fixture
# ---------------------------------------------------------------------------


def test_sanity_passes_on_clean_fixture(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Sanity check PASSES on the clean stride6 fixture."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    # Act
    result = run_raw_database_sanity(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert result.passed, f"Expected PASS on clean fixture: {result}"


def test_sanity_fixture_has_no_material_negatives(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """No material negative Y values in the stride6 fixture."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    # Act
    result = run_raw_database_sanity(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert result.n_material_negative_y == 0


def test_sanity_fixture_row_sum_fraction_ok(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """At least 99% of rows have ΣY ∈ [0.999, 1.001] on the clean fixture."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    # Act
    result = run_raw_database_sanity(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert result.row_sum_fraction_ok >= 0.99, (
        f"row_sum_fraction_ok too low: {result.row_sum_fraction_ok}"
    )


def test_sanity_fixture_absorption_non_negative(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Stride6 fixture has no negative absorption values (positivity lineage check)."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    # Act
    result = run_raw_database_sanity(stride6_df, stride6_schema, tmp_path)

    # Assert — stride6 is strictly positive (all 87 rows)
    assert result.absorption_negative_count == 0
    assert not result.absorption_flag


def test_sanity_reports_case_count(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Sanity report records the correct number of cases (4 in stride6 sample)."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    # Act
    result = run_raw_database_sanity(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert result.n_cases == 4


def test_sanity_writes_report_files(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Sanity check writes both .md and .csv report files."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    # Act
    run_raw_database_sanity(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert (tmp_path / "sanity_report.md").exists()
    assert (tmp_path / "sanity_report.csv").exists()


# ---------------------------------------------------------------------------
# FAIL case — corrupted frame (injected Y=2.0)
# ---------------------------------------------------------------------------


def test_sanity_fails_on_corrupted_frame(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Sanity check FAILS when Y > 1 is injected into the data."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    corrupted = stride6_df.copy()
    # Inject Y_H2 = 2.0 in the first row (clear violation)
    y_col = "Y_H2"
    corrupted.loc[corrupted.index[0], y_col] = 2.0

    # Act
    result = run_raw_database_sanity(corrupted, stride6_schema, tmp_path / "corrupted")

    # Assert
    assert not result.passed, "Expected FAIL on corrupted frame with Y=2.0"
    assert result.n_y_above_one >= 1


def test_sanity_fails_on_material_negative_y(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Sanity check FAILS when Y = −0.01 (material negative) is injected."""
    # Arrange
    from scarfs.diagnostics import run_raw_database_sanity

    corrupted = stride6_df.copy()
    y_col = "Y_C2H4"
    corrupted.loc[corrupted.index[0], y_col] = -0.01  # well below -1e-5

    # Act
    result = run_raw_database_sanity(corrupted, stride6_schema, tmp_path / "neg_y")

    # Assert
    assert not result.passed
    assert result.n_material_negative_y >= 1

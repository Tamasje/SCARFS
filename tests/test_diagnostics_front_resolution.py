"""Tests for scarfs.diagnostics.front_resolution — AAA pattern, one function per test.

Uses the stride6_sample.parquet fixture.  The stride-6 database stores every
6th reactor point, so consecutive |ΔS_E| jumps are expected to be substantial
(Phase-1 on stride5: 39% median / 82% p95; stride6 sample is similar or worse
due to the shifted distribution).
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
# Front resolution audit on real stride-6 fixture
# ---------------------------------------------------------------------------


def test_front_resolution_computes_finite_stats(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Front-resolution audit computes finite median and p95 jump fractions."""
    # Arrange
    from scarfs.diagnostics import run_front_resolution_audit

    # Act
    result = run_front_resolution_audit(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert np.isfinite(result.median_jump_frac), "median_jump_frac not finite"
    assert np.isfinite(result.p95_jump_frac), "p95_jump_frac not finite"


def test_front_resolution_median_gt_zero_on_stride6(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Stride-6 storage produces material S_E jumps (median jump > 0)."""
    # Arrange
    from scarfs.diagnostics import run_front_resolution_audit

    # Act
    result = run_front_resolution_audit(stride6_df, stride6_schema, tmp_path)

    # Assert — stride-6 should produce jumps; median > 0.01 (1% of peak)
    assert result.median_jump_frac > 0.01, (
        f"Expected material jumps on stride-6, got median={result.median_jump_frac:.4f}"
    )


def test_front_resolution_has_flagged_cases(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """At least one case is flagged for max jump > 20% of peak on stride-6."""
    # Arrange
    from scarfs.diagnostics import run_front_resolution_audit

    # Act
    result = run_front_resolution_audit(stride6_df, stride6_schema, tmp_path)

    # Assert — stride-6 storage is coarse; at least 1 case should exceed 20% flag
    assert result.n_flagged_cases >= 1, (
        f"Expected ≥1 flagged case, got {result.n_flagged_cases}"
    )


def test_front_resolution_step_pair_count(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Step-pair count equals total points minus one per case (87 rows, 4 cases)."""
    # Arrange
    from scarfs.diagnostics import run_front_resolution_audit

    # Act
    result = run_front_resolution_audit(stride6_df, stride6_schema, tmp_path)

    # Assert — 87 total rows, 4 cases of ~21 points each → ~83 step pairs
    assert result.n_step_pairs == 87 - 4, (
        f"Expected {87 - 4} step pairs, got {result.n_step_pairs}"
    )


def test_front_resolution_writes_report_files(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Front-resolution audit writes both .md and .csv report files."""
    # Arrange
    from scarfs.diagnostics import run_front_resolution_audit

    # Act
    run_front_resolution_audit(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert (tmp_path / "front_resolution_audit.md").exists()
    assert (tmp_path / "front_resolution_audit.csv").exists()

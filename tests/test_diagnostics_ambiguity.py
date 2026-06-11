"""Tests for scarfs.diagnostics.ambiguity — AAA pattern, one function per test.

Uses the stride6_sample.parquet fixture.  The collapse-direction property
(full-state |ΔS_E| p50 ≤ retained-state p50) is the key assertion — we
tolerate ties on this small sample.
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
# run_state_ambiguity — retained state
# ---------------------------------------------------------------------------


def test_state_ambiguity_retained_returns_finite_stats(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Retained-state ambiguity report returns finite p50/p95 statistics."""
    # Arrange
    from scarfs.diagnostics import run_state_ambiguity

    # Act
    result = run_state_ambiguity(stride6_df, stride6_schema, tmp_path, state="retained")

    # Assert
    assert np.isfinite(result.delta_se_p50), "delta_se_p50 not finite"
    assert np.isfinite(result.delta_se_p95), "delta_se_p95 not finite"
    assert np.isfinite(result.dist_p50), "dist_p50 not finite"
    assert result.n_pairs > 0


def test_state_ambiguity_full_returns_finite_stats(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Full-state ambiguity report returns finite p50/p95 statistics."""
    # Arrange
    from scarfs.diagnostics import run_state_ambiguity

    # Act
    result = run_state_ambiguity(stride6_df, stride6_schema, tmp_path, state="full")

    # Assert
    assert np.isfinite(result.delta_se_p50)
    assert np.isfinite(result.delta_se_p95)
    assert result.n_state_dims > 0


def test_state_ambiguity_writes_report_files(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Ambiguity report writes .md and .csv files for the chosen state."""
    # Arrange
    from scarfs.diagnostics import run_state_ambiguity

    # Act
    run_state_ambiguity(stride6_df, stride6_schema, tmp_path, state="retained")

    # Assert
    assert (tmp_path / "ambiguity_retained.md").exists()
    assert (tmp_path / "ambiguity_retained.csv").exists()


def test_state_ambiguity_full_delta_se_le_retained(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Full-state |ΔS_E| p50 ≤ retained-state p50 (collapse direction; ties accepted)."""
    # Arrange
    from scarfs.diagnostics import run_state_ambiguity

    # Act
    retained = run_state_ambiguity(stride6_df, stride6_schema, tmp_path, state="retained")
    full = run_state_ambiguity(stride6_df, stride6_schema, tmp_path, state="full")

    # Assert — tolerate ties; full state should resolve *at least as much* ambiguity
    assert full.delta_se_p50 <= retained.delta_se_p50 * 1.05, (
        f"Expected full.p50={full.delta_se_p50:.4e} <= "
        f"retained.p50={retained.delta_se_p50:.4e} (5% tolerance)"
    )


# ---------------------------------------------------------------------------
# run_ambiguity_collapse
# ---------------------------------------------------------------------------


def test_ambiguity_collapse_returns_both_reports(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Collapse report contains both retained and full sub-reports."""
    # Arrange
    from scarfs.diagnostics import run_ambiguity_collapse

    # Act
    result = run_ambiguity_collapse(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert result.retained is not None
    assert result.full is not None
    assert result.retained.state == "retained"
    assert result.full.state == "full"


def test_ambiguity_collapse_writes_all_report_files(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Collapse audit writes .md and .csv for retained, full, and collapse summary."""
    # Arrange
    from scarfs.diagnostics import run_ambiguity_collapse

    # Act
    run_ambiguity_collapse(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert (tmp_path / "ambiguity_retained.md").exists()
    assert (tmp_path / "ambiguity_full.md").exists()
    assert (tmp_path / "ambiguity_collapse.md").exists()


def test_ambiguity_collapse_spearman_finite(
    stride6_df: pd.DataFrame,
    stride6_schema,
    tmp_path: Path,
) -> None:
    """Spearman correlation in both sub-reports is finite (enough pairs in fixture)."""
    # Arrange
    from scarfs.diagnostics import run_ambiguity_collapse

    # Act
    result = run_ambiguity_collapse(stride6_df, stride6_schema, tmp_path)

    # Assert
    assert np.isfinite(result.retained.spearman_r), "retained Spearman r not finite"
    assert np.isfinite(result.full.spearman_r), "full Spearman r not finite"

"""Tests for scarfs.benchmark.feasibility — kNN feasibility pre-gate.

Each test follows arrange/act/assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.benchmark.feasibility import feasibility_table, FeasibilityReport


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_feasibility_df(n_species: int = 8, n_cases: int = 4, n_pts: int = 20,
                          seed: int = 42) -> pd.DataFrame:
    """Build a synthetic DataFrame with Y_*, R_*, T, P, CaseID, absorption columns."""
    rng = np.random.default_rng(seed)
    n = n_cases * n_pts
    rows: dict[str, np.ndarray] = {}

    species = [f"SP{i}" for i in range(n_species)]
    Y = np.abs(rng.normal(0.1, 0.05, (n, n_species)))
    Y = Y / Y.sum(axis=1, keepdims=True)  # normalise to sum ~1
    for i, sp in enumerate(species):
        rows[f"Y_{sp}"] = Y[:, i]
        # R_ columns are required by Schema.from_columns for consistency
        rows[f"R_{sp}"] = rng.normal(0, 0.1, n)

    rows["T [K]"] = rng.uniform(823, 1200, n)
    rows["P [Pa]"] = rng.uniform(1.5e5, 3.5e5, n)
    rows["CaseID"] = np.repeat(np.arange(n_cases), n_pts)
    rows["rho [kg/m3]"] = rng.uniform(0.3, 1.0, n)
    rows["tau [s]"] = np.tile(np.linspace(0, 0.5, n_pts), n_cases)
    rows["z [m]"] = np.tile(np.linspace(0, 10, n_pts), n_cases)

    # Absorption that correlates strongly with T and one species
    rows["Reaction heat absorption [J/s/m3]"] = (
        1e8 * (rows["T [K]"] - 823) / 377 + 5e7 * Y[:, 0]
        + rng.normal(0, 1e6, n)
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feasibility_table_returns_report():
    """feasibility_table returns a FeasibilityReport with the right structure."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4, 6))

    # Assert
    assert isinstance(report, FeasibilityReport)
    assert len(report.entries) == 2 * 2  # pca + pls for each k
    assert report.ks == [4, 6]


def test_feasibility_table_entry_spaces():
    """Each k produces a 'pca' and a 'pls' entry."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4,))
    spaces = {e.space for e in report.entries}

    # Assert
    assert "pca" in spaces
    assert "pls" in spaces


def test_feasibility_table_r2_finite():
    """All global_r2 and tail_r2 values are finite numbers."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4, 6))

    # Assert
    for entry in report.entries:
        assert np.isfinite(entry.global_r2), f"NaN global_r2 for {entry}"
        assert np.isfinite(entry.tail_r2), f"NaN tail_r2 for {entry}"


def test_feasibility_table_baseline_finite():
    """Baseline (full-state) R² values are finite."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4,))

    # Assert
    assert np.isfinite(report.baseline_global_r2)
    assert np.isfinite(report.baseline_tail_r2)


def test_feasibility_table_margin_zero_pass_condition():
    """With margin=0, entries with tail_r2 >= baseline pass."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4,), margin=0.0)

    # Assert: entries whose tail_r2 >= baseline_tail_r2 should have passes=True
    for entry in report.entries:
        expected = entry.tail_r2 >= report.baseline_tail_r2 - 0.0
        assert entry.passes == expected, f"Passes mismatch for {entry}"


def test_feasibility_table_summary_contains_table():
    """summary() contains the baseline R² and a column header."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4,))
    text = report.summary()

    # Assert
    assert "FEASIBILITY PRE-GATE" in text
    assert "space" in text.lower() or "pca" in text.lower()


def test_feasibility_table_to_dataframe():
    """to_dataframe() returns a DataFrame with expected columns."""
    # Arrange
    df = _make_feasibility_df()
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4, 6))
    tbl = report.to_dataframe()

    # Assert
    assert "space" in tbl.columns
    assert "k" in tbl.columns
    assert "global_r2" in tbl.columns
    assert "tail_r2" in tbl.columns
    assert "passes" in tbl.columns
    assert len(tbl) == 4  # 2 ks × 2 spaces


def test_feasibility_table_single_case_fallback():
    """Single-case data falls back gracefully without raising."""
    # Arrange
    df = _make_feasibility_df(n_cases=1, n_pts=30)
    df["CaseID"] = 0
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act — must not raise
    report = feasibility_table(df, schema, ks=(4,))

    # Assert
    assert len(report.entries) == 2  # pca + pls


def test_feasibility_table_stride6_fixture():
    """Smoke test on the real stride6 parquet fixture."""
    from pathlib import Path

    fixture = Path(__file__).parent / "data" / "stride6_sample.parquet"
    if not fixture.exists():
        pytest.skip("stride6_sample.parquet not found")

    # Arrange
    df = pd.read_parquet(str(fixture))
    from scarfs.benchmark.loader import infer_schema
    schema = infer_schema(df)

    # Act
    report = feasibility_table(df, schema, ks=(4, 6))

    # Assert: report must be well-formed
    assert isinstance(report, FeasibilityReport)
    assert report.n_folds >= 2
    assert len(report.entries) == 4  # 2 ks × 2 spaces
    for entry in report.entries:
        assert np.isfinite(entry.global_r2)
        assert np.isfinite(entry.tail_r2)

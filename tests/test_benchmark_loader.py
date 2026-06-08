"""Tests for scarfs.benchmark.loader — load_database and infer_schema.

Each test follows arrange / act / assert.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from scarfs.benchmark.loader import load_database, infer_schema
from scarfs.schema import Schema

# Skip parquet-based tests if no engine is available
try:
    import pyarrow  # noqa: F401
    _PARQUET_AVAILABLE = True
except ImportError:
    try:
        import fastparquet  # noqa: F401
        _PARQUET_AVAILABLE = True
    except ImportError:
        _PARQUET_AVAILABLE = False

requires_parquet = pytest.mark.skipif(
    not _PARQUET_AVAILABLE,
    reason="pyarrow or fastparquet required for Parquet tests",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_df() -> pd.DataFrame:
    """Return a minimal DataFrame that satisfies the Y_*/R_* contract."""
    return pd.DataFrame(
        {
            "Y_A": [0.6, 0.5],
            "Y_B": [0.4, 0.5],
            "R_A": [1.0, 2.0],
            "R_B": [-1.0, -2.0],
            "T [K]": [800.0, 900.0],
            "P [Pa]": [1e5, 1e5],
            "z [m]": [0.0, 1.0],
            "rho [kg/m3]": [0.7, 0.7],
            "Mass flow [kg/s]": [1.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# load_database
# ---------------------------------------------------------------------------

class TestLoadDatabase:
    def test_load_csv(self, tmp_path: Path) -> None:
        """Arrange: write a minimal CSV. Act: load it. Assert: DataFrame matches."""
        df_expected = _minimal_df()
        csv_path = tmp_path / "test.csv"
        df_expected.to_csv(csv_path, index=False)

        df_loaded = load_database(csv_path)

        assert list(df_loaded.columns) == list(df_expected.columns)
        assert len(df_loaded) == len(df_expected)

    @requires_parquet
    def test_load_parquet(self, tmp_path: Path) -> None:
        """Arrange: write a Parquet file. Act: load it. Assert: DataFrame matches."""
        df_expected = _minimal_df()
        pq_path = tmp_path / "test.parquet"
        df_expected.to_parquet(pq_path, index=False)

        df_loaded = load_database(pq_path)

        assert list(df_loaded.columns) == list(df_expected.columns)
        assert len(df_loaded) == len(df_expected)

    @requires_parquet
    def test_load_pq_extension(self, tmp_path: Path) -> None:
        """Arrange: use .pq extension. Act/Assert: loads without error."""
        df = _minimal_df()
        path = tmp_path / "test.pq"
        df.to_parquet(path, index=False)

        df_loaded = load_database(path)

        assert len(df_loaded) == len(df)

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        """Arrange: .xlsx path. Act/Assert: ValueError raised."""
        fake_path = tmp_path / "test.xlsx"
        fake_path.write_text("dummy")

        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_database(fake_path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """Arrange: non-existent path. Act/Assert: FileNotFoundError raised."""
        missing = tmp_path / "ghost.csv"

        with pytest.raises(FileNotFoundError):
            load_database(missing)

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        """Arrange: path as str, not Path. Act/Assert: loads without error."""
        df = _minimal_df()
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        df_loaded = load_database(str(csv_path))

        assert len(df_loaded) == len(df)


# ---------------------------------------------------------------------------
# infer_schema
# ---------------------------------------------------------------------------

class TestInferSchema:
    def test_returns_schema_instance(self) -> None:
        """Arrange: minimal DataFrame. Act: infer_schema. Assert: Schema returned."""
        df = _minimal_df()

        schema = infer_schema(df)

        assert isinstance(schema, Schema)

    def test_species_extracted(self) -> None:
        """Arrange: DataFrame with two Y_ columns. Act/Assert: species=(A, B)."""
        df = _minimal_df()

        schema = infer_schema(df)

        assert schema.species == ("A", "B")

    def test_state_columns_resolved(self) -> None:
        """Arrange: DataFrame with T, P, z, rho state columns. Act/Assert: state keys present."""
        df = _minimal_df()

        schema = infer_schema(df)

        assert "T" in schema.state
        assert "P" in schema.state
        assert "z" in schema.state
        assert "rho" in schema.state

    def test_missing_r_column_raises(self) -> None:
        """Arrange: Y_A present but R_A absent. Act/Assert: ValueError from Schema."""
        df = pd.DataFrame({"Y_A": [1.0], "T [K]": [800.0]})

        with pytest.raises(ValueError, match="no R_"):
            infer_schema(df)

    def test_roundtrip_csv(self, tmp_path: Path) -> None:
        """Arrange: write CSV, load it, infer schema. Assert: species consistent."""
        df = _minimal_df()
        path = tmp_path / "db.csv"
        df.to_csv(path, index=False)

        df_loaded = load_database(path)
        schema = infer_schema(df_loaded)

        assert schema.species == ("A", "B")

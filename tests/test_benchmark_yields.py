"""Tests for scarfs.benchmark.yields — integrate_yields and integrate_yields_per_case.

Each test follows arrange / act / assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.benchmark.yields import integrate_yields, integrate_yields_per_case
from scarfs.schema import Schema


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_single_case_df(n: int = 20, L: float = 8.0) -> pd.DataFrame:
    """Return a synthetic single-case PFR DataFrame with n axial points."""
    z = np.linspace(0.0, L, n)
    return pd.DataFrame(
        {
            "Y_A": np.linspace(0.7, 0.5, n),
            "Y_B": np.linspace(0.1, 0.3, n),
            "R_A": np.full(n, -1.0),   # constant depletion rate
            "R_B": np.full(n, 1.0),    # constant production rate
            "T [K]": np.linspace(823.0, 973.0, n),
            "P [Pa]": np.full(n, 2e5),
            "z [m]": z,
            "rho [kg/m3]": np.full(n, 0.73),
            "Mass flow [kg/s]": np.full(n, 1.6),
            "mdot [kg/s]": np.full(n, 1.6),
            "U_in [m/s]": np.full(n, 11.0),
            "CaseID": np.ones(n, dtype=int),
            "L [m]": np.full(n, L),
        }
    )


def _make_schema_single() -> Schema:
    return Schema.from_columns(list(_make_single_case_df().columns))


# ---------------------------------------------------------------------------
# integrate_yields
# ---------------------------------------------------------------------------

class TestIntegrateYields:
    def test_returns_1d_array_length_k(self) -> None:
        """Arrange: single case with 2 species. Act: integrate. Assert: shape (2,)."""
        df = _make_single_case_df()
        schema = _make_schema_single()
        rates = df[["R_A", "R_B"]].to_numpy()

        result = integrate_yields(df, rates, schema)

        assert result.shape == (2,)

    def test_constant_rate_analytic(self) -> None:
        """Arrange: constant rate R_A=-1, A/mdot can be computed.
        Assert: ΔY_A == R_A * (A/mdot) * L (trapezoidal = exact for linear integrand)."""
        df = _make_single_case_df(n=100, L=8.0)
        schema = _make_schema_single()

        # A = mdot / (rho_in * U_in)
        rho_in = float(df["rho [kg/m3]"].iloc[0])
        mdot = float(df["mdot [kg/s]"].iloc[0])
        u_in = float(df["U_in [m/s]"].iloc[0])
        A = mdot / (rho_in * u_in)
        L = 8.0

        rates = np.column_stack([
            np.full(100, -1.0),   # R_A
            np.full(100, 1.0),    # R_B
        ])

        result = integrate_yields(df, rates, schema)

        expected_dYA = -1.0 * (A / mdot) * L
        expected_dYB = 1.0 * (A / mdot) * L

        assert result[0] == pytest.approx(expected_dYA, rel=1e-4)
        assert result[1] == pytest.approx(expected_dYB, rel=1e-4)

    def test_zero_rates_give_zero_delta(self) -> None:
        """Arrange: all rates=0. Act/Assert: ΔY == 0 for all species."""
        df = _make_single_case_df()
        schema = _make_schema_single()
        rates = np.zeros((len(df), 2))

        result = integrate_yields(df, rates, schema)

        assert np.allclose(result, 0.0)

    def test_sign_conventions(self) -> None:
        """Arrange: positive R_B rate. Act/Assert: ΔY_B > 0 (production)."""
        df = _make_single_case_df()
        schema = _make_schema_single()
        rates = np.column_stack([np.zeros(len(df)), np.full(len(df), 1.0)])

        result = integrate_yields(df, rates, schema)

        assert result[1] > 0.0  # production → positive yield change

    def test_missing_column_raises(self) -> None:
        """Arrange: schema without U_in. Act/Assert: KeyError raised."""
        df_no_uin = _make_single_case_df().drop(columns=["U_in [m/s]"])
        schema_bad = Schema.from_columns(list(df_no_uin.columns))
        rates = np.zeros((len(df_no_uin), 2))

        with pytest.raises(KeyError):
            integrate_yields(df_no_uin, rates, schema_bad)


# ---------------------------------------------------------------------------
# integrate_yields_per_case
# ---------------------------------------------------------------------------

class TestIntegrateYieldsPerCase:
    def test_single_case_returns_one_row(self) -> None:
        """Arrange: single case. Act: integrate_per_case. Assert: 1-row DataFrame."""
        df = _make_single_case_df()
        schema = _make_schema_single()
        rates = df[["R_A", "R_B"]].to_numpy()

        result = integrate_yields_per_case(df, rates, schema, active_species=("A", "B"))

        assert len(result) == 1

    def test_multi_case_returns_correct_row_count(self) -> None:
        """Arrange: 3 cases. Act: integrate_per_case. Assert: 3-row DataFrame."""
        dfs = []
        for cid in [1, 2, 3]:
            df_case = _make_single_case_df(n=10)
            df_case["CaseID"] = cid
            dfs.append(df_case)
        df = pd.concat(dfs, ignore_index=True)
        schema = Schema.from_columns(list(df.columns))
        rates = df[["R_A", "R_B"]].to_numpy()

        result = integrate_yields_per_case(df, rates, schema, active_species=("A", "B"))

        assert len(result) == 3

    def test_columns_contain_dy_prefix(self) -> None:
        """Arrange: species A, B. Act: integrate_per_case. Assert: columns dY_A, dY_B present."""
        df = _make_single_case_df()
        schema = _make_schema_single()
        rates = df[["R_A", "R_B"]].to_numpy()

        result = integrate_yields_per_case(df, rates, schema, active_species=("A", "B"))

        assert "dY_A" in result.columns
        assert "dY_B" in result.columns

    def test_caseid_in_output(self) -> None:
        """Arrange: single case CaseID=1. Act/Assert: CaseID column present in output."""
        df = _make_single_case_df()
        schema = _make_schema_single()
        rates = df[["R_A", "R_B"]].to_numpy()

        result = integrate_yields_per_case(df, rates, schema, active_species=("A", "B"))

        assert "CaseID" in result.columns

"""Tests for the column-schema contract."""

from __future__ import annotations

from scarfs.schema import Schema, column_base, is_radical, r_column, y_column


def test_from_columns_extracts_species_in_order(synthetic_df):
    # arrange
    cols = list(synthetic_df.columns)
    # act
    schema = Schema.from_columns(cols)
    # assert
    assert schema.species == ("C2H6", "C2H4", "H2", "H2O", "CH3.")


def test_molecular_and_radical_split(synthetic_schema):
    # arrange / act
    molecular = synthetic_schema.molecular_species()
    radicals = synthetic_schema.radical_species()
    # assert
    assert "CH3." in radicals and "CH3." not in molecular
    assert "C2H4" in molecular


def test_active_species_excludes_diluent_and_radicals(synthetic_schema):
    # act
    active = synthetic_schema.active_species()
    # assert
    assert "H2O" not in active and "CH3." not in active
    assert set(active) == {"C2H6", "C2H4", "H2"}


def test_state_columns_resolved_with_unit_suffix(synthetic_schema):
    # assert canonical keys map to the actual unit-suffixed column names
    assert synthetic_schema.state["T"] == "T [K]"
    assert synthetic_schema.state["rho"] == "rho [kg/m3]"
    assert synthetic_schema.meta["CaseID"] == "CaseID"


def test_column_base_strips_units_and_handles_middot():
    # act / assert — the non-ASCII '·' in 'mu [Pa·s]' must not break base resolution
    assert column_base("mu [Pa·s]") == "mu"
    assert column_base("T_in [K]") == "t_in"


def test_y_and_r_column_helpers():
    assert y_column("C2H4") == "Y_C2H4"
    assert r_column("C2H4") == "R_C2H4"


def test_is_radical():
    assert is_radical("CH3.") and not is_radical("C2H4")


def test_missing_rate_column_raises(synthetic_df):
    # arrange — drop a rate column to make the DB inconsistent
    cols = [c for c in synthetic_df.columns if c != "R_C2H4"]
    # act / assert
    import pytest

    with pytest.raises(ValueError):
        Schema.from_columns(cols)

"""Tests for schema.py parquet-contract extensions.

Tests cover:
- Parquet column-name resolution from a representative subset of the colleague's parquet schema.
  Exact column names are used (copied from the parquet file); the species subset is self-consistent
  (each Y_ species has a matching dYdt_ or R_ entry).
- Pseudo-species exclusion (Y_C2H6_in [-] / Y_H2O_in [-]).
- dYdt_* column mapping (has_dydt, dydt_columns, rate_unit_convention).
- energy_target_column (present / missing).
- rate_unit_convention for each family combination.
- Relaxed coverage: dYdt_ covers species when R_ is absent.
- Strict coverage: require_r=True raises when R_ is missing.
- New state/meta keys (tau, u, S_reaction_absorption, S_wall, diameter, etc.).
"""

from __future__ import annotations

import pytest

from scarfs.schema import (
    DYDT_PREFIX,
    R_PREFIX,
    Schema,
    dydt_column,
    wdot_column,
)

# ---------------------------------------------------------------------------
# Representative column subset from TEST_ETHANE_LOW_sobol_stride6.parquet.
#
# Only 5 species are included (H2, CH4, C2H6, C3H8, IC4H10) because the full
# parquet has 213 Y_ species each matched by a dYdt_ entry.  This subset keeps
# tests self-consistent: every Y_ species has a matching dYdt_ column.
# The 5 R_ columns cover the same species (parquet carries both families).
# ---------------------------------------------------------------------------

# Species included in the representative subset (all matched by dYdt_ and R_)
_SPECIES_5 = ["H2", "CH4", "C2H6", "C3H8", "IC4H10"]

_Y_COLS_5 = [f"Y_{s}" for s in _SPECIES_5]

# 2 pseudo-species (carry [ in their name — exact parquet column names)
_Y_PSEUDO = [
    "Y_C2H6_in [-]",
    "Y_H2O_in [-]",
]

# dYdt_ columns for the 5 species (exact parquet names)
_DYDT_COLS_5 = [
    "dYdt_H2 [1/s]",
    "dYdt_CH4 [1/s]",
    "dYdt_C2H6 [1/s]",
    "dYdt_C3H8 [1/s]",
    "dYdt_IC4H10 [1/s]",
]

# wdot_ columns (exact parquet names)
_WDOT_COLS = [
    "wdot_H2 [kmol/m3/s]",
    "wdot_CH4 [kmol/m3/s]",
    "wdot_C2H6 [kmol/m3/s]",
]

# R_ columns — no unit suffix in parquet (raw DLL units)
_R_COLS_5 = [f"R_{s}" for s in _SPECIES_5]

# D_ columns (exact parquet names)
_D_COLS = [
    "D_H2 [m2/s]",
    "D_CH4 [m2/s]",
    "D_C2H6 [m2/s]",
]

# All state/meta columns from the parquet file (exact names)
_STATE_COLS = [
    "T [K]",
    "P [Pa]",
    "S Energy [J/s/m3]",
    "S Energy source label",
    "Reaction heat absorption [J/s/m3]",
    "sum_h_wdot [J/s/m3]",
    "S Wall imposed [J/s/m3]",
    "Heat input [W/m2]",
    "z [m]",
    "tau [s]",
    "PFR point index",
    "PFR points solved",
    "Storage stride",
    "u [m/s]",
    "cp_mass [J/kg/K]",
    "cv_mass [J/kg/K]",
    "rho [kg/m3]",
    "mu [Pa-s]",
    "k [W/m/K]",
    "W_mean [kg/kmol]",
    "CaseID",
    "mdot [kg/s]",
    "Mass flow [kg/s]",
    "diameter [m]",
    "Area [m2]",
    "steam_to_ethane [kg/kg]",
    "T_in [K]",
    "P_in [Pa]",
    "shape",
    "H_peak [W/m2]",
    "Re_in [-]",
    "U_in [m/s]",
]

# Full consistent parquet representative column list:
# Y_(5 species) + Y_(2 pseudo) + dYdt_(5) + wdot_(3) + R_(5) + D_(3) + all state/meta
_PARQUET_COLS = (
    _Y_COLS_5 + _Y_PSEUDO + _DYDT_COLS_5 + _WDOT_COLS + _R_COLS_5 + _D_COLS + _STATE_COLS
)

# dYdt-only variant: no R_ columns; dYdt_ covers all 5 species
_DYDT_ONLY_COLS = _Y_COLS_5 + _Y_PSEUDO + _DYDT_COLS_5 + _STATE_COLS


# ---------------------------------------------------------------------------
# Pseudo-species exclusion
# ---------------------------------------------------------------------------

class TestPseudoSpeciesExclusion:
    def test_pseudo_species_excluded_from_species(self):
        # arrange / act
        schema = Schema.from_columns(_PARQUET_COLS)
        # assert — pseudo-species names must not appear in schema.species
        assert "C2H6_in" not in schema.species
        assert "H2O_in" not in schema.species

    def test_pseudo_count_recorded(self):
        # arrange / act
        schema = Schema.from_columns(_PARQUET_COLS)
        # assert
        assert schema.n_pseudo_excluded == 2

    def test_real_species_count_correct(self):
        # arrange / act
        schema = Schema.from_columns(_PARQUET_COLS)
        # assert — 5 real Y_ species in the subset
        assert len(schema.species) == 5

    def test_no_pseudo_when_none_present(self):
        # arrange — classic CSV columns without pseudo-species
        cols = ["Y_A", "Y_B", "R_A", "R_B", "T [K]"]
        # act
        schema = Schema.from_columns(cols)
        # assert
        assert schema.n_pseudo_excluded == 0

    def test_species_order_preserved(self):
        # arrange / act
        schema = Schema.from_columns(_PARQUET_COLS)
        # assert — species derived from Y_ column order; pseudo-species excluded
        assert schema.species == tuple(_SPECIES_5)


# ---------------------------------------------------------------------------
# dYdt_* column mapping
# ---------------------------------------------------------------------------

class TestDydtColumns:
    def test_has_dydt_true_when_present(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert schema.has_dydt() is True

    def test_has_dydt_false_when_absent(self):
        cols = ["Y_A", "Y_B", "R_A", "R_B"]
        schema = Schema.from_columns(cols)
        assert schema.has_dydt() is False

    def test_dydt_column_helper_name(self):
        assert dydt_column("C2H6") == "dYdt_C2H6 [1/s]"

    def test_dydt_columns_for_subset(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        cols = schema.dydt_columns(["H2", "C2H6"])
        assert cols == ["dYdt_H2 [1/s]", "dYdt_C2H6 [1/s]"]

    def test_dydt_columns_all_species(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        all_cols = schema.dydt_columns()
        assert all_cols == [f"dYdt_{s} [1/s]" for s in _SPECIES_5]

    def test_dydt_columns_empty_when_no_dydt(self):
        schema = Schema.from_columns(["Y_A", "Y_B", "R_A", "R_B"])
        assert schema.dydt_columns() == []

    def test_rate_families_includes_dydt(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert DYDT_PREFIX in schema.rate_families

    def test_rate_families_includes_r_when_both_present(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert R_PREFIX in schema.rate_families


# ---------------------------------------------------------------------------
# rate_unit_convention
# ---------------------------------------------------------------------------

class TestRateUnitConvention:
    def test_both_when_r_and_dydt(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert schema.rate_unit_convention() == "both"

    def test_mass_kg_m3_s_when_r_only(self):
        schema = Schema.from_columns(["Y_A", "Y_B", "R_A", "R_B"])
        assert schema.rate_unit_convention() == "mass_kg_m3_s"

    def test_dydt_per_s_when_dydt_only(self):
        schema = Schema.from_columns(_DYDT_ONLY_COLS)
        assert schema.rate_unit_convention() == "dydt_per_s"

    def test_raises_when_no_rate_family(self):
        # no Y_ columns → no rate families
        schema = Schema.from_columns(["T [K]", "P [Pa]"])
        with pytest.raises(ValueError, match="no rate family"):
            schema.rate_unit_convention()


# ---------------------------------------------------------------------------
# energy_target_column
# ---------------------------------------------------------------------------

class TestEnergyTargetColumn:
    def test_returns_absorption_column(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        col = schema.energy_target_column()
        assert col == "Reaction heat absorption [J/s/m3]"

    def test_raises_when_absent(self):
        cols_no_abs = [c for c in _PARQUET_COLS if "absorption" not in c.lower()]
        schema = Schema.from_columns(cols_no_abs)
        with pytest.raises(KeyError, match="Reaction heat absorption"):
            schema.energy_target_column()

    def test_s_energy_still_resolvable(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        # S_energy must be present as a state key (deprecated-for-training, but accessible)
        assert "S_energy" in schema.state

    def test_s_energy_is_not_energy_target(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        target = schema.energy_target_column()
        assert "S Energy" not in target


# ---------------------------------------------------------------------------
# New state keys (tau, u, S_reaction_absorption, S_wall, meta)
# ---------------------------------------------------------------------------

class TestParquetStateKeys:
    def test_tau_resolved(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "tau" in schema.state
        assert schema.state["tau"] == "tau [s]"

    def test_u_resolved(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "u" in schema.state
        assert schema.state["u"] == "u [m/s]"

    def test_s_reaction_absorption_resolved(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "S_reaction_absorption" in schema.state
        assert schema.state["S_reaction_absorption"] == "Reaction heat absorption [J/s/m3]"

    def test_s_wall_resolved(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "S_wall" in schema.state

    def test_diameter_resolved_to_meta(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "diameter" in schema.meta

    def test_steam_to_ethane_resolved_to_meta(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "steam_to_ethane" in schema.meta

    def test_storage_stride_resolved_to_meta(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "storage_stride" in schema.meta

    def test_pfr_point_index_resolved_to_meta(self):
        schema = Schema.from_columns(_PARQUET_COLS)
        assert "pfr_point_index" in schema.meta

    def test_legacy_csv_state_keys_still_work(self):
        # Legacy CSV columns must still resolve as before
        cols = ["Y_A", "Y_B", "R_A", "R_B", "T [K]", "P [Pa]", "rho [kg/m3]", "z [m]", "CaseID"]
        schema = Schema.from_columns(cols)
        assert "T" in schema.state
        assert "P" in schema.state
        assert "rho" in schema.state
        assert "z" in schema.state
        assert "CaseID" in schema.meta

    def test_mu_pa_s_dash_resolves(self):
        # parquet uses "mu [Pa-s]" (dash), legacy CSV uses "mu [Pa·s]" (middle-dot); both must work
        schema_dash = Schema.from_columns(_PARQUET_COLS)  # contains "mu [Pa-s]"
        assert "mu" in schema_dash.state


# ---------------------------------------------------------------------------
# Rate coverage rules
# ---------------------------------------------------------------------------

class TestRateCoverage:
    def test_dydt_only_covers_species_no_error(self):
        # dYdt_ family alone must satisfy coverage when require_r is None
        schema = Schema.from_columns(_DYDT_ONLY_COLS)
        assert len(schema.species) == 5

    def test_require_r_true_raises_when_r_absent(self):
        with pytest.raises(ValueError, match="no R_"):
            Schema.from_columns(_DYDT_ONLY_COLS, require_r=True)

    def test_require_r_false_skips_r_check(self):
        # require_r=False: no R_ columns present but no error raised (dYdt_ satisfies coverage)
        schema = Schema.from_columns(_DYDT_ONLY_COLS, require_r=False)
        assert len(schema.species) == 5

    def test_uncovered_species_no_families_no_error(self):
        # No rate families at all → coverage check skipped; species extracted normally
        cols = ["Y_A", "Y_B", "T [K]"]
        schema = Schema.from_columns(cols, require_r=False)
        assert schema.species == ("A", "B")

    def test_partial_dydt_coverage_raises(self):
        # One species has Y_ but no dYdt_ and no R_ → should raise when dYdt_ family is present
        cols = [
            "Y_A",
            "Y_B",
            "dYdt_A [1/s]",  # covers A but not B
            "T [K]",
        ]
        with pytest.raises(ValueError, match="not covered by any rate family"):
            Schema.from_columns(cols)

    def test_r_and_dydt_partial_union_coverage_ok(self):
        # R_ covers A, dYdt_ covers B → union covers both → no error
        cols = [
            "Y_A",
            "Y_B",
            "R_A",
            "dYdt_B [1/s]",
            "T [K]",
        ]
        schema = Schema.from_columns(cols)
        assert set(schema.species) == {"A", "B"}

    def test_automatic_relaxed_mode_with_dydt(self):
        # When dYdt_ columns are present, require_r defaults to False
        schema = Schema.from_columns(_DYDT_ONLY_COLS)
        # must not raise; both families detected or only dYdt_
        assert schema.has_dydt() is True


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------

def test_dydt_column_helper():
    assert dydt_column("C2H4") == "dYdt_C2H4 [1/s]"


def test_dydt_column_custom_suffix():
    assert dydt_column("C2H4", " [s^-1]") == "dYdt_C2H4 [s^-1]"


def test_wdot_column_helper():
    assert wdot_column("C2H4") == "wdot_C2H4 [kmol/m3/s]"

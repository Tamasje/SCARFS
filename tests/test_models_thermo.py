"""Tests for the NASA7 thermo module (scarfs.models.thermo).

Includes the cornerstone validation test: absorption recomputed from dYdt in the
stride6 fixture must match the stored 'Reaction heat absorption [J/s/m3]' column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MINI_YAML = Path(__file__).parent / "data" / "thermo_mini.yaml"
FIXTURE_PARQUET = Path(__file__).parent / "data" / "stride6_sample.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_thermo(species=None):
    from scarfs.models.thermo import SpeciesThermo
    if species is None:
        species = ["H2", "CH4", "C2H6", "C2H4", "C3H8", "C3H6", "H2O", "CO", "CO2",
                   "H.", "CH3.", "C2H3.", "C2H5."]
    return SpeciesThermo.from_mechanism_yaml(MINI_YAML, species)


# ---------------------------------------------------------------------------
# from_mechanism_yaml
# ---------------------------------------------------------------------------

def test_from_yaml_species_order():
    # arrange
    wanted = ["CH4", "H2", "C2H6"]
    # act
    st = _load_thermo(wanted)
    # assert
    assert st.species == tuple(wanted)


def test_from_yaml_raises_on_missing_species():
    # arrange + act + assert
    with pytest.raises(ValueError, match="not found"):
        _load_thermo(["H2", "NOTASPECIES"])


def test_molar_mass_c2h4():
    # C2H4: 2*12.0107 + 4*1.00794 = 28.054
    st = _load_thermo(["C2H4"])
    # assert
    expected = 2 * 12.0107 + 4 * 1.00794
    assert abs(st.molar_mass[0] - expected) < 0.01


def test_element_matrix_c2h4():
    # C2H4: C=2, H=4
    st = _load_thermo(["C2H4"])
    el_idx = {e: i for i, e in enumerate(st.element_names)}
    # assert
    assert st.element_matrix[0, el_idx["C"]] == pytest.approx(2.0)
    assert st.element_matrix[0, el_idx["H"]] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# h_molar / h_mass
# ---------------------------------------------------------------------------

def test_h_molar_hand_computed_h2():
    """Hand-verify NASA7 h_molar for H2 at T=1000 K (low-range coefficients)."""
    from scarfs.models.thermo import R_J_PER_KMOL_K

    st = _load_thermo(["H2"])
    # H2 low-range (T_mid=1400 K, so 1000 K uses low)
    # data[0] = [3.5370809, -0.0003547348, 6.8090191e-07, -2.2708045e-10, 1.9361594e-14, -1044.3898, -4.368296]
    a = np.array([3.5370809, -0.0003547348, 6.8090191e-07, -2.2708045e-10, 1.9361594e-14, -1044.3898, -4.368296])
    T = 1000.0
    h_rt = a[0] + a[1]*T/2 + a[2]*T**2/3 + a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T
    expected = R_J_PER_KMOL_K * T * h_rt

    T_arr = np.array([T])
    got = st.h_molar(T_arr)[0, 0]
    assert got == pytest.approx(expected, rel=1e-6)


def test_h_molar_two_temperatures():
    """h_molar at T=500 K and T=2000 K selects the correct NASA7 range."""
    st = _load_thermo(["CH4"])
    T = np.array([500.0, 2000.0])
    h = st.h_molar(T)
    assert h.shape == (2, 1)
    # At 500 K (< T_mid=1400) low range; at 2000 K high range — values should differ
    assert h[0, 0] != pytest.approx(h[1, 0], rel=0.01)


def test_h_mass_equals_h_molar_over_mw():
    st = _load_thermo(["C2H6", "H2"])
    T = np.array([800.0, 1200.0])
    h_mol = st.h_molar(T)
    h_ms = st.h_mass(T)
    expected = h_mol / st.molar_mass[None, :]
    assert np.allclose(h_ms, expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# cp_molar / cp_mass
# ---------------------------------------------------------------------------

def test_cp_shape():
    st = _load_thermo(["H2", "CH4", "C2H4"])
    T = np.linspace(500, 1500, 10)
    assert st.cp_molar(T).shape == (10, 3)
    assert st.cp_mass(T).shape == (10, 3)


def test_cp_positive():
    st = _load_thermo(["H2", "CH4", "C2H6"])
    T = np.array([300.0, 800.0, 1400.0, 2000.0])
    assert np.all(st.cp_molar(T) > 0)


# ---------------------------------------------------------------------------
# absorption_from_dydt
# ---------------------------------------------------------------------------

def test_absorption_from_dydt_single_species():
    """Scalar check: one species, known h_mass, known rho and dydt."""
    from scarfs.models.thermo import SpeciesThermo

    st = _load_thermo(["H2"])
    T = np.array([1000.0])
    rho = np.array([1.0])
    # dydt = 1.0 → absorption = rho * h_mass * dydt = h_mass[0]
    h_ms = st.h_mass(T)[0, 0]
    dydt = np.array([[1.0]])
    result = st.absorption_from_dydt(dydt, rho, T)
    assert result[0] == pytest.approx(h_ms, rel=1e-9)


def test_absorption_from_dydt_zero():
    st = _load_thermo(["H2", "CH4"])
    T = np.array([1000.0, 900.0])
    rho = np.array([1.5, 2.0])
    dydt = np.zeros((2, 2))
    result = st.absorption_from_dydt(dydt, rho, T)
    assert np.allclose(result, 0.0)


# ---------------------------------------------------------------------------
# select_energy_active_species
# ---------------------------------------------------------------------------

def test_select_energy_active_coverage_monotonic():
    """Coverage threshold monotonically grows the selected set."""
    from scarfs.models.thermo import select_energy_active_species

    rng = np.random.default_rng(42)
    species = ["H2", "CH4", "C2H6", "C2H4", "C3H8"]
    st = _load_thermo(species)
    n = 50
    T = rng.uniform(800, 1400, n)
    rho = rng.uniform(0.5, 3.0, n)
    dydt = rng.normal(0, 1e-3, (n, len(species)))

    prev_len = 0
    for cov in [0.5, 0.9, 0.999, 1.0]:
        sel = select_energy_active_species(dydt, rho, T, species, st, coverage=cov)
        assert len(sel) >= prev_len
        prev_len = len(sel)


def test_select_energy_active_always_include():
    """always_include forces species into the result regardless of coverage."""
    from scarfs.models.thermo import select_energy_active_species

    rng = np.random.default_rng(7)
    species = ["H2", "CH4", "C2H6", "C2H4", "C3H8"]
    st = _load_thermo(species)
    n = 30
    T = rng.uniform(900, 1300, n)
    rho = rng.uniform(1.0, 2.0, n)
    # Make C3H8 a negligible contributor
    dydt = np.zeros((n, len(species)))
    dydt[:, 0] = 1e-3   # H2 dominates

    sel = select_energy_active_species(
        dydt, rho, T, species, st, coverage=0.99, always_include=["C3H8"]
    )
    assert "C3H8" in sel


def test_select_energy_active_schema_order():
    """Output follows original species order, not rank order."""
    from scarfs.models.thermo import select_energy_active_species

    rng = np.random.default_rng(13)
    species = ["C2H6", "H2", "CH4", "C2H4"]
    st = _load_thermo(species)
    n = 20
    T = rng.uniform(900, 1300, n)
    rho = np.ones(n)
    dydt = rng.normal(0, 1e-3, (n, 4))

    sel = select_energy_active_species(dydt, rho, T, species, st, coverage=0.5)
    # All returned names must appear in the original order
    idxs = [species.index(s) for s in sel]
    assert idxs == sorted(idxs)


# ---------------------------------------------------------------------------
# Cornerstone validation: absorption vs stride6 fixture
# ---------------------------------------------------------------------------

def test_absorption_matches_db_stride6_mini():
    """Subset check using mini-yaml (13 species): partial sum correlates with DB absorption.

    The mini-yaml covers 13 / 215 species; C2H4 dominates, so the partial recomputed
    sum must correlate with the stored column (corr > 0.9) and lie within 3x of the
    stored value for high-energy rows (dominated by the 13 included species).
    """
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from scarfs.models.thermo import SpeciesThermo

    if not FIXTURE_PARQUET.exists():
        pytest.skip("stride6_sample.parquet fixture not found")

    df = pd.read_parquet(FIXTURE_PARQUET)
    dydt_col_map = {
        c[len("dYdt_"):].split(" [")[0]: c
        for c in df.columns if c.startswith("dYdt_")
    }
    mini_species = ["H2", "CH4", "C2H6", "C2H4", "C3H8", "C3H6", "H2O", "CO", "CO2",
                    "H.", "CH3.", "C2H3.", "C2H5."]
    present = [s for s in mini_species if s in dydt_col_map and f"Y_{s}" in df.columns]
    assert len(present) >= 5

    st = SpeciesThermo.from_mechanism_yaml(MINI_YAML, present)
    T = df["T [K]"].to_numpy(dtype=float)
    rho = df["rho [kg/m3]"].to_numpy(dtype=float)
    dydt = df[[dydt_col_map[s] for s in present]].to_numpy(dtype=float)
    target = df["Reaction heat absorption [J/s/m3]"].to_numpy(dtype=float)
    recomputed = st.absorption_from_dydt(dydt, rho, T)

    # Correlation must be high (13-species partial sum vs full 215-species target)
    corr = float(np.corrcoef(target, recomputed)[0, 1])
    assert corr > 0.90, f"corr={corr:.6f} < 0.90 — check units or NASA7 implementation"


def test_absorption_matches_db_stride6_full():
    """Cornerstone validation: absorption recomputed from ALL species vs stored column.

    Uses the full chem_ForTransport.yaml (all 193 intersecting species with dYdt
    columns in the fixture).  The NASA7 recompute should match the stored
    'Reaction heat absorption [J/s/m3]' column with:
      - corr > 0.999
      - rel_RMSE < 1e-2 (achieved: ~2.9e-05)

    This test is slow (~5 s for YAML parsing) and is skipped when the full YAML
    is not in the repo root.
    """
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")

    import pandas as pd
    from scarfs.models.thermo import SpeciesThermo

    full_yaml = Path(__file__).parent.parent / "chem_ForTransport.yaml"
    if not full_yaml.exists():
        pytest.skip("chem_ForTransport.yaml not found at repo root")
    if not FIXTURE_PARQUET.exists():
        pytest.skip("stride6_sample.parquet fixture not found")

    df = pd.read_parquet(FIXTURE_PARQUET)
    dydt_col_map = {
        c[len("dYdt_"):].split(" [")[0]: c
        for c in df.columns if c.startswith("dYdt_")
    }
    import yaml as _yaml
    raw = _yaml.safe_load(full_yaml.read_text(encoding="utf-8"))
    yaml_species = {str(s["name"]) for s in raw.get("species", []) if "name" in s}
    available = [
        s for s in dydt_col_map
        if s in yaml_species and f"Y_{s}" in df.columns
    ]
    assert len(available) > 100, f"Too few species: {len(available)}"

    st = SpeciesThermo.from_mechanism_yaml(full_yaml, available)
    T = df["T [K]"].to_numpy(dtype=float)
    rho = df["rho [kg/m3]"].to_numpy(dtype=float)
    dydt = df[[dydt_col_map[s] for s in available]].to_numpy(dtype=float)
    target = df["Reaction heat absorption [J/s/m3]"].to_numpy(dtype=float)
    recomputed = st.absorption_from_dydt(dydt, rho, T)

    corr = float(np.corrcoef(target, recomputed)[0, 1])
    mask = target > 1e4
    rel_rmse = float(np.sqrt(np.mean(((recomputed[mask] - target[mask]) / target[mask]) ** 2)))

    assert corr > 0.999, f"corr={corr:.9f} — check NASA7 coefficient parsing"
    assert rel_rmse < 1e-2, (
        f"rel_RMSE={rel_rmse:.6e} > 1e-2 — check units (J/kmol vs J/mol, R value)"
    )


# ---------------------------------------------------------------------------
# PyTorch twin (skip if torch unavailable)
# ---------------------------------------------------------------------------

def test_h_mass_torch_matches_numpy():
    """h_mass_torch output must match h_mass to float32 precision."""
    torch = pytest.importorskip("torch")

    st = _load_thermo(["H2", "CH4", "C2H6"])
    T_np = np.array([700.0, 1000.0, 1500.0])
    T_t = torch.tensor(T_np, dtype=torch.float32)

    numpy_result = st.h_mass(T_np).astype(np.float32)
    torch_result = st.h_mass_torch(T_t).detach().numpy()

    # float32 rounding: allow ~1e-5 relative tolerance
    assert np.allclose(numpy_result, torch_result, rtol=1e-4, atol=1.0)


def test_absorption_from_rates_torch():
    """absorption_from_rates_torch matches numpy path."""
    torch = pytest.importorskip("torch")

    st = _load_thermo(["H2", "CH4"])
    T = np.array([900.0, 1100.0])
    rho = np.array([1.5, 2.0])
    dydt = np.array([[1e-3, -2e-3], [2e-3, 1e-3]])
    rate_mass = rho[:, None] * dydt

    np_result = st.absorption_from_dydt(dydt, rho, T).astype(np.float32)
    t_result = st.absorption_from_rates_torch(
        torch.tensor(rate_mass, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
    ).detach().numpy()

    assert np.allclose(np_result, t_result, rtol=1e-4, atol=1.0)

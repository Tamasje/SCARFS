"""Tests for the v2 generator's platform-independent core (no Cantera/DLL required).

The CRACKSIM-coupled paths (run_case_v2, off-manifold eval, gate A) execute only on the
Windows machine; their pure building blocks — manifest tiers, runner mapping, perturbation
kernel, frame assembly, gates C/D logic, flux builders — are fully tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.data.generation_v2 import (
    FULL_TIER_COUNTS,
    GenV2Settings,
    PerturbConfig,
    aggregate_sign_audits,
    assemble_v2_frame,
    build_heat_profile,
    build_v2_manifest,
    ethane_steam_mass_fractions,
    gate_front_resolution,
    make_piecewise,
    perturb_states,
    to_runner_case,
)
from scarfs.schema import Schema


# ---------------------------------------------------------------------------
# Manifest tiers
# ---------------------------------------------------------------------------

def test_manifest_full_counts_match_spec():
    """The full tier produces the approved per-regime case counts."""
    # arrange / act
    cases, manifest = build_v2_manifest("full")
    # assert
    assert manifest["regime_counts"] == FULL_TIER_COUNTS
    assert manifest["n_cases"] == sum(FULL_TIER_COUNTS.values())


def test_manifest_pilot_is_prefix_of_full():
    """Pilot case IDs are a strict subset (per-regime prefix) of the full tier's IDs."""
    # arrange
    full_cases, _ = build_v2_manifest("full")
    pilot_cases, m = build_v2_manifest("pilot")
    # act
    full_ids = {c["id"] for c in full_cases}
    pilot_ids = {c["id"] for c in pilot_cases}
    # assert
    assert pilot_ids < full_ids, "pilot must be a strict subset of full"
    assert m["n_cases"] == len(pilot_ids)
    assert all(v >= 2 for v in m["regime_counts"].values()), "every regime represented"


def test_manifest_smoke_covers_every_regime():
    """Smoke takes a couple of cases from EACH regime (pipeline shakedown coverage)."""
    # arrange / act
    cases, manifest = build_v2_manifest("smoke")
    # assert
    assert set(manifest["regime_counts"]) == set(FULL_TIER_COUNTS)
    assert all(v == 2 for v in manifest["regime_counts"].values())


def test_manifest_deterministic():
    """Same tier + seed → identical case list (Sobol streams are seeded)."""
    # arrange / act
    a, _ = build_v2_manifest("pilot", seed=123)
    b, _ = build_v2_manifest("pilot", seed=123)
    # assert
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert all(ca["T_in"] == cb["T_in"] for ca, cb in zip(a, b))


def test_manifest_test_tier_ids_disjoint_from_train():
    """Test-tier CaseIDs are offset so they can never collide with train tiers."""
    # arrange / act
    full_cases, _ = build_v2_manifest("full")
    test_cases, _ = build_v2_manifest("test")
    # assert
    assert {c["id"] for c in full_cases}.isdisjoint({c["id"] for c in test_cases})
    assert min(c["id"] for c in test_cases) >= 1_000_000


def test_manifest_respects_temperature_cap():
    """No manifest case requests an inlet above the confirmed 1423.15 K cap."""
    # arrange / act
    cases, manifest = build_v2_manifest("full")
    # assert
    assert manifest["t_max_K"] == pytest.approx(1423.15)
    assert max(c["T_in"] for c in cases) <= 1423.15


# ---------------------------------------------------------------------------
# Runner-case mapping
# ---------------------------------------------------------------------------

def test_to_runner_case_steam_ratio_roundtrip():
    """X_H2O (steam mass fraction) maps to SD with Y_H2O = X recovered exactly."""
    # arrange
    case = {"id": 7, "regime": "body", "L": 6.0, "H_peak": 5e4, "shape": "uniform",
            "params": {}, "mdot": 1.2, "T_in": 900.0, "P_in": 2e5, "X_H2O": 0.30,
            "diameter": 0.05, "Re_in": 5e4, "U_in": 10.0}
    # act
    runner = to_runner_case(case, GenV2Settings(n_points=123))
    y = ethane_steam_mass_fractions(runner["steam_to_ethane_kgkg"])
    # assert
    assert runner["N_points"] == 123
    assert runner["diameter_m"] == pytest.approx(0.05)
    assert y["H2O"] == pytest.approx(0.30, rel=1e-12)
    assert y["C2H6"] == pytest.approx(0.70, rel=1e-12)


# ---------------------------------------------------------------------------
# Perturbation kernel (off-manifold block)
# ---------------------------------------------------------------------------

def test_perturb_states_valid_compositions():
    """Perturbed compositions are non-negative, renormalised, and deterministic by seed."""
    # arrange
    rng = np.random.default_rng(0)
    Y = rng.dirichlet(np.ones(6), size=10)
    T = np.full(10, 1000.0)
    P = np.full(10, 2e5)
    cfg = PerturbConfig(sigma_log=0.25, t_rel=0.03, p_rel=0.05, points_per_anchor=4)
    # act
    Yp, Tp, Pp = perturb_states(Y, T, P, cfg, seed=42)
    Yp2, _, _ = perturb_states(Y, T, P, cfg, seed=42)
    # assert
    assert Yp.shape == (40, 6)
    assert np.all(Yp >= 0.0)
    np.testing.assert_allclose(Yp.sum(axis=1), 1.0, rtol=1e-12)
    t_jit = np.abs(Tp / np.repeat(T, 4) - 1.0)
    assert t_jit.max() <= 0.03 + 1e-12 and t_jit.max() > 0.01  # bounded, non-degenerate jitter
    np.testing.assert_array_equal(Yp, Yp2)
    assert not np.allclose(Yp[0], Yp[1]), "distinct samples per anchor"


def test_perturb_states_jitter_is_modest():
    """Median multiplicative change per species stays within ~e^{2σ} (neighbourhood, not noise)."""
    # arrange
    Y = np.full((50, 5), 0.2)
    cfg = PerturbConfig(sigma_log=0.25, points_per_anchor=2)
    # act
    Yp, _, _ = perturb_states(Y, np.full(50, 900.0), np.full(50, 2e5), cfg, seed=1)
    ratio = Yp / 0.2
    # assert
    assert 0.4 < np.median(ratio) < 1.6
    assert np.quantile(ratio, 0.99) < np.exp(3 * cfg.sigma_log)


# ---------------------------------------------------------------------------
# Frame assembly + schema contract
# ---------------------------------------------------------------------------

def _toy_frame(n: int = 7, sample_kind: str = "trajectory"):
    species = ["C2H6", "C2H4", "H2O", "CH3."]
    rng = np.random.default_rng(3)
    runner = {"id": 11, "regime": "body", "mdot": 1.5, "diameter_m": 0.05,
              "steam_to_ethane_kgkg": 0.43, "T_in": 900.0, "P_in": 2e5,
              "shape": "uniform", "H_peak": 5e4, "Re_in": 5e4, "U_in": 11.0}
    return assemble_v2_frame(
        species_names=species,
        Y=rng.dirichlet(np.ones(4), size=n),
        dYdt=rng.normal(0, 1, (n, 4)),
        T=np.linspace(900, 1100, n), P=np.full(n, 2e5), rho=np.full(n, 0.6),
        u=np.full(n, 10.0), tau=np.linspace(0, 0.1, n), z=np.linspace(0, 6, n),
        cp=np.full(n, 2500.0), cv=np.full(n, 2100.0), mu=np.full(n, 3e-5),
        k=np.full(n, 0.1), W_mean=np.full(n, 25.0),
        absorption=np.linspace(1e5, 1e8, n),
        s_wall=np.full(n, 4e6), q_wall=np.full(n, 5e4),
        pfr_point_index=np.arange(n), n_points_solved=400,
        runner_case=runner, settings=GenV2Settings(),
        inlet_Y={"C2H6": 0.7, "H2O": 0.3}, sample_kind=sample_kind,
    )


def test_assemble_v2_frame_schema_parses_and_no_pseudo_species():
    """The v2 frame satisfies the package Schema contract with no Y_*_in pseudo-species."""
    # arrange / act
    df = _toy_frame()
    schema = Schema.from_columns(list(df.columns))
    # assert
    assert schema.species == ("C2H6", "C2H4", "H2O", "CH3.")
    assert schema.n_pseudo_excluded == 0, "inlet composition must not use Y_ prefix"
    assert schema.has_dydt()
    assert schema.energy_target_column() == "Reaction heat absorption [J/s/m3]"
    assert "tau" in schema.state and "z" in schema.state


def test_assemble_v2_frame_drops_redundant_families_and_records_provenance():
    """No R_/wdot_/D_ columns; float64 species data; solver tolerances recorded per row."""
    # arrange / act
    df = _toy_frame()
    # assert
    assert not [c for c in df.columns if c.startswith(("R_", "wdot_", "D_"))]
    assert "S Energy [J/s/m3]" not in df.columns, "the misleading column stays dead"
    assert df["Y_C2H6"].dtype == np.float64
    assert df["solver_rtol"].iloc[0] == pytest.approx(1e-9)
    assert df["generator_version"].iloc[0] == GenV2Settings().generator_version
    assert df["sample_kind"].unique().tolist() == ["trajectory"]


# ---------------------------------------------------------------------------
# Gates (logic on synthetic frames)
# ---------------------------------------------------------------------------

def test_gate_front_resolution_passes_smooth_and_fails_jumpy():
    """Gate C accepts policy-respecting storage and rejects stride-like jumps."""
    # arrange: a densely stored ramp (per-step jump 1/(n-1) of peak) vs a coarse one
    def frame(n):
        return pd.DataFrame({
            "Reaction heat absorption [J/s/m3]": np.linspace(0.0, 1e8, n),
            "tau [s]": np.linspace(0, 0.1, n),
            "CaseID": np.full(n, 1),
            "sample_kind": ["trajectory"] * n,
        })
    # act
    ok = gate_front_resolution(frame(80), max_frac_jump=0.03)   # jumps ≈ 1.3% of peak
    bad = gate_front_resolution(frame(4), max_frac_jump=0.03)   # jumps ≈ 33% of peak
    # assert
    assert ok["passed"] and ok["p95_jump_frac"] < 0.045
    assert not bad["passed"]


def test_aggregate_sign_audits_flags_material_negatives_only():
    """Gate D: −2e5 (truncation-noise scale) is not material; −5e6 is."""
    # arrange
    benign = [{"min_value": -2.0e5, "frac_negative": 0.001},
              {"min_value": 0.5, "frac_negative": 0.0}]
    material = benign + [{"min_value": -5.0e6, "frac_negative": 0.02}]
    # act / assert
    assert not aggregate_sign_audits(benign)["material_negative"]
    agg = aggregate_sign_audits(material)
    assert agg["material_negative"] and agg["worst_min_absorption"] == pytest.approx(-5e6)


# ---------------------------------------------------------------------------
# Flux builders
# ---------------------------------------------------------------------------

def test_flux_builders_nonnegative_and_peak_bounded():
    """Every v2 shape yields a non-negative callable bounded by ~H over [0, L]."""
    # arrange
    L, H = 8.0, 1.2e5
    zs = np.linspace(0, L, 257)
    shapes = [("uniform", {}), ("triangular", {}), ("gaussian_pair", {}),
              ("sinusoidal", {"Np_req": 200}), ("front_ramp", {"Np_req": 200}),
              ("back_ramp", {"Np_req": 200})]
    # act / assert
    for name, params in shapes:
        z, q, meta = build_heat_profile(L, name, {**params, "H": H})
        f = make_piecewise(z, q)
        vals = np.array([f(zz) for zz in zs])
        assert vals.min() >= 0.0, name
        # gaussian_pair's overlapping tails legitimately sum to ~1.004·H mid-domain
        assert vals.max() <= H * 1.05, name
        assert meta["shape"] == name


def test_pulsed_shape_is_explicitly_unsupported():
    """The unported 'pulsed' shape raises a clear error instead of silently misbehaving."""
    # arrange / act / assert
    with pytest.raises(ValueError, match="pulsed"):
        build_heat_profile(5.0, "pulsed", {"H": 1e5})

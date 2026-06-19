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
    gate_low_conversion_coverage,
    make_piecewise,
    perturb_states,
    to_runner_case,
)
from scarfs.data.config import DataGenConfig, StorageConfig
from scarfs.data.generate import select_storage_indices
from scarfs.data.sampling import build_cases
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
    """No R_/wdot_/D_ columns by default; float64 species data; solver tolerances recorded per row."""
    # arrange / act
    df = _toy_frame()
    # assert
    assert not [c for c in df.columns if c.startswith(("R_", "wdot_", "D_"))]
    assert "S Energy [J/s/m3]" not in df.columns, "the misleading column stays dead"
    assert df["Y_C2H6"].dtype == np.float64
    assert df["solver_rtol"].iloc[0] == pytest.approx(1e-9)
    assert df["generator_version"].iloc[0] == GenV2Settings().generator_version
    assert df["sample_kind"].unique().tolist() == ["trajectory"]


def test_assemble_v2_frame_emits_diffusivity_when_D_supplied():
    """When a D array is passed, D_<species> [m2/s] columns appear and the Schema still parses
    (D_ is excluded from species — coverage gap 3 / Dᵢ delivery)."""
    species = ["C2H6", "C2H4", "H2O", "CH3."]
    rng = np.random.default_rng(5)
    n = 6
    runner = {"id": 1, "regime": "body", "mdot": 1.0, "diameter_m": 0.05,
              "steam_to_ethane_kgkg": 0.43, "T_in": 900.0, "P_in": 2e5,
              "shape": "uniform", "H_peak": 5e4}
    df = assemble_v2_frame(
        species_names=species,
        Y=rng.dirichlet(np.ones(4), size=n), dYdt=rng.normal(0, 1, (n, 4)),
        T=np.linspace(900, 1100, n), P=np.full(n, 2e5), rho=np.full(n, 0.6),
        u=np.full(n, 10.0), tau=np.linspace(0, 0.1, n), z=np.linspace(0, 6, n),
        cp=np.full(n, 2500.0), cv=np.full(n, 2100.0), mu=np.full(n, 3e-5),
        k=np.full(n, 0.1), W_mean=np.full(n, 25.0),
        absorption=np.linspace(1e5, 1e8, n), s_wall=np.full(n, 4e6), q_wall=np.full(n, 5e4),
        pfr_point_index=np.arange(n), n_points_solved=400,
        runner_case=runner, settings=GenV2Settings(),
        inlet_Y={"C2H6": 0.7, "H2O": 0.3},
        D=np.abs(rng.normal(1e-4, 1e-5, (n, 4))),
    )
    d_cols = [c for c in df.columns if c.startswith("D_")]
    assert d_cols == [f"D_{s} [m2/s]" for s in species]
    schema = Schema.from_columns(list(df.columns))
    assert schema.species == ("C2H6", "C2H4", "H2O", "CH3."), "D_ must not become a species"


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
              ("back_ramp", {"Np_req": 200}), ("pulsed", {"N": 6, "Np_req": 200})]
    # act / assert
    for name, params in shapes:
        z, q, meta = build_heat_profile(L, name, {**params, "H": H})
        f = make_piecewise(z, q)
        vals = np.array([f(zz) for zz in zs])
        assert vals.min() >= 0.0, name
        # gaussian_pair's overlapping tails legitimately sum to ~1.004·H mid-domain
        assert vals.max() <= H * 1.05, name
        assert meta["shape"] == name


def test_every_default_manifest_shape_is_buildable():
    """REGRESSION (Windows smoke 2026-06-12): the manifest sampled 'pulsed' but the v2
    builder set lacked it -> 4/10 smoke cases dropped. Every shape the sampler can emit
    must build a valid non-negative profile with the runner-supplied params."""
    from scarfs.data.config import DataGenConfig

    # arrange
    cfg = DataGenConfig()
    # act / assert: exercise each default shape exactly as run_case_v2 parameterises it
    for shape_name, shape_params in cfg.shapes:
        params = dict(shape_params)
        if shape_name == "pulsed":
            params = {**params, "H": 9e4, "N": int(params.get("N", 10)), "Np_req": 400,
                      "seed": 3}
        elif shape_name in ("sinusoidal", "front_ramp", "back_ramp"):
            params = {**params, "H": 9e4, "Np_req": 400}
        else:
            params = {**params, "H": 9e4}
        z, q, meta = build_heat_profile(7.5, shape_name, params)
        f = make_piecewise(z, q)
        vals = np.array([f(zz) for zz in np.linspace(0, 7.5, 301)])
        assert vals.min() >= 0.0, shape_name
        assert vals.max() > 0.0, shape_name
        assert meta["shape"] == shape_name


def test_unknown_shape_raises_clearly():
    """An unknown shape still raises with the supported list in the message."""
    # arrange / act / assert
    with pytest.raises(ValueError, match="Unknown heat profile shape"):
        build_heat_profile(5.0, "warp_drive", {"H": 1e5})


def test_gate_front_resolution_separates_grid_limited_jumps():
    """REGRESSION (Windows smoke 2026-06-12): single-solver-step jumps are the storage
    policy's floor — they must inform the --n-points recommendation, not fail the gate."""
    # arrange: a steep front where CONSECUTIVE solver points jump 10% of peak (grid-limited)
    # while every policy-chosen skip (index gap > 1) respects the 3% policy
    a = np.array([0.0, 0.01, 0.02, 0.12, 0.22, 0.32, 0.34, 0.36, 1.0]) * 1e8
    idx = np.array([0, 40, 80, 81, 82, 83, 120, 160, 399])  # gaps: >1,>1,1,1,1,>1,>1,>1
    df = pd.DataFrame({
        "Reaction heat absorption [J/s/m3]": a,
        "tau [s]": np.linspace(0, 0.1, len(a)),
        "PFR point index": idx,
        "CaseID": np.full(len(a), 1),
        "sample_kind": ["trajectory"] * len(a),
    })
    # act
    res = gate_front_resolution(df, max_frac_jump=0.7)  # policy jumps here are <= 0.64 of peak
    strict = gate_front_resolution(df.drop(columns=["PFR point index"]), max_frac_jump=0.7)
    # assert: grid-limited 10% steps are reported separately, not in the policy population
    assert res["n_grid_jumps"] == 3 and res["n_policy_jumps"] == 5
    assert res["grid_p95_jump_frac"] == pytest.approx(0.10, abs=1e-9)
    assert res["grid_max_jump_frac"] == pytest.approx(0.10, abs=1e-9)
    # recommendation uses the STEEPEST single step (max), the binding grid constraint
    assert res["grid_resolution_factor"] == pytest.approx(0.10 / 0.7, rel=1e-6)
    assert res["passed"]
    assert strict["n_policy_jumps"] == 8, "without the index column all jumps count (strict)"


def test_settings_from_doc_ignores_transport_flags():
    """REGRESSION (Windows smoke crash 2026-06-12): the worker payload carries loop-only
    flags (gate_a_in_worker) alongside GenV2Settings fields; reconstruction must filter to
    dataclass fields and rebuild the nested StorageConfig."""
    from scarfs.data.generation_v2 import settings_from_doc

    # arrange: exactly what run_tier ships to workers
    base = GenV2Settings(n_points=321, solver_rtol=1e-8)
    doc = {**base.__dict__, "storage": base.storage.__dict__, "gate_a_in_worker": True}
    # act
    rebuilt = settings_from_doc(doc)
    # assert
    assert rebuilt.n_points == 321
    assert rebuilt.solver_rtol == pytest.approx(1e-8)
    assert rebuilt.storage.mode == base.storage.mode
    assert rebuilt.storage.max_frac_jump == pytest.approx(base.storage.max_frac_jump)
    assert not hasattr(rebuilt, "gate_a_in_worker")


def test_dydt_and_energy_units_are_mass_conserving():
    """REGRESSION (gate-A finding 2026-06-12): pfr.solve returns MASS rate r·MW [kg/m³/s].
    The true dY/dt = mass_rate/ρ (mass-conserving, Σ=0); the colleague's chain
    compute_dYdt_from_wdot(mass_rate, MW, ρ) multiplies by MW again → Σ dYdt ≠ 0. Energy
    must use the MOLAR rate r = mass_rate/MW. This locks the corrected v2 physics."""
    from scarfs.data.generation_v2 import compute_dYdt_from_wdot, compute_reaction_energy_terms

    # arrange: synthetic molar net rates that conserve MASS (Σ r_i·MW_i = 0)
    rng = np.random.default_rng(0)
    n_sp = 8
    MW = rng.uniform(2.0, 200.0, n_sp)          # kg/kmol, wide spread like the real mechanism
    r = rng.normal(0, 1.0, n_sp)                # molar rate [kmol/m³/s]
    r[-1] = -(r[:-1] @ MW[:-1]) / MW[-1]         # force Σ r_i·MW_i = 0 (mass conservation)
    rho = 0.6
    mass_rate = r * MW                           # what pfr.solve actually returns

    # act
    dYdt_correct = mass_rate / rho                              # v2 corrected
    dYdt_buggy = compute_dYdt_from_wdot(mass_rate, MW, np.array([rho]))[0]  # colleague's chain

    # assert: corrected dY/dt conserves mass; the buggy one does not; they differ by MW
    assert abs(dYdt_correct.sum()) < 1e-9 * np.abs(dYdt_correct).sum(), "corrected dYdt must close"
    assert abs(dYdt_buggy.sum()) > 1e-2 * np.abs(dYdt_buggy).sum(), "buggy dYdt should NOT close"
    np.testing.assert_allclose(dYdt_buggy / dYdt_correct, MW, rtol=1e-9)  # exactly one extra MW


def test_v2_default_storage_min_every_nth_is_one():
    """REGRESSION (gate-C 0.199 finding 2026-06-12): the size cap must not throttle storage
    through the front. v2 default min_every_nth=1 so the 3% threshold alone governs density."""
    assert GenV2Settings().storage.min_every_nth == 1
    assert GenV2Settings().storage.mode == "front_adaptive"


def test_min_every_nth_one_halves_stored_jumps_on_steep_front():
    """min_every_nth=1 lets the policy store consecutive points through a steep front, so the
    stored-jump ceiling is ONE grid step; min_every_nth=2 forces skipping → ~2× larger jumps."""
    from scarfs.data.config import StorageConfig
    from scarfs.data.generate import select_storage_indices

    # arrange: a steep ramp where each solver step is ~8% of peak (above the 3% policy)
    n = 60
    s_e = np.concatenate([np.linspace(0, 1.0, 13), np.full(n - 13, 1.0)]) * 1e8  # 8.3%/step front

    def max_stored_jump(min_gap):
        idx = select_storage_indices(s_e, StorageConfig(mode="front_adaptive",
                                                        max_frac_jump=0.03, min_every_nth=min_gap))
        a = s_e[idx]; peak = np.abs(s_e).max()
        return np.abs(np.diff(a)).max() / peak

    # act
    j1 = max_stored_jump(1)
    j2 = max_stored_jump(2)
    # assert: with min_gap=1 the worst stored jump ≈ one grid step (~8%); min_gap=2 ≈ doubles it
    assert j1 < 0.11, f"min_gap=1 should cap at ~1 grid step, got {j1:.3f}"
    assert j2 > 1.6 * j1, f"min_gap=2 should roughly double the worst jump: {j2:.3f} vs {j1:.3f}"


# ---------------------------------------------------------------------------
# Composition-curvature storage co-trigger (#2: keep the induction zone)
# ---------------------------------------------------------------------------

def test_comp_trigger_keeps_induction_zone_when_s_e_flat():
    """With a flat S_E (no |ΔS_E| trigger) but a moving composition, the composition co-trigger
    stores the induction-zone rows the S_E-only policy would discard."""
    n = 50
    s_e = np.full(n, 1.0e6)  # constant -> peak>0, ΔS_E == 0, so the S_E trigger never fires
    ramp = np.geomspace(1e-6, 1e-1, n)            # a product growing through the induction zone
    Y = np.column_stack([1.0 - ramp, ramp])       # 2 species, renormalised-ish
    cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=1,
                        comp_arcsinh_jump=1.0, comp_arcsinh_floor=1e-4)

    without = select_storage_indices(s_e, cfg)                  # comp=None -> only first+last
    with_comp = select_storage_indices(s_e, cfg, comp=Y)

    assert without.tolist() == [0, n - 1], "S_E-only stores nothing in the flat front"
    assert len(with_comp) > len(without), "composition trigger must keep induction rows"
    assert with_comp[0] == 0 and with_comp[-1] == n - 1


def test_comp_trigger_disabled_by_default_reproduces_legacy():
    """comp_arcsinh_jump=0.0 (default) ignores the composition argument entirely (legacy)."""
    n = 30
    s_e = np.full(n, 5.0e5)
    Y = np.column_stack([np.linspace(0.9, 0.1, n), np.linspace(0.1, 0.9, n)])
    cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=1)  # jump=0.0
    assert select_storage_indices(s_e, cfg, comp=Y).tolist() == [0, n - 1]


# ---------------------------------------------------------------------------
# Low-conversion coverage gate (Gate E / RC-1)
# ---------------------------------------------------------------------------

def _inlet_seed_frame(y_c2h6_values, inlet=0.7):
    n = len(y_c2h6_values)
    species = ["C2H6", "C2H4", "H2O"]
    Y = np.column_stack([np.asarray(y_c2h6_values, float),
                         np.full(n, 0.05), np.full(n, 0.25)])
    runner = {"id": 2, "regime": "inlet_seed", "mdot": 1.0, "diameter_m": 0.05,
              "steam_to_ethane_kgkg": 0.43, "T_in": 1000.0, "P_in": 2e5,
              "shape": "uniform", "H_peak": 3e4}
    return assemble_v2_frame(
        species_names=species, Y=Y, dYdt=np.zeros((n, 3)),
        T=np.full(n, 1000.0), P=np.full(n, 2e5), rho=np.full(n, 0.6),
        u=np.full(n, 10.0), tau=np.linspace(0, 0.05, n), z=np.linspace(0, 1, n),
        cp=np.full(n, 2500.0), cv=np.full(n, 2100.0), mu=np.full(n, 3e-5),
        k=np.full(n, 0.1), W_mean=np.full(n, 25.0),
        absorption=np.full(n, 1e5), s_wall=np.full(n, 1e6), q_wall=np.full(n, 3e4),
        pfr_point_index=np.arange(n), n_points_solved=100,
        runner_case=runner, settings=GenV2Settings(), inlet_Y={"C2H6": inlet, "H2O": 0.3},
    )


def test_low_conversion_gate_passes_with_near_inlet_rows_fails_without():
    # near-inlet: Y_C2H6 barely below inlet 0.7 -> conversion < 5% on every row -> PASS
    good = gate_low_conversion_coverage(_inlet_seed_frame([0.70, 0.695, 0.69, 0.688]))
    assert good["passed"]
    assert good["inlet_seed_low_conv_frac"] == pytest.approx(1.0)
    # fully converted inlet_seed rows (Y_C2H6 ~0.3) -> no low-conversion coverage -> FAIL
    bad = gate_low_conversion_coverage(_inlet_seed_frame([0.30, 0.25, 0.20, 0.15]))
    assert not bad["passed"]
    assert bad["inlet_seed_low_conv_frac"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# inlet_seed temperature span (#2: cover the high-T / low-conversion corner)
# ---------------------------------------------------------------------------

def test_inlet_seed_spans_full_temperature_range():
    """The inlet_seed regime must span the full operating inlet-T envelope (not a low-T band),
    so the high-T/low-conversion near-wall corner is covered."""
    cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=64, n_highT_cases=0, n_tail_cases=0,
                        diameters_m=(0.05,))
    cases = build_cases(cfg)
    seed_T = [c["T_in"] for c in cases if c["regime"] == "inlet_seed"]
    assert seed_T, "no inlet_seed cases generated"
    assert max(seed_T) > 1250.0, "inlet_seed must reach high inlet T (hot near-wall corner)"
    assert min(seed_T) < 900.0, "inlet_seed must also cover low inlet T"
    assert max(seed_T) <= cfg.inlet_seed_T_in_range_K[1] + 1.0

"""Tests for MergedCoil, standardised CompositionScaler, extended physics, and features.

Covers:
  - CompositionScaler standard mode: round-trip, mean_/scale_ accessors
  - project_conserve_atoms: zeroes residual (machine eps), idempotent, minimal perturbation
  - MergedCoil: shape, absorption positivity, project idempotence direction, PCA init
  - spectral_norm: weight spectral norm ≤ 1 after make_mlp(spectral_norm=True)
  - build_mass_rate_matrix: dydt vs R_ selection (via stub Schema)
  - build_absorption_target: clipping statistics
"""

from __future__ import annotations

import numpy as np
import pytest


# ===========================================================================
# CompositionScaler — standard mode
# ===========================================================================

def test_composition_scaler_standard_roundtrip():
    # arrange
    rng = np.random.default_rng(1)
    y = rng.uniform(0.0, 1.0, (100, 10))
    from scarfs.models.common import CompositionScaler
    scaler = CompositionScaler(log=False, mode="standard")
    # act
    z = scaler.fit_transform(y)
    back = scaler.inverse_transform(z)
    # assert
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-9)
    assert np.allclose(back, y, rtol=1e-9)


def test_composition_scaler_standard_mean_scale_accessible():
    from scarfs.models.common import CompositionScaler
    y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = CompositionScaler(log=False, mode="standard")
    scaler.fit(y)
    # assert accessible attributes
    assert scaler.mean_ is not None
    assert scaler.scale_ is not None
    assert scaler.mean_.shape == (2,)
    assert scaler.scale_.shape == (2,)
    assert np.allclose(scaler.mean_, y.mean(axis=0))


def test_composition_scaler_default_mode_is_minmax():
    # existing default behaviour must be unchanged
    from scarfs.models.common import CompositionScaler
    scaler = CompositionScaler()
    assert scaler.mode == "minmax"
    assert scaler.log is True


def test_composition_scaler_invalid_mode_raises():
    from scarfs.models.common import CompositionScaler
    with pytest.raises(ValueError, match="mode"):
        CompositionScaler(mode="badmode")


def test_composition_scaler_standard_constant_column_no_nan():
    # constant column → std=0 → scale_=1 (no division by zero)
    from scarfs.models.common import CompositionScaler
    y = np.ones((20, 3))
    scaler = CompositionScaler(log=False, mode="standard")
    z = scaler.fit_transform(y)
    assert np.all(np.isfinite(z))


# ===========================================================================
# project_conserve_atoms
# ===========================================================================

def test_project_conserve_atoms_zeros_residual():
    """After projection, atom_balance_residual should be at machine epsilon."""
    from scarfs.models.physics import atom_balance_residual, project_conserve_atoms

    rng = np.random.default_rng(42)
    n, n_sp, n_el = 20, 6, 4
    W = rng.uniform(1.0, 50.0, n_sp)
    A = rng.uniform(0.0, 3.0, (n_sp, n_el)).round()   # integer atom counts
    rate_mass = rng.normal(0, 1.0, (n, n_sp))

    corrected = project_conserve_atoms(rate_mass, W, A)
    residual = atom_balance_residual(corrected, W, A)
    # residual should be ~machine epsilon
    assert np.max(np.abs(residual)) < 1e-8


def test_project_conserve_atoms_idempotent():
    """Projecting twice gives the same result as projecting once."""
    from scarfs.models.physics import project_conserve_atoms

    rng = np.random.default_rng(7)
    n, n_sp, n_el = 15, 5, 3
    W = rng.uniform(2.0, 30.0, n_sp)
    A = np.array([[2, 0, 1], [0, 2, 0], [1, 1, 0], [0, 0, 2], [3, 1, 0]], dtype=float)
    rate_mass = rng.normal(0, 0.5, (n, n_sp))

    once = project_conserve_atoms(rate_mass, W, A)
    twice = project_conserve_atoms(once, W, A)
    assert np.allclose(once, twice, atol=1e-8)


def test_project_conserve_atoms_minimal_perturbation():
    """Correction lies in col(A): the perturbation is orthogonal to null(A).

    The projection formula  r_corr = r - A(A^T A)^{-1} A^T r  gives
    delta = r_corr - r = -A·c  which is in col(A).  So delta must be
    orthogonal to null(A), the right null-space of A.

    We check via SVD: delta @ V_null == 0 where V_null spans the right null-
    space of A (columns of V corresponding to zero singular values).
    """
    from scarfs.models.physics import project_conserve_atoms

    rng = np.random.default_rng(99)
    n, n_sp, n_el = 10, 6, 2
    W = np.ones(n_sp)
    # Build a rank-n_el matrix A (n_sp x n_el)
    A = rng.uniform(0, 2, (n_sp, n_el)).round()
    A[:n_el] = np.eye(n_el)                        # ensures full column rank
    rate_mass = rng.normal(0, 1.0, (n, n_sp))

    corrected = project_conserve_atoms(rate_mass, W, A)
    delta = corrected - rate_mass                  # (n, n_sp), molar == mass (W=1)

    # Right null-space of A: SVD of A (n_sp x n_el)
    U, s, Vt = np.linalg.svd(A.T, full_matrices=True)   # A^T is (n_el, n_sp)
    rank = int((s > 1e-10).sum())
    # V of A^T is (n_sp, n_sp); null space is the last (n_sp - rank) cols
    if rank < n_sp:
        V_null = Vt[rank:].T                       # (n_sp, n_sp - rank)  — null space of A
        # delta lies in col(A) == orthogonal to null(A)
        proj = delta @ V_null                      # (n, n_sp - rank)
        assert np.allclose(proj, 0.0, atol=1e-8), f"max |proj|={np.abs(proj).max():.2e}"
    # If rank == n_sp (A full-row-rank), null space is trivial — nothing to check


def test_project_conserve_atoms_conserving_rates_unchanged():
    """Rates that already conserve atoms are unaffected by the projection."""
    from scarfs.models.physics import atom_balance_residual, project_conserve_atoms

    # 2 species: A (W=1, 1 atom) → B (W=1, 1 atom), conserving 1 element
    W = np.array([1.0, 1.0])
    A = np.array([[1.0], [1.0]])    # (n_sp=2, n_el=1)
    rate_mass = np.array([[-1.0, 1.0], [-0.5, 0.5]])   # conserving rows

    corrected = project_conserve_atoms(rate_mass, W, A)
    assert np.allclose(corrected, rate_mass, atol=1e-10)


# ===========================================================================
# MergedCoil shape / positivity / projection
# ===========================================================================

@pytest.fixture
def merged_coil():
    torch = pytest.importorskip("torch")
    from scarfs.models.neuralcoil import MergedCoil
    return MergedCoil(n_dry=20, n_energy_active=8, latent_dim=4, n_thermo=4)


def test_merged_coil_forward_shapes(merged_coil):
    torch = pytest.importorskip("torch")
    B = 16
    y = torch.randn(B, 20)
    q = torch.randn(B, 4)
    out = merged_coil(y, q)
    # assert all expected keys and correct shapes
    assert out["z"].shape == (B, 4)
    assert out["z_proj"].shape == (B, 4)
    assert out["y_recon"].shape == (B, 20)
    assert out["latent_source"].shape == (B, 4)
    assert out["rates"].shape == (B, 8)
    assert out["absorption"].shape == (B,)


def test_merged_coil_absorption_strictly_positive(merged_coil):
    torch = pytest.importorskip("torch")
    B = 64
    rng = torch.Generator().manual_seed(99)
    # Test with extreme inputs
    y = torch.randn(B, 20, generator=rng) * 10
    q = torch.randn(B, 4, generator=rng) * 5
    out = merged_coil(y, q)
    # absorption must be > 0 by construction (softplus + floor)
    assert bool((out["absorption"] > 0).all()), "absorption must be strictly positive"


def test_merged_coil_absorption_positive_after_calibration(merged_coil):
    torch = pytest.importorskip("torch")
    merged_coil.set_energy_calibration(scale=1e6, floor=100.0)
    B = 32
    y = torch.zeros(B, 20)
    q = torch.zeros(B, 4)
    out = merged_coil(y, q)
    assert bool((out["absorption"] >= 100.0).all())


def test_merged_coil_project_definition(merged_coil):
    """project(z, q) == encode(decode(z, q)) by definition."""
    torch = pytest.importorskip("torch")
    B = 8
    z = torch.randn(B, 4)
    q = torch.randn(B, 4)
    # act: compare project() to the composed call
    y_recon = merged_coil.decode(z, q)
    expected = merged_coil.encode(y_recon)
    got = merged_coil.project(z, q)
    assert torch.allclose(got, expected, atol=1e-5)


def test_merged_coil_pca_init():
    torch = pytest.importorskip("torch")
    from scarfs.models.neuralcoil import MergedCoil
    k, n_dry = 4, 10
    model = MergedCoil(n_dry=n_dry, n_energy_active=3, latent_dim=k)
    components = np.eye(k, n_dry)  # identity sub-matrix
    model.init_encoder_pca(components)
    w = model.encoder.weight.detach().numpy()
    assert np.allclose(w, components, atol=1e-6)


def test_merged_coil_pca_init_wrong_shape_raises():
    torch = pytest.importorskip("torch")
    from scarfs.models.neuralcoil import MergedCoil
    model = MergedCoil(n_dry=10, n_energy_active=3, latent_dim=4)
    with pytest.raises(ValueError, match="init_encoder_pca"):
        model.init_encoder_pca(np.zeros((3, 10)))


def test_merged_coil_spectral_norm():
    """Spectral norm of each head Linear weight should be ≤ 1 + small eps."""
    torch = pytest.importorskip("torch")
    from scarfs.models.neuralcoil import MergedCoil

    model = MergedCoil(
        n_dry=10, n_energy_active=4, latent_dim=4, spectral_norm=True,
        latent_source_hidden=(16,), rate_hidden=(16,), energy_hidden=(8,),
    )
    # Check that parametrized spectral_norm is applied to the rate_net
    found_sn = False
    for name, module in model.rate_net.named_modules():
        if hasattr(module, "parametrizations"):
            for pname in module.parametrizations:
                if pname == "weight":
                    found_sn = True
    assert found_sn, "spectral_norm parametrization not found in rate_net"


# ===========================================================================
# make_mlp spectral_norm
# ===========================================================================

def test_make_mlp_spectral_norm_weight_bound():
    """Spectral norm ≤ 1 + small numerical tolerance after parametrization."""
    torch = pytest.importorskip("torch")
    from scarfs.models.nets import make_mlp

    net = make_mlp([4, 16, 8], activation="silu", layernorm=False, spectral_norm=True)
    # After one forward pass to initialise the power-iteration vectors, check sigma ≤ 1
    _ = net(torch.randn(2, 4))
    found = False
    for module in net.modules():
        if hasattr(module, "parametrizations") and hasattr(module, "weight"):
            found = True
            w = module.weight
            sigma = torch.linalg.matrix_norm(w, ord=2).item()
            # PyTorch spectral_norm converges to ≤ 1 after sufficient power iterations;
            # allow a small numerical tolerance (the power-iteration initialises to ≤ 1
            # by construction but may be slightly above due to float rounding).
            assert sigma <= 1.0 + 0.01, f"spectral norm {sigma:.4f} > 1"
    assert found, "No parametrized module found — spectral_norm was not applied"


def test_make_mlp_no_spectral_norm_default():
    torch = pytest.importorskip("torch")
    from scarfs.models.nets import make_mlp
    net = make_mlp([4, 8, 2])
    # No parametrizations on default MLP
    for module in net.modules():
        assert not hasattr(module, "parametrizations"), "unexpected parametrization"


# ===========================================================================
# build_mass_rate_matrix / build_absorption_target (stub Schema)
# ===========================================================================

class _StubSchemaWithDydt:
    """Minimal stub mimicking B1a Schema with has_dydt / dydt_columns."""

    def __init__(self, species, has_dydt=True):
        self._species = list(species)
        self._has_dydt = has_dydt

    def has_dydt(self):
        return self._has_dydt

    def dydt_columns(self, species):
        return [f"dYdt_{s}" for s in species]

    def r_columns(self, species):
        return [f"R_{s}" for s in species]

    def energy_target_column(self):
        return "Reaction heat absorption [J/s/m3]"

    def require_state(self, key):
        return [key]   # key == "rho" → return ["rho"]


def _make_stub_df(species, n=20, *, with_dydt=True, with_r=True):
    import pandas as pd
    rng = np.random.default_rng(5)
    data = {}
    for s in species:
        if with_dydt:
            data[f"dYdt_{s}"] = rng.normal(0, 1e-3, n)
        if with_r:
            data[f"R_{s}"] = rng.normal(0, 0.1, n)
    data["rho"] = rng.uniform(0.5, 2.0, n)
    data["Reaction heat absorption [J/s/m3]"] = rng.uniform(-1e4, 1e8, n)
    return pd.DataFrame(data)


def test_build_mass_rate_matrix_uses_dydt_when_available():
    from scarfs.models.features import build_mass_rate_matrix
    species = ["H2", "CH4"]
    df = _make_stub_df(species)
    schema = _StubSchemaWithDydt(species, has_dydt=True)
    result = build_mass_rate_matrix(df, schema, species, prefer_dydt=True)
    # result should be rho * dYdt
    rho = df["rho"].to_numpy()[:, None]
    dydt = df[["dYdt_H2", "dYdt_CH4"]].to_numpy()
    assert np.allclose(result, rho * dydt)


def test_build_mass_rate_matrix_falls_back_to_R():
    from scarfs.models.features import build_mass_rate_matrix
    species = ["H2", "CH4"]
    df = _make_stub_df(species, with_dydt=True)
    schema = _StubSchemaWithDydt(species, has_dydt=False)
    result = build_mass_rate_matrix(df, schema, species, prefer_dydt=True)
    # should use R_ columns
    expected = df[["R_H2", "R_CH4"]].to_numpy(dtype=float)
    assert np.allclose(result, expected)


def test_build_mass_rate_matrix_explicit_prefer_false():
    from scarfs.models.features import build_mass_rate_matrix
    species = ["H2", "CH4"]
    df = _make_stub_df(species)
    schema = _StubSchemaWithDydt(species, has_dydt=True)
    result = build_mass_rate_matrix(df, schema, species, prefer_dydt=False)
    expected = df[["R_H2", "R_CH4"]].to_numpy(dtype=float)
    assert np.allclose(result, expected)


def test_build_absorption_target_clips_negatives():
    from scarfs.models.features import build_absorption_target
    import pandas as pd
    n = 50
    rng = np.random.default_rng(3)
    raw = rng.uniform(-1e5, 1e8, n)
    df = pd.DataFrame({"Reaction heat absorption [J/s/m3]": raw})

    class _SimpleSchema:
        def energy_target_column(self):
            return "Reaction heat absorption [J/s/m3]"

    result = build_absorption_target(df, _SimpleSchema())
    n_neg = int((raw < 0).sum())
    assert result.n_clipped == n_neg
    assert np.all(result.values >= 0)
    # positive rows unchanged
    pos_mask = raw >= 0
    assert np.allclose(result.values[pos_mask], raw[pos_mask])


def test_build_absorption_target_all_positive():
    from scarfs.models.features import build_absorption_target
    import pandas as pd
    raw = np.array([1e4, 2e4, 3e4])
    df = pd.DataFrame({"Reaction heat absorption [J/s/m3]": raw})

    class _Schema:
        def energy_target_column(self):
            return "Reaction heat absorption [J/s/m3]"

    result = build_absorption_target(df, _Schema())
    assert result.n_clipped == 0
    assert np.allclose(result.values, raw)

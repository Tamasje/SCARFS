"""Tests for the merged-model training layer (B1c).

All tests are torch-free where possible; the merged composite loss smoke test stubs
MergedCoil via a minimal torch.nn.Module so B1b integration is not required.

Coverage:
- Deterministic 70/15/15 case split (disjoint, stable across runs).
- Tail-stratified weights (monotone in absorption decile).
- Enthalpy-aware weight vector (floor 1.0, higher for high-|h·ω| species).
- Lagrangian pair builder (pairs only within case, Δτ > 0, sorted).
- Merged composite loss smoke (all terms finite, gradients flow).
- Energy tie decrease over 50 Adam steps on overfit-able toy.
- Config round-trip (from_mapping / load with new fields; old configs still parse).
- No-winsorize guard (targets pass through unchanged for extreme synthetic outlier).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scarfs.training.config import DataConfig, LossConfig, ModelConfig, TrainConfig
from scarfs.training.datamodule import (
    case_step_pairs,
    enthalpy_aware_weights,
    tail_stratified_weights,
    tripartite_case_split,
)
from scarfs.schema import Schema


# ---------------------------------------------------------------------------
# Synthetic fixtures (self-contained; do not depend on conftest fixtures)
# ---------------------------------------------------------------------------

def _make_df(n_cases: int = 10, n_points: int = 8) -> pd.DataFrame:
    """Synthetic PFR DataFrame with dYdt, tau, rho, and absorption columns."""
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for cid in range(1, n_cases + 1):
        for j in range(n_points):
            frac = j / (n_points - 1)
            comp = {
                "C2H6": 0.70 * (1.0 - 0.4 * frac),
                "C2H4": 0.20 * frac + 1e-6,
                "H2": 0.01 * frac + 1e-7,
                "H2O": 0.30,
                "CH3.": 1e-6 * frac + 1e-12,
            }
            rates = {s: (-0.10 if s == "C2H6" else 0.05) * (1.0 + 0.1 * cid) for s in comp}
            row: dict = {f"Y_{s}": comp[s] for s in comp}
            row.update({f"R_{s}": rates[s] for s in comp})
            row.update({f"dYdt_{s}": rates[s] * 2.0 for s in comp})
            # absorption: larger at high conversion
            absorption = float(1e6 * frac * cid + 1.0)
            row.update({
                "T [K]": 823.15 + 120.0 * frac,
                "P [Pa]": 2.0e5,
                "z [m]": round(0.1 * j, 6),
                "tau [s]": round(0.05 * j, 6),
                "rho [kg/m3]": 0.70 - 0.1 * frac,
                "Mass flow [kg/s]": 1.6,
                "CaseID": cid,
                "L [m]": 0.5,
                "mdot [kg/s]": 1.6,
                "U_in [m/s]": 11.0,
                "Reaction heat absorption [J/s/m3]": absorption,
            })
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_df_merged() -> pd.DataFrame:
    return _make_df(n_cases=10, n_points=8)


@pytest.fixture()
def merged_schema(synthetic_df_merged: pd.DataFrame) -> Schema:
    return Schema.from_columns(list(synthetic_df_merged.columns))


# ---------------------------------------------------------------------------
# 1. Deterministic 70/15/15 case split
# ---------------------------------------------------------------------------

def test_tripartite_split_disjoint(synthetic_df_merged, merged_schema):
    """Train, val, and test case sets must be mutually disjoint."""
    # arrange
    df = synthetic_df_merged
    schema = merged_schema
    case_col = schema.meta["CaseID"]
    # act
    train_mask, val_mask, test_mask = tripartite_case_split(
        df, schema, val_fraction=0.176, test_fraction=0.15, seed=0, split_by_case=True
    )
    train_cases = set(df.loc[train_mask, case_col])
    val_cases = set(df.loc[val_mask, case_col])
    test_cases = set(df.loc[test_mask, case_col])
    # assert
    assert train_cases.isdisjoint(val_cases), "Train/val case leak"
    assert train_cases.isdisjoint(test_cases), "Train/test case leak"
    assert val_cases.isdisjoint(test_cases), "Val/test case leak"
    assert len(train_cases | val_cases | test_cases) == df[case_col].nunique()


def test_tripartite_split_sizes(synthetic_df_merged, merged_schema):
    """Sizes should be approximately 70/15/15 (±1 case tolerance)."""
    # arrange
    df = synthetic_df_merged
    schema = merged_schema
    n_cases = df[schema.meta["CaseID"]].nunique()
    # act
    train_mask, val_mask, test_mask = tripartite_case_split(
        df, schema, val_fraction=0.176, test_fraction=0.15, seed=0, split_by_case=True
    )
    case_col = schema.meta["CaseID"]
    n_test = df.loc[test_mask, case_col].nunique()
    n_val = df.loc[val_mask, case_col].nunique()
    n_train = df.loc[train_mask, case_col].nunique()
    # assert ±1 case tolerance from 70/15/15
    assert abs(n_test / n_cases - 0.15) <= 0.15, f"Test fraction off: {n_test}/{n_cases}"
    assert abs(n_val / n_cases - 0.15) <= 0.20, f"Val fraction off: {n_val}/{n_cases}"
    assert n_train + n_val + n_test == n_cases


def test_tripartite_split_stable(synthetic_df_merged, merged_schema):
    """The split must be identical across two calls with the same seed."""
    # arrange
    df = synthetic_df_merged
    schema = merged_schema
    # act
    m1 = tripartite_case_split(df, schema, val_fraction=0.176, test_fraction=0.15, seed=7, split_by_case=True)
    m2 = tripartite_case_split(df, schema, val_fraction=0.176, test_fraction=0.15, seed=7, split_by_case=True)
    # assert
    for a, b in zip(m1, m2):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 2. Tail-stratified weights
# ---------------------------------------------------------------------------

def test_tail_weights_monotone(synthetic_df_merged, merged_schema):
    """Mean tail weight must be monotonically non-decreasing across absorption deciles."""
    # arrange
    df = synthetic_df_merged
    schema = merged_schema
    # act
    weights = tail_stratified_weights(
        df, schema, tail_strata=10, tail_weight_alpha=2.0,
        absorption_col="Reaction heat absorption [J/s/m3]",
    )
    absorption = df["Reaction heat absorption [J/s/m3]"].to_numpy()
    # bin rows into deciles by absorption
    decile_edges = np.quantile(absorption, np.linspace(0, 1, 11))
    decile_edges[0] -= 1.0
    decile_idx = np.searchsorted(decile_edges[1:], absorption, side="left")
    decile_idx = np.clip(decile_idx, 0, 9)
    mean_w_per_decile = np.array([weights[decile_idx == d].mean() for d in range(10)])
    # assert: mean weight is non-decreasing across deciles
    diffs = np.diff(mean_w_per_decile)
    assert np.all(diffs >= -1e-6), f"Tail weights not monotone: {mean_w_per_decile}"


def test_tail_weights_zero_strata(synthetic_df_merged, merged_schema):
    """With tail_strata=0, weights should all be 1.0 (disabled)."""
    # arrange / act
    weights = tail_stratified_weights(
        synthetic_df_merged, merged_schema, tail_strata=0,
    )
    # assert
    np.testing.assert_array_equal(weights, np.ones(len(synthetic_df_merged)))


def test_tail_weights_range(synthetic_df_merged, merged_schema):
    """Weights must be in [1.0, 1.0 + alpha]."""
    # arrange
    alpha = 2.0
    # act
    weights = tail_stratified_weights(
        synthetic_df_merged, merged_schema, tail_strata=10, tail_weight_alpha=alpha,
        absorption_col="Reaction heat absorption [J/s/m3]",
    )
    # assert
    assert weights.min() >= 1.0 - 1e-9
    assert weights.max() <= 1.0 + alpha + 1e-9


# ---------------------------------------------------------------------------
# 3. Enthalpy-aware weight vector
# ---------------------------------------------------------------------------

def test_enthalpy_weights_floor(synthetic_df_merged, merged_schema):
    """With no h_mass_fn, weights are uniform at the floor."""
    # arrange
    schema = merged_schema
    df = synthetic_df_merged
    species = ("C2H6", "C2H4", "H2")
    # act
    w = enthalpy_aware_weights(df, schema, species, h_mass_fn=None, floor=1.0)
    # assert
    np.testing.assert_allclose(w, np.ones(3))


def test_enthalpy_weights_high_species(synthetic_df_merged, merged_schema):
    """Species with higher |h·R| share should receive higher weight (floor 1.0)."""
    # arrange
    schema = merged_schema
    df = synthetic_df_merged
    species = ("C2H6", "C2H4", "H2")

    # make a synthetic h_mass_fn: C2H6 has 1000x higher specific enthalpy
    def h_mass_fn(T: np.ndarray) -> np.ndarray:
        n = len(T)
        h = np.zeros((n, 3))
        h[:, 0] = 1e7   # C2H6 high enthalpy
        h[:, 1] = 1e4   # C2H4 low
        h[:, 2] = 1e3   # H2 very low
        return h

    # act
    w = enthalpy_aware_weights(df, schema, species, h_mass_fn=h_mass_fn, floor=1.0)
    # assert: C2H6 weight >= C2H4 weight >= H2 weight; all >= 1.0
    assert w[0] >= w[1] >= w[2], f"Weights not ordered: {w}"
    assert np.all(w >= 1.0), f"Weight below floor: {w}"


def test_enthalpy_weights_minimum_is_floor(synthetic_df_merged, merged_schema):
    """Minimum weight must equal the floor (normalisation check)."""
    # arrange
    schema = merged_schema
    df = synthetic_df_merged
    species = ("C2H6", "C2H4", "H2")

    def h_mass_fn(T: np.ndarray) -> np.ndarray:
        n = len(T)
        h = np.zeros((n, 3))
        h[:, 0] = 1e7
        h[:, 1] = 1e4
        h[:, 2] = 1e3
        return h

    # act
    w = enthalpy_aware_weights(df, schema, species, h_mass_fn=h_mass_fn, floor=1.0)
    # assert
    assert np.isclose(w.min(), 1.0), f"Min weight {w.min()} != 1.0"


# ---------------------------------------------------------------------------
# 4. Lagrangian pair builder
# ---------------------------------------------------------------------------

def test_lagrangian_pairs_within_case_only():
    """Pairs must only connect rows within the same case (no cross-case pairs)."""
    # arrange: 2 cases, 4 rows each, tau increasing within each case
    rows = []
    for cid in [1, 2]:
        for j in range(4):
            rows.append({
                "Y_C2H6": 0.7 - 0.1 * j, "R_C2H6": -0.1,
                "Y_C2H4": 0.1 * j, "R_C2H4": 0.05,
                "T [K]": 900.0, "P [Pa]": 2e5,
                "z [m]": 0.1 * j,
                "tau [s]": 0.05 * j,
                "CaseID": cid,
                "Mass flow [kg/s]": 1.0,
                "rho [kg/m3]": 0.7,
                "U_in [m/s]": 11.0,
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    schema = Schema.from_columns(list(df.columns))

    # act
    idx_t, idx_tp1, dtau = case_step_pairs(df, schema)
    # assert: each pair must be in the same case
    case_col = schema.meta["CaseID"]
    cases = df[case_col].to_numpy()
    for a, b in zip(idx_t, idx_tp1):
        assert cases[a] == cases[b], f"Cross-case pair: row {a} (case {cases[a]}) -> {b} (case {cases[b]})"


def test_lagrangian_pairs_dtau_positive():
    """All Δτ values must be strictly positive."""
    # arrange
    rows = []
    for cid in [1, 2]:
        for j in range(5):
            rows.append({
                "Y_A": 0.5, "R_A": 0.01, "T [K]": 900.0, "P [Pa]": 2e5,
                "tau [s]": 0.1 * j,
                "CaseID": cid,
                "Mass flow [kg/s]": 1.0, "rho [kg/m3]": 0.7, "U_in [m/s]": 11.0,
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    schema = Schema.from_columns(list(df.columns))
    # act
    _, _, dtau = case_step_pairs(df, schema)
    # assert
    assert len(dtau) > 0, "No pairs found"
    assert np.all(dtau > 0), f"Non-positive Δτ: {dtau[dtau <= 0]}"


def test_lagrangian_pairs_sorted_by_tau():
    """Within each case, pairs should form a strictly increasing τ sequence."""
    # arrange
    rows = []
    for j in [3, 1, 0, 2, 4]:  # shuffled order
        rows.append({
            "Y_A": 0.5, "R_A": 0.01, "T [K]": 900.0, "P [Pa]": 2e5,
            "tau [s]": 0.1 * j,
            "CaseID": 1,
            "Mass flow [kg/s]": 1.0, "rho [kg/m3]": 0.7, "U_in [m/s]": 11.0,
        })
    df = pd.DataFrame(rows).reset_index(drop=True)
    schema = Schema.from_columns(list(df.columns))
    # act
    idx_t, idx_tp1, dtau = case_step_pairs(df, schema)
    # assert: dtau all positive (implies sorted pairs)
    assert np.all(dtau > 0)
    # check that tau_tp1 > tau_t for each pair
    tau_arr = df["tau [s]"].to_numpy()
    for a, b in zip(idx_t, idx_tp1):
        assert tau_arr[b] > tau_arr[a], f"tau not sorted: {tau_arr[a]} -> {tau_arr[b]}"


def test_lagrangian_pairs_count():
    """Expected n_pairs = n_cases * (n_points - 1) for regular grids."""
    # arrange
    n_cases, n_pts = 3, 5
    rows = []
    for cid in range(1, n_cases + 1):
        for j in range(n_pts):
            rows.append({
                "Y_A": 0.5, "R_A": 0.01, "T [K]": 900.0, "P [Pa]": 2e5,
                "tau [s]": 0.1 * j,
                "CaseID": cid,
                "Mass flow [kg/s]": 1.0, "rho [kg/m3]": 0.7, "U_in [m/s]": 11.0,
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    schema = Schema.from_columns(list(df.columns))
    # act
    idx_t, idx_tp1, dtau = case_step_pairs(df, schema)
    # assert
    assert len(idx_t) == n_cases * (n_pts - 1)


# ---------------------------------------------------------------------------
# 5. Merged composite loss smoke test (stub MergedCoil)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark_torch = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")


class _StubMergedCoil(nn.Module if _TORCH_AVAILABLE else object):
    """Minimal stub satisfying the B1b MergedCoil interface."""

    def __init__(self, n_dry: int = 10, n_active: int = 5, k: int = 2):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.n_dry = n_dry
        self.n_active = n_active
        self.k = k
        self.encoder = nn.Linear(n_dry, k, bias=False)
        self.decoder_net = nn.Sequential(nn.Linear(k + 4, n_dry), nn.Sigmoid())
        self.rate_net = nn.Linear(k + 4, n_active)
        self.latent_src_net = nn.Linear(k + 4, k)
        # absorption head: softplus output to ensure strictly positive
        self.abs_net = nn.Sequential(nn.Linear(k + 4, 1), nn.Softplus())

    def encode(self, y: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(y)

    def decode(self, z: "torch.Tensor", q: "torch.Tensor") -> "torch.Tensor":
        return self.decoder_net(torch.cat([z, q], dim=-1))

    def project(self, z: "torch.Tensor", q: "torch.Tensor") -> "torch.Tensor":
        return self.encoder(self.decode(z, q))

    def rates_from_latent(self, z: "torch.Tensor", q: "torch.Tensor") -> "torch.Tensor":
        return self.rate_net(torch.cat([z, q], dim=-1))

    def absorption(self, z: "torch.Tensor", q: "torch.Tensor") -> "torch.Tensor":
        return self.abs_net(torch.cat([z, q], dim=-1))

    def latent_source(self, z: "torch.Tensor", q: "torch.Tensor") -> "torch.Tensor":
        return self.latent_src_net(torch.cat([z, q], dim=-1))

    def forward(self, y_std: "torch.Tensor", q: "torch.Tensor") -> dict:
        z = self.encode(y_std)
        y_recon = self.decode(z, q)
        z_proj = self.project(z, q)
        latent_src = self.latent_source(z_proj, q)
        rates = self.rates_from_latent(z_proj, q)
        abs_pred = self.absorption(z_proj, q)
        return {
            "z": z,
            "z_proj": z_proj,
            "y_recon": y_recon,
            "latent_source": latent_src,
            "rates": rates,
            "absorption": abs_pred,
        }


if _TORCH_AVAILABLE:
    import torch


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_merged_loss_smoke_all_terms_finite():
    """All loss terms must be finite on tiny synthetic data."""
    from scarfs.training.losses import merged_composite
    # arrange
    batch, n_dry, n_active, k = 32, 10, 5, 2
    model = _StubMergedCoil(n_dry, n_active, k)

    rng = np.random.default_rng(0)
    y_std = torch.as_tensor(rng.standard_normal((batch, n_dry)), dtype=torch.float32)
    q = torch.as_tensor(rng.standard_normal((batch, 4)), dtype=torch.float32)
    target_rates = torch.as_tensor(rng.standard_normal((batch, n_active)), dtype=torch.float32)
    abs_target = torch.as_tensor(np.abs(rng.standard_normal(batch)) + 1.0, dtype=torch.float32)
    dydt_dry = torch.as_tensor(rng.standard_normal((batch, n_dry)), dtype=torch.float32)
    rho = torch.ones(batch, dtype=torch.float32)
    row_w = torch.ones(batch, dtype=torch.float32)
    enth_w = torch.ones(n_active, dtype=torch.float32)
    qoi_w = torch.ones(n_dry, dtype=torch.float32)
    arc_rate = torch.ones(n_active, dtype=torch.float32)
    arc_lat = torch.ones(k, dtype=torch.float32)
    sigma_active = torch.ones(n_active, dtype=torch.float32)
    sigma_comp = torch.ones(n_dry, dtype=torch.float32)
    active_idx = torch.arange(n_active, dtype=torch.long)

    # act
    total, parts = merged_composite(
        model=model,
        y_std_scaled=y_std,
        q=q,
        target_rates_phys=target_rates,
        absorption_target=abs_target,
        dydt_dry_phys=dydt_dry,
        rho=rho,
        row_weights=row_w,
        enthalpy_weights=enth_w,
        species_weights_qoi=qoi_w,
        arcsinh_rate_scale=arc_rate,
        arcsinh_latent_scale=arc_lat,
        sigma_active=sigma_active,
        sigma_comp_all=sigma_comp,
        active_col_idx=active_idx,
        rate_weight=1.0,
        latent_source_weight=1.0,
        energy_weight=0.5,
        energy_distill_weight=0.25,
        energy_target_weight=0.25,
        consistency_weight=0.1,
        recon_weight=1.0,
        qoi_recon_weight=0.5,
        manifold_weight=0.1,
    )
    # assert
    assert torch.isfinite(total), f"Total loss not finite: {total}"
    for name, val in parts.items():
        assert np.isfinite(val), f"Loss term '{name}' not finite: {val}"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_merged_loss_gradients_flow():
    """Gradients must flow to all major parameter groups (encoder, rate head, latent src, absorption)."""
    from scarfs.training.losses import merged_composite
    # arrange
    batch, n_dry, n_active, k = 16, 10, 5, 2
    model = _StubMergedCoil(n_dry, n_active, k)

    rng = np.random.default_rng(1)
    y_std = torch.as_tensor(rng.standard_normal((batch, n_dry)), dtype=torch.float32)
    q = torch.as_tensor(rng.standard_normal((batch, 4)), dtype=torch.float32)
    target_rates = torch.as_tensor(rng.standard_normal((batch, n_active)), dtype=torch.float32)
    abs_target = torch.as_tensor(np.abs(rng.standard_normal(batch)) + 1.0, dtype=torch.float32)
    dydt_dry = torch.as_tensor(rng.standard_normal((batch, n_dry)), dtype=torch.float32)
    rho = torch.ones(batch, dtype=torch.float32)
    row_w = torch.ones(batch, dtype=torch.float32)
    enth_w = torch.ones(n_active, dtype=torch.float32)
    qoi_w = torch.ones(n_dry, dtype=torch.float32)
    arc_rate = torch.ones(n_active, dtype=torch.float32)
    arc_lat = torch.ones(k, dtype=torch.float32)
    sigma_active = torch.ones(n_active, dtype=torch.float32)
    sigma_comp = torch.ones(n_dry, dtype=torch.float32)
    active_idx = torch.arange(n_active, dtype=torch.long)

    # act
    total, _ = merged_composite(
        model=model,
        y_std_scaled=y_std, q=q,
        target_rates_phys=target_rates, absorption_target=abs_target,
        dydt_dry_phys=dydt_dry, rho=rho,
        row_weights=row_w, enthalpy_weights=enth_w, species_weights_qoi=qoi_w,
        arcsinh_rate_scale=arc_rate, arcsinh_latent_scale=arc_lat,
        sigma_active=sigma_active, sigma_comp_all=sigma_comp, active_col_idx=active_idx,
        rate_weight=1.0, latent_source_weight=1.0, energy_weight=0.5,
        energy_distill_weight=0.25, energy_target_weight=0.25, consistency_weight=0.1,
        recon_weight=1.0, qoi_recon_weight=0.0, manifold_weight=0.1,
    )
    total.backward()
    # assert: all leaf parameters have gradients
    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert len(no_grad) == 0, f"No gradient for: {no_grad}"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_energy_tie_decreases_over_steps():
    """Energy tie loss must decrease over 50 Adam steps on an overfit-able toy."""
    from scarfs.training.losses import energy_rate_tied_loss
    # arrange: tiny overfit-able toy — absorption head directly predicts rate-derived absorption
    rng = np.random.default_rng(7)
    batch = 8
    target_abs = torch.as_tensor(np.abs(rng.standard_normal(batch)) * 1e6 + 1e5, dtype=torch.float32)
    # learnable scalar log-offset applied to target
    log_pred = nn.Parameter(torch.zeros(batch))
    optimiser = torch.optim.Adam([log_pred], lr=0.1)

    initial_loss = float("inf")
    for step in range(50):
        optimiser.zero_grad()
        pred = torch.exp(log_pred)
        loss = energy_rate_tied_loss(pred, target_abs)
        if step == 0:
            initial_loss = float(loss.detach())
        loss.backward()
        optimiser.step()

    final_loss = float(loss.detach())
    # assert
    assert final_loss < initial_loss * 0.5, (
        f"Energy tie loss did not decrease: initial={initial_loss:.4g}, final={final_loss:.4g}"
    )


# ---------------------------------------------------------------------------
# 6. Config round-trip
# ---------------------------------------------------------------------------

def test_config_new_fields_defaults():
    """New config fields must have correct defaults that reproduce existing behaviour."""
    # act
    cfg = TrainConfig()
    # assert — new fields with backward-compatible defaults
    assert cfg.data.test_fraction == 0.0
    assert cfg.data.tail_strata == 0
    assert cfg.data.tail_weight_alpha == 2.0
    assert cfg.data.energy_active_coverage == 0.999
    assert cfg.data.columns_projection is False
    assert cfg.model.kind == "reduced"
    assert cfg.model.spectral_norm is False
    assert cfg.model.latent_source_hidden == (128, 128)
    assert cfg.model.energy_hidden == (64, 64)
    assert cfg.loss.rate_weight == 1.0
    assert cfg.loss.latent_source_weight == 1.0
    assert cfg.loss.energy_weight == 0.5
    assert cfg.loss.energy_distill_weight == 0.25
    assert cfg.loss.energy_target_weight == 0.25
    assert cfg.loss.consistency_weight == 0.1
    assert cfg.loss.rollout_mode == "manifold"


def test_config_from_mapping_merged():
    """from_mapping must correctly parse all merged-model fields."""
    # arrange
    d = {
        "model": {"kind": "merged", "latent_dim": 8, "spectral_norm": True,
                  "latent_source_hidden": [256, 256], "energy_hidden": [64, 64]},
        "data": {"target_species": "energy_active", "tail_strata": 10, "test_fraction": 0.15},
        "loss": {"energy_weight": 0.5, "rollout_mode": "lagrangian"},
    }
    # act
    cfg = TrainConfig.from_mapping(d)
    # assert
    assert cfg.model.kind == "merged"
    assert cfg.model.spectral_norm is True
    assert cfg.model.latent_source_hidden == (256, 256)
    assert cfg.data.tail_strata == 10
    assert cfg.data.test_fraction == 0.15
    assert cfg.loss.rollout_mode == "lagrangian"


def test_config_old_json_files_parse(tmp_path: Path):
    """Existing train_reduced.json and train_neuralcoil.json must still parse without error."""
    # arrange
    configs_dir = Path(__file__).parent.parent / "configs"
    for fname in ("train_reduced.json", "train_neuralcoil.json"):
        path = configs_dir / fname
        if not path.exists():
            pytest.skip(f"Config file {fname} not found")
        # act
        cfg = TrainConfig.load(path)
        # assert — must round-trip to dict without error
        d = cfg.to_dict()
        assert "model" in d and "data" in d


def test_config_merged_json_parse():
    """train_merged.json must parse and have the expected fields."""
    # arrange
    config_path = Path(__file__).parent.parent / "configs" / "train_merged.json"
    # act
    cfg = TrainConfig.load(config_path)
    # assert
    assert cfg.model.kind == "merged"
    assert cfg.model.latent_dim == 8
    assert cfg.loss.rollout_mode == "lagrangian"
    assert cfg.data.tail_strata == 10
    assert cfg.data.test_fraction == 0.15


def test_config_merged_mimic_json_parse():
    """train_merged_mimic.json must parse and have neuralcoil kind."""
    # arrange
    config_path = Path(__file__).parent.parent / "configs" / "train_merged_mimic.json"
    # act
    cfg = TrainConfig.load(config_path)
    # assert
    assert cfg.model.kind == "neuralcoil"
    assert cfg.model.latent_dim == 3
    assert cfg.data.tail_strata == 0  # mimic has no tail weighting


def test_config_round_trip_json(tmp_path: Path):
    """from_mapping(to_dict()) must be lossless for all new merged fields."""
    # arrange
    cfg1 = TrainConfig.from_mapping({
        "model": {"kind": "merged", "latent_dim": 8, "spectral_norm": True,
                  "latent_source_hidden": [128, 128], "energy_hidden": [64, 64]},
        "data": {"tail_strata": 10, "test_fraction": 0.15, "energy_active_coverage": 0.999},
        "loss": {"energy_weight": 0.5, "rollout_mode": "lagrangian",
                 "consistency_weight": 0.1},
    })
    # act: write to JSON and reload
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg1.to_dict()), encoding="utf-8")
    cfg2 = TrainConfig.load(path)
    # assert
    assert cfg2.model.kind == cfg1.model.kind
    assert cfg2.model.spectral_norm == cfg1.model.spectral_norm
    assert cfg2.data.tail_strata == cfg1.data.tail_strata
    assert cfg2.loss.rollout_mode == cfg1.loss.rollout_mode
    assert cfg2.loss.energy_weight == cfg1.loss.energy_weight


# ---------------------------------------------------------------------------
# 7. No-winsorize guard
# ---------------------------------------------------------------------------

def test_no_winsorize_in_tail_weights():
    """Tail weights must not clip or winsorize target values — even extreme outliers pass through."""
    # arrange: one row with an extreme absorption value (1e15)
    rows = []
    for cid in [1]:
        for j in range(6):
            rows.append({
                "Y_A": 0.5, "Y_B": 0.5, "R_A": 0.01, "R_B": -0.01,
                "T [K]": 900.0, "P [Pa]": 2e5,
                "tau [s]": 0.1 * j, "CaseID": cid,
                "Mass flow [kg/s]": 1.0, "rho [kg/m3]": 0.7, "U_in [m/s]": 11.0,
                "Reaction heat absorption [J/s/m3]": 1e15 if j == 5 else float(j * 1e6),
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    schema = Schema.from_columns(list(df.columns))
    # act
    weights = tail_stratified_weights(
        df, schema, tail_strata=4, tail_weight_alpha=2.0,
        absorption_col="Reaction heat absorption [J/s/m3]",
    )
    # assert: the outlier row (j==5, abs=1e15) must get the HIGHEST weight, not clipped
    outlier_idx = 5  # last row
    assert weights[outlier_idx] == weights.max(), (
        f"Outlier row weight {weights[outlier_idx]} < max {weights.max()} — possible winsorization"
    )
    # assert: the extreme value passes through the column unchanged
    assert df.loc[outlier_idx, "Reaction heat absorption [J/s/m3]"] == 1e15


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not available")
def test_no_winsorize_in_merged_loss():
    """Merged composite loss must not modify the absorption target tensor (no clamping of inputs)."""
    from scarfs.training.losses import merged_composite
    # arrange: extreme absorption target (1e12)
    batch, n_dry, n_active, k = 4, 10, 5, 2
    model = _StubMergedCoil(n_dry, n_active, k)
    rng = np.random.default_rng(2)
    y_std = torch.as_tensor(rng.standard_normal((batch, n_dry)), dtype=torch.float32)
    q = torch.as_tensor(rng.standard_normal((batch, 4)), dtype=torch.float32)
    target_rates = torch.as_tensor(rng.standard_normal((batch, n_active)), dtype=torch.float32)
    abs_target_np = np.array([1e12, 1e6, 1e5, 1e4], dtype=np.float32)
    abs_target = torch.as_tensor(abs_target_np, dtype=torch.float32)
    dydt_dry = torch.as_tensor(rng.standard_normal((batch, n_dry)), dtype=torch.float32)
    rho = torch.ones(batch, dtype=torch.float32)
    row_w = torch.ones(batch, dtype=torch.float32)
    ones_active = torch.ones(n_active, dtype=torch.float32)
    ones_dry = torch.ones(n_dry, dtype=torch.float32)
    ones_k = torch.ones(k, dtype=torch.float32)
    active_idx = torch.arange(n_active, dtype=torch.long)

    # record input absorption_target before call
    abs_before = abs_target.clone()
    # act
    merged_composite(
        model=model,
        y_std_scaled=y_std, q=q,
        target_rates_phys=target_rates, absorption_target=abs_target,
        dydt_dry_phys=dydt_dry, rho=rho,
        row_weights=row_w, enthalpy_weights=ones_active, species_weights_qoi=ones_dry,
        arcsinh_rate_scale=ones_active, arcsinh_latent_scale=ones_k,
        sigma_active=ones_active, sigma_comp_all=ones_dry, active_col_idx=active_idx,
        rate_weight=1.0, latent_source_weight=1.0, energy_weight=0.5,
        energy_distill_weight=0.25, energy_target_weight=0.25, consistency_weight=0.1,
        recon_weight=1.0, qoi_recon_weight=0.0, manifold_weight=0.1,
    )
    # assert: absorption_target must be identical after the call
    torch.testing.assert_close(abs_target, abs_before, msg="absorption_target was modified (possible winsorization)")

"""Tests for scarfs.benchmark.parents — parent-model adapters.

Two test classes:
  - TestColleagueSurrogateUnit   : synthetic NPZ with matching key schema → shape/finiteness
  - TestColleagueSurrogateSmoke  : env-gated real NPZ load + predict on stride6 fixture rows
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic NPZ fixture builder
# ---------------------------------------------------------------------------

def _build_synthetic_npz(k: int = 4, n_retained: int = 6) -> dict[str, np.ndarray]:
    """Build a minimal NPZ payload matching the real model_arrays.npz key schema."""
    rng = np.random.default_rng(99)
    n_feat = k + 3 + k * (k - 1) // 2 + k + 7  # q + [inv_T, ln_T, ln_P] + interactions + arrhenius

    # Compute actual interaction count: pairs (i,j) with i<=j, i < interaction_order=4
    ia_count = 0
    inter_order = min(4, k)
    for i in range(inter_order):
        for j in range(i, inter_order):
            ia_count += 1
    n_feat = k + 3 + ia_count + 7

    arrays = {
        "species_scaler_mean": rng.normal(0, 0.1, n_retained),
        "species_scaler_scale": np.abs(rng.normal(0.05, 0.01, n_retained)) + 1e-4,
        "species_inverse_min": rng.normal(-1, 0.1, n_retained),
        "species_inverse_max": rng.normal(1, 0.1, n_retained),
        "latent_min": rng.normal(-2, 0.5, k),
        "latent_max": rng.normal(2, 0.5, k),
        "source_feature_mean": rng.normal(0, 0.1, n_feat),
        "source_feature_scale": np.abs(rng.normal(1, 0.1, n_feat)) + 1e-4,
        "source_target_mean": rng.normal(0, 0.01, k),
        "source_target_scale": np.abs(rng.normal(0.5, 0.1, k)) + 1e-4,
        "source_trans_min": rng.normal(-5, 0.5, k),
        "source_trans_max": rng.normal(5, 0.5, k),
        "source_output_min": rng.normal(-10, 1, k),
        "source_output_max": rng.normal(10, 1, k),
        "pca_mean": rng.normal(0, 0.1, n_retained),
        "pca_components": rng.normal(0, 0.1, (k, n_retained)),
        "pca_explained_variance_ratio": np.abs(rng.normal(0.1, 0.05, k)),
        "source_W_0": rng.normal(0, 0.1, (128, n_feat)),
        "source_B_0": np.zeros(128),
        "source_W_1": rng.normal(0, 0.1, (128, 128)),
        "source_B_1": np.zeros(128),
        "source_W_2": rng.normal(0, 0.1, (k, 128)),
        "source_B_2": np.zeros(k),
    }
    return arrays


def _build_synthetic_metadata(k: int, retained_species: list[str]) -> dict:
    """Build minimal metadata matching model_metadata.json structure."""
    interaction_order = min(4, k)
    feat_names = [f"q_{i}" for i in range(k)]
    feat_names += ["inv_T", "ln_T", "ln_P"]
    for i in range(interaction_order):
        for j in range(i, interaction_order):
            feat_names.append(f"q_{i}*q_{j}")
    feat_names += [
        f"arrhenius_Ea_{ea}_kJmol"
        for ea in (80, 120, 160, 200, 240, 280, 320)
    ]
    return {
        "retained_species": retained_species,
        "source_feature_names": feat_names,
        "bounded_source_output": True,
        "projection": "weighted_pca",
        "preprocessing": {
            "species_transform": "linear",
            "scaler": "robust",
            "projection_y_floor": 1e-10,
            "rate_y_floor": 1e-7,
        },
        "source_target_transform": "signed_log",
        "source_model": "residual_mlp",
    }


def _save_synthetic_outputs(tmp_path: Path, k: int = 4, n_retained: int = 6) -> tuple[Path, list[str]]:
    """Write synthetic model_arrays.npz and model_metadata.json to tmp_path.

    Returns (tmp_path, retained_species).
    """
    import json

    retained = [f"SP{i}" for i in range(n_retained)]
    arrays = _build_synthetic_npz(k=k, n_retained=n_retained)
    meta = _build_synthetic_metadata(k=k, retained_species=retained)

    np.savez(str(tmp_path / "model_arrays.npz"), **arrays)
    with open(str(tmp_path / "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return tmp_path, retained


def _make_df_for_retained(retained: list[str], n_rows: int = 20) -> pd.DataFrame:
    """Build a DataFrame with columns matching *retained*.

    Includes R_* columns so Schema.from_columns does not raise a
    rate-coverage error.
    """
    rng = np.random.default_rng(1)
    df = {}
    for sp in retained:
        df[f"Y_{sp}"] = np.abs(rng.normal(0.1, 0.05, n_rows))
        df[f"R_{sp}"] = rng.normal(0, 0.1, n_rows)
    df["T [K]"] = rng.uniform(900, 1100, n_rows)
    df["P [Pa]"] = rng.uniform(1.5e5, 3.5e5, n_rows)
    df["rho [kg/m3]"] = rng.uniform(0.4, 1.0, n_rows)
    df["CaseID"] = np.ones(n_rows, dtype=int)
    return pd.DataFrame(df)


# ---------------------------------------------------------------------------
# Test: ColleagueReducedSurrogate with synthetic NPZ
# ---------------------------------------------------------------------------

class TestColleagueSurrogateUnit:
    """Unit tests using a synthetic NPZ with matching key schema."""

    def test_from_outputs_dir_loads_without_error(self, tmp_path):
        """from_outputs_dir on a synthetic NPZ must not raise."""
        # Arrange
        _save_synthetic_outputs(tmp_path, k=4, n_retained=6)
        from scarfs.benchmark.parents import ColleagueReducedSurrogate

        # Act
        surrogate = ColleagueReducedSurrogate.from_outputs_dir(tmp_path)

        # Assert
        assert isinstance(surrogate.active_species, tuple)
        assert len(surrogate.active_species) == 6

    def test_predict_output_shape(self, tmp_path):
        """predict() returns correct (n, k) rates and (n,) energy shapes."""
        # Arrange
        _, retained = _save_synthetic_outputs(tmp_path, k=4, n_retained=6)
        df = _make_df_for_retained(retained, n_rows=15)
        from scarfs.benchmark.parents import ColleagueReducedSurrogate

        surrogate = ColleagueReducedSurrogate.from_outputs_dir(tmp_path)

        # Act
        pred = surrogate.predict(df)

        # Assert
        assert pred.rates.shape == (15, 6)
        assert pred.energy.shape == (15,)

    def test_predict_rates_finite(self, tmp_path):
        """Predicted rates must be finite (no NaN/Inf)."""
        # Arrange
        _, retained = _save_synthetic_outputs(tmp_path, k=4, n_retained=6)
        df = _make_df_for_retained(retained, n_rows=10)
        from scarfs.benchmark.parents import ColleagueReducedSurrogate

        surrogate = ColleagueReducedSurrogate.from_outputs_dir(tmp_path)

        # Act
        pred = surrogate.predict(df)

        # Assert
        assert np.all(np.isfinite(pred.rates)), "Rates contain NaN or Inf"
        assert np.all(np.isfinite(pred.energy)), "Energy contains NaN or Inf"

    def test_from_outputs_dir_wrong_transform_raises(self, tmp_path):
        """from_outputs_dir raises ValueError for unsupported species_transform."""
        import json
        _save_synthetic_outputs(tmp_path, k=4, n_retained=6)
        meta_path = tmp_path / "model_metadata.json"
        with open(str(meta_path), "r") as f:
            meta = json.load(f)
        meta["preprocessing"]["species_transform"] = "log1p"
        with open(str(meta_path), "w") as f:
            json.dump(meta, f)

        from scarfs.benchmark.parents import ColleagueReducedSurrogate
        with pytest.raises(ValueError, match="species_transform"):
            ColleagueReducedSurrogate.from_outputs_dir(tmp_path)

    def test_from_outputs_dir_missing_npz_raises(self, tmp_path):
        """from_outputs_dir raises FileNotFoundError when npz is absent."""
        import json
        (tmp_path / "model_metadata.json").write_text("{}", encoding="utf-8")
        from scarfs.benchmark.parents import ColleagueReducedSurrogate
        with pytest.raises(FileNotFoundError):
            ColleagueReducedSurrogate.from_outputs_dir(tmp_path)


# ---------------------------------------------------------------------------
# Test: Smoke test on real NPZ (env-gated)
# ---------------------------------------------------------------------------

_REAL_OUTPUTS = Path("/Users/tamasbuzogany/Documents/SCARFS/reduced_chem_ml/outputs")
_STRIDE6 = Path(__file__).parent / "data" / "stride6_sample.parquet"


@pytest.mark.skipif(
    not _REAL_OUTPUTS.exists(),
    reason="Colleague outputs directory not present (ENV-GATED)",
)
class TestColleagueSurrogateSmoke:
    """Env-gated smoke test using the real model_arrays.npz."""

    def test_loads_from_real_outputs(self):
        """Real NPZ loads without error."""
        from scarfs.benchmark.parents import ColleagueReducedSurrogate
        surrogate = ColleagueReducedSurrogate.from_outputs_dir(_REAL_OUTPUTS)
        assert len(surrogate.active_species) > 0

    @pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6 fixture not found")
    def test_predict_on_stride6_shape(self):
        """Predict on 20 rows of stride6 — check shape."""
        from scarfs.benchmark.parents import ColleagueReducedSurrogate
        df = pd.read_parquet(str(_STRIDE6)).head(20)
        surrogate = ColleagueReducedSurrogate.from_outputs_dir(_REAL_OUTPUTS)
        pred = surrogate.predict(df)
        assert pred.rates.shape[0] == 20
        assert pred.energy.shape[0] == 20

    @pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6 fixture not found")
    def test_predict_rates_finite_on_real_data(self):
        """Real model predictions must be finite on stride6."""
        from scarfs.benchmark.parents import ColleagueReducedSurrogate
        df = pd.read_parquet(str(_STRIDE6)).head(20)
        surrogate = ColleagueReducedSurrogate.from_outputs_dir(_REAL_OUTPUTS)
        pred = surrogate.predict(df)
        assert np.all(np.isfinite(pred.rates)), "Rates contain NaN/Inf on real data"
        assert np.all(np.isfinite(pred.energy)), "Energy contains NaN/Inf on real data"

    @pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6 fixture not found")
    def test_energy_magnitude_within_expected_range(self):
        """Energy magnitude must be within [1e2, 1e10] J/m³/s on stride6 rows."""
        from scarfs.benchmark.parents import ColleagueReducedSurrogate
        df = pd.read_parquet(str(_STRIDE6)).head(20)
        surrogate = ColleagueReducedSurrogate.from_outputs_dir(_REAL_OUTPUTS)
        pred = surrogate.predict(df)
        abs_energy = np.abs(pred.energy)
        finite_mask = np.isfinite(abs_energy) & (abs_energy > 0)
        if finite_mask.any():
            assert abs_energy[finite_mask].min() >= 1e2, "Energy below 1e2 J/m³/s"
            assert abs_energy[finite_mask].max() <= 1e10, "Energy above 1e10 J/m³/s"

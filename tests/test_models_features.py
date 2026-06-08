"""Tests for feature/target assembly."""

from __future__ import annotations

import numpy as np

from scarfs.models.features import (
    FeatureSpec,
    build_features,
    build_rate_targets,
    fit_scalers,
    invert_rate_targets,
)


def _scalers(df, schema):
    active = schema.active_species()
    return fit_scalers(df, schema, active, active), active


def test_build_features_has_composition_plus_four_thermo(synthetic_df, synthetic_schema):
    # arrange
    scalers, active = _scalers(synthetic_df, synthetic_schema)
    # act
    X = build_features(synthetic_df, synthetic_schema, scalers)
    # assert
    assert X.shape == (len(synthetic_df), len(active) + 4)


def test_rate_targets_shape_matches_active_species(synthetic_df, synthetic_schema):
    # arrange
    scalers, active = _scalers(synthetic_df, synthetic_schema)
    # act
    Y = build_rate_targets(synthetic_df, synthetic_schema, scalers)
    # assert
    assert Y.shape == (len(synthetic_df), len(active))


def test_rate_target_roundtrip_recovers_physical_rates(synthetic_df, synthetic_schema):
    # arrange
    scalers, active = _scalers(synthetic_df, synthetic_schema)
    true_rates = synthetic_df[synthetic_schema.r_columns(active)].to_numpy(dtype=float)
    # act
    Y = build_rate_targets(synthetic_df, synthetic_schema, scalers)
    back = invert_rate_targets(Y, scalers)
    # assert
    assert np.allclose(back, true_rates, rtol=1e-6, atol=1e-12)


def test_feature_spec_counts():
    # arrange / act
    spec = FeatureSpec(input_species=("A", "B", "C"), target_species=("A", "B"))
    # assert
    assert spec.n_input_features == 3 + 4
    assert spec.n_targets == 2

"""Tests for the torch-free training datamodule (feature/target/weight assembly)."""

from __future__ import annotations

import numpy as np
import pytest

from scarfs.training.config import DataConfig
from scarfs.training.datamodule import (
    case_aware_split,
    compute_conversion,
    importance_weights,
    prepare_data,
    resolve_species,
)


def test_resolve_species_selectors(synthetic_schema):
    # assert each selector resolves correctly
    assert set(resolve_species(synthetic_schema, "active")) == {"C2H6", "C2H4", "H2"}
    assert "H2O" not in resolve_species(synthetic_schema, "dry_all")
    assert set(resolve_species(synthetic_schema, "all")) == set(synthetic_schema.species)
    assert resolve_species(synthetic_schema, ["C2H4"]) == ("C2H4",)


def test_resolve_species_rejects_unknown(synthetic_schema):
    with pytest.raises(ValueError):
        resolve_species(synthetic_schema, ["NOT_A_SPECIES"])


def test_compute_conversion_inlet_zero_and_increasing(synthetic_df, synthetic_schema):
    # act
    conv = compute_conversion(synthetic_df, synthetic_schema)
    # assert — first row of each case is the inlet (conversion 0); values stay in [0,1]
    assert np.isclose(conv[0], 0.0)
    assert 0.0 < conv.max() <= 1.0


def test_importance_weights_upweight_near_inlet(synthetic_df, synthetic_schema):
    # arrange
    cfg = DataConfig(inlet_conversion_threshold=0.05, inlet_weight=5.0)
    # act
    w = importance_weights(synthetic_df, synthetic_schema, cfg)
    # assert
    assert w.max() == 5.0 and w.min() == 1.0


def test_case_aware_split_has_no_case_leakage(synthetic_df, synthetic_schema):
    # act
    train_mask, val_mask = case_aware_split(synthetic_df, synthetic_schema, 0.5, seed=0, split_by_case=True)
    train_cases = set(synthetic_df.loc[train_mask, "CaseID"])
    val_cases = set(synthetic_df.loc[val_mask, "CaseID"])
    # assert
    assert train_cases.isdisjoint(val_cases)


def test_prepare_data_consistent_shapes(synthetic_df, synthetic_schema):
    # arrange
    cfg = DataConfig(input_species="active", target_species="active", val_fraction=0.5, split_by_case=True)
    # act
    bundle = prepare_data(cfg, synthetic_df, synthetic_schema)
    # assert
    assert bundle.X_train.shape[1] == len(bundle.spec.input_species) + 4
    assert bundle.Y_train.shape[1] == len(bundle.spec.target_species)
    assert bundle.X_train.shape[0] == bundle.Y_train.shape[0] == bundle.w_train.shape[0]
    assert bundle.species_weights.shape[0] == len(bundle.spec.target_species)

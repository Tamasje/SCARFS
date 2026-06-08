"""Database -> scaled tensors, with near-inlet importance weighting (F1). NumPy/pandas only.

This is the correctness-critical part of training (which columns become features/targets, how rows
are weighted, how the train/val split avoids leakage). It is torch-free and unit-tested locally; the
PyTorch loop in ``train.py`` simply consumes the arrays produced here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import DILUENT_SPECIES, Schema, y_column
from ..models.features import FeatureScalers, FeatureSpec, build_features, build_rate_targets, fit_scalers
from .config import DataConfig


def resolve_species(schema: Schema, selector: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve a species selector to a concrete tuple of species names.

    Selectors: ``"active"`` (molecular, diluent excluded), ``"molecular"``, ``"dry_all"`` (all
    species except the diluent), ``"all"``, or an explicit list (validated against the schema).
    """
    if isinstance(selector, str):
        if selector == "active":
            return schema.active_species()
        if selector == "molecular":
            return schema.molecular_species()
        if selector == "dry_all":
            return tuple(s for s in schema.species if s != DILUENT_SPECIES)
        if selector == "all":
            return schema.species
        raise ValueError(f"Unknown species selector {selector!r}")
    present = set(schema.species)
    missing = [s for s in selector if s not in present]
    if missing:
        raise ValueError(f"Requested species absent from database: {missing}")
    return tuple(selector)


def compute_conversion(df: pd.DataFrame, schema: Schema, key_species: str = "C2H6") -> np.ndarray:
    """Per-row conversion of *key_species* relative to its per-case inlet (min-z) value.

    Returns an array of conversions in ``[0, 1]`` (clipped). If the key species or the ``CaseID`` /
    ``z`` columns are unavailable, returns zeros and warns (never fabricates).
    """
    col = y_column(key_species)
    if col not in df.columns or "CaseID" not in schema.meta or "z" not in schema.state:
        warnings.warn("compute_conversion: missing C2H6/CaseID/z columns; returning zeros.")
        return np.zeros(len(df), dtype=float)
    case_col = schema.meta["CaseID"]
    z_col = schema.state["z"]
    inlet = df.loc[df.groupby(case_col)[z_col].idxmin(), [case_col, col]].set_index(case_col)[col]
    y_in = df[case_col].map(inlet).to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        conv = np.where(y_in > 0, (y_in - y) / y_in, 0.0)
    return np.clip(conv, 0.0, 1.0)


def importance_weights(df: pd.DataFrame, schema: Schema, cfg: DataConfig) -> np.ndarray:
    """Per-row training weights that up-weight near-inlet / low-conversion states (RC-1/F1)."""
    conv = compute_conversion(df, schema)
    w = np.ones(len(df), dtype=float)
    w[conv < cfg.inlet_conversion_threshold] = cfg.inlet_weight
    return w


def species_weight_vector(target_species: Sequence[str], species_weights: dict[str, float]) -> np.ndarray:
    """Per-target weight vector (default 1.0), e.g. emphasising C2H4/C3H6 (thesis upweights)."""
    return np.array([float(species_weights.get(s, 1.0)) for s in target_species], dtype=float)


def case_aware_split(
    df: pd.DataFrame, schema: Schema, val_fraction: float, seed: int, split_by_case: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (train, val) masks.

    When ``split_by_case`` and a ``CaseID`` column exist, whole cases go to one side (prevents the
    same PFR profile leaking across the split). Otherwise rows are split randomly.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if split_by_case and "CaseID" in schema.meta:
        case_col = schema.meta["CaseID"]
        cases = df[case_col].to_numpy()
        unique = np.unique(cases)
        rng.shuffle(unique)
        n_val = max(1, int(round(val_fraction * len(unique)))) if len(unique) > 1 else 0
        val_cases = set(unique[:n_val].tolist())
        val_mask = np.array([c in val_cases for c in cases])
    else:
        val_mask = rng.random(n) < val_fraction
    return ~val_mask, val_mask


@dataclass
class DataBundle:
    """Prepared training arrays + the fitted scalers and resolved spec."""

    X_train: np.ndarray
    Y_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    w_val: np.ndarray
    species_weights: np.ndarray
    scalers: FeatureScalers
    spec: FeatureSpec
    schema: Schema


def prepare_data(cfg: DataConfig, df: pd.DataFrame, schema: Schema) -> DataBundle:
    """Build a :class:`DataBundle` from a loaded DataFrame and its schema.

    Scalers are fit on the **train** split only (no leakage). Importance weights combine the
    per-row near-inlet weight (F1) so the optimiser sees low-conversion states more often.
    """
    input_species = resolve_species(schema, cfg.input_species)
    target_species = resolve_species(schema, cfg.target_species)

    train_mask, val_mask = case_aware_split(df, schema, cfg.val_fraction, cfg.seed, cfg.split_by_case)
    df_train, df_val = df.loc[train_mask], df.loc[val_mask]

    scalers = fit_scalers(
        df_train, schema, input_species, target_species,
        composition_kwargs={"feature_range": cfg.composition_feature_range},
    )

    bundle = DataBundle(
        X_train=build_features(df_train, schema, scalers),
        Y_train=build_rate_targets(df_train, schema, scalers),
        w_train=importance_weights(df_train, schema, cfg),
        X_val=build_features(df_val, schema, scalers) if len(df_val) else np.empty((0, len(input_species) + 4)),
        Y_val=build_rate_targets(df_val, schema, scalers) if len(df_val) else np.empty((0, len(target_species))),
        w_val=importance_weights(df_val, schema, cfg) if len(df_val) else np.empty((0,)),
        species_weights=species_weight_vector(target_species, cfg.species_weights),
        scalers=scalers,
        spec=FeatureSpec(input_species=input_species, target_species=target_species),
        schema=schema,
    )
    return bundle

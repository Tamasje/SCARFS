"""Feature and target assembly (NumPy only — no PyTorch).

Turns database rows into the scaled input/target matrices both surrogates consume. Keeping this
torch-free makes the correctness-critical part (which columns become features, how they are scaled,
how rates/energy become targets) unit-testable in the minimal local environment; the PyTorch
networks (``scarfs.models.nets``) only consume the resulting arrays.

Input feature vector = ``[scaled composition (selected species) | scaled thermo block]`` where the
thermo block is the Arrhenius-motivated ``[T, p, 1/T, ln T]`` (thesis Ch. 5). Targets are the net
production rates of the active species (a source term, kg m-3 s-1) — never a yield or increment
(RC-5). The energy source is *derived* from the rates downstream (see ``physics.py``), not a free
head, which removes the energy/rate inconsistency diagnosed in RC-3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import Schema
from .common import ArcsinhScaler, CompositionScaler, StandardScaler, thermo_features


@dataclass
class FeatureScalers:
    """Fitted scalers for one model's features and targets."""

    composition: CompositionScaler
    thermo: StandardScaler
    rate: ArcsinhScaler
    input_species: tuple[str, ...]
    target_species: tuple[str, ...]


def _composition_matrix(df: pd.DataFrame, schema: Schema, species: Sequence[str]) -> np.ndarray:
    """Extract the ``Y_<species>`` block as a float array in *species* order."""
    return df[schema.y_columns(species)].to_numpy(dtype=float)


def _rate_matrix(df: pd.DataFrame, schema: Schema, species: Sequence[str]) -> np.ndarray:
    """Extract the ``R_<species>`` block as a float array in *species* order."""
    return df[schema.r_columns(species)].to_numpy(dtype=float)


def _thermo_block(df: pd.DataFrame, schema: Schema) -> np.ndarray:
    """Build the raw ``[T, p, 1/T, ln T]`` block from the database state columns."""
    (t_col,) = schema.require_state("T")
    (p_col,) = schema.require_state("P")
    return thermo_features(df[t_col].to_numpy(dtype=float), df[p_col].to_numpy(dtype=float))


def fit_scalers(
    df: pd.DataFrame,
    schema: Schema,
    input_species: Sequence[str],
    target_species: Sequence[str],
    *,
    composition_kwargs: dict | None = None,
) -> FeatureScalers:
    """Fit composition / thermo / rate scalers on a training DataFrame.

    Parameters
    ----------
    df
        Training rows.
    schema
        Column contract for *df*.
    input_species
        Species whose mass fractions feed the model input (e.g. active/molecular set).
    target_species
        Species whose rates are predicted.
    composition_kwargs
        Extra kwargs for :class:`CompositionScaler` (e.g. ``feature_range``).
    """
    comp = CompositionScaler(**(composition_kwargs or {}))
    comp.fit(_composition_matrix(df, schema, input_species))
    thermo = StandardScaler().fit(_thermo_block(df, schema))
    rate = ArcsinhScaler().fit(_rate_matrix(df, schema, target_species))
    return FeatureScalers(
        composition=comp,
        thermo=thermo,
        rate=rate,
        input_species=tuple(input_species),
        target_species=tuple(target_species),
    )


def build_features(df: pd.DataFrame, schema: Schema, scalers: FeatureScalers) -> np.ndarray:
    """Assemble the scaled input feature matrix ``[comp_scaled | thermo_scaled]``."""
    comp = scalers.composition.transform(_composition_matrix(df, schema, scalers.input_species))
    thermo = scalers.thermo.transform(_thermo_block(df, schema))
    return np.concatenate([comp, thermo], axis=1)


def build_rate_targets(df: pd.DataFrame, schema: Schema, scalers: FeatureScalers) -> np.ndarray:
    """Assemble the scaled rate-target matrix for the active species."""
    return scalers.rate.transform(_rate_matrix(df, schema, scalers.target_species))


def invert_rate_targets(scaled: np.ndarray, scalers: FeatureScalers) -> np.ndarray:
    """Invert scaled model outputs back to physical rates [kg m-3 s-1]."""
    return scalers.rate.inverse_transform(scaled)


@dataclass
class FeatureSpec:
    """Resolved input/target species selection for a model (for logging/reproducibility)."""

    input_species: tuple[str, ...]
    target_species: tuple[str, ...]
    n_input_features: int = field(init=False)
    n_targets: int = field(init=False)

    def __post_init__(self) -> None:
        # composition block + 4 thermo features (T, p, 1/T, ln T)
        object.__setattr__(self, "n_input_features", len(self.input_species) + 4)
        object.__setattr__(self, "n_targets", len(self.target_species))

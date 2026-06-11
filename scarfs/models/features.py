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

For the merged model the composition contract is: linear (log=False) + standard z-score mode
(no log, no minmax); mass rates are assembled from ``dYdt_*`` columns when available, falling back
to ``R_*`` (legacy).  The absorption target is read from the ``Reaction heat absorption [J/s/m³]``
column and clipped at zero (E-c: positivity assumption).
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
    """Fitted scalers for one model's features and targets.

    ``rate_source`` records which database family the rate scaler was fitted on, so
    :func:`build_rate_targets` always transforms the *same* physical quantity:
    ``"r_mass"`` — legacy ``R_<species>`` columns [kg m-3 s-1]; ``"dydt_rho"`` —
    mass rates assembled as ``ρ·dY/dt`` from the parquet ``dYdt_*`` columns.
    """

    composition: CompositionScaler
    thermo: StandardScaler
    rate: ArcsinhScaler
    input_species: tuple[str, ...]
    target_species: tuple[str, ...]
    rate_source: str = "r_mass"


@dataclass
class AbsorptionTarget:
    """Result of :func:`build_absorption_target`.

    Attributes
    ----------
    values
        ``(n,)`` absorption values [J/m³/s], clipped at zero.
    n_clipped
        Number of rows that were negative before clipping.
    """

    values: np.ndarray
    n_clipped: int


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
    prefer_dydt: bool = False,
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
    prefer_dydt
        If ``True`` and the schema carries ``dYdt_*`` columns, fit the rate scaler on the
        mass rates ``ρ·dY/dt`` (the parquet path; legacy ``R_`` columns on the parquet are
        raw kmol m-3 s-1 and must NOT feed the mass-rate contract).
    """
    comp = CompositionScaler(**(composition_kwargs or {}))
    comp.fit(_composition_matrix(df, schema, input_species))
    thermo = StandardScaler().fit(_thermo_block(df, schema))
    use_dydt = prefer_dydt and hasattr(schema, "has_dydt") and schema.has_dydt()
    rate_matrix = build_mass_rate_matrix(df, schema, target_species, prefer_dydt=use_dydt)
    rate = ArcsinhScaler().fit(rate_matrix)
    return FeatureScalers(
        composition=comp,
        thermo=thermo,
        rate=rate,
        input_species=tuple(input_species),
        target_species=tuple(target_species),
        rate_source="dydt_rho" if use_dydt else "r_mass",
    )


def build_features(df: pd.DataFrame, schema: Schema, scalers: FeatureScalers) -> np.ndarray:
    """Assemble the scaled input feature matrix ``[comp_scaled | thermo_scaled]``."""
    comp = scalers.composition.transform(_composition_matrix(df, schema, scalers.input_species))
    thermo = scalers.thermo.transform(_thermo_block(df, schema))
    return np.concatenate([comp, thermo], axis=1)


def build_rate_targets(df: pd.DataFrame, schema: Schema, scalers: FeatureScalers) -> np.ndarray:
    """Assemble the scaled rate-target matrix for the active species.

    Routes through the same rate family the scaler was fitted on (``scalers.rate_source``),
    so fit and transform always see the same physical quantity.
    """
    use_dydt = getattr(scalers, "rate_source", "r_mass") == "dydt_rho"
    matrix = build_mass_rate_matrix(df, schema, scalers.target_species, prefer_dydt=use_dydt)
    return scalers.rate.transform(matrix)


def invert_rate_targets(scaled: np.ndarray, scalers: FeatureScalers) -> np.ndarray:
    """Invert scaled model outputs back to physical rates [kg m-3 s-1]."""
    return scalers.rate.inverse_transform(scaled)


def build_mass_rate_matrix(
    df: pd.DataFrame,
    schema: Schema,
    species: Sequence[str],
    *,
    prefer_dydt: bool = True,
) -> np.ndarray:
    """Return mass production rates ``ρ·dY/dt`` [kg/m³/s].

    Prefers ``dYdt_<species>`` columns (multiplied by ``rho``) when
    *prefer_dydt* is ``True`` and the schema reports their availability via
    ``schema.has_dydt()``, falling back to ``R_<species>`` (legacy) otherwise.

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract.  Must expose ``has_dydt()``, ``dydt_columns(species)``,
        and ``r_columns(species)`` (B1a interface; a minimal stub suffices for
        independent testing).
    species
        Species whose rates are assembled.
    prefer_dydt
        If ``True`` (default), use dYdt when available.
    """
    use_dydt = prefer_dydt and hasattr(schema, "has_dydt") and schema.has_dydt()
    if use_dydt:
        dydt_cols = schema.dydt_columns(list(species))
        dydt = df[dydt_cols].to_numpy(dtype=float)
        rho_col = schema.require_state("rho")[0]
        rho = df[rho_col].to_numpy(dtype=float).reshape(-1, 1)
        return rho * dydt
    return _rate_matrix(df, schema, species)


def build_absorption_target(
    df: pd.DataFrame,
    schema: Schema,
) -> AbsorptionTarget:
    """Read the absorption target column and clip at zero.

    The column is accessed via ``schema.energy_target_column()`` (B1a
    interface). Negative values are truncation noise (E-c); they are clipped
    to zero and counted.

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract; must expose ``energy_target_column() -> str``.

    Returns
    -------
    :class:`AbsorptionTarget` with clipped values and the clip count.
    """
    col = schema.energy_target_column()
    raw = df[col].to_numpy(dtype=float)
    n_clipped = int((raw < 0).sum())
    values = np.clip(raw, 0.0, None)
    return AbsorptionTarget(values=values, n_clipped=n_clipped)


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

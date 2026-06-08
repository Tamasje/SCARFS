"""Trivial surrogate baselines for the SCARFS benchmark harness.

Three baselines implement the :class:`~scarfs.models.common.Surrogate` protocol
and provide a sanity floor against which a trained model must compete:

- :class:`FrozenComposition` — always predicts zero rates (the "failed model"
  floor: if a candidate model cannot beat this, it adds no predictive value).
- :class:`MeanRate` — predicts the per-species mean rate computed over a
  training set (slightly stronger than zero; captures the average tendency).
- :class:`NearestNeighborRates` — retrieves the training row nearest in a
  scaled (log Y, T, P) feature space and returns its rates verbatim (a
  non-parametric upper bound on what a simple look-up achieves without any
  extrapolation).

All three accept the full database row (``Y_*`` + state columns) as their
``predict`` input, matching the surrogate contract.  None require PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from scarfs.models.common import SurrogatePrediction
from scarfs.schema import Schema


# ---------------------------------------------------------------------------
# Helper: build (logY, T, P) feature matrix
# ---------------------------------------------------------------------------

def _build_features(
    df: pd.DataFrame,
    active_species: tuple[str, ...],
    schema: Schema,
    log_floor: float = 1e-30,
) -> np.ndarray:
    """Build the scaled feature matrix used for nearest-neighbour lookup.

    Features are ``[log10(Y_i) for i in active_species, T_scaled, P_scaled]``
    where T and P are standardised by their training statistics.  The log
    transform compresses the many-decades span of mass fractions so that all
    species contribute comparably to the distance calculation.

    Parameters
    ----------
    df
        DataFrame containing ``Y_*`` mass-fraction columns and state columns.
    active_species
        Species whose log-mass-fractions form the composition features.
    schema
        Resolved :class:`~scarfs.schema.Schema` for the DataFrame.
    log_floor
        Lower clip applied before ``log10`` to avoid -inf.

    Returns
    -------
    numpy.ndarray
        ``(n, n_active + 2)`` float array.
    """
    y_cols = schema.y_columns(active_species)
    Y = df[y_cols].to_numpy(dtype=float)
    logY = np.log10(np.clip(Y, log_floor, None))

    T_col = schema.state["T"]
    P_col = schema.state["P"]
    T = df[T_col].to_numpy(dtype=float).reshape(-1, 1)
    P = df[P_col].to_numpy(dtype=float).reshape(-1, 1)

    return np.hstack([logY, T, P])


# ---------------------------------------------------------------------------
# Baseline 1: FrozenComposition
# ---------------------------------------------------------------------------

@dataclass
class FrozenComposition:
    """Surrogate that always predicts zero net production rates.

    This is the "failed model" sanity floor: a surrogate that cannot beat
    ``FrozenComposition`` is no better than assuming nothing reacts.

    Parameters
    ----------
    active_species
        Species for which rates are predicted.  Determines the column order of
        :attr:`~scarfs.models.common.SurrogatePrediction.rates`.
    """

    active_species: tuple[str, ...]

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Return zero rates and zero energy source for every row of *df*.

        Parameters
        ----------
        df
            Input DataFrame (columns not used — only row count is read).

        Returns
        -------
        SurrogatePrediction
            ``rates`` shape ``(n, k)`` of zeros; ``energy`` shape ``(n,)`` of
            zeros.
        """
        n = len(df)
        k = len(self.active_species)
        return SurrogatePrediction(
            species=self.active_species,
            rates=np.zeros((n, k), dtype=float),
            energy=np.zeros(n, dtype=float),
        )


# ---------------------------------------------------------------------------
# Baseline 2: MeanRate
# ---------------------------------------------------------------------------

@dataclass
class MeanRate:
    """Surrogate that predicts the per-species mean rate from the training set.

    Slightly stronger than :class:`FrozenComposition` because it captures the
    average tendency (net production vs net consumption) of each species.

    After calling :meth:`fit`, calling :meth:`predict` on any input returns
    the same constant row-vector tiled to the batch size.

    Parameters
    ----------
    active_species
        Species for which rates are predicted.
    """

    active_species: tuple[str, ...]
    _mean_rates: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _mean_energy: float = field(default=0.0, repr=False, compare=False)

    def fit(self, train_df: pd.DataFrame, schema: Schema) -> "MeanRate":
        """Compute per-species mean rate and mean energy source from *train_df*.

        Parameters
        ----------
        train_df
            Training DataFrame with ``R_*`` and ``S Energy`` columns.
        schema
            Schema describing *train_df*'s columns.

        Returns
        -------
        MeanRate
            ``self`` (for chaining).
        """
        r_cols = schema.r_columns(self.active_species)
        self._mean_rates = train_df[r_cols].to_numpy(dtype=float).mean(axis=0)

        if "S_energy" in schema.state:
            self._mean_energy = float(
                train_df[schema.state["S_energy"]].mean()
            )
        else:
            self._mean_energy = 0.0
        return self

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Return the training mean rates for every row of *df*.

        Parameters
        ----------
        df
            Input DataFrame (only row count is read).

        Returns
        -------
        SurrogatePrediction
            ``rates`` shape ``(n, k)`` with each row equal to the training
            mean; ``energy`` shape ``(n,)`` of the training mean energy.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._mean_rates is None:
            raise RuntimeError("MeanRate.predict called before fit().")
        n = len(df)
        rates = np.tile(self._mean_rates, (n, 1))
        energy = np.full(n, self._mean_energy, dtype=float)
        return SurrogatePrediction(
            species=self.active_species,
            rates=rates,
            energy=energy,
        )


# ---------------------------------------------------------------------------
# Baseline 3: NearestNeighborRates
# ---------------------------------------------------------------------------

@dataclass
class NearestNeighborRates:
    """Surrogate that returns training rates from the nearest-neighbour row.

    The look-up space is ``(log10(Y_i), T, P)`` for *active_species*, with T
    and P standardised by their training mean/std.  This provides a
    non-parametric upper bound on what a simple look-up table can achieve
    without any interpolation or extrapolation.

    If ``scipy`` is available, a :class:`scipy.spatial.cKDTree` is used for
    efficient O(log n) queries.  Otherwise, a brute-force NumPy fallback is
    used (suitable for small databases).

    Parameters
    ----------
    active_species
        Species for which rates are predicted.
    log_floor
        Clip floor for log10 of mass fractions.
    """

    active_species: tuple[str, ...]
    log_floor: float = 1e-30

    _train_features: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _train_rates: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _train_energy: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _feat_mean: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _feat_std: Optional[np.ndarray] = field(default=None, repr=False, compare=False)
    _tree: object = field(default=None, repr=False, compare=False)

    def fit(self, train_df: pd.DataFrame, schema: Schema) -> "NearestNeighborRates":
        """Index the training set for nearest-neighbour queries.

        Parameters
        ----------
        train_df
            Training DataFrame.
        schema
            Schema describing *train_df*'s columns.

        Returns
        -------
        NearestNeighborRates
            ``self`` (for chaining).
        """
        raw = _build_features(train_df, self.active_species, schema, self.log_floor)
        self._feat_mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        self._feat_std = np.where(std > 0, std, 1.0)
        self._train_features = (raw - self._feat_mean) / self._feat_std

        r_cols = schema.r_columns(self.active_species)
        self._train_rates = train_df[r_cols].to_numpy(dtype=float)

        if "S_energy" in schema.state:
            self._train_energy = train_df[schema.state["S_energy"]].to_numpy(dtype=float)
        else:
            self._train_energy = np.zeros(len(train_df), dtype=float)

        # Try to build a cKDTree for fast queries
        try:
            from scipy.spatial import cKDTree  # type: ignore[import]
            self._tree = cKDTree(self._train_features)
        except ImportError:
            self._tree = None

        return self

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Return training rates from the nearest row in feature space.

        Parameters
        ----------
        df
            Input DataFrame with ``Y_*`` and state columns.

        Returns
        -------
        SurrogatePrediction
            Rates and energy from the nearest training row for each input row.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if self._train_features is None:
            raise RuntimeError("NearestNeighborRates.predict called before fit().")

        schema = Schema.from_columns(list(df.columns))
        raw = _build_features(df, self.active_species, schema, self.log_floor)
        query = (raw - self._feat_mean) / self._feat_std

        if self._tree is not None:
            _, idx = self._tree.query(query, k=1)  # type: ignore[union-attr]
        else:
            # Brute-force: compute pairwise squared distances
            # query: (n, d), train: (m, d)
            diff = query[:, np.newaxis, :] - self._train_features[np.newaxis, :, :]
            dist2 = (diff ** 2).sum(axis=2)  # (n, m)
            idx = dist2.argmin(axis=1)

        rates = self._train_rates[idx]
        energy = self._train_energy[idx]

        return SurrogatePrediction(
            species=self.active_species,
            rates=rates,
            energy=energy,
        )

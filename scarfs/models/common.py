"""Framework-agnostic scalers and the surrogate I/O contract.

This module deliberately imports **only NumPy** so that the benchmark, coupling and plotting code
can depend on the contract without pulling in PyTorch. The concrete networks live in
``scarfs.models.nets`` (PyTorch) and are imported lazily by the training pipeline.

The contract is small and explicit (see ``DIAGNOSIS.md`` / ``BENCHMARK_PLAN.md``):

- A surrogate consumes the **local thermochemical state** (composition + T + P) and returns
  **instantaneous net production rates** ``R_i`` [kg m-3 s-1] for its active species, plus the
  volumetric energy source ``S_E`` [J m-3 s-1]. No residence time, no Δt — the CFD solver integrates.
- The energy source should be *consistent* with the rates, ``S_E = -Σ h_i · ω̇_i`` (enforced in the
  training losses; checked in the a-posteriori harness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


# ----------------------------------------------------------------------------------------------
# Scalers (sklearn-style fit/transform/inverse_transform, NumPy only)
# ----------------------------------------------------------------------------------------------
class CompositionScaler:
    """Scale species mass fractions for use as NN inputs.

    Composition spans many orders of magnitude (trace radicals to bulk diluent), so a log transform
    is applied before an affine map to ``feature_range``. Tiny/negative numerical values (the
    machine-epsilon floor seen in the database) are clipped to ``floor`` before the log.

    Parameters
    ----------
    log
        If ``True`` (default), transform ``log10(clip(Y, floor))`` before the affine map.
    floor
        Lower clip applied to mass fractions prior to the log.
    feature_range
        Target range of the affine map (default ``(-1.0, 1.0)``).  Ignored when
        ``mode="standard"``.
    mode
        Scaling mode applied after the optional log transform:

        - ``"minmax"`` (default): per-species affine map to *feature_range*.
        - ``"standard"``: per-species z-score (zero mean, unit variance).  Use
          with ``log=False`` for the merged model's standardised linear
          encoder contract (plan §3 / E-a).  ``mean_`` and ``scale_`` are
          accessible for the UDF exporter.
    sigma_floor
        Standard mode only. Species whose per-column std is ``<= sigma_floor`` get
        ``scale_ = 1.0`` — i.e. they are de-activated as encoder inputs instead of being
        standardised to unit-variance numerical noise. On the stride5 database, 62 of 212
        dry species have std < 1e-10 (3 exactly constant incl. inert N2, 59 trace radicals
        at 1e-17..1e-11): standardising them (a) feeds ~28% pure-noise columns to the PCA
        basis and (b) amplifies dYdt solver noise by up to 1e15 in the latent-source
        target ż = E·(Ẏ⊘σ), which made ω_Z regression noise-bound (overnight diagnosis
        2026-06-12; the clean formulation's kNN ceiling is R²≈0.82 vs ≈−0.10 polluted).
        Default 0.0 preserves the original ``std > 0`` behaviour exactly.

    Attributes
    ----------
    data_min_, data_max_
        Set by ``fit()`` in minmax mode; ``None`` in standard mode.
    mean_, scale_
        Set by ``fit()`` in standard mode; ``None`` in minmax mode.
    """

    def __init__(
        self,
        log: bool = True,
        floor: float = 1e-30,
        feature_range: tuple[float, float] = (-1.0, 1.0),
        mode: str = "minmax",
        sigma_floor: float = 0.0,
    ):
        if mode not in ("minmax", "standard"):
            raise ValueError(f"CompositionScaler mode must be 'minmax' or 'standard', got {mode!r}")
        self.log = log
        self.floor = float(floor)
        self.feature_range = (float(feature_range[0]), float(feature_range[1]))
        self.mode = mode
        self.sigma_floor = float(sigma_floor)
        self.data_min_: np.ndarray | None = None
        self.data_max_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _pre(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self.log:
            return np.log10(np.clip(y, self.floor, None))
        return y

    def fit(self, y: np.ndarray) -> "CompositionScaler":
        """Fit per-species statistics on the (log-)transformed composition."""
        t = self._pre(y)
        if self.mode == "standard":
            self.mean_ = t.mean(axis=0)
            std = t.std(axis=0)
            self.scale_ = np.where(std > self.sigma_floor, std, 1.0)
        else:  # minmax
            self.data_min_ = t.min(axis=0)
            self.data_max_ = t.max(axis=0)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """Map composition to the fitted scale."""
        t = self._pre(y)
        if self.mode == "standard":
            if self.mean_ is None:
                raise RuntimeError("CompositionScaler.transform called before fit().")
            return (t - self.mean_) / self.scale_
        # minmax
        if self.data_min_ is None:
            raise RuntimeError("CompositionScaler.transform called before fit().")
        span = np.where(self.data_max_ > self.data_min_, self.data_max_ - self.data_min_, 1.0)
        unit = (t - self.data_min_) / span
        lo, hi = self.feature_range
        return unit * (hi - lo) + lo

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        return self.fit(y).transform(y)

    def inverse_transform(self, s: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform` back to mass fractions (or log-transformed values)."""
        s = np.asarray(s, dtype=float)
        if self.mode == "standard":
            if self.mean_ is None:
                raise RuntimeError("CompositionScaler.inverse_transform called before fit().")
            t = s * self.scale_ + self.mean_
        else:
            if self.data_min_ is None:
                raise RuntimeError("CompositionScaler.inverse_transform called before fit().")
            lo, hi = self.feature_range
            span = np.where(self.data_max_ > self.data_min_, self.data_max_ - self.data_min_, 1.0)
            t = (s - lo) / (hi - lo) * span + self.data_min_
        return np.power(10.0, t) if self.log else t


class StandardScaler:
    """Zero-mean unit-variance scaler (NumPy)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        x = np.asarray(x, dtype=float)
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler.transform called before fit().")
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("StandardScaler.inverse_transform called before fit().")
        return np.asarray(z, dtype=float) * self.scale_ + self.mean_


class ArcsinhScaler:
    """Signed-log scaler for reaction rates / energy sources.

    Rates span many orders of magnitude AND change sign (production vs consumption). ``arcsinh`` is
    smooth through zero and handles both signs, avoiding the discontinuous magnitude+sign-head
    design whose fragility contributed to the reduced surrogate's instability (RC-3). The transform
    is ``z = standardize(arcsinh(x / scale))`` with a per-column ``scale`` (median absolute value).

    Parameters
    ----------
    min_scale
        Floor on the per-column scale, to avoid division blow-up on near-zero columns.
    """

    def __init__(self, min_scale: float = 1e-12):
        self.min_scale = float(min_scale)
        self.scale_: np.ndarray | None = None
        self._std = StandardScaler()

    def fit(self, x: np.ndarray) -> "ArcsinhScaler":
        x = np.asarray(x, dtype=float)
        med = np.median(np.abs(x), axis=0)
        self.scale_ = np.where(med > self.min_scale, med, self.min_scale)
        self._std.fit(np.arcsinh(x / self.scale_))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("ArcsinhScaler.transform called before fit().")
        return self._std.transform(np.arcsinh(np.asarray(x, dtype=float) / self.scale_))

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.scale_ is None:
            raise RuntimeError("ArcsinhScaler.inverse_transform called before fit().")
        return np.sinh(self._std.inverse_transform(z)) * self.scale_


def thermo_features(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Arrhenius-motivated thermodynamic feature block ``[T, p, 1/T, ln T]`` (thesis Ch. 5).

    Returns an ``(n, 4)`` array; the caller standardises it (e.g. with :class:`StandardScaler`).
    """
    T = np.asarray(T, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float).reshape(-1)
    return np.column_stack([T, P, 1.0 / T, np.log(T)])


# ----------------------------------------------------------------------------------------------
# Surrogate contract
# ----------------------------------------------------------------------------------------------
@dataclass
class SurrogatePrediction:
    """Output of a surrogate over a batch of states.

    Attributes
    ----------
    species
        Active species in the column order of ``rates``.
    rates
        ``(n, n_active)`` net production rates [kg m-3 s-1].
    energy
        ``(n,)`` volumetric energy source [J m-3 s-1].
    """

    species: tuple[str, ...]
    rates: np.ndarray
    energy: np.ndarray


@runtime_checkable
class Surrogate(Protocol):
    """The interface every trained surrogate (and every benchmark baseline) exposes.

    Implementations receive a pandas ``DataFrame`` of database rows (full ``Y_*`` + state columns,
    so they can build their own feature vector) and return a :class:`SurrogatePrediction`.
    """

    active_species: tuple[str, ...]

    def predict(self, df) -> SurrogatePrediction:  # noqa: D401,ANN001 - protocol
        """Predict rates and energy source for every row of *df*."""
        ...

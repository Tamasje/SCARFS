"""Pure-NumPy metric functions for the a-priori benchmark harness.

All functions operate on plain NumPy arrays (``y_true``, ``y_pred``) and are
intentionally free of pandas / torch / sklearn so that the benchmark can run
even when those libraries are not installed.

Metric definitions match the ChemZIP paper (Tables 2-3 in the thesis):

- **R²** — coefficient of determination (``1 - SS_res / SS_tot``).
- **NRMSE** — RMSE normalised by the *standard deviation* of the true values.
  Normalising by std (rather than range) makes the metric independent of the
  absolute scale of a species, which varies by orders of magnitude.  A value
  of 1.0 means the model is no better than predicting the mean.
- **NMdAE** — median absolute error normalised by the *median absolute value*
  of the true signal.  This is the robust analogue of the relative error; it
  is insensitive to outliers and avoids division by near-zero values (a common
  failure mode for trace species whose rates are almost always zero outside a
  narrow temperature window).
- **relative_error** / **median_relative_error** — element-wise signed and
  unsigned relative errors with a safe denominator (``max(|true|, epsilon)``
  to avoid NaN for species that are identically zero in the test set).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar metrics (operate on 1-D arrays)
# ---------------------------------------------------------------------------

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R².

    Computes ``1 - SS_res / SS_tot`` where ``SS_tot`` is the variance of
    *y_true*.  Returns ``0.0`` when ``SS_tot == 0`` (constant true signal)
    because the model cannot improve over the mean and any finite prediction
    error is equally bad.

    Parameters
    ----------
    y_true, y_pred
        1-D (or ravelled) arrays of equal length.

    Returns
    -------
    float
        R² in the range (-∞, 1].
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalised RMSE (normalised by the standard deviation of *y_true*).

    **Normalisation choice:** using the standard deviation (instead of the
    range) makes the metric robust to outliers and independent of the absolute
    scale of the species.  A value of 1.0 means the model matches the
    "predict the mean" baseline exactly; values >1 are worse than that baseline.

    Parameters
    ----------
    y_true, y_pred
        1-D (or ravelled) arrays of equal length.

    Returns
    -------
    float
        NRMSE ≥ 0.  Returns 0.0 when ``std(y_true) == 0`` (constant signal
        and no prediction error) or the raw RMSE when ``std(y_true) == 0``
        but there *is* a non-zero error.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    std = float(np.std(y_true))
    if std == 0.0:
        return rmse  # return raw RMSE when signal is constant
    return rmse / std


def nmdae(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-30) -> float:
    """Normalised Median Absolute Error (NMdAE).

    Computes ``median(|y_true - y_pred|) / max(median(|y_true|), eps)``.

    **Design rationale:** The median (rather than the mean) makes this metric
    resistant to outliers that are common in species rates during ignition-like
    events.  Normalising by the *median absolute value* of the true signal
    (rather than by std or range) gives a dimensionless relative scale that
    matches the ChemZIP paper's definition (see thesis Table 2).

    Parameters
    ----------
    y_true, y_pred
        1-D (or ravelled) arrays of equal length.
    eps
        Floor on the denominator to avoid division by zero.

    Returns
    -------
    float
        NMdAE ≥ 0.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = float(np.median(np.abs(y_true - y_pred)))
    scale = max(float(np.median(np.abs(y_true))), eps)
    return mae / scale


def relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    """Element-wise signed relative error ``(y_pred - y_true) / max(|y_true|, eps)``.

    The safe denominator ``max(|y_true|, eps)`` avoids NaN for species that
    are identically zero in the test set (e.g. trace products at the reactor
    inlet where no cracking has occurred yet).

    Parameters
    ----------
    y_true, y_pred
        Arrays of equal shape.
    eps
        Denominator floor.

    Returns
    -------
    numpy.ndarray
        Array of signed relative errors, same shape as inputs.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) > eps, np.abs(y_true), eps)
    return (y_pred - y_true) / denom


def median_relative_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-30,
) -> float:
    """Median of the *absolute* relative errors.

    Uses :func:`relative_error` internally and then takes the median of
    absolute values, giving a single robust scalar per column.

    Parameters
    ----------
    y_true, y_pred
        1-D (or ravelled) arrays of equal length.
    eps
        Denominator floor, forwarded to :func:`relative_error`.

    Returns
    -------
    float
        Median absolute relative error ≥ 0.
    """
    return float(np.median(np.abs(relative_error(y_true, y_pred, eps=eps))))


# ---------------------------------------------------------------------------
# Per-species aggregate
# ---------------------------------------------------------------------------

def per_species_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species: Sequence[str],
) -> pd.DataFrame:
    """Compute R², NRMSE, NMdAE, and median relative error for each species column.

    Parameters
    ----------
    y_true
        ``(n, k)`` array of true values (one column per species).
    y_pred
        ``(n, k)`` array of predicted values (same column order).
    species
        Sequence of *k* species names (used as the DataFrame index).

    Returns
    -------
    pandas.DataFrame
        Index: species names.
        Columns: ``["R2", "NRMSE", "NMdAE", "MedianRelErr"]``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    n_species = len(species)
    if y_true.shape[1] != n_species or y_pred.shape[1] != n_species:
        raise ValueError(
            f"per_species_metrics: species list length ({n_species}) does not match "
            f"array columns (y_true: {y_true.shape[1]}, y_pred: {y_pred.shape[1]})."
        )

    rows = []
    for i, sp in enumerate(species):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append(
            {
                "species": sp,
                "R2": r2_score(yt, yp),
                "NRMSE": nrmse(yt, yp),
                "NMdAE": nmdae(yt, yp),
                "MedianRelErr": median_relative_error(yt, yp),
            }
        )

    df = pd.DataFrame(rows).set_index("species")
    return df

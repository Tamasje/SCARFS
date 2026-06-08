"""Publication-quality benchmark figures for the CRACKSIM/PFR surrogate evaluation.

Each function returns a :class:`matplotlib.figure.Figure` and, when a *path* is
supplied, saves a PNG at 400 DPI via the shared :func:`_save` helper.

Temperature convention (steam-cracking):
    °C is always the primary axis; Kelvin is shown on the secondary axis via
    :func:`~scarfs.plotting.plot_defaults.dual_temperature_xaxis` /
    :func:`~scarfs.plotting.plot_defaults.dual_temperature_axis`.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot_defaults import apply_defaults, dual_temperature_axis, dual_temperature_xaxis, palette

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _save(fig: Figure, path: str | os.PathLike | None) -> None:
    """Save *fig* to *path* as a 400-DPI PNG when *path* is not ``None``."""
    if path is not None:
        fig.savefig(str(path), dpi=400)


# ---------------------------------------------------------------------------
# 1. Parity plot
# ---------------------------------------------------------------------------


def parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_labels: Sequence[str],
    *,
    log_scale: bool = False,
    title: str = "Parity plot",
    path: str | os.PathLike | None = None,
) -> Figure:
    """Per-species parity plot with 1:1 line and ±10 % tolerance band.

    Parameters
    ----------
    y_true:
        Array of shape ``(N, S)`` — reference values for *S* species.
    y_pred:
        Array of shape ``(N, S)`` — predicted values; same layout as *y_true*.
    species_labels:
        Length-*S* sequence of species name strings used for the colour legend.
    log_scale:
        When ``True`` use symlog scaling (useful for net-production rates that
        span many orders of magnitude, including negative values).
    title:
        Figure title.
    path:
        Optional file path; when given the figure is saved as a 400-DPI PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()
    fig, ax = plt.subplots(figsize=(7, 7))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Plot per-species scatter
    for i, label in enumerate(species_labels):
        col = colors[i % len(colors)]
        ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            s=8,
            alpha=0.5,
            color=col,
            label=label,
            rasterized=True,
        )

    # Determine axis limits for the reference lines
    all_vals = np.concatenate([y_true.ravel(), y_pred.ravel()])
    finite = all_vals[np.isfinite(all_vals)]
    if finite.size:
        lo, hi = finite.min(), finite.max()
        margin = (hi - lo) * 0.05
        lims = (lo - margin, hi + margin)
    else:
        lims = (-1.0, 1.0)

    ref = np.linspace(lims[0], lims[1], 300)

    # 1:1 line
    ax.plot(ref, ref, "k-", linewidth=1.5, label="1:1")
    # ±10 % band
    ax.plot(ref, ref * 1.10, "k--", linewidth=1.0, alpha=0.6, label="±10 %")
    ax.plot(ref, ref * 0.90, "k--", linewidth=1.0, alpha=0.6)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Reference")
    ax.set_ylabel("Predicted")
    ax.set_title(title)

    if log_scale:
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")

    ax.legend(fontsize=11, markerscale=2)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# 2. Error vs temperature
# ---------------------------------------------------------------------------


def error_vs_temperature(
    T_K: np.ndarray,
    rel_error: np.ndarray,
    *,
    title: str = "Relative error vs temperature",
    path: str | os.PathLike | None = None,
) -> Figure:
    """Relative error as a function of temperature with °C primary axis and K secondary.

    Parameters
    ----------
    T_K:
        Temperature in Kelvin, shape ``(N,)`` or ``(N, S)``.
    rel_error:
        Relative error (dimensionless), same shape as *T_K*.
    title:
        Figure title.
    path:
        Optional save path (400-DPI PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    T_K = np.asarray(T_K)
    rel_error = np.asarray(rel_error)

    T_C = T_K - 273.15  # convert to °C for primary axis

    fig, ax = plt.subplots(figsize=(8, 5))

    if T_C.ndim == 1:
        ax.scatter(T_C, rel_error, s=8, alpha=0.5, color=colors[0], rasterized=True)
    else:
        for i in range(T_C.shape[1]):
            ax.scatter(
                T_C[:, i],
                rel_error[:, i],
                s=8,
                alpha=0.4,
                color=colors[i % len(colors)],
                rasterized=True,
            )

    ax.axhline(0.0, color="k", linewidth=1.0)
    ax.axhline(0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="±10 %")
    ax.axhline(-0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Relative error [-]")
    ax.set_title(title)

    # Dual axis: primary °C, secondary K (top)
    dual_temperature_xaxis(ax, primary="C")

    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# 3. Error vs residence time
# ---------------------------------------------------------------------------


def error_vs_residence_time(
    tau_s: np.ndarray,
    rel_error: np.ndarray,
    *,
    title: str = "Relative error vs residence time",
    path: str | os.PathLike | None = None,
) -> Figure:
    """Relative error as a function of residence time.

    Parameters
    ----------
    tau_s:
        Residence time in seconds, shape ``(N,)`` or ``(N, S)``.
    rel_error:
        Relative error (dimensionless), same shape as *tau_s*.
    title:
        Figure title.
    path:
        Optional save path (400-DPI PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    tau_s = np.asarray(tau_s)
    rel_error = np.asarray(rel_error)

    fig, ax = plt.subplots(figsize=(8, 5))

    if tau_s.ndim == 1:
        ax.scatter(tau_s, rel_error, s=8, alpha=0.5, color=colors[0], rasterized=True)
    else:
        for i in range(tau_s.shape[1]):
            ax.scatter(
                tau_s[:, i],
                rel_error[:, i],
                s=8,
                alpha=0.4,
                color=colors[i % len(colors)],
                rasterized=True,
            )

    ax.axhline(0.0, color="k", linewidth=1.0)
    ax.axhline(0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="±10 %")
    ax.axhline(-0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Residence time [s]")
    ax.set_ylabel("Relative error [-]")
    ax.set_title(title)
    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# 4. Error vs conversion
# ---------------------------------------------------------------------------


def error_vs_conversion(
    conversion: np.ndarray,
    rel_error: np.ndarray,
    *,
    title: str = "Relative error vs conversion",
    path: str | os.PathLike | None = None,
) -> Figure:
    """Relative error vs feed conversion — exposes near-inlet / low-conversion deficit (RC-1).

    Parameters
    ----------
    conversion:
        Feed-species conversion fraction [0, 1], shape ``(N,)`` or ``(N, S)``.
    rel_error:
        Relative error (dimensionless), same shape as *conversion*.
    title:
        Figure title.
    path:
        Optional save path (400-DPI PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    conversion = np.asarray(conversion)
    rel_error = np.asarray(rel_error)

    fig, ax = plt.subplots(figsize=(8, 5))

    if conversion.ndim == 1:
        ax.scatter(
            conversion, rel_error, s=8, alpha=0.5, color=colors[0], rasterized=True
        )
    else:
        for i in range(conversion.shape[1]):
            ax.scatter(
                conversion[:, i],
                rel_error[:, i],
                s=8,
                alpha=0.4,
                color=colors[i % len(colors)],
                rasterized=True,
            )

    ax.axhline(0.0, color="k", linewidth=1.0)
    ax.axhline(0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.6, label="±10 %")
    ax.axhline(-0.10, color="k", linestyle="--", linewidth=1.0, alpha=0.6)

    ax.set_xlabel("Conversion [-]")
    ax.set_ylabel("Relative error [-]")
    ax.set_title(title)
    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# 5. Relative-error histogram
# ---------------------------------------------------------------------------


def relative_error_histogram(
    rel_error: np.ndarray,
    *,
    bins: int = 50,
    title: str = "Relative error distribution",
    path: str | os.PathLike | None = None,
) -> Figure:
    """Histogram of relative errors with median marker and ±10 % reference lines.

    The ±10 % ChemZIP tolerance is shown as dashed vertical lines. A solid
    vertical line marks the median.

    Parameters
    ----------
    rel_error:
        Relative error values (dimensionless), any shape — flattened internally.
    bins:
        Number of histogram bins.
    title:
        Figure title.
    path:
        Optional save path (400-DPI PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    rel_error = np.asarray(rel_error).ravel()
    finite_err = rel_error[np.isfinite(rel_error)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(finite_err, bins=bins, color=colors[0], edgecolor="white", linewidth=0.4)

    median_val = float(np.median(finite_err))
    ax.axvline(median_val, color=colors[3], linewidth=2.0, label=f"Median = {median_val:.3f}")
    ax.axvline(0.10, color="k", linestyle="--", linewidth=1.5, alpha=0.7, label="±10 %")
    ax.axvline(-0.10, color="k", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("Relative error [-]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, path)
    return fig


# ---------------------------------------------------------------------------
# 6. Axial profiles
# ---------------------------------------------------------------------------


def axial_profiles(
    z_m: np.ndarray,
    series_dict: Mapping[str, np.ndarray],
    *,
    temperature_keys: Sequence[str] | None = None,
    ylabel: str = "Value",
    title: str = "Axial profiles",
    path: str | os.PathLike | None = None,
) -> Figure:
    """Overlay axial profiles along the reactor coordinate *z*.

    Predicted and reference series are plotted on the same axes.  Any series
    whose key is listed in *temperature_keys* is assumed to be in °C and will
    receive a dual Kelvin axis on the right via
    :func:`~scarfs.plotting.plot_defaults.dual_temperature_axis`.

    Parameters
    ----------
    z_m:
        Axial coordinate in metres, shape ``(N,)``.
    series_dict:
        Mapping ``{label: values}`` where ``values`` has shape ``(N,)``.
        Typical keys: ``"C2H4 ref"``, ``"C2H4 pred"``, ``"T [°C] ref"``, etc.
    temperature_keys:
        Labels in *series_dict* that represent temperature in °C.  When any
        match, a secondary Kelvin y-axis is added (first matching series
        determines placement).
    ylabel:
        Primary y-axis label.
    title:
        Figure title.
    path:
        Optional save path (400-DPI PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    z_m = np.asarray(z_m)
    temperature_keys = list(temperature_keys) if temperature_keys is not None else []

    fig, ax = plt.subplots(figsize=(9, 5))

    has_temp = False
    for i, (label, values) in enumerate(series_dict.items()):
        values = np.asarray(values)
        col = colors[i % len(colors)]
        # Distinguish predicted vs reference by line style heuristic
        ls = "--" if "pred" in label.lower() else "-"
        ax.plot(z_m, values, linestyle=ls, color=col, linewidth=1.8, label=label)
        if label in temperature_keys:
            has_temp = True

    ax.set_xlabel("Axial position [m]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if has_temp:
        # Primary axis is °C; add secondary K axis on the right
        dual_temperature_axis(ax, primary="C")

    ax.legend(fontsize=12)
    fig.tight_layout()
    _save(fig, path)
    return fig

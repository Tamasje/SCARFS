"""Publication-quality benchmark figures for the CRACKSIM/PFR surrogate evaluation.

Each function returns a :class:`matplotlib.figure.Figure` and, when a *path* is
supplied, saves a PNG at 400 DPI via the shared :func:`_save` helper.

Temperature convention (steam-cracking):
    °C is always the primary axis; Kelvin is shown on the secondary axis via
    :func:`~scarfs.plotting.plot_defaults.dual_temperature_xaxis` /
    :func:`~scarfs.plotting.plot_defaults.dual_temperature_axis`.

Diagnostics figures (added in B2c):
    ``energy_parity_figure``, ``tail_rel_err_hist_figure``,
    ``front_localization_figure``, ``accuracy_vs_k_figure``.
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


# ---------------------------------------------------------------------------
# 7. Energy parity figure (B2c diagnostics)
# ---------------------------------------------------------------------------


def energy_parity_figure(
    pred: np.ndarray,
    target: np.ndarray,
    out_path: str | os.PathLike | None = None,
    *,
    title: str = "Energy source parity",
) -> Figure:
    """Log-log parity plot for energy absorption with ±10% and ±25% tolerance bands.

    Both axes are shown on a log10 scale (absolute values; sign assumed
    positive for the absorption column per plan E-c / §4).  The ±10% and ±25%
    bands delimit the ChemZIP-floor tolerance and the provisional tail target
    respectively (plan §5 items 1–2).

    Parameters
    ----------
    pred
        ``(N,)`` predicted absorption [J/m³/s].
    target
        ``(N,)`` reference absorption [J/m³/s].
    out_path
        Optional file path; written as 400-DPI PNG when given.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)

    # Work in absolute value space (absorption ≥ 0 by lineage; clip tiny values)
    floor = 1.0  # J/m³/s
    pred_pos = np.maximum(np.abs(pred), floor)
    tgt_pos = np.maximum(np.abs(target), floor)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(tgt_pos, pred_pos, s=6, alpha=0.5, color=colors[0], rasterized=True)

    finite = np.concatenate([pred_pos, tgt_pos])
    lo = float(np.log10(np.minimum(finite.min(), 1.0)))
    hi = float(np.log10(finite.max())) + 0.1
    ref = np.logspace(lo, hi, 300)

    ax.plot(ref, ref, "k-", linewidth=1.5, label="1:1")
    ax.plot(ref, ref * 1.10, "k--", linewidth=1.0, alpha=0.7, label="±10 %")
    ax.plot(ref, ref * 0.90, "k--", linewidth=1.0, alpha=0.7)
    ax.plot(ref, ref * 1.25, color="gray", linestyle=":", linewidth=1.0, label="±25 %")
    ax.plot(ref, ref * 0.80, color="gray", linestyle=":", linewidth=1.0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference absorption [J/m³/s]")
    ax.set_ylabel("Predicted absorption [J/m³/s]")
    ax.set_title(title)
    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Tail relative-error histogram (B2c diagnostics)
# ---------------------------------------------------------------------------


def tail_rel_err_hist_figure(
    rel_errs: np.ndarray,
    out_path: str | os.PathLike | None = None,
    *,
    bins: int = 50,
    title: str = "Tail relative-error distribution",
) -> Figure:
    """Histogram of relative errors for the high-|S_E| tail, with the 10% target line.

    The top-20% rows (by |S_E|) are the critical tail from plan §5 item 2.
    This figure shows the distribution of their per-row relative errors and
    marks the 10% target / 25% provisional thresholds.

    Parameters
    ----------
    rel_errs
        ``(N,)`` relative errors for the tail rows (dimensionless).
    out_path
        Optional file path; written as 400-DPI PNG when given.
    bins
        Number of histogram bins.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    rel_errs = np.asarray(rel_errs, dtype=float).ravel()
    finite_err = rel_errs[np.isfinite(rel_errs)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(finite_err, bins=bins, color=colors[0], edgecolor="white", linewidth=0.4)

    median_val = float(np.median(finite_err)) if finite_err.size > 0 else 0.0
    ax.axvline(median_val, color=colors[3], linewidth=2.0, label=f"Median = {median_val:.3f}")
    ax.axvline(0.10, color="k", linestyle="--", linewidth=1.5, alpha=0.8, label="10 % target")
    ax.axvline(0.25, color="gray", linestyle=":", linewidth=1.5, alpha=0.7, label="25 % provisional")
    ax.axvline(0.0, color="k", linewidth=0.8, alpha=0.4)

    ax.set_xlabel("Relative error (tail rows) [-]")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Front localisation figure (B2c diagnostics)
# ---------------------------------------------------------------------------


def front_localization_figure(
    tau: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    case_id: str | int,
    out_path: str | os.PathLike | None = None,
) -> Figure:
    """Per-case S_E vs τ overlay with a CDF inset.

    Shows the reference and predicted energy absorption profiles along the
    reactor residence time axis, plus a normalised cumulative-∫S_E CDF inset
    to assess front localisation quality (plan §5 item 3).

    Parameters
    ----------
    tau
        ``(N,)`` residence time [s].
    target
        ``(N,)`` reference absorption [J/m³/s].
    pred
        ``(N,)`` predicted absorption [J/m³/s].
    case_id
        Case identifier for the figure title.
    out_path
        Optional file path; written as 400-DPI PNG when given.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    tau = np.asarray(tau, dtype=float)
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)

    order = np.argsort(tau)
    tau = tau[order]
    target = target[order]
    pred = pred[order]

    fig, (ax_main, ax_cdf) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]}
    )

    # --- Main panel: S_E vs τ ---
    ax_main.plot(tau, target, "-", color=colors[0], linewidth=2.0, label="Reference")
    ax_main.plot(tau, pred, "--", color=colors[3], linewidth=1.8, label="Predicted")
    ax_main.set_xlabel("Residence time τ [s]")
    ax_main.set_ylabel("Absorption [J/m³/s]")
    ax_main.set_title(f"Case {case_id} — S_E profile")
    ax_main.legend(fontsize=13)

    # --- CDF inset panel: normalised cumulative ∫S_E ---
    dtau = np.diff(tau, prepend=tau[0])
    dtau[0] = dtau[1] if len(dtau) > 1 else 1.0
    tgt_int = np.cumsum(np.abs(target) * dtau)
    pred_int = np.cumsum(np.abs(pred) * dtau)
    tgt_norm = tgt_int / max(float(tgt_int[-1]), 1.0)
    pred_norm = pred_int / max(float(pred_int[-1]), 1.0)

    ax_cdf.plot(tau, tgt_norm, "-", color=colors[0], linewidth=2.0, label="Reference CDF")
    ax_cdf.plot(tau, pred_norm, "--", color=colors[3], linewidth=1.8, label="Predicted CDF")
    ax_cdf.set_xlabel("τ [s]")
    ax_cdf.set_ylabel("Normalised cumulative ∫|S_E|dτ")
    ax_cdf.set_title("CDF")
    ax_cdf.legend(fontsize=11)

    fig.tight_layout()
    _save(fig, out_path)
    return fig


# ---------------------------------------------------------------------------
# 10. Accuracy vs k ablation figure (B2c diagnostics)
# ---------------------------------------------------------------------------


def accuracy_vs_k_figure(
    ks: Sequence[int],
    metric_by_variant: dict[str, list[float]],
    out_path: str | os.PathLike | None = None,
    metric_name: str = "R²",
    *,
    title: str | None = None,
) -> Figure:
    """Metric vs latent dimension k for the k-ablation study.

    Shows one line per model variant (e.g. ``"tail R²"``, ``"full R²"``)
    across the tested latent dimensions k ∈ {4, 6, 8, 12, 16} (plan §4 E-e).

    Parameters
    ----------
    ks
        Sequence of k values on the x-axis (e.g. ``[4, 6, 8, 12, 16]``).
    metric_by_variant
        Dict mapping variant name to a list of metric values (same length as
        *ks*).
    out_path
        Optional file path; written as 400-DPI PNG when given.
    metric_name
        Label for the y-axis (e.g. ``"R²"`` or ``"relRMSE"``).
    title
        Figure title; defaults to ``"<metric_name> vs latent dimension k"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    apply_defaults()
    colors = palette()

    ks = list(ks)
    title = title or f"{metric_name} vs latent dimension k"

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (variant, vals) in enumerate(metric_by_variant.items()):
        color = colors[i % len(colors)]
        ax.plot(ks, vals, "o-", color=color, linewidth=2.0, markersize=7, label=variant)

    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.set_xticks(ks)
    ax.legend(fontsize=13)
    fig.tight_layout()
    _save(fig, out_path)
    return fig

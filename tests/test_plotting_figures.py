"""Tests for scarfs.plotting — one function per test, AAA pattern.

Uses the Agg (non-interactive) backend so tests run without a display.
Each test renders a figure to a temporary PNG and checks:
  - the file exists and is non-empty,
  - any dual temperature axis is present where required.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must come before importing pyplot

import numpy as np
import pytest

import scarfs.plotting as sp
from scarfs.plotting import (
    axial_profiles,
    error_vs_conversion,
    error_vs_residence_time,
    error_vs_temperature,
    parity_plot,
    relative_error_histogram,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    """Return a temporary PNG path inside pytest's temp dir."""
    return tmp_path / "out.png"


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# plot_defaults
# ---------------------------------------------------------------------------

def test_apply_defaults_sets_savefig_dpi() -> None:
    """apply_defaults sets savefig.dpi to 400."""
    # Arrange
    import matplotlib.pyplot as plt
    # Act
    sp.apply_defaults()
    # Assert
    assert plt.rcParams["savefig.dpi"] == 400


def test_palette_returns_twelve_hex_colours() -> None:
    """palette() returns 12 hex colour strings."""
    # Arrange / Act
    pal = sp.palette()
    # Assert
    assert len(pal) == 12
    for colour in pal:
        assert colour.startswith("#")


def test_dual_temperature_axis_celsius_primary() -> None:
    """dual_temperature_axis with primary='C' attaches a secondary K y-axis."""
    # Arrange
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylabel("Temperature [°C]")
    # Act
    secax = sp.dual_temperature_axis(ax, primary="C")
    # Assert
    assert secax is not None
    assert secax.get_ylabel() == "Temperature [K]"
    plt.close(fig)


def test_dual_temperature_axis_kelvin_primary() -> None:
    """dual_temperature_axis with primary='K' attaches a secondary °C y-axis."""
    # Arrange
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_ylabel("Temperature [K]")
    # Act
    secax = sp.dual_temperature_axis(ax, primary="K")
    # Assert
    assert secax.get_ylabel() == "Temperature [°C]"
    plt.close(fig)


def test_dual_temperature_axis_invalid_primary() -> None:
    """dual_temperature_axis raises ValueError for unknown primary unit."""
    # Arrange
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # Act / Assert
    with pytest.raises(ValueError, match="primary must be"):
        sp.dual_temperature_axis(ax, primary="F")
    plt.close(fig)


def test_dual_temperature_xaxis_celsius_primary() -> None:
    """dual_temperature_xaxis with primary='C' attaches a secondary K x-axis."""
    # Arrange
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel("Temperature [°C]")
    # Act
    secax = sp.dual_temperature_xaxis(ax, primary="C")
    # Assert
    assert secax.get_xlabel() == "Temperature [K]"
    plt.close(fig)


def test_dual_temperature_xaxis_invalid_primary() -> None:
    """dual_temperature_xaxis raises ValueError for unknown primary unit."""
    # Arrange
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # Act / Assert
    with pytest.raises(ValueError, match="primary must be"):
        sp.dual_temperature_xaxis(ax, primary="R")
    plt.close(fig)


# ---------------------------------------------------------------------------
# parity_plot
# ---------------------------------------------------------------------------

def test_parity_plot_saves_non_empty_png(tmp_png: Path) -> None:
    """parity_plot writes a non-empty PNG when path is given."""
    # Arrange
    rng = _rng(1)
    y_true = rng.uniform(0, 1, (50, 3))
    y_pred = y_true + rng.normal(0, 0.05, y_true.shape)
    labels = ["H2", "CH4", "C2H4"]
    # Act
    fig = parity_plot(y_true, y_pred, labels, path=tmp_png)
    # Assert
    assert tmp_png.exists()
    assert tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


def test_parity_plot_returns_figure() -> None:
    """parity_plot returns a matplotlib Figure."""
    # Arrange
    rng = _rng(2)
    y_true = rng.uniform(0, 1, (20, 2))
    y_pred = y_true * 1.05
    # Act
    fig = parity_plot(y_true, y_pred, ["A", "B"])
    # Assert
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)
    import matplotlib.pyplot as plt; plt.close(fig)


def test_parity_plot_log_scale(tmp_png: Path) -> None:
    """parity_plot with log_scale=True saves a non-empty PNG."""
    # Arrange
    rng = _rng(3)
    y_true = rng.uniform(-100, 100, (30, 2))
    y_pred = y_true + rng.normal(0, 5, y_true.shape)
    # Act
    fig = parity_plot(y_true, y_pred, ["R1", "R2"], log_scale=True, path=tmp_png)
    # Assert
    assert tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# error_vs_temperature
# ---------------------------------------------------------------------------

def test_error_vs_temperature_saves_png(tmp_png: Path) -> None:
    """error_vs_temperature writes a non-empty PNG."""
    # Arrange
    rng = _rng(4)
    T_K = rng.uniform(900, 1200, 80)
    rel_error = rng.normal(0, 0.08, 80)
    # Act
    fig = error_vs_temperature(T_K, rel_error, path=tmp_png)
    # Assert
    assert tmp_png.exists() and tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


def test_error_vs_temperature_has_dual_x_axis(tmp_png: Path) -> None:
    """error_vs_temperature produces a figure with a secondary (K) x-axis."""
    # Arrange
    rng = _rng(5)
    T_K = rng.uniform(900, 1200, 60)
    rel_error = rng.normal(0, 0.05, 60)
    # Act
    fig = error_vs_temperature(T_K, rel_error, path=tmp_png)
    ax = fig.axes[0]
    # Assert: a secondary xaxis is present as a child of the primary axis
    # (matplotlib secondary_xaxis/secondary_yaxis live in ax.child_axes, not fig.axes)
    assert len(ax.child_axes) >= 1, "Expected at least one secondary axis (K)"
    import matplotlib.pyplot as plt; plt.close(fig)


def test_error_vs_temperature_secondary_xlabel_is_kelvin() -> None:
    """The secondary x-axis is labelled 'Temperature [K]'."""
    # Arrange
    rng = _rng(6)
    T_K = rng.uniform(900, 1200, 40)
    rel_error = rng.normal(0, 0.05, 40)
    # Act
    fig = error_vs_temperature(T_K, rel_error)
    ax = fig.axes[0]
    # secondary_xaxis lives in ax.child_axes, not fig.axes
    assert any("K" in a.get_xlabel() for a in ax.child_axes), (
        "No secondary axis with 'K' label found"
    )
    import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# error_vs_residence_time
# ---------------------------------------------------------------------------

def test_error_vs_residence_time_saves_png(tmp_png: Path) -> None:
    """error_vs_residence_time writes a non-empty PNG."""
    # Arrange
    rng = _rng(7)
    tau = rng.uniform(0.01, 0.5, 60)
    err = rng.normal(0, 0.07, 60)
    # Act
    fig = error_vs_residence_time(tau, err, path=tmp_png)
    # Assert
    assert tmp_png.exists() and tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


def test_error_vs_residence_time_xlabel() -> None:
    """error_vs_residence_time labels the x-axis with 'Residence time'."""
    # Arrange
    rng = _rng(8)
    tau = rng.uniform(0.01, 0.5, 30)
    err = rng.normal(0, 0.05, 30)
    # Act
    fig = error_vs_residence_time(tau, err)
    ax = fig.axes[0]
    # Assert
    assert "Residence time" in ax.get_xlabel()
    import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# error_vs_conversion
# ---------------------------------------------------------------------------

def test_error_vs_conversion_saves_png(tmp_png: Path) -> None:
    """error_vs_conversion writes a non-empty PNG."""
    # Arrange
    rng = _rng(9)
    conv = rng.uniform(0, 1, 70)
    err = rng.normal(0, 0.09, 70)
    # Act
    fig = error_vs_conversion(conv, err, path=tmp_png)
    # Assert
    assert tmp_png.exists() and tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


def test_error_vs_conversion_xlabel() -> None:
    """error_vs_conversion labels the x-axis with 'Conversion'."""
    # Arrange
    rng = _rng(10)
    conv = rng.uniform(0, 1, 40)
    err = rng.normal(0, 0.05, 40)
    # Act
    fig = error_vs_conversion(conv, err)
    ax = fig.axes[0]
    # Assert
    assert "Conversion" in ax.get_xlabel()
    import matplotlib.pyplot as plt; plt.close(fig)


def test_error_vs_conversion_2d_input(tmp_png: Path) -> None:
    """error_vs_conversion handles 2-D (N, S) input without error."""
    # Arrange
    rng = _rng(11)
    conv = rng.uniform(0, 1, (50, 3))
    err = rng.normal(0, 0.06, (50, 3))
    # Act
    fig = error_vs_conversion(conv, err, path=tmp_png)
    # Assert
    assert tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# relative_error_histogram
# ---------------------------------------------------------------------------

def test_relative_error_histogram_saves_png(tmp_png: Path) -> None:
    """relative_error_histogram writes a non-empty PNG."""
    # Arrange
    rng = _rng(12)
    err = rng.normal(0, 0.08, 200)
    # Act
    fig = relative_error_histogram(err, path=tmp_png)
    # Assert
    assert tmp_png.exists() and tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


def test_relative_error_histogram_has_median_line() -> None:
    """relative_error_histogram adds a vertical median line."""
    # Arrange
    rng = _rng(13)
    err = rng.normal(0.02, 0.06, 150)
    # Act
    fig = relative_error_histogram(err)
    ax = fig.axes[0]
    # Assert: at least one vertical line (the median)
    vlines = [line for line in ax.get_lines()]
    assert len(vlines) >= 1, "Expected vertical line(s) for median / ±10% references"
    import matplotlib.pyplot as plt; plt.close(fig)


def test_relative_error_histogram_label_content() -> None:
    """relative_error_histogram legend contains 'Median' and '±10 %'."""
    # Arrange
    rng = _rng(14)
    err = rng.normal(0, 0.05, 100)
    # Act
    fig = relative_error_histogram(err)
    ax = fig.axes[0]
    legend = ax.get_legend()
    legend_text = " ".join(t.get_text() for t in legend.get_texts())
    # Assert
    assert "Median" in legend_text
    assert "10" in legend_text
    import matplotlib.pyplot as plt; plt.close(fig)


# ---------------------------------------------------------------------------
# axial_profiles
# ---------------------------------------------------------------------------

def test_axial_profiles_saves_png(tmp_png: Path) -> None:
    """axial_profiles writes a non-empty PNG."""
    # Arrange
    rng = _rng(15)
    z = np.linspace(0, 10, 100)
    series = {
        "C2H4 ref": rng.uniform(0.1, 0.3, 100),
        "C2H4 pred": rng.uniform(0.1, 0.3, 100),
    }
    # Act
    fig = axial_profiles(z, series, path=tmp_png)
    # Assert
    assert tmp_png.exists() and tmp_png.stat().st_size > 0
    import matplotlib.pyplot as plt; plt.close(fig)


def test_axial_profiles_temperature_dual_axis() -> None:
    """axial_profiles adds a secondary K y-axis when temperature_keys are given."""
    # Arrange
    rng = _rng(16)
    z = np.linspace(0, 10, 80)
    T_C = rng.uniform(700, 900, 80)
    series = {"T_ref": T_C}
    # Act
    fig = axial_profiles(z, series, temperature_keys=["T_ref"], ylabel="Temperature [°C]")
    ax = fig.axes[0]
    # Assert: secondary y-axis present in ax.child_axes
    assert len(ax.child_axes) >= 1, "Expected a secondary K y-axis for temperature profile"
    import matplotlib.pyplot as plt; plt.close(fig)


def test_axial_profiles_secondary_ylabel_kelvin() -> None:
    """When a temperature key is provided, the secondary y-axis is labelled 'Temperature [K]'."""
    # Arrange
    rng = _rng(17)
    z = np.linspace(0, 5, 60)
    T_C = rng.uniform(700, 900, 60)
    series = {"T_ref": T_C}
    # Act
    fig = axial_profiles(z, series, temperature_keys=["T_ref"], ylabel="Temperature [°C]")
    ax_main = fig.axes[0]
    # secondary_yaxis lives in ax.child_axes, not fig.axes
    assert any("K" in a.get_ylabel() for a in ax_main.child_axes), (
        "Secondary y-axis should be labelled with K"
    )
    import matplotlib.pyplot as plt; plt.close(fig)


def test_axial_profiles_no_dual_axis_without_temp_keys() -> None:
    """axial_profiles does NOT add a secondary axis when temperature_keys is empty."""
    # Arrange
    rng = _rng(18)
    z = np.linspace(0, 5, 50)
    series = {"Y_C2H4 ref": rng.uniform(0, 0.3, 50)}
    # Act
    fig = axial_profiles(z, series)
    ax = fig.axes[0]
    # Assert: no child secondary axes attached
    assert len(ax.child_axes) == 0, f"Expected no secondary axes, got {ax.child_axes}"
    import matplotlib.pyplot as plt; plt.close(fig)

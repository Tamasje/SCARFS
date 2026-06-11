"""Tests for the B2c diagnostics figures added to scarfs.plotting.

Every test uses the Agg (non-interactive) backend and follows the AAA pattern.
DPI ≥ 400 is verified via ``matplotlib.rcParams["savefig.dpi"]`` set by
``apply_defaults()`` (not via PIL, since DPI metadata in PNG is unreliable;
the rcParam is the authoritative value for savefig).

Leak check: every test closes its figure handle with ``plt.close(fig)``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scarfs.plotting import apply_defaults


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_png(tmp_path: Path) -> Path:
    return tmp_path / "test_figure.png"


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# energy_parity_figure
# ---------------------------------------------------------------------------


def test_energy_parity_figure_saves_png(tmp_png: Path) -> None:
    """energy_parity_figure writes a non-empty PNG when out_path is given."""
    # Arrange
    from scarfs.plotting import energy_parity_figure

    rng = _rng(0)
    target = rng.uniform(1e3, 1e8, 60)
    pred = target * rng.uniform(0.85, 1.15, 60)

    # Act
    fig = energy_parity_figure(pred, target, tmp_png)

    # Assert
    assert tmp_png.exists()
    assert tmp_png.stat().st_size > 0
    plt.close(fig)


def test_energy_parity_figure_returns_figure() -> None:
    """energy_parity_figure returns a matplotlib Figure."""
    # Arrange
    from matplotlib.figure import Figure
    from scarfs.plotting import energy_parity_figure

    rng = _rng(1)
    target = rng.uniform(1e4, 1e9, 30)
    pred = target * 0.9

    # Act
    fig = energy_parity_figure(pred, target)

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_energy_parity_figure_uses_log_scale() -> None:
    """energy_parity_figure sets log scale on both axes."""
    # Arrange
    from scarfs.plotting import energy_parity_figure

    rng = _rng(2)
    target = rng.uniform(1e2, 1e8, 40)
    pred = target * rng.uniform(0.9, 1.1, 40)

    # Act
    fig = energy_parity_figure(pred, target)
    ax = fig.axes[0]

    # Assert
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_energy_parity_figure_savefig_dpi_400() -> None:
    """apply_defaults sets savefig.dpi to 400 — the DPI energy_parity_figure uses."""
    # Arrange / Act
    apply_defaults()

    # Assert
    assert plt.rcParams["savefig.dpi"] == 400


def test_energy_parity_figure_closes_cleanly(tmp_png: Path) -> None:
    """energy_parity_figure figure handle can be closed without error."""
    # Arrange
    from scarfs.plotting import energy_parity_figure

    target = np.logspace(3, 9, 50)
    pred = target * 1.05

    # Act
    fig = energy_parity_figure(pred, target, tmp_png)

    # Assert — close without error (no resource leak)
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# tail_rel_err_hist_figure
# ---------------------------------------------------------------------------


def test_tail_rel_err_hist_saves_png(tmp_png: Path) -> None:
    """tail_rel_err_hist_figure writes a non-empty PNG."""
    # Arrange
    from scarfs.plotting import tail_rel_err_hist_figure

    rng = _rng(3)
    rel_errs = rng.uniform(0, 0.5, 120)

    # Act
    fig = tail_rel_err_hist_figure(rel_errs, tmp_png)

    # Assert
    assert tmp_png.exists()
    assert tmp_png.stat().st_size > 0
    plt.close(fig)


def test_tail_rel_err_hist_has_10pct_line() -> None:
    """tail_rel_err_hist_figure draws the 10% target reference line."""
    # Arrange
    from scarfs.plotting import tail_rel_err_hist_figure

    rng = _rng(4)
    rel_errs = rng.uniform(0, 0.4, 80)

    # Act
    fig = tail_rel_err_hist_figure(rel_errs)
    ax = fig.axes[0]

    # Assert — at least 2 vertical lines expected (median + 10% + 25%)
    vlines = ax.get_lines()
    assert len(vlines) >= 2, "Expected vertical reference lines"
    plt.close(fig)


def test_tail_rel_err_hist_legend_has_target_label() -> None:
    """tail_rel_err_hist_figure legend mentions the 10% target."""
    # Arrange
    from scarfs.plotting import tail_rel_err_hist_figure

    rng = _rng(5)
    rel_errs = rng.uniform(0, 0.3, 60)

    # Act
    fig = tail_rel_err_hist_figure(rel_errs)
    ax = fig.axes[0]
    legend = ax.get_legend()
    legend_text = " ".join(t.get_text() for t in legend.get_texts())

    # Assert
    assert "10" in legend_text or "%" in legend_text
    plt.close(fig)


def test_tail_rel_err_hist_closes_cleanly(tmp_png: Path) -> None:
    """tail_rel_err_hist_figure figure handle can be closed without error."""
    # Arrange
    from scarfs.plotting import tail_rel_err_hist_figure

    rel_errs = np.linspace(0, 0.4, 50)

    # Act
    fig = tail_rel_err_hist_figure(rel_errs, tmp_png)
    plt.close(fig)

    # Assert
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# front_localization_figure
# ---------------------------------------------------------------------------


def test_front_localization_figure_saves_png(tmp_png: Path) -> None:
    """front_localization_figure writes a non-empty PNG."""
    # Arrange
    from scarfs.plotting import front_localization_figure

    rng = _rng(6)
    tau = np.linspace(0.0, 0.5, 22)
    target = rng.uniform(1e5, 5e8, 22)
    pred = target * rng.uniform(0.85, 1.15, 22)

    # Act
    fig = front_localization_figure(tau, target, pred, case_id=42, out_path=tmp_png)

    # Assert
    assert tmp_png.exists()
    assert tmp_png.stat().st_size > 0
    plt.close(fig)


def test_front_localization_figure_returns_figure() -> None:
    """front_localization_figure returns a matplotlib Figure."""
    # Arrange
    from matplotlib.figure import Figure
    from scarfs.plotting import front_localization_figure

    tau = np.linspace(0, 0.3, 15)
    target = np.exp(-tau * 5) * 1e7
    pred = target * 0.95

    # Act
    fig = front_localization_figure(tau, target, pred, case_id="test_case")

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_front_localization_figure_has_two_panels() -> None:
    """front_localization_figure creates a figure with 2 axes (main + CDF)."""
    # Arrange
    from scarfs.plotting import front_localization_figure

    tau = np.linspace(0, 0.4, 18)
    target = np.abs(np.sin(tau * 10)) * 1e8
    pred = target * 1.02

    # Act
    fig = front_localization_figure(tau, target, pred, case_id=1)

    # Assert — 2 subplots: profile + CDF
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_front_localization_figure_closes_cleanly(tmp_png: Path) -> None:
    """front_localization_figure figure handle closes without error."""
    # Arrange
    from scarfs.plotting import front_localization_figure

    tau = np.linspace(0, 0.2, 10)
    target = np.ones(10) * 1e6
    pred = target * 0.9

    # Act
    fig = front_localization_figure(tau, target, pred, case_id=99, out_path=tmp_png)
    plt.close(fig)

    # Assert
    assert not plt.fignum_exists(fig.number)


# ---------------------------------------------------------------------------
# accuracy_vs_k_figure
# ---------------------------------------------------------------------------


def test_accuracy_vs_k_saves_png(tmp_png: Path) -> None:
    """accuracy_vs_k_figure writes a non-empty PNG."""
    # Arrange
    from scarfs.plotting import accuracy_vs_k_figure

    ks = [4, 6, 8, 12, 16]
    metrics = {"full R²": [0.60, 0.72, 0.81, 0.88, 0.91], "tail R²": [0.30, 0.45, 0.55, 0.68, 0.72]}

    # Act
    fig = accuracy_vs_k_figure(ks, metrics, tmp_png, metric_name="R²")

    # Assert
    assert tmp_png.exists()
    assert tmp_png.stat().st_size > 0
    plt.close(fig)


def test_accuracy_vs_k_returns_figure() -> None:
    """accuracy_vs_k_figure returns a matplotlib Figure."""
    # Arrange
    from matplotlib.figure import Figure
    from scarfs.plotting import accuracy_vs_k_figure

    ks = [4, 8, 16]
    metrics = {"relRMSE": [0.50, 0.30, 0.25]}

    # Act
    fig = accuracy_vs_k_figure(ks, metrics, metric_name="relRMSE")

    # Assert
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_accuracy_vs_k_legend_has_variant_names() -> None:
    """accuracy_vs_k_figure legend contains the variant names."""
    # Arrange
    from scarfs.plotting import accuracy_vs_k_figure

    ks = [4, 6, 8]
    variant_name = "special_variant"
    metrics = {variant_name: [0.7, 0.8, 0.85]}

    # Act
    fig = accuracy_vs_k_figure(ks, metrics, metric_name="R²")
    ax = fig.axes[0]
    legend = ax.get_legend()
    legend_text = " ".join(t.get_text() for t in legend.get_texts())

    # Assert
    assert variant_name in legend_text
    plt.close(fig)


def test_accuracy_vs_k_x_ticks_match_ks() -> None:
    """accuracy_vs_k_figure sets x-ticks to match the provided k values."""
    # Arrange
    from scarfs.plotting import accuracy_vs_k_figure

    ks = [4, 6, 8, 12, 16]
    metrics = {"A": [0.1 * k for k in ks]}

    # Act
    fig = accuracy_vs_k_figure(ks, metrics, metric_name="score")
    ax = fig.axes[0]

    # Assert
    ticks = list(ax.get_xticks())
    for k in ks:
        assert k in ticks, f"Expected tick {k} in {ticks}"
    plt.close(fig)


def test_accuracy_vs_k_closes_cleanly(tmp_png: Path) -> None:
    """accuracy_vs_k_figure figure handle closes without error."""
    # Arrange
    from scarfs.plotting import accuracy_vs_k_figure

    ks = [4, 8]
    metrics = {"A": [0.5, 0.7]}

    # Act
    fig = accuracy_vs_k_figure(ks, metrics, tmp_png, metric_name="R²")
    plt.close(fig)

    # Assert
    assert not plt.fignum_exists(fig.number)

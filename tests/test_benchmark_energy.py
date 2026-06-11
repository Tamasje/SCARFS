"""Tests for scarfs.benchmark.energy — §5 energy acceptance suite.

Each test follows arrange/act/assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from scarfs.benchmark.energy import (
    EnergyThresholds,
    EnergyReport,
    evaluate_energy,
    ABS_FLOOR_GENERATOR_NOISE,
    ABS_FLOOR_SPECIES_TRUNCATION,
    ABS_FLOOR_INFORMATION_LOW,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_perfect_data(n_cases: int = 4, n_pts: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (pred, target, case_ids, tau) where pred == target (perfect model)."""
    rng = np.random.default_rng(42)
    target = np.abs(rng.normal(1e8, 5e7, n_cases * n_pts))
    pred = target.copy()
    case_ids = np.repeat(np.arange(n_cases), n_pts)
    tau = np.tile(np.linspace(0, 0.5, n_pts), n_cases)
    return pred, target, case_ids, tau


def _make_poor_data(n_cases: int = 4, n_pts: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return data where pred is 3× target (large error)."""
    rng = np.random.default_rng(7)
    target = np.abs(rng.normal(1e8, 5e7, n_cases * n_pts))
    pred = 3.0 * target
    case_ids = np.repeat(np.arange(n_cases), n_pts)
    tau = np.tile(np.linspace(0, 0.5, n_pts), n_cases)
    return pred, target, case_ids, tau


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_evaluate_energy_perfect_model_passes():
    """A perfect surrogate (pred == target) must pass all §5 criteria."""
    # Arrange
    pred, target, case_ids, tau = _make_perfect_data()

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert
    assert report.passed, f"Perfect model should PASS.  Failures: {[k for k,v in report.pass_fail.items() if not v]}"
    assert report.global_r2 == pytest.approx(1.0, abs=1e-10)
    assert report.global_rel_rmse == pytest.approx(0.0, abs=1e-10)


def test_evaluate_energy_poor_model_fails():
    """A bad surrogate (pred = 3 × target) must fail the global R² criterion."""
    # Arrange
    pred, target, case_ids, tau = _make_poor_data()

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert
    assert not report.passed
    # Global R² for 3x scale error is negative
    assert report.global_r2 < 0.95


def test_evaluate_energy_returns_correct_shape():
    """EnergyReport has the right number of cases and rows."""
    # Arrange
    pred, target, case_ids, tau = _make_perfect_data(n_cases=3, n_pts=10)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert
    assert report.n_cases == 3
    assert report.n_rows == 30
    assert len(report.case_results) == 3


def test_evaluate_energy_single_row_case_no_crash():
    """Single-row cases must not raise NaN-related exceptions."""
    # Arrange
    rng = np.random.default_rng(0)
    n = 10
    target = np.abs(rng.normal(1e8, 5e7, n))
    pred = target * 0.9
    case_ids = np.arange(n)  # one row per case
    tau = np.linspace(0, 0.1, n)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert — must not raise; all NaN values are acceptable for degenerate cases
    assert report.n_cases == n
    assert np.isfinite(report.global_r2)


def test_evaluate_energy_zero_target_no_crash():
    """All-zero target must not crash (degenerate case)."""
    # Arrange
    pred = np.ones(20, dtype=float)
    target = np.zeros(20, dtype=float)
    case_ids = np.zeros(20, dtype=int)
    tau = np.linspace(0, 1.0, 20)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert — must return without NaN crash; global_r2 is 0 for constant target
    assert report.n_rows == 20


def test_evaluate_energy_custom_thresholds():
    """Custom thresholds are respected."""
    # Arrange
    pred, target, case_ids, tau = _make_poor_data()
    loose = EnergyThresholds(r2_min=-100.0, rel_rmse_max=1000.0,
                             tail_median_rel_err_max=1000.0, tail_p95_rel_err_max=1000.0,
                             tail_rel_rmse_max=1000.0, peak_tau_position_err_max=1000.0,
                             cdf_max_dev_max=1000.0, integral_rel_err_max_median=1000.0,
                             integral_rel_err_max_p95=1000.0)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau, thresholds=loose)

    # Assert — with very loose thresholds even poor data passes
    assert report.passed


def test_evaluate_energy_tail_fraction_kwarg():
    """tail_fraction parameter is honoured."""
    # Arrange
    pred, target, case_ids, tau = _make_perfect_data()

    # Act: use a very small tail fraction (extreme tail only)
    report_tight = evaluate_energy(pred, target, case_ids, tau, tail_fraction=0.05)
    report_wide = evaluate_energy(pred, target, case_ids, tau, tail_fraction=0.50)

    # Assert: n_tail_pooled differs
    assert report_tight.n_tail_pooled <= report_wide.n_tail_pooled


def test_evaluate_energy_sign_negative_fraction():
    """sign_negative_fraction is computed correctly."""
    # Arrange
    n = 40
    pred = np.concatenate([np.ones(20) * 1e8, -np.ones(20) * 1e8])  # 50% negative
    target = np.abs(pred)
    case_ids = np.repeat([0, 1], 20)
    tau = np.tile(np.linspace(0, 0.5, 20), 2)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert
    assert report.sign_negative_fraction == pytest.approx(0.5, abs=1e-10)


def test_evaluate_energy_summary_string():
    """summary() returns a non-empty string containing key headings."""
    # Arrange
    pred, target, case_ids, tau = _make_perfect_data()

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)
    summary = report.summary()

    # Assert
    assert "ENERGY ACCEPTANCE REPORT" in summary
    assert "PASS" in summary
    assert str(ABS_FLOOR_GENERATOR_NOISE) in summary or "5.1" in summary


def test_abs_floor_constants_are_ordered():
    """Absolute floor constants satisfy the expected ordering."""
    # Arrange / Act / Assert — all in one line; these are invariant physics facts
    assert ABS_FLOOR_GENERATOR_NOISE < ABS_FLOOR_SPECIES_TRUNCATION < ABS_FLOOR_INFORMATION_LOW


def test_evaluate_energy_integral_budget_correct():
    """For a model where pred = 2 × target, the integral rel-err should be ~1."""
    # Arrange
    n_cases, n_pts = 2, 15
    rng = np.random.default_rng(1)
    target = np.abs(rng.normal(1e8, 3e7, n_cases * n_pts))
    pred = 2.0 * target
    case_ids = np.repeat(np.arange(n_cases), n_pts)
    tau = np.tile(np.linspace(0, 0.5, n_pts), n_cases)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert: integral rel-err ≈ 1 (pred has twice the total energy)
    assert report.integral_rel_err_median == pytest.approx(1.0, abs=0.01)


def test_evaluate_energy_uses_stride6_fixture():
    """Smoke test on the real stride6 parquet fixture."""
    import pandas as pd
    from pathlib import Path
    from scarfs.benchmark.loader import infer_schema

    fixture = Path(__file__).parent / "data" / "stride6_sample.parquet"
    if not fixture.exists():
        pytest.skip("stride6_sample.parquet not found")

    # Arrange
    df = pd.read_parquet(str(fixture))
    schema = infer_schema(df)
    abs_col = schema.energy_target_column()
    target = df[abs_col].to_numpy(dtype=float)
    pred = target * 0.9  # 10% under-prediction

    case_col = schema.meta.get("CaseID")
    tau_col = schema.state.get("tau")
    case_ids = df[case_col].to_numpy() if case_col else np.zeros(len(df), dtype=int)
    tau = df[tau_col].to_numpy(dtype=float) if tau_col else np.arange(len(df), dtype=float)

    # Act
    report = evaluate_energy(pred, target, case_ids, tau)

    # Assert — basic sanity: R² close to 1 for 90% prediction
    assert report.global_r2 > 0.95
    assert report.n_rows == len(df)
    assert report.n_cases == df[case_col].nunique() if case_col else 1

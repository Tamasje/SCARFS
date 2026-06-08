"""Tests for scarfs.benchmark.metrics — pure metric functions.

Each test follows arrange / act / assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.benchmark.metrics import (
    r2_score,
    nrmse,
    nmdae,
    relative_error,
    median_relative_error,
    per_species_metrics,
)


# ---------------------------------------------------------------------------
# r2_score
# ---------------------------------------------------------------------------

class TestR2Score:
    def test_perfect_prediction(self) -> None:
        """Arrange: y_pred == y_true. Act: r2_score. Assert: 1.0."""
        y = np.array([1.0, 2.0, 3.0, 4.0])

        result = r2_score(y, y)

        assert result == pytest.approx(1.0)

    def test_mean_predictor(self) -> None:
        """Arrange: y_pred = mean(y_true). Act/Assert: R² == 0.0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, y_true.mean())

        result = r2_score(y_true, y_pred)

        assert result == pytest.approx(0.0)

    def test_negative_r2(self) -> None:
        """Arrange: adversarial predictor. Act/Assert: R² < 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])  # perfectly anti-correlated

        result = r2_score(y_true, y_pred)

        assert result < 0.0

    def test_constant_true_returns_zero(self) -> None:
        """Arrange: constant y_true (ss_tot = 0). Act/Assert: 0.0 (defined behaviour)."""
        y_true = np.full(5, 3.0)
        y_pred = np.full(5, 2.0)

        result = r2_score(y_true, y_pred)

        assert result == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """Arrange: known analytic case. Act/Assert: R² matches hand calculation."""
        # y_true = [1, 2, 3], mean = 2, SS_tot = 2, y_pred = [1.5, 2, 2.5]
        # SS_res = 0.25 + 0 + 0.25 = 0.5, R2 = 1 - 0.5/2 = 0.75
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])

        result = r2_score(y_true, y_pred)

        assert result == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# nrmse
# ---------------------------------------------------------------------------

class TestNrmse:
    def test_perfect_prediction_zero(self) -> None:
        """Arrange: y_pred == y_true. Act/Assert: NRMSE == 0."""
        y = np.array([1.0, 2.0, 3.0])

        result = nrmse(y, y)

        assert result == pytest.approx(0.0)

    def test_normalisation_by_std(self) -> None:
        """Arrange: RMSE == std. Act/Assert: NRMSE == 1.0."""
        y_true = np.array([0.0, 2.0])  # std = 1.0
        y_pred = np.array([1.0, 3.0])  # RMSE = 1.0

        result = nrmse(y_true, y_pred)

        assert result == pytest.approx(1.0)

    def test_constant_true_returns_raw_rmse(self) -> None:
        """Arrange: std(y_true)==0, pred != true. Act/Assert: returns raw RMSE."""
        y_true = np.full(4, 2.0)
        y_pred = np.full(4, 3.0)  # RMSE = 1.0

        result = nrmse(y_true, y_pred)

        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# nmdae
# ---------------------------------------------------------------------------

class TestNmdae:
    def test_perfect_prediction_zero(self) -> None:
        """Arrange: y_pred == y_true. Act/Assert: NMdAE == 0."""
        y = np.array([1.0, -1.0, 2.0])

        result = nmdae(y, y)

        assert result == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """Arrange: y_true=[1,2,3], y_pred=[2,2,2]. Act/Assert: matches hand calc.

        Errors: [1, 0, 1], median = 1.0.
        Scale: median(|y_true|) = 2.0.
        NMdAE = 1.0 / 2.0 = 0.5.
        """
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.full(3, 2.0)

        result = nmdae(y_true, y_pred)

        assert result == pytest.approx(0.5)

    def test_zero_true_uses_eps(self) -> None:
        """Arrange: y_true all zeros, y_pred non-zero. Act/Assert: no ZeroDivisionError."""
        y_true = np.zeros(4)
        y_pred = np.ones(4)

        result = nmdae(y_true, y_pred)

        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# relative_error
# ---------------------------------------------------------------------------

class TestRelativeError:
    def test_shape_preserved(self) -> None:
        """Arrange: 2-D arrays. Act/Assert: output shape equals input shape."""
        y_true = np.ones((3, 4))
        y_pred = np.ones((3, 4)) * 2.0

        result = relative_error(y_true, y_pred)

        assert result.shape == (3, 4)

    def test_values(self) -> None:
        """Arrange: y_true=2, y_pred=3. Act/Assert: rel_err=(3-2)/2=0.5."""
        y_true = np.array([2.0])
        y_pred = np.array([3.0])

        result = relative_error(y_true, y_pred)

        assert result[0] == pytest.approx(0.5)

    def test_safe_denominator(self) -> None:
        """Arrange: y_true=0. Act/Assert: no NaN, denominator = eps."""
        y_true = np.array([0.0])
        y_pred = np.array([1.0])

        result = relative_error(y_true, y_pred, eps=1e-10)

        assert np.isfinite(result[0])
        assert result[0] > 0


# ---------------------------------------------------------------------------
# median_relative_error
# ---------------------------------------------------------------------------

class TestMedianRelativeError:
    def test_perfect_prediction_zero(self) -> None:
        """Arrange: y_pred == y_true. Act/Assert: 0.0."""
        y = np.array([1.0, 2.0, 3.0])

        result = median_relative_error(y, y)

        assert result == pytest.approx(0.0)

    def test_known_value(self) -> None:
        """Arrange: y_true=[1,2,4], y_pred=[1,3,4]. Act/Assert: median(0,0.5,0)=0.0."""
        # relative errors: |(1-1)/1|=0, |(3-2)/2|=0.5, |(4-4)/4|=0
        # median = 0.0
        y_true = np.array([1.0, 2.0, 4.0])
        y_pred = np.array([1.0, 3.0, 4.0])

        result = median_relative_error(y_true, y_pred)

        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# per_species_metrics
# ---------------------------------------------------------------------------

class TestPerSpeciesMetrics:
    def test_returns_dataframe(self) -> None:
        """Arrange: 2 species, 10 rows. Act/Assert: DataFrame with correct shape."""
        y_true = np.random.default_rng(0).standard_normal((10, 2))
        y_pred = np.random.default_rng(1).standard_normal((10, 2))

        result = per_species_metrics(y_true, y_pred, ["A", "B"])

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 4)

    def test_index_is_species(self) -> None:
        """Arrange: species = ['X', 'Y']. Act/Assert: index equals species list."""
        y_true = np.ones((5, 2))
        y_pred = np.ones((5, 2))

        result = per_species_metrics(y_true, y_pred, ["X", "Y"])

        assert list(result.index) == ["X", "Y"]

    def test_columns_correct(self) -> None:
        """Arrange: any arrays. Act/Assert: columns = R2, NRMSE, NMdAE, MedianRelErr."""
        y_true = np.random.default_rng(42).standard_normal((5, 3))
        y_pred = np.random.default_rng(43).standard_normal((5, 3))

        result = per_species_metrics(y_true, y_pred, ["A", "B", "C"])

        assert set(result.columns) == {"R2", "NRMSE", "NMdAE", "MedianRelErr"}

    def test_perfect_prediction_r2_one(self) -> None:
        """Arrange: y_pred == y_true. Act/Assert: all R2 == 1."""
        y = np.random.default_rng(5).standard_normal((8, 3))

        result = per_species_metrics(y, y, ["A", "B", "C"])

        assert all(v == pytest.approx(1.0) for v in result["R2"].values)

    def test_mismatch_species_raises(self) -> None:
        """Arrange: 3-column arrays but 2 species. Act/Assert: ValueError raised."""
        y_true = np.ones((4, 3))
        y_pred = np.ones((4, 3))

        with pytest.raises(ValueError, match="species list length"):
            per_species_metrics(y_true, y_pred, ["A", "B"])

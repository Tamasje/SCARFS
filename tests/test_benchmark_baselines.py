"""Tests for scarfs.benchmark.baselines — FrozenComposition, MeanRate, NearestNeighborRates.

Each test follows arrange / act / assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.benchmark.baselines import (
    FrozenComposition,
    MeanRate,
    NearestNeighborRates,
)
from scarfs.models.common import SurrogatePrediction
from scarfs.schema import Schema


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_schema() -> Schema:
    """Return a minimal Schema with species A, B and state T, P, z, rho."""
    return Schema.from_columns(
        [
            "Y_A", "Y_B",
            "R_A", "R_B",
            "T [K]", "P [Pa]",
            "z [m]", "rho [kg/m3]",
            "Mass flow [kg/s]",
            "mdot [kg/s]",
            "U_in [m/s]",
            "CaseID",
        ]
    )


def _make_df(n: int, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic DataFrame with n rows matching the minimal schema."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Y_A": rng.uniform(0.3, 0.7, n),
            "Y_B": rng.uniform(0.3, 0.7, n),
            "R_A": rng.standard_normal(n),
            "R_B": rng.standard_normal(n),
            "T [K]": rng.uniform(800, 1000, n),
            "P [Pa]": np.full(n, 2e5),
            "z [m]": np.linspace(0, 8, n),
            "rho [kg/m3]": np.full(n, 0.73),
            "Mass flow [kg/s]": np.full(n, 1.6),
            "mdot [kg/s]": np.full(n, 1.6),
            "U_in [m/s]": np.full(n, 11.0),
            "CaseID": np.ones(n, dtype=int),
        }
    )


ACTIVE = ("A", "B")


# ---------------------------------------------------------------------------
# FrozenComposition
# ---------------------------------------------------------------------------

class TestFrozenComposition:
    def test_returns_surrogate_prediction(self) -> None:
        """Arrange: n=5 rows. Act: predict. Assert: SurrogatePrediction returned."""
        df = _make_df(5)
        baseline = FrozenComposition(active_species=ACTIVE)

        result = baseline.predict(df)

        assert isinstance(result, SurrogatePrediction)

    def test_rates_are_zero(self) -> None:
        """Arrange: n=8 rows. Act: predict. Assert: all rates == 0."""
        df = _make_df(8)
        baseline = FrozenComposition(active_species=ACTIVE)

        result = baseline.predict(df)

        assert np.all(result.rates == 0.0)

    def test_rates_shape(self) -> None:
        """Arrange: n=10 rows, 2 species. Act: predict. Assert: rates shape (10, 2)."""
        df = _make_df(10)
        baseline = FrozenComposition(active_species=ACTIVE)

        result = baseline.predict(df)

        assert result.rates.shape == (10, 2)

    def test_energy_shape(self) -> None:
        """Arrange: n=6 rows. Act: predict. Assert: energy shape (6,)."""
        df = _make_df(6)
        baseline = FrozenComposition(active_species=ACTIVE)

        result = baseline.predict(df)

        assert result.energy.shape == (6,)

    def test_species_matches(self) -> None:
        """Arrange: active_species=('A','B'). Act: predict. Assert: result.species same."""
        df = _make_df(3)
        baseline = FrozenComposition(active_species=ACTIVE)

        result = baseline.predict(df)

        assert result.species == ACTIVE


# ---------------------------------------------------------------------------
# MeanRate
# ---------------------------------------------------------------------------

class TestMeanRate:
    def test_predict_before_fit_raises(self) -> None:
        """Arrange: unfitted MeanRate. Act/Assert: RuntimeError raised."""
        baseline = MeanRate(active_species=ACTIVE)
        df = _make_df(5)

        with pytest.raises(RuntimeError, match="fit"):
            baseline.predict(df)

    def test_fit_returns_self(self) -> None:
        """Arrange: MeanRate. Act: fit. Assert: returns same object (chainable)."""
        schema = _make_schema()
        df = _make_df(10)
        baseline = MeanRate(active_species=ACTIVE)

        result = baseline.fit(df, schema)

        assert result is baseline

    def test_predict_constant_rows(self) -> None:
        """Arrange: fit on 10 rows, predict 5 rows. Assert: all rows identical."""
        schema = _make_schema()
        train_df = _make_df(10, seed=0)
        test_df = _make_df(5, seed=1)
        baseline = MeanRate(active_species=ACTIVE).fit(train_df, schema)

        result = baseline.predict(test_df)

        # Every predicted row should be identical (the training mean)
        assert np.allclose(result.rates, result.rates[0])

    def test_predict_rates_equal_training_mean(self) -> None:
        """Arrange: known training rates. Act: fit & predict. Assert: output == mean."""
        schema = _make_schema()
        train_df = _make_df(4, seed=7)
        baseline = MeanRate(active_species=ACTIVE).fit(train_df, schema)

        expected_mean = train_df[["R_A", "R_B"]].to_numpy().mean(axis=0)
        result = baseline.predict(train_df)

        assert np.allclose(result.rates[0], expected_mean)

    def test_output_shape(self) -> None:
        """Arrange: fit 10 rows, predict 3 rows. Assert: rates shape (3, 2)."""
        schema = _make_schema()
        baseline = MeanRate(active_species=ACTIVE).fit(_make_df(10), schema)

        result = baseline.predict(_make_df(3))

        assert result.rates.shape == (3, 2)


# ---------------------------------------------------------------------------
# NearestNeighborRates
# ---------------------------------------------------------------------------

class TestNearestNeighborRates:
    def test_predict_before_fit_raises(self) -> None:
        """Arrange: unfitted NN. Act/Assert: RuntimeError raised."""
        baseline = NearestNeighborRates(active_species=ACTIVE)
        df = _make_df(3)

        with pytest.raises(RuntimeError, match="fit"):
            baseline.predict(df)

    def test_fit_returns_self(self) -> None:
        """Arrange: NN. Act: fit. Assert: returns same object (chainable)."""
        schema = _make_schema()
        df = _make_df(10)
        baseline = NearestNeighborRates(active_species=ACTIVE)

        result = baseline.fit(df, schema)

        assert result is baseline

    def test_predict_on_training_rows_returns_training_rates(self) -> None:
        """Arrange: fit on training, predict exact same rows. Assert: rates match training.

        Each query row is itself in the training set, so the nearest neighbour
        should return the row's own rates (assuming all rows are unique).
        """
        schema = _make_schema()
        # Use a small, well-separated training set to guarantee uniqueness
        train_df = pd.DataFrame(
            {
                "Y_A": [0.9, 0.1],
                "Y_B": [0.1, 0.9],
                "R_A": [5.0, -5.0],
                "R_B": [-5.0, 5.0],
                "T [K]": [800.0, 1000.0],
                "P [Pa]": [2e5, 2e5],
                "z [m]": [0.0, 1.0],
                "rho [kg/m3]": [0.73, 0.73],
                "Mass flow [kg/s]": [1.6, 1.6],
                "mdot [kg/s]": [1.6, 1.6],
                "U_in [m/s]": [11.0, 11.0],
                "CaseID": [1, 1],
            }
        )
        baseline = NearestNeighborRates(active_species=ACTIVE).fit(train_df, schema)

        result = baseline.predict(train_df)

        assert np.allclose(result.rates, train_df[["R_A", "R_B"]].to_numpy())

    def test_output_shape(self) -> None:
        """Arrange: fit 20, predict 5. Assert: rates shape (5, 2)."""
        schema = _make_schema()
        baseline = NearestNeighborRates(active_species=ACTIVE).fit(_make_df(20), schema)

        result = baseline.predict(_make_df(5))

        assert result.rates.shape == (5, 2)

    def test_rates_from_training_set(self) -> None:
        """Arrange: fit on training, predict on new points.
        Assert: each predicted row is one of the training rows (look-up, not interpolation)."""
        schema = _make_schema()
        train_df = _make_df(15, seed=3)
        test_df = _make_df(5, seed=99)
        baseline = NearestNeighborRates(active_species=ACTIVE).fit(train_df, schema)

        result = baseline.predict(test_df)

        train_rates = train_df[["R_A", "R_B"]].to_numpy()
        for i in range(len(test_df)):
            # The predicted row must match one of the training rows exactly
            matches = np.any(
                np.all(np.isclose(result.rates[i], train_rates), axis=1)
            )
            assert matches, f"Row {i}: predicted rate not found in training set"

    def test_surrogate_protocol_satisfied(self) -> None:
        """Arrange: NearestNeighborRates. Act: isinstance check. Assert: Surrogate protocol."""
        from scarfs.models.common import Surrogate
        schema = _make_schema()
        baseline = NearestNeighborRates(active_species=ACTIVE).fit(_make_df(10), schema)

        assert isinstance(baseline, Surrogate)

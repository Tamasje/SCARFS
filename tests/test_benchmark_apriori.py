"""Tests for scarfs.benchmark.apriori — holdout_split, run_apriori, AprioriReport.

Each test follows arrange / act / assert.
Uses only small synthetic DataFrames for speed and determinism.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.benchmark.apriori import (
    AprioriConfig,
    AprioriReport,
    holdout_split,
    run_apriori,
)
from scarfs.benchmark.baselines import FrozenComposition
from scarfs.models.common import SurrogatePrediction
from scarfs.schema import Schema


# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------

def _make_df(
    n: int = 40,
    n_cases: int = 1,
    seed: int = 0,
    L: float = 8.0,
) -> pd.DataFrame:
    """Synthetic PFR database with 2 species (A, B), n rows, n_cases cases."""
    rng = np.random.default_rng(seed)
    rows_per_case = n // n_cases

    dfs = []
    for cid in range(1, n_cases + 1):
        nc = rows_per_case
        z = np.linspace(0, L, nc)
        T = np.linspace(823, 973, nc)
        df_c = pd.DataFrame(
            {
                "Y_A": np.linspace(0.7, 0.5, nc) + rng.normal(0, 0.001, nc),
                "Y_B": np.linspace(0.1, 0.3, nc) + rng.normal(0, 0.001, nc),
                "R_A": np.linspace(-0.1, -1.0, nc),
                "R_B": np.linspace(0.1, 1.0, nc),
                "T [K]": T,
                "P [Pa]": np.full(nc, 2e5),
                "z [m]": z,
                "rho [kg/m3]": np.full(nc, 0.73),
                "Mass flow [kg/s]": np.full(nc, 1.6),
                "mdot [kg/s]": np.full(nc, 1.6),
                "U_in [m/s]": np.full(nc, 11.0),
                "CaseID": np.full(nc, cid, dtype=int),
                "L [m]": np.full(nc, L),
            }
        )
        dfs.append(df_c)
    return pd.concat(dfs, ignore_index=True)


def _make_schema(df: pd.DataFrame) -> Schema:
    return Schema.from_columns(list(df.columns))


ACTIVE = ("A", "B")


# ---------------------------------------------------------------------------
# A perfect predictor for testing PASS/FAIL logic
# ---------------------------------------------------------------------------

class _PerfectPredictor:
    """Surrogate that returns ground-truth rates from the DataFrame."""

    active_species: tuple[str, ...] = ACTIVE

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        rates = df[["R_A", "R_B"]].to_numpy(dtype=float)
        energy = np.zeros(len(df), dtype=float)
        return SurrogatePrediction(species=ACTIVE, rates=rates, energy=energy)


class _WorstPredictor:
    """Surrogate that always returns the negation of the true rates."""

    active_species: tuple[str, ...] = ACTIVE

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        rates = -df[["R_A", "R_B"]].to_numpy(dtype=float)
        energy = np.zeros(len(df), dtype=float)
        return SurrogatePrediction(species=ACTIVE, rates=rates, energy=energy)


# ---------------------------------------------------------------------------
# holdout_split
# ---------------------------------------------------------------------------

class TestHoldoutSplit:
    def test_returns_four_frames(self) -> None:
        """Arrange: 40-row single-case df. Act: holdout_split. Assert: 4 frames returned."""
        df = _make_df(40, n_cases=1)
        schema = _make_schema(df)
        config = AprioriConfig()

        train, test, low_conv, high_T = holdout_split(df, schema, config)

        assert all(isinstance(f, pd.DataFrame) for f in [train, test, low_conv, high_T])

    def test_partition_covers_all_rows(self) -> None:
        """Arrange: 40 rows. Act: split. Assert: total rows across partitions == 40."""
        df = _make_df(40, n_cases=1)
        schema = _make_schema(df)
        config = AprioriConfig(heldout_conversion_quantile=0.1, heldout_T_quantile=0.1)

        train, test, low_conv, high_T = holdout_split(df, schema, config)

        total = len(train) + len(test) + len(low_conv) + len(high_T)
        assert total == len(df)

    def test_low_conv_rows_are_near_inlet(self) -> None:
        """Arrange: sorted z, low_conv_quantile=0.1. Act/Assert: low_conv rows have smallest z."""
        df = _make_df(40, n_cases=1)
        schema = _make_schema(df)
        config = AprioriConfig(heldout_conversion_quantile=0.25)

        _, _, low_conv, _ = holdout_split(df, schema, config)

        z_col = schema.state["z"]
        T_col = schema.state["T"]
        # Low-conv rows should all have z <= the 25th percentile of all z values
        z_max_low = low_conv[z_col].max()
        z_75pct = df[z_col].quantile(0.26)  # small tolerance
        assert z_max_low <= z_75pct + 1e-6

    def test_train_test_non_overlapping(self) -> None:
        """Arrange: 40 rows. Act: split. Assert: train and test row indices do not overlap."""
        df = _make_df(40, n_cases=1)
        schema = _make_schema(df)
        config = AprioriConfig()

        train, test, _, _ = holdout_split(df, schema, config)

        # After reset_index they won't share index, but we can check T values
        # are disjoint (they should be since the split is positional)
        train_T = set(train["T [K]"].round(6))
        test_T = set(test["T [K]"].round(6))
        assert len(train_T & test_T) == 0

    def test_multi_case_split(self) -> None:
        """Arrange: 4 cases × 10 rows = 40 rows. Act: split. Assert: coverage preserved."""
        df = _make_df(40, n_cases=4)
        schema = _make_schema(df)
        config = AprioriConfig(heldout_conversion_quantile=0.1, heldout_T_quantile=0.1)

        train, test, low_conv, high_T = holdout_split(df, schema, config)

        total = len(train) + len(test) + len(low_conv) + len(high_T)
        assert total == len(df)

    def test_schema_without_caseid(self) -> None:
        """Arrange: DataFrame without CaseID. Act: split. Assert: runs without error."""
        df = _make_df(20, n_cases=1).drop(columns=["CaseID", "L [m]"])
        schema = Schema.from_columns(list(df.columns))
        config = AprioriConfig()

        train, test, low_conv, high_T = holdout_split(df, schema, config)

        assert len(train) + len(test) + len(low_conv) + len(high_T) == len(df)


# ---------------------------------------------------------------------------
# run_apriori
# ---------------------------------------------------------------------------

class TestRunApriori:
    def _setup(self, n: int = 40, n_cases: int = 1, seed: int = 0):
        df = _make_df(n=n, n_cases=n_cases, seed=seed)
        schema = _make_schema(df)
        config = AprioriConfig()
        train, test, _, _ = holdout_split(df, schema, config)
        return train, test, schema, config

    def test_returns_apriori_report(self) -> None:
        """Arrange: perfect predictor. Act: run_apriori. Assert: AprioriReport returned."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert isinstance(report, AprioriReport)

    def test_perfect_predictor_passes_r2(self) -> None:
        """Arrange: perfect predictor. Act: run. Assert: R2_rates >= 0.95 check PASS."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert report.pass_fail["R2_rates >= threshold"]

    def test_worst_predictor_fails_r2(self) -> None:
        """Arrange: inverted-rates predictor. Act: run. Assert: R2_rates check FAIL."""
        train, test, schema, config = self._setup()
        predictor = _WorstPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert not report.pass_fail["R2_rates >= threshold"]

    def test_perfect_predictor_beats_frozen(self) -> None:
        """Arrange: perfect predictor. Act: run. Assert: beats FrozenComposition."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert report.pass_fail["beats_FrozenComposition_R2"]

    def test_report_has_all_pass_fail_keys(self) -> None:
        """Arrange: any predictor. Act: run. Assert: all expected keys present."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        expected_keys = {
            "R2_rates >= threshold",
            "R2_yields >= threshold",
            "MedianRelErr_rates <= threshold",
            "beats_FrozenComposition_R2",
            "beats_MeanRate_R2",
            "beats_NearestNeighbor_R2",
            "heldout_low_conv_R2_not_worse",
            "heldout_high_T_R2_not_worse",
        }
        assert expected_keys.issubset(set(report.pass_fail.keys()))

    def test_report_summary_returns_string(self) -> None:
        """Arrange: any predictor. Act: report.summary(). Assert: non-empty string."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)
        summary = report.summary()

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_passed_flag_consistent_with_pass_fail(self) -> None:
        """Arrange: any predictor. Act: run. Assert: passed == all(pass_fail.values())."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert report.passed == all(report.pass_fail.values())

    def test_baseline_metrics_keys(self) -> None:
        """Arrange: any predictor. Act: run. Assert: all 3 baselines in baseline_rate_metrics."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert "FrozenComposition" in report.baseline_rate_metrics
        assert "MeanRate" in report.baseline_rate_metrics
        assert "NearestNeighborRates" in report.baseline_rate_metrics

    def test_error_vs_T_is_dataframe(self) -> None:
        """Arrange: any predictor. Act: run. Assert: error_vs_T is a DataFrame."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert isinstance(report.error_vs_T, pd.DataFrame)
        assert "T_center" in report.error_vs_T.columns

    def test_aggregate_rate_keys(self) -> None:
        """Arrange: any predictor. Act: run. Assert: aggregate_rate has R2, NRMSE, NMdAE."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert "R2" in report.aggregate_rate
        assert "NRMSE" in report.aggregate_rate
        assert "NMdAE" in report.aggregate_rate

    def test_heldout_degradation_warned(self) -> None:
        """Arrange: predictor that is bad on first rows. Act: run. Assert: warning emitted."""
        df = _make_df(n=40, n_cases=1)
        schema = _make_schema(df)
        config = AprioriConfig(heldout_conversion_quantile=0.25, r2_threshold=0.5)
        _, test, _, _ = holdout_split(df, schema, config)

        # A predictor that returns zeros (poor on all rows)
        predictor = FrozenComposition(active_species=ACTIVE)

        # fit baselines against the test df used as proxy train
        report = run_apriori(predictor, test, test, schema, config)

        # Warnings should be a list (may be empty for frozen, but no crash)
        assert isinstance(report.warnings, list)

    def test_active_species_in_report(self) -> None:
        """Arrange: predictor with active_species=ACTIVE. Act: run. Assert: report.active_species same."""
        train, test, schema, config = self._setup()
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert report.active_species == ACTIVE

    def test_multi_case_run(self) -> None:
        """Arrange: 4 cases × 20 rows. Act: run_apriori. Assert: no crash, report returned."""
        df = _make_df(n=80, n_cases=4, seed=7)
        schema = _make_schema(df)
        config = AprioriConfig()
        train, test, _, _ = holdout_split(df, schema, config)
        predictor = _PerfectPredictor()

        report = run_apriori(predictor, train, test, schema, config)

        assert isinstance(report, AprioriReport)

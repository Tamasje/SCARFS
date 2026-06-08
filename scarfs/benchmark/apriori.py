"""A-priori (offline) evaluation harness for the SCARFS benchmark.

The a-priori harness evaluates a surrogate's quality **without** running the
CFD solver.  It compares predictions against ground-truth rates stored in the
database, computes per-species and aggregate metrics, checks against
ChemZIP's published targets, and reports two deliberately difficult
held-out regions to diagnose coverage gaps:

1. **Near-inlet / low-conversion rows** — earliest reactor positions where
   conversion is low and rates are just igniting.  The surrogate must handle
   the near-zero-rate regime without predicting spurious chemistry.
2. **High-temperature rows** — the top ``heldout_T_quantile`` fraction by
   temperature (hottest reactions).  The surrogate must not silently fail when
   extrapolating to extreme conditions seen only rarely in training.

Typical usage::

    config = AprioriConfig(focus_species=("C2H4", "C2H6", "H2", "CH4"))
    report = run_apriori(my_surrogate, train_df, test_df, schema, config)
    print(report.summary())
    assert report.passed

Public API
----------
- :class:`AprioriConfig` — configuration dataclass (thresholds, quantiles,
  focus species).
- :class:`AprioriReport` — results dataclass; ``.passed`` is ``True`` iff all
  PASS/FAIL checks succeed.
- :func:`holdout_split` — case-aware split producing train / test /
  held-out-region frames.
- :func:`run_apriori` — main entry point.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from scarfs.models.common import Surrogate, SurrogatePrediction
from scarfs.schema import Schema
from scarfs.benchmark.metrics import (
    per_species_metrics,
    r2_score,
    nrmse,
    nmdae,
    median_relative_error,
)
from scarfs.benchmark.yields import integrate_yields
from scarfs.benchmark.baselines import FrozenComposition, MeanRate, NearestNeighborRates


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AprioriConfig:
    """Configuration for :func:`run_apriori` and :func:`holdout_split`.

    Attributes
    ----------
    r2_threshold
        Minimum R² required across *all* active-species rates and yields.
        ChemZIP target: 0.95.
    median_rel_err_threshold
        Maximum allowed median relative error.  ChemZIP target: 0.10.
    heldout_conversion_quantile
        Fraction of reactor length (from inlet) to withhold as the
        "near-inlet / low-conversion" region.  Default: first 10 % of each
        case's axial points.
    heldout_T_quantile
        Fraction of rows by *descending* temperature to withhold as the
        "high-temperature" region.  Default: top 10 % by temperature.
    focus_species
        Species to highlight in per-species tables and PASS/FAIL checks.
        If empty, all active species are used.
    random_seed
        Seed for any random operations (currently not used; reserved for
        future train/test random splits).
    """

    r2_threshold: float = 0.95
    median_rel_err_threshold: float = 0.10
    heldout_conversion_quantile: float = 0.10
    heldout_T_quantile: float = 0.10
    focus_species: Tuple[str, ...] = ()
    random_seed: int = 42


# ---------------------------------------------------------------------------
# AprioriReport
# ---------------------------------------------------------------------------

@dataclass
class AprioriReport:
    """Results of one a-priori evaluation run.

    Attributes
    ----------
    predictor_rate_metrics
        Per-species DataFrame (R², NRMSE, NMdAE, MedianRelErr) on the full
        test set **rates**.
    predictor_yield_metrics
        Per-species DataFrame on the integrated-yield error.
    baseline_rate_metrics
        Dict mapping baseline name to per-species rate metric DataFrame.
    baseline_yield_metrics
        Dict mapping baseline name to per-species yield metric DataFrame.
    aggregate_rate
        Dict with scalar aggregate metrics (``R2``, ``NRMSE``, ``NMdAE``,
        ``MedianRelErr``) on rates, averaged across active species.
    aggregate_yield
        Same but for integrated yields.
    error_vs_T
        DataFrame of median absolute rate error binned by temperature.
    error_vs_tau
        DataFrame of median absolute rate error binned by cumulative
        residence-time proxy τ.
    heldout_low_conv_metrics
        Per-species rate metrics on the near-inlet held-out region.
    heldout_high_T_metrics
        Per-species rate metrics on the high-temperature held-out region.
    pass_fail
        Dict of individual check name → bool.
    passed
        ``True`` iff all checks in :attr:`pass_fail` are ``True``.
    config
        The :class:`AprioriConfig` used for this run.
    active_species
        The species evaluated.
    warnings
        List of warning strings emitted during evaluation.
    """

    predictor_rate_metrics: pd.DataFrame
    predictor_yield_metrics: pd.DataFrame
    baseline_rate_metrics: dict[str, pd.DataFrame]
    baseline_yield_metrics: dict[str, pd.DataFrame]
    aggregate_rate: dict[str, float]
    aggregate_yield: dict[str, float]
    error_vs_T: pd.DataFrame
    error_vs_tau: pd.DataFrame
    heldout_low_conv_metrics: pd.DataFrame
    heldout_high_T_metrics: pd.DataFrame
    pass_fail: dict[str, bool]
    passed: bool
    config: AprioriConfig
    active_species: tuple[str, ...]
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string.

        Suitable for logging or printing to the console.  Does *not* depend on
        any plotting library.
        """
        lines = [
            "=" * 70,
            "A-PRIORI BENCHMARK REPORT",
            "=" * 70,
            f"Active species evaluated : {len(self.active_species)}",
            f"Overall PASS             : {self.passed}",
            "",
            "--- Aggregate Rate Metrics ---",
        ]
        for k, v in self.aggregate_rate.items():
            lines.append(f"  {k:20s}: {v:.4f}")
        lines += ["", "--- Aggregate Yield Metrics ---"]
        for k, v in self.aggregate_yield.items():
            lines.append(f"  {k:20s}: {v:.4f}")
        lines += ["", "--- PASS / FAIL Checks ---"]
        for check, result in self.pass_fail.items():
            status = "PASS" if result else "FAIL"
            lines.append(f"  [{status}] {check}")
        if self.warnings:
            lines += ["", "--- Warnings ---"]
            for w in self.warnings:
                lines.append(f"  ! {w}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# holdout_split
# ---------------------------------------------------------------------------

def holdout_split(
    df: pd.DataFrame,
    schema: Schema,
    config: AprioriConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a database DataFrame into train / test / heldout-low-conv / heldout-high-T.

    The split is **case-aware**: the near-inlet and high-temperature held-out
    regions are identified per case (so each case contributes rows to each
    partition rather than entire cases being withheld).

    Partitioning logic
    ------------------
    1. **Near-inlet / low-conversion region** — for each case, the first
       ``heldout_conversion_quantile`` fraction of axial rows (sorted by ``z``)
       are withheld as ``heldout_low_conv``.
    2. **High-temperature region** — rows in the top ``heldout_T_quantile``
       of the temperature distribution *after* excluding low-conv rows are
       withheld as ``heldout_high_T``.
    3. The remaining rows form the union of **train** and **test**, split 80/20
       by case (whole cases are assigned to train or test to avoid data
       leakage between rows of the same case).  When there is only one case,
       the split is done by row index (first 80 % → train, last 20 % → test).

    Parameters
    ----------
    df
        Full database DataFrame.
    schema
        Schema describing *df*.
    config
        :class:`AprioriConfig` controlling quantile thresholds.

    Returns
    -------
    train_df, test_df, heldout_low_conv_df, heldout_high_T_df
        Four non-overlapping DataFrames whose union is the full input *df*.
    """
    df = df.copy()

    if "CaseID" not in schema.meta:
        # No CaseID column: treat entire frame as a single case
        df["_CaseID"] = 0
        case_col = "_CaseID"
    else:
        case_col = schema.meta["CaseID"]

    z_col = schema.state["z"]
    T_col = schema.state["T"]

    low_conv_idx: list[int] = []
    remaining_idx: list[int] = []

    for _case_id, grp_idx in df.groupby(case_col).groups.items():
        grp = df.loc[grp_idx].sort_values(z_col)
        n = len(grp)
        cutoff = max(1, int(np.floor(config.heldout_conversion_quantile * n)))
        low_conv_idx.extend(grp.index[:cutoff].tolist())
        remaining_idx.extend(grp.index[cutoff:].tolist())

    low_conv_df = df.loc[low_conv_idx]
    remaining = df.loc[remaining_idx]

    # High-T from the remaining pool (not the low-conv rows)
    T_threshold = remaining[T_col].quantile(1.0 - config.heldout_T_quantile)
    high_T_mask = remaining[T_col] >= T_threshold
    high_T_df = remaining[high_T_mask]
    core = remaining[~high_T_mask]

    # Train / test split on the core (case-aware when >1 case)
    cases_in_core = core[case_col].unique()
    rng = np.random.default_rng(config.random_seed)

    if len(cases_in_core) > 1:
        rng.shuffle(cases_in_core)
        n_train = max(1, int(np.floor(0.8 * len(cases_in_core))))
        train_cases = set(cases_in_core[:n_train])
        train_df = core[core[case_col].isin(train_cases)]
        test_df = core[~core[case_col].isin(train_cases)]
    else:
        # Single case: row-wise 80/20 split
        n_train = max(1, int(np.floor(0.8 * len(core))))
        train_df = core.iloc[:n_train]
        test_df = core.iloc[n_train:]

    # Drop the synthetic case column if we added it
    for frame in [train_df, test_df, low_conv_df, high_T_df]:
        if "_CaseID" in frame.columns:
            frame.drop(columns=["_CaseID"], inplace=True)

    return (
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        low_conv_df.reset_index(drop=True),
        high_T_df.reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _predict_rates_array(
    predictor: Surrogate,
    df: pd.DataFrame,
) -> np.ndarray:
    """Call predictor.predict and return the rates array (n, k)."""
    pred: SurrogatePrediction = predictor.predict(df)
    return np.asarray(pred.rates, dtype=float)


def _true_rates_array(
    df: pd.DataFrame,
    schema: Schema,
    active_species: tuple[str, ...],
) -> np.ndarray:
    """Extract ground-truth rates from *df* for *active_species*."""
    r_cols = schema.r_columns(active_species)
    return df[r_cols].to_numpy(dtype=float)


def _aggregate_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Compute mean of R2, NRMSE, NMdAE, MedianRelErr across species."""
    return {col: float(metrics_df[col].mean()) for col in metrics_df.columns}


def _compute_integrated_yields(
    df: pd.DataFrame,
    rates: np.ndarray,
    schema: Schema,
    active_species: tuple[str, ...],
) -> np.ndarray:
    """Integrate rates per case and stack into (n_cases, k)."""
    if "CaseID" not in schema.meta:
        return integrate_yields(df, rates, schema).reshape(1, -1)

    case_col = schema.meta["CaseID"]
    results = []
    for _cid, idx in df.groupby(case_col).groups.items():
        df_case = df.loc[idx].reset_index(drop=True)
        r_case = rates[idx]
        dY = integrate_yields(df_case, r_case, schema)
        results.append(dY)
    return np.vstack(results)  # (n_cases, k)


def _error_vs_bin(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_col_values: np.ndarray,
    n_bins: int = 10,
    bin_label: str = "bin_center",
) -> pd.DataFrame:
    """Bin rows by *bin_col_values* and report median absolute rate error per bin.

    Parameters
    ----------
    df
        Input DataFrame (only used for its length).
    y_true, y_pred
        ``(n, k)`` arrays of true and predicted rates.
    bin_col_values
        ``(n,)`` values to bin along (e.g. temperature or tau).
    n_bins
        Number of equal-quantile bins.
    bin_label
        Name for the bin-center column in the output.

    Returns
    -------
    pandas.DataFrame
        Columns: ``[bin_label, "median_abs_err"]``.
    """
    errors = np.abs(y_true - y_pred).mean(axis=1)  # per-row MAE across species
    bins = pd.qcut(bin_col_values, q=n_bins, duplicates="drop")
    result = pd.DataFrame(
        {"bin": bins, "median_abs_err": errors}
    ).groupby("bin")["median_abs_err"].median().reset_index()
    result[bin_label] = result["bin"].apply(lambda iv: iv.mid)
    result = result[[bin_label, "median_abs_err"]]
    return result


def _tau_from_df(df: pd.DataFrame, schema: Schema) -> np.ndarray:
    """Compute cumulative residence-time proxy τ = ∫ dz / u [s].

    τ is not stored in the database and is not a model feature, but it
    localises coverage gaps on a physical time axis.  We approximate:

        u(z) = mdot / (rho(z) * A)    where A = mdot / (rho_in * U_in)
        τ(z) = ∫₀ᶻ 1/u dz   (trapezoidal)

    The computation is per-case; rows from different cases are stacked in
    order.

    Parameters
    ----------
    df
        Full (potentially multi-case) DataFrame.
    schema
        Schema describing *df*.

    Returns
    -------
    numpy.ndarray
        ``(n,)`` cumulative residence-time proxy [s] for each row.
    """
    rho_col = schema.state["rho"]
    z_col = schema.state["z"]
    if "mass_flow" in schema.state:
        mdot_col = schema.state["mass_flow"]
    else:
        mdot_col = schema.meta["mdot"]
    u_in_col = schema.meta["U_in"]

    tau_all = np.empty(len(df), dtype=float)

    if "CaseID" in schema.meta:
        case_col = schema.meta["CaseID"]
        groups = df.groupby(case_col)
    else:
        groups = {0: df}.items()  # type: ignore[assignment]

    for _cid, grp_or_idx in groups:
        if hasattr(grp_or_idx, "index"):
            idx = grp_or_idx.index
            grp = grp_or_idx
        else:
            idx = grp_or_idx
            grp = df.loc[idx]

        grp = grp.sort_values(z_col)
        idx_sorted = grp.index

        z = grp[z_col].to_numpy(dtype=float)
        rho = grp[rho_col].to_numpy(dtype=float)
        mdot = float(grp[mdot_col].iloc[0])
        rho_in = float(rho[0])
        u_in = float(grp[u_in_col].iloc[0])

        A = mdot / (rho_in * u_in)
        u = mdot / (rho * A)
        inv_u = 1.0 / u

        # Cumulative trapezoidal integration
        tau = np.zeros(len(z))
        for j in range(1, len(z)):
            dz = z[j] - z[j - 1]
            tau[j] = tau[j - 1] + 0.5 * (inv_u[j] + inv_u[j - 1]) * dz

        tau_all[idx_sorted] = tau

    return tau_all


# ---------------------------------------------------------------------------
# run_apriori
# ---------------------------------------------------------------------------

def run_apriori(
    predictor: Surrogate,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    config: AprioriConfig,
) -> AprioriReport:
    """Run the full a-priori offline benchmark.

    Steps
    -----
    1. Determine the active species from *predictor.active_species*.
    2. Extract ground-truth rates from *test_df*.
    3. Call *predictor.predict* on *test_df*.
    4. Fit and predict all three baselines on *test_df*.
    5. Compute per-species + aggregate metrics (rates and integrated yields).
    6. Compute error-vs-T and error-vs-τ diagnostic tables.
    7. Evaluate predictor on the held-out low-conversion and high-T sub-frames
       (derived from *test_df* using the configured quantiles).
    8. Run PASS/FAIL checks (described below).

    PASS/FAIL Logic
    ---------------
    All of the following must hold for ``report.passed == True``:

    - **R2_rates** : mean R² across active species on test rates ≥
      ``config.r2_threshold`` (default 0.95).
    - **R2_yields** : mean R² across active species on integrated yields ≥
      ``config.r2_threshold``.
    - **MedianRelErr_rates** : mean median relative error on rates ≤
      ``config.median_rel_err_threshold`` (default 0.10).
    - **beats_FrozenComposition_R2** : mean R² of predictor > mean R² of
      FrozenComposition baseline on rates.
    - **beats_MeanRate_R2** : mean R² of predictor > mean R² of MeanRate
      baseline on rates.
    - **beats_NearestNeighbor_R2** : mean R² of predictor > mean R² of
      NearestNeighborRates baseline on rates.
    - **heldout_low_conv_R2_not_worse** : mean R² on low-conversion held-out
      region ≥ ``config.r2_threshold * 0.90`` (10 % grace degradation allowed).
    - **heldout_high_T_R2_not_worse** : same for high-T held-out region.

    All degradations are reported explicitly in :attr:`AprioriReport.pass_fail`
    and :attr:`AprioriReport.summary` — **no silent capping**.

    Parameters
    ----------
    predictor
        Any object satisfying the :class:`~scarfs.models.common.Surrogate`
        protocol.
    train_df
        Training rows (used to fit baselines; *predictor* is assumed already
        trained).
    test_df
        Evaluation rows (predictor and baselines are evaluated here).
    schema
        Schema for both *train_df* and *test_df* (same columns assumed).
    config
        :class:`AprioriConfig` controlling thresholds and held-out quantiles.

    Returns
    -------
    AprioriReport
        Full results dataclass.
    """
    warn_list: list[str] = []
    active_species: tuple[str, ...] = predictor.active_species

    # Determine which species to focus on
    focus = (
        tuple(s for s in config.focus_species if s in active_species)
        if config.focus_species
        else active_species
    )

    # ------------------------------------------------------------------ #
    # 1.  Ground-truth rates on test set
    # ------------------------------------------------------------------ #
    y_true_rates = _true_rates_array(test_df, schema, active_species)

    # ------------------------------------------------------------------ #
    # 2.  Predictor predictions
    # ------------------------------------------------------------------ #
    y_pred_rates = _predict_rates_array(predictor, test_df)

    # ------------------------------------------------------------------ #
    # 3.  Fit & predict baselines
    # ------------------------------------------------------------------ #
    b_frozen = FrozenComposition(active_species=active_species)
    b_mean = MeanRate(active_species=active_species).fit(train_df, schema)
    b_nn = NearestNeighborRates(active_species=active_species).fit(train_df, schema)

    baselines: dict[str, Surrogate] = {
        "FrozenComposition": b_frozen,
        "MeanRate": b_mean,
        "NearestNeighborRates": b_nn,
    }
    baseline_preds: dict[str, np.ndarray] = {
        name: _predict_rates_array(bl, test_df)
        for name, bl in baselines.items()
    }

    # ------------------------------------------------------------------ #
    # 4.  Per-species rate metrics
    # ------------------------------------------------------------------ #
    predictor_rate_metrics = per_species_metrics(y_true_rates, y_pred_rates, list(active_species))
    baseline_rate_metrics: dict[str, pd.DataFrame] = {
        name: per_species_metrics(y_true_rates, bp, list(active_species))
        for name, bp in baseline_preds.items()
    }

    # ------------------------------------------------------------------ #
    # 5.  Integrated-yield metrics (per-case)
    # ------------------------------------------------------------------ #
    # Only compute yields if all required columns are present
    try:
        Y_true_yields = _compute_integrated_yields(test_df, y_true_rates, schema, active_species)
        Y_pred_yields = _compute_integrated_yields(test_df, y_pred_rates, schema, active_species)
        predictor_yield_metrics = per_species_metrics(
            Y_true_yields, Y_pred_yields, list(active_species)
        )
        baseline_yield_metrics: dict[str, pd.DataFrame] = {
            name: per_species_metrics(
                Y_true_yields,
                _compute_integrated_yields(test_df, bp, schema, active_species),
                list(active_species),
            )
            for name, bp in baseline_preds.items()
        }
    except (KeyError, Exception) as exc:
        warn_list.append(f"Yield integration skipped: {exc}")
        _empty = per_species_metrics(
            np.zeros((1, len(active_species))),
            np.zeros((1, len(active_species))),
            list(active_species),
        )
        predictor_yield_metrics = _empty
        baseline_yield_metrics = {name: _empty for name in baselines}

    # ------------------------------------------------------------------ #
    # 6.  Aggregate metrics
    # ------------------------------------------------------------------ #
    agg_rate = _aggregate_metrics(predictor_rate_metrics)
    agg_yield = _aggregate_metrics(predictor_yield_metrics)

    # ------------------------------------------------------------------ #
    # 7.  Error-vs-T and error-vs-tau diagnostic tables
    # ------------------------------------------------------------------ #
    T_col = schema.state["T"]
    T_vals = test_df[T_col].to_numpy(dtype=float)

    error_vs_T = _error_vs_bin(
        test_df, y_true_rates, y_pred_rates, T_vals, n_bins=10, bin_label="T_center"
    )

    try:
        tau_vals = _tau_from_df(test_df, schema)
        error_vs_tau = _error_vs_bin(
            test_df, y_true_rates, y_pred_rates, tau_vals, n_bins=10, bin_label="tau_center"
        )
    except (KeyError, Exception) as exc:
        warn_list.append(f"tau computation skipped: {exc}")
        error_vs_tau = pd.DataFrame({"tau_center": [], "median_abs_err": []})

    # ------------------------------------------------------------------ #
    # 8.  Held-out region diagnostics (from test_df)
    # ------------------------------------------------------------------ #
    # --- low-conversion region (near-inlet) ----------------------------
    z_col = schema.state["z"]
    heldout_low_conv_dfs: list[pd.DataFrame] = []
    heldout_high_T_dfs: list[pd.DataFrame] = []

    if "CaseID" in schema.meta:
        case_col = schema.meta["CaseID"]
        for _cid, idx in test_df.groupby(case_col).groups.items():
            grp = test_df.loc[idx].sort_values(z_col)
            n = len(grp)
            cutoff = max(1, int(np.floor(config.heldout_conversion_quantile * n)))
            heldout_low_conv_dfs.append(grp.iloc[:cutoff])

            remaining = grp.iloc[cutoff:]
            if len(remaining) > 0:
                T_thr = remaining[T_col].quantile(1.0 - config.heldout_T_quantile)
                heldout_high_T_dfs.append(remaining[remaining[T_col] >= T_thr])
    else:
        n = len(test_df)
        cutoff = max(1, int(np.floor(config.heldout_conversion_quantile * n)))
        sorted_df = test_df.sort_values(z_col)
        heldout_low_conv_dfs.append(sorted_df.iloc[:cutoff])
        remaining = sorted_df.iloc[cutoff:]
        if len(remaining) > 0:
            T_thr = remaining[T_col].quantile(1.0 - config.heldout_T_quantile)
            heldout_high_T_dfs.append(remaining[remaining[T_col] >= T_thr])

    if heldout_low_conv_dfs:
        heldout_low_conv = pd.concat(heldout_low_conv_dfs).reset_index(drop=True)
        yt_lc = _true_rates_array(heldout_low_conv, schema, active_species)
        yp_lc = _predict_rates_array(predictor, heldout_low_conv)
        heldout_low_conv_metrics = per_species_metrics(yt_lc, yp_lc, list(active_species))
    else:
        warn_list.append("No low-conversion held-out rows found in test_df.")
        heldout_low_conv_metrics = predictor_rate_metrics.copy()

    if heldout_high_T_dfs:
        heldout_high_T = pd.concat(heldout_high_T_dfs).reset_index(drop=True)
        yt_ht = _true_rates_array(heldout_high_T, schema, active_species)
        yp_ht = _predict_rates_array(predictor, heldout_high_T)
        heldout_high_T_metrics = per_species_metrics(yt_ht, yp_ht, list(active_species))
    else:
        warn_list.append("No high-T held-out rows found in test_df.")
        heldout_high_T_metrics = predictor_rate_metrics.copy()

    # ------------------------------------------------------------------ #
    # 9.  PASS/FAIL checks
    # ------------------------------------------------------------------ #
    r2_rate = agg_rate["R2"]
    r2_yield = agg_yield["R2"]
    med_rel_rate = agg_rate["MedianRelErr"]

    # Baseline R² values
    bl_r2: dict[str, float] = {
        name: float(baseline_rate_metrics[name]["R2"].mean())
        for name in baselines
    }

    # Held-out mean R² values
    r2_lc = float(heldout_low_conv_metrics["R2"].mean())
    r2_ht = float(heldout_high_T_metrics["R2"].mean())
    grace = 0.90  # 10 % degradation grace

    pass_fail: dict[str, bool] = {
        "R2_rates >= threshold": r2_rate >= config.r2_threshold,
        "R2_yields >= threshold": r2_yield >= config.r2_threshold,
        "MedianRelErr_rates <= threshold": med_rel_rate <= config.median_rel_err_threshold,
        "beats_FrozenComposition_R2": r2_rate > bl_r2["FrozenComposition"],
        "beats_MeanRate_R2": r2_rate > bl_r2["MeanRate"],
        "beats_NearestNeighbor_R2": r2_rate > bl_r2["NearestNeighborRates"],
        "heldout_low_conv_R2_not_worse": r2_lc >= config.r2_threshold * grace,
        "heldout_high_T_R2_not_worse": r2_ht >= config.r2_threshold * grace,
    }

    # Explicitly warn about degradations — NO silent capping
    if r2_lc < r2_rate - 0.01:
        warn_list.append(
            f"Low-conversion region degradation: R²={r2_lc:.4f} vs full-test R²={r2_rate:.4f}"
        )
    if r2_ht < r2_rate - 0.01:
        warn_list.append(
            f"High-T region degradation: R²={r2_ht:.4f} vs full-test R²={r2_rate:.4f}"
        )

    passed = all(pass_fail.values())

    return AprioriReport(
        predictor_rate_metrics=predictor_rate_metrics,
        predictor_yield_metrics=predictor_yield_metrics,
        baseline_rate_metrics=baseline_rate_metrics,
        baseline_yield_metrics=baseline_yield_metrics,
        aggregate_rate=agg_rate,
        aggregate_yield=agg_yield,
        error_vs_T=error_vs_T,
        error_vs_tau=error_vs_tau,
        heldout_low_conv_metrics=heldout_low_conv_metrics,
        heldout_high_T_metrics=heldout_high_T_metrics,
        pass_fail=pass_fail,
        passed=passed,
        config=config,
        active_species=active_species,
        warnings=warn_list,
    )

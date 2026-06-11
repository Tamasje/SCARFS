"""Energy acceptance suite for the SCARFS merged-model benchmark.

This module implements the §5 energy criteria from the SCARFS merge plan:

1. Global:   R²(S_E) ≥ 0.95 and relRMSE ≤ 0.23 (ChemZIP Φ̇ parity).
2. Tail:     per-case tail (top-20% |target|) median rel-err ≤ 10%; p95 ≤ 25% (provisional,
             target 10%); tail relRMSE ≤ 0.30.
3. Front:    per-case τ-position of peak S_E error ≤ 5% of case τ-span (median);
             per-case normalized CDF max deviation ≤ 0.05 (median).
4. Budget:   per-case ∫S_E dτ rel-err ≤ 5% median / 10% p95.
5. Sign:     fraction pred<0 reported only (head path architecturally non-negative; raw rate-tied
             S_E may have small negatives from trace species).

Sign convention
---------------
``evaluate_energy`` expects **absorption** values in J/m³/s (positive = endothermic), NOT the
Fluent source term S_h = −absorption.  In the DB, ``Reaction heat absorption [J/s/m³]``
is positive; the SurrogatePrediction.energy field carries S_E = −absorption (Fluent convention).
The caller must negate: pass ``-pred.energy`` as ``pred_absorption`` when calling this function.
For the 'head' path (MergedCoil absorption head), TorchSurrogate returns absorption directly.
Check adapter.py: energy = absorption (already positive) for MergedCoil, NOT negated.

Absolute-floor reference lines (J/m³/s)
-----------------------------------------
These are REPORTED, never used as gates:

- **5.1e3** : generator self-noise p95 (1.38e4 is his gate ≈ 1.21 × this value; plan §2).
- **1.61e5** : species-truncation information floor (energy from dropped species; measured in
              plan §2 diagnosis, root cause #3).
- **3e5–6e5**: information floor at full retained state (plan §4 expected position).

Reference: SCARFS merge plan §5 (steady-meandering-waterfall.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Absolute floor constants (REPORTED only, never gated)
# ---------------------------------------------------------------------------

#: Generator self-noise p95 (J/m³/s).  Plan §2: his gate 1.67e4 ≈ 1.21× this value.
ABS_FLOOR_GENERATOR_NOISE: float = 5.1e3

#: Species-truncation information floor (J/m³/s).  Plan §2 root cause #3: energy from
#: dropped species is 1.61e5 on stride6.
ABS_FLOOR_SPECIES_TRUNCATION: float = 1.61e5

#: Lower bound of the full-state information floor range (J/m³/s).  Plan §4 expected position.
ABS_FLOOR_INFORMATION_LOW: float = 3.0e5

#: Upper bound of the full-state information floor range (J/m³/s).
ABS_FLOOR_INFORMATION_HIGH: float = 6.0e5


# ---------------------------------------------------------------------------
# Threshold dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnergyThresholds:
    """Gate values for the §5 energy acceptance criteria.

    Attributes
    ----------
    r2_min
        Minimum global R² (§5.1). Default 0.95 (ChemZIP Φ̇ parity).
    rel_rmse_max
        Maximum global relRMSE (§5.1). Default 0.23.
    tail_median_rel_err_max
        Maximum median per-row |rel-err| in the tail region, taken as the
        median across cases (§5.2). Default 0.10.
    tail_p95_rel_err_max
        Maximum p95 per-row |rel-err| in the tail region, median across
        cases (§5.2 provisional). Default 0.25; final target is 0.10.
    tail_p95_rel_err_target
        Recorded aspirational target for tail p95, not gated. Default 0.10.
    tail_rel_rmse_max
        Maximum relRMSE within the tail region (§5.2). Default 0.30.
    peak_tau_position_err_max
        Maximum median (across cases) of |τ(argmax pred) − τ(argmax target)|
        normalised by case τ-span (§5.3 front localisation). Default 0.05.
    cdf_max_dev_max
        Maximum median (across cases) of max |CDF_pred − CDF_target| over τ
        (§5.3 cumulative-CDF criterion). Default 0.05.
    integral_rel_err_max_median
        Maximum median (across cases) of |∫S_E dτ rel-err| (§5.4 energy
        budget). Default 0.05.
    integral_rel_err_max_p95
        Maximum p95 (across cases) of |∫S_E dτ rel-err| (§5.4). Default 0.10.
    """

    r2_min: float = 0.95
    rel_rmse_max: float = 0.23
    tail_median_rel_err_max: float = 0.10
    tail_p95_rel_err_max: float = 0.25         # provisional; target = 0.10
    tail_p95_rel_err_target: float = 0.10      # aspirational, not gated
    tail_rel_rmse_max: float = 0.30
    peak_tau_position_err_max: float = 0.05
    cdf_max_dev_max: float = 0.05
    integral_rel_err_max_median: float = 0.05
    integral_rel_err_max_p95: float = 0.10


# ---------------------------------------------------------------------------
# Per-case result holder
# ---------------------------------------------------------------------------

@dataclass
class CaseEnergyResult:
    """Per-case intermediate metrics (internal; exposed inside EnergyReport)."""

    case_id: object
    n_rows: int
    n_tail_rows: int
    # tail metrics
    tail_median_rel_err: float
    tail_p95_rel_err: float
    tail_rel_rmse: float
    # front localization
    peak_tau_pos_err: float   # |τ(peak_pred) − τ(peak_target)| / τ_span; NaN if degenerate
    cdf_max_dev: float        # max |CDF_pred − CDF_target|; NaN if degenerate
    # energy budget
    integral_pred: float
    integral_target: float
    integral_rel_err: float


# ---------------------------------------------------------------------------
# Main report dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnergyReport:
    """Result of :func:`evaluate_energy`.

    Attributes
    ----------
    global_r2
        R² over all rows and all cases pooled.
    global_rel_rmse
        relRMSE = RMSE / std(target) pooled across all rows.
    tail_median_rel_err_median
        Median (across cases) of per-case tail median |rel-err|.
    tail_p95_rel_err_median
        Median (across cases) of per-case tail p95 |rel-err|.
    tail_rel_rmse_pooled
        relRMSE computed on the pooled tail rows (diagnostic; gameable).
    peak_tau_position_err_median
        Median (across cases) of |τ(argmax pred) − τ(argmax target)| / case τ-span.
    cdf_max_dev_median
        Median (across cases) of max |CDF_pred − CDF_target|.
    integral_rel_err_median
        Median (across cases) of |∫S_E dτ rel-err|.
    integral_rel_err_p95
        p95 (across cases) of |∫S_E dτ rel-err|.
    sign_negative_fraction
        Fraction of predictions that are negative (reported only; §5.5).
    pass_fail
        Mapping of criterion name → bool.
    passed
        True iff all entries in :attr:`pass_fail` are True.
    thresholds
        The :class:`EnergyThresholds` used.
    case_results
        Per-case detail (list of :class:`CaseEnergyResult`).
    n_rows
        Total row count.
    n_cases
        Number of unique cases.
    n_tail_pooled
        Number of rows in the pooled tail across all cases.
    warnings
        Diagnostic warnings emitted during evaluation.
    """

    global_r2: float
    global_rel_rmse: float
    tail_median_rel_err_median: float
    tail_p95_rel_err_median: float
    tail_rel_rmse_pooled: float        # reported-only diagnostic
    peak_tau_position_err_median: float
    cdf_max_dev_median: float
    integral_rel_err_median: float
    integral_rel_err_p95: float
    sign_negative_fraction: float      # reported only (§5.5)
    pass_fail: dict[str, bool]
    passed: bool
    thresholds: EnergyThresholds
    case_results: list[CaseEnergyResult]
    n_rows: int
    n_cases: int
    n_tail_pooled: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        thr = self.thresholds
        lines = [
            "=" * 70,
            "ENERGY ACCEPTANCE REPORT  (§5 SCARFS merge plan)",
            "=" * 70,
            f"Rows: {self.n_rows}   Cases: {self.n_cases}   Tail rows (pooled): {self.n_tail_pooled}",
            f"Sign (pred<0 fraction): {self.sign_negative_fraction:.4f}  [reported only]",
            "",
            "--- Global ---",
            f"  R²            : {self.global_r2:+.4f}  (threshold >= {thr.r2_min})",
            f"  relRMSE       : {self.global_rel_rmse:.4f}  (threshold <= {thr.rel_rmse_max})",
            "",
            "--- Tail (per-case, median across cases) ---",
            f"  Median |rel-err|  : {self.tail_median_rel_err_median:.4f}  (threshold <= {thr.tail_median_rel_err_max})",
            f"  p95 |rel-err|     : {self.tail_p95_rel_err_median:.4f}  (provisional threshold <= {thr.tail_p95_rel_err_max}, target {thr.tail_p95_rel_err_target})",
            f"  relRMSE (pooled)  : {self.tail_rel_rmse_pooled:.4f}  (threshold <= {thr.tail_rel_rmse_max})",
            "",
            "--- Front Localization ---",
            f"  Peak-τ position err (median) : {self.peak_tau_position_err_median:.4f}  (threshold <= {thr.peak_tau_position_err_max})",
            f"  CDF max deviation (median)   : {self.cdf_max_dev_median:.4f}  (threshold <= {thr.cdf_max_dev_max})",
            "",
            "--- Energy Budget ---",
            f"  ∫S_E dτ rel-err median : {self.integral_rel_err_median:.4f}  (threshold <= {thr.integral_rel_err_max_median})",
            f"  ∫S_E dτ rel-err p95    : {self.integral_rel_err_p95:.4f}  (threshold <= {thr.integral_rel_err_max_p95})",
            "",
            "--- Absolute Floors (reported only) ---",
            f"  Generator noise p95         : {ABS_FLOOR_GENERATOR_NOISE:.2e} J/m³/s",
            f"  Species-truncation floor    : {ABS_FLOOR_SPECIES_TRUNCATION:.2e} J/m³/s",
            f"  Information floor (full)    : {ABS_FLOOR_INFORMATION_LOW:.2e}–{ABS_FLOOR_INFORMATION_HIGH:.2e} J/m³/s",
            "",
            "--- PASS / FAIL ---",
        ]
        for criterion, result in self.pass_fail.items():
            status = "PASS" if result else "FAIL"
            lines.append(f"  [{status}] {criterion}")
        lines.append(f"\nOverall: {'PASS' if self.passed else 'FAIL'}")
        if self.warnings:
            lines.append("")
            lines.append("--- Warnings ---")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        lines.append("=" * 70)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRAPEZOID = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _safe_rel_err(pred: np.ndarray, target: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Element-wise |pred - target| / max(|target|, eps) — safe denominator."""
    denom = np.where(np.abs(target) > eps, np.abs(target), eps)
    return np.abs(pred - target) / denom


def _rel_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """relRMSE = RMSE / std(target).  Returns 0 for constant targets."""
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    std = float(np.std(target))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(np.mean((pred - target) ** 2))) / std


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    """R² = 1 - SS_res / SS_tot.  Returns 0 for constant target."""
    target = np.asarray(target, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - np.sum((target - pred) ** 2) / ss_tot)


def _tail_mask(target: np.ndarray, tail_fraction: float) -> np.ndarray:
    """Boolean mask selecting the top *tail_fraction* of rows by |target|."""
    if len(target) == 0 or tail_fraction <= 0.0:
        return np.zeros(len(target), dtype=bool)
    threshold = np.percentile(np.abs(target), 100.0 * (1.0 - tail_fraction))
    return np.abs(target) >= threshold


def _normalized_cdf(values: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the normalised cumulative integral of *values* over sorted *tau*.

    Returns ``(tau_sorted, cdf)`` where ``cdf[-1] == 1`` (or 0 if integral is
    zero).  Degenerate: if the total integral is zero returns all-zeros.
    """
    order = np.argsort(tau)
    tau_s = tau[order]
    val_s = values[order]
    cumsum = np.zeros(len(val_s), dtype=float)
    for i in range(1, len(val_s)):
        dt = tau_s[i] - tau_s[i - 1]
        cumsum[i] = cumsum[i - 1] + 0.5 * (val_s[i] + val_s[i - 1]) * dt
    total = float(cumsum[-1])
    if total == 0.0 or not np.isfinite(total):
        return tau_s, np.zeros_like(cumsum)
    return tau_s, cumsum / total


def _process_case(
    case_id: object,
    pred: np.ndarray,
    target: np.ndarray,
    tau: np.ndarray,
    tail_fraction: float,
) -> CaseEnergyResult:
    """Compute all per-case metrics for a single PFR case.

    Parameters
    ----------
    case_id
        Identifier (for bookkeeping).
    pred, target
        1-D absorption arrays [J/m³/s] for this case.
    tau
        Residence time [s] for each row (monotonically increasing within case).
    tail_fraction
        Fraction of rows (by |target|) that define the 'tail' region.
    """
    n = len(pred)

    # --- degenerate guard ---
    if n == 0:
        return CaseEnergyResult(
            case_id=case_id, n_rows=0, n_tail_rows=0,
            tail_median_rel_err=np.nan, tail_p95_rel_err=np.nan, tail_rel_rmse=np.nan,
            peak_tau_pos_err=np.nan, cdf_max_dev=np.nan,
            integral_pred=0.0, integral_target=0.0, integral_rel_err=np.nan,
        )

    # --- tail region ---
    tmask = _tail_mask(target, tail_fraction)
    n_tail = int(tmask.sum())

    if n_tail > 0:
        rel_errs = _safe_rel_err(pred[tmask], target[tmask])
        tail_med = float(np.median(rel_errs))
        tail_p95 = float(np.percentile(rel_errs, 95))
        tail_rrmse = _rel_rmse(pred[tmask], target[tmask])
    else:
        tail_med = tail_p95 = tail_rrmse = np.nan

    # --- front localization ---
    tau_span = float(tau.max() - tau.min()) if n > 1 else 1.0
    if tau_span <= 0.0:
        tau_span = 1.0

    # peak-τ position error
    peak_target_idx = int(np.argmax(target))
    peak_pred_idx = int(np.argmax(pred))
    tau_target_peak = float(tau[peak_target_idx])
    tau_pred_peak = float(tau[peak_pred_idx])
    peak_tau_err = abs(tau_pred_peak - tau_target_peak) / tau_span

    # CDF max deviation
    if n > 1:
        tau_s_t, cdf_t = _normalized_cdf(target, tau)
        tau_s_p, cdf_p = _normalized_cdf(pred, tau)
        # interpolate pred CDF at target tau points
        cdf_p_interp = np.interp(tau_s_t, tau_s_p, cdf_p)
        cdf_dev = float(np.max(np.abs(cdf_t - cdf_p_interp)))
    else:
        cdf_dev = np.nan

    # --- energy budget ---
    if n > 1:
        integ_pred = float(_TRAPEZOID(pred, x=tau))
        integ_target = float(_TRAPEZOID(target, x=tau))
    else:
        integ_pred = float(pred[0])
        integ_target = float(target[0])

    denom_budget = max(abs(integ_target), 1.0)
    integ_rel_err = abs(integ_pred - integ_target) / denom_budget

    return CaseEnergyResult(
        case_id=case_id,
        n_rows=n,
        n_tail_rows=n_tail,
        tail_median_rel_err=tail_med,
        tail_p95_rel_err=tail_p95,
        tail_rel_rmse=tail_rrmse,
        peak_tau_pos_err=peak_tau_err,
        cdf_max_dev=cdf_dev,
        integral_pred=integ_pred,
        integral_target=integ_target,
        integral_rel_err=integ_rel_err,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_energy(
    pred_absorption: np.ndarray,
    target_absorption: np.ndarray,
    case_ids: Sequence,
    tau: np.ndarray,
    *,
    tail_fraction: float = 0.20,
    thresholds: EnergyThresholds | None = None,
) -> EnergyReport:
    """Evaluate the §5 energy acceptance criteria on a set of predictions.

    Caller convention
    -----------------
    ``pred_absorption`` and ``target_absorption`` are both in the **absorption**
    sign: positive = endothermic (same sign as ``Reaction heat absorption [J/s/m³]``
    in the database).  For a ``SurrogatePrediction`` whose ``.energy`` field carries
    S_E = −absorption (Fluent convention), pass ``-pred.energy``.
    For the MergedCoil 'head' path, TorchSurrogate already returns ``energy =
    absorption`` (positive), so pass it directly.

    Parameters
    ----------
    pred_absorption
        ``(n,)`` predicted absorption values [J/m³/s].
    target_absorption
        ``(n,)`` ground-truth absorption values [J/m³/s].
    case_ids
        ``(n,)`` case-ID array (int or str) identifying which rows belong to
        which PFR case.  Consecutive rows in the same case should be ordered
        by increasing τ for the front-localisation metrics to be meaningful,
        but the function sorts internally.
    tau
        ``(n,)`` residence-time [s] for each row.
    tail_fraction
        Fraction of rows per case (by |target|) that define the 'tail' region.
        Default 0.20 (top-20% per §5.2).
    thresholds
        :class:`EnergyThresholds`; uses defaults when ``None``.

    Returns
    -------
    EnergyReport
        Full metrics, PASS/FAIL flags, and a :meth:`~EnergyReport.summary` method.
    """
    if thresholds is None:
        thresholds = EnergyThresholds()

    pred = np.asarray(pred_absorption, dtype=float)
    target = np.asarray(target_absorption, dtype=float)
    case_ids_arr = np.asarray(case_ids)
    tau_arr = np.asarray(tau, dtype=float)

    n_total = len(pred)
    warn_list: list[str] = []

    # --- global metrics ---
    global_r2 = _r2(pred, target)
    global_rrmse = _rel_rmse(pred, target)

    # --- sign consistency ---
    sign_neg_frac = float(np.mean(pred < 0)) if n_total > 0 else 0.0

    # --- per-case processing ---
    unique_cases = np.unique(case_ids_arr)
    n_cases = len(unique_cases)
    case_results: list[CaseEnergyResult] = []

    pooled_tail_pred: list[np.ndarray] = []
    pooled_tail_target: list[np.ndarray] = []

    for cid in unique_cases:
        mask = case_ids_arr == cid
        c_pred = pred[mask]
        c_target = target[mask]
        c_tau = tau_arr[mask]
        # sort by tau within case
        order = np.argsort(c_tau)
        cr = _process_case(cid, c_pred[order], c_target[order], c_tau[order], tail_fraction)
        case_results.append(cr)

        tmask = _tail_mask(c_target, tail_fraction)
        if tmask.any():
            pooled_tail_pred.append(c_pred[tmask])
            pooled_tail_target.append(c_target[tmask])

    # --- aggregate per-case scalars ---
    def _finite_median(values: list[float]) -> float:
        arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
        return float(np.median(arr)) if len(arr) > 0 else np.nan

    def _finite_percentile(values: list[float], q: float) -> float:
        arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
        return float(np.percentile(arr, q)) if len(arr) > 0 else np.nan

    tail_med_median = _finite_median([cr.tail_median_rel_err for cr in case_results])
    tail_p95_median = _finite_median([cr.tail_p95_rel_err for cr in case_results])
    peak_tau_median = _finite_median([cr.peak_tau_pos_err for cr in case_results])
    cdf_dev_median = _finite_median([cr.cdf_max_dev for cr in case_results])
    integ_rel_median = _finite_median([cr.integral_rel_err for cr in case_results])
    integ_rel_p95 = _finite_percentile([cr.integral_rel_err for cr in case_results], 95)

    # --- pooled tail relRMSE (diagnostic only) ---
    n_tail_pooled = 0
    if pooled_tail_pred:
        all_tp = np.concatenate(pooled_tail_pred)
        all_tt = np.concatenate(pooled_tail_target)
        n_tail_pooled = len(all_tp)
        tail_pooled_rrmse = _rel_rmse(all_tp, all_tt)
    else:
        tail_pooled_rrmse = np.nan
        warn_list.append("No tail rows found across all cases.")

    # warn for degenerate cases
    n_degen = sum(1 for cr in case_results if cr.n_rows <= 1)
    if n_degen:
        warn_list.append(f"{n_degen} case(s) with ≤1 row — front-localisation metrics are NaN for those.")

    # --- PASS / FAIL ---
    def _pass(val: float, threshold: float, direction: str) -> bool:
        """Return True if val satisfies the gate; NaN always FAILS."""
        if not np.isfinite(val):
            return False
        return val >= threshold if direction == "ge" else val <= threshold

    pass_fail: dict[str, bool] = {
        f"global_r2 >= {thresholds.r2_min}":
            _pass(global_r2, thresholds.r2_min, "ge"),
        f"global_rel_rmse <= {thresholds.rel_rmse_max}":
            _pass(global_rrmse, thresholds.rel_rmse_max, "le"),
        f"tail_median_rel_err_median <= {thresholds.tail_median_rel_err_max}":
            _pass(tail_med_median, thresholds.tail_median_rel_err_max, "le"),
        f"tail_p95_rel_err_median <= {thresholds.tail_p95_rel_err_max} (provisional)":
            _pass(tail_p95_median, thresholds.tail_p95_rel_err_max, "le"),
        f"tail_rel_rmse_pooled <= {thresholds.tail_rel_rmse_max}":
            _pass(tail_pooled_rrmse, thresholds.tail_rel_rmse_max, "le"),
        f"peak_tau_position_err_median <= {thresholds.peak_tau_position_err_max}":
            _pass(peak_tau_median, thresholds.peak_tau_position_err_max, "le"),
        f"cdf_max_dev_median <= {thresholds.cdf_max_dev_max}":
            _pass(cdf_dev_median, thresholds.cdf_max_dev_max, "le"),
        f"integral_rel_err_median <= {thresholds.integral_rel_err_max_median}":
            _pass(integ_rel_median, thresholds.integral_rel_err_max_median, "le"),
        f"integral_rel_err_p95 <= {thresholds.integral_rel_err_max_p95}":
            _pass(integ_rel_p95, thresholds.integral_rel_err_max_p95, "le"),
    }
    passed = all(pass_fail.values())

    return EnergyReport(
        global_r2=global_r2,
        global_rel_rmse=global_rrmse,
        tail_median_rel_err_median=tail_med_median,
        tail_p95_rel_err_median=tail_p95_median,
        tail_rel_rmse_pooled=tail_pooled_rrmse,
        peak_tau_position_err_median=peak_tau_median,
        cdf_max_dev_median=cdf_dev_median,
        integral_rel_err_median=integ_rel_median,
        integral_rel_err_p95=integ_rel_p95,
        sign_negative_fraction=sign_neg_frac,
        pass_fail=pass_fail,
        passed=passed,
        thresholds=thresholds,
        case_results=case_results,
        n_rows=n_total,
        n_cases=n_cases,
        n_tail_pooled=n_tail_pooled,
        warnings=warn_list,
    )

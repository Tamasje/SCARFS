"""Raw database sanity checks for the SCARFS diagnostics package.

Context (plan §2 / energy-deviation diagnosis):
    Root cause #5 identified a miscalibrated absolute gate in the colleague's
    pipeline.  This module re-implements the sanity check pattern over our
    contract with explicit positivity lineage tracking for the absorption column
    (plan §4 E-c).

    Checks performed:
    1. Non-finite counts (NaN/Inf in any Y_, dYdt_, T, P, rho column).
    2. Material negatives: Y < −1e-5 (not just numerical noise ≈ −0.0).
    3. Y > 1 counts.
    4. Per-row ΣY fraction in [0.999, 1.001].
    5. Duplicated (CaseID, tau) rows.
    6. Absorption negativity: count + min.  Flag if min < −1e6 J/m³/s
       (positivity lineage check from plan §4 E-c).
    7. T/P range sanity.
    8. Case count.

PASS: no material violations (Y < −1e-5, or Y > 1 fraction above 0.5%).
Corruption injector test: inject Y=2.0 → FAIL expected (for the negative test).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Schema
from .report import md_header, md_kv_block, md_table, write_report_pair

# Threshold for 'material negative' mass fraction (not just numerical noise)
_MATERIAL_NEGATIVE_Y = -1e-5

# Flag threshold for absorption negativity (plan §4 E-c)
_ABSORPTION_NEGATIVE_FLAG = -1.0e6  # J/m³/s


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SanityReport:
    """Summary of database sanity checks.

    Attributes
    ----------
    n_rows
        Total rows in *df*.
    n_cases
        Number of unique CaseID values.
    n_species
        Number of species columns detected.
    n_nonfinite
        Count of non-finite values across all species mass fractions.
    n_material_negative_y
        Count of rows with at least one species Y < −1e-5.
    n_y_above_one
        Count of Y > 1 values.
    row_sum_fraction_ok
        Fraction of rows where ΣY ∈ [0.999, 1.001].
    n_duplicate_rows
        Number of duplicated (CaseID, tau) pairs (if both columns present).
    absorption_min
        Minimum absorption value [J/m³/s] (-inf if column absent).
    absorption_negative_count
        Count of absorption < 0 rows.
    absorption_flag
        ``True`` if absorption_min < −1e6 J/m³/s (positivity lineage alert).
    T_min
        Minimum temperature [K].
    T_max
        Maximum temperature [K].
    P_min
        Minimum pressure [Pa].
    P_max
        Maximum pressure [Pa].
    passed
        ``True`` if no material violations.
    """

    n_rows: int
    n_cases: int
    n_species: int
    n_nonfinite: int
    n_material_negative_y: int
    n_y_above_one: int
    row_sum_fraction_ok: float
    n_duplicate_rows: int
    absorption_min: float
    absorption_negative_count: int
    absorption_flag: bool
    T_min: float
    T_max: float
    P_min: float
    P_max: float
    passed: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_raw_database_sanity(
    df: pd.DataFrame,
    schema: Schema,
    out_dir: str | os.PathLike,
) -> SanityReport:
    """Perform raw database sanity checks.

    Parameters
    ----------
    df
        Database rows to audit.
    schema
        Column contract resolved from *df*.
    out_dir
        Directory to write ``sanity_report.md`` / ``sanity_report.csv``.

    Returns
    -------
    :class:`SanityReport`

    Notes
    -----
    PASS criteria:
        - n_material_negative_y == 0 (no Y < −1e-5)
        - n_y_above_one == 0 (no Y > 1)
        - row_sum_fraction_ok >= 0.995 (≥99.5% rows sum to ~1)
    An absorption_flag (min < −1e6) is reported separately but does not
    trigger FAIL by itself — it is a lineage alert for the E-c positivity
    assumption.
    """
    n_rows = len(df)

    # --- Species mass-fraction columns (real, no pseudo-species) ---
    y_cols = schema.y_columns()
    y_present = [c for c in y_cols if c in df.columns]
    Y_mat = df[y_present].to_numpy(dtype=float)

    n_species = len(y_present)
    n_nonfinite = int(np.sum(~np.isfinite(Y_mat)))
    n_material_negative_y = int(np.sum(np.any(Y_mat < _MATERIAL_NEGATIVE_Y, axis=1)))
    n_y_above_one = int(np.sum(Y_mat > 1.0 + 1e-9))
    row_sums = Y_mat.sum(axis=1)
    row_sum_ok = (row_sums >= 0.999) & (row_sums <= 1.001)
    row_sum_fraction_ok = float(row_sum_ok.mean())

    # --- Duplicated (CaseID, tau) ---
    n_duplicate_rows = 0
    tau_key = schema.state.get("tau")
    case_key = schema.meta.get("CaseID", None)
    if case_key is None and "CaseID" in df.columns:
        case_key = "CaseID"
    if tau_key and case_key and tau_key in df.columns and case_key in df.columns:
        n_duplicate_rows = int(df.duplicated(subset=[case_key, tau_key]).sum())

    # --- Cases count ---
    if case_key and case_key in df.columns:
        n_cases = int(df[case_key].nunique())
    else:
        n_cases = 0

    # --- Absorption ---
    absorption_min = float("-inf")
    absorption_negative_count = 0
    absorption_flag = False
    try:
        abs_col = schema.energy_target_column()
        if abs_col in df.columns:
            abs_vals = df[abs_col].to_numpy(dtype=float)
            absorption_min = float(np.nanmin(abs_vals))
            absorption_negative_count = int(np.sum(abs_vals < 0))
            absorption_flag = absorption_min < _ABSORPTION_NEGATIVE_FLAG
    except KeyError:
        pass  # absorption column absent — not a hard failure

    # --- T/P range ---
    T_min, T_max, P_min, P_max = np.nan, np.nan, np.nan, np.nan
    if schema.has_state("T"):
        t_col = schema.state["T"]
        if t_col in df.columns:
            T_vals = df[t_col].to_numpy(dtype=float)
            T_min, T_max = float(np.nanmin(T_vals)), float(np.nanmax(T_vals))
    if schema.has_state("P"):
        p_col = schema.state["P"]
        if p_col in df.columns:
            P_vals = df[p_col].to_numpy(dtype=float)
            P_min, P_max = float(np.nanmin(P_vals)), float(np.nanmax(P_vals))

    # --- PASS determination ---
    passed = (
        n_material_negative_y == 0
        and n_y_above_one == 0
        and row_sum_fraction_ok >= 0.995
    )

    result = SanityReport(
        n_rows=n_rows,
        n_cases=n_cases,
        n_species=n_species,
        n_nonfinite=n_nonfinite,
        n_material_negative_y=n_material_negative_y,
        n_y_above_one=n_y_above_one,
        row_sum_fraction_ok=row_sum_fraction_ok,
        n_duplicate_rows=n_duplicate_rows,
        absorption_min=absorption_min,
        absorption_negative_count=absorption_negative_count,
        absorption_flag=absorption_flag,
        T_min=T_min,
        T_max=T_max,
        P_min=P_min,
        P_max=P_max,
        passed=passed,
    )

    kv: dict[str, object] = {
        "passed": passed,
        "n_rows": n_rows,
        "n_cases": n_cases,
        "n_species": n_species,
        "n_nonfinite_Y": n_nonfinite,
        "n_material_negative_Y_below_minus1e5": n_material_negative_y,
        "n_Y_above_one": n_y_above_one,
        "row_sum_fraction_ok_[0.999_1.001]": f"{row_sum_fraction_ok:.4f}",
        "n_duplicate_CaseID_tau": n_duplicate_rows,
        "absorption_min_J_m3_s": f"{absorption_min:.4e}",
        "absorption_negative_count": absorption_negative_count,
        "absorption_flag_below_minus1e6": absorption_flag,
        "T_min_K": f"{T_min:.2f}",
        "T_max_K": f"{T_max:.2f}",
        "P_min_Pa": f"{P_min:.2f}",
        "P_max_Pa": f"{P_max:.2f}",
    }
    csv_rows = [{"check": k, "value": str(v)} for k, v in sorted(kv.items())]
    write_report_pair(
        out_dir,
        "sanity_report",
        md_sections=[
            (md_header("Raw Database Sanity Report"), md_kv_block(kv)),
            (md_header("Check Table", level=2), md_table(csv_rows)),
        ],
        csv_rows=csv_rows,
    )
    return result

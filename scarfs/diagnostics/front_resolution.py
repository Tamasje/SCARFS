"""Front-resolution audit for the SCARFS diagnostics package.

Context (plan §2 / root cause #4):
    Front under-resolution in storage was identified as root cause #4.  On the
    stride-5 database, consecutive stored points jump 39% median / 82% p95 of
    the per-case S_E peak.  This motivates front-adaptive storage (plan §3 data
    side) and this standing diagnostic that flags cases with large S_E jumps.

    The check measures:
    - Per-case consecutive |ΔS_E| as a fraction of the case's S_E peak.
    - Reports median and p95 across all steps (pooled across cases).
    - Flags individual cases whose maximum jump > 20% of their peak.

    The 20% flag threshold is chosen as the boundary below which front-adaptive
    storage would not trigger (plan: store where inter-point ΔS_E > ~2–5% of
    case max); 20% is a conservative flag for gross under-resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import Schema
from .report import md_header, md_kv_block, md_table, write_report_pair

# Flag threshold: cases with max jump > this fraction of case peak are flagged.
_JUMP_FLAG_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class FrontResolutionReport:
    """Summary of the front-resolution audit.

    Attributes
    ----------
    median_jump_frac
        Median consecutive |ΔS_E| / case_peak across all step pairs.
    p95_jump_frac
        95th-percentile of the same.
    n_flagged_cases
        Number of cases with max_jump > 20% of peak.
    n_total_cases
        Total number of cases processed.
    flagged_case_ids
        CaseID values of flagged cases (up to 20 listed; full list in CSV).
    n_step_pairs
        Total number of consecutive step pairs used in the statistics.
    """

    median_jump_frac: float
    p95_jump_frac: float
    n_flagged_cases: int
    n_total_cases: int
    flagged_case_ids: list
    n_step_pairs: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_front_resolution_audit(
    df: pd.DataFrame,
    schema: Schema,
    out_dir: str | os.PathLike,
    *,
    flag_threshold: float = _JUMP_FLAG_THRESHOLD,
) -> FrontResolutionReport:
    """Audit the S_E front resolution implied by the storage stride.

    For each case, sorts rows by τ, computes consecutive |ΔS_E| normalised by
    the case peak, and accumulates the distribution.  Cases whose maximum jump
    exceeds *flag_threshold* are flagged.

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract.  Must have the absorption column and tau.
    out_dir
        Output directory for ``front_resolution_audit.md`` /
        ``front_resolution_audit.csv``.
    flag_threshold
        Fraction of case peak above which a maximum step jump triggers a flag.
        Default 0.20 (20%).

    Returns
    -------
    :class:`FrontResolutionReport`
    """
    abs_col = schema.energy_target_column()
    tau_col = schema.state.get("tau")
    case_key = schema.meta.get("CaseID", None)
    if case_key is None and "CaseID" in df.columns:
        case_key = "CaseID"

    if tau_col is None or tau_col not in df.columns:
        raise KeyError(
            "front_resolution_audit: tau column not found in schema state. "
            "The database must have a 'tau [s]' column."
        )
    if case_key is None or case_key not in df.columns:
        raise KeyError(
            "front_resolution_audit: CaseID column not found. "
            "The database must have a 'CaseID' column."
        )

    all_jump_fracs: list[float] = []
    flagged_case_ids: list = []
    n_total_cases = 0
    case_rows: list[dict] = []

    for cid, grp in df.groupby(case_key):
        grp_sorted = grp.sort_values(tau_col)
        se = grp_sorted[abs_col].to_numpy(dtype=float)
        peak = float(np.max(np.abs(se)))
        n_total_cases += 1

        if peak < 1.0 or len(se) < 2:
            # Degenerate case (no meaningful energy activity)
            continue

        diffs = np.abs(np.diff(se))
        jumps = diffs / peak
        all_jump_fracs.extend(jumps.tolist())

        max_jump = float(np.max(jumps)) if len(jumps) > 0 else 0.0
        flagged = max_jump > flag_threshold
        if flagged:
            flagged_case_ids.append(cid)

        case_rows.append({
            "CaseID": cid,
            "n_steps": len(se),
            "peak_S_E": f"{peak:.4e}",
            "median_jump_frac": f"{float(np.median(jumps)):.4f}",
            "p95_jump_frac": f"{float(np.percentile(jumps, 95)):.4f}",
            "max_jump_frac": f"{max_jump:.4f}",
            "flagged": flagged,
        })

    if all_jump_fracs:
        arr = np.array(all_jump_fracs, dtype=float)
        median_jump_frac = float(np.median(arr))
        p95_jump_frac = float(np.percentile(arr, 95))
    else:
        median_jump_frac = np.nan
        p95_jump_frac = np.nan

    result = FrontResolutionReport(
        median_jump_frac=median_jump_frac,
        p95_jump_frac=p95_jump_frac,
        n_flagged_cases=len(flagged_case_ids),
        n_total_cases=n_total_cases,
        flagged_case_ids=flagged_case_ids[:20],
        n_step_pairs=len(all_jump_fracs),
    )

    kv: dict[str, object] = {
        "median_jump_frac_of_peak": f"{median_jump_frac:.4f}",
        "p95_jump_frac_of_peak": f"{p95_jump_frac:.4f}",
        "n_flagged_cases_max_jump_gt_{flag_threshold}".format(flag_threshold=flag_threshold):
            len(flagged_case_ids),
        "n_total_cases": n_total_cases,
        "n_step_pairs": len(all_jump_fracs),
        "flag_threshold": flag_threshold,
        "context": (
            "stride5 baseline: 39% median / 82% p95; "
            "front-adaptive storage targets <5% median"
        ),
    }
    write_report_pair(
        out_dir,
        "front_resolution_audit",
        md_sections=[
            (md_header("Front Resolution Audit"), md_kv_block(kv)),
            (md_header("Per-case Table", level=2), md_table(case_rows)),
        ],
        csv_rows=case_rows,
    )
    return result

"""State-space ambiguity diagnostics for the SCARFS diagnostics package.

Context (plan §2 / root cause #3):
    Hidden-state ambiguity from species dropping was identified as root cause #3
    of the colleague's energy failure.  At matched retained-state distance,
    dropped-near pairs show 23–98× smaller |ΔS_E|; kNN tail R² rises 0.10 → 0.53
    retained → full state.

    These diagnostics quantify how well the state space used by the surrogate
    identifies points that have similar energy source terms.  Two modes:

    - ``run_state_ambiguity`` — NN statistics for a single state definition
      (``"retained"`` = species with max Y > threshold, or ``"full"`` = all dry
      species + T + P).
    - ``run_ambiguity_collapse`` — runs both states, adds the dropped-subspace
      conditional check: at matched retained-distance quartiles, compare |ΔS_E|
      for dropped-near vs dropped-far pairs.  This is the Phase-1 'collapse test'.

Caveat (documented here and in report output):
    Small-n NN floors overestimate ambiguity — with 87 rows in the stride6
    sample the nearest-neighbour distance distribution reflects sparse sampling,
    not the true information floor.  Numbers are meaningful only on the full
    stride5 database (102 k rows).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

from ..schema import Schema
from .report import md_header, md_kv_block, md_table, write_report_pair


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AmbiguityReport:
    """NN-based ambiguity statistics for a single state definition.

    Attributes
    ----------
    state
        State name used: ``"retained"`` or ``"full"``.
    dist_p50
        Median NN distance in the standardised state space.
    dist_p95
        95th-percentile NN distance.
    delta_se_p50
        Median |ΔS_E| across all NN pairs [J/m³/s].
    delta_se_p95
        95th-percentile |ΔS_E| [J/m³/s].
    spearman_r
        Spearman(|ΔS_E|, NN distance).  Positive = higher S_E uncertainty in
        sparse regions of state space.
    spearman_p
        Two-sided p-value of the Spearman test.
    n_pairs
        Number of NN pairs used.
    n_state_dims
        Dimensionality of the standardised state space.
    """

    state: str
    dist_p50: float
    dist_p95: float
    delta_se_p50: float
    delta_se_p95: float
    spearman_r: float
    spearman_p: float
    n_pairs: int
    n_state_dims: int


@dataclass
class AmbiguityCollapseReport:
    """Results of the full Phase-1 collapse test.

    Attributes
    ----------
    retained
        :class:`AmbiguityReport` for the retained state.
    full
        :class:`AmbiguityReport` for the full state.
    collapse_detected
        ``True`` when ``full.delta_se_p50 <= retained.delta_se_p50``
        (the expected collapse direction: full state resolves more ambiguity).
    dropped_near_delta_se_p50
        Median |ΔS_E| for NN pairs that are close in the *dropped* subspace
        at the same retained-distance quartile.  Lower = dropped subspace
        carries information beyond the retained state.
    dropped_far_delta_se_p50
        Same statistic for pairs that are far in the dropped subspace.
    conditional_ratio
        ``dropped_far_delta_se_p50 / dropped_near_delta_se_p50``.  Values
        substantially above 1.0 indicate the dropped species carry additional
        S_E information (root cause #3 evidence).
    """

    retained: AmbiguityReport
    full: AmbiguityReport
    collapse_detected: bool
    dropped_near_delta_se_p50: float
    dropped_far_delta_se_p50: float
    conditional_ratio: float


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _standardise(X: np.ndarray) -> np.ndarray:
    """Standardise columns to zero mean, unit variance (clip std floor at 1e-12)."""
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-12)
    return (X - mean) / std


def _build_state_matrix(
    df: pd.DataFrame,
    schema: Schema,
    *,
    state: str,
    threshold_y: float,
) -> np.ndarray:
    """Build the raw (non-standardised) feature matrix for *state*.

    Parameters
    ----------
    state
        ``"retained"`` (species with max Y > threshold_y) or ``"full"`` (all
        dry species, i.e. all real Y_ columns).
    threshold_y
        Threshold for ``"retained"`` mode.
    """
    t_col = schema.require_state("T")[0]
    p_col = schema.require_state("P")[0]
    y_cols = [c for c in df.columns if c.startswith("Y_") and "[" not in c]

    if state == "retained":
        y_arr = df[y_cols].to_numpy(dtype=float)
        max_y = y_arr.max(axis=0)
        y_cols = [y_cols[i] for i in range(len(y_cols)) if max_y[i] > threshold_y]
    elif state == "full":
        pass  # keep all dry Y_ columns
    else:
        raise ValueError(f"state must be 'retained' or 'full', got {state!r}")

    return df[y_cols + [t_col, p_col]].to_numpy(dtype=float)


def _nn_pairs(
    X_std: np.ndarray,
    se: np.ndarray,
    case_ids: np.ndarray | None,
    *,
    n_neighbors: int,
    exclude_same_case: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nn_distances, nn_indices, delta_se) arrays.

    When *exclude_same_case* is ``True`` and *case_ids* is provided, the
    nearest neighbour is chosen as the closest point from a *different* case.

    Returns
    -------
    distances
        ``(n,)`` distance to the nearest acceptable neighbour.
    indices
        ``(n,)`` index of that neighbour.
    delta_se
        ``(n,)`` |S_E[i] - S_E[nn_index[i]]|.
    """
    n = X_std.shape[0]
    k = min(n_neighbors + 10, n - 1)  # fetch extra neighbours for exclusion fallback
    nn_model = NearestNeighbors(n_neighbors=k + 1)
    nn_model.fit(X_std)
    dists_all, inds_all = nn_model.kneighbors(X_std)

    distances = np.empty(n, dtype=float)
    indices = np.empty(n, dtype=int)

    for i in range(n):
        chosen_dist = np.nan
        chosen_idx = -1
        for j in range(1, dists_all.shape[1]):  # skip self (j=0)
            cand = inds_all[i, j]
            if exclude_same_case and case_ids is not None:
                if case_ids[i] == case_ids[cand]:
                    continue
            chosen_dist = dists_all[i, j]
            chosen_idx = cand
            break
        distances[i] = chosen_dist
        indices[i] = chosen_idx

    valid = indices >= 0
    delta_se = np.where(valid, np.abs(se - se[np.where(valid, indices, 0)]), np.nan)
    distances = np.where(valid, distances, np.nan)
    return distances, indices, delta_se


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_state_ambiguity(
    df: pd.DataFrame,
    schema: Schema,
    out_dir: str | os.PathLike,
    *,
    state: str = "retained",
    threshold_y: float = 1e-4,
    n_neighbors: int = 1,
    exclude_same_case: bool = True,
) -> AmbiguityReport:
    """Compute NN-based ambiguity statistics for a single state definition.

    Standardises the state space, finds 1-NN pairs (optionally cross-case),
    and reports NN-distance p50/p95, |ΔS_E| p50/p95, and Spearman correlation.

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract.
    out_dir
        Output directory for the report pair.
    state
        ``"retained"`` or ``"full"`` (see module docstring).
    threshold_y
        Max-Y threshold for ``"retained"`` mode (default 1e-4, the colleague's
        species-dropping gate).
    n_neighbors
        Number of neighbours to find (1 = closest neighbour).
    exclude_same_case
        If ``True``, nearest neighbours from the same CaseID are skipped.
        This prevents trivially close neighbours along the same reactor profile.

    Returns
    -------
    :class:`AmbiguityReport`

    Notes
    -----
    Caveat: small-n NN floors overestimate ambiguity (see module docstring).
    """
    abs_col = schema.energy_target_column()
    se = df[abs_col].to_numpy(dtype=float)
    case_ids: np.ndarray | None = None
    if "CaseID" in schema.meta:
        case_ids = df[schema.meta["CaseID"]].to_numpy()
    elif "CaseID" in df.columns:
        case_ids = df["CaseID"].to_numpy()

    X = _build_state_matrix(df, schema, state=state, threshold_y=threshold_y)
    X_std = _standardise(X)

    dists, inds, delta_se = _nn_pairs(
        X_std, se, case_ids,
        n_neighbors=n_neighbors,
        exclude_same_case=exclude_same_case,
    )

    valid = np.isfinite(dists) & np.isfinite(delta_se)
    dists_v = dists[valid]
    delta_se_v = delta_se[valid]

    dist_p50 = float(np.percentile(dists_v, 50)) if dists_v.size > 0 else np.nan
    dist_p95 = float(np.percentile(dists_v, 95)) if dists_v.size > 0 else np.nan
    delta_se_p50 = float(np.percentile(delta_se_v, 50)) if delta_se_v.size > 0 else np.nan
    delta_se_p95 = float(np.percentile(delta_se_v, 95)) if delta_se_v.size > 0 else np.nan

    if dists_v.size >= 4:
        r_val, p_val = spearmanr(delta_se_v, dists_v)
        spearman_r = float(r_val)
        spearman_p = float(p_val)
    else:
        spearman_r = np.nan
        spearman_p = np.nan

    result = AmbiguityReport(
        state=state,
        dist_p50=dist_p50,
        dist_p95=dist_p95,
        delta_se_p50=delta_se_p50,
        delta_se_p95=delta_se_p95,
        spearman_r=spearman_r,
        spearman_p=spearman_p,
        n_pairs=int(valid.sum()),
        n_state_dims=X.shape[1],
    )

    kv: dict[str, object] = {
        "state": state,
        "n_state_dims": X.shape[1],
        "n_pairs": result.n_pairs,
        "threshold_y": threshold_y,
        "exclude_same_case": exclude_same_case,
        "dist_p50": f"{dist_p50:.4f}",
        "dist_p95": f"{dist_p95:.4f}",
        "delta_se_p50_J_m3_s": f"{delta_se_p50:.4e}",
        "delta_se_p95_J_m3_s": f"{delta_se_p95:.4e}",
        "spearman_r": f"{spearman_r:.4f}",
        "spearman_p": f"{spearman_p:.4e}",
        "caveat": "small-n NN floors overestimate ambiguity; use stride5 full DB for certification",
    }
    csv_rows = [{"quantity": k, "value": str(v)} for k, v in sorted(kv.items())]
    stem = f"ambiguity_{state}"
    write_report_pair(
        out_dir,
        stem,
        md_sections=[
            (md_header(f"State Ambiguity Report ({state})"), md_kv_block(kv)),
            (md_header("Row Table", level=2), md_table(csv_rows)),
        ],
        csv_rows=csv_rows,
    )
    return result


def run_ambiguity_collapse(
    df: pd.DataFrame,
    schema: Schema,
    out_dir: str | os.PathLike,
    *,
    threshold_y: float = 1e-4,
    n_neighbors: int = 1,
    exclude_same_case: bool = True,
) -> AmbiguityCollapseReport:
    """Run the Phase-1 ambiguity collapse test.

    Runs ``run_state_ambiguity`` for both ``"retained"`` and ``"full"`` states,
    then performs the dropped-subspace conditional check: at matched
    retained-distance quartiles, compares |ΔS_E| for pairs that are *close*
    vs *far* in the dropped subspace.

    A ``conditional_ratio`` substantially above 1.0 is evidence that the
    dropped species carry information the surrogate cannot see (root cause #3,
    plan §2).

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract.
    out_dir
        Output directory for all report pairs.
    threshold_y
        Max-Y threshold for ``"retained"`` mode.
    n_neighbors
        Number of neighbours (passed to ``run_state_ambiguity``).
    exclude_same_case
        Skip same-case neighbours (passed through).

    Returns
    -------
    :class:`AmbiguityCollapseReport`
    """
    ret_report = run_state_ambiguity(
        df, schema, out_dir,
        state="retained",
        threshold_y=threshold_y,
        n_neighbors=n_neighbors,
        exclude_same_case=exclude_same_case,
    )
    full_report = run_state_ambiguity(
        df, schema, out_dir,
        state="full",
        threshold_y=threshold_y,
        n_neighbors=n_neighbors,
        exclude_same_case=exclude_same_case,
    )

    collapse_detected = (
        np.isfinite(full_report.delta_se_p50)
        and np.isfinite(ret_report.delta_se_p50)
        and full_report.delta_se_p50 <= ret_report.delta_se_p50
    )

    # --- Dropped-subspace conditional check ---
    abs_col = schema.energy_target_column()
    se = df[abs_col].to_numpy(dtype=float)
    case_ids: np.ndarray | None = None
    if "CaseID" in schema.meta:
        case_ids = df[schema.meta["CaseID"]].to_numpy()
    elif "CaseID" in df.columns:
        case_ids = df["CaseID"].to_numpy()

    X_ret = _build_state_matrix(df, schema, state="retained", threshold_y=threshold_y)
    X_full = _build_state_matrix(df, schema, state="full", threshold_y=threshold_y)
    # dropped subspace = columns present in full but not retained
    # We approximate as: X_full standardised minus the retained columns
    # In practice: full has more columns; compute dropped distances separately
    n_ret_dims = X_ret.shape[1]
    X_dropped = X_full[:, n_ret_dims:]  # columns beyond retained block (if any)

    dropped_near_p50 = np.nan
    dropped_far_p50 = np.nan
    conditional_ratio = np.nan

    if X_dropped.shape[1] > 0:
        X_ret_std = _standardise(X_ret)
        X_dropped_std = _standardise(X_dropped)

        # NN distances in retained space (cross-case)
        dists_ret, inds_ret, delta_se_ret = _nn_pairs(
            X_ret_std, se, case_ids,
            n_neighbors=n_neighbors,
            exclude_same_case=exclude_same_case,
        )
        valid = np.isfinite(dists_ret) & (inds_ret >= 0)
        if valid.sum() >= 4:
            # Partition by retained-distance quartile median
            dists_v = dists_ret[valid]
            q50_ret = float(np.median(dists_v))
            idx_v = np.where(valid)[0]

            # For pairs in the lower half of retained distance:
            # compute dropped-space distance between i and its NN
            dropped_dists_for_pairs = np.array([
                float(np.linalg.norm(X_dropped_std[i] - X_dropped_std[inds_ret[i]]))
                for i in idx_v
            ])
            dropped_q50 = float(np.median(dropped_dists_for_pairs))

            delta_se_v = delta_se_ret[valid]
            near_mask = dropped_dists_for_pairs <= dropped_q50
            far_mask = ~near_mask

            if near_mask.sum() >= 2 and far_mask.sum() >= 2:
                dropped_near_p50 = float(np.median(delta_se_v[near_mask]))
                dropped_far_p50 = float(np.median(delta_se_v[far_mask]))
                conditional_ratio = (
                    dropped_far_p50 / max(dropped_near_p50, 1.0)
                )

    result = AmbiguityCollapseReport(
        retained=ret_report,
        full=full_report,
        collapse_detected=collapse_detected,
        dropped_near_delta_se_p50=dropped_near_p50,
        dropped_far_delta_se_p50=dropped_far_p50,
        conditional_ratio=conditional_ratio,
    )

    kv: dict[str, object] = {
        "collapse_detected": collapse_detected,
        "retained_delta_se_p50_J_m3_s": f"{ret_report.delta_se_p50:.4e}",
        "full_delta_se_p50_J_m3_s": f"{full_report.delta_se_p50:.4e}",
        "dropped_near_delta_se_p50": f"{dropped_near_p50:.4e}",
        "dropped_far_delta_se_p50": f"{dropped_far_p50:.4e}",
        "conditional_ratio": f"{conditional_ratio:.3f}",
        "caveat": "small-n NN floors overestimate ambiguity; ratio meaningful on full stride5 DB",
    }
    csv_rows = [{"quantity": k, "value": str(v)} for k, v in sorted(kv.items())]
    write_report_pair(
        out_dir,
        "ambiguity_collapse",
        md_sections=[
            (md_header("Ambiguity Collapse Report"), md_kv_block(kv)),
            (md_header("Row Table", level=2), md_table(csv_rows)),
        ],
        csv_rows=csv_rows,
    )
    return result

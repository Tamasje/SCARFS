"""kNN feasibility pre-gate for the SCARFS benchmark (plan §4 E-e).

This module implements the kNN feasibility table that answers: for a given
latent-space size k, does the compressed state (q + T, P) contain enough
information to predict absorption?  If the k-dimensional encoder cannot beat the
full-state kNN baseline, training with that k is informationally hopeless.

Three state-space representations are compared via out-of-fold 5-NN regression R²:

(a) Full standardised dry composition + standardised T, P  (baseline).
(b) PCA-k of (a)'s composition block + standardised T, P.
(c) PLS-k of (a) supervised by asinh(absorption) + standardised T, P  (supervised-
    encoder emulation — measures how much the energy signal itself helps).

The GroupKFold split is by CaseID for at least 3 folds (or leave-one-case-out
for tiny data), so no case contributes to both train and test within a fold.

A candidate (space, k) PASSES when its tail R² ≥ baseline tail R² − margin.

Public API
----------
- :class:`FeasibilityReport` — compact table + :meth:`~FeasibilityReport.summary`.
- :func:`feasibility_table` — main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import Schema, DILUENT_SPECIES


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityEntry:
    """One row of the feasibility table."""

    space: str              # "full", "pca", "pls"
    k: int                  # latent dimension (1 for "full")
    global_r2: float        # global out-of-fold 5-NN R²
    tail_r2: float          # tail out-of-fold 5-NN R² (top tail_fraction by |target|)
    passes: bool            # tail_r2 >= baseline_tail_r2 - margin


@dataclass
class FeasibilityReport:
    """Result of :func:`feasibility_table`.

    Attributes
    ----------
    entries
        List of :class:`FeasibilityEntry` rows.
    baseline_global_r2
        Global R² of the full-state baseline.
    baseline_tail_r2
        Tail R² of the full-state baseline.
    ks
        Latent dimensions evaluated.
    margin
        PASS margin: entry passes when tail_r2 >= baseline_tail_r2 - margin.
    n_folds
        Number of GroupKFold folds used.
    tail_fraction
        Fraction of rows (by |target|) counted as 'tail'.
    warnings
        Diagnostic messages.
    """

    entries: list[FeasibilityEntry]
    baseline_global_r2: float
    baseline_tail_r2: float
    ks: Sequence[int]
    margin: float
    n_folds: int
    tail_fraction: float
    warnings: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the feasibility table as a DataFrame."""
        rows = [
            {
                "space": e.space,
                "k": e.k,
                "global_r2": e.global_r2,
                "tail_r2": e.tail_r2,
                "passes": e.passes,
            }
            for e in self.entries
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Return a compact human-readable table."""
        lines = [
            "=" * 60,
            "FEASIBILITY PRE-GATE TABLE  (§4 E-e)",
            "=" * 60,
            f"Folds: {self.n_folds}   tail_fraction: {self.tail_fraction}   margin: {self.margin}",
            f"Baseline (full state) global R² = {self.baseline_global_r2:.4f}",
            f"Baseline (full state) tail   R² = {self.baseline_tail_r2:.4f}",
            "",
            f"{'space':6s}  {'k':>4s}  {'global_r2':>10s}  {'tail_r2':>10s}  {'pass':>6s}",
            "-" * 50,
        ]
        for e in self.entries:
            lines.append(
                f"{e.space:6s}  {e.k:4d}  {e.global_r2:10.4f}  {e.tail_r2:10.4f}  "
                f"{'PASS' if e.passes else 'FAIL':>6s}"
            )
        lines.append("=" * 60)
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² with degenerate guard."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def _tail_mask(y: np.ndarray, tail_fraction: float) -> np.ndarray:
    """Select top *tail_fraction* rows by |y|."""
    if len(y) == 0 or tail_fraction <= 0.0:
        return np.zeros(len(y), dtype=bool)
    threshold = np.percentile(np.abs(y), 100.0 * (1.0 - tail_fraction))
    return np.abs(y) >= threshold


def _knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, k: int) -> np.ndarray:
    """Brute-force k-NN regression (no scipy required).

    Uses squared Euclidean distance; averages target values of k nearest neighbours.
    For small folds (< k training samples) falls back to k = len(X_train).
    """
    k = min(k, len(X_train))
    # (n_test, n_train) squared distances
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]  # (n_t, n_tr, d)
    dist2 = np.sum(diff ** 2, axis=2)  # (n_t, n_tr)
    idx = np.argpartition(dist2, kth=min(k - 1, dist2.shape[1] - 1), axis=1)[:, :k]
    return np.mean(y_train[idx], axis=1)


def _group_kfold_masks(
    case_ids: np.ndarray,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (train_idx, test_idx) pairs for GroupKFold.

    If *n_folds* > n_unique_cases, falls back to leave-one-case-out.
    Randomises case assignment with *seed*.
    """
    unique_cases = np.unique(case_ids)
    n_cases = len(unique_cases)
    n_folds = min(n_folds, n_cases)
    if n_folds < 2:
        # degenerate: only 1 case — 80/20 row split
        n = len(case_ids)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split = max(1, int(0.8 * n))
        return [(idx[:split], idx[split:])]

    rng = np.random.default_rng(seed)
    shuffled_cases = rng.permutation(unique_cases)
    fold_assignments = {c: int(i % n_folds) for i, c in enumerate(shuffled_cases)}

    result = []
    for fold in range(n_folds):
        test_cases = {c for c, f in fold_assignments.items() if f == fold}
        test_mask = np.array([c in test_cases for c in case_ids])
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            result.append((train_idx, test_idx))
    return result


def _oof_r2(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_neighbors: int,
    tail_fraction: float,
) -> tuple[float, float]:
    """Compute out-of-fold 5-NN R² (global and tail).

    Returns ``(global_r2, tail_r2)``.
    """
    oof_pred = np.full(len(y), np.nan, dtype=float)

    for train_idx, test_idx in folds:
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        X_te = X[test_idx]
        if len(X_tr) == 0 or len(X_te) == 0:
            continue
        # standardise features on train fold
        mu = X_tr.mean(axis=0)
        sigma = X_tr.std(axis=0)
        sigma = np.where(sigma > 0, sigma, 1.0)
        X_tr_s = (X_tr - mu) / sigma
        X_te_s = (X_te - mu) / sigma
        oof_pred[test_idx] = _knn_predict(X_tr_s, y_tr, X_te_s, n_neighbors)

    valid = np.isfinite(oof_pred)
    if not valid.any():
        return np.nan, np.nan

    global_r2 = _r2_score(y[valid], oof_pred[valid])

    tail_m = _tail_mask(y, tail_fraction)
    combined = valid & tail_m
    if combined.sum() < 2:
        tail_r2 = global_r2
    else:
        tail_r2 = _r2_score(y[combined], oof_pred[combined])

    return global_r2, tail_r2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def feasibility_table(
    df: pd.DataFrame,
    schema: Schema,
    ks: Sequence[int] = (4, 6, 8, 12, 16),
    *,
    n_neighbors: int = 5,
    tail_fraction: float = 0.20,
    margin: float = 0.0,
    seed: int = 0,
) -> FeasibilityReport:
    """Compute the kNN feasibility table for a range of latent dimensions.

    For each k and each state-space representation (full / PCA-k / PLS-k),
    the out-of-fold 5-NN regression R² on absorption is evaluated to determine
    whether the compressed state retains sufficient energy-relevant information.

    Parameters
    ----------
    df
        Database rows including composition columns (``Y_*``) and the
        absorption column (resolved via ``schema.energy_target_column()``).
    schema
        Column contract for *df*.
    ks
        Latent dimensions to evaluate.  Default ``(4, 6, 8, 12, 16)``
        matching the ablation range in plan §4 E-e.
    n_neighbors
        Number of neighbours for the kNN regressor.  Default 5.
    tail_fraction
        Fraction of rows (by |target|) that define the 'tail' region.
        Default 0.20.
    margin
        A candidate (space, k) PASSES when ``tail_r2 >= baseline_tail_r2 - margin``.
        Default 0.0 (must match or exceed baseline).
    seed
        Random seed for GroupKFold fold assignment.

    Returns
    -------
    FeasibilityReport
        Table of results with :meth:`~FeasibilityReport.summary`.
    """
    from sklearn.cross_decomposition import PLSRegression  # optional dependency

    warn_list: list[str] = []

    # --- extract features and target ---
    try:
        abs_col = schema.energy_target_column()
    except KeyError as exc:
        raise KeyError(f"feasibility_table: {exc}") from exc

    target = df[abs_col].to_numpy(dtype=float)

    # dry composition: all species except diluent (H2O)
    dry_species = [s for s in schema.species if s != DILUENT_SPECIES]
    if not dry_species:
        dry_species = list(schema.species)
        warn_list.append("No dry species found (diluent exclusion produced empty set); using all species.")

    y_cols = [f"Y_{s}" for s in dry_species]
    present = [c for c in y_cols if c in df.columns]
    missing = [c for c in y_cols if c not in df.columns]
    if missing:
        warn_list.append(f"{len(missing)} composition columns absent from DataFrame; excluded.")
        present = [c for c in y_cols if c in df.columns]
    Y = df[present].to_numpy(dtype=float)

    T_col = schema.state.get("T")
    P_col = schema.state.get("P")
    T = df[T_col].to_numpy(dtype=float).reshape(-1, 1) if T_col else np.ones((len(df), 1))
    P = df[P_col].to_numpy(dtype=float).reshape(-1, 1) if P_col else np.ones((len(df), 1))

    if T_col is None:
        warn_list.append("T column not found; using constant T=1 for all features.")
    if P_col is None:
        warn_list.append("P column not found; using constant P=1 for all features.")

    # standardise composition globally (before fold splits for feature construction;
    # fold-internal standardisation is applied inside _oof_r2)
    Y_std_global = Y.copy()
    mu_Y = Y_std_global.mean(axis=0)
    sig_Y = Y_std_global.std(axis=0)
    sig_Y = np.where(sig_Y > 0, sig_Y, 1.0)
    Y_std_global = (Y_std_global - mu_Y) / sig_Y

    # standardise T and P globally
    T_s = (T - T.mean()) / max(float(T.std()), 1.0)
    P_s = (P - P.mean()) / max(float(P.std()), 1.0)

    # --- case-ID folds ---
    if "CaseID" in schema.meta:
        case_ids = df[schema.meta["CaseID"]].to_numpy()
    else:
        case_ids = np.zeros(len(df), dtype=int)
        warn_list.append("No CaseID column found; using single-case fallback (row-wise split).")

    n_unique = len(np.unique(case_ids))
    n_folds_actual = max(2, min(n_unique, 5))  # at least 3-fold or leave-one-case-out
    if n_unique < 3:
        n_folds_actual = n_unique
        warn_list.append(f"Only {n_unique} unique cases; using leave-one-case-out ({n_folds_actual} folds).")

    folds = _group_kfold_masks(case_ids, n_folds=n_folds_actual, seed=seed)

    # --- (a) full baseline ---
    X_full = np.hstack([Y_std_global, T_s, P_s])
    baseline_global, baseline_tail = _oof_r2(X_full, target, folds, n_neighbors, tail_fraction)

    entries: list[FeasibilityEntry] = []

    for k in ks:
        # (b) PCA-k of composition + T, P
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=min(int(k), Y_std_global.shape[1], Y_std_global.shape[0] - 1))
            pca.fit(Y_std_global)
            Y_pca = pca.transform(Y_std_global)
            X_pca = np.hstack([Y_pca, T_s, P_s])
            pca_global, pca_tail = _oof_r2(X_pca, target, folds, n_neighbors, tail_fraction)
        except Exception as exc:
            warn_list.append(f"PCA k={k} failed: {exc}")
            pca_global, pca_tail = np.nan, np.nan

        entries.append(FeasibilityEntry(
            space="pca",
            k=k,
            global_r2=pca_global,
            tail_r2=pca_tail,
            passes=np.isfinite(pca_tail) and pca_tail >= baseline_tail - margin,
        ))

        # (c) PLS-k supervised by asinh(target)
        try:
            pls_k = min(int(k), Y_std_global.shape[1], Y_std_global.shape[0] - 1)
            if pls_k < 1:
                raise ValueError("k < 1 for PLS")
            y_pls = np.arcsinh(target / (np.percentile(np.abs(target), 75) + 1e-12))
            pls = PLSRegression(n_components=pls_k, max_iter=500)
            pls.fit(Y_std_global, y_pls.reshape(-1, 1))
            Y_pls = pls.transform(Y_std_global)
            X_pls = np.hstack([Y_pls, T_s, P_s])
            pls_global, pls_tail = _oof_r2(X_pls, target, folds, n_neighbors, tail_fraction)
        except Exception as exc:
            warn_list.append(f"PLS k={k} failed: {exc}")
            pls_global, pls_tail = np.nan, np.nan

        entries.append(FeasibilityEntry(
            space="pls",
            k=k,
            global_r2=pls_global,
            tail_r2=pls_tail,
            passes=np.isfinite(pls_tail) and pls_tail >= baseline_tail - margin,
        ))

    return FeasibilityReport(
        entries=entries,
        baseline_global_r2=float(baseline_global),
        baseline_tail_r2=float(baseline_tail),
        ks=list(ks),
        margin=float(margin),
        n_folds=n_folds_actual,
        tail_fraction=float(tail_fraction),
        warnings=warn_list,
    )

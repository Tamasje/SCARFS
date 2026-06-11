"""Database -> scaled tensors, with importance / tail-stratified weighting. NumPy/pandas only.

This is the correctness-critical part of training (which columns become features/targets, how rows
are weighted, how the train/val/test split avoids leakage).  It is torch-free and unit-tested
locally; the PyTorch loop in ``train.py`` simply consumes the arrays produced here.

Design decisions
----------------
- **No winsorization anywhere** — energy absorption targets pass through unchanged.
- **Tail-stratified row weights** (merged model, opt-in): ``weight = 1 + α·(decile_rank / 9)``
  over ``log10(absorption + 1)`` deciles; multiplied into the inlet-weight.
- **Enthalpy-aware species weight** (merged model, opt-in): per-species weight ∝ train-mean share
  of |h_i · rate_i|, normalised so the minimum weight is 1.0.  Exposed as a separate vector;
  always computed on the train split only (no leakage).
- **70/15/15 GroupKFold-style split**: hash-based deterministic by ``(CaseID, seed)`` to avoid any
  sklearn dependency; test fold held out completely (the benchmark consumes it via the saved split).
- **Lagrangian pair index builder**: consecutive same-case rows sorted by τ, yielding
  ``(idx_t, idx_t+1, Δτ)`` for the Lagrangian rollout loss.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import DILUENT_SPECIES, MAJOR_SPECIES, Schema, y_column
from ..models.features import FeatureScalers, FeatureSpec, build_features, build_rate_targets, fit_scalers
from .config import DataConfig


# ---------------------------------------------------------------------------
# Species resolution
# ---------------------------------------------------------------------------

def resolve_species(schema: Schema, selector: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve a species selector to a concrete tuple of species names.

    Selectors:
    - ``"active"``       — molecular species, diluent excluded.
    - ``"molecular"``    — all molecular (non-radical) species.
    - ``"dry_all"``      — all species except the diluent.
    - ``"all"``          — all species.
    - ``"energy_active"``— placeholder; resolved by :func:`select_energy_active_species`
                           inside the merged training path (requires thermo); raises here.
    - explicit list      — validated against the schema.
    """
    if isinstance(selector, str):
        if selector == "active":
            return schema.active_species()
        if selector == "molecular":
            return schema.molecular_species()
        if selector == "dry_all":
            return tuple(s for s in schema.species if s != DILUENT_SPECIES)
        if selector == "all":
            return schema.species
        if selector == "energy_active":
            raise ValueError(
                "selector='energy_active' must be resolved by select_energy_active_species() "
                "inside the merged training path (requires thermo and train data)."
            )
        raise ValueError(f"Unknown species selector {selector!r}")
    present = set(schema.species)
    missing = [s for s in selector if s not in present]
    if missing:
        raise ValueError(f"Requested species absent from database: {missing}")
    return tuple(selector)


# ---------------------------------------------------------------------------
# Conversion / inlet weighting (unchanged from baseline)
# ---------------------------------------------------------------------------

def compute_conversion(df: pd.DataFrame, schema: Schema, key_species: str = "C2H6") -> np.ndarray:
    """Per-row conversion of *key_species* relative to its per-case inlet (min-z) value.

    Returns an array of conversions in ``[0, 1]`` (clipped).  If the key species or the
    ``CaseID``/``z`` columns are unavailable, returns zeros and warns (never fabricates).
    """
    col = y_column(key_species)
    if col not in df.columns or "CaseID" not in schema.meta or "z" not in schema.state:
        warnings.warn("compute_conversion: missing C2H6/CaseID/z columns; returning zeros.")
        return np.zeros(len(df), dtype=float)
    case_col = schema.meta["CaseID"]
    z_col = schema.state["z"]
    inlet = df.loc[df.groupby(case_col)[z_col].idxmin(), [case_col, col]].set_index(case_col)[col]
    y_in = df[case_col].map(inlet).to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        conv = np.where(y_in > 0, (y_in - y) / y_in, 0.0)
    return np.clip(conv, 0.0, 1.0)


def importance_weights(df: pd.DataFrame, schema: Schema, cfg: DataConfig) -> np.ndarray:
    """Per-row training weights that up-weight near-inlet / low-conversion states (RC-1/F1)."""
    conv = compute_conversion(df, schema)
    w = np.ones(len(df), dtype=float)
    w[conv < cfg.inlet_conversion_threshold] = cfg.inlet_weight
    return w


def species_weight_vector(target_species: Sequence[str], species_weights: dict[str, float]) -> np.ndarray:
    """Per-target weight vector (default 1.0), e.g. emphasising C2H4/C3H6."""
    return np.array([float(species_weights.get(s, 1.0)) for s in target_species], dtype=float)


# ---------------------------------------------------------------------------
# 70 / 15 / 15 deterministic case split
# ---------------------------------------------------------------------------

def _case_hash(case_id: Any, seed: int) -> float:  # noqa: ANN001
    """Return a deterministic pseudo-random float in [0, 1) for a (case_id, seed) pair.

    Using SHA-256 so the mapping is stable across Python versions and machines.
    """
    raw = f"{case_id}_{seed}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16) / (16 ** 16)


def tripartite_case_split(
    df: pd.DataFrame,
    schema: Schema,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    split_by_case: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean (train, val, test) masks with no case leakage.

    When ``test_fraction = 0`` the test mask is all False and the existing 80/20 behaviour is
    reproduced exactly (backward compatible).

    The split is deterministic and hash-based — no sklearn dependency required.  Cases with
    hash < test_fraction go to test; of the remainder, cases with rescaled hash < val_fraction
    go to val; the rest are train.
    """
    n = len(df)
    if not split_by_case or "CaseID" not in schema.meta:
        rng = np.random.default_rng(seed)
        r = rng.random(n)
        test_mask = r < test_fraction
        remaining = ~test_mask
        r2 = rng.random(n)
        val_mask = remaining & (r2 < val_fraction)
        train_mask = ~test_mask & ~val_mask
        return train_mask, val_mask, test_mask

    case_col = schema.meta["CaseID"]
    cases = df[case_col].to_numpy()
    unique_cases = np.unique(cases)

    # hash each case to [0, 1)
    hashes = np.array([_case_hash(c, seed) for c in unique_cases])
    test_cases: set = set(unique_cases[hashes < test_fraction].tolist())
    # rescale remaining hashes to decide val
    remaining_mask = hashes >= test_fraction
    val_cases: set = set()
    if remaining_mask.any():
        h_rem = hashes[remaining_mask]
        # re-normalise: map [test_fraction, 1] -> [0, 1]
        span = 1.0 - test_fraction
        h_scaled = (h_rem - test_fraction) / span if span > 0 else h_rem
        val_set_cases = unique_cases[remaining_mask][h_scaled < val_fraction]
        val_cases = set(val_set_cases.tolist())

    test_mask = np.array([c in test_cases for c in cases])
    val_mask = np.array([c in val_cases for c in cases])
    train_mask = ~test_mask & ~val_mask
    return train_mask, val_mask, test_mask


def case_aware_split(
    df: pd.DataFrame, schema: Schema, val_fraction: float, seed: int, split_by_case: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean (train, val) masks — backward-compatible wrapper around tripartite split.

    When ``split_by_case`` and a ``CaseID`` column exist, whole cases go to one side (prevents the
    same PFR profile leaking across the split).  Otherwise rows are split randomly.
    """
    train_mask, val_mask, _ = tripartite_case_split(
        df, schema, val_fraction=val_fraction, test_fraction=0.0, seed=seed, split_by_case=split_by_case
    )
    return train_mask, val_mask


# ---------------------------------------------------------------------------
# Tail-stratified row weights
# ---------------------------------------------------------------------------

def tail_stratified_weights(
    df: pd.DataFrame,
    schema: Schema,
    *,
    tail_strata: int = 10,
    tail_weight_alpha: float = 2.0,
    absorption_col: str = "Reaction heat absorption [J/s/m3]",
) -> np.ndarray:
    """Per-row tail weights over log10(absorption + 1) deciles.

    weight_i = 1 + alpha * (decile_rank_i / (tail_strata - 1))

    Rows in the lowest decile (rank 0) get weight 1.0; rows in the top decile get
    weight 1 + alpha.  This is additive to the inlet weight (caller multiplies).

    Parameters
    ----------
    df
        DataFrame slice (train split only — fit on train, no leakage).
    schema
        Ignored (absorption_col resolved directly); kept for API symmetry.
    tail_strata
        Number of equal-count bins over log10(absorption + 1).  Must be >= 2.
        Pass 0 to skip (returns ones).
    tail_weight_alpha
        Alpha parameter (see formula above).
    absorption_col
        Column name for the energy absorption target.
    """
    if tail_strata <= 0:
        return np.ones(len(df), dtype=float)

    # Prefer the schema-resolved energy target (unit-suffix tolerant); the keyword stays as a
    # fallback for frames whose schema carries no absorption column.
    try:
        resolved = schema.energy_target_column()
        if resolved in df.columns:
            absorption_col = resolved
    except (KeyError, AttributeError):
        pass
    if absorption_col not in df.columns:
        warnings.warn(f"tail_stratified_weights: column '{absorption_col}' not found; returning ones.")
        return np.ones(len(df), dtype=float)

    absorption = df[absorption_col].to_numpy(dtype=float)
    # clip negatives to 0 (design: absorption strictly positive; count handled upstream)
    absorption = np.clip(absorption, 0.0, None)
    log_abs = np.log10(absorption + 1.0)  # +1 to handle zero safely

    # compute equal-count decile ranks
    n = len(df)
    sorted_idx = np.argsort(log_abs)
    ranks = np.empty(n, dtype=float)
    for rank_i, idx in enumerate(sorted_idx):
        ranks[idx] = rank_i

    # map [0, n-1] -> [0, tail_strata-1] using quantile
    bins = np.quantile(log_abs, np.linspace(0, 1, tail_strata + 1))
    bins[0] -= 1e-9  # ensure leftmost bin is inclusive
    decile_rank = np.searchsorted(bins[1:], log_abs, side="left")
    decile_rank = np.clip(decile_rank, 0, tail_strata - 1)

    weights = 1.0 + tail_weight_alpha * (decile_rank.astype(float) / (tail_strata - 1))
    return weights


# ---------------------------------------------------------------------------
# Enthalpy-aware species weight vector
# ---------------------------------------------------------------------------

def enthalpy_aware_weights(
    df: pd.DataFrame,
    schema: Schema,
    target_species: Sequence[str],
    *,
    h_mass_fn=None,
    floor: float = 1.0,
) -> np.ndarray:
    """Per-species weight vector ∝ train-mean share of |h_i · rate_i| (ONLY on physical-rate head).

    If ``h_mass_fn`` is None or enthalpy data is unavailable, returns uniform weights (all 1.0).
    The weight vector is normalised so the minimum is ``floor`` (default 1.0).

    Parameters
    ----------
    df
        Training-split DataFrame.
    schema
        Column contract for *df*.
    target_species
        Species whose rates are predicted (energy-active set for the merged model).
    h_mass_fn
        Optional callable ``h_mass_fn(T) -> (n_rows, n_species)`` giving specific enthalpies
        [J/kg] for the target_species in order.  When provided, weights are computed from
        the mean |h_i * R_i| share.  Expected interface: B1b ``SpeciesThermo.h_mass``.
    floor
        Minimum weight value (normalisation floor).
    """
    if h_mass_fn is None:
        return np.full(len(target_species), floor, dtype=float)

    # get T values
    try:
        (t_col,) = schema.require_state("T")
        T = df[t_col].to_numpy(dtype=float)
    except KeyError:
        warnings.warn("enthalpy_aware_weights: T column missing; returning uniform weights.")
        return np.full(len(target_species), floor, dtype=float)

    # get MASS rates for target_species — ρ·dYdt on the parquet (legacy R_ columns there are raw
    # kmol m-3 s-1 and would skew the |h·ω| shares by molar-mass factors).
    try:
        from ..models.features import build_mass_rate_matrix
        rates = build_mass_rate_matrix(df, schema, target_species, prefer_dydt=True)
    except (KeyError, ValueError) as exc:
        warnings.warn(f"enthalpy_aware_weights: rate columns unavailable ({exc}); returning uniform.")
        return np.full(len(target_species), floor, dtype=float)

    try:
        h = h_mass_fn(T)  # (n, n_species)
        h = np.asarray(h, dtype=float)
        if h.ndim == 1:
            h = h[np.newaxis, :]  # broadcast
    except Exception as exc:
        warnings.warn(f"enthalpy_aware_weights: h_mass_fn failed ({exc}); returning uniform.")
        return np.full(len(target_species), floor, dtype=float)

    # mean |h_i * R_i| per species
    energy_per_species = np.abs(h * rates).mean(axis=0)  # (n_species,)
    total = energy_per_species.sum()
    if total <= 0:
        return np.full(len(target_species), floor, dtype=float)

    shares = energy_per_species / total  # normalised shares
    # normalise so minimum == floor
    min_share = shares.min()
    if min_share <= 0:
        return np.full(len(target_species), floor, dtype=float)
    weights = (shares / min_share) * floor
    weights = np.clip(weights, floor, None)
    return weights


# ---------------------------------------------------------------------------
# Lagrangian pair index builder
# ---------------------------------------------------------------------------

def case_step_pairs(
    df: pd.DataFrame,
    schema: Schema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build index pairs for the Lagrangian rollout loss.

    For each case, rows are sorted by τ.  Consecutive same-case rows (sorted by τ) yield a pair
    ``(idx_t, idx_{t+1}, Δτ)`` where ``Δτ = τ_{t+1} - τ_t > 0``.

    Parameters
    ----------
    df
        DataFrame that contains the τ/CaseID columns (whole train split, index reset to 0..n-1).
    schema
        Column contract.

    Returns
    -------
    idx_t : ndarray of int, shape (n_pairs,)
        Row indices of the *current* step.
    idx_tp1 : ndarray of int, shape (n_pairs,)
        Row indices of the *next* step.
    dtau : ndarray of float, shape (n_pairs,)
        Δτ [s] (strictly positive).

    Raises
    ------
    KeyError
        If CaseID or tau/z columns are missing from *schema*.
    """
    if "CaseID" not in schema.meta:
        raise KeyError("case_step_pairs: 'CaseID' not in schema.meta")
    # prefer tau if present, fall back to z
    if "tau" in schema.state:
        tau_col = schema.state["tau"]
    elif "z" in schema.state:
        tau_col = schema.state["z"]
    else:
        raise KeyError("case_step_pairs: neither 'tau' nor 'z' found in schema.state")

    case_col = schema.meta["CaseID"]
    df = df.reset_index(drop=True)

    idx_t_list: list[int] = []
    idx_tp1_list: list[int] = []
    dtau_list: list[float] = []

    for case_id, group in df.groupby(case_col, sort=False):
        sorted_group = group.sort_values(tau_col)
        idxs = sorted_group.index.to_numpy()
        taus = sorted_group[tau_col].to_numpy(dtype=float)
        for i in range(len(idxs) - 1):
            dt = taus[i + 1] - taus[i]
            if dt > 0:
                idx_t_list.append(int(idxs[i]))
                idx_tp1_list.append(int(idxs[i + 1]))
                dtau_list.append(float(dt))

    return (
        np.array(idx_t_list, dtype=np.intp),
        np.array(idx_tp1_list, dtype=np.intp),
        np.array(dtau_list, dtype=float),
    )


# ---------------------------------------------------------------------------
# DataBundle and prepare_data
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Prepared training arrays + the fitted scalers and resolved spec."""

    X_train: np.ndarray
    Y_train: np.ndarray
    w_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    w_val: np.ndarray
    species_weights: np.ndarray
    scalers: FeatureScalers
    spec: FeatureSpec
    schema: Schema
    # -- merged-model additions (None when not computed) --
    #: Enthalpy-aware per-species weight vector (energy-active species; train split only).
    enthalpy_weights: np.ndarray | None = None
    #: Lagrangian pair indices (idx_t, idx_tp1, dtau) built from the train split.
    lagrangian_pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    #: Boolean mask of test cases rows (held-out; not included in X/Y arrays above).
    test_case_ids: list | None = None
    #: Row indices of train and val in the original df (for benchmark split tracing).
    train_indices: np.ndarray | None = None
    val_indices: np.ndarray | None = None


def prepare_data(
    cfg: DataConfig,
    df: pd.DataFrame,
    schema: Schema,
    *,
    composition_kwargs: dict | None = None,
    prefer_dydt: bool = False,
) -> DataBundle:
    """Build a :class:`DataBundle` from a loaded DataFrame and its schema.

    Scalers are fit on the **train** split only (no leakage).  Importance weights combine the
    per-row near-inlet weight (F1) and, when enabled, the tail-stratified absorption weight.

    When ``cfg.test_fraction > 0`` the split uses the 70/15/15 tripartite logic; otherwise falls
    back to the existing 80/20 train/val behaviour (full backward compatibility).

    ``composition_kwargs`` overrides the :class:`CompositionScaler` construction entirely (the
    merged model passes ``{"log": False, "mode": "standard"}`` so X is built with the SAME scaler
    that is exported — never fit one scaler and ship another).  ``prefer_dydt=True`` makes targets
    mass rates ``ρ·dY/dt`` on dYdt-carrying databases (see ``features.fit_scalers``).
    """
    input_species = resolve_species(schema, cfg.input_species)
    # "energy_active" deferred to train.py; use "active" fallback here
    target_sel = cfg.target_species
    if isinstance(target_sel, str) and target_sel == "energy_active":
        # will be replaced in train.py after species selection; use active as placeholder
        target_species = schema.active_species()
    else:
        target_species = resolve_species(schema, target_sel)

    if cfg.test_fraction > 0.0:
        train_mask, val_mask, test_mask = tripartite_case_split(
            df, schema,
            val_fraction=cfg.val_fraction,
            test_fraction=cfg.test_fraction,
            seed=cfg.seed,
            split_by_case=cfg.split_by_case,
        )
    else:
        train_mask, val_mask = case_aware_split(df, schema, cfg.val_fraction, cfg.seed, cfg.split_by_case)
        test_mask = np.zeros(len(df), dtype=bool)

    df_train = df.loc[train_mask].reset_index(drop=True)
    df_val = df.loc[val_mask].reset_index(drop=True)

    scalers = fit_scalers(
        df_train, schema, input_species, target_species,
        composition_kwargs=(
            composition_kwargs if composition_kwargs is not None
            else {"feature_range": cfg.composition_feature_range}
        ),
        prefer_dydt=prefer_dydt,
    )

    # per-row importance weights (inlet up-weight)
    w_train = importance_weights(df_train, schema, cfg)
    w_val = importance_weights(df_val, schema, cfg) if len(df_val) else np.empty((0,))

    # tail-stratified weights (opt-in; no winsorization)
    if cfg.tail_strata > 0:
        tail_w = tail_stratified_weights(
            df_train, schema,
            tail_strata=cfg.tail_strata,
            tail_weight_alpha=cfg.tail_weight_alpha,
        )
        w_train = w_train * tail_w

    n_in = len(input_species)
    n_target = len(target_species)
    bundle = DataBundle(
        X_train=build_features(df_train, schema, scalers),
        Y_train=build_rate_targets(df_train, schema, scalers),
        w_train=w_train,
        X_val=build_features(df_val, schema, scalers) if len(df_val) else np.empty((0, n_in + 4)),
        Y_val=build_rate_targets(df_val, schema, scalers) if len(df_val) else np.empty((0, n_target)),
        w_val=w_val,
        species_weights=species_weight_vector(target_species, cfg.species_weights),
        scalers=scalers,
        spec=FeatureSpec(input_species=input_species, target_species=target_species),
        schema=schema,
        train_indices=np.where(train_mask)[0],
        val_indices=np.where(val_mask)[0],
    )

    # test case IDs (for benchmark split tracing)
    if test_mask.any() and "CaseID" in schema.meta:
        case_col = schema.meta["CaseID"]
        bundle.test_case_ids = sorted(set(df.loc[test_mask, case_col].tolist()))

    return bundle


# ---------------------------------------------------------------------------
# Typing helper (re-exported for test stubs)
# ---------------------------------------------------------------------------
from typing import Any  # noqa: E402 – placed here to avoid shadowing above

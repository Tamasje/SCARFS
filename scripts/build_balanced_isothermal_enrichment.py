#!/usr/bin/env python3
# SCRIPT_VERSION = "v3_1600K_coverage_refine_print_cases"
"""
Build a lean, balanced ChemZIP-like training database for ethane steam cracking.

Purpose
-------
This script is meant to sit next to your existing SCARFS / CRACKSIM / PCAfold / CMLM
code and work with the large databases you already generated:

    full.parquet
    offmanifold_1000000.parquet

It does four things:

1. Reads only a controllable slice/sample of the large parquet files to map the
   already-covered thermo-chemical state space.
2. Designs additional *isothermal constant-pressure ethane steam cracking cases*
   in low-filled regions instead of blindly adding more PFR trajectories.
3. Optionally solves those enrichment cases with Cantera if a mechanism is supplied.
   If Cantera is not available, it still writes a manifest that can be consumed by
   your CRACKSIM / SCARFS harness.
4. Creates a balanced training parquet by down-sampling oversampled bins and adding
   a sample_weight column, plus paper-quality coverage plots.

Why isothermal enrichment?
-------------------------
The old heat-input PFRs are excellent for physics-consistent trajectories, but they
only populate a thin correlated manifold: T, conversion, heat-release/absorption,
and residence time are tied together by the imposed thermal history. A CFD-coupled
latent chemistry model does not only see physically ideal PFR histories; it sees
near-wall, high-gradient, partially converted, and numerically perturbed states.
Isothermal reactors are a cheap way to independently sweep temperature, pressure,
steam dilution, and residence time, and therefore cover the state space with fewer
cases. Keep heat-input PFRs for validation/certification and for deliberately testing
wall-heat-flux edge cases.

Typical usage
-------------
Fast coverage design only, no Cantera solve:

    python scripts/build_balanced_isothermal_enrichment.py \
        --full full.parquet \
        --off offmanifold_1000000.parquet \
        --out out_balanced_iso \
        --mode design \
        --max-existing-rows 300000 \
        --n-candidates 50000 \
        --n-new-cases 2500

Solve selected isothermal enrichment cases with Cantera:

    python scripts/build_balanced_isothermal_enrichment.py \
        --full full.parquet \
        --off offmanifold_1000000.parquet \
        --out out_balanced_iso \
        --mode solve \
        --mech chem.yaml \
        --n-time-points 160

Merge and balance for training:

    python scripts/build_balanced_isothermal_enrichment.py \
        --full full.parquet \
        --off offmanifold_1000000.parquet \
        --enriched out_balanced_iso/enriched_isothermal.parquet \
        --out out_balanced_iso \
        --mode balance \
        --bin-cap 250

All-in-one, if Cantera is available:

    python scripts/build_balanced_isothermal_enrichment.py \
        --full full.parquet \
        --off offmanifold_1000000.parquet \
        --out out_balanced_iso \
        --mode all \
        --mech chem.yaml

Notes for integration with your existing repo
---------------------------------------------
- The script is intentionally standalone. It does not import scarfs.data.generation_v2,
  because your CRACKSIM DLL workflow is Windows-specific and stateful.
- The manifest it writes is deliberately simple JSON/CSV. CODEX can wire this manifest
  into your existing generation_v2.run_case_v2(...) or CRACKSIM harness later.
- Column detection is heuristic but robust for common columns such as:
  T / Temperature [K], p / Pressure [Pa], tau / Residence time [s], Y_C2H6,
  Y_H2O, Y_C2H4, dYdt_*, and Reaction heat absorption [J/s/m3].
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires pyarrow. Install with: pip install pyarrow") from exc

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires scikit-learn. Install with: pip install scikit-learn") from exc

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires matplotlib. Install with: pip install matplotlib") from exc


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EPS = 1.0e-300

TEMP_CANDIDATES = [
    "T", "T [K]", "Temperature", "Temperature [K]", "temperature", "temperature_K", "T_K",
]
PRESSURE_CANDIDATES = [
    "p", "P", "Pressure", "Pressure [Pa]", "pressure", "pressure_Pa", "P_Pa",
]
TAU_CANDIDATES = [
    "tau", "tau [s]", "Residence time [s]", "residence_time_s", "time", "t", "t [s]",
]
HEAT_CANDIDATES = [
    "Reaction heat absorption [J/s/m3]",
    "Reaction heat absorption [J/m3/s]",
    "heat_absorption",
    "S_E",
    "energy_source",
]

SPECIES_ALIASES = {
    "C2H6": ["C2H6", "ETHANE"],
    "H2O": ["H2O", "WATER"],
    "C2H4": ["C2H4", "ETHYLENE"],
    "CH4": ["CH4", "METHANE"],
    "H2": ["H2", "HYDROGEN"],
    "C2H2": ["C2H2", "ACETYLENE"],
    "C3H6": ["C3H6", "PROPYLENE", "C3H6-1"],
    "C3H8": ["C3H8", "PROPANE"],
}

DEFAULT_FEATURES = [
    "T_K",
    "log10_p_Pa",
    "log10_tau_s",
    "conversion_proxy",
    "log10_y_c2h6",
    "log10_y_h2o",
    "log10_product_pool",
]


@dataclass
class DesignConfig:
    seed: int = 20260704
    max_existing_rows: int = 300_000
    n_candidates: int = 50_000
    n_new_cases: int = 2_500
    # Safety cap per command requested by user: never design more than this many
    # additional cases in a single run. Keep <= 5000 from the bash command.
    max_new_cases_per_run: int = 5_000
    # After the first low-coverage selection, re-check the T-conversion holes and
    # add targeted fillers until the bin-count target is met or the cap is reached.
    coverage_refine_rounds: int = 4
    target_gap_min_bin_count: int = 4
    tx_T_bin_width_K: float = 50.0
    tx_X_bin_width: float = 0.05
    n_time_points: int = 160
    # Conventional ethane steam cracking + CFD-near-wall safety margin.
    # New enrichment states are hard-capped by T_hard_max_K.
    T_min_K: float = 800.0
    T_max_K: float = 1450.0
    T_wall_max_K: float = 1600.0
    T_hard_max_K: float = 1600.0
    final_T_max_K: float = 1600.0
    focus_gap_fraction: float = 0.60
    pressure_min_Pa: float = 80_000.0
    pressure_max_Pa: float = 600_000.0
    steam_ethane_min_mass: float = 0.20
    steam_ethane_max_mass: float = 1.20
    tau_min_s: float = 1.0e-5
    tau_max_s: float = 2.0
    wall_extreme_fraction: float = 0.15
    bin_count_per_feature: int = 12
    bin_cap: int = 250
    batch_size: int = 65_536


# -----------------------------------------------------------------------------
# Utilities: column detection and parquet sampling
# -----------------------------------------------------------------------------


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _is_numeric_arrow_type(t: pa.DataType) -> bool:
    return pa.types.is_floating(t) or pa.types.is_integer(t)


def _existing_paths(paths: Sequence[str | Path | Sequence[str | Path] | None]) -> list[Path]:
    """Flatten path arguments and keep only existing paths.

    --enriched is repeatable, so this helper also accepts nested lists/tuples.
    """
    out: list[Path] = []
    for p in paths:
        if not p:
            continue
        if isinstance(p, (list, tuple, set)):
            out.extend(_existing_paths(list(p)))
            continue
        pp = Path(p)
        if pp.exists():
            out.append(pp)
        else:
            warnings.warn(f"Input path does not exist and will be skipped: {pp}")
    # de-duplicate while preserving order
    dedup: list[Path] = []
    seen: set[str] = set()
    for pp in out:
        key = str(pp.resolve()) if pp.exists() else str(pp)
        if key not in seen:
            seen.add(key)
            dedup.append(pp)
    return dedup


def get_schema_columns(paths: Sequence[Path]) -> dict[str, pa.DataType]:
    merged: dict[str, pa.DataType] = {}
    for p in paths:
        dataset = ds.dataset(str(p), format="parquet")
        for field in dataset.schema:
            if field.name not in merged:
                merged[field.name] = field.type
    return merged


def find_first_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    exact = {c: c for c in columns}
    for c in candidates:
        if c in exact:
            return exact[c]
    nmap = {_norm_name(c): c for c in columns}
    for c in candidates:
        nc = _norm_name(c)
        if nc in nmap:
            return nmap[nc]
    return None


def species_column_map(columns: Sequence[str]) -> dict[str, str]:
    """Map canonical species names to likely mass/mole fraction columns."""
    norm_to_col = {_norm_name(c): c for c in columns}
    out: dict[str, str] = {}
    prefixes = ["Y_", "X_", "Y-", "X-", "Mass fraction ", "Mole fraction "]

    for canonical, aliases in SPECIES_ALIASES.items():
        candidates = []
        for alias in aliases:
            candidates.extend([
                alias,
                f"Y_{alias}", f"X_{alias}", f"Y-{alias}", f"X-{alias}",
                f"Mass fraction {alias}", f"Mole fraction {alias}",
                f"mass_fraction_{alias}", f"mole_fraction_{alias}",
            ])
        for cand in candidates:
            nc = _norm_name(cand)
            if nc in norm_to_col:
                out[canonical] = norm_to_col[nc]
                break
        if canonical not in out:
            # Last resort: a column that ends with the species name and starts with Y/X.
            for c in columns:
                cn = _norm_name(c)
                an = _norm_name(canonical)
                if cn.endswith(an) and any(_norm_name(c).startswith(_norm_name(p)) for p in prefixes):
                    out[canonical] = c
                    break
    return out


def select_state_columns(schema_cols: dict[str, pa.DataType]) -> list[str]:
    cols = list(schema_cols)
    numeric_cols = [c for c, t in schema_cols.items() if _is_numeric_arrow_type(t)]
    selected: set[str] = set()
    for group in [TEMP_CANDIDATES, PRESSURE_CANDIDATES, TAU_CANDIDATES, HEAT_CANDIDATES]:
        c = find_first_column(cols, group)
        if c and c in numeric_cols:
            selected.add(c)
    for c in species_column_map(cols).values():
        if c in numeric_cols:
            selected.add(c)
    # Keep common scalar metadata if present.
    for c in numeric_cols:
        n = _norm_name(c)
        if n in {"caseid", "id", "z", "zms", "x", "samplekindid"}:
            selected.add(c)
    return sorted(selected)


def stream_parquet_batches(
    paths: Sequence[Path], columns: Sequence[str] | None, batch_size: int
) -> Iterable[pd.DataFrame]:
    for p in paths:
        dataset = ds.dataset(str(p), format="parquet")
        available = set(dataset.schema.names)
        use_cols = None if columns is None else [c for c in columns if c in available]
        scanner = dataset.scanner(columns=use_cols, batch_size=batch_size)
        for batch in scanner.to_batches():
            if batch.num_rows:
                yield batch.to_pandas()


def sample_parquet_rows(
    paths: Sequence[Path],
    columns: Sequence[str],
    max_rows: int,
    batch_size: int,
    seed: int,
) -> pd.DataFrame:
    """Reservoir-like streaming sample from one or more parquet files."""
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    total_seen = 0
    for batch in stream_parquet_batches(paths, columns, batch_size=batch_size):
        n = len(batch)
        total_seen += n
        if n == 0:
            continue
        if len(pieces) == 0 and n >= max_rows:
            pieces = [batch.sample(max_rows, random_state=seed)]
            break
        # Oversample modestly during streaming, then downsample at the end.
        keep_prob = min(1.0, max_rows / max(total_seen, 1) * 1.5)
        if keep_prob < 1.0:
            mask = rng.random(n) < keep_prob
            batch = batch.loc[mask]
        pieces.append(batch)
        current = sum(len(x) for x in pieces)
        if current > max_rows * 3:
            merged = pd.concat(pieces, ignore_index=True)
            pieces = [merged.sample(max_rows, random_state=int(rng.integers(0, 2**31 - 1)))]
    if not pieces:
        return pd.DataFrame(columns=columns)
    merged = pd.concat(pieces, ignore_index=True)
    if len(merged) > max_rows:
        merged = merged.sample(max_rows, random_state=seed)
    return merged.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------


def build_feature_frame(df: pd.DataFrame, *, fallback_tau_s: float = 1.0e-3) -> tuple[pd.DataFrame, dict]:
    """Build coverage features from a raw dataframe.

    Returns
    -------
    features: dataframe with standardized semantic columns.
    meta: detected source column names.
    """
    cols = list(df.columns)
    meta: dict[str, str | None] = {}
    cT = find_first_column(cols, TEMP_CANDIDATES)
    cp = find_first_column(cols, PRESSURE_CANDIDATES)
    ctau = find_first_column(cols, TAU_CANDIDATES)
    cheat = find_first_column(cols, HEAT_CANDIDATES)
    smap = species_column_map(cols)
    meta.update({"T": cT, "p": cp, "tau": ctau, "heat": cheat, "species": smap})

    if cT is None:
        raise ValueError(
            "Could not detect a temperature column. Rename it to e.g. 'T', 'T [K]', or 'Temperature [K]'."
        )

    f = pd.DataFrame(index=df.index)
    f["T_K"] = pd.to_numeric(df[cT], errors="coerce")
    if cp is None:
        f["p_Pa"] = np.nan
        f["log10_p_Pa"] = math.log10(101325.0)
    else:
        f["p_Pa"] = pd.to_numeric(df[cp], errors="coerce")
        f["log10_p_Pa"] = np.log10(np.clip(f["p_Pa"].to_numpy(float), 1.0, None))
    if ctau is None:
        f["tau_s"] = fallback_tau_s
    else:
        f["tau_s"] = pd.to_numeric(df[ctau], errors="coerce")
    f["log10_tau_s"] = np.log10(np.clip(f["tau_s"].to_numpy(float), 1.0e-12, None))

    for sp in SPECIES_ALIASES:
        col = smap.get(sp)
        if col is not None:
            f[f"y_{sp.lower()}"] = pd.to_numeric(df[col], errors="coerce").clip(lower=0.0)
        else:
            f[f"y_{sp.lower()}"] = 0.0

    y_c2h6 = f["y_c2h6"].to_numpy(float)
    y_h2o = f["y_h2o"].to_numpy(float)
    # Use a robust high quantile as the unconverted-feed proxy, because full/off-manifold
    # files may include already-perturbed data.
    feed_ref = np.nanquantile(y_c2h6[y_c2h6 > 0], 0.98) if np.any(y_c2h6 > 0) else 1.0
    f["conversion_proxy"] = np.clip(1.0 - y_c2h6 / max(feed_ref, 1.0e-20), 0.0, 1.2)
    product_pool = (
        f["y_c2h4"].to_numpy(float)
        + f["y_ch4"].to_numpy(float)
        + f["y_h2"].to_numpy(float)
        + f["y_c2h2"].to_numpy(float)
        + f["y_c3h6"].to_numpy(float)
    )
    f["product_pool"] = product_pool
    f["log10_product_pool"] = np.log10(np.clip(product_pool, 1.0e-12, None))
    f["log10_y_c2h6"] = np.log10(np.clip(y_c2h6, 1.0e-12, None))
    f["log10_y_h2o"] = np.log10(np.clip(y_h2o, 1.0e-12, None))
    f["steam_to_ethane_mass_proxy"] = y_h2o / np.clip(y_c2h6, 1.0e-12, None)
    f["log10_steam_to_ethane_mass_proxy"] = np.log10(
        np.clip(f["steam_to_ethane_mass_proxy"].to_numpy(float), 1.0e-6, None)
    )
    if cheat is not None:
        h = pd.to_numeric(df[cheat], errors="coerce")
        f["heat_absorption"] = h
        f["signed_log10_abs_heat"] = np.sign(h) * np.log10(np.clip(np.abs(h), 1.0, None))
    else:
        f["heat_absorption"] = np.nan
        f["signed_log10_abs_heat"] = np.nan

    # Drop rows with impossible / missing T.
    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.dropna(subset=["T_K"])
    return f.reset_index(drop=True), meta


def feature_matrix(features: pd.DataFrame, feature_cols: Sequence[str] = DEFAULT_FEATURES) -> np.ndarray:
    present = [c for c in feature_cols if c in features.columns]
    if not present:
        raise ValueError("No feature columns available for coverage analysis.")
    X = features[present].to_numpy(float)
    # Fill remaining NaNs with column medians.
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = ~np.isfinite(X)
    if np.any(idx):
        X[idx] = np.take(med, np.where(idx)[1])
    return X


# -----------------------------------------------------------------------------
# Enrichment design
# -----------------------------------------------------------------------------


def sobol_or_random(n: int, d: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats import qmc

        m = int(math.ceil(math.log2(max(n, 2))))
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        u = sampler.random_base2(m=m)[:n]
        return np.asarray(u)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, d))


def _composition_from_conversion(conv: np.ndarray, steam_to_ethane_mass: np.ndarray) -> dict[str, np.ndarray]:
    """Approximate mass-fraction composition for arbitrary state-probe enrichment.

    This is deliberately not a stoichiometric reactor model. It creates plausible
    product-rich or reactant-rich states so the rate surrogate sees the states that
    CFD may visit, including off-manifold near-wall states. Cantera/CRACKSIM will
    then evaluate the true source terms at those states.
    """
    conv = np.clip(np.asarray(conv, dtype=float), 0.0, 0.999999)
    s2e = np.clip(np.asarray(steam_to_ethane_mass, dtype=float), 1.0e-8, None)
    ethane_unconverted = 1.0 - conv
    product_mass = conv
    steam_mass = s2e
    # Product distribution for steam-cracking-like converted hydrocarbon mass.
    # Kept broad on purpose; the balancing stage and off-manifold data will decide weights.
    splits = {
        "C2H4": 0.58,
        "CH4": 0.18,
        "H2": 0.04,
        "C2H2": 0.08,
        "C3H6": 0.07,
        "C3H8": 0.02,
        "C2H6": 0.03,  # small back-mixing / residual ethane-like component
    }
    total = ethane_unconverted + product_mass + steam_mass
    y = {"H2O": steam_mass / total}
    y["C2H6"] = (ethane_unconverted + splits["C2H6"] * product_mass) / total
    for sp, frac in splits.items():
        if sp == "C2H6":
            continue
        y[sp] = frac * product_mass / total
    # Numerical renormalization.
    summ = np.zeros_like(conv, dtype=float)
    for arr in y.values():
        summ += arr
    for sp in y:
        y[sp] = np.clip(y[sp] / np.clip(summ, 1.0e-300, None), 0.0, 1.0)
    return y


def _assemble_candidate_frame(
    *,
    T: np.ndarray,
    p: np.ndarray,
    tau: np.ndarray,
    s2e: np.ndarray,
    conv: np.ndarray,
    design_region: str,
    design_kind: str,
    start_case_id: int,
) -> pd.DataFrame:
    y = _composition_from_conversion(conv, s2e)
    product_pool = y.get("C2H4", 0.0) + y.get("CH4", 0.0) + y.get("H2", 0.0) + y.get("C2H2", 0.0) + y.get("C3H6", 0.0)
    df = pd.DataFrame(
        {
            "case_id": np.arange(start_case_id, start_case_id + len(T), dtype=int),
            "design_region": design_region,
            "design_kind": design_kind,
            "T_K": T,
            "p_Pa": p,
            "tau_end_s": tau,
            "steam_to_ethane_mass": s2e,
            "target_conversion": conv,
            "wall_extreme": T >= 1450.0,
            "conversion_proxy": conv,
            "log10_p_Pa": np.log10(np.clip(p, 1.0, None)),
            "log10_tau_s": np.log10(np.clip(tau, 1.0e-12, None)),
            "log10_y_c2h6": np.log10(np.clip(y["C2H6"], 1.0e-12, None)),
            "log10_y_h2o": np.log10(np.clip(y["H2O"], 1.0e-12, None)),
            "log10_product_pool": np.log10(np.clip(product_pool, 1.0e-12, None)),
        }
    )
    for sp, arr in y.items():
        df[f"Y_{sp}"] = arr
    return df


def _sample_common(n: int, cfg: DesignConfig, seed_offset: int, dims: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = sobol_or_random(max(n, 1), dims, cfg.seed + seed_offset)[:n]
    logp_min, logp_max = np.log10(cfg.pressure_min_Pa), np.log10(cfg.pressure_max_Pa)
    p = 10.0 ** (logp_min + u[:, 1] * (logp_max - logp_min))
    s2e = cfg.steam_ethane_min_mass + u[:, 2] * (cfg.steam_ethane_max_mass - cfg.steam_ethane_min_mass)
    return u, p, s2e


def _tau_from_proxy(T: np.ndarray, conv: np.ndarray, cfg: DesignConfig) -> np.ndarray:
    # Same cheap monotone proxy used only for case placement. It is clipped to the user bounds.
    kproxy = 5.0e8 * np.exp(-28_000.0 / np.clip(T, 300.0, None))
    tau = -np.log(np.clip(1.0 - conv, 1.0e-12, 1.0)) / np.clip(kproxy, 1.0e-30, None)
    return np.clip(tau, cfg.tau_min_s, cfg.tau_max_s)


def design_candidate_cases(cfg: DesignConfig) -> pd.DataFrame:
    """Generate candidates with explicit filling of the two visible holes.

    The original plot shows two important sparse regions:
      1) high-temperature / low-conversion states around 1200--1400 K;
      2) low-temperature / high-conversion states around 800--1000 K.

    Region (1) can be reached by short isothermal PFR residence times.
    Region (2) is usually not reachable from fresh ethane feed at realistic residence
    time, but it is highly relevant for CFD robustness: product-rich gas can cool or
    mix into colder cells. Therefore those candidates are generated as direct
    state probes, i.e. set T, p, composition and evaluate source terms.
    """
    n_total = int(cfg.n_candidates)
    n_focus = int(round(cfg.focus_gap_fraction * n_total))
    n_general = max(0, n_total - n_focus)
    n_highT_lowX = n_focus // 2
    n_lowT_highX = n_focus - n_highT_lowX

    pieces: list[pd.DataFrame] = []
    cid0 = 0

    # General background: mostly physical isothermal PFR cases across the conventional space,
    # with a small capped high-T tail up to T_hard_max_K, never above 1600 K by default.
    if n_general:
        u, p, s2e = _sample_common(n_general, cfg, 11, dims=5)
        wall = u[:, 4] < cfg.wall_extreme_fraction
        T = np.empty(n_general)
        T[~wall] = cfg.T_min_K + u[~wall, 0] * (cfg.T_max_K - cfg.T_min_K)
        T[wall] = cfg.T_max_K + u[wall, 0] * (cfg.T_wall_max_K - cfg.T_max_K)
        T = np.clip(T, cfg.T_min_K, cfg.T_hard_max_K)
        logt_min, logt_max = np.log10(cfg.tau_min_s), np.log10(cfg.tau_max_s)
        tau = 10.0 ** (logt_min + u[:, 3] * (logt_max - logt_min))
        theta = tau * np.exp(-28_000.0 / np.clip(T, 300.0, None)) * 5.0e8
        conv = np.clip(1.0 - np.exp(-theta), 0.0, 1.0)
        df = _assemble_candidate_frame(T=T, p=p, tau=tau, s2e=s2e, conv=conv, design_region="general", design_kind="isothermal_pfr", start_case_id=cid0)
        pieces.append(df); cid0 += len(df)

    # Hole 1: 1200--1400 K, low conversion. These are important for hot near-wall cells
    # before chemistry has caught up. Use very short PFR trajectories.
    if n_highT_lowX:
        u, p, s2e = _sample_common(n_highT_lowX, cfg, 23, dims=5)
        T = 1200.0 + u[:, 0] * 200.0
        T = np.clip(T, cfg.T_min_K, min(cfg.T_hard_max_K, 1400.0))
        conv = 0.0 + u[:, 3] * 0.20
        tau = _tau_from_proxy(T, conv, cfg)
        df = _assemble_candidate_frame(T=T, p=p, tau=tau, s2e=s2e, conv=conv, design_region="gap_highT_lowX_1200_1400", design_kind="isothermal_pfr", start_case_id=cid0)
        pieces.append(df); cid0 += len(df)

    # Hole 2: 800--1000 K, high conversion. A fresh-feed isothermal PFR cannot usually
    # create this at realistic tau, so evaluate these as direct state probes.
    if n_lowT_highX:
        u, p, s2e = _sample_common(n_lowT_highX, cfg, 37, dims=5)
        T = 800.0 + u[:, 0] * 200.0
        T = np.clip(T, cfg.T_min_K, min(cfg.T_hard_max_K, 1000.0))
        conv = 0.70 + u[:, 3] * 0.30
        tau = cfg.tau_min_s * (cfg.tau_max_s / cfg.tau_min_s) ** u[:, 4]
        df = _assemble_candidate_frame(T=T, p=p, tau=tau, s2e=s2e, conv=conv, design_region="gap_lowT_highX_800_1000", design_kind="state_probe", start_case_id=cid0)
        pieces.append(df); cid0 += len(df)

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    # Absolute safety cap requested by user.
    out = out[out["T_K"] <= cfg.T_hard_max_K].reset_index(drop=True)
    out["case_id"] = np.arange(len(out), dtype=int)
    return out



def _target_gap_specs(cfg: DesignConfig) -> list[dict]:
    """Sparse regions to fill explicitly in the T-conversion plane.

    These are based on the user-inspected original coverage plot:
      - 1200--1400 K at low conversion;
      - 800--1000 K at high conversion.
    All temperatures are clipped by cfg.T_hard_max_K, normally 1600 K.
    """
    return [
        {
            "name": "gap_highT_lowX_1200_1400",
            "T_lo": max(cfg.T_min_K, 1200.0),
            "T_hi": min(cfg.T_hard_max_K, 1400.0),
            "X_lo": 0.00,
            "X_hi": 0.25,
            "design_kind": "isothermal_pfr",
        },
        {
            "name": "gap_lowT_highX_800_1000",
            "T_lo": max(cfg.T_min_K, 800.0),
            "T_hi": min(cfg.T_hard_max_K, 1000.0),
            "X_lo": 0.65,
            "X_hi": 1.00,
            "design_kind": "state_probe",
        },
    ]


def _tx_edges(lo: float, hi: float, width: float) -> np.ndarray:
    if hi <= lo:
        return np.asarray([lo, hi], dtype=float)
    n = max(1, int(math.ceil((hi - lo) / max(width, 1.0e-12))))
    return np.linspace(lo, hi, n + 1)


def target_gap_coverage_report(features: pd.DataFrame, cfg: DesignConfig) -> dict:
    """Return bin occupancy and underfilled bins for the two target holes."""
    report: dict[str, dict] = {}
    if features is None or len(features) == 0:
        base = pd.DataFrame({"T_K": [], "conversion_proxy": []})
    else:
        base = features.copy()
    if "T_K" not in base.columns or "conversion_proxy" not in base.columns:
        return {"error": "features must contain T_K and conversion_proxy"}

    T_all = pd.to_numeric(base["T_K"], errors="coerce").to_numpy(float)
    X_all = pd.to_numeric(base["conversion_proxy"], errors="coerce").to_numpy(float)
    finite = np.isfinite(T_all) & np.isfinite(X_all) & (T_all <= cfg.T_hard_max_K)
    T_all = T_all[finite]
    X_all = X_all[finite]

    for spec in _target_gap_specs(cfg):
        T_edges = _tx_edges(spec["T_lo"], spec["T_hi"], cfg.tx_T_bin_width_K)
        X_edges = _tx_edges(spec["X_lo"], spec["X_hi"], cfg.tx_X_bin_width)
        mask = (
            (T_all >= spec["T_lo"]) & (T_all <= spec["T_hi"]) &
            (X_all >= spec["X_lo"]) & (X_all <= spec["X_hi"])
        )
        if np.any(mask):
            H, _, _ = np.histogram2d(T_all[mask], X_all[mask], bins=[T_edges, X_edges])
        else:
            H = np.zeros((len(T_edges) - 1, len(X_edges) - 1), dtype=float)
        under = np.argwhere(H < cfg.target_gap_min_bin_count)
        report[spec["name"]] = {
            "T_range_K": [float(spec["T_lo"]), float(spec["T_hi"])],
            "conversion_range": [float(spec["X_lo"]), float(spec["X_hi"])],
            "n_T_bins": int(len(T_edges) - 1),
            "n_X_bins": int(len(X_edges) - 1),
            "n_bins": int(H.size),
            "min_bin_count_target": int(cfg.target_gap_min_bin_count),
            "empty_bins": int(np.sum(H == 0)),
            "underfilled_bins": int(len(under)),
            "min_count": int(np.min(H)) if H.size else 0,
            "median_count": float(np.median(H)) if H.size else 0.0,
            "p10_count": float(np.percentile(H, 10)) if H.size else 0.0,
        }
    return report


def _gap_fill_points_for_underfilled_bins(existing_plus_selected: pd.DataFrame, cfg: DesignConfig, budget: int, seed_offset: int) -> pd.DataFrame:
    """Generate targeted cases for underfilled T-conversion bins.

    The function never returns more than `budget` rows and never creates T > cfg.T_hard_max_K.
    """
    if budget <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.seed + seed_offset)
    rows: list[pd.DataFrame] = []
    current = existing_plus_selected.copy()
    cid0 = 0

    for spec in _target_gap_specs(cfg):
        if len(rows) >= budget:
            break
        T_edges = _tx_edges(spec["T_lo"], spec["T_hi"], cfg.tx_T_bin_width_K)
        X_edges = _tx_edges(spec["X_lo"], spec["X_hi"], cfg.tx_X_bin_width)
        T_all = pd.to_numeric(current.get("T_K", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
        X_all = pd.to_numeric(current.get("conversion_proxy", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
        finite = np.isfinite(T_all) & np.isfinite(X_all) & (T_all <= cfg.T_hard_max_K)
        T_all = T_all[finite]
        X_all = X_all[finite]
        mask = (
            (T_all >= spec["T_lo"]) & (T_all <= spec["T_hi"]) &
            (X_all >= spec["X_lo"]) & (X_all <= spec["X_hi"])
        )
        if np.any(mask):
            H, _, _ = np.histogram2d(T_all[mask], X_all[mask], bins=[T_edges, X_edges])
        else:
            H = np.zeros((len(T_edges) - 1, len(X_edges) - 1), dtype=float)

        under = np.argwhere(H < cfg.target_gap_min_bin_count)
        # Fill emptiest bins first, but add only a few per bin so diversity remains broad.
        under = sorted(under, key=lambda ij: H[tuple(ij)])
        for iT, iX in under:
            if sum(len(r) for r in rows) >= budget:
                break
            missing = int(cfg.target_gap_min_bin_count - H[iT, iX])
            n_here = min(missing, budget - sum(len(r) for r in rows))
            if n_here <= 0:
                continue
            T_lo, T_hi = T_edges[iT], T_edges[iT + 1]
            X_lo, X_hi = X_edges[iX], X_edges[iX + 1]
            T = rng.uniform(T_lo, T_hi, size=n_here)
            T = np.clip(T, cfg.T_min_K, cfg.T_hard_max_K)
            conv = rng.uniform(X_lo, X_hi, size=n_here)
            logp_min, logp_max = np.log10(cfg.pressure_min_Pa), np.log10(cfg.pressure_max_Pa)
            p = 10.0 ** rng.uniform(logp_min, logp_max, size=n_here)
            s2e = rng.uniform(cfg.steam_ethane_min_mass, cfg.steam_ethane_max_mass, size=n_here)
            if spec["design_kind"] == "isothermal_pfr":
                tau = _tau_from_proxy(T, conv, cfg)
            else:
                tau = cfg.tau_min_s * (cfg.tau_max_s / cfg.tau_min_s) ** rng.random(n_here)
            df = _assemble_candidate_frame(
                T=T, p=p, tau=tau, s2e=s2e, conv=conv,
                design_region=f"{spec['name']}_iterative_fill",
                design_kind=spec["design_kind"],
                start_case_id=cid0,
            )
            cid0 += len(df)
            rows.append(df)
            # Update current immediately so subsequent bins/regions see these fillers.
            current = pd.concat([current, df[["T_K", "conversion_proxy"]]], ignore_index=True)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out = out[out["T_K"] <= cfg.T_hard_max_K].reset_index(drop=True)
    out["case_id"] = np.arange(len(out), dtype=int)
    return out


def refine_selected_gap_coverage(existing_features: pd.DataFrame, selected: pd.DataFrame, cfg: DesignConfig) -> tuple[pd.DataFrame, dict]:
    """Re-check target-hole coverage after selection and add fillers if still needed."""
    hard_cap = min(int(cfg.n_new_cases), int(cfg.max_new_cases_per_run), 5000)
    if cfg.n_new_cases > hard_cap:
        warnings.warn(f"Requested --n-new-cases {cfg.n_new_cases}, but capped at {hard_cap} by --max-new-cases-per-run / hard 5000 cap.")
    selected = selected.head(hard_cap).copy().reset_index(drop=True)

    def _combined(sel: pd.DataFrame) -> pd.DataFrame:
        cols = ["T_K", "conversion_proxy"]
        return pd.concat([existing_features[cols], sel[cols]], ignore_index=True)

    history: list[dict] = []
    before = target_gap_coverage_report(existing_features, cfg)
    history.append({"stage": "before_enrichment", "n_selected": 0, "gap_report": before})

    after_initial = target_gap_coverage_report(_combined(selected), cfg)
    history.append({"stage": "after_initial_selection", "n_selected": int(len(selected)), "gap_report": after_initial})

    for r in range(max(0, int(cfg.coverage_refine_rounds))):
        remaining = hard_cap - len(selected)
        if remaining <= 0:
            break
        current_report = target_gap_coverage_report(_combined(selected), cfg)
        under_total = sum(v.get("underfilled_bins", 0) for v in current_report.values() if isinstance(v, dict))
        if under_total == 0:
            break
        filler = _gap_fill_points_for_underfilled_bins(_combined(selected), cfg, budget=remaining, seed_offset=1000 + r)
        if len(filler) == 0:
            break
        selected = pd.concat([selected, filler], ignore_index=True)
        selected = selected[selected["T_K"] <= cfg.T_hard_max_K].reset_index(drop=True)
        selected["case_id"] = np.arange(len(selected), dtype=int)
        new_report = target_gap_coverage_report(_combined(selected), cfg)
        history.append({"stage": f"after_refine_round_{r + 1}", "n_selected": int(len(selected)), "added_this_round": int(len(filler)), "gap_report": new_report})

    final = target_gap_coverage_report(_combined(selected), cfg)
    history.append({"stage": "final", "n_selected": int(len(selected)), "gap_report": final})
    return selected.head(hard_cap).copy().reset_index(drop=True), {"hard_cap_per_command": hard_cap, "history": history}

def choose_low_coverage_cases(existing_features: pd.DataFrame, candidates: pd.DataFrame, cfg: DesignConfig) -> pd.DataFrame:
    X_old = feature_matrix(existing_features)
    X_cand = feature_matrix(candidates)
    scaler = StandardScaler().fit(np.vstack([X_old, X_cand]))
    Xo = scaler.transform(X_old)
    Xc = scaler.transform(X_cand)

    # Distance to nearest existing state identifies extrapolation-prone regions.
    n_neighbors = min(8, len(Xo))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(Xo)
    dists, _ = nbrs.kneighbors(Xc)
    far_score = dists[:, -1]

    # Explicitly boost the visually missing regions pointed out by the user.
    region = candidates.get("design_region", pd.Series("general", index=candidates.index)).astype(str)
    boost = np.zeros(len(candidates), dtype=float)
    boost[region.str.contains("gap_highT_lowX", regex=False).to_numpy()] += np.nanpercentile(far_score, 80) * 0.75
    boost[region.str.contains("gap_lowT_highX", regex=False).to_numpy()] += np.nanpercentile(far_score, 80) * 0.75
    score = far_score + boost

    # Guarantee that both sparse regions survive the diversity filter.
    min_each_gap = max(1, int(0.20 * cfg.n_new_cases))
    forced: list[int] = []
    for pat in ["gap_highT_lowX", "gap_lowT_highX"]:
        idx = np.where(region.str.contains(pat, regex=False).to_numpy())[0]
        if len(idx):
            idx = idx[np.argsort(-score[idx])]
            forced.extend([int(i) for i in idx[:min_each_gap]])
    forced = list(dict.fromkeys(forced))

    # Penalize candidate-candidate clustering by greedily taking high-score candidates
    # while rejecting points too close to already selected enrichment points.
    order = np.argsort(-score)
    selected: list[int] = []
    selected_X: list[np.ndarray] = []
    min_sep = np.nanpercentile(far_score, 35) * 0.30
    min_sep = max(float(min_sep), 0.04)

    for idx in forced + [int(i) for i in order]:
        if idx in selected:
            continue
        x = Xc[idx]
        if selected_X:
            sx = np.vstack(selected_X)
            if np.min(np.linalg.norm(sx - x, axis=1)) < min_sep and idx not in forced:
                continue
        selected.append(int(idx))
        selected_X.append(x)
        if len(selected) >= cfg.n_new_cases:
            break
    if len(selected) < cfg.n_new_cases:
        missing = cfg.n_new_cases - len(selected)
        selected_set = set(selected)
        selected.extend([int(i) for i in order if int(i) not in selected_set][:missing])

    out = candidates.iloc[selected].copy().reset_index(drop=True)
    out["case_id"] = np.arange(len(out), dtype=int)
    out["coverage_distance"] = far_score[selected]
    out["coverage_score"] = score[selected]
    return out


# -----------------------------------------------------------------------------
# Optional Cantera isothermal solver
# -----------------------------------------------------------------------------


def ethane_steam_X(steam_to_ethane_mass: float) -> dict[str, float]:
    mw_c2h6 = 30.069  # kg/kmol
    mw_h2o = 18.01528
    n_c2h6 = 1.0
    n_h2o = steam_to_ethane_mass * mw_c2h6 / mw_h2o
    total = n_c2h6 + n_h2o
    return {"C2H6": n_c2h6 / total, "H2O": n_h2o / total}


def solve_isothermal_cases(
    manifest: pd.DataFrame,
    mech: str | Path,
    out_parquet: Path,
    n_time_points: int,
    seed: int,
    print_cases: bool = False,
    case_log_every: int = 100,
) -> pd.DataFrame:
    try:
        import cantera as ct
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Cantera is required for --mode solve/all. Install Cantera or run --mode design.") from exc

    gas = ct.Solution(str(mech))
    species_names = list(gas.species_names)
    mw = gas.molecular_weights  # kg/kmol
    rows: list[dict] = []
    rng = np.random.default_rng(seed)

    # Mechanism-specific species availability.
    has = {s: (s in species_names) for s in ["C2H6", "H2O"]}
    if not has["C2H6"] or not has["H2O"]:
        raise ValueError("Mechanism must contain species C2H6 and H2O for the default ethane/steam feed.")

    case_log_every = max(1, int(case_log_every))
    for case_index, (_, case) in enumerate(manifest.iterrows(), start=1):
        cid = int(case["case_id"])
        if print_cases and (case_index == 1 or case_index == len(manifest) or ((case_index - 1) % case_log_every == 0)):
            print(_case_log_line(case, prefix="SOLVE"), flush=True)
        T = float(case["T_K"])
        p = float(case["p_Pa"])
        tau_end = float(case["tau_end_s"])
        s2e = float(case["steam_to_ethane_mass"])
        design_kind = str(case.get("design_kind", "isothermal_pfr"))
        design_region = str(case.get("design_region", "general"))

        def append_state_row(g, t_value: float, sample_kind: str) -> None:
            Y = g.Y
            wdot = g.net_production_rates  # kmol/m3/s
            rho = g.density  # kg/m3
            dydt = wdot * mw / max(rho, 1.0e-300)  # 1/s mass-fraction source
            # Cantera partial_molar_enthalpies: J/kmol; wdot: kmol/m3/s -> J/m3/s.
            heat_absorption = float(np.dot(g.partial_molar_enthalpies, wdot))
            row = {
                "case_id": cid,
                "sample_kind": sample_kind,
                "design_kind": design_kind,
                "design_region": design_region,
                "target_conversion": float(case.get("target_conversion", np.nan)),
                "T": float(g.T),
                "T [K]": float(g.T),
                "Pressure [Pa]": float(g.P),
                "tau": float(t_value),
                "Residence time [s]": float(t_value),
                "tau_end_s": tau_end,
                "steam_to_ethane_mass": s2e,
                "wall_extreme": bool(case.get("wall_extreme", False)),
                "Reaction heat absorption [J/s/m3]": heat_absorption,
                "rho [kg/m3]": float(rho),
                "cp [J/kg/K]": float(g.cp_mass),
                "MW [kg/kmol]": float(g.mean_molecular_weight),
            }
            # Store species mass fractions and source terms with SCARFS-friendly names.
            for name, y, dy in zip(species_names, Y, dydt):
                row[f"Y_{name}"] = float(max(y, 0.0))
                row[f"dYdt_{name}"] = float(dy)
            rows.append(row)

        if design_kind == "state_probe":
            Ydict = {}
            for sp in species_names:
                col = f"Y_{sp}"
                if col in case.index and pd.notna(case[col]):
                    val = float(case[col])
                    if val > 0.0:
                        Ydict[sp] = val
            if not Ydict:
                Ydict = {"C2H6": 1.0 / (1.0 + s2e), "H2O": s2e / (1.0 + s2e)}
            gas.TPY = T, p, Ydict
            append_state_row(gas, tau_end, "state_probe_enrichment")
        else:
            X0 = ethane_steam_X(s2e)
            gas.TPX = T, p, X0
            # Constant-pressure, energy-off homogeneous reactor: isothermal chemistry trajectory.
            reactor = ct.IdealGasConstPressureReactor(gas, energy="off")
            sim = ct.ReactorNet([reactor])
            # Denser near t=0 and near the final high-conversion tail.
            t_eval = np.unique(
                np.r_[
                    np.geomspace(max(tau_end * 1.0e-6, 1.0e-12), tau_end, n_time_points),
                    np.linspace(0.0, tau_end, max(16, n_time_points // 8)),
                ]
            )
            t_eval = t_eval[t_eval >= 0.0]
            for t in t_eval:
                sim.advance(float(t))
                append_state_row(reactor.thermo, float(t), "isothermal_enrichment")

        if (case_index % case_log_every == 0) or (case_index == len(manifest)):
            print(f"[SOLVED] {case_index}/{len(manifest)} cases completed; last case_id={cid:05d}", flush=True)

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False, compression="snappy")
    return df


# -----------------------------------------------------------------------------
# Balanced database creation
# -----------------------------------------------------------------------------


def make_bin_edges(features: pd.DataFrame, feature_cols: Sequence[str], n_bins: int) -> dict[str, np.ndarray]:
    edges: dict[str, np.ndarray] = {}
    for col in feature_cols:
        if col not in features.columns:
            continue
        x = features[col].to_numpy(float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        # Robust quantile bins reduce the damage of long tails.
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        e = np.nanquantile(x, qs)
        e = np.unique(e)
        if len(e) < 3:
            lo, hi = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                continue
            e = np.linspace(lo, hi, n_bins + 1)
        e[0] = -np.inf
        e[-1] = np.inf
        edges[col] = e
    return edges


def bin_keys(features: pd.DataFrame, edges: dict[str, np.ndarray]) -> np.ndarray:
    if not edges:
        return np.zeros(len(features), dtype=np.int64)
    codes = []
    mult = []
    base = 1
    for col, e in edges.items():
        x = features[col].to_numpy(float) if col in features else np.full(len(features), np.nan)
        # digitize returns 0..len(e)-2 after clipping.
        c = np.digitize(x, e[1:-1], right=False)
        c = np.where(np.isfinite(x), c, 0).astype(np.int64)
        codes.append(c)
        mult.append(base)
        base *= max(1, len(e) - 1)
    key = np.zeros(len(features), dtype=np.int64)
    for c, m in zip(codes, mult):
        key += c * m
    return key


def count_bins(paths: Sequence[Path], columns: Sequence[str], edges: dict[str, np.ndarray], cfg: DesignConfig) -> dict[int, int]:
    counts: dict[int, int] = {}
    for batch in stream_parquet_batches(paths, columns, cfg.batch_size):
        if len(batch) == 0:
            continue
        # Keep the training/enrichment domain bounded; default user-requested cap is 1600 K.
        if cfg.final_T_max_K is not None:
            tcol = find_first_column(list(batch.columns), TEMP_CANDIDATES)
            if tcol is not None:
                batch = batch[pd.to_numeric(batch[tcol], errors="coerce") <= cfg.final_T_max_K]
                if len(batch) == 0:
                    continue
        f, _ = build_feature_frame(batch)
        keys = bin_keys(f, edges)
        vals, cnts = np.unique(keys, return_counts=True)
        for k, c in zip(vals, cnts):
            counts[int(k)] = counts.get(int(k), 0) + int(c)
    return counts


def write_balanced_parquet(
    paths: Sequence[Path],
    out_file: Path,
    columns: Sequence[str] | None,
    edges: dict[str, np.ndarray],
    counts: dict[int, int],
    cfg: DesignConfig,
) -> tuple[int, dict]:
    rng = np.random.default_rng(cfg.seed + 17)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    kept_per_bin: dict[int, int] = {}
    total_in = 0
    total_out = 0

    for batch in stream_parquet_batches(paths, columns, cfg.batch_size):
        total_in += len(batch)
        if cfg.final_T_max_K is not None:
            tcol = find_first_column(list(batch.columns), TEMP_CANDIDATES)
            if tcol is not None:
                batch = batch[pd.to_numeric(batch[tcol], errors="coerce") <= cfg.final_T_max_K]
                if len(batch) == 0:
                    continue
        f, _ = build_feature_frame(batch)
        keys = bin_keys(f, edges)
        keep = np.zeros(len(batch), dtype=bool)
        weights = np.ones(len(batch), dtype=float)
        for i, k in enumerate(keys):
            kk = int(k)
            nbin = max(counts.get(kk, 1), 1)
            # Probabilistic cap: expected kept rows per bin <= bin_cap.
            pkeep = min(1.0, cfg.bin_cap / nbin)
            if rng.random() < pkeep:
                keep[i] = True
                kept_per_bin[kk] = kept_per_bin.get(kk, 0) + 1
                # This can be used instead of hard balancing during training.
                weights[i] = 1.0 / nbin
        if not np.any(keep):
            continue
        out = batch.loc[keep].copy()
        out["coverage_bin"] = keys[keep]
        out["sample_weight_inverse_bin_count"] = weights[keep]
        table = pa.Table.from_pandas(out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_file), table.schema, compression="snappy")
        writer.write_table(table)
        total_out += len(out)

    if writer is not None:
        writer.close()
    stats = {
        "total_input_rows_seen": total_in,
        "total_output_rows": total_out,
        "n_bins_seen": len(counts),
        "n_bins_kept": len(kept_per_bin),
        "bin_cap": cfg.bin_cap,
    }
    return total_out, stats


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def ensure_fig_dir(out: Path) -> Path:
    d = out / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_for_plot(paths: Sequence[Path], cfg: DesignConfig, seed: int, max_rows: int = 120_000) -> pd.DataFrame:
    schema = get_schema_columns(paths)
    cols = select_state_columns(schema)
    raw = sample_parquet_rows(paths, cols, max_rows=max_rows, batch_size=cfg.batch_size, seed=seed)
    f, _ = build_feature_frame(raw)
    return f


def pca_plot(feature_sets: dict[str, pd.DataFrame], out_png: Path, title: str) -> None:
    # Fit PCA on all feature sets combined.
    mats = []
    labels = []
    for label, f in feature_sets.items():
        if len(f) == 0:
            continue
        X = feature_matrix(f)
        mats.append(X)
        labels.extend([label] * len(X))
    if not mats:
        return
    Xall = np.vstack(mats)
    scaler = StandardScaler().fit(Xall)
    Z = PCA(n_components=2, random_state=0).fit_transform(scaler.transform(Xall))
    labels_arr = np.asarray(labels)

    plt.figure(figsize=(7.2, 5.2), dpi=180)
    start = 0
    for label, f in feature_sets.items():
        n = len(f)
        if n == 0:
            continue
        z = Z[start : start + n]
        start += n
        if n > 40_000:
            idx = np.random.default_rng(0).choice(n, 40_000, replace=False)
            z = z[idx]
        plt.scatter(z[:, 0], z[:, 1], s=2, alpha=0.25, label=label)
    plt.xlabel("PCA-1 of coverage features")
    plt.ylabel("PCA-2 of coverage features")
    plt.title(title)
    plt.legend(markerscale=4, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def hexbin_plot(f: pd.DataFrame, out_png: Path, title: str) -> None:
    if len(f) == 0:
        return
    x = f["T_K"].to_numpy(float)
    y = f["conversion_proxy"].to_numpy(float)
    plt.figure(figsize=(7.2, 5.2), dpi=180)
    plt.hexbin(x, y, gridsize=75, bins="log", mincnt=1)
    plt.xlabel("Temperature [K]")
    plt.ylabel("Ethane conversion proxy [-]")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("log10(N) per hexbin")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def tau_temp_plot(f: pd.DataFrame, out_png: Path, title: str) -> None:
    if len(f) == 0:
        return
    plt.figure(figsize=(7.2, 5.2), dpi=180)
    plt.hexbin(f["T_K"], f["log10_tau_s"], gridsize=75, bins="log", mincnt=1)
    plt.xlabel("Temperature [K]")
    plt.ylabel("log10(residence time [s])")
    plt.title(title)
    cb = plt.colorbar()
    cb.set_label("log10(N) per hexbin")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def occupancy_plot(counts_before: dict[int, int], counts_after: dict[int, int], out_png: Path) -> None:
    b = np.array(list(counts_before.values()), dtype=float)
    a = np.array(list(counts_after.values()), dtype=float)
    plt.figure(figsize=(7.2, 5.2), dpi=180)
    if len(b):
        plt.hist(np.log10(b + 1.0), bins=60, alpha=0.45, label="before balancing")
    if len(a):
        plt.hist(np.log10(a + 1.0), bins=60, alpha=0.45, label="after balancing")
    plt.xlabel("log10(rows per coverage bin + 1)")
    plt.ylabel("number of bins")
    plt.title("Coverage-bin occupancy before/after balancing")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def make_plots(
    out: Path,
    original: pd.DataFrame,
    candidates: pd.DataFrame | None = None,
    balanced: pd.DataFrame | None = None,
    counts_before: dict[int, int] | None = None,
    counts_after: dict[int, int] | None = None,
) -> None:
    fig_dir = ensure_fig_dir(out)
    hexbin_plot(original, fig_dir / "01_original_T_vs_conversion.png", "Original database state space")
    tau_temp_plot(original, fig_dir / "02_original_T_vs_tau.png", "Original database: temperature vs residence time")
    sets = {"original": original}
    if candidates is not None and len(candidates):
        sets["selected isothermal cases"] = candidates
        hexbin_plot(
            pd.concat([original, candidates], ignore_index=True),
            fig_dir / "03_after_enrichment_T_vs_conversion.png",
            "State space after selected isothermal enrichment",
        )
    if balanced is not None and len(balanced):
        sets["balanced final"] = balanced
        hexbin_plot(balanced, fig_dir / "04_balanced_T_vs_conversion.png", "Balanced final training database")
        tau_temp_plot(balanced, fig_dir / "05_balanced_T_vs_tau.png", "Balanced final database: T vs tau")
    pca_plot(sets, fig_dir / "06_pca_state_space_overlay.png", "Coverage-feature PCA overlay")
    if counts_before is not None and counts_after is not None:
        occupancy_plot(counts_before, counts_after, fig_dir / "07_bin_occupancy_before_after.png")


# -----------------------------------------------------------------------------
# Main modes
# -----------------------------------------------------------------------------


def write_manifest(manifest: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out / "isothermal_enrichment_manifest.csv", index=False)
    (out / "isothermal_enrichment_manifest.json").write_text(
        json.dumps(manifest.to_dict(orient="records"), indent=2), encoding="utf-8"
    )


def _case_log_line(row: pd.Series, prefix: str = "ADDED") -> str:
    """Compact one-line case description for terminal follow-up."""
    def val(name: str, default=np.nan) -> float:
        try:
            return float(row.get(name, default))
        except Exception:
            return float("nan")

    cid = int(val("case_id", -1))
    p_bar = val("p_Pa") / 1.0e5
    tau = val("tau_end_s")
    T = val("T_K")
    s2e = val("steam_to_ethane_mass")
    X = val("target_conversion")
    dist = val("coverage_distance")
    score = val("coverage_score")
    region = str(row.get("design_region", "?"))
    kind = str(row.get("design_kind", "?"))
    dist_txt = f" dist={dist:.3g}" if np.isfinite(dist) else ""
    score_txt = f" score={score:.3g}" if np.isfinite(score) else ""
    return (
        f"[{prefix}] case={cid:05d} region={region} kind={kind} "
        f"T={T:.1f}K p={p_bar:.3f}bar steam/C2H6_mass={s2e:.3f} "
        f"tau_end={tau:.3e}s target_X={X:.3f}{dist_txt}{score_txt}"
    )


def print_case_log(manifest: pd.DataFrame, *, prefix: str, every: int, max_rows: int | None = None) -> None:
    """Print selected/solved case IDs without flooding unless explicitly requested."""
    if manifest is None or len(manifest) == 0:
        print(f"[{prefix}] no cases", flush=True)
        return
    every = max(1, int(every))
    n = len(manifest) if max_rows is None else min(len(manifest), int(max_rows))
    for i, (_, row) in enumerate(manifest.head(n).iterrows(), start=1):
        if i == 1 or i == n or ((i - 1) % every == 0):
            print(_case_log_line(row, prefix=prefix), flush=True)
    if n < len(manifest):
        print(f"[{prefix}] printed {n}/{len(manifest)} cases; full list is in isothermal_enrichment_manifest.csv", flush=True)


def write_human_case_log(manifest: pd.DataFrame, out: Path, filename: str = "isothermal_enrichment_case_log.txt") -> None:
    """Always write a human-readable one-line-per-case log beside the manifest."""
    lines = [_case_log_line(row, prefix="ADDED") for _, row in manifest.iterrows()]
    (out / filename).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_design(args: argparse.Namespace, cfg: DesignConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = _existing_paths([args.full, args.off, args.enriched])
    if not paths:
        raise SystemExit("No valid --full/--off parquet paths were found.")
    schema = get_schema_columns(paths)
    state_cols = select_state_columns(schema)
    if not state_cols:
        raise SystemExit("Could not detect usable state columns in the input parquet files.")
    print(f"sampling existing database columns: {state_cols}")
    raw = sample_parquet_rows(paths, state_cols, cfg.max_existing_rows, cfg.batch_size, cfg.seed)
    existing_features, meta = build_feature_frame(raw)
    candidates = design_candidate_cases(cfg)
    selected_initial = choose_low_coverage_cases(existing_features, candidates, cfg)
    selected, refinement_report = refine_selected_gap_coverage(existing_features, selected_initial, cfg)
    write_manifest(selected, out)
    write_human_case_log(selected, out)
    print("selected enrichment cases by region/kind:", flush=True)
    if len(selected):
        summary_tbl = selected.groupby(["design_region", "design_kind"]).size().reset_index(name="n_cases")
        print(summary_tbl.to_string(index=False), flush=True)
    print(f"human-readable case log: {out / 'isothermal_enrichment_case_log.txt'}", flush=True)
    print(f"machine-readable manifest: {out / 'isothermal_enrichment_manifest.csv'}", flush=True)
    if getattr(args, "print_added_cases", False):
        max_rows = None if int(getattr(args, "max_print_cases", 0)) <= 0 else int(args.max_print_cases)
        print_case_log(selected, prefix="ADDED", every=int(getattr(args, "case_log_every", 1)), max_rows=max_rows)
    (out / "coverage_refinement_report.json").write_text(json.dumps(refinement_report, indent=2), encoding="utf-8")
    report = {
        "config": asdict(cfg),
        "detected_columns": meta,
        "input_paths": [str(p) for p in paths],
        "state_columns_sampled": state_cols,
        "n_existing_sample_rows": int(len(existing_features)),
        "n_candidates": int(len(candidates)),
        "n_selected_isothermal_cases": int(len(selected)),
        "coverage_refinement_report": str(out / "coverage_refinement_report.json"),
        "manifest_csv": str(out / "isothermal_enrichment_manifest.csv"),
        "manifest_json": str(out / "isothermal_enrichment_manifest.json"),
    }
    (out / "coverage_design_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    make_plots(out, original=existing_features, candidates=selected)
    return existing_features, selected, report


def run_solve(args: argparse.Namespace, cfg: DesignConfig) -> Path:
    out = Path(args.out)
    manifest_path = Path(args.manifest) if args.manifest else out / "isothermal_enrichment_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}. Run --mode design first.")
    if not args.mech:
        raise SystemExit("--mode solve/all requires --mech chem.yaml")
    manifest = pd.read_csv(manifest_path)
    out_parquet = out / "enriched_isothermal.parquet"
    solve_isothermal_cases(
        manifest,
        args.mech,
        out_parquet,
        cfg.n_time_points,
        cfg.seed,
        print_cases=bool(getattr(args, "print_solve_cases", False)),
        case_log_every=int(getattr(args, "case_log_every", 100)),
    )
    return out_parquet


def run_balance(args: argparse.Namespace, cfg: DesignConfig) -> dict:
    out = Path(args.out)
    paths = _existing_paths([args.full, args.off, args.enriched])
    if not paths:
        raise SystemExit("No valid input parquet paths for balancing.")
    schema = get_schema_columns(paths)
    # For writing the balanced DB, read all columns. For binning, sample state columns first.
    state_cols = select_state_columns(schema)
    raw_sample = sample_parquet_rows(paths, state_cols, cfg.max_existing_rows, cfg.batch_size, cfg.seed + 99)
    feat_sample, meta = build_feature_frame(raw_sample)
    edges = make_bin_edges(feat_sample, DEFAULT_FEATURES, cfg.bin_count_per_feature)
    counts_before = count_bins(paths, state_cols, edges, cfg)

    out_file = out / "balanced_training.parquet"
    # Preserve all columns from all inputs. pyarrow handles missing columns per input file poorly
    # when schemas differ, so stream with columns=None per file and let pandas/pyarrow write chunks.
    total_out, stats = write_balanced_parquet(paths, out_file, None, edges, counts_before, cfg)
    # Plot after balancing from a sample of the new output.
    balanced_sample = sample_for_plot([out_file], cfg, cfg.seed + 123, max_rows=120_000) if total_out else pd.DataFrame()
    counts_after = count_bins([out_file], select_state_columns(get_schema_columns([out_file])), edges, cfg) if total_out else {}
    make_plots(out, original=feat_sample, balanced=balanced_sample, counts_before=counts_before, counts_after=counts_after)
    report = {
        "config": asdict(cfg),
        "detected_columns": meta,
        "input_paths": [str(p) for p in paths],
        "balanced_parquet": str(out_file),
        **stats,
    }
    (out / "balance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Design isothermal enrichment and balance a ChemZIP-like ethane cracking database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["design", "solve", "balance", "all"], default="design")
    p.add_argument("--full", default="full.parquet", help="Existing trajectory database.")
    p.add_argument("--off", default="offmanifold_1000000.parquet", help="Existing off-manifold database.")
    p.add_argument("--enriched", action="append", default=[], help="Optional enriched isothermal parquet for design/balancing. Repeat this flag for multiple enrichment rounds.")
    p.add_argument("--out", default="out_balanced_iso")
    p.add_argument("--manifest", default=None, help="Manifest CSV for --mode solve.")
    p.add_argument("--mech", default=None, help="Cantera mechanism, e.g. chem.yaml.")
    p.add_argument("--print-added-cases", action="store_true", help="Print selected enrichment cases to the terminal during --mode design.")
    p.add_argument("--print-solve-cases", action="store_true", help="Print each/periodic case before it is solved during --mode solve/all.")
    p.add_argument("--case-log-every", type=int, default=25, help="Print one case every N cases when --print-added-cases or --print-solve-cases is used; use 1 for every case.")
    p.add_argument("--max-print-cases", type=int, default=0, help="Maximum added cases to print in design mode. 0 means print all selected cases.")

    p.add_argument("--seed", type=int, default=DesignConfig.seed)
    p.add_argument("--max-existing-rows", type=int, default=DesignConfig.max_existing_rows)
    p.add_argument("--n-candidates", type=int, default=DesignConfig.n_candidates)
    p.add_argument("--n-new-cases", type=int, default=DesignConfig.n_new_cases, help="Requested new enrichment cases. This is capped by --max-new-cases-per-run and by a hard limit of 5000.")
    p.add_argument("--max-new-cases-per-run", type=int, default=DesignConfig.max_new_cases_per_run, help="Maximum additional cases designed by one command. Use <=5000.")
    p.add_argument("--coverage-refine-rounds", type=int, default=DesignConfig.coverage_refine_rounds, help="After initial selection, re-check target T-conversion holes and add fillers for this many rounds.")
    p.add_argument("--target-gap-min-bin-count", type=int, default=DesignConfig.target_gap_min_bin_count, help="Minimum number of states per T-conversion bin inside the explicit sparse gap regions.")
    p.add_argument("--tx-T-bin-width-K", type=float, default=DesignConfig.tx_T_bin_width_K, help="Temperature bin width used to check/fill the explicit T-conversion holes.")
    p.add_argument("--tx-X-bin-width", type=float, default=DesignConfig.tx_X_bin_width, help="Conversion bin width used to check/fill the explicit T-conversion holes.")
    p.add_argument("--n-time-points", type=int, default=DesignConfig.n_time_points)
    p.add_argument("--batch-size", type=int, default=DesignConfig.batch_size)

    p.add_argument("--T-min-K", type=float, default=DesignConfig.T_min_K)
    p.add_argument("--T-max-K", type=float, default=DesignConfig.T_max_K)
    p.add_argument("--T-wall-max-K", type=float, default=DesignConfig.T_wall_max_K)
    p.add_argument("--T-hard-max-K", type=float, default=DesignConfig.T_hard_max_K, help="Absolute maximum temperature for newly generated enrichment states.")
    p.add_argument("--final-T-max-K", type=float, default=DesignConfig.final_T_max_K, help="Drop rows above this temperature when building the balanced final database.")
    p.add_argument("--focus-gap-fraction", type=float, default=DesignConfig.focus_gap_fraction, help="Fraction of candidate pool explicitly assigned to the sparse T-conversion holes.")
    p.add_argument("--pressure-min-Pa", type=float, default=DesignConfig.pressure_min_Pa)
    p.add_argument("--pressure-max-Pa", type=float, default=DesignConfig.pressure_max_Pa)
    p.add_argument("--steam-ethane-min-mass", type=float, default=DesignConfig.steam_ethane_min_mass)
    p.add_argument("--steam-ethane-max-mass", type=float, default=DesignConfig.steam_ethane_max_mass)
    p.add_argument("--tau-min-s", type=float, default=DesignConfig.tau_min_s)
    p.add_argument("--tau-max-s", type=float, default=DesignConfig.tau_max_s)
    p.add_argument("--wall-extreme-fraction", type=float, default=DesignConfig.wall_extreme_fraction)
    p.add_argument("--bin-count-per-feature", type=int, default=DesignConfig.bin_count_per_feature)
    p.add_argument("--bin-cap", type=int, default=DesignConfig.bin_cap)
    return p.parse_args(argv)


def cfg_from_args(args: argparse.Namespace) -> DesignConfig:
    return DesignConfig(
        seed=args.seed,
        max_existing_rows=args.max_existing_rows,
        n_candidates=args.n_candidates,
        n_new_cases=args.n_new_cases,
        max_new_cases_per_run=args.max_new_cases_per_run,
        coverage_refine_rounds=args.coverage_refine_rounds,
        target_gap_min_bin_count=args.target_gap_min_bin_count,
        tx_T_bin_width_K=args.tx_T_bin_width_K,
        tx_X_bin_width=args.tx_X_bin_width,
        n_time_points=args.n_time_points,
        T_min_K=args.T_min_K,
        T_max_K=args.T_max_K,
        T_wall_max_K=args.T_wall_max_K,
        T_hard_max_K=args.T_hard_max_K,
        final_T_max_K=args.final_T_max_K,
        focus_gap_fraction=args.focus_gap_fraction,
        pressure_min_Pa=args.pressure_min_Pa,
        pressure_max_Pa=args.pressure_max_Pa,
        steam_ethane_min_mass=args.steam_ethane_min_mass,
        steam_ethane_max_mass=args.steam_ethane_max_mass,
        tau_min_s=args.tau_min_s,
        tau_max_s=args.tau_max_s,
        wall_extreme_fraction=args.wall_extreme_fraction,
        bin_count_per_feature=args.bin_count_per_feature,
        bin_cap=args.bin_cap,
        batch_size=args.batch_size,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = cfg_from_args(args)
    if cfg.max_new_cases_per_run > 5000:
        raise SystemExit("--max-new-cases-per-run must be <= 5000, as requested for one bash prompt.")
    if cfg.T_hard_max_K > 1600.0:
        raise SystemExit("--T-hard-max-K must be <= 1600 K for this enrichment run.")
    cfg.T_wall_max_K = min(cfg.T_wall_max_K, cfg.T_hard_max_K, 1600.0)
    cfg.T_max_K = min(cfg.T_max_K, cfg.T_hard_max_K, 1600.0)
    cfg.final_T_max_K = min(cfg.final_T_max_K, cfg.T_hard_max_K, 1600.0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode in {"design", "all"}:
        print("[1/3] Designing isothermal enrichment cases in low-coverage regions")
        run_design(args, cfg)

    enriched_path: Path | None = None
    if args.mode in {"solve", "all"}:
        print("[2/3] Solving selected isothermal enrichment cases")
        enriched_path = run_solve(args, cfg)
        args.enriched = str(enriched_path)

    if args.mode in {"balance", "all"}:
        print("[3/3] Building balanced training database")
        if args.mode == "all" and enriched_path is not None:
            args.enriched = str(enriched_path)
        report = run_balance(args, cfg)
        print(json.dumps(report, indent=2))

    print(f"Done. Outputs are in: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

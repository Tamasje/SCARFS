#!/usr/bin/env python3
# SCRIPT_VERSION = "identify_isothermal_empty_regions_r3_20k_max_physical_coverage_v6"
"""
R3 20k max-physical-coverage identifier. Auto-discovers existing
out_v2_iso_r*/isothermal_enrichment_cracksim.parquet databases and treats them
as already-covered state space before designing the third isothermal round.

Identify sparse / empty regions in the existing SCARFS state space using an
adaptive Sobol/bin-deficit design in T-X and T-log(tau) space. It writes PFR-first target requests and real-database fallback anchor compositions; it does not pre-classify reachability with a hardcoded Arrhenius model and write an isothermal enrichment manifest to be consumed by scripts/generate_database_Isothermal.py. If a previously generated isothermal enrichment parquet is available, it is included in the coverage accounting so subsequent rounds target remaining gaps instead of refilling already enriched regions.

This script DOES NOT call Cantera or CRACKSIM. It only designs modelling conditions. Fallback state-probe compositions are copied from real full-species Y_* anchor states in the existing CRACKSIM databases rather than from synthetic product splits. The CRACKSIM generator decides after an actual PFR attempt whether the fallback probe is needed.
The production database is generated in the second step with generate_database_Isothermal.py.

Main assumptions requested by the user:
- New enrichment states are limited to T <= 1600 K.
- Steam dilution is H2O/C2H6 mass ratio and is capped between 0 and 1.
- Pressure is capped between 1.5 and 3.5 bar.
- No special drop is applied above 1400 K; only the 1600 K hard cap is enforced.
- Maximum number of new conditions per command is controlled by --max-new-cases-per-run
  and hard-limited to 20000.

Typical use from SCARFS repo root:

  python scripts/identify_isothermal_empty_regions.py --full out_v2/full.parquet --off out_v2/offmanifold_1000000.parquet --out out_balanced_iso_r3 --n-new-cases 20000 --max-new-cases-per-run 20000 --max-existing-rows 30000000 --state-probe-anchor-rows 8000000 --T-min-K 800 --T-hard-max-K 1600 --pressure-min-Pa 150000 --pressure-max-Pa 350000 --steam-ethane-min-mass 0 --steam-ethane-max-mass 1 --tau-min-s 1e-5 --tau-max-s 1.0 --coverage-refine-rounds 8 --target-log10-count 2.3 --target-ttau-log10-count 2.3 --ttau-logtau-bin-width-decades 0.25

Outputs:
- isothermal_enrichment_manifest.csv
- isothermal_enrichment_manifest.json
- coverage_refinement_report.json
- figures/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires pyarrow. Install with: pip install pyarrow") from exc

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

SCRIPT_VERSION = "identify_isothermal_empty_regions_r3_20k_max_physical_coverage_v6"
EPS = 1.0e-300
DESIGN_SPECIES = ["H2O", "C2H6", "C2H4", "CH4", "H2", "C2H2", "C3H6", "C3H8"]
# Approximate molecular weights [kg/mol] used only for hydrodynamic design metadata.
# generate_database_Isothermal.py recomputes rho/mu with Cantera before writing the final parquet.
MW_KG_PER_MOL = {"H2O": 0.01801528, "C2H6": 0.03006904, "C2H4": 0.02805316, "CH4": 0.01604246, "H2": 0.00201588, "C2H2": 0.02603728, "C3H6": 0.04207974, "C3H8": 0.04409562}
R_UNIVERSAL = 8.31446261815324

TEMP_CANDIDATES = ["T", "T [K]", "Temperature", "Temperature [K]", "temperature", "temperature_K", "T_K"]
PRESSURE_CANDIDATES = ["p", "P", "Pressure", "Pressure [Pa]", "pressure", "pressure_Pa", "P_Pa", "p_Pa"]
TAU_CANDIDATES = ["tau", "tau [s]", "Residence time [s]", "residence_time_s", "time", "t", "t [s]", "tau_end_s"]
SPECIES_ALIASES = {
    "H2O": ["H2O", "WATER"],
    "C2H6": ["C2H6", "ETHANE"],
    "C2H4": ["C2H4", "ETHYLENE"],
    "CH4": ["CH4", "METHANE"],
    "H2": ["H2", "HYDROGEN"],
    "C2H2": ["C2H2", "ACETYLENE"],
    "C3H6": ["C3H6", "PROPYLENE", "C3H6-1"],
    "C3H8": ["C3H8", "PROPANE"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    if not cols:
        return None
    exact = {c: c for c in cols}
    for cand in candidates:
        if cand in exact:
            return cand
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        out = nmap.get(_norm(cand))
        if out is not None:
            return out
    return None


def find_species_columns(columns: Iterable[str]) -> dict[str, str]:
    cols = list(columns)
    nmap = {_norm(c): c for c in cols}
    out: dict[str, str] = {}
    for sp, aliases in SPECIES_ALIASES.items():
        candidates: list[str] = []
        for a in aliases:
            candidates += [a, f"Y_{a}", f"X_{a}", f"Y-{a}", f"X-{a}", f"Mass fraction {a}", f"mass_fraction_{a}", f"Mole fraction {a}", f"mole_fraction_{a}"]
        for cand in candidates:
            c = nmap.get(_norm(cand))
            if c is not None:
                out[sp] = c
                break
    return out


def find_all_y_columns(columns: Iterable[str]) -> list[str]:
    """Return all Y_* mass-fraction columns in the reference/database order."""
    out: list[str] = []
    for c in columns:
        cs = str(c)
        if cs.startswith("Y_") and not cs.startswith("Y_dot") and not cs.startswith("Ydot"):
            out.append(cs)
    return out


STEAM_TO_ETHANE_CANDIDATES = [
    "steam_to_ethane [kg/kg]", "steam_to_ethane_mass", "steam/C2H6 [kg/kg]",
    "steam_to_ethane", "steam_C2H6", "H2O/C2H6 [kg/kg]",
]


@dataclass
class ColumnMap:
    T_col: str
    p_col: str | None
    tau_col: str | None
    species_cols: dict[str, str]
    all_y_cols: list[str]
    steam_to_ethane_col: str | None = None


def inspect_schema(paths: list[Path]) -> ColumnMap:
    for path in paths:
        if not path.exists():
            continue
        schema = pq.read_schema(str(path))
        cols = list(schema.names)
        T_col = find_first(cols, TEMP_CANDIDATES)
        if T_col is None:
            continue
        return ColumnMap(
            T_col=T_col,
            p_col=find_first(cols, PRESSURE_CANDIDATES),
            tau_col=find_first(cols, TAU_CANDIDATES),
            species_cols=find_species_columns(cols),
            all_y_cols=find_all_y_columns(cols),
            steam_to_ethane_col=find_first(cols, STEAM_TO_ETHANE_CANDIDATES),
        )
    raise SystemExit("Could not inspect any valid parquet schema. Check --full/--off paths.")


def required_columns(cmap: ColumnMap) -> list[str]:
    cols = [cmap.T_col]
    if cmap.p_col:
        cols.append(cmap.p_col)
    if cmap.tau_col:
        cols.append(cmap.tau_col)
    for sp in ["H2O", "C2H6", "C2H4", "CH4", "H2", "C2H2", "C3H6", "C3H8"]:
        if sp in cmap.species_cols:
            cols.append(cmap.species_cols[sp])
    # Preserve order / uniqueness
    return list(dict.fromkeys(cols))


def reservoir_sample_parquet(path: Path, columns: list[str], n: int, seed: int) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"Input path does not exist and will be skipped: {path}")
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pf = pq.ParquetFile(str(path))
    selected: list[pd.DataFrame] = []
    total_seen = 0
    # Batch-level approximate reservoir: cheap and robust for multi-GB parquet.
    for batch in pf.iter_batches(batch_size=65536, columns=[c for c in columns if c in pf.schema_arrow.names]):
        df = batch.to_pandas()
        if df.empty:
            continue
        total_seen += len(df)
        if len(df) > n:
            df = df.iloc[rng.choice(len(df), size=n, replace=False)].copy()
        selected.append(df)
        # Periodically downsample pooled selection to keep memory bounded.
        if sum(len(x) for x in selected) > 3 * n:
            pooled = pd.concat(selected, ignore_index=True)
            if len(pooled) > n:
                pooled = pooled.iloc[rng.choice(len(pooled), size=n, replace=False)].copy()
            selected = [pooled]
    if not selected:
        return pd.DataFrame()
    pooled = pd.concat(selected, ignore_index=True)
    if len(pooled) > n:
        pooled = pooled.iloc[rng.choice(len(pooled), size=n, replace=False)].copy()
    pooled.reset_index(drop=True, inplace=True)
    pooled.attrs["total_seen"] = total_seen
    return pooled


def existing_state_sample(paths: list[Path], cmap: ColumnMap, max_rows: int, seed: int, T_hard_max_K: float) -> pd.DataFrame:
    cols = required_columns(cmap)
    n_per = max(1, int(math.ceil(max_rows / max(1, len(paths)))))
    frames = []
    for i, p in enumerate(paths):
        df = reservoir_sample_parquet(p, cols, n_per, seed + i * 7919)
        if not df.empty:
            df["_source_file"] = str(p)
            frames.append(df)
    if not frames:
        raise SystemExit("No valid --full/--off parquet paths were found or sampled.")
    out = pd.concat(frames, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    # Only remove states above the true hard cap. Do NOT drop >1400 K.
    T = pd.to_numeric(out[cmap.T_col], errors="coerce")
    out = out[np.isfinite(T) & (T <= T_hard_max_K + 1e-9)].copy()
    out.reset_index(drop=True, inplace=True)
    return out


def compute_conversion_proxy(df: pd.DataFrame, cmap: ColumnMap) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.array([], dtype=float)
    y_h2o = np.zeros(n, dtype=float)
    y_c2h6 = np.zeros(n, dtype=float)
    if "H2O" in cmap.species_cols and cmap.species_cols["H2O"] in df.columns:
        y_h2o = pd.to_numeric(df[cmap.species_cols["H2O"]], errors="coerce").fillna(0.0).to_numpy(float)
    if "C2H6" in cmap.species_cols and cmap.species_cols["C2H6"] in df.columns:
        y_c2h6 = pd.to_numeric(df[cmap.species_cols["C2H6"]], errors="coerce").fillna(0.0).to_numpy(float)
    dry = np.clip(1.0 - y_h2o, EPS, None)
    ethane_on_dry_basis = np.clip(y_c2h6 / dry, 0.0, 1.0)
    X = 1.0 - ethane_on_dry_basis
    return np.clip(X, 0.0, 1.0)


def compute_pressure(df: pd.DataFrame, cmap: ColumnMap) -> np.ndarray:
    if cmap.p_col and cmap.p_col in df.columns:
        return pd.to_numeric(df[cmap.p_col], errors="coerce").to_numpy(float)
    return np.full(len(df), np.nan)


def compute_tau(df: pd.DataFrame, cmap: ColumnMap) -> np.ndarray:
    if cmap.tau_col and cmap.tau_col in df.columns:
        return pd.to_numeric(df[cmap.tau_col], errors="coerce").to_numpy(float)
    return np.full(len(df), np.nan)



def compute_steam_to_ethane(df: pd.DataFrame, cmap: ColumnMap) -> np.ndarray:
    """Return an approximate steam/ethane mass-ratio metadata vector for anchor matching.

    Prefer the explicit database metadata column if present. Falling back to Y_H2O/Y_C2H6
    is intentionally clipped because product-rich states can have very low C2H6 and the
    instantaneous ratio then stops representing the inlet dilution.
    """
    n = len(df)
    if cmap.steam_to_ethane_col and cmap.steam_to_ethane_col in df.columns:
        val = pd.to_numeric(df[cmap.steam_to_ethane_col], errors="coerce").to_numpy(float)
        return np.clip(val, 0.0, 1.0)
    if "H2O" in cmap.species_cols and "C2H6" in cmap.species_cols:
        yh = pd.to_numeric(df[cmap.species_cols["H2O"]], errors="coerce").fillna(0.0).to_numpy(float)
        ye = pd.to_numeric(df[cmap.species_cols["C2H6"]], errors="coerce").fillna(0.0).to_numpy(float)
        approx = yh / np.clip(ye, 1.0e-6, None)
        return np.clip(approx, 0.0, 1.0)
    return np.full(n, 0.5, dtype=float)


def anchor_required_columns(cmap: ColumnMap) -> list[str]:
    """Columns needed to select real full-species composition anchors."""
    cols = [cmap.T_col]
    if cmap.p_col:
        cols.append(cmap.p_col)
    if cmap.tau_col:
        cols.append(cmap.tau_col)
    if cmap.steam_to_ethane_col:
        cols.append(cmap.steam_to_ethane_col)
    # Add the full mechanism Y vector, not only the 8 design species.
    cols.extend(cmap.all_y_cols)
    # Make sure important aliases are included even if not in all_y_cols for some reason.
    cols.extend(required_columns(cmap))
    return list(dict.fromkeys(cols))


def build_anchor_pool(paths: list[Path], cmap: ColumnMap, n_rows: int, seed: int, T_hard_max_K: float) -> pd.DataFrame:
    """Sample real CRACKSIM states with the full Y_* vector for state-probe composition anchors."""
    if not cmap.all_y_cols:
        warnings.warn("No full Y_* columns found; state probes will fall back to synthetic composition. This is not recommended.")
        return pd.DataFrame()
    cols = anchor_required_columns(cmap)
    n_per = max(1, int(math.ceil(max(1, n_rows) / max(1, len(paths)))))
    frames = []
    for i, path in enumerate(paths):
        df = reservoir_sample_parquet(path, cols, n_per, seed + 314159 + i * 2713)
        if df.empty:
            continue
        df["_anchor_source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True)
    if len(pool) > n_rows:
        pool = pool.sample(n=n_rows, random_state=seed + 314159).reset_index(drop=True)
    T = pd.to_numeric(pool[cmap.T_col], errors="coerce")
    pool = pool[np.isfinite(T) & (T <= T_hard_max_K + 1e-9)].copy()
    if pool.empty:
        return pool
    pool.reset_index(drop=True, inplace=True)
    pool["_anchor_pool_id"] = np.arange(len(pool), dtype=int)
    pool["_anchor_conversion_proxy"] = compute_conversion_proxy(pool, cmap)
    pool["_anchor_steam_to_ethane_mass"] = compute_steam_to_ethane(pool, cmap)
    if cmap.p_col and cmap.p_col in pool.columns:
        pool["_anchor_pressure_Pa"] = pd.to_numeric(pool[cmap.p_col], errors="coerce").to_numpy(float)
    else:
        pool["_anchor_pressure_Pa"] = np.nan
    if cmap.tau_col and cmap.tau_col in pool.columns:
        pool["_anchor_tau_s"] = pd.to_numeric(pool[cmap.tau_col], errors="coerce").to_numpy(float)
    else:
        pool["_anchor_tau_s"] = np.nan
    # Remove obviously invalid composition rows.
    ysum = np.zeros(len(pool), dtype=float)
    for c in cmap.all_y_cols:
        if c in pool.columns:
            ysum += pd.to_numeric(pool[c], errors="coerce").fillna(0.0).to_numpy(float)
    pool = pool[np.isfinite(ysum) & (ysum > 0.5) & (ysum < 1.5)].copy()
    pool.reset_index(drop=True, inplace=True)
    pool["_anchor_pool_id"] = np.arange(len(pool), dtype=int)
    return pool


def choose_real_state_anchor(
    anchor_pool: pd.DataFrame,
    cmap: ColumnMap,
    target_T: float,
    target_X: float,
    target_s2e: float,
    target_p: float,
    rng: np.random.Generator,
    max_steam_abs_diff: float = 0.05,
    max_conversion_abs_diff: float = 0.20,
    max_temperature_abs_diff_K: float = 350.0,
    conversion_scale: float = 0.05,
    steam_scale: float = 0.025,
    pressure_scale_Pa: float = 7.5e4,
    temperature_scale_K: float = 125.0,
    random_top_k: int = 16,
) -> pd.Series | None:
    """Choose a real full-species CRACKSIM composition anchor for a state probe.

    v8 change: anchor selection is now strict on steam dilution and more chemically
    consistent.  We do NOT use a latent/PCA space because that model does not exist yet.
    Instead we use a physically transparent nearest-neighbour metric over conversion,
    inlet steam dilution, pressure and temperature.

    The hard steam gate is important: if the copied full-species Y-vector was generated
    at H2O/C2H6 = 0.30, we should not pretend it represents a H2O/C2H6 = 0.90 state.
    If no suitable anchor exists, return None and let the caller skip that state_probe
    instead of falling back to a synthetic product split.
    """
    if anchor_pool is None or anchor_pool.empty:
        return None

    X = anchor_pool["_anchor_conversion_proxy"].to_numpy(float)
    S = anchor_pool["_anchor_steam_to_ethane_mass"].to_numpy(float)
    T = pd.to_numeric(anchor_pool[cmap.T_col], errors="coerce").to_numpy(float) if cmap.T_col in anchor_pool.columns else np.full(len(anchor_pool), np.nan)
    P = anchor_pool["_anchor_pressure_Pa"].to_numpy(float) if "_anchor_pressure_Pa" in anchor_pool.columns else np.full(len(anchor_pool), np.nan)

    valid = np.isfinite(X) & np.isfinite(S) & (np.abs(S - float(target_s2e)) <= float(max_steam_abs_diff))
    if np.isfinite(max_conversion_abs_diff):
        valid &= np.abs(X - float(target_X)) <= float(max_conversion_abs_diff)
    if np.isfinite(target_T) and np.isfinite(max_temperature_abs_diff_K) and np.isfinite(T).any():
        # Temperature is a soft/medium constraint: we prefer a similar radical/intermediate
        # pool, but still allow transported/cooled states within a wide physical window.
        valid &= np.where(np.isfinite(T), np.abs(T - float(target_T)) <= float(max_temperature_abs_diff_K), True)

    if not valid.any():
        return None

    # Weighted transparent distance. Conversion and steam dominate; T and p break ties
    # and reduce selection of very different radical/intermediate pools.
    d = np.full(len(anchor_pool), np.inf, dtype=float)
    d_valid = ((X[valid] - float(target_X)) / max(float(conversion_scale), 1e-12)) ** 2
    d_valid += ((S[valid] - float(target_s2e)) / max(float(steam_scale), 1e-12)) ** 2
    if np.isfinite(target_p) and np.isfinite(P).any():
        psel = np.where(np.isfinite(P[valid]), P[valid], target_p)
        d_valid += 0.35 * ((psel - float(target_p)) / max(float(pressure_scale_Pa), 1.0)) ** 2
    if np.isfinite(target_T) and np.isfinite(T).any():
        tsel = np.where(np.isfinite(T[valid]), T[valid], target_T)
        d_valid += 0.50 * ((tsel - float(target_T)) / max(float(temperature_scale_K), 1.0)) ** 2
    d[valid] = d_valid

    if not np.isfinite(d).any():
        return None

    # Pick among top-k very close candidates to avoid reusing the exact same row too often,
    # but keep k small so anchors remain close.
    k = min(max(1, int(random_top_k)), int(np.isfinite(d).sum()))
    idx = np.argpartition(d, k - 1)[:k]
    idx = idx[np.argsort(d[idx])]
    chosen = int(idx[int(rng.integers(0, k))])
    return anchor_pool.iloc[chosen].copy()


def composition_from_conversion(X: np.ndarray, steam_to_ethane_mass: np.ndarray) -> dict[str, np.ndarray]:
    X = np.clip(np.asarray(X, dtype=float), 0.0, 0.999999)
    s2e = np.clip(np.asarray(steam_to_ethane_mass, dtype=float), 0.0, None)
    ethane_unconverted = 1.0 - X
    product_mass = X
    steam_mass = s2e
    # Approximate product split used only to generate states. CRACKSIM calculates exact rates later.
    splits = {"C2H4": 0.58, "CH4": 0.18, "H2": 0.04, "C2H2": 0.08, "C3H6": 0.07, "C3H8": 0.02, "C2H6": 0.03}
    total = np.clip(ethane_unconverted + product_mass + steam_mass, EPS, None)
    y = {sp: np.zeros_like(X, dtype=float) for sp in DESIGN_SPECIES}
    y["H2O"] = steam_mass / total
    y["C2H6"] = (ethane_unconverted + splits["C2H6"] * product_mass) / total
    for sp, frac in splits.items():
        if sp == "C2H6":
            continue
        y[sp] = frac * product_mass / total
    ysum = np.zeros_like(X, dtype=float)
    for val in y.values():
        ysum += val
    for sp in y:
        y[sp] = np.clip(y[sp] / np.clip(ysum, EPS, None), 0.0, 1.0)
    return y



def approximate_mixture_mw_kg_per_mol(y: dict[str, np.ndarray]) -> np.ndarray:
    """Return mixture molecular weight from mass fractions for design-only metadata."""
    first = next(iter(y.values()))
    denom = np.zeros_like(first, dtype=float)
    for sp, arr in y.items():
        mw = MW_KG_PER_MOL.get(sp)
        if mw is None:
            continue
        denom += np.asarray(arr, dtype=float) / max(mw, EPS)
    return 1.0 / np.clip(denom, EPS, None)


def approximate_rho_mu(T: np.ndarray, p: np.ndarray, y: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Approximate density and viscosity for selecting D/Re/L/mdot in the manifest.

    This script intentionally does not initialise Cantera/CRACKSIM. The values are
    used only to choose a plausible hydrodynamic design. The production generator
    recomputes rho and mu with Cantera and writes the authoritative metadata.
    """
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    mw = approximate_mixture_mw_kg_per_mol(y)
    rho = p * mw / np.clip(R_UNIVERSAL * T, EPS, None)
    # Simple high-temperature gas viscosity estimate, Pa.s. Good enough for Re-based design.
    mu = 3.5e-5 * np.clip(T / 1000.0, 0.2, 3.0) ** 0.70
    return rho, mu


def choose_hydrodynamics_for_tau(
    tau_s: float,
    T_K: float,
    p_Pa: float,
    comp: dict[str, np.ndarray],
    u_diam: float,
    u_re: float,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    """Choose D and Re, then compute U, mdot and L so that tau=L/U.

    The selected point therefore carries a physically interpretable residence-time
    coordinate instead of a free off-manifold tag.  The source terms are still local
    f(T,p,Y); these flow quantities are metadata for coverage/CFD relevance.
    """
    diameters = [float(d) for d in getattr(args, "diameters_m", [0.0306, 0.05, 0.1])]
    diameters = [d for d in diameters if d > 0]
    if not diameters:
        diameters = [0.0306]
    idx = min(len(diameters) - 1, int(np.floor(float(u_diam) * len(diameters))))
    D = float(diameters[idx])
    A = math.pi * D * D / 4.0
    rho_arr, mu_arr = approximate_rho_mu(np.array([T_K]), np.array([p_Pa]), comp)
    rho = float(rho_arr[0])
    mu = float(mu_arr[0])
    tau = max(float(tau_s), EPS)

    # Feasible U from direct U bounds and L=tau*U bounds.
    U_low = max(float(args.U_min_m_s), float(args.L_min_m) / tau)
    U_high = min(float(args.U_max_m_s), float(args.L_max_m) / tau)
    if U_low > U_high:
        # If the requested tau is too extreme for the requested L/U limits, use the
        # closest feasible compromise instead of dropping the case.
        U_star = math.sqrt(max(float(args.U_min_m_s), EPS) * max(float(args.U_max_m_s), EPS))
        U_star = float(np.clip(U_star, float(args.U_min_m_s), float(args.U_max_m_s)))
        L = U_star * tau
        L = float(np.clip(L, float(args.L_min_m), float(args.L_max_m)))
        U_low = U_high = max(L / tau, EPS)

    Re_low_from_U = U_low * rho * D / max(mu, EPS)
    Re_high_from_U = U_high * rho * D / max(mu, EPS)
    Re_low = max(float(args.Re_min), Re_low_from_U)
    Re_high = min(float(args.Re_max), Re_high_from_U)
    if Re_low <= Re_high:
        Re = float(np.exp(np.log(max(Re_low, EPS)) + float(u_re) * (np.log(max(Re_high, EPS)) - np.log(max(Re_low, EPS)))))
        U = Re * mu / max(rho * D, EPS)
    else:
        # No overlap with requested Re range. Preserve tau/L/U feasibility and report the resulting Re.
        U = float(np.exp(np.log(max(U_low, EPS)) + float(u_re) * (np.log(max(U_high, EPS)) - np.log(max(U_low, EPS))))) if U_low < U_high else float(U_low)
        Re = U * rho * D / max(mu, EPS)
    U = float(np.clip(U, U_low, U_high))
    L = float(U * tau)
    mdot = float(rho * A * U)
    return {
        "hydro_design_mode": "D_Re_sampled__L_mdot_from_tau",
        "diameter_m": D,
        "area_m2": A,
        "target_Re": float(Re),
        "estimated_rho_kg_m3": rho,
        "estimated_mu_Pa_s": mu,
        "estimated_U_m_s": float(U),
        "estimated_mdot_kg_s": mdot,
        "estimated_length_m": L,
    }

def deprecated_tau_ratio_placeholder(*args, **kwargs):
    """v9 removed hardcoded Arrhenius reachability classification.

    The design stage no longer decides whether a point is reachable by a
    fresh-feed isothermal PFR. It writes pfr_first_request rows. The production
    generator must run the actual CRACKSIM isothermal PFR and then decide
    whether the requested sparse bin was hit.
    """
    return np.nan


def design_kind_for_request() -> str:
    return "pfr_first_request"


def physicality_flag_for_request() -> str:
    return "pending_cracksim_pfr_reachability_check"


def region_name(T: float, X: float) -> str:
    if 1200.0 <= T <= 1400.0 and X <= 0.30:
        return "gap_1200_1400_low_conversion"
    if 800.0 <= T <= 1000.0 and X >= 0.65:
        return "gap_800_1000_high_conversion"
    if T >= 1400.0:
        return "global_high_temperature_1400_1600"
    return "global_sparse_tx"


def design_kind(T: float, X: float) -> str:
    # Low-T/high-conversion states are not efficiently reachable from fresh-feed isothermal PFRs;
    # keep them as direct state probes for CFD/OOD protection.
    if T <= 1050.0 and X >= 0.45:
        return "state_probe"
    return "isothermal_pfr"


def design_kind_with_tau_gate(T: float, X: float, tau_ratio: float, args: argparse.Namespace) -> tuple[str, str]:
    """Classify whether a designed T-X-tau point may be represented as a PFR trajectory.

    A large/small tau/tau_estimate means the requested residence time is deliberately
    off the crude kinetic manifold. Treating such a point as a 1200-row PFR trajectory
    would create a fake trajectory where conversion is forced to evolve too fast/slow.
    Therefore, by default, off-trajectory T-X-tau points are direct CRACKSIM state probes
    with physically interpretable hydrodynamic metadata.
    """
    base = design_kind(T, X)
    if base == "state_probe":
        return "state_probe", "lowT_highX_not_reachable_from_fresh_feed"
    if getattr(args, "allow_offtrajectory_pfr", False):
        return "isothermal_pfr", "offtrajectory_PFR_allowed_by_user"
    lo = float(getattr(args, "pfr_tau_ratio_min", 0.2))
    hi = float(getattr(args, "pfr_tau_ratio_max", 5.0))
    if not np.isfinite(tau_ratio):
        return "state_probe", "nonfinite_tau_ratio"
    if tau_ratio < lo:
        return "state_probe", "tau_much_shorter_than_kinetic_estimate"
    if tau_ratio > hi:
        return "state_probe", "tau_much_longer_than_kinetic_estimate"
    return "isothermal_pfr", "near_kinetic_tau_estimate"


def bin_edges(T_min: float, T_max: float, T_bin: float, X_bin: float) -> tuple[np.ndarray, np.ndarray]:
    T_edges = np.arange(T_min, T_max + 0.5 * T_bin, T_bin)
    if T_edges[-1] < T_max:
        T_edges = np.r_[T_edges, T_max]
    X_edges = np.arange(0.0, 1.0 + 0.5 * X_bin, X_bin)
    if X_edges[-1] < 1.0:
        X_edges = np.r_[X_edges, 1.0]
    return T_edges, X_edges


def tx_hist(T: np.ndarray, X: np.ndarray, T_edges: np.ndarray, X_edges: np.ndarray) -> np.ndarray:
    mask = np.isfinite(T) & np.isfinite(X)
    H, _, _ = np.histogram2d(T[mask], X[mask], bins=[T_edges, X_edges])
    return H.astype(int)






def tau_bin_edges(tau_min: float, tau_max: float, logtau_bin_width: float) -> np.ndarray:
    """Log10 residence-time bin edges."""
    tau_min = max(float(tau_min), 1e-12)
    tau_max = max(float(tau_max), tau_min * 1.000001)
    w = max(float(logtau_bin_width), 1e-6)
    lo = math.floor(math.log10(tau_min) / w) * w
    hi = math.ceil(math.log10(tau_max) / w) * w
    edges = np.arange(lo, hi + 0.5 * w, w)
    if edges[-1] < hi:
        edges = np.r_[edges, hi]
    return edges.astype(float)


def ttau_hist(T: np.ndarray, tau: np.ndarray, T_edges: np.ndarray, logtau_edges: np.ndarray) -> np.ndarray:
    """2D histogram over T and log10(tau)."""
    T = np.asarray(T, dtype=float)
    tau = np.asarray(tau, dtype=float)
    logtau = np.log10(np.clip(tau, 1e-300, None))
    mask = np.isfinite(T) & np.isfinite(logtau)
    H, _, _ = np.histogram2d(T[mask], logtau[mask], bins=[T_edges, logtau_edges])
    return H.astype(int)


def choose_logtau_bin_for_T(
    H_ttau: np.ndarray,
    T_bin_i: int,
    logtau_edges: np.ndarray,
    target_count: int,
    u_select: float,
) -> tuple[int, float, float, int]:
    """Choose a tau bin for the selected temperature bin by water-filling.

    This is the key v5 change: tau is no longer derived only from a kinetic proxy.
    Inside each selected T-X bin, we choose a residence-time bin that is least
    populated in T-log(tau) space.  The Sobol variable u_select spreads choices
    across equally sparse bins instead of repeatedly picking the first one.
    """
    i = int(np.clip(T_bin_i, 0, H_ttau.shape[0] - 1))
    counts = H_ttau[i, :].astype(int)
    target = int(max(1, target_count))
    deficits = np.maximum(0, target - counts)
    candidates = np.where(deficits > 0)[0]
    if len(candidates) == 0:
        # All tau bins for this T bin already satisfy the requested target. Continue
        # by filling the lowest-count tau bins so the plot stays uniform when the
        # global T-X objective still asks for more cases.
        min_count = counts.min() if len(counts) else 0
        candidates = np.where(counts == min_count)[0]
    # Prefer the largest deficits, then use Sobol selection among ties.
    if len(candidates) > 0:
        max_def = deficits[candidates].max()
        if max_def > 0:
            candidates = candidates[deficits[candidates] == max_def]
    if len(candidates) == 0:
        k = 0
    else:
        k = int(candidates[min(len(candidates) - 1, int(np.floor(float(u_select) * len(candidates))))])
    lo, hi = float(logtau_edges[k]), float(logtau_edges[k + 1])
    return k, lo, hi, int(counts[k])


def effective_ttau_target_count(args: argparse.Namespace) -> int:
    """Target count per T-log(tau) bin.

    Kept separate from the T-X target because the tau grid may be coarser/finer
    than the T-X grid. By default, it follows --target-log10-count.
    """
    if getattr(args, "target_ttau_log10_count", None) is not None:
        val = float(args.target_ttau_log10_count)
        if not np.isfinite(val) or val <= 0:
            raise SystemExit("--target-ttau-log10-count must be a positive finite number")
        return int(max(1, math.ceil(10.0 ** val - 1.0)))
    return effective_target_count(args)


def sobol_unit(n: int, d: int, seed: int) -> np.ndarray:
    """Return n quasi-random points in [0, 1]^d.

    Uses scipy Sobol when available and falls back to a deterministic pseudo-random
    generator if scipy is not installed. The script remains usable on lightweight
    Windows environments, while still preferring Sobol coverage when possible.
    """
    n = int(max(0, n))
    d = int(max(1, d))
    if n == 0:
        return np.empty((0, d), dtype=float)
    try:
        from scipy.stats import qmc  # type: ignore
        m = int(math.ceil(math.log2(max(1, n))))
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        pts = sampler.random_base2(m=m)
        return np.asarray(pts[:n], dtype=float)
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, d))


def tx_bin_priority(Tc: float, Xc: float) -> int:
    """Priority for ordering deficits. It does not exclude global gaps."""
    if 1200.0 <= Tc <= 1400.0 and Xc <= 0.30:
        return 6
    if 800.0 <= Tc <= 1000.0 and Xc >= 0.65:
        return 6
    if Tc >= 1400.0:
        return 2
    return 1


def underfilled_bins(
    H: np.ndarray,
    T_edges: np.ndarray,
    X_edges: np.ndarray,
    target_count: int,
    priority_boost: np.ndarray | None = None,
) -> list[dict]:
    """Return all bins below target_count with their deficit and priority.

    priority_boost is a case-level feedback map from previous isothermal rounds.
    It does not manufacture coverage; it only changes which still-underfilled bins
    are tried first.
    """
    bins: list[dict] = []
    target = int(max(1, target_count))
    for i in range(H.shape[0]):
        Tc = 0.5 * (T_edges[i] + T_edges[i + 1])
        for j in range(H.shape[1]):
            Xc = 0.5 * (X_edges[j] + X_edges[j + 1])
            count = int(H[i, j])
            if count >= target:
                continue
            deficit = target - count
            extra = 0
            if priority_boost is not None and i < priority_boost.shape[0] and j < priority_boost.shape[1]:
                extra = int(priority_boost[i, j])
            bins.append({
                "i": i,
                "j": j,
                "Tc": float(Tc),
                "Xc": float(Xc),
                "count": count,
                "deficit": int(deficit),
                "priority": int(tx_bin_priority(Tc, Xc) + extra),
                "case_feedback_boost": int(extra),
            })
    return bins


def choose_bins_by_deficit_round(H: np.ndarray, T_edges: np.ndarray, X_edges: np.ndarray, target_count: int,
                                 n_remaining: int, seed: int,
                                 priority_boost: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Choose bins to fill in a balanced, round-based way.

    The algorithm is deliberately not pure random sampling:
    - First it identifies the bins that are below the requested minimum count.
    - It then allocates one point per missing level before allocating second/third
      points to the same bin. This maximizes terrain coverage when the case budget is
      limited.
    - Within each deficit level the ordering is quasi-random Sobol-ranked, with a
      mild priority boost for the two user-identified gaps and the high-T tail.

    If n_remaining is larger than the total deficit, the returned list is shorter;
    this lets the script stop early instead of blindly using all requested cases.
    """
    target = int(max(1, target_count))
    n_remaining = int(max(0, n_remaining))
    if n_remaining == 0:
        return []
    bins = underfilled_bins(H, T_edges, X_edges, target, priority_boost=priority_boost)
    if not bins:
        return []

    # Sobol rank over bin centres: gives deterministic spread across T-X terrain.
    # We use the distance from each bin centre to a Sobol point ordering surrogate.
    # This avoids marching systematically from low T to high T.
    sob = sobol_unit(max(1, len(bins)), 2, seed)
    for k, b in enumerate(bins):
        Tc_norm = (b["Tc"] - T_edges[0]) / max(T_edges[-1] - T_edges[0], EPS)
        Xc_norm = (b["Xc"] - X_edges[0]) / max(X_edges[-1] - X_edges[0], EPS)
        # Match each bin with its own Sobol jitter. The exact value only defines ordering.
        b["sobol_rank"] = float((Tc_norm - sob[k, 0]) ** 2 + (Xc_norm - sob[k, 1]) ** 2)

    chosen: list[tuple[int, int, int]] = []
    # Fill by deficit level: first all bins with at least one missing point, then those
    # with at least two missing points, etc. This is what makes the spacing controlled
    # by the user's max-case budget.
    for level in range(1, target + 1):
        level_bins = [b for b in bins if b["deficit"] >= level]
        if not level_bins:
            continue
        level_bins.sort(key=lambda b: (-b["priority"], b["sobol_rank"], b["Tc"], b["Xc"]))
        for b in level_bins:
            if len(chosen) >= n_remaining:
                return chosen
            chosen.append((int(b["i"]), int(b["j"]), int(b["priority"])))
    return chosen


def effective_target_count(args: argparse.Namespace) -> int:
    """Return the target bin count used by the T-X water-filling design.

    --target-log10-count is the preferred scientific-control knob: for example
    2.3 means target count ≈ 10**2.3 - 1 ≈ 199 per T-X bin.  This will usually
    be budget-limited when --n-new-cases is capped by the requested campaign budget, but it forces the
    algorithm to spend the available budget on the least populated bins rather
    than stopping after a few points per empty bin.
    """
    if getattr(args, "target_log10_count", None) is not None:
        val = float(args.target_log10_count)
        if not np.isfinite(val) or val <= 0:
            raise SystemExit("--target-log10-count must be a positive finite number")
        return int(max(1, math.ceil(10.0 ** val - 1.0)))
    return int(max(1, args.target_gap_min_bin_count))


def make_manifest(existing: pd.DataFrame, cmap: ColumnMap, args: argparse.Namespace, anchor_pool: pd.DataFrame | None = None, iso_case_feedback: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(args.seed)
    T_existing = pd.to_numeric(existing[cmap.T_col], errors="coerce").to_numpy(float)
    X_existing = compute_conversion_proxy(existing, cmap)
    T_edges, X_edges = bin_edges(args.T_min_K, args.T_hard_max_K, args.tx_T_bin_width_K, args.tx_X_bin_width)
    logtau_edges = tau_bin_edges(args.tau_min_s, args.tau_max_s, args.ttau_logtau_bin_width_decades)
    tau_existing = compute_tau(existing, cmap)
    # Existing off-manifold rows may have missing/zero tau. Only finite tau values are
    # allowed to count toward T-log(tau) coverage. This prevents fake zeros/NaNs from
    # making the tau map look filled when it is actually untracked.
    tau_existing = np.where(np.isfinite(tau_existing) & (tau_existing > 0), tau_existing, np.nan)
    H0 = tx_hist(T_existing, X_existing, T_edges, X_edges)
    Htau0 = ttau_hist(T_existing, tau_existing, T_edges, logtau_edges)
    iso_priority_boost_TX, iso_priority_boost_Ttau, iso_feedback_summary = iso_case_feedback_maps(
        iso_case_feedback if iso_case_feedback is not None else pd.DataFrame(),
        T_edges, X_edges, logtau_edges, args,
    )

    H = H0.copy()
    Htau = Htau0.copy()
    rows: list[dict] = []
    n_target = min(int(args.n_new_cases), int(args.max_new_cases_per_run))
    if n_target > 20000:
        raise SystemExit("Refusing to design more than 20000 cases per command. Use --max-new-cases-per-run <= 20000.")

    target_count = effective_target_count(args)
    ttau_target_count = effective_ttau_target_count(args)
    n_rounds = max(1, int(args.coverage_refine_rounds))
    # Make rounds real: a 20000-case design with 8 rounds adds roughly 2500 cases,
    # updates coverage, then redistributes the next 1250 to the bins that are still
    # least populated. This behaves like water-filling under a finite case budget.
    per_round_budget = max(1, int(math.ceil(n_target / n_rounds)))

    round_summaries: list[dict] = []
    skipped_no_anchor_total = 0
    for round_id in range(n_rounds):
        if len(rows) >= n_target:
            break
        n_remaining = n_target - len(rows)
        n_this_round = min(n_remaining, per_round_budget)
        # Overdraw candidate bins because v4 can skip candidates without a valid
        # fallback anchor. This avoids spending the case budget on targets that the
        # previous round already showed cannot be filled safely by a probe.
        n_candidate_bins = min(
            max(n_this_round, int(math.ceil(n_this_round * float(args.candidate_overdraw_factor)))),
            max(n_this_round, int(np.maximum(0, target_count - H).sum())),
        )
        bins = choose_bins_by_deficit_round(
            H, T_edges, X_edges, target_count, n_candidate_bins,
            seed=int(args.seed + 1009 * round_id),
            priority_boost=iso_priority_boost_TX,
        )
        if not bins:
            round_summaries.append({
                "round": round_id,
                "status": "TX_coverage_target_reached",
                "cases_added": 0,
                "underfilled_TX_bins": int((H < target_count).sum()),
                "underfilled_Ttau_bins": int((Htau < ttau_target_count).sum()),
            })
            break

        # Sobol dimensions: T within selected bin, X within bin, pressure, dilution,
        # tau bin selection, tau-in-bin coordinate, composition jitter, diameter choice, Re choice.
        # v5 change: tau is an explicit water-filled design dimension in T-log(tau) space.
        sob = sobol_unit(len(bins), 9, seed=int(args.seed + 7919 + 101 * round_id))
        added_this_round = 0
        skipped_no_anchor_round = 0
        for local_id, (i, j, priority) in enumerate(bins):
            if len(rows) >= n_target or added_this_round >= n_this_round:
                break
            u = sob[local_id]
            Tlo, Thi = float(T_edges[i]), float(T_edges[i + 1])
            Xlo, Xhi = float(X_edges[j]), float(X_edges[j + 1])

            # Keep away from exact bin boundaries to reduce accidental duplicate bin edges.
            T = Tlo + (0.05 + 0.90 * u[0]) * (Thi - Tlo)
            X = Xlo + (0.05 + 0.90 * u[1]) * (Xhi - Xlo)
            T = float(np.clip(T, args.T_min_K, args.T_hard_max_K))
            X = float(np.clip(X, 0.0, 0.999))
            p = float(args.pressure_min_Pa + u[2] * (args.pressure_max_Pa - args.pressure_min_Pa))
            s2e = float(args.steam_ethane_min_mass + u[3] * (args.steam_ethane_max_mass - args.steam_ethane_min_mass))
            ttau_k, logtau_lo, logtau_hi, ttau_count_before = choose_logtau_bin_for_T(
                Htau, i, logtau_edges, ttau_target_count, u[4]
            )
            # v9: tau is an explicit water-filled design dimension.  No hardcoded
            # Arrhenius estimate is used.  Reachability is tested later by the
            # CRACKSIM generator with an actual isothermal PFR attempt.
            logtau = logtau_lo + (0.05 + 0.90 * u[5]) * (logtau_hi - logtau_lo)
            tau = float(np.clip(10.0 ** logtau, args.tau_min_s, args.tau_max_s))
            tau_design_mode = "T_logtau_waterfill_explicit_tau_pfr_first"
            tau_est = np.nan
            tau_ratio = np.nan
            kind = design_kind_for_request()
            physicality_flag = physicality_flag_for_request()

            # Composition definition for fallback state probes.  We do not invent an
            # 8-species product split.  Instead, each PFR-first request carries an
            # optional fallback full-Y anchor copied from a real CRACKSIM database state.
            # The production generator uses it only if the actual PFR attempt misses
            # the requested sparse bin.
            X_for_comp = float(np.clip(X + (u[6] - 0.5) * min(args.tx_X_bin_width, 0.04), 0.0, 0.999))
            comp = composition_from_conversion(np.array([0.0]), np.array([s2e]))  # fresh feed for hydrodynamic estimate
            anchor = None
            if anchor_pool is not None and not anchor_pool.empty:
                anchor = choose_real_state_anchor(
                    anchor_pool, cmap, T, X_for_comp, s2e, p, rng,
                    max_steam_abs_diff=args.anchor_steam_max_abs_diff,
                    max_conversion_abs_diff=args.anchor_conversion_max_abs_diff,
                    max_temperature_abs_diff_K=args.anchor_temperature_max_abs_diff_K,
                    conversion_scale=args.anchor_conversion_scale,
                    steam_scale=args.anchor_steam_scale,
                    pressure_scale_Pa=args.anchor_pressure_scale_Pa,
                    temperature_scale_K=args.anchor_temperature_scale_K,
                    random_top_k=args.anchor_random_top_k,
                )
            if anchor is None and getattr(args, "require_fallback_anchor", True):
                skipped_no_anchor_round += 1
                skipped_no_anchor_total += 1
                continue

            hydro = choose_hydrodynamics_for_tau(tau, T, p, comp, u[7], u[8], args)

            row = {
                "case_id": len(rows),
                "design_region": region_name(T, X),
                "design_kind": kind,
                "T_K": T,
                "p_Pa": p,
                "pressure_bar": p / 1.0e5,
                "steam_to_ethane_mass": s2e,
                "tau_end_s": tau,
                "target_tau_s": tau,
                "tau_design_mode": tau_design_mode,
                "target_conversion": X,
                "conversion_proxy": X,
                "coverage_round": round_id,
                "coverage_priority": int(priority),
                "tx_bin_i": int(i),
                "tx_bin_j": int(j),
                "tx_bin_T_low_K": Tlo,
                "tx_bin_T_high_K": Thi,
                "tx_bin_X_low": Xlo,
                "tx_bin_X_high": Xhi,
                "ttau_bin_k": int(ttau_k),
                "ttau_bin_logtau_low": float(logtau_lo),
                "ttau_bin_logtau_high": float(logtau_hi),
                "ttau_bin_tau_low_s": float(10.0 ** logtau_lo),
                "ttau_bin_tau_high_s": float(10.0 ** logtau_hi),
                "ttau_bin_count_before": int(ttau_count_before),
                "tau_estimate_s": tau_est,
                "log10_tau_s": float(math.log10(max(tau, EPS))),
                "log10_tau_estimate_s": np.nan,
                "tau_over_tau_estimate": np.nan,
                "pfr_tau_ratio_min": np.nan,
                "pfr_tau_ratio_max": np.nan,
                "tau_physicality_flag": physicality_flag,
                "script_version": SCRIPT_VERSION,
            }
            row.update(hydro)
            # Fresh-feed columns are supplied for the PFR attempt.  Fallback anchor
            # columns are supplied separately as full Y_* columns and used only if the
            # PFR attempt misses the target bin.
            row["pfr_attempt_composition_source"] = "fresh_feed_ethane_steam"
            row["state_probe_composition_source"] = "real_database_anchor_full_Y_vector" if anchor is not None else "no_suitable_anchor_available"
            for sp in DESIGN_SPECIES:
                row[f"fresh_Y_{sp}"] = float(comp[sp][0])
            if anchor is not None:
                row["anchor_pool_id"] = int(anchor.get("_anchor_pool_id", -1))
                row["anchor_source_file"] = str(anchor.get("_anchor_source_file", "unknown"))
                row["anchor_conversion_proxy"] = float(anchor.get("_anchor_conversion_proxy", np.nan))
                row["anchor_steam_to_ethane_mass"] = float(anchor.get("_anchor_steam_to_ethane_mass", np.nan))
                row["anchor_T_K"] = float(anchor.get(cmap.T_col, np.nan))
                row["anchor_p_Pa"] = float(anchor.get(cmap.p_col, np.nan)) if cmap.p_col else np.nan
                row["anchor_tau_s"] = float(anchor.get(cmap.tau_col, np.nan)) if cmap.tau_col else np.nan
                row["anchor_abs_steam_diff"] = abs(float(row["anchor_steam_to_ethane_mass"]) - float(s2e))
                row["anchor_abs_conversion_diff"] = abs(float(row["anchor_conversion_proxy"]) - float(X_for_comp))
                row["anchor_abs_T_diff_K"] = abs(float(row["anchor_T_K"]) - float(T)) if np.isfinite(row["anchor_T_K"]) else np.nan
                row["anchor_abs_p_diff_Pa"] = abs(float(row["anchor_p_Pa"]) - float(p)) if np.isfinite(row["anchor_p_Pa"]) else np.nan
                row["anchor_selection_rule"] = "strict_steam_gate_plus_weighted_X_T_p_distance"
                row["anchor_steam_max_abs_diff"] = float(args.anchor_steam_max_abs_diff)
                # Copy the full mechanism state vector for possible fallback state probe.
                for ycol in cmap.all_y_cols:
                    row[ycol] = float(anchor.get(ycol, 0.0))
            rows.append(row)
            added_this_round += 1
            # Update mutable coverage immediately. The next round sees what this round filled.
            if 0 <= i < H.shape[0] and 0 <= j < H.shape[1]:
                H[i, j] += 1
            if 0 <= i < Htau.shape[0] and 0 <= ttau_k < Htau.shape[1]:
                Htau[i, ttau_k] += 1

        round_summaries.append({
            "round": round_id,
            "status": "added",
            "cases_added": int(added_this_round),
            "candidate_bins_considered": int(len(bins)),
            "skipped_no_valid_anchor": int(skipped_no_anchor_round),
            "underfilled_bins_after_round": int((H < target_count).sum()),
            "total_deficit_after_round": int(np.maximum(0, target_count - H).sum()),
        })

        # Stop early when all requested bins are filled. This is intentional: 5000 is the max,
        # not a requirement to spend cases in already adequate regions.
        if (not underfilled_bins(H, T_edges, X_edges, target_count, priority_boost=iso_priority_boost_TX)) and int(np.maximum(0, ttau_target_count - Htau).sum()) == 0:
            break

    out = pd.DataFrame(rows)
    H_after = H.copy()
    Htau_after = Htau.copy()
    deficit0 = np.maximum(0, target_count - H0)
    deficit_after = np.maximum(0, target_count - H_after)
    deficit_tau0 = np.maximum(0, ttau_target_count - Htau0)
    deficit_tau_after = np.maximum(0, ttau_target_count - Htau_after)
    report = {
        "script_version": SCRIPT_VERSION,
        "n_existing_sampled": int(len(existing)),
        "n_cases_requested": int(args.n_new_cases),
        "max_new_cases_per_run": int(args.max_new_cases_per_run),
        "n_cases_designed": int(len(out)),
        "stopped_early_because_target_reached": bool(len(out) < n_target and int(deficit_after.sum()) == 0 and int(deficit_tau_after.sum()) == 0),
        "T_range_K": [float(args.T_min_K), float(args.T_hard_max_K)],
        "pressure_range_Pa": [float(args.pressure_min_Pa), float(args.pressure_max_Pa)],
        "steam_to_ethane_mass_range": [float(args.steam_ethane_min_mass), float(args.steam_ethane_max_mass)],
        "tau_range_s": [float(args.tau_min_s), float(args.tau_max_s)],
        "diameters_m": [float(d) for d in args.diameters_m],
        "Re_range": [float(args.Re_min), float(args.Re_max)],
        "U_range_m_s": [float(args.U_min_m_s), float(args.U_max_m_s)],
        "state_probe_anchor_rows": int(0 if anchor_pool is None else len(anchor_pool)),
        "require_fallback_anchor_for_new_cases": bool(getattr(args, "require_fallback_anchor", True)),
        "new_candidates_skipped_no_valid_anchor": int(skipped_no_anchor_total),
        "candidate_overdraw_factor": float(getattr(args, "candidate_overdraw_factor", 4.0)),
        "isothermal_case_feedback": iso_feedback_summary,
        "state_probe_composition_rule": "fallback_real_database_anchor_full_Y_vector_for_pfr_first_requests",
        "L_range_m": [float(args.L_min_m), float(args.L_max_m)],
        "target_gap_min_bin_count": int(args.target_gap_min_bin_count),
        "target_log10_count": None if getattr(args, "target_log10_count", None) is None else float(args.target_log10_count),
        "effective_target_count_per_T_X_bin": int(target_count),
        "target_ttau_log10_count": None if getattr(args, "target_ttau_log10_count", None) is None else float(args.target_ttau_log10_count),
        "effective_target_count_per_T_logtau_bin": int(ttau_target_count),
        "ttau_logtau_bin_width_decades": float(args.ttau_logtau_bin_width_decades),
        "budget_limited": bool((int(deficit0.sum()) + int(deficit_tau0.sum())) > int(args.n_new_cases)),
        "tx_T_bin_width_K": float(args.tx_T_bin_width_K),
        "tx_X_bin_width": float(args.tx_X_bin_width),
        "under_target_bins_before": int((H0 < target_count).sum()),
        "under_target_bins_after_design": int((H_after < target_count).sum()),
        "empty_bins_before": int((H0 == 0).sum()),
        "empty_bins_after_design": int((H_after == 0).sum()),
        "total_TX_deficit_before": int(deficit0.sum()),
        "total_TX_deficit_after_design": int(deficit_after.sum()),
        "under_target_Ttau_bins_before": int((Htau0 < ttau_target_count).sum()),
        "under_target_Ttau_bins_after_design": int((Htau_after < ttau_target_count).sum()),
        "empty_Ttau_bins_before": int((Htau0 == 0).sum()),
        "empty_Ttau_bins_after_design": int((Htau_after == 0).sum()),
        "total_Ttau_deficit_before": int(deficit_tau0.sum()),
        "total_Ttau_deficit_after_design": int(deficit_tau_after.sum()),
        "pfr_tau_ratio_min": float(args.pfr_tau_ratio_min),
        "pfr_tau_ratio_max": float(args.pfr_tau_ratio_max),
        "reachability_classification": "deferred_to_generate_database_Isothermal_CRACKSIM_PFR_attempt",
        "tau_physicality_flag_counts": out["tau_physicality_flag"].value_counts().to_dict() if (not out.empty and "tau_physicality_flag" in out.columns) else {},
        "tau_over_tau_estimate_quantiles": "not_used_pfr_first_no_arrhenius",
        "round_summaries": round_summaries,
        "region_counts": out["design_region"].value_counts().to_dict() if not out.empty else {},
        "kind_counts": out["design_kind"].value_counts().to_dict() if not out.empty else {},
    }
    return out, report

def plot_outputs(existing: pd.DataFrame, manifest: pd.DataFrame, cmap: ColumnMap, out_dir: Path, args: argparse.Namespace) -> None:
    if plt is None:
        warnings.warn("matplotlib not available; skipping plots")
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    T_existing = pd.to_numeric(existing[cmap.T_col], errors="coerce").to_numpy(float)
    X_existing = compute_conversion_proxy(existing, cmap)
    T_edges, X_edges = bin_edges(args.T_min_K, args.T_hard_max_K, args.tx_T_bin_width_K, args.tx_X_bin_width)
    logtau_edges = tau_bin_edges(args.tau_min_s, args.tau_max_s, args.ttau_logtau_bin_width_decades)
    tau_existing = compute_tau(existing, cmap)
    tau_existing = np.where(np.isfinite(tau_existing) & (tau_existing > 0), tau_existing, np.nan)
    H0 = tx_hist(T_existing, X_existing, T_edges, X_edges)
    Htau0 = ttau_hist(T_existing, tau_existing, T_edges, logtau_edges)

    # Original T-X scatter
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)
    idx = np.isfinite(T_existing) & np.isfinite(X_existing)
    if idx.sum() > 0:
        take = np.random.default_rng(args.seed).choice(np.where(idx)[0], size=min(idx.sum(), 80000), replace=False)
        ax.scatter(X_existing[take], T_existing[take], s=1, alpha=0.25)
    ax.set_xlabel("Ethane conversion proxy [-]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Original sampled state space")
    ax.set_xlim(0, 1)
    ax.set_ylim(args.T_min_K, args.T_hard_max_K)
    fig.tight_layout()
    fig.savefig(fig_dir / "01_original_T_vs_conversion.png", dpi=300)
    plt.close(fig)

    # Original + added
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)
    if idx.sum() > 0:
        ax.scatter(X_existing[take], T_existing[take], s=1, alpha=0.15, label="existing sample")
    if not manifest.empty:
        ax.scatter(manifest["target_conversion"], manifest["T_K"], s=8, alpha=0.75, label="new isothermal conditions")
    ax.set_xlabel("Ethane conversion proxy / target conversion [-]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("State space after designed isothermal additions")
    ax.set_xlim(0, 1)
    ax.set_ylim(args.T_min_K, args.T_hard_max_K)
    ax.legend(loc="best", markerscale=3)
    fig.tight_layout()
    fig.savefig(fig_dir / "02_original_plus_isothermal_design_T_vs_conversion.png", dpi=300)
    plt.close(fig)

    # Heatmap before/after
    H_after = H0.copy()
    if not manifest.empty:
        H_add = tx_hist(manifest["T_K"].to_numpy(float), manifest["target_conversion"].to_numpy(float), T_edges, X_edges)
        H_after += H_add
    for name, H in [("03_coverage_before_heatmap.png", H0), ("04_coverage_after_design_heatmap.png", H_after)]:
        fig = plt.figure(figsize=(7.4, 5.2))
        ax = fig.add_subplot(111)
        im = ax.imshow(np.log10(H.T + 1), origin="lower", aspect="auto", extent=[T_edges[0], T_edges[-1], X_edges[0], X_edges[-1]])
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Ethane conversion proxy [-]")
        ax.set_title(name.replace("_", " ").replace(".png", ""))
        fig.colorbar(im, ax=ax, label="log10(count + 1)")
        fig.tight_layout()
        fig.savefig(fig_dir / name, dpi=300)
        plt.close(fig)

    # Occupancy distribution makes it visible whether the design is still
    # dominated by a few over-represented original bins or has been water-filled
    # toward a more uniform log-count level.
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.hist(np.log10(H0.ravel() + 1), bins=40, alpha=0.55, label="before")
    ax.hist(np.log10(H_after.ravel() + 1), bins=40, alpha=0.55, label="after design")
    if getattr(args, "target_log10_count", None) is not None:
        ax.axvline(float(args.target_log10_count), linestyle="--", linewidth=1.2, label="target log10(count+1)")
    ax.set_xlabel("log10(count + 1) per T-X bin")
    ax.set_ylabel("Number of T-X bins")
    ax.set_title("T-X occupancy distribution before/after design")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "05_TX_occupancy_distribution.png", dpi=300)
    plt.close(fig)

    if not manifest.empty:
        fig = plt.figure(figsize=(7.2, 5.0))
        ax = fig.add_subplot(111)
        ax.scatter(manifest["steam_to_ethane_mass"], manifest["pressure_bar"], c=manifest["T_K"], s=10, alpha=0.8)
        ax.set_xlabel("Steam / ethane mass ratio [-]")
        ax.set_ylabel("Pressure [bar]")
        ax.set_title("New condition coverage in pressure-dilution space")
        fig.tight_layout()
        fig.savefig(fig_dir / "06_new_conditions_pressure_dilution.png", dpi=300)
        plt.close(fig)

        fig = plt.figure(figsize=(7.2, 5.0))
        ax = fig.add_subplot(111)
        ax.scatter(manifest["tau_end_s"], manifest["T_K"], s=10, alpha=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("Residence time / target tau_end [s]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title("New condition coverage in explicit T-log(tau) water-filled space")
        fig.tight_layout()
        fig.savefig(fig_dir / "07_new_conditions_T_vs_tau.png", dpi=300)
        plt.close(fig)

        # T-log(tau) heatmaps before/after. These show whether the gaps in the
        # residence-time coverage have actually been removed.
        Htau_after = Htau0.copy()
        if not manifest.empty:
            Htau_after += ttau_hist(manifest["T_K"].to_numpy(float), manifest["tau_end_s"].to_numpy(float), T_edges, logtau_edges)
        for name, HH in [("10_Ttau_coverage_before_heatmap.png", Htau0), ("11_Ttau_coverage_after_design_heatmap.png", Htau_after)]:
            fig = plt.figure(figsize=(7.4, 5.2))
            ax = fig.add_subplot(111)
            im = ax.imshow(np.log10(HH.T + 1), origin="lower", aspect="auto", extent=[T_edges[0], T_edges[-1], logtau_edges[0], logtau_edges[-1]])
            ax.set_xlabel("Temperature [K]")
            ax.set_ylabel("log10(residence time [s])")
            ax.set_title(name.replace("_", " ").replace(".png", ""))
            fig.colorbar(im, ax=ax, label="log10(count + 1)")
            fig.tight_layout()
            fig.savefig(fig_dir / name, dpi=300)
            plt.close(fig)

        fig = plt.figure(figsize=(7.2, 4.6))
        ax = fig.add_subplot(111)
        ax.hist(np.log10(Htau0.ravel() + 1), bins=40, alpha=0.55, label="before")
        ax.hist(np.log10(Htau_after.ravel() + 1), bins=40, alpha=0.55, label="after design")
        if getattr(args, "target_ttau_log10_count", None) is not None:
            ax.axvline(float(args.target_ttau_log10_count), linestyle="--", linewidth=1.2, label="T-log(tau) target")
        elif getattr(args, "target_log10_count", None) is not None:
            ax.axvline(float(args.target_log10_count), linestyle="--", linewidth=1.2, label="target log10(count+1)")
        ax.set_xlabel("log10(count + 1) per T-log(tau) bin")
        ax.set_ylabel("Number of T-log(tau) bins")
        ax.set_title("T-log(tau) occupancy distribution before/after design")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / "12_Ttau_occupancy_distribution.png", dpi=300)
        plt.close(fig)

        if "target_Re" in manifest.columns and "estimated_length_m" in manifest.columns:
            fig = plt.figure(figsize=(7.2, 5.0))
            ax = fig.add_subplot(111)
            ax.scatter(manifest["tau_end_s"], manifest["estimated_length_m"], c=np.log10(manifest["target_Re"].clip(lower=1)), s=10, alpha=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Designed residence time [s]")
            ax.set_ylabel("Estimated reactor length L = U tau [m]")
            ax.set_title("Hydrodynamic interpretation of designed residence time")
            fig.tight_layout()
            fig.savefig(fig_dir / "08_hydrodynamic_tau_vs_length.png", dpi=300)
            plt.close(fig)

            fig = plt.figure(figsize=(7.2, 5.0))
            ax = fig.add_subplot(111)
            ax.scatter(manifest["estimated_U_m_s"], manifest["estimated_mdot_kg_s"], c=manifest["diameter_m"], s=10, alpha=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Estimated velocity [m/s]")
            ax.set_ylabel("Estimated mass flow [kg/s]")
            ax.set_title("Hydrodynamic spread of new conditions")
            fig.tight_layout()
            fig.savefig(fig_dir / "09_hydrodynamic_U_vs_mdot.png", dpi=300)
            plt.close(fig)



def _iso_round_sort_key(path: Path) -> tuple[int, str]:
    """Sort out_v2_iso_r1, out_v2_iso_r2, ... in numerical round order."""
    text = str(path).replace("\\", "/")
    m = re.search(r"out_v2_iso_r(\d+)", text, flags=re.IGNORECASE)
    return (int(m.group(1)) if m else 10**9, text)


def discover_isothermal_paths(args: argparse.Namespace, base_input_paths: list[Path]) -> list[Path]:
    """Return existing isothermal enrichment parquet paths to include in coverage.

    The project convention is fixed to:

        out_v2_iso_r*/isothermal_enrichment_cracksim.parquet

    The next identification round therefore automatically accounts for all previous
    isothermal rounds, e.g. r1, r2, r3, ... . Explicit --iso paths are also accepted,
    and duplicates are removed.
    """
    candidates: list[Path] = []

    for item in getattr(args, "iso", []) or []:
        if item:
            candidates.append(Path(item))

    if not getattr(args, "no_auto_iso", False):
        pattern = "out_v2_iso_r*/isothermal_enrichment_cracksim.parquet"
        for root in [Path.cwd()]:
            try:
                candidates.extend(sorted(root.glob(pattern), key=_iso_round_sort_key))
            except Exception:
                pass

    seen = {str(p.resolve()).lower() for p in base_input_paths if p.exists()}
    out: list[Path] = []
    for p in candidates:
        try:
            pp = p.resolve()
            key = str(pp).lower()
        except Exception:
            pp = p
            key = str(p).lower()
        if key in seen:
            continue
        if pp.exists() and pp.suffix.lower() == ".parquet":
            seen.add(key)
            out.append(pp)
    return sorted(out, key=_iso_round_sort_key)



# ---------------------------------------------------------------------------
# Existing isothermal enrichment case-level feedback
# ---------------------------------------------------------------------------

ISO_FEEDBACK_COLUMNS = [
    "CaseID", "iso_case_id", "sample_kind", "iso_final_design_kind",
    "iso_pfr_hit_target", "iso_fallback_probe_status",
    "iso_native_truncation_reason", "iso_native_truncated_before_L",
    "iso_T_K", "iso_target_conversion", "iso_manifest_target_tau_s",
    "iso_manifest_tx_bin_i", "iso_manifest_tx_bin_j", "iso_manifest_ttau_bin_k",
]


def read_existing_iso_case_feedback(paths: list[Path]) -> pd.DataFrame:
    """Read compact case-level outcome information from existing isothermal parquets.

    This is intentionally metadata-only: it does not read the 213 Y_* columns or rate
    columns.  The output is used to understand whether a previous target was actually
    filled by a PFR hit / fallback probe, or remained unresolved because no suitable
    anchor was available.  Row-level coverage is still computed from the actual parquet
    states through existing_state_sample(...); this function only adds case-level
    diagnostics and prioritisation for the next design round.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists() or path.suffix.lower() != ".parquet":
            continue
        try:
            schema = pq.read_schema(str(path))
            available = [c for c in ISO_FEEDBACK_COLUMNS if c in schema.names]
            if not available:
                continue
            tbl = pq.read_table(str(path), columns=available)
            df = tbl.to_pandas()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not read isothermal case feedback from {path}: {exc}")
            continue
        if df.empty:
            continue
        df["_iso_feedback_source"] = str(path)
        # Normalise the case id column name.
        if "CaseID" not in df.columns and "iso_case_id" in df.columns:
            df["CaseID"] = df["iso_case_id"]
        if "CaseID" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)

    def _any_eq(x: pd.Series, value: str) -> bool:
        return bool((x.astype(str) == value).any())

    def _contains(x: pd.Series, needle: str) -> bool:
        return bool(x.astype(str).str.contains(needle, case=False, na=False, regex=False).any())

    def _first_num(x: pd.Series) -> float:
        v = pd.to_numeric(x, errors="coerce")
        v = v[np.isfinite(v)]
        return float(v.iloc[0]) if len(v) else np.nan

    rows: list[dict] = []
    for case_id, g in raw.groupby("CaseID", sort=False):
        sample = g.get("sample_kind", pd.Series(dtype=object))
        kinds = g.get("iso_final_design_kind", pd.Series(dtype=object))
        fallback = g.get("iso_fallback_probe_status", pd.Series(dtype=object))
        pfr_hit_col = g.get("iso_pfr_hit_target", pd.Series(dtype=object))
        pfr_hit = bool(pd.Series(pfr_hit_col).fillna(False).astype(bool).any()) if len(pfr_hit_col) else False
        has_probe = _any_eq(sample, "state_probe") if len(sample) else False
        has_traj = _any_eq(sample, "trajectory") if len(sample) else False
        native_failed = _contains(kinds, "native_pfr_failure") or _contains(fallback, "native_pfr_failure")
        no_anchor = _contains(fallback, "no_suitable_anchor_available")
        filled_target = bool(pfr_hit or has_probe)
        unresolved = bool(not filled_target)
        if pfr_hit:
            outcome = "pfr_hit_target"
        elif has_probe and native_failed:
            outcome = "native_pfr_failure_replaced_by_probe"
        elif has_probe:
            outcome = "pfr_missed_target_replaced_by_probe"
        elif native_failed:
            outcome = "native_pfr_failure_no_probe"
        elif no_anchor:
            outcome = "pfr_missed_target_no_anchor"
        else:
            outcome = "pfr_missed_target_unresolved"
        row = {
            "CaseID": case_id,
            "has_trajectory": has_traj,
            "has_state_probe": has_probe,
            "pfr_hit_target": pfr_hit,
            "native_pfr_failed": native_failed,
            "no_suitable_anchor": no_anchor,
            "filled_target": filled_target,
            "unresolved_target": unresolved,
            "case_outcome": outcome,
            "source_files": ";".join(sorted(set(map(str, g.get("_iso_feedback_source", pd.Series(dtype=object)).dropna())))),
        }
        for col in ["iso_T_K", "iso_target_conversion", "iso_manifest_target_tau_s",
                    "iso_manifest_tx_bin_i", "iso_manifest_tx_bin_j", "iso_manifest_ttau_bin_k"]:
            row[col] = _first_num(g[col]) if col in g.columns else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in ["iso_manifest_tx_bin_i", "iso_manifest_tx_bin_j", "iso_manifest_ttau_bin_k"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def iso_case_feedback_maps(
    case_feedback: pd.DataFrame,
    T_edges: np.ndarray,
    X_edges: np.ndarray,
    logtau_edges: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build priority maps from previous isothermal case-level outcomes.

    The actual row-level states already count in H/Htau.  This function adds *case-level*
    knowledge: bins where previous PFR attempts missed and no fallback probe was possible
    are promoted in the next search, but only if they are still underfilled after actual
    row-level coverage is considered.
    """
    boost_tx = np.zeros((len(T_edges) - 1, len(X_edges) - 1), dtype=int)
    boost_ttau = np.zeros((len(T_edges) - 1, len(logtau_edges) - 1), dtype=int)
    summary = {
        "n_iso_feedback_cases": 0,
        "case_outcome_counts": {},
        "n_unresolved_target_cases": 0,
        "n_previous_no_anchor_cases": 0,
        "n_previous_successful_target_cases": 0,
        "unresolved_TX_bins": 0,
        "unresolved_Ttau_bins": 0,
    }
    if case_feedback is None or case_feedback.empty:
        return boost_tx, boost_ttau, summary
    summary["n_iso_feedback_cases"] = int(len(case_feedback))
    if "case_outcome" in case_feedback.columns:
        summary["case_outcome_counts"] = case_feedback["case_outcome"].value_counts(dropna=False).to_dict()
    unresolved = case_feedback[case_feedback.get("unresolved_target", False).astype(bool)].copy()
    summary["n_unresolved_target_cases"] = int(len(unresolved))
    if "no_suitable_anchor" in case_feedback.columns:
        summary["n_previous_no_anchor_cases"] = int(case_feedback["no_suitable_anchor"].astype(bool).sum())
    if "filled_target" in case_feedback.columns:
        summary["n_previous_successful_target_cases"] = int(case_feedback["filled_target"].astype(bool).sum())
    if unresolved.empty or not getattr(args, "boost_previous_unfilled_target_bins", True):
        return boost_tx, boost_ttau, summary
    boost = int(max(0, getattr(args, "previous_unfilled_bin_priority_boost", 8)))
    if boost == 0:
        return boost_tx, boost_ttau, summary

    tx_seen: set[tuple[int, int]] = set()
    ttau_seen: set[tuple[int, int]] = set()
    for _, r in unresolved.iterrows():
        ti = r.get("iso_manifest_tx_bin_i", np.nan)
        xj = r.get("iso_manifest_tx_bin_j", np.nan)
        tk = r.get("iso_manifest_ttau_bin_k", np.nan)
        if not (np.isfinite(ti) and np.isfinite(xj)):
            T = float(r.get("iso_T_K", np.nan))
            X = float(r.get("iso_target_conversion", np.nan))
            if np.isfinite(T) and np.isfinite(X):
                ti = int(np.searchsorted(T_edges, T, side="right") - 1)
                xj = int(np.searchsorted(X_edges, X, side="right") - 1)
        if np.isfinite(ti) and np.isfinite(xj):
            i, j = int(ti), int(xj)
            if 0 <= i < boost_tx.shape[0] and 0 <= j < boost_tx.shape[1]:
                boost_tx[i, j] += boost
                tx_seen.add((i, j))
        if not (np.isfinite(ti) and np.isfinite(tk)):
            T = float(r.get("iso_T_K", np.nan))
            tau = float(r.get("iso_manifest_target_tau_s", np.nan))
            if np.isfinite(T) and np.isfinite(tau) and tau > 0:
                ti = int(np.searchsorted(T_edges, T, side="right") - 1)
                tk = int(np.searchsorted(logtau_edges, math.log10(max(tau, EPS)), side="right") - 1)
        if np.isfinite(ti) and np.isfinite(tk):
            i, k = int(ti), int(tk)
            if 0 <= i < boost_ttau.shape[0] and 0 <= k < boost_ttau.shape[1]:
                boost_ttau[i, k] += boost
                ttau_seen.add((i, k))
    summary["unresolved_TX_bins"] = int(len(tx_seen))
    summary["unresolved_Ttau_bins"] = int(len(ttau_seen))
    return boost_tx, boost_ttau, summary

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Identify empty regions and write isothermal enrichment conditions for CRACKSIM")
    ap.add_argument("--full", required=True, help="Existing full trajectory parquet")
    ap.add_argument("--off", default=None, help="Existing off-manifold parquet")
    ap.add_argument("--iso", action="append", default=[], help="Optional extra generated isothermal enrichment parquet(s) to include in coverage and anchor selection. Can be passed multiple times. In addition, out_v2_iso_r*/isothermal_enrichment_cracksim.parquet is auto-discovered by default.")
    ap.add_argument("--no-auto-iso", action="store_true", help="Disable automatic discovery of out_v2_iso_r*/isothermal_enrichment_cracksim.parquet")
    ap.add_argument("--no-iso-case-feedback", action="store_true", help="Use existing isothermal parquet rows for coverage but ignore case-level outcome feedback when prioritising the next round")
    ap.add_argument("--allow-no-anchor-cases", action="store_true", help="Allow new PFR-first requests even if no valid fallback state-probe anchor is available. Default v4 behavior is to skip such candidates so missed PFRs do not produce unresolved targets.")
    ap.add_argument("--candidate-overdraw-factor", type=float, default=10.0, help="How many more candidate bins to inspect per round to compensate for skipped no-anchor candidates")
    ap.add_argument("--previous-unfilled-bin-priority-boost", type=int, default=8, help="Priority boost for bins where previous isothermal cases missed/failed and had no fallback probe")
    ap.add_argument("--no-boost-previous-unfilled-target-bins", dest="boost_previous_unfilled_target_bins", action="store_false", help="Disable priority boosting of target bins that previous isothermal rounds left unresolved")
    ap.set_defaults(boost_previous_unfilled_target_bins=True)
    ap.add_argument("--previous-manifest", action="append", default=[], help="Optional previous enrichment manifest(s) to include in coverage")
    ap.add_argument("--out", default="out_balanced_iso_r3", help="Output directory")
    ap.add_argument("--max-existing-rows", type=int, default=30000000)
    ap.add_argument("--n-new-cases", type=int, default=20000)
    ap.add_argument("--max-new-cases-per-run", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260706)
    ap.add_argument("--T-min-K", type=float, default=800.0)
    ap.add_argument("--T-hard-max-K", type=float, default=1600.0)
    ap.add_argument("--pressure-min-Pa", type=float, default=150000.0)
    ap.add_argument("--pressure-max-Pa", type=float, default=350000.0)
    ap.add_argument("--steam-ethane-min-mass", type=float, default=0.0)
    ap.add_argument("--steam-ethane-max-mass", type=float, default=1.0)
    ap.add_argument("--tau-min-s", type=float, default=1e-5)
    ap.add_argument("--tau-max-s", type=float, default=1.0)
    ap.add_argument("--coverage-refine-rounds", type=int, default=8)
    ap.add_argument("--target-gap-min-bin-count", type=int, default=4)
    ap.add_argument("--target-log10-count", type=float, default=2.3, help="Preferred target log10(count+1) per T-X bin. Example: 2.3 -> target count about 199. If set, this overrides --target-gap-min-bin-count for design allocation and keeps adding until the max case budget is reached or this target is met.")
    ap.add_argument("--tx-T-bin-width-K", type=float, default=50.0)
    ap.add_argument("--tx-X-bin-width", type=float, default=0.05)
    ap.add_argument("--tau-jitter-decades", type=float, default=0.35, help="Deprecated compatibility option; v5 uses explicit T-log(tau) water-filling by default")
    ap.add_argument("--ttau-logtau-bin-width-decades", type=float, default=0.25, help="Bin width in log10(tau/s) for explicit T-log(tau) water-filling")
    ap.add_argument("--target-ttau-log10-count", type=float, default=2.3, help="Optional separate target log10(count+1) for T-log(tau) bins. Defaults to --target-log10-count.")
    ap.add_argument("--pfr-tau-ratio-min", type=float, default=0.2, help="Deprecated/ignored in PFR-first mode; retained for command compatibility. No Arrhenius/tau-estimate classification is used.")
    ap.add_argument("--pfr-tau-ratio-max", type=float, default=5.0, help="Deprecated/ignored in PFR-first mode; retained for command compatibility. No Arrhenius/tau-estimate classification is used.")
    ap.add_argument("--allow-offtrajectory-pfr", action="store_true", help="Deprecated/ignored in PFR-first mode; every row is a pfr_first_request.")
    ap.add_argument("--state-probe-anchor-rows", type=int, default=8000000, help="Number of real full-species CRACKSIM states sampled from full/off parquets to supply state-probe compositions")
    ap.add_argument("--anchor-steam-max-abs-diff", type=float, default=0.05, help="Hard maximum |target steam/C2H6 - anchor steam/C2H6| for state-probe composition anchors")
    ap.add_argument("--anchor-conversion-max-abs-diff", type=float, default=0.20, help="Hard maximum |target conversion - anchor conversion| for state-probe anchors; use a large value to disable")
    ap.add_argument("--anchor-temperature-max-abs-diff-K", type=float, default=350.0, help="Maximum |target T - anchor T| for state-probe anchors; wide because transported/cooled product states are allowed")
    ap.add_argument("--anchor-conversion-scale", type=float, default=0.05, help="Scale used in weighted anchor nearest-neighbour distance for conversion")
    ap.add_argument("--anchor-steam-scale", type=float, default=0.025, help="Scale used in weighted anchor nearest-neighbour distance for steam/C2H6")
    ap.add_argument("--anchor-pressure-scale-Pa", type=float, default=75000.0, help="Scale used in weighted anchor nearest-neighbour distance for pressure")
    ap.add_argument("--anchor-temperature-scale-K", type=float, default=125.0, help="Scale used in weighted anchor nearest-neighbour distance for temperature")
    ap.add_argument("--anchor-random-top-k", type=int, default=8, help="Randomly choose among the top-k anchor matches to avoid overusing one anchor while keeping matches close")
    ap.add_argument("--diameters-m", nargs="+", type=float, default=[1.0e-05, 0.0306, 0.05, 0.10, 0.2, 0.5, 0.75, 1.0], help="Candidate reactor diameters [m] used to assign physically interpretable tau/Re/L/mdot metadata")
    ap.add_argument("--Re-min", type=float, default=1.0e-3, help="Minimum target Reynolds number for hydrodynamic metadata")
    ap.add_argument("--Re-max", type=float, default=5.0e8, help="Maximum target Reynolds number for hydrodynamic metadata")
    ap.add_argument("--U-min-m-s", type=float, default=1.0e-8, help="Minimum plausible velocity used when translating tau into L/mdot")
    ap.add_argument("--U-max-m-s", type=float, default=1000.0, help="Maximum plausible velocity used when translating tau into L/mdot")
    ap.add_argument("--L-min-m", type=float, default=1.0e-8, help="Minimum plausible reactor length used when translating tau into U/mdot")
    ap.add_argument("--L-max-m", type=float, default=1000.0, help="Maximum plausible reactor length used when translating tau into U/mdot")
    ap.add_argument("--print-added-cases", action="store_true")
    ap.add_argument("--case-log-every", type=int, default=25)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_new_cases_per_run > 20000:
        raise SystemExit("--max-new-cases-per-run cannot exceed 20000")
    if args.n_new_cases > args.max_new_cases_per_run:
        warnings.warn(f"--n-new-cases={args.n_new_cases} exceeds --max-new-cases-per-run={args.max_new_cases_per_run}; capping.")
        args.n_new_cases = args.max_new_cases_per_run
    if args.T_hard_max_K > 1600.0 + 1e-9:
        raise SystemExit("For this enrichment campaign, --T-hard-max-K must be <= 1600 K")
    if not (0.0 <= args.steam_ethane_min_mass <= args.steam_ethane_max_mass <= 1.0):
        raise SystemExit("Steam/ethane mass range must lie within 0--1")
    if not (150000.0 <= args.pressure_min_Pa <= args.pressure_max_Pa <= 350000.0):
        raise SystemExit("Pressure range must lie within 150000--350000 Pa")
    if args.Re_min <= 0 or args.Re_max <= args.Re_min:
        raise SystemExit("Require 0 < --Re-min < --Re-max")
    if args.U_min_m_s <= 0 or args.U_max_m_s <= args.U_min_m_s:
        raise SystemExit("Require 0 < --U-min-m-s < --U-max-m-s")
    if args.L_min_m <= 0 or args.L_max_m <= args.L_min_m:
        raise SystemExit("Require 0 < --L-min-m < --L-max-m")
    if args.candidate_overdraw_factor < 1.0:
        raise SystemExit("Require --candidate-overdraw-factor >= 1")
    # Internal positive form used by make_manifest.  The default is now conservative:
    # every new PFR-first request should have a valid real-Y fallback anchor so a PFR
    # miss/failure can still fill the target via a state probe.
    args.require_fallback_anchor = not bool(args.allow_no_anchor_cases)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_input_paths = [Path(args.full)]
    if args.off:
        base_input_paths.append(Path(args.off))
    iso_paths = discover_isothermal_paths(args, base_input_paths)
    input_paths = base_input_paths + iso_paths
    if iso_paths:
        print("[iso] including existing isothermal enrichment parquet(s) in coverage/anchor selection:")
        for p in iso_paths:
            print(f"      {p}")
    else:
        print("[iso] no existing isothermal enrichment parquet found; using only --full/--off for coverage")
    cmap = inspect_schema(input_paths)
    print(f"[schema] T={cmap.T_col!r}, p={cmap.p_col!r}, tau={cmap.tau_col!r}, species={cmap.species_cols}, n_Y={len(cmap.all_y_cols)}, steam_col={cmap.steam_to_ethane_col!r}")
    existing = existing_state_sample(input_paths, cmap, args.max_existing_rows, args.seed, args.T_hard_max_K)
    anchor_pool = build_anchor_pool(input_paths, cmap, args.state_probe_anchor_rows, args.seed, args.T_hard_max_K)
    print(f"[coverage] sampled {len(existing)} states from {len(input_paths)} input parquet(s), including {len(iso_paths)} isothermal parquet(s)")
    print(f"[anchors] sampled {len(anchor_pool)} real full-species states for state_probe compositions")

    iso_case_feedback = pd.DataFrame()
    if iso_paths and not args.no_iso_case_feedback:
        iso_case_feedback = read_existing_iso_case_feedback(iso_paths)
        if not iso_case_feedback.empty:
            fb_path = out_dir / "existing_isothermal_case_feedback.csv"
            iso_case_feedback.to_csv(fb_path, index=False)
            print(f"[iso-feedback] read {len(iso_case_feedback)} previous isothermal cases -> {fb_path}")
            print(f"[iso-feedback] outcome counts: {iso_case_feedback['case_outcome'].value_counts(dropna=False).to_dict()}")
        else:
            print("[iso-feedback] no case-level metadata found in existing isothermal parquet(s)")
    elif iso_paths:
        print("[iso-feedback] disabled by --no-iso-case-feedback")

    # Add previous manifests to the coverage map if provided, so round-2 design does not
    # keep refilling the same T-X bins. The previous target conversion is converted back
    # to lightweight Y_H2O/Y_C2H6 columns so compute_conversion_proxy works unchanged.
    prev_frames = []
    for pm in args.previous_manifest:
        p = Path(pm)
        if not p.exists():
            warnings.warn(f"Previous manifest does not exist and will be skipped: {p}")
            continue
        m = pd.read_csv(p)
        if "T_K" not in m.columns or "target_conversion" not in m.columns:
            warnings.warn(f"Previous manifest lacks T_K/target_conversion and will be skipped: {p}")
            continue
        s2e = m["steam_to_ethane_mass"].to_numpy(float) if "steam_to_ethane_mass" in m.columns else np.zeros(len(m))
        comp = composition_from_conversion(m["target_conversion"].to_numpy(float), s2e)
        tmp = pd.DataFrame({cmap.T_col: m["T_K"].to_numpy(float)})
        if cmap.p_col and "p_Pa" in m.columns:
            tmp[cmap.p_col] = m["p_Pa"].to_numpy(float)
        if cmap.tau_col and "tau_end_s" in m.columns:
            tmp[cmap.tau_col] = m["tau_end_s"].to_numpy(float)
        for sp, col in cmap.species_cols.items():
            if sp in comp:
                tmp[col] = comp[sp]
        prev_frames.append(tmp)
    if prev_frames:
        n_prev = sum(len(x) for x in prev_frames)
        existing = pd.concat([existing] + prev_frames, ignore_index=True)
        print(f"[coverage] included {n_prev} previously designed manifest rows in coverage accounting")

    manifest, report = make_manifest(existing, cmap, args, anchor_pool=anchor_pool, iso_case_feedback=iso_case_feedback)
    report["input_parquets_used"] = [str(p) for p in input_paths]
    report["isothermal_parquets_used"] = [str(p) for p in iso_paths]
    report["isothermal_case_feedback_used"] = bool(not iso_case_feedback.empty)
    manifest_path = out_dir / "isothermal_enrichment_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (out_dir / "isothermal_enrichment_manifest.json").write_text(json.dumps(manifest.to_dict(orient="records"), indent=2), encoding="utf-8")
    (out_dir / "coverage_refinement_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_outputs(existing, manifest, cmap, out_dir, args)

    if args.print_added_cases:
        every = max(1, int(args.case_log_every))
        with (out_dir / "isothermal_enrichment_case_log.txt").open("w", encoding="utf-8") as fh:
            for _, r in manifest.iterrows():
                line = (f"[ADDED] case={int(r.case_id):05d} region={r.design_region} kind={r.design_kind} "
                        f"T={r.T_K:.1f}K p={r.pressure_bar:.2f}bar steam/C2H6_mass={r.steam_to_ethane_mass:.3f} "
                        f"tau_end={r.tau_end_s:.3e}s logtau_bin={getattr(r, 'ttau_bin_k', -1)} target_X={r.target_conversion:.3f} "
                        f"tau/tau_est={getattr(r, 'tau_over_tau_estimate', float('nan')):.2e} reason={getattr(r, 'tau_physicality_flag', 'n/a')} "
                        f"D={getattr(r, 'diameter_m', float('nan')):.4f}m Re={getattr(r, 'target_Re', float('nan')):.2e} "
                        f"L~={getattr(r, 'estimated_length_m', float('nan')):.3e}m mdot~={getattr(r, 'estimated_mdot_kg_s', float('nan')):.3e}kg/s")
                fh.write(line + "\n")
                if int(r.case_id) % every == 0:
                    print(line, flush=True)
    print(f"[done] wrote {len(manifest)} conditions -> {manifest_path}")
    print(f"[done] empty T-X bins before={report['empty_bins_before']} after_design={report['empty_bins_after_design']}")
    print(f"[done] under-target T-X bins before={report['under_target_bins_before']} after_design={report['under_target_bins_after_design']} target_count={report['effective_target_count_per_T_X_bin']}")
    print(f"[done] under-target T-log(tau) bins before={report['under_target_Ttau_bins_before']} after_design={report['under_target_Ttau_bins_after_design']} target_count={report['effective_target_count_per_T_logtau_bin']}")
    print(f"[done] kind counts={report.get('kind_counts', {})}")
    print(f"[done] tau physicality counts={report.get('tau_physicality_flag_counts', {})}")
    print(f"[done] wrote T-log(tau) diagnostics -> {out_dir / 'figures' / '10_Ttau_coverage_before_heatmap.png'} and 11_Ttau_coverage_after_design_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

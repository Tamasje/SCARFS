"""Generate ISAT-inspired interpolated state probes for SCARFS/CRACKSIM.

Purpose
-------
This script fills remaining sparse regions of the local chemistry state space with
single-point CRACKSIM evaluations at interpolated/mixed states:

    input  : T, p, tau, Y_interpolated
    output : dY/dt, reaction heat absorption, Cantera properties

It does NOT interpolate the rates.  It only interpolates/proposes chemically
plausible input states from existing full/off-manifold/isothermal databases and
then evaluates the kinetic model directly with CRACKSIM.

Designed to run on the Windows machine where SA_CRACKSIM.dll loads.

Typical use after a new identify_isothermal_empty_regions.py run:

    python scripts/generate_database_interpolated_state_probes.py ^
        --manifest out_balanced_iso_r2/isothermal_enrichment_manifest.csv ^
        --full out_v2/full.parquet ^
        --off out_v2/offmanifold_1000000.parquet ^
        --schema-reference out_v2/full.parquet ^
        --out out_v2_interp_r1 ^
        --out-name interpolated_state_probes ^
        --n-probes 5000 --n-cpu 8 --skip-existing

By default this version automatically discovers all existing:

    out_v2_iso_r*/isothermal_enrichment_cracksim.parquet

and uses them as anchor sources. You can still pass extra --iso paths explicitly,
or disable discovery with --no-auto-iso.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from multiprocessing import Process, Queue, set_start_method
from pathlib import Path
from typing import Any, Iterable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scarfs.data import generation_v3 as g2
except Exception:  # noqa: BLE001
    from scarfs.data import generation_v2 as g2  # fallback for older checkouts

SCRIPT_VERSION = "interpolated_state_probes_v2_auto_out_v2_iso_rstar"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_mechanism(base_dir: Path) -> str:
    """Build chem.yaml from chem.inp/transport via ck2yaml if absent."""
    chem_yaml = base_dir / "chem.yaml"
    if not chem_yaml.exists():
        import subprocess

        with (base_dir / "C2KYAML_log.txt").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    "ck2yaml",
                    f"--input={base_dir / 'chem.inp'}",
                    f"--transport={base_dir / 'transport_chemkin.DAT'}",
                    "--permissive",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                text=True,
            )
    return str(chem_yaml.resolve())


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.read_schema(str(path)).names)


def _choose_col(columns: Iterable[str], names: list[str]) -> str | None:
    cols = set(columns)
    for n in names:
        if n in cols:
            return n
    return None


def _read_parquet_sample(path: Path, columns: list[str], max_rows: int, seed: int) -> pd.DataFrame:
    """Approximate uniform row sample from a parquet file without loading all rows when possible."""
    path = Path(path)
    available = _parquet_columns(path)
    use_cols = [c for c in columns if c in available]
    pf = pq.ParquetFile(str(path))
    n_total = pf.metadata.num_rows if pf.metadata is not None else None
    if n_total is None or n_total <= max_rows:
        return pd.read_parquet(path, columns=use_cols)

    rng = np.random.default_rng(seed)
    p = min(1.0, max_rows / max(float(n_total), 1.0))
    pieces: list[pd.DataFrame] = []
    for batch in pf.iter_batches(batch_size=100_000, columns=use_cols):
        df = batch.to_pandas()
        if len(df) == 0:
            continue
        mask = rng.random(len(df)) < p
        if mask.any():
            pieces.append(df.loc[mask])
    if not pieces:
        # Fallback: read the first batch only. This should be rare.
        for batch in pf.iter_batches(batch_size=max_rows, columns=use_cols):
            return batch.to_pandas()
        return pd.DataFrame(columns=use_cols)
    out = pd.concat(pieces, ignore_index=True)
    if len(out) > max_rows:
        out = out.sample(max_rows, random_state=seed).reset_index(drop=True)
    return out


def _normalize_Y(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, dtype=float)
    Y = np.where(np.isfinite(Y), Y, 0.0)
    Y = np.maximum(Y, 0.0)
    s = float(Y.sum())
    if not np.isfinite(s) or s <= 0.0:
        raise ValueError("invalid Y vector: non-positive sum")
    return Y / s


def _safe_log10(x: np.ndarray | float, floor: float = 1e-300):
    return np.log10(np.clip(x, floor, None))


def _source_tag(path: Path) -> str:
    s = str(path).replace("\\", "/")
    if "offmanifold" in s.lower():
        return "offmanifold"
    if "iso" in s.lower() or "isothermal" in s.lower():
        return "isothermal"
    if "full" in s.lower():
        return "full"
    return Path(path).stem


def _iso_round_sort_key(path: Path) -> tuple[int, str]:
    """Sort out_v2_iso_r1, out_v2_iso_r2, ... in numerical round order."""
    text = str(path).replace("\\", "/")
    m = re.search(r"out_v2_iso_r(\d+)", text, flags=re.IGNORECASE)
    return (int(m.group(1)) if m else 10**9, text)


def discover_auto_iso_paths(args: argparse.Namespace) -> list[Path]:
    """Discover existing isothermal enrichment parquets from out_v2_iso_r* folders.

    The project convention is fixed to:

        out_v2_iso_r*/isothermal_enrichment_cracksim.parquet

    Discovery is relative to both the repository root and the current working
    directory. Explicit --iso paths are kept as well. Duplicates are removed.
    """
    candidates: list[Path] = []

    # Explicit user-provided paths first.
    for item in getattr(args, "iso", []) or []:
        if item:
            candidates.append(Path(item))

    # Automatic discovery of all completed isothermal rounds.
    if not getattr(args, "no_auto_iso", False):
        pattern = "out_v2_iso_r*/isothermal_enrichment_cracksim.parquet"
        for root in [Path.cwd(), REPO]:
            try:
                candidates.extend(sorted(root.glob(pattern), key=_iso_round_sort_key))
            except Exception:
                pass

    out: list[Path] = []
    seen: set[str] = set()
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
# Anchor cloud + target requests
# ---------------------------------------------------------------------------

def load_anchor_cloud(
    *,
    sources: list[Path],
    species_names: list[str],
    max_rows_per_file: int,
    seed: int,
) -> pd.DataFrame:
    y_cols = [f"Y_{s}" for s in species_names]
    optional_cols = [
        "T [K]", "P [Pa]", "tau [s]", "CaseID", "sample_kind", "regime",
        "inlet_Y_C2H6 [-]", "steam_to_ethane [kg/kg]",
        "iso_steam_to_ethane_mass", "iso_manifest_target_tau_s", "iso_target_conversion",
        "iso_final_design_kind", "iso_pfr_hit_target", "iso_fallback_probe_status",
    ]
    pieces: list[pd.DataFrame] = []
    for i, p in enumerate(sources):
        if not p or not Path(p).exists():
            continue
        cols = _parquet_columns(Path(p))
        required_present = all(c in cols for c in ["T [K]", "P [Pa]"]) and any(c in cols for c in y_cols)
        if not required_present:
            print(f"[anchors] skip {p}: missing T/P/Y columns")
            continue
        use_cols = [c for c in (y_cols + optional_cols) if c in cols]
        df = _read_parquet_sample(Path(p), use_cols, max_rows_per_file, seed + 1009 * i)
        if df.empty:
            continue
        df["anchor_source_file"] = _source_tag(Path(p))
        pieces.append(df)
        print(f"[anchors] {p}: sampled {len(df):,} rows")
    if not pieces:
        raise RuntimeError("no anchor rows loaded")
    df = pd.concat(pieces, ignore_index=True)

    # Required derived fields.
    if "tau [s]" not in df.columns and "iso_manifest_target_tau_s" in df.columns:
        df["tau [s]"] = df["iso_manifest_target_tau_s"]
    if "tau [s]" not in df.columns:
        df["tau [s]"] = np.nan

    if "steam_to_ethane [kg/kg]" in df.columns:
        steam = pd.to_numeric(df["steam_to_ethane [kg/kg]"], errors="coerce")
    elif "iso_steam_to_ethane_mass" in df.columns:
        steam = pd.to_numeric(df["iso_steam_to_ethane_mass"], errors="coerce")
    else:
        Yh2o = pd.to_numeric(df.get("Y_H2O", np.nan), errors="coerce")
        Yc2h6 = pd.to_numeric(df.get("Y_C2H6", np.nan), errors="coerce")
        steam = Yh2o / np.clip(Yc2h6, 1e-300, None)
    df["anchor_steam_to_ethane_mass"] = steam

    if "inlet_Y_C2H6 [-]" in df.columns:
        inlet = pd.to_numeric(df["inlet_Y_C2H6 [-]"], errors="coerce")
    else:
        inlet = 1.0 / (1.0 + np.clip(steam, 0.0, None))
    df["anchor_inlet_Y_C2H6_ref"] = inlet

    Yc2h6 = pd.to_numeric(df.get("Y_C2H6", np.nan), errors="coerce")
    X = 1.0 - Yc2h6 / np.clip(inlet, 1e-300, None)
    df["anchor_conversion_proxy"] = np.clip(X, -0.25, 1.25)
    df["anchor_log10_tau_s"] = _safe_log10(pd.to_numeric(df["tau [s]"], errors="coerce").to_numpy(float))
    df["anchor_log10_p_Pa"] = _safe_log10(pd.to_numeric(df["P [Pa]"], errors="coerce").to_numpy(float))

    # Drop rows with unusable core values or invalid Y.
    for c in ["T [K]", "P [Pa]", "anchor_conversion_proxy", "anchor_steam_to_ethane_mass"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    y_sum = df[y_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    mask = (
        np.isfinite(df["T [K]"]) & np.isfinite(df["P [Pa]"]) &
        np.isfinite(df["anchor_conversion_proxy"]) &
        np.isfinite(df["anchor_steam_to_ethane_mass"]) &
        (y_sum > 0.1) & (y_sum < 1.1)
    )
    df = df.loc[mask].reset_index(drop=True)
    print(f"[anchors] usable anchor cloud: {len(df):,} rows")
    return df


def load_target_manifest(path: Path, n_probes: int | None, seed: int) -> pd.DataFrame:
    m = pd.read_csv(path)
    cols = m.columns
    c_T = _choose_col(cols, ["T_K", "T_target_K", "target_T_K", "iso_T_K"])
    c_p = _choose_col(cols, ["p_Pa", "P_Pa", "p_target_Pa", "target_p_Pa", "iso_p_Pa"])
    c_tau = _choose_col(cols, ["target_tau_s", "tau_s", "tau_target_s", "iso_manifest_target_tau_s"])
    c_X = _choose_col(cols, ["target_conversion", "X_target", "conversion_target", "iso_target_conversion"])
    c_sd = _choose_col(cols, ["steam_to_ethane_mass", "steam_to_ethane [kg/kg]", "iso_steam_to_ethane_mass"])
    if not all([c_T, c_p, c_tau, c_X]):
        raise RuntimeError(
            "manifest must contain target T, p, tau and conversion columns; "
            f"found T={c_T}, p={c_p}, tau={c_tau}, X={c_X}"
        )
    out = pd.DataFrame({
        "target_T_K": pd.to_numeric(m[c_T], errors="coerce"),
        "target_p_Pa": pd.to_numeric(m[c_p], errors="coerce"),
        "target_tau_s": pd.to_numeric(m[c_tau], errors="coerce"),
        "target_conversion": pd.to_numeric(m[c_X], errors="coerce"),
    })
    if c_sd:
        out["target_steam_to_ethane_mass"] = pd.to_numeric(m[c_sd], errors="coerce")
    else:
        out["target_steam_to_ethane_mass"] = np.nan
    out["target_source_manifest_row"] = np.arange(len(m), dtype=int)
    out["target_source_case_id"] = pd.to_numeric(m.get("case_id", m.get("CaseID", np.arange(len(m)))), errors="coerce")
    for opt in [
        "design_region", "iso_design_region", "tx_bin_i", "tx_bin_j", "ttau_bin_k",
        "coverage_priority", "coverage_round", "design_kind",
    ]:
        if opt in m.columns:
            out[f"target_{opt}"] = m[opt].values
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["target_T_K", "target_p_Pa", "target_tau_s", "target_conversion"])
    out = out[(out["target_tau_s"] > 0) & (out["target_p_Pa"] > 0)]
    if n_probes is not None and len(out) > n_probes:
        out = out.sample(n_probes, random_state=seed).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    print(f"[targets] loaded {len(out):,} target rows from {path}")
    return out


@dataclass
class InterpConfig:
    anchor_steam_max_abs_diff: float = 0.08
    anchor_T_window_K: float = 350.0
    anchor_logtau_window_decades: float = 1.5
    anchor_logp_window_decades: float = 0.50
    min_candidates: int = 20
    random_top_k: int = 24
    allow_relaxed_anchor_search: bool = True
    pair_mode: str = "bracket_conversion"


def _candidate_pool(anchor_df: pd.DataFrame, target: pd.Series, cfg: InterpConfig) -> pd.DataFrame:
    T = float(target["target_T_K"])
    p = float(target["target_p_Pa"])
    tau = float(target["target_tau_s"])
    steam = target.get("target_steam_to_ethane_mass", np.nan)
    logtau = math.log10(max(tau, 1e-300))
    logp = math.log10(max(p, 1e-300))

    df = anchor_df
    mask = (
        (np.abs(df["T [K]"].to_numpy(float) - T) <= cfg.anchor_T_window_K) &
        (np.abs(df["anchor_log10_tau_s"].to_numpy(float) - logtau) <= cfg.anchor_logtau_window_decades) &
        (np.abs(df["anchor_log10_p_Pa"].to_numpy(float) - logp) <= cfg.anchor_logp_window_decades)
    )
    if np.isfinite(steam):
        mask &= np.abs(df["anchor_steam_to_ethane_mass"].to_numpy(float) - float(steam)) <= cfg.anchor_steam_max_abs_diff
    cand = df.loc[mask].copy()
    if len(cand) >= cfg.min_candidates or not cfg.allow_relaxed_anchor_search:
        return cand

    # Relax gradually.  The source state is still a real state; looser matching only affects
    # how far we allow the proposed state to move from the anchor manifold.
    mask = (
        (np.abs(df["T [K]"].to_numpy(float) - T) <= 2.0 * cfg.anchor_T_window_K) &
        (np.abs(df["anchor_log10_tau_s"].to_numpy(float) - logtau) <= 2.0 * cfg.anchor_logtau_window_decades) &
        (np.abs(df["anchor_log10_p_Pa"].to_numpy(float) - logp) <= 2.0 * cfg.anchor_logp_window_decades)
    )
    if np.isfinite(steam):
        mask &= np.abs(df["anchor_steam_to_ethane_mass"].to_numpy(float) - float(steam)) <= max(0.15, 2.0 * cfg.anchor_steam_max_abs_diff)
    return df.loc[mask].copy()


def propose_interpolated_state(
    *,
    target: pd.Series,
    anchors: pd.DataFrame,
    species_names: list[str],
    rng: np.random.Generator,
    cfg: InterpConfig,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    y_cols = [f"Y_{s}" for s in species_names]
    cand = _candidate_pool(anchors, target, cfg)
    if len(cand) < 2:
        return None

    Tt = float(target["target_T_K"])
    pt = float(target["target_p_Pa"])
    taut = float(target["target_tau_s"])
    Xt = float(target["target_conversion"])
    steam_t = target.get("target_steam_to_ethane_mass", np.nan)
    logtaut = math.log10(max(taut, 1e-300))
    logpt = math.log10(max(pt, 1e-300))

    # Dimensionless ranking score for nearest anchors.
    score = (
        np.abs(cand["T [K]"].to_numpy(float) - Tt) / max(cfg.anchor_T_window_K, 1.0) +
        np.abs(cand["anchor_conversion_proxy"].to_numpy(float) - Xt) / 0.10 +
        np.abs(cand["anchor_log10_tau_s"].to_numpy(float) - logtaut) / max(cfg.anchor_logtau_window_decades, 1e-9) +
        np.abs(cand["anchor_log10_p_Pa"].to_numpy(float) - logpt) / max(cfg.anchor_logp_window_decades, 1e-9)
    )
    if np.isfinite(steam_t):
        score += np.abs(cand["anchor_steam_to_ethane_mass"].to_numpy(float) - float(steam_t)) / max(cfg.anchor_steam_max_abs_diff, 1e-9)
    cand = cand.assign(_interp_score=score).sort_values("_interp_score").head(max(2, cfg.random_top_k))

    low = cand[cand["anchor_conversion_proxy"] <= Xt]
    high = cand[cand["anchor_conversion_proxy"] >= Xt]
    mode = "convex_local_interpolation_no_bracket"
    alpha = np.nan
    if len(low) and len(high):
        a = low.sort_values("_interp_score").iloc[0]
        b = high.sort_values("_interp_score").iloc[0]
        Xa = float(a["anchor_conversion_proxy"])
        Xb = float(b["anchor_conversion_proxy"])
        denom = Xb - Xa
        alpha = 0.5 if abs(denom) < 1e-12 else float(np.clip((Xt - Xa) / denom, 0.0, 1.0))
        Ya = a[y_cols].to_numpy(float)
        Yb = b[y_cols].to_numpy(float)
        Y = (1.0 - alpha) * Ya + alpha * Yb
        inlet_ref = (1.0 - alpha) * float(a["anchor_inlet_Y_C2H6_ref"]) + alpha * float(b["anchor_inlet_Y_C2H6_ref"])
        steam_ref = (1.0 - alpha) * float(a["anchor_steam_to_ethane_mass"]) + alpha * float(b["anchor_steam_to_ethane_mass"])
        anchor_ids = [a.get("CaseID", np.nan), b.get("CaseID", np.nan)]
        anchor_sources = [a.get("anchor_source_file", "?"), b.get("anchor_source_file", "?")]
        anchor_X = [Xa, Xb]
        anchor_T = [float(a["T [K]"]), float(b["T [K]"])]
        anchor_p = [float(a["P [Pa]"]), float(b["P [Pa]"])]
        anchor_tau = [float(a.get("tau [s]", np.nan)), float(b.get("tau [s]", np.nan))]
        mode = "convex_bracketed_conversion_interpolation"
    else:
        # Use a small random convex combination of 2-4 nearest states.
        k = int(min(len(cand), rng.integers(2, min(4, len(cand)) + 1)))
        sub = cand.head(max(k, 2)).sample(k, random_state=int(rng.integers(0, 2**31 - 1)))
        w = rng.dirichlet(np.ones(k))
        Ymat = sub[y_cols].to_numpy(float)
        Y = np.sum(w[:, None] * Ymat, axis=0)
        inlet_ref = float(np.sum(w * sub["anchor_inlet_Y_C2H6_ref"].to_numpy(float)))
        steam_ref = float(np.sum(w * sub["anchor_steam_to_ethane_mass"].to_numpy(float)))
        anchor_ids = list(sub.get("CaseID", pd.Series([np.nan] * k)).values)
        anchor_sources = list(sub.get("anchor_source_file", pd.Series(["?"] * k)).values)
        anchor_X = list(sub["anchor_conversion_proxy"].to_numpy(float))
        anchor_T = list(sub["T [K]"].to_numpy(float))
        anchor_p = list(sub["P [Pa]"].to_numpy(float))
        anchor_tau = list(sub.get("tau [s]", pd.Series([np.nan] * k)).values)
        alpha = np.nan

    Y = _normalize_Y(Y)
    X_new = float(1.0 - Y[species_names.index("C2H6")] / max(inlet_ref, 1e-300)) if "C2H6" in species_names else np.nan
    meta = {
        "interp_probe_kind": mode,
        "interp_alpha": alpha,
        "interp_anchor_case_ids": json.dumps([None if pd.isna(x) else int(x) for x in anchor_ids]),
        "interp_anchor_sources": json.dumps([str(x) for x in anchor_sources]),
        "interp_anchor_conversions": json.dumps([float(x) for x in anchor_X]),
        "interp_anchor_T_K": json.dumps([float(x) for x in anchor_T]),
        "interp_anchor_p_Pa": json.dumps([float(x) for x in anchor_p]),
        "interp_anchor_tau_s": json.dumps([None if pd.isna(x) else float(x) for x in anchor_tau]),
        "interp_target_T_K": Tt,
        "interp_target_p_Pa": pt,
        "interp_target_tau_s": taut,
        "interp_target_conversion": Xt,
        "interp_actual_conversion_proxy": X_new,
        "interp_steam_to_ethane_mass_ref": steam_ref,
        "interp_inlet_Y_C2H6_ref": inlet_ref,
        "interp_state_selection_note": "Y is a convex combination of real database states; T/p/tau are target gap coordinates; CRACKSIM labels are newly evaluated, not interpolated.",
    }
    return Y, meta


# ---------------------------------------------------------------------------
# CRACKSIM single-state evaluation
# ---------------------------------------------------------------------------

def evaluate_probe_row(
    *,
    item: dict[str, Any],
    species_names: list[str],
    canonical_columns: list[str],
    keep_d_mix: bool,
) -> pd.DataFrame:
    import cantera as ct

    gas = ct.Solution(g2.REAC_MECH_PATH)
    Y = _normalize_Y(np.asarray(item["Y"], dtype=float))
    T = float(item["T_K"])
    P = float(item["p_Pa"])
    tau = float(item["tau_s"])
    gas.TPY = T, P, Y

    rates_raw = np.asarray(g2.CRACKSIM_rates_DLL(gas), dtype=float)
    wdot = g2.convert_raw_rates_to_kmol_m3_s(rates_raw).reshape(1, -1)
    rho = float(gas.density)
    dYdt = g2.compute_dYdt_from_wdot(wdot, gas.molecular_weights, rho).reshape(-1)
    absorption = float(g2.compute_reaction_energy_terms(
        gas,
        np.array([T], dtype=float),
        np.array([P], dtype=float),
        Y.reshape(1, -1),
        wdot,
    )[0])

    # Reset state because some property calls may depend on the current gas state after the loop.
    gas.TPY = T, P, Y
    row: dict[str, Any] = {
        "T [K]": T,
        "P [Pa]": P,
        "Reaction heat absorption [J/s/m3]": absorption,
        "S Wall imposed [J/s/m3]": 0.0,
        "Heat input [W/m2]": 0.0,
        "z [m]": 0.0,
        "tau [s]": tau,
        "u [m/s]": np.nan,
        "cp_mass [J/kg/K]": float(gas.cp_mass),
        "cv_mass [J/kg/K]": float(gas.cv_mass),
        "rho [kg/m3]": rho,
        "mu [Pa-s]": float(gas.viscosity),
        "k [W/m/K]": float(gas.thermal_conductivity),
        "W_mean [kg/kmol]": float(gas.mean_molecular_weight),
        "PFR point index": 0,
        "PFR points solved": 1,
        "Storage policy": "single_state_interpolated_probe",
        "CaseID": int(item["case_id"]),
        "regime": "interpolated_gap_probe",
        "sample_kind": "state_probe_interpolated",
        "mdot [kg/s]": np.nan,
        "Mass flow [kg/s]": np.nan,
        "diameter [m]": np.nan,
        "Area [m2]": np.nan,
        "steam_to_ethane [kg/kg]": item.get("interp_steam_to_ethane_mass_ref", np.nan),
        "inlet_Y_C2H6 [-]": item.get("interp_inlet_Y_C2H6_ref", np.nan),
        "inlet_Y_H2O [-]": np.nan,
        "T_in [K]": T,
        "P_in [Pa]": P,
        "shape": "interpolated_state_probe",
        "H_peak [W/m2]": 0.0,
        "solver_rtol": np.nan,
        "solver_atol": np.nan,
        "generator_version": SCRIPT_VERSION,
        "Re_in [-]": np.nan,
        "U_in [m/s]": np.nan,
        # Extra provenance.
        "interp_case_id": int(item["case_id"]),
        "interp_generator_version": SCRIPT_VERSION,
        "interp_target_source_manifest_row": item.get("target_source_manifest_row", np.nan),
        "interp_target_source_case_id": item.get("target_source_case_id", np.nan),
    }
    for s, val in zip(species_names, Y):
        row[f"Y_{s}"] = float(val)
    for s, val in zip(species_names, dYdt):
        row[f"dYdt_{s} [1/s]"] = float(val)
    if keep_d_mix:
        try:
            Dmix = np.asarray(gas.mix_diff_coeffs, dtype=float)
            for s, val in zip(species_names, Dmix):
                row[f"D_{s} [m2/s]"] = float(val)
        except Exception:  # noqa: BLE001
            pass
    for k_, v_ in item.get("meta", {}).items():
        row[k_] = v_
    for k_, v_ in item.get("target_extra", {}).items():
        row[k_] = v_

    # Build a fixed schema/order. Missing reference columns become nullable NA.
    data = {c: row.get(c, pd.NA) for c in canonical_columns}
    return pd.DataFrame([data], columns=canonical_columns)


def worker_loop(
    worker_id: int,
    task_q: Queue,
    ready_q: Queue,
    status_q: Queue,
    dll_path: str,
    mech_path: str,
    base_dir: str,
    scratch_root: str,
    species_names: list[str],
    canonical_columns: list[str],
    keep_d_mix: bool,
) -> None:
    # CRACKSIM initialisation is intentionally per process.
    g2.init_worker_cracksim(dll_path, mech_path, Path(base_dir), Path(scratch_root), ready_q)
    scratch = Path(scratch_root)
    try:
        while True:
            item = task_q.get()
            if item is None:
                break
            case_id = int(item["case_id"])
            t0 = time.monotonic()
            try:
                df = evaluate_probe_row(item=item, species_names=species_names,
                                        canonical_columns=canonical_columns,
                                        keep_d_mix=keep_d_mix)
                out_tmp = scratch / f"case_{case_id}.tmp.parquet"
                out_file = scratch / f"case_{case_id}.parquet"
                pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(out_tmp), compression="snappy")
                out_tmp.rename(out_file)
                dt = time.monotonic() - t0
                status_q.put(("done", worker_id, case_id, dt, ""))
                print(f"[w{worker_id:02d}] DONE interpolated probe case={case_id} {dt:.2f}s", flush=True)
            except Exception as exc:  # noqa: BLE001
                dt = time.monotonic() - t0
                status_q.put(("drop", worker_id, case_id, dt, str(exc)))
                warnings.warn(f"[w{worker_id:02d}] DROP interpolated probe case={case_id}: {exc}")
    finally:
        status_q.put(("exit", worker_id, None, None, None))


# ---------------------------------------------------------------------------
# Merge + main driver
# ---------------------------------------------------------------------------

def merge_case_files(out_root: Path, out_name: str, expected_ids: set[int] | None = None) -> Path:
    scratch = out_root / "scratch"
    files = sorted(scratch.glob("case_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    if expected_ids is not None:
        files = [p for p in files if int(p.stem.split("_")[1]) in expected_ids]
    if not files:
        raise RuntimeError(f"no case_*.parquet files found in {scratch}")
    out_file = out_root / f"{out_name}.parquet"
    writer = None
    n_rows = 0
    first_schema = None
    for p in files:
        t = pq.read_table(str(p))
        if writer is None:
            first_schema = t.schema
            writer = pq.ParquetWriter(str(out_file), first_schema, compression="snappy")
        else:
            if t.schema.names != first_schema.names:
                missing = [c for c in first_schema.names if c not in t.schema.names]
                extra = [c for c in t.schema.names if c not in first_schema.names]
                raise RuntimeError(
                    f"Scratch schema mismatch in {p}. Missing vs first: {missing}; extra vs first: {extra}. "
                    "Delete the output scratch folder if it contains files from an older script version."
                )
            t = t.cast(first_schema, safe=False)
        writer.write_table(t)
        n_rows += t.num_rows
    if writer is not None:
        writer.close()
    print(f"[merge] {len(files):,} probe files -> {out_file} ({n_rows:,} rows)")
    return out_file


def build_canonical_columns(schema_reference: Path | None, species_names: list[str], keep_d_mix: bool) -> list[str]:
    ref_cols: list[str] = []
    if schema_reference and Path(schema_reference).exists():
        ref_cols = _parquet_columns(Path(schema_reference))
    base_cols = []
    for s in species_names:
        base_cols.append(f"Y_{s}")
    for s in species_names:
        base_cols.append(f"dYdt_{s} [1/s]")
    base_cols += [
        "T [K]", "P [Pa]", "Reaction heat absorption [J/s/m3]",
        "S Wall imposed [J/s/m3]", "Heat input [W/m2]", "z [m]", "tau [s]", "u [m/s]",
        "cp_mass [J/kg/K]", "cv_mass [J/kg/K]", "rho [kg/m3]", "mu [Pa-s]", "k [W/m/K]",
        "W_mean [kg/kmol]", "PFR point index", "PFR points solved", "Storage policy",
        "CaseID", "regime", "sample_kind", "mdot [kg/s]", "Mass flow [kg/s]",
        "diameter [m]", "Area [m2]", "steam_to_ethane [kg/kg]", "inlet_Y_C2H6 [-]",
        "inlet_Y_H2O [-]", "T_in [K]", "P_in [Pa]", "shape", "H_peak [W/m2]",
        "solver_rtol", "solver_atol", "generator_version", "Re_in [-]", "U_in [m/s]",
    ]
    if keep_d_mix:
        base_cols += [f"D_{s} [m2/s]" for s in species_names]
    extra_cols = [
        "interp_case_id", "interp_generator_version", "interp_probe_kind", "interp_alpha",
        "interp_anchor_case_ids", "interp_anchor_sources", "interp_anchor_conversions",
        "interp_anchor_T_K", "interp_anchor_p_Pa", "interp_anchor_tau_s",
        "interp_target_T_K", "interp_target_p_Pa", "interp_target_tau_s", "interp_target_conversion",
        "interp_actual_conversion_proxy", "interp_steam_to_ethane_mass_ref",
        "interp_inlet_Y_C2H6_ref", "interp_target_source_manifest_row",
        "interp_target_source_case_id", "interp_state_selection_note",
    ]
    cols = []
    for c in ref_cols + base_cols + extra_cols:
        if c not in cols:
            cols.append(c)
    return cols


def make_items(args, species_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iso_paths = discover_auto_iso_paths(args)

    sources = [Path(args.full)]
    if args.off:
        sources.append(Path(args.off))
    sources.extend(iso_paths)

    # Deduplicate full/off/iso paths while preserving order.
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in sources:
        if not p:
            continue
        try:
            pp = Path(p).resolve()
            key = str(pp).lower()
        except Exception:
            pp = Path(p)
            key = str(pp).lower()
        if key in seen or not pp.exists():
            continue
        seen.add(key)
        deduped.append(pp)
    sources = deduped

    if iso_paths:
        print("[auto-iso] including discovered isothermal database(s):")
        for p in iso_paths:
            print(f"  - {p}")
    else:
        print("[auto-iso] no out_v2_iso_r*/isothermal_enrichment_cracksim.parquet databases found")

    anchors = load_anchor_cloud(
        sources=sources,
        species_names=species_names,
        max_rows_per_file=args.max_anchor_rows_per_file,
        seed=args.seed,
    )
    targets = load_target_manifest(Path(args.manifest), args.n_probes, args.seed)

    cfg = InterpConfig(
        anchor_steam_max_abs_diff=args.anchor_steam_max_abs_diff,
        anchor_T_window_K=args.anchor_T_window_K,
        anchor_logtau_window_decades=args.anchor_logtau_window_decades,
        anchor_logp_window_decades=args.anchor_logp_window_decades,
        min_candidates=args.min_candidates,
        random_top_k=args.anchor_random_top_k,
        allow_relaxed_anchor_search=not args.no_relaxed_anchor_search,
    )

    rng = np.random.default_rng(args.seed)
    items: list[dict[str, Any]] = []
    skipped = []
    for i, target in targets.iterrows():
        proposed = propose_interpolated_state(
            target=target,
            anchors=anchors,
            species_names=species_names,
            rng=rng,
            cfg=cfg,
        )
        if proposed is None:
            skipped.append({"target_index": int(i), "reason": "no_suitable_interpolation_anchors"})
            continue
        Y, meta = proposed
        case_id = int(args.case_id_offset + len(items))
        extra = {f"interp_target_{c}": target[c] for c in target.index if c.startswith("target_") and c not in meta}
        items.append({
            "case_id": case_id,
            "Y": Y.tolist(),
            "T_K": float(target["target_T_K"]),
            "p_Pa": float(target["target_p_Pa"]),
            "tau_s": float(target["target_tau_s"]),
            "meta": meta,
            "target_source_manifest_row": target.get("target_source_manifest_row", np.nan),
            "target_source_case_id": target.get("target_source_case_id", np.nan),
            "target_extra": extra,
        })
        if args.print_cases and (len(items) <= 20 or len(items) % args.case_log_every == 0):
            print(
                f"[design] probe {case_id}: T={float(target['target_T_K']):.1f}K "
                f"p={float(target['target_p_Pa'])/1e5:.2f}bar tau={float(target['target_tau_s']):.3e}s "
                f"Xtarget={float(target['target_conversion']):.3f} kind={meta['interp_probe_kind']} "
                f"Xactual={meta['interp_actual_conversion_proxy']:.3f}"
            )
    report = {
        "script_version": SCRIPT_VERSION,
        "n_targets_loaded": int(len(targets)),
        "n_probe_items": int(len(items)),
        "n_skipped": int(len(skipped)),
        "skipped_head": skipped[:50],
        "interp_config": asdict(cfg),
        "sources": [str(p) for p in sources],
        "auto_isothermal_parquets_used": [str(p) for p in iso_paths],
        "auto_iso_pattern": "out_v2_iso_r*/isothermal_enrichment_cracksim.parquet",
    }
    return items, report


def run_generate(args) -> int:
    base_dir = REPO
    dll_path = str((base_dir / "SA_CRACKSIM.dll").resolve())
    if not Path(dll_path).exists():
        print(f"ERROR: {dll_path} not found — place the CRACKSIM DLL in the repo root.", file=sys.stderr)
        return 1
    mech_path = _ensure_mechanism(base_dir)

    # Use Cantera only in the parent to get mechanism species names; worker creates its own gas.
    import cantera as ct
    ct.suppress_thermo_warnings()
    gas = ct.Solution(mech_path)
    species_names = list(gas.species_names)
    print(f"[schema] mechanism species: {len(species_names)}")

    out_root = Path(args.out)
    scratch = out_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    canonical_cols = build_canonical_columns(Path(args.schema_reference) if args.schema_reference else None,
                                             species_names, keep_d_mix=not args.no_d_mix)

    if args.merge_only:
        merge_case_files(out_root, args.out_name, None)
        return 0

    items, report = make_items(args, species_names)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "interpolated_probe_design_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    if not items:
        print("ERROR: no interpolated probe items generated", file=sys.stderr)
        return 1

    expected_ids = {int(x["case_id"]) for x in items}
    if args.skip_existing:
        existing = {int(p.stem.split("_")[1]) for p in scratch.glob("case_*.parquet")}
        before = len(items)
        items = [x for x in items if int(x["case_id"]) not in existing]
        print(f"[resume] {len(existing):,} existing case files, {len(items):,}/{before:,} probes left to evaluate")
    if not items:
        print("[resume] nothing left to evaluate; merging existing files")
        merge_case_files(out_root, args.out_name, expected_ids)
        return 0

    n_workers = max(1, int(args.n_cpu))
    task_q: Queue = Queue()
    status_q: Queue = Queue()
    workers = []
    for i in range(n_workers):
        rq: Queue = Queue(maxsize=1)
        w = Process(
            target=worker_loop,
            args=(
                i, task_q, rq, status_q, dll_path, mech_path, str(base_dir), str(scratch),
                species_names, canonical_cols, not args.no_d_mix,
            ),
            daemon=False,
        )
        w.start()
        msg = rq.get()
        if msg != "READY":
            print(f"ERROR: worker {i} init failed: {msg}", file=sys.stderr)
            for _ in workers:
                task_q.put(None)
            return 1
        print(f"[w{i:02d}] READY", flush=True)
        workers.append(w)

    print(f"[run] interpolated CRACKSIM state probes: {len(items):,} probes, {n_workers} workers")
    t0 = time.time()
    for item in items:
        task_q.put(item)
    for _ in workers:
        task_q.put(None)

    done, dropped, exited = 0, 0, 0
    drops = []
    while exited < n_workers:
        kind, wid, cid, dt, payload = status_q.get()
        if kind == "done":
            done += 1
        elif kind == "drop":
            dropped += 1
            drops.append({"worker": wid, "case_id": cid, "dt_s": dt, "reason": payload})
        elif kind == "exit":
            exited += 1
        if (done + dropped) and (done + dropped) % max(1, args.progress_every) == 0:
            rate = (done + dropped) / max(time.time() - t0, 1e-9)
            print(f"... {done:,} done / {dropped:,} dropped ({rate * 3600:.0f} probes/h)", flush=True)
    for w in workers:
        w.join()

    (out_root / "interpolated_probe_drops.json").write_text(json.dumps(drops, indent=2), encoding="utf-8")
    print(f"[run] finished: {done:,} done / {dropped:,} dropped")
    if dropped and args.fail_on_drops:
        print("ERROR: some probes dropped. Inspect interpolated_probe_drops.json.", file=sys.stderr)
        return 2
    merge_case_files(out_root, args.out_name, expected_ids)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate ISAT-inspired interpolated CRACKSIM state probes")
    ap.add_argument("--manifest", required=True, help="Manifest from identify_isothermal_empty_regions.py containing target T,p,tau,conversion")
    ap.add_argument("--full", required=True, help="Original full.parquet anchor source")
    ap.add_argument("--off", default=None, help="Off-manifold parquet anchor source")
    ap.add_argument("--iso", action="append", default=[], help="Extra isothermal/interpolated parquet anchor source; can be repeated. In addition, out_v2_iso_r*/isothermal_enrichment_cracksim.parquet is auto-discovered by default.")
    ap.add_argument("--no-auto-iso", action="store_true", help="Disable automatic discovery of out_v2_iso_r*/isothermal_enrichment_cracksim.parquet")
    ap.add_argument("--schema-reference", default=None, help="Reference parquet whose columns/order should be preserved")
    ap.add_argument("--out", default="out_v2_interp_r1")
    ap.add_argument("--out-name", default="interpolated_state_probes")
    ap.add_argument("--n-probes", type=int, default=5000)
    ap.add_argument("--n-cpu", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--case-id-offset", type=int, default=3_000_000)
    ap.add_argument("--max-anchor-rows-per-file", type=int, default=500_000)
    ap.add_argument("--anchor-steam-max-abs-diff", type=float, default=0.08)
    ap.add_argument("--anchor-T-window-K", type=float, default=350.0)
    ap.add_argument("--anchor-logtau-window-decades", type=float, default=1.5)
    ap.add_argument("--anchor-logp-window-decades", type=float, default=0.50)
    ap.add_argument("--min-candidates", type=int, default=20)
    ap.add_argument("--anchor-random-top-k", type=int, default=24)
    ap.add_argument("--no-relaxed-anchor-search", action="store_true")
    ap.add_argument("--no-d-mix", action="store_true", help="Do not compute D_<species> mixture diffusion columns")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--fail-on-drops", action="store_true")
    ap.add_argument("--print-cases", action="store_true")
    ap.add_argument("--case-log-every", type=int, default=100)
    ap.add_argument("--progress-every", type=int, default=100)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run_generate(args)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())

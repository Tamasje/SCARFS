#!/usr/bin/env python3
# SCRIPT_VERSION = "visualize_isothermal_enrichment_coverage_v3_multi_iso_rstar"
"""
Visualize and quantify where isothermal enrichment added data relative to the
original SCARFS databases.

The script is intentionally read-only: it never calls Cantera/CRACKSIM and never
modifies the databases. It scans parquet files in batches for exact occupancy
histograms and uses reservoir sampling only for scatter plots.

Typical use from SCARFS repo root:

  python scripts/visualize_isothermal_enrichment_coverage.py --full out_v2/full.parquet --off out_v2/offmanifold_1000000.parquet --out out_iso_coverage_report_multi_iso --T-min-K 800 --T-max-K 1600 --tau-min-s 1e-5 --tau-max-s 1 --tx-T-bin-width-K 50 --tx-X-bin-width 0.05 --ttau-logtau-bin-width-decades 0.25 --sample-rows-per-group 150000

By default the script auto-discovers:
  out_v2_iso_r*/isothermal_enrichment_cracksim.parquet

You can also pass --iso multiple times to include explicit isothermal databases.

Outputs:
- coverage_summary.md
- coverage_metrics.json
- bin_metrics_TX.csv
- bin_metrics_Ttau.csv
- sample_original.csv / sample_isothermal.csv
- figures/*.png
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
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
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires matplotlib. Install with: pip install matplotlib") from exc

SCRIPT_VERSION = "visualize_isothermal_enrichment_coverage_v3_multi_iso_rstar"
EPS = 1.0e-300

TEMP_CANDIDATES = ["T [K]", "T", "T_K", "Temperature [K]", "temperature_K"]
PRESSURE_CANDIDATES = ["P [Pa]", "P", "p", "P_Pa", "p_Pa", "Pressure [Pa]"]
TAU_CANDIDATES = ["tau [s]", "tau", "iso_tau_s", "tau_end_s", "residence_time_s", "time", "t [s]"]
STEAM_CANDIDATES = ["steam_to_ethane [kg/kg]", "steam_to_ethane_mass", "iso_steam_to_ethane_mass", "steam_to_ethane", "steam/C2H6 [kg/kg]"]
Y_C2H6_CANDIDATES = ["Y_C2H6"]
Y_H2O_CANDIDATES = ["Y_H2O"]
INLET_Y_C2H6_CANDIDATES = ["inlet_Y_C2H6 [-]", "inlet_Y_C2H6", "Y_C2H6_in"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    exact = {c: c for c in cols}
    for c in candidates:
        if c in exact:
            return c
    nmap = {_norm(c): c for c in cols}
    for cand in candidates:
        hit = nmap.get(_norm(cand))
        if hit is not None:
            return hit
    return None


class SchemaMap:
    def __init__(self, cols: list[str]) -> None:
        self.cols = cols
        self.T = find_first(cols, TEMP_CANDIDATES)
        self.P = find_first(cols, PRESSURE_CANDIDATES)
        self.tau = find_first(cols, TAU_CANDIDATES)
        self.steam = find_first(cols, STEAM_CANDIDATES)
        self.Y_C2H6 = find_first(cols, Y_C2H6_CANDIDATES)
        self.Y_H2O = find_first(cols, Y_H2O_CANDIDATES)
        self.inlet_Y_C2H6 = find_first(cols, INLET_Y_C2H6_CANDIDATES)
        self.sample_kind = "sample_kind" if "sample_kind" in cols else None
        self.case_id = "CaseID" if "CaseID" in cols else None
        self.iso_kind = "iso_final_design_kind" if "iso_final_design_kind" in cols else None
        self.iso_hit = "iso_pfr_hit_target" if "iso_pfr_hit_target" in cols else None
        self.iso_fallback = "iso_fallback_probe_status" if "iso_fallback_probe_status" in cols else None
        self.iso_native_fail_reason = "iso_native_truncation_reason" if "iso_native_truncation_reason" in cols else None
        self.iso_native_truncated = "iso_native_truncated_before_L" if "iso_native_truncated_before_L" in cols else None
        self.iso_target_T = find_first(cols, ["iso_T_K", "T_target_K", "T_K", "T [K]"])
        self.iso_target_X = find_first(cols, ["iso_target_conversion", "target_conversion", "X_target", "conversion_target"])
        self.iso_target_tau = find_first(cols, ["iso_manifest_target_tau_s", "target_tau_s", "iso_tau_s", "tau [s]"])
        missing = []
        if self.T is None: missing.append("temperature")
        if self.tau is None: missing.append("tau")
        if self.Y_C2H6 is None: missing.append("Y_C2H6")
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available examples: {cols[:20]}")

    def required_columns(self, include_meta: bool = True) -> list[str]:
        cols = [self.T, self.tau, self.Y_C2H6]
        for c in [self.P, self.steam, self.Y_H2O, self.inlet_Y_C2H6]:
            if c: cols.append(c)
        if include_meta:
            for c in [self.sample_kind, self.case_id, self.iso_kind, self.iso_hit, self.iso_fallback, self.iso_native_fail_reason, self.iso_native_truncated, self.iso_target_T, self.iso_target_X, self.iso_target_tau]:
                if c: cols.append(c)
        return list(dict.fromkeys([c for c in cols if c]))


def read_schema(path: Path) -> SchemaMap:
    return SchemaMap(list(pq.read_schema(str(path)).names))


def conversion_proxy(df: pd.DataFrame, smap: SchemaMap, mode: str) -> np.ndarray:
    y_c2h6 = pd.to_numeric(df[smap.Y_C2H6], errors="coerce").fillna(0.0).to_numpy(float)
    if mode == "inlet" and smap.inlet_Y_C2H6 and smap.inlet_Y_C2H6 in df.columns:
        yin = pd.to_numeric(df[smap.inlet_Y_C2H6], errors="coerce").replace(0, np.nan).to_numpy(float)
        X = 1.0 - y_c2h6 / np.clip(yin, EPS, None)
    else:
        if smap.Y_H2O and smap.Y_H2O in df.columns:
            y_h2o = pd.to_numeric(df[smap.Y_H2O], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            y_h2o = np.zeros(len(df), dtype=float)
        dry = np.clip(1.0 - y_h2o, EPS, None)
        X = 1.0 - y_c2h6 / dry
    return np.clip(X, 0.0, 1.0)


def numeric_col(df: pd.DataFrame, col: str | None, default: float = np.nan) -> np.ndarray:
    if col is None or col not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)



def _natural_key(path_or_text) -> list:
    """Natural sort key so r2 sorts before r10."""
    txt = str(path_or_text)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", txt)]


def iso_label_from_path(path: Path) -> str:
    """Return a compact source label such as out_v2_iso_r1 for plotting."""
    path = Path(path)
    if path.parent.name:
        return path.parent.name
    return path.stem


def unique_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate paths while preserving natural sorted order where possible."""
    seen = set()
    out = []
    for p in paths:
        pp = Path(p)
        try:
            key = str(pp.resolve())
        except Exception:
            key = str(pp)
        if key not in seen:
            seen.add(key)
            out.append(pp)
    return out


def reservoir_update(current: pd.DataFrame | None, incoming: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if incoming.empty or n <= 0:
        return current if current is not None else pd.DataFrame()
    if current is None or current.empty:
        if len(incoming) > n:
            incoming = incoming.iloc[rng.choice(len(incoming), size=n, replace=False)].copy()
        return incoming.reset_index(drop=True)
    pooled = pd.concat([current, incoming], ignore_index=True)
    if len(pooled) > n:
        pooled = pooled.iloc[rng.choice(len(pooled), size=n, replace=False)].copy()
    return pooled.reset_index(drop=True)


def scan_group(paths: list[Path], group_name: str, args: argparse.Namespace, T_edges: np.ndarray, X_edges: np.ndarray, logtau_edges: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Scan paths and return metrics, exact histograms, sample dataframe, and per-case metadata."""
    rng = np.random.default_rng(args.seed + (0 if group_name == "original" else 100003))
    H_TX = np.zeros((len(T_edges) - 1, len(X_edges) - 1), dtype=np.int64)
    H_Ttau = np.zeros((len(T_edges) - 1, len(logtau_edges) - 1), dtype=np.int64)
    total_rows = 0
    finite_rows = 0
    sample: pd.DataFrame | None = None
    case_meta_frames: list[pd.DataFrame] = []
    path_stats: list[dict] = []

    for path_i, path in enumerate(paths):
        if not path.exists():
            warnings.warn(f"Skipping missing {group_name} path: {path}")
            continue
        smap = read_schema(path)
        cols = smap.required_columns(include_meta=(group_name == "isothermal"))
        pf = pq.ParquetFile(str(path))
        row_count_meta = pf.metadata.num_rows if pf.metadata is not None else None
        path_rows = 0
        path_finite = 0
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=[c for c in cols if c in pf.schema_arrow.names]):
            df = batch.to_pandas()
            if df.empty:
                continue
            path_rows += len(df)
            total_rows += len(df)
            T = numeric_col(df, smap.T)
            tau = numeric_col(df, smap.tau)
            X = conversion_proxy(df, smap, args.conversion_mode)
            logtau = np.log10(np.clip(tau, args.tau_min_s * 1.0e-6, None))
            mask = (
                np.isfinite(T) & np.isfinite(X) & np.isfinite(tau) &
                (T >= args.T_min_K) & (T <= args.T_max_K) &
                (X >= 0.0) & (X <= 1.0) &
                (tau >= args.tau_min_s) & (tau <= args.tau_max_s)
            )
            path_finite += int(mask.sum())
            finite_rows += int(mask.sum())
            if mask.any():
                htx, _, _ = np.histogram2d(T[mask], X[mask], bins=[T_edges, X_edges])
                httau, _, _ = np.histogram2d(T[mask], logtau[mask], bins=[T_edges, logtau_edges])
                H_TX += htx.astype(np.int64)
                H_Ttau += httau.astype(np.int64)

            # Reservoir scatter sample. Keep only lightweight diagnostic columns.
            if args.sample_rows_per_group > 0:
                take = min(len(df), max(1, args.batch_sample_cap))
                if len(df) > take:
                    df_s = df.iloc[rng.choice(len(df), size=take, replace=False)].copy()
                    T_s = T[df_s.index.to_numpy()]
                    tau_s = tau[df_s.index.to_numpy()]
                    X_s = X[df_s.index.to_numpy()]
                else:
                    df_s = df.copy()
                    T_s = T
                    tau_s = tau
                    X_s = X
                slim = pd.DataFrame({
                    "source_group": group_name,
                    "source_file": str(path),
                    "source_label": iso_label_from_path(path) if group_name == "isothermal" else group_name,
                    "T_K": T_s,
                    "conversion_proxy": X_s,
                    "tau_s": tau_s,
                    "log10_tau_s": np.log10(np.clip(tau_s, EPS, None)),
                    "P_Pa": numeric_col(df_s, smap.P),
                    "steam_to_ethane": numeric_col(df_s, smap.steam),
                })
                if group_name == "isothermal":
                    slim["iso_round"] = iso_label_from_path(path)
                if smap.sample_kind and smap.sample_kind in df_s.columns:
                    slim["sample_kind"] = df_s[smap.sample_kind].astype(str).to_numpy()
                if smap.iso_kind and smap.iso_kind in df_s.columns:
                    slim["iso_final_design_kind"] = df_s[smap.iso_kind].astype(str).to_numpy()
                sample = reservoir_update(sample, slim, args.sample_rows_per_group, rng)

            # Isothermal case-level metadata from batch, dedup later.
            if group_name == "isothermal" and smap.case_id and smap.case_id in df.columns:
                keep_cols = [smap.case_id]
                for c in [smap.sample_kind, smap.iso_kind, smap.iso_hit, smap.iso_fallback, smap.iso_native_fail_reason, smap.iso_native_truncated, smap.iso_target_T, smap.iso_target_X, smap.iso_target_tau]:
                    if c and c in df.columns:
                        keep_cols.append(c)
                cm = df[keep_cols].drop_duplicates().copy()
                cm.rename(columns={smap.case_id: "CaseID"}, inplace=True)
                cm["source_file"] = str(path)
                cm["iso_round"] = iso_label_from_path(path)
                case_meta_frames.append(cm)
        path_stats.append({"path": str(path), "metadata_rows": row_count_meta, "scanned_rows": path_rows, "rows_inside_requested_window": path_finite})
    metrics = {"group": group_name, "paths": path_stats, "scanned_rows": int(total_rows), "rows_inside_requested_window": int(finite_rows)}
    case_meta = pd.concat(case_meta_frames, ignore_index=True).drop_duplicates() if case_meta_frames else pd.DataFrame()
    return metrics, H_TX, H_Ttau, (sample if sample is not None else pd.DataFrame()), case_meta


def heatmap(ax, H: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, title: str, xlabel: str, ylabel: str, log_counts: bool = True):
    data = np.log10(H.T + 1.0) if log_counts else H.T
    im = ax.imshow(data, origin="lower", aspect="auto", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return im


def make_plots(out_dir: Path, sample_orig: pd.DataFrame, sample_iso: pd.DataFrame, case_level: pd.DataFrame, H_orig_TX: np.ndarray, H_iso_TX: np.ndarray, H_orig_Ttau: np.ndarray, H_iso_Ttau: np.ndarray, T_edges: np.ndarray, X_edges: np.ndarray, logtau_edges: np.ndarray, args: argparse.Namespace) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Scatter overlays.
    if not sample_orig.empty or not sample_iso.empty:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        if not sample_orig.empty:
            ax.scatter(sample_orig["T_K"], sample_orig["conversion_proxy"], s=3, alpha=0.18, label="original sample")
        if not sample_iso.empty:
            group_col = "iso_round" if "iso_round" in sample_iso.columns else None
            if group_col:
                for label, g in sample_iso.groupby(group_col, dropna=False):
                    ax.scatter(g["T_K"], g["conversion_proxy"], s=5, alpha=0.35, label=f"{label} sample")
            else:
                ax.scatter(sample_iso["T_K"], sample_iso["conversion_proxy"], s=5, alpha=0.35, label="isothermal enrichment sample")
        ax.set_xlabel("T [K]")
        ax.set_ylabel("C2H6 conversion proxy [-]")
        ax.set_xlim(args.T_min_K, args.T_max_K)
        ax.set_ylim(0, 1)
        ax.set_title("State-space overlay: T vs conversion")
        ax.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(fig_dir / "01_overlay_T_vs_conversion.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        if not sample_orig.empty:
            ax.scatter(sample_orig["T_K"], sample_orig["log10_tau_s"], s=3, alpha=0.18, label="original sample")
        if not sample_iso.empty:
            group_col = "iso_round" if "iso_round" in sample_iso.columns else None
            if group_col:
                for label, g in sample_iso.groupby(group_col, dropna=False):
                    ax.scatter(g["T_K"], g["log10_tau_s"], s=5, alpha=0.35, label=f"{label} sample")
            else:
                ax.scatter(sample_iso["T_K"], sample_iso["log10_tau_s"], s=5, alpha=0.35, label="isothermal enrichment sample")
        ax.set_xlabel("T [K]")
        ax.set_ylabel("log10(tau / s)")
        ax.set_xlim(args.T_min_K, args.T_max_K)
        ax.set_ylim(math.log10(args.tau_min_s), math.log10(args.tau_max_s))
        ax.set_title("State-space overlay: T vs residence time")
        ax.legend(markerscale=3)
        fig.tight_layout()
        fig.savefig(fig_dir / "02_overlay_T_vs_logtau.png", dpi=300)
        plt.close(fig)

    H_comb_TX = H_orig_TX + H_iso_TX
    H_comb_Ttau = H_orig_Ttau + H_iso_Ttau
    gain_TX = (H_iso_TX > 0).astype(int) + ((H_orig_TX == 0) & (H_iso_TX > 0)).astype(int)
    gain_Ttau = (H_iso_Ttau > 0).astype(int) + ((H_orig_Ttau == 0) & (H_iso_Ttau > 0)).astype(int)

    panels = [
        (H_orig_TX, "Original occupancy: T-X", "03_TX_original_occupancy.png", X_edges, T_edges, "Conversion proxy [-]", "T [K]"),
        (H_iso_TX, "Isothermal added occupancy: T-X", "04_TX_isothermal_occupancy.png", X_edges, T_edges, "Conversion proxy [-]", "T [K]"),
        (H_comb_TX, "Combined occupancy after enrichment: T-X", "05_TX_combined_occupancy.png", X_edges, T_edges, "Conversion proxy [-]", "T [K]"),
        (gain_TX, "Coverage gain code: 1=added occupied, 2=new bin", "06_TX_coverage_gain.png", X_edges, T_edges, "Conversion proxy [-]", "T [K]"),
        (H_orig_Ttau, "Original occupancy: T-log(tau)", "07_Ttau_original_occupancy.png", logtau_edges, T_edges, "log10(tau / s)", "T [K]"),
        (H_iso_Ttau, "Isothermal added occupancy: T-log(tau)", "08_Ttau_isothermal_occupancy.png", logtau_edges, T_edges, "log10(tau / s)", "T [K]"),
        (H_comb_Ttau, "Combined occupancy after enrichment: T-log(tau)", "09_Ttau_combined_occupancy.png", logtau_edges, T_edges, "log10(tau / s)", "T [K]"),
        (gain_Ttau, "Coverage gain code: 1=added occupied, 2=new bin", "10_Ttau_coverage_gain.png", logtau_edges, T_edges, "log10(tau / s)", "T [K]"),
    ]
    for H, title, fname, xedges, yedges, xlabel, ylabel in panels:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        im = heatmap(ax, H, xedges, yedges, title, xlabel, ylabel, log_counts=("gain" not in fname))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=300)
        plt.close(fig)

    # Isothermal composition by final design kind.
    if not sample_iso.empty and "iso_final_design_kind" in sample_iso.columns:
        counts = sample_iso["iso_final_design_kind"].value_counts(dropna=False).head(20)
        fig, ax = plt.subplots(figsize=(9, max(4.5, 0.28 * len(counts))))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("sampled rows")
        ax.set_title("Isothermal enrichment sample by final design kind")
        fig.tight_layout()
        fig.savefig(fig_dir / "11_iso_final_design_kind_sample_counts.png", dpi=300)
        plt.close(fig)

    if not sample_iso.empty and "iso_round" in sample_iso.columns:
        counts = sample_iso["iso_round"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(counts))))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("sampled rows")
        ax.set_title("Isothermal enrichment sample rows by database round")
        fig.tight_layout()
        fig.savefig(fig_dir / "11b_iso_sample_rows_by_round.png", dpi=300)
        plt.close(fig)

    # Case-level diagnostics. These prevent trajectory row counts from hiding one-row probes.
    if case_level is not None and not case_level.empty and "case_outcome" in case_level.columns:
        counts = case_level["case_outcome"].value_counts(dropna=False)
        fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(counts))))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel("cases")
        ax.set_title("Case-level enrichment outcome")
        fig.tight_layout()
        fig.savefig(fig_dir / "12_case_level_outcome_counts.png", dpi=300)
        plt.close(fig)

        if "iso_round" in case_level.columns:
            tab = pd.crosstab(case_level["iso_round"], case_level["case_outcome"])
            if not tab.empty:
                fig, ax = plt.subplots(figsize=(10, max(4.0, 0.45 * len(tab))))
                tab.plot(kind="barh", stacked=True, ax=ax)
                ax.set_xlabel("cases")
                ax.set_ylabel("isothermal database")
                ax.set_title("Case-level enrichment outcome by isothermal round")
                ax.legend(fontsize=8, loc="best")
                fig.tight_layout()
                fig.savefig(fig_dir / "12b_case_level_outcome_counts_by_round.png", dpi=300)
                plt.close(fig)

        # One target point per CaseID: T-X map, colored by case outcome via Matplotlib defaults.
        if {"target_T_K", "target_conversion"}.issubset(case_level.columns):
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for outcome, g in case_level.groupby("case_outcome", dropna=False):
                gg = g[np.isfinite(g["target_T_K"]) & np.isfinite(g["target_conversion"])]
                if len(gg):
                    ax.scatter(gg["target_T_K"], gg["target_conversion"], s=14, alpha=0.65, label=str(outcome))
            ax.set_xlabel("Target T [K]")
            ax.set_ylabel("Target C2H6 conversion [-]")
            ax.set_xlim(args.T_min_K, args.T_max_K)
            ax.set_ylim(0, 1)
            ax.set_title("Case-level target map: T vs conversion")
            ax.legend(markerscale=2, fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(fig_dir / "13_case_level_targets_T_vs_conversion.png", dpi=300)
            plt.close(fig)

        # One target point per CaseID: T-logtau map, same outcome categories.
        if {"target_T_K", "target_log10_tau_s"}.issubset(case_level.columns):
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for outcome, g in case_level.groupby("case_outcome", dropna=False):
                gg = g[np.isfinite(g["target_T_K"]) & np.isfinite(g["target_log10_tau_s"])]
                if len(gg):
                    ax.scatter(gg["target_T_K"], gg["target_log10_tau_s"], s=14, alpha=0.65, label=str(outcome))
            ax.set_xlabel("Target T [K]")
            ax.set_ylabel("Target log10(tau / s)")
            ax.set_xlim(args.T_min_K, args.T_max_K)
            ax.set_ylim(math.log10(args.tau_min_s), math.log10(args.tau_max_s))
            ax.set_title("Case-level target map: T vs residence time")
            ax.legend(markerscale=2, fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(fig_dir / "14_case_level_targets_T_vs_logtau.png", dpi=300)
            plt.close(fig)


def bin_metrics_dataframe(H_orig: np.ndarray, H_iso: np.ndarray, T_edges: np.ndarray, other_edges: np.ndarray, other_name: str, target_count: int) -> pd.DataFrame:
    records = []
    for i in range(H_orig.shape[0]):
        for j in range(H_orig.shape[1]):
            o = int(H_orig[i, j])
            a = int(H_iso[i, j])
            c = o + a
            records.append({
                "T_low_K": T_edges[i],
                "T_high_K": T_edges[i + 1],
                f"{other_name}_low": other_edges[j],
                f"{other_name}_high": other_edges[j + 1],
                "original_count": o,
                "isothermal_added_count": a,
                "combined_count": c,
                "was_empty_before": bool(o == 0),
                "is_empty_after": bool(c == 0),
                "newly_occupied_by_isothermal": bool(o == 0 and a > 0),
                "under_target_before": bool(o < target_count),
                "under_target_after": bool(c < target_count),
            })
    return pd.DataFrame.from_records(records)


def _truthy_series(s: pd.Series) -> bool:
    """Robust truthiness for booleans stored as bools, strings, or pandas NA."""
    for x in s:
        if x is True:
            return True
        if isinstance(x, (int, np.integer)) and int(x) == 1:
            return True
        if isinstance(x, str) and x.strip().lower() in {"true", "1", "yes", "y"}:
            return True
    return False


def build_case_level_dataframe(case_meta: pd.DataFrame) -> pd.DataFrame:
    """Collapse row-level isothermal metadata into one row per CaseID.

    This is the key diagnostic for PFR-first enrichment because one missed PFR
    can contain hundreds/thousands of trajectory rows, whereas the fallback
    state probe is often a single row. Row-level counts therefore strongly
    over-represent `isothermal_pfr_missed_target`. This case-level table asks:
    did the case hit, did it receive a probe, or did it remain unfilled?
    """
    if case_meta.empty or "CaseID" not in case_meta.columns:
        return pd.DataFrame()

    cm = case_meta.copy()
    # Normalise optional columns so downstream code is simple.
    for c in [
        "sample_kind", "iso_final_design_kind", "iso_fallback_probe_status",
        "iso_native_truncation_reason", "iso_native_truncated_before_L",
        "iso_pfr_hit_target", "iso_T_K", "iso_target_conversion",
        "iso_manifest_target_tau_s", "iso_round", "source_file",
    ]:
        if c not in cm.columns:
            cm[c] = pd.NA

    group_cols = ["iso_round", "CaseID"] if "iso_round" in cm.columns else ["CaseID"]
    rows = []
    for key, g in cm.groupby(group_cols, dropna=False):
        if isinstance(key, tuple):
            iso_round, case_id = key
        else:
            iso_round, case_id = pd.NA, key
        sample_kinds = {str(x) for x in g["sample_kind"].dropna().unique()}
        kinds = {str(x) for x in g["iso_final_design_kind"].dropna().unique()}
        fallback = {str(x) for x in g["iso_fallback_probe_status"].dropna().unique()}
        reasons = {str(x) for x in g["iso_native_truncation_reason"].dropna().unique() if str(x).strip() and str(x) != "<NA>"}
        has_trajectory = "trajectory" in sample_kinds
        has_state_probe = "state_probe" in sample_kinds
        pfr_hit = _truthy_series(g["iso_pfr_hit_target"])
        native_failed = (
            any("native_pfr_failure" in k for k in kinds)
            or any("native_pfr_failure" in f for f in fallback)
            or _truthy_series(g["iso_native_truncated_before_L"])
            or bool(reasons)
        )
        no_anchor = any("no_suitable_anchor" in f for f in fallback)
        used_probe = has_state_probe or any("used_after" in f for f in fallback)

        if pfr_hit:
            outcome = "PFR hit target"
        elif native_failed and used_probe:
            outcome = "Native PFR failed → state probe"
        elif native_failed and not used_probe:
            outcome = "Native PFR failed → no probe"
        elif has_trajectory and used_probe:
            outcome = "PFR missed → state probe"
        elif has_trajectory and no_anchor:
            outcome = "PFR missed → no suitable anchor"
        elif has_trajectory:
            outcome = "PFR missed → no probe/unknown"
        elif used_probe:
            outcome = "State probe only"
        else:
            outcome = "Other/unknown"

        def first_numeric(col: str) -> float:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            return float(vals.iloc[0]) if len(vals) else np.nan

        rows.append({
            "iso_round": iso_round,
            "source_file": str(g["source_file"].dropna().iloc[0]) if "source_file" in g.columns and len(g["source_file"].dropna()) else "",
            "CaseID": case_id,
            "case_outcome": outcome,
            "has_trajectory": bool(has_trajectory),
            "has_state_probe": bool(has_state_probe),
            "pfr_hit_target": bool(pfr_hit),
            "native_pfr_failed": bool(native_failed),
            "no_suitable_anchor": bool(no_anchor),
            "used_probe": bool(used_probe),
            "sample_kinds": ", ".join(sorted(sample_kinds)),
            "final_design_kinds": ", ".join(sorted(kinds)),
            "fallback_statuses": ", ".join(sorted(fallback)),
            "native_failure_reasons": ", ".join(sorted(reasons)),
            "target_T_K": first_numeric("iso_T_K"),
            "target_conversion": first_numeric("iso_target_conversion"),
            "target_tau_s": first_numeric("iso_manifest_target_tau_s"),
        })
    out = pd.DataFrame(rows)
    if "target_tau_s" in out.columns:
        out["target_log10_tau_s"] = np.log10(np.clip(out["target_tau_s"].astype(float), EPS, None))
    return out


def summarize_case_meta(case_meta: pd.DataFrame) -> dict:
    out: dict = {}
    if case_meta.empty or "CaseID" not in case_meta.columns:
        return out
    case_level = build_case_level_dataframe(case_meta)
    if "iso_round" in case_meta.columns:
        out["n_cases_with_isothermal_rows"] = int(case_meta[["iso_round", "CaseID"]].drop_duplicates().shape[0])
    else:
        out["n_cases_with_isothermal_rows"] = int(case_meta["CaseID"].nunique())
    if not case_level.empty and "case_outcome" in case_level.columns:
        out["case_level_outcome_counts"] = {str(k): int(v) for k, v in case_level["case_outcome"].value_counts(dropna=False).items()}
        out["case_level_probe_used_cases"] = int(case_level["used_probe"].sum())
        out["case_level_unfilled_cases"] = int(((~case_level["pfr_hit_target"]) & (~case_level["used_probe"])).sum())
    if "sample_kind" in case_meta.columns:
        group_cols = ["iso_round", "CaseID"] if "iso_round" in case_meta.columns else ["CaseID"]
        by_case = case_meta.groupby(group_cols)["sample_kind"].apply(lambda s: tuple(sorted(set(map(str, s))))).value_counts()
        out["case_sample_kind_patterns"] = {str(k): int(v) for k, v in by_case.items()}
    if "iso_final_design_kind" in case_meta.columns:
        dedup_cols = (["iso_round", "CaseID", "iso_final_design_kind"] if "iso_round" in case_meta.columns else ["CaseID", "iso_final_design_kind"])
        kinds = case_meta.drop_duplicates(dedup_cols)["iso_final_design_kind"].value_counts(dropna=False)
        out["case_final_design_kind_counts"] = {str(k): int(v) for k, v in kinds.items()}
    if "iso_pfr_hit_target" in case_meta.columns:
        group_cols = ["iso_round", "CaseID"] if "iso_round" in case_meta.columns else ["CaseID"]
        hit_by_case = case_meta.groupby(group_cols)["iso_pfr_hit_target"].apply(lambda s: any(str(x).lower() == "true" or x is True for x in s))
        out["pfr_hit_cases"] = int(hit_by_case.sum())
        out["pfr_hit_fraction"] = float(hit_by_case.mean()) if len(hit_by_case) else None
    if "iso_fallback_probe_status" in case_meta.columns:
        dedup_cols = (["iso_round", "CaseID", "iso_fallback_probe_status"] if "iso_round" in case_meta.columns else ["CaseID", "iso_fallback_probe_status"])
        status = case_meta.drop_duplicates(dedup_cols)["iso_fallback_probe_status"].value_counts(dropna=False)
        out["fallback_status_case_counts"] = {str(k): int(v) for k, v in status.items()}
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Visualize original vs isothermal enrichment coverage")
    ap.add_argument("--full", required=True, help="Original full trajectory parquet")
    ap.add_argument("--off", default=None, help="Original off-manifold parquet")
    ap.add_argument("--iso", action="append", default=[], help="Generated isothermal enrichment parquet. May be repeated. If omitted, auto-discovery is used.")
    ap.add_argument("--iso-glob", default="out_v2_iso_r*/isothermal_enrichment_cracksim.parquet", help="Glob pattern for automatic isothermal database discovery from the repo root")
    ap.add_argument("--no-auto-iso", action="store_true", help="Disable automatic discovery of out_v2_iso_r*/isothermal_enrichment_cracksim.parquet")
    ap.add_argument("--out", default="out_iso_coverage_report")
    ap.add_argument("--T-min-K", type=float, default=800.0)
    ap.add_argument("--T-max-K", type=float, default=1600.0)
    ap.add_argument("--tau-min-s", type=float, default=1.0e-5)
    ap.add_argument("--tau-max-s", type=float, default=1.0)
    ap.add_argument("--tx-T-bin-width-K", type=float, default=50.0)
    ap.add_argument("--tx-X-bin-width", type=float, default=0.05)
    ap.add_argument("--ttau-logtau-bin-width-decades", type=float, default=0.25)
    ap.add_argument("--target-count-per-bin", type=int, default=200, help="Qualitative under-coverage target for bin metrics; 200 corresponds roughly to log10(count+1)=2.3")
    ap.add_argument("--sample-rows-per-group", type=int, default=150000)
    ap.add_argument("--batch-sample-cap", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--conversion-mode", choices=["dry", "inlet"], default="dry", help="dry = 1 - Y_C2H6/(1-Y_H2O), consistent with identify script; inlet = 1 - Y_C2H6/inlet_Y_C2H6 when available")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_paths = [Path(args.full)]
    if args.off:
        original_paths.append(Path(args.off))

    explicit_iso_paths = [Path(p) for p in (args.iso or [])]
    auto_iso_paths: list[Path] = []
    if not args.no_auto_iso:
        auto_iso_paths = sorted(Path(".").glob(args.iso_glob), key=_natural_key)
    iso_paths = unique_paths(explicit_iso_paths + auto_iso_paths)
    if not iso_paths:
        raise SystemExit("No isothermal database found. Pass --iso explicitly or make sure out_v2_iso_r*/isothermal_enrichment_cracksim.parquet exists.")
    print("[iso] using isothermal database(s):")
    for p in iso_paths:
        print(f"  - {p}")

    T_edges = np.arange(args.T_min_K, args.T_max_K + 0.5 * args.tx_T_bin_width_K, args.tx_T_bin_width_K)
    X_edges = np.arange(0.0, 1.0 + 0.5 * args.tx_X_bin_width, args.tx_X_bin_width)
    logtau_edges = np.arange(math.log10(args.tau_min_s), math.log10(args.tau_max_s) + 0.5 * args.ttau_logtau_bin_width_decades, args.ttau_logtau_bin_width_decades)

    print("[scan] original database(s)")
    met_orig, H_orig_TX, H_orig_Ttau, sample_orig, _ = scan_group(original_paths, "original", args, T_edges, X_edges, logtau_edges)
    print("[scan] isothermal enrichment database")
    met_iso, H_iso_TX, H_iso_Ttau, sample_iso, case_meta = scan_group(iso_paths, "isothermal", args, T_edges, X_edges, logtau_edges)

    sample_orig.to_csv(out_dir / "sample_original.csv", index=False)
    sample_iso.to_csv(out_dir / "sample_isothermal.csv", index=False)
    case_level = build_case_level_dataframe(case_meta)
    if not case_meta.empty:
        case_meta.to_csv(out_dir / "isothermal_case_metadata_unique.csv", index=False)
    if not case_level.empty:
        case_level.to_csv(out_dir / "isothermal_case_level_summary.csv", index=False)

    tx_df = bin_metrics_dataframe(H_orig_TX, H_iso_TX, T_edges, X_edges, "X", args.target_count_per_bin)
    ttau_df = bin_metrics_dataframe(H_orig_Ttau, H_iso_Ttau, T_edges, logtau_edges, "log10_tau", args.target_count_per_bin)
    tx_df.to_csv(out_dir / "bin_metrics_TX.csv", index=False)
    ttau_df.to_csv(out_dir / "bin_metrics_Ttau.csv", index=False)

    def occ_stats(Ho: np.ndarray, Ha: np.ndarray) -> dict:
        Hc = Ho + Ha
        return {
            "total_bins": int(Ho.size),
            "occupied_before": int((Ho > 0).sum()),
            "occupied_after": int((Hc > 0).sum()),
            "empty_before": int((Ho == 0).sum()),
            "empty_after": int((Hc == 0).sum()),
            "newly_occupied_by_isothermal": int(((Ho == 0) & (Ha > 0)).sum()),
            "isothermal_bins_occupied": int((Ha > 0).sum()),
            "isothermal_rows_in_previously_empty_bins": int(Ha[Ho == 0].sum()),
            "isothermal_rows_in_previously_occupied_bins": int(Ha[Ho > 0].sum()),
            "under_target_before": int((Ho < args.target_count_per_bin).sum()),
            "under_target_after": int((Hc < args.target_count_per_bin).sum()),
            "isothermal_rows_in_under_target_before_bins": int(Ha[Ho < args.target_count_per_bin].sum()),
        }

    metrics = {
        "script_version": SCRIPT_VERSION,
        "inputs": {"original": [str(p) for p in original_paths], "isothermal": [str(p) for p in iso_paths]},
        "window": {"T_min_K": args.T_min_K, "T_max_K": args.T_max_K, "tau_min_s": args.tau_min_s, "tau_max_s": args.tau_max_s},
        "scan_metrics": {"original": met_orig, "isothermal": met_iso},
        "TX_coverage": occ_stats(H_orig_TX, H_iso_TX),
        "Ttau_coverage": occ_stats(H_orig_Ttau, H_iso_Ttau),
        "isothermal_case_metrics": summarize_case_meta(case_meta),
    }
    (out_dir / "coverage_metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    make_plots(out_dir, sample_orig, sample_iso, case_level, H_orig_TX, H_iso_TX, H_orig_Ttau, H_iso_Ttau, T_edges, X_edges, logtau_edges, args)

    # Human-readable summary.
    lines = []
    lines.append(f"# Isothermal enrichment coverage summary\n")
    lines.append(f"Script version: `{SCRIPT_VERSION}`\n")
    lines.append("## Quantitative coverage\n")
    for name in ["TX_coverage", "Ttau_coverage"]:
        s = metrics[name]
        lines.append(f"### {name}\n")
        lines.append(f"- Occupied bins before: {s['occupied_before']} / {s['total_bins']}\n")
        lines.append(f"- Occupied bins after: {s['occupied_after']} / {s['total_bins']}\n")
        lines.append(f"- Newly occupied by isothermal enrichment: {s['newly_occupied_by_isothermal']} bins\n")
        lines.append(f"- Empty bins before → after: {s['empty_before']} → {s['empty_after']}\n")
        lines.append(f"- Under-target bins before → after: {s['under_target_before']} → {s['under_target_after']} using target_count={args.target_count_per_bin}\n")
        lines.append(f"- Isothermal rows in previously empty bins: {s['isothermal_rows_in_previously_empty_bins']}\n")
        lines.append(f"- Isothermal rows in previously under-target bins: {s['isothermal_rows_in_under_target_before_bins']}\n")
    if metrics["isothermal_case_metrics"]:
        lines.append("## Isothermal case diagnostics\n")
        cm = metrics["isothermal_case_metrics"]
        for k, v in cm.items():
            lines.append(f"- `{k}`: {v}\n")
    lines.append("## Generated figures\n")
    lines.append("See `figures/` for overlay scatter plots, occupancy heatmaps, coverage-gain maps, and case-level outcome plots.\n")
    if not case_level.empty:
        lines.append("\nCase-level data are written to `isothermal_case_level_summary.csv`; this is the preferred table for judging PFR hit/miss/probe outcomes because it counts each CaseID once.\n")
    (out_dir / "coverage_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] wrote coverage report to {out_dir}")
    print(json.dumps({"TX": metrics["TX_coverage"], "Ttau": metrics["Ttau_coverage"], "case_metrics": metrics["isothermal_case_metrics"]}, indent=2, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

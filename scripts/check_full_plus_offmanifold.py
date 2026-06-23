"""
SCARFS database check: full.parquet + offmanifold_1000000.parquet (v2 nearest-neighbour fix)

Purpose
-------
Compare the original on-manifold trajectory database with the generated
off-manifold cloud and quantify whether the off-manifold file actually thickens
coverage around the physical ethane/steam-cracking state manifold.

Default input expected by Mike:
    C:\\Users\\mbonheur\\OneDrive - UGent\\Documenten\\GitHub\\SCARFS\\out_v2\\full.parquet
    C:\\Users\\mbonheur\\OneDrive - UGent\\Documenten\\GitHub\\SCARFS\\out_v2\\offmanifold_1000000.parquet

Outputs are written to:
    C:\\Users\\mbonheur\\OneDrive - UGent\\Documenten\\GitHub\\SCARFS\\out_v2\\coverage_report_full_plus_offmanifold

Required packages:
    pip install pandas pyarrow numpy matplotlib scipy scikit-learn

Notes
-----
- The script reads only the columns needed for coverage diagnostics.
- Large files are sampled for PCA, pairwise-bin and nearest-neighbour metrics,
  but exact range/quality checks are computed from all rows read for the chosen columns.
- v2 fix: nearest-neighbour distances use chemistry-state columns only and ignore tau/wall-source sentinel columns in off-manifold rows.
- Python 3.8+ compatible.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations
import json
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    cKDTree = None

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
    PCA = None


# ============================================================
# User settings
# ============================================================

DATA_DIR = Path(
    r"C:\Users\mbonheur\OneDrive - UGent\Documenten\GitHub\SCARFS\out_v2"
)
FULL_FILE = DATA_DIR / "full.parquet"
OFF_FILE = DATA_DIR / "offmanifold_1000000.parquet"
OUT_DIR = DATA_DIR / "coverage_report_full_plus_offmanifold"

# Sampling limits for expensive metrics/plots. Exact min/p01/p50/p99/max tables use all rows.
FULL_SAMPLE_MAX = 250_000
OFF_SAMPLE_MAX = 250_000
PCA_SAMPLE_MAX = 140_000
PAIR_SAMPLE_MAX = 180_000
NN_FULL_REF_MAX = 80_000
NN_OFF_QUERY_MAX = 80_000
CORNER_SAMPLE_MAX = 50_000

# Pairwise occupancy bins per axis. 30x30 is a good compromise.
BINS_2D = 30
RANDOM_SEED = 20260623

# Selected major species for diagnostics. Keep this list focused to avoid reading 100+ species.
MAJOR_SPECIES = ["C2H6", "C2H4", "C2H2", "C3H6", "C3H8", "H2O"]

# Molecular weights [kg/kmol], numerically equivalent to g/mol.
MW = {
    "C2H6": 30.069,
    "C2H4": 28.054,
    "C2H2": 26.038,
    "C3H6": 42.081,
    "C3H8": 44.097,
    "H2O": 18.01528,
}


# ============================================================
# Column utilities
# ============================================================

def choose_col(names, candidates, required=False, role="column"):
    """Return the first matching column from candidates, with case-insensitive fallback."""
    names = list(names)
    for c in candidates:
        if c in names:
            return c
    lower = {n.lower().strip(): n for n in names}
    for c in candidates:
        key = c.lower().strip()
        if key in lower:
            return lower[key]
    if required:
        raise KeyError(
            "Could not find {}. Tried {}. Available columns include: {} ...".format(
                role, candidates, names[:80]
            )
        )
    return None


def keep_existing(names, cols):
    seen = set()
    out = []
    for c in cols:
        if c is not None and c in names and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def sample_df(df, n, seed=RANDOM_SEED):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def infer_columns(parquet_file):
    pf = pq.ParquetFile(parquet_file)
    names = pf.schema_arrow.names

    col = {
        "CaseID": choose_col(names, ["CaseID", "case_id", "id"], required=False, role="case id"),
        "regime": choose_col(names, ["regime"], required=False, role="regime"),
        "sample_kind": choose_col(names, ["sample_kind", "sample kind", "kind"], required=False, role="sample kind"),
        "T": choose_col(names, ["T [K]", "Temperature [K]", "Temperature", "T"], required=True, role="temperature"),
        "P": choose_col(names, ["P [Pa]", "Pressure [Pa]", "Pressure", "P"], required=True, role="pressure"),
        "heat_abs": choose_col(
            names,
            ["Reaction heat absorption [J/s/m3]", "!Reaction heat absorption [J/s/m3]"],
            required=False,
            role="reaction heat absorption",
        ),
        "wall_source": choose_col(
            names,
            ["S Wall imposed [J/s/m3]", "Energy source term [J/s/m3]", "S_h [J/s/m3]", "S_h"],
            required=False,
            role="wall/source term",
        ),
        "z": choose_col(names, ["z [m]", "Z [m]", "z"], required=False, role="axial coordinate"),
        "tau": choose_col(names, ["tau [s]", "residence_time [s]", "time [s]", "tau"], required=False, role="residence time"),
        "pfr_index": choose_col(names, ["PFR point index", "point_index", "index"], required=False, role="PFR point index"),
        "T_in": choose_col(names, ["T_in [K]", "Tin [K]", "T0 [K]"], required=False, role="inlet temperature"),
        "P_in": choose_col(names, ["P_in [Pa]", "Pin [Pa]", "P0 [Pa]"], required=False, role="inlet pressure"),
        "steam": choose_col(names, ["steam_to_ethane [kg/kg]", "steam_to_ethane", "S/E"], required=False, role="steam-to-ethane"),
        "H_peak": choose_col(names, ["H_peak [W/m2]", "Heat input [W/m2]"], required=False, role="peak heat flux"),
        "diameter": choose_col(names, ["diameter [m]", "D [m]"], required=False, role="diameter"),
        "Re_in": choose_col(names, ["Re_in [-]", "Re_in"], required=False, role="Reynolds"),
        "U_in": choose_col(names, ["U_in [m/s]", "u [m/s]", "U [m/s]"], required=False, role="velocity"),
        "mdot": choose_col(names, ["mdot [kg/s]", "Mass flow [kg/s]"], required=False, role="mass flow"),
        "inlet_C2H6": choose_col(names, ["inlet_Y_C2H6 [-]", "inlet_Y_C2H6"], required=False, role="inlet ethane"),
        "inlet_H2O": choose_col(names, ["inlet_Y_H2O [-]", "inlet_Y_H2O"], required=False, role="inlet steam"),
    }

    species_cols = {}
    for sp in MAJOR_SPECIES:
        species_cols[sp] = choose_col(names, ["Y_{}".format(sp)], required=(sp != "H2O"), role="mass fraction {}".format(sp))

    read_cols = list(col.values()) + list(species_cols.values())
    read_cols = keep_existing(names, read_cols)
    return pf, names, col, species_cols, read_cols


# ============================================================
# Data loading and derived variables
# ============================================================

def add_derived_quantities(df, col, species_cols):
    df = df.copy()

    if col.get("P") in df.columns:
        df["P [bar]"] = safe_numeric(df[col["P"]]) / 1e5
    if col.get("tau") in df.columns:
        df["tau [ms]"] = safe_numeric(df[col["tau"]]) * 1e3
    if col.get("heat_abs") in df.columns:
        df["Heat absorption [MW/m3]"] = safe_numeric(df[col["heat_abs"]]) / 1e6
    if col.get("wall_source") in df.columns:
        df["Wall source [MW/m3]"] = safe_numeric(df[col["wall_source"]]) / 1e6
    if col.get("H_peak") in df.columns:
        df["H_peak [MW/m2]"] = safe_numeric(df[col["H_peak"]]) / 1e6
    if col.get("Re_in") in df.columns:
        df["Re_in [k]"] = safe_numeric(df[col["Re_in"]]) / 1e3
    if col.get("P_in") in df.columns:
        df["P_in [bar]"] = safe_numeric(df[col["P_in"]]) / 1e5

    # Keep canonical plotting aliases.
    if col.get("T") in df.columns and "T [K]" not in df.columns:
        df["T [K]"] = safe_numeric(df[col["T"]])

    # Selected-species sum: this is not necessarily the sum over all mechanism species.
    selected = [c for c in species_cols.values() if c is not None and c in df.columns]
    if selected:
        df["sum_selected_Y [-]"] = df[selected].apply(safe_numeric).sum(axis=1)

    # Ethane conversion and selected molar yields, only if inlet C2H6 is available.
    c2h6 = species_cols.get("C2H6")
    if c2h6 in df.columns and col.get("inlet_C2H6") in df.columns:
        yin = safe_numeric(df[col["inlet_C2H6"]])
        n_c2h6_in = yin / MW["C2H6"]
        n_c2h6 = safe_numeric(df[c2h6]) / MW["C2H6"]
        df["X_C2H6 [-]"] = 1.0 - n_c2h6 / (n_c2h6_in + 1e-300)
        df["X_C2H6 [%]"] = 100.0 * df["X_C2H6 [-]"]

        for sp in ["C2H4", "C2H2", "C3H6", "C3H8"]:
            ycol = species_cols.get(sp)
            if ycol in df.columns:
                df["Yld_{} [mol/mol C2H6_in]".format(sp)] = (
                    safe_numeric(df[ycol]) / MW[sp]
                ) / (n_c2h6_in + 1e-300)

    return df


def read_relevant_parquet(parquet_file, label, sample_max):
    pf, names, col, species_cols, read_cols = infer_columns(parquet_file)
    print("\nReading {}".format(label))
    print("  file: {}".format(parquet_file))
    print("  metadata rows: {:,}".format(pf.metadata.num_rows))
    print("  reading {} / {} columns".format(len(read_cols), len(names)))

    table = pq.read_table(str(parquet_file), columns=read_cols)
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    df["dataset"] = label
    df = add_derived_quantities(df, col, species_cols)

    sample = sample_df(df, sample_max, seed=RANDOM_SEED + (0 if label == "full" else 1))
    return {
        "path": str(parquet_file),
        "metadata_rows": int(pf.metadata.num_rows),
        "columns_available": names,
        "columns_used": {k: v for k, v in col.items() if v is not None},
        "species_cols": {k: v for k, v in species_cols.items() if v is not None},
        "read_cols": read_cols,
        "df": df,
        "sample": sample,
    }


# ============================================================
# Metrics
# ============================================================

def range_table_by_dataset(df, cols):
    rows = []
    cols = [c for c in cols if c in df.columns]
    for dataset, g in df.groupby("dataset"):
        for col in cols:
            x = safe_numeric(g[col]).dropna()
            if len(x) == 0:
                continue
            qs = x.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
            rows.append({
                "dataset": dataset,
                "variable": col,
                "n": int(len(x)),
                "min": float(x.min()),
                "p01": float(qs.loc[0.01]),
                "p05": float(qs.loc[0.05]),
                "p50": float(qs.loc[0.50]),
                "p95": float(qs.loc[0.95]),
                "p99": float(qs.loc[0.99]),
                "max": float(x.max()),
            })
    # Add combined row.
    for col in cols:
        x = safe_numeric(df[col]).dropna()
        if len(x) == 0:
            continue
        qs = x.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
        rows.append({
            "dataset": "combined",
            "variable": col,
            "n": int(len(x)),
            "min": float(x.min()),
            "p01": float(qs.loc[0.01]),
            "p05": float(qs.loc[0.05]),
            "p50": float(qs.loc[0.50]),
            "p95": float(qs.loc[0.95]),
            "p99": float(qs.loc[0.99]),
            "max": float(x.max()),
        })
    return pd.DataFrame(rows)


def quality_checks(df, col_names):
    rows = []
    for dataset, g in df.groupby("dataset"):
        row = {"dataset": dataset, "n_rows": int(len(g))}
        if "CaseID" in g.columns:
            row["n_cases"] = int(g["CaseID"].nunique())
        for col in col_names:
            if col not in g.columns:
                continue
            x = safe_numeric(g[col])
            row["{}_nan_frac".format(col)] = float(x.isna().mean())
            row["{}_neg_frac".format(col)] = float((x < 0).mean())
            row["{}_gt1_frac".format(col)] = float((x > 1).mean())
        for col in ["T [K]", "P [bar]", "Heat absorption [MW/m3]", "sum_selected_Y [-]"]:
            if col in g.columns:
                x = safe_numeric(g[col]).dropna()
                if len(x):
                    row["{}_min".format(col)] = float(x.min())
                    row["{}_p50".format(col)] = float(x.quantile(0.50))
                    row["{}_p99".format(col)] = float(x.quantile(0.99))
                    row["{}_max".format(col)] = float(x.max())
        rows.append(row)
    return pd.DataFrame(rows)


def outside_full_envelope(full_sample, off_sample, cols):
    rows = []
    cols = [c for c in cols if c in full_sample.columns and c in off_sample.columns]
    for col in cols:
        xf = safe_numeric(full_sample[col]).dropna()
        xo = safe_numeric(off_sample[col]).dropna()
        if len(xf) == 0 or len(xo) == 0:
            continue
        fmin, fmax = float(xf.min()), float(xf.max())
        fp01, fp99 = float(xf.quantile(0.01)), float(xf.quantile(0.99))
        rows.append({
            "variable": col,
            "full_min": fmin,
            "full_p01": fp01,
            "full_p99": fp99,
            "full_max": fmax,
            "off_n": int(len(xo)),
            "off_below_full_min_frac": float((xo < fmin).mean()),
            "off_above_full_max_frac": float((xo > fmax).mean()),
            "off_outside_full_minmax_frac": float(((xo < fmin) | (xo > fmax)).mean()),
            "off_outside_full_p01p99_frac": float(((xo < fp01) | (xo > fp99)).mean()),
        })
    return pd.DataFrame(rows).sort_values("off_outside_full_p01p99_frac", ascending=False)


def _signed_log1p_array(x):
    """Compress heavy-tailed signed source terms before distance calculations."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def _prepare_nn_matrix(df, cols, transform_cols=None):
    """Return a numeric matrix for nearest-neighbour calculations.

    The original version used DataFrame.dropna() across all state columns. That is too fragile
    for mixed trajectory/off-manifold files: one non-physical/sentinel column can remove all rows.
    This helper only uses chemistry-state columns and drops columns that are mostly missing.
    """
    transform_cols = set(transform_cols or [])
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        x = safe_numeric(df[col])
        if col in transform_cols:
            x = pd.Series(_signed_log1p_array(x.to_numpy(float)), index=x.index)
        out[col] = x
    return out


def normalise_by_reference(df, cols, ref_lo, ref_hi):
    X = df[cols].to_numpy(float)
    span = ref_hi - ref_lo
    span[span == 0] = 1.0
    Xn = (X - ref_lo) / span
    return Xn


def nearest_off_to_full(full_sample, off_sample, cols):
    """Quantify how far off-manifold states sit from the full trajectory manifold.

    Important: this metric is deliberately based on local chemistry state variables only.
    It excludes tau, z, wall-source and design variables because off-manifold rows are not
    reactor trajectory points: tau is a sentinel in the off-manifold file and wall source is
    not physically meaningful there. Including those columns would either empty the table or
    make all off-manifold points look artificially far away.
    """

    preferred = [
        "T [K]",
        "P [bar]",
        "Y_C2H6",
        "Y_C2H4",
        "Y_C2H2",
        "Y_C3H6",
        "Y_C3H8",
        "Heat absorption [MW/m3]",
    ]
    cols = [c for c in preferred if c in full_sample.columns and c in off_sample.columns]

    full_raw = sample_df(full_sample[cols], NN_FULL_REF_MAX, seed=RANDOM_SEED)
    off_raw = sample_df(off_sample[cols], NN_OFF_QUERY_MAX, seed=RANDOM_SEED + 1)

    transform_cols = {"Heat absorption [MW/m3]"}
    full_num = _prepare_nn_matrix(full_raw, cols, transform_cols=transform_cols)
    off_num = _prepare_nn_matrix(off_raw, cols, transform_cols=transform_cols)

    # Keep only columns with enough valid values in both datasets and finite spread in the full reference.
    usable = []
    for col in cols:
        f = full_num[col]
        o = off_num[col]
        if f.notna().mean() < 0.90 or o.notna().mean() < 0.90:
            continue
        if f.nunique(dropna=True) < 5 or o.nunique(dropna=True) < 5:
            continue
        usable.append(col)

    if len(usable) < 2:
        msg = pd.DataFrame([
            {
                "metric": "nearest_not_computed",
                "reason": "fewer than two usable chemistry-state columns after NaN/spread filtering",
                "candidate_cols": ";".join(cols),
                "usable_cols": ";".join(usable),
            }
        ])
        return msg, pd.DataFrame(columns=["off_to_nearest_full_distance"]), usable

    full = full_num[usable].dropna(axis=0, how="any")
    off = off_num[usable].dropna(axis=0, how="any")

    if len(full) < 10 or len(off) < 10:
        msg = pd.DataFrame([
            {
                "metric": "nearest_not_computed",
                "reason": "too few complete rows after filtering",
                "n_full_complete": int(len(full)),
                "n_off_complete": int(len(off)),
                "candidate_cols": ";".join(cols),
                "usable_cols": ";".join(usable),
            }
        ])
        return msg, pd.DataFrame(columns=["off_to_nearest_full_distance"]), usable

    # Robust full-trajectory envelope scaling. Quantiles are computed on transformed values for heat absorption.
    ref_lo = full.quantile(0.01).to_numpy(float)
    ref_hi = full.quantile(0.99).to_numpy(float)
    good = (ref_hi - ref_lo) > 1e-15
    cols_good = [c for c, g in zip(usable, good) if g]
    ref_lo = ref_lo[good]
    ref_hi = ref_hi[good]

    Xf = normalise_by_reference(full[cols_good], cols_good, ref_lo, ref_hi)
    Xo = normalise_by_reference(off[cols_good], cols_good, ref_lo, ref_hi)
    Xf = np.clip(Xf, -5.0, 6.0)
    Xo = np.clip(Xo, -5.0, 6.0)

    if SCIPY_AVAILABLE:
        tree = cKDTree(Xf)
        off_d, _ = tree.query(Xo, k=1)
        ff_d, _ = tree.query(Xf, k=2)
        full_nn = ff_d[:, 1]
    else:
        def min_dist(A, B, chunk=1000):
            out = []
            for i in range(0, A.shape[0], chunk):
                D = ((A[i:i+chunk, None, :] - B[None, :, :]) ** 2).sum(axis=2) ** 0.5
                out.append(D.min(axis=1))
            return np.concatenate(out)
        off_d = min_dist(Xo, Xf)
        full_ref = Xf[:min(len(Xf), 5000)]
        full_nn = []
        for i in range(0, len(full_ref), 1000):
            D = ((full_ref[i:i+1000, None, :] - Xf[None, :, :]) ** 2).sum(axis=2) ** 0.5
            D.sort(axis=1)
            full_nn.append(D[:, 1])
        full_nn = np.concatenate(full_nn)

    q = [0.05, 0.50, 0.90, 0.95, 0.99]
    full_ref = pd.Series(full_nn)
    off_ref = pd.Series(off_d)
    summary = pd.DataFrame([
        {
            "metric": "full_to_nearest_full_reference",
            "n": int(len(full_ref)),
            "n_columns": int(len(cols_good)),
            "columns_used": ";".join(cols_good),
            "p05": float(full_ref.quantile(q[0])),
            "p50": float(full_ref.quantile(q[1])),
            "p90": float(full_ref.quantile(q[2])),
            "p95": float(full_ref.quantile(q[3])),
            "p99": float(full_ref.quantile(q[4])),
            "max": float(full_ref.max()),
        },
        {
            "metric": "off_to_nearest_full",
            "n": int(len(off_ref)),
            "n_columns": int(len(cols_good)),
            "columns_used": ";".join(cols_good),
            "p05": float(off_ref.quantile(q[0])),
            "p50": float(off_ref.quantile(q[1])),
            "p90": float(off_ref.quantile(q[2])),
            "p95": float(off_ref.quantile(q[3])),
            "p99": float(off_ref.quantile(q[4])),
            "max": float(off_ref.max()),
            "frac_farther_than_full_p95": float((off_ref > full_ref.quantile(0.95)).mean()),
            "frac_farther_than_full_p99": float((off_ref > full_ref.quantile(0.99)).mean()),
        },
    ])

    distances = pd.DataFrame({"off_to_nearest_full_distance": off_d})
    return summary, distances, cols_good


def pairwise_bin_gain(full_sample, off_sample, cols, bins=BINS_2D):
    rows = []
    cols = [c for c in cols if c in full_sample.columns and c in off_sample.columns]
    f = sample_df(full_sample[cols], PAIR_SAMPLE_MAX, seed=RANDOM_SEED).apply(safe_numeric)
    o = sample_df(off_sample[cols], PAIR_SAMPLE_MAX, seed=RANDOM_SEED + 1).apply(safe_numeric)

    for xcol, ycol in combinations(cols, 2):
        xy_f = f[[xcol, ycol]].dropna()
        xy_o = o[[xcol, ycol]].dropna()
        if len(xy_f) < 20 or len(xy_o) < 20:
            continue
        xy_all = pd.concat([xy_f, xy_o], ignore_index=True)

        # Robust bin limits: 0.5-99.5% of combined sample.
        lo = xy_all.quantile(0.005).to_numpy(float)
        hi = xy_all.quantile(0.995).to_numpy(float)
        span = hi - lo
        if np.any(span <= 0):
            continue

        def occ(xy):
            X = xy.to_numpy(float)
            Xn = (X - lo) / span
            Xn = np.clip(Xn, 0.0, 1.0)
            H, _, _ = np.histogram2d(Xn[:, 0], Xn[:, 1], bins=bins, range=[[0, 1], [0, 1]])
            return H > 0

        of = occ(xy_f)
        oo = occ(xy_o)
        oc = of | oo
        total = bins * bins
        new_bins = int((oc & ~of).sum())
        rows.append({
            "x": xcol,
            "y": ycol,
            "bins_per_axis": bins,
            "full_occupied_bins": int(of.sum()),
            "off_occupied_bins": int(oo.sum()),
            "combined_occupied_bins": int(oc.sum()),
            "new_bins_added_by_off": new_bins,
            "full_occupied_fraction": float(of.sum() / total),
            "off_occupied_fraction": float(oo.sum() / total),
            "combined_occupied_fraction": float(oc.sum() / total),
            "absolute_coverage_gain": float(new_bins / total),
            "relative_gain_vs_full": float(new_bins / max(int(of.sum()), 1)),
            "n_full": int(len(xy_f)),
            "n_off": int(len(xy_o)),
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("absolute_coverage_gain", ascending=False)


# ============================================================
# Plotting
# ============================================================

def plot_counts(full_info, off_info, out_dir):
    rows = [
        {"dataset": "full", "metadata_rows": full_info["metadata_rows"], "sample_rows_used": len(full_info["sample"])},
        {"dataset": "offmanifold", "metadata_rows": off_info["metadata_rows"], "sample_rows_used": len(off_info["sample"])},
    ]
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tab["dataset"], tab["metadata_rows"])
    ax.set_ylabel("Rows")
    ax.set_title("Rows in full and off-manifold parquet files")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(tab["metadata_rows"]):
        ax.text(i, v, "{:,.0f}".format(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "01_row_counts.png", dpi=250)
    plt.close(fig)


def plot_pca(combined_sample, cols, out_dir):
    cols = [c for c in cols if c in combined_sample.columns]
    work = sample_df(combined_sample[["dataset"] + cols], PCA_SAMPLE_MAX, seed=RANDOM_SEED)
    X = work[cols].apply(safe_numeric)
    good = X.notna().all(axis=1)
    X = X.loc[good]
    meta = work.loc[good]
    if len(X) < 10 or len(cols) < 2:
        return None, None, None

    Xv = X.to_numpy(float)
    mu = Xv.mean(axis=0)
    sig = Xv.std(axis=0)
    sig[sig == 0] = 1.0
    Xs = (Xv - mu) / sig

    if SKLEARN_AVAILABLE:
        pca = PCA(n_components=2, random_state=RANDOM_SEED)
        Z = pca.fit_transform(Xs)
        evr = pca.explained_variance_ratio_
    else:
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        Z = U[:, :2] * S[:2]
        evr = (S[:2] ** 2) / np.sum(S ** 2)

    p = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "dataset": meta["dataset"].values})
    p.to_csv(out_dir / "pca_projection_sample.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, g in p.groupby("dataset"):
        ax.scatter(g["PC1"], g["PC2"], s=4, alpha=0.25, label=label, rasterized=True)
    ax.set_xlabel("PC1 ({:.1f}% variance)".format(100.0 * evr[0]))
    ax.set_ylabel("PC2 ({:.1f}% variance)".format(100.0 * evr[1]))
    ax.set_title("PCA of full + off-manifold state space")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "02_pca_full_vs_offmanifold.png", dpi=250)
    plt.close(fig)

    return p, evr, cols


def plot_nearest_distance(distances, summary, out_dir):
    if distances is None or distances.empty:
        return
    d = distances["off_to_nearest_full_distance"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(d, bins=80, alpha=0.85)
    ax.set_xlabel("Normalised distance from off-manifold point to nearest full trajectory point")
    ax.set_ylabel("Count")
    ax.set_title("How far the off-manifold cloud sits from the full trajectory manifold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "03_off_to_full_nearest_distance.png", dpi=250)
    plt.close(fig)


def plot_outside_envelope(outside, out_dir):
    if outside is None or outside.empty:
        return
    work = outside.head(12).copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(work["variable"], work["off_outside_full_p01p99_frac"])
    ax.set_xlabel("Fraction of off-manifold rows outside full p01-p99 envelope")
    ax.set_title("Off-manifold spread beyond the central full-trajectory envelope")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "04_off_outside_full_envelope.png", dpi=250)
    plt.close(fig)


def plot_pairwise_gain(pair_gain, out_dir):
    if pair_gain is None or pair_gain.empty:
        return
    work = pair_gain.head(15).copy()
    labels = ["{} vs {}".format(a, b) for a, b in zip(work["x"], work["y"])]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, work["absolute_coverage_gain"])
    ax.invert_yaxis()
    ax.set_xlabel("Absolute 2D bin-coverage gain from off-manifold points")
    ax.set_title("Which projections are most improved by the off-manifold cloud?")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "05_pairwise_coverage_gain.png", dpi=250)
    plt.close(fig)


def density_panel(full_sample, off_sample, xcol, ycol, out_path):
    if xcol not in full_sample.columns or ycol not in full_sample.columns:
        return
    if xcol not in off_sample.columns or ycol not in off_sample.columns:
        return
    f = sample_df(full_sample[[xcol, ycol]], 100_000, seed=RANDOM_SEED).apply(safe_numeric).dropna()
    o = sample_df(off_sample[[xcol, ycol]], 100_000, seed=RANDOM_SEED + 1).apply(safe_numeric).dropna()
    if len(f) < 20 or len(o) < 20:
        return
    combined = pd.concat([f, o], ignore_index=True)
    xlim = combined[xcol].quantile([0.005, 0.995]).to_numpy(float)
    ylim = combined[ycol].quantile([0.005, 0.995]).to_numpy(float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, data, title in [
        (axes[0], f, "full"),
        (axes[1], o, "off-manifold"),
        (axes[2], combined, "combined"),
    ]:
        ax.hexbin(data[xcol], data[ycol], gridsize=60, mincnt=1)
        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.2)
        ax.set_xlabel(xcol)
    axes[0].set_ylabel(ycol)
    fig.suptitle("{} versus {}".format(ycol, xcol), y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def corr_heatmap(combined_sample, cols, out_dir):
    cols = [c for c in cols if c in combined_sample.columns]
    if len(cols) < 2:
        return
    work = sample_df(combined_sample[cols], CORNER_SAMPLE_MAX, seed=RANDOM_SEED).apply(safe_numeric).dropna()
    if len(work) < 10:
        return
    corr = work.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.7 * len(cols) + 4, 0.7 * len(cols) + 3))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    ax.set_title("Combined state-space correlation")
    if len(cols) <= 10:
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, "{:.2f}".format(corr.values[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "09_combined_state_correlation_heatmap.png", dpi=250)
    plt.close(fig)


# ============================================================
# Report
# ============================================================

def write_report(out_dir, metadata, state_cols, quality, ranges, outside, pair_gain, nn_summary, pca_evr):
    lines = []
    lines.append("# SCARFS full + off-manifold coverage check\n")
    lines.append("## Files\n")
    lines.append("- full: `{}`\n".format(metadata["full"]["path"]))
    lines.append("- offmanifold: `{}`\n".format(metadata["offmanifold"]["path"]))
    lines.append("\n## Row counts\n")
    lines.append("- full rows from parquet metadata: {:,}\n".format(metadata["full"]["metadata_rows"]))
    lines.append("- off-manifold rows from parquet metadata: {:,}\n".format(metadata["offmanifold"]["metadata_rows"]))
    lines.append("- full sample rows used: {:,}\n".format(metadata["full"]["sample_rows_used"]))
    lines.append("- off-manifold sample rows used: {:,}\n".format(metadata["offmanifold"]["sample_rows_used"]))

    lines.append("\n## State variables used\n")
    for c in state_cols:
        lines.append("- {}\n".format(c))

    if pca_evr is not None:
        lines.append("\n## PCA\n")
        lines.append("- PC1 explained variance: {:.2f}%\n".format(100.0 * pca_evr[0]))
        lines.append("- PC2 explained variance: {:.2f}%\n".format(100.0 * pca_evr[1]))
        lines.append("- PC1 + PC2 explained variance: {:.2f}%\n".format(100.0 * np.sum(pca_evr[:2])))

    lines.append("\n## Nearest-distance interpretation\n")
    if nn_summary is not None and not nn_summary.empty:
        lines.append(nn_summary.to_markdown(index=False))
        lines.append("\n\nInterpretation: `off_to_nearest_full` measures how far perturbed points sit from the original trajectory manifold after robust normalisation by the full trajectory envelope. If the off-manifold median is close to the full-to-full nearest-neighbour scale, the cloud is mainly local thickening. If the p95/p99 values are much larger, the cloud also explores wider drift regions.\n")
    else:
        lines.append("Nearest-distance metric could not be computed.\n")

    lines.append("\n## Off-manifold outside the full envelope\n")
    if outside is not None and not outside.empty:
        top = outside.head(12)
        lines.append(top.to_markdown(index=False))
        lines.append("\n\nInterpretation: values outside the full min-max envelope are genuinely outside the original trajectory range for that variable. Values outside p01-p99 but inside min-max usually indicate useful enrichment of tails rather than unphysical extrapolation.\n")

    lines.append("\n## Pairwise bin-coverage gain\n")
    if pair_gain is not None and not pair_gain.empty:
        lines.append(pair_gain.head(15).to_markdown(index=False))
        lines.append("\n\nInterpretation: high absolute gain means that the off-manifold cloud adds new occupied 2D regions that were absent in `full.parquet`. Focus especially on projections involving heat absorption, acetylene, conversion and temperature.\n")

    lines.append("\n## Automatic conclusion\n")
    conclusion = []
    if nn_summary is not None and not nn_summary.empty and "frac_farther_than_full_p95" in nn_summary.columns:
        row = nn_summary[nn_summary["metric"] == "off_to_nearest_full"]
        if len(row):
            frac_far = float(row["frac_farther_than_full_p95"].iloc[0])
            p50 = float(row["p50"].iloc[0])
            p95 = float(row["p95"].iloc[0])
            conclusion.append("The off-manifold file is not just a duplicate of the full trajectory file: its median nearest distance to the full manifold is {:.4g}, with p95 {:.4g}.".format(p50, p95))
            if frac_far < 0.30:
                conclusion.append("Most off-manifold points remain relatively close to the trajectory manifold, which is good for local surrogate robustness.")
            elif frac_far < 0.70:
                conclusion.append("The off-manifold cloud gives a mixed near- and wider-perturbation dataset; this is probably useful, but wide points should not dominate the training weights.")
            else:
                conclusion.append("A large fraction of off-manifold points are farther from the full manifold than the full manifold's own p95 spacing; use these points carefully and consider downweighting wide perturbations.")
    if pair_gain is not None and not pair_gain.empty:
        mean_gain = float(pair_gain["absolute_coverage_gain"].mean())
        max_gain = float(pair_gain["absolute_coverage_gain"].max())
        conclusion.append("The mean pairwise bin-coverage gain from off-manifold data is {:.1f}% of all 2D bins, with a maximum gain of {:.1f}%.".format(100.0 * mean_gain, 100.0 * max_gain))
    if outside is not None and not outside.empty:
        max_out = float(outside["off_outside_full_minmax_frac"].max())
        conclusion.append("The largest fraction outside the full min-max envelope for any checked variable is {:.1f}%.".format(100.0 * max_out))
    if not conclusion:
        conclusion.append("The script produced tables and plots, but not enough summary metrics for an automatic judgement.")
    lines.append("\n".join("- " + c for c in conclusion))

    lines.append("\n\n## Output files to inspect first\n")
    for name in [
        "coverage_report.md",
        "range_comparison_by_dataset.csv",
        "off_outside_full_envelope.csv",
        "pairwise_2d_bin_gain.csv",
        "nearest_off_to_full_summary.csv",
        "02_pca_full_vs_offmanifold.png",
        "03_off_to_full_nearest_distance.png",
        "04_off_outside_full_envelope.png",
        "05_pairwise_coverage_gain.png",
        "06_T_vs_heat_density.png",
        "07_X_or_YC2H6_vs_heat_density.png",
        "08_C2H2_vs_heat_density.png",
    ]:
        lines.append("- `{}`\n".format(name))

    (out_dir / "coverage_report.md").write_text("".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not FULL_FILE.exists():
        raise FileNotFoundError("Could not find full parquet: {}".format(FULL_FILE))
    if not OFF_FILE.exists():
        raise FileNotFoundError("Could not find off-manifold parquet: {}".format(OFF_FILE))

    full = read_relevant_parquet(FULL_FILE, "full", FULL_SAMPLE_MAX)
    off = read_relevant_parquet(OFF_FILE, "offmanifold", OFF_SAMPLE_MAX)

    # Work on samples for expensive metrics/plots.
    full_sample = full["sample"]
    off_sample = off["sample"]
    combined_sample = pd.concat([full_sample, off_sample], ignore_index=True)

    # Candidate physical state variables. Only existing columns are used.
    state_cols = [
        "T [K]",
        "P [bar]",
        "tau [ms]",
        "X_C2H6 [%]",
        "Y_C2H6",
        "Y_C2H4",
        "Y_C2H2",
        "Y_C3H6",
        "Y_C3H8",
        "Heat absorption [MW/m3]",
        "Wall source [MW/m3]",
    ]
    state_cols = [c for c in state_cols if c in combined_sample.columns]

    major_y_cols = ["Y_{}".format(sp) for sp in MAJOR_SPECIES if "Y_{}".format(sp) in combined_sample.columns]
    design_cols = [
        "T_in [K]",
        "P_in [bar]",
        "steam_to_ethane [kg/kg]",
        "H_peak [MW/m2]",
        "diameter [m]",
        "Re_in [k]",
        "U_in [m/s]",
        "mdot [kg/s]",
    ]
    design_cols = [c for c in design_cols if c in combined_sample.columns]
    range_cols = list(dict.fromkeys(state_cols + design_cols + ["sum_selected_Y [-]"]))

    print("\nWriting report to: {}".format(OUT_DIR))

    # Tables.
    quality = quality_checks(combined_sample, major_y_cols)
    quality.to_csv(OUT_DIR / "quality_checks_sample.csv", index=False)

    ranges = range_table_by_dataset(combined_sample, range_cols)
    ranges.to_csv(OUT_DIR / "range_comparison_by_dataset.csv", index=False)

    outside = outside_full_envelope(full_sample, off_sample, state_cols)
    outside.to_csv(OUT_DIR / "off_outside_full_envelope.csv", index=False)

    pair_gain = pairwise_bin_gain(full_sample, off_sample, state_cols, bins=BINS_2D)
    pair_gain.to_csv(OUT_DIR / "pairwise_2d_bin_gain.csv", index=False)

    nn_summary, nn_distances, nn_cols = nearest_off_to_full(full_sample, off_sample, state_cols)
    nn_summary.to_csv(OUT_DIR / "nearest_off_to_full_summary.csv", index=False)
    nn_distances.to_csv(OUT_DIR / "nearest_off_to_full_distances_sample.csv", index=False)

    # Plots.
    plot_counts(full, off, OUT_DIR)
    pca_projection, pca_evr, pca_cols = plot_pca(combined_sample, state_cols, OUT_DIR)
    plot_nearest_distance(nn_distances, nn_summary, OUT_DIR)
    plot_outside_envelope(outside, OUT_DIR)
    plot_pairwise_gain(pair_gain, OUT_DIR)
    corr_heatmap(combined_sample, state_cols, OUT_DIR)

    # Key density panels.
    if "T [K]" in state_cols and "Heat absorption [MW/m3]" in state_cols:
        density_panel(full_sample, off_sample, "T [K]", "Heat absorption [MW/m3]", OUT_DIR / "06_T_vs_heat_density.png")
    x_eth = "X_C2H6 [%]" if "X_C2H6 [%]" in state_cols else "Y_C2H6"
    if x_eth in state_cols and "Heat absorption [MW/m3]" in state_cols:
        density_panel(full_sample, off_sample, x_eth, "Heat absorption [MW/m3]", OUT_DIR / "07_X_or_YC2H6_vs_heat_density.png")
    if "Y_C2H2" in state_cols and "Heat absorption [MW/m3]" in state_cols:
        density_panel(full_sample, off_sample, "Y_C2H2", "Heat absorption [MW/m3]", OUT_DIR / "08_C2H2_vs_heat_density.png")

    metadata = {
        "full": {
            "path": full["path"],
            "metadata_rows": full["metadata_rows"],
            "sample_rows_used": int(len(full_sample)),
            "columns_used": full["columns_used"],
            "species_cols": full["species_cols"],
        },
        "offmanifold": {
            "path": off["path"],
            "metadata_rows": off["metadata_rows"],
            "sample_rows_used": int(len(off_sample)),
            "columns_used": off["columns_used"],
            "species_cols": off["species_cols"],
        },
        "state_cols_used": state_cols,
        "nearest_distance_cols_used": nn_cols,
        "pca_cols_used": pca_cols,
        "pca_explained_variance_ratio": None if pca_evr is None else [float(x) for x in pca_evr],
        "bins_2d": BINS_2D,
        "random_seed": RANDOM_SEED,
    }
    (OUT_DIR / "coverage_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    write_report(
        OUT_DIR,
        metadata,
        state_cols,
        quality,
        ranges,
        outside,
        pair_gain,
        nn_summary,
        pca_evr,
    )

    print("\nDone. Key outputs:")
    print("  {}".format(OUT_DIR / "coverage_report.md"))
    print("  {}".format(OUT_DIR / "range_comparison_by_dataset.csv"))
    print("  {}".format(OUT_DIR / "off_outside_full_envelope.csv"))
    print("  {}".format(OUT_DIR / "pairwise_2d_bin_gain.csv"))
    print("  {}".format(OUT_DIR / "nearest_off_to_full_summary.csv"))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()

#!/usr/bin/env python3
# SCRIPT_VERSION = "generate_database_Isothermal_final_fast_v3native_pfr_first_fallback_v7_native_failure_to_state_probe"
"""
Generate a CRACKSIM-backed isothermal enrichment parquet from the manifest produced by
identify_isothermal_empty_regions.py.

This script is deliberately the production step that the design script is NOT:

    isothermal_enrichment_manifest.csv  --CRACKSIM-->  isothermal_enrichment_cracksim.parquet

It reuses the existing SCARFS v2 CRACKSIM infrastructure:
- SA_CRACKSIM.dll worker initialisation via scarfs.data.generation_v3.init_worker_cracksim
- sequential worker READY handshake to avoid CRACKSIM init races
- GenV2Settings / PerturbConfig
- CRACKSIM source-term evaluation through generation_v3.eval_offmanifold_points(...)
- scratch/case_*.parquet atomic commits and --skip-existing resume
- final parquet merge and schema alignment to a reference database

Important modelling convention
------------------------------
For each manifest row:
- design_kind = pfr_first_request:
    A fresh-feed, fixed-T, fixed-p isothermal PFR attempt is integrated using CRACKSIM
    source terms.  The trajectory is checked against the requested T-X-log(tau) bin.
    If it misses the requested sparse bin, an anchored state-probe row is appended using
    the full Y_* vector supplied in the manifest.
- design_kind = state_probe:
    The supplied T, p, tau and full Y_* composition copied from a real database anchor are evaluated directly with CRACKSIM.
- design_kind = isothermal_pfr:
    Legacy compatibility path.  Prefer pfr_first_request for new campaigns.

This is intended for source-term surrogate training, where the local mapping is:

    source = f(T, p, Y)

so exact CRACKSIM source terms at well-chosen states are the critical requirement.

Typical usage, from the SCARFS repo root:

    python scripts/generate_database_Isothermal.py --manifest out_balanced_iso_r1/isothermal_enrichment_manifest.csv --schema-reference out_v2/full.parquet --out out_v2_iso_r1 --out-name isothermal_enrichment_cracksim --n-cpu 10 --n-points 160 --skip-existing

Then balance with:

    python scripts/build_balanced_isothermal_enrichment.py --full out_v2/full.parquet --off out_v2/offmanifold_1000000.parquet --enriched out_v2_iso_r1/isothermal_enrichment_cracksim.parquet --out out_balanced_iso_final --mode balance --max-existing-rows 500000 --bin-cap 250 --T-hard-max-K 1600 --final-T-max-K 1600
"""
from __future__ import annotations

import argparse
import copy
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

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception as exc:  # pragma: no cover
    raise SystemExit("This script requires pyarrow. Install with: pip install pyarrow") from exc

from scarfs.data import generation_v3 as g2
from scarfs.data.generation_v3 import GenV2Settings
try:
    from scarfs.data.generate import finalize_flow
except Exception:  # pragma: no cover
    finalize_flow = None


# -----------------------------
# Constants / species handling
# -----------------------------

EPS = 1.0e-300
SCRIPT_VERSION = "generate_database_Isothermal_final_fast_v3native_pfr_first_fallback_v7_native_failure_to_state_probe"

SPECIES = ["H2O", "C2H6", "C2H4", "CH4", "H2", "C2H2", "C3H6", "C3H8"]


class NativePfrAttemptFailure(RuntimeError):
    """Native v3 PFR attempt failed before producing a usable trajectory.

    In the isothermal enrichment workflow this is not a fatal case failure:
    it means the requested PFR realization is numerically/physically unstable
    under the native solver, so the sparse target should be filled by the
    already-selected real-Y anchored state probe instead.
    """

    def __init__(self, reason: Any):
        self.reason = str(reason)
        super().__init__(self.reason)

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
TEMP_CANDIDATES = ["T", "T [K]", "Temperature", "Temperature [K]", "temperature", "temperature_K", "T_K"]
PRESSURE_CANDIDATES = ["p", "P", "Pressure", "Pressure [Pa]", "pressure", "pressure_Pa", "P_Pa", "p_Pa"]
TAU_CANDIDATES = ["tau", "tau [s]", "Residence time [s]", "residence_time_s", "time", "t", "t [s]"]
CASE_CANDIDATES = ["CaseID", "case_id", "id", "case", "Case ID"]
SAMPLE_KIND_CANDIDATES = ["sample_kind", "Sample kind", "sample type"]
HEAT_CANDIDATES = ["Reaction heat absorption [J/s/m3]", "Reaction heat absorption [J/m3/s]", "heat_absorption", "S_E"]


@dataclass
class ReferenceMap:
    columns: list[str]
    dtypes: dict[str, str]
    T_col: str
    p_col: str
    tau_col: str
    case_col: str | None
    sample_kind_col: str | None
    species_cols: dict[str, str]
    # All mass-fraction columns in the exact order used by the reference parquet.
    # This is critical because generation_v3.eval_offmanifold_points checks that
    # the anchor Y_* columns match the CRACKSIM/Cantera mechanism species order.
    all_y_cols: list[str]
    # Preferred Y_* columns in Cantera/mechanism order. This is what the CRACKSIM state evaluator expects.
    mechanism_y_cols: list[str] | None = None


# -----------------------------
# Naming helpers
# -----------------------------


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_first(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    columns = list(columns)
    exact = {c: c for c in columns}
    for cand in candidates:
        if cand in exact:
            return cand
    nmap = {_norm(c): c for c in columns}
    for cand in candidates:
        val = nmap.get(_norm(cand))
        if val is not None:
            return val
    return None


def find_species_columns(columns: Iterable[str]) -> dict[str, str]:
    columns = list(columns)
    nmap = {_norm(c): c for c in columns}
    out: dict[str, str] = {}
    for sp, aliases in SPECIES_ALIASES.items():
        candidates: list[str] = []
        for a in aliases:
            candidates += [
                a,
                f"Y_{a}", f"X_{a}", f"Y-{a}", f"X-{a}",
                f"Mass fraction {a}", f"Mole fraction {a}",
                f"mass_fraction_{a}", f"mole_fraction_{a}",
            ]
        for cand in candidates:
            c = nmap.get(_norm(cand))
            if c is not None:
                out[sp] = c
                break
    return out


def find_all_y_columns(columns: Iterable[str]) -> list[str]:
    """Return all mechanism mass-fraction columns in the reference order.

    generation_v3.eval_offmanifold_points expects the anchor parquet to contain
    Y_* columns for the full mechanism, and usually in the same order as the
    mechanism species list. The v1 script only wrote the 8 design species, which
    caused: 'anchor parquet species do not match the mechanism ordering'.
    """
    out: list[str] = []
    for c in columns:
        cs = str(c)
        if cs.startswith("Y_") and not cs.startswith("Y_dot") and not cs.startswith("Ydot"):
            out.append(cs)
    return out


def species_name_from_y_col(col: str) -> str:
    return str(col)[2:] if str(col).startswith("Y_") else str(col)


def read_reference_map(reference_path: str | Path | None) -> ReferenceMap:
    if reference_path is None:
        # Fallback to a conventional schema if a reference parquet is not supplied.
        all_y_cols = [f"Y_{sp}" for sp in SPECIES]
        cols = ["CaseID", "sample_kind", "T", "Pressure [Pa]", "tau"] + all_y_cols
        return ReferenceMap(cols, {}, "T", "Pressure [Pa]", "tau", "CaseID", "sample_kind", {sp: f"Y_{sp}" for sp in SPECIES}, all_y_cols)

    schema = pq.read_schema(str(reference_path))
    columns = list(schema.names)
    dtypes = {field.name: str(field.type) for field in schema}

    T_col = find_first(columns, TEMP_CANDIDATES) or "T"
    p_col = find_first(columns, PRESSURE_CANDIDATES) or "Pressure [Pa]"
    tau_col = find_first(columns, TAU_CANDIDATES) or "tau"
    case_col = find_first(columns, CASE_CANDIDATES)
    sample_kind_col = find_first(columns, SAMPLE_KIND_CANDIDATES)
    species_cols = find_species_columns(columns)
    all_y_cols = find_all_y_columns(columns)
    if not all_y_cols:
        # Fallback, but this is less safe than a real reference parquet because CRACKSIM
        # normally expects the full mechanism species vector.
        all_y_cols = [f"Y_{sp}" for sp in SPECIES]

    # If a design species column is absent from the reference, still create a conventional Y_* column.
    for sp in SPECIES:
        species_cols.setdefault(sp, f"Y_{sp}")

    missing_core = []
    if T_col not in columns:
        missing_core.append("temperature")
    if p_col not in columns:
        missing_core.append("pressure")
    if tau_col not in columns:
        missing_core.append("tau/residence time")
    if missing_core:
        warnings.warn(
            f"Could not identify these core fields in schema reference {reference_path}: {missing_core}. "
            "The script will create conventional names, but you should inspect the output schema."
        )
    return ReferenceMap(columns, dtypes, T_col, p_col, tau_col, case_col, sample_kind_col, species_cols, all_y_cols)


# -----------------------------
# Mechanism / CRACKSIM helpers
# -----------------------------


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



def mechanism_y_columns(mech_path: str, fallback: list[str]) -> list[str]:
    """Return Y_* columns in the exact mechanism species order.

    The existing off-manifold CRACKSIM evaluator usually validates that the anchor
    dataframe contains Y_* columns in mechanism order. Using the reference parquet
    schema alone is not always sufficient, so this function prefers chem.yaml.
    """
    try:
        import cantera as ct
        gas = ct.Solution(mech_path)
        return [f"Y_{sp}" for sp in gas.species_names]
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Could not read mechanism species from {mech_path}: {exc}; falling back to reference Y_* columns.")
        return list(fallback)


def settings_doc_from_args(args: argparse.Namespace) -> dict[str, Any]:
    settings = GenV2Settings(n_points=args.n_points, solver_rtol=args.rtol, solver_atol=args.atol)
    if hasattr(settings, "storage") and hasattr(settings.storage, "max_frac_jump"):
        settings.storage.max_frac_jump = args.max_frac_jump
    _set_temperature_cap_attrs(settings, float(args.T_hard_max_K))
    doc = {**settings.__dict__}
    if hasattr(settings, "storage"):
        doc["storage"] = settings.storage.__dict__
    # pfr_ode_steps is deprecated in the fast native-v3 backend but kept in the
    # settings document for backward-compatible schema/reporting.
    doc["pfr_ode_steps"] = int(args.pfr_ode_steps or args.n_points)
    doc["native_T_cap_K"] = float(args.T_hard_max_K)
    doc["T_hard_max_K"] = float(args.T_hard_max_K)
    # Isothermal enrichment should not drop a native PFR solely because the solver
    # stopped before the requested L. The partial trajectory is retained and, if it
    # misses the sparse bin, the anchored state-probe fallback fills that target.
    doc["keep_truncated_pfr"] = True
    for name in _TEMP_CAP_ATTR_NAMES:
        doc[name] = float(args.T_hard_max_K)
    return doc


def _native_settings_doc_for_g2(doc: dict[str, Any]) -> dict[str, Any]:
    """Return only keys accepted by the native g2.settings_from_doc/GenV2Settings.

    The isothermal enrichment generator carries additional campaign metadata such
    as T_hard_max_K/native_T_cap_K/pfr_ode_steps. Older SCARFS versions rebuild
    GenV2Settings with **doc, so those extra keys cause errors like
    "__init__() got an unexpected keyword argument 'T_hard_max_K'" during worker
    startup.  Keep the native settings document clean and apply the 1600 K cap
    separately via the temperature-cap patching functions.
    """
    allowed: set[str] = set()
    try:
        import dataclasses
        if dataclasses.is_dataclass(GenV2Settings):
            allowed |= {f.name for f in dataclasses.fields(GenV2Settings)}
    except Exception:
        pass
    try:
        import inspect
        sig = inspect.signature(GenV2Settings)
        allowed |= {k for k in sig.parameters if k != "self"}
    except Exception:
        pass
    # Conservative fallback for the known v2 settings names.
    allowed |= {"n_points", "solver_rtol", "solver_atol", "storage", "keep_truncated_pfr"}
    clean = {k: v for k, v in doc.items() if k in allowed}
    if "storage" in doc:
        clean["storage"] = doc["storage"]
    return clean


def settings_from_doc(doc: dict[str, Any]):
    native_doc = _native_settings_doc_for_g2(doc)
    try:
        if hasattr(g2, "settings_from_doc"):
            settings = g2.settings_from_doc(native_doc)
        else:
            raise AttributeError("g2.settings_from_doc unavailable")
    except Exception as exc:  # noqa: BLE001
        # Robust fallback for SCARFS revisions where settings_from_doc is strict.
        msg = str(exc)
        if "unexpected keyword" not in msg and "settings_from_doc unavailable" not in msg:
            warnings.warn(f"native g2.settings_from_doc failed; falling back to GenV2Settings: {exc}")
        settings = GenV2Settings(
            n_points=int(doc.get("n_points", 160)),
            solver_rtol=float(doc.get("solver_rtol", 1e-9)),
            solver_atol=float(doc.get("solver_atol", 1e-16)),
        )
        if hasattr(settings, "storage") and isinstance(doc.get("storage"), dict):
            for k, v in doc["storage"].items():
                if hasattr(settings.storage, k):
                    try:
                        setattr(settings.storage, k, v)
                    except Exception:
                        pass
    _set_temperature_cap_attrs(settings, float(doc.get("native_T_cap_K", doc.get("T_hard_max_K", 1600.0))))
    try:
        setattr(settings, "keep_truncated_pfr", bool(doc.get("keep_truncated_pfr", True)))
    except Exception:
        pass
    return settings



# -----------------------------
# Native-v2 temperature-cap override for isothermal enrichment
# -----------------------------

_TEMP_CAP_ATTR_NAMES = [
    "T_hard_max_K", "T_cap_K", "Tmax_K", "T_max_K", "max_T_K",
    "max_temperature_K", "temperature_max_K", "temperature_cap_K",
    "T_stop_K", "T_drop_K", "T_safety_cap_K", "hard_T_cap_K",
]

def _set_temperature_cap_attrs(obj: Any, cap_K: float) -> None:
    """Best-effort patch of settings/runner/module objects with a higher T cap.

    generate_database_v2.py intentionally caps trajectories around 1150 degC
    (~1423 K). For this isothermal enrichment campaign we explicitly allow
    fixed-temperature CRACKSIM evaluations up to --T-hard-max-K, normally 1600 K.
    Different SCARFS revisions used different attribute names, so we set all
    plausible names when possible. This does not alter the requested chemistry;
    it only prevents the native v3 safety gate from dropping valid enrichment
    cases above 1150 degC.
    """
    if obj is None:
        return
    for name in _TEMP_CAP_ATTR_NAMES:
        try:
            setattr(obj, name, float(cap_K))
        except Exception:
            pass
    try:
        storage = getattr(obj, "storage", None)
        if storage is not None:
            for name in _TEMP_CAP_ATTR_NAMES:
                try:
                    setattr(storage, name, float(cap_K))
                except Exception:
                    pass
    except Exception:
        pass

_NATIVE_CAP_PATCH_DONE = False

def _patch_generation_v3_temperature_cap(settings: Any, cap_K: float) -> None:
    """Patch generation_v3's 1150 degC safety cap to the campaign cap.

    Preferred path: set settings/module attributes such as T_cap_K/max_T_K.
    Fallback path: if a SCARFS revision hard-coded the ~1423 K constant inside a
    Python helper, replace only numeric constants in the 1410--1435 K band in
    functions whose source/constants indicate a temperature-cap check. This keeps
    the fast g2.run_case_v2(...) architecture without reverting to the slow
    Python-loop PFR prototype.
    """
    global _NATIVE_CAP_PATCH_DONE
    cap_K = float(cap_K)
    _set_temperature_cap_attrs(settings, cap_K)
    _set_temperature_cap_attrs(g2, cap_K)

    # Patch any module-level cap-like numeric global close to 1150 degC.
    for name, value in list(vars(g2).items()):
        lname = name.lower()
        if any(token in lname for token in ["cap", "max", "limit", "stop"]) and any(token in lname for token in ["t", "temp"]):
            if isinstance(value, (int, float, np.integer, np.floating)) and 1410.0 <= float(value) <= 1435.0:
                try:
                    setattr(g2, name, cap_K)
                except Exception:
                    pass

    # Last-resort patch for hard-coded Python constants. This is deliberately
    # narrow: only functions with cap/exceeded/drop source text or constants and
    # only constants in the old temperature-cap window are modified.
    patched_funcs = []
    try:
        import inspect
        import types
        for fname, func in list(vars(g2).items()):
            if not inspect.isfunction(func):
                continue
            try:
                consts = func.__code__.co_consts
            except Exception:
                continue
            numeric_hits = [c for c in consts if isinstance(c, (int, float)) and 1410.0 <= float(c) <= 1435.0]
            if not numeric_hits:
                continue
            try:
                source = inspect.getsource(func).lower()
            except Exception:
                source = ""
            text = source + " " + " ".join(str(c).lower() for c in consts if isinstance(c, str))
            if not ("cap" in text or "exceeded" in text or "drop" in text or "temperature" in text):
                continue
            new_consts = tuple(float(cap_K) if isinstance(c, float) and 1410.0 <= float(c) <= 1435.0 else (int(round(cap_K)) if isinstance(c, int) and 1410 <= int(c) <= 1435 else c) for c in consts)
            if new_consts != consts:
                try:
                    func.__code__ = func.__code__.replace(co_consts=new_consts)
                    patched_funcs.append(fname)
                except Exception:
                    pass
    except Exception:
        patched_funcs = []

    if patched_funcs and not _NATIVE_CAP_PATCH_DONE:
        warnings.warn(
            "Patched generation_v3 native temperature-cap constants for isothermal enrichment: "
            + ", ".join(sorted(set(patched_funcs))[:12])
            + (" ..." if len(set(patched_funcs)) > 12 else "")
        )
    _NATIVE_CAP_PATCH_DONE = True


# -----------------------------
# Manifest -> CRACKSIM state anchors
# -----------------------------


def _composition_from_conversion(conv: np.ndarray, steam_to_ethane_mass: np.ndarray) -> dict[str, np.ndarray]:
    """Same broad state-space composition model as the design script.

    The CRACKSIM call computes exact source terms at the generated states.
    """
    conv = np.clip(np.asarray(conv, dtype=float), 0.0, 0.999999)
    s2e = np.clip(np.asarray(steam_to_ethane_mass, dtype=float), 1.0e-12, None)
    ethane_unconverted = 1.0 - conv
    product_mass = conv
    steam_mass = s2e
    splits = {
        "C2H4": 0.58,
        "CH4": 0.18,
        "H2": 0.04,
        "C2H2": 0.08,
        "C3H6": 0.07,
        "C3H8": 0.02,
        "C2H6": 0.03,
    }
    total = ethane_unconverted + product_mass + steam_mass
    y = {"H2O": steam_mass / np.clip(total, EPS, None)}
    y["C2H6"] = (ethane_unconverted + splits["C2H6"] * product_mass) / np.clip(total, EPS, None)
    for sp, frac in splits.items():
        if sp == "C2H6":
            continue
        y[sp] = frac * product_mass / np.clip(total, EPS, None)
    summ = np.zeros_like(conv, dtype=float)
    for arr in y.values():
        summ += arr
    for sp in y:
        y[sp] = np.clip(y[sp] / np.clip(summ, EPS, None), 0.0, 1.0)
    return y


def _fresh_feed_composition(steam_to_ethane_mass: float) -> dict[str, float]:
    s2e = max(float(steam_to_ethane_mass), 0.0)
    total = 1.0 + s2e
    y = {sp: 0.0 for sp in SPECIES}
    y["C2H6"] = 1.0 / total
    y["H2O"] = s2e / total
    return y



def _first_finite(case_row: dict[str, Any], keys: Iterable[str], default: float) -> float:
    for k in keys:
        if k in case_row:
            try:
                v = float(case_row.get(k))
                if math.isfinite(v):
                    return v
            except Exception:
                pass
    return float(default)


def _gas_y_from_anchor_row(row: pd.Series, gas: Any) -> np.ndarray:
    """Build a Cantera mass-fraction vector from anchor Y_* columns."""
    y = np.zeros(int(gas.n_species), dtype=float)
    for i, sp in enumerate(gas.species_names):
        col = f"Y_{sp}"
        if col in row.index:
            try:
                y[i] = max(float(row[col]), 0.0)
            except Exception:
                y[i] = 0.0
    s = float(y.sum())
    if not math.isfinite(s) or s <= 0.0:
        # Safe fallback; should not happen if the manifest/schema are valid.
        if "C2H6" in gas.species_names:
            y[gas.species_index("C2H6")] = 1.0
    else:
        y /= s
    return y


def _compute_flow_metadata(gas: Any, anchors: pd.DataFrame, case_row: dict[str, Any], tau_vals: np.ndarray) -> dict[str, Any]:
    """Compute physically consistent D/Re/U/mdot/L metadata for a designed tau.

    The design script chooses a target residence time and target hydrodynamic family
    (diameter + target Reynolds number).  Here we recompute rho and mu with Cantera
    and convert that design into actual flow metadata:

        U_in = Re_target * mu_in / (rho_in * D)
        mdot = rho_in * A * U_in
        L = U_in * tau_end

    For PFR rows, mdot and D are case constants and local U/Re are recomputed from
    local rho/mu. For state probes, the same relation gives a physically meaningful
    tau tag. v9 also preserves explicit T-log(tau) water-fill manifest fields: tau = L / U for that local state.
    """
    n = len(anchors)
    D = _first_finite(case_row, ["diameter_m", "D_m", "diameter [m]"], 0.0306)
    D = max(D, 1e-6)
    A = math.pi * D * D / 4.0
    target_Re = _first_finite(case_row, ["target_Re", "Re_target", "Re"], 1.0e5)
    target_Re = max(target_Re, 1.0)
    tau_end = _first_finite(case_row, ["tau_end_s", "tau_s", "tau"], float(np.nanmax(tau_vals)) if len(tau_vals) else 0.0)
    tau_end = max(tau_end, 1e-12)

    # Inlet/design state for mdot/L. For isothermal_pfr the first anchor is fresh feed;
    # for state_probe it is the designed local state.
    first = anchors.iloc[0]
    T0 = float(first.iloc[0]) if False else _first_finite(case_row, ["T_K"], np.nan)
    p0 = _first_finite(case_row, ["p_Pa"], np.nan)
    if not math.isfinite(T0):
        for c in ["T", "T [K]", "Temperature [K]", "temperature_K"]:
            if c in first.index:
                T0 = float(first[c]); break
    if not math.isfinite(p0):
        for c in ["Pressure [Pa]", "P", "p", "P_Pa", "p_Pa"]:
            if c in first.index:
                p0 = float(first[c]); break
    y0 = _gas_y_from_anchor_row(first, gas)
    gas.TPY = T0, p0, y0
    rho_in = float(gas.density)
    try:
        mu_in = float(gas.viscosity)
    except Exception:
        mu_in = 3.5e-5 * max(T0 / 1000.0, 0.2) ** 0.70

    U_in = float(target_Re * mu_in / max(rho_in * D, EPS))
    # Optional manifest estimates can be used if target_Re was deliberately clipped by the design script.
    if not math.isfinite(U_in) or U_in <= 0.0:
        U_in = _first_finite(case_row, ["estimated_U_m_s", "U_in_m_s"], 1.0)
    mdot = float(rho_in * A * U_in)
    L = float(U_in * tau_end)

    rho_local = np.empty(n, dtype=float)
    mu_local = np.empty(n, dtype=float)
    T_local = np.empty(n, dtype=float)
    p_local = np.empty(n, dtype=float)
    for i, (_, r) in enumerate(anchors.iterrows()):
        # Use the columns present in anchors. They were built with reference-compatible names.
        T_val = None
        p_val = None
        for c in ["T", "T [K]", "Temperature [K]", "temperature_K", "T_K"]:
            if c in r.index:
                T_val = float(r[c]); break
        for c in ["Pressure [Pa]", "P", "p", "P_Pa", "p_Pa", "pressure_Pa"]:
            if c in r.index:
                p_val = float(r[c]); break
        if T_val is None:
            T_val = T0
        if p_val is None:
            p_val = p0
        yy = _gas_y_from_anchor_row(r, gas)
        gas.TPY = float(T_val), float(p_val), yy
        T_local[i] = float(T_val)
        p_local[i] = float(p_val)
        rho_local[i] = float(gas.density)
        try:
            mu_local[i] = float(gas.viscosity)
        except Exception:
            mu_local[i] = 3.5e-5 * max(float(T_val) / 1000.0, 0.2) ** 0.70

    U_local = mdot / np.clip(rho_local * A, EPS, None)
    Re_local = rho_local * U_local * D / np.clip(mu_local, EPS, None)
    return {
        "diameter_m": D,
        "area_m2": A,
        "length_m": L,
        "mdot_kg_s": mdot,
        "U_in_m_s": U_in,
        "target_Re": target_Re,
        "rho_in_kg_m3": rho_in,
        "mu_in_Pa_s": mu_in,
        "rho_local_kg_m3": rho_local,
        "mu_local_Pa_s": mu_local,
        "U_local_m_s": U_local,
        "Re_local": Re_local,
        "T_local_K": T_local,
        "p_local_Pa": p_local,
    }

def _tau_grid(tau_end_s: float, n: int) -> np.ndarray:
    tau_end_s = max(float(tau_end_s), 1e-12)
    n = max(1, int(n))
    if n == 1:
        return np.array([tau_end_s], dtype=float)
    # Include zero-like inlet state but avoid exactly zero if downstream code logs tau.
    t0 = min(1e-12, tau_end_s * 1e-9)
    return np.unique(np.r_[t0, np.geomspace(max(t0, tau_end_s * 1e-6), tau_end_s, n - 1)]).astype(float)


def _build_case_anchors(case_row: dict[str, Any], ref: ReferenceMap, n_points: int) -> pd.DataFrame:
    case_id = int(case_row.get("case_id", case_row.get("id", 0)))
    kind = str(case_row.get("design_kind", "state_probe"))
    T = float(case_row.get("T_K"))
    p = float(case_row.get("p_Pa"))
    tau_end = float(case_row.get("tau_end_s", 0.0))
    s2e = float(case_row.get("steam_to_ethane_mass", 0.0))
    target_conversion = float(case_row.get("target_conversion", case_row.get("conversion_proxy", 0.0)))

    if T > 1600.0 + 1e-9:
        raise ValueError(f"Case {case_id}: T_K={T} exceeds hard 1600 K cap")
    if not (150000.0 <= p <= 350000.0):
        warnings.warn(f"Case {case_id}: p_Pa={p:g} is outside requested 1.5--3.5 bar range")
    if not (0.0 <= s2e <= 1.0):
        warnings.warn(f"Case {case_id}: steam_to_ethane_mass={s2e:g} is outside requested 0--1 range")

    if kind == "isothermal_pfr":
        tau = _tau_grid(tau_end, n_points)
        # Smooth conversion placement from fresh feed to target conversion.
        # This deliberately fills the designed state-space path. CRACKSIM evaluates rates at each state.
        frac = np.clip(tau / max(tau_end, EPS), 0.0, 1.0)
        conv = target_conversion * (1.0 - np.exp(-5.0 * frac)) / (1.0 - np.exp(-5.0))
        conv[0] = 0.0
        y = _composition_from_conversion(conv, np.full_like(conv, s2e, dtype=float))
        n = len(tau)
    else:
        tau = np.array([max(tau_end, 1e-12)], dtype=float)
        n = 1
        # Prefer explicit Y_* from the manifest for state probes.
        y_scalar: dict[str, float] = {}
        for sp in SPECIES:
            key = f"Y_{sp}"
            val = case_row.get(key, np.nan)
            if pd.isna(val):
                y_scalar[sp] = 0.0
            else:
                y_scalar[sp] = max(float(val), 0.0)
        total_y = sum(y_scalar.values())
        if total_y <= 0.0:
            y_scalar = {k: float(v[0]) for k, v in _composition_from_conversion(np.array([target_conversion]), np.array([s2e])).items()}
            total_y = sum(y_scalar.values())
        y_scalar = {k: v / max(total_y, EPS) for k, v in y_scalar.items()}
        y = {sp: np.array([y_scalar.get(sp, 0.0)], dtype=float) for sp in SPECIES}

    data: dict[str, Any] = {}
    # Put core columns under reference-compatible names.
    data[ref.T_col] = np.full(n, T, dtype=float)
    data[ref.p_col] = np.full(n, p, dtype=float)
    data[ref.tau_col] = tau
    if ref.case_col:
        data[ref.case_col] = np.full(n, case_id, dtype=int)
    else:
        data["CaseID"] = np.full(n, case_id, dtype=int)
    if ref.sample_kind_col:
        data[ref.sample_kind_col] = np.full(n, "isothermal_enrichment", dtype=object)
    else:
        data["sample_kind"] = np.full(n, "isothermal_enrichment", dtype=object)
    # Write the full mechanism mass-fraction vector in the reference/mechanism order.
    # Unknown species are set to zero; the design species are inserted by name.
    # This fixes the common CRACKSIM off-manifold error:
    #     anchor parquet species do not match the mechanism ordering
    y_by_col: dict[str, np.ndarray] = {}
    y_cols_for_anchor = ref.mechanism_y_cols or ref.all_y_cols
    if kind == "state_probe":
        # Key v11 fix: use the full Y_* vector supplied by the design manifest.
        # For anchor-based state probes this vector was copied from a real solved CRACKSIM state,
        # preserving radicals/minors/heavies instead of zeroing all non-design species.
        has_full_manifest_y = False
        for y_col in y_cols_for_anchor:
            val = case_row.get(y_col, np.nan)
            if not pd.isna(val):
                y_by_col[y_col] = np.array([max(float(val), 0.0)], dtype=float)
                has_full_manifest_y = True
            else:
                y_by_col[y_col] = np.zeros(n, dtype=float)
        if not has_full_manifest_y:
            warnings.warn(f"Case {case_id}: state_probe manifest has no full Y_* columns; using legacy 8-species fallback")
            for y_col in y_cols_for_anchor:
                sp_name = species_name_from_y_col(y_col)
                y_by_col[y_col] = y.get(sp_name, np.zeros(n, dtype=float))
    else:
        for y_col in y_cols_for_anchor:
            sp_name = species_name_from_y_col(y_col)
            y_by_col[y_col] = y.get(sp_name, np.zeros(n, dtype=float))
    # If the reference map used aliases for some design species, also honor those columns.
    for sp in SPECIES:
        col = ref.species_cols.get(sp, f"Y_{sp}")
        if col not in y_by_col:
            y_by_col[col] = y.get(sp, np.zeros(n, dtype=float))
    # Normalize across the full Y_* vector to avoid tiny roundoff deviations.
    y_sum = np.zeros(n, dtype=float)
    for arr in y_by_col.values():
        y_sum += np.asarray(arr, dtype=float)
    for col, arr in y_by_col.items():
        data[col] = np.asarray(arr, dtype=float) / np.clip(y_sum, EPS, None)

    # Metadata columns. These are allowed in addition to the reference schema and help debugging.
    data["iso_case_id"] = np.full(n, case_id, dtype=int)
    data["iso_design_region"] = np.full(n, str(case_row.get("design_region", "unknown")), dtype=object)
    data["iso_design_kind"] = np.full(n, kind, dtype=object)
    data["iso_target_conversion"] = np.full(n, target_conversion, dtype=float)
    data["iso_steam_to_ethane_mass"] = np.full(n, s2e, dtype=float)
    data["iso_tau_end_s"] = np.full(n, tau_end, dtype=float)
    data["iso_tau_s"] = tau.astype(float)
    data["iso_row_in_case"] = np.arange(n, dtype=int)
    data["iso_anchor_conversion"] = conv.astype(float) if kind == "isothermal_pfr" else np.array([target_conversion], dtype=float)
    data["iso_manifest_version"] = np.full(n, SCRIPT_VERSION, dtype=object)
    # Preserve explicit T-log(tau) water-fill metadata in the anchors as well;
    # the CRACKSIM evaluator may drop these, but _stamp_manifest_metadata restores them.
    for key in [
        "target_tau_s", "tau_design_mode", "coverage_round", "coverage_priority",
        "tx_bin_i", "tx_bin_j", "tx_bin_T_low_K", "tx_bin_T_high_K",
        "tx_bin_X_low", "tx_bin_X_high", "ttau_bin_k",
        "ttau_bin_logtau_low", "ttau_bin_logtau_high",
        "ttau_bin_tau_low_s", "ttau_bin_tau_high_s", "ttau_bin_count_before",
        "tau_estimate_s", "log10_tau_s", "log10_tau_estimate_s", "tau_over_tau_estimate",
        "pfr_tau_ratio_min", "pfr_tau_ratio_max", "tau_physicality_flag",
        "state_probe_composition_source", "anchor_pool_id", "anchor_source_file",
        "anchor_conversion_proxy", "anchor_steam_to_ethane_mass", "anchor_T_K", "anchor_p_Pa", "anchor_tau_s",
        "state_probe_composition_source", "anchor_pool_id", "anchor_source_file",
        "anchor_conversion_proxy", "anchor_steam_to_ethane_mass", "anchor_T_K", "anchor_p_Pa", "anchor_tau_s",
        "hydro_design_mode", "estimated_length_m", "estimated_U_m_s", "estimated_mdot_kg_s",
    ]:
        if key in case_row:
            data[f"iso_manifest_{key}"] = np.full(n, case_row.get(key), dtype=object)
    anchors = pd.DataFrame(data)
    return anchors



def _rate_column_for_y_col(y_col: str, columns: Iterable[str]) -> str | None:
    """Find the CRACKSIM dY/dt column corresponding to a Y_* column."""
    sp = species_name_from_y_col(y_col)
    candidates = [
        f"dYdt_{sp}", f"dYdt_{y_col}", f"dYdt_Y_{sp}",
        f"dYdt_{sp} [-/s]", f"dYdt_{sp} [1/s]", f"dYdt_{sp} [s-1]",
        f"dY_dt_{sp}", f"Ydot_{sp}", f"Y_dot_{sp}",
    ]
    nmap = {_norm(c): c for c in columns}
    for cand in candidates:
        if _norm(cand) in nmap:
            return nmap[_norm(cand)]
    # Fallback: look for a dYdt column whose normalized tail contains the species name.
    nsp = _norm(sp)
    for c in columns:
        nc = _norm(c)
        if nc.startswith('dydt') and nsp in nc:
            return c
    return None


def _fresh_feed_y_by_col(ref: ReferenceMap, steam_to_ethane_mass: float) -> dict[str, float]:
    """Fresh ethane/steam mass-fraction vector in the reference Y-column order."""
    s2e = max(float(steam_to_ethane_mass), 0.0)
    y = {c: 0.0 for c in (ref.mechanism_y_cols or ref.all_y_cols)}
    c2h6_col = ref.species_cols.get('C2H6', 'Y_C2H6')
    h2o_col = ref.species_cols.get('H2O', 'Y_H2O')
    if c2h6_col in y:
        y[c2h6_col] = 1.0 / max(1.0 + s2e, EPS)
    if h2o_col in y:
        y[h2o_col] = s2e / max(1.0 + s2e, EPS)
    total = sum(y.values())
    if total <= 0:
        raise ValueError('Could not build fresh-feed Y vector: C2H6/H2O columns not found in reference schema')
    return {k: v / total for k, v in y.items()}


def _single_state_anchor_from_y(case_row: dict[str, Any], ref: ReferenceMap, y_by_col: dict[str, float], tau_s: float, row_id: int = 0) -> pd.DataFrame:
    """Build one direct-evaluation anchor row at fixed T,p,Y,tau."""
    case_id = int(case_row.get('case_id', case_row.get('id', 0)))
    T = float(case_row.get('T_K'))
    p = float(case_row.get('p_Pa'))
    data: dict[str, Any] = {
        ref.T_col: [T],
        ref.p_col: [p],
        ref.tau_col: [float(max(tau_s, 1e-300))],
        'iso_tau_s': [float(max(tau_s, 1e-300))],
        'iso_row_in_case': [int(row_id)],
        'iso_case_id': [case_id],
    }
    if ref.case_col:
        data[ref.case_col] = [case_id]
    if ref.sample_kind_col:
        data[ref.sample_kind_col] = ['isothermal_enrichment']
    y_cols = ref.mechanism_y_cols or ref.all_y_cols
    y_sum = sum(max(float(y_by_col.get(c, 0.0)), 0.0) for c in y_cols)
    if y_sum <= 0:
        raise ValueError(f'Case {case_id}: nonpositive Y sum in anchor')
    for c in y_cols:
        data[c] = [max(float(y_by_col.get(c, 0.0)), 0.0) / y_sum]
    return pd.DataFrame(data)


def _extract_rates_for_y(evaluated: pd.DataFrame, y_cols: list[str]) -> dict[str, float]:
    """Extract dY/dt from a one-row CRACKSIM-evaluated dataframe."""
    rates: dict[str, float] = {}
    missing: list[str] = []
    for ycol in y_cols:
        rcol = _rate_column_for_y_col(ycol, evaluated.columns)
        if rcol is None:
            missing.append(ycol)
            rates[ycol] = 0.0
        else:
            rates[ycol] = float(pd.to_numeric(evaluated[rcol], errors='coerce').iloc[0])
    if len(missing) == len(y_cols):
        raise RuntimeError('Could not find any dYdt_* columns in CRACKSIM evaluation output; cannot integrate true PFR attempt')
    return rates


def _conversion_from_y_by_col(y_by_col: dict[str, float], ref: ReferenceMap, steam_to_ethane_mass: float) -> float:
    c2h6_col = ref.species_cols.get('C2H6', 'Y_C2H6')
    y_c2h6 = max(float(y_by_col.get(c2h6_col, 0.0)), 0.0)
    y_c2h6_in = 1.0 / max(1.0 + float(steam_to_ethane_mass), EPS)
    return float(np.clip(1.0 - y_c2h6 / max(y_c2h6_in, EPS), 0.0, 1.0))


def _run_explicit_cracksim_isothermal_pfr(case_row: dict[str, Any], ref: ReferenceMap, settings: Any, n_points: int, seed: int, pfr_ode_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Integrate a fresh-feed isothermal PFR attempt using CRACKSIM dY/dt.

    This removes the previous hardcoded Arrhenius reachability proxy.  It is a
    conservative explicit integration intended for target-bin reachability and
    enrichment generation.  It may be slower than repository-native PFR solvers;
    if your SCARFS repository later exposes a native isothermal PFR runner, replace
    this function with that backend while keeping the same metadata interface.
    """
    case_id = int(case_row.get('case_id', case_row.get('id', 0)))
    tau_end = float(case_row.get('tau_end_s', case_row.get('target_tau_s', 0.0)))
    s2e = float(case_row.get('steam_to_ethane_mass', 0.0))
    y_cols = ref.mechanism_y_cols or ref.all_y_cols
    if not y_cols:
        raise RuntimeError('No reference Y_* columns found; cannot integrate PFR')
    n_steps = max(2, int(pfr_ode_steps))
    tau = _tau_grid(max(tau_end, 1e-12), n_steps)
    y = _fresh_feed_y_by_col(ref, s2e)
    evaluated_rows: list[pd.DataFrame] = []
    anchor_rows: list[pd.DataFrame] = []
    t_prev = 0.0
    for k, t in enumerate(tau):
        # Evaluate current state and store it at tau=t.
        anchor = _single_state_anchor_from_y(case_row, ref, y, float(t), row_id=k)
        ev = _eval_cracksim_states(anchor, settings, seed=seed + case_id * 100000 + k)
        evaluated_rows.append(ev)
        anchor_rows.append(anchor)
        if k == len(tau) - 1:
            break
        dt = max(float(tau[k + 1] - t), 0.0)
        rates = _extract_rates_for_y(ev, y_cols)
        for c in y_cols:
            y[c] = max(float(y.get(c, 0.0)) + dt * float(rates.get(c, 0.0)), 0.0)
        # Explicit integration can introduce tiny mass drift or negative clipping.
        # Renormalize to remain a valid mass-fraction vector.
        total = sum(y.values())
        if total <= 0 or not np.isfinite(total):
            raise RuntimeError(f'Case {case_id}: PFR integration produced invalid Y sum at step {k}')
        y = {c: float(v) / total for c, v in y.items()}
    evaluated = pd.concat(evaluated_rows, ignore_index=True)
    anchors = pd.concat(anchor_rows, ignore_index=True)
    # Stamp a physically integrated conversion coordinate for diagnostics.
    c2h6_col = ref.species_cols.get('C2H6', 'Y_C2H6')
    if c2h6_col in anchors.columns:
        y_in = 1.0 / max(1.0 + s2e, EPS)
        anchors['iso_anchor_conversion'] = np.clip(1.0 - pd.to_numeric(anchors[c2h6_col], errors='coerce').to_numpy(float) / max(y_in, EPS), 0.0, 1.0)
    return evaluated, anchors


def _trajectory_hits_target_bin(out: pd.DataFrame, ref: ReferenceMap, case_row: dict[str, Any]) -> bool:
    """Check whether a PFR trajectory reaches the requested X-log(tau) bin."""
    target_X = float(case_row.get('target_conversion', case_row.get('conversion_proxy', np.nan)))
    target_tau = float(case_row.get('target_tau_s', case_row.get('tau_end_s', np.nan)))
    X_half = 0.5 * float(case_row.get('tx_bin_X_high', target_X + 0.05) - case_row.get('tx_bin_X_low', target_X - 0.05))
    loglo = float(case_row.get('ttau_bin_logtau_low', np.log10(max(target_tau, 1e-300)) - 0.125))
    loghi = float(case_row.get('ttau_bin_logtau_high', np.log10(max(target_tau, 1e-300)) + 0.125))
    s2e = float(case_row.get('steam_to_ethane_mass', 0.0))
    ycol = ref.species_cols.get('C2H6', 'Y_C2H6')
    if ycol not in out.columns or 'iso_tau_s' not in out.columns:
        return False
    y_in = 1.0 / max(1.0 + s2e, EPS)
    X = np.clip(1.0 - pd.to_numeric(out[ycol], errors='coerce').to_numpy(float) / max(y_in, EPS), 0.0, 1.0)
    tau = pd.to_numeric(out['iso_tau_s'], errors='coerce').to_numpy(float)
    logtau = np.log10(np.clip(tau, 1e-300, None))
    return bool(np.any((np.abs(X - target_X) <= max(X_half, 1e-12)) & (logtau >= loglo) & (logtau <= loghi)))


def _build_fallback_state_probe_anchors(case_row: dict[str, Any], ref: ReferenceMap) -> pd.DataFrame | None:
    """Build a one-row fallback state probe from the full Y_* vector in the manifest."""
    if str(case_row.get('state_probe_composition_source', '')) != 'real_database_anchor_full_Y_vector':
        return None
    y_cols = ref.mechanism_y_cols or ref.all_y_cols
    y = {c: max(float(case_row.get(c, 0.0)) if not pd.isna(case_row.get(c, np.nan)) else 0.0, 0.0) for c in y_cols}
    if sum(y.values()) <= 0:
        return None
    tau = float(case_row.get('target_tau_s', case_row.get('tau_end_s', 0.0)))
    return _single_state_anchor_from_y(case_row, ref, y, tau, row_id=0)

def _eval_cracksim_states(anchors: pd.DataFrame, settings: Any, seed: int) -> pd.DataFrame:
    """Evaluate source terms at supplied states.

    Prefer a direct state-evaluation function if the repo has one; otherwise use the existing
    off-manifold evaluator with sigma_log=0 and one point per anchor. This is the same CRACKSIM
    path used to generate offmanifold_1000000.parquet, but without perturbation.
    """
    if hasattr(g2, "eval_states_v2"):
        return g2.eval_states_v2(anchors, settings)
    if hasattr(g2, "eval_states"):
        return g2.eval_states(anchors, settings)
    if not hasattr(g2, "eval_offmanifold_points"):
        raise RuntimeError(
            "scarfs.data.generation_v3 does not expose eval_states_v2/eval_states/"
            "eval_offmanifold_points. Add a direct state-evaluation function there, or use your "
            "existing off-manifold evaluator as eval_offmanifold_points."
        )
    if not hasattr(g2, "PerturbConfig"):
        raise RuntimeError("generation_v3.PerturbConfig is required for sigma_log=0 state evaluation.")
    cfg = g2.PerturbConfig(sigma_log=0.0, points_per_anchor=1)
    return g2.eval_offmanifold_points(anchors, len(anchors), settings, cfg, seed=seed)


# -----------------------------------------------------------------------------
# Fast PFR-first backend: reuse the native v3 case solver instead of Python-loop
# source-term marching. This is the key difference versus the explicit PFR
# prototype: the PFR attempt is solved through generation_v3.run_case_v2(...),
# so n_points has the same meaning/density style as generate_database_v2.py.
# -----------------------------------------------------------------------------

_NATIVE_TEMPLATE_RUNNER_CACHE: dict[str, dict[str, Any]] = {}


def _set_if_key_matches(obj: Any, candidates: list[str], value: Any) -> int:
    """Recursively set existing keys/attributes whose normalized name matches candidates."""
    wanted = {_norm(c) for c in candidates}
    n_set = 0
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if _norm(k) in wanted:
                obj[k] = value
                n_set += 1
            else:
                n_set += _set_if_key_matches(obj[k], candidates, value)
    elif isinstance(obj, list):
        for x in obj:
            n_set += _set_if_key_matches(x, candidates, value)
    elif hasattr(obj, "__dict__"):
        for k in list(vars(obj).keys()):
            if _norm(k) in wanted:
                try:
                    setattr(obj, k, value)
                    n_set += 1
                except Exception:
                    pass
            else:
                try:
                    n_set += _set_if_key_matches(getattr(obj, k), candidates, value)
                except Exception:
                    pass
    return n_set


def _set_existing_or_add_root(d: dict[str, Any], candidates: list[str], canonical: str, value: Any) -> None:
    n = _set_if_key_matches(d, candidates, value)
    if n == 0:
        d[canonical] = value


def _native_template_runner(settings: Any, mech_path: str, seed: int) -> dict[str, Any]:
    """Build one finalized v2 runner case as a mutable template.

    We deliberately derive the template from generation_v3 itself. This makes the
    isothermal enrichment generator robust against internal changes in the v2 case
    schema: all obscure fields required by run_case_v2 remain present; we only patch
    the physical degrees of freedom from the sparse-region manifest.
    """
    key = f"{int(getattr(settings, 'n_points', 0))}_{float(getattr(settings, 'solver_rtol', 0.0))}_{float(getattr(settings, 'solver_atol', 0.0))}_{seed}"
    if key in _NATIVE_TEMPLATE_RUNNER_CACHE:
        return copy.deepcopy(_NATIVE_TEMPLATE_RUNNER_CACHE[key])
    if not hasattr(g2, "build_v2_manifest") or not hasattr(g2, "to_runner_case") or not hasattr(g2, "run_case_v2"):
        raise RuntimeError("Fast native backend requires generation_v3.build_v2_manifest, to_runner_case and run_case_v2")
    cases, _manifest = g2.build_v2_manifest("smoke", seed=seed)
    if not cases:
        raise RuntimeError("generation_v3.build_v2_manifest('smoke') returned no cases")
    # The v2 script finalizes flow before converting cases to runner dictionaries.
    # Do the same for the template if the project exposes finalize_flow.
    if finalize_flow is not None:
        try:
            finalize_flow(cases, mech_path)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Could not finalize native v3 template flow before patching ({exc}); continuing with raw template.")
    runner = g2.to_runner_case(cases[0], settings)
    if not isinstance(runner, dict):
        try:
            runner = dict(runner)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"g2.to_runner_case returned unsupported runner type {type(runner)}") from exc
    _NATIVE_TEMPLATE_RUNNER_CACHE[key] = copy.deepcopy(runner)
    return copy.deepcopy(runner)


def _patch_native_runner_from_manifest(runner: dict[str, Any], case_row: dict[str, Any]) -> dict[str, Any]:
    """Patch a v2 runner template with one sparse-region PFR request."""
    r = copy.deepcopy(runner)
    case_id = int(case_row.get("case_id", case_row.get("id", 0)))
    T = float(case_row.get("T_K"))
    p = float(case_row.get("p_Pa"))
    tau_end = float(case_row.get("tau_end_s", case_row.get("target_tau_s", 0.0)))
    s2e = float(case_row.get("steam_to_ethane_mass", 0.0))
    D = _first_finite(case_row, ["diameter_m", "D_m", "diameter [m]"], 0.0306)
    mdot = _first_finite(case_row, ["estimated_mdot_kg_s", "mdot_kg_s", "mdot [kg/s]", "Mass flow [kg/s]"], np.nan)
    U = _first_finite(case_row, ["estimated_U_m_s", "U_in_m_s", "u [m/s]"], np.nan)
    L = _first_finite(case_row, ["estimated_length_m", "length_m", "L_m", "z_end_m"], np.nan)
    Re = _first_finite(case_row, ["target_Re", "Re_target", "Re"], np.nan)

    _set_existing_or_add_root(r, ["id", "case_id", "CaseID", "case"], "id", case_id)
    _set_existing_or_add_root(r, ["regime", "design_region"], "regime", str(case_row.get("design_region", "isothermal_enrichment")))

    # Temperature/pressure/feed. We set all common inlet/outlet/target names to the
    # same T to request a fixed-temperature/native-PFR trajectory when the backend
    # supports it. If the backend ignores some of these aliases, the existing v2
    # template still supplies the other fields it needs.
    _set_existing_or_add_root(r, ["T", "T_K", "T_in", "Tin", "T_in_K", "T0", "temperature_K", "T_target", "T_target_K", "T_out", "Tout", "T_end", "T_wall", "T_wall_K", "T_peak"], "T_K", T)
    native_T_cap = float(case_row.get("native_T_cap_K", case_row.get("T_hard_max_K", 1600.0)))
    _set_existing_or_add_root(r, ["T_cap", "T_cap_K", "T_hard_max_K", "Tmax_K", "T_max_K", "max_T_K", "max_temperature_K", "temperature_cap_K", "temperature_max_K", "T_stop_K", "T_drop_K", "T_safety_cap_K"], "T_cap_K", native_T_cap)
    _set_existing_or_add_root(r, ["p", "P", "P_in", "Pin", "P_in_Pa", "p_Pa", "P_Pa", "pressure", "pressure_Pa"], "p_Pa", p)
    _set_existing_or_add_root(r, ["steam_to_ethane", "steam_to_ethane_mass", "steam_to_ethane_kg_kg", "s2e", "dilution", "steam_dilution"], "steam_to_ethane_mass", s2e)

    # Geometry/hydrodynamics. The identification script already chose D/Re/U/mdot/L
    # such that tau=L/U is consistent. Patch all common names, but keep the template
    # otherwise unchanged.
    _set_existing_or_add_root(r, ["diameter_m", "diameter", "D", "D_m", "hydraulic_diameter_m"], "diameter_m", D)
    if math.isfinite(mdot):
        _set_existing_or_add_root(r, ["mdot", "mdot_kg_s", "mass_flow", "mass_flow_kg_s", "Mass flow [kg/s]"], "mdot_kg_s", mdot)
    if math.isfinite(U):
        _set_existing_or_add_root(r, ["U", "U_in", "U_in_m_s", "u", "u_in", "velocity", "velocity_m_s"], "U_in_m_s", U)
    if math.isfinite(L):
        _set_existing_or_add_root(r, ["L", "L_m", "length", "length_m", "z_end", "z_end_m", "reactor_length_m"], "length_m", L)
    if math.isfinite(Re):
        _set_existing_or_add_root(r, ["Re", "Re_in", "Re_target", "target_Re"], "target_Re", Re)
    _set_existing_or_add_root(r, ["tau", "tau_s", "tau_end", "tau_end_s", "tau_final", "tau_final_s", "residence_time", "residence_time_s"], "tau_end_s", tau_end)

    # Request isothermal/fixed-temperature behaviour if the backend knows such flags.
    # These extra keys are harmless if ignored.
    r.setdefault("isothermal", True)
    r.setdefault("force_isothermal", True)
    r.setdefault("fixed_T", True)
    r.setdefault("T_profile", "isothermal")
    r.setdefault("target_conversion", float(case_row.get("target_conversion", np.nan)))
    return r


def _metadata_anchors_from_native_pfr(out: pd.DataFrame, ref: ReferenceMap, case_row: dict[str, Any]) -> pd.DataFrame:
    """Build metadata anchors matching the native v3 PFR output row count.

    These anchors are only used for stamping iso_* metadata and flow provenance;
    the actual Y_*/dYdt_* fields remain those computed by g2.run_case_v2.
    """
    n = len(out)
    case_id = int(case_row.get("case_id", case_row.get("id", 0)))
    T_req = float(case_row.get("T_K"))
    p_req = float(case_row.get("p_Pa"))
    s2e = float(case_row.get("steam_to_ethane_mass", 0.0))
    # Prefer the native trajectory coordinates if present.
    T_vals = pd.to_numeric(out[ref.T_col], errors="coerce").to_numpy(float) if ref.T_col in out.columns else np.full(n, T_req)
    p_vals = pd.to_numeric(out[ref.p_col], errors="coerce").to_numpy(float) if ref.p_col in out.columns else np.full(n, p_req)
    tau_vals = pd.to_numeric(out[ref.tau_col], errors="coerce").to_numpy(float) if ref.tau_col in out.columns else _tau_grid(float(case_row.get("tau_end_s", case_row.get("target_tau_s", 1e-12))), n)
    # Fill any NaN tau with a monotone grid to keep hit detection usable.
    if not np.all(np.isfinite(tau_vals)):
        fallback_tau = _tau_grid(float(case_row.get("tau_end_s", case_row.get("target_tau_s", 1e-12))), n)
        tau_vals = np.where(np.isfinite(tau_vals), tau_vals, fallback_tau[:n])
    data: dict[str, Any] = {
        ref.T_col: T_vals,
        ref.p_col: p_vals,
        ref.tau_col: tau_vals,
        "iso_tau_s": tau_vals,
        "iso_row_in_case": np.arange(n, dtype=int),
        "iso_case_id": np.full(n, case_id, dtype=int),
    }
    if ref.case_col:
        data[ref.case_col] = np.full(n, case_id, dtype=int)
    if ref.sample_kind_col:
        data[ref.sample_kind_col] = np.full(n, "isothermal_enrichment", dtype=object)
    y_cols = ref.mechanism_y_cols or ref.all_y_cols
    for c in y_cols:
        if c in out.columns:
            data[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            data[c] = np.zeros(n, dtype=float)
    c2h6_col = ref.species_cols.get("C2H6", "Y_C2H6")
    if c2h6_col in data:
        y_in = 1.0 / max(1.0 + s2e, EPS)
        conv = np.clip(1.0 - np.asarray(data[c2h6_col], dtype=float) / max(y_in, EPS), 0.0, 1.0)
    else:
        conv = np.full(n, np.nan)
    data["iso_anchor_conversion"] = conv
    return pd.DataFrame(data)


def _run_native_v2_pfr_attempt(case_row: dict[str, Any], ref: ReferenceMap, settings: Any, mech_path: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one PFR-first request through the fast native v3 solver path.

    Important enrichment-policy detail:
    if the native PFR solver fails/drops the case (for example ``truncated at z=...``
    after a DVODE/VODE excess-work failure), that is *not* treated as a failed
    database case here.  It means this sparse point is not robustly reachable with
    the requested PFR realisation, so the caller should replace it by the real-Y
    anchored state-probe fallback.
    """
    template = _native_template_runner(settings, mech_path, seed=seed)
    runner = _patch_native_runner_from_manifest(template, case_row)
    _patch_generation_v3_temperature_cap(settings, float(case_row.get("native_T_cap_K", case_row.get("T_hard_max_K", 1600.0))))
    df, audit = g2.run_case_v2(runner, settings)
    if df is None:
        raise NativePfrAttemptFailure(audit)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    anchors = _metadata_anchors_from_native_pfr(df, ref, case_row)
    return df, anchors, dict(audit or {})


def _set_or_add(df: pd.DataFrame, col: str, values: Any) -> None:
    """Set a dataframe column even if the CRACKSIM evaluator created stale metadata."""
    df[col] = values


def _maybe_set_many(df: pd.DataFrame, candidate_cols: Iterable[str], values: Any, ref: ReferenceMap) -> None:
    """Set any candidate metadata columns that are present in the output or reference schema."""
    ref_cols = set(ref.columns)
    for col in candidate_cols:
        if col in df.columns or col in ref_cols:
            df[col] = values


def _stamp_manifest_metadata(
    out: pd.DataFrame,
    anchors: pd.DataFrame,
    case_row: dict[str, Any],
    ref: ReferenceMap,
    settings: Any,
    n_points_requested: int,
    gas: Any | None = None,
) -> pd.DataFrame:
    """Restore manifest/PFR metadata after CRACKSIM state evaluation.

    The fallback path uses generation_v3.eval_offmanifold_points(..., sigma_log=0) as a
    direct CRACKSIM state evaluator. That function intentionally labels rows as
    offmanifold and may replace CaseID by -1 and inlet/tau metadata by NaN/zero.
    For this isothermal enrichment database that is wrong: every row must remain
    traceable to the manifest row and to the designed residence-time coordinate.

    This function therefore stamps deterministic metadata from the anchors back onto
    the CRACKSIM-evaluated output. It does not alter Y_*, dYdt_*, wdot_* or heat-source
    columns; it only repairs labels and plotting/provenance columns.
    """
    out = out.copy()
    n = len(out)
    case_id = int(case_row.get("case_id", case_row.get("id", 0)))
    kind = str(case_row.get("design_kind", "state_probe"))
    region = str(case_row.get("design_region", "isothermal_enrichment"))
    T = float(case_row.get("T_K"))
    p = float(case_row.get("p_Pa"))
    tau_end = float(case_row.get("tau_end_s", 0.0))
    s2e = float(case_row.get("steam_to_ethane_mass", 0.0))
    target_conversion = float(case_row.get("target_conversion", case_row.get("conversion_proxy", np.nan)))

    flow = None
    if gas is not None:
        try:
            # tau_vals is defined below; initialise with anchors if available for flow calculation.
            _tau_for_flow = anchors["iso_tau_s"].to_numpy(float) if "iso_tau_s" in anchors.columns else np.array([tau_end], dtype=float)
            flow = _compute_flow_metadata(gas, anchors, case_row, _tau_for_flow)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Case {case_id}: failed to compute Cantera flow metadata ({exc}); using manifest estimates/fallbacks.")

    # eval_offmanifold_points with points_per_anchor=1 should preserve row count. If a
    # repository-specific direct evaluator returns a different row count, fall back to a
    # sensible monotone tau grid rather than leaving NaNs.
    if len(anchors) == n and "iso_tau_s" in anchors.columns:
        tau_vals = anchors["iso_tau_s"].to_numpy(float)
        row_vals = anchors.get("iso_row_in_case", pd.Series(np.arange(n))).to_numpy(int)
        conv_vals = anchors.get("iso_anchor_conversion", pd.Series(np.full(n, target_conversion))).to_numpy(float)
    else:
        tau_vals = _tau_grid(tau_end, n) if kind == "isothermal_pfr" else np.full(n, max(tau_end, 1e-12))
        row_vals = np.arange(n, dtype=int)
        conv_vals = np.linspace(0.0, target_conversion, n) if kind == "isothermal_pfr" else np.full(n, target_conversion)

    fresh_c2h6 = 1.0 / max(1.0 + s2e, EPS)
    fresh_h2o = s2e / max(1.0 + s2e, EPS)
    sample_kind = "trajectory" if kind == "isothermal_pfr" else "state_probe"

    # Core identifiers / labels
    _maybe_set_many(out, ["CaseID", "case_id", "id", "Case ID"], np.full(n, case_id, dtype=int), ref)
    _maybe_set_many(out, ["regime"], np.full(n, region, dtype=object), ref)
    _maybe_set_many(out, ["sample_kind", "Sample kind", "sample type"], np.full(n, sample_kind, dtype=object), ref)

    # Residence time / plotting coordinate. This is a physical residence-time coordinate
    # for isothermal_pfr and a design/probe residence-time tag for state_probe rows.
    _maybe_set_many(out, [ref.tau_col, "tau", "tau [s]", "Residence time [s]", "residence_time_s", "time", "t", "t [s]"], tau_vals, ref)
    _maybe_set_many(out, [ref.T_col, "T", "T [K]", "Temperature [K]", "temperature_K"], np.full(n, T), ref)
    _maybe_set_many(out, [ref.p_col, "p", "P", "Pressure [Pa]", "pressure_Pa", "P_Pa", "p_Pa"], np.full(n, p), ref)

    # Common v2 metadata shown by the user's schema inspection. These are normally not
    # used for the source mapping, but leaving them NaN/0/offmanifold makes debugging and
    # coverage plots impossible.
    pfr_points = n_points_requested if kind == "isothermal_pfr" else 1
    _maybe_set_many(out, ["PFR points solved"], np.full(n, pfr_points, dtype=int), ref)
    _maybe_set_many(out, ["Storage policy"], np.full(n, f"isothermal_manifest:n_points={n_points_requested}", dtype=object), ref)
    if flow is None:
        D_fb = _first_finite(case_row, ["diameter_m", "D_m", "diameter [m]"], 0.0306)
        A_fb = math.pi * D_fb * D_fb / 4.0
        mdot_fb = _first_finite(case_row, ["estimated_mdot_kg_s", "mdot_kg_s", "mdot [kg/s]"], 0.0)
        L_fb = _first_finite(case_row, ["estimated_length_m", "length_m", "Length [m]"], np.nan)
        U_fb = _first_finite(case_row, ["estimated_U_m_s", "U_in_m_s"], np.nan)
        Re_fb = _first_finite(case_row, ["target_Re", "Re_target"], np.nan)
        flow = {
            "diameter_m": D_fb, "area_m2": A_fb, "mdot_kg_s": mdot_fb, "length_m": L_fb,
            "U_in_m_s": U_fb, "target_Re": Re_fb, "rho_in_kg_m3": np.nan, "mu_in_Pa_s": np.nan,
            "rho_local_kg_m3": np.full(n, np.nan), "mu_local_Pa_s": np.full(n, np.nan),
            "U_local_m_s": np.full(n, U_fb), "Re_local": np.full(n, Re_fb),
        }
    _maybe_set_many(out, ["mdot [kg/s]", "Mass flow [kg/s]"], np.full(n, float(flow["mdot_kg_s"]), dtype=float), ref)
    _maybe_set_many(out, ["diameter [m]"], np.full(n, float(flow["diameter_m"]), dtype=float), ref)
    _maybe_set_many(out, ["Area [m2]"], np.full(n, float(flow["area_m2"]), dtype=float), ref)
    _maybe_set_many(out, ["Length [m]", "Reactor length [m]"], np.full(n, float(flow["length_m"]), dtype=float), ref)
    _maybe_set_many(out, ["Velocity [m/s]", "U_in [m/s]", "Inlet velocity [m/s]"], np.full(n, float(flow["U_in_m_s"]), dtype=float), ref)
    _maybe_set_many(out, ["Re", "Reynolds number [-]", "Reynolds [-]"], np.asarray(flow["Re_local"], dtype=float), ref)
    _maybe_set_many(out, ["steam_to_ethane [kg/kg]", "steam_to_ethane_mass", "steam/C2H6 [kg/kg]"], np.full(n, s2e, dtype=float), ref)
    _maybe_set_many(out, ["inlet_Y_C2H6 [-]"], np.full(n, fresh_c2h6, dtype=float), ref)
    _maybe_set_many(out, ["inlet_Y_H2O [-]"], np.full(n, fresh_h2o, dtype=float), ref)
    _maybe_set_many(out, ["T_in [K]"], np.full(n, T, dtype=float), ref)
    _maybe_set_many(out, ["P_in [Pa]"], np.full(n, p, dtype=float), ref)
    _maybe_set_many(out, ["shape"], np.full(n, kind, dtype=object), ref)
    _maybe_set_many(out, ["H_peak [W/m2]"], np.zeros(n, dtype=float), ref)
    _maybe_set_many(out, ["solver_rtol"], np.full(n, float(getattr(settings, "solver_rtol", np.nan))), ref)
    _maybe_set_many(out, ["solver_atol"], np.full(n, float(getattr(settings, "solver_atol", np.nan))), ref)
    _maybe_set_many(out, ["generator_version"], np.full(n, SCRIPT_VERSION, dtype=object), ref)

    # Extra traceability columns. Keep them by default; pass --no-keep-extra-columns to
    # write only the reference schema.
    out["iso_case_id"] = np.full(n, case_id, dtype=int)
    out["iso_design_kind"] = np.full(n, kind, dtype=object)
    out["iso_design_region"] = np.full(n, region, dtype=object)
    out["iso_row_in_case"] = row_vals
    out["iso_tau_s"] = tau_vals
    out["iso_tau_end_s"] = np.full(n, tau_end, dtype=float)
    out["iso_target_conversion"] = np.full(n, target_conversion, dtype=float)
    out["iso_anchor_conversion"] = conv_vals
    out["iso_steam_to_ethane_mass"] = np.full(n, s2e, dtype=float)
    out["iso_T_K"] = np.full(n, T, dtype=float)
    out["iso_p_Pa"] = np.full(n, p, dtype=float)
    out["iso_diameter_m"] = np.full(n, float(flow["diameter_m"]), dtype=float)
    out["iso_area_m2"] = np.full(n, float(flow["area_m2"]), dtype=float)
    out["iso_length_m"] = np.full(n, float(flow["length_m"]), dtype=float)
    out["iso_mdot_kg_s"] = np.full(n, float(flow["mdot_kg_s"]), dtype=float)
    out["iso_U_in_m_s"] = np.full(n, float(flow["U_in_m_s"]), dtype=float)
    out["iso_target_Re"] = np.full(n, float(flow["target_Re"]), dtype=float)
    out["iso_rho_in_kg_m3"] = np.full(n, float(flow["rho_in_kg_m3"]), dtype=float)
    out["iso_mu_in_Pa_s"] = np.full(n, float(flow["mu_in_Pa_s"]), dtype=float)
    out["iso_rho_local_kg_m3"] = np.asarray(flow["rho_local_kg_m3"], dtype=float)
    out["iso_mu_local_Pa_s"] = np.asarray(flow["mu_local_Pa_s"], dtype=float)
    out["iso_U_local_m_s"] = np.asarray(flow["U_local_m_s"], dtype=float)
    out["iso_Re_local"] = np.asarray(flow["Re_local"], dtype=float)
    out["iso_flow_relation"] = np.full(n, "tau=L/U; U=Re*mu/(rho*D); mdot=rho*A*U", dtype=object)
    # Preserve explicit T-X/T-log(tau) design provenance from the manifest. These
    # columns are essential for checking that a solved CRACKSIM row still maps back
    # to the sparse bin selected by identify_isothermal_empty_regions.py.
    for key in [
        "target_tau_s", "tau_design_mode", "coverage_round", "coverage_priority",
        "tx_bin_i", "tx_bin_j", "tx_bin_T_low_K", "tx_bin_T_high_K",
        "tx_bin_X_low", "tx_bin_X_high", "ttau_bin_k",
        "ttau_bin_logtau_low", "ttau_bin_logtau_high",
        "ttau_bin_tau_low_s", "ttau_bin_tau_high_s", "ttau_bin_count_before",
        "tau_estimate_s", "log10_tau_s", "log10_tau_estimate_s", "tau_over_tau_estimate",
        "pfr_tau_ratio_min", "pfr_tau_ratio_max", "tau_physicality_flag",
        "state_probe_composition_source", "anchor_pool_id", "anchor_source_file",
        "anchor_conversion_proxy", "anchor_steam_to_ethane_mass", "anchor_T_K", "anchor_p_Pa", "anchor_tau_s",
        "state_probe_composition_source", "anchor_pool_id", "anchor_source_file",
        "anchor_conversion_proxy", "anchor_steam_to_ethane_mass", "anchor_T_K", "anchor_p_Pa", "anchor_tau_s",
        "hydro_design_mode", "estimated_length_m", "estimated_U_m_s", "estimated_mdot_kg_s",
        "estimated_rho_kg_m3", "estimated_mu_Pa_s",
    ]:
        if key in case_row:
            out[f"iso_manifest_{key}"] = np.full(n, case_row.get(key), dtype=object)
    out["iso_manifest_version"] = np.full(n, SCRIPT_VERSION, dtype=object)
    return out


# -----------------------------
# Output schema alignment / audits
# -----------------------------


# Fixed extra columns for v2-style scratch files.
#
# The original v2 generator can merge by simply streaming case_*.parquet files because
# every worker writes the same schema for every case.  We keep that contract here: each
# isothermal scratch parquet is first aligned to [reference schema + canonical iso_* schema]
# and written with an explicit Arrow schema.  The merge step can then use the same simple
# ParquetWriter pattern as generate_database_v2.py, without unioning columns at merge time.

ISO_EXTRA_FLOAT_COLUMNS = [
    "iso_tau_s", "iso_tau_end_s", "iso_target_conversion", "iso_anchor_conversion",
    "iso_steam_to_ethane_mass", "iso_T_K", "iso_p_Pa", "iso_diameter_m", "iso_area_m2",
    "iso_length_m", "iso_mdot_kg_s", "iso_U_in_m_s", "iso_target_Re", "iso_rho_in_kg_m3",
    "iso_mu_in_Pa_s", "iso_rho_local_kg_m3", "iso_mu_local_Pa_s", "iso_U_local_m_s",
    "iso_Re_local",
    "iso_native_z_final_m", "iso_native_L_requested_m",
    "iso_manifest_target_tau_s", "iso_manifest_coverage_priority",
    "iso_manifest_tx_bin_T_low_K", "iso_manifest_tx_bin_T_high_K",
    "iso_manifest_tx_bin_X_low", "iso_manifest_tx_bin_X_high",
    "iso_manifest_ttau_bin_logtau_low", "iso_manifest_ttau_bin_logtau_high",
    "iso_manifest_ttau_bin_tau_low_s", "iso_manifest_ttau_bin_tau_high_s",
    "iso_manifest_ttau_bin_count_before",
    # Kept for backward compatibility with earlier manifests. These are no longer used
    # for the PFR-first/no-Arrhenius decision, but old manifest CSVs may still contain them.
    "iso_manifest_tau_estimate_s", "iso_manifest_log10_tau_s", "iso_manifest_log10_tau_estimate_s",
    "iso_manifest_tau_over_tau_estimate", "iso_manifest_pfr_tau_ratio_min", "iso_manifest_pfr_tau_ratio_max",
    "iso_manifest_anchor_conversion_proxy", "iso_manifest_anchor_steam_to_ethane_mass",
    "iso_manifest_anchor_T_K", "iso_manifest_anchor_p_Pa", "iso_manifest_anchor_tau_s",
    "iso_manifest_estimated_length_m", "iso_manifest_estimated_U_m_s", "iso_manifest_estimated_mdot_kg_s",
    "iso_manifest_estimated_rho_kg_m3", "iso_manifest_estimated_mu_Pa_s",
    # Strict-anchor v8/v9+ diagnostics, if present.
    "iso_manifest_anchor_abs_steam_diff", "iso_manifest_anchor_abs_conversion_diff",
    "iso_manifest_anchor_abs_T_diff_K", "iso_manifest_anchor_abs_p_diff_Pa",
    "iso_manifest_anchor_distance", "iso_manifest_anchor_score",
]

ISO_EXTRA_INT_COLUMNS = [
    "iso_case_id", "iso_row_in_case", "iso_manifest_coverage_round",
    "iso_manifest_tx_bin_i", "iso_manifest_tx_bin_j", "iso_manifest_ttau_bin_k",
]

ISO_EXTRA_BOOL_COLUMNS = [
    "iso_pfr_hit_target", "iso_native_truncated_before_L",
]

ISO_EXTRA_STRING_COLUMNS = [
    "iso_design_kind", "iso_design_region", "iso_flow_relation",
    "iso_manifest_tau_design_mode", "iso_manifest_tau_physicality_flag",
    "iso_manifest_state_probe_composition_source", "iso_manifest_anchor_pool_id",
    "iso_manifest_anchor_source_file", "iso_manifest_hydro_design_mode",
    "iso_manifest_version", "iso_final_design_kind", "iso_fallback_probe_status",
    "iso_native_truncation_reason",
    # Strict-anchor v8/v9+ diagnostics, if present.
    "iso_manifest_anchor_quality_flag", "iso_manifest_anchor_match_status",
]

CANONICAL_ISO_EXTRA_COLUMNS = (
    ISO_EXTRA_INT_COLUMNS + ISO_EXTRA_FLOAT_COLUMNS + ISO_EXTRA_BOOL_COLUMNS + ISO_EXTRA_STRING_COLUMNS
)


def _missing_float_series(n: int) -> pd.Series:
    return pd.Series(np.full(n, np.nan, dtype=float), dtype="float64")


def _missing_int_series(n: int) -> pd.Series:
    return pd.Series([pd.NA] * n, dtype="Int64")


def _missing_bool_series(n: int) -> pd.Series:
    return pd.Series([pd.NA] * n, dtype="boolean")


def _missing_string_series(n: int) -> pd.Series:
    return pd.Series([pd.NA] * n, dtype="string")


def _coerce_extra_column(df: pd.DataFrame, col: str) -> None:
    if col in ISO_EXTRA_FLOAT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    elif col in ISO_EXTRA_INT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    elif col in ISO_EXTRA_BOOL_COLUMNS:
        # Preserve nullable bool.  Strings such as "True"/"False" are handled explicitly.
        s = df[col]
        if s.dtype == object or str(s.dtype).startswith("string"):
            low = s.astype("string").str.lower()
            mapped = low.map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
            df[col] = mapped.astype("boolean")
        else:
            df[col] = s.astype("boolean")
    else:
        df[col] = df[col].astype("string")


def _add_missing_extra_column(df: pd.DataFrame, col: str, n: int) -> None:
    if col in ISO_EXTRA_FLOAT_COLUMNS:
        df[col] = _missing_float_series(n)
    elif col in ISO_EXTRA_INT_COLUMNS:
        df[col] = _missing_int_series(n)
    elif col in ISO_EXTRA_BOOL_COLUMNS:
        df[col] = _missing_bool_series(n)
    else:
        df[col] = _missing_string_series(n)


def case_output_columns(ref: ReferenceMap, keep_extra: bool = True) -> list[str]:
    if not keep_extra:
        return list(ref.columns)
    extras = [c for c in CANONICAL_ISO_EXTRA_COLUMNS if c not in ref.columns]
    return list(ref.columns) + extras


def _arrow_type_for_extra(col: str) -> pa.DataType:
    if col in ISO_EXTRA_FLOAT_COLUMNS:
        return pa.float64()
    if col in ISO_EXTRA_INT_COLUMNS:
        return pa.int64()
    if col in ISO_EXTRA_BOOL_COLUMNS:
        return pa.bool_()
    return pa.string()


def case_output_schema(ref: ReferenceMap, keep_extra: bool = True) -> pa.Schema:
    fields: list[pa.Field] = []
    for c in ref.columns:
        fields.append(pa.field(c, _arrow_type_from_ref_dtype(ref.dtypes.get(c))))
    if keep_extra:
        for c in CANONICAL_ISO_EXTRA_COLUMNS:
            if c not in ref.columns:
                fields.append(pa.field(c, _arrow_type_for_extra(c)))
    return pa.schema(fields)


def align_schema(df: pd.DataFrame, ref: ReferenceMap, keep_extra: bool = True) -> pd.DataFrame:
    # Add missing reference columns as NaN so downstream code sees exactly the reference columns available.
    # Reference columns are written using the reference Arrow schema in write_table_atomic(..., schema=...).
    n = len(df)
    for c in ref.columns:
        if c not in df.columns:
            df[c] = np.nan

    if keep_extra:
        for c in CANONICAL_ISO_EXTRA_COLUMNS:
            if c not in df.columns:
                _add_missing_extra_column(df, c, n)
            else:
                _coerce_extra_column(df, c)
        return df[case_output_columns(ref, keep_extra=True)]

    return df[list(ref.columns)]


def source_columns_from_names(columns: Iterable[str]) -> list[str]:
    """Return source/rate/heat columns from column names, without reading a truncated sample.

    Earlier versions inspected only the first ~200 columns of the parquet sample. With
    a 213-species mechanism, the first 200 columns can be mostly Y_* columns, so the
    diagnostic could incorrectly warn that no dYdt/heat columns were present even when
    they existed later in the schema.
    """
    out: list[str] = []
    heat_norm = {_norm(c) for c in HEAT_CANDIDATES}
    for c in columns:
        cs = str(c)
        ncs = _norm(cs)
        if (
            cs.startswith("dYdt_")
            or cs.startswith("wdot_")
            or cs.startswith("NetRate_")
            or cs.startswith("net_rate_")
            or ncs in heat_norm
            or ("reactionheat" in ncs and ("absorption" in ncs or "source" in ncs))
        ):
            out.append(cs)
    return out


def source_columns(df: pd.DataFrame) -> list[str]:
    return source_columns_from_names(df.columns)


def write_table_atomic(df: pd.DataFrame, path: Path, schema: pa.Schema | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if schema is None:
        table = pa.Table.from_pandas(df, preserve_index=False)
    else:
        # Explicit schema is the important difference versus the failed union-merge version:
        # every case_*.parquet receives identical field names and Arrow types up front,
        # so merge-only can remain the simple v2 streaming concatenation.
        table = pa.Table.from_pandas(df, preserve_index=False, schema=schema, safe=False)
    pq.write_table(table, str(tmp), compression="snappy")
    tmp.replace(path)


def _arrow_type_from_ref_dtype(dtype: str | None) -> pa.DataType:
    """Map the reference parquet dtype string to a stable Arrow type.

    This is intentionally conservative.  Reference columns should keep their original
    broad type.  Extra metadata columns are inferred from the scratch schemas.
    """
    if not dtype:
        return pa.float64()
    d = str(dtype).lower()
    if "string" in d or "utf8" in d or "large_string" in d:
        return pa.string()
    if "bool" in d:
        return pa.bool_()
    if "int" in d:
        return pa.int64()
    if "float" in d or "double" in d or "decimal" in d:
        return pa.float64()
    return pa.string()


def _choose_arrow_type(col: str, types: list[pa.DataType], ref: ReferenceMap) -> pa.DataType:
    """Choose a single robust Arrow type for a column across all case files."""
    if col in ref.dtypes:
        return _arrow_type_from_ref_dtype(ref.dtypes.get(col))

    nn = [t for t in types if not pa.types.is_null(t)]
    if not nn:
        return pa.float64()
    if any(pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t) for t in nn):
        return pa.string()
    if any(pa.types.is_floating(t) or pa.types.is_decimal(t) for t in nn):
        return pa.float64()
    if any(pa.types.is_integer(t) for t in nn):
        return pa.int64()
    if any(pa.types.is_boolean(t) for t in nn):
        return pa.bool_()
    return pa.string()


def _series_to_arrow_array(series: pd.Series, typ: pa.DataType) -> pa.Array:
    """Convert a pandas series to a nullable Arrow array with the requested type."""
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        obj = series.astype("object")
        vals = [None if pd.isna(v) else str(v) for v in obj.to_numpy(dtype=object)]
        return pa.array(vals, type=typ)
    if pa.types.is_boolean(typ):
        obj = series.astype("object")
        vals = [None if pd.isna(v) else bool(v) for v in obj.to_numpy(dtype=object)]
        return pa.array(vals, type=typ)
    if pa.types.is_integer(typ):
        num = pd.to_numeric(series, errors="coerce")
        vals = [None if pd.isna(v) else int(v) for v in num.to_numpy(dtype=object)]
        return pa.array(vals, type=typ)
    if pa.types.is_floating(typ):
        num = pd.to_numeric(series, errors="coerce")
        return pa.array(num.to_numpy(dtype="float64"), type=typ)
    obj = series.astype("object")
    vals = [None if pd.isna(v) else str(v) for v in obj.to_numpy(dtype=object)]
    return pa.array(vals, type=pa.string())


def _build_union_merge_schema(files: list[Path], ref: ReferenceMap, keep_extra_columns: bool) -> tuple[list[str], pa.Schema]:
    """Build one stable schema for all scratch case files before streaming the merge.

    The previous merge used the first case file as writer schema.  That fails when case 0
    has e.g. no fallback probe but case 1 has an extra fallback-status column, or vice versa.
    This function first scans all scratch schemas, creates the union of extra columns, and
    then every case file is written with the same field names and compatible types.
    """
    seen: list[str] = []
    types_by_col: dict[str, list[pa.DataType]] = {}
    for f in files:
        schema = pq.read_schema(str(f))
        for field in schema:
            c = field.name
            if (c in ref.columns) or keep_extra_columns:
                if c not in types_by_col:
                    types_by_col[c] = []
                types_by_col[c].append(field.type)
                if c not in seen:
                    seen.append(c)

    if keep_extra_columns:
        extra_cols = [c for c in seen if c not in ref.columns]
        all_cols = list(ref.columns) + extra_cols
    else:
        all_cols = list(ref.columns)

    fields = [pa.field(c, _choose_arrow_type(c, types_by_col.get(c, []), ref)) for c in all_cols]
    return all_cols, pa.schema(fields)


def _dataframe_to_schema_table(df: pd.DataFrame, all_cols: list[str], schema: pa.Schema) -> pa.Table:
    """Return an Arrow table with exactly all_cols/schema, adding nullable missing columns."""
    arrays: list[pa.Array] = []
    n = len(df)
    for field in schema:
        c = field.name
        if c in df.columns:
            arrays.append(_series_to_arrow_array(df[c], field.type))
        else:
            arrays.append(pa.nulls(n, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def merge_case_files(out_root: Path, out_name: str, ref: ReferenceMap, keep_extra_columns: bool) -> Path:
    """Merge per-case parquets using the same simple pattern as generate_database_v2.py.

    This intentionally does NOT build a union schema at merge time.  The contract is that
    each worker already wrote every scratch file with the canonical case schema.  If this
    function detects an older inconsistent scratch file, delete that scratch folder and rerun
    the cases with this fixed-schema generator.
    """
    scratch = out_root / "scratch"
    files = sorted(scratch.glob("case_*.parquet"), key=lambda p: int(re.findall(r"case_(\d+)", p.stem)[0]))
    if not files:
        raise RuntimeError(f"No case_*.parquet files found in {scratch}")

    out_file = out_root / f"{out_name}.parquet"
    writer = None
    writer_schema = None
    n_rows = 0
    n_files = 0
    try:
        for p in files:
            t = pq.read_table(str(p))
            if writer is None:
                writer_schema = t.schema
                writer = pq.ParquetWriter(str(out_file), writer_schema, compression="snappy")
            else:
                if t.schema != writer_schema:
                    # Fail loudly.  This mirrors v2's assumption of same-schema scratch files,
                    # but gives a clearer explanation than pyarrow's raw field-name error.
                    first_names = list(writer_schema.names)
                    this_names = list(t.schema.names)
                    missing = [c for c in first_names if c not in this_names][:20]
                    extra = [c for c in this_names if c not in first_names][:20]
                    raise RuntimeError(
                        f"Scratch schema mismatch in {p}. This usually means some case files were written by an older generator. "
                        f"Delete {scratch} and rerun, or rerun only the affected cases. Missing vs first: {missing}; extra vs first: {extra}"
                    )
            writer.write_table(t)
            n_rows += t.num_rows
            n_files += 1
    finally:
        if writer is not None:
            writer.close()
    print(f"[merge] {n_files} case files -> {out_file} ({n_rows} rows)", flush=True)
    return out_file


def run_schema_report(out_file: Path, reference_path: Path | None, out_root: Path) -> None:
    report: dict[str, Any] = {"output": str(out_file), "script_version": SCRIPT_VERSION}
    out_schema = pq.read_schema(str(out_file))
    out_cols = list(out_schema.names)
    report["n_output_columns"] = len(out_cols)

    ref_cols: list[str] = []
    ref_source_cols: list[str] = []
    if reference_path and reference_path.exists():
        ref_schema = pq.read_schema(str(reference_path))
        ref_cols = list(ref_schema.names)
        ref_source_cols = source_columns_from_names(ref_cols)
        report["reference"] = str(reference_path)
        report["n_reference_columns"] = len(ref_cols)
        report["missing_reference_columns"] = [c for c in ref_cols if c not in out_cols]
        report["extra_columns"] = [c for c in out_cols if c not in ref_cols]
        report["reference_source_like_columns"] = ref_source_cols[:50]
        report["n_reference_source_like_columns"] = len(ref_source_cols)

    out_source_cols = source_columns_from_names(out_cols)
    report["source_like_columns"] = out_source_cols[:50]
    report["n_source_like_columns"] = len(out_source_cols)
    report["has_source_like_columns"] = bool(out_source_cols)

    # If a reference has CRACKSIM source columns, verify that the output actually has them too.
    missing_source_cols = [c for c in ref_source_cols if c not in out_cols]
    report["missing_reference_source_like_columns"] = missing_source_cols[:100]
    report["n_missing_reference_source_like_columns"] = len(missing_source_cols)

    # Read a tiny table only for numerical sanity checks of a few detected source columns.
    finite_report: dict[str, Any] = {}
    cols_to_check = out_source_cols[: min(len(out_source_cols), 20)]
    if cols_to_check:
        try:
            sample = pq.read_table(str(out_file), columns=cols_to_check).to_pandas()
            for c in cols_to_check:
                arr = pd.to_numeric(sample[c], errors="coerce").to_numpy(float)
                finite_report[c] = {
                    "finite_fraction": float(np.isfinite(arr).mean()) if arr.size else 0.0,
                    "nonzero_fraction": float((np.nan_to_num(arr) != 0.0).mean()) if arr.size else 0.0,
                }
        except Exception as exc:  # noqa: BLE001
            finite_report["error"] = str(exc)
    report["source_column_value_check"] = finite_report

    (out_root / "isothermal_schema_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[schema] wrote {out_root / 'isothermal_schema_report.json'}", flush=True)
    print(f"[schema] detected {len(out_source_cols)} source-like columns in output schema", flush=True)
    if ref_source_cols:
        print(f"[schema] reference has {len(ref_source_cols)} source-like columns; missing in output: {len(missing_source_cols)}", flush=True)
    if not out_source_cols:
        warnings.warn(
            "No dYdt_*/wdot_*/Reaction heat absorption columns detected in the OUTPUT SCHEMA. "
            "This means CRACKSIM source-term evaluation likely did not run as expected. "
            "If the reference parquet also has no such columns, check the exact column names in out_v2/full.parquet."
        )


# -----------------------------
# Worker loop
# -----------------------------


def worker_loop(
    worker_id: int,
    task_q: Queue,
    ready_q: Queue,
    status_q: Queue,
    dll_path: str,
    mech_path: str,
    base_dir: str,
    scratch_root: str,
    settings_doc: dict[str, Any],
    ref_doc: dict[str, Any],
    n_points: int,
    keep_extra_columns: bool,
    print_cases: bool,
) -> None:
    try:
        settings = settings_from_doc(settings_doc)
        _patch_generation_v3_temperature_cap(settings, float(settings_doc.get("native_T_cap_K", settings_doc.get("T_hard_max_K", 1600.0))))
        ref = ReferenceMap(**ref_doc)
        fixed_case_schema = case_output_schema(ref, keep_extra=keep_extra_columns)
        g2.init_worker_cracksim(dll_path, mech_path, Path(base_dir), Path(scratch_root), ready_q)
        try:
            import cantera as ct  # type: ignore
            gas = ct.Solution(mech_path)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"[w{worker_id:02d}] Could not create Cantera gas for flow metadata: {exc}")
            gas = None
        scratch = Path(scratch_root)
        solved = 0
        while True:
            item = task_q.get()
            if item is None:
                break
            case_id = int(item.get("case_id", item.get("id", solved)))
            t0 = time.monotonic()
            try:
                kind = str(item.get("design_kind", "state_probe"))
                if kind == "pfr_first_request":
                    # 1) Fast native v3 PFR attempt.  If the native PFR is numerically
                    # unstable (e.g. DVODE excess work / truncated before L), we do NOT
                    # drop the database case.  For this enrichment workflow that means:
                    # the requested fresh-feed PFR realization is not robustly reachable,
                    # so replace this request by the pre-selected real-Y anchored state
                    # probe.  This avoids double PFR computation and avoids storing a
                    # misleading partial trajectory.
                    parts: list[pd.DataFrame] = []
                    native_failed = False
                    native_failure_reason = ""
                    hit = False
                    try:
                        pfr_eval, pfr_anchors, _native_audit = _run_native_v2_pfr_attempt(
                            item, ref, settings, mech_path=mech_path,
                            seed=int(20260704 + worker_id * 1000003 + case_id),
                        )
                        pfr_item = dict(item)
                        pfr_item["design_kind"] = "isothermal_pfr"
                        out_pfr = _stamp_manifest_metadata(pfr_eval, pfr_anchors, pfr_item, ref, settings, n_points_requested=len(pfr_anchors), gas=gas)
                        native_truncated = bool(_native_audit.get("truncated_before_L", False))
                        out_pfr["iso_native_truncated_before_L"] = native_truncated
                        out_pfr["iso_native_truncation_reason"] = str(_native_audit.get("truncation_reason", ""))
                        out_pfr["iso_native_z_final_m"] = float(_native_audit.get("z_final_m", np.nan))
                        out_pfr["iso_native_L_requested_m"] = float(_native_audit.get("L_requested_m", np.nan))
                        hit = _trajectory_hits_target_bin(out_pfr, ref, item)
                        out_pfr["iso_pfr_hit_target"] = bool(hit)
                        out_pfr["iso_final_design_kind"] = "isothermal_pfr_hit_target" if hit else "isothermal_pfr_missed_target"
                        parts.append(out_pfr)
                    except NativePfrAttemptFailure as exc:
                        native_failed = True
                        native_failure_reason = str(exc.reason)
                        hit = False
                        # Intentionally no PFR rows: the PFR attempt failed before producing
                        # a trustworthy trajectory.  The fallback state probe below becomes
                        # the replacement data for this sparse target.

                    # 2) If the PFR missed OR failed, append one anchored probe.
                    if not hit:
                        fb_anchors = _build_fallback_state_probe_anchors(item, ref)
                        if fb_anchors is not None:
                            fb_eval = _eval_cracksim_states(fb_anchors, settings, seed=int(909090 + worker_id * 1000003 + case_id))
                            fb_item = dict(item)
                            fb_item["design_kind"] = "state_probe"
                            fb_item["design_region"] = str(item.get("design_region", "isothermal_enrichment")) + ("__fallback_after_native_pfr_failure" if native_failed else "__fallback_anchor")
                            out_fb = _stamp_manifest_metadata(fb_eval, fb_anchors, fb_item, ref, settings, n_points_requested=1, gas=gas)
                            out_fb["iso_pfr_hit_target"] = False
                            out_fb["iso_native_truncated_before_L"] = bool(native_failed)
                            out_fb["iso_native_truncation_reason"] = native_failure_reason if native_failed else ""
                            out_fb["iso_native_z_final_m"] = np.nan
                            out_fb["iso_native_L_requested_m"] = float(item.get("estimated_length_m", item.get("length_m", np.nan)))
                            out_fb["iso_final_design_kind"] = "anchored_state_probe_after_native_pfr_failure" if native_failed else "anchored_state_probe_after_pfr_miss"
                            out_fb["iso_fallback_probe_status"] = "used_after_native_pfr_failure" if native_failed else "used_after_pfr_miss"
                            parts.append(out_fb)
                        elif native_failed:
                            raise RuntimeError(f"native PFR failed ({native_failure_reason}) and no suitable real-Y fallback anchor was available")
                        else:
                            # Keep the missed PFR if it was solved, but flag that the target was not filled.
                            if parts:
                                parts[0]["iso_fallback_probe_status"] = "no_suitable_anchor_available"
                            else:
                                raise RuntimeError("PFR missed target and no suitable real-Y fallback anchor was available")
                    out = pd.concat(parts, ignore_index=True)
                else:
                    anchors = _build_case_anchors(item, ref, n_points=n_points)
                    out = _eval_cracksim_states(anchors, settings, seed=int(20260704 + worker_id * 1000003 + case_id))
                    out = _stamp_manifest_metadata(out, anchors, item, ref, settings, n_points_requested=n_points, gas=gas)
                    out["iso_pfr_hit_target"] = np.nan
                    out["iso_final_design_kind"] = kind
                out = align_schema(out, ref, keep_extra=keep_extra_columns)
                out_file = scratch / f"case_{case_id}.parquet"
                write_table_atomic(out, out_file, schema=fixed_case_schema)
                dt = time.monotonic() - t0
                solved += 1
                status_q.put(("done", worker_id, case_id, len(out), dt, None))
                if print_cases:
                    print(
                        f"[w{worker_id:02d}] DONE case={case_id} kind={item.get('design_kind')} "
                        f"region={item.get('design_region')} rows={len(out)} T={float(item.get('T_K')):.1f}K "
                        f"p={float(item.get('p_Pa'))/1e5:.2f}bar tau={float(item.get('tau_end_s', 0.0)):.3e}s "
                        f"steam/C2H6={float(item.get('steam_to_ethane_mass', 0.0)):.3f} "
                        f"D={float(item.get('diameter_m', 0.0306)):.4f}m Re={float(item.get('target_Re', float('nan'))):.2e} {dt:.1f}s",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                dt = time.monotonic() - t0
                status_q.put(("drop", worker_id, case_id, 0, dt, str(exc)))
                warnings.warn(f"[w{worker_id:02d}] DROP case={case_id}: {exc}")
    except Exception as exc:  # noqa: BLE001
        ready_q.put(f"ERROR: {exc}")
    finally:
        status_q.put(("exit", worker_id, None, None, None, None))


# -----------------------------
# Main run modes
# -----------------------------


def load_manifest(args: argparse.Namespace) -> pd.DataFrame:
    m = pd.read_csv(args.manifest)
    if args.limit_cases is not None:
        m = m.head(int(args.limit_cases)).copy()
    if "case_id" not in m.columns:
        m.insert(0, "case_id", np.arange(len(m), dtype=int))
    if "design_kind" not in m.columns:
        m["design_kind"] = "state_probe"
    required = ["T_K", "p_Pa", "tau_end_s", "steam_to_ethane_mass"]
    missing = [c for c in required if c not in m.columns]
    if missing:
        raise SystemExit(f"Manifest is missing required columns: {missing}")
    if "diameter_m" not in m.columns:
        m["diameter_m"] = float(args.default_diameter_m)
    if "target_Re" not in m.columns:
        m["target_Re"] = float(args.default_target_Re)
    if "area_m2" not in m.columns:
        m["area_m2"] = math.pi * m["diameter_m"].astype(float) ** 2 / 4.0
    # Only enforce the hard campaign cap. Do not drop cases above 1400 K.
    if m["T_K"].max() > args.T_hard_max_K + 1e-9:
        raise SystemExit(f"Manifest contains T_K > {args.T_hard_max_K} K. Refusing to run.")
    if m["p_Pa"].min() < args.pressure_min_Pa - 1e-9 or m["p_Pa"].max() > args.pressure_max_Pa + 1e-9:
        raise SystemExit(
            f"Manifest pressure range {m['p_Pa'].min():g}--{m['p_Pa'].max():g} Pa is outside "
            f"allowed range {args.pressure_min_Pa:g}--{args.pressure_max_Pa:g} Pa."
        )
    if m["steam_to_ethane_mass"].min() < args.steam_ethane_min_mass - 1e-12 or m["steam_to_ethane_mass"].max() > args.steam_ethane_max_mass + 1e-12:
        raise SystemExit(
            f"Manifest steam_to_ethane_mass range {m['steam_to_ethane_mass'].min():g}--{m['steam_to_ethane_mass'].max():g} "
            f"is outside allowed range {args.steam_ethane_min_mass:g}--{args.steam_ethane_max_mass:g}."
        )
    return m


def run_generate(args: argparse.Namespace) -> int:
    base_dir = REPO
    dll_path = base_dir / "SA_CRACKSIM.dll"
    if not dll_path.exists():
        print(f"ERROR: {dll_path} not found — place SA_CRACKSIM.dll in the repo root.", file=sys.stderr)
        return 1
    mech_path = _ensure_mechanism(base_dir)

    out_root = Path(args.out)
    scratch = out_root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(args)
    ref = read_reference_map(args.schema_reference)
    ref.mechanism_y_cols = mechanism_y_columns(mech_path, ref.all_y_cols)
    print(f"[schema] reference Y_* columns detected: {len(ref.all_y_cols)}", flush=True)
    print(f"[schema] mechanism Y_* columns used for CRACKSIM anchors: {len(ref.mechanism_y_cols or [])}", flush=True)
    if len(ref.all_y_cols) <= len(SPECIES):
        warnings.warn(
            f"Only {len(ref.all_y_cols)} Y_* columns were detected from the schema reference. "
            "That may still be too few for the full mechanism. Check --schema-reference."
        )
    ref_doc = asdict(ref)

    if args.skip_existing:
        existing = {int(re.findall(r"case_(\d+)", p.stem)[0]) for p in scratch.glob("case_*.parquet") if re.findall(r"case_(\d+)", p.stem)}
        before = len(manifest)
        manifest = manifest[~manifest["case_id"].astype(int).isin(existing)].copy()
        print(f"[resume] {len(existing)} existing case files, {len(manifest)}/{before} cases left to solve", flush=True)

    # Write a frozen copy of the manifest actually used.
    out_root.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_root / "isothermal_manifest_used.csv", index=False)
    (out_root / "isothermal_run_settings.json").write_text(
        json.dumps({"script_version": SCRIPT_VERSION, "args": vars(args), "n_cases_to_solve": len(manifest)}, indent=2, default=str),
        encoding="utf-8",
    )

    if manifest.empty:
        print("[run] No new cases to solve. Use --merge-only to merge existing scratch files.", flush=True)
        return 0

    n_workers = max(1, int(args.n_cpu))
    print(f"[run] CRACKSIM isothermal enrichment FAST native-v3 PFR-first: {len(manifest)} cases, {n_workers} workers, n_points={args.n_points}", flush=True)
    print(f"[run] output scratch: {scratch}", flush=True)

    task_q: Queue = Queue()
    status_q: Queue = Queue()
    workers: list[Process] = []
    settings_doc = settings_doc_from_args(args)

    # Same sequential READY handshake as your v2 generator, because CRACKSIM init is stateful.
    for i in range(n_workers):
        ready_q: Queue = Queue(maxsize=1)
        w = Process(
            target=worker_loop,
            args=(
                i,
                task_q,
                ready_q,
                status_q,
                str(dll_path.resolve()),
                mech_path,
                str(base_dir.resolve()),
                str(scratch.resolve()),
                settings_doc,
                ref_doc,
                int(args.n_points),
                bool(args.keep_extra_columns),
                bool(args.print_cases),
            ),
            daemon=False,
        )
        w.start()
        msg = ready_q.get()
        if msg != "READY":
            print(f"ERROR: worker {i} init failed: {msg}", file=sys.stderr)
            for _ in workers:
                task_q.put(None)
            return 1
        print(f"[w{i:02d}] READY", flush=True)
        workers.append(w)

    t_start = time.time()
    for _, row in manifest.iterrows():
        task_q.put(row.to_dict())
    for _ in workers:
        task_q.put(None)

    done = dropped = exited = 0
    drops: list[dict[str, Any]] = []
    while exited < len(workers):
        kind, wid, cid, nrows, dt, payload = status_q.get()
        if kind == "done":
            done += 1
        elif kind == "drop":
            dropped += 1
            drops.append({"worker": wid, "case_id": cid, "seconds": dt, "error": payload})
        elif kind == "exit":
            exited += 1
        completed = done + dropped
        if completed and (completed % max(1, int(args.progress_every)) == 0):
            elapsed = max(time.time() - t_start, 1e-9)
            print(f"[progress] {done} done / {dropped} dropped / {len(manifest)} total ({completed/elapsed*3600:.0f} cases/h)", flush=True)

    for w in workers:
        w.join()

    (out_root / "isothermal_drops.json").write_text(json.dumps(drops, indent=2), encoding="utf-8")
    print(f"[run] finished: {done} done / {dropped} dropped", flush=True)
    if dropped and not args.allow_drops:
        print("ERROR: some cases dropped. Inspect isothermal_drops.json; rerun with --skip-existing after fixing.", file=sys.stderr)
        return 2

    out_file = merge_case_files(out_root, args.out_name, ref, keep_extra_columns=args.keep_extra_columns)
    run_schema_report(out_file, Path(args.schema_reference) if args.schema_reference else None, out_root)
    if args.gates:
        return run_gates_on(out_file, args)
    return 0


def run_merge_only(args: argparse.Namespace) -> int:
    out_root = Path(args.out)
    ref = read_reference_map(args.schema_reference)
    out_file = merge_case_files(out_root, args.out_name, ref, keep_extra_columns=args.keep_extra_columns)
    run_schema_report(out_file, Path(args.schema_reference) if args.schema_reference else None, out_root)
    if args.gates:
        return run_gates_on(out_file, args)
    return 0


def _print_gate_a(a: dict) -> None:
    if "error" in a:
        print(f"  ERROR: {a['error']}")
        return
    print(
        f"  max_rel={a.get('max_rel_diff', float('nan')):.3e}  "
        f"median={a.get('median_rel_diff', float('nan')):.3e}  "
        f"p95={a.get('p95_rel_diff', float('nan')):.3e}  "
        f"on {a.get('n_compared', 0)} entries -> {'PASS' if a.get('passed') else 'FAIL'}"
    )
    if "mass_closure_p95" in a:
        print(f"  stored-dYdt mass closure p95: {a['mass_closure_p95']:.3e}")


def run_gates_on(parquet_path: Path, args: argparse.Namespace) -> int:
    if not hasattr(g2, "gate_front_resolution"):
        print("[gates] generation_v3.gate_front_resolution not available; skipping gates.", flush=True)
        return 0
    base_dir = REPO
    dll_path = base_dir / "SA_CRACKSIM.dll"
    mech_path = _ensure_mechanism(base_dir)
    ready_q: Queue = Queue()
    g2.init_worker_cracksim(str(dll_path.resolve()), mech_path, base_dir, Path(args.out) / "scratch", ready_q)
    msg = ready_q.get()
    if msg != "READY":
        print(f"ERROR: gate init failed: {msg}", file=sys.stderr)
        return 1
    df = pd.read_parquet(parquet_path)
    ok = True
    if hasattr(g2, "gate_dll_consistency"):
        ref = df.sample(min(64, len(df)), random_state=0)
        a = g2.gate_dll_consistency(ref)
        print("GATE A (DLL/dYdt consistency):")
        _print_gate_a(a)
        ok = ok and bool(a.get("passed", False))
    c = g2.gate_front_resolution(df, max_frac_jump=args.max_frac_jump)
    print(
        f"GATE C (front resolution): policy-jump p95 {c.get('p95_jump_frac', float('nan')):.3f} "
        f"<= {c.get('threshold', float('nan')):.3f} -> {'PASS' if c.get('passed') else 'FAIL'}"
    )
    ok = ok and bool(c.get("passed", False))
    print(f"GATES: {'ALL PASS' if ok else 'FAILED'}")
    return 0 if ok else 2


# -----------------------------
# CLI
# -----------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="CRACKSIM-backed isothermal enrichment database generator")
    ap.add_argument("--manifest", required=True, help="CSV produced by build_balanced_isothermal_enrichment.py")
    ap.add_argument("--schema-reference", default="out_v2/full.parquet", help="Reference parquet whose schema/column names must be matched")
    ap.add_argument("--out", default="out_v2_iso", help="Output directory")
    ap.add_argument("--out-name", default="isothermal_enrichment_cracksim", help="Output parquet basename without .parquet")
    ap.add_argument("--n-cpu", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 2))
    ap.add_argument("--n-points", type=int, default=160, help="Requested stored/integration points per PFR-first attempt")
    ap.add_argument("--pfr-ode-steps", type=int, default=None, help="Deprecated/ignored by the fast native-v3 backend. Kept for CLI compatibility with older explicit-PFR scripts.")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-16)
    ap.add_argument("--max-frac-jump", type=float, default=0.03)
    ap.add_argument("--skip-existing", action="store_true", help="Resume from existing scratch/case_*.parquet")
    ap.add_argument("--merge-only", action="store_true", help="Only merge existing scratch/case_*.parquet")
    ap.add_argument("--gates", action="store_true", help="Run available v2 gates after merge")
    ap.add_argument("--limit-cases", type=int, default=None, help="For smoke tests: only solve the first N manifest cases")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--print-cases", action="store_true", help="Print every solved case")
    ap.add_argument("--allow-drops", action="store_true", help="Still merge if some cases dropped")
    ap.add_argument("--keep-extra-columns", dest="keep_extra_columns", action="store_true", default=True, help="Keep iso_* debug columns in addition to reference schema")
    ap.add_argument("--no-keep-extra-columns", dest="keep_extra_columns", action="store_false", help="Write only columns present in --schema-reference")
    # Physical safety bounds requested by user.
    ap.add_argument("--T-hard-max-K", type=float, default=1600.0)
    ap.add_argument("--pressure-min-Pa", type=float, default=150000.0)
    ap.add_argument("--pressure-max-Pa", type=float, default=350000.0)
    ap.add_argument("--steam-ethane-min-mass", type=float, default=0.0)
    ap.add_argument("--steam-ethane-max-mass", type=float, default=1.0)
    ap.add_argument("--default-diameter-m", type=float, default=0.0306, help="Fallback diameter if an older manifest lacks diameter_m")
    ap.add_argument("--default-target-Re", type=float, default=1.0e5, help="Fallback target Reynolds number if an older manifest lacks target_Re")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.merge_only:
        return run_merge_only(args)
    return run_generate(args)


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    raise SystemExit(main())

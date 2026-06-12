"""v2 database generation core — the ideal-DB spec (front-adaptive + off-manifold + gates).

Implements the data plan approved on 2026-06-12: front-adaptive trajectory storage, the
off-manifold single-point state cloud, regime enrichment (near-inlet, high-T, tail,
deep-conversion/aromatics), a tier system (smoke ⊂ pilot ⊂ full as prefixes of the SAME
Sobol streams, so a pilot training run proves the pipeline and the full run extends it
without waste), and pre-production verification gates.

Execution targets Windows (where ``SA_CRACKSIM.dll`` loads); everything that does not
need Cantera/the DLL is pure NumPy and unit-tested on any platform. Cantera and ctypes
are imported lazily inside the functions that need them.

Provenance: the proven execution pieces — heat-flux builders, CRACKSIM DLL callback and
worker initialisation, raw-rate/energy unit conversions, the per-case PFR flow — are
ported from the colleague's ``Database_Generation_MB_NEW_ethane_sobol_stride.py`` (the
generator that produced stride5), with these deliberate changes:

- storage: ``select_storage_indices`` (front-adaptive on |ΔS_E|) replaces every-Nth;
- solve grid raised (default 400 points) so the front has points to select;
- columns: ``Y_*`` + ``dYdt_* [1/s]`` + state/meta only — the redundant ``R_*`` (raw DLL
  units), ``wdot_*`` and ``D_*`` families are dropped (~640 columns), the misleading
  ``S Energy`` column is dropped (`Reaction heat absorption [J/s/m3]` is the one truth),
  and the ``Y_*_in`` pseudo-species trap is renamed to ``inlet_Y_* [-]``;
- per-case sign audit + solver tolerances recorded in-row;
"""

from __future__ import annotations

import math
import os
import tempfile
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .config import DataGenConfig, StorageConfig
from .generate import select_storage_indices, sign_audit
from .sampling import build_cases

#: CRACKSIM ``NetRates_C`` output units (per the colleague's generator).
RATES_RAW_UNITS = "kmol/m3/s"

#: Mechanism validity cap confirmed by the user (2026-06-12): states above are dropped.
T_MAX_K_V2 = 1423.15

# Module-level worker state (set by init_worker_cracksim; one process = one DLL handle).
fortlib = None
_gas_cache = None
REAC_MECH_PATH: str | None = None


# ---------------------------------------------------------------------------
# Heat-flux profile builders (ported)
# ---------------------------------------------------------------------------

def _interp_func(x, y):
    """Return a pure-Python callable f(z) = interp(x, y)(z)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def f(z):
        return float(np.interp(z, x, y))

    return f


def make_piecewise(z, q):
    """Normalise inputs and return a callable non-negative wall heat-flux function."""
    z = np.asarray(z, dtype=float)
    q = np.maximum(0.0, np.asarray(q, dtype=float))  # no wall cooling in this database
    return _interp_func(z, q)


def hf_uniform(L, H):
    z = np.array([0.0, L], dtype=float)
    q = np.full_like(z, H, dtype=float)
    return z, q, {"shape": "uniform", "H": float(H)}


def hf_pulsed(L, H, N, Np_req=None, mode="uniform", jitter=0.35,
              seed=None, w_samples=2, snap_to_grid=False):
    """Non-negative heating pulses along the reactor (ported verbatim; localized heating).

    N = number of pulses at peak flux H; widths ≈ ``w_samples`` grid cells when Np_req given.
    """
    N = int(max(1, N))
    dz = (L / max(Np_req - 1, 1)) if (Np_req and Np_req > 1) else None

    base_w = 0.02 * L / max(N, 1)
    w = max(base_w, (w_samples * dz) if dz else base_w)
    margin = 0.5 * w
    usable = max(L - 2.0 * margin, 1e-9)
    s_nom = usable / max(N - 1, 1)

    if mode == "uniform" or N == 1:
        centres = margin + np.arange(N) * s_nom
    elif mode == "jitter":
        rng = np.random.default_rng(seed)
        centres = margin + np.arange(N) * s_nom
        centres = centres + rng.uniform(-jitter * s_nom, jitter * s_nom, size=N)
        min_gap = 0.60 * w
        centres = np.sort(np.clip(centres, margin, L - margin))
        for i in range(1, N):
            if centres[i] - centres[i - 1] < min_gap:
                centres[i] = centres[i - 1] + min_gap
        overflow = centres[-1] - (L - margin)
        if overflow > 0:
            centres -= overflow
            under = margin - centres[0]
            if under > 0:
                centres += under
        for i in range(1, N):
            if centres[i] - centres[i - 1] < min_gap:
                centres[i] = centres[i - 1] + min_gap
        centres = np.clip(centres, margin, L - margin)
    else:
        raise ValueError(f"hf_pulsed: unknown mode '{mode}'")

    if snap_to_grid and dz:
        centres = np.round(centres / dz) * dz
        centres = np.clip(centres, margin, L - margin)
        for i in range(1, N):
            if centres[i] <= centres[i - 1]:
                centres[i] = min(L - margin, centres[i - 1] + max(0.60 * w, dz))

    z = [0.0]
    q = [0.0]
    half_w = 0.5 * w
    for c in centres:
        z.extend([max(0.0, c - half_w), c, min(L, c + half_w)])
        q.extend([0.0, H, 0.0])
    z = np.array(z, dtype=float)
    q = np.maximum(np.array(q, dtype=float), 0.0)
    idx = np.argsort(z)
    return z[idx], q[idx], {"shape": "pulsed", "H": float(H), "N": N, "w": float(w),
                            "mode": str(mode)}


def hf_front_ramp(L, H, k=3.0, Np_req=None, samples_per_cell=4):
    ncp = max(200, (samples_per_cell * (Np_req or 0)) or 0)
    z = np.linspace(0.0, L, ncp)
    q = H * np.exp(-k * (z / L))
    return z, q, {"shape": "front_ramp", "H": float(H), "k": float(k)}


def hf_back_ramp(L, H, k=3.0, Np_req=None, samples_per_cell=4):
    ncp = max(200, (samples_per_cell * (Np_req or 0)) or 0)
    z = np.linspace(0.0, L, ncp)
    q = H * np.exp(-k * (1.0 - z / L))
    return z, q, {"shape": "back_ramp", "H": float(H), "k": float(k)}


def hf_triangular(L, H, peak_s=0.5):
    z = np.array([0.0, float(peak_s) * L, L], dtype=float)
    q = np.array([0.0, H, 0.0], dtype=float)
    return z, q, {"shape": "triangular", "H": float(H), "peak_s": float(peak_s)}


def hf_gaussian_pair(L, H, w_frac=0.12, c1=0.3, c2=0.7, normalise_peak=False):
    z = np.linspace(0.0, L, 400)
    w = max(1e-6, w_frac * L)
    q = (np.exp(-0.5 * ((z - c1 * L) / w) ** 2) + np.exp(-0.5 * ((z - c2 * L) / w) ** 2))
    if normalise_peak and q.max() > 0:
        q = q / q.max()
    q = H * q
    return z, q, {"shape": "gaussian_pair", "H": float(H), "w_frac": float(w_frac),
                  "c1": float(c1), "c2": float(c2), "normalise_peak": bool(normalise_peak)}


def hf_sinusoidal(L, H, cycles=1, mode="offset", Np_req=None, samples_per_cell=6, phase=0.0):
    cycles = max(1, int(cycles))
    n = int(max(10, samples_per_cell * (Np_req - 1) + 1)) if (Np_req and Np_req > 1) else 400
    z = np.linspace(0.0, L, n)
    s = np.sin(2.0 * np.pi * cycles / max(L, 1e-12) * z + float(phase))
    if mode == "offset":
        q = H * 0.5 * (1.0 + s)
    elif mode == "pure":
        q = H * s
    elif mode == "half-wave":
        q = H * np.maximum(0.0, s)
    else:
        raise ValueError(f"hf_sinusoidal: unknown mode '{mode}'")
    return z, q, {"shape": "sinusoidal", "H": float(H), "cycles": cycles, "mode": str(mode)}


HF_BUILDERS = {
    "uniform": hf_uniform,
    "pulsed": hf_pulsed,
    "front_ramp": hf_front_ramp,
    "back_ramp": hf_back_ramp,
    "triangular": hf_triangular,
    "gaussian_pair": hf_gaussian_pair,
    "sinusoidal": hf_sinusoidal,
}


def build_heat_profile(L, shape_name, params):
    if shape_name not in HF_BUILDERS:
        raise ValueError(f"Unknown heat profile shape: {shape_name} "
                         f"(v2 supports {sorted(HF_BUILDERS)})")
    return HF_BUILDERS[shape_name](L=L, **params)


# ---------------------------------------------------------------------------
# Small physics helpers (ported)
# ---------------------------------------------------------------------------

def _prop_val(obj, name):
    attr = getattr(obj, name)
    return attr() if callable(attr) else float(attr)


def ethane_steam_mass_fractions(steam_to_ethane_kgkg: float) -> dict[str, float]:
    """Mass fractions for SD = kg steam per kg ethane (Y_C2H6 = 1/(1+SD))."""
    sd = float(steam_to_ethane_kgkg)
    if sd < 0.0:
        raise ValueError("steam_to_ethane_kgkg must be non-negative")
    return {"C2H6": 1.0 / (1.0 + sd), "H2O": sd / (1.0 + sd)}


def circular_area(diameter_m: float) -> float:
    return 0.25 * math.pi * float(diameter_m) ** 2


def circular_wall_area_per_volume(diameter_m: float) -> float:
    """Circular tube: perimeter / cross-section = 4/D [1/m]."""
    return 4.0 / float(diameter_m)


def convert_raw_rates_to_kmol_m3_s(rates_raw: np.ndarray) -> np.ndarray:
    """Convert CRACKSIM raw rates to kmol/m3/s according to RATES_RAW_UNITS."""
    if RATES_RAW_UNITS == "kmol/m3/s":
        return np.asarray(rates_raw, dtype=float)
    if RATES_RAW_UNITS == "mol/m3/s":
        return np.asarray(rates_raw, dtype=float) / 1000.0
    raise ValueError(f"Unsupported RATES_RAW_UNITS={RATES_RAW_UNITS!r}")


def compute_dYdt_from_wdot(wdot_kmol_m3_s, molecular_weights_kg_kmol, rho_kg_m3) -> np.ndarray:
    """dY/dt [1/s] = wdot [kmol/m3/s] · MW [kg/kmol] / ρ [kg/m3]."""
    rho = np.asarray(rho_kg_m3, dtype=float).reshape(-1, 1)
    mw = np.asarray(molecular_weights_kg_kmol, dtype=float).reshape(1, -1)
    return np.asarray(wdot_kmol_m3_s, dtype=float) * mw / np.clip(rho, 1.0e-300, None)


def compute_reaction_energy_terms(gas, T_raw, P_raw, Y_raw, wdot_kmol_m3_s):
    """``sum_h_wdot = Σ_k h_k·ẇ_k`` [J/m3/s] from Cantera partial molar enthalpies (J/kmol).

    ``heat_absorption = +sum_h_wdot`` is the canonical energy target (positive = endothermic);
    the Fluent energy source is ``S_h = −absorption``.
    """
    n = len(T_raw)
    sum_h_wdot = np.empty(n, dtype=float)
    for j in range(n):
        gas.TPY = float(T_raw[j]), float(P_raw[j]), Y_raw[j, :]
        h_j = np.asarray(gas.partial_molar_enthalpies, dtype=float)  # J/kmol
        sum_h_wdot[j] = float(np.dot(h_j, wdot_kmol_m3_s[j, :]))
    return sum_h_wdot


# ---------------------------------------------------------------------------
# CRACKSIM DLL callback / worker init (ported; Windows runtime only)
# ---------------------------------------------------------------------------

def CRACKSIM_rates_DLL(gas):
    """state→rates callback: NetRates_C(T, concentrations) → raw rates (RATES_RAW_UNITS)."""
    import ctypes as xt

    global fortlib
    if fortlib is None:
        raise RuntimeError("CRACKSIM not initialised in this worker")
    T_point = xt.byref(xt.c_double(gas.T))
    C_point = gas.concentrations.ctypes
    status = xt.pointer(xt.c_int(0))
    R_point = (xt.c_double * gas.n_species)()
    _ = fortlib.NetRates_C(T_point, C_point, R_point, status)
    return np.ctypeslib.as_array(R_point)


def init_worker_cracksim(dll_path: str, mech_path: str, base_dir, scratch_root, ready_q) -> None:
    """Per-worker CRACKSIM + Cantera initialisation (ported: scratch dirs, FORT45/100 env)."""
    import ctypes as xt
    import cantera as ct

    global fortlib, _gas_cache, REAC_MECH_PATH
    os.chdir(base_dir)
    w_scratch = Path(tempfile.mkdtemp(prefix=f"cracksim_w_{os.getpid()}_", dir=str(scratch_root)))
    for var in ("FORT45", "FOR45"):
        os.environ[var] = str(w_scratch / "fort45.log")
    for var in ("FORT100", "FOR100"):
        os.environ[var] = str(w_scratch / "fort100.log")

    ct.suppress_thermo_warnings()
    fort = xt.CDLL(dll_path)
    status = xt.pointer(xt.c_int(0))
    option = (xt.c_int * 20)()
    option[0] = 2
    _ = fort.Initialise_CRACKSIM(status, option)
    if status[0] != 1:
        ready_q.put("ERROR: CRACKSIM initialise failed in worker")
        return
    fortlib = fort
    REAC_MECH_PATH = mech_path
    _gas_cache = ct.Solution(REAC_MECH_PATH)
    missing = [s for s in ("C2H6", "H2O") if s not in _gas_cache.species_names]
    if missing:
        ready_q.put(f"ERROR: mechanism lacks required species {missing}")
        return
    ready_q.put("READY")


# ---------------------------------------------------------------------------
# v2 settings + manifest tiers
# ---------------------------------------------------------------------------

@dataclass
class GenV2Settings:
    """Execution settings for the v2 generator (defaults = the approved ideal-DB spec)."""

    n_points: int = 400
    solver_rtol: float = 1.0e-9
    solver_atol: float = 1.0e-16
    t_max_K: float = T_MAX_K_V2
    storage: StorageConfig = field(
        default_factory=lambda: StorageConfig(mode="front_adaptive", max_frac_jump=0.03,
                                              min_every_nth=2))
    case_timeout_s: float = 900.0
    keep_d_mix: bool = False
    generator_version: str = "v2.0"


#: Per-regime FULL-tier case counts (the approved spec, ~20.5k cases).
FULL_TIER_COUNTS = {
    "body": 12000, "inlet_seed": 2500, "high_T": 2500, "tail": 2000, "deep_conversion": 1500,
}
#: Pilot = the same Sobol streams, first ~6% per regime (≈1.2k cases — prove-it-works scale).
PILOT_FRACTION = 0.06
#: Smoke = first N per regime (pipeline shakedown, minutes).
SMOKE_PER_REGIME = 2
#: Independent-stream test tier (certification set).
TEST_TIER_COUNTS = {
    "body": 2000, "inlet_seed": 250, "high_T": 250, "tail": 250, "deep_conversion": 250,
}
TEST_SEED_OFFSET = 7777


def _base_config(counts: dict[str, int], seed: int) -> DataGenConfig:
    """The broad v2 envelope (extends the campaign defaults to the merged spec)."""
    return DataGenConfig(
        n_body_cases=counts["body"],
        n_inlet_seed_cases=counts["inlet_seed"],
        n_highT_cases=counts["high_T"],
        n_tail_cases=counts["tail"],
        P_in_range_Pa=(1.5e5, 3.5e5),
        X_H2O_values=(0.10, 0.23, 0.30, 0.43, 0.55),
        H_peak_range_W_m2=(25.0e3, 250.0e3),
        L_range_m=(3.0, 12.0),
        T_max_K=T_MAX_K_V2,
        seed=seed,
    )


def _deep_conversion_config(n: int, seed: int) -> DataGenConfig:
    """Block 4+5: long-τ, hot, low-dilution cases where aromatics form and exothermic
    recombination (the E-c positivity escape hatch) would show if it exists."""
    return DataGenConfig(
        n_body_cases=n, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0,
        T_in_range_K=(950.0, 1100.0),
        P_in_range_Pa=(1.5e5, 3.5e5),
        X_H2O_values=(0.10, 0.23),
        H_peak_range_W_m2=(25.0e3, 120.0e3),
        L_range_m=(12.0, 20.0),
        T_max_K=T_MAX_K_V2,
        seed=seed + 99,
    )


def build_v2_manifest(tier: str, *, seed: int = 20260612) -> tuple[list[dict], dict[str, Any]]:
    """Build the case list for *tier* ∈ {"smoke", "pilot", "full", "test"}.

    Tiers are PREFIXES of the same per-regime Sobol streams (low-discrepancy prefixes are
    themselves well-spread), so smoke ⊂ pilot ⊂ full case-ID sets and a full run after a
    pilot run can skip already-solved cases. The "test" tier uses an independent seed.
    """
    if tier not in ("smoke", "pilot", "full", "test"):
        raise ValueError(f"unknown tier {tier!r}")
    counts = dict(TEST_TIER_COUNTS) if tier == "test" else dict(FULL_TIER_COUNTS)
    eff_seed = seed + TEST_SEED_OFFSET if tier == "test" else seed

    cases = list(build_cases(_base_config(counts, eff_seed)))
    deep = build_cases(_deep_conversion_config(counts["deep_conversion"], eff_seed))
    next_id = (max(c["id"] for c in cases) + 1) if cases else 0
    for i, c in enumerate(deep):
        c["id"] = next_id + i
        c["seed"] = c["id"]
        c["regime"] = "deep_conversion"
    cases.extend(deep)

    if tier in ("smoke", "pilot"):
        keep_n = {r: (SMOKE_PER_REGIME if tier == "smoke"
                      else max(SMOKE_PER_REGIME, int(round(PILOT_FRACTION * n))))
                  for r, n in counts.items()}
        by_regime: dict[str, int] = {r: 0 for r in keep_n}
        kept = []
        for c in cases:  # build_cases emits each regime's stream in order → prefix per regime
            r = c["regime"]
            if by_regime.get(r, 0) < keep_n.get(r, 0):
                kept.append(c)
                by_regime[r] += 1
        cases = kept

    # test-tier IDs offset so they can never collide with train-tier CaseIDs
    if tier == "test":
        for c in cases:
            c["id"] += 1_000_000
            c["seed"] = c["id"]

    regime_counts: dict[str, int] = {}
    for c in cases:
        regime_counts[c["regime"]] = regime_counts.get(c["regime"], 0) + 1
    manifest = {
        "tier": tier, "seed": seed, "n_cases": len(cases), "regime_counts": regime_counts,
        "t_max_K": T_MAX_K_V2, "generator_version": GenV2Settings().generator_version,
    }
    return cases, manifest


def to_runner_case(case: dict, settings: GenV2Settings) -> dict:
    """Map a manifest case dict to the runner's expected keys (steam ratio from X_H2O)."""
    x = float(case["X_H2O"])  # steam MASS fraction of the binary inlet
    if not 0.0 <= x < 1.0:
        raise ValueError(f"X_H2O out of range: {x}")
    runner = {
        "id": int(case["id"]),
        "seed": int(case.get("seed", case["id"])),
        "regime": str(case.get("regime", "body")),
        "L": float(case["L"]),
        "H_peak": float(case["H_peak"]),
        "shape": str(case["shape"]),
        "params": dict(case.get("params", {})),
        "mdot": float(case["mdot"]),
        "T_in": float(case["T_in"]),
        "P_in": float(case["P_in"]),
        "steam_to_ethane_kgkg": x / (1.0 - x),
        "N_points": int(settings.n_points),
        "diameter_m": float(case["diameter"]),
    }
    if "Re_in" in case:
        runner["Re_in"] = float(case["Re_in"])
    if "U_in" in case:
        runner["U_in"] = float(case["U_in"])
    return runner


# ---------------------------------------------------------------------------
# v2 frame assembly (pure — unit-testable without Cantera)
# ---------------------------------------------------------------------------

def assemble_v2_frame(
    *,
    species_names: Sequence[str],
    Y: np.ndarray,
    dYdt: np.ndarray,
    T: np.ndarray,
    P: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    tau: np.ndarray,
    z: np.ndarray,
    cp: np.ndarray,
    cv: np.ndarray,
    mu: np.ndarray,
    k: np.ndarray,
    W_mean: np.ndarray,
    absorption: np.ndarray,
    s_wall: np.ndarray,
    q_wall: np.ndarray,
    pfr_point_index: np.ndarray,
    n_points_solved: int,
    runner_case: dict,
    settings: GenV2Settings,
    inlet_Y: dict[str, float],
    sample_kind: str = "trajectory",
):
    """Assemble the v2 export DataFrame (float64 species/rates; no pseudo-species columns).

    Pure function of arrays → testable anywhere; ``Schema.from_columns`` must accept the
    result (regression-tested).
    """
    import pandas as pd

    n = len(T)
    idx = pd.RangeIndex(n)
    dfY = pd.DataFrame(np.asarray(Y, dtype=np.float64),
                       columns=[f"Y_{s}" for s in species_names], index=idx)
    df_dYdt = pd.DataFrame(np.asarray(dYdt, dtype=np.float64),
                           columns=[f"dYdt_{s} [1/s]" for s in species_names], index=idx)
    df_main = pd.DataFrame({
        "T [K]": T, "P [Pa]": P,
        "Reaction heat absorption [J/s/m3]": absorption,
        "S Wall imposed [J/s/m3]": s_wall,
        "Heat input [W/m2]": q_wall,
        "z [m]": z, "tau [s]": tau, "u [m/s]": u,
        "cp_mass [J/kg/K]": cp, "cv_mass [J/kg/K]": cv,
        "rho [kg/m3]": rho, "mu [Pa-s]": mu, "k [W/m/K]": k, "W_mean [kg/kmol]": W_mean,
        "PFR point index": np.asarray(pfr_point_index, dtype=int),
        "PFR points solved": np.full(n, int(n_points_solved), dtype=int),
        "Storage policy": np.full(
            n, f"{settings.storage.mode}:{settings.storage.max_frac_jump}", dtype=object),
        "CaseID": np.full(n, int(runner_case["id"]), dtype=int),
        "regime": np.full(n, runner_case.get("regime", "body"), dtype=object),
        "sample_kind": np.full(n, sample_kind, dtype=object),
        "mdot [kg/s]": np.full(n, float(runner_case["mdot"])),
        "Mass flow [kg/s]": np.full(n, float(runner_case["mdot"])),
        "diameter [m]": np.full(n, float(runner_case["diameter_m"])),
        "Area [m2]": np.full(n, circular_area(runner_case["diameter_m"])),
        "steam_to_ethane [kg/kg]": np.full(n, float(runner_case["steam_to_ethane_kgkg"])),
        "inlet_Y_C2H6 [-]": np.full(n, float(inlet_Y["C2H6"])),
        "inlet_Y_H2O [-]": np.full(n, float(inlet_Y["H2O"])),
        "T_in [K]": np.full(n, float(runner_case["T_in"])),
        "P_in [Pa]": np.full(n, float(runner_case["P_in"])),
        "shape": np.full(n, runner_case["shape"], dtype=object),
        "H_peak [W/m2]": np.full(n, float(runner_case["H_peak"])),
        "solver_rtol": np.full(n, settings.solver_rtol),
        "solver_atol": np.full(n, settings.solver_atol),
        "generator_version": np.full(n, settings.generator_version, dtype=object),
    }, index=idx)
    if "Re_in" in runner_case:
        df_main["Re_in [-]"] = float(runner_case["Re_in"])
    if "U_in" in runner_case:
        df_main["U_in [m/s]"] = float(runner_case["U_in"])
    return pd.concat([dfY, df_dYdt, df_main], axis=1, copy=False)


# ---------------------------------------------------------------------------
# Per-case PFR execution (Cantera + DLL; Windows runtime)
# ---------------------------------------------------------------------------

def run_case_v2(case: dict, settings: GenV2Settings):
    """Solve one PFR case and return ``(DataFrame, audit_dict)`` or ``(None, reason)``.

    Differences from the ported ``run_case`` are listed in the module docstring.
    ``case`` must be a runner dict (see :func:`to_runner_case`).
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ideal_reactor_models import customPFR  # noqa: E402  (repo root, F5-fixed)

    try:
        L = float(case["L"])
        Np_req = int(case["N_points"])
        shape = str(case["shape"])
        params = dict(case.get("params", {}))
        if shape == "pulsed":
            params = {**params, "H": case["H_peak"], "N": int(params.get("N", 10)),
                      "Np_req": Np_req, "seed": int(case.get("seed", 0))}
        elif shape in ("sinusoidal", "front_ramp", "back_ramp"):
            params = {**params, "H": case["H_peak"], "Np_req": Np_req}
        else:
            params = {**params, "H": case["H_peak"]}
        z_cp, q_cp, _meta = build_heat_profile(L, shape, params)
        hf_func = make_piecewise(z_cp, q_cp)

        pfr = customPFR(
            REAC_MECH_PATH, "gas", float(case["mdot"]), float(case["diameter_m"]),
            CRACKSIM_rates_DLL, energy_type="heat-flux-profile", heat_flux=hf_func,
            U=None, Tr=None, friction_factors=None,
        )
        try:
            pfr.gas = _gas_cache
        except Exception:
            pass
        inlet_Y = ethane_steam_mass_fractions(case["steam_to_ethane_kgkg"])
        pfr.gas.TPY = float(case["T_in"]), float(case["P_in"]), inlet_Y
        states, rates, _ = pfr.solve(
            L, Np_req, solver_rtol=settings.solver_rtol, solver_atol=settings.solver_atol)

        T_full = np.asarray(states.T, dtype=float).ravel()
        P_full = np.asarray(states.P, dtype=float).ravel()
        Y_full = np.asarray(states.Y, dtype=float)
        rates_full = np.asarray(rates, dtype=float)
        z_full = getattr(states, "z", None)
        if z_full is None:
            z_full = getattr(states, "grid", None)
        if z_full is None:
            return None, "missing axial grid"
        z_full = np.asarray(z_full, dtype=float).ravel()

        n = min(T_full.size, P_full.size, z_full.size, Y_full.shape[0], rates_full.shape[0])
        if n <= 1:
            return None, "invalid array sizes"
        T_full, P_full, z_full = T_full[:n], P_full[:n], z_full[:n]
        Y_full, rates_full = Y_full[:n, :], rates_full[:n, :]

        if np.any(T_full > settings.t_max_K):
            return None, f"T exceeded cap {settings.t_max_K:.1f} K"
        dz_nom = L / max(Np_req - 1, 1)
        if z_full[-1] < L - 0.5 * dz_nom:
            return None, f"truncated at z={z_full[-1]:.4f} < L={L:.4f}"

        gas = pfr.gas
        species_names = list(gas.species_names)
        mw = np.asarray(gas.molecular_weights, dtype=float)

        q_wall_full = np.array([hf_func(zz) for zz in z_full], dtype=float)
        s_wall_full = q_wall_full * circular_wall_area_per_volume(case["diameter_m"])

        cp_f = np.empty(n); cv_f = np.empty(n); rho_f = np.empty(n)
        mu_f = np.empty(n); k_f = np.empty(n); Wm_f = np.empty(n)
        for j in range(n):
            gas.TPY = float(T_full[j]), float(P_full[j]), Y_full[j, :]
            cp_f[j] = float(gas.cp_mass); cv_f[j] = float(gas.cv_mass)
            rho_f[j] = float(gas.density)
            mu_f[j] = _prop_val(gas, "viscosity"); k_f[j] = _prop_val(gas, "thermal_conductivity")
            Wm_f[j] = float(gas.mean_molecular_weight)

        area = circular_area(case["diameter_m"])
        u_f = case["mdot"] / np.clip(rho_f * area, 1.0e-300, None)
        tau_f = np.zeros(n, dtype=float)
        if n > 1:
            inv_u_mid = 0.5 * (1.0 / np.clip(u_f[1:], 1e-300, None)
                               + 1.0 / np.clip(u_f[:-1], 1e-300, None))
            tau_f[1:] = np.cumsum(inv_u_mid * np.diff(z_full))

        wdot_f = convert_raw_rates_to_kmol_m3_s(rates_full)
        dYdt_f = compute_dYdt_from_wdot(wdot_f, mw, rho_f)
        absorption_f = compute_reaction_energy_terms(gas, T_full, P_full, Y_full, wdot_f)

        # FRONT-ADAPTIVE storage on the full solved grid (the v2 point).
        store_idx = select_storage_indices(absorption_f, settings.storage)
        if store_idx.size <= 0:
            return None, "no storage indices"

        audit = sign_audit(absorption_f)
        peak_abs = float(np.abs(absorption_f).max())
        grid_fr = (np.abs(np.diff(absorption_f)) / peak_abs) if peak_abs > 0 else np.zeros(1)
        audit.update({"case_id": int(case["id"]), "regime": case.get("regime", "body"),
                      "n_solved": int(n), "n_stored": int(store_idx.size),
                      # single-solver-step |Δabsorption|/peak — the storage policy's floor;
                      # if p95 here exceeds the policy, raise --n-points (grid-limited front)
                      "grid_p95_jump_frac": float(np.percentile(grid_fr, 95)),
                      "grid_max_jump_frac": float(grid_fr.max())})

        df = assemble_v2_frame(
            species_names=species_names,
            Y=Y_full[store_idx, :], dYdt=dYdt_f[store_idx, :],
            T=T_full[store_idx], P=P_full[store_idx], rho=rho_f[store_idx],
            u=u_f[store_idx], tau=tau_f[store_idx], z=z_full[store_idx],
            cp=cp_f[store_idx], cv=cv_f[store_idx], mu=mu_f[store_idx], k=k_f[store_idx],
            W_mean=Wm_f[store_idx],
            absorption=absorption_f[store_idx],
            s_wall=s_wall_full[store_idx], q_wall=q_wall_full[store_idx],
            pfr_point_index=store_idx, n_points_solved=n,
            runner_case=case, settings=settings, inlet_Y=inlet_Y,
        )
        return df, audit
    except Exception as e:  # noqa: BLE001 — worker resilience: report, never crash the pool
        return None, f"exception: {e}"


# ---------------------------------------------------------------------------
# Off-manifold point cloud (block 2)
# ---------------------------------------------------------------------------

@dataclass
class PerturbConfig:
    """Off-manifold perturbation kernel around trajectory anchor states.

    Multiplicative log-normal jitter on every species mass fraction (σ in log-space),
    renormalised to a valid composition; relative T and P jitter. Modest by design —
    the goal is the CFD-visited NEIGHBOURHOOD of the manifold, not random chemistry.
    """

    sigma_log: float = 0.25
    t_rel: float = 0.03
    p_rel: float = 0.05
    points_per_anchor: int = 4


def perturb_states(Y_anchor: np.ndarray, T_anchor: np.ndarray, P_anchor: np.ndarray,
                   cfg: PerturbConfig, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return perturbed (Y, T, P) with ``points_per_anchor`` samples per anchor row.

    Pure NumPy (unit-tested): compositions stay non-negative and renormalised to sum 1;
    T respects the mechanism cap upstream (caller clips).
    """
    Y_anchor = np.asarray(Y_anchor, dtype=float)
    T_anchor = np.asarray(T_anchor, dtype=float).ravel()
    P_anchor = np.asarray(P_anchor, dtype=float).ravel()
    n, m = Y_anchor.shape
    k = int(cfg.points_per_anchor)
    rng = np.random.default_rng(seed)

    Y = np.repeat(Y_anchor, k, axis=0)
    T = np.repeat(T_anchor, k)
    P = np.repeat(P_anchor, k)
    Y = Y * np.exp(rng.normal(0.0, cfg.sigma_log, size=Y.shape))
    Y = np.maximum(Y, 0.0)
    Y = Y / np.clip(Y.sum(axis=1, keepdims=True), 1e-300, None)
    T = T * (1.0 + rng.uniform(-cfg.t_rel, cfg.t_rel, size=T.shape))
    P = P * (1.0 + rng.uniform(-cfg.p_rel, cfg.p_rel, size=P.shape))
    return Y, T, P


def eval_offmanifold_points(anchors_df, n_target: int, settings: GenV2Settings,
                            cfg: PerturbConfig, seed: int):
    """Single-point DLL evaluations at perturbed anchor states → v2-schema DataFrame.

    Anchors are sampled from a freshly generated trajectory parquet (front-biased by
    construction: front-adaptive storage already concentrates rows there). Synthetic rows
    carry ``sample_kind="offmanifold"``, ``CaseID = −(anchor CaseID)`` and ``tau = z = −1``
    so the Lagrangian pair builder can never chain them (Δτ ≤ 0 within a case).
    """
    import pandas as pd

    gas = _gas_cache
    if gas is None:
        raise RuntimeError("worker not initialised (gas cache missing)")
    species_names = list(gas.species_names)
    mw = np.asarray(gas.molecular_weights, dtype=float)
    y_cols = [c for c in anchors_df.columns if c.startswith("Y_")]
    anchor_species = [c[2:] for c in y_cols]
    if anchor_species != species_names:
        raise RuntimeError("anchor parquet species do not match the mechanism ordering")

    n_anchors = max(1, n_target // cfg.points_per_anchor)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(anchors_df), size=min(n_anchors, len(anchors_df)), replace=False)
    sub = anchors_df.iloc[sel]
    Yp, Tp, Pp = perturb_states(
        sub[y_cols].to_numpy(float),
        sub["T [K]"].to_numpy(float), sub["P [Pa]"].to_numpy(float), cfg, seed)
    Tp = np.clip(Tp, 300.0, settings.t_max_K)
    anchor_case = np.repeat(sub["CaseID"].to_numpy(int), cfg.points_per_anchor)

    n = len(Tp)
    rates_raw = np.empty((n, len(species_names)))
    rho = np.empty(n); cp = np.empty(n); cv = np.empty(n)
    mu = np.empty(n); kk = np.empty(n); Wm = np.empty(n)
    for j in range(n):
        gas.TPY = float(Tp[j]), float(Pp[j]), Yp[j, :]
        rates_raw[j, :] = CRACKSIM_rates_DLL(gas)
        rho[j] = float(gas.density); cp[j] = float(gas.cp_mass); cv[j] = float(gas.cv_mass)
        mu[j] = _prop_val(gas, "viscosity"); kk[j] = _prop_val(gas, "thermal_conductivity")
        Wm[j] = float(gas.mean_molecular_weight)
    wdot = convert_raw_rates_to_kmol_m3_s(rates_raw)
    dYdt = compute_dYdt_from_wdot(wdot, mw, rho)
    absorption = compute_reaction_energy_terms(gas, Tp, Pp, Yp, wdot)

    frames = []
    neg = np.zeros(n)
    for case_id in np.unique(anchor_case):
        m_sel = anchor_case == case_id
        nn = int(m_sel.sum())
        runner_stub = {
            "id": -int(case_id), "regime": "offmanifold", "mdot": 0.0, "diameter_m": 1.0,
            "steam_to_ethane_kgkg": 0.0, "T_in": float("nan"), "P_in": float("nan"),
            "shape": "n/a", "H_peak": 0.0,
        }
        frames.append(assemble_v2_frame(
            species_names=species_names,
            Y=Yp[m_sel], dYdt=dYdt[m_sel], T=Tp[m_sel], P=Pp[m_sel], rho=rho[m_sel],
            u=neg[m_sel], tau=np.full(nn, -1.0), z=np.full(nn, -1.0),
            cp=cp[m_sel], cv=cv[m_sel], mu=mu[m_sel], k=kk[m_sel], W_mean=Wm[m_sel],
            absorption=absorption[m_sel], s_wall=neg[m_sel], q_wall=neg[m_sel],
            pfr_point_index=np.full(nn, -1), n_points_solved=0,
            runner_case=runner_stub, settings=settings,
            inlet_Y={"C2H6": float("nan"), "H2O": float("nan")},
            sample_kind="offmanifold",
        ))
    return pd.concat(frames, axis=0, ignore_index=True)


# ---------------------------------------------------------------------------
# Verification gates (run before any production tier)
# ---------------------------------------------------------------------------

def gate_dll_consistency(reference_df, n_rows: int = 64, rel_tol: float = 1.0e-6,
                         dydt_floor: float = 1.0e-8) -> dict[str, Any]:
    """Gate A: re-evaluate the DLL at stored states and compare to stored dYdt columns.

    Catches species-ordering and unit mistakes (the Phase-1 unverified hypothesis): the
    stored trajectory rows came through this exact pipeline, so a faithful single-point
    re-evaluation must reproduce dYdt to numerical precision on the same machine.
    Relative differences are evaluated where |dYdt| > *dydt_floor* (below that, solver
    noise dominates and relative comparison is meaningless).
    """
    gas = _gas_cache
    if gas is None:
        raise RuntimeError("worker not initialised")
    species_names = list(gas.species_names)
    mw = np.asarray(gas.molecular_weights, dtype=float)
    y_cols = [f"Y_{s}" for s in species_names]
    d_cols = [f"dYdt_{s} [1/s]" for s in species_names]
    missing = [c for c in (y_cols[:1] + d_cols[:1]) if c not in reference_df.columns]
    if missing:
        raise ValueError(f"reference frame lacks expected columns, e.g. {missing}")

    sub = reference_df.iloc[: int(n_rows)]
    Y = sub[y_cols].to_numpy(float)
    T = sub["T [K]"].to_numpy(float)
    P = sub["P [Pa]"].to_numpy(float)
    stored = sub[d_cols].to_numpy(float)

    recomputed = np.empty_like(stored)
    for j in range(len(sub)):
        gas.TPY = float(T[j]), float(P[j]), Y[j, :]
        raw = CRACKSIM_rates_DLL(gas)
        wdot = convert_raw_rates_to_kmol_m3_s(raw[np.newaxis, :])
        recomputed[j, :] = compute_dYdt_from_wdot(wdot, mw, np.array([gas.density]))[0]

    mask = np.abs(stored) > dydt_floor
    rel = np.abs(recomputed[mask] - stored[mask]) / np.abs(stored[mask])

    # diagnostics: distinguish "DLL returns zeros out of solve context" (statefulness),
    # ordering/unit errors (uniform large rel), and isolated numerical disagreements.
    zero_recompute_frac = float(np.mean(np.abs(recomputed[mask]) < 1e-300)) if mask.any() else 0.0
    per_species_worst: list[tuple[str, float]] = []
    if mask.any():
        rel_full = np.where(mask, np.abs(recomputed - stored) / np.maximum(np.abs(stored), 1e-300), 0.0)
        worst_idx = np.argsort(rel_full.max(axis=0))[::-1][:5]
        per_species_worst = [(species_names[i], float(rel_full[:, i].max())) for i in worst_idx]

    # statefulness probe: the same state evaluated twice must agree exactly; a drift
    # implies NetRates_C carries internal state between calls.
    gas.TPY = float(T[0]), float(P[0]), Y[0, :]
    raw_a = np.array(CRACKSIM_rates_DLL(gas), dtype=float, copy=True)
    gas.TPY = float(T[0]), float(P[0]), Y[0, :]
    raw_b = np.array(CRACKSIM_rates_DLL(gas), dtype=float, copy=True)
    double_call_max_abs_diff = float(np.abs(raw_a - raw_b).max())
    raw_nonzero_frac = float(np.mean(np.abs(raw_a) > 0.0))

    out = {
        "n_rows": int(len(sub)), "n_compared": int(mask.sum()),
        "max_rel_diff": float(rel.max()) if mask.any() else 0.0,
        "median_rel_diff": float(np.median(rel)) if mask.any() else 0.0,
        "p95_rel_diff": float(np.percentile(rel, 95)) if mask.any() else 0.0,
        "zero_recompute_frac": zero_recompute_frac,
        "raw_nonzero_frac_row0": raw_nonzero_frac,
        "double_call_max_abs_diff": double_call_max_abs_diff,
        "per_species_worst": per_species_worst,
        "passed": bool(mask.any() and float(rel.max()) < rel_tol),
        "rel_tol": rel_tol,
    }
    return out


def gate_front_resolution(df, max_frac_jump: float, slack: float = 1.5) -> dict[str, Any]:
    """Gate C: POLICY-chosen stored jumps must respect the storage policy (× slack).

    Stored jumps split into two populations using ``PFR point index`` when available:
    consecutive solver points (index gap == 1) are **grid-limited** — no storage policy can
    make a single solver step smaller, so they are reported as a solve-resolution
    recommendation, not failed; multi-step jumps (gap > 1) are the policy's own skipping
    decisions and are the pass/fail population. Without the index column (legacy frames),
    all jumps count as policy jumps — strictest interpretation.
    """
    col = "Reaction heat absorption [J/s/m3]"
    traj = df[df["sample_kind"] == "trajectory"] if "sample_kind" in df.columns else df
    has_idx = "PFR point index" in traj.columns
    policy_jumps: list[float] = []
    grid_jumps: list[float] = []
    for _cid, g in traj.groupby("CaseID"):
        g = g.sort_values("tau [s]")
        a = g[col].to_numpy(float)
        peak = np.abs(a).max()
        if peak <= 0 or len(a) < 2:
            continue
        fr = np.abs(np.diff(a)) / peak
        if has_idx:
            gaps = np.diff(g["PFR point index"].to_numpy(int))
            policy_jumps.extend(fr[gaps > 1])
            grid_jumps.extend(fr[gaps <= 1])
        else:
            policy_jumps.extend(fr)
    pj = np.asarray(policy_jumps) if policy_jumps else np.array([0.0])
    gj = np.asarray(grid_jumps) if grid_jumps else np.array([0.0])
    grid_p95 = float(np.percentile(gj, 95))
    # recommended solve-grid multiplier so single grid steps land within the policy
    grid_factor = grid_p95 / max_frac_jump if max_frac_jump > 0 else float("nan")
    return {
        "median_jump_frac": float(np.median(pj)),
        "p95_jump_frac": float(np.percentile(pj, 95)),
        "grid_p95_jump_frac": grid_p95,
        "grid_resolution_factor": float(grid_factor),
        "n_policy_jumps": int(len(policy_jumps)),
        "n_grid_jumps": int(len(grid_jumps)),
        "passed": bool(np.percentile(pj, 95) <= max_frac_jump * slack),
        "threshold": max_frac_jump * slack,
    }


def aggregate_sign_audits(audits: Sequence[dict]) -> dict[str, Any]:
    """Gate D: per-case sign-audit roll-up (the E-c positivity-lineage check)."""
    mins = [a["min_value"] for a in audits if np.isfinite(a.get("min_value", np.nan))]
    fracs = [a["frac_negative"] for a in audits]
    worst = min(mins) if mins else float("nan")
    return {
        "n_cases": len(audits),
        "worst_min_absorption": worst,
        "max_frac_negative": float(max(fracs)) if fracs else float("nan"),
        "material_negative": bool(mins and worst < -1.0e6),
    }

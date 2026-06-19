"""Cantera-backed flow finalisation and HPC generation driver.

:func:`build_cases` (in ``sampling.py``) produces cases parameterised by inlet Reynolds number. The
mass flow ``mdot`` and inlet velocity ``U_in`` depend on gas properties, so they are computed here
with Cantera, mirroring the formula already used in ``Database_Generation_MB.py`` (lines 558-563):

    A   = pi * D**2 / 4
    U_in = Re_in * mu_in / (rho_in * D)
    mdot = rho_in * U_in * A

The finalised cases are written to a manifest the HPC run consumes; the actual CRACKSIM integration
(``run_case`` + the multiprocessing harness in ``Database_Generation_MB.py``) is launched on the HPC.
Cantera is imported lazily so this module can be imported (and the non-Cantera helpers tested)
without it installed.

Storage modes (see :class:`~scarfs.data.config.StorageConfig`)
--------------------------------------------------------------
:func:`select_storage_indices` implements both storage policies as a **pure function** over a
per-case trajectory.  It is testable without Cantera.

Sign audit
----------
:func:`sign_audit` computes per-case sign statistics for the ``Reaction heat absorption`` trajectory.
The generator should call this per case so the positivity assumption (E-c) is checkable before
training.  Stride-6 data are strictly positive; stride-5 show a −2.06e5 minimum (truncation noise).
A negative fraction above ~1e-3 or a sum-of-negatives below −1e6 warrants investigation.

Export-column contract
----------------------
The generator assembles per-row output columns following these rules:

- ``Reaction heat absorption [J/s/m3]`` = ``+Σ h_i_mass · ρ · dY_i/dt`` (positive = endothermic).
  **Never export-rename this column.**
- ``S Energy [J/s/m3]`` is a CRACKSIM-internal term.  It may be read from the DLL output for
  diagnostic purposes but must **never be used as a training target** and should not be
  the sole exported energy column.
- ``dYdt_<species> [1/s]`` columns are exported when
  :attr:`~scarfs.data.config.ExportColumnsConfig.dydt` is ``True``.
- ``wdot_<species> [kmol/m3/s]`` columns are exported when
  :attr:`~scarfs.data.config.ExportColumnsConfig.wdot` is ``True``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .config import DataGenConfig, StorageConfig
from .sampling import build_cases, coverage_summary


# ---------------------------------------------------------------------------
# Storage index selection
# ---------------------------------------------------------------------------

def select_storage_indices(
    s_e: np.ndarray, cfg: StorageConfig, comp: np.ndarray | None = None
) -> np.ndarray:
    """Return the subset of point indices to store for a single reactor case.

    This is a **pure function** with no side effects — it can be unit-tested
    with synthetic trajectories without Cantera.

    Parameters
    ----------
    s_e
        1-D array of ``Reaction heat absorption [J/s/m3]`` values along the
        case trajectory (one entry per solved PFR grid point, in order).
    cfg
        Storage configuration controlling the mode and thresholds.
    comp
        Optional ``(n_points, n_species)`` composition (mass fractions) along the trajectory.
        When supplied AND ``cfg.comp_arcsinh_jump > 0`` (front_adaptive only), a point is also
        stored whenever any species' change in ``arcsinh(Y / cfg.comp_arcsinh_floor)`` since the
        last stored point exceeds ``cfg.comp_arcsinh_jump`` — this keeps the radical-chain
        INDUCTION zone (composition moving, |S_E| still ~0) that the S_E-only policy discards
        (the RC-1 near-inlet coverage fix).

    Returns
    -------
    np.ndarray
        Sorted array of integer indices into *s_e* to store.  Always includes
        the first and last index.

    Notes
    -----
    **Stride mode** (``cfg.mode == "stride"``)
        Returns every ``cfg.every_nth``-th index, plus the last point.

    **Front-adaptive mode** (``cfg.mode == "front_adaptive"``)
        Stores a point at index *i* when the absolute change in *s_e* from
        the last stored index exceeds ``cfg.max_frac_jump × peak`` (``peak = max(|s_e|)``),
        OR — when *comp* is given — when the composition-curvature co-trigger fires.  The first
        and last points are always stored, and the ``cfg.min_every_nth`` size cap applies to both
        triggers.  When *s_e* is all-zero and no composition trigger is active, every
        ``cfg.min_every_nth``-th point is stored.

    Raises
    ------
    ValueError
        If *s_e* is empty.
    """
    n = len(s_e)
    if n == 0:
        raise ValueError("select_storage_indices: s_e must not be empty.")
    if n == 1:
        return np.array([0], dtype=np.intp)

    if cfg.mode == "stride":
        indices = list(range(0, n, cfg.every_nth))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        return np.array(indices, dtype=np.intp)

    # front_adaptive mode
    peak = float(np.max(np.abs(s_e)))
    threshold = cfg.max_frac_jump * peak if peak > 0.0 else 0.0
    min_gap = int(cfg.min_every_nth)  # must have elapsed ≥ min_gap points between stores

    use_comp = comp is not None and getattr(cfg, "comp_arcsinh_jump", 0.0) > 0.0
    if use_comp:
        comp_t = np.arcsinh(np.asarray(comp, dtype=float) / cfg.comp_arcsinh_floor)  # (n, n_species)
        comp_jump = float(cfg.comp_arcsinh_jump)

    stored: list[int] = [0]  # always store first
    last_stored_val = float(s_e[0])
    last_stored_idx = 0

    for i in range(1, n - 1):
        # Size cap: skip if fewer than min_gap points have passed since last stored point
        if (i - last_stored_idx) < min_gap:
            continue
        delta = abs(float(s_e[i]) - last_stored_val)
        fire = peak == 0.0 or delta > threshold
        if not fire and use_comp:
            comp_move = float(np.max(np.abs(comp_t[i] - comp_t[last_stored_idx])))
            fire = comp_move > comp_jump
        if fire:
            stored.append(i)
            last_stored_val = float(s_e[i])
            last_stored_idx = i

    # Always store last
    stored.append(n - 1)
    return np.array(sorted(set(stored)), dtype=np.intp)


# ---------------------------------------------------------------------------
# Sign audit
# ---------------------------------------------------------------------------

def sign_audit(absorption: np.ndarray) -> dict:
    """Compute sign statistics for a per-case ``Reaction heat absorption`` trajectory.

    Used to verify the positivity assumption (E-c) stated in the design doc: stride-6 data are
    strictly positive; stride-5 show a −2.06e5 minimum (truncation noise from species dropping).
    A ``frac_negative`` above ~1e-3 or ``sum_negative`` below −1e6 warrants investigation before
    using the case for training.

    Parameters
    ----------
    absorption
        1-D array of ``Reaction heat absorption [J/s/m3]`` values for one case.

    Returns
    -------
    dict with keys:

    - ``"n_points"``       : int — number of grid points in this case.
    - ``"min_value"``      : float — minimum value (most negative).
    - ``"frac_negative"``  : float — fraction of points with absorption < 0.
    - ``"sum_negative"``   : float — sum of negative values (≤ 0).
    - ``"max_abs_negative"``: float — largest magnitude negative value (≥ 0).
    """
    arr = np.asarray(absorption, dtype=float)
    neg_mask = arr < 0.0
    n_neg = int(np.sum(neg_mask))
    return {
        "n_points": int(len(arr)),
        "min_value": float(arr.min()) if len(arr) > 0 else float("nan"),
        "frac_negative": float(n_neg) / max(len(arr), 1),
        "sum_negative": float(arr[neg_mask].sum()) if n_neg > 0 else 0.0,
        "max_abs_negative": float(np.abs(arr[neg_mask]).max()) if n_neg > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Flow finalisation (Cantera)
# ---------------------------------------------------------------------------

def _inlet_composition(x_h2o: float) -> dict[str, float]:
    """Ethane/steam inlet mass-fraction dict for steam dilution *x_h2o* (mass basis)."""
    return {"C2H6": 1.0 - float(x_h2o), "H2O": float(x_h2o)}


def finalize_flow(cases: list[dict], mech_path: str, diam_m: float | None = None) -> list[dict]:
    """Fill ``mdot`` and ``U_in`` for each case from Cantera gas properties.

    Supports D-sweep: each case may carry its own ``"diameter"`` key (set by
    :func:`scarfs.data.sampling.build_cases`).  When ``"diameter"`` is present it is used in
    preference to *diam_m*.  Pass *diam_m* as a fallback for legacy case dicts that lack the key.

    Parameters
    ----------
    cases
        Case dicts from :func:`scarfs.data.sampling.build_cases` (must have ``T_in``/``P_in``/
        ``X_H2O``/``Re_in``).
    mech_path
        Path to the Cantera mechanism (``chem.yaml``).
    diam_m
        Fallback reactor diameter [m] used when a case does not carry a ``"diameter"`` key.
        Required when any case is missing the key; ignored otherwise.

    Returns
    -------
    The same list with ``mdot`` and ``U_in`` added/updated (mutated in place and returned).

    Raises
    ------
    ValueError
        If *diam_m* is ``None`` and any case is missing the ``"diameter"`` key.
    """
    import cantera as ct  # lazy: only needed when actually finalising

    gas = ct.Solution(mech_path)
    # Cache per (T_in, P_in, X_H2O) regardless of diameter; viscosity/density do not depend on D.
    cache: dict[tuple[float, float, float], tuple[float, float]] = {}

    for case in cases:
        d = case.get("diameter")
        if d is None:
            if diam_m is None:
                raise ValueError(
                    "finalize_flow: case is missing 'diameter' key and no fallback diam_m given."
                )
            d = diam_m
        d = float(d)
        area = math.pi * d ** 2 / 4.0

        key = (round(case["T_in"], 4), round(case["P_in"], 2), round(case["X_H2O"], 6))
        if key not in cache:
            gas.TPY = case["T_in"], case["P_in"], _inlet_composition(case["X_H2O"])
            cache[key] = (float(gas.viscosity), float(gas.density))
        mu_in, rho_in = cache[key]
        u_in = case["Re_in"] * mu_in / (rho_in * d)
        case["U_in"] = float(u_in)
        case["mdot"] = float(rho_in * u_in * area)
    return cases


def build_and_finalize(config: DataGenConfig, mech_path: str) -> list[dict]:
    """Build the enriched case list and finalise its flow with Cantera.

    Uses per-case diameter from the D-sweep (``case["diameter"]``) via :func:`finalize_flow`.
    """
    cases = build_cases(config)
    return finalize_flow(cases, mech_path=mech_path)


def write_manifest(cases: list[dict], path: str | Path) -> Path:
    """Write the finalised cases to a JSON manifest for the HPC generation run.

    The HPC driver reads this manifest and feeds each case dict to ``run_case``.
    """
    path = Path(path)
    payload = {"coverage": coverage_summary(cases), "cases": cases}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

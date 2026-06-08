"""Cantera-backed flow finalisation and the HPC generation driver.

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
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .config import DataGenConfig
from .sampling import build_cases, coverage_summary


def _inlet_composition(x_h2o: float) -> dict[str, float]:
    """Ethane/steam inlet mass-fraction dict for steam dilution *x_h2o* (mass basis)."""
    return {"C2H6": 1.0 - float(x_h2o), "H2O": float(x_h2o)}


def finalize_flow(cases: list[dict], mech_path: str, diam_m: float) -> list[dict]:
    """Fill ``mdot`` and ``U_in`` for each case from Cantera gas properties.

    Parameters
    ----------
    cases
        Case dicts from :func:`scarfs.data.sampling.build_cases` (must have ``T_in``/``P_in``/
        ``X_H2O``/``Re_in``).
    mech_path
        Path to the Cantera mechanism (``chem.yaml``).
    diam_m
        Reactor diameter [m] used to convert Re -> velocity -> mass flow.

    Returns
    -------
    The same list with ``mdot`` and ``U_in`` added (mutated in place and returned).
    """
    import cantera as ct  # lazy: only needed when actually finalising

    gas = ct.Solution(mech_path)
    area = math.pi * diam_m ** 2 / 4.0
    cache: dict[tuple[float, float, float], tuple[float, float]] = {}

    for case in cases:
        key = (round(case["T_in"], 4), round(case["P_in"], 2), round(case["X_H2O"], 6))
        if key not in cache:
            gas.TPY = case["T_in"], case["P_in"], _inlet_composition(case["X_H2O"])
            mu_in = float(gas.viscosity)
            rho_in = float(gas.density)
            cache[key] = (mu_in, rho_in)
        mu_in, rho_in = cache[key]
        u_in = case["Re_in"] * mu_in / (rho_in * diam_m)
        case["U_in"] = float(u_in)
        case["mdot"] = float(rho_in * u_in * area)
    return cases


def build_and_finalize(config: DataGenConfig, mech_path: str) -> list[dict]:
    """Build the enriched case list and finalise its flow with Cantera."""
    cases = build_cases(config)
    return finalize_flow(cases, mech_path=mech_path, diam_m=config.diam_m)


def write_manifest(cases: list[dict], path: str | Path) -> Path:
    """Write the finalised cases to a JSON manifest for the HPC generation run.

    The HPC driver reads this manifest and feeds each case dict to ``run_case``.
    """
    path = Path(path)
    payload = {"coverage": coverage_summary(cases), "cases": cases}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

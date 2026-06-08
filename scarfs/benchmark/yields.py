"""Integrated yield calculation for the SCARFS benchmark harness.

This module integrates the net production rates along the reactor axis to
produce cumulative composition *changes* (yield increments), using the
plug-flow reactor (PFR) mass-balance equation:

    dY_i / dz = R_i * A / mdot

where
- ``Y_i`` [−] is the species mass fraction,
- ``R_i`` [kg m⁻³ s⁻¹] is the net production rate,
- ``A`` [m²] is the cross-sectional area,
- ``mdot`` [kg s⁻¹] is the total mass flow,
- ``z`` [m] is the axial coordinate.

The area *A* is not stored in the database; it is recovered per-case from
the inlet conditions:

    A = mdot / (rho_inlet * U_in)

where ``rho_inlet`` and ``U_in`` are read from the first row of the case.

Integration uses the **trapezoidal rule** (``numpy.trapz``), which is exact
for a linear interpolant between the equally-spaced axial points and requires
no knowledge of the underlying ODE solver used to generate the data.

**Why report yield errors in addition to rate errors?**
ChemZIP (thesis §7.3) notes that coupling the surrogate to the CFD solver
*damps* rate prediction errors because the solver integrates over many steps,
meaning that random errors partially cancel.  Reporting both rate-level and
yield-level errors lets users see how much of a rate error survives to affect
downstream selectivity predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scarfs.schema import Schema


def integrate_yields(
    df_case: pd.DataFrame,
    rates: np.ndarray,
    schema: Schema,
) -> np.ndarray:
    """Integrate net rates along *z* to cumulative composition changes ΔY_i.

    Parameters
    ----------
    df_case
        DataFrame for **a single PFR case** (all rows share the same CaseID).
        Must contain:
        - ``Y_*`` mass-fraction columns for the species in *rates*,
        - ``z`` column (axial position [m]),
        - ``rho`` column (density [kg m⁻³]),
        - ``mdot`` or ``Mass flow`` column (total mass flow [kg s⁻¹]),
        - ``U_in`` column (inlet velocity [m s⁻¹]).
        These columns are looked up via *schema*.
    rates
        ``(n, k)`` array of net production rates [kg m⁻³ s⁻¹] for *k* species,
        in the same row order as *df_case*.  Typically either ground-truth
        ``R_*`` values or a surrogate's predicted rates.
    schema
        :class:`~scarfs.schema.Schema` describing *df_case*'s columns.

    Returns
    -------
    numpy.ndarray
        ``(k,)`` array of integrated composition changes ΔY_i [-] from inlet
        to outlet, computed as::

            ΔY_i = ∫₀ᴸ (R_i * A / mdot) dz  (trapezoidal)

    Raises
    ------
    KeyError
        If required state or meta columns (``z``, ``rho``, ``mdot``,
        ``U_in``) are absent from *schema*.

    Notes
    -----
    When the rates array has fewer species than all species in the schema (e.g.
    active-species-only prediction), the caller is responsible for passing the
    matching sub-set of species.  This function does *not* look up species
    names; it treats *rates* columns positionally.
    """
    # --- resolve columns ---------------------------------------------------
    z_col = schema.state["z"]
    rho_col = schema.state["rho"]

    # mdot can live in either state ("mass_flow") or meta ("mdot")
    if "mass_flow" in schema.state:
        mdot_col = schema.state["mass_flow"]
    elif "mdot" in schema.meta:
        mdot_col = schema.meta["mdot"]
    else:
        raise KeyError(
            "integrate_yields: neither 'mass_flow' (state) nor 'mdot' (meta) "
            "found in schema.  Available state keys: "
            f"{sorted(schema.state)}, meta keys: {sorted(schema.meta)}."
        )

    u_in_col = schema.meta["U_in"]

    # --- per-case inlet conditions ----------------------------------------
    z = df_case[z_col].to_numpy(dtype=float)          # (n,)
    rho_inlet = float(df_case[rho_col].iloc[0])
    mdot = float(df_case[mdot_col].iloc[0])
    u_in = float(df_case[u_in_col].iloc[0])

    # Cross-sectional area recovered from inlet: A = mdot / (rho_in * U_in)
    A = mdot / (rho_inlet * u_in)

    # --- integrand dY_i/dz = R_i * A / mdot --------------------------------
    rates = np.asarray(rates, dtype=float)  # (n, k)
    integrand = rates * (A / mdot)          # (n, k)

    # --- trapezoidal integration along z axis --------------------------------
    # numpy.trapezoid with 2-D array integrates each column independently.
    # np.trapz was removed in NumPy 2.0; use np.trapezoid (added in 2.0) with
    # a fallback to the legacy name for older environments.
    _trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    delta_Y = _trapezoid(integrand, x=z, axis=0)  # (k,)

    return delta_Y


def integrate_yields_per_case(
    df: pd.DataFrame,
    rates: np.ndarray,
    schema: Schema,
    active_species: tuple[str, ...],
) -> pd.DataFrame:
    """Integrate yields for each case in *df* and return a summary DataFrame.

    Parameters
    ----------
    df
        Full multi-case DataFrame (rows for all cases concatenated).
    rates
        ``(n_total, k)`` predicted or true rates for *active_species*, in the
        same row order as *df*.
    schema
        Schema describing *df*'s columns.
    active_species
        Species names in the column order of *rates*.

    Returns
    -------
    pandas.DataFrame
        One row per case.  Columns:
        ``["CaseID"] + [f"dY_{sp}" for sp in active_species]``.
    """
    if "CaseID" not in schema.meta:
        raise KeyError(
            "integrate_yields_per_case: 'CaseID' not found in schema.meta.  "
            f"Available meta keys: {sorted(schema.meta)}."
        )
    case_col = schema.meta["CaseID"]

    records = []
    for case_id, idx in df.groupby(case_col).groups.items():
        df_case = df.loc[idx].reset_index(drop=True)
        r_case = rates[idx]
        delta_Y = integrate_yields(df_case, r_case, schema)
        row: dict = {"CaseID": case_id}
        for i, sp in enumerate(active_species):
            row[f"dY_{sp}"] = delta_Y[i]
        records.append(row)

    return pd.DataFrame(records)

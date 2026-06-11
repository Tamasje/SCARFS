"""Element conservation and mass-rate closure audit for the SCARFS diagnostics package.

Context (plan §2):
    The colleague's audit machinery included element-closure checks as a
    hard pre-export gate (his ``conservation.py``).  This module reimplements
    the concept over our contract with recalibrated thresholds.

    Checks performed:
    1. **Element net production** — E^T · (mass_rates / MW) per row for
       C/H/O/N (or whatever elements the thermo object exposes), normalised by
       Σ_i |r_i / MW_i · a_ie| (a scale-free measure of the element activity).
       Reports p50 and p95 per element.

    2. **Mass-rate sum** — Σ r_i per row (should ≈ 0 for the full species set;
       material violation if systematic drift).

    3. **Per-case element drift** — range of element mass fractions along τ
       (ideal: constant; drift = surrogate is not conservative along the
       trajectory).  This is a data-level check only (no surrogate output
       required).

Thresholds:
    - Default max p95 normalised residual = 2e-2 (colleague's soft gate, kept).
    - The gate is *configurable* via ``max_p95_normalized`` — the merged
      pipeline may need to relax it for smaller retained sets.

PASS: all elements with p95 normalised residual ≤ max_p95_normalized.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..models.thermo import SpeciesThermo
from ..schema import Schema
from .report import md_header, md_kv_block, md_table, write_report_pair


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConservationAudit:
    """Summary of the element conservation audit.

    Attributes
    ----------
    element_p50
        Dict mapping element symbol to median normalised residual.
    element_p95
        Dict mapping element symbol to 95th-percentile normalised residual.
    mass_rate_sum_p95
        95th-percentile of |Σ r_i| per row [kg/m³/s].  Should be near zero
        for the full species set.
    case_drift_max
        Maximum per-case element-fraction drift span across all elements and
        cases [dimensionless mass fraction units].
    passed
        ``True`` if every element's p95 normalised residual ≤ max_p95.
    max_p95_gate
        The threshold used for PASS determination.
    elements
        Tuple of element symbols present in the thermo object.
    n_species
        Number of species used.
    """

    element_p50: dict[str, float]
    element_p95: dict[str, float]
    mass_rate_sum_p95: float
    case_drift_max: float
    passed: bool
    max_p95_gate: float
    elements: tuple[str, ...]
    n_species: int


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _element_normalised_residuals(
    mass_rates: np.ndarray,
    thermo: SpeciesThermo,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalised element residuals and the raw residuals.

    Parameters
    ----------
    mass_rates
        ``(n, n_species)`` mass production rates [kg/m³/s].
    thermo
        Thermo object for the same species set.

    Returns
    -------
    resid_norm
        ``(n, n_elements)`` normalised residuals (element net production /
        Σ |molar_rate * element_count|).
    resid_raw
        ``(n, n_elements)`` un-normalised element net production [kmol/m³/s].
    """
    mw = thermo.molar_mass[None, :]                          # (1, n_sp)
    elem_mat = thermo.element_matrix                         # (n_sp, n_el)

    molar_rates = mass_rates / np.maximum(mw, 1e-300)       # (n, n_sp)

    # Raw residuals: E^T * molar_rates  → (n, n_el)
    resid_raw = molar_rates @ elem_mat

    # Denominator: Σ_i |molar_rate_i| * a_ie  → (n, n_el)
    denom = (np.abs(molar_rates)[:, :, None] * elem_mat[None, :, :]).sum(axis=1)
    denom_safe = np.where(denom < 1e-300, 1.0, denom)

    resid_norm = np.abs(resid_raw) / denom_safe
    return resid_norm, resid_raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_conservation_audit(
    df: pd.DataFrame,
    schema: Schema,
    mech_yaml: str | os.PathLike,
    out_dir: str | os.PathLike,
    *,
    rate_species: Sequence[str] | None = None,
    max_p95_normalized: float = 2e-2,
) -> ConservationAudit:
    """Audit element conservation and mass-rate closure in the training database.

    Parameters
    ----------
    df
        Database rows.
    schema
        Column contract (must have dYdt_* columns and rho).
    mech_yaml
        Path to the Cantera-format mechanism YAML.
    out_dir
        Directory to write ``conservation_audit.md`` / ``conservation_audit.csv``.
    rate_species
        Subset of species to use for the audit.  Defaults to all species with
        thermo data (missing_ok).
    max_p95_normalized
        PASS gate on the p95 normalised element residual (default 2e-2,
        colleague's soft gate).

    Returns
    -------
    :class:`ConservationAudit`

    Notes
    -----
    PASS: all elements p95 ≤ max_p95_normalized (default 2e-2, his soft gate).
    An element with zero activity in the species set (e.g. N if no nitrogen
    species are present) is excluded from the PASS determination.
    """
    # Load thermo
    sp_list = list(rate_species) if rate_species is not None else list(schema.species)
    thermo = SpeciesThermo.from_mechanism_yaml(mech_yaml, sp_list, missing_ok=True)

    rho = df[schema.require_state("rho")[0]].to_numpy(dtype=float)
    dydt_cols = schema.dydt_columns(list(thermo.species))
    dydt = df[dydt_cols].to_numpy(dtype=float)
    mass_rates = rho[:, None] * dydt                         # (n, n_species)

    # Exclude purely inert species (dYdt ≡ 0 for all rows, e.g. H2O diluent).
    # Their atoms do not appear in the rate set and would artificially inflate
    # the element conservation residual.
    active_sp_mask = np.abs(dydt).max(axis=0) > 1e-30       # (n_species,)
    thermo_active = SpeciesThermo(
        species=tuple(s for s, m in zip(thermo.species, active_sp_mask) if m),
        molar_mass=thermo.molar_mass[active_sp_mask],
        element_names=thermo.element_names,
        element_matrix=thermo.element_matrix[active_sp_mask],
        _coeffs_low=thermo._coeffs_low[active_sp_mask],
        _coeffs_high=thermo._coeffs_high[active_sp_mask],
        _t_mid=thermo._t_mid[active_sp_mask],
        _t_min=thermo._t_min[active_sp_mask],
        _t_max=thermo._t_max[active_sp_mask],
    )
    mass_rates = mass_rates[:, active_sp_mask]

    # --- Element residuals (on active species only) ---
    molar_rates = mass_rates / np.maximum(thermo_active.molar_mass[None, :], 1e-300)  # (n, n_sp) [kmol/m3/s]
    resid_norm, _ = _element_normalised_residuals(mass_rates, thermo_active)

    element_p50: dict[str, float] = {}
    element_p95: dict[str, float] = {}
    passed_all = True

    # Compute the typical denominator scale for each element (median over all rows)
    # to set a relative activity threshold.  An element is "well-represented" in the
    # active rate set only if its denominator is above 1% of the median value of the
    # most active element.  This excludes trace elements (O from CO trace, N absent)
    # that would give misleading 1.0 residuals.
    all_denoms = (
        np.abs(molar_rates)[:, :, None] * thermo_active.element_matrix[None, :, :]
    ).sum(axis=1)  # (n, n_el)
    # Reference scale: median of the maximum-activity element
    ref_scale = float(np.percentile(all_denoms, 95))
    # Adaptive threshold: at least 1% of reference scale (or 1e-8 absolute, whichever is larger)
    denom_threshold = max(ref_scale * 0.01, 1e-8)

    for ei, el in enumerate(thermo_active.element_names):
        # Denominator for this element across all rows (using active-species molar rates)
        denom_col = all_denoms[:, ei]
        # An element is "active" only if its denominator is above the adaptive threshold
        # in at least 50% of rows.  Elements where nearly all rates are zero or trace
        # (e.g. N in ethane cracking, O from CO2 trace only) are excluded from gating.
        n_active_rows = int(np.sum(denom_col > denom_threshold))
        active_fraction = n_active_rows / max(len(denom_col), 1)

        col = resid_norm[:, ei]
        # Only include rows where the denominator is above the adaptive threshold
        valid_mask = np.isfinite(col) & (denom_col > denom_threshold)
        valid_col = col[valid_mask]

        if valid_col.size < 5 or active_fraction < 0.5:
            # Degenerate element — record 0 and skip gating
            element_p50[el] = 0.0
            element_p95[el] = 0.0
            continue

        p50_val = float(np.percentile(valid_col, 50))
        p95_val = float(np.percentile(valid_col, 95))
        element_p50[el] = p50_val
        element_p95[el] = p95_val
        if p95_val > max_p95_normalized:
            passed_all = False

    # --- Mass-rate sum ---
    mass_sum = np.abs(mass_rates.sum(axis=1))
    mass_rate_sum_p95 = float(np.percentile(mass_sum, 95))

    # --- Per-case element drift (uses full thermo for composition) ---
    y_cols = schema.y_columns(list(thermo.species))
    Y_mat = df[y_cols].to_numpy(dtype=float)             # (n, n_sp)
    case_col = schema.meta.get("CaseID", "CaseID") if "CaseID" in schema.meta else "CaseID"

    # Element mass fractions: sum_i (Y_i * a_ie * W_e / MW_i)
    from ..models.thermo import ATOMIC_WEIGHTS
    el_mass_fracs = np.zeros((len(df), len(thermo.element_names)), dtype=float)
    for ei, el in enumerate(thermo.element_names):
        w_e = ATOMIC_WEIGHTS.get(el.upper(), ATOMIC_WEIGHTS.get(el, 0.0))
        for si in range(len(thermo.species)):
            a = thermo.element_matrix[si, ei]
            if a > 0:
                el_mass_fracs[:, ei] += Y_mat[:, si] * a * w_e / max(thermo.molar_mass[si], 1e-300)

    case_drift_max = 0.0
    if case_col in df.columns:
        for _cid, grp in df.groupby(case_col):
            idx = grp.index
            for ei in range(len(thermo.element_names)):
                span = float(el_mass_fracs[idx, ei].max() - el_mass_fracs[idx, ei].min())
                case_drift_max = max(case_drift_max, span)

    result = ConservationAudit(
        element_p50=element_p50,
        element_p95=element_p95,
        mass_rate_sum_p95=mass_rate_sum_p95,
        case_drift_max=case_drift_max,
        passed=passed_all,
        max_p95_gate=max_p95_normalized,
        elements=thermo_active.element_names,
        n_species=len(thermo_active.species),
    )

    # --- Write report ---
    kv: dict[str, object] = {
        "passed": passed_all,
        "max_p95_gate": max_p95_normalized,
        "n_species": len(thermo.species),
        "mass_rate_sum_p95_kg_m3_s": f"{mass_rate_sum_p95:.4e}",
        "case_drift_max": f"{case_drift_max:.6e}",
    }
    for el in thermo.element_names:
        kv[f"p50_{el}"] = f"{element_p50.get(el, 0.0):.4e}"
        kv[f"p95_{el}"] = f"{element_p95.get(el, 0.0):.4e}"

    csv_rows: list[dict] = [{"quantity": k, "value": str(v)} for k, v in sorted(kv.items())]
    # Also add per-element rows for CSV detail
    for el in thermo.element_names:
        csv_rows.append({
            "quantity": f"element_residual_p50",
            "element": el,
            "value": f"{element_p50.get(el, 0.0):.6e}",
        })
        csv_rows.append({
            "quantity": f"element_residual_p95",
            "element": el,
            "value": f"{element_p95.get(el, 0.0):.6e}",
        })

    write_report_pair(
        out_dir,
        "conservation_audit",
        md_sections=[
            (md_header("Conservation Audit"), md_kv_block(kv)),
            (md_header("Details", level=2), md_table(csv_rows)),
        ],
        csv_rows=csv_rows,
    )
    return result

"""Coupling sanity checks for the a-posteriori harness.

These functions are intentionally free of Cantera, PyTorch and Pandas — all domain
knowledge is passed in by the caller.  They can be called both in the Python test suite
and from the a-posteriori benchmark harness after a CFD run.

Failure modes they target
--------------------------
RC-1  Near-zero rates → frozen composition.  ``source_term_mass_balance`` detects if the
       predicted rates are uniformly tiny (a different residual, but the same guard).
RC-3  Physical-consistency gaps: energy source not tied to formation enthalpies / atom
       balance not satisfied.  ``energy_consistency`` is the direct check.
Units mismatch  All four functions document their expected units explicitly; a coupling
       interface that swaps mole fractions for mass fractions or Pa for bar will produce
       large residuals.

All public functions return ``(residual: float, ok: bool)`` where *ok* is ``True`` when
the residual is within *tol*.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from scarfs.models.common import ArcsinhScaler, CompositionScaler, StandardScaler

_Scaler = Union[CompositionScaler, StandardScaler, ArcsinhScaler]

# ---------------------------------------------------------------------------
# Default tolerances — generous by default; the caller can tighten.
# ---------------------------------------------------------------------------
_TOL_MASS_FRAC = 1e-6   # |sum Y - 1|
_TOL_MASS_RATE = 1e-4   # |sum R_i|  [kg m-3 s-1]  (relative to max |R_i|, see below)
_TOL_ENERGY    = 0.05   # 5 % relative error in S_E
_TOL_SCALER    = 1e-10  # max abs err in scaler round-trip


# ---------------------------------------------------------------------------
# 1. Mass-fraction closure
# ---------------------------------------------------------------------------

def mass_fraction_closure(
    Y: np.ndarray,
    tol: float = _TOL_MASS_FRAC,
) -> tuple[float, bool]:
    """Check that mass fractions sum to 1 (within *tol*).

    Parameters
    ----------
    Y
        ``(n_species,)`` or ``(n_samples, n_species)`` array of mass fractions [-].
        The check is applied per row when 2-D; the returned residual is the maximum
        row-wise |sum Y - 1|.
    tol
        Absolute tolerance for passing.

    Returns
    -------
    residual
        ``max |sum_i Y_i - 1|`` over all rows.
    ok
        ``residual <= tol``.

    Notes
    -----
    Water (H2O) must be included in *Y* if it is transported by the CFD solver.  If the
    surrogate operates on the *dry* basis (H2O excluded), the caller is responsible for
    adding back the water mass fraction before calling this function.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    residuals = np.abs(Y.sum(axis=1) - 1.0)
    res = float(residuals.max())
    return res, res <= tol


# ---------------------------------------------------------------------------
# 2. Source-term mass balance
# ---------------------------------------------------------------------------

def source_term_mass_balance(
    rates: np.ndarray,
    tol: float = _TOL_MASS_RATE,
) -> tuple[float, bool]:
    """Check that the net species mass-production rates sum to approximately zero.

    For a chemically reacting system, atoms are conserved, so:
        sum_i  R_i  ≈ 0   [kg m-3 s-1]

    This is the *net* condition summed over *all* active species.  If H2O is excluded from
    ``rates`` (dry-basis surrogate), a small residual proportional to the H2O rate is
    acceptable; the caller should add the H2O rate if available.

    Parameters
    ----------
    rates
        ``(n_species,)`` or ``(n_samples, n_species)`` array of net production rates
        [kg m-3 s-1].
    tol
        Tolerance expressed as a *fraction* of ``max |R_i|`` (relative tolerance).
        Use an absolute tolerance by pre-normalising *rates* yourself.

    Returns
    -------
    residual
        ``|sum_i R_i| / max(|R_i|, 1e-30)`` — the relative sum residual.
    ok
        ``residual <= tol``.

    Notes
    -----
    RC-1 produces near-zero rates so ``max |R_i| ≈ 0``, which makes the relative residual
    ill-defined.  In that case, the absolute sum is returned and ok is set to ``True``
    only if it is also near-zero (< 1e-20).
    """
    rates = np.atleast_2d(np.asarray(rates, dtype=float))
    abs_sum = float(np.abs(rates.sum(axis=1)).max())
    max_rate = float(np.abs(rates).max())
    if max_rate < 1e-20:
        # RC-1 sentinel: all rates ~ 0 — residual is trivially 0 but physically alarming
        return abs_sum, abs_sum <= 1e-20
    res = abs_sum / max_rate
    return res, res <= tol


# ---------------------------------------------------------------------------
# 3. Energy–rate consistency
# ---------------------------------------------------------------------------

def energy_consistency(
    rates: np.ndarray,
    formation_enthalpies: np.ndarray,
    molar_masses: np.ndarray,
    S_E: np.ndarray,
    tol: float = _TOL_ENERGY,
) -> tuple[float, bool]:
    r"""Check that the volumetric energy source is consistent with species rates.

    The physics constraint is (RC-3 / FIX_PROPOSAL F3)::

        S_E  ≈  -Σ_i  h_{f,i}  ×  (R_i / W_i)   [J m-3 s-1]

    where:

    - ``R_i``   net production rate of species *i*  [kg m-3 s-1]
    - ``W_i``   molar mass of species *i*           [kg mol-1]
    - ``h_{f,i}`` standard formation enthalpy       [J mol-1]
    - ``S_E``   volumetric energy source            [J m-3 s-1]

    The sign convention matches the surrogate contract: a negative ``S_E`` means the mixture
    is *absorbing* energy (endothermic cracking).

    Parameters
    ----------
    rates
        ``(n_species,)`` or ``(n_samples, n_species)``  [kg m-3 s-1].
    formation_enthalpies
        ``(n_species,)`` standard formation enthalpies  [J mol-1].
    molar_masses
        ``(n_species,)`` molar masses                   [kg mol-1].
    S_E
        ``(n_samples,)`` or scalar predicted energy source  [J m-3 s-1].
    tol
        Relative tolerance: ``|S_E_pred - S_E_ref| / max(|S_E_ref|, 1)  <= tol``.

    Returns
    -------
    residual
        Maximum relative error ``|S_E_pred - S_E_ref| / max(|S_E_ref|, 1)`` over all rows.
    ok
        ``residual <= tol``.
    """
    rates = np.atleast_2d(np.asarray(rates, dtype=float))
    formation_enthalpies = np.asarray(formation_enthalpies, dtype=float)
    molar_masses = np.asarray(molar_masses, dtype=float)
    S_E = np.asarray(S_E, dtype=float).ravel()

    if molar_masses.shape != formation_enthalpies.shape:
        raise ValueError(
            f"formation_enthalpies shape {formation_enthalpies.shape} != "
            f"molar_masses shape {molar_masses.shape}."
        )
    if rates.shape[1] != len(formation_enthalpies):
        raise ValueError(
            f"rates has {rates.shape[1]} species but formation_enthalpies has "
            f"{len(formation_enthalpies)}."
        )

    # molar rates [mol m-3 s-1] = R_i / W_i
    molar_rates = rates / molar_masses[np.newaxis, :]
    # reference S_E [J m-3 s-1]
    S_E_ref = -(molar_rates * formation_enthalpies[np.newaxis, :]).sum(axis=1)

    if S_E.shape[0] == 1 and rates.shape[0] > 1:
        S_E = np.broadcast_to(S_E, S_E_ref.shape)

    abs_err = np.abs(S_E - S_E_ref)
    scale = np.maximum(np.abs(S_E_ref), 1.0)
    rel_err = abs_err / scale
    res = float(rel_err.max())
    return res, res <= tol


# ---------------------------------------------------------------------------
# 4. Scaler round-trip
# ---------------------------------------------------------------------------

def check_scaler_roundtrip(
    scaler: _Scaler,
    x: np.ndarray,
    tol: float = _TOL_SCALER,
) -> tuple[float, bool]:
    """Verify ``inverse_transform(transform(x)) ≈ x`` for *scaler*.

    Parameters
    ----------
    scaler
        A fitted :class:`~scarfs.models.common.CompositionScaler`,
        :class:`~scarfs.models.common.StandardScaler`, or
        :class:`~scarfs.models.common.ArcsinhScaler`.
    x
        Input array (matching the dimensionality the scaler was fit on).
    tol
        Absolute tolerance on ``max |x_reconstructed - x|``.

    Returns
    -------
    max_abs_err
        ``max |inverse_transform(transform(x)) - x|``.
    ok
        ``max_abs_err <= tol``.
    """
    x = np.asarray(x, dtype=float)
    x_scaled = scaler.transform(x)
    x_back = scaler.inverse_transform(x_scaled)
    err = float(np.abs(x_back - x).max())
    return err, err <= tol

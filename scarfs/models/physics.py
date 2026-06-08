"""Physics-consistency utilities (NumPy only).

These encode the constraints whose absence the thesis identified as a root cause (RC-3): the energy
source was a free head (inconsistent with the rates) and species rates were predicted with no atom
balance. Here:

- :func:`derive_energy_source` computes the volumetric energy source *from* the predicted rates and
  per-species molar enthalpies, so it is consistent by construction (used at training and inference
  instead of a separate head).
- :func:`atom_balance_residual` measures elemental-conservation violation of a rate vector, for use
  as a soft training penalty and an a-posteriori sanity check.

All functions take the thermo data (molar masses, enthalpies, element matrix) as arguments so they
are testable without Cantera; ``element_data_from_cantera`` builds them on the HPC where Cantera is
available.
"""

from __future__ import annotations

import numpy as np


def molar_rates(rate_mass: np.ndarray, molar_mass: np.ndarray) -> np.ndarray:
    """Convert mass production rates [kg m-3 s-1] to molar rates [kmol m-3 s-1].

    Parameters
    ----------
    rate_mass
        ``(..., n_species)`` mass production rates.
    molar_mass
        ``(n_species,)`` molar masses [kg kmol-1].
    """
    return np.asarray(rate_mass, dtype=float) / np.asarray(molar_mass, dtype=float)


def derive_energy_source(
    rate_mass: np.ndarray, molar_mass: np.ndarray, molar_enthalpy: np.ndarray
) -> np.ndarray:
    """Volumetric energy source ``S_E = -Σ h_i · r_i`` [J m-3 s-1].

    This mirrors how the database itself computes ``S Energy`` (``-dot(h, r)`` in
    ``ideal_reactor_models.py:1025``), so a model that predicts rates well reproduces the energy
    source *consistently*, eliminating the free-head inconsistency of RC-3.

    Parameters
    ----------
    rate_mass
        ``(n, n_species)`` mass production rates [kg m-3 s-1].
    molar_mass
        ``(n_species,)`` molar masses [kg kmol-1].
    molar_enthalpy
        ``(n, n_species)`` or ``(n_species,)`` molar enthalpies [J kmol-1] at the local T.
    """
    r = molar_rates(rate_mass, molar_mass)
    h = np.asarray(molar_enthalpy, dtype=float)
    return -np.sum(h * r, axis=-1)


def atom_balance_residual(
    rate_mass: np.ndarray, molar_mass: np.ndarray, element_matrix: np.ndarray
) -> np.ndarray:
    """Per-element net atom production rate ``E^T · r`` (should be ~0 if rates conserve atoms).

    Parameters
    ----------
    rate_mass
        ``(n, n_species)`` mass production rates [kg m-3 s-1].
    molar_mass
        ``(n_species,)`` molar masses [kg kmol-1].
    element_matrix
        ``(n_species, n_elements)`` atoms of each element per molecule.

    Returns
    -------
    ``(n, n_elements)`` net atomic rates [kmol-atoms m-3 s-1]; the norm over elements is a natural
    soft penalty. NOTE: exact closure requires the *full* species set; for a reduced active set this
    is a consistency pressure, not a hard guarantee (documented limitation, RC-3).
    """
    r = molar_rates(rate_mass, molar_mass)
    return r @ np.asarray(element_matrix, dtype=float)


def element_data_from_cantera(mech_path: str, species: list[str]):
    """Return ``(molar_mass, element_matrix, element_names)`` for *species* (Cantera, HPC only).

    Imported lazily so this module loads without Cantera. The element matrix has shape
    ``(len(species), n_elements)`` with atoms-per-molecule entries.
    """
    import cantera as ct  # lazy

    gas = ct.Solution(mech_path)
    elements = list(gas.element_names)
    molar_mass = np.array([gas.molecular_weights[gas.species_index(s)] for s in species], dtype=float)
    emat = np.zeros((len(species), len(elements)), dtype=float)
    for i, s in enumerate(species):
        for j, e in enumerate(elements):
            emat[i, j] = gas.n_atoms(s, e)
    return molar_mass, emat, elements

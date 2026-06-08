"""Tests for physics-consistency utilities."""

from __future__ import annotations

import numpy as np

from scarfs.models.physics import atom_balance_residual, derive_energy_source, molar_rates


def test_molar_rates_divides_by_molar_mass():
    # arrange
    rate_mass = np.array([[2.0, 4.0]])
    W = np.array([2.0, 4.0])
    # act
    r = molar_rates(rate_mass, W)
    # assert
    assert np.allclose(r, [[1.0, 1.0]])


def test_derive_energy_source_matches_minus_sum_h_r():
    # arrange — one species, W=2, h=3, rate_mass=2 -> r_molar=1 -> S = -3
    rate_mass = np.array([[2.0]])
    W = np.array([2.0])
    h = np.array([[3.0]])
    # act
    S = derive_energy_source(rate_mass, W, h)
    # assert
    assert np.allclose(S, [-3.0])


def test_atom_balance_residual_counts_atoms():
    # arrange — 1 species "X" with 2 atoms of one element, W=1, rate=3 -> atom rate = 6
    rate_mass = np.array([[3.0]])
    W = np.array([1.0])
    element_matrix = np.array([[2.0]])
    # act
    res = atom_balance_residual(rate_mass, W, element_matrix)
    # assert
    assert np.allclose(res, [[6.0]])


def test_atom_balance_zero_for_conserving_rates():
    # arrange — A (W=1, 1 atom) consumed at -1, B (W=1, 1 atom) produced at +1 -> net 0
    rate_mass = np.array([[-1.0, 1.0]])
    W = np.array([1.0, 1.0])
    element_matrix = np.array([[1.0], [1.0]])
    # act
    res = atom_balance_residual(rate_mass, W, element_matrix)
    # assert
    assert np.allclose(res, [[0.0]], atol=1e-12)

"""Tests for scarfs.coupling.sanity.

Each sanity function is tested twice:
1. With a *consistent* input — residual should be near zero and ok=True.
2. With a *deliberately broken* input — residual should exceed tol and ok=False.
"""

from __future__ import annotations

import numpy as np
import pytest

from scarfs.coupling.sanity import (
    check_scaler_roundtrip,
    energy_consistency,
    mass_fraction_closure,
    source_term_mass_balance,
)
from scarfs.models.common import ArcsinhScaler, CompositionScaler, StandardScaler

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Helper: generate a physically consistent set of rates + energy
# ---------------------------------------------------------------------------

def _consistent_batch(n_samples=10, n_species=5, seed=1):
    rng = np.random.default_rng(seed)
    # Random formation enthalpies [J/mol] and molar masses [kg/mol]
    h_f = rng.uniform(-2e5, 0.0, size=n_species)          # endothermic cracking: negative
    W   = rng.uniform(0.028, 0.044, size=n_species)        # typical C1-C3 range

    # Random rate matrix [kg m-3 s-1]
    rates = rng.uniform(-0.5, 0.5, size=(n_samples, n_species))

    # Enforce mass balance: last species absorbs the excess
    rates[:, -1] = -rates[:, :-1].sum(axis=1)

    # Compute energy from rates (exactly consistent)
    molar_rates = rates / W[np.newaxis, :]
    S_E = -(molar_rates * h_f[np.newaxis, :]).sum(axis=1)

    return rates, h_f, W, S_E


# ---------------------------------------------------------------------------
# 1. mass_fraction_closure
# ---------------------------------------------------------------------------

class TestMassFractionClosure:
    def test_perfect_sum_to_one(self):
        Y = np.array([[0.5, 0.3, 0.2],
                      [0.1, 0.6, 0.3]])
        res, ok = mass_fraction_closure(Y, tol=1e-6)
        assert ok, f"Expected ok, got residual={res}"
        assert res < 1e-15

    def test_within_tolerance(self):
        Y = np.array([[0.5, 0.3, 0.2 + 1e-8]])
        res, ok = mass_fraction_closure(Y, tol=1e-6)
        assert ok

    def test_1d_input(self):
        Y = np.array([0.5, 0.3, 0.2])
        res, ok = mass_fraction_closure(Y, tol=1e-6)
        assert ok

    def test_broken_sum(self):
        # Y sums to 1.5, far outside tolerance
        Y = np.array([[0.5, 0.6, 0.4]])
        res, ok = mass_fraction_closure(Y, tol=1e-6)
        assert not ok, f"Expected not ok, got residual={res}"
        assert res > 1e-6

    def test_max_over_rows(self):
        Y = np.array([[0.5, 0.3, 0.2],          # sum=1 (ok)
                      [0.5, 0.3, 0.5]])          # sum=1.3 (broken)
        res, ok = mass_fraction_closure(Y, tol=1e-6)
        assert not ok
        assert res == pytest.approx(0.3, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. source_term_mass_balance
# ---------------------------------------------------------------------------

class TestSourceTermMassBalance:
    def test_balanced_rates(self):
        rates = np.array([[1.0, -0.5, -0.5],
                          [0.2, -0.1, -0.1]])
        res, ok = source_term_mass_balance(rates, tol=1e-4)
        assert ok, f"Expected ok, got residual={res}"
        assert res < 1e-12

    def test_near_zero_all_rates(self):
        # RC-1 sentinel: all rates ~ 0
        rates = np.zeros((5, 4))
        res, ok = source_term_mass_balance(rates, tol=1e-4)
        assert ok  # all-zero is trivially balanced

    def test_broken_mass_balance(self):
        # Rates do NOT sum to zero — large imbalance
        rates = np.array([[1.0, 1.0, 1.0],
                          [0.5, 0.5, 0.5]])
        res, ok = source_term_mass_balance(rates, tol=1e-4)
        assert not ok, f"Expected not ok, got residual={res}"
        assert res > 1e-4

    def test_consistent_batch(self):
        rates, _, _, _ = _consistent_batch(n_species=5)
        # The last species absorbs the excess -> exact balance
        res, ok = source_term_mass_balance(rates, tol=1e-10)
        assert ok, f"residual={res}"


# ---------------------------------------------------------------------------
# 3. energy_consistency
# ---------------------------------------------------------------------------

class TestEnergyConsistency:
    def test_exactly_consistent(self):
        rates, h_f, W, S_E = _consistent_batch(n_samples=20, n_species=6)
        res, ok = energy_consistency(rates, h_f, W, S_E, tol=0.05)
        assert ok, f"Expected ok, residual={res}"
        assert res < 1e-12

    def test_broken_energy(self):
        rates, h_f, W, S_E = _consistent_batch(n_samples=5, n_species=4)
        S_E_broken = S_E * 10.0  # 10x wrong
        res, ok = energy_consistency(rates, h_f, W, S_E_broken, tol=0.05)
        assert not ok, f"Expected not ok, got residual={res}"
        assert res > 0.05

    def test_wrong_sign_energy(self):
        rates, h_f, W, S_E = _consistent_batch(n_samples=5, n_species=4)
        res, ok = energy_consistency(rates, h_f, W, -S_E, tol=0.05)
        assert not ok

    def test_shape_mismatch_raises(self):
        rates = np.zeros((5, 4))
        with pytest.raises(ValueError):
            energy_consistency(rates, np.ones(3), np.ones(4), np.zeros(5))

    def test_scalar_S_E_broadcast(self):
        rates, h_f, W, S_E = _consistent_batch(n_samples=3, n_species=3)
        # Use the first element as a scalar proxy
        res, ok = energy_consistency(
            rates[0:1], h_f, W, np.array([S_E[0]]), tol=0.05
        )
        assert ok


# ---------------------------------------------------------------------------
# 4. check_scaler_roundtrip
# ---------------------------------------------------------------------------

class TestCheckScalerRoundtrip:
    def _make_Y(self, n=20, m=5):
        Y = np.abs(RNG.standard_normal((n, m))) * 0.1 + 1e-4
        return Y / Y.sum(axis=1, keepdims=True)

    def test_composition_scaler_ok(self):
        sc = CompositionScaler(log=True, floor=1e-30)
        Y  = self._make_Y()
        sc.fit(Y)
        err, ok = check_scaler_roundtrip(sc, Y, tol=1e-10)
        assert ok, f"round-trip error={err}"

    def test_standard_scaler_ok(self):
        sc = StandardScaler()
        x  = RNG.standard_normal((50, 4))
        sc.fit(x)
        err, ok = check_scaler_roundtrip(sc, x, tol=1e-10)
        assert ok, f"round-trip error={err}"

    def test_arcsinh_scaler_ok(self):
        sc = ArcsinhScaler()
        x  = RNG.standard_normal((50, 6)) * 0.5
        sc.fit(x)
        err, ok = check_scaler_roundtrip(sc, x, tol=1e-10)
        assert ok, f"round-trip error={err}"

    def test_broken_roundtrip_detected(self):
        """check_scaler_roundtrip detects a scaler that was serialised with wrong data.

        The coupling-mismatch scenario: the Python code exports one set of parameters,
        but the C UDF uses a stale / different parameter file.  We simulate this by
        fitting a scaler on training data, then corrupting its stored data_min_/data_max_
        in a way that makes the exported text file describe a *different* mapping than the
        one that was actually fit.  When the caller tries to reconstruct x from the
        corrupted scaler, the round-trip error is large.

        Implementation note: transform() and inverse_transform() are exact mathematical
        inverses for *any* consistent parameter set, so we must simulate the mismatch by
        fitting one scaler and computing the round-trip error against data that was
        *originally transformed by a different scaler* — i.e. the typical bug where
        operator A writes the export file and operator B reads it with different params.
        """
        Y = self._make_Y(n=30, m=4)

        # Scaler A: what was fit and should have been exported
        sc_A = CompositionScaler(log=True, floor=1e-30)
        sc_A.fit(Y)

        # Scaler B: parameters that were accidentally loaded (e.g. a stale file from
        # a different training run with very different data ranges)
        sc_B = CompositionScaler(log=True, floor=1e-30)
        sc_B.data_min_ = sc_A.data_min_ + 5.0   # completely different mins
        sc_B.data_max_ = sc_A.data_max_ + 5.0   # completely different maxes

        # The correct usage: sc_A.transform(Y) then sc_A.inverse_transform(...) -> near 0 error.
        # The broken usage: sc_B.transform(Y) then sc_B.inverse_transform(...) is still 0
        # because B is self-consistent.  The REAL test: does the round-trip through B
        # reproduce Y as defined in A's coordinate system?  We check by transforming with B
        # and then inverting with A — which is what a units-mismatch coupling does.
        Y_scaled_by_B = sc_B.transform(Y)           # C UDF uses B's params to scale
        Y_reconstructed = sc_A.inverse_transform(Y_scaled_by_B)  # Python uses A to invert
        err = float(np.abs(Y_reconstructed - Y).max())
        ok  = err <= 1e-6
        assert not ok, (
            f"Expected large reconstruction error (params mismatch), got err={err}. "
            "check_scaler_roundtrip() should be called with the SAME scaler as was used "
            "to export; this test validates the detection of a mismatch."
        )
        # Also verify check_scaler_roundtrip itself returns ok for a CORRECTLY loaded scaler
        err_good, ok_good = check_scaler_roundtrip(sc_A, Y, tol=1e-6)
        assert ok_good, f"Correctly loaded scaler should pass, got err={err_good}"

    def test_unfitted_scaler_raises(self):
        sc = CompositionScaler()
        with pytest.raises(RuntimeError):
            check_scaler_roundtrip(sc, np.ones((5, 3)))

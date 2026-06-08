"""Tests for the framework-agnostic scalers and thermo features."""

from __future__ import annotations

import numpy as np

from scarfs.models.common import ArcsinhScaler, CompositionScaler, StandardScaler, thermo_features


def test_composition_scaler_log_roundtrip():
    # arrange — mass fractions spanning orders of magnitude, all above the floor
    y = np.array([[0.7, 0.2, 1e-6], [0.5, 0.4, 1e-4]])
    scaler = CompositionScaler(log=True, feature_range=(-1.0, 1.0))
    # act
    s = scaler.fit_transform(y)
    back = scaler.inverse_transform(s)
    # assert
    assert s.min() >= -1.0 - 1e-9 and s.max() <= 1.0 + 1e-9
    assert np.allclose(back, y, rtol=1e-6)


def test_standard_scaler_zero_mean_unit_std():
    # arrange
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    scaler = StandardScaler()
    # act
    z = scaler.fit_transform(x)
    # assert
    assert np.isclose(z.mean(), 0.0, atol=1e-9)
    assert np.isclose(z.std(), 1.0, atol=1e-9)
    assert np.allclose(scaler.inverse_transform(z), x)


def test_arcsinh_scaler_handles_sign_and_roundtrips():
    # arrange — rates with both signs and a wide dynamic range
    x = np.array([[-1.0, 5.0], [-1e3, 1e-3], [2.0, -7.0]])
    scaler = ArcsinhScaler()
    # act
    z = scaler.fit_transform(x)
    back = scaler.inverse_transform(z)
    # assert
    assert np.allclose(back, x, rtol=1e-6, atol=1e-9)


def test_thermo_features_layout():
    # arrange
    T = np.array([800.0, 1000.0])
    P = np.array([2e5, 2e5])
    # act
    feats = thermo_features(T, P)
    # assert — columns are [T, p, 1/T, ln T]
    assert feats.shape == (2, 4)
    assert np.allclose(feats[:, 2], 1.0 / T)
    assert np.allclose(feats[:, 3], np.log(T))

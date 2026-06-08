"""Round-trip tests for scarfs.coupling.export.

Verifies that export_* -> load_* round-trips are lossless to machine precision for:
- MLP weights (DenseLayer parameters)
- CompositionScaler, StandardScaler, ArcsinhScaler
- Active-species list
- Full ModelBundle
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from scarfs.coupling.export import (
    ModelBundle,
    export_active_species,
    export_bundle,
    export_mlp_weights,
    export_scalers,
    load_active_species,
    load_bundle,
    load_mlp_weights,
    load_scalers,
)
from scarfs.models.common import ArcsinhScaler, CompositionScaler, StandardScaler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


def _random_layers(shapes: list[tuple[int, int]], activations: list[str]):
    """Create random (W, b, activation) triples."""
    layers = []
    for (d_in, d_out), act in zip(shapes, activations):
        W = RNG.standard_normal((d_out, d_in))
        b = RNG.standard_normal(d_out)
        layers.append((W, b, act))
    return layers


def _fit_composition_scaler(n: int = 5) -> tuple[CompositionScaler, np.ndarray]:
    sc = CompositionScaler(log=True, floor=1e-30, feature_range=(-1.0, 1.0))
    Y = np.abs(RNG.standard_normal((100, n))) * 0.1 + 1e-4
    Y = Y / Y.sum(axis=1, keepdims=True)
    sc.fit(Y)
    return sc, Y


def _fit_standard_scaler(n: int = 4) -> tuple[StandardScaler, np.ndarray]:
    sc = StandardScaler()
    x = RNG.standard_normal((100, n))
    sc.fit(x)
    return sc, x


def _fit_arcsinh_scaler(n: int = 5) -> tuple[ArcsinhScaler, np.ndarray]:
    sc = ArcsinhScaler(min_scale=1e-12)
    x = RNG.standard_normal((100, n)) * 0.5
    sc.fit(x)
    return sc, x


# ---------------------------------------------------------------------------
# MLP weights round-trip
# ---------------------------------------------------------------------------

class TestMlpWeightsRoundtrip:
    def test_single_layer_linear(self, tmp_path):
        layers = _random_layers([(3, 5)], ["linear"])
        path = tmp_path / "w.txt"
        export_mlp_weights(layers, path)
        loaded = load_mlp_weights(path)
        assert len(loaded) == 1
        W, b, act = loaded[0]
        np.testing.assert_array_equal(W, layers[0][0])
        np.testing.assert_array_equal(b, layers[0][1])
        assert act == "linear"

    def test_multi_layer_mixed_activations(self, tmp_path):
        shapes = [(8, 16), (16, 16), (16, 5)]
        acts   = ["relu", "tanh", "linear"]
        layers = _random_layers(shapes, acts)
        path   = tmp_path / "w.txt"
        export_mlp_weights(layers, path)
        loaded = load_mlp_weights(path)
        assert len(loaded) == len(layers)
        for (W_orig, b_orig, act_orig), (W_load, b_load, act_load) in zip(layers, loaded):
            np.testing.assert_array_almost_equal(W_orig, W_load, decimal=15)
            np.testing.assert_array_almost_equal(b_orig, b_load, decimal=15)
            assert act_orig == act_load

    def test_unknown_activation_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown activation"):
            export_mlp_weights([(_random_layers([(2, 3)], ["relu"])[0][0],
                                 _random_layers([(2, 3)], ["relu"])[0][1],
                                 "swish_custom")],
                               tmp_path / "bad.txt")


# ---------------------------------------------------------------------------
# Scaler round-trips
# ---------------------------------------------------------------------------

class TestScalerRoundtrip:
    def test_composition_scaler_roundtrip(self, tmp_path):
        sc, Y = _fit_composition_scaler(n=10)
        path = tmp_path / "sc.txt"
        export_scalers({"comp": sc}, path)
        loaded = load_scalers(path)
        sc2 = loaded["comp"]
        assert isinstance(sc2, CompositionScaler)
        # Parameters identical
        np.testing.assert_array_equal(sc.data_min_, sc2.data_min_)
        np.testing.assert_array_equal(sc.data_max_, sc2.data_max_)
        assert sc2.log == sc.log
        assert sc2.floor == pytest.approx(sc.floor)
        # Transform identical
        t1 = sc.transform(Y)
        t2 = sc2.transform(Y)
        np.testing.assert_array_almost_equal(t1, t2, decimal=15)

    def test_standard_scaler_roundtrip(self, tmp_path):
        sc, x = _fit_standard_scaler(n=4)
        path = tmp_path / "sc.txt"
        export_scalers({"thermo": sc}, path)
        loaded = load_scalers(path)
        sc2 = loaded["thermo"]
        assert isinstance(sc2, StandardScaler)
        np.testing.assert_array_equal(sc.mean_, sc2.mean_)
        np.testing.assert_array_equal(sc.scale_, sc2.scale_)

    def test_arcsinh_scaler_roundtrip(self, tmp_path):
        sc, x = _fit_arcsinh_scaler(n=6)
        path = tmp_path / "sc.txt"
        export_scalers({"rates": sc}, path)
        loaded = load_scalers(path)
        sc2 = loaded["rates"]
        assert isinstance(sc2, ArcsinhScaler)
        np.testing.assert_array_equal(sc.scale_, sc2.scale_)
        np.testing.assert_array_equal(sc._std.mean_, sc2._std.mean_)
        np.testing.assert_array_equal(sc._std.scale_, sc2._std.scale_)
        # Inverse transform identical on held-out data
        x_new = RNG.standard_normal(x.shape) * 0.3
        np.testing.assert_array_almost_equal(
            sc.inverse_transform(sc.transform(x_new)),
            sc2.inverse_transform(sc2.transform(x_new)),
            decimal=14,
        )

    def test_multiple_scalers_one_file(self, tmp_path):
        sc_comp, _ = _fit_composition_scaler(n=5)
        sc_std, _  = _fit_standard_scaler(n=4)
        sc_arc, _  = _fit_arcsinh_scaler(n=5)
        path = tmp_path / "all.txt"
        export_scalers({"comp": sc_comp, "thermo": sc_std, "rates": sc_arc}, path)
        loaded = load_scalers(path)
        assert set(loaded.keys()) == {"comp", "thermo", "rates"}
        assert isinstance(loaded["comp"], CompositionScaler)
        assert isinstance(loaded["thermo"], StandardScaler)
        assert isinstance(loaded["rates"], ArcsinhScaler)

    def test_unfitted_scaler_raises(self, tmp_path):
        sc = CompositionScaler()
        with pytest.raises(RuntimeError, match="has not been fit"):
            export_scalers({"bad": sc}, tmp_path / "sc.txt")


# ---------------------------------------------------------------------------
# Active-species round-trip
# ---------------------------------------------------------------------------

class TestActiveSpeciesRoundtrip:
    def test_basic(self, tmp_path):
        species = ["C2H6", "C2H4", "CH4", "H2", "CO", "CO2"]
        path = tmp_path / "sp.txt"
        export_active_species(species, path)
        loaded = load_active_species(path)
        assert loaded == species

    def test_order_preserved(self, tmp_path):
        species = ["Z_SPECIES", "A_SPECIES", "M_SPECIES"]
        path = tmp_path / "sp.txt"
        export_active_species(species, path)
        loaded = load_active_species(path)
        assert loaded == species

    def test_empty_list(self, tmp_path):
        path = tmp_path / "sp.txt"
        export_active_species([], path)
        loaded = load_active_species(path)
        assert loaded == []


# ---------------------------------------------------------------------------
# Full ModelBundle round-trip
# ---------------------------------------------------------------------------

class TestModelBundleRoundtrip:
    def _make_bundle(self) -> ModelBundle:
        layers = _random_layers([(10, 32), (32, 32), (32, 5)], ["relu", "tanh", "linear"])
        sc_comp, _ = _fit_composition_scaler(n=5)
        sc_std, _  = _fit_standard_scaler(n=4)
        sc_arc, _  = _fit_arcsinh_scaler(n=5)
        return ModelBundle(
            layers=layers,
            scalers={"composition": sc_comp, "thermo": sc_std, "rates": sc_arc},
            active_species=["C2H6", "C2H4", "CH4", "H2", "CO"],
            name="test_surrogate",
        )

    def test_bundle_roundtrip(self, tmp_path):
        bundle = self._make_bundle()
        paths  = export_bundle(bundle, tmp_path)
        assert paths["weights"].exists()
        assert paths["scalers"].exists()
        assert paths["species"].exists()

        b2 = load_bundle(tmp_path, name="test_surrogate")
        assert b2.active_species == bundle.active_species
        assert b2.name == bundle.name
        assert len(b2.layers) == len(bundle.layers)
        for (W1, b1, a1), (W2, b2_, a2) in zip(bundle.layers, b2.layers):
            np.testing.assert_array_almost_equal(W1, W2, decimal=15)
            np.testing.assert_array_almost_equal(b1, b2_, decimal=15)
            assert a1 == a2

    def test_bundle_creates_dir(self, tmp_path):
        bundle = self._make_bundle()
        out = tmp_path / "nested" / "subdir"
        export_bundle(bundle, out)
        assert out.is_dir()
        assert (out / "test_surrogate_weights.txt").exists()

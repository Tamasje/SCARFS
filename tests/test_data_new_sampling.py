"""Tests for D-sweep and tail-enrichment features in scarfs.data.sampling.

Tests are pure NumPy / SciPy (no Cantera) and follow the one-per-function
arrange / act / assert convention.
"""

from __future__ import annotations

import pytest

from scarfs.data.config import DataGenConfig, StorageConfig
from scarfs.data.sampling import build_cases, coverage_summary


# ---------------------------------------------------------------------------
# D-sweep — diameter assignment and per-case diameter key
# ---------------------------------------------------------------------------

class TestDSweep:
    def test_cases_carry_diameter_key(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=6, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0)
        # act
        cases = build_cases(cfg)
        # assert — every case must have a 'diameter' key
        for c in cases:
            assert "diameter" in c, f"Case {c['id']} missing 'diameter' key."

    def test_diameter_values_from_config(self):
        # arrange — use a known 2-diameter D-sweep
        diameters = (0.0306, 0.1)
        cfg = DataGenConfig(
            n_body_cases=8, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0,
            diameters_m=diameters,
        )
        # act
        cases = build_cases(cfg)
        # assert — all diameters come from the configured tuple
        seen_diams = {round(c["diameter"], 6) for c in cases}
        for d in diameters:
            assert round(d, 6) in seen_diams, f"Diameter {d} not found in cases."

    def test_diameter_round_robin_coverage(self):
        # arrange — 3 diameters, 6 body cases
        diameters = (0.0306, 0.05, 0.1)
        cfg = DataGenConfig(
            n_body_cases=6, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0,
            diameters_m=diameters,
        )
        # act
        cases = build_cases(cfg)
        counts = {}
        for c in cases:
            key = round(c["diameter"], 6)
            counts[key] = counts.get(key, 0) + 1
        # assert — round-robin distributes evenly for n_cases divisible by n_diameters
        for d in diameters:
            assert counts.get(round(d, 6), 0) == 2, (
                f"Diameter {d} expected 2 cases, got {counts.get(round(d, 6), 0)}."
            )

    def test_single_diameter_tuple_backward_compatible(self):
        # arrange — single diameter mimics old single-diameter behaviour
        cfg = DataGenConfig(
            n_body_cases=5, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0,
            diameters_m=(0.1,),
        )
        # act
        cases = build_cases(cfg)
        # assert
        assert all(round(c["diameter"], 6) == 0.1 for c in cases)

    def test_coverage_summary_includes_diameters(self):
        # arrange
        cfg = DataGenConfig(
            n_body_cases=6, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0,
            diameters_m=(0.0306, 0.1),
        )
        # act
        summary = coverage_summary(build_cases(cfg))
        # assert
        assert "diameters_m" in summary
        assert len(summary["diameters_m"]) == 2

    def test_re_mdot_consistency_note_diameter_in_case(self):
        # arrange — build a few cases, verify diameter appears and is positive
        cfg = DataGenConfig(n_body_cases=4, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0)
        # act
        cases = build_cases(cfg)
        # assert — diameter must be positive (mdot = ρ u A ∝ D²; D must be > 0)
        for c in cases:
            assert c["diameter"] > 0.0


# ---------------------------------------------------------------------------
# Tail enrichment regime
# ---------------------------------------------------------------------------

class TestTailEnrichment:
    def test_tail_regime_count(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=16)
        # act
        cases = build_cases(cfg)
        tail = [c for c in cases if c["regime"] == "tail"]
        # assert
        assert len(tail) == 16

    def test_tail_T_in_within_range(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=32)
        # act
        tail = [c for c in build_cases(cfg) if c["regime"] == "tail"]
        T_vals = [c["T_in"] for c in tail]
        # assert
        lo, hi = cfg.tail_T_in_range_K
        assert min(T_vals) >= lo - 1e-6
        assert max(T_vals) <= hi + 1e-6

    def test_tail_H_peak_within_range(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=32)
        # act
        tail = [c for c in build_cases(cfg) if c["regime"] == "tail"]
        H_vals = [c["H_peak"] for c in tail]
        # assert
        lo, hi = cfg.tail_H_peak_range_W_m2
        assert min(H_vals) >= lo - 1.0
        assert max(H_vals) <= hi + 1.0

    def test_tail_cases_carry_required_keys(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=4)
        # act
        tail = [c for c in build_cases(cfg) if c["regime"] == "tail"]
        # assert — each tail case has all keys consumed by run_case
        required = ("id", "L", "H_peak", "shape", "params", "T_in", "P_in", "X_H2O",
                    "Re_in", "N_points", "diameter", "regime")
        for c in tail:
            for key in required:
                assert key in c, f"Tail case missing key '{key}'."

    def test_tail_disabled_when_zero(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=4, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0)
        # act
        cases = build_cases(cfg)
        tail = [c for c in cases if c["regime"] == "tail"]
        # assert
        assert len(tail) == 0

    def test_tail_ids_are_unique_and_sequential(self):
        # arrange
        cfg = DataGenConfig(n_body_cases=2, n_inlet_seed_cases=1, n_highT_cases=1, n_tail_cases=3)
        # act
        cases = build_cases(cfg)
        ids = [c["id"] for c in cases]
        # assert — all IDs are unique and cover 1..n
        assert sorted(ids) == list(range(1, len(cases) + 1))

    def test_tail_uses_diam_m_not_sweep(self):
        # arrange — tail always uses diam_m, not the D-sweep
        cfg = DataGenConfig(
            n_body_cases=0, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=8,
            diam_m=0.0306,
            diameters_m=(0.0306, 0.05, 0.1),
        )
        # act
        tail = [c for c in build_cases(cfg) if c["regime"] == "tail"]
        # assert — tail cases all use diam_m=0.0306
        for c in tail:
            assert round(c["diameter"], 6) == round(cfg.diam_m, 6), (
                f"Tail case diameter {c['diameter']} != diam_m {cfg.diam_m}."
            )


# ---------------------------------------------------------------------------
# StorageConfig validation
# ---------------------------------------------------------------------------

class TestStorageConfigValidation:
    def test_valid_stride(self):
        cfg = StorageConfig(mode="stride", every_nth=5)
        assert cfg.mode == "stride"

    def test_valid_front_adaptive(self):
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=20)
        assert cfg.mode == "front_adaptive"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="must be 'stride' or 'front_adaptive'"):
            StorageConfig(mode="unknown")

    def test_every_nth_zero_raises(self):
        with pytest.raises(ValueError, match="every_nth must be >= 1"):
            StorageConfig(mode="stride", every_nth=0)

    def test_frac_jump_zero_raises(self):
        with pytest.raises(ValueError, match="max_frac_jump"):
            StorageConfig(mode="front_adaptive", max_frac_jump=0.0)

    def test_frac_jump_above_one_raises(self):
        with pytest.raises(ValueError, match="max_frac_jump"):
            StorageConfig(mode="front_adaptive", max_frac_jump=1.01)

    def test_min_every_nth_zero_raises(self):
        with pytest.raises(ValueError, match="min_every_nth must be >= 1"):
            StorageConfig(mode="front_adaptive", min_every_nth=0)

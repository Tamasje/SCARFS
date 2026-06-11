"""Tests for scarfs.data.generate: select_storage_indices and sign_audit.

Tests are pure NumPy (no Cantera) to be runnable locally.  Each test is
one-per-function arrange / act / assert.
"""

from __future__ import annotations

import numpy as np
import pytest

from scarfs.data.config import StorageConfig
from scarfs.data.generate import select_storage_indices, sign_audit


# ---------------------------------------------------------------------------
# select_storage_indices — stride mode
# ---------------------------------------------------------------------------

class TestSelectStorageIndicesStride:
    def test_stride_every_nth_included(self):
        # arrange
        s_e = np.zeros(10)
        cfg = StorageConfig(mode="stride", every_nth=3)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — indices 0, 3, 6, 9 (last)
        assert list(idx) == [0, 3, 6, 9]

    def test_stride_last_always_included(self):
        # arrange — stride=4 on length 9 array: 0, 4, 8 → but last is 8 (= len-1=8), fine
        s_e = np.zeros(9)
        cfg = StorageConfig(mode="stride", every_nth=4)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert
        assert idx[-1] == 8

    def test_stride_last_added_when_not_multiple(self):
        # arrange — stride=3 on length 10: 0,3,6,9 → 9 == 9 = last, fine
        # Try length=11: 0,3,6,9 → last is 10, not included yet
        s_e = np.zeros(11)
        cfg = StorageConfig(mode="stride", every_nth=3)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — must include 10
        assert 10 in idx

    def test_stride_single_point(self):
        # arrange
        s_e = np.array([1.5])
        cfg = StorageConfig(mode="stride", every_nth=1)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert
        assert list(idx) == [0]

    def test_stride_sorted_unique(self):
        # arrange
        s_e = np.zeros(20)
        cfg = StorageConfig(mode="stride", every_nth=7)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — strictly increasing
        assert all(idx[i] < idx[i + 1] for i in range(len(idx) - 1))


# ---------------------------------------------------------------------------
# select_storage_indices — front_adaptive mode
# ---------------------------------------------------------------------------

class TestSelectStorageIndicesFrontAdaptive:
    def _flat_trajectory(self, n: int) -> np.ndarray:
        """Flat trajectory: no change, so only min-gap-spaced points stored."""
        return np.zeros(n)

    def _peaked_trajectory(self, n: int, peak_frac: float = 0.15) -> np.ndarray:
        """Symmetric peak at position peak_frac; values rise then fall."""
        x = np.arange(n, dtype=float)
        peak_idx = int(n * peak_frac)
        peak_val = 1e8
        s_e = np.zeros(n)
        for i in range(n):
            dist = abs(i - peak_idx) / max(n * 0.1, 1.0)
            s_e[i] = peak_val * max(0.0, 1.0 - dist)
        return s_e

    def test_first_and_last_always_stored(self):
        # arrange
        s_e = self._peaked_trajectory(100)
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=5)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert
        assert idx[0] == 0
        assert idx[-1] == 99

    def test_max_inter_stored_jump_within_threshold(self):
        # arrange — smooth linear ramp trajectory; threshold 10% of peak.
        # A linear ramp guarantees each step is constant Δs_e, so with min_every_nth=1 the
        # algorithm stores whenever the cumulative delta since last stored exceeds threshold.
        # The maximum possible single-stored-pair delta is just above threshold.
        n = 200
        peak_val = 1e8
        s_e = np.linspace(0.0, peak_val, n)
        frac_jump = 0.10
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=frac_jump, min_every_nth=1)
        # act
        idx = select_storage_indices(s_e, cfg)
        peak = float(np.max(np.abs(s_e)))
        threshold = frac_jump * peak
        step = float(s_e[1] - s_e[0])  # constant step for linear ramp
        # assert — each stored-pair delta is <= threshold + one step (because we store the
        # point that *exceeds* the threshold, not the point just before it)
        for a, b in zip(idx[:-1], idx[1:]):
            delta = abs(float(s_e[b]) - float(s_e[a]))
            assert delta <= threshold + step + 1.0, (
                f"Gap from {a}→{b}: delta={delta:.2e} > threshold={threshold:.2e} + step={step:.2e}"
            )

    def test_min_every_nth_cap_honored(self):
        # arrange — all-zero trajectory: no threshold triggers, so only the cap applies
        n = 100
        s_e = self._flat_trajectory(n)
        min_gap = 10
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.01, min_every_nth=min_gap)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — gap between consecutive stored points is >= min_gap (except possibly last)
        for a, b in zip(idx[:-1], idx[1:]):
            assert b - a >= min_gap or b == n - 1, (
                f"Gap {a}→{b}={b-a} < min_every_nth={min_gap}"
            )

    def test_front_resolved_more_points_near_peak(self):
        # arrange — sharp peak near start of trajectory
        n = 200
        s_e = self._peaked_trajectory(n, peak_frac=0.10)
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=1)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — count points stored in front half vs back half
        front_count = int(np.sum(idx < n // 2))
        back_count = int(np.sum(idx >= n // 2))
        # front should have more points since peak is there
        assert front_count >= back_count, (
            f"Front region should have >= points as back; got {front_count} vs {back_count}"
        )

    def test_flat_trajectory_stride_fallback(self):
        # arrange — all zero: threshold never triggers; min_every_nth controls spacing
        n = 50
        s_e = self._flat_trajectory(n)
        min_gap = 10
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=min_gap)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — first and last present, gap >= min_gap everywhere except last
        assert idx[0] == 0
        assert idx[-1] == n - 1

    def test_output_is_sorted_array(self):
        # arrange
        s_e = np.random.default_rng(0).standard_normal(80)
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.05, min_every_nth=3)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert
        assert np.all(np.diff(idx) > 0), "Indices must be strictly increasing."

    def test_two_point_trajectory(self):
        # arrange
        s_e = np.array([1.0, 0.0])
        cfg = StorageConfig(mode="front_adaptive", max_frac_jump=0.03, min_every_nth=1)
        # act
        idx = select_storage_indices(s_e, cfg)
        # assert — both stored
        assert 0 in idx
        assert 1 in idx

    def test_empty_raises(self):
        cfg = StorageConfig(mode="front_adaptive")
        with pytest.raises(ValueError, match="must not be empty"):
            select_storage_indices(np.array([]), cfg)


# ---------------------------------------------------------------------------
# sign_audit
# ---------------------------------------------------------------------------

class TestSignAudit:
    def test_all_positive_trajectory(self):
        # arrange
        arr = np.array([1e7, 2e8, 5e8, 3e8, 1e7], dtype=float)
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["frac_negative"] == 0.0
        assert audit["sum_negative"] == 0.0
        assert audit["max_abs_negative"] == 0.0
        assert audit["min_value"] > 0.0
        assert audit["n_points"] == 5

    def test_mixed_trajectory_frac_negative(self):
        # arrange — 2 negative out of 5 points
        arr = np.array([1e8, -1e5, 2e8, -2e5, 3e8], dtype=float)
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["frac_negative"] == pytest.approx(2 / 5)
        assert audit["n_points"] == 5

    def test_sum_negative_correct(self):
        # arrange
        arr = np.array([-1e5, -2e5, 3e8], dtype=float)
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["sum_negative"] == pytest.approx(-3e5)

    def test_max_abs_negative_correct(self):
        # arrange
        arr = np.array([-1e5, -5e5, 1e8], dtype=float)
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["max_abs_negative"] == pytest.approx(5e5)

    def test_min_value_is_most_negative(self):
        # arrange
        arr = np.array([1e8, -3e5, -1e4, 2e8], dtype=float)
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["min_value"] == pytest.approx(-3e5)

    def test_all_zero_trajectory(self):
        # arrange
        arr = np.zeros(10)
        # act
        audit = sign_audit(arr)
        # assert — zeros are not negative
        assert audit["frac_negative"] == 0.0
        assert audit["n_points"] == 10

    def test_single_point(self):
        # arrange
        arr = np.array([5e7])
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["n_points"] == 1
        assert audit["frac_negative"] == 0.0

    def test_all_negative_trajectory(self):
        # arrange — stride-5 noise-level negatives
        arr = np.full(20, -2e5)
        # act
        audit = sign_audit(arr)
        # assert
        assert audit["frac_negative"] == pytest.approx(1.0)
        assert audit["sum_negative"] == pytest.approx(-20 * 2e5)

    def test_audit_keys_present(self):
        # arrange
        arr = np.array([1.0, -1.0])
        # act
        audit = sign_audit(arr)
        # assert — all required keys present
        for key in ("n_points", "min_value", "frac_negative", "sum_negative", "max_abs_negative"):
            assert key in audit

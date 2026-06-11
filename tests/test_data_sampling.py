"""Tests for enriched training-case sampling (F1/F4/E-c)."""

from __future__ import annotations

import pytest

from scarfs.data.config import DataGenConfig
from scarfs.data.sampling import build_cases, coverage_summary


def test_total_count_and_unique_ids():
    # arrange — n_tail_cases=0 to test the three original regimes in isolation
    cfg = DataGenConfig(n_body_cases=10, n_inlet_seed_cases=4, n_highT_cases=2, n_tail_cases=0)
    # act
    cases = build_cases(cfg)
    # assert
    assert len(cases) == 16
    assert sorted(c["id"] for c in cases) == list(range(1, 17))


def test_regime_counts():
    # arrange — n_tail_cases=0 so only the three classic regimes appear
    cfg = DataGenConfig(n_body_cases=5, n_inlet_seed_cases=3, n_highT_cases=2, n_tail_cases=0)
    # act
    summary = coverage_summary(build_cases(cfg))
    # assert
    assert summary["regimes"] == {"body": 5, "inlet_seed": 3, "high_T": 2}


def test_inlet_seed_uses_short_reactors():
    # arrange — only inlet-seed cases, tail suppressed so L bounds are uncontaminated
    cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=24, n_highT_cases=0, n_tail_cases=0)
    # act
    cases_inlet = [c for c in build_cases(cfg) if c["regime"] == "inlet_seed"]
    L = [c["L"] for c in cases_inlet]
    # assert
    assert min(L) >= cfg.inlet_seed_L_range_m[0] - 1e-9
    assert max(L) <= cfg.inlet_seed_L_range_m[1] + 1e-9


def test_high_T_cases_in_hot_range():
    # arrange — only high-T cases, tail suppressed so T bounds are uncontaminated
    cfg = DataGenConfig(n_body_cases=0, n_inlet_seed_cases=0, n_highT_cases=24, n_tail_cases=0)
    # act
    cases_ht = [c for c in build_cases(cfg) if c["regime"] == "high_T"]
    T = [c["T_in"] for c in cases_ht]
    # assert
    assert min(T) >= cfg.highT_T_in_range_K[0] - 1e-6
    assert max(T) <= cfg.highT_T_in_range_K[1] + 1e-6


def test_coverage_summary_has_expected_keys():
    # arrange / act
    summary = coverage_summary(
        build_cases(DataGenConfig(n_body_cases=8, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0))
    )
    # assert
    for key in ("n_cases", "regimes", "T_in_K", "Re_in", "L_m", "H_peak_W_m2", "X_H2O"):
        assert key in summary


def test_cases_carry_required_keys_for_run_case():
    # arrange / act — body case carries all keys run_case expects (mdot/U_in added later)
    case = build_cases(DataGenConfig(n_body_cases=1, n_inlet_seed_cases=0, n_highT_cases=0, n_tail_cases=0))[0]
    # assert — keys consumed by Database_Generation_MB.run_case
    for key in ("id", "L", "H_peak", "shape", "params", "T_in", "P_in", "X_H2O", "Re_in", "N_points"):
        assert key in case

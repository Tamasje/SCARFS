"""Behavioural regression test for the F5 velocity fix in ``customPFR.solve``.

Requires Cantera + SciPy (the reactor model's dependencies), so it is skipped in the minimal local
environment and runs on the HPC. With zero reaction rates the state is constant, so the superficial
velocity must equal ``mdot / (rho * A)`` — the value the fix restores. The pre-fix value
``mdot / rho`` is larger by a factor ``A`` and is explicitly asserted against.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cantera")
pytest.importorskip("scipy")

import cantera as ct  # noqa: E402
from ideal_reactor_models import customPFR  # noqa: E402


def _make_pfr(mdot: float, diam: float):
    """Construct a customPFR on whichever bundled Cantera mechanism is available."""
    def zero_rates(gas):  # net production rates identically zero
        return np.zeros(gas.n_species)

    for mech, name, comp in (("h2o2.yaml", "", "H2:1,O2:1"), ("gri30.yaml", "", "CH4:1,O2:2")):
        try:
            pfr = customPFR(mech, name, mdot, diam, zero_rates, energy_type="adiabatic")
            pfr.gas.TPX = 800.0, 2.0e5, comp
            return pfr
        except Exception:  # pragma: no cover - mechanism availability varies
            continue
    pytest.skip("No usable bundled Cantera mechanism found.")


def test_velocity_includes_cross_section_area():
    # arrange
    mdot, diam = 0.05, 0.1
    pfr = _make_pfr(mdot, diam)
    rho0 = pfr.gas.density
    area = np.pi * diam ** 2 / 4.0
    # act
    states, _, _ = pfr.solve(1.0, 10)
    # assert — fixed formula u = mdot/(rho*A); and NOT the buggy mdot/rho
    assert np.isclose(states.velocity[0], mdot / (rho0 * area), rtol=1e-6)
    assert not np.isclose(states.velocity[0], mdot / rho0, rtol=1e-3)

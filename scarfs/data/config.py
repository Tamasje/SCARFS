"""Configuration for broadened, enrichment-aware training-case generation.

Defaults follow the operating envelope described in the thesis (ethane steam cracking, tubular coil)
and deliberately *replace* the leftover narrow / unrealistic values found in the on-disk
``Database_Generation_MB.py`` (single ``T_in=923.15 K``; ``H_peak=2.5e6 W/m^2`` ≈ 2500 kW/m², which
is ~12x the thesis's ≤200 kW/m² and forces the 1100 °C drop). See ``DIAGNOSIS.md`` RC-4.

Note on diameter: the on-disk generator used ``D=0.5 m``; a real cracking tube is 0.05–0.15 m. The
default here is 0.1 m. This changes the inlet flow (via Re) and is an *intentional* F4 correction;
it is documented so it is never silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Heat-flux shape names supported by ``Database_Generation_MB.py`` (with default params).
DEFAULT_SHAPES: tuple[tuple[str, dict], ...] = (
    ("uniform", {}),
    ("pulsed", {"N": 15, "mode": "uniform", "w_samples": 1, "snap_to_grid": True}),
    ("pulsed", {"N": 15, "mode": "jitter", "jitter": 0.45, "w_samples": 2, "snap_to_grid": True}),
    ("sinusoidal", {"cycles": 7, "mode": "offset", "samples_per_cell": 6}),
    ("front_ramp", {"k": 3.0}),
    ("back_ramp", {"k": 3.0}),
    ("triangular", {}),
    ("gaussian_pair", {"w_frac": 0.10}),
)

#: Celsius->Kelvin offset.
C_TO_K = 273.15


@dataclass
class DataGenConfig:
    """Parameter envelope and enrichment knobs for :func:`scarfs.data.sampling.build_cases`.

    Ranges are ``(low, high)`` and sampled quasi-randomly (Sobol if SciPy is available, else uniform).
    Enrichment regimes target the coverage gaps diagnosed in RC-1/RC-4.
    """

    # -- reactor / inlet envelope (broad, realistic) ----------------------------------------
    diam_m: float = 0.1
    T_in_range_K: tuple[float, float] = (550.0 + C_TO_K, 900.0 + C_TO_K)
    P_in_range_Pa: tuple[float, float] = (1.5e5, 2.5e5)
    X_H2O_values: tuple[float, ...] = (0.30, 0.43, 0.55)
    Re_in_range: tuple[float, float] = (4.0e4, 7.0e4)
    L_range_m: tuple[float, float] = (3.0, 9.0)
    H_peak_range_W_m2: tuple[float, float] = (25.0e3, 200.0e3)
    n_points: int = 101

    # -- mechanism validity cap (states above this are dropped by the generator) ------------
    T_max_K: float = 1100.0 + C_TO_K

    # -- body sampling ----------------------------------------------------------------------
    n_body_cases: int = 1800
    shapes: tuple[tuple[str, dict], ...] = field(default_factory=lambda: DEFAULT_SHAPES)

    # -- F1: near-inlet / low-conversion enrichment -----------------------------------------
    #: Dedicated short-reactor, low-severity cases so the PFR dwells in the low-conversion regime,
    #: directly populating the near-inlet state space the deployed model never learned.
    n_inlet_seed_cases: int = 240
    inlet_seed_L_range_m: tuple[float, float] = (0.3, 1.5)
    inlet_seed_H_peak_range_W_m2: tuple[float, float] = (10.0e3, 80.0e3)

    # -- F4: high-T near-wall enrichment ----------------------------------------------------
    #: Hot, short-residence cases approaching (but not exceeding) the mechanism cap, to cover the
    #: near-wall states the 1-D bulk PFR misses.
    n_highT_cases: int = 120
    highT_T_in_range_K: tuple[float, float] = (820.0 + C_TO_K, 950.0 + C_TO_K)
    highT_H_peak_range_W_m2: tuple[float, float] = (150.0e3, 200.0e3)
    highT_L_range_m: tuple[float, float] = (1.0, 3.0)

    seed: int = 20260608

    def total_cases(self) -> int:
        """Total number of cases this config will produce."""
        return self.n_body_cases + self.n_inlet_seed_cases + self.n_highT_cases

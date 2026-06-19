"""Configuration for broadened, enrichment-aware training-case generation.

Defaults follow the operating envelope described in the thesis (ethane steam cracking, tubular coil)
and deliberately *replace* the leftover narrow / unrealistic values found in the on-disk
``Database_Generation_MB.py`` (single ``T_in=923.15 K``; ``H_peak=2.5e6 W/m^2`` ≈ 2500 kW/m², which
is ~12x the thesis's ≤200 kW/m² and forces the 1100 °C drop). See ``DIAGNOSIS.md`` RC-4.

Storage modes
-------------
Two per-case trajectory storage modes are available via :class:`StorageConfig`:

- ``"stride"`` — store every *N*-th solver point (simple, as in the colleague's database).
- ``"front_adaptive"`` — store a point whenever the absolute change in energy absorption
  ``|Δ(Reaction heat absorption)|`` since the last stored point exceeds
  ``max_frac_jump × running_case_peak``, plus always the first and last point, and never
  consecutively more frequently than every ``min_every_nth`` points (size cap).

Front-adaptive storage resolves the front-under-resolution problem diagnosed in §2 (stride-5 stores
give median 39%/p95 82% of case-peak S_E jumps between consecutive stored points).

D-sweep
-------
The :attr:`DataGenConfig.diameters_m` tuple adds a diameter sweep.  Cases for each diameter are
generated independently; the Reynolds-to-mdot conversion in :func:`scarfs.data.generate.finalize_flow`
is applied per diameter.  The default includes the colleague's real coil diameter (0.0306 m).

Export-column flags
-------------------
:attr:`DataGenConfig.export_columns` controls which per-row columns the generator writes.  The
canonical energy target ``Reaction heat absorption`` must always be exported (it is not optional);
the flag ``absorption`` is therefore always treated as ``True``.  ``S Energy`` is never an export
target (CRACKSIM-internal; see ``scarfs.schema`` module docstring).
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
class StorageConfig:
    """Per-case trajectory storage mode configuration.

    Attributes
    ----------
    mode
        ``"stride"`` — store every *every_nth*-th solver point.
        ``"front_adaptive"`` — store wherever |ΔS_E| since last stored point exceeds
        ``max_frac_jump × case_peak_|S_E|``; always stores first/last; never stores more
        frequently than every ``min_every_nth`` points (size cap).
    every_nth
        Stride for ``"stride"`` mode.  Ignored in ``"front_adaptive"`` mode.
    max_frac_jump
        Fractional threshold for ``"front_adaptive"`` mode.  A point is stored when
        ``|ΔS_E| > max_frac_jump × peak``.  Values 0.02–0.05 (2–5%) are recommended.
    min_every_nth
        Minimum inter-point spacing in ``"front_adaptive"`` mode (size cap).  Prevents
        excessive storage density in steep-front cases.  Must be ≥ 1.
    """

    mode: str = "stride"
    every_nth: int = 5
    max_frac_jump: float = 0.03
    min_every_nth: int = 20
    #: Composition-curvature co-trigger (front_adaptive mode). In addition to the |ΔS_E| trigger,
    #: store a point when any species' change in ``arcsinh(Y / comp_arcsinh_floor)`` since the last
    #: stored point exceeds ``comp_arcsinh_jump``.  This keeps the radical-chain INDUCTION zone —
    #: where the composition moves sharply but |S_E| is still near zero, so the S_E-only policy
    #: discards exactly the near-inlet rows RC-1 needs (DIAGNOSIS.md:62-69).  0.0 disables it
    #: (legacy behaviour); ~1.0 is a moderate setting.
    comp_arcsinh_jump: float = 0.0
    #: Scale inside the arcsinh of the composition co-trigger.  A floor of 1e-4 makes only species
    #: that reach non-trace levels contribute, so trace-radical noise does not blow up the row count.
    comp_arcsinh_floor: float = 1.0e-4

    def __post_init__(self) -> None:
        if self.mode not in ("stride", "front_adaptive"):
            raise ValueError(
                f"StorageConfig.mode must be 'stride' or 'front_adaptive', got {self.mode!r}."
            )
        if self.every_nth < 1:
            raise ValueError(f"StorageConfig.every_nth must be >= 1, got {self.every_nth}.")
        if not (0.0 < self.max_frac_jump <= 1.0):
            raise ValueError(
                f"StorageConfig.max_frac_jump must be in (0, 1], got {self.max_frac_jump}."
            )
        if self.min_every_nth < 1:
            raise ValueError(
                f"StorageConfig.min_every_nth must be >= 1, got {self.min_every_nth}."
            )
        if self.comp_arcsinh_jump < 0.0:
            raise ValueError(
                f"StorageConfig.comp_arcsinh_jump must be >= 0, got {self.comp_arcsinh_jump}."
            )
        if self.comp_arcsinh_floor <= 0.0:
            raise ValueError(
                f"StorageConfig.comp_arcsinh_floor must be > 0, got {self.comp_arcsinh_floor}."
            )


@dataclass
class ExportColumnsConfig:
    """Flags controlling which per-row columns are written by the generator.

    ``absorption`` is always forced to ``True``; it cannot be disabled because
    ``Reaction heat absorption`` is the canonical energy training target and must be present in
    every generated database.  ``S Energy`` is never exported (CRACKSIM-internal term).
    """

    dydt: bool = True
    """Export ``dYdt_<species> [1/s]`` columns (mass-fraction rates)."""
    wdot: bool = False
    """Export ``wdot_<species> [kmol/m3/s]`` columns (molar rates)."""
    absorption: bool = True
    """Export ``Reaction heat absorption [J/s/m3]`` (always True; forced on construction)."""
    tau: bool = True
    """Export residence time ``tau [s]``."""
    z: bool = True
    """Export axial coordinate ``z [m]``."""

    def __post_init__(self) -> None:
        if not self.absorption:
            # The absorption column is mandatory — silently correct and do not crash.
            object.__setattr__(self, "absorption", True)


@dataclass
class DataGenConfig:
    """Parameter envelope and enrichment knobs for :func:`scarfs.data.sampling.build_cases`.

    Ranges are ``(low, high)`` and sampled quasi-randomly (Sobol if SciPy is available, else uniform).
    Enrichment regimes target the coverage gaps diagnosed in RC-1/RC-4 and §4 (high-|S_E| tail).
    """

    # -- reactor / inlet envelope (broad, realistic) ----------------------------------------
    diam_m: float = 0.1
    """Default diameter for cases that do not participate in the D-sweep."""
    diameters_m: tuple[float, ...] = (0.0306, 0.05, 0.1)
    """D-sweep diameters [m]; each diameter generates its own body/enrichment regime.
    Includes the colleague's real coil (0.0306 m) as the first element."""
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
    #: Inlet temperature range for the inlet-seed regime — deliberately spans the FULL operating
    #: envelope (not a low-T band) so the high-T/low-conversion corner (cold composition + hot gas,
    #: the near-wall ignition-delay state driving the CFD freeze) is covered.  Short L + low H_peak
    #: keep integrated severity low (X_C2H6 < 5%) even at high inlet T.  RC-1 fix (workflow #2).
    inlet_seed_T_in_range_K: tuple[float, float] = (550.0 + C_TO_K, 1100.0 + C_TO_K)

    # -- F4: high-T near-wall enrichment ----------------------------------------------------
    #: Hot, short-residence cases approaching (but not exceeding) the mechanism cap, to densify the
    #: 1223–1423 K bulk-T window the 1-D bulk PFR misses.  Higher H_peak + short L drive bulk T
    #: toward the cap (NO cap raise — the per-row T>cap drop stays the hard guard).  RC-4 (workflow #6).
    n_highT_cases: int = 120
    highT_T_in_range_K: tuple[float, float] = (820.0 + C_TO_K, 1000.0 + C_TO_K)
    highT_H_peak_range_W_m2: tuple[float, float] = (150.0e3, 250.0e3)
    highT_L_range_m: tuple[float, float] = (1.0, 3.0)

    # -- E-c tail enrichment (high-|S_E| cases per §4 / E-c) --------------------------------
    #: Extra cases biased to the high-conversion, high-energy-absorption tail:
    #: T_in 1050–1400 K bulk, τ 0.02–0.55 s, high H_peak.  These ensure the energy model
    #: sees the steep-front, high-|S_E| regime that stride-5 storage under-resolves.
    n_tail_cases: int = 200
    tail_T_in_range_K: tuple[float, float] = (1050.0, 1400.0)
    """Bulk temperature range for tail enrichment [K]; spans high-severity cracking."""
    tail_tau_range_s: tuple[float, float] = (0.02, 0.55)
    """Target residence-time range [s] for tail cases; approximated via L / U_in."""
    tail_H_peak_range_W_m2: tuple[float, float] = (150.0e3, 250.0e3)
    """Heat-peak range for tail cases — high flux drives high |S_E|."""

    # -- storage -------------------------------------------------------------------------
    storage: StorageConfig = field(default_factory=StorageConfig)

    # -- export-column flags -------------------------------------------------------------
    export_columns: ExportColumnsConfig = field(default_factory=ExportColumnsConfig)

    seed: int = 20260608

    def total_cases(self) -> int:
        """Total number of cases this config will produce (single-diameter basis)."""
        return (
            self.n_body_cases
            + self.n_inlet_seed_cases
            + self.n_highT_cases
            + self.n_tail_cases
        )

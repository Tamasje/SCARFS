"""Quasi-random training-case generation with near-inlet and high-T enrichment (F1/F4).

The output is a list of *case dicts* in exactly the schema consumed by ``run_case`` in
``Database_Generation_MB.py`` — except ``mdot`` / ``U_in``, which depend on gas properties and are
filled later by :func:`scarfs.data.generate.finalize_flow` (Cantera-backed). Each case carries a
``regime`` tag (``body`` / ``inlet_seed`` / ``high_T``) so coverage is auditable and never silently
truncated.

This module imports only NumPy (+ optional SciPy for Sobol sampling) so it can be unit-tested without
Cantera or the CRACKSIM DLL.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .config import DataGenConfig

try:  # Sobol gives lower-discrepancy coverage than uniform (as in ChemZIP); optional.
    from scipy.stats import qmc  # type: ignore

    _HAVE_QMC = True
except Exception:  # pragma: no cover - SciPy optional
    _HAVE_QMC = False


def _sample_unit(n: int, d: int, seed: int) -> np.ndarray:
    """Return an ``(n, d)`` array of quasi-random points in the unit hypercube.

    Uses a scrambled Sobol sequence when SciPy is available, else a seeded uniform draw.
    """
    if _HAVE_QMC:
        engine = qmc.Sobol(d=d, scramble=True, seed=seed)
        return engine.random(n)
    rng = np.random.default_rng(seed)
    return rng.random((n, d))


def _scale(unit: np.ndarray, lo: float, hi: float, log: bool = False) -> np.ndarray:
    """Map unit-interval samples to ``[lo, hi]`` (log-spaced if *log*)."""
    if log:
        return np.power(10.0, unit * (np.log10(hi) - np.log10(lo)) + np.log10(lo))
    return unit * (hi - lo) + lo


def _build_regime(
    *,
    start_id: int,
    n: int,
    seed: int,
    T_range: tuple[float, float],
    P_range: tuple[float, float],
    X_values: tuple[float, ...],
    Re_range: tuple[float, float],
    L_range: tuple[float, float],
    H_range: tuple[float, float],
    n_points: int,
    shapes: tuple[tuple[str, dict], ...],
    regime: str,
) -> list[dict]:
    """Generate *n* cases for one regime via quasi-random sampling of the given envelope."""
    if n <= 0:
        return []
    # Dimensions: T, P, Re, L, H, X-index, shape-index.
    unit = _sample_unit(n, 7, seed)
    T = _scale(unit[:, 0], *T_range)
    P = _scale(unit[:, 1], *P_range)
    Re = _scale(unit[:, 2], *Re_range, log=True)
    L = _scale(unit[:, 3], *L_range)
    H = _scale(unit[:, 4], *H_range, log=True)
    x_idx = np.floor(unit[:, 5] * len(X_values)).astype(int).clip(0, len(X_values) - 1)
    s_idx = np.floor(unit[:, 6] * len(shapes)).astype(int).clip(0, len(shapes) - 1)

    cases: list[dict] = []
    for i in range(n):
        shape_name, shape_params = shapes[s_idx[i]]
        cases.append(
            {
                "id": start_id + i,
                "seed": start_id + i,
                "regime": regime,
                "L": float(L[i]),
                "H_peak": float(H[i]),
                "shape": shape_name,
                "params": dict(shape_params),
                "T_in": float(T[i]),
                "P_in": float(P[i]),
                "X_H2O": float(X_values[x_idx[i]]),
                "Re_in": float(Re[i]),
                "N_points": int(n_points),
                # mdot / U_in filled by generate.finalize_flow (needs Cantera gas properties).
            }
        )
    return cases


def build_cases(config: DataGenConfig | None = None) -> list[dict]:
    """Build the full enriched case list (body + inlet-seed + high-T near-wall).

    Parameters
    ----------
    config
        Sampling configuration; defaults to :class:`DataGenConfig`.

    Returns
    -------
    list of case dicts (without ``mdot``/``U_in``), each tagged with a ``regime``.
    """
    cfg = config or DataGenConfig()
    cases: list[dict] = []
    next_id = 1

    body = _build_regime(
        start_id=next_id, n=cfg.n_body_cases, seed=cfg.seed,
        T_range=cfg.T_in_range_K, P_range=cfg.P_in_range_Pa, X_values=cfg.X_H2O_values,
        Re_range=cfg.Re_in_range, L_range=cfg.L_range_m, H_range=cfg.H_peak_range_W_m2,
        n_points=cfg.n_points, shapes=cfg.shapes, regime="body",
    )
    cases += body
    next_id += len(body)

    inlet = _build_regime(
        start_id=next_id, n=cfg.n_inlet_seed_cases, seed=cfg.seed + 1,
        T_range=cfg.T_in_range_K, P_range=cfg.P_in_range_Pa, X_values=cfg.X_H2O_values,
        Re_range=cfg.Re_in_range, L_range=cfg.inlet_seed_L_range_m,
        H_range=cfg.inlet_seed_H_peak_range_W_m2,
        n_points=cfg.n_points, shapes=cfg.shapes, regime="inlet_seed",
    )
    cases += inlet
    next_id += len(inlet)

    high_t = _build_regime(
        start_id=next_id, n=cfg.n_highT_cases, seed=cfg.seed + 2,
        T_range=cfg.highT_T_in_range_K, P_range=cfg.P_in_range_Pa, X_values=cfg.X_H2O_values,
        Re_range=cfg.Re_in_range, L_range=cfg.highT_L_range_m,
        H_range=cfg.highT_H_peak_range_W_m2,
        n_points=cfg.n_points, shapes=cfg.shapes, regime="high_T",
    )
    cases += high_t

    return cases


def coverage_summary(cases: list[dict]) -> dict:
    """Summarise the sampled envelope for logging (so coverage is explicit, never silently capped).

    Returns counts per regime and min/max of the key sampled parameters.
    """
    def _rng(key: str) -> tuple[float, float]:
        vals = [c[key] for c in cases if key in c]
        return (min(vals), max(vals)) if vals else (float("nan"), float("nan"))

    regimes: dict[str, int] = {}
    for c in cases:
        regimes[c.get("regime", "body")] = regimes.get(c.get("regime", "body"), 0) + 1

    return {
        "n_cases": len(cases),
        "regimes": regimes,
        "T_in_K": _rng("T_in"),
        "P_in_Pa": _rng("P_in"),
        "Re_in": _rng("Re_in"),
        "L_m": _rng("L"),
        "H_peak_W_m2": _rng("H_peak"),
        "X_H2O": sorted({c["X_H2O"] for c in cases}),
    }


def iter_cases(config: DataGenConfig | None = None) -> Iterator[dict]:
    """Yield enriched cases one at a time (convenience for streaming into the generator)."""
    yield from build_cases(config)

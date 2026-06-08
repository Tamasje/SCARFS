"""SCARFS benchmark harness — a-priori offline evaluation of rate surrogates.

This package provides tooling to evaluate any :class:`~scarfs.models.common.Surrogate`
implementation offline (without running the CFD solver) against the ground-truth
rates stored in the training database.

Public API
----------

**loader**

- :func:`~scarfs.benchmark.loader.load_database` — load ``.csv`` / ``.parquet`` database.
- :func:`~scarfs.benchmark.loader.infer_schema` — derive :class:`~scarfs.schema.Schema`
  from a loaded DataFrame.

**metrics**

- :func:`~scarfs.benchmark.metrics.r2_score` — R² (coefficient of determination).
- :func:`~scarfs.benchmark.metrics.nrmse` — RMSE normalised by std of true values.
- :func:`~scarfs.benchmark.metrics.nmdae` — median absolute error normalised by
  median absolute value of true values.
- :func:`~scarfs.benchmark.metrics.relative_error` — element-wise signed relative error.
- :func:`~scarfs.benchmark.metrics.median_relative_error` — median absolute relative error.
- :func:`~scarfs.benchmark.metrics.per_species_metrics` — per-species metric DataFrame.

**baselines**

- :class:`~scarfs.benchmark.baselines.FrozenComposition` — zero-rate sanity floor.
- :class:`~scarfs.benchmark.baselines.MeanRate` — training-mean rate predictor.
- :class:`~scarfs.benchmark.baselines.NearestNeighborRates` — nearest-neighbour look-up
  in scaled (logY, T, P) space.

**yields**

- :func:`~scarfs.benchmark.yields.integrate_yields` — trapezoidal PFR yield integration
  for a single case.
- :func:`~scarfs.benchmark.yields.integrate_yields_per_case` — yield integration for
  all cases in a multi-case DataFrame.

**apriori**

- :class:`~scarfs.benchmark.apriori.AprioriConfig` — configuration dataclass.
- :class:`~scarfs.benchmark.apriori.AprioriReport` — results dataclass with
  ``.passed`` and ``.summary()``.
- :func:`~scarfs.benchmark.apriori.holdout_split` — case-aware train/test/heldout split.
- :func:`~scarfs.benchmark.apriori.run_apriori` — main a-priori evaluation entry point.
"""

from scarfs.benchmark.loader import load_database, infer_schema
from scarfs.benchmark.metrics import (
    r2_score,
    nrmse,
    nmdae,
    relative_error,
    median_relative_error,
    per_species_metrics,
)
from scarfs.benchmark.baselines import (
    FrozenComposition,
    MeanRate,
    NearestNeighborRates,
)
from scarfs.benchmark.yields import integrate_yields, integrate_yields_per_case
from scarfs.benchmark.apriori import (
    AprioriConfig,
    AprioriReport,
    holdout_split,
    run_apriori,
)

__all__ = [
    # loader
    "load_database",
    "infer_schema",
    # metrics
    "r2_score",
    "nrmse",
    "nmdae",
    "relative_error",
    "median_relative_error",
    "per_species_metrics",
    # baselines
    "FrozenComposition",
    "MeanRate",
    "NearestNeighborRates",
    # yields
    "integrate_yields",
    "integrate_yields_per_case",
    # apriori
    "AprioriConfig",
    "AprioriReport",
    "holdout_split",
    "run_apriori",
]

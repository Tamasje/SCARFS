"""SCARFS — ML source-term surrogates for detailed-chemistry steam-cracking CFD.

This package supports the Tier-3 fix of the ChemZIP-style surrogate replication:

- ``scarfs.schema``       — the canonical column contract for the CRACKSIM/PFR database.
- ``scarfs.data``         — broadened, near-inlet-enriched training-case generation (F1/F4).
- ``scarfs.models``       — the two surrogates: reduced source-term and NeuralCoil (F2/F3).
- ``scarfs.training``     — HPC-runnable training pipeline, physics-consistent losses.
- ``scarfs.benchmark``    — a-priori (offline) and a-posteriori (CFD) benchmark harness.
- ``scarfs.coupling``     — Fluent UDF/UDS coupling scaffolding.
- ``scarfs.plotting``     — publication figures (user palette, dual °C/K, 400 DPI).
- ``scarfs.diagnostics``  — standing audit writers (energy unit/coverage, ambiguity,
                            conservation, sanity, front-resolution) with recalibrated
                            thresholds; grafted from the colleague's audit machinery (B2c).

Diagnosis and rationale: see ``DIAGNOSIS.md``, ``FIX_PROPOSAL.md``, ``BENCHMARK_PLAN.md``.
"""

from __future__ import annotations

__version__ = "0.1.0"

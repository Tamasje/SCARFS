"""Training-data generation: broadened, near-inlet-enriched case sampling (F1/F4).

- ``scarfs.data.config``   — :class:`DataGenConfig` (sampling ranges + enrichment fractions).
- ``scarfs.data.sampling`` — :func:`build_cases` (quasi-random body + inlet-seed + high-T near-wall).
- ``scarfs.data.generate`` — Cantera-backed flow finalisation + a documented driver that plugs the
  cases into the existing ``Database_Generation_MB.py`` multiprocessing harness (run on the HPC).

Rationale (see ``DIAGNOSIS.md`` RC-1/RC-4): the deployed model under-predicted rates from
low-conversion near-inlet states because those states were under-represented along the 1-D PFR grid.
This module enriches that regime and restores a broad, realistic operating envelope; it does NOT add
residence time as a feature (RC-5 refuted).
"""

from __future__ import annotations

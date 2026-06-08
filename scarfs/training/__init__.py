"""HPC-runnable training pipeline for the two surrogates.

- ``scarfs.training.config``     — typed configuration (JSON/YAML loadable).
- ``scarfs.training.datamodule`` — DB -> scaled features/targets + importance weights (F1); NumPy.
- ``scarfs.training.losses``     — composite physics-aware losses (F2/F3); PyTorch.
- ``scarfs.training.train``      — entry point ``python -m scarfs.training.train --config <file>``.

Only ``losses`` and ``train`` require PyTorch; ``config`` and ``datamodule`` are torch-free so the
data path is testable in a minimal environment. Full training runs on the HPC.
"""

from __future__ import annotations

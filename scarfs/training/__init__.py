"""HPC-runnable training pipeline for the two surrogates (and the new merged model).

- ``scarfs.training.config``     — typed configuration (JSON/YAML loadable).
- ``scarfs.training.datamodule`` — DB -> scaled features/targets + importance weights (F1); NumPy.
- ``scarfs.training.losses``     — composite physics-aware losses (F2/F3 + merged); PyTorch.
- ``scarfs.training.train``      — entry point ``python -m scarfs.training.train --config <file>``.

Only ``losses`` and ``train`` require PyTorch; ``config`` and ``datamodule`` are torch-free so the
data path is testable in a minimal environment.  Full training runs on the HPC.

Merged-model additions (B1c)
------------------------------
``kind="merged"`` activates the split-head energy path with:
- tail-stratified sample weights (log|S_E| deciles, opt-in via ``DataConfig.tail_strata``);
- enthalpy-aware species weights on the physical-rate head;
- arcsinh-space rate + latent-source losses (no winsorization, no output bounds);
- energy ties (rate-tied F3 + distillation + direct head);
- split-head consistency penalty;
- Lagrangian rollout loss (same-case τ steps; ``rollout_mode="lagrangian"``);
- 70/15/15 GroupKFold-style deterministic case split.

B1b (``scarfs.models.thermo`` / ``MergedCoil``) is imported lazily at train time; stubs are
used in unit tests via monkeypatching.
"""

from __future__ import annotations

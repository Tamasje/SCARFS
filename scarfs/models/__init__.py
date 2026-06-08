"""Surrogate models and the shared model-I/O contract.

- ``scarfs.models.common`` — framework-agnostic scalers + the :class:`Surrogate` protocol.
- ``scarfs.models.nets``   — PyTorch network factories (imported lazily; torch only needed to train).
- ``scarfs.models.reduced``    — the reduced source-term surrogate (thesis Ch. 6).
- ``scarfs.models.neuralcoil`` — the latent-space NeuralCoil surrogate (thesis Ch. 5, F2).
"""

from __future__ import annotations

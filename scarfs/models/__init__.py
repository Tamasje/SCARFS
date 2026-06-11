"""Surrogate models and the shared model-I/O contract.

- ``scarfs.models.common``      — framework-agnostic scalers + the :class:`Surrogate` protocol.
- ``scarfs.models.thermo``      — NASA7 thermo module (NumPy core + torch adapter, no Cantera).
- ``scarfs.models.nets``        — PyTorch network factories (imported lazily; torch only needed to train).
- ``scarfs.models.physics``     — physics-consistency utilities (molar rates, energy, atom balance).
- ``scarfs.models.features``    — feature / target assembly (NumPy only).
- ``scarfs.models.reduced``     — the reduced source-term surrogate (thesis Ch. 6).
- ``scarfs.models.neuralcoil``  — the latent-space NeuralCoil + MergedCoil surrogates.
- ``scarfs.models.adapter``     — :class:`TorchSurrogate` wrapper (NeuralCoil + MergedCoil).
"""

from __future__ import annotations

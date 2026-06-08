"""PyTorch network building blocks (imported only when training/inferring).

Kept separate from the rest of ``scarfs.models`` so that importing the contract (``common``),
feature assembly (``features``) and physics (``physics``) never requires PyTorch — only the actual
network code does. Run on the HPC where PyTorch is installed.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

#: Name -> activation module factory.
ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}


def make_mlp(
    sizes: Sequence[int],
    activation: str = "silu",
    layernorm: bool = True,
    final_activation: str | None = None,
) -> nn.Sequential:
    """Build a feed-forward MLP.

    Parameters
    ----------
    sizes
        Layer widths ``[in, h1, ..., out]`` (at least two entries).
    activation
        Hidden-layer activation name (see :data:`ACTIVATIONS`).
    layernorm
        Apply :class:`torch.nn.LayerNorm` before each hidden activation (thesis rate net used
        LayerNorm + SiLU, which stabilised training).
    final_activation
        Optional activation on the output layer (e.g. ``"sigmoid"`` for [0,1] decoders); ``None``
        leaves the output linear (the default for rate regression).
    """
    if len(sizes) < 2:
        raise ValueError(f"make_mlp needs >= 2 sizes, got {sizes!r}")
    act = ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layernorm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(act())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if final_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    elif final_activation is not None:
        layers.append(ACTIVATIONS[final_activation]())
    return nn.Sequential(*layers)


def count_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters in *module*."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

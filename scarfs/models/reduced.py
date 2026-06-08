"""Reduced source-term surrogate (thesis Ch. 6), with the RC-3 fixes applied.

Predicts the scaled net production rates of the active species directly from the local state. Two
deliberate improvements over the thesis version:

1. **Signed-log (arcsinh) rate targets instead of magnitude + separate sign head.** The thesis used a
   log-magnitude head plus a binary sign head; the sign discontinuity was a source of fragility. The
   :class:`~scarfs.models.common.ArcsinhScaler` represents production and consumption smoothly in one
   regression target.
2. **Energy source derived from the predicted rates** (``physics.derive_energy_source``) rather than a
   free, separately-trained head — so the energy equation is consistent with the species rates
   (addresses RC-3). The network therefore outputs *only* rates.
"""

from __future__ import annotations

import torch
from torch import nn

from .nets import make_mlp


class ReducedSurrogate(nn.Module):
    """MLP mapping the scaled state feature vector to scaled active-species rates.

    Parameters
    ----------
    n_features
        Input width = (number of input species) + 4 thermo features.
    n_targets
        Number of active species whose rates are predicted.
    hidden
        Hidden-layer widths.
    activation
        Hidden activation name.
    """

    def __init__(
        self,
        n_features: int,
        n_targets: int,
        hidden: tuple[int, ...] = (256, 256, 128),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_targets = n_targets
        self.net = make_mlp(
            [n_features, *hidden, n_targets],
            activation=activation,
            layernorm=True,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scaled rate predictions ``(batch, n_targets)``."""
        return self.net(x)

"""NeuralCoil latent-space surrogate (thesis Ch. 5) with the RC-2 stability fixes (F2).

Architecture (ChemZIP-faithful, with two deliberate corrections of the thesis deviations that caused
the divergence in DIAGNOSIS.md RC-2):

- **Linear encoder** ``E``: ``Y_dry (n_dry) -> Z (k)``, no bias (≈ PCA), as in ChemZIP.
- **Decoder** ``D``: ``[Z, q] -> Y_dry`` (sigmoid output for [0,1] MinMax composition).
- **Latent rate network** ``[Z, q] -> scaled rates`` — **takes the latent Z directly** (the
  ChemZIP design), NOT the decoded physical species. The thesis fed the *decoded* species to its
  rate net, forcing a decode every iteration and amplifying latent drift; using Z directly removes
  that coupling (F2).
- **Manifold projection** :meth:`project` recomputes ``Z <- E · D(Z)`` so a transported latent that
  has drifted off the encoder manifold is snapped back before the rate net is queried (the CFD-side
  fix for RC-2; mirrored in the Fluent UDS template).

``q`` is the standardised thermo block ``[T, p, 1/T, ln T]``.
"""

from __future__ import annotations

import torch
from torch import nn

from .nets import make_mlp


class NeuralCoil(nn.Module):
    """Latent-space source-term surrogate.

    Parameters
    ----------
    n_dry
        Number of (dry, diluent-excluded) species in the encoder input / decoder output.
    n_targets
        Number of active species whose rates are predicted by the latent rate net.
    latent_dim
        Latent dimension ``k`` (thesis used 6).
    n_thermo
        Width of the thermo block (``[T, p, 1/T, ln T]`` -> 4).
    decoder_hidden, rate_hidden
        Hidden widths of the decoder and rate networks.
    activation
        Hidden activation name.
    """

    def __init__(
        self,
        n_dry: int,
        n_targets: int,
        latent_dim: int = 6,
        n_thermo: int = 4,
        decoder_hidden: tuple[int, ...] = (128, 256),
        rate_hidden: tuple[int, ...] = (128, 128),
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.n_dry = n_dry
        self.n_targets = n_targets
        self.latent_dim = latent_dim
        self.encoder = nn.Linear(n_dry, latent_dim, bias=False)
        self.decoder = make_mlp(
            [latent_dim + n_thermo, *decoder_hidden, n_dry],
            activation=activation, layernorm=False, final_activation="sigmoid",
        )
        self.rate_net = make_mlp(
            [latent_dim + n_thermo, *rate_hidden, n_targets],
            activation=activation, layernorm=True, final_activation=None,
        )

    def encode(self, y_dry: torch.Tensor) -> torch.Tensor:
        """Project scaled dry composition to the latent space."""
        return self.encoder(y_dry)

    def decode(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Reconstruct scaled dry composition from latent + thermo."""
        return self.decoder(torch.cat([z, q], dim=-1))

    def project(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Snap a (possibly drifted) latent back onto the encoder manifold: ``Z <- E · D(Z)``.

        This is the F2 fix for RC-2 — applied before the rate net both in training (as a consistency
        regulariser) and at CFD inference (in the UDS DEFINE_ADJUST hook).
        """
        return self.encoder(self.decode(z, q))

    def rates_from_latent(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Scaled active-species rates from latent + thermo (the ChemZIP-faithful path, F2)."""
        return self.rate_net(torch.cat([z, q], dim=-1))

    def forward(self, y_dry: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass used during training.

        Returns a dict with the latent ``z``, projected latent ``z_proj``, reconstruction
        ``y_recon`` and scaled ``rates`` (computed from the projected latent for stability).
        """
        z = self.encode(y_dry)
        y_recon = self.decode(z, q)
        z_proj = self.encoder(y_recon)
        rates = self.rates_from_latent(z_proj, q)
        return {"z": z, "z_proj": z_proj, "y_recon": y_recon, "rates": rates}

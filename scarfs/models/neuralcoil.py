"""NeuralCoil latent-space surrogate (thesis Ch. 5) with the RC-2 stability fixes (F2),
and ``MergedCoil`` — the split-head merged surrogate (plan §3 / §4 E-b).

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

``MergedCoil`` extends the design with:

- Per-species standardised encoder input (linear, log=False, mode="standard") to preserve
  trace-species information (energy fix E-a).
- Three heads computed on ``cat([z_proj, q])``:
  1. Latent source ``ω_Z`` for CFD transport.
  2. Physical-rate head restricted to the energy-active species subset.
  3. Distilled strictly-positive absorption head (softplus-scaled).
"""

from __future__ import annotations

import numpy as np
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


class MergedCoil(nn.Module):
    """Split-head merged latent-space surrogate (plan §3 / §4 E-a – E-e).

    Extends :class:`NeuralCoil` with:

    - Per-species **standardised** (linear, no log) encoder input — preserves
      trace-species energy information (E-a).
    - Decoder outputs in **standardised** space (no sigmoid — the composition
      is no longer min–max scaled) with final activation = None.
    - **Three heads** on ``cat([z_proj, q])``:

      1. ``latent_source_net``: ``(B, k)`` scaled ω_Z for CFD transport
         (dimensionless; the CFD solver forms the actual source term).
      2. ``rate_net``: ``(B, n_energy_active)`` scaled physical rates for the
         energy-active species subset; this head drives the energy computation.
      3. ``energy_net_raw``: ``(B, 1)`` un-activated output mapped through
         softplus + calibration to give a strictly-positive absorption.

    Absorption = softplus(energy_raw) · energy_scale + energy_floor.

    Parameters
    ----------
    n_dry
        Number of (dry, diluent-excluded) species — the encoder input width.
    n_energy_active
        Number of energy-active species — the physical-rate head output width.
    latent_dim
        Latent dimension ``k`` (default 8, per plan E-e ablation default).
    n_thermo
        Width of the thermo block (default 4: ``[T, p, 1/T, ln T]``).
    decoder_hidden, rate_hidden, latent_source_hidden, energy_hidden
        Hidden layer widths for the respective MLPs.
    activation
        Hidden activation name.
    spectral_norm
        If ``True``, apply spectral normalisation to head Linear layers.
    """

    def __init__(
        self,
        n_dry: int,
        n_energy_active: int,
        latent_dim: int = 8,
        n_thermo: int = 4,
        decoder_hidden: tuple[int, ...] = (128, 256),
        rate_hidden: tuple[int, ...] = (128, 128),
        latent_source_hidden: tuple[int, ...] = (128, 128),
        energy_hidden: tuple[int, ...] = (64, 64),
        activation: str = "silu",
        spectral_norm: bool = False,
        n_transport: int = 0,
        transport_hidden: tuple[int, ...] = (64, 64),
    ) -> None:
        super().__init__()
        self.n_dry = n_dry
        self.n_energy_active = n_energy_active
        self.latent_dim = latent_dim
        self.n_transport = int(n_transport)

        k = latent_dim
        self.encoder = nn.Linear(n_dry, k, bias=False)
        # Decoder: standardised-space output (no sigmoid — caller handles composition contract)
        self.decoder = make_mlp(
            [k + n_thermo, *decoder_hidden, n_dry],
            activation=activation,
            layernorm=False,
            final_activation=None,
        )
        # Head 1: latent source ω_Z
        self.latent_source_net = make_mlp(
            [k + n_thermo, *latent_source_hidden, k],
            activation=activation,
            layernorm=True,
            final_activation=None,
            spectral_norm=spectral_norm,
        )
        # Head 2: physical rates for energy-active species
        self.rate_net = make_mlp(
            [k + n_thermo, *rate_hidden, n_energy_active],
            activation=activation,
            layernorm=True,
            final_activation=None,
            spectral_norm=spectral_norm,
        )
        # Head 3: distilled absorption (strictly positive via softplus)
        self.energy_net = make_mlp(
            [k + n_thermo, *energy_hidden, 1],
            activation=activation,
            layernorm=False,
            final_activation=None,
            spectral_norm=spectral_norm,
        )
        # Calibration buffers (set by set_energy_calibration after seeing training data)
        self.register_buffer("energy_scale", torch.tensor(1.0))
        self.register_buffer("energy_floor", torch.tensor(0.0))

        # Head 4 (optional): transport properties (μ, k, …) — strictly positive via softplus.
        # n_transport == 0 disables it entirely (state-dict unchanged for the legacy contract).
        if self.n_transport > 0:
            self.transport_net = make_mlp(
                [k + n_thermo, *transport_hidden, self.n_transport],
                activation=activation,
                layernorm=False,
                final_activation=None,
                spectral_norm=spectral_norm,
            )
            self.register_buffer("transport_scale", torch.ones(self.n_transport))
            self.register_buffer("transport_floor", torch.zeros(self.n_transport))

    # ------------------------------------------------------------------
    # Encoder / decoder / projection
    # ------------------------------------------------------------------
    def encode(self, y_std: torch.Tensor) -> torch.Tensor:
        """Map per-species standardised composition to the latent space."""
        return self.encoder(y_std)

    def decode(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Reconstruct standardised composition from latent + thermo."""
        return self.decoder(torch.cat([z, q], dim=-1))

    def project(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Snap a (possibly drifted) latent onto the encoder manifold: Z ← E·D(Z).

        Identical in purpose to the :class:`NeuralCoil` manifold projection (F2 fix
        for RC-2) but operates in the standardised-composition space.
        """
        return self.encoder(self.decode(z, q))

    # ------------------------------------------------------------------
    # Heads (take z_proj not raw z, for stability)
    # ------------------------------------------------------------------
    def latent_source(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Scaled latent-source ω_Z (shape ``(B, k)``) for CFD transport."""
        return self.latent_source_net(torch.cat([z, q], dim=-1))

    def rates_from_latent(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Scaled physical production rates for energy-active species (``(B, n_energy_active)``)."""
        return self.rate_net(torch.cat([z, q], dim=-1))

    def _energy_raw(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.energy_net(torch.cat([z, q], dim=-1)).squeeze(-1)  # (B,)

    def absorption(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Strictly-positive absorption [J/m³/s] via softplus + calibration.

        absorption = softplus(energy_raw) * energy_scale + energy_floor

        ``energy_scale`` and ``energy_floor`` are registered buffers initialised
        to 1.0 / 0.0 and set to representative training-data statistics via
        :meth:`set_energy_calibration`.
        """
        raw = self._energy_raw(z, q)
        return torch.nn.functional.softplus(raw) * self.energy_scale + self.energy_floor

    def transport(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Strictly-positive transport properties ``(B, n_transport)`` via softplus + calibration.

        ``property = softplus(raw) * transport_scale + transport_floor`` — the same C-expressible
        template as the absorption head, one extra net+softplus per cell (cheap).  Column order is
        whatever the trainer fed targets in (μ, k, …).  Raises if the head was not built
        (``n_transport == 0``).
        """
        if self.n_transport <= 0:
            raise RuntimeError("MergedCoil.transport called but n_transport == 0 (head not built).")
        raw = self.transport_net(torch.cat([z, q], dim=-1))      # (B, n_transport)
        return torch.nn.functional.softplus(raw) * self.transport_scale + self.transport_floor

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def set_energy_calibration(self, scale: float, floor: float = 0.0) -> None:
        """Set the energy head calibration buffers.

        Parameters
        ----------
        scale
            Typical absorption magnitude (e.g. median of training absorption
            column); makes the softplus output numerically comparable to the
            target distribution before training.
        floor
            Minimum guaranteed output [J/m³/s] (default 0.0).
        """
        self.energy_scale.fill_(float(scale))
        self.energy_floor.fill_(float(floor))

    def set_transport_calibration(self, scale, floor=None) -> None:
        """Set per-property calibration for the transport head.

        Parameters
        ----------
        scale
            ``(n_transport,)`` typical magnitudes (e.g. train medians of μ, k, …) so the softplus
            output is numerically comparable to each property before training.
        floor
            ``(n_transport,)`` minimum guaranteed outputs (default zeros).
        """
        if self.n_transport <= 0:
            raise RuntimeError("set_transport_calibration called but n_transport == 0.")
        s = torch.as_tensor(scale, dtype=self.transport_scale.dtype).reshape(-1)
        self.transport_scale.copy_(s)
        if floor is not None:
            f = torch.as_tensor(floor, dtype=self.transport_floor.dtype).reshape(-1)
            self.transport_floor.copy_(f)

    # ------------------------------------------------------------------
    # PCA initialisation helper
    # ------------------------------------------------------------------
    def init_encoder_pca(self, components: np.ndarray) -> None:
        """Initialise the encoder weight from PCA components.

        Parameters
        ----------
        components
            ``(k, n_dry)`` array of the top-k PCA components (rows = components,
            as returned by ``sklearn.decomposition.PCA.components_``).
        """
        w = torch.as_tensor(components, dtype=torch.float32)
        if w.shape != (self.latent_dim, self.n_dry):
            raise ValueError(
                f"init_encoder_pca: expected ({self.latent_dim}, {self.n_dry}), got {tuple(w.shape)}"
            )
        with torch.no_grad():
            self.encoder.weight.copy_(w)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, y_std: torch.Tensor, q: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Returns
        -------
        dict with keys:
        - ``z``: raw latent ``(B, k)``.
        - ``z_proj``: manifold-projected latent ``(B, k)``.
        - ``y_recon``: reconstructed standardised composition ``(B, n_dry)``.
        - ``latent_source``: ω_Z head output ``(B, k)``.
        - ``rates``: physical-rate head output ``(B, n_energy_active)``.
        - ``absorption``: strictly-positive absorption head ``(B,)``.
        """
        z = self.encode(y_std)
        y_recon = self.decode(z, q)
        z_proj = self.encoder(y_recon)
        zq = torch.cat([z_proj, q], dim=-1)  # shared input for all heads
        # compute heads from z_proj (stability: manifold-consistent state)
        ls = self.latent_source_net(zq)
        rates = self.rate_net(zq)
        raw = self.energy_net(zq).squeeze(-1)
        abs_ = torch.nn.functional.softplus(raw) * self.energy_scale + self.energy_floor
        return {
            "z": z,
            "z_proj": z_proj,
            "y_recon": y_recon,
            "latent_source": ls,
            "rates": rates,
            "absorption": abs_,
        }

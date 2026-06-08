"""Composite, physics-aware training losses (PyTorch).

Encodes the F2/F3 fixes:

- **Importance- and species-weighted rate loss** — row weights up-weight near-inlet states (F1, set in
  the datamodule); per-species weights emphasise the key products (thesis upweighted C2H4/C3H6/...).
- **Atom-balance penalty** (F3) — soft elemental-conservation pressure on the predicted physical
  rates (optional; needs element data).
- **NeuralCoil stabilisers** (F2) — decoder reconstruction, manifold-consistency ``||z - E·D(z)||``,
  and latent **noise injection** so the rate net learns to recover from off-manifold drift (the
  failure mode behind RC-2). Full data-rollout unrolling is a documented extension hook.
"""

from __future__ import annotations

import torch
from torch import nn

_mse = nn.MSELoss(reduction="none")


def weighted_rate_loss(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    row_weights: torch.Tensor,
    species_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on scaled rates, weighted per row (F1) and per species (product emphasis).

    Parameters
    ----------
    pred_scaled, target_scaled
        ``(batch, n_targets)`` scaled rate predictions / targets.
    row_weights
        ``(batch,)`` per-sample weights (>=1 for near-inlet rows).
    species_weights
        ``(n_targets,)`` per-species weights.
    """
    err = _mse(pred_scaled, target_scaled)                 # (batch, n_targets)
    err = err * species_weights.unsqueeze(0)
    err = err.mean(dim=1) * row_weights
    return err.mean()


def atom_balance_penalty(
    rates_phys: torch.Tensor, molar_mass: torch.Tensor, element_matrix: torch.Tensor
) -> torch.Tensor:
    """Soft elemental-conservation penalty ``mean ||E^T (rate/W)||^2`` on physical rates (F3).

    Exact closure needs the full species set; on a reduced active set this is a consistency
    pressure (documented limitation). All tensors share the active-species ordering.
    """
    molar = rates_phys / molar_mass.unsqueeze(0)           # (batch, n_species)
    atom_rate = molar @ element_matrix                     # (batch, n_elements)
    return (atom_rate ** 2).sum(dim=1).mean()


def manifold_consistency(z: torch.Tensor, z_proj: torch.Tensor) -> torch.Tensor:
    """``mean ||z - E·D(z)||^2`` — keeps the latent on the encoder manifold (F2, RC-2)."""
    return _mse(z, z_proj).mean()


def reconstruction_loss(y_recon: torch.Tensor, y_true_scaled: torch.Tensor) -> torch.Tensor:
    """Decoder reconstruction MSE on scaled composition."""
    return _mse(y_recon, y_true_scaled).mean()


def neuralcoil_composite(
    model,
    y_dry_scaled: torch.Tensor,
    q: torch.Tensor,
    target_scaled: torch.Tensor,
    row_weights: torch.Tensor,
    species_weights: torch.Tensor,
    *,
    recon_weight: float,
    manifold_weight: float,
    noise_std: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full NeuralCoil training loss with the F2 stabilisers.

    Noise injection: latent is perturbed by ``noise_std`` Gaussian noise before the rate net, so the
    network learns to map slightly off-manifold states back to correct rates — directly targeting the
    drift that diverged the thesis model (RC-2).
    """
    z = model.encode(y_dry_scaled)
    y_recon = model.decode(z, q)
    z_proj = model.encoder(y_recon)
    z_used = z_proj
    if noise_std > 0:
        z_used = z_used + noise_std * torch.randn_like(z_used)
    rates = model.rates_from_latent(z_used, q)

    rate = weighted_rate_loss(rates, target_scaled, row_weights, species_weights)
    recon = reconstruction_loss(y_recon, y_dry_scaled)
    manifold = manifold_consistency(z, z_proj)
    total = rate + recon_weight * recon + manifold_weight * manifold
    parts = {"rate": float(rate.detach()), "recon": float(recon.detach()), "manifold": float(manifold.detach())}
    return total, parts

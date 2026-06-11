"""Composite, physics-aware training losses (PyTorch).

Encodes the F2/F3 fixes:

- **Importance- and species-weighted rate loss** — row weights up-weight near-inlet states (F1,
  set in the datamodule); per-species weights emphasise the key products.
- **Atom-balance penalty** (F3) — soft elemental-conservation pressure on the predicted physical
  rates (optional; needs element data).
- **NeuralCoil stabilisers** (F2) — decoder reconstruction, manifold-consistency
  ``||z - E·D(z)||``, and latent **noise injection** so the rate net learns to recover from
  off-manifold drift (the failure mode behind RC-2).  Full data-rollout unrolling is a documented
  extension hook.
- **Merged composite** (B1c) — adds the split-head energy path, latent-source loss, energy ties,
  consistency penalty, and Lagrangian rollout.  All existing reduced/neuralcoil paths unchanged.

Energy-path no-winsorize guarantee
------------------------------------
No function in this module applies quantile clipping, winsorization, or train-quantile output
bounds to either rates or absorption targets.  The design relies on arcsinh scaling and tail-
stratified *row weights* instead (set in the datamodule).
"""

from __future__ import annotations

import torch
from torch import nn

_mse = nn.MSELoss(reduction="none")
_l1 = nn.L1Loss(reduction="none")


# ---------------------------------------------------------------------------
# Shared primitives (used by reduced, neuralcoil, and merged)
# ---------------------------------------------------------------------------

def weighted_rate_loss(
    pred_scaled: torch.Tensor,
    target_scaled: torch.Tensor,
    row_weights: torch.Tensor,
    species_weights: torch.Tensor,
) -> torch.Tensor:
    """Mean squared error on scaled rates, weighted per row (F1) and per species.

    Parameters
    ----------
    pred_scaled, target_scaled
        ``(batch, n_targets)`` scaled rate predictions / targets.
    row_weights
        ``(batch,)`` per-sample weights (>=1 for near-inlet rows; may include tail weights).
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
    pressure (documented limitation).  All tensors share the active-species ordering.
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


# ---------------------------------------------------------------------------
# NeuralCoil composite (unchanged baseline)
# ---------------------------------------------------------------------------

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

    Noise injection: latent is perturbed by ``noise_std`` Gaussian noise before the rate net, so
    the network learns to map slightly off-manifold states back to correct rates — directly
    targeting the drift that diverged the thesis model (RC-2).
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


# ---------------------------------------------------------------------------
# Merged-model loss terms
# ---------------------------------------------------------------------------

def arcsinh_rate_loss(
    pred_rates: torch.Tensor,
    target_rates: torch.Tensor,
    row_weights: torch.Tensor,
    species_weights: torch.Tensor,
    arcsinh_scale: torch.Tensor,
) -> torch.Tensor:
    """Arcsinh-space MSE on physical (unscaled) mass rates with per-row and per-species weights.

    The arcsinh transform is applied inside the loss so the scale constants ``arcsinh_scale`` are
    frozen from the PCA-init pre-pass and remain constant throughout training (unlike the normal
    ArcsinhScaler which standardises further).

    Parameters
    ----------
    pred_rates, target_rates
        ``(batch, n_species)`` physical mass rates [kg m-3 s-1].
    row_weights
        ``(batch,)`` combined importance + tail weights.
    species_weights
        ``(n_species,)`` enthalpy-aware per-species weights (floor 1.0).
    arcsinh_scale
        ``(n_species,)`` frozen per-species scale constants (median |rate| from PCA-init pass).
    """
    pred_t = torch.arcsinh(pred_rates / arcsinh_scale.unsqueeze(0))
    tgt_t = torch.arcsinh(target_rates / arcsinh_scale.unsqueeze(0))
    err = _mse(pred_t, tgt_t)                              # (batch, n_species)
    err = err * species_weights.unsqueeze(0)
    err = err.mean(dim=1) * row_weights
    return err.mean()


def latent_source_loss(
    pred_latent_src: torch.Tensor,
    target_latent_src: torch.Tensor,
    arcsinh_scale: torch.Tensor,
) -> torch.Tensor:
    """Arcsinh-space MSE for the latent-source head (ω_Z).

    Target is ``E·(dYdt ⊘ σ_comp)`` computed with the CURRENT encoder weights inside the
    composite loss (so E evolves).  ``arcsinh_scale`` are frozen per-dim scale constants.

    Parameters
    ----------
    pred_latent_src
        ``(batch, k)`` predicted latent source.
    target_latent_src
        ``(batch, k)`` target latent source (computed dynamically from encoder + dYdt).
    arcsinh_scale
        ``(k,)`` frozen per-dim arcsinh scale constants.
    """
    pred_t = torch.arcsinh(pred_latent_src / arcsinh_scale.unsqueeze(0))
    tgt_t = torch.arcsinh(target_latent_src / arcsinh_scale.unsqueeze(0))
    return _mse(pred_t, tgt_t).mean()


def _weighted_mean(err: torch.Tensor, row_weights: torch.Tensor | None) -> torch.Tensor:
    """Mean of per-row errors, optionally weighted (weights broadcast over the batch)."""
    if row_weights is None:
        return err.mean()
    return (err * row_weights).sum() / row_weights.sum().clamp(min=1e-12)


def energy_rate_tied_loss(
    absorption_from_rates: torch.Tensor,
    absorption_target: torch.Tensor,
    scale: torch.Tensor | float = 1.0,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """F3 rate-tied energy tie in arcsinh space: rates-derived absorption vs the DB target.

    Both tensors are ``(batch,)`` [J m-3 s-1].  ``scale`` is a fixed physical scale (median
    train absorption) so the loss is O(1) and scale-invariant in the tail.  arcsinh — not
    ``log(clamp(·, 1))`` — keeps the gradient ALIVE when the rate-derived value is negative
    (early training: Σh·ω̂ of random rates is negative for ~half the rows; a clamped log
    zeroes those gradients and the energy tie never pulls the rates positive).
    No winsorization applied.  ``row_weights`` carries the tail-stratified emphasis.
    """
    pred_t = torch.arcsinh(absorption_from_rates / scale)
    tgt_t = torch.arcsinh(absorption_target / scale)
    return _weighted_mean(_mse(pred_t, tgt_t), row_weights)


def energy_distill_loss(
    absorption_head: torch.Tensor,
    absorption_from_rates_stopgrad: torch.Tensor,
    scale: torch.Tensor | float = 1.0,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distillation term: head vs stopgrad(rate-derived), arcsinh space.

    Raw-unit MSE on absorptions (1e6–1e9 J m-3 s-1) reaches 1e16+ and swamps the composite
    total (breaking the early-stopping monitor); the arcsinh form is O(1) like every other
    term. ``absorption_from_rates_stopgrad`` should have its gradient detached by the caller.
    """
    pred_t = torch.arcsinh(absorption_head / scale)
    tgt_t = torch.arcsinh(absorption_from_rates_stopgrad / scale)
    return _weighted_mean(_mse(pred_t, tgt_t), row_weights)


def energy_direct_loss(
    absorption_head: torch.Tensor,
    absorption_target: torch.Tensor,
    scale: torch.Tensor | float = 1.0,
    row_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Direct head-vs-target term in arcsinh space (head is strictly positive)."""
    pred_t = torch.arcsinh(absorption_head / scale)
    tgt_t = torch.arcsinh(absorption_target / scale)
    return _weighted_mean(_mse(pred_t, tgt_t), row_weights)


def split_head_consistency(
    z_t: torch.Tensor,
    omega_hat_mass: torch.Tensor,
    rho: torch.Tensor,
    dydt_dry_true: torch.Tensor,
    z_dot_true: torch.Tensor,
    encoder_weight_full: torch.Tensor,
    active_col_idx: torch.Tensor,
    sigma_active: torch.Tensor,
    arcsinh_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """MSE penalty ensuring the latent-source head agrees with the physical-rate head.

    The consistency target is:

        C = E_act · ((ω̂_mass / ρ) ⊘ σ_active)
              + stopgrad(z_dot_true − E_act · (dYdt_true_active ⊘ σ_active))

    i.e. the active-species contribution of the PREDICTED rates (converted from mass
    rates back to dY/dt with the local density) plus the stop-gradient residual carried
    by the non-active species.  ``encoder_weight_full`` must already be DETACHED — the
    encoder must not be able to game agreement between its two heads.

    Parameters
    ----------
    z_t
        ``(batch, k)`` latent-source head output (ω_Z, physical latent units).
    omega_hat_mass
        ``(batch, n_active)`` predicted physical MASS rates ρ·dY/dt [kg m-3 s-1].
    rho
        ``(batch,)`` density [kg m-3] — converts mass rates back to dY/dt.
    dydt_dry_true
        ``(batch, n_dry)`` ground-truth dY/dt for ALL dry input species [1/s].
    z_dot_true
        ``(batch, k)`` ground-truth latent source = E · (dYdt_dry ⊘ σ_comp_all).
    encoder_weight_full
        ``(k, n_dry)`` DETACHED encoder weight.
    active_col_idx
        ``(n_active,)`` long indices of the energy-active species within the dry ordering.
    sigma_active
        ``(n_active,)`` composition-scaler σ for the active species.
    """
    e_act = encoder_weight_full[:, active_col_idx]                                       # (k, n_active)
    dydt_pred_active = omega_hat_mass / rho.unsqueeze(1)                                 # (batch, n_active)
    pred_contrib = (dydt_pred_active / sigma_active.unsqueeze(0)) @ e_act.t()            # (batch, k)
    true_contrib = (dydt_dry_true[:, active_col_idx] / sigma_active.unsqueeze(0)) @ e_act.t()
    residual = (z_dot_true - true_contrib).detach()
    consistency_target = pred_contrib + residual                                         # (batch, k)
    # arcsinh space at the frozen latent scale: raw z-dot components reach ~1e8 (trace-species
    # σ-amplification), so a raw-unit MSE here dominates the whole composite by ~10 orders of
    # magnitude and hijacks every gradient (observed: this single term at 2.3e11).
    pred_t = torch.arcsinh(z_t / arcsinh_scale)
    tgt_t = torch.arcsinh(consistency_target / arcsinh_scale)
    return _mse(pred_t, tgt_t).mean()


def lagrangian_rollout_loss(
    z_t: torch.Tensor,
    z_tp1_true: torch.Tensor,
    latent_src_t: torch.Tensor,
    dtau: torch.Tensor,
) -> torch.Tensor:
    """Lagrangian continuity penalty: z_{t+1}^pred = z_t + Δτ·ω_Z(z_t) ≈ z_{t+1}^true.

    The latent-source head outputs PHYSICAL latent units by convention (the arcsinh
    transform lives inside :func:`latent_source_loss`, not in the head), so no scale
    conversion is applied here — rescaling would double-scale the source.

    Parameters
    ----------
    z_t
        ``(n_pairs, k)`` latent at step t.
    z_tp1_true
        ``(n_pairs, k)`` true latent at step t+1 (stop-gradient applied by the caller).
    latent_src_t
        ``(n_pairs, k)`` predicted latent source ω_Z at step t [1/s, physical].
    dtau
        ``(n_pairs,)`` Δτ [s].
    """
    z_tp1_pred = z_t + dtau.unsqueeze(1) * latent_src_t                  # (n_pairs, k)
    return _mse(z_tp1_pred, z_tp1_true).mean()


# ---------------------------------------------------------------------------
# Merged composite (B1c)
# ---------------------------------------------------------------------------

def merged_composite(
    model,
    y_std_scaled: torch.Tensor,
    q: torch.Tensor,
    target_rates_phys: torch.Tensor,
    absorption_target: torch.Tensor,
    dydt_dry_phys: torch.Tensor,
    rho: torch.Tensor,
    row_weights: torch.Tensor,
    enthalpy_weights: torch.Tensor,
    species_weights_qoi: torch.Tensor,
    *,
    arcsinh_rate_scale: torch.Tensor,
    arcsinh_latent_scale: torch.Tensor,
    sigma_active: torch.Tensor,
    sigma_comp_all: torch.Tensor,
    active_col_idx: torch.Tensor,
    energy_arcsinh_scale: torch.Tensor | float = 1.0,
    # loss weights
    rate_weight: float = 1.0,
    latent_source_weight: float = 1.0,
    energy_weight: float = 0.5,
    energy_distill_weight: float = 0.25,
    energy_target_weight: float = 0.25,
    consistency_weight: float = 0.1,
    recon_weight: float = 1.0,
    qoi_recon_weight: float = 0.0,
    manifold_weight: float = 0.1,
    atom_balance_weight: float = 0.0,
    noise_std: float = 0.0,
    rollout_mode: str = "manifold",
    # lagrangian rollout (optional)
    idx_t: torch.Tensor | None = None,
    idx_tp1: torch.Tensor | None = None,
    dtau: torch.Tensor | None = None,
    # atom balance (optional)
    molar_mass: torch.Tensor | None = None,
    element_matrix: torch.Tensor | None = None,
    # absorption_from_rates_fn: callable that converts predicted mass rates + T to absorption
    absorption_from_rates_fn=None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Full merged-model training loss.

    Loss terms (see module docstring formula):
    1. ``rate_weight``         × arcsinh-space MSE on physical rates with enthalpy+tail weights.
    2. ``latent_source_weight``× arcsinh-space MSE on the latent-source head (ω_Z).
    3. ``energy_weight``       × rate-tied log-MSE: absorption_from_rates(ω̂_phys, T) vs target.
    4. ``energy_distill_weight``× head–stopgrad(rate-derived absorption) L2.
    5. ``energy_target_weight`` × direct head vs DB absorption target (log-MSE).
    6. ``consistency_weight``  × split-head consistency penalty.
    7. ``recon_weight``        × decoder reconstruction MSE (composition).
    8. ``qoi_recon_weight``    × QoI-weighted decoder reconstruction.
    9. ``manifold_weight``     × manifold consistency ||z – E·D(z)||².
    10. ``atom_balance_weight``× atom-balance penalty (optional).
    11. Rollout term           (manifold | lagrangian).

    Parameters
    ----------
    model
        A ``MergedCoil``-compatible model (B1b contract).  Must expose ``.encode``,
        ``.decode``, ``.project``, ``.latent_source``, ``.rates_from_latent``,
        ``.absorption``, and ``.encoder`` attribute.
    y_std_scaled
        ``(batch, n_dry)`` standardised (linear, no log) input composition.
    q
        ``(batch, n_thermo)`` standardised thermo block.
    target_rates_phys
        ``(batch, n_active)`` physical mass rates for the energy-active species [kg m-3 s-1].
    absorption_target
        ``(batch,)`` DB absorption target [J m-3 s-1] (clipped negatives to 0 upstream).
    dydt_dry_phys
        ``(batch, n_dry)`` physical dY/dt for ALL dry input species [1/s] — the latent-source
        target needs the full dry vector: under per-species standardisation, trace species
        contribute to ż = E·(Ẏ⊘σ) through the 1/σ amplification, so truncating to the
        energy-active set would corrupt the transport target.
    rho
        ``(batch,)`` density [kg m-3] (mass rate ↔ dY/dt conversion in the consistency term).
    row_weights
        ``(batch,)`` combined importance + tail weights.
    enthalpy_weights
        ``(n_active,)`` enthalpy-aware per-species weights (floor 1.0; only for rate head).
    species_weights_qoi
        ``(n_dry,)`` QoI per-species weights for the decoder recon term (from DataConfig).
    arcsinh_rate_scale
        ``(n_active,)`` frozen arcsinh scale constants for the rate head.
    arcsinh_latent_scale
        ``(k,)`` frozen arcsinh scale constants for the latent-source head.
    sigma_active
        ``(n_active,)`` composition-scaler σ for the energy-active species.
    sigma_comp_all
        ``(n_dry,)`` composition-scaler σ for all dry input species (for z_dot_true target).
    active_col_idx
        ``(n_active,)`` long indices of the energy-active species within the dry ordering.
        The latent-source target and the consistency term read the encoder weight LIVE
        (detached) from ``model.encoder`` each batch, so the target evolves with E without
        opening a gradient shortcut through the target side.
    rollout_mode
        ``"manifold"`` — existing multi-step consistency; ``"lagrangian"`` — τ-step rollout.
    idx_t, idx_tp1, dtau
        Lagrangian pair indices and Δτ values (required when ``rollout_mode="lagrangian"``).
    absorption_from_rates_fn
        Optional callable ``(rates_phys, q) -> (batch,)`` giving rate-derived absorption
        [J m-3 s-1].  If None, energy tie terms are skipped (e.g. B1b not yet integrated).
    """
    # -- forward pass (single pass; heads read the noise-injected projected latent) -------
    z = model.encode(y_std_scaled)
    y_recon = model.decode(z, q)
    z_proj = model.encoder(y_recon)                # manifold projection, reusing the decode

    z_used = z_proj
    if noise_std > 0:
        z_used = z_used + noise_std * torch.randn_like(z_used)

    z_out, z_proj_out, y_recon_out = z, z_proj, y_recon
    latent_src = model.latent_source(z_used, q)    # (batch, k)
    rates_pred = model.rates_from_latent(z_used, q)  # (batch, n_active)
    absorption_head = model.absorption(z_used, q)  # (batch, 1) strictly positive

    # ground-truth latent source from the LIVE (detached) encoder: ż = E·(Ẏ_dry ⊘ σ_dry)
    encoder_weight_full = model.encoder.weight.detach()            # (k, n_dry)
    z_dot_true = (dydt_dry_phys / sigma_comp_all.unsqueeze(0)) @ encoder_weight_full.t()

    parts: dict[str, float] = {}
    total = torch.tensor(0.0, device=y_std_scaled.device)

    # -- 1. Physical-rate loss (arcsinh + enthalpy weights + tail weights) ---------------
    if rates_pred is not None and rate_weight > 0.0:
        rl = arcsinh_rate_loss(
            rates_pred, target_rates_phys, row_weights, enthalpy_weights, arcsinh_rate_scale
        )
        total = total + rate_weight * rl
        parts["rate"] = float(rl.detach())

    # -- 2. Latent-source loss -----------------------------------------------------------
    if latent_src is not None and latent_source_weight > 0.0:
        lsl = latent_source_loss(latent_src, z_dot_true, arcsinh_latent_scale)
        total = total + latent_source_weight * lsl
        parts["latent_source"] = float(lsl.detach())

    # -- 3–5. Energy terms (arcsinh space at the fixed physical scale; tail row-weighted) --
    if absorption_from_rates_fn is not None:
        # rate-derived absorption (differentiable wrt rates_pred)
        abs_from_rates = absorption_from_rates_fn(rates_pred, q)  # (batch,) or (batch,1)
        abs_from_rates = abs_from_rates.squeeze(-1)               # (batch,)

        # 3. Rate-tied energy tie (F3)
        if energy_weight > 0.0:
            etl = energy_rate_tied_loss(
                abs_from_rates, absorption_target, energy_arcsinh_scale, row_weights)
            total = total + energy_weight * etl
            parts["energy_rate_tied"] = float(etl.detach())

        if absorption_head is not None:
            abs_head = absorption_head.squeeze(-1)  # (batch,)

            # 4. Distillation: head → stopgrad(rate-derived)
            if energy_distill_weight > 0.0:
                edl = energy_distill_loss(
                    abs_head, abs_from_rates.detach(), energy_arcsinh_scale, row_weights)
                total = total + energy_distill_weight * edl
                parts["energy_distill"] = float(edl.detach())

            # 5. Direct head vs target (always runs when head is present)
            if energy_target_weight > 0.0:
                etgt = energy_direct_loss(
                    abs_head, absorption_target, energy_arcsinh_scale, row_weights)
                total = total + energy_target_weight * etgt
                parts["energy_direct"] = float(etgt.detach())

    elif absorption_head is not None:
        # Fallback: no rate-derived fn available yet (B1b pending) — still train the absorption
        # head directly against the DB target so its parameters receive gradients.
        abs_head = absorption_head.squeeze(-1)
        if energy_target_weight > 0.0:
            etgt = energy_direct_loss(
                abs_head, absorption_target, energy_arcsinh_scale, row_weights)
            total = total + energy_target_weight * etgt
            parts["energy_direct"] = float(etgt.detach())

    # -- 6. Split-head consistency -------------------------------------------------------
    if latent_src is not None and rates_pred is not None and consistency_weight > 0.0:
        cpen = split_head_consistency(
            latent_src, rates_pred, rho, dydt_dry_phys, z_dot_true,
            encoder_weight_full, active_col_idx, sigma_active,
            arcsinh_scale=arcsinh_latent_scale,
        )
        total = total + consistency_weight * cpen
        parts["consistency"] = float(cpen.detach())

    # -- 7. Decoder reconstruction -------------------------------------------------------
    if recon_weight > 0.0:
        recon = reconstruction_loss(y_recon_out, y_std_scaled)
        total = total + recon_weight * recon
        parts["recon"] = float(recon.detach())

    # -- 8. QoI-weighted decoder reconstruction -----------------------------------------
    if qoi_recon_weight > 0.0:
        err_qoi = _mse(y_recon_out, y_std_scaled)                  # (batch, n_dry)
        err_qoi = (err_qoi * species_weights_qoi.unsqueeze(0)).mean()
        total = total + qoi_recon_weight * err_qoi
        parts["qoi_recon"] = float(err_qoi.detach())

    # -- 9. Manifold consistency ---------------------------------------------------------
    if manifold_weight > 0.0:
        manifold = manifold_consistency(z_out, z_proj_out)
        total = total + manifold_weight * manifold
        parts["manifold"] = float(manifold.detach())

    # -- 10. Atom-balance ----------------------------------------------------------------
    if atom_balance_weight > 0.0 and molar_mass is not None and element_matrix is not None and rates_pred is not None:
        abp = atom_balance_penalty(rates_pred, molar_mass, element_matrix)
        total = total + atom_balance_weight * abp
        parts["atom_balance"] = float(abp.detach())

    # -- 11. Rollout term ----------------------------------------------------------------
    if rollout_mode == "lagrangian" and idx_t is not None and len(idx_t) > 0:
        if latent_src is not None:
            z_t_lag = z_out[idx_t]
            z_tp1_true = z_out.detach()[idx_tp1]  # treat future true as stopgrad
            src_t_lag = latent_src[idx_t]
            roll_loss = lagrangian_rollout_loss(z_t_lag, z_tp1_true, src_t_lag, dtau)
            total = total + roll_loss
            parts["lagrangian_rollout"] = float(roll_loss.detach())

    parts["total"] = float(total.detach())
    return total, parts

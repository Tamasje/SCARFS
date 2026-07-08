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

def scaled_rate_loss(
    pred_rates_scaled: torch.Tensor,
    target_rates_scaled: torch.Tensor,
    row_weights: torch.Tensor,
    species_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted MSE on rates in the SCALED (ArcsinhScaler) space — the head's native space.

    The rate head emits ArcsinhScaler-space values (standardised arcsinh, O(1) per output).
    Training the head to emit PHYSICAL rates was tried and fails structurally: physical mass
    rates span ~9 decades, which a (spectral-normed) linear-output MLP cannot express — the
    observed result was a rate term stuck ~1e3 in arcsinh-diff units and near-zero rate-derived
    energy. The scaled-head convention also matches the adapter, the bundle scalers and the
    UDF exporter (one inversion point, ``rates_to_physical_fn``, owned by the trainer).

    Parameters
    ----------
    pred_rates_scaled, target_rates_scaled
        ``(batch, n_species)`` ArcsinhScaler-space rates (the bundle's Y arrays).
    row_weights
        ``(batch,)`` combined importance + tail weights.
    species_weights
        ``(n_species,)`` enthalpy-aware per-species weights (bounded [floor, cap]).
    """
    err = _mse(pred_rates_scaled, target_rates_scaled)     # (batch, n_species)
    err = err * species_weights.unsqueeze(0)
    err = err.mean(dim=1) * row_weights
    return err.mean()


def latent_source_loss(
    pred_latent_src_scaled: torch.Tensor,
    target_latent_src_phys: torch.Tensor,
    arcsinh_scale: torch.Tensor,
) -> torch.Tensor:
    """MSE for the latent-source head in its native arcsinh space.

    The head emits ``arcsinh(ω_Z / s_Z)`` (scaled); the target is the PHYSICAL
    ``E·(dYdt ⊘ σ_comp)`` computed with the CURRENT (detached) encoder weights inside the
    composite, transformed here with the frozen per-dim scales ``s_Z``. Physical ω_Z for
    transport/rollout/UDF is recovered as ``sinh(head)·s_Z``.

    Parameters
    ----------
    pred_latent_src_scaled
        ``(batch, k)`` head output (scaled space).
    target_latent_src_phys
        ``(batch, k)`` physical target latent source [1/s].
    arcsinh_scale
        ``(k,)`` frozen per-dim arcsinh scale constants.
    """
    tgt_t = torch.arcsinh(target_latent_src_phys / arcsinh_scale.unsqueeze(0))
    return _mse(pred_latent_src_scaled, tgt_t).mean()


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
    # The latent head output z_t is ALREADY in arcsinh space; transform only the (physical)
    # consistency target. Raw z-dot components reach ~1e8 (trace-species σ-amplification), so a
    # raw-unit MSE here dominates the whole composite by ~10 orders of magnitude and hijacks
    # every gradient (observed: this single term at 2.3e11).
    tgt_t = torch.arcsinh(consistency_target / arcsinh_scale)
    return _mse(z_t, tgt_t).mean()


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


def pushforward_species_loss(
    model,
    seq_ystd: torch.Tensor,      # (M, K+1, n_dry)  standardised composition along the τ-window
    seq_q: torch.Tensor,         # (M, K+1, n_thermo) standardised thermo per step
    seq_dtau: torch.Tensor,      # (M, K)            Δτ between steps [s]
    arcsinh_latent_scale: torch.Tensor,   # (k,)
    clamp: float = 50.0,
    *,
    seq_abs: torch.Tensor | None = None,          # (M, K+1) true absorption S_E along the window [J m-3 s-1]
    absorption_from_rates_fn=None,                # (rates_phys, q) -> S_E [J m-3 s-1]
    rates_to_physical_fn=None,                    # scaled rate-head output -> physical mass rates
    energy_arcsinh_scale: torch.Tensor | float = 1.0,
    energy_weight: float = 0.0,
    transport_mode: str = "reproject",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-step pushforward rollout loss in DECODED-SPECIES space (closed-loop tracking, F2/RC-2).

    ``transport_mode``:
    - ``"reproject"`` (default): the legacy loop z ← P(z)+Δτ·ω_Z(P(z)) with P=E∘D re-projection.
    - ``"direct"`` (stable latent ODE): z ← z + Δτ·model.latent_field(z,q), NO re-projection; the
      decoder/rate/energy heads read the RAW transported z. Stability comes from the field's
      structural −β·z damping (+ the dynamics_contraction penalty), so the decoder stays faithful.

    Pushforward trick (Brandstetter et al.): roll the latent K steps feeding the model its OWN output
    WITHOUT gradient (clamped, to reach the drifted states the deployed loop actually visits), then
    take a SINGLE gradient step from each visited state, decode to standardised composition, and match
    the TRUE next composition. Back-prop touches only the single step (not the whole unstable unroll →
    no gradient explosion), yet trains E/D/ω_Z to CORRECT their own drift — the closed-loop tracking
    that 1-step (Lagrangian) and contraction-only training cannot give.

        z_0 = E·y(inlet);  visited[j] reached by the no-grad rollout
        loss = mean_j ‖ decode( project(visited[j]) + Δτ·ω_Z(project(visited[j])) ) − y_true[j+1] ‖²

    Energy-aware term (``energy_weight`` > 0): the species term above trains only COMPOSITION
    tracking, so the deployed energy S_E = Σ hᵢ·ω̇ᵢ is optimised only indirectly and its rollout
    integral lags. When enabled, at each rolled step we ALSO evaluate the rate-derived absorption at
    the DEPLOYED input — ``project(z_next)`` (the same E∘D state the UDF feeds its rate head) — and
    match it (arcsinh-space, at the fixed physical scale) to the true S_E at that trajectory step.
    Gradients reach the decoder, ω_Z AND the rate head, training them to keep the energy on track
    while the composition drifts. Returns ``(total_loss, parts)`` for per-term logging.
    """
    K = seq_dtau.shape[1]
    do_energy = (energy_weight > 0.0 and seq_abs is not None
                 and absorption_from_rates_fn is not None and rates_to_physical_fn is not None)

    direct = (transport_mode == "direct")

    def omega(zp, q):  # physical ω_Z = sinh(arcsinh-head)·s_Z (head emits arcsinh-space)
        return torch.sinh(model.latent_source(zp, q).clamp(-20.0, 20.0)) * arcsinh_latent_scale.unsqueeze(0)

    def step(zin, dtau_col, q):
        """One transport step. direct: z + Δτ·f(z) (no reproject); reproject: P(z) + Δτ·ω_Z(P(z))."""
        if direct:
            return zin + dtau_col * model.latent_field(zin, q)
        zp = model.project(zin, q)
        return zp + dtau_col * omega(zp, q)

    with torch.no_grad():                                  # collect the visited (drifted) states
        z = model.encode(seq_ystd[:, 0, :]).clamp(-clamp, clamp)
        visited = [z]
        for j in range(1, K):
            z = step(z, seq_dtau[:, j - 1].unsqueeze(1), seq_q[:, j, :]).clamp(-clamp, clamp)
            visited.append(z)

    s_loss = seq_ystd.new_zeros(())
    e_loss = seq_ystd.new_zeros(())
    for j in range(K):                                     # 1-step grad from each visited state
        zin = visited[j].detach()
        qj1 = seq_q[:, j + 1, :]
        z_next = step(zin, seq_dtau[:, j].unsqueeze(1), qj1)
        y_pred_std = model.decode(z_next, qj1)
        s_loss = s_loss + _mse(y_pred_std, seq_ystd[:, j + 1, :]).mean()
        if do_energy:
            # rate-head input at the rolled step: raw z_next (direct) or project(z_next) (reproject)
            zpe = z_next if direct else model.project(z_next, qj1)
            rates_phys = rates_to_physical_fn(model.rates_from_latent(zpe, qj1))
            abs_pred = absorption_from_rates_fn(rates_phys, qj1).squeeze(-1)
            e_loss = e_loss + energy_rate_tied_loss(
                abs_pred, seq_abs[:, j + 1], energy_arcsinh_scale)

    s_loss = s_loss / K
    total = s_loss
    parts = {"pf_species": float(s_loss.detach())}
    if do_energy:
        e_loss = e_loss / K
        total = total + energy_weight * e_loss
        parts["pf_energy"] = float(e_loss.detach())
    return total, parts


def dynamics_contraction_penalty(
    model,
    z: torch.Tensor,
    q: torch.Tensor,
    dtau: float,
    gain_target: float = 1.0,
    eps: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contraction of the DIRECT-transport map g(z) = z + Δτ·f(z,q) (stable-latent-ODE stability).

    The stable-ODE deployment advances the latent WITHOUT the E∘D re-projection, so stability must
    live in the field f = model.latent_field (which already carries a structural −β·z damping floor).
    This penalises the residual expansion relu(‖g(z+δ)−g(z)‖/‖δ‖ − gain_target)² for random unit
    perturbations δ, tightening the field so the deployed advance is non-expansive — the analogue of
    the E∘D contraction penalty, but on the dynamics map, leaving the decoder free to be faithful.

    Δτ is a representative (coarse storage-grid) step; the deployed CFD Δτ is finer ⇒ z+Δτ·f is closer
    to identity ⇒ strictly more contractive, so enforcing it here is conservative. Returns
    ``(penalty, mean_gain)``.
    """
    delta = torch.randn_like(z)
    delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-12) * eps
    g0 = z + dtau * model.latent_field(z, q)
    g1 = (z + delta) + dtau * model.latent_field(z + delta, q)
    gain = (g1 - g0).norm(dim=1) / (delta.norm(dim=1) + 1e-12)
    return (torch.relu(gain - gain_target) ** 2).mean(), gain.mean().detach()


def atom_projection_penalty(
    rates_mass: torch.Tensor,
    molar_mass: torch.Tensor,
    nonconserving_projector: torch.Tensor,
) -> torch.Tensor:
    """Mean ``‖r_molar · Q‖²`` — the element-violating component of the predicted molar rates.

    ``Q`` (a CONSTANT matrix from :func:`scarfs.models.physics.atom_conservation_projector`) is the
    orthogonal projector onto ``col(A)``; ``r_molar · Q`` is the part of the molar rates that does
    NOT conserve atoms.  This is the better-conditioned sibling of :func:`atom_balance_penalty`
    (which weights elements by their raw atom-count magnitude); both vanish exactly for an
    atom-conserving rate vector, but the projector form is the scale-free L2 distance from the
    conserving subspace, so a single ``atom_projection_weight`` behaves consistently across element
    sets.  Closure is exact only over the full carrier set; on the reduced active set it is a
    consistency pressure (documented limitation, RC-3).

    Parameters
    ----------
    rates_mass
        ``(batch, n_active)`` predicted physical mass rates ρ·dY/dt [kg m-3 s-1].
    molar_mass
        ``(n_active,)`` molar masses [kg kmol-1].
    nonconserving_projector
        ``(n_active, n_active)`` constant projector ``Q`` onto the non-conserving subspace.
    """
    molar = rates_mass / molar_mass.unsqueeze(0)                  # (batch, n_active) molar rates
    nonconserving = molar @ nonconserving_projector              # (batch, n_active)
    return (nonconserving ** 2).sum(dim=1).mean()


def realizability_penalty(
    rates_mass: torch.Tensor,
    rho: torch.Tensor,
    y_active_mass: torch.Tensor,
    dt: float,
    rho_floor: float = 1e-6,
) -> torch.Tensor:
    """Soft physical-realizability floor: a species cannot be consumed faster than it exists.

    Over a representative timestep ``dt`` the *mass fraction* of species *i* consumed is
    ``(−R_i)⁺·dt / ρ``; it cannot exceed the mass fraction present ``Y_i``.  We penalise the
    squared overshoot in mass-fraction units::

        penalty = mean Σ_i relu( (−R_i)⁺·dt / ρ − Y_i )²

    Dividing by ``ρ`` (not ``ρ·Y_i``) keeps the term dimensionless AND finite for trace species
    where ``Y_i → 0`` (a relative ``/ρ·Y_i`` form overflows float32).  Zero unless a predicted
    consumption rate would drive a mass fraction negative within ``dt``.  Magnitude-only and
    per-cell cheap, so it respects the source-term + plain-C constraints; arcsinh scaling is
    sign-preserving but NOT realizability-aware, which is the gap this closes.

    Parameters
    ----------
    rates_mass
        ``(batch, n_active)`` predicted mass rates ρ·dY/dt [kg m-3 s-1] (negative = consumption).
    rho
        ``(batch,)`` mixture density [kg m-3].
    y_active_mass
        ``(batch, n_active)`` mass fractions of the active species [-].
    dt
        Representative timestep [s] over which depletion is bounded.
    """
    consumed_frac = (-rates_mass).clamp(min=0.0) * dt / (rho.unsqueeze(1) + rho_floor)
    over = torch.relu(consumed_frac - y_active_mass.clamp(min=0.0))
    return (over ** 2).sum(dim=1).mean()


def keq_consistency_penalty(
    omega_molar_rxn: torch.Tensor,
    ln_quotient: torch.Tensor,
    ln_keq: torch.Tensor,
    stoich: torch.Tensor,
    extent_scale: torch.Tensor | float = 1.0,
    width: float = 1.0,
) -> torch.Tensor:
    """Equilibrium-consistency penalty for ONE reversible elementary step.

    Thermodynamics forbids a net forward rate once the reaction quotient ``Q`` reaches the
    equilibrium constant ``Keq(T)``.  We form the step's net extent rate by projecting the molar
    rates of the involved species onto the stoichiometric vector,

        ξ̇ = (Σ νᵢ·ω_molar,i) / (Σ νᵢ²)

    and penalise its squared, scale-normalised magnitude weighted by a Gaussian bump that is
    peaked at equilibrium (``Δ = lnQ − lnKeq → 0``)::

        penalty = mean[ exp(−(Δ/width)²) · (ξ̇ / extent_scale)² ]

    Far from equilibrium the weight ≈ 0, so kinetics are left to the data; near equilibrium the
    net extent is driven toward zero.  This is a one-sided thermodynamic *consistency pressure*,
    not a sign-correcting constraint (inference-time damping cannot flip a wrong-sign rate — see
    the proposal's risk note); scope it to elementally-exact overall steps only.

    Parameters
    ----------
    omega_molar_rxn
        ``(batch, m)`` molar net rates [kmol m-3 s-1] of the *m* species in the step.
    ln_quotient, ln_keq
        ``(batch,)`` ln of the reaction quotient and equilibrium constant (same convention).
    stoich
        ``(m,)`` signed stoichiometric coefficients (reactants negative, products positive).
    extent_scale
        Scalar normaliser for ξ̇ (e.g. median |ξ̇| on the train split) so the term is O(1).
    width
        Gaussian width ``ε`` in ln-units around equilibrium.
    """
    w = torch.exp(-((ln_quotient - ln_keq) / width) ** 2)                     # (batch,)
    xi_dot = (omega_molar_rxn * stoich.unsqueeze(0)).sum(dim=-1) / (stoich ** 2).sum()
    return (w * (xi_dot / extent_scale) ** 2).mean()


def transport_property_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    log_floor: float = 1e-30,
) -> torch.Tensor:
    """Log-space MSE for strictly-positive transport properties (μ, k, …).

    Viscosity and conductivity are positive and span well under a decade across the front, so a
    log transform gives a well-scaled, sign-safe target without arcsinh machinery.

    Parameters
    ----------
    pred, target
        ``(batch, n_props)`` predicted / database transport properties (same column order).
    """
    lp = torch.log(pred.clamp(min=log_floor))
    lt = torch.log(target.clamp(min=log_floor))
    return _mse(lp, lt).mean()


# ---------------------------------------------------------------------------
# Merged composite (B1c)
# ---------------------------------------------------------------------------

def merged_composite(
    model,
    y_std_scaled: torch.Tensor,
    q: torch.Tensor,
    target_rates_scaled: torch.Tensor,
    absorption_target: torch.Tensor,
    dydt_dry_phys: torch.Tensor,
    rho: torch.Tensor,
    row_weights: torch.Tensor,
    enthalpy_weights: torch.Tensor,
    species_weights_qoi: torch.Tensor,
    *,
    arcsinh_latent_scale: torch.Tensor,
    sigma_active: torch.Tensor,
    sigma_comp_all: torch.Tensor,
    active_col_idx: torch.Tensor,
    energy_arcsinh_scale: torch.Tensor | float = 1.0,
    rates_to_physical_fn=None,
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
    atom_projection_weight: float = 0.0,
    keq_weight: float = 0.0,
    realizability_weight: float = 0.0,
    realizability_dt: float = 1.0e-3,
    transport_weight: float = 0.0,
    noise_std: float = 0.0,
    rollout_mode: str = "manifold",
    # contraction penalty on the round-trip projection P=E∘D (a-posteriori stability)
    contraction_weight: float = 0.0,
    contraction_gain: float = 0.9,
    contraction_eps: float = 0.1,
    slow_manifold_weight: float = 0.0,
    slow_manifold_gain: float = 0.5,
    # stable latent ODE (direct transport, no E∘D re-projection)
    transport_mode: str = "reproject",
    dynamics_contraction_weight: float = 0.0,
    dynamics_contraction_gain: float = 1.0,
    dynamics_dtau: float = 1.0e-3,
    # lagrangian rollout (optional)
    idx_t: torch.Tensor | None = None,
    idx_tp1: torch.Tensor | None = None,
    dtau: torch.Tensor | None = None,
    # atom balance (optional)
    molar_mass: torch.Tensor | None = None,
    element_matrix: torch.Tensor | None = None,
    # atom-projection (constant null-space projector; optional)
    atom_nonconserving_projector: torch.Tensor | None = None,
    # Keq equilibrium consistency (optional) — closure (rates_phys, y_std, q, rho, n_total) -> scalar
    keq_penalty_fn=None,
    keq_n_total: torch.Tensor | None = None,
    # realizability (optional) — needs active-species composition mean to un-standardise Y
    comp_mean_active: torch.Tensor | None = None,
    # transport-property heads (optional) — DB targets in (μ, k, …) column order
    transport_targets: torch.Tensor | None = None,
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
    target_rates_scaled
        ``(batch, n_active)`` ArcsinhScaler-space rate targets (the bundle's Y arrays) — the
        rate head's native output space. ``rates_to_physical_fn`` recovers physical rates
        where the physics needs them (energy tie, consistency, atom balance).
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
    # -- forward pass (single pass; heads read the noise-injected latent) ------------------
    z = model.encode(y_std_scaled)
    y_recon = model.decode(z, q)
    z_proj = model.encoder(y_recon)                # manifold projection, reusing the decode

    # DIRECT (stable-ODE) transport feeds heads the RAW latent (that is what the UDF transports —
    # no E∘D re-projection); REPROJECT mode feeds the manifold-projected latent (legacy).
    z_used = z if transport_mode == "direct" else z_proj
    if noise_std > 0:
        z_used = z_used + noise_std * torch.randn_like(z_used)

    z_out, z_proj_out, y_recon_out = z, z_proj, y_recon
    latent_src = model.latent_source(z_used, q)    # (batch, k)
    rates_pred = model.rates_from_latent(z_used, q)  # (batch, n_active)
    absorption_head = model.absorption(z_used, q)  # (batch, 1) strictly positive

    # ground-truth latent source ż = d/dt E(Y_std) = J_E·(Ẏ_dry ⊘ σ_dry). For a LINEAR encoder
    # J_E = W (constant) → the matmul below. For a NONLINEAR (MLP) encoder J_E is state-dependent →
    # use a forward-mode JVP of the encoder at Y_std in the direction (Ẏ ⊘ σ). encoder_weight_full
    # stays None in the nonlinear case, which auto-disables the linear-weight split-head consistency.
    _ydot_std = dydt_dry_phys / sigma_comp_all.unsqueeze(0)
    if getattr(model, "encoder_is_linear", True):
        encoder_weight_full = model.encoder.weight.detach()        # (k, n_dry)
        z_dot_true = _ydot_std @ encoder_weight_full.t()
    else:
        import torch.func as _tfunc
        encoder_weight_full = None
        _, z_dot_true = _tfunc.jvp(lambda _y: model.encode(_y), (y_std_scaled,), (_ydot_std,))
        z_dot_true = z_dot_true.detach()

    # single inversion point: physical rates for the physics-facing terms (energy tie,
    # consistency, atom balance); identity when no inverse fn is supplied (tests/stubs).
    rates_pred_phys = (
        rates_to_physical_fn(rates_pred) if rates_to_physical_fn is not None else rates_pred
    )

    parts: dict[str, float] = {}
    total = torch.tensor(0.0, device=y_std_scaled.device)

    # -- 1. Rate loss in the head's native scaled space (enthalpy + tail weights) --------
    if rates_pred is not None and rate_weight > 0.0:
        rl = scaled_rate_loss(rates_pred, target_rates_scaled, row_weights, enthalpy_weights)
        total = total + rate_weight * rl
        parts["rate"] = float(rl.detach())

    # -- 2. Latent-source loss -----------------------------------------------------------
    if latent_src is not None and latent_source_weight > 0.0:
        lsl = latent_source_loss(latent_src, z_dot_true, arcsinh_latent_scale)
        total = total + latent_source_weight * lsl
        parts["latent_source"] = float(lsl.detach())

    # -- 3–5. Energy terms (arcsinh space at the fixed physical scale; tail row-weighted) --
    if absorption_from_rates_fn is not None:
        # rate-derived absorption (differentiable wrt rates_pred via the physical inverse)
        abs_from_rates = absorption_from_rates_fn(rates_pred_phys, q)  # (batch,) or (batch,1)
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
    if (latent_src is not None and rates_pred is not None and consistency_weight > 0.0
            and encoder_weight_full is not None):  # linear-encoder-weight term; skipped for MLP encoder
        cpen = split_head_consistency(
            latent_src, rates_pred_phys, rho, dydt_dry_phys, z_dot_true,
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

    # -- 9b. Contraction of the round-trip projection P=E∘D (closed-loop stability) -------
    # The deployed rollout advances z<-P(z)+Δτ·ω_Z(P(z)); it is stable iff P is non-expansive.
    # Measured gain ≈6.6 ⇒ exponential drift. Penalise the directional gain exceeding the target
    # for random latent perturbations, pushing grads into encoder+decoder to flatten the geometry.
    if contraction_weight > 0.0:
        delta = torch.randn_like(z_out)
        delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-12) * contraction_eps
        p0 = model.project(z_out, q)
        p1 = model.project(z_out + delta, q)
        gain = (p1 - p0).norm(dim=1) / (delta.norm(dim=1) + 1e-12)
        contraction = (torch.relu(gain - contraction_gain) ** 2).mean()
        total = total + contraction_weight * contraction
        parts["contraction"] = float(contraction.detach())
        parts["contraction_gain_mean"] = float(gain.mean().detach())

    # -- 9c. Slow-manifold (anisotropic) contraction: contract ONLY transverse-to-tangent modes ----
    # The isotropic contraction over-damps the SLOW along-trajectory direction (→ wrong attractor).
    # Here we contract only the FAST directions transverse to the trajectory tangent z_dot_true,
    # leaving the slow mode free — the slow-manifold structure that gives Layer-2 attraction.
    if slow_manifold_weight > 0.0:
        t = z_dot_true / (z_dot_true.norm(dim=1, keepdim=True) + 1e-8)   # unit trajectory tangent
        d = torch.randn_like(z_out)
        d = d - (d * t).sum(dim=1, keepdim=True) * t                     # remove tangent ⇒ transverse
        d = d / (d.norm(dim=1, keepdim=True) + 1e-12) * contraction_eps
        p0 = model.project(z_out, q)
        p1 = model.project(z_out + d, q)
        gain_perp = (p1 - p0).norm(dim=1) / (d.norm(dim=1) + 1e-12)
        slow = (torch.relu(gain_perp - slow_manifold_gain) ** 2).mean()
        total = total + slow_manifold_weight * slow
        parts["slow_manifold"] = float(slow.detach())
        parts["slow_gain_perp_mean"] = float(gain_perp.mean().detach())

    # -- 9d. Dynamics-map contraction (stable latent ODE, DIRECT transport) ----------------
    # For direct transport the stability lives in the field: penalise the gain of z↦z+Δτ·f(z,q)
    # (f=model.latent_field, already carrying the structural −β·z floor) so the deployed advance is
    # non-expansive WITHOUT E∘D — leaving the decoder free to reconstruct composition faithfully.
    if dynamics_contraction_weight > 0.0 and transport_mode == "direct":
        dcp, dgain = dynamics_contraction_penalty(
            model, z_out, q, dynamics_dtau, dynamics_contraction_gain, contraction_eps)
        total = total + dynamics_contraction_weight * dcp
        parts["dyn_contraction"] = float(dcp.detach())
        parts["dyn_gain_mean"] = float(dgain)

    # -- 10. Atom-balance (physical rates) -------------------------------------------------
    if atom_balance_weight > 0.0 and molar_mass is not None and element_matrix is not None and rates_pred is not None:
        abp = atom_balance_penalty(rates_pred_phys, molar_mass, element_matrix)
        total = total + atom_balance_weight * abp
        parts["atom_balance"] = float(abp.detach())

    # -- 10b. Atom-projection (constant null-space projector; better-conditioned) ----------
    if (atom_projection_weight > 0.0 and atom_nonconserving_projector is not None
            and molar_mass is not None and rates_pred is not None):
        app = atom_projection_penalty(rates_pred_phys, molar_mass, atom_nonconserving_projector)
        total = total + atom_projection_weight * app
        parts["atom_projection"] = float(app.detach())

    # -- 10c. Keq equilibrium consistency (scoped, opt-in) --------------------------------
    if keq_weight > 0.0 and keq_penalty_fn is not None and rates_pred is not None:
        kqp = keq_penalty_fn(rates_pred_phys, y_std_scaled, q, rho, keq_n_total)
        total = total + keq_weight * kqp
        parts["keq"] = float(kqp.detach())

    # -- 10d. Realizability floor (consumption cannot deplete a species within dt) ---------
    if (realizability_weight > 0.0 and comp_mean_active is not None and rates_pred is not None):
        y_active = y_std_scaled[:, active_col_idx] * sigma_active.unsqueeze(0) + comp_mean_active.unsqueeze(0)
        rzp = realizability_penalty(rates_pred_phys, rho, y_active, realizability_dt)
        total = total + realizability_weight * rzp
        parts["realizability"] = float(rzp.detach())

    # -- 10e. Transport-property heads (μ, k, …) — log-space MSE ---------------------------
    if transport_weight > 0.0 and transport_targets is not None and hasattr(model, "transport"):
        tp_pred = model.transport(z_used, q)
        tpl = transport_property_loss(tp_pred, transport_targets)
        total = total + transport_weight * tpl
        parts["transport"] = float(tpl.detach())

    # -- 11. Rollout term ----------------------------------------------------------------
    if rollout_mode == "lagrangian" and idx_t is not None and len(idx_t) > 0:
        if latent_src is not None:
            z_t_lag = z_out[idx_t]
            z_tp1_true = z_out.detach()[idx_tp1]  # treat future true as stopgrad
            # physical ω_Z = sinh(head)·s_Z; pre-sinh clamp at ±20 is a pure inf-guard
            # (sinh(20) ≈ 2.4e8 — far beyond any physical latent source magnitude).
            src_t_lag = torch.sinh(latent_src[idx_t].clamp(-20.0, 20.0)) * arcsinh_latent_scale.unsqueeze(0)
            roll_loss = lagrangian_rollout_loss(z_t_lag, z_tp1_true, src_t_lag, dtau)
            total = total + roll_loss
            parts["lagrangian_rollout"] = float(roll_loss.detach())

    parts["total"] = float(total.detach())
    return total, parts

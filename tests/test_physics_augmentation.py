"""Unit tests for the physics-augmentation additions.

Covers the constant atom-conservation projector (physics.py), the NASA7 entropy/Gibbs twins
(thermo.py), and the four new training penalties (losses.py): atom-projection, realizability,
Keq equilibrium-consistency, and the transport-property loss.  These are the focused guards;
the end-to-end wiring is exercised by ``test_train_merged_integration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from scarfs.models.physics import atom_conservation_projector, atom_balance_residual
from scarfs.models.thermo import SpeciesThermo, R_J_PER_KMOL_K

torch = pytest.importorskip("torch")

from scarfs.training import losses as L  # noqa: E402


# ---------------------------------------------------------------------------
# Constant atom-conservation projector (physics.py)
# ---------------------------------------------------------------------------

def test_projector_symmetric_and_idempotent():
    A = np.array([[2.0, 4.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # 4 species, 2 elements
    Q = atom_conservation_projector(A)
    assert np.allclose(Q, Q.T)
    assert np.allclose(Q @ Q, Q, atol=1e-10)


def test_projector_zeroes_conserving_rate():
    # C2H4 (C2H4) consumed, 2x CH2-equivalent produced — choose A so a known r conserves atoms.
    A = np.array([[1.0], [1.0]])  # 2 species, 1 element, 1 atom each
    Q = atom_conservation_projector(A)
    r_conserving = np.array([[-3.0, 3.0]])  # net element rate 0
    assert np.allclose(r_conserving @ Q, 0.0, atol=1e-10)
    r_violating = np.array([[1.0, 1.0]])    # net element rate 2 != 0
    assert np.linalg.norm(r_violating @ Q) > 1e-6


def test_projection_penalty_matches_balance_zero_set():
    # A conserving molar rate gives ~0 for BOTH penalties; a violating one gives >0 for both.
    W = torch.tensor([1.0, 1.0])
    A_np = np.array([[1.0], [1.0]])
    Q = torch.as_tensor(atom_conservation_projector(A_np), dtype=torch.float32)
    rates_conserving = torch.tensor([[-2.0, 2.0]])
    rates_violating = torch.tensor([[2.0, 3.0]])
    assert float(L.atom_projection_penalty(rates_conserving, W, Q)) < 1e-10
    assert float(L.atom_projection_penalty(rates_violating, W, Q)) > 1e-6
    # consistent with the element-residual sign (both zero together)
    assert np.allclose(atom_balance_residual(rates_conserving.numpy(), W.numpy(), A_np), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# NASA7 entropy / Gibbs (thermo.py)
# ---------------------------------------------------------------------------

def test_gibbs_equals_h_minus_ts_and_torch_twin_matches():
    th = SpeciesThermo.from_mechanism_yaml("chem_ForTransport.yaml", ["C2H6", "C2H4", "H2"])
    T = np.array([900.0, 1100.0, 1350.0])
    g = th.g_molar(T)
    expected = th.h_molar(T) - T[:, None] * th.s_molar(T)
    assert np.allclose(g, expected, rtol=1e-10)
    g_torch = th.g_molar_torch(torch.as_tensor(T)).numpy()
    assert np.allclose(g, g_torch, rtol=1e-4)


def test_keq_sign_for_dehydrogenation():
    # C2H6 <-> C2H4 + H2 is endothermic; ln Kp = -ΔG/RT should INCREASE with T (more dissociation).
    th = SpeciesThermo.from_mechanism_yaml("chem_ForTransport.yaml", ["C2H6", "C2H4", "H2"])
    lnK = []
    for T in (900.0, 1400.0):
        g = th.g_molar(np.array([T]))[0]  # [C2H6, C2H4, H2]
        dG = g[1] + g[2] - g[0]
        lnK.append(-dG / (R_J_PER_KMOL_K * T))
    assert lnK[1] > lnK[0]


# ---------------------------------------------------------------------------
# Keq consistency penalty (losses.py)
# ---------------------------------------------------------------------------

def test_keq_penalty_active_at_equilibrium_inactive_far_away():
    stoich = torch.tensor([-1.0, 1.0, 1.0])
    omega = torch.tensor([[-1.0, 1.0, 1.0]])  # unit extent in the dehydrogenation direction
    ln_keq = torch.tensor([0.0])
    # at equilibrium (Δ=0): weight=1, extent=1 -> penalty=1
    at_eq = float(L.keq_consistency_penalty(omega, ln_keq, ln_keq, stoich, extent_scale=1.0, width=1.0))
    assert at_eq == pytest.approx(1.0, rel=1e-5)
    # far from equilibrium: Gaussian weight ~0 -> penalty ~0 even with a large net extent
    far = float(L.keq_consistency_penalty(omega, ln_keq + 10.0, ln_keq, stoich, 1.0, 1.0))
    assert far < 1e-6


# ---------------------------------------------------------------------------
# Realizability penalty (losses.py)
# ---------------------------------------------------------------------------

def test_realizability_zero_when_safe_positive_when_over_depleting():
    rho = torch.tensor([1.0])
    Y = torch.tensor([[1e-3, 1e-3]])
    # safe: small consumption within dt
    safe = float(L.realizability_penalty(torch.tensor([[-1e-4, 0.0]]), rho, Y, dt=1.0))
    assert safe == pytest.approx(0.0, abs=1e-12)
    # over-depleting: consumes 10000x the available mass within dt
    over = float(L.realizability_penalty(torch.tensor([[-10.0, 0.0]]), rho, Y, dt=1.0))
    assert over > 1.0
    # production (positive rate) is never penalised
    prod = float(L.realizability_penalty(torch.tensor([[+10.0, +10.0]]), rho, Y, dt=1.0))
    assert prod == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Transport-property loss (losses.py)
# ---------------------------------------------------------------------------

def test_transport_loss_zero_on_match_positive_otherwise():
    target = torch.tensor([[2.0e-5, 0.1], [3.0e-5, 0.12]])
    assert float(L.transport_property_loss(target, target)) == pytest.approx(0.0, abs=1e-12)
    worse = float(L.transport_property_loss(2.0 * target, target))
    assert worse == pytest.approx(np.log(2.0) ** 2, rel=1e-4)


# ---------------------------------------------------------------------------
# merged_composite wiring of the new terms (real MergedCoil, incl. transport head + grads)
# ---------------------------------------------------------------------------

def test_merged_composite_runs_all_new_terms_and_grads_flow():
    from scarfs.models.neuralcoil import MergedCoil

    batch, n_dry, n_active, k = 24, 8, 5, 3
    model = MergedCoil(n_dry=n_dry, n_energy_active=n_active, latent_dim=k, n_transport=2)
    model.set_transport_calibration([1e-5, 0.1])

    rng = np.random.default_rng(0)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    y_std = t(rng.standard_normal((batch, n_dry)))
    q = t(rng.standard_normal((batch, 4)))
    target_rates = t(rng.standard_normal((batch, n_active)))
    abs_target = t(np.abs(rng.standard_normal(batch)) + 1.0)
    dydt_dry = t(rng.standard_normal((batch, n_dry)))
    rho = torch.ones(batch)
    molar_mass = t(np.array([28.0, 16.0, 2.0, 30.0, 42.0]))
    element_matrix = t(np.array([[2, 4], [1, 4], [0, 2], [2, 6], [3, 6]], dtype=float))  # C,H
    from scarfs.models.physics import atom_conservation_projector
    Q = t(atom_conservation_projector(element_matrix.numpy()))
    comp_mean_active = t(rng.standard_normal(n_active))
    transport_targets = t(np.abs(rng.standard_normal((batch, 2))) + 1e-3)

    def keq_fn(rp, ys, qq, rho_b, n_tot):
        stoich = t([-1.0, 1.0, 1.0])
        return L.keq_consistency_penalty(
            rp[:, :3], torch.zeros(rp.shape[0]), torch.zeros(rp.shape[0]), stoich, 1.0, 1.0)

    total, parts = L.merged_composite(
        model=model, y_std_scaled=y_std, q=q,
        target_rates_scaled=target_rates, absorption_target=abs_target,
        dydt_dry_phys=dydt_dry, rho=rho,
        row_weights=torch.ones(batch), enthalpy_weights=torch.ones(n_active),
        species_weights_qoi=torch.ones(n_dry),
        arcsinh_latent_scale=torch.ones(k), sigma_active=torch.ones(n_active),
        sigma_comp_all=torch.ones(n_dry), active_col_idx=torch.arange(n_active),
        # new physics terms — all on at once
        atom_balance_weight=0.0, atom_projection_weight=5e-3,
        keq_weight=1e-2, realizability_weight=1e-2, realizability_dt=1e-3,
        transport_weight=0.05,
        molar_mass=molar_mass, element_matrix=element_matrix,
        atom_nonconserving_projector=Q, comp_mean_active=comp_mean_active,
        keq_penalty_fn=keq_fn, keq_n_total=torch.ones(batch),
        transport_targets=transport_targets,
    )

    for term in ("atom_projection", "keq", "realizability", "transport"):
        assert term in parts, f"{term} missing from {sorted(parts)}"
        assert np.isfinite(parts[term])
    assert torch.isfinite(total)
    total.backward()
    # transport head must receive gradient
    grads = [p.grad for p in model.transport_net.parameters()]
    assert all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)

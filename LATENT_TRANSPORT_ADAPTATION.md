# Latent transport — why it fails a-posteriori, and how to adapt it

Status 2026-06-29. The merged surrogate is **a-priori excellent** (best model `merged_contract090`:
held-out TEST energy relRMSE **0.0531 = 10.15× / R² 0.997** — the 10× goal met) but **free-running
latent transport** (CFD integrating dZ/dτ = ω_Z(Z), ChemZIP-style) is **a-posteriori unstable**.
Five principled fixes failed to make it stable *and* accurate. This is the diagnosis and the path.

## The two-layer root cause (measured, not assumed)

The deployed rollout advances `z ← F(z) = P(z) + Δτ·ω_Z(P(z))`, with `P = E∘D` the manifold
projection (encode∘decode). Two distinct failures, in series:

**Layer 1 — geometric expansion.** `scripts/diag_projection_gain.py` measured the one-step Lipschitz
gain `‖F(z+δ)−F(z)‖/‖δ‖ ≈ 6.6` (and `G_P ≈ G_F`, so it's the projection `E∘D`, not the source). A
gain ≫ 1 amplifies every perturbation → exponential drift. **Fixed** by the contraction penalty
(`contraction_weight`): median gain → ~0.8, and the latent no longer blows up (env-clamp → 0).

**Layer 2 — non-tracking (revealed once Layer 1 is fixed).** Even *bounded*, the rolled latent settles
onto a **wrong** trajectory (0D rollout relRMSE in the **hundreds** at every contraction gain 0.8/0.9/
0.97). The learned `(E, D, ω_Z)` define an autonomous dynamical system whose attractor, off the data
manifold, is **not** the true chemistry path. Per-point (a-priori) training never constrained the
*closed-loop* attractor to follow truth — so "bounded" ≠ "tracking". Contraction alone cannot fix this
(it controls boundedness, not which trajectory the system contracts toward).

## Why each attempt addressed only one layer

| attempt | layer targeted | outcome |
|---|---|---|
| spectral-norm | 1 (weak — per-layer ≠ composite `E∘D`) | gain stayed ≫1; unstable |
| diffusion (1D PDE) | 1 (external) | advection-dominated (Pe~1.9e4); no realistic D stabilises |
| contraction penalty | 1 | **works** → bounded, but Layer 2 then dominates (relRMSE ~hundreds) |
| Lagrangian 1-step | 2 | trains only at TRUE states; can't correct off-manifold drift; hurt a-priori |
| pushforward (full) | 2 | back-prop through the *unstable* unroll explodes (1e28) |
| pushforward-trick (ω_Z only) | 2 | bounded but didn't track (frozen E/D can't reshape the attractor) |

The pattern: every method fixed at most ONE layer. **Nothing has yet fixed both at once.**

## Recommended adaptation #1 — contraction + pushforward, JOINTLY, in species space

The key unlock: **contraction (Layer 1) makes the unrolled trajectory bounded, which is exactly the
precondition that makes pushforward training (Layer 2) feasible** — bounded rollouts have bounded
gradients, so the explosion that killed full pushforward disappears. Train `E, D, ω_Z` **jointly**
(not ω_Z alone) with BOTH:
- the **contraction penalty** (keeps the rollout bounded), AND
- a **multi-step pushforward rollout loss evaluated in DECODED SPECIES space** — roll K steps feeding
  the model its own output (pushforward trick: no-grad rollout to reach drifted states, gradient step
  from each), decode to Y, penalise `‖Y_pred − Y_true‖` along the trajectory. Species space is bounded
  by the simplex and physically meaningful (avoids the latent-MSE blow-up), and joint E/D training lets
  the **representation co-adapt** so its autonomous dynamics track truth.

We ran contraction and pushforward *separately* (each fixed one layer); the **combination, jointly
trained**, is the logical next experiment and is now feasible. Moderate effort: extend the existing
gated `contraction_weight` path with a species-space pushforward term (reuse `case_step_pairs` →
K-step sequences). Verify with the existing `aposteriori_rollout.py` (0D) + `pfr_1d_diffusion.py` (1D).

## Recommended adaptation #2 — slow-manifold latent (addresses the cause, not the symptom)

Deeper reason the dynamics are ill-conditioned: the latent is a **variance**-based compression (≈PCA
encoder), not a **timescale**-aware one. Cracking chemistry is stiff — fast radical modes (quasi-steady
state) + slow molecular modes. Proper ChemZIP/ILDM/REDIM parametrises the **slow invariant manifold**
with the fast modes *slaved*; then (a) the latent dynamics are non-stiff, and (b) the fast directions
**contract toward the manifold by construction** — which is precisely the Layer-2 attraction that's
missing. Adaptation: train the encoder so the latent dynamics are non-stiff / fast modes slaved (e.g.
penalise fast latent eigen-directions, or learn the latent as the slow manifold), and pair with
**implicit/semi-implicit integration** in the UDF for residual stiffness. This is "do ChemZIP properly";
the current generic autoencoder skipped the slow-manifold structure, which is the true architectural gap.

## Guaranteed fallback — species-transport local closure

If the above research doesn't converge: deploy the (excellent) model as a **local source-term closure**.
CFD transports the 61 energy-active species; the rate head supplies dY/dt + S_E at each *resolved* state.
No latent integration ⇒ neither Layer 1 nor Layer 2 exists ⇒ **stable by construction**. Cost: ~61 UDS
vs 32 latent (loses the compression saving), but it ships with the 10× model.

## Recommended sequence

1. **Contraction + pushforward (species-space, joint)** — build on current code; feasible now.
2. If insufficient: **slow-manifold latent + implicit integration** — architectural; addresses the root.
3. Fallback: **species-transport** — guaranteed, uses the 10× model as-is.

Test protocol for each (already built): `full_test_eval.py` (a-priori), `diag_projection_gain.py`
(Layer-1 gain < 1?), `aposteriori_rollout.py` (0D rollout relRMSE → O(1)?), `pfr_1d_diffusion.py`
(1D converges to truth?). Success = a-priori ≥ ~8× AND 0D relRMSE → O(1) with G_F < 1.

## RESULTS (2026-06-29) — what each adaptation actually did

All trained on the full DB (k=32), evaluated with the protocol above. The two latent adaptations hit
a **Pareto wall**; species-transport (the fallback) is the one that works.

| config | a-priori | G_F (Layer 1) | 0D rollout (Layer 2) | verdict |
|---|--:|--:|--:|---|
| `merged_contract090` (contraction only) | **10.15×** | 0.78 ✓ | relRMSE ~hundreds | bounded, no tracking |
| **#1** `merged_joint` (contraction+pushforward) | 5.29× | 0.49–0.81 ✓ | relRMSE **158** (best latent); ∫S_E **0.34**; 1D **87** | best latent tracking, but a-priori too low |
| **#2** `merged_slowmanifold` (anisotropic) | **8.10×** | 0.70 ✓ | relRMSE 182; 1D 333 | kept a-priori, did NOT track |
| **#3** `aposteriori_species.py` (species transport) | — (uses contract090, 10.15×) | n/a (no E∘D loop) | **ABS traj RMSE median 7e-4, p95 2.4e-2, max 3.25e-2**; outlet 0.14%; ∫S_E 12%; max\|Y\|=1.0 | **STABLE + ACCURATE** |

**#1** (joint, species-space pushforward): contraction made the rollout bounded → pushforward trained
without exploding → **first real Layer-2 movement** (∫S_E rollout error 14→0.34, 1D 330→87). But the
trajectory-matching loss competes with accuracy → a-priori fell 10.15×→5.29×. Tracking-vs-accuracy
tension a single loss can't escape.

**#2** (slow-manifold, anisotropic transverse-only contraction using the `z_dot_true` tangent): kept
a-priori high (8.10×) because a geometric penalty competes less than a trajectory loss — but the
1-D-tangent approximation didn't reshape the attractor onto the true path (the slow manifold is
multi-D), so 0D relRMSE stayed ~hundreds. Did not track.

**Conclusion — free-running latent transport is not crackable here** (7 principled attempts: spectral,
diffusion, Lagrangian, pushforward-ω_Z, contraction sweep, #1, #2). Layer 1 (boundedness) is solved
many ways; **Layer 2 (closed-loop tracking) is not**, by any training-side fix — you get a-priori OR
tracking, never both.

**#3 species-transport WORKS and is the deployment path.** Transporting the energy-active species and
reading the rate head at the resolved composition (`z = E(Y)`, **no decode→re-encode**, so the 6.6×
amplifier never exists) tracks truth to **≤3.25% mass-fraction absolute error** (median 0.07%),
bounded, with the **10.15× model as-is**. The apparent "divergent tail" (relRMSE p95 ~1e9) is purely
the near-zero-species relRMSE artifact — the absolute error proves Y stays bounded and accurate.

### Deployment decision
Ship `merged_contract090` (a-priori **10.15×**, R² 0.997) as a **species-transport local closure**: CFD
transports the 61 energy-active species; the rate head supplies dYᵢ/dτ + S_E per cell. Cost ~61 UDS vs
32 latent (loses the compression saving) but **stable by construction**. Remaining work: regenerate the
UDF/codegen for species transport (currently latent-UDS), and handle the ~151 non-energy-active species
(trace; prescribe/quasi-steady or transport with decoded values). The latent-compression idea would
need a true slow-manifold autoencoder built with the dynamics as a first-class objective (future work).

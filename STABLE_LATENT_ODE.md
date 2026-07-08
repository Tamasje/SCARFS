# Stable latent ODE — breaking the stability–fidelity wall (2026-07-08)

## The problem
The deployed latent-transport surrogate advanced the CFD latent by re-projecting every step:
`z ← P(z) + Δτ·ω_Z(P(z))` with `P = E∘D`. Closed-loop stability then *required* `P` to be
non-expansive (contraction penalty on the E∘D round-trip), which forces the **decoder flat** and caps
composition reconstruction at ~2.3e-2 — which in turn caps the deployed energy `∫S_E` at ~0.25.

We proved this is a genuine Pareto wall, not under-optimisation: **13 models across 6 levers** — energy-aware
pushforward (×2), closed-loop composition checkpoint selection, slow-manifold contraction, stronger/longer
pushforward, the k-sweep (k=8→16), the integration-substep sweep (1→200), and a decoder-capacity +
recon-priority pass — none moved the floor. The reconstruction-vs-contraction frontier is monotone:
*any* decoder faithful enough to represent composition below ~2.3e-2 is expansive (G_F > 1), and forcing
it contractive (G_F < 1, the CFD stability requirement) reverts reconstruction to the floor. Faithful and
stable were mutually exclusive.

## The fix: move stability onto the field, drop the re-projection
Advance the latent **directly** as a genuine ODE, with no E∘D each step:

    dz/dτ = f(z,q) = sinh(ω_Z(z,q))·s_Z − β·z          (MergedCoil.latent_field)

- The **arcsinh term** `sinh(ω_Z)·s_Z` carries the wide (~1e8) latent-velocity dynamic range.
- The **structural damping** `−β·z` (β = softplus(field_damping_raw) ≥ 0) contributes `−β·I` to `∂f/∂z`
  everywhere — a construction-guaranteed contraction floor — tightened by a soft dynamics-map
  contraction penalty on `z ↦ z + Δτ·f`.
- The **decoder is now free** (no contraction on E∘D): it only reads the latent out for the species
  field, never gates stability. Decoder fidelity and transport stability become independent knobs.

Deployment stays trivial plain-C and **8 UDS**: the UDF advances `z += Δτ·(sinh(ω_Z)·s_Z − β·z)` with
`mc_project` as identity (no re-projection); readout heads read the raw latent.

## Results — deployment-faithful (numpy C-primitives + envelope clamp, 40 test cases, substep=50)

| metric | stageBt2 (reproject) | **stableode_k8_bal (direct)** | factor |
|---|---|---|---|
| composition reconstruction | 2.3e-2 | **6.6e-4** | 35× |
| closed-loop composition drift | 2.49e-2 | **7.0e-3** | 3.5× |
| ∫S_E rollout | 0.328 | **0.100** | 3.3× |
| latent envelope clamp (max) | 0.93 | **7e-4** | barely clamps |
| point-wise energy a-priori (TEST) | 13.0× | 5.9× | **worse** |

The stable ODE also **converges with finer Δτ** (∫S_E 0.235→0.100 from substep 1→50), because it is a true
continuous ODE — whereas the reproject scheme plateaued (0.32→0.33). Real CFD uses fine CFL-limited steps,
so the deployed advantage should be at least this large. §5 energy acceptance: **8/9 gates pass** (same
profile as stageBt2; only the known near-zero-inlet integral-p95 artifact fails; integral median 1.9%).

## The one honest cost
Point-wise energy a-priori dropped 13.0×→5.9×. This is a real k=8 capacity split (spending the latent on
near-perfect composition leaves less for point-wise energy) and does **not** recover at k=8 (rebalancing
and warm-starting stageBt2's energy heads both tried). But it is a *point-wise ceiling proxy* that
deployment beats: stageBt2's 13× collapsed to ∫S_E 0.33 in closed loop because its composition drifted;
the stable ODE's 5.9× holds up to ∫S_E 0.10 because composition barely drifts. Watch-item: local hot-spot
energy accuracy. If the point-wise headline is required, higher k (now that stability is k-independent) is
the lever to recover it — at extra UDS cost.

## Deployment artifacts
`runs/stableode_k8_bal/` — model + `udf_export/` (direct-transport C-UDF: `#define MC_DIRECT_TRANSPORT`,
`MC_BETA`, `mc_project` identity, UDS source `sinh(ω_Z)·s_Z − β·z`). numpy↔torch consistency Y 2.9e-4 /
field 2.2e-5 / S_h 4.5e-7; standalone `merged_coil_forward_test.c` compiles and passes all 6 states at 1e-6.

Reproduce: `bash scripts/run_recon.sh` (→ recon_k8_A faithful decoder) → train `configs/train_stableode_k8_bal.json`.
Validate: `scripts/validate_stable_ode.py`, `scripts/aposteriori_rollout.py`, `scripts/diag_reconstruction.py`.

## The real arbiter
Everything here is the 0D/1D prescribed-thermo proxy. The Fluent run remains the final test — but the
stable ODE's fine-Δτ convergence + barely-clamping stability + near-perfect species transport make it the
better-founded deployment candidate. All new behaviour is gated behind `transport_mode="direct"`
(default `"reproject"`), so every prior run and all 496 tests are preserved.

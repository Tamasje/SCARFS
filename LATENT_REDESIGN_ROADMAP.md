# SCARFS latent-transport redesign roadmap (workflow output, 2026-06-29)

22-agent workflow (5 map / 8 ideate / 8 adversarial-verify / 1 synthesize). The synthesizer verified
claims against the repo and ran two measurements on the real 20k-row data that **reorder the priorities**.

## 0. Ground truth established by measurement (this reframes the problem)

- **The "10×" is a val artifact; honest TEST is 7.75× at k=32, 5.30× at k=16. Accuracy RISES with k.**
  Any plan that assumes k≤4 recovers 10× is betting against the only honest number on file.
- **Intrinsic dimensionality (measured on 20k rows):**
  - Linear-PCA composition variance: k=4→91.3%, k=6→95.6%, k=16→99.6%.
  - Slow-manifold tangent (unit dY/dt directions, reacting rows): **participation ratio = 3.80**;
    cumulative k=6→91%. **The dynamical manifold is genuinely low-D (~4–6).**
  - Energy-info ceiling (linear readout of log S_E from top-k PCA scores): k=4→R²0.13, k=6→0.28,
    k=16→0.52, k=32→0.60. **Energy info keeps climbing with k far past the manifold dimension.**
- **The reconciliation (the crux):** the manifold is ~4–6D, but a **LINEAR** encoder cannot
  parametrize a *curved* 4–6D manifold compactly, so it spends k=16–32 buying back the energy
  accuracy curvature destroys. **The k-vs-accuracy wall is a LINEARITY artifact.** The encoder has
  only ever been `nn.Linear` (neuralcoil.py:174) — **a nonlinear encoder is the one untried lever.**
- **Deployment reality (verified):** the C-UDF's `mc_net_eval` is already a generic Linear+LayerNorm
  +SiLU evaluator passing a 1e-4 consistency check → **a nonlinear MLP encoder is C-exportable with
  zero new primitives.** The semi-implicit `dS[eqn]` slot is unused (free stabilization if needed).

## 1. Recommended redesign — two untried levers that combine

**Direction A — nonlinear encoder.** Replace `nn.Linear(n_dry, k)` with a shallow MLP
(`[n_dry, 96, k]`, PCA-warm-started final layer). Lets a **k≈6** latent curve onto the ~3.8-D manifold
and recover the energy info linear PCA needs k=32 for. The only architectural change.

**Direction B — staged training with the encoder FROZEN (the Pareto-wall escape).** Attempt 6's
10.15→5.29× collapse came from *one joint loss* where the trajectory term dragged the shared
representation off the accuracy optimum. The fix is **order of operations**:
- **Stage A** — train a-priori to convergence (current `merged_composite`, no rollout/contraction).
- **Stage B** — `encoder.requires_grad_(False)`, then fine-tune **decoder + ω_Z only** with
  `contraction_weight≈0.9` (Layer 1, already drives G_F 6.6→0.78) + species-space `pushforward`
  (Layer 2, already built). Freezing the encoder fences off a-priori accuracy *structurally* while the
  dynamics are reshaped → tracking without the accuracy trade. **Both levers are cheap recombinations
  of code that already exists.**

**Why it breaks the wall:** accuracy (Stage A) and tracking (Stage B) are never in one competing loss
at the same time; contraction in the frozen stage costs ~0 a-priori (encoder fixed, heads converged);
pushforward reshapes only the decoder + ω_Z — the components that set the autonomous flow, not the
manifold geometry.

**Rejected (sound adversarial + measured reasons):** Koopman/Hurwitz (an affine generator can't
represent the ~40% of points moving outward in an open PFR), ICNN-Lyapunov (attractor ill-posed for a
non-equilibrium outlet), isometric tied-weight AE (the P=A⁺A retraction is false — the decoder takes
the thermo block q as input), fixed RPV coordinates.

## 2. Expected k and acceptance
Target **k=6**; sweep {4,6,8}. Accept the smallest k clearing **a-priori ≥7× AND 0D rollout
relRMSE→O(0.1)**. Do not promise k<5 — the honest evidence is against it. (6 UDS ≈ 1.2× base-flow cost
vs 6.4× at k=32.)

## 3. Fast de-risking experiment (ONE run, decisive, ~zero new code)
Isolate **Direction B alone** on the EXISTING k=32 `merged_contract090`:
1. Load it (already 10.15× val, G_F 0.78, contraction trained).
2. **Freeze the encoder**; fine-tune decoder + ω_Z with `pushforward_weight>0` (steps 4→16),
   ~50–100 epochs.
3. Gate with the existing harness: `aposteriori_rollout.py` (does 0D relRMSE fall from hundreds toward
   O(0.1)?) + `full_test_eval.py` (does a-priori stay ≥8×, vs attempt 6's collapse to 5.29×?).

**Decision rule:** 0D drops AND a-priori holds → the staged/frozen escape works → proceed to the
nonlinear-encoder k-sweep to also win low-k. Else → freezing is insufficient → **ship species-transport**
(proven, numpy closure exists). Decisive in one run, on an existing model, with a `requires_grad_(False)`
plus the already-wired pushforward flag.

## 4. Honest risks
1. k=6 nonlinear may still miss 10× — 7.75× at k=32 is the best *anywhere*; treat ≥7× at k≤8 as success.
2. Freezing the encoder may cap tracking (a-priori-optimal manifold ≠ dynamically-optimal); mitigate
   with a short low-LR Stage C on the encoder.
3. Pushforward on sparse trajectories (median 32 pts/case); denser re-export of a subset is possible.
4. Decoder may drift off the simplex (no atom conservation in latent transport); keep recon +
   atom_projection active in Stage B.
5. **It might not crack** (the repo's standing conclusion after 7 attempts). Both levers are genuinely
   new, but if the de-risk fails, **species-transport (61 UDS, stable, ships the 7.75× model) is the
   committed fallback.**

## Execution order
1. Freeze-encoder + pushforward de-risk on the existing k=32 model (decides if Layer 2 is crackable by staging).
2. If yes → nonlinear-encoder k={4,6,8} two-stage sweep (also delivers low-k).
3. Ship species-transport as the guaranteed fallback in parallel.

Key files: `neuralcoil.py:174` (encoder swap), `losses.py:318,734` (pushforward + contraction, both
built), `train.py:132,597` (build_model + pushforward wiring), `codegen.py:1575,1648,1390` (generic MLP
eval, encoder export, unused dS slot), `scripts/aposteriori_rollout.py` + `aposteriori_species.py` (gates).

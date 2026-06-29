"""Quantify the closed-loop instability: the Lipschitz GAIN of the deployed one-step latent map.

The rollout advances z <- F(z) = P(z) + Δτ·ω_Z(P(z)),  P = E∘D (manifold projection).
A map is contractive (stable) iff ‖F(z+δ)-F(z)‖/‖δ‖ < 1. We measure, at TRUE test latents, the
gain of:
  G_P  = ‖P(z+δ)-P(z)‖/‖δ‖           (the projection alone — the representation geometry)
  G_F  = ‖F(z+δ)-F(z)‖/‖δ‖           (the full one-step rollout map, representative Δτ)
for small random δ. G_F>1 ⇒ the deployed map AMPLIFIES perturbations every step ⇒ exponential drift
(the observed instability). Where the gain lives dictates the contractive fix:
  - G_P>1  ⇒ the encoder/decoder geometry is expansive → must retrain E/D to be contractive.
  - G_P<1, G_F>1 ⇒ the source term drives the expansion → a contractive design can damp/constrain ω_Z.

Run: .venv/bin/python scripts/diag_projection_gain.py [runs/merged_best]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.training.datamodule import tripartite_case_split
from aposteriori_rollout import Surrogate


def main() -> None:
    bundle = sys.argv[1] if len(sys.argv) > 1 else "runs/merged_best"
    db = "Database_FINAL.parquet"
    df = load_database(db); sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    sg = Surrogate(bundle)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    tau_col = sch.state["tau"]; cid_col = sch.meta["CaseID"]
    rng = np.random.default_rng(0)

    # representative Δτ (median consecutive step in the test set)
    dts = []
    for c in dft[cid_col].drop_duplicates().to_numpy()[:200]:
        t = np.sort(dft.loc[(dft[cid_col] == c).to_numpy(), tau_col].to_numpy())
        if len(t) > 1:
            dts.append(np.median(np.diff(t)))
    dtau = float(np.median(dts))

    # sample test states
    idx = rng.choice(len(dft), size=min(4000, len(dft)), replace=False)
    Y = dft[[f"Y_{s}" for s in sg.input]].to_numpy(float)[idx]
    T = dft[tc].to_numpy(float)[idx]; P = dft[pc].to_numpy(float)[idx]
    qn = sg.q(T, P)
    z = sg.encode(Y)                       # (n,k) true latents (clamped)
    span = np.maximum(sg.env_hi - sg.env_lo, 1e-12)

    def F(zz):
        zp = sg.project(zz, qn)
        return zp + dtau * sg.omega_z(zp, qn)

    Pz = sg.project(z, qn); Fz = F(z)
    print(f"=== one-step latent-map Lipschitz gain — {bundle} ===")
    print(f"representative Δτ = {dtau:.3e} s ; states n={len(idx)} ; k={z.shape[1]}\n")
    print(f"  {'‖δ‖/span':>10} {'G_P=‖ΔP‖/‖δ‖':>16} {'G_F=‖ΔF‖/‖δ‖':>16} {'frac G_F>1':>11}")
    for eps in (1e-3, 1e-2, 1e-1):
        d = rng.standard_normal(z.shape); d = d / np.linalg.norm(d, axis=1, keepdims=True) * (eps * np.linalg.norm(span))
        gP = np.linalg.norm(sg.project(z + d, qn) - Pz, axis=1) / np.linalg.norm(d, axis=1)
        gF = np.linalg.norm(F(z + d) - Fz, axis=1) / np.linalg.norm(d, axis=1)
        print(f"  {eps:>10.0e} {np.median(gP):>16.3f} {np.median(gF):>16.3f} {np.mean(gF>1.0):>11.2f}")
    print("\n  G_F>1 (median) ⇒ the deployed map amplifies every step ⇒ exponential drift (the instability).")
    print("  If G_P already >1, the encoder/decoder geometry is the culprit → contraction must be")
    print("  trained INTO E/D; if only G_F>1, the source ω_Z drives it → a contractive design can damp it.")


if __name__ == "__main__":
    main()

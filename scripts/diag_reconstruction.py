"""Reconstruction-floor diagnostic: how well does the autoencoder represent composition?

The a-posteriori investigation showed the closed-loop composition drift (which caps ∫S_E) EQUALS the
reconstruction floor decode(encode(Y_true)) — dynamics (ω_Z) add almost nothing. So the surrogate's
accuracy ceiling is set by how well the k-dim latent + decoder can REPRESENT composition, not by the
transport. This measures that floor on held-out TEST states, in two metrics:

  per-row     : median over states of sqrt(mean_over_majors (Ŷ−Y)²)         (worst-major-dominated)
  per-species : mean over majors of sqrt(mean_over_states (Ŷ−Y)²)           (rollout-aligned)

Run: .venv/bin/python scripts/diag_reconstruction.py runs/merged_nlenc_k8_stageBt2 [more bundles...]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.schema import MAJOR_SPECIES
from scarfs.training.datamodule import tripartite_case_split
from scripts.aposteriori_rollout import Surrogate


def reconstruction_floor(bundle: str, dft, sch, idx, T, P):
    sg = Surrogate(bundle)
    mi = [sg.input.index(s) for s in MAJOR_SPECIES if s in sg.input]
    Y = dft[[f"Y_{s}" for s in sg.input]].to_numpy(float)[idx]
    Yrec = np.zeros_like(Y)
    for i in range(len(idx)):
        q = sg.q(T[i], P[i])
        Yrec[i] = sg.decode_mass(sg.encode(Y[i]), q)[0]
    per_row = np.sqrt(np.mean((Yrec[:, mi] - Y[:, mi]) ** 2, axis=1))          # (n,)
    per_species = [np.sqrt(np.mean((Yrec[:, j] - Y[:, j]) ** 2)) for j in mi]  # (n_maj,)
    return float(np.median(per_row)), float(np.mean(per_species))


def main() -> None:
    bundles = sys.argv[1:] or ["runs/merged_nlenc_k8_stageBt2"]
    df = load_database("Database_FINAL.parquet"); sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=0.176, test_fraction=0.15,
                                         seed=0, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    rng = np.random.default_rng(0); idx = rng.choice(len(dft), min(4000, len(dft)), replace=False)
    T = dft[tc].to_numpy()[idx]; P = dft[pc].to_numpy()[idx]
    print("=== RECONSTRUCTION FLOOR — decode(encode(Y_true)), major species, held-out TEST ===")
    print(f"    {'bundle':40s} {'per-row(med)':>14s} {'per-species(mean)':>18s}")
    for b in bundles:
        try:
            pr, ps = reconstruction_floor(b, dft, sch, idx, T, P)
            print(f"    {b:40s} {pr:>14.4e} {ps:>18.4e}")
        except Exception as e:
            print(f"    {b:40s} ERROR: {e}")


if __name__ == "__main__":
    main()

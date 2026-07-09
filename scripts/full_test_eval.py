"""HONEST evaluator on the regenerated full-tier DB's held-out TEST split.

Production analogue of ``goal_test_eval.py`` (which is pinned to the retired pilot). Reconstructs the
SAME deterministic case split that training held out (``tripartite_case_split`` with the config's
val/test fractions + seed), then scores the rate-derived energy relRMSE — the deployed quantity
(S_E = Σ hᵢ·ω̇ᵢ via NASA7) — on cases never seen in training OR checkpoint selection.

Factor is reported against the pilot's honest 1.00x anchor (k16/100ep baseline test relRMSE 0.5390)
so the "Nx" number is continuous with GOAL_ARCHITECTURE_SEARCH.md; 10x target = relRMSE 0.0539.

Run: .venv/bin/python scripts/full_test_eval.py runs/merged_best [runs/merged_k16 ...]
     [--db Database_FINAL.parquet --val-frac 0.176 --test-frac 0.15 --seed 0]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.models.common import thermo_features
from scarfs.models.neuralcoil import MergedCoil
from scarfs.models.thermo import SpeciesThermo
from scarfs.training.datamodule import tripartite_case_split

MECH = "chem_ForTransport.yaml"
BASELINE_TEST = 0.5390  # pilot k16/100ep honest test relRMSE — the 1.00x reference (continuity)
TARGET_10X = BASELINE_TEST / 10.0


def _test_df(db: str, val_frac: float, test_frac: float, seed: int):
    df = load_database(db)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=val_frac, test_fraction=test_frac,
                                         seed=seed, split_by_case=True)
    return df.loc[te].reset_index(drop=True), sch


def test_relrmse(bundle: str, dft, sch) -> dict:
    b = Path(bundle)
    spec = json.loads((b / "spec.json").read_text())
    sc = pickle.load((b / "scalers.pkl").open("rb"))
    insp, ea = spec["input"], spec["energy_active"]
    cm = np.asarray(spec["composition_mean"], float); cs = np.asarray(spec["composition_scale"], float)
    rs = sc["rate_scaler"]; tsc = sc["legacy_scalers"].thermo
    (tc,) = sch.require_state("T"); (pc,) = sch.require_state("P")
    T = dft[tc].to_numpy(float); P = dft[pc].to_numpy(float)
    Y = dft[[f"Y_{s}" for s in insp]].to_numpy(float)
    ystd = (Y - cm) / cs
    q = (thermo_features(T, P) - tsc.mean_) / tsc.scale_
    cmod = spec["config_echo"]["model"]
    m = MergedCoil(n_dry=len(insp), n_energy_active=len(ea), latent_dim=int(cmod["latent_dim"]), n_thermo=4,
                   decoder_hidden=tuple(cmod["decoder_hidden"]), rate_hidden=tuple(cmod["rate_hidden"]),
                   latent_source_hidden=tuple(cmod["latent_source_hidden"]), energy_hidden=tuple(cmod["energy_hidden"]),
                   n_transport=int(cmod.get("transport_outputs", 0) or 0),
                   transport_hidden=tuple(cmod.get("transport_hidden", (64, 64))),
                   encoder_hidden=tuple(cmod.get("encoder_hidden", ()) or ()))
    m.load_state_dict(torch.load(b / "model.pt", map_location="cpu", weights_only=True)); m.eval()
    with torch.no_grad():
        out = m.forward(torch.tensor(ystd, dtype=torch.float32), torch.tensor(q, dtype=torch.float32))
    rmass = rs.inverse_transform(out["rates"].numpy())
    ta = SpeciesThermo.from_mechanism_yaml(MECH, list(ea))
    pred = (rmass * ta.h_mass(T)).sum(1)
    tgt = np.clip(dft[sch.energy_target_column()].to_numpy(float), 0.0, None)
    rel = float(np.sqrt(np.mean((pred - tgt) ** 2)) / np.sqrt(np.mean(tgt ** 2)))
    r2 = float(1.0 - np.sum((pred - tgt) ** 2) / np.sum((tgt - tgt.mean()) ** 2))
    return {"test_relRMSE": rel, "test_R2": r2, "factor": BASELINE_TEST / rel, "latent_dim": int(cmod["latent_dim"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*", default=["runs/merged_best"])
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--val-frac", type=float, default=0.176)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/full_test_eval.json")
    args = ap.parse_args()
    bundles = args.bundles or ["runs/merged_best"]
    dft, sch = _test_df(args.db, args.val_frac, args.test_frac, args.seed)
    print(f"=== HONEST TEST-split relRMSE ({args.db}, n_rows={len(dft)}; "
          f"baseline={BASELINE_TEST}, 10x target={TARGET_10X:.4f}) ===")
    rows = {}
    for bnd in bundles:
        try:
            r = test_relrmse(bnd, dft, sch)
            rows[bnd] = r
            hit = "  <-- >=10x!" if r["test_relRMSE"] <= TARGET_10X else ""
            print(f"  {Path(bnd).name:<22} k={r['latent_dim']:<3} relRMSE={r['test_relRMSE']:.4f}  "
                  f"R2={r['test_R2']:.4f}  factor={r['factor']:.2f}x{hit}")
        except Exception as e:  # noqa: BLE001
            print(f"  {bnd}: FAILED {e}")
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

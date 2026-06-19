"""HONEST evaluator: rate-derived energy relRMSE on the fully-held-out PILOT TEST split.

The goal_search ledger reports VAL relRMSE, which on the small pilot is optimistically biased
(easy val split + checkpoint selection). This evaluates any trained bundle on the pilot's 15% TEST
cases — never seen in training OR checkpoint selection — for the unbiased factor. The pilot test
cases hash identically whether or not a bundle was trained on augmented data, so test numbers are
comparable across experiments.

Run: .venv/bin/python scripts/goal_test_eval.py runs/goal_combo_eck runs/goal_combo ...
"""

from __future__ import annotations

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

PILOT = "pilot.parquet"
MECH = "chem_ForTransport.yaml"
SEED, VAL_FRAC, TEST_FRAC = 0, 0.176, 0.15
BASELINE_TEST = 0.5390  # baseline (k16, 100ep) test relRMSE — the honest 1.00x reference


def _test_df():
    df = load_database(PILOT)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=VAL_FRAC, test_fraction=TEST_FRAC,
                                         seed=SEED, split_by_case=True)
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
                   transport_hidden=tuple(cmod.get("transport_hidden", (64, 64))))
    m.load_state_dict(torch.load(b / "model.pt", map_location="cpu", weights_only=True)); m.eval()
    with torch.no_grad():
        out = m.forward(torch.tensor(ystd, dtype=torch.float32), torch.tensor(q, dtype=torch.float32))
    rmass = rs.inverse_transform(out["rates"].numpy())
    ta = SpeciesThermo.from_mechanism_yaml(MECH, list(ea))
    pred = (rmass * ta.h_mass(T)).sum(1)
    tgt = np.clip(dft[sch.energy_target_column()].to_numpy(float), 0.0, None)
    rel = float(np.sqrt(np.mean((pred - tgt) ** 2)) / np.sqrt(np.mean(tgt ** 2)))
    r2 = float(1.0 - np.sum((pred - tgt) ** 2) / np.sum((tgt - tgt.mean()) ** 2))
    return {"test_relRMSE": rel, "test_R2": r2, "factor": BASELINE_TEST / rel}


def main() -> None:
    bundles = sys.argv[1:] or ["runs/goal_baseline", "runs/goal_combo", "runs/goal_combo_eck"]
    dft, sch = _test_df()
    print(f"=== HONEST TEST-split relRMSE (pilot, n={len(dft)}; baseline={BASELINE_TEST}) ===")
    rows = {}
    for bnd in bundles:
        try:
            r = test_relrmse(bnd, dft, sch)
            rows[bnd] = r
            print(f"  {Path(bnd).name:<26} relRMSE={r['test_relRMSE']:.4f}  R2={r['test_R2']:.4f}  factor={r['factor']:.2f}x")
        except Exception as e:  # noqa: BLE001
            print(f"  {bnd}: FAILED {e}")
    Path("runs/goal_test_eval.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

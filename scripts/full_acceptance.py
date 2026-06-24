"""§5 energy acceptance gates + C-UDF forward-consistency on the held-out TEST split.

Runs the deployment-readiness suite on a trained merged bundle, on the regenerated DB's TEST cases
(the same deterministic split training held out). Two checks:

1. §5 energy acceptance (scarfs.benchmark.energy.evaluate_energy) on the RATE-DERIVED absorption
   S_E = Σ hᵢ·ω̇ᵢ — the DEPLOYED path (the C-UDF computes S_h from rates; the distilled head is a
   diagnostic). Note: the §5 global relRMSE is std-normalized (gate ≤0.23); the headline
   RMS-normalized relRMSE (the 0.539→10× story) comes from scripts/full_test_eval.py.
2. C-UDF export forward-consistency (numpy-vs-torch) via export_merged_udf — confirms the generated
   Fluent UDF reproduces the Python forward pass (Y, ω_Z, S_h max rel-diff).

Run: .venv/bin/python scripts/full_acceptance.py runs/merged_best [--db Database_FINAL.parquet]
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

from scarfs.benchmark.energy import evaluate_energy
from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.coupling.codegen import export_merged_udf
from scarfs.models.common import thermo_features
from scarfs.models.neuralcoil import MergedCoil
from scarfs.models.thermo import SpeciesThermo
from scarfs.training.datamodule import tripartite_case_split

MECH = "chem_ForTransport.yaml"


def _rate_derived_absorption(bundle: str, dft, sch):
    """Return (pred_absorption, target_absorption, case_ids, tau) on the test rows."""
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
    tau = dft[sch.state["tau"]].to_numpy(float)
    cases = dft[sch.meta["CaseID"]].to_numpy()
    return pred, tgt, cases, tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", nargs="?", default="runs/merged_best")
    ap.add_argument("--db", default="Database_FINAL.parquet")
    ap.add_argument("--val-frac", type=float, default=0.176)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-udf", action="store_true")
    args = ap.parse_args()

    df = load_database(args.db)
    sch = infer_schema(df)
    _tr, _va, te = tripartite_case_split(df, sch, val_fraction=args.val_frac,
                                         test_fraction=args.test_frac, seed=args.seed, split_by_case=True)
    dft = df.loc[te].reset_index(drop=True)
    n_cases = int(dft[sch.meta["CaseID"]].nunique())
    print(f"=== §5 ENERGY ACCEPTANCE (rate-derived / deployed path) — {args.bundle} ===")
    print(f"    DB={args.db}  TEST rows={len(dft)}  TEST cases={n_cases}\n")

    pred, tgt, cases, tau = _rate_derived_absorption(args.bundle, dft, sch)
    report = evaluate_energy(pred, tgt, cases, tau)
    print(report.summary())

    if not args.skip_udf:
        print("\n=== C-UDF forward-consistency (numpy-vs-torch) ===")
        out_dir = Path(args.bundle) / "udf_export"
        export = export_merged_udf(Path(args.bundle), out_dir)
        print(f"  artifacts: {sorted(p.name for p in export.artifacts.values())}")
        print(f"  max rel-diff:  Y={export.consistency_max_rel_diff_y:.2e}  "
              f"omega_Z={export.consistency_max_rel_diff_omega_z:.2e}  "
              f"S_h={export.consistency_max_rel_diff_sh:.2e}")

    out = Path(args.bundle) / "acceptance_s5.json"
    out.write_text(json.dumps({
        "bundle": args.bundle, "db": args.db, "test_rows": len(dft), "test_cases": n_cases,
        "global_r2": report.global_r2, "global_rel_rmse_std_norm": report.global_rel_rmse,
        "tail_median_rel_err_median": report.tail_median_rel_err_median,
        "tail_rel_rmse_pooled": report.tail_rel_rmse_pooled,
        "peak_tau_position_err_median": report.peak_tau_position_err_median,
        "cdf_max_dev_median": report.cdf_max_dev_median,
        "integral_rel_err_median": report.integral_rel_err_median,
        "integral_rel_err_p95": report.integral_rel_err_p95,
        "sign_negative_fraction": report.sign_negative_fraction,
        "passed": report.passed, "pass_fail": report.pass_fail,
    }, indent=2))
    print(f"\nwrote {out}\nOVERALL §5: {'PASS' if report.passed else 'FAIL'}")


if __name__ == "__main__":
    main()

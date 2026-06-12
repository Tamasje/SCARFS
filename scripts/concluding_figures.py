"""Concluding evaluation + figure set for the GPU-trained merged model.

Selects the best bundle from a k-ablation (or takes --bundle), scores it on its held-out
TEST cases with the §5 energy acceptance suite (rate-derived path primary, distilled head
reported), and writes the concluding figures: energy parity, per-case trajectories (incl.
the colleague's historical worst cases when present in the split), tail errors, error vs
temperature, QoI rate parity, reconstruction parity, latent-source parity (target space),
training curves, and accuracy-vs-k.

Usage:
  .venv/bin/python scripts/concluding_figures.py [--ablation-dir runs/gpu_ablation]
      [--bundle PATH] [--database PATH] [--out sanity_figures/concluding]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scarfs.benchmark.energy import evaluate_energy
from scarfs.benchmark.loader import infer_schema
from scarfs.models.adapter import TorchSurrogate
from scarfs.models.common import thermo_features
from scarfs.models.features import build_absorption_target, build_mass_rate_matrix
from scarfs.models.thermo import SpeciesThermo
from scarfs.plotting import figures
from scarfs.plotting.plot_defaults import apply_defaults, palette

DEFAULT_DB = "/Users/tamasbuzogany/Documents/SCARFS/TEST_ETHANE_LOW_sobol_stride5.parquet"
HISTORICAL_WORST_CASES = (4818, 2967)  # the colleague's documented sign-flip cases


def load_frame(db: str) -> tuple[pd.DataFrame, object]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(db)
    needed = [c for c in pf.schema_arrow.names
              if c.startswith(("Y_", "dYdt_")) or c == "CaseID"
              or c.split("[")[0].strip() in ("T", "P", "rho", "tau", "z", "u")
              or "absorption" in c.lower() or c.startswith("R_")]
    df = pd.read_parquet(db, columns=needed)
    return df, infer_schema(df)


def pick_winner(ablation_dir: Path) -> tuple[Path, dict]:
    summary = json.loads((ablation_dir / "summary.json").read_text(encoding="utf-8"))
    best_k = max(summary, key=lambda k: summary[k]["absorption_val"]["rate_derived"]["r2"])
    return Path(summary[best_k]["bundle"]), summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default="runs/gpu_ablation")
    ap.add_argument("--bundle", default=None)
    ap.add_argument("--database", default=DEFAULT_DB)
    ap.add_argument("--out", default="sanity_figures/concluding")
    args = ap.parse_args()

    apply_defaults()
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    ablation_dir = REPO / args.ablation_dir
    summary: dict | None = None
    if args.bundle:
        bundle = Path(args.bundle)
    else:
        bundle, summary = pick_winner(ablation_dir)
    print(f"winner bundle: {bundle}")

    df, schema = load_frame(args.database)
    case_col = schema.meta["CaseID"]
    spec = json.loads((Path(bundle) / "spec.json").read_text(encoding="utf-8"))
    test_cases = set(spec["split_case_ids"]["test"])
    dft = df[df[case_col].isin(test_cases)].reset_index(drop=True)
    print(f"test split: {len(dft)} rows / {dft[case_col].nunique()} cases")

    surr = TorchSurrogate.from_merged_bundle(bundle, schema)
    pred = surr.predict(dft)
    T = dft[schema.state["T"]].to_numpy(float)
    tau = dft[schema.state["tau"]].to_numpy(float)
    cases = dft[case_col].to_numpy()
    target = build_absorption_target(dft, schema).values
    thermo = SpeciesThermo.from_mechanism_yaml(REPO / "chem_ForTransport.yaml", surr.active_species)
    abs_rate = (thermo.h_mass(T) * pred.rates).sum(axis=1)

    print("\n=== §5 ENERGY ACCEPTANCE — TEST split, rate-derived path (primary) ===")
    rep = evaluate_energy(abs_rate, target, cases, tau)
    print(rep.summary())
    (out / "energy_acceptance_test.txt").write_text(rep.summary(), encoding="utf-8")

    print("=== distilled head path (UDF fast path) ===")
    rep_h = evaluate_energy(pred.energy, target, cases, tau)
    print(f"global R2={rep_h.global_r2:.4f} relRMSE={rep_h.global_rel_rmse:.4f} "
          f"tail_median={rep_h.tail_median_rel_err_median:.4f}")

    # --- figures -------------------------------------------------------------------------
    figures.energy_parity_figure(abs_rate, target, str(out / "energy_parity_test.png"))
    rel = np.abs(abs_rate - target) / np.maximum(np.abs(target), 1.0)
    figures.tail_rel_err_hist_figure(rel, str(out / "energy_tail_rel_err_hist_test.png"))
    figures.error_vs_temperature(T, rel, path=str(out / "energy_error_vs_T_test.png"))

    dft = dft.assign(_abs=target)
    peaks = dft.groupby(case_col)["_abs"].max().sort_values(ascending=False)
    traj_cases = list(peaks.index[:3]) + [c for c in HISTORICAL_WORST_CASES
                                          if c in set(cases) and c not in set(peaks.index[:3])]
    for cid in traj_cases:
        m = cases == cid
        order = np.argsort(tau[m])
        figures.front_localization_figure(
            tau[m][order], target[m][order], abs_rate[m][order], cid,
            str(out / f"trajectory_case_{cid}_energy.png"))

    qoi = [s for s in ("C2H4", "C2H6", "C3H6", "CH4", "H2", "C2H2", "BENZENE")
           if s in surr.active_species]
    truth = build_mass_rate_matrix(dft, schema, surr.active_species, prefer_dydt=True)
    idx = [list(surr.active_species).index(s) for s in qoi]
    figures.parity_plot(truth[:, idx], pred.rates[:, idx], qoi, log_scale=True,
                        path=str(out / "parity_qoi_rates_test.png"))
    print("\nQoI mass-rate R2 (test):")
    for s, i in zip(qoi, idx):
        sst = np.sum((truth[:, i] - truth[:, i].mean()) ** 2)
        r2 = 1 - np.sum((pred.rates[:, i] - truth[:, i]) ** 2) / sst if sst > 0 else float("nan")
        print(f"  {s:10s} {r2:8.4f}")

    # reconstruction + latent-source parity (model internals, target space)
    model, scalers = surr.model, surr.scalers
    y_all = dft[schema.y_columns(surr.spec.input_species)].to_numpy(float)
    x_std = scalers.composition.transform(y_all)
    q_std = scalers.thermo.transform(thermo_features(T, dft[schema.state["P"]].to_numpy(float)))
    with torch.no_grad():
        ct = torch.as_tensor(x_std, dtype=torch.float32)
        qt = torch.as_tensor(q_std, dtype=torch.float32)
        z = model.encode(ct)
        y_rec_std = model.decode(z, qt)
        z_proj = model.encoder(y_rec_std)
        zdot_pred_scaled = model.latent_source(z_proj, qt).numpy()
    y_rec = scalers.composition.inverse_transform(y_rec_std.numpy())
    his_majors = [s for s in ("C2H6", "C2H4", "CH4", "C2H2", "BENZENE", "H2", "STYRENE", "__1.3C4H6")
                  if s in surr.spec.input_species]
    ridx = [list(surr.spec.input_species).index(s) for s in his_majors]
    figures.parity_plot(y_all[:, ridx], y_rec[:, ridx], his_majors, log_scale=False,
                        path=str(out / "parity_species_reconstruction_test.png"))

    sigma = np.asarray(scalers.composition.scale_, dtype=float)
    dydt_dry = dft[schema.dydt_columns(list(surr.spec.input_species))].to_numpy(float)
    enc_w = model.encoder.weight.detach().numpy()
    zdot_true = (dydt_dry / sigma) @ enc_w.T
    s_z = np.asarray(spec["arcsinh_latent_scale"], dtype=float)
    t_true = np.arcsinh(zdot_true / s_z)
    k = t_true.shape[1]
    names_z = [f"Z{i+1}" for i in range(k)]
    figures.parity_plot(t_true, zdot_pred_scaled, names_z, log_scale=False,
                        path=str(out / "parity_latent_sources_test.png"))
    lat_r2 = [1 - np.sum((zdot_pred_scaled[:, i] - t_true[:, i]) ** 2)
              / max(np.sum((t_true[:, i] - t_true[:, i].mean()) ** 2), 1e-12) for i in range(k)]
    print("\nlatent-source target-space R2 per dim (test):",
          [round(float(r), 3) for r in lat_r2], "| mean:", round(float(np.mean(lat_r2)), 4))

    # training curves + accuracy-vs-k
    hist = json.loads((Path(bundle) / "metrics.json").read_text(encoding="utf-8"))["history"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, color in (("train", palette()[0]), ("val", palette()[3])):
        ax.semilogy([h["epoch"] for h in hist], [h[key] for h in hist], color=color, label=key)
    ax.semilogy([h["epoch"] for h in hist],
                [h["val_parts"].get("latent_source", np.nan) for h in hist],
                color=palette()[6], label="val latent_source")
    ax.set_xlabel("epoch"); ax.set_ylabel("composite loss"); ax.legend()
    ax.set_title(f"Training history — {Path(bundle).name}")
    fig.savefig(out / "training_history.png"); plt.close(fig)

    if summary is not None:
        ks = sorted(int(x) for x in summary)
        curves = {
            "absorption R² (rate-derived)": [summary[str(x)]["absorption_val"]["rate_derived"]["r2"] for x in ks],
            "absorption R² (distilled head)": [summary[str(x)]["absorption_val"]["head"]["r2"] for x in ks],
        }
        figures.accuracy_vs_k_figure(ks, curves, str(out / "accuracy_vs_k.png"),
                                     metric_name="val absorption R²")

    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()

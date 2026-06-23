"""A-priori A/B of the physics-augmentation changes on the pilot DB (directional, NON-certifying).

Trains capped MergedCoil models on pilot.parquet with IDENTICAL case splits and seed and evaluates
each on the held-out val cases, reporting per-term metrics on the axis appropriate for each change:
  - energy   : rate-derived vs distilled-head absorption R²            (proposal #1)
  - rate     : pooled R² in the head's native (scaled) space           (regression guard)
  - omega_z  : latent-source R² (pooled AND per-dim median)            (k=16 lever)
  - conserve : atom-balance residual of the PREDICTED rates (rel)      (proposal #4)
  - realize  : predicted realizability-violation fraction              (gap-5)
  - transport: μ/k R²                                                  (proposal #5)
  - keq      : net dehydrogenation extent ξ̇ vs the data, binned by |lnQ-lnKeq|  (proposal #3)

Modes:
  --mode ab    : train baseline (physics OFF) and all_on (physics ON), eval both, compare  [default]
  --mode eval  : evaluate the EXISTING runs/ab_baseline + runs/ab_all_on bundles (no training)
  --mode long  : train ONE long all_on run (push past the ω_Z floor; see if energy recovers)

Pilot-scale, front-under-resolved → DIRECTION, not certified accuracy (see README_SCARFS_ML.md).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on path when run as a script

import numpy as np
import torch

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.models.common import thermo_features
from scarfs.models.neuralcoil import MergedCoil
from scarfs.models.thermo import R_J_PER_KMOL_K, SpeciesThermo
from scarfs.training.config import DataConfig, LossConfig, ModelConfig, OptimConfig, TrainConfig
from scarfs.training.datamodule import tripartite_case_split
from scarfs.training.train import train

DB = "pilot.parquet"
MECH = "chem_ForTransport.yaml"
SEED = 0
BATCH = 4096
VAL_FRAC = 0.176
TEST_FRAC = 0.15
REALIZ_DT = 1.0e-3
MAJOR = ("C2H4", "C3H6", "CH4", "H2", "C2H6", "C2H2", "BENZENE")
KEQ_SP = ["C2H6", "C2H4", "H2"]
KEQ_STOICH = np.array([-1.0, 1.0, 1.0])  # C2H6 -> C2H4 + H2


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return float(1.0 - np.sum((pred - true) ** 2) / ss_tot) if ss_tot > 0 else float("nan")


def make_cfg(out_dir: str, physics_on: bool, epochs: int, patience: int, head_ft: int) -> TrainConfig:
    loss = LossConfig(
        rate_weight=1.0, latent_source_weight=1.0, energy_weight=0.5,
        energy_distill_weight=0.25, energy_target_weight=0.25, consistency_weight=0.02,
        recon_weight=1.0, manifold_weight=0.1, noise_std=0.05, rollout_mode="manifold",
        atom_projection_weight=(5e-3 if physics_on else 0.0),
        keq_weight=(1e-2 if physics_on else 0.0), keq_width=1.0,
        realizability_weight=(1e-2 if physics_on else 0.0), realizability_dt=REALIZ_DT,
        transport_weight=(0.05 if physics_on else 0.0),
    )
    return TrainConfig(
        data=DataConfig(
            database_path=DB, input_species="dry_all", target_species="energy_active",
            val_fraction=VAL_FRAC, test_fraction=TEST_FRAC, split_by_case=True,
            inlet_conversion_threshold=0.05, inlet_weight=5.0,
            species_weights={"C2H4": 150, "C3H6": 100, "CH4": 20, "H2": 10},
            seed=SEED, energy_active_coverage=0.999, mech_yaml=MECH,
            tail_strata=10, tail_weight_alpha=2.0, columns_projection=True,
            composition_sigma_floor=1e-10,
        ),
        model=ModelConfig(
            kind="merged", latent_dim=16, decoder_hidden=(128, 256), rate_hidden=(128, 128),
            latent_source_hidden=(128, 128), energy_hidden=(64, 64), activation="silu",
            spectral_norm=False,
            transport_outputs=(2 if physics_on else 0), transport_hidden=(64, 64),
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=1e-6, epochs=epochs, batch_size=BATCH,
                          grad_clip=1.0, patience=patience, head_finetune_epochs=head_ft),
        loss=loss, output_dir=out_dir,
    )


def _keq_diag(dfv, schema, input_species, energy_active, rates_pred_mass, true_mass, T, P) -> dict:
    """Net dehydrogenation extent ξ̇=(ω_C2H4+ω_H2−ω_C2H6)/3 vs the data, binned by |lnQ−lnKeq|.

    A working Keq pressure makes the PREDICTED |ξ̇| track the TRUE |ξ̇| collapse near equilibrium
    (small |lnQ−lnKeq|). Normalised by the val-median |ξ̇_true| so numbers are O(1).
    """
    if not (all(s in input_species for s in KEQ_SP) and all(s in energy_active for s in KEQ_SP)):
        return {}
    tk = SpeciesThermo.from_mechanism_yaml(MECH, KEQ_SP)
    Wk = tk.molar_mass
    in_idx = [list(input_species).index(s) for s in KEQ_SP]
    act_idx = [list(energy_active).index(s) for s in KEQ_SP]
    Y3 = dfv[[f"Y_{s}" for s in KEQ_SP]].to_numpy(float)

    td = SpeciesThermo.from_mechanism_yaml(MECH, list(input_species), missing_ok=True)
    Wmap = dict(zip(td.species, td.molar_mass)); Wmap.setdefault("H2O", 18.01528)
    ntot = np.zeros(len(dfv))
    for s, w in Wmap.items():
        col = f"Y_{s}"
        if col in dfv.columns and w > 0:
            ntot += dfv[col].to_numpy(float) / w
    ntot = np.maximum(ntot, 1e-30)

    X = np.clip((Y3 / Wk) / ntot[:, None], 1e-12, None)
    lnQ = np.log(X[:, 1]) + np.log(X[:, 2]) - np.log(X[:, 0]) + np.log(np.clip(P / 101325.0, 1e-30, None))
    g3 = tk.g_molar(T)
    lnKeq = -(g3[:, 1] + g3[:, 2] - g3[:, 0]) / (R_J_PER_KMOL_K * np.maximum(T, 1.0))
    dlt = np.abs(lnQ - lnKeq)

    xi_pred = (rates_pred_mass[:, act_idx] / Wk * KEQ_STOICH).sum(1) / 3.0
    xi_true = (true_mass[:, act_idx] / Wk * KEQ_STOICH).sum(1) / 3.0
    scale = float(np.median(np.abs(xi_true))) + 1e-300

    edges = [0.0, 1.0, 2.0, 4.0, np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dlt >= lo) & (dlt < hi)
        band = (f"|d|<{hi:g}" if lo == 0 else (f"|d|>={lo:g}" if np.isinf(hi) else f"{lo:g}<=|d|<{hi:g}"))
        bins.append({
            "band": band, "n": int(m.sum()),
            "pred_xi_norm": float(np.mean(np.abs(xi_pred[m])) / scale) if m.any() else float("nan"),
            "true_xi_norm": float(np.mean(np.abs(xi_true[m])) / scale) if m.any() else float("nan"),
        })
    near = dlt < 1.0
    return {
        "scale_kmol_m3_s": scale, "bins": bins,
        "near_eq_pred_norm": float(np.mean(np.abs(xi_pred[near])) / scale) if near.any() else float("nan"),
        "near_eq_true_norm": float(np.mean(np.abs(xi_true[near])) / scale) if near.any() else float("nan"),
        "near_eq_n": int(near.sum()),
    }


def evaluate(bundle_dir: str) -> dict:
    bundle = Path(bundle_dir)
    spec = json.loads((bundle / "spec.json").read_text())
    with (bundle / "scalers.pkl").open("rb") as fh:
        scalers = pickle.load(fh)

    input_species = spec["input"]
    energy_active = spec["energy_active"]
    comp_mean = np.asarray(spec["composition_mean"], float)
    comp_scale = np.asarray(spec["composition_scale"], float)
    s_z = np.asarray(spec["arcsinh_latent_scale"], float)
    rate_scaler = scalers["rate_scaler"]
    thermo_sc = scalers["legacy_scalers"].thermo

    df = load_database(DB)
    schema = infer_schema(df)
    _tr, va, _te = tripartite_case_split(df, schema, val_fraction=VAL_FRAC,
                                         test_fraction=TEST_FRAC, seed=SEED, split_by_case=True)
    dfv = df.loc[va].reset_index(drop=True)
    (t_col,) = schema.require_state("T")
    (p_col,) = schema.require_state("P")
    rho_col = schema.require_state("rho")[0]

    Y_in = dfv[[f"Y_{s}" for s in input_species]].to_numpy(float)
    y_std = (Y_in - comp_mean) / comp_scale
    T = dfv[t_col].to_numpy(float)
    P = dfv[p_col].to_numpy(float)
    rho = dfv[rho_col].to_numpy(float)
    q = (thermo_features(T, P) - thermo_sc.mean_) / thermo_sc.scale_

    cm = spec["config_echo"]["model"]
    n_tp = int(cm.get("transport_outputs", 0) or 0)
    model = MergedCoil(
        n_dry=len(input_species), n_energy_active=len(energy_active), latent_dim=int(cm["latent_dim"]),
        n_thermo=4, decoder_hidden=tuple(cm["decoder_hidden"]), rate_hidden=tuple(cm["rate_hidden"]),
        latent_source_hidden=tuple(cm["latent_source_hidden"]), energy_hidden=tuple(cm["energy_hidden"]),
        activation="silu", n_transport=n_tp, transport_hidden=tuple(cm.get("transport_hidden", (64, 64))),
    )
    model.load_state_dict(torch.load(bundle / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        out = model.forward(torch.as_tensor(y_std, dtype=torch.float32), torch.as_tensor(q, dtype=torch.float32))
        rates_scaled = out["rates"].cpu().numpy()
        ls_head = out["latent_source"].cpu().numpy()
        tp_pred = model.transport(out["z_proj"], torch.as_tensor(q, dtype=torch.float32)).cpu().numpy() if n_tp > 0 else None

    res: dict = {}
    m = json.loads((bundle / "metrics.json").read_text()).get("absorption_metrics_val", {})
    res["energy_rate_derived_R2"] = m.get("rate_derived", {}).get("r2", float("nan"))
    res["energy_rate_derived_relRMSE"] = m.get("rate_derived", {}).get("rel_rmse", float("nan"))
    res["energy_tail_relRMSE"] = m.get("rate_derived", {}).get("tail_rel_rmse", float("nan"))
    res["energy_tail_median_rel_err"] = m.get("rate_derived", {}).get("tail_median_rel_err", float("nan"))
    res["energy_head_R2"] = m.get("head", {}).get("r2", float("nan"))

    true_mass = rho[:, None] * dfv[schema.dydt_columns(list(energy_active))].to_numpy(float)
    tgt_scaled = rate_scaler.transform(true_mass)
    res["rate_R2_scaled_pooled"] = _r2(rates_scaled, tgt_scaled)
    maj = [s for s in MAJOR if s in energy_active]
    if maj:
        idx = [list(energy_active).index(s) for s in maj]
        res["rate_R2_major_scaled"] = _r2(rates_scaled[:, idx], tgt_scaled[:, idx])

    omega_pred = np.sinh(np.clip(ls_head, -20.0, 20.0)) * s_z
    enc_W = model.encoder.weight.detach().cpu().numpy()
    dydt_dry = dfv[schema.dydt_columns(list(input_species))].to_numpy(float)
    z_dot_true = (dydt_dry / comp_scale[None, :]) @ enc_W.T
    res["omega_z_R2_pooled"] = _r2(omega_pred, z_dot_true)
    res["omega_z_R2_perdim_median"] = float(np.nanmedian(
        [_r2(omega_pred[:, d], z_dot_true[:, d]) for d in range(omega_pred.shape[1])]))

    thermo_a = SpeciesThermo.from_mechanism_yaml(MECH, list(energy_active))
    rates_pred_mass = rate_scaler.inverse_transform(rates_scaled)
    r_molar = rates_pred_mass / thermo_a.molar_mass[None, :]
    elem = r_molar @ thermo_a.element_matrix
    res["atom_residual_rel"] = float(
        np.mean(np.linalg.norm(elem, axis=1)) / (np.mean(np.linalg.norm(r_molar, axis=1)) + 1e-300))

    Y_active = Y_in[:, [list(input_species).index(s) for s in energy_active]]
    consumed_frac = np.clip(-rates_pred_mass, 0.0, None) * REALIZ_DT / (rho[:, None] + 1e-6)
    res["realizability_violation_frac"] = float(np.mean(consumed_frac > Y_active))

    if tp_pred is not None and "mu [Pa-s]" in dfv.columns and "k [W/m/K]" in dfv.columns:
        res["mu_R2"] = _r2(tp_pred[:, 0], dfv["mu [Pa-s]"].to_numpy(float))
        res["k_R2"] = _r2(tp_pred[:, 1], dfv["k [W/m/K]"].to_numpy(float))

    res["keq_diag"] = _keq_diag(dfv, schema, input_species, energy_active,
                                rates_pred_mass, true_mass, T, P)
    res["n_val_rows"] = int(len(dfv))
    return res


def _print_keq(label: str, diag: dict) -> None:
    if not diag:
        print(f"  [{label}] keq diag unavailable")
        return
    print(f"  [{label}] net-dehydrogenation extent |ξ̇| / median|ξ̇_true|, binned by |lnQ-lnKeq|:")
    print(f"    {'band':<14}{'n':>7}{'pred':>10}{'true':>10}")
    for b in diag["bins"]:
        print(f"    {b['band']:<14}{b['n']:>7}{b['pred_xi_norm']:>10.3g}{b['true_xi_norm']:>10.3g}")
    print(f"    near-eq (|Δ|<1): pred={diag['near_eq_pred_norm']:.3g} true={diag['near_eq_true_norm']:.3g} "
          f"(n={diag['near_eq_n']})")


def _print_summary(out: dict) -> None:
    keys = ["energy_rate_derived_R2", "energy_head_R2", "rate_R2_scaled_pooled", "rate_R2_major_scaled",
            "omega_z_R2_pooled", "omega_z_R2_perdim_median", "atom_residual_rel",
            "realizability_violation_frac", "mu_R2", "k_R2"]
    labels = list(out.keys())
    print(f"\n{'='*86}\nA/B SUMMARY (pilot, directional/non-certifying)  n_val={out[labels[0]]['n_val_rows']}\n{'='*86}")
    hdr = f"{'metric':<32}" + "".join(f"{l:>14}" for l in labels)
    if len(labels) == 2:
        hdr += f"{'delta':>14}"
    print(hdr)
    for k in keys:
        vals = [out[l].get(k, float('nan')) for l in labels]
        row = f"{k:<32}" + "".join(f"{v:>14.4g}" for v in vals)
        if len(labels) == 2 and all(np.isfinite(v) for v in vals):
            row += f"{vals[1]-vals[0]:>14.4g}"
        print(row)
    print(f"\n{'-'*86}\nKeq equilibrium-collapse diagnostic (#3):")
    for l in labels:
        _print_keq(l, out[l].get("keq_diag", {}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ab", "eval", "long"], default="ab")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--head-ft", type=int, default=None)
    args = ap.parse_args()

    if args.mode == "eval":
        out = {"baseline": evaluate("runs/ab_baseline"), "all_on": evaluate("runs/ab_all_on")}
        _print_summary(out)
        Path("runs/ab_pilot_eval.json").write_text(json.dumps(out, indent=2))
        print("\nwrote runs/ab_pilot_eval.json")
        return

    if args.mode == "long":
        ep = args.epochs or 100
        pat = args.patience or 25
        hft = args.head_ft if args.head_ft is not None else 40
        print(f"\n{'='*70}\nLONG all_on run: epochs={ep} patience={pat} head_ft={hft}\n{'='*70}", flush=True)
        cfg = make_cfg("runs/ab_all_on_long", True, ep, pat, hft)
        train(cfg)
        out = {"all_on_long": evaluate(cfg.output_dir)}
        _print_summary(out)
        Path("runs/ab_pilot_long.json").write_text(json.dumps(out, indent=2))
        print("\nwrote runs/ab_pilot_long.json")
        return

    ep = args.epochs or 20
    pat = args.patience or 8
    hft = args.head_ft if args.head_ft is not None else 8
    out = {}
    for label, physics_on in (("baseline", False), ("all_on", True)):
        print(f"\n{'='*70}\nTRAIN {label} (physics_on={physics_on}) ep={ep}\n{'='*70}", flush=True)
        cfg = make_cfg(f"runs/ab_{label}", physics_on, ep, pat, hft)
        train(cfg)
        out[label] = evaluate(cfg.output_dir)
    _print_summary(out)
    Path("runs/ab_pilot_summary.json").write_text(json.dumps(out, indent=2))
    print("\nwrote runs/ab_pilot_summary.json")


if __name__ == "__main__":
    main()

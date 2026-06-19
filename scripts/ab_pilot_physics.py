"""A-priori A/B of the physics-augmentation changes on the pilot DB (directional, NON-certifying).

Trains two capped MergedCoil models on pilot.parquet with IDENTICAL case splits and seed:
  - baseline : k=16, physics terms OFF (atom_projection / keq / realizability / transport all 0)
  - all_on   : k=16, the new physics terms ON (the production config weights)

Then evaluates BOTH on the same held-out val cases and reports per-term metrics on the axis that
is appropriate for each change (energy: rate-derived vs distilled head R²; conservation: atom-balance
residual of the PREDICTED rates; realizability: predicted-violation fraction; transport: μ/k R²;
ω_Z: latent-source R²; rate: scaled-space pooled R²).

This is a pilot-scale, short-training, front-under-resolved comparison — it shows DIRECTION, not
certified accuracy (see README_SCARFS_ML.md). Run: .venv/bin/python scripts/ab_pilot_physics.py
"""

from __future__ import annotations

import copy
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
from scarfs.models.physics import atom_conservation_projector
from scarfs.models.thermo import SpeciesThermo
from scarfs.training.config import DataConfig, LossConfig, ModelConfig, OptimConfig, TrainConfig
from scarfs.training.datamodule import tripartite_case_split
from scarfs.training.train import train

DB = "pilot.parquet"
SEED = 0
EPOCHS = 20
PATIENCE = 8
HEAD_FT = 8
BATCH = 4096
VAL_FRAC = 0.176
TEST_FRAC = 0.15
REALIZ_DT = 1.0e-3
MAJOR = ("C2H4", "C3H6", "CH4", "H2", "C2H6", "C2H2", "BENZENE")


def _r2(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, float).ravel()
    true = np.asarray(true, float).ravel()
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - np.sum((pred - true) ** 2) / ss_tot)


def make_cfg(out_dir: str, physics_on: bool) -> TrainConfig:
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
            seed=SEED, energy_active_coverage=0.999, mech_yaml="chem_ForTransport.yaml",
            tail_strata=10, tail_weight_alpha=2.0, columns_projection=True,
            composition_sigma_floor=1e-10,
        ),
        model=ModelConfig(
            kind="merged", latent_dim=16, decoder_hidden=(128, 256), rate_hidden=(128, 128),
            latent_source_hidden=(128, 128), energy_hidden=(64, 64), activation="silu",
            spectral_norm=False,
            transport_outputs=(2 if physics_on else 0), transport_hidden=(64, 64),
        ),
        optim=OptimConfig(lr=1e-3, weight_decay=1e-6, epochs=EPOCHS, batch_size=BATCH,
                          grad_clip=1.0, patience=PATIENCE, head_finetune_epochs=HEAD_FT),
        loss=loss, output_dir=out_dir,
    )


def evaluate(bundle_dir: str) -> dict:
    """Load a trained bundle and compute per-term metrics on the held-out val cases."""
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

    cfg_echo = spec["config_echo"]["model"]
    n_tp = int(cfg_echo.get("transport_outputs", 0) or 0)
    model = MergedCoil(
        n_dry=len(input_species), n_energy_active=len(energy_active),
        latent_dim=int(cfg_echo["latent_dim"]), n_thermo=4,
        decoder_hidden=tuple(cfg_echo["decoder_hidden"]), rate_hidden=tuple(cfg_echo["rate_hidden"]),
        latent_source_hidden=tuple(cfg_echo["latent_source_hidden"]),
        energy_hidden=tuple(cfg_echo["energy_hidden"]), activation="silu",
        n_transport=n_tp, transport_hidden=tuple(cfg_echo.get("transport_hidden", (64, 64))),
    )
    model.load_state_dict(torch.load(bundle / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    yt = torch.as_tensor(y_std, dtype=torch.float32)
    qt = torch.as_tensor(q, dtype=torch.float32)
    with torch.no_grad():
        out = model.forward(yt, qt)
        rates_scaled = out["rates"].cpu().numpy()
        ls_head = out["latent_source"].cpu().numpy()
        z_proj = out["z_proj"]
        tp_pred = model.transport(z_proj, qt).cpu().numpy() if n_tp > 0 else None

    res: dict = {}

    # --- energy: rate-derived vs distilled head (read straight from metrics.json) ---
    m = json.loads((bundle / "metrics.json").read_text()).get("absorption_metrics_val", {})
    res["energy_rate_derived_R2"] = m.get("rate_derived", {}).get("r2", float("nan"))
    res["energy_head_R2"] = m.get("head", {}).get("r2", float("nan"))

    # --- rate head: pooled R² in the head's native (scaled) space over active species ---
    true_mass = rho[:, None] * dfv[schema.dydt_columns(list(energy_active))].to_numpy(float)
    tgt_scaled = rate_scaler.transform(true_mass)
    res["rate_R2_scaled_pooled"] = _r2(rates_scaled, tgt_scaled)
    maj = [s for s in MAJOR if s in energy_active]
    if maj:
        idx = [list(energy_active).index(s) for s in maj]
        res["rate_R2_major_scaled"] = _r2(rates_scaled[:, idx], tgt_scaled[:, idx])

    # --- ω_Z latent source: physical pred sinh(clip)*s_Z vs true ż = E·(Ẏ_dry ⊘ σ) ---
    omega_pred = np.sinh(np.clip(ls_head, -20.0, 20.0)) * s_z
    enc_W = model.encoder.weight.detach().cpu().numpy()           # (k, n_dry)
    dydt_dry = dfv[schema.dydt_columns(list(input_species))].to_numpy(float)
    z_dot_true = (dydt_dry / comp_scale[None, :]) @ enc_W.T
    res["omega_z_R2"] = _r2(omega_pred, z_dot_true)

    # --- atom-balance residual of the PREDICTED rates (relative; lower = better) ---
    thermo_a = SpeciesThermo.from_mechanism_yaml("chem_ForTransport.yaml", list(energy_active))
    W = thermo_a.molar_mass[None, :]
    rates_pred_mass = rate_scaler.inverse_transform(rates_scaled)
    r_molar = rates_pred_mass / W
    elem = r_molar @ thermo_a.element_matrix                       # (n, n_elem)
    res["atom_residual_rel"] = float(
        np.mean(np.linalg.norm(elem, axis=1)) / (np.mean(np.linalg.norm(r_molar, axis=1)) + 1e-300)
    )

    # --- realizability: fraction of (row, species) where predicted consumption depletes within dt ---
    Y_active = Y_in[:, [list(input_species).index(s) for s in energy_active]]
    consumed_frac = np.clip(-rates_pred_mass, 0.0, None) * REALIZ_DT / (rho[:, None] + 1e-6)
    res["realizability_violation_frac"] = float(np.mean(consumed_frac > Y_active))

    # --- transport μ/k R² (all-on only) ---
    if tp_pred is not None and "mu [Pa-s]" in dfv.columns and "k [W/m/K]" in dfv.columns:
        res["mu_R2"] = _r2(tp_pred[:, 0], dfv["mu [Pa-s]"].to_numpy(float))
        res["k_R2"] = _r2(tp_pred[:, 1], dfv["k [W/m/K]"].to_numpy(float))

    res["n_val_rows"] = int(len(dfv))
    return res


def main() -> None:
    out = {}
    for label, physics_on in (("baseline", False), ("all_on", True)):
        print(f"\n{'='*70}\nTRAIN {label} (physics_on={physics_on})\n{'='*70}", flush=True)
        cfg = make_cfg(f"runs/ab_{label}", physics_on)
        train(cfg)
        out[label] = evaluate(cfg.output_dir)
        print(f"{label} metrics: {json.dumps(out[label], indent=2)}", flush=True)

    keys = ["energy_rate_derived_R2", "energy_head_R2", "rate_R2_scaled_pooled",
            "rate_R2_major_scaled", "omega_z_R2", "atom_residual_rel",
            "realizability_violation_frac", "mu_R2", "k_R2"]
    print(f"\n{'='*78}\nA/B SUMMARY (pilot, directional/non-certifying)  n_val={out['baseline']['n_val_rows']}\n{'='*78}")
    print(f"{'metric':<32}{'baseline':>14}{'all_on':>14}{'delta':>14}")
    for k in keys:
        b = out["baseline"].get(k, float("nan"))
        a = out["all_on"].get(k, float("nan"))
        d = (a - b) if (isinstance(a, float) and isinstance(b, float)
                        and np.isfinite(a) and np.isfinite(b)) else float("nan")
        print(f"{k:<32}{b:>14.4g}{a:>14.4g}{d:>14.4g}")
    Path("runs/ab_pilot_summary.json").write_text(json.dumps(out, indent=2))
    print("\nwrote runs/ab_pilot_summary.json")


if __name__ == "__main__":
    main()

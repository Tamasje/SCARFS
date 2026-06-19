"""Architecture/training search to drive the energy relRMSE down (the 10× goal).

Each experiment is a physically/chemically/computationally-motivated change applied to the current
master config (keq OFF), trained on the pilot at a FIXED budget, and scored on the held-out val
split. Results accumulate in runs/goal_ledger.json with the factor vs the baseline relRMSE.

The energy target is deterministic in the inputs (energy identity relRMSE ~3e-5), so the floor is
solver noise, not information — relRMSE is model/training-limited and improvable. Pilot-scale,
directional, NON-certifying (re-confirm winners on the regenerated DB at certification).

Run: .venv/bin/python scripts/goal_search.py --exps baseline,rate_cap,k24 [--epochs 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ab_pilot_physics import evaluate, make_cfg  # reuse the scorer + config builder
from scarfs.training.train import train

LEDGER = Path("runs/goal_ledger.json")
PRIMARY = "energy_rate_derived_relRMSE"   # lower is better; the 10× target metric
BASKET = ["energy_rate_derived_relRMSE", "energy_rate_derived_R2", "energy_tail_relRMSE",
          "rate_R2_major_scaled", "omega_z_R2_perdim_median", "atom_residual_rel",
          "realizability_violation_frac"]


def _set(obj, **kw):
    for k, v in kw.items():
        setattr(obj, k, v)


# Each experiment mutates a fresh master-config (keq already forced off in run()).
EXPERIMENTS = {
    "baseline":   lambda c: None,
    "baseline_ref": lambda c: None,  # identical config; a distinct ledger key for a matched-budget reference
    # --- capacity (deterministic map may need more approximation power) ---
    "rate_cap":   lambda c: _set(c.model, rate_hidden=(256, 256, 128)),
    "deep_all":   lambda c: _set(c.model, rate_hidden=(256, 256, 128),
                                 energy_hidden=(128, 128), decoder_hidden=(256, 256)),
    # --- latent width (more composition info to the rate head) ---
    "k24":        lambda c: _set(c.model, latent_dim=24),
    "k32":        lambda c: _set(c.model, latent_dim=32),
    # --- loss emphasis on the energy-relevant signal / the high-|S_E| tail ---
    "energyw1":   lambda c: _set(c.loss, energy_weight=1.0),
    "tailw4":     lambda c: _set(c.data, tail_weight_alpha=4.0),
    # --- combine the batch-1 winners (k32 + capacity + tail + energy emphasis) ---
    "combo":      lambda c: (_set(c.model, latent_dim=32, rate_hidden=(256, 256, 128)),
                             _set(c.loss, energy_weight=1.0), _set(c.data, tail_weight_alpha=4.0)),
    "k48":        lambda c: _set(c.model, latent_dim=48),
    "combo_k48":  lambda c: (_set(c.model, latent_dim=48, rate_hidden=(256, 256, 128)),
                             _set(c.loss, energy_weight=1.0), _set(c.data, tail_weight_alpha=6.0)),
    # --- computational: cosine LR + warmup (deterministic map anneals to a lower final error) ---
    "combo_cos":  lambda c: (_set(c.model, latent_dim=32, rate_hidden=(256, 256, 128)),
                             _set(c.loss, energy_weight=1.0), _set(c.data, tail_weight_alpha=4.0),
                             _set(c.optim, lr_schedule="cosine", warmup_epochs=10)),
    # --- checkpoint on the deployed metric (val energy relRMSE), not latent-dominated total loss ---
    "combo_eck":  lambda c: (_set(c.model, latent_dim=32, rate_hidden=(256, 256, 128)),
                             _set(c.loss, energy_weight=1.0), _set(c.data, tail_weight_alpha=4.0),
                             _set(c.optim, checkpoint_metric="energy_relrmse")),
}


def run(names: list[str], epochs: int, patience: int, head_ft: int) -> None:
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
    for name in names:
        if name not in EXPERIMENTS:
            print(f"!! unknown experiment {name!r}; skipping", flush=True)
            continue
        print(f"\n{'='*72}\nEXP {name}  (epochs={epochs})\n{'='*72}", flush=True)
        try:
            cfg = make_cfg(f"runs/goal_{name}", physics_on=True, epochs=epochs,
                           patience=patience, head_ft=head_ft)
            cfg.loss.keq_weight = 0.0  # validated off (2026-06-19)
            EXPERIMENTS[name](cfg)
            train(cfg)
            res = evaluate(cfg.output_dir)
            res["_config"] = {"latent_dim": cfg.model.latent_dim, "rate_hidden": list(cfg.model.rate_hidden),
                              "energy_hidden": list(cfg.model.energy_hidden),
                              "decoder_hidden": list(cfg.model.decoder_hidden),
                              "energy_weight": cfg.loss.energy_weight,
                              "tail_weight_alpha": cfg.data.tail_weight_alpha, "epochs": epochs}
            ledger[name] = res
            LEDGER.write_text(json.dumps(ledger, indent=2))
            print(f"{name}: {PRIMARY}={res.get(PRIMARY):.4f}  R2={res.get('energy_rate_derived_R2'):.4f}", flush=True)
        except Exception as e:  # noqa: BLE001 — keep the batch alive
            print(f"!! EXP {name} FAILED: {e}", flush=True)

    # summary
    base = ledger.get("baseline", {}).get(PRIMARY)
    print(f"\n{'='*92}\nGOAL LEDGER (energy relRMSE; lower better)  baseline={base}\n{'='*92}")
    hdr = f"{'exp':<14}{'relRMSE':>10}{'factor':>9}{'R2':>8}{'tailRMSE':>10}{'rateR2maj':>11}{'omegaZ':>9}{'atomRes':>9}{'realizV':>9}"
    print(hdr)
    for name, r in ledger.items():
        rr = r.get(PRIMARY, float("nan"))
        fac = (base / rr) if (base and rr and np.isfinite(base) and np.isfinite(rr) and rr > 0) else float("nan")
        print(f"{name:<14}{rr:>10.4f}{fac:>9.2f}{r.get('energy_rate_derived_R2', float('nan')):>8.3f}"
              f"{r.get('energy_tail_relRMSE', float('nan')):>10.3f}{r.get('rate_R2_major_scaled', float('nan')):>11.3f}"
              f"{r.get('omega_z_R2_perdim_median', float('nan')):>9.3f}{r.get('atom_residual_rel', float('nan')):>9.3f}"
              f"{r.get('realizability_violation_frac', float('nan')):>9.3f}")
    print(f"\nledger: {LEDGER}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="baseline,rate_cap,deep_all,k24,k32,energyw1,tailw4")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--head-ft", type=int, default=30)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.summary_only:
        run([], args.epochs, args.patience, args.head_ft)
        return
    run([s.strip() for s in args.exps.split(",") if s.strip()], args.epochs, args.patience, args.head_ft)


if __name__ == "__main__":
    main()

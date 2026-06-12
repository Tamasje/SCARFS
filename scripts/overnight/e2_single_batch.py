"""E2/E3 (overnight diagnosis): the decisive single-batch overfit discriminator.

Question (per the brief): can a fresh model drive the latent_source loss to ~0 on ONE batch
with every other loss off?  Run the identical seeded experiment under
  (E2) the SAVED s_Z (the bundle's mis-scaled value — latent-STATE median), and
  (E3) the CORRECTED s_Z* (per-dim median |ż| computed on the same batch),
and compare descent. If E2 cannot overfit while E3 can, the failure is target
conditioning (H1), not capacity/optimization (H4/H5).

Usage: .venv/bin/python scripts/overnight/e2_single_batch.py [--steps 400] [--batch 2048]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import torch

from scarfs.benchmark.loader import infer_schema
from scarfs.models.adapter import TorchSurrogate
from scarfs.models.common import thermo_features
from scarfs.models.neuralcoil import MergedCoil
from scarfs.training.datamodule import tripartite_case_split

BUNDLE = REPO / "runs" / "merged_bootstrap_stride5"
DEFAULT_DB = "/Users/tamasbuzogany/Documents/SCARFS/TEST_ETHANE_LOW_sobol_stride5.parquet"


def overfit(tag: str, x_std: np.ndarray, q_std: np.ndarray, t_target: np.ndarray,
            steps: int, lr: float, seed: int, *, direct_z: bool = False,
            full_input: bool = False) -> list[tuple[int, float]]:
    """Train a fresh model on ONE batch, latent_source loss only; return (step, loss).

    direct_z
        Feed the head ``z = E·x`` directly instead of ``z_proj = E·D(z, q)``. In real
        training the decoder is trained by the recon/manifold losses; with those off,
        z_proj passes through an UNTRAINED decoder and the head input is noise — this
        flag removes that experiment confound.
    full_input
        Bypass the k-dim bottleneck entirely: a plain MLP on ``[x_std, q]`` (212+4 in).
        Tests whether the target is a function of the FULL state at all (input-information
        limit vs optimization limit).
    """
    torch.manual_seed(seed)
    k = t_target.shape[1]
    if full_input:
        from scarfs.models.nets import make_mlp
        net = make_mlp([x_std.shape[1] + q_std.shape[1], 256, 256, k],
                       activation="silu", layernorm=True, final_activation=None)
        params = net.parameters()
    else:
        model = MergedCoil(
            n_dry=x_std.shape[1], n_energy_active=8, latent_dim=k,
            decoder_hidden=(128, 256), rate_hidden=(16,), latent_source_hidden=(128, 128),
            energy_hidden=(16,), activation="silu", spectral_norm=False,
        )
        _, _, Vt = np.linalg.svd(x_std - x_std.mean(axis=0), full_matrices=False)
        with torch.no_grad():
            model.encoder.weight.copy_(torch.as_tensor(Vt[:k], dtype=torch.float32))
        params = model.parameters()

    xb = torch.as_tensor(x_std, dtype=torch.float32)
    qb = torch.as_tensor(q_std, dtype=torch.float32)
    tb = torch.as_tensor(t_target, dtype=torch.float32)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-6)
    curve: list[tuple[int, float]] = []
    for step in range(steps + 1):
        if full_input:
            pred = net(torch.cat([xb, qb], dim=-1))
        else:
            z = model.encode(xb)
            head_in = z if direct_z else model.encoder(model.decode(z, qb))
            pred = model.latent_source(head_in, qb)
        loss = ((pred - tb) ** 2).mean()
        if step % max(steps // 12, 1) == 0 or step == steps:
            curve.append((step, float(loss.detach())))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params if full_input else model.parameters(), 1.0)
        opt.step()
    print(f"[{tag}] loss curve (step, MSE):")
    for s, v in curve:
        print(f"   {s:5d}  {v:10.4f}")
    print(f"[{tag}] initial -> final: {curve[0][1]:.3f} -> {curve[-1][1]:.4f} "
          f"(reduction x{curve[0][1] / max(curve[-1][1], 1e-9):.1f})")
    return curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database", default=DEFAULT_DB)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = json.loads((BUNDLE / "spec.json").read_text(encoding="utf-8"))
    cfg = spec["config_echo"]["data"]
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(args.database)
    needed = [c for c in pf.schema_arrow.names
              if c.startswith(("Y_", "dYdt_")) or c == "CaseID"
              or c.split("[")[0].strip() in ("T", "P", "rho", "tau")
              or "absorption" in c.lower() or c.startswith("R_")]
    df = pd.read_parquet(args.database, columns=needed)
    schema = infer_schema(df)
    train_mask, _, _ = tripartite_case_split(
        df, schema, val_fraction=cfg["val_fraction"], test_fraction=cfg["test_fraction"],
        seed=cfg["seed"], split_by_case=cfg["split_by_case"])
    dft = df.loc[train_mask].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(dft), size=args.batch, replace=False)
    dfb = dft.iloc[sel].reset_index(drop=True)
    print(f"single batch: {len(dfb)} rows from {dfb[schema.meta['CaseID']].nunique()} train cases "
          f"(seed {args.seed})")

    # exact training-time scalers/encoder geometry from the bundle (read-only)
    surr = TorchSurrogate.from_merged_bundle(BUNDLE, schema)
    scalers = surr.scalers
    input_species = list(surr.spec.input_species)
    y = dfb[schema.y_columns(input_species)].to_numpy(float)
    x_std = scalers.composition.transform(y)
    sigma = np.asarray(scalers.composition.scale_, dtype=float)
    dydt = dfb[schema.dydt_columns(input_species)].to_numpy(float)
    q_std = scalers.thermo.transform(thermo_features(
        dfb[schema.state["T"]].to_numpy(float), dfb[schema.state["P"]].to_numpy(float)))

    # ż with the batch-PCA encoder (same E used to init the fresh model)
    _, _, Vt = np.linalg.svd(x_std - x_std.mean(axis=0), full_matrices=False)
    E0 = Vt[: len(spec["arcsinh_latent_scale"])]
    z_dot = (dydt / sigma) @ E0.T

    s_saved = np.asarray(spec["arcsinh_latent_scale"], dtype=float)
    s_corr = np.maximum(np.median(np.abs(z_dot), axis=0), 1e-12)
    print(f"s_Z saved (bundle): {np.array2string(s_saved, precision=3)}")
    print(f"s_Z corrected (median|zdot| on batch): {np.array2string(s_corr, precision=3)}")

    t_saved = np.arcsinh(z_dot / s_saved)
    t_corr = np.arcsinh(z_dot / s_corr)
    print(f"target Var — saved: {t_saved.var(axis=0).mean():.2f}, corrected: {t_corr.var(axis=0).mean():.2f}")

    variants = [
        ("E2b saved-sZ z-direct", t_saved, {"direct_z": True}),
        ("E3b corrected-sZ z-direct", t_corr, {"direct_z": True}),
        ("E4b corrected-sZ FULL-input", t_corr, {"full_input": True}),
    ]
    results = []
    for tag, tt, kw in variants:
        c = overfit(tag, x_std, q_std, tt, args.steps, args.lr, args.seed, **kw)
        frac = c[-1][1] / tt.var(axis=0).mean()
        results.append((tag, c[0][1], c[-1][1], frac))

    print("\nVERDICT inputs (final/Var(target) — 0=memorized, 1=mean-prediction):")
    for tag, ini, fin, frac in results:
        print(f"  {tag:32s} {ini:8.2f} -> {fin:8.3f}   final/Var = {frac:.3f}")


if __name__ == "__main__":
    main()

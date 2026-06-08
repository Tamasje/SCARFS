"""Training entry point (PyTorch).

Usage (on the HPC, where PyTorch + the database are available)::

    python -m scarfs.training.train --config configs/train_reduced.yaml
    python -m scarfs.training.train --config configs/train_neuralcoil.yaml

Produces, under ``output_dir``: ``model.pt`` (state dict), ``scalers.pkl`` (fitted scalers),
``spec.json`` (resolved input/target species), ``metrics.json`` (train/val curves), and a portable
``bundle/`` for the Fluent UDF via :mod:`scarfs.coupling.export`.

The data path (loading, feature/target assembly, weighting) is the torch-free, tested
``scarfs.training.datamodule``; this module adds the optimisation loop only.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..benchmark.loader import infer_schema, load_database
from ..models.features import FeatureSpec
from .config import TrainConfig
from .datamodule import DataBundle, prepare_data


def build_model(cfg: TrainConfig, spec: FeatureSpec):
    """Instantiate the configured surrogate (PyTorch)."""
    if cfg.model.kind == "reduced":
        from ..models.reduced import ReducedSurrogate

        return ReducedSurrogate(
            n_features=spec.n_input_features, n_targets=spec.n_targets,
            hidden=tuple(cfg.model.hidden), activation=cfg.model.activation,
        )
    if cfg.model.kind == "neuralcoil":
        from ..models.neuralcoil import NeuralCoil

        return NeuralCoil(
            n_dry=len(spec.input_species), n_targets=spec.n_targets,
            latent_dim=cfg.model.latent_dim, n_thermo=4,
            decoder_hidden=tuple(cfg.model.decoder_hidden),
            rate_hidden=tuple(cfg.model.rate_hidden), activation=cfg.model.activation,
        )
    raise ValueError(f"Unknown model kind {cfg.model.kind!r}")


def _epoch(model, cfg, bundle: DataBundle, optimiser, train: bool) -> float:
    """Run one epoch (train or eval); return the mean loss."""
    import torch
    from . import losses as L

    X = bundle.X_train if train else bundle.X_val
    Y = bundle.Y_train if train else bundle.Y_val
    W = bundle.w_train if train else bundle.w_val
    if len(X) == 0:
        return float("nan")

    device = next(model.parameters()).device
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)
    Wt = torch.as_tensor(W, dtype=torch.float32, device=device)
    sw = torch.as_tensor(bundle.species_weights, dtype=torch.float32, device=device)
    n_in = len(bundle.spec.input_species)

    model.train(train)
    idx = np.arange(len(X))
    if train:
        np.random.default_rng(cfg.data.seed).shuffle(idx)
    total, count = 0.0, 0
    bs = cfg.optim.batch_size
    for start in range(0, len(idx), bs):
        sel = idx[start : start + bs]
        xb, yb, wb = Xt[sel], Yt[sel], Wt[sel]
        if cfg.model.kind == "neuralcoil":
            loss, _ = L.neuralcoil_composite(
                model, xb[:, :n_in], xb[:, n_in:], yb, wb, sw,
                recon_weight=cfg.loss.recon_weight, manifold_weight=cfg.loss.manifold_weight,
                noise_std=cfg.loss.noise_std,
            )
        else:
            loss = L.weighted_rate_loss(model(xb), yb, wb, sw)
        if train:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimiser.step()
        total += float(loss.detach()) * len(sel)
        count += len(sel)
    return total / max(count, 1)


def train(cfg: TrainConfig) -> dict:
    """Run the full training loop and persist artifacts. Returns the metrics dict."""
    import torch

    df = load_database(cfg.data.database_path)
    schema = infer_schema(df)
    bundle = prepare_data(cfg.data, df, schema)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, bundle.spec).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    history: list[dict] = []
    best_val, best_state, since = float("inf"), None, 0
    for epoch in range(cfg.optim.epochs):
        tr = _epoch(model, cfg, bundle, optimiser, train=True)
        va = _epoch(model, cfg, bundle, optimiser, train=False)
        history.append({"epoch": epoch, "train": tr, "val": va})
        monitor = va if np.isfinite(va) else tr
        if monitor < best_val:
            best_val, since = monitor, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.optim.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {"best_val": best_val, "epochs_run": len(history), "history": history,
               "config": asdict(cfg), "spec": {"input": list(bundle.spec.input_species),
                                                "target": list(bundle.spec.target_species)}}
    _save_artifacts(cfg, model, bundle, metrics)
    return metrics


def _save_artifacts(cfg: TrainConfig, model, bundle: DataBundle, metrics: dict) -> None:
    """Persist model, scalers, spec, metrics, and the portable UDF bundle."""
    import torch

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with (out / "scalers.pkl").open("wb") as fh:
        pickle.dump(bundle.scalers, fh)
    (out / "spec.json").write_text(
        json.dumps({"input": list(bundle.spec.input_species), "target": list(bundle.spec.target_species)}, indent=2),
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a SCARFS source-term surrogate.")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML TrainConfig.")
    args = parser.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    metrics = train(cfg)
    print(f"Done. best_val={metrics['best_val']:.6g}, epochs={metrics['epochs_run']}, out={cfg.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

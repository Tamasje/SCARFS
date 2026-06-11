"""Training entry point (PyTorch).

Usage (on the HPC, where PyTorch + the database are available)::

    python -m scarfs.training.train --config configs/train_reduced.yaml
    python -m scarfs.training.train --config configs/train_neuralcoil.yaml
    python -m scarfs.training.train --config configs/train_merged.json

Produces, under ``output_dir``: ``model.pt`` (state dict), ``scalers.pkl`` (fitted scalers),
``spec.json`` (resolved input/target species, split case IDs, config echo), ``metrics.json``
(train/val curves + per-term losses + absorption metrics on val).

The data path (loading, feature/target assembly, weighting) is the torch-free, tested
``scarfs.training.datamodule``; this module adds the optimisation loop only.

Merged-model path (kind="merged")
-----------------------------------
1. Build a StandardScaler (linear, no log) over the dry composition on the train split.
2. Select energy-active species on the train split (coverage + always_include).
3. Fit arcsinh scale constants for rates and latent source from the PCA-init pre-pass.
4. PCA-init the encoder; freeze latent-source scale constants.
5. Train with the merged composite loss.
6. Compute val-set absorption metrics (R², relRMSE, tail relRMSE/median per §5 of the plan).
7. Save bundle: model.pt, scalers.pkl, spec.json, metrics.json.

Notes
-----
- No winsorization or train-quantile output bounds anywhere in the energy path.
- Benchmark code (B2a) is NOT imported here to avoid coupling; absorption metrics are
  computed locally (3–4 metrics).
- B1b ``scarfs.models.thermo`` and ``scarfs.models.neuralcoil.MergedCoil`` are imported
  lazily (train-time only); missing imports raise a clear error with the stub note.
"""

from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..benchmark.loader import infer_schema, load_database
from ..models.features import FeatureSpec
from ..schema import MAJOR_SPECIES
from .config import TrainConfig
from .datamodule import DataBundle, prepare_data, resolve_species, enthalpy_aware_weights


# ---------------------------------------------------------------------------
# Helpers shared by all paths
# ---------------------------------------------------------------------------

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
    if cfg.model.kind == "merged":
        return _build_merged_model(cfg, spec)
    raise ValueError(f"Unknown model kind {cfg.model.kind!r}")


def _build_merged_model(cfg: TrainConfig, spec: FeatureSpec):
    """Instantiate a MergedCoil (B1b contract).

    Raises ImportError with a helpful message if B1b is not yet integrated.
    """
    try:
        from ..models.neuralcoil import MergedCoil
    except ImportError as exc:
        raise ImportError(
            "MergedCoil is not yet available (B1b integration pending). "
            "Stub it in tests via monkeypatching. "
            f"Original error: {exc}"
        ) from exc
    return MergedCoil(
        n_dry=len(spec.input_species),
        n_energy_active=spec.n_targets,
        latent_dim=cfg.model.latent_dim,
        n_thermo=4,
        decoder_hidden=tuple(cfg.model.decoder_hidden),
        rate_hidden=tuple(cfg.model.rate_hidden),
        latent_source_hidden=tuple(cfg.model.latent_source_hidden),
        energy_hidden=tuple(cfg.model.energy_hidden),
        activation=cfg.model.activation,
        spectral_norm=cfg.model.spectral_norm,
    )


# ---------------------------------------------------------------------------
# Energy-active species selection
# ---------------------------------------------------------------------------

def select_energy_active_species(
    df_train,
    schema,
    all_dry_species: tuple[str, ...],
    coverage: float = 0.999,
    always_include: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Select species whose cumulative |R_i| share covers *coverage* of total mass rate flux.

    This is a data-driven approximation to the plan's ``Σ|h_i·ω_i|`` share selection; the
    full enthalpy-aware version requires B1b thermo integration and is deferred to ``train.py``
    when ``SpeciesThermo`` is available.

    The result always includes:
    - Species whose cumulative share (sorted descending by mean |R_i|) reaches *coverage*.
    - All species in *always_include*.
    - All species in ``MAJOR_SPECIES`` present in the database with max Y > 1e-4.

    Parameters
    ----------
    df_train
        Training-split DataFrame.
    schema
        Column contract.
    all_dry_species
        All dry (non-diluent) species to consider.
    coverage
        Fraction of total |R| flux to cover (default 0.999).
    always_include
        Species names to always include regardless of flux share.
    """
    # try to use rate columns; fall back to Y-based importance
    available = []
    mean_abs_rate = {}
    for s in all_dry_species:
        r_col = f"R_{s}"
        if r_col in df_train.columns:
            available.append(s)
            mean_abs_rate[s] = float(np.abs(df_train[r_col].to_numpy(dtype=float)).mean())

    if not available:
        warnings.warn("select_energy_active_species: no R_ columns found; using all dry species.")
        return tuple(all_dry_species)

    total_flux = sum(mean_abs_rate.values())
    if total_flux <= 0:
        return tuple(all_dry_species)

    sorted_species = sorted(available, key=lambda s: mean_abs_rate[s], reverse=True)
    cumulative = 0.0
    selected: list[str] = []
    for s in sorted_species:
        selected.append(s)
        cumulative += mean_abs_rate[s] / total_flux
        if cumulative >= coverage:
            break

    # always_include and MAJOR_SPECIES (present + Y>1e-4)
    must_include: set[str] = set(always_include)
    for s in MAJOR_SPECIES:
        if s in schema.species:
            y_col = f"Y_{s}"
            if y_col in df_train.columns and float(df_train[y_col].max()) > 1e-4:
                must_include.add(s)

    selected_set = set(selected) | must_include
    # preserve database ordering
    return tuple(s for s in all_dry_species if s in selected_set)


# ---------------------------------------------------------------------------
# Merged-model scalers
# ---------------------------------------------------------------------------

def build_linear_composition_scaler(X_comp: np.ndarray):
    """Fit a StandardScaler (mean/std, linear — no log) over the composition block.

    Returns the fitted ``scarfs.models.common.StandardScaler``.
    """
    from ..models.common import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_comp)
    return scaler


def build_arcsinh_rate_scaler(rates_phys: np.ndarray):
    """Fit an ArcsinhScaler over physical mass rates for the energy-active species.

    Returns the fitted ``scarfs.models.common.ArcsinhScaler``.
    """
    from ..models.common import ArcsinhScaler
    scaler = ArcsinhScaler()
    scaler.fit(rates_phys)
    return scaler


def pca_init_encoder(model, X_comp_train: np.ndarray) -> np.ndarray:
    """Initialise the encoder weight matrix with the top-k PCA components.

    Runs SVD on the standardised composition; sets model.encoder.weight.data.
    Returns the ``(k, n_dry)`` component matrix (same shape as encoder weight).
    """
    import torch
    _, _, Vt = np.linalg.svd(X_comp_train - X_comp_train.mean(axis=0), full_matrices=False)
    k = model.encoder.weight.shape[0]
    components = Vt[:k]  # (k, n_dry)
    with torch.no_grad():
        model.encoder.weight.copy_(torch.as_tensor(components, dtype=torch.float32))
    return components


def freeze_latent_arcsinh_scale(model, X_comp_train: np.ndarray) -> np.ndarray:
    """Compute per-dim arcsinh scale constants for the latent source from the PCA-init encoder.

    Returns a ``(k,)`` array: per-dimension median |E·X_comp| used as arcsinh scale.
    These are frozen after the PCA init pass and do not change during training.
    """
    import torch
    with torch.no_grad():
        enc_w = model.encoder.weight.detach().cpu().numpy()  # (k, n_dry)
        z = X_comp_train @ enc_w.T                            # (n_train, k)
    scale = np.maximum(np.median(np.abs(z), axis=0), 1e-12)
    return scale


# ---------------------------------------------------------------------------
# Val-set absorption metrics (no import from scarfs.benchmark)
# ---------------------------------------------------------------------------

def absorption_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    tail_fraction: float = 0.20,
) -> dict[str, float]:
    """Compute the §5 energy acceptance metrics on a val/test split.

    Parameters
    ----------
    pred, target
        ``(n,)`` arrays [J m-3 s-1].
    tail_fraction
        Top-*tail_fraction* of ``|target|`` rows define the tail subset.

    Returns
    -------
    dict with keys: r2, rel_rmse, tail_rel_rmse, tail_median_rel_err.
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(target) == 0:
        return {"r2": float("nan"), "rel_rmse": float("nan"),
                "tail_rel_rmse": float("nan"), "tail_median_rel_err": float("nan")}

    # global R²
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # global relRMSE
    denom = np.sqrt(np.mean(target ** 2)) if np.any(target != 0) else 1.0
    rel_rmse = float(np.sqrt(np.mean((pred - target) ** 2)) / denom)

    # tail subset
    n_tail = max(1, int(round(tail_fraction * len(target))))
    tail_idx = np.argpartition(np.abs(target), -n_tail)[-n_tail:]
    pred_tail = pred[tail_idx]
    target_tail = target[tail_idx]

    denom_tail = np.sqrt(np.mean(target_tail ** 2)) if np.any(target_tail != 0) else 1.0
    tail_rel_rmse = float(np.sqrt(np.mean((pred_tail - target_tail) ** 2)) / denom_tail)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err_tail = np.abs(pred_tail - target_tail) / np.maximum(np.abs(target_tail), 1.0)
    tail_median_rel_err = float(np.nanmedian(rel_err_tail))

    return {
        "r2": r2,
        "rel_rmse": rel_rmse,
        "tail_rel_rmse": tail_rel_rmse,
        "tail_median_rel_err": tail_median_rel_err,
    }


# ---------------------------------------------------------------------------
# Training epochs
# ---------------------------------------------------------------------------

def _epoch(model, cfg: TrainConfig, bundle: DataBundle, optimiser, train: bool) -> tuple[float, dict]:
    """Run one epoch (train or eval); return (mean_loss, per-term dict)."""
    import torch
    from . import losses as L

    X = bundle.X_train if train else bundle.X_val
    Y = bundle.Y_train if train else bundle.Y_val
    W = bundle.w_train if train else bundle.w_val
    if len(X) == 0:
        return float("nan"), {}

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
    last_parts: dict[str, float] = {}
    bs = cfg.optim.batch_size
    for start in range(0, len(idx), bs):
        sel = idx[start: start + bs]
        xb, yb, wb = Xt[sel], Yt[sel], Wt[sel]
        if cfg.model.kind == "neuralcoil":
            loss, parts = L.neuralcoil_composite(
                model, xb[:, :n_in], xb[:, n_in:], yb, wb, sw,
                recon_weight=cfg.loss.recon_weight,
                manifold_weight=cfg.loss.manifold_weight,
                noise_std=cfg.loss.noise_std,
            )
        elif cfg.model.kind == "merged":
            loss, parts = _merged_batch(model, cfg, bundle, xb, yb, wb, sw, device, sel, train)
        else:
            loss = L.weighted_rate_loss(model(xb), yb, wb, sw)
            parts = {"rate": float(loss.detach())}
        if train:
            optimiser.zero_grad()
            loss.backward()
            import torch as _torch
            _torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimiser.step()
        total += float(loss.detach()) * len(sel)
        count += len(sel)
        last_parts = parts
    return total / max(count, 1), last_parts


def _merged_batch(model, cfg: TrainConfig, bundle: DataBundle, xb, yb, wb, sw, device, sel, train: bool):
    """Compute the merged composite loss for a single batch.

    All tensors that depend on training-set statistics (arcsinh scales, sigma, encoder weight)
    are stored on the bundle's merged_extras dict (set up in train_merged).
    """
    import torch
    from . import losses as L

    extras: dict[str, Any] = getattr(bundle, "_merged_extras", {})

    n_in = len(bundle.spec.input_species)
    y_std = xb[:, :n_in]
    q = xb[:, n_in:]

    # physical rate targets and dydt (yb is scaled; invert to physical for the loss)
    # yb is arcsinh-scaled; invert to physical
    rate_scaler = extras.get("rate_scaler")
    if rate_scaler is not None:
        target_rates_phys = torch.as_tensor(
            rate_scaler.inverse_transform(yb.detach().cpu().numpy()),
            dtype=torch.float32, device=device,
        )
    else:
        target_rates_phys = yb  # fallback: use scaled as proxy

    # absorption / dYdt / density arrays are SPLIT-LOCAL: `sel` indexes the train arrays during
    # train epochs and the val arrays during eval epochs — mixing them mis-labels every row.
    split = "train" if train else "val"
    absorption_target = extras.get(f"absorption_target_{split}")
    if absorption_target is not None:
        at = torch.as_tensor(absorption_target[sel], dtype=torch.float32, device=device)
    else:
        at = torch.ones(len(sel), dtype=torch.float32, device=device)

    # dydt physical for ALL dry input species (latent-source target needs the full vector)
    dydt_dry_np = extras.get(f"dydt_dry_{split}")
    if dydt_dry_np is not None:
        dydt_dry_phys = torch.as_tensor(dydt_dry_np[sel], dtype=torch.float32, device=device)
    else:
        dydt_dry_phys = torch.zeros((len(sel), n_in), dtype=torch.float32, device=device)

    # density (mass-rate <-> dY/dt conversion in the consistency term)
    rho_np = extras.get(f"rho_{split}")
    if rho_np is not None:
        rho = torch.as_tensor(rho_np[sel], dtype=torch.float32, device=device)
    else:
        rho = torch.ones(len(sel), dtype=torch.float32, device=device)

    arcsinh_rate_scale = torch.as_tensor(
        extras.get("arcsinh_rate_scale", np.ones(yb.shape[1])),
        dtype=torch.float32, device=device,
    )
    arcsinh_latent_scale = torch.as_tensor(
        extras.get("arcsinh_latent_scale", np.ones(cfg.model.latent_dim)),
        dtype=torch.float32, device=device,
    )
    sigma_active = torch.as_tensor(
        extras.get("sigma_active", np.ones(yb.shape[1])),
        dtype=torch.float32, device=device,
    )
    sigma_comp_all = torch.as_tensor(
        extras.get("sigma_comp_all", np.ones(n_in)),
        dtype=torch.float32, device=device,
    )
    active_col_idx = torch.as_tensor(
        extras.get("active_col_idx", np.arange(yb.shape[1])),
        dtype=torch.long, device=device,
    )
    enthalpy_weights = torch.as_tensor(
        extras.get("enthalpy_weights", np.ones(yb.shape[1])),
        dtype=torch.float32, device=device,
    )
    species_weights_qoi = torch.as_tensor(
        extras.get("species_weights_qoi", np.ones(n_in)),
        dtype=torch.float32, device=device,
    )
    molar_mass_t = element_matrix_t = None
    if cfg.loss.atom_balance_weight > 0.0:
        mm = extras.get("molar_mass")
        em = extras.get("element_matrix")
        if mm is not None and em is not None:
            molar_mass_t = torch.as_tensor(mm, dtype=torch.float32, device=device)
            element_matrix_t = torch.as_tensor(em, dtype=torch.float32, device=device)

    # Lagrangian pairs (if any) – train-split indices only; never applied during val epochs
    lagrangian_idx_t = lagrangian_idx_tp1 = lagrangian_dtau = None
    if train and cfg.loss.rollout_mode == "lagrangian" and bundle.lagrangian_pairs is not None:
        idx_t_all, idx_tp1_all, dtau_all = bundle.lagrangian_pairs
        # get batch row indices in the original train-split array
        sel_set = set(sel.tolist())
        # remap: find pairs where both idx_t and idx_tp1 are in this batch
        sel_arr = np.asarray(sel)
        # build local index map
        local_map = {v: i for i, v in enumerate(sel_arr.tolist())}
        valid = np.array([i for i, (a, b) in enumerate(zip(idx_t_all, idx_tp1_all))
                          if a in local_map and b in local_map])
        if len(valid) > 0:
            local_t = np.array([local_map[idx_t_all[i]] for i in valid], dtype=np.intp)
            local_tp1 = np.array([local_map[idx_tp1_all[i]] for i in valid], dtype=np.intp)
            lagrangian_idx_t = torch.as_tensor(local_t, dtype=torch.long, device=device)
            lagrangian_idx_tp1 = torch.as_tensor(local_tp1, dtype=torch.long, device=device)
            lagrangian_dtau = torch.as_tensor(dtau_all[valid], dtype=torch.float32, device=device)

    # absorption_from_rates_fn: use B1b if available, else None
    abs_fn = extras.get("absorption_from_rates_fn", None)

    return L.merged_composite(
        model=model,
        y_std_scaled=y_std,
        q=q,
        target_rates_phys=target_rates_phys,
        absorption_target=at,
        dydt_dry_phys=dydt_dry_phys,
        rho=rho,
        row_weights=wb,
        enthalpy_weights=enthalpy_weights,
        species_weights_qoi=species_weights_qoi,
        arcsinh_rate_scale=arcsinh_rate_scale,
        arcsinh_latent_scale=arcsinh_latent_scale,
        sigma_active=sigma_active,
        sigma_comp_all=sigma_comp_all,
        active_col_idx=active_col_idx,
        rate_weight=cfg.loss.rate_weight,
        latent_source_weight=cfg.loss.latent_source_weight,
        energy_weight=cfg.loss.energy_weight,
        energy_distill_weight=cfg.loss.energy_distill_weight,
        energy_target_weight=cfg.loss.energy_target_weight,
        consistency_weight=cfg.loss.consistency_weight,
        recon_weight=cfg.loss.recon_weight,
        qoi_recon_weight=cfg.loss.qoi_recon_weight,
        manifold_weight=cfg.loss.manifold_weight,
        atom_balance_weight=cfg.loss.atom_balance_weight,
        noise_std=cfg.loss.noise_std,
        rollout_mode=cfg.loss.rollout_mode,
        idx_t=lagrangian_idx_t,
        idx_tp1=lagrangian_idx_tp1,
        dtau=lagrangian_dtau,
        molar_mass=molar_mass_t,
        element_matrix=element_matrix_t,
        absorption_from_rates_fn=abs_fn,
    )


# ---------------------------------------------------------------------------
# Main train entry point
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig) -> dict:
    """Run the full training loop and persist artifacts.  Returns the metrics dict."""
    import torch

    df = load_database(cfg.data.database_path)
    schema = infer_schema(df)

    if cfg.model.kind == "merged":
        return _train_merged(cfg, df, schema)

    bundle = prepare_data(cfg.data, df, schema)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, bundle.spec).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)

    history: list[dict] = []
    best_val, best_state, since = float("inf"), None, 0
    for epoch in range(cfg.optim.epochs):
        tr, _ = _epoch(model, cfg, bundle, optimiser, train=True)
        va, _ = _epoch(model, cfg, bundle, optimiser, train=False)
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

    metrics = {
        "best_val": best_val, "epochs_run": len(history), "history": history,
        "config": asdict(cfg),
        "spec": {"input": list(bundle.spec.input_species), "target": list(bundle.spec.target_species)},
    }
    _save_artifacts(cfg, model, bundle, metrics)
    return metrics


def _train_merged(cfg: TrainConfig, df, schema) -> dict:
    """Merged-model training path (kind='merged').

    Steps:
    1. 70/15/15 case split (test held out).
    2. StandardScaler (linear) for dry composition on train split.
    3. Select energy-active species.
    4. Fit arcsinh rate scaler on train split.
    5. PCA-init encoder; freeze latent arcsinh scales.
    6. Train with merged composite loss.
    7. Val-set absorption metrics.
    8. Save bundle.
    """
    import torch

    # Override data config for merged: use test_fraction if not set
    if cfg.data.test_fraction <= 0.0:
        import copy
        cfg = copy.deepcopy(cfg)
        cfg.data.test_fraction = 0.15
        cfg.data.val_fraction = 0.15 / (1.0 - 0.15)  # 15% of total → ~0.176 of remaining

    from ..models.features import build_mass_rate_matrix
    from ..models.thermo import SpeciesThermo
    from ..models.thermo import select_energy_active_species as thermo_select

    all_dry = resolve_species(schema, "dry_all")

    # -- thermo over the dry species (drop species the mechanism YAML lacks) -------------
    thermo_dry = SpeciesThermo.from_mechanism_yaml(cfg.data.mech_yaml, all_dry, missing_ok=True)
    if thermo_dry.missing:
        warnings.warn(
            f"_train_merged: {len(thermo_dry.missing)} species lack thermo data in "
            f"{cfg.data.mech_yaml} and cannot become energy-active (their direct enthalpy-flux "
            f"contribution was measured negligible in Phase 1): {list(thermo_dry.missing)[:8]}..."
        )
    found_species = thermo_dry.species

    # -- deterministic split (identical to prepare_data's: same hash, seed, fractions) ----
    from .datamodule import tripartite_case_split
    train_mask, _val_mask, _test_mask = tripartite_case_split(
        df, schema, val_fraction=cfg.data.val_fraction, test_fraction=cfg.data.test_fraction,
        seed=cfg.data.seed, split_by_case=cfg.data.split_by_case,
    )
    df_train_sel = df.loc[train_mask]

    # -- energy-active selection: rank by share of Σ|ρ·h·dYdt| on the train split --------
    always_incl = set(s for s in MAJOR_SPECIES if s in schema.species)
    y_found = df_train_sel[schema.y_columns(found_species)].to_numpy(dtype=float)
    retained_by_y = tuple(s for s, ymax in zip(found_species, y_found.max(axis=0)) if ymax > 1e-4)
    always_incl.update(retained_by_y)
    if schema.has_dydt():
        dydt_sel = df_train_sel[schema.dydt_columns(found_species)].to_numpy(dtype=float)
        rho_col = schema.require_state("rho")[0]
        (t_col,) = schema.require_state("T")
        energy_active = thermo_select(
            dydt_sel,
            df_train_sel[rho_col].to_numpy(dtype=float),
            df_train_sel[t_col].to_numpy(dtype=float),
            found_species,
            thermo_dry,
            coverage=cfg.data.energy_active_coverage,
            always_include=tuple(s for s in always_incl if s in found_species),
        )
    else:
        warnings.warn("_train_merged: no dYdt columns; falling back to |R|-share selection.")
        energy_active = select_energy_active_species(
            df_train_sel, schema, found_species,
            coverage=cfg.data.energy_active_coverage,
            always_include=tuple(always_incl),
        )

    # -- bundle: standard/linear composition + mass-rate targets, fitted on train --------
    import copy
    cfg2 = copy.deepcopy(cfg)
    cfg2.data.target_species = list(energy_active)
    bundle = prepare_data(
        cfg2.data, df, schema,
        composition_kwargs={"log": False, "mode": "standard"},
        prefer_dydt=True,
    )
    df_train = df.iloc[bundle.train_indices].reset_index(drop=True)
    df_val = df.iloc[bundle.val_indices].reset_index(drop=True)
    linear_comp_scaler = bundle.scalers.composition
    rate_scaler = bundle.scalers.rate
    n_in = len(bundle.spec.input_species)
    X_comp_train = bundle.X_train[:, :n_in]  # standardized composition block

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg, bundle.spec).to(device)

    # PCA-init encoder on the standardized composition; freeze latent arcsinh scales
    pca_init_encoder(model, X_comp_train)
    arcsinh_latent_scale = freeze_latent_arcsinh_scale(model, X_comp_train)
    arcsinh_rate_scale = rate_scaler.scale_

    # thermo restricted to the energy-active set (selection guarantees thermo coverage)
    thermo_active = SpeciesThermo.from_mechanism_yaml(cfg.data.mech_yaml, energy_active)

    # enthalpy-aware species weights (mass rates × h_mass shares; floor 1.0)
    enth_w = enthalpy_aware_weights(df_train, schema, energy_active, h_mass_fn=thermo_active.h_mass)

    # sigma vectors from the SAME composition scaler the features were built with
    all_dry_list = list(bundle.spec.input_species)
    sigma_comp_all = np.asarray(linear_comp_scaler.scale_, dtype=float)
    ea_col_idx = np.array([all_dry_list.index(s) for s in energy_active], dtype=np.intp)
    sigma_active = sigma_comp_all[ea_col_idx]

    # absorption target via the schema contract (never the CRACKSIM-internal `S Energy`)
    from ..models.features import build_absorption_target
    n_abs_clipped = 0
    try:
        abs_train = build_absorption_target(df_train, schema)
        absorption_train = abs_train.values
        n_abs_clipped = abs_train.n_clipped
        if n_abs_clipped:
            warnings.warn(f"Clipped {n_abs_clipped} negative absorption train values to 0.")
        absorption_val = build_absorption_target(df_val, schema).values if len(df_val) else None
    except (KeyError, AttributeError):
        absorption_train = np.ones(len(df_train))
        absorption_val = None
        warnings.warn("Absorption column not found; energy terms train against placeholder ones.")

    # energy-head calibration: scale = median positive train absorption
    pos = absorption_train[absorption_train > 0]
    energy_scale = float(np.median(pos)) if len(pos) else 1.0
    if hasattr(model, "set_energy_calibration"):
        model.set_energy_calibration(energy_scale, floor=0.0)

    # full-dry dYdt (latent-source target) and density, per split
    if schema.has_dydt():
        dydt_cols_dry = schema.dydt_columns(all_dry_list)
        rho_col = schema.require_state("rho")[0]
        dydt_dry_train = df_train[dydt_cols_dry].to_numpy(dtype=float)
        rho_train = df_train[rho_col].to_numpy(dtype=float)
        dydt_dry_val = df_val[dydt_cols_dry].to_numpy(dtype=float) if len(df_val) else None
        rho_val = df_val[rho_col].to_numpy(dtype=float) if len(df_val) else None
    else:
        dydt_dry_train = np.zeros((len(df_train), n_in))
        rho_train = np.ones(len(df_train))
        dydt_dry_val = rho_val = None

    # differentiable rate-derived absorption: unstandardize T from the thermo block
    import torch as _torch
    t_mean = float(bundle.scalers.thermo.mean_[0])
    t_scale = float(bundle.scalers.thermo.scale_[0])

    def _absorption_from_rates(rates_phys, q):
        T = q[..., 0] * t_scale + t_mean
        return thermo_active.absorption_from_rates_torch(rates_phys, T)

    # Lagrangian pairs
    if cfg.loss.rollout_mode == "lagrangian":
        from .datamodule import case_step_pairs
        try:
            lag_pairs = case_step_pairs(df_train, schema)
        except KeyError:
            lag_pairs = None
            warnings.warn("case_step_pairs failed (missing tau/z); Lagrangian rollout disabled.")
    else:
        lag_pairs = None
    bundle.lagrangian_pairs = lag_pairs

    # store merged extras on bundle (split-local arrays keyed _train / _val)
    bundle._merged_extras = {
        "rate_scaler": rate_scaler,
        "absorption_target_train": absorption_train,
        "absorption_target_val": absorption_val,
        "dydt_dry_train": dydt_dry_train,
        "dydt_dry_val": dydt_dry_val,
        "rho_train": rho_train,
        "rho_val": rho_val,
        "arcsinh_rate_scale": arcsinh_rate_scale,
        "arcsinh_latent_scale": arcsinh_latent_scale,
        "sigma_active": sigma_active,
        "sigma_comp_all": sigma_comp_all,
        "active_col_idx": ea_col_idx,
        "enthalpy_weights": enth_w,
        "species_weights_qoi": np.ones(n_in),
        "molar_mass": thermo_active.molar_mass,
        "element_matrix": thermo_active.element_matrix,
        "absorption_from_rates_fn": _absorption_from_rates,
    }

    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    history: list[dict] = []
    best_val, best_state, since = float("inf"), None, 0

    for epoch in range(cfg.optim.epochs):
        tr, tr_parts = _epoch(model, cfg, bundle, optimiser, train=True)
        va, va_parts = _epoch(model, cfg, bundle, optimiser, train=False)
        history.append({"epoch": epoch, "train": tr, "val": va,
                        "train_parts": tr_parts, "val_parts": va_parts})
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

    # val-set absorption metrics: both energy paths (rate-derived tie and distilled head)
    abs_metrics: dict[str, Any] = {}
    if absorption_val is not None and len(bundle.X_val):
        model.eval()
        preds_rate, preds_head = [], []
        with _torch.no_grad():
            Xv = _torch.as_tensor(bundle.X_val, dtype=_torch.float32, device=device)
            for start in range(0, len(Xv), cfg.optim.batch_size):
                xb = Xv[start: start + cfg.optim.batch_size]
                out = model.forward(xb[:, :n_in], xb[:, n_in:])
                preds_rate.append(
                    _absorption_from_rates(out["rates"], xb[:, n_in:]).cpu().numpy()
                )
                preds_head.append(out["absorption"].squeeze(-1).cpu().numpy())
        abs_metrics = {
            "rate_derived": absorption_metrics(np.concatenate(preds_rate), absorption_val),
            "head": absorption_metrics(np.concatenate(preds_head), absorption_val),
        }

    metrics = {
        "best_val": best_val,
        "epochs_run": len(history),
        "history": history,
        "config": asdict(cfg),
        "spec": {
            "input": list(bundle.spec.input_species),
            "target_energy_active": list(energy_active),
            "target": list(bundle.spec.target_species),
        },
        "split_case_ids": {
            "test": bundle.test_case_ids or [],
        },
        "absorption_metrics_val": abs_metrics,
        "thermo_missing_species": list(thermo_dry.missing),
        "n_absorption_clipped_train": int(n_abs_clipped),
    }
    # export-time safety stats: train latent envelope (UDF OOD clamp) and the energy-clamp
    # bound (plan §3: clamp from the FULL data range x1.3 — never a prediction-falsifying p99)
    with _torch.no_grad():
        z_train = (
            model.encode(_torch.as_tensor(X_comp_train, dtype=_torch.float32, device=device))
            .cpu().numpy()
        )
    (t_col_x,) = schema.require_state("T")
    (p_col_x,) = schema.require_state("P")
    export_stats = {
        "latent_env_min": z_train.min(axis=0).tolist(),
        "latent_env_max": z_train.max(axis=0).tolist(),
        "absorption_train_max": float(absorption_train.max()),
        "energy_clamp": float(1.3 * absorption_train.max()),
        "T_train_min": float(df_train[t_col_x].min()),
        "T_train_max": float(df_train[t_col_x].max()),
        "P_train_min": float(df_train[p_col_x].min()),
        "P_train_max": float(df_train[p_col_x].max()),
    }
    _save_artifacts_merged(cfg, model, bundle, metrics, energy_active,
                           linear_comp_scaler, rate_scaler, arcsinh_latent_scale,
                           energy_calibration={"scale": energy_scale, "floor": 0.0},
                           export_stats=export_stats)
    return metrics


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _save_artifacts(cfg: TrainConfig, model, bundle: DataBundle, metrics: dict) -> None:
    """Persist model, scalers, spec, metrics (reduced/neuralcoil paths)."""
    import torch

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    with (out / "scalers.pkl").open("wb") as fh:
        pickle.dump(bundle.scalers, fh)
    (out / "spec.json").write_text(
        json.dumps({"input": list(bundle.spec.input_species),
                    "target": list(bundle.spec.target_species)}, indent=2),
        encoding="utf-8",
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def _save_artifacts_merged(
    cfg: TrainConfig,
    model,
    bundle: DataBundle,
    metrics: dict,
    energy_active: tuple[str, ...],
    comp_scaler,
    rate_scaler,
    arcsinh_latent_scale: np.ndarray,
    energy_calibration: dict | None = None,
    export_stats: dict | None = None,
) -> None:
    """Persist merged-model artifacts.

    Saves:
    - ``model.pt`` — model state dict.
    - ``scalers.pkl`` — dict of all scalers + scale arrays.
    - ``spec.json`` — species lists, energy_active, composition mode (standard), split case IDs,
      energy calibration, config echo.
    - ``metrics.json`` — training curves + per-term losses + absorption metrics on val.
    """
    import torch

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")

    scalers_bundle = {
        "composition_scaler": comp_scaler,
        "rate_scaler": rate_scaler,
        "arcsinh_latent_scale": arcsinh_latent_scale,
        "composition_mode": "standard",  # linear per-species standardisation
        "legacy_scalers": bundle.scalers,  # keep original bundle scalers for backward compat
    }
    with (out / "scalers.pkl").open("wb") as fh:
        pickle.dump(scalers_bundle, fh)

    comp_mean = comp_scaler.mean_.tolist() if hasattr(comp_scaler, "mean_") else []
    comp_scale = comp_scaler.scale_.tolist() if hasattr(comp_scaler, "scale_") else []
    spec_doc = {
        "kind": "merged",
        "input": list(bundle.spec.input_species),
        "target": list(bundle.spec.target_species),
        "energy_active": list(energy_active),
        "composition_mode": "standard",
        "composition_mean": comp_mean,
        "composition_scale": comp_scale,
        "arcsinh_latent_scale": arcsinh_latent_scale.tolist(),
        "energy_calibration": energy_calibration or {"scale": 1.0, "floor": 0.0},
        "rate_source": getattr(bundle.scalers, "rate_source", "r_mass"),
        "mech_yaml": cfg.data.mech_yaml,
        "export_stats": export_stats or {},
        "split_case_ids": metrics.get("split_case_ids", {}),
        "config_echo": asdict(cfg),
    }
    (out / "spec.json").write_text(json.dumps(spec_doc, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m scarfs.training.train --config <path>``."""
    parser = argparse.ArgumentParser(description="Train a SCARFS source-term surrogate.")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML TrainConfig.")
    args = parser.parse_args(argv)
    cfg = TrainConfig.load(args.config)
    metrics = train(cfg)
    print(f"Done. best_val={metrics['best_val']:.6g}, epochs={metrics['epochs_run']}, out={cfg.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()

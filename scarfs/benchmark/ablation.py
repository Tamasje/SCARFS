"""k-ablation runner for the SCARFS benchmark (plan §4 E-e).

Trains a separate model for each latent dimension k in a given list, collecting
per-k validation metrics so that the best k can be selected before the full HPC
training run.

The ablation is deliberately lightweight: it calls ``scarfs.training.train.train``
(imported lazily to avoid hard PyTorch dependency at import time) with a deep-
copied :class:`~scarfs.training.config.TrainConfig` whose ``model.latent_dim``
is patched for each k.  The results include:

- Validation absorption R² and relRMSE (both 'rate' and 'head' paths if the
  bundle's metrics.json records them).
- Per-major-species rate R² on the validation split via
  :class:`~scarfs.models.adapter.TorchSurrogate`.
- The feasibility-gate verdict (:func:`~scarfs.benchmark.feasibility.feasibility_table`)
  for this k.

Public API
----------
- :class:`AblationResult` — result for one k.
- :class:`AblationReport` — collection of results + :meth:`~AblationReport.to_table`.
- :func:`run_k_ablation` — main entry point.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..schema import Schema, MAJOR_SPECIES


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Per-k training result.

    Attributes
    ----------
    k
        Latent dimension trained.
    val_absorption_r2_rate
        R² of absorption derived from the rate head on the val split.
    val_absorption_r2_head
        R² of absorption from the direct distilled head (NaN if not present).
    val_absorption_rel_rmse
        relRMSE of absorption on the val split.
    major_species_rate_r2
        Dict mapping major-species name → R² of mass rate on val split.
    feasibility_baseline_tail_r2
        kNN feasibility tail R² of the full-state baseline.
    feasibility_pca_tail_r2
        kNN feasibility tail R² for PCA-k state.
    feasibility_pls_tail_r2
        kNN feasibility tail R² for PLS-k state.
    feasibility_passes
        Whether PCA-k or PLS-k passes the feasibility gate.
    bundle_dir
        Path where the bundle was written.
    error
        Non-None if training raised an exception.
    """

    k: int
    val_absorption_r2_rate: float = np.nan
    val_absorption_r2_head: float = np.nan
    val_absorption_rel_rmse: float = np.nan
    major_species_rate_r2: dict[str, float] = field(default_factory=dict)
    feasibility_baseline_tail_r2: float = np.nan
    feasibility_pca_tail_r2: float = np.nan
    feasibility_pls_tail_r2: float = np.nan
    feasibility_passes: bool = False
    bundle_dir: str = ""
    error: str | None = None


@dataclass
class AblationReport:
    """Collection of per-k results.

    Attributes
    ----------
    results
        List of :class:`AblationResult`, one per k.
    ks
        Latent dimensions evaluated.
    """

    results: list[AblationResult]
    ks: list[int]

    def to_table(self) -> pd.DataFrame:
        """Return a tidy DataFrame with one row per k."""
        rows = []
        for r in self.results:
            row: dict = {
                "k": r.k,
                "val_absorption_r2_rate": r.val_absorption_r2_rate,
                "val_absorption_r2_head": r.val_absorption_r2_head,
                "val_absorption_rel_rmse": r.val_absorption_rel_rmse,
                "feasibility_baseline_tail_r2": r.feasibility_baseline_tail_r2,
                "feasibility_pca_tail_r2": r.feasibility_pca_tail_r2,
                "feasibility_pls_tail_r2": r.feasibility_pls_tail_r2,
                "feasibility_passes": r.feasibility_passes,
                "bundle_dir": r.bundle_dir,
                "error": r.error or "",
            }
            for sp, r2 in r.major_species_rate_r2.items():
                row[f"r2_{sp}"] = r2
            rows.append(row)
        return pd.DataFrame(rows)

    def accuracy_vs_k(self) -> dict[str, np.ndarray]:
        """Return arrays for plotting (no plotting done here; diagnostics builder uses these).

        Returns a dict with keys:
        - ``"k"``  : latent dimensions.
        - ``"val_r2_rate"`` : val absorption R² (rate path).
        - ``"val_r2_head"`` : val absorption R² (head path; NaN if absent).
        - ``"feasibility_pca_tail_r2"`` : feasibility PCA tail R².
        - ``"feasibility_pls_tail_r2"`` : feasibility PLS tail R².
        """
        ks_arr = np.array([r.k for r in self.results], dtype=float)
        return {
            "k": ks_arr,
            "val_r2_rate": np.array([r.val_absorption_r2_rate for r in self.results]),
            "val_r2_head": np.array([r.val_absorption_r2_head for r in self.results]),
            "feasibility_pca_tail_r2": np.array([r.feasibility_pca_tail_r2 for r in self.results]),
            "feasibility_pls_tail_r2": np.array([r.feasibility_pls_tail_r2 for r in self.results]),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² with degenerate guard."""
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / ss_tot)


def _rel_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """relRMSE = RMSE / std(target)."""
    std = float(np.std(y_true))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) / std


def _evaluate_val_metrics(
    bundle_dir: Path,
    val_df: pd.DataFrame,
    schema: Schema,
) -> tuple[float, float, float, dict[str, float]]:
    """Evaluate a trained bundle on *val_df* and return metrics.

    Returns
    -------
    (val_r2_rate, val_r2_head, val_rel_rmse, major_species_r2)
    """
    import pickle
    import torch

    from ..models.adapter import TorchSurrogate
    from ..models.features import FeatureSpec

    scalers_path = bundle_dir / "scalers.pkl"
    model_path = bundle_dir / "model.pt"
    metrics_path = bundle_dir / "metrics.json"
    spec_path = bundle_dir / "spec.json"

    if not model_path.exists() or not scalers_path.exists():
        return np.nan, np.nan, np.nan, {}

    with open(str(scalers_path), "rb") as f:
        scalers = pickle.load(f)

    import json

    spec_dict: dict = {}
    if spec_path.exists():
        with open(str(spec_path), "r", encoding="utf-8") as f:
            spec_dict = json.load(f)

    target_species = tuple(spec_dict.get("target", scalers.target_species if hasattr(scalers, "target_species") else ()))
    input_species = tuple(spec_dict.get("input", scalers.input_species if hasattr(scalers, "input_species") else ()))
    feat_spec = FeatureSpec(input_species=input_species, target_species=target_species)

    # Try to load absorption metrics from metrics.json
    val_r2_rate = np.nan
    val_r2_head = np.nan
    val_rel_rmse = np.nan
    if metrics_path.exists():
        with open(str(metrics_path), "r", encoding="utf-8") as f:
            mj = json.load(f)
        rate_m = mj.get("absorption_metrics_val", {}).get("rate_derived", {})
        head_m = mj.get("absorption_metrics_val", {}).get("head", {})
        val_r2_rate = float(rate_m.get("r2", np.nan))
        val_r2_head = float(head_m.get("r2", np.nan))
        val_rel_rmse = float(rate_m.get("rel_rmse", np.nan))

    # Per-species R² on major species present in this schema
    major_r2: dict[str, float] = {}
    if len(val_df) == 0 or not target_species:
        return val_r2_rate, val_r2_head, val_rel_rmse, major_r2

    try:
        kind = spec_dict.get("kind", "merged")
        if kind == "merged":
            surrogate = TorchSurrogate.from_merged_bundle(bundle_dir, schema, device="cpu")
        else:
            # Legacy bundles store FeatureScalers directly; reconstruct from config echo.
            from ..models.neuralcoil import NeuralCoil
            state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
            model_cfg = spec_dict.get("config_echo", {}).get("model", {})
            model = NeuralCoil(
                n_dry=len(input_species),
                n_targets=len(target_species),
                latent_dim=int(model_cfg.get("latent_dim", 6)),
            )
            model.load_state_dict(state_dict)
            surrogate = TorchSurrogate(model=model, scalers=scalers, spec=feat_spec, schema=schema, device="cpu")
        pred = surrogate.predict(val_df)
        pred_rates = pred.rates  # (n, n_active)
        pred_absorption = pred.energy  # (n,) — absorption [J/m³/s] for MergedCoil

        # Major species rate R²
        from ..models.features import build_mass_rate_matrix
        true_rates = build_mass_rate_matrix(val_df, schema, list(target_species), prefer_dydt=True)
        major_in_db = schema.major_species()
        for sp in major_in_db:
            if sp in target_species:
                idx = list(target_species).index(sp)
                major_r2[sp] = _r2(true_rates[:, idx], pred_rates[:, idx])

        # Update absorption metrics from live prediction if not in metrics.json
        if schema.has_state("S_reaction_absorption") and np.isnan(val_r2_rate):
            abs_col = schema.energy_target_column()
            true_abs = val_df[abs_col].to_numpy(dtype=float)
            val_r2_rate = _r2(true_abs, pred_absorption)
            val_rel_rmse = _rel_rmse(true_abs, pred_absorption)

    except Exception as exc:
        warnings.warn(f"_evaluate_val_metrics: failed to load/predict model: {exc}")

    return val_r2_rate, val_r2_head, val_rel_rmse, major_r2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_k_ablation(
    base_cfg: "TrainConfig",  # noqa: F821 – imported lazily
    ks: Sequence[int],
    df: pd.DataFrame,
    schema: Schema,
) -> AblationReport:
    """Train one model per k and collect per-k validation metrics.

    Parameters
    ----------
    base_cfg
        Base :class:`~scarfs.training.config.TrainConfig`; a deep copy is made
        for each k (original is not modified).
    ks
        Latent dimensions to evaluate, e.g. ``[4, 6, 8, 12, 16]``.
    df
        Full database DataFrame (will be split internally by the training
        pipeline using ``base_cfg.data.val_fraction`` / ``test_fraction``).
    schema
        Column contract for *df*.

    Returns
    -------
    AblationReport
    """
    # Lazy import of training pipeline to avoid hard torch dependency
    from ..training.train import train as train_fn
    from .feasibility import feasibility_table

    ks = list(ks)
    results: list[AblationResult] = []

    # Compute feasibility table once for all ks
    try:
        feas_report = feasibility_table(df, schema, ks=ks)
    except Exception as exc:
        warnings.warn(f"run_k_ablation: feasibility_table failed: {exc}")
        feas_report = None

    # Persist the dataframe to a temporary parquet file so train() can load it
    # via cfg.data.database_path (train() always loads from disk).
    import tempfile
    import os

    _tmp_db_dir = tempfile.mkdtemp(prefix="scarfs_ablation_")
    _tmp_db_path = os.path.join(_tmp_db_dir, "ablation_db.parquet")
    try:
        df.to_parquet(_tmp_db_path, index=False)
    except Exception as exc:
        warnings.warn(f"run_k_ablation: failed to write temp parquet ({exc}); falling back to CSV")
        _tmp_db_path = _tmp_db_path.replace(".parquet", ".csv")
        df.to_csv(_tmp_db_path, index=False)

    for k in ks:
        cfg = copy.deepcopy(base_cfg)
        cfg.model.latent_dim = int(k)
        cfg.data.database_path = _tmp_db_path
        # Use a per-k output dir to avoid collisions
        cfg.output_dir = str(Path(cfg.output_dir) / f"k{k}")

        try:
            train_fn(cfg)
            bundle_dir = Path(cfg.output_dir)

            # Split off a val frame for live evaluation
            from ..training.datamodule import tripartite_case_split
            val_frac = cfg.data.val_fraction
            test_frac = cfg.data.test_fraction
            seed = cfg.data.seed
            split_by_case = cfg.data.split_by_case
            _, val_mask, _ = tripartite_case_split(df, schema, val_frac, test_frac, seed, split_by_case)
            val_df = df[val_mask].reset_index(drop=True)

            val_r2_rate, val_r2_head, val_rel_rmse, major_r2 = _evaluate_val_metrics(
                bundle_dir, val_df, schema
            )

            # Feasibility for this k
            feas_base = np.nan
            feas_pca = np.nan
            feas_pls = np.nan
            feas_passes = False
            if feas_report is not None:
                feas_base = feas_report.baseline_tail_r2
                for entry in feas_report.entries:
                    if entry.k == k:
                        if entry.space == "pca":
                            feas_pca = entry.tail_r2
                            feas_passes = feas_passes or entry.passes
                        elif entry.space == "pls":
                            feas_pls = entry.tail_r2
                            feas_passes = feas_passes or entry.passes

            results.append(AblationResult(
                k=k,
                val_absorption_r2_rate=val_r2_rate,
                val_absorption_r2_head=val_r2_head,
                val_absorption_rel_rmse=val_rel_rmse,
                major_species_rate_r2=major_r2,
                feasibility_baseline_tail_r2=feas_base,
                feasibility_pca_tail_r2=feas_pca,
                feasibility_pls_tail_r2=feas_pls,
                feasibility_passes=feas_passes,
                bundle_dir=str(bundle_dir),
                error=None,
            ))
        except Exception as exc:
            results.append(AblationResult(k=k, error=str(exc)))

    return AblationReport(results=results, ks=ks)

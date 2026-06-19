"""End-to-end integration test for the merged training path (kind="merged").

Runs the REAL pipeline on the real-data fixture (87 rows / 4 cases sliced from the
colleague's stride6 parquet) with the real mechanism YAML: schema resolution →
thermo-weighted energy-active selection → standardized-linear composition scalers →
mass-rate targets (ρ·dYdt) → MergedCoil with split heads → composite loss incl. the
rate-tied energy term, Lagrangian rollout and atom balance → artifact persistence.

This is the wiring guard: any unit/column/scaler mismatch between the data, models and
training layers fails here before it can reach the HPC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from scarfs.benchmark.loader import infer_schema, load_database
from scarfs.training.config import DataConfig, LossConfig, ModelConfig, OptimConfig, TrainConfig
from scarfs.training.datamodule import tripartite_case_split
from scarfs.training.train import train

FIXTURE = Path(__file__).parent / "data" / "stride6_sample.parquet"
MECH_YAML = Path(__file__).parents[1] / "chem_ForTransport.yaml"


def _non_degenerate_seed(df, schema, val_fraction: float, test_fraction: float) -> int:
    """Pick a deterministic seed whose case split leaves all three folds non-empty.

    With only 4 fixture cases, some seeds hash every case into one fold; the test wants
    train, val AND test populated so the val-metrics path is exercised.
    """
    for seed in range(20):
        tr, va, te = tripartite_case_split(
            df, schema, val_fraction=val_fraction, test_fraction=test_fraction,
            seed=seed, split_by_case=True,
        )
        if tr.any() and va.any() and te.any():
            return seed
    pytest.skip("no seed in range produced a non-degenerate 3-way case split")


def test_train_merged_end_to_end_on_stride6_fixture(tmp_path):
    """The merged path trains on real parquet rows and persists a coherent bundle."""
    # arrange
    df = load_database(FIXTURE)
    schema = infer_schema(df)
    seed = _non_degenerate_seed(df, schema, val_fraction=0.3, test_fraction=0.25)
    cfg = TrainConfig(
        data=DataConfig(
            database_path=str(FIXTURE),
            input_species="dry_all",
            target_species="energy_active",
            val_fraction=0.3,
            test_fraction=0.25,
            split_by_case=True,
            energy_active_coverage=0.999,
            mech_yaml=str(MECH_YAML),
            tail_strata=5,
            tail_weight_alpha=2.0,
            seed=seed,
        ),
        model=ModelConfig(
            kind="merged",
            latent_dim=2,
            decoder_hidden=(16,),
            rate_hidden=(16,),
            latent_source_hidden=(16,),
            energy_hidden=(8,),
            spectral_norm=False,
        ),
        optim=OptimConfig(lr=1e-3, epochs=2, batch_size=256, patience=5),
        loss=LossConfig(
            rollout_mode="lagrangian",
            noise_std=0.01,
            atom_balance_weight=0.1,
        ),
        output_dir=str(tmp_path / "run"),
    )

    # act
    metrics = train(cfg)

    # assert — training ran, artifacts exist, and the spec records the merged contract
    assert metrics["epochs_run"] >= 1
    assert np.isfinite(metrics["best_val"])
    out = tmp_path / "run"
    for artifact in ("model.pt", "scalers.pkl", "spec.json", "metrics.json"):
        assert (out / artifact).exists(), f"missing artifact {artifact}"

    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    assert spec["kind"] == "merged"
    assert spec["composition_mode"] == "standard"
    assert spec["rate_source"] == "dydt_rho"
    assert spec["energy_calibration"]["scale"] > 0.0
    energy_active = spec["energy_active"]
    assert 0 < len(energy_active) < len(spec["input"])
    assert set(energy_active) <= set(schema.species)
    assert "C2H4" in energy_active and "C2H6" in energy_active

    # the composite ran with every merged term active (incl. the rate-tied energy path)
    last_parts = metrics["history"][-1]["train_parts"]
    for term in ("rate", "latent_source", "energy_rate_tied", "energy_distill",
                 "energy_direct", "consistency", "recon", "manifold", "atom_balance",
                 "lagrangian_rollout"):
        assert term in last_parts, f"loss term {term!r} missing: {sorted(last_parts)}"
        assert np.isfinite(last_parts[term]), f"loss term {term!r} not finite"

    # val absorption metrics computed for BOTH energy paths
    abs_val = metrics["absorption_metrics_val"]
    assert set(abs_val) == {"rate_derived", "head"}
    for path_name, m in abs_val.items():
        assert np.isfinite(m["rel_rmse"]), f"{path_name} rel_rmse not finite"

    # held-out test cases recorded for the benchmark layer
    assert len(metrics["split_case_ids"]["test"]) >= 1


def test_train_merged_with_physics_augmentation(tmp_path):
    """The merged path runs end-to-end with the new physics terms enabled (atom-projection,
    Keq equilibrium consistency, realizability) and reports each as a finite loss term."""
    # arrange
    df = load_database(FIXTURE)
    schema = infer_schema(df)
    seed = _non_degenerate_seed(df, schema, val_fraction=0.3, test_fraction=0.25)
    cfg = TrainConfig(
        data=DataConfig(
            database_path=str(FIXTURE),
            input_species="dry_all",
            target_species="energy_active",
            val_fraction=0.3,
            test_fraction=0.25,
            split_by_case=True,
            energy_active_coverage=0.999,
            mech_yaml=str(MECH_YAML),
            seed=seed,
        ),
        model=ModelConfig(
            kind="merged", latent_dim=2,
            decoder_hidden=(16,), rate_hidden=(16,),
            latent_source_hidden=(16,), energy_hidden=(8,),
        ),
        optim=OptimConfig(lr=1e-3, epochs=2, batch_size=256, patience=5),
        loss=LossConfig(
            atom_projection_weight=5e-3,
            keq_weight=1e-2,
            keq_width=1.0,
            realizability_weight=1e-2,
            realizability_dt=1e-3,
        ),
        output_dir=str(tmp_path / "run"),
    )

    # act
    metrics = train(cfg)

    # assert — the new physics terms ran and stayed finite
    assert np.isfinite(metrics["best_val"])
    last_parts = metrics["history"][-1]["train_parts"]
    for term in ("atom_projection", "keq", "realizability"):
        assert term in last_parts, f"{term!r} missing: {sorted(last_parts)}"
        assert np.isfinite(last_parts[term]), f"{term!r} not finite"

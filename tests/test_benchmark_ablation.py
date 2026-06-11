"""Tests for scarfs.benchmark.ablation — k-ablation runner.

The ablation runner calls scarfs.training.train.train() for each k, which requires
PyTorch.  The slow integration test (marked `slow`) exercises this against the
stride6 fixture with epochs=1 and ks=[4, 6].  All other tests cover the public
API surface (dataclasses, to_table, accuracy_vs_k) with synthetic stub data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.benchmark.ablation import AblationResult, AblationReport


# ---------------------------------------------------------------------------
# Unit tests — dataclass API only (no training)
# ---------------------------------------------------------------------------

class TestAblationResult:
    """Minimal unit tests for AblationResult dataclass."""

    def test_default_nans(self):
        """Default numeric fields are NaN."""
        r = AblationResult(k=4)
        assert np.isnan(r.val_absorption_r2_rate)
        assert np.isnan(r.val_absorption_r2_head)
        assert np.isnan(r.val_absorption_rel_rmse)

    def test_error_field(self):
        """error field defaults to None; can be set."""
        r = AblationResult(k=6, error="training failed")
        assert r.error == "training failed"


class TestAblationReport:
    """Unit tests for AblationReport.to_table() and accuracy_vs_k()."""

    def _make_report(self) -> AblationReport:
        results = [
            AblationResult(
                k=4,
                val_absorption_r2_rate=0.88,
                val_absorption_r2_head=0.87,
                val_absorption_rel_rmse=0.18,
                major_species_rate_r2={"CH4": 0.92, "C2H4": 0.85},
                feasibility_pca_tail_r2=0.75,
                feasibility_pls_tail_r2=0.78,
                feasibility_passes=True,
                bundle_dir="/tmp/runs/k4",
            ),
            AblationResult(
                k=6,
                val_absorption_r2_rate=0.91,
                val_absorption_r2_head=np.nan,
                val_absorption_rel_rmse=0.15,
                major_species_rate_r2={"CH4": 0.94, "C2H4": 0.88},
                feasibility_pca_tail_r2=0.82,
                feasibility_pls_tail_r2=0.80,
                feasibility_passes=True,
                bundle_dir="/tmp/runs/k6",
            ),
        ]
        return AblationReport(results=results, ks=[4, 6])

    def test_to_table_shape(self):
        """to_table() has one row per k and standard columns."""
        report = self._make_report()
        tbl = report.to_table()
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) == 2
        assert "k" in tbl.columns
        assert "val_absorption_r2_rate" in tbl.columns
        assert "feasibility_passes" in tbl.columns

    def test_to_table_species_columns(self):
        """Species R² columns appear as r2_<species>."""
        report = self._make_report()
        tbl = report.to_table()
        assert "r2_CH4" in tbl.columns
        assert "r2_C2H4" in tbl.columns

    def test_to_table_values(self):
        """to_table values match the input AblationResult objects."""
        report = self._make_report()
        tbl = report.to_table()
        row_k4 = tbl[tbl["k"] == 4].iloc[0]
        assert row_k4["val_absorption_r2_rate"] == pytest.approx(0.88)
        assert row_k4["feasibility_pca_tail_r2"] == pytest.approx(0.75)

    def test_accuracy_vs_k_keys(self):
        """accuracy_vs_k returns dict with expected keys."""
        report = self._make_report()
        arrays = report.accuracy_vs_k()
        assert "k" in arrays
        assert "val_r2_rate" in arrays
        assert "val_r2_head" in arrays
        assert "feasibility_pca_tail_r2" in arrays
        assert "feasibility_pls_tail_r2" in arrays

    def test_accuracy_vs_k_lengths(self):
        """accuracy_vs_k arrays have same length as number of ks."""
        report = self._make_report()
        arrays = report.accuracy_vs_k()
        assert len(arrays["k"]) == 2
        assert len(arrays["val_r2_rate"]) == 2

    def test_accuracy_vs_k_values(self):
        """k array values match input ks."""
        report = self._make_report()
        arrays = report.accuracy_vs_k()
        np.testing.assert_array_equal(arrays["k"], [4.0, 6.0])

    def test_to_table_error_field(self):
        """AblationResult with error shows empty string in table error column."""
        results = [AblationResult(k=4, error="training failed")]
        report = AblationReport(results=results, ks=[4])
        tbl = report.to_table()
        assert tbl.iloc[0]["error"] == "training failed"

    def test_to_table_no_error_field(self):
        """AblationResult without error shows empty string in table."""
        results = [AblationResult(k=4)]
        report = AblationReport(results=results, ks=[4])
        tbl = report.to_table()
        assert tbl.iloc[0]["error"] == ""

    def test_report_ks_matches(self):
        """AblationReport.ks attribute matches input list."""
        report = self._make_report()
        assert report.ks == [4, 6]


# ---------------------------------------------------------------------------
# Slow integration test: run_k_ablation on stride6 fixture, epochs=1
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_k_ablation_stride6_smoke(tmp_path):
    """Slow: run_k_ablation trains the MERGED model per k on the stride6 fixture.

    The ablation's design target is the merged model — k is selected on energy/tail
    metrics (plan §4 E-e). Runs in the default suite (~20 s; the `slow` marker is
    informational, nothing deselects it by default).
    """
    from pathlib import Path
    fixture = Path(__file__).parent / "data" / "stride6_sample.parquet"
    if not fixture.exists():
        pytest.skip("stride6_sample.parquet not found")

    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("PyTorch not installed")

    import pandas as pd
    from scarfs.benchmark.loader import infer_schema
    from scarfs.benchmark.ablation import run_k_ablation
    from scarfs.training.config import TrainConfig, DataConfig, ModelConfig, OptimConfig, LossConfig
    from scarfs.training.datamodule import tripartite_case_split

    df = pd.read_parquet(str(fixture))
    schema = infer_schema(df)
    mech = Path(__file__).parents[1] / "chem_ForTransport.yaml"

    # pick a deterministic seed with all three case folds non-empty (4 fixture cases)
    seed = next(
        s for s in range(20)
        if all(m.any() for m in tripartite_case_split(
            df, schema, val_fraction=0.3, test_fraction=0.25, seed=s, split_by_case=True))
    )

    cfg = TrainConfig(
        data=DataConfig(
            input_species="dry_all",
            target_species="energy_active",
            val_fraction=0.3,
            test_fraction=0.25,
            mech_yaml=str(mech),
            seed=seed,
            split_by_case=True,
        ),
        model=ModelConfig(
            kind="merged",
            latent_dim=2,
            decoder_hidden=(8,),
            rate_hidden=(8,),
            latent_source_hidden=(8,),
            energy_hidden=(8,),
        ),
        optim=OptimConfig(
            epochs=1,
            batch_size=256,
            patience=10,
        ),
        loss=LossConfig(),
        output_dir=str(tmp_path / "ablation"),
    )

    report = run_k_ablation(cfg, ks=[2, 3], df=df, schema=schema)

    # Verify structural properties — metrics may be NaN for 1-epoch run
    assert isinstance(report, AblationReport)
    assert report.ks == [2, 3]
    assert len(report.results) == 2
    assert report.results[0].k == 2
    assert report.results[1].k == 3
    # No hard crash — error field should be None for successful runs
    for r in report.results:
        assert r.error is None, f"Training failed for k={r.k}: {r.error}"

    # to_table must work without raising
    tbl = report.to_table()
    assert len(tbl) == 2
    assert "k" in tbl.columns

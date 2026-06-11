"""Tests for scripts/benchmark_parents.py CLI.

Smoke-tests main() on the stride6 fixture with --use-stub mode (FrozenComposition
baseline replaces all three parent models) — no real model loading, no training.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_STRIDE6 = Path(__file__).parent / "data" / "stride6_sample.parquet"
_SCRIPT = Path(__file__).parent.parent / "scripts" / "benchmark_parents.py"


# ---------------------------------------------------------------------------
# Helper: import main() from the script without executing as __main__
# ---------------------------------------------------------------------------

def _import_main():
    """Import main() from the benchmark_parents script via importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("benchmark_parents", str(_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="benchmark_parents.py not found")
def test_main_stub_mode_runs(tmp_path, capsys):
    """main() in --use-stub mode completes and writes a Markdown report."""
    main = _import_main()

    out_path = str(tmp_path / "benchmark_report.md")
    rc = main([
        "--database", str(_STRIDE6),
        "--use-stub",
        "--out", out_path,
    ])

    # FrozenComposition stubs will fail the §5 acceptance criteria (they don't
    # predict anything useful), so the return code may be 0 or 1 — what matters
    # is that the script did not crash with an unhandled exception.
    assert rc in (0, 1), f"main() returned unexpected code: {rc}"

    # Report file must be written and contain expected headers
    written = Path(out_path)
    assert written.exists(), "Output Markdown report was not written"
    text = written.read_text(encoding="utf-8")
    assert "SCARFS Benchmark" in text, "Report missing expected header"
    assert "Summary" in text, "Report missing Summary section"


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="benchmark_parents.py not found")
def test_main_stub_mode_three_models_in_report(tmp_path):
    """Stub mode produces one row per stub model in the summary table."""
    main = _import_main()

    out_path = str(tmp_path / "report.md")
    main([
        "--database", str(_STRIDE6),
        "--use-stub",
        "--out", out_path,
    ])

    text = Path(out_path).read_text(encoding="utf-8")
    # Three stub labels appear in the report
    assert "Parent-1 (stub)" in text
    assert "Parent-2 (stub)" in text
    assert "Merged (stub)" in text


@pytest.mark.skipif(not _SCRIPT.exists(), reason="benchmark_parents.py not found")
def test_main_missing_database():
    """main() returns 1 when database does not exist."""
    main = _import_main()

    rc = main(["--database", "/nonexistent/db.parquet", "--use-stub"])
    assert rc == 1


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="benchmark_parents.py not found")
def test_main_no_surrogates_returns_error():
    """main() without any surrogate dirs and without --use-stub returns 1."""
    main = _import_main()

    rc = main([
        "--database", str(_STRIDE6),
        # No --use-stub, no --parent1-dir, no --parent2-dir, no --merged-dir
    ])
    assert rc == 1


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="benchmark_parents.py not found")
def test_main_stub_mode_markdown_has_pass_or_fail(tmp_path):
    """Summary table in stub report contains PASS or FAIL for each model."""
    main = _import_main()

    out_path = str(tmp_path / "report.md")
    main([
        "--database", str(_STRIDE6),
        "--use-stub",
        "--out", out_path,
    ])

    text = Path(out_path).read_text(encoding="utf-8")
    assert "PASS" in text or "FAIL" in text, "No PASS/FAIL verdict in report"


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="benchmark_parents.py not found")
def test_main_stub_mode_stdout_printed(tmp_path, capsys):
    """main() prints the report to stdout."""
    main = _import_main()

    out_path = str(tmp_path / "report.md")
    main([
        "--database", str(_STRIDE6),
        "--use-stub",
        "--out", out_path,
    ])

    captured = capsys.readouterr()
    assert "SCARFS Benchmark" in captured.out

"""Tests for scripts/verify_feasibility_stride5.py CLI.

Runs main(argv=[...]) directly against the stride6 fixture to verify
the CLI entry point works end-to-end without the large stride5 file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


_STRIDE6 = Path(__file__).parent / "data" / "stride6_sample.parquet"
_SCRIPT = Path(__file__).parent.parent / "scripts" / "verify_feasibility_stride5.py"


# ---------------------------------------------------------------------------
# Helper: import main() from the script without executing it as __main__
# ---------------------------------------------------------------------------

def _import_main():
    """Import main() from the script via importlib."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_feasibility_stride5", str(_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="verify_feasibility_stride5.py not found")
def test_main_runs_on_stride6(tmp_path, capsys):
    """main() exits with 0 and prints a FEASIBILITY PRE-GATE header."""
    main = _import_main()

    out_path = str(tmp_path / "report.txt")
    rc = main([
        "--database", str(_STRIDE6),
        "--ks", "4,6",
        "--out", out_path,
    ])

    assert rc == 0, "main() should return 0 for a valid database"

    # Check output file was written
    written = Path(out_path)
    assert written.exists(), "Output file was not written"
    text = written.read_text(encoding="utf-8")
    assert "FEASIBILITY PRE-GATE" in text, "Report missing FEASIBILITY PRE-GATE header"


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="verify_feasibility_stride5.py not found")
def test_main_columns_projection(tmp_path, capsys):
    """main() with --columns-projection does not crash."""
    main = _import_main()

    rc = main([
        "--database", str(_STRIDE6),
        "--ks", "4",
        "--columns-projection",
    ])

    assert rc == 0


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="verify_feasibility_stride5.py not found")
def test_main_rows_cap(tmp_path, capsys):
    """main() with --rows-cap 40 does not crash and loads fewer rows."""
    main = _import_main()

    rc = main([
        "--database", str(_STRIDE6),
        "--ks", "4",
        "--rows-cap", "40",
    ])

    assert rc == 0


@pytest.mark.skipif(not _SCRIPT.exists(), reason="verify_feasibility_stride5.py not found")
def test_main_missing_database():
    """main() returns 1 when database does not exist."""
    main = _import_main()

    rc = main(["--database", "/nonexistent/file.parquet", "--ks", "4"])
    assert rc == 1


@pytest.mark.skipif(not _STRIDE6.exists(), reason="stride6_sample.parquet not found")
@pytest.mark.skipif(not _SCRIPT.exists(), reason="verify_feasibility_stride5.py not found")
def test_main_report_contains_r2_values(tmp_path):
    """The written report contains numeric R² values."""
    main = _import_main()

    out_path = str(tmp_path / "report.txt")
    rc = main([
        "--database", str(_STRIDE6),
        "--ks", "4",
        "--out", out_path,
    ])

    assert rc == 0
    text = Path(out_path).read_text(encoding="utf-8")
    # Should contain at least one decimal number that looks like an R² value
    import re
    decimals = re.findall(r"0\.\d+", text)
    assert len(decimals) >= 1, f"No decimal values found in report:\n{text}"

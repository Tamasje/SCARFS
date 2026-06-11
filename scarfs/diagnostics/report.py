"""Shared markdown/CSV writer helpers for the SCARFS diagnostics package.

All diagnostic writers emit paired markdown + CSV reports.  This module
provides tiny, deterministic helpers (sorted keys) that every writer uses so
the output format is consistent.

Design note (plan §3 / §2):
    The audit machinery was grafted from the colleague's reduced_chem_ml audit
    pattern, reimplemented cleanly over our contract.  The report helpers are
    deliberately thin so that callers control the *content* while this module
    only controls *formatting*.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------


def md_header(title: str, level: int = 1) -> str:
    """Return a markdown section header string."""
    return f"{'#' * level} {title}\n"


def md_kv_block(pairs: dict[str, Any]) -> str:
    """Return a sorted key-value block as markdown list items.

    Keys are sorted for deterministic output.
    """
    lines = [f"- **{k}**: `{v}`" for k, v in sorted(pairs.items())]
    return "\n".join(lines) + "\n"


def md_table(rows: list[dict[str, Any]]) -> str:
    """Format a list of dicts as a markdown table.

    Column order is taken from the sorted union of all keys across rows.
    Values are converted to strings.
    """
    if not rows:
        return ""
    keys = sorted({k for row in rows for k in row})
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = "\n".join(
        "| " + " | ".join(str(row.get(k, "")) for k in keys) + " |"
        for row in rows
    )
    return "\n".join([header, sep, body]) + "\n"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_markdown(out_path: str | os.PathLike, sections: list[tuple[str, str]]) -> None:
    """Write a markdown file from a list of (heading, body) sections.

    Parameters
    ----------
    out_path
        Destination ``.md`` file.
    sections
        List of ``(heading, body)`` tuples.  *heading* may be ``""`` to emit
        body without a heading.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for heading, body in sections:
            if heading:
                fh.write(heading + "\n")
            if body:
                fh.write(body + "\n")


def write_csv(
    out_path: str | os.PathLike,
    rows: list[dict[str, Any]],
) -> None:
    """Write a list of dicts to CSV with sorted column order.

    Parameters
    ----------
    out_path
        Destination ``.csv`` file.
    rows
        List of row dicts; columns are the sorted union of all keys.
    """
    if not rows:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_report_pair(
    out_dir: str | os.PathLike,
    stem: str,
    md_sections: list[tuple[str, str]],
    csv_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    """Write both a ``.md`` and a ``.csv`` report next to each other.

    Parameters
    ----------
    out_dir
        Directory to write into.
    stem
        File name stem (no extension).
    md_sections
        Markdown sections as ``(heading, body)`` pairs.
    csv_rows
        Row dicts for the CSV.

    Returns
    -------
    ``(md_path, csv_path)``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{stem}.md"
    csv_path = out_dir / f"{stem}.csv"
    write_markdown(md_path, md_sections)
    write_csv(csv_path, csv_rows)
    return md_path, csv_path

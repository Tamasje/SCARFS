"""Database loading and schema inference for the SCARFS benchmark harness.

This module provides two public functions:

- :func:`load_database` — reads a ``.csv`` or ``.parquet`` file and returns a
  ``pandas.DataFrame`` with the raw database rows (one row = one axial PFR
  point).
- :func:`infer_schema` — wraps :meth:`scarfs.schema.Schema.from_columns` to
  derive the typed column contract from an already-loaded DataFrame.

Both functions intentionally do *no* filtering or normalisation so that the
benchmark harness operates on the raw data exactly as produced by
``Database_Generation_MB.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from scarfs.schema import Schema


def load_database(path: Union[str, Path]) -> pd.DataFrame:
    """Load a database file (CSV or Parquet) and return the raw DataFrame.

    Parameters
    ----------
    path
        Absolute or relative path to the database file.  Accepts ``.csv``
        (read with :func:`pandas.read_csv`) and ``.parquet`` / ``.pq`` (read
        with :func:`pandas.read_parquet`).  Other extensions raise
        :class:`ValueError`.

    Returns
    -------
    pandas.DataFrame
        Raw rows exactly as stored on disk.  No column renaming, no type
        coercion, no row filtering.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not ``.csv``, ``.parquet``, or ``.pq``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'.  "
            "Expected '.csv', '.parquet', or '.pq'."
        )


def infer_schema(df: pd.DataFrame) -> Schema:
    """Derive a :class:`~scarfs.schema.Schema` from a loaded DataFrame.

    Delegates to :meth:`~scarfs.schema.Schema.from_columns`.  Any column that
    does not match the ``Y_*`` / ``R_*`` prefixes or the known state/meta base
    names is silently ignored (same tolerance as the ``Schema`` itself).

    Parameters
    ----------
    df
        A DataFrame produced by :func:`load_database` (or any frame whose
        columns follow the database convention).

    Returns
    -------
    Schema
        The resolved schema for *df*.

    Raises
    ------
    ValueError
        Propagated from :meth:`Schema.from_columns` if the Y_/R_ species sets
        are inconsistent.
    """
    return Schema.from_columns(list(df.columns))

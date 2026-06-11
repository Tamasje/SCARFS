"""Database loading and schema inference for the SCARFS benchmark harness.

This module provides two public functions:

- :func:`load_database` — reads a ``.csv`` or ``.parquet`` file and returns a
  ``pandas.DataFrame`` with the raw database rows (one row = one axial PFR
  point).  Parquet files are read via ``pyarrow`` for column-projection support.
- :func:`infer_schema` — wraps :meth:`scarfs.schema.Schema.from_columns` to
  derive the typed column contract from an already-loaded DataFrame.

Both functions intentionally do *no* filtering or normalisation so that the
benchmark harness operates on the raw data exactly as produced by
``Database_Generation_MB.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from scarfs.schema import Schema


def load_database(
    path: Union[str, Path],
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load a database file (CSV or Parquet) and return the raw DataFrame.

    Parameters
    ----------
    path
        Absolute or relative path to the database file.  Accepts ``.csv``
        (read with :func:`pandas.read_csv`) and ``.parquet`` / ``.pq`` (read
        via ``pyarrow.parquet``).  Other extensions raise :class:`ValueError`.
    columns
        Optional list of column names to load (column projection).  When
        provided, only the specified columns are read from disk — useful for
        large parquet files where reading 1,100 columns at once is expensive.
        For CSV files, the projection is applied after loading (all columns are
        read from disk; narrow the file on disk if bandwidth matters).
        ``None`` (default) loads all columns.

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
        df = pd.read_csv(path)
        if columns is not None:
            df = df[list(columns)]
        return df
    elif suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq  # preferred: native column projection
            return pq.read_table(path, columns=list(columns) if columns is not None else None).to_pandas()
        except ImportError:
            # Fall back to pandas / fastparquet (no native projection)
            df = pd.read_parquet(path)
            if columns is not None:
                df = df[list(columns)]
            return df
    else:
        raise ValueError(
            f"Unsupported file extension '{path.suffix}'.  "
            "Expected '.csv', '.parquet', or '.pq'."
        )


def infer_schema(df: pd.DataFrame) -> Schema:
    """Derive a :class:`~scarfs.schema.Schema` from a loaded DataFrame.

    Delegates to :meth:`~scarfs.schema.Schema.from_columns`.  Any column that
    does not match the ``Y_*`` / ``R_*`` / ``dYdt_*`` prefixes or the known
    state/meta base names is silently ignored (same tolerance as the
    ``Schema`` itself).  Pseudo-species columns (e.g. ``Y_C2H6_in [-]``) are
    automatically excluded from the returned :attr:`~scarfs.schema.Schema.species`.

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
        are inconsistent and no ``dYdt_`` family is available to satisfy
        coverage.
    """
    return Schema.from_columns(list(df.columns))

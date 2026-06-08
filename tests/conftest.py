"""Shared pytest fixtures: a small synthetic database matching the real column schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scarfs.schema import Schema

#: Species incl. one radical (``CH3.``) and the diluent (``H2O``) to exercise the molecular/radical
#: and diluent-exclusion logic.
_SPECIES = ("C2H6", "C2H4", "H2", "H2O", "CH3.")


def build_synthetic_df(n_cases: int = 2, n_points: int = 6) -> pd.DataFrame:
    """Build a small PFR-like database: per case, C2H6 decreases and C2H4 rises along z."""
    rows: list[dict] = []
    for cid in range(1, n_cases + 1):
        for j in range(n_points):
            frac = j / (n_points - 1)
            comp = {
                "C2H6": 0.70 * (1.0 - 0.4 * frac),
                "C2H4": 0.20 * frac,
                "H2": 0.01 * frac,
                "H2O": 0.30,
                "CH3.": 1e-6 * frac,
            }
            rates = {s: (-0.10 if s == "C2H6" else 0.05) * (1.0 + 0.1 * cid) for s in _SPECIES}
            row: dict[str, float] = {f"Y_{s}": comp[s] for s in _SPECIES}
            row.update({f"R_{s}": rates[s] for s in _SPECIES})
            row.update(
                {
                    "T [K]": 823.15 + 120.0 * frac,
                    "P [Pa]": 2.0e5,
                    "z [m]": round(0.1 * j, 6),
                    "rho [kg/m3]": 0.70 - 0.1 * frac,
                    "Mass flow [kg/s]": 1.6,
                    "CaseID": cid,
                    "L [m]": 0.5,
                    "mdot [kg/s]": 1.6,
                    "U_in [m/s]": 11.0,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    return build_synthetic_df()


@pytest.fixture()
def synthetic_schema(synthetic_df: pd.DataFrame) -> Schema:
    return Schema.from_columns(list(synthetic_df.columns))

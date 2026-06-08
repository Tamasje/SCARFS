"""Canonical column contract for the CRACKSIM/PFR training database.

The database (e.g. ``Database_Validation3.csv`` / ``Database_FINAL.parquet``) is produced by
``Database_Generation_MB.py``. Each **row is one axial point** of a 1-D plug-flow reactor, holding:

- ``Y_<species>``  — species mass fraction [-] (sum over species -> 1).
- ``R_<species>``  — species net production rate [kg m-3 s-1] (the CFD source term).
- a block of thermochemical state scalars (T, P, density, viscosity, ...).
- a block of per-case metadata (CaseID, reactor length, inlet conditions, ...).

This module turns a raw list of column names into a typed :class:`Schema` so every downstream
module (training, benchmark, coupling) shares one source of truth. Resolution is tolerant of the
unit suffix and of the non-ASCII ``·`` in ``mu [Pa·s]`` by matching on the *base* name (the text
before the first ``[``), so minor unit-string drift does not break the contract.

Key modelling facts (see ``DIAGNOSIS.md``):

- The surrogate target is the **rate** ``R_*`` (a source term), not a yield or a state increment —
  hence residence time is *not* a feature (RC-5, refuted).
- Radical species (names ending in ``.``) are transported in the detailed mechanism but are usually
  excluded from the reduced CFD active set; :meth:`Schema.molecular_species` / ``radical_species``
  expose that split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

#: Prefix marking a species mass-fraction column.
Y_PREFIX = "Y_"
#: Prefix marking a species net-production-rate column.
R_PREFIX = "R_"
#: Trailing character marking a radical species (e.g. ``CH3.``, ``H.``).
RADICAL_SUFFIX = "."
#: Default diluent — water vapour is inert in the cracking mechanism and is excluded from the
#: NeuralCoil encoder (see thesis Ch. 5.6: including it produced spurious net rates).
DILUENT_SPECIES = "H2O"

#: Canonical key -> set of accepted *base* names (lower-case, unit-suffix stripped).
#: The base name is ``column.split("[")[0].strip().lower()``.
_STATE_BASE_TO_KEY: dict[str, str] = {
    "t": "T",
    "p": "P",
    "s energy": "S_energy",
    "mass flow": "mass_flow",
    "z": "z",
    "heat input": "heat_input",
    "cp_mass": "cp_mass",
    "cv_mass": "cv_mass",
    "rho": "rho",
    "mu": "mu",
    "k": "k",
    "w_mean": "W_mean",
}

_META_BASE_TO_KEY: dict[str, str] = {
    "caseid": "CaseID",
    "l": "L",
    "h_peak": "H_peak",
    "shape": "shape",
    "shape_params": "shape_params",
    "mdot": "mdot",
    "t_in": "T_in",
    "p_in": "P_in",
    "x_h2o": "X_H2O",
    "re_in": "Re_in",
    "u_in": "U_in",
}

#: Principal steam-cracking species, used to *focus* benchmark tables and figures. Only those
#: actually present in a given database are used; missing names are ignored, never invented.
MAJOR_SPECIES: tuple[str, ...] = (
    "H2", "CH4", "C2H4", "C2H6", "C2H2", "C3H6", "C3H8",
    "C3H4_MA", "C3H4_PD", "__1.3C4H6", "__1C4H8", "IC4H8",
    "NC4H10", "IC4H10", "BENZENE", "TOLUENE", "STYRENE", "CO", "CO2",
)


def column_base(column: str) -> str:
    """Return the matching key of a column: text before ``[``, stripped and lower-cased.

    >>> column_base("mu [Pa·s]")
    'mu'
    >>> column_base("T_in [K]")
    't_in'
    """
    return column.split("[", 1)[0].strip().lower()


def y_column(species: str) -> str:
    """Return the mass-fraction column name for *species* (e.g. ``"C2H4" -> "Y_C2H4"``)."""
    return f"{Y_PREFIX}{species}"


def r_column(species: str) -> str:
    """Return the net-rate column name for *species* (e.g. ``"C2H4" -> "R_C2H4"``)."""
    return f"{R_PREFIX}{species}"


def is_radical(species: str) -> bool:
    """Return ``True`` if *species* is a radical (its name ends in ``.``)."""
    return species.endswith(RADICAL_SUFFIX)


@dataclass(frozen=True)
class Schema:
    """A resolved view of one database's columns.

    Attributes
    ----------
    species
        Species names in database order, derived from the ``Y_`` columns.
    state
        Canonical-key -> actual-column-name for the thermochemical state scalars present.
    meta
        Canonical-key -> actual-column-name for the per-case metadata present.
    """

    species: tuple[str, ...]
    state: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, str] = field(default_factory=dict)

    # -- construction ----------------------------------------------------------------------
    @classmethod
    def from_columns(cls, columns: Sequence[str]) -> "Schema":
        """Build a :class:`Schema` from a sequence of column names.

        The species order is taken from the ``Y_`` columns. ``R_`` columns are validated to cover
        the same species (a mismatch is reported, never silently reconciled).
        """
        cols = list(columns)
        species = tuple(c[len(Y_PREFIX):] for c in cols if c.startswith(Y_PREFIX))
        rate_species = {c[len(R_PREFIX):] for c in cols if c.startswith(R_PREFIX)}
        missing_rates = [s for s in species if s not in rate_species]
        if missing_rates:
            raise ValueError(
                f"Schema: {len(missing_rates)} species have Y_ but no R_ column "
                f"(first few: {missing_rates[:5]}). The database is inconsistent."
            )

        state: dict[str, str] = {}
        meta: dict[str, str] = {}
        for col in cols:
            if col.startswith(Y_PREFIX) or col.startswith(R_PREFIX):
                continue
            base = column_base(col)
            if base in _STATE_BASE_TO_KEY:
                state[_STATE_BASE_TO_KEY[base]] = col
            elif base in _META_BASE_TO_KEY:
                meta[_META_BASE_TO_KEY[base]] = col
        return cls(species=species, state=state, meta=meta)

    # -- species views ---------------------------------------------------------------------
    def molecular_species(self) -> tuple[str, ...]:
        """Species that are NOT radicals (do not end in ``.``)."""
        return tuple(s for s in self.species if not is_radical(s))

    def radical_species(self) -> tuple[str, ...]:
        """Radical species (end in ``.``)."""
        return tuple(s for s in self.species if is_radical(s))

    def active_species(self, exclude_diluent: bool = True) -> tuple[str, ...]:
        """Default CFD-active set: molecular species, optionally minus the diluent.

        The thesis transports ~30 molecular species in CFD (radicals excluded). This returns the
        molecular set as a safe default; narrow it with an explicit list in the training config to
        reproduce the exact 30 if needed.
        """
        species = self.molecular_species()
        if exclude_diluent:
            species = tuple(s for s in species if s != DILUENT_SPECIES)
        return species

    def major_species(self) -> tuple[str, ...]:
        """The principal cracking species (:data:`MAJOR_SPECIES`) present in this database."""
        present = set(self.species)
        return tuple(s for s in MAJOR_SPECIES if s in present)

    # -- column-name helpers ---------------------------------------------------------------
    def y_columns(self, species: Sequence[str] | None = None) -> list[str]:
        """Mass-fraction column names for *species* (all species if ``None``)."""
        return [y_column(s) for s in (species if species is not None else self.species)]

    def r_columns(self, species: Sequence[str] | None = None) -> list[str]:
        """Net-rate column names for *species* (all species if ``None``)."""
        return [r_column(s) for s in (species if species is not None else self.species)]

    def require_state(self, *keys: str) -> list[str]:
        """Return actual column names for the requested state *keys*, or raise if any are absent."""
        missing = [k for k in keys if k not in self.state]
        if missing:
            raise KeyError(f"Schema missing required state columns: {missing}. Present: {sorted(self.state)}")
        return [self.state[k] for k in keys]

    def has_state(self, key: str) -> bool:
        """Return ``True`` if state scalar *key* is present in this database."""
        return key in self.state

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Schema(n_species={len(self.species)}, "
            f"n_molecular={len(self.molecular_species())}, "
            f"n_radical={len(self.radical_species())}, "
            f"state={sorted(self.state)}, meta={sorted(self.meta)})"
        )

"""Canonical column contract for the CRACKSIM/PFR training database.

The database (e.g. ``Database_Validation3.csv`` / ``Database_FINAL.parquet``) is produced by
``Database_Generation_MB.py``. Each **row is one axial point** of a 1-D plug-flow reactor, holding:

- ``Y_<species>``  — species mass fraction [-] (sum over species -> 1).
- ``R_<species>``  — species net production rate [kg m-3 s-1] (CFD source term, legacy CSV convention).
- ``dYdt_<species> [1/s]`` — species mass-fraction rate (parquet databases).
- ``wdot_<species> [kmol/m3/s]`` — molar production rate (parquet databases).
- ``D_<species> [m2/s]`` — species diffusivity (parquet databases).
- a block of thermochemical state scalars (T, P, density, viscosity, ...).
- a block of per-case metadata (CaseID, reactor length, inlet conditions, ...).

This module turns a raw list of column names into a typed :class:`Schema` so every downstream
module (training, benchmark, coupling) shares one source of truth. Resolution is tolerant of the
unit suffix and of the non-ASCII ``·`` in ``mu [Pa·s]`` by matching on the *base* name (the text
before the first ``[``), so minor unit-string drift does not break the contract.

Parquet pseudo-species: ``Y_C2H6_in [-]`` and ``Y_H2O_in [-]`` carry inlet metadata, not species
transport. They carry a ``[`` in the column name and are excluded from :attr:`Schema.species`
automatically by :meth:`Schema.from_columns`.

Energy columns (parquet databases):

- ``Reaction heat absorption [J/s/m3]`` is the **canonical training target** — equal to
  ``Σ h_i · ω̇_i`` (NASA7 enthalpies including formation; positive = endothermic absorption).
  The Fluent source term is ``S_h = −absorption``.
- ``S Energy [J/s/m3]`` is a CRACKSIM-internal bookkeeping term with only ~0.92 correlation to
  ``−absorption``; it must **never** be used as a training target.  It is kept resolvable via
  :attr:`Schema.state` but marked deprecated-for-training in the canonical key ``S_energy``.
  Call :meth:`Schema.energy_target_column` to get the correct column or raise a clear error.

Rate unit convention:

- Legacy CSV (``R_*``): ``kg m-3 s-1`` (mass source term).
- Parquet (``dYdt_*``): ``1/s`` (mass-fraction rate). Use :meth:`Schema.rate_unit_convention`.

Key modelling facts (see ``DIAGNOSIS.md``):

- The surrogate target is the **rate** ``R_*`` or ``dYdt_*``, not a yield or a state increment —
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
#: Prefix marking a species net-production-rate column (legacy CSV, kg m-3 s-1).
R_PREFIX = "R_"
#: Prefix marking a species mass-fraction rate column (parquet, 1/s).
DYDT_PREFIX = "dYdt_"
#: Prefix marking a species molar production rate column (parquet, kmol/m3/s).
WDOT_PREFIX = "wdot_"
#: Prefix marking a species diffusivity column (parquet, m2/s).
D_PREFIX = "D_"

#: Trailing character marking a radical species (e.g. ``CH3.``, ``H.``).
RADICAL_SUFFIX = "."
#: Default diluent — water vapour is inert in the cracking mechanism and is excluded from the
#: NeuralCoil encoder (see thesis Ch. 5.6: including it produced spurious net rates).
DILUENT_SPECIES = "H2O"

#: Pseudo-species present in parquet databases as ``Y_<name> [-]`` columns.  These carry inlet
#: metadata, not transported species — they must be excluded from :attr:`Schema.species`.
PSEUDO_SPECIES: tuple[str, ...] = ("C2H6_in", "H2O_in")

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
    # parquet-specific state columns
    "tau": "tau",
    "u": "u",
    "reaction heat absorption": "S_reaction_absorption",
    "s wall imposed": "S_wall",
    "sum_h_wdot": "sum_h_wdot",
    # additional parquet scalars
    "pfr points solved": "pfr_points_solved",
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
    # parquet meta columns
    "diameter": "diameter",
    "area": "area",
    "steam_to_ethane": "steam_to_ethane",
    "pfr point index": "pfr_point_index",
    "storage stride": "storage_stride",
    "s energy source label": "s_energy_source_label",
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


def dydt_column(species: str, unit_suffix: str = " [1/s]") -> str:
    """Return the mass-fraction-rate column name for *species*.

    Parameters
    ----------
    species
        Species base name (e.g. ``"C2H4"``).
    unit_suffix
        Appended after the species name; must match what the generator emits.
    """
    return f"{DYDT_PREFIX}{species}{unit_suffix}"


def wdot_column(species: str, unit_suffix: str = " [kmol/m3/s]") -> str:
    """Return the molar-rate column name for *species*.

    Parameters
    ----------
    species
        Species base name (e.g. ``"C2H4"``).
    unit_suffix
        Appended after the species name; must match what the generator emits.
    """
    return f"{WDOT_PREFIX}{species}{unit_suffix}"


def is_radical(species: str) -> bool:
    """Return ``True`` if *species* is a radical (its name ends in ``.``)."""
    return species.endswith(RADICAL_SUFFIX)


def _species_from_prefixed_col(col: str, prefix: str) -> str:
    """Extract the species name from a prefixed, unit-suffixed column.

    The convention is ``<prefix><species> [<unit>]`` — species is the text between the prefix
    and the first ``" ["`` (or end-of-string if no unit suffix is present).

    Parameters
    ----------
    col
        Full column name (e.g. ``"dYdt_C2H4 [1/s]"``).
    prefix
        Column prefix to strip (e.g. ``"dYdt_"``).
    """
    inner = col[len(prefix):]
    return inner.split(" [")[0] if " [" in inner else inner


@dataclass(frozen=True)
class Schema:
    """A resolved view of one database's columns.

    Attributes
    ----------
    species
        Species names in database order, derived from the ``Y_`` columns; pseudo-species
        (``Y_*_in [-]``) are excluded.  The count reported in :meth:`__repr__` includes
        the number of excluded pseudo-species.
    state
        Canonical-key -> actual-column-name for the thermochemical state scalars present.
        Includes energy columns if present (``S_reaction_absorption``, ``S_energy``, ``S_wall``).
    meta
        Canonical-key -> actual-column-name for the per-case metadata present.
    rate_families
        Frozenset of rate-family prefixes detected (``"R_"`` and/or ``"dYdt_"``).  Used by
        :meth:`rate_unit_convention` and coverage validation.
    n_pseudo_excluded
        Number of pseudo-species columns excluded from :attr:`species`.
    """

    species: tuple[str, ...]
    state: Mapping[str, str] = field(default_factory=dict)
    meta: Mapping[str, str] = field(default_factory=dict)
    rate_families: frozenset[str] = field(default_factory=frozenset)
    n_pseudo_excluded: int = 0
    #: species -> ACTUAL ``dYdt_`` column name as found in the database. The unit suffix is
    #: not guaranteed (`` [1/s]`` on the parquet, absent elsewhere), so columns are recorded
    #: at resolution time rather than reconstructed by name synthesis.
    dydt_cols_by_species: Mapping[str, str] = field(default_factory=dict)

    # -- construction ----------------------------------------------------------------------
    @classmethod
    def from_columns(
        cls,
        columns: Sequence[str],
        *,
        require_r: bool | None = None,
    ) -> "Schema":
        """Build a :class:`Schema` from a sequence of column names.

        Species order is taken from the ``Y_`` columns, **excluding pseudo-species** (those whose
        column name contains ``[``, e.g. ``Y_C2H6_in [-]``).

        Rate coverage rules
        -------------------
        - If ``require_r`` is ``True``: every species must have an ``R_`` column (legacy strict
          behaviour; raises if any are missing).
        - If ``require_r`` is ``False``: ``R_`` coverage is not validated.
        - If ``require_r`` is ``None`` (default): strict ``R_`` validation is used **only when no
          ``dYdt_`` columns exist**; if ``dYdt_`` columns are present the check is relaxed —
          the union of ``R_``-covered and ``dYdt_``-covered species must equal :attr:`species`
          (raises only if no rate family fully covers the species set).

        Parameters
        ----------
        columns
            Raw column names from the database (CSV header or Parquet schema).
        require_r
            Override the automatic rate-coverage policy (see above).
        """
        cols = list(columns)

        # ── species (Y_ columns, pseudo-species excluded) ──────────────────────────────────
        y_all: list[str] = [c for c in cols if c.startswith(Y_PREFIX)]
        pseudo_excluded = [c for c in y_all if "[" in c]  # pseudo-species carry unit suffix
        y_real = [c for c in y_all if "[" not in c]
        species = tuple(c[len(Y_PREFIX):] for c in y_real)
        n_pseudo = len(pseudo_excluded)

        # ── rate families ──────────────────────────────────────────────────────────────────
        r_species: set[str] = {c[len(R_PREFIX):] for c in cols if c.startswith(R_PREFIX)}
        dydt_cols: list[str] = [c for c in cols if c.startswith(DYDT_PREFIX)]
        dydt_map: dict[str, str] = {_species_from_prefixed_col(c, DYDT_PREFIX): c for c in dydt_cols}
        dydt_species: set[str] = set(dydt_map)

        families: set[str] = set()
        if r_species:
            families.add(R_PREFIX)
        if dydt_species:
            families.add(DYDT_PREFIX)

        # ── rate coverage validation ───────────────────────────────────────────────────────
        has_dydt = bool(dydt_species)
        effective_require_r: bool
        if require_r is None:
            effective_require_r = not has_dydt  # strict only when no dYdt_ family present
        else:
            effective_require_r = require_r

        if effective_require_r:
            missing_rates = [s for s in species if s not in r_species]
            if missing_rates:
                raise ValueError(
                    f"Schema: {len(missing_rates)} species have Y_ but no R_ column "
                    f"(first few: {missing_rates[:5]}). The database is inconsistent."
                )
        elif families:
            # Relaxed: union coverage must cover all species
            covered = r_species | dydt_species
            uncovered = [s for s in species if s not in covered]
            if uncovered:
                raise ValueError(
                    f"Schema: {len(uncovered)} species not covered by any rate family "
                    f"(R_ or dYdt_); first few: {uncovered[:5]}.  "
                    f"R_ covers {len(r_species)}, dYdt_ covers {len(dydt_species)} species."
                )

        # ── state and meta resolution ──────────────────────────────────────────────────────
        skip_prefixes = (Y_PREFIX, R_PREFIX, DYDT_PREFIX, WDOT_PREFIX, D_PREFIX)
        state: dict[str, str] = {}
        meta: dict[str, str] = {}
        for col in cols:
            if any(col.startswith(p) for p in skip_prefixes):
                continue
            base = column_base(col)
            if base in _STATE_BASE_TO_KEY:
                state[_STATE_BASE_TO_KEY[base]] = col
            elif base in _META_BASE_TO_KEY:
                meta[_META_BASE_TO_KEY[base]] = col

        return cls(
            species=species,
            state=state,
            meta=meta,
            rate_families=frozenset(families),
            n_pseudo_excluded=n_pseudo,
            dydt_cols_by_species=dydt_map,
        )

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

    def dydt_columns(self, species: Sequence[str] | None = None) -> list[str]:
        """ACTUAL ``dYdt_`` column names for *species* (all species if ``None``).

        Returns the column names as recorded at :meth:`from_columns` time — the unit suffix
        varies between databases (`` [1/s]`` on the parquet, none on synthetic frames), so
        names are never synthesised here. Returns an empty list if no ``dYdt_`` family is
        present; raises ``KeyError`` for a species the family does not cover.

        Parameters
        ----------
        species
            Subset of :attr:`species` to return columns for; defaults to all.
        """
        if DYDT_PREFIX not in self.rate_families:
            return []
        wanted = species if species is not None else self.species
        missing = [s for s in wanted if s not in self.dydt_cols_by_species]
        if missing:
            raise KeyError(
                f"Schema.dydt_columns: no dYdt_ column for {len(missing)} species "
                f"(first few: {missing[:5]})."
            )
        return [self.dydt_cols_by_species[s] for s in wanted]

    def has_dydt(self) -> bool:
        """Return ``True`` if the ``dYdt_`` rate family is present in this database."""
        return DYDT_PREFIX in self.rate_families

    def rate_unit_convention(self) -> str:
        """Return the rate unit convention of the primary rate family.

        Returns
        -------
        ``"mass_kg_m3_s"``
            Legacy CSV convention: ``R_*`` columns in kg m-3 s-1.
        ``"dydt_per_s"``
            Parquet convention: ``dYdt_*`` columns in 1/s.
        ``"both"``
            Both families present (parquet file that also retains ``R_*``).

        Raises
        ------
        ValueError
            If no rate family was detected (database has no Y_ or rate columns).
        """
        has_r = R_PREFIX in self.rate_families
        has_d = DYDT_PREFIX in self.rate_families
        if has_r and has_d:
            return "both"
        if has_r:
            return "mass_kg_m3_s"
        if has_d:
            return "dydt_per_s"
        raise ValueError(
            "Schema has no rate family (neither R_ nor dYdt_ columns were detected).  "
            "Call from_columns on the actual database column list."
        )

    def energy_target_column(self) -> str:
        """Return the canonical energy training-target column name.

        The canonical target is ``Reaction heat absorption [J/s/m3]`` — equal to
        ``Σ h_i · ω̇_i`` (positive = endothermic; Fluent ``S_h = −absorption``).

        Raises
        ------
        KeyError
            If no ``S_reaction_absorption`` column was found.  This means the database was
            generated without the absorption export flag — use ``S_energy`` only as a fallback
            (with caution; it has only ~0.92 correlation to the true absorption and must not
            be used as a primary target; see module-level docstring).
        """
        if "S_reaction_absorption" in self.state:
            return self.state["S_reaction_absorption"]
        raise KeyError(
            "Schema: 'Reaction heat absorption' column not found in this database.  "
            "The canonical energy training target requires a parquet database generated with "
            "absorption export enabled.  'S Energy' (S_energy) is a CRACKSIM-internal term "
            "with ~0.92 correlation to the true absorption and must not be used as a training "
            "target."
        )

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
        pseudo_note = f", n_pseudo_excluded={self.n_pseudo_excluded}" if self.n_pseudo_excluded else ""
        return (
            f"Schema(n_species={len(self.species)}, "
            f"n_molecular={len(self.molecular_species())}, "
            f"n_radical={len(self.radical_species())}, "
            f"rate_families={sorted(self.rate_families)}, "
            f"state={sorted(self.state)}, meta={sorted(self.meta)}"
            f"{pseudo_note})"
        )

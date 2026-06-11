"""NASA7 thermodynamic property module (NumPy core; optional PyTorch adapter).

Parses a Cantera-format mechanism YAML **without** importing Cantera and exposes
vectorised enthalpy / cp / absorption computations for both NumPy and PyTorch
call sites.  The torch twin is loaded lazily so this module stays importable in
pure-NumPy environments.

NASA7 polynomial (h/RT form):
    h/RT = a0 + a1·T/2 + a2·T²/3 + a3·T³/4 + a4·T⁴/5 + a5/T
    h_molar [J/kmol] = R·T·(h/RT),   R = 8314.462618 J/(kmol·K)

Enthalpies include formation enthalpy (consistent with Cantera's convention and
with ``Reaction heat absorption [J/s/m³]`` = +Σ h_i·ω̇_i in the database).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

# Universal gas constant  [J / (kmol·K)]
R_J_PER_KMOL_K: float = 8314.462618

#: Standard atomic weights  [kg/kmol]
ATOMIC_WEIGHTS: dict[str, float] = {
    "C": 12.0107,
    "H": 1.00794,
    "O": 15.9994,
    "N": 14.0067,
    "S": 32.065,
    "AR": 39.948,
    "HE": 4.0026,
    "CL": 35.453,
    "F": 18.9984,
    "P": 30.9738,
}


class SpeciesThermo:
    """Per-species NASA7 thermochemical data for a requested set of species.

    Attributes
    ----------
    species
        Species names in the order requested.
    molar_mass
        ``(n_species,)`` molar masses [kg/kmol].
    element_names
        Unique element symbols present in this species set.
    element_matrix
        ``(n_species, n_elements)`` atoms of each element per molecule.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        species: tuple[str, ...],
        molar_mass: np.ndarray,
        element_names: tuple[str, ...],
        element_matrix: np.ndarray,
        # NASA7 arrays  (n_species, 7)  low/high range + Tmid + Trange
        _coeffs_low: np.ndarray,
        _coeffs_high: np.ndarray,
        _t_mid: np.ndarray,
        _t_min: np.ndarray,
        _t_max: np.ndarray,
    ) -> None:
        self.species = species
        self.molar_mass = np.asarray(molar_mass, dtype=float)
        self.element_names = element_names
        self.element_matrix = np.asarray(element_matrix, dtype=float)
        self._coeffs_low = np.asarray(_coeffs_low, dtype=float)   # (n, 7)
        self._coeffs_high = np.asarray(_coeffs_high, dtype=float)  # (n, 7)
        self._t_mid = np.asarray(_t_mid, dtype=float)              # (n,)
        self._t_min = np.asarray(_t_min, dtype=float)              # (n,)
        self._t_max = np.asarray(_t_max, dtype=float)              # (n,)
        #: Species requested but absent from the mechanism YAML (``missing_ok=True`` path).
        self.missing: tuple[str, ...] = ()

    @classmethod
    def from_mechanism_yaml(
        cls,
        path: str | Path,
        species: Sequence[str],
        *,
        missing_ok: bool = False,
    ) -> "SpeciesThermo":
        """Parse a Cantera-format YAML mechanism and extract NASA7 data.

        Parameters
        ----------
        path
            Path to the mechanism YAML file (Cantera format, ``species`` key).
        species
            Species names to extract, **in the desired output order**.
        missing_ok
            If ``True``, species absent from the YAML are dropped (order of the
            found ones preserved) and recorded on the returned instance as
            ``.missing`` — the database can name species the transport mechanism
            file does not carry thermo for. If ``False`` (default), any absence
            raises.

        Raises
        ------
        ValueError
            If a requested species is not found in the YAML (``missing_ok=False``),
            or if *no* requested species is found at all.
        """
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        db: dict[str, dict] = {
            str(entry["name"]): entry
            for entry in raw.get("species", [])
            if "name" in entry
        }

        missing = [s for s in species if s not in db]
        if missing and not missing_ok:
            raise ValueError(
                f"SpeciesThermo.from_mechanism_yaml: {len(missing)} species not found "
                f"in {path.name}: {missing}"
            )
        if missing:
            species = [s for s in species if s in db]
            if not species:
                raise ValueError(
                    f"SpeciesThermo.from_mechanism_yaml: none of the requested species "
                    f"exist in {path.name}."
                )

        n = len(species)
        coeffs_low = np.zeros((n, 7), dtype=float)
        coeffs_high = np.zeros((n, 7), dtype=float)
        t_mid = np.zeros(n, dtype=float)
        t_min = np.zeros(n, dtype=float)
        t_max = np.zeros(n, dtype=float)
        compositions: list[dict[str, float]] = []

        for i, name in enumerate(species):
            entry = db[name]
            thermo = entry.get("thermo", {})
            if thermo.get("model") != "NASA7":
                raise ValueError(
                    f"Species {name!r} does not have a NASA7 thermo block (got {thermo.get('model')!r})"
                )
            ranges = thermo["temperature-ranges"]
            data = thermo["data"]
            t_min[i] = float(ranges[0])
            t_max[i] = float(ranges[-1])
            t_mid[i] = float(ranges[1]) if len(ranges) > 2 else t_max[i]
            coeffs_low[i] = np.asarray(data[0], dtype=float)
            # If only one range is given use it for both
            coeffs_high[i] = np.asarray(data[-1], dtype=float)
            comp = {str(k): float(v) for k, v in entry.get("composition", {}).items()}
            compositions.append(comp)

        # Build element set (preserve encounter order, then sort for stability)
        all_elems: list[str] = []
        seen: set[str] = set()
        for comp in compositions:
            for el in comp:
                if el not in seen:
                    all_elems.append(el)
                    seen.add(el)
        element_names = tuple(sorted(all_elems))

        # Molar masses and element matrix
        molar_mass = np.zeros(n, dtype=float)
        element_matrix = np.zeros((n, len(element_names)), dtype=float)
        el_idx = {e: j for j, e in enumerate(element_names)}
        for i, comp in enumerate(compositions):
            for el, cnt in comp.items():
                w = ATOMIC_WEIGHTS.get(el.upper(), ATOMIC_WEIGHTS.get(el, 0.0))
                molar_mass[i] += cnt * w
                j = el_idx.get(el, el_idx.get(el.upper(), -1))
                if j >= 0:
                    element_matrix[i, j] = cnt

        inst = cls(
            species=tuple(species),
            molar_mass=molar_mass,
            element_names=element_names,
            element_matrix=element_matrix,
            _coeffs_low=coeffs_low,
            _coeffs_high=coeffs_high,
            _t_mid=t_mid,
            _t_min=t_min,
            _t_max=t_max,
        )
        inst.missing = tuple(missing)
        return inst

    # ------------------------------------------------------------------
    # Internal NASA7 helpers
    # ------------------------------------------------------------------
    def _select_coeffs(self, T: np.ndarray) -> np.ndarray:
        """Return ``(n_rows, n_species, 7)`` coefficients selected by range."""
        T = T.reshape(-1)                       # (n,)
        use_high = T[:, None] > self._t_mid[None, :]   # (n, n_species)
        # Broadcast: (n, n_species, 7)
        return np.where(
            use_high[:, :, None],
            self._coeffs_high[None, :, :],
            self._coeffs_low[None, :, :],
        )

    # ------------------------------------------------------------------
    # Thermodynamic properties (NumPy)
    # ------------------------------------------------------------------
    def h_molar(self, T: np.ndarray) -> np.ndarray:
        """Molar enthalpies [J/kmol] at temperatures *T*.

        Parameters
        ----------
        T
            ``(n,)`` temperatures [K].

        Returns
        -------
        ``(n, n_species)`` molar enthalpies [J/kmol] (includes formation enthalpy).
        """
        T = np.asarray(T, dtype=float).reshape(-1)
        a = self._select_coeffs(T)              # (n, n_species, 7)
        a0, a1, a2, a3, a4, a5 = [a[:, :, i] for i in range(6)]
        T_ = T[:, None]
        h_rt = (
            a0
            + a1 * T_ / 2.0
            + a2 * T_**2 / 3.0
            + a3 * T_**3 / 4.0
            + a4 * T_**4 / 5.0
            + a5 / np.maximum(T_, 1e-300)
        )
        return R_J_PER_KMOL_K * T_ * h_rt   # (n, n_species)

    def h_mass(self, T: np.ndarray) -> np.ndarray:
        """Mass-specific enthalpies [J/kg] at temperatures *T* → ``(n, n_species)``."""
        return self.h_molar(T) / self.molar_mass[None, :]

    def cp_molar(self, T: np.ndarray) -> np.ndarray:
        """Molar heat capacities [J/(kmol·K)] at temperatures *T* → ``(n, n_species)``."""
        T = np.asarray(T, dtype=float).reshape(-1)
        a = self._select_coeffs(T)
        a0, a1, a2, a3, a4 = [a[:, :, i] for i in range(5)]
        T_ = T[:, None]
        cp_r = a0 + a1 * T_ + a2 * T_**2 + a3 * T_**3 + a4 * T_**4
        return R_J_PER_KMOL_K * cp_r

    def cp_mass(self, T: np.ndarray) -> np.ndarray:
        """Mass-specific heat capacities [J/(kg·K)] → ``(n, n_species)``."""
        return self.cp_molar(T) / self.molar_mass[None, :]

    # ------------------------------------------------------------------
    # Absorption (NumPy)
    # ------------------------------------------------------------------
    def absorption_from_dydt(
        self,
        dydt: np.ndarray,
        rho: np.ndarray,
        T: np.ndarray,
    ) -> np.ndarray:
        """Volumetric heat absorption  Σ ρ·h_mass·(dY/dt)  [J/m³/s].

        Parameters
        ----------
        dydt
            ``(n, n_species)`` species mass-fraction time derivatives [1/s].
        rho
            ``(n,)`` mixture density [kg/m³].
        T
            ``(n,)`` temperature [K].

        Returns
        -------
        ``(n,)`` absorption [J/m³/s].  Positive = endothermic (energy extracted
        from the flow).
        """
        h = self.h_mass(T)                          # (n, n_species)
        dydt = np.asarray(dydt, dtype=float)
        rho_ = np.asarray(rho, dtype=float).reshape(-1, 1)
        return np.sum(rho_ * h * dydt, axis=1)

    # ------------------------------------------------------------------
    # PyTorch twins (lazy import)
    # ------------------------------------------------------------------
    def h_mass_torch(self, T):  # T: torch.Tensor
        """Mass-specific enthalpies [J/kg] as a differentiable PyTorch tensor.

        Parameters
        ----------
        T
            ``(n,)`` or ``(n,1)`` float tensor [K].

        Returns
        -------
        ``(n, n_species)`` tensor.
        """
        import torch

        T_ = T.reshape(-1, 1).float()
        t_mid = torch.as_tensor(self._t_mid, dtype=torch.float32, device=T.device)
        cl = torch.as_tensor(self._coeffs_low, dtype=torch.float32, device=T.device)
        ch = torch.as_tensor(self._coeffs_high, dtype=torch.float32, device=T.device)
        mw = torch.as_tensor(self.molar_mass, dtype=torch.float32, device=T.device)

        use_high = T_ > t_mid.unsqueeze(0)          # (n, n_species)
        a = torch.where(use_high.unsqueeze(-1), ch.unsqueeze(0), cl.unsqueeze(0))
        a0, a1, a2, a3, a4, a5 = a[..., 0], a[..., 1], a[..., 2], a[..., 3], a[..., 4], a[..., 5]
        h_rt = (
            a0
            + a1 * T_ / 2.0
            + a2 * T_**2 / 3.0
            + a3 * T_**3 / 4.0
            + a4 * T_**4 / 5.0
            + a5 / T_.clamp(min=1e-300)
        )
        return (R_J_PER_KMOL_K * T_ * h_rt) / mw.unsqueeze(0)

    def absorption_from_rates_torch(self, rate_mass, T):
        """Differentiable heat absorption  Σ rate_mass_i · h_mass_i.

        Parameters
        ----------
        rate_mass
            ``(n, n_species)`` mass production rates ρ·dY/dt [kg/m³/s].
        T
            ``(n,)`` temperature [K].

        Returns
        -------
        ``(n,)`` absorption [J/m³/s].
        """
        h = self.h_mass_torch(T)         # (n, n_species)
        return (rate_mass.float() * h).sum(dim=-1)

    # ------------------------------------------------------------------
    # Energy-active species selection
    # ------------------------------------------------------------------


def select_energy_active_species(
    dydt: np.ndarray,
    rho: np.ndarray,
    T: np.ndarray,
    species: Sequence[str],
    thermo: "SpeciesThermo",
    *,
    coverage: float = 0.999,
    always_include: Sequence[str] = (),
) -> tuple[str, ...]:
    """Select the minimal species subset that accounts for *coverage* of total absorption.

    Ranks species by their share of ``Σ_rows |ρ·h_mass·dY/dt|``, takes the
    smallest prefix reaching *coverage*, unions with *always_include*, and
    returns names in their original schema order.

    Parameters
    ----------
    dydt
        ``(n, n_species)`` dY/dt [1/s].
    rho
        ``(n,)`` density [kg/m³].
    T
        ``(n,)`` temperature [K].
    species
        Full species list (same column order as *dydt*).
    thermo
        Pre-loaded :class:`SpeciesThermo` for the same species set.
    coverage
        Target fraction of total absolute absorption contribution (default 0.999).
    always_include
        Species that must appear in the result regardless of their rank.

    Returns
    -------
    Tuple of species names, in their original schema order.
    """
    h = thermo.h_mass(T)                            # (n, n_species)
    rho_ = np.asarray(rho, dtype=float).reshape(-1, 1)
    contrib = np.abs(rho_ * h * np.asarray(dydt, dtype=float))  # (n, n_species)
    per_species = contrib.sum(axis=0)                # (n_species,)
    total = per_species.sum()

    if total == 0.0:
        # Degenerate case — return always_include + first species
        forced = set(always_include)
        result = [s for s in species if s in forced]
        if not result:
            result = [species[0]]
        return tuple(result)

    order = np.argsort(per_species)[::-1]            # descending
    cumulative = np.cumsum(per_species[order]) / total
    n_needed = int(np.searchsorted(cumulative, coverage)) + 1

    selected_idx = set(order[:n_needed].tolist())
    species_list = list(species)
    forced = set(always_include)
    for s in forced:
        if s in species_list:
            selected_idx.add(species_list.index(s))

    # Restore schema order
    return tuple(species_list[i] for i in sorted(selected_idx))

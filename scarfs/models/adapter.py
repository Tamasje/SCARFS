"""Adapter wrapping a trained PyTorch surrogate as a :class:`~scarfs.models.common.Surrogate`.

Lets the benchmark harness treat a trained model exactly like a baseline: given a DataFrame of
database rows it returns predicted physical rates (and, when thermo data is supplied, a *derived*
energy source consistent with those rates — F3). PyTorch is imported lazily.

For :class:`~scarfs.models.neuralcoil.MergedCoil` models the ``predict`` method additionally
returns the distilled absorption head output, and can apply
:func:`~scarfs.models.physics.project_conserve_atoms` when *conserve_atoms* is ``True`` and thermo
data is provided.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..schema import Schema
from .common import SurrogatePrediction
from .features import FeatureScalers, FeatureSpec, build_features, invert_rate_targets
from .physics import derive_energy_source, project_conserve_atoms


class TorchSurrogate:
    """Wrap a trained model + its scalers so it satisfies the ``Surrogate`` protocol.

    Supports both :class:`~scarfs.models.neuralcoil.NeuralCoil` / ``ReducedSurrogate``
    (original behaviour, unchanged) and :class:`~scarfs.models.neuralcoil.MergedCoil`
    (new split-head behaviour enabled automatically when the model has an
    ``absorption`` method).

    Parameters
    ----------
    model
        Trained PyTorch module.
    scalers
        Fitted :class:`FeatureScalers` (composition / thermo / rate).
    spec
        :class:`FeatureSpec` describing input / target species.
    schema
        Column contract.
    device
        PyTorch device string.
    molar_mass
        ``(n_active,)`` molar masses [kg/kmol] for derived energy.
    molar_enthalpy_fn
        Callable ``df -> (n, n_active)`` molar enthalpies; used by legacy path.
    conserve_atoms
        If ``True`` **and** *molar_mass* + *element_matrix* are provided, apply
        :func:`~scarfs.models.physics.project_conserve_atoms` to the rate
        predictions before returning (only for MergedCoil / rate-head output).
    element_matrix
        ``(n_active, n_elements)`` element matrix; required when *conserve_atoms*
        is ``True``.
    """

    def __init__(
        self,
        model,
        scalers: FeatureScalers,
        spec: FeatureSpec,
        schema: Schema,
        *,
        device: str = "cpu",
        molar_mass: np.ndarray | None = None,
        molar_enthalpy_fn: Callable[[pd.DataFrame], np.ndarray] | None = None,
        conserve_atoms: bool = False,
        element_matrix: np.ndarray | None = None,
    ) -> None:
        self.model = model
        self.scalers = scalers
        self.spec = spec
        self.schema = schema
        self.device = device
        self.active_species = spec.target_species
        self._molar_mass = molar_mass
        self._enthalpy_fn = molar_enthalpy_fn
        self._conserve_atoms = conserve_atoms
        self._element_matrix = element_matrix
        self._n_input = len(spec.input_species)

    def _is_merged_coil(self) -> bool:
        """Return True if the wrapped model is a MergedCoil (has the absorption head)."""
        return hasattr(self.model, "absorption") and callable(getattr(self.model, "absorption", None))

    def _predict_scaled(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """Return (scaled_rates, absorption_or_None)."""
        import torch

        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            y_in, q = xt[:, : self._n_input], xt[:, self._n_input :]
            if self._is_merged_coil():
                out = self.model(y_in, q)
                return out["rates"].detach().cpu().numpy(), out["absorption"].detach().cpu().numpy()
            if hasattr(self.model, "rates_from_latent"):  # NeuralCoil
                out = self.model(y_in, q)["rates"]
            else:  # ReducedSurrogate
                out = self.model(xt)
            return out.detach().cpu().numpy(), None

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict physical rates [kg m-3 s-1] and energy [J m-3 s-1].

        For :class:`~scarfs.models.neuralcoil.MergedCoil` models the energy is taken
        directly from the distilled absorption head (strictly positive by construction).
        For other models the energy is derived from the rates via
        :func:`~scarfs.models.physics.derive_energy_source` when thermo data is
        available, or set to NaN otherwise.

        When *conserve_atoms* is ``True`` and both *molar_mass* and *element_matrix*
        are set, the rate predictions are projected onto the atom-conserving
        null-space before returning.
        """
        X = build_features(df, self.schema, self.scalers)
        scaled_rates, absorption = self._predict_scaled(X)
        rates = invert_rate_targets(scaled_rates, self.scalers)

        # Optional atom-conservation projection
        if (
            self._conserve_atoms
            and self._molar_mass is not None
            and self._element_matrix is not None
        ):
            rates = project_conserve_atoms(rates, self._molar_mass, self._element_matrix)

        # Energy
        if absorption is not None:
            # MergedCoil: use distilled head (already in J/m³/s, positive)
            energy = absorption
        elif self._molar_mass is not None and self._enthalpy_fn is not None:
            energy = derive_energy_source(rates, self._molar_mass, self._enthalpy_fn(df))
        else:
            energy = np.full(len(df), np.nan, dtype=float)

        return SurrogatePrediction(species=self.active_species, rates=rates, energy=energy)

"""Adapter wrapping a trained PyTorch surrogate as a :class:`~scarfs.models.common.Surrogate`.

Lets the benchmark harness treat a trained model exactly like a baseline: given a DataFrame of
database rows it returns predicted physical rates (and, when thermo data is supplied, a *derived*
energy source consistent with those rates — F3). PyTorch is imported lazily.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..schema import Schema
from .common import SurrogatePrediction
from .features import FeatureScalers, FeatureSpec, build_features, invert_rate_targets
from .physics import derive_energy_source


class TorchSurrogate:
    """Wrap a trained model + its scalers so it satisfies the ``Surrogate`` protocol."""

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
    ) -> None:
        self.model = model
        self.scalers = scalers
        self.spec = spec
        self.schema = schema
        self.device = device
        self.active_species = spec.target_species
        self._molar_mass = molar_mass
        self._enthalpy_fn = molar_enthalpy_fn
        self._n_input = len(spec.input_species)

    def _predict_scaled(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.model.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            if hasattr(self.model, "rates_from_latent"):  # NeuralCoil: split [dry | thermo]
                y_dry, q = xt[:, : self._n_input], xt[:, self._n_input :]
                out = self.model(y_dry, q)["rates"]
            else:  # ReducedSurrogate
                out = self.model(xt)
        return out.detach().cpu().numpy()

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict physical rates [kg m-3 s-1] (and derived energy if thermo data is available)."""
        X = build_features(df, self.schema, self.scalers)
        rates = invert_rate_targets(self._predict_scaled(X), self.scalers)
        if self._molar_mass is not None and self._enthalpy_fn is not None:
            energy = derive_energy_source(rates, self._molar_mass, self._enthalpy_fn(df))
        else:
            energy = np.full(len(df), np.nan, dtype=float)
        return SurrogatePrediction(species=self.active_species, rates=rates, energy=energy)

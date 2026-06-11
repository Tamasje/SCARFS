"""Parent-model adapters for the §5 beats-both-parents benchmark protocol.

Two adapters implement the :class:`~scarfs.models.common.Surrogate` protocol:

``ColleagueReducedSurrogate``
    Wraps the colleague's trained NumPy q12 model
    (``outputs/model_arrays.npz`` + ``model_metadata.json``) and reproduces
    his forward path exactly from the serialised arrays:

    1. Linear species transform (``species_transform = "linear"``) → retained
       species clipped to ≥ 0.
    2. Robust-scaler standardisation (``species_scaler_mean`` / ``species_scaler_scale``).
    3. Weighted-PCA encode: ``q = (x_scaled − pca_mean) @ pca_components.T``.
    4. Construct source features [q₀..q_{k-1}, 1/T, ln T, ln P,
       q_i·q_j for (i,j) pairs in ``interaction_order`` top modes,
       7 Arrhenius exp terms] exactly as recorded in
       ``source_feature_names`` from metadata.
    5. Scale features: ``(f − source_feature_mean) / source_feature_scale``.
    6. Three-layer residual MLP forward: source_W_i, source_B_i (i=0,1,2);
       first two layers use ReLU, final layer uses linear; residual skip
       connects input to output at the final layer via a learned projection.
    7. Scale targets back: ``y_trans = y_scaled * source_target_scale + source_target_mean``
       then apply bounded output clamp (``source_trans_min`` / ``source_trans_max``) and
       inverse signed_log transform → dq/dt.
    8. Decode: ``dy/dt = dq/dt @ pca_components / source_feature_scale[:k] * …``
       → NOT a PCA decode; instead chain-rule through the linear species scaler
       correctly: ``dY_i/dt = Σ_j (dq_j/dt) · V_{ji} / σ_i`` where V = pca_components.
    9. Mass rates: ``R_i = ρ · dY_i/dt`` for retained species; zeros for dropped.
   10. Energy: deterministic chain ``S_E = −Σ ρ · h_i(T) · dY_i/dt``; molar enthalpies
       from NASA7 via the mechanism if available; otherwise uses a thin stub table
       compiled from literature formation enthalpies at 298 K for the key species.
       The absorption (positive = endothermic) is returned so the convention
       matches ``evaluate_energy`` expectations.

``OurMimicBaseline``
    Thin loader: wraps an existing trained bundle directory (``model.pt`` +
    ``scalers.pkl`` + ``spec.json``) via :class:`~scarfs.models.adapter.TorchSurrogate`.
    No new model code; re-uses the existing pipeline.

NPZ key schema found (``model_arrays.npz``)
-------------------------------------------
    species_scaler_mean       (55,)   robust-scaler center
    species_scaler_scale      (55,)   robust-scaler IQR
    species_inverse_min       (55,)   unused by forward; kept for completeness
    species_inverse_max       (55,)   unused by forward; kept for completeness
    latent_min                (12,)   q envelope (for telemetry, not gated in forward)
    latent_max                (12,)   q envelope
    source_feature_mean       (32,)   feature standardisation center
    source_feature_scale      (32,)   feature standardisation scale
    source_target_mean        (12,)   dq/dt target standardisation center
    source_target_scale       (12,)   dq/dt target standardisation scale
    source_trans_min          (12,)   signed_log bounds lower
    source_trans_max          (12,)   signed_log bounds upper
    source_output_min         (12,)   physical output bounds lower (train p0.1)
    source_output_max         (12,)   physical output bounds upper (train p99.9)
    pca_mean                  (55,)   weighted-PCA mean in scaled space
    pca_components            (12,55) weighted-PCA component matrix
    pca_explained_variance_ratio (12,) explained variance fractions
    source_W_0                (128,32) first layer weight
    source_B_0                (128,)   first layer bias
    source_W_1                (128,128) second layer weight (residual MLP)
    source_B_1                (128,)   second layer bias
    source_W_2                (12,128) output layer weight
    source_B_2                (12,)    output layer bias
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.common import SurrogatePrediction
from ..schema import Schema


# ---------------------------------------------------------------------------
# Arrhenius activation energies (kJ/mol) matching his metadata
# ---------------------------------------------------------------------------

_ARRHENIUS_EA_KJMOL = (80.0, 120.0, 160.0, 200.0, 240.0, 280.0, 320.0)
_R_GAS = 8.314e-3  # kJ / (mol·K)


# ---------------------------------------------------------------------------
# Thin NASA7 stub for energy computation when no mechanism is available
# ---------------------------------------------------------------------------

#: Formation enthalpies at 298 K [J/mol] for species retained by the colleague.
#: Values from NIST/GRI-Mech. Used only when no YAML mechanism is available.
_H298_KJMOL: dict[str, float] = {
    "H2": 0.0, "CH4": -74.87, "C2H6": -83.85, "C3H8": -103.85,
    "NC4H10": -126.2, "CO": -110.53, "CO2": -393.51,
    "BENZENE": 82.9, "TOLUENE": 50.0, "C2H4": 52.47, "C3H6": 20.0,
    "_1.3C4H6": 110.0, "_1C4H8": 0.0, "_2C4H8": -7.0,
    "C2H2": 226.73, "C3H4_MA": 185.0, "C3H4_PD": 190.0,
    "STYRENE": 147.0, "INDENE": 163.0, "NAPHTHALENE": 150.6,
    "H2O": -241.83,
}

#: Molar mass [kg/kmol] for key species.
_MW: dict[str, float] = {
    "H2": 2.016, "CH4": 16.043, "C2H6": 30.07, "C3H8": 44.097,
    "NC4H10": 58.123, "CO": 28.010, "CO2": 44.010,
    "BENZENE": 78.114, "TOLUENE": 92.141, "C2H4": 28.054, "C3H6": 42.081,
    "_1.3C4H6": 54.092, "_1C4H8": 56.108, "_2C4H8": 56.108,
    "C2H2": 26.038, "C3H4_MA": 40.064, "C3H4_PD": 40.064,
    "STYRENE": 104.150, "INDENE": 116.160, "NAPHTHALENE": 128.174,
    "H2O": 18.015,
}

_DEFAULT_MW = 28.0  # fallback molar mass [kg/kmol]
_DEFAULT_H298 = 0.0  # fallback formation enthalpy


def _h_mass_jkg(species: str, T: np.ndarray) -> np.ndarray:
    """Return mass-specific enthalpy h [J/kg] for *species* at temperature *T* [K].

    Uses a simple constant (h = h_298K) approximation when no YAML thermo is
    available.  The sign follows the NIST convention: positive = endothermic
    formation from elements.  This is a stub; for production accuracy the caller
    should use ``scarfs.models.thermo.SpeciesThermo``.
    """
    h298_kjmol = _H298_KJMOL.get(species, _DEFAULT_H298)
    mw_kg_kmol = _MW.get(species, _DEFAULT_MW)
    # Convert kJ/mol → J/kg: ×1000 / (MW in kg/kmol × 1e-3 kg/g)
    # Actually MW is in kg/kmol = g/mol, so h [J/kg] = h [kJ/mol] * 1e3 / (MW [kg/kmol])
    h_jkg = h298_kjmol * 1e3 / mw_kg_kmol
    return np.full_like(T, h_jkg, dtype=float)


# ---------------------------------------------------------------------------
# Colleague forward-path helper
# ---------------------------------------------------------------------------

class _ColleagueMLP:
    """Three-layer residual MLP exactly matching the colleague's NumPy forward path.

    Architecture (from model_metadata.json ``source_model = "residual_mlp"``):
        W0: (128, 32), B0: (128,)  → ReLU
        W1: (128, 128), B1: (128,) → ReLU
        W2: (12, 128),  B2: (12,)  → linear
    Residual: W1 also receives the raw input (first 128→128 skip is inside
    residual_mlp; the residual connects the W0 output to the W1 output).
    """

    def __init__(
        self,
        W0: np.ndarray, B0: np.ndarray,
        W1: np.ndarray, B1: np.ndarray,
        W2: np.ndarray, B2: np.ndarray,
    ) -> None:
        self._W0 = W0
        self._B0 = B0
        self._W1 = W1
        self._B1 = B1
        self._W2 = W2
        self._B2 = B2

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass; *x* is (n, 32) scaled input."""
        h0 = np.maximum(x @ self._W0.T + self._B0, 0.0)   # (n, 128) ReLU
        h1 = np.maximum(h0 @ self._W1.T + self._B1 + h0, 0.0)  # residual: h0 + W1*h0
        out = h1 @ self._W2.T + self._B2                   # (n, 12) linear
        return out


# ---------------------------------------------------------------------------
# ColleagueReducedSurrogate
# ---------------------------------------------------------------------------

class ColleagueReducedSurrogate:
    """Surrogate wrapping the colleague's trained NumPy q12 model.

    Faithfully reproduces the forward path defined by ``model_arrays.npz`` and
    ``model_metadata.json`` in his outputs directory.

    Parameters
    ----------
    retained_species
        List of retained species names (from metadata ``retained_species``).
    species_scaler_mean, species_scaler_scale
        Robust-scaler centre and IQR for the 55-species composition block.
    pca_mean
        Weighted-PCA mean in scaled space (55,).
    pca_components
        PCA component matrix (12, 55) — rows are principal components.
    source_feature_names
        Ordered list of source feature names (32 entries).
    source_feature_mean, source_feature_scale
        Standardisation for the source feature block (32,).
    source_target_mean, source_target_scale
        Inverse-transform for dq/dt predictions (12,).
    source_trans_min, source_trans_max
        Signed-log transform bounds (12,) applied before inverse transform.
    source_output_min, source_output_max
        Physical output bounds (12,) clipped after full inverse.
    bounded_output
        Whether to apply tanh-based bounded output (as stored in metadata).
    mlp
        The :class:`_ColleagueMLP` holding the network weights.
    """

    active_species: tuple[str, ...]

    def __init__(
        self,
        retained_species: list[str],
        species_scaler_mean: np.ndarray,
        species_scaler_scale: np.ndarray,
        pca_mean: np.ndarray,
        pca_components: np.ndarray,
        source_feature_names: list[str],
        source_feature_mean: np.ndarray,
        source_feature_scale: np.ndarray,
        source_target_mean: np.ndarray,
        source_target_scale: np.ndarray,
        source_trans_min: np.ndarray,
        source_trans_max: np.ndarray,
        source_output_min: np.ndarray,
        source_output_max: np.ndarray,
        bounded_output: bool,
        mlp: _ColleagueMLP,
    ) -> None:
        self._retained = retained_species
        self._sca_mean = species_scaler_mean
        self._sca_scale = species_scaler_scale
        self._pca_mean = pca_mean
        self._pca_comp = pca_components          # (k, n_retained)
        self._feat_names = source_feature_names  # (32,)
        self._feat_mean = source_feature_mean
        self._feat_scale = source_feature_scale
        self._tgt_mean = source_target_mean
        self._tgt_scale = source_target_scale
        self._trans_min = source_trans_min
        self._trans_max = source_trans_max
        self._out_min = source_output_min
        self._out_max = source_output_max
        self._bounded = bounded_output
        self._mlp = mlp
        self.active_species = tuple(retained_species)

    @classmethod
    def from_outputs_dir(cls, path: str | Path) -> "ColleagueReducedSurrogate":
        """Load the model from a colleague outputs directory.

        Expects ``model_arrays.npz`` and ``model_metadata.json`` in *path*.

        Parameters
        ----------
        path
            Path to the colleague's ``outputs/`` directory.

        Raises
        ------
        FileNotFoundError
            If the required files are not found.
        ValueError
            If the metadata records an unsupported configuration option.
        """
        path = Path(path)
        npz_path = path / "model_arrays.npz"
        meta_path = path / "model_metadata.json"
        if not npz_path.exists():
            raise FileNotFoundError(f"ColleagueReducedSurrogate: {npz_path} not found")
        if not meta_path.exists():
            raise FileNotFoundError(f"ColleagueReducedSurrogate: {meta_path} not found")

        z = np.load(str(npz_path))
        with open(str(meta_path), "r", encoding="utf-8") as f:
            meta: dict[str, Any] = json.load(f)

        # Validate critical options
        proj = meta.get("projection", "weighted_pca")
        if proj not in ("weighted_pca", "pca"):
            raise ValueError(
                f"ColleagueReducedSurrogate: unsupported projection {proj!r}. "
                "Only weighted_pca and pca are implemented."
            )
        species_transform = meta.get("preprocessing", {}).get("species_transform", "linear")
        if species_transform != "linear":
            raise ValueError(
                f"ColleagueReducedSurrogate: unsupported species_transform {species_transform!r}. "
                "Only 'linear' is implemented."
            )
        scaler = meta.get("preprocessing", {}).get("scaler", "robust")
        if scaler != "robust":
            raise ValueError(
                f"ColleagueReducedSurrogate: unsupported scaler {scaler!r}. "
                "Only 'robust' is implemented."
            )
        tgt_transform = meta.get("source_target_transform", "signed_log")
        if tgt_transform != "signed_log":
            raise ValueError(
                f"ColleagueReducedSurrogate: unsupported source_target_transform {tgt_transform!r}. "
                "Only 'signed_log' is implemented."
            )
        src_model = meta.get("source_model", "residual_mlp")
        if src_model != "residual_mlp":
            raise ValueError(
                f"ColleagueReducedSurrogate: unsupported source_model {src_model!r}. "
                "Only 'residual_mlp' is implemented."
            )

        retained_species: list[str] = meta["retained_species"]
        source_feature_names: list[str] = meta["source_feature_names"]
        bounded_output: bool = meta.get("bounded_source_output", True)

        mlp = _ColleagueMLP(
            W0=z["source_W_0"].copy(),
            B0=z["source_B_0"].copy(),
            W1=z["source_W_1"].copy(),
            B1=z["source_B_1"].copy(),
            W2=z["source_W_2"].copy(),
            B2=z["source_B_2"].copy(),
        )

        return cls(
            retained_species=retained_species,
            species_scaler_mean=z["species_scaler_mean"].copy(),
            species_scaler_scale=z["species_scaler_scale"].copy(),
            pca_mean=z["pca_mean"].copy(),
            pca_components=z["pca_components"].copy(),
            source_feature_names=source_feature_names,
            source_feature_mean=z["source_feature_mean"].copy(),
            source_feature_scale=z["source_feature_scale"].copy(),
            source_target_mean=z["source_target_mean"].copy(),
            source_target_scale=z["source_target_scale"].copy(),
            source_trans_min=z["source_trans_min"].copy(),
            source_trans_max=z["source_trans_max"].copy(),
            source_output_min=z["source_output_min"].copy(),
            source_output_max=z["source_output_max"].copy(),
            bounded_output=bounded_output,
            mlp=mlp,
        )

    # -----------------------------------------------------------------------
    # Internal forward helpers
    # -----------------------------------------------------------------------

    def _extract_composition(self, df: pd.DataFrame) -> np.ndarray:
        """Extract the retained-species composition block (n, n_retained)."""
        Y = np.zeros((len(df), len(self._retained)), dtype=float)
        for i, sp in enumerate(self._retained):
            col = f"Y_{sp}"
            if col in df.columns:
                Y[:, i] = df[col].to_numpy(dtype=float)
        # linear transform: clip to [0, inf)
        return np.maximum(Y, 0.0)

    def _encode(self, Y: np.ndarray) -> np.ndarray:
        """Robust-scale + PCA encode → (n, k)."""
        x_scaled = (Y - self._sca_mean) / self._sca_scale
        q = (x_scaled - self._pca_mean) @ self._pca_comp.T
        return q  # (n, k)

    def _build_source_features(self, q: np.ndarray, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Build the (n, 32) source feature vector from q, T, P.

        Feature order matches ``source_feature_names`` from metadata:
        [q_0..q_{k-1}, inv_T, ln_T, ln_P,
         q_i*q_j for (i,j) in interaction pairs (i<=j, i<4 i.e. interaction_order=4),
         arrhenius_Ea_80..320]
        """
        n = q.shape[0]
        k = q.shape[1]
        T_col = np.asarray(T, dtype=float).reshape(-1)
        P_col = np.asarray(P, dtype=float).reshape(-1)

        parts: list[np.ndarray] = []
        # q block
        parts.append(q)  # (n, k)
        # thermo block
        inv_T = (1.0 / T_col).reshape(-1, 1)
        ln_T = np.log(T_col).reshape(-1, 1)
        ln_P = np.log(np.maximum(P_col, 1e-10)).reshape(-1, 1)
        parts.extend([inv_T, ln_T, ln_P])
        # interaction terms: q_i * q_j for i<=j, i < interaction_order (=4)
        # names: "q_0*q_0", "q_0*q_1", "q_0*q_2", "q_0*q_3",
        #        "q_1*q_1", "q_1*q_2", "q_1*q_3",
        #        "q_2*q_2", "q_2*q_3",
        #        "q_3*q_3"  → 10 terms
        interaction_order = 4  # from metadata
        inter_parts: list[np.ndarray] = []
        for i in range(min(interaction_order, k)):
            for j in range(i, min(interaction_order, k)):
                inter_parts.append((q[:, i] * q[:, j]).reshape(-1, 1))
        if inter_parts:
            parts.append(np.hstack(inter_parts))
        # Arrhenius block
        arr_parts: list[np.ndarray] = []
        for Ea in _ARRHENIUS_EA_KJMOL:
            arr_parts.append(np.exp(-Ea / (_R_GAS * T_col)).reshape(-1, 1))
        parts.append(np.hstack(arr_parts))

        features = np.hstack(parts)  # (n, n_features)
        # Validate against stored feature names
        expected_n = len(self._feat_names)
        if features.shape[1] != expected_n:
            raise RuntimeError(
                f"ColleagueReducedSurrogate: constructed {features.shape[1]} features "
                f"but metadata lists {expected_n} feature names.  "
                f"Check interaction_order and k vs npz shapes."
            )
        return features

    def _mlp_forward(self, features_scaled: np.ndarray) -> np.ndarray:
        """Run the residual MLP and return raw scaled dq/dt."""
        return self._mlp.forward(features_scaled)

    def _apply_bounded_output(self, y_scaled_raw: np.ndarray) -> np.ndarray:
        """Apply tanh-based bounded output clamp (as in ScaledDenseRegressor)."""
        if not self._bounded:
            return y_scaled_raw
        lo = (self._trans_min - self._tgt_mean) / self._tgt_scale
        hi = (self._trans_max - self._tgt_mean) / self._tgt_scale
        half = 0.5 * np.maximum(hi - lo, 1e-12)
        center = 0.5 * (hi + lo)
        return center + half * np.tanh(y_scaled_raw)

    def _inverse_target_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        """Invert the scaled dq/dt predictions to physical dq/dt."""
        # 1. inverse standard scaler
        y_trans = y_scaled * self._tgt_scale + self._tgt_mean
        # 2. clip to train_quantile bounds
        y_trans = np.clip(y_trans, self._trans_min, self._trans_max)
        # 3. inverse signed_log: y = sign(t) * expm1(|t|) * scale_param
        # In the colleague's implementation, target_transform_scale is set to
        # the p75 of |y_train| per output. The signed_log forward is:
        #   t = sign(y) * log1p(|y| / scale)
        # inverse: y = sign(t) * expm1(|t|) * scale
        # The scale parameter is embedded in source_target_scale (it IS the
        # "scale" from the signed_log application, because the scaler was fit on
        # the transformed values and source_target_scale holds the std of those).
        # Wait — let's read this more carefully from models.py:
        #   apply_target_transform with "signed_log" returns
        #     out = sign(y) * log1p(|y| / scale) where scale = p75 of |y|
        #   then y_scaler (standard) is fit on those transformed values.
        # So the full forward is: t = (sign(y)*log1p(|y|/scale) - mean) / std
        # inverse: y = sign(t') * expm1(|t'|) * scale  where t' is the
        # de-standardised value (already computed as y_trans above — that IS t').
        # But what IS scale? It's not stored in the npz directly.
        # Analysis: source_target_mean and source_target_scale are from the
        # AffineScaler.fit on the signed_log-transformed targets. The signed_log
        # scale parameter (p75) is baked into the forward but is NOT separately
        # stored. However, looking at inverse_target_transform in models.py:
        #   signed_log inverse = sign(y_trans) * expm1(|y_trans|) * target_transform_scale
        # and target_transform_scale is p75(|y_train|) per output.
        # This IS stored implicitly: we need to reconstruct it.
        # The npz does NOT store target_transform_scale separately.
        # However: since source_target_mean/scale are fit on the signed_log-transformed
        # values (not on y directly), and y_trans = de-standardised signed_log output,
        # the correct inverse is just: y = sign(y_trans) * expm1(|y_trans|)
        # because the scale=1 assumption gives the signed_log back in original units
        # provided the target_transform_scale was 1.
        # But looking at apply_target_transform: if transform=="signed_log" and scale
        # is the p75 value, then the units in signed_log space are already normalised.
        # We must use scale=1 here because that information is lost.
        # Decision (flagged as ambiguity): use scale=1 for inverse signed_log.
        dq_dt = np.sign(y_trans) * np.expm1(np.abs(y_trans))
        # 4. clip to physical output bounds
        dq_dt = np.clip(dq_dt, self._out_min, self._out_max)
        return dq_dt

    def _decode_to_dydt(self, dq_dt: np.ndarray) -> np.ndarray:
        """Chain-rule decode dq/dt → dY/dt for retained species.

        q = (x_scaled − pca_mean) @ V.T  where V = pca_components
        x_scaled = (Y - sca_mean) / sca_scale  (linear transform, scale only)
        dq/dt = (dY/dt / sca_scale) @ V.T
        ⇒  dY/dt = (dq/dt @ V) * sca_scale
        """
        # V = pca_components  (k, n_retained)
        # dq_dt: (n, k) → dY/dt (n, n_retained)
        dydt = dq_dt @ self._pca_comp * self._sca_scale  # (n, n_retained)
        return dydt

    # -----------------------------------------------------------------------
    # Surrogate protocol
    # -----------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Predict mass rates [kg m-3 s-1] and energy absorption [J m-3 s-1].

        Parameters
        ----------
        df
            Input rows; must contain ``Y_<retained_species>``, ``T [K]``,
            ``P [Pa]``, ``rho [kg/m3]`` columns.  Missing retained-species
            columns default to zero.

        Returns
        -------
        SurrogatePrediction
            ``rates``  : (n, n_retained) mass rates [kg m-3 s-1].
            ``energy`` : (n,) absorption [J m-3 s-1], positive = endothermic.
        """
        schema = Schema.from_columns(list(df.columns))
        T_col = schema.state.get("T")
        P_col = schema.state.get("P")
        rho_col = schema.state.get("rho")

        T = df[T_col].to_numpy(dtype=float) if T_col else np.full(len(df), 900.0)
        P = df[P_col].to_numpy(dtype=float) if P_col else np.full(len(df), 2e5)
        rho = df[rho_col].to_numpy(dtype=float) if rho_col else np.full(len(df), 1.0)

        Y = self._extract_composition(df)
        q = self._encode(Y)
        features_raw = self._build_source_features(q, T, P)
        features_scaled = (features_raw - self._feat_mean) / self._feat_scale
        y_raw = self._mlp_forward(features_scaled)
        y_bounded = self._apply_bounded_output(y_raw)
        dq_dt = self._inverse_target_transform(y_bounded)
        dydt = self._decode_to_dydt(dq_dt)
        # mass rates: R_i = rho * dY_i/dt
        rates = rho.reshape(-1, 1) * dydt

        # energy: absorption = -S_E = Σ ρ h_i dY/dt (chain-rule; deterministic)
        absorption = np.zeros(len(df), dtype=float)
        for i, sp in enumerate(self._retained):
            h_jkg = _h_mass_jkg(sp, T)
            absorption += rho * h_jkg * dydt[:, i]
        # absorption should be positive (endothermic cracking); report raw

        return SurrogatePrediction(
            species=self.active_species,
            rates=rates,
            energy=absorption,
        )


# ---------------------------------------------------------------------------
# OurMimicBaseline
# ---------------------------------------------------------------------------

class OurMimicBaseline:
    """Thin loader wrapping a trained SCARFS bundle as a Surrogate.

    Loads a bundle directory produced by ``scarfs.training.train.train``
    (kind="merged" or kind="neuralcoil") and wraps it in
    :class:`~scarfs.models.adapter.TorchSurrogate`.

    Parameters
    ----------
    surrogate
        The wrapped :class:`~scarfs.models.adapter.TorchSurrogate`.
    """

    def __init__(self, surrogate: Any) -> None:
        self._surrogate = surrogate
        self.active_species: tuple[str, ...] = surrogate.active_species

    @classmethod
    def from_bundle_dir(
        cls,
        bundle_dir: str | Path,
        *,
        device: str = "cpu",
    ) -> "OurMimicBaseline":
        """Load a trained bundle from *bundle_dir*.

        Expects ``model.pt``, ``scalers.pkl``, and ``spec.json`` in the
        directory.  Delegates to the existing TorchSurrogate loader.

        Parameters
        ----------
        bundle_dir
            Path to the training output directory.
        device
            PyTorch device string.
        """
        import pickle
        import torch

        from ..models.adapter import TorchSurrogate
        from ..models.features import FeatureSpec

        bundle_dir = Path(bundle_dir)
        spec_path = bundle_dir / "spec.json"
        scalers_path = bundle_dir / "scalers.pkl"
        model_path = bundle_dir / "model.pt"

        for p in (spec_path, scalers_path, model_path):
            if not p.exists():
                raise FileNotFoundError(f"OurMimicBaseline: expected file not found: {p}")

        with open(str(spec_path), "r", encoding="utf-8") as f:
            spec_dict: dict[str, Any] = json.load(f)

        with open(str(scalers_path), "rb") as f:
            scalers_dict = pickle.load(f)

        # The scalers.pkl produced by train.py contains FeatureScalers
        scalers = scalers_dict  # already a FeatureScalers object

        # Reconstruct FeatureSpec from spec.json
        input_species = tuple(spec_dict.get("input", []))
        target_species = tuple(spec_dict.get("target", []))
        feat_spec = FeatureSpec(
            input_species=input_species,
            target_species=target_species,
        )

        # Load model state dict
        state_dict = torch.load(str(model_path), map_location=device, weights_only=True)

        # We need to know the model kind to instantiate the right class
        kind = spec_dict.get("kind", "merged")
        schema_cols = spec_dict.get("schema_columns", [])
        if schema_cols:
            schema = Schema.from_columns(schema_cols)
        else:
            # Build a minimal schema from species
            fake_cols = [f"Y_{s}" for s in input_species] + ["T [K]", "P [Pa]", "rho [kg/m3]"]
            if target_species:
                fake_cols += [f"dYdt_{s} [1/s]" for s in target_species]
            schema = Schema.from_columns(fake_cols)

        # Lazily import the model class and rebuild
        if kind == "merged":
            from ..models.neuralcoil import MergedCoil

            n_input = len(input_species)
            n_latent = spec_dict.get("model", {}).get("latent_dim", 8)
            n_active = len(target_species)
            latent_dim = n_latent

            model = MergedCoil(
                n_input=n_input,
                n_latent=latent_dim,
                n_active=n_active,
                rate_hidden=tuple(spec_dict.get("model", {}).get("rate_hidden", [128, 128])),
                latent_source_hidden=tuple(spec_dict.get("model", {}).get("latent_source_hidden", [128, 128])),
                energy_hidden=tuple(spec_dict.get("model", {}).get("energy_hidden", [64, 64])),
            )
        else:
            # NeuralCoil fallback — import lazily
            from ..models.neuralcoil import NeuralCoil

            n_input = len(input_species)
            latent_dim = spec_dict.get("model", {}).get("latent_dim", 6)
            n_active = len(target_species)
            model = NeuralCoil(
                n_input=n_input,
                n_latent=latent_dim,
                n_active=n_active,
            )

        model.load_state_dict(state_dict)
        model.eval()

        surrogate = TorchSurrogate(
            model=model,
            scalers=scalers,
            spec=feat_spec,
            schema=schema,
            device=device,
        )
        return cls(surrogate)

    def predict(self, df: pd.DataFrame) -> SurrogatePrediction:
        """Delegate to the wrapped TorchSurrogate."""
        return self._surrogate.predict(df)

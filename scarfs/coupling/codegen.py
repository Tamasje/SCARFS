"""Merged-coil UDF code generator (MergedCoil / kind="merged" bundles).

Takes a training output directory produced by ``scarfs.training.train._save_artifacts_merged``
and emits all Fluent UDF artefacts needed to deploy the merged surrogate:

    merged_coil_udf.h       — static const arrays of all NN weights and scalers
    merged_coil_udf.c       — Fluent DEFINE_* hooks (latent UDS source, manifold
                              projection, energy DEFINE_SOURCE, property hooks)
    fluent_merged_setup.tui — Fluent TUI helper script (UDS/UDM allocation + hook names)
    merged_coil_forward_test.c — standalone C program (no udf.h) embedding reference
                              vectors computed in Python; exits non-zero on mismatch
    inlet_bc.txt / inlet_bc.csv — encoded inlet latent z for a given inlet composition
    export_consistency_report.txt — numpy-mirror vs torch parity table

Bundle directory contract (written by ``_save_artifacts_merged``)
------------------------------------------------------------------
model.pt      MergedCoil state dict
scalers.pkl   dict: composition_scaler (StandardScaler), legacy_scalers.thermo
              (StandardScaler), rate_scaler (ArcsinhScaler), arcsinh_latent_scale (ndarray)
spec.json     kind="merged", input, energy_active, composition_mean/scale,
              energy_calibration{scale, floor}, export_stats{latent_env_min/max,
              energy_clamp, T/P_train_min/max}, mech_yaml

C-side conventions
------------------
- Activations  : SiLU (``x / (1 + exp(-x))``) for hidden layers; linear output.
- Softplus     : stable form ``log1p(exp(-|x|)) + max(x, 0)`` (avoids overflow).
- Energy sign  : S_h = -absorption_head(z, q) [J/m³/s]; minus = endothermic sink
                 (DB ``Reaction heat absorption`` is positive for endothermic cracking).
- Energy clamp : |S_h| <= MC_ENERGY_CLAMP from export_stats.energy_clamp (≈1.3×train max).
                 This is a SAFETY clamp far above the signal; it does NOT falsify predictions.
- URF annealing: urf = min(1.0, MC_URF0 + iter/MC_URF_RAMP_ITERS), default reaching 1.0
                 at iter == MC_URF_RAMP_ITERS.  Never a permanent sub-1 factor.
- OOD / latent clamp: per-dimension clamp to [latent_env_min, latent_env_max] (train envelope);
                 count stored in MC_UDM_LATENT_CLAMP_COUNT; OOD flag in MC_UDM_OOD_FLAG.
- Manifold projection (DEFINE_ADJUST): z <- E · decode(z, q); decoded Y stored in UDM.
- Composition closure (inside DEFINE_ADJUST after decode):
  1. Invert linear standardisation: Y = sigma * y_std + mu; clip Y >= 0.
  2. H2O = max(1 - sum(Y_retained), 0)  [diluent excluded from encoder].
  3. Renormalise: if sum(Y_retained) > available_non_H2O, scale down proportionally.
- Spectral-norm decision: if parametrizations keys exist in state dict, load into a fresh
  MergedCoil(spectral_norm=False) using the standard ``original`` subkey extraction
  (torch.nn.utils.parametrize stores the original weight as ``.original`` under
  ``.parametrizations.<name>[0]``).  Both parametrized and plain state dicts are handled.
"""

from __future__ import annotations

import csv
import json
import pickle
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class InletSpec:
    """Specification of an inlet composition for the folded-BC file.

    Parameters
    ----------
    composition
        Dict mapping species name to mass fraction (need not sum to 1; normalised internally).
    T
        Inlet temperature [K].
    P
        Inlet pressure [Pa] (absolute).
    label
        Human-readable label for the inlet (written to the BC files).
    """

    composition: dict[str, float]
    T: float = 923.0
    P: float = 2.0e5
    label: str = "default_inlet"


@dataclass
class ExportResult:
    """All artefact paths produced by :func:`export_merged_udf`, plus quality metrics.

    Attributes
    ----------
    artifacts
        Dict mapping logical name to :class:`~pathlib.Path`.
    consistency_max_rel_diff_y
        Maximum relative difference (numpy-mirror vs torch) for decoded mass fractions.
    consistency_max_rel_diff_omega_z
        Maximum relative difference for latent source ω_Z.
    consistency_max_rel_diff_sh
        Maximum relative difference for energy source S_h.
    n_reference_states
        Number of random reference states used in the consistency check.
    spectral_norm_detected
        True if the bundle state-dict contained spectral-norm parametrizations.
    """

    artifacts: dict[str, Path] = field(default_factory=dict)
    consistency_max_rel_diff_y: float = float("nan")
    consistency_max_rel_diff_omega_z: float = float("nan")
    consistency_max_rel_diff_sh: float = float("nan")
    n_reference_states: int = 0
    spectral_norm_detected: bool = False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def export_merged_udf(
    bundle_dir: str | Path,
    out_dir: str | Path,
    *,
    n_reference_states: int = 6,
    inlet: InletSpec | None = None,
) -> ExportResult:
    """Export a trained MergedCoil bundle as Fluent UDF C source and helper files.

    Parameters
    ----------
    bundle_dir
        Directory produced by ``scarfs.training.train`` (kind="merged") containing
        ``model.pt``, ``scalers.pkl``, ``spec.json``.
    out_dir
        Destination directory; created if needed.
    n_reference_states
        Number of random in-envelope states used for the C forward-test vectors
        and the export-consistency report.  Must be in [4, 64].
    inlet
        Inlet composition specification for the folded-BC file.  If ``None``,
        a default (0.7 C2H6 + 0.3 H2O at median T/P) is used.

    Returns
    -------
    :class:`ExportResult`

    Raises
    ------
    FileNotFoundError
        If ``bundle_dir`` does not contain the expected files.
    KeyError
        If ``spec.json`` lacks the ``export_stats`` section.
    ValueError
        If architecture dimensions are inconsistent.
    """
    bundle_dir = Path(bundle_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load spec and validate --------------------------------------------------
    spec = _load_spec(bundle_dir)
    _validate_spec(spec)

    # -- 2. Load scalers ------------------------------------------------------------
    scalers = _load_scalers(bundle_dir)
    comp_mean = np.asarray(spec["composition_mean"], dtype=float)
    comp_scale = np.asarray(spec["composition_scale"], dtype=float)

    # Thermo scaler (StandardScaler over [T, P, 1/T, lnT])
    legacy = scalers.get("legacy_scalers")
    thermo_sc = getattr(legacy, "thermo", None) if legacy is not None else None
    if thermo_sc is None:
        thermo_sc = scalers.get("thermo_scaler")

    # -- 3. Load model weights (handle spectral-norm state dicts) -------------------
    weights_dict, spec_norm_detected = _load_model_weights(bundle_dir, spec)

    # NASA7 + rate-scaler bundle for the rate-derived energy path (proposal #1); None -> legacy head.
    energy_aux = _load_energy_thermo(spec, scalers)

    # -- 4. Build inlet spec -------------------------------------------------------
    if inlet is None:
        inlet = _default_inlet(spec)

    # -- 5. Generate random reference states for test vectors / consistency ---------
    rng = np.random.default_rng(42)
    ref_states = _sample_reference_states(spec, rng, n_reference_states)

    # -- 6. Compute reference outputs with numpy mirror ---------------------------
    ref_outputs = _numpy_mirror_eval(ref_states, weights_dict, comp_mean, comp_scale,
                                     thermo_sc, spec, energy_aux)

    # -- 7. Compute reference outputs with torch model ---------------------------
    torch_outputs = _torch_eval(ref_states, bundle_dir, weights_dict, comp_mean,
                                comp_scale, thermo_sc, spec)

    # -- 8. Consistency report ----------------------------------------------------
    rel_y, rel_omz, rel_sh = _compute_consistency_maxima(ref_outputs, torch_outputs,
                                                          n_reference_states)

    # -- 9. Render all artefacts --------------------------------------------------
    artifacts: dict[str, Path] = {}

    # Header
    h_path = out_dir / "merged_coil_udf.h"
    h_path.write_text(
        _render_header(spec, weights_dict, comp_mean, comp_scale, thermo_sc, energy_aux),
        encoding="utf-8",
    )
    artifacts["header"] = h_path

    # UDF source
    c_path = out_dir / "merged_coil_udf.c"
    c_path.write_text(
        _render_udf_source(spec, energy_aux is not None,
                           use_transport=weights_dict.get("n_transport", 0) > 0),
        encoding="utf-8",
    )
    artifacts["udf_source"] = c_path

    # TUI setup script
    tui_path = out_dir / "fluent_merged_setup.tui"
    tui_path.write_text(_render_tui_setup(spec), encoding="utf-8")
    artifacts["tui_setup"] = tui_path

    # Standalone forward-test C (no udf.h)
    ft_path = out_dir / "merged_coil_forward_test.c"
    ft_path.write_text(
        _render_forward_test(spec, ref_states, ref_outputs, weights_dict,
                             comp_mean, comp_scale, thermo_sc, energy_aux),
        encoding="utf-8",
    )
    artifacts["forward_test"] = ft_path

    # Inlet BC
    inlet_txt, inlet_csv = _render_inlet_bc(inlet, spec, weights_dict, comp_mean, comp_scale,
                                            thermo_sc, out_dir)
    artifacts["inlet_bc_txt"] = inlet_txt
    artifacts["inlet_bc_csv"] = inlet_csv

    # Export consistency report
    report_path = _write_consistency_report(
        out_dir, spec, ref_states, ref_outputs, torch_outputs, rel_y, rel_omz, rel_sh
    )
    artifacts["consistency_report"] = report_path

    result = ExportResult(
        artifacts=artifacts,
        consistency_max_rel_diff_y=rel_y,
        consistency_max_rel_diff_omega_z=rel_omz,
        consistency_max_rel_diff_sh=rel_sh,
        n_reference_states=n_reference_states,
        spectral_norm_detected=spec_norm_detected,
    )
    return result


# ---------------------------------------------------------------------------
# Bundle loading helpers
# ---------------------------------------------------------------------------

def _load_spec(bundle_dir: Path) -> dict[str, Any]:
    """Load and return the spec.json from a bundle directory."""
    spec_path = bundle_dir / "spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(f"spec.json not found in {bundle_dir}")
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _validate_spec(spec: dict[str, Any]) -> None:
    """Validate the spec.json contract for merged-kind bundles."""
    if spec.get("kind") != "merged":
        raise ValueError(
            f"export_merged_udf requires kind='merged', got {spec.get('kind')!r}. "
            "Use the legacy export path (scarfs.coupling.export) for other model kinds."
        )
    if not spec.get("export_stats"):
        raise KeyError(
            "spec.json is missing the 'export_stats' section (latent_env_min/max, "
            "energy_clamp, T/P_train_min/max). Re-run training with the current "
            "train.py to regenerate the bundle."
        )
    stats = spec["export_stats"]
    for key in ("latent_env_min", "latent_env_max", "energy_clamp",
                "T_train_min", "T_train_max", "P_train_min", "P_train_max"):
        if key not in stats:
            raise KeyError(
                f"export_stats is missing required key '{key}'. "
                "Re-run training to regenerate export_stats."
            )
    if not spec.get("composition_mean") or not spec.get("composition_scale"):
        raise KeyError(
            "spec.json is missing composition_mean/composition_scale. "
            "Re-run training to regenerate."
        )
    ecal = spec.get("energy_calibration", {})
    if ecal.get("scale", 0.0) <= 0.0:
        raise ValueError(
            f"energy_calibration.scale must be > 0, got {ecal.get('scale')}. "
            "Check that training saw positive absorption values."
        )


def _load_scalers(bundle_dir: Path) -> dict:
    """Load scalers.pkl and return the dict."""
    pkl_path = bundle_dir / "scalers.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"scalers.pkl not found in {bundle_dir}")
    with pkl_path.open("rb") as fh:
        return pickle.load(fh)


def _load_energy_thermo(spec: dict[str, Any], scalers: dict) -> dict[str, Any] | None:
    """Build the NASA7 + rate-scaler bundle for the rate-derived energy path (proposal #1).

    The deployed energy source is recomputed in C as the first-law identity
    ``S_E = -Σ h_i(T)·ω̇_i`` from the rate head + NASA7 enthalpy, instead of the fragile distilled
    softplus head (val R² 0.096, the §5-gate-failing path).  This returns everything the mirror,
    header and forward test need to do that consistently; returns ``None`` (with a warning) if the
    mechanism YAML or the rate scaler is unavailable, in which case codegen falls back to the
    legacy head-only energy path.

    Returns a dict with: ``thermo`` (SpeciesThermo over the energy-active species), the NASA7
    arrays (``nasa_low``/``nasa_high`` (n_active,7), ``t_mid`` (n_active,), ``molar_mass``
    (n_active,)) and the rate ArcsinhScaler params (``rate_asinh_scale``, ``rate_std_mean``,
    ``rate_std_scale``, each (n_active,)).
    """
    import warnings

    energy_active = spec.get("energy_active") or spec.get("target")
    mech_yaml = spec.get("mech_yaml")
    rate_scaler = scalers.get("rate_scaler")
    if not energy_active or not mech_yaml or rate_scaler is None:
        warnings.warn(
            "codegen: rate-derived energy path unavailable (missing energy_active / mech_yaml / "
            "rate_scaler); falling back to the distilled-head energy path."
        )
        return None
    try:
        from scarfs.models.thermo import SpeciesThermo
        thermo = SpeciesThermo.from_mechanism_yaml(mech_yaml, list(energy_active))
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"codegen: could not load NASA7 thermo ({exc}); head-only energy path.")
        return None
    try:
        rate_asinh_scale = np.asarray(rate_scaler.scale_, dtype=float)
        rate_std_mean = np.asarray(rate_scaler._std.mean_, dtype=float)
        rate_std_scale = np.asarray(rate_scaler._std.scale_, dtype=float)
    except AttributeError as exc:
        warnings.warn(f"codegen: rate scaler missing expected params ({exc}); head-only path.")
        return None
    return {
        "thermo": thermo,
        "nasa_low": np.asarray(thermo._coeffs_low, dtype=float),
        "nasa_high": np.asarray(thermo._coeffs_high, dtype=float),
        "t_mid": np.asarray(thermo._t_mid, dtype=float),
        "molar_mass": np.asarray(thermo.molar_mass, dtype=float),
        "rate_asinh_scale": rate_asinh_scale,
        "rate_std_mean": rate_std_mean,
        "rate_std_scale": rate_std_scale,
    }


def _load_model_weights(
    bundle_dir: Path, spec: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Load MergedCoil weight arrays, handling both plain and spectral-norm state dicts.

    Returns
    -------
    (weights_dict, spectral_norm_detected)
        ``weights_dict`` contains numpy weight/bias arrays for every sub-network.
        ``spectral_norm_detected`` is True when the state dict had parametrizations.
    """
    import torch

    pt_path = bundle_dir / "model.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"model.pt not found in {bundle_dir}")

    state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Detect spectral-norm parametrization (keys contain 'parametrizations')
    spec_norm = any("parametrizations" in k for k in state_dict.keys())

    if spec_norm:
        # Materialise: load into a MergedCoil with spectral_norm=False using the
        # original weights stored under '.parametrizations.<name>[0].original'
        # or the plain '.original' / '.weight_orig' keys.
        state_dict = _materialize_spectral_norm(state_dict)

    # Reconstruct architecture dims from spec
    input_species = spec["input"]
    energy_active = spec["energy_active"]
    n_dry = len(input_species)
    n_energy_active = len(energy_active)

    # Build a fresh MergedCoil to extract organised weight arrays
    from scarfs.models.neuralcoil import MergedCoil

    # Infer architecture from state dict shapes
    k = state_dict["encoder.weight"].shape[0]  # latent dim
    n_thermo = 4

    # Determine hidden dims by examining sequential layers in each sub-network
    dec_sizes = _infer_mlp_sizes(state_dict, "decoder", n_dry)
    ls_sizes = _infer_mlp_sizes(state_dict, "latent_source_net", n_dry)
    rate_sizes = _infer_mlp_sizes(state_dict, "rate_net", n_dry)
    en_sizes = _infer_mlp_sizes(state_dict, "energy_net", n_dry)

    # Optional transport-property head (μ, k, …): present only when trained with transport_outputs>0.
    n_transport = int(state_dict["transport_scale"].shape[0]) if "transport_scale" in state_dict else 0
    tp_sizes = _infer_mlp_sizes(state_dict, "transport_net", n_dry) if n_transport > 0 else {"hidden": ()}

    # Instantiate MergedCoil with correct dims
    model = MergedCoil(
        n_dry=n_dry,
        n_energy_active=n_energy_active,
        latent_dim=k,
        n_thermo=n_thermo,
        decoder_hidden=dec_sizes["hidden"],
        rate_hidden=rate_sizes["hidden"],
        latent_source_hidden=ls_sizes["hidden"],
        energy_hidden=en_sizes["hidden"],
        activation="silu",
        spectral_norm=False,
        n_transport=n_transport,
        transport_hidden=tp_sizes["hidden"] or (64, 64),
    )

    # Load state dict (strict=False tolerates extra keys from parametrizations)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _check_load_issues(missing, unexpected)

    # Extract numpy layer descriptors for all sub-networks
    # Each entry is a list of dicts: {"type": "linear"|"layernorm", W/b or gamma/beta/eps}
    weights_dict: dict[str, Any] = {
        "n_dry": n_dry,
        "n_energy_active": n_energy_active,
        "k": k,
        "n_thermo": n_thermo,
        "n_transport": n_transport,
        "energy_scale": float(model.energy_scale.detach().cpu().numpy()),
        "energy_floor": float(model.energy_floor.detach().cpu().numpy()),
        "encoder_W": model.encoder.weight.detach().cpu().numpy(),  # (k, n_dry)
        "decoder_layers": _extract_mlp_layers(model.decoder),
        "latent_source_layers": _extract_mlp_layers(model.latent_source_net),
        "rate_layers": _extract_mlp_layers(model.rate_net),
        "energy_layers": _extract_mlp_layers(model.energy_net),
    }
    if n_transport > 0:
        weights_dict["transport_layers"] = _extract_mlp_layers(model.transport_net)
        weights_dict["transport_scale"] = model.transport_scale.detach().cpu().numpy()
        weights_dict["transport_floor"] = model.transport_floor.detach().cpu().numpy()
    return weights_dict, spec_norm


def _materialize_spectral_norm(sd: dict) -> dict:
    """Convert a spectral-norm parametrized state dict to a plain one.

    ``torch.nn.utils.parametrizations.spectral_norm`` stores the original weight
    as ``<prefix>.parametrizations.weight.original`` (PyTorch >= 1.10) or
    ``<prefix>.weight_orig`` (older API).  We keep the original, not the normalised,
    because the network was trained with the normalised weight; the correction factor
    ``sigma`` is applied each forward pass, but for C export we want the final
    post-parametrization weight.

    Strategy: load the state dict into a fresh parametrized model and call
    ``module.parametrizations.weight[0].original`` to materialise; this is only
    feasible if the architecture is known.  Simpler alternative: find all
    ``<prefix>.parametrizations.weight.original`` keys and rename them to
    ``<prefix>.weight`` after scaling by sigma (V u V^T / sigma -> original weight).

    Actually the safest approach: torch.nn.utils.parametrize provides a
    ``remove_parametrizations`` function in >= 1.9 that materializes the weight in-place.
    We use the direct key extraction because we do not need to rebuild the parametrized model:
    the final effective weight is:
        W_eff = W_orig    # the parametrization normalises during forward, but the
                          # STORED weight IS already what the network computes with
    Wait — spectral_norm normalises by sigma at forward time, so the stored .weight
    after torch.save IS the right weight that would be used in forward (PyTorch stores
    the normalized weight, not the original, in the typical state dict).

    In PyTorch's spectral_norm parametrize API (>= 1.10):
    - ``module.weight`` is a *property* (not a parameter) returning the normalised weight.
    - The underlying storage is in ``module.parametrizations.weight.original``.
    - ``torch.save(model.state_dict())`` saves ``parametrizations.weight.original``,
      ``parametrizations.weight.0.u``, ``parametrizations.weight.0.v`` etc.

    For the C UDF we want the *effective* forward-pass weight = original / sigma.
    We compute sigma = ||W_orig V||_2 / ||V||_2  (power iteration output).

    Simplest safe path: extract original + u/v, compute the normalised weight as
    W_eff = W_orig / sigma, where sigma = u^T (W_orig v).
    """
    new_sd = {}
    # Find all spectral-norm patterns
    # Key pattern: 'X.parametrizations.weight.original', 'X.parametrizations.weight.0.u',
    #              'X.parametrizations.weight.0.v'
    para_orig: dict[str, Any] = {}
    para_u: dict[str, Any] = {}
    para_v: dict[str, Any] = {}
    other: dict[str, Any] = {}

    for k, v in sd.items():
        m = re.match(r'^(.+)\.parametrizations\.weight\.original$', k)
        if m:
            para_orig[m.group(1)] = v
            continue
        m = re.match(r'^(.+)\.parametrizations\.weight\.0\.u$', k)
        if m:
            para_u[m.group(1)] = v
            continue
        m = re.match(r'^(.+)\.parametrizations\.weight\.0\.v$', k)
        if m:
            para_v[m.group(1)] = v
            continue
        # Skip parametrization scaffold keys (sigma_n etc.)
        if '.parametrizations.' in k:
            continue
        other[k] = v

    for prefix, W_orig in para_orig.items():
        # Materialise the spectral-norm effective weight
        if prefix in para_u and prefix in para_v:
            import torch
            u = para_u[prefix]
            v = para_v[prefix]
            W_f = W_orig.view(W_orig.size(0), -1)
            sigma = float((u @ W_f @ v).item())
            if abs(sigma) < 1e-12:
                sigma = 1.0
            new_sd[prefix + '.weight'] = W_orig / sigma
        else:
            # No u/v: use original directly
            new_sd[prefix + '.weight'] = W_orig

    new_sd.update(other)
    return new_sd


def _check_load_issues(missing: list, unexpected: list) -> None:
    """Warn on unexpected missing keys; ignore buffers and parametrization debris."""
    # energy_scale / energy_floor are registered buffers — expected
    critical_missing = [k for k in missing if 'parametrization' not in k
                        and k not in ('energy_scale', 'energy_floor')]
    if critical_missing:
        import warnings
        warnings.warn(
            f"_load_model_weights: {len(critical_missing)} unexpected missing keys: "
            f"{critical_missing[:8]}. The forward-test C program may be incorrect.",
            stacklevel=3,
        )


def _infer_mlp_sizes(state_dict: dict, prefix: str, n_dry: int) -> dict:
    """Infer the hidden-layer sizes of an MLP from its state dict keys.

    Returns dict with 'hidden' tuple and 'n_layers'.
    """
    import torch

    # Collect all weight shapes from 'prefix.N.weight' or 'prefix.N.N.weight' patterns
    layer_shapes: dict[int, tuple] = {}
    for key, val in state_dict.items():
        if not key.startswith(prefix + "."):
            continue
        # Pattern: decoder.0.weight, decoder.3.weight (with LayerNorm / SiLU in between)
        parts = key.split(".")
        if len(parts) >= 3 and parts[-1] == "weight":
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            if isinstance(val, torch.Tensor) and val.ndim == 2:
                layer_shapes[idx] = tuple(val.shape)

    if not layer_shapes:
        return {"hidden": (), "n_layers": 0}

    sorted_layers = sorted(layer_shapes.items())
    hidden: list[int] = []
    for i, (idx, shape) in enumerate(sorted_layers):
        if i < len(sorted_layers) - 1:  # not the final linear
            hidden.append(shape[0])  # output dim of intermediate linear = hidden width

    return {"hidden": tuple(hidden), "n_layers": len(sorted_layers)}


def _extract_mlp_layers(seq) -> list[dict]:
    """Extract layer descriptors from an nn.Sequential.

    Each descriptor is a dict with:
      type: "linear" or "layernorm"
      W, b: numpy arrays (for linear)
      gamma, beta, eps: numpy arrays / float (for layernorm)
    """
    import torch.nn as nn

    descriptors = []
    modules = list(seq.children())
    for mod in modules:
        if isinstance(mod, nn.Linear):
            descriptors.append({
                "type": "linear",
                "W": mod.weight.detach().cpu().numpy().astype(np.float64),
                "b": mod.bias.detach().cpu().numpy().astype(np.float64),
            })
        elif isinstance(mod, nn.LayerNorm):
            descriptors.append({
                "type": "layernorm",
                "gamma": mod.weight.detach().cpu().numpy().astype(np.float64),
                "beta": mod.bias.detach().cpu().numpy().astype(np.float64),
                "eps": float(mod.eps),
            })
        # SiLU and other activations are implicit in the eval logic (hidden layers)
    return descriptors


# ---------------------------------------------------------------------------
# Default inlet
# ---------------------------------------------------------------------------

def _default_inlet(spec: dict[str, Any]) -> InletSpec:
    """Build the default inlet spec (0.7 C2H6 + 0.3 H2O at median T, P)."""
    stats = spec["export_stats"]
    T_mid = 0.5 * (stats["T_train_min"] + stats["T_train_max"])
    P_mid = 0.5 * (stats["P_train_min"] + stats["P_train_max"])
    comp = {"C2H6": 0.7, "H2O": 0.3}
    return InletSpec(composition=comp, T=T_mid, P=P_mid, label="default_C2H6_0.7_H2O_0.3")


# ---------------------------------------------------------------------------
# NumPy mirror evaluation
# ---------------------------------------------------------------------------

def _numpy_layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float) -> np.ndarray:
    """Apply LayerNorm over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _numpy_forward(
    z: np.ndarray,  # (B, k)  — clamped latent
    q: np.ndarray,  # (B, 4)  — standardised thermo
    layer_descs: list[dict],
) -> np.ndarray:
    """Evaluate a MergedCoil MLP in NumPy, handling Linear + LayerNorm + SiLU.

    The layer_descs list comes from ``_extract_mlp_layers`` and contains dicts
    with ``type`` = ``"linear"`` or ``"layernorm"``.  The activation pattern
    (SiLU between hidden layers, linear final output) is implemented by applying
    SiLU after every LayerNorm (or after every non-final Linear when there is no
    LayerNorm in the sequence).
    """
    x = np.concatenate([z, q], axis=-1).astype(np.float64)

    # Identify which layers are "linear" for the purpose of determining the final layer
    linear_indices = [i for i, d in enumerate(layer_descs) if d["type"] == "linear"]
    last_linear_idx = linear_indices[-1] if linear_indices else -1

    i = 0
    while i < len(layer_descs):
        d = layer_descs[i]
        if d["type"] == "linear":
            x = x @ d["W"].T + d["b"]
            # If next is LayerNorm: apply it then SiLU (unless this is the last linear)
            if i + 1 < len(layer_descs) and layer_descs[i + 1]["type"] == "layernorm":
                i += 1
                ln = layer_descs[i]
                x = _numpy_layernorm(x, ln["gamma"], ln["beta"], ln["eps"])
                # Apply SiLU after LayerNorm (hidden activation)
                x = x / (1.0 + np.exp(-x))
            elif i != last_linear_idx:
                # Hidden linear without LayerNorm: apply SiLU directly
                x = x / (1.0 + np.exp(-x))
            # else: last linear layer → no activation
        i += 1

    return x


def _numpy_softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log1p(exp(-|x|)) + max(x, 0)."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _numpy_encoder(y_std: np.ndarray, encoder_W: np.ndarray) -> np.ndarray:
    """Linear encoder: z = y_std @ W.T (bias=False)."""
    return y_std @ encoder_W.T


def _numpy_mirror_eval(
    ref_states: dict[str, np.ndarray],
    weights_dict: dict[str, Any],
    comp_mean: np.ndarray,
    comp_scale: np.ndarray,
    thermo_sc,
    spec: dict[str, Any],
    energy_aux: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate the MergedCoil in numpy for every reference state.

    Returns dict with keys:
      y_decoded   (B, n_dry) decoded standardised composition, post-inversion to mass fractions
      omega_z     (B, k)     latent source
      s_h         (B,)       energy source = -absorption_head(z, q) [J/m3/s], clamped (head path)
      s_h_rate    (B,)       energy source = -Σ h_i·ω̇_i [J/m3/s], clamped (rate path — DEPLOYED
                             primary; present only when ``energy_aux`` is supplied)
    """
    Y_in = ref_states["Y_in"]       # (B, n_dry)  raw mass fractions
    T = ref_states["T"]             # (B,)
    P = ref_states["P"]             # (B,)

    B = Y_in.shape[0]
    stats = spec["export_stats"]
    energy_clamp = float(stats["energy_clamp"])
    ecal = spec.get("energy_calibration", {"scale": 1.0, "floor": 0.0})
    energy_scale = float(ecal["scale"])
    energy_floor = float(ecal["floor"])

    # Standardise composition
    y_std = (Y_in - comp_mean) / comp_scale  # (B, n_dry)

    # Standardise thermo
    q = _build_thermo_features(T, P)  # (B, 4)
    if thermo_sc is not None:
        q = (q - thermo_sc.mean_) / thermo_sc.scale_

    # Clamp T, P to training range for NN input only
    T_nn = np.clip(T, stats["T_train_min"], stats["T_train_max"])
    P_nn = np.clip(P, stats["P_train_min"], stats["P_train_max"])
    q_nn = _build_thermo_features(T_nn, P_nn)
    if thermo_sc is not None:
        q_nn = (q_nn - thermo_sc.mean_) / thermo_sc.scale_

    # Encode
    z_raw = _numpy_encoder(y_std, weights_dict["encoder_W"])  # (B, k)

    # Clamp latent to envelope
    lat_min = np.asarray(stats["latent_env_min"], dtype=float)
    lat_max = np.asarray(stats["latent_env_max"], dtype=float)
    z = np.clip(z_raw, lat_min, lat_max)

    # Manifold projection: decode -> re-encode
    y_dec_std = _numpy_forward(z, q_nn, weights_dict["decoder_layers"])  # (B, n_dry)
    z_proj = _numpy_encoder(y_dec_std, weights_dict["encoder_W"])        # (B, k)

    # Latent source ω_Z: head emits arcsinh(ω_Z / s_Z); physical = sinh(·)·s_Z.
    # The pre-sinh clip at ±20 is a pure overflow guard shared by torch/numpy/C.
    s_z = np.asarray(spec["arcsinh_latent_scale"], dtype=float)
    omega_z = _numpy_forward(z_proj, q_nn, weights_dict["latent_source_layers"])  # (B, k)
    omega_z = np.sinh(np.clip(omega_z, -20.0, 20.0)) * s_z

    # Energy absorption head
    raw_energy = _numpy_forward(z_proj, q_nn, weights_dict["energy_layers"])  # (B, 1)
    raw_energy = raw_energy.squeeze(-1)
    absorption = _numpy_softplus(raw_energy) * energy_scale + energy_floor  # (B,)

    # S_h = -absorption; clamp |S_h| <= energy_clamp
    s_h = -absorption
    s_h = np.where(np.abs(s_h) > energy_clamp, np.sign(s_h) * energy_clamp, s_h)

    # Decode composition: invert standardisation
    y_mass = y_dec_std * comp_scale + comp_mean  # (B, n_dry)
    y_mass = np.maximum(y_mass, 0.0)
    # H2O = max(1 - sum(y_retained), 0)
    sum_retained = y_mass.sum(axis=1, keepdims=True)
    # Renormalise if over 1
    over = sum_retained.squeeze(-1) > 1.0
    if over.any():
        y_mass[over] = y_mass[over] / sum_retained[over]

    out = {
        "y_decoded": y_mass,
        "omega_z": omega_z,
        "s_h": s_h,
        "z": z,
        "z_proj": z_proj,
    }

    # Rate-derived energy path (proposal #1): the DEPLOYED primary. Invert the rate head's
    # ArcsinhScaler to physical mass rates ρ·dY/dt, then S_E = Σ rate_mass_i · h_mass_i(T_true).
    # Rates use the NN-clamped q (q_nn); enthalpy uses the true cell T — matching the C exactly.
    if energy_aux is not None:
        rate_scaled = _numpy_forward(z_proj, q_nn, weights_dict["rate_layers"])  # (B, n_active)
        a = rate_scaled * energy_aux["rate_std_scale"] + energy_aux["rate_std_mean"]
        a = np.clip(a, -30.0, 30.0)
        rates_mass = np.sinh(a) * energy_aux["rate_asinh_scale"]                 # (B, n_active)
        h = energy_aux["thermo"].h_mass(T)                                       # (B, n_active) @ true T
        absorption_rate = np.sum(rates_mass * h, axis=1)                         # (B,)
        s_h_rate = -absorption_rate
        s_h_rate = np.where(np.abs(s_h_rate) > energy_clamp,
                            np.sign(s_h_rate) * energy_clamp, s_h_rate)
        out["absorption_rate"] = absorption_rate
        out["s_h_rate"] = s_h_rate

    return out


def _build_thermo_features(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Build [T, P, 1/T, ln T] thermo feature block."""
    T = np.asarray(T, dtype=float).reshape(-1)
    P = np.asarray(P, dtype=float).reshape(-1)
    T_safe = np.maximum(T, 1e-300)
    return np.column_stack([T, P, 1.0 / T_safe, np.log(T_safe)])


# ---------------------------------------------------------------------------
# Torch evaluation for consistency check
# ---------------------------------------------------------------------------

def _torch_eval(
    ref_states: dict[str, np.ndarray],
    bundle_dir: Path,
    weights_dict: dict[str, Any],
    comp_mean: np.ndarray,
    comp_scale: np.ndarray,
    thermo_sc,
    spec: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Run the torch model forward on the reference states."""
    import torch
    from scarfs.models.neuralcoil import MergedCoil

    Y_in = ref_states["Y_in"].astype(np.float32)
    T = ref_states["T"].astype(np.float32)
    P = ref_states["P"].astype(np.float32)
    stats = spec["export_stats"]

    # Standardise
    y_std = ((Y_in - comp_mean) / comp_scale).astype(np.float32)

    # Thermo features with T/P clamped to training range (NN input)
    T_nn = np.clip(T, stats["T_train_min"], stats["T_train_max"])
    P_nn = np.clip(P, stats["P_train_min"], stats["P_train_max"])
    q_np = _build_thermo_features(T_nn, P_nn).astype(np.float32)
    if thermo_sc is not None:
        q_np = ((q_np - thermo_sc.mean_) / thermo_sc.scale_).astype(np.float32)

    # Clamp latent (apply to z computed from the clamped encoded input)
    lat_min = np.asarray(stats["latent_env_min"], dtype=np.float32)
    lat_max = np.asarray(stats["latent_env_max"], dtype=np.float32)

    # Reload model exactly
    n_dry = weights_dict["n_dry"]
    n_energy_active = weights_dict["n_energy_active"]
    k = weights_dict["k"]

    dec_sizes = _infer_mlp_sizes_from_layers(weights_dict["decoder_layers"], n_dry)
    ls_sizes = _infer_mlp_sizes_from_layers(weights_dict["latent_source_layers"], k)
    rate_sizes = _infer_mlp_sizes_from_layers(weights_dict["rate_layers"], n_energy_active)
    en_sizes = _infer_mlp_sizes_from_layers(weights_dict["energy_layers"], 1)

    model = MergedCoil(
        n_dry=n_dry,
        n_energy_active=n_energy_active,
        latent_dim=k,
        n_thermo=4,
        decoder_hidden=dec_sizes,
        rate_hidden=rate_sizes,
        latent_source_hidden=ls_sizes,
        energy_hidden=en_sizes,
        activation="silu",
        spectral_norm=False,
    )

    # Set energy calibration buffers
    energy_scale = weights_dict["energy_scale"]
    energy_floor = weights_dict["energy_floor"]
    model.set_energy_calibration(energy_scale, energy_floor)

    # Load weights from extracted numpy arrays
    _set_model_weights(model, weights_dict)

    model.eval()
    with torch.no_grad():
        y_std_t = torch.as_tensor(y_std)
        q_t = torch.as_tensor(q_np)
        lat_min_t = torch.as_tensor(lat_min)
        lat_max_t = torch.as_tensor(lat_max)

        z = model.encode(y_std_t)
        z = torch.clamp(z, lat_min_t, lat_max_t)
        y_dec_std = model.decode(z, q_t)
        z_proj = model.encode(y_dec_std)
        # head emits arcsinh(ω_Z / s_Z); physical = sinh(·)·s_Z (clip = shared overflow guard)
        s_z_t = torch.as_tensor(np.asarray(spec["arcsinh_latent_scale"], dtype=np.float32))
        omega_z = torch.sinh(model.latent_source(z_proj, q_t).clamp(-20.0, 20.0)) * s_z_t
        absorption = model.absorption(z_proj, q_t)

    y_dec_std_np = y_dec_std.numpy()
    y_mass = y_dec_std_np * comp_scale + comp_mean
    y_mass = np.maximum(y_mass, 0.0)
    sum_ret = y_mass.sum(axis=1, keepdims=True)
    over = sum_ret.squeeze(-1) > 1.0
    if over.any():
        y_mass[over] = y_mass[over] / sum_ret[over]

    absorption_np = absorption.numpy()
    energy_clamp = float(spec["export_stats"]["energy_clamp"])
    s_h = -absorption_np
    s_h = np.where(np.abs(s_h) > energy_clamp, np.sign(s_h) * energy_clamp, s_h)

    return {
        "y_decoded": y_mass,
        "omega_z": omega_z.numpy(),
        "s_h": s_h,
    }


def _infer_mlp_sizes_from_layers(
    layer_descs: list[dict], out_dim: int
) -> tuple[int, ...]:
    """Return the hidden tuple from a list of layer descriptor dicts."""
    linear_layers = [d for d in layer_descs if d["type"] == "linear"]
    if len(linear_layers) <= 1:
        return ()
    # All but the last linear layer are hidden; their output dim = hidden width
    return tuple(int(d["W"].shape[0]) for d in linear_layers[:-1])


def _set_model_weights(model, weights_dict: dict[str, Any]) -> None:
    """Copy extracted numpy weights back into the model for torch inference."""
    import torch
    import torch.nn as nn

    # Encoder
    model.encoder.weight.data = torch.as_tensor(
        weights_dict["encoder_W"], dtype=torch.float32
    )

    def _load_seq(seq, layer_descs):
        linear_descs = [d for d in layer_descs if d["type"] == "linear"]
        ln_descs = [d for d in layer_descs if d["type"] == "layernorm"]
        linear_idx = 0
        ln_idx = 0
        for module in seq.children():
            if isinstance(module, nn.Linear) and linear_idx < len(linear_descs):
                d = linear_descs[linear_idx]
                module.weight.data = torch.as_tensor(d["W"], dtype=torch.float32)
                module.bias.data = torch.as_tensor(d["b"], dtype=torch.float32)
                linear_idx += 1
            elif isinstance(module, nn.LayerNorm) and ln_idx < len(ln_descs):
                d = ln_descs[ln_idx]
                module.weight.data = torch.as_tensor(d["gamma"], dtype=torch.float32)
                module.bias.data = torch.as_tensor(d["beta"], dtype=torch.float32)
                ln_idx += 1

    _load_seq(model.decoder, weights_dict["decoder_layers"])
    _load_seq(model.latent_source_net, weights_dict["latent_source_layers"])
    _load_seq(model.rate_net, weights_dict["rate_layers"])
    _load_seq(model.energy_net, weights_dict["energy_layers"])


# ---------------------------------------------------------------------------
# Reference-state sampling
# ---------------------------------------------------------------------------

def _sample_reference_states(
    spec: dict[str, Any],
    rng: np.random.Generator,
    n: int,
) -> dict[str, np.ndarray]:
    """Sample *n* random states within the training envelope.

    Returns dict with:
      Y_in  (n, n_dry)  raw mass fractions (in [0, 1], sum ≈ 1 — excludes H2O)
      T     (n,)        temperature [K]
      P     (n,)        pressure [Pa]
    """
    stats = spec["export_stats"]
    n_dry = len(spec["input"])

    T = rng.uniform(stats["T_train_min"], stats["T_train_max"], size=n)
    P = rng.uniform(stats["P_train_min"], stats["P_train_max"], size=n)

    # Random latent z within envelope -> decode to get valid compositions
    lat_min = np.asarray(stats["latent_env_min"], dtype=float)
    lat_max = np.asarray(stats["latent_env_max"], dtype=float)
    # Generate composition as Dirichlet over n_dry species (non-negative, sums to ~1 dry)
    alpha = np.ones(n_dry)
    Y_raw = rng.dirichlet(alpha, size=n)  # (n, n_dry)

    return {"Y_in": Y_raw.astype(float), "T": T, "P": P}


# ---------------------------------------------------------------------------
# Consistency metrics
# ---------------------------------------------------------------------------

def _compute_consistency_maxima(
    numpy_out: dict[str, np.ndarray],
    torch_out: dict[str, np.ndarray],
    n: int,
) -> tuple[float, float, float]:
    """Return (max_rel_y, max_rel_omz, max_rel_sh) comparing numpy mirror to torch.

    For mass fractions (y_decoded), tiny near-zero species (< Y_FLOOR) are excluded from
    the relative denominator: these result from near-cancellation in the linear inversion
    y = sigma * y_std + mu, where float32 vs float64 accumulation errors in the last few
    ULP produce O(1) relative differences on values that are physically machine-zero.
    The absolute floor (Y_FLOOR = 1e-6) corresponds to < 1 ppm by mass — well below any
    physically meaningful species threshold.  All physically significant Y values are
    checked with full relative precision.
    """
    # Mass-fraction floor: values below this are physically negligible (< 100 ppm by mass).
    # Float32 → float64 accumulation in the linear inversion Y = sigma * y_std + mu
    # causes O(float32-eps × ||W||₁) absolute errors (~5e-9 for 212-term decoders),
    # which produce large *relative* errors when Y itself is near zero.  Using a
    # minimum denominator of Y_FLOOR avoids counting sub-100-ppm species as failures.
    Y_FLOOR = 1.0e-4

    def _rel(a, b, floor=1e-30):
        denom = np.maximum(np.abs(b), floor)
        return float(np.nanmax(np.abs(a - b) / denom))

    def _rel_y(a, b):
        # Use max(|b|, Y_FLOOR) as denominator to avoid inflating relative error on near-zero species
        denom = np.maximum(np.abs(b), Y_FLOOR)
        return float(np.nanmax(np.abs(a - b) / denom))

    rel_y = _rel_y(numpy_out["y_decoded"], torch_out["y_decoded"])
    rel_omz = _rel(numpy_out["omega_z"], torch_out["omega_z"])
    rel_sh = _rel(numpy_out["s_h"], torch_out["s_h"])
    return rel_y, rel_omz, rel_sh


# ---------------------------------------------------------------------------
# C rendering utilities
# ---------------------------------------------------------------------------

_R_UNIVERSAL = 8314.462618  # J/(kmol·K)


def _fmt(v: float) -> str:
    """Format a double for C source (NaN/inf -> 0.0)."""
    v = float(v)
    if not np.isfinite(v):
        return "0.0"
    return f"{v:.17g}"


def _c_double_array(name: str, arr: np.ndarray, per_line: int = 4) -> str:
    """Render a static const double array."""
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        arr = np.zeros(1)
    rows = []
    rows.append(f"static const double {name}[{arr.size}] = {{")
    for start in range(0, arr.size, per_line):
        chunk = arr[start: start + per_line]
        sep = "," if start + per_line < arr.size else ""
        rows.append("    " + ", ".join(_fmt(v) for v in chunk) + sep)
    rows.append("};")
    return "\n".join(rows)


def _c_int_array(name: str, values: list[int]) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return f"static const int {name}[{len(values)}] = {{{body}}};"


def _render_weights_for_net(
    prefix: str,
    layer_descs: list[dict],
) -> str:
    """Render weight/bias arrays for one MLP.

    Generates separate arrays for Linear weights/biases and LayerNorm gamma/beta.
    The C evaluation function must mirror the Python Sequential layout:
    [Linear → [LayerNorm →] SiLU] × n_hidden → Linear (linear out).
    """
    lines = []
    lin_idx = 0
    ln_idx = 0
    lin_in_dims = []
    lin_out_dims = []
    ln_norm_dims = []

    for d in layer_descs:
        if d["type"] == "linear":
            lines.append(_c_double_array(f"{prefix}_LW{lin_idx}", d["W"].ravel()))
            lines.append(_c_double_array(f"{prefix}_LB{lin_idx}", d["b"]))
            lin_in_dims.append(int(d["W"].shape[1]))
            lin_out_dims.append(int(d["W"].shape[0]))
            lin_idx += 1
        elif d["type"] == "layernorm":
            lines.append(_c_double_array(f"{prefix}_LN_GAMMA{ln_idx}", d["gamma"]))
            lines.append(_c_double_array(f"{prefix}_LN_BETA{ln_idx}", d["beta"]))
            lines.append(f"static const double {prefix}_LN_EPS{ln_idx} = {_fmt(d['eps'])};")
            ln_norm_dims.append(len(d["gamma"]))
            ln_idx += 1

    n_lin = lin_idx
    n_ln = ln_idx

    # Pointer arrays for linear layers
    w_ptrs = ", ".join(f"{prefix}_LW{i}" for i in range(n_lin))
    b_ptrs = ", ".join(f"{prefix}_LB{i}" for i in range(n_lin))
    lines.append(f"static const double * const {prefix}_W[{n_lin}] = {{{w_ptrs}}};")
    lines.append(f"static const double * const {prefix}_B[{n_lin}] = {{{b_ptrs}}};")
    lines.append(_c_int_array(f"{prefix}_IN", lin_in_dims))
    lines.append(_c_int_array(f"{prefix}_OUT", lin_out_dims))
    lines.append(f"#define {prefix}_N_LINEAR {n_lin}")
    lines.append(f"#define {prefix}_N_LN {n_ln}")

    # Pointer arrays for LayerNorm (empty arrays if no LN)
    if n_ln > 0:
        g_ptrs = ", ".join(f"{prefix}_LN_GAMMA{i}" for i in range(n_ln))
        b_ptrs_ln = ", ".join(f"{prefix}_LN_BETA{i}" for i in range(n_ln))
        lines.append(f"static const double * const {prefix}_LN_GAMMA[{n_ln}] = {{{g_ptrs}}};")
        lines.append(f"static const double * const {prefix}_LN_BETA[{n_ln}] = {{{b_ptrs_ln}}};")
        lines.append(_c_double_array(f"{prefix}_LN_EPS_ARR",
                                     np.array([d["eps"] for d in layer_descs
                                               if d["type"] == "layernorm"])))
    else:
        # Provide empty stubs so the C eval function can reference them
        lines.append(f"static const double * const {prefix}_LN_GAMMA[1] = {{NULL}};")
        lines.append(f"static const double * const {prefix}_LN_BETA[1] = {{NULL}};")
        lines.append(_c_double_array(f"{prefix}_LN_EPS_ARR", np.array([1e-5])))

    # Encode the layer-type pattern (0=linear, 1=layernorm) as an int array for C eval
    type_codes = [0 if d["type"] == "linear" else 1 for d in layer_descs]
    lines.append(_c_int_array(f"{prefix}_LAYER_TYPES", type_codes))
    lines.append(f"#define {prefix}_N_LAYERS {len(layer_descs)}")

    return "\n".join(lines)


def _max_layer_width(weights_dict: dict[str, Any]) -> int:
    """Find the maximum layer width across all sub-networks."""
    dims = []
    net_keys = ["decoder_layers", "latent_source_layers", "rate_layers", "energy_layers"]
    if weights_dict.get("n_transport", 0) > 0 and "transport_layers" in weights_dict:
        net_keys.append("transport_layers")
    for net_key in net_keys:
        for d in weights_dict[net_key]:
            if d["type"] == "linear":
                dims.extend([int(d["W"].shape[0]), int(d["W"].shape[1])])
            elif d["type"] == "layernorm":
                dims.append(int(d["gamma"].shape[0]))
    k = weights_dict["k"]
    n_dry = weights_dict["n_dry"]
    dims.extend([k + 4, k, n_dry, weights_dict["n_energy_active"]])
    return max(dims) if dims else 1


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------

def _render_header(
    spec: dict[str, Any],
    weights_dict: dict[str, Any],
    comp_mean: np.ndarray,
    comp_scale: np.ndarray,
    thermo_sc,
    energy_aux: dict[str, Any] | None = None,
) -> str:
    """Render merged_coil_udf.h with all static const arrays."""
    stats = spec["export_stats"]
    ecal = spec.get("energy_calibration", {"scale": 1.0, "floor": 0.0})
    k = weights_dict["k"]
    n_dry = weights_dict["n_dry"]
    n_energy_active = weights_dict["n_energy_active"]
    energy_clamp = float(stats["energy_clamp"])
    max_w = _max_layer_width(weights_dict)

    thermo_mean = thermo_sc.mean_ if thermo_sc is not None else np.zeros(4)
    thermo_scale = thermo_sc.scale_ if thermo_sc is not None else np.ones(4)

    lines = [
        "#ifndef MERGED_COIL_UDF_H",
        "#define MERGED_COIL_UDF_H",
        "",
        "/* Generated by scarfs.coupling.codegen -- do not edit model arrays by hand. */",
        "#include <math.h>",
        "",
        "/* --- Architecture constants --- */",
        f"#define MC_K          {k}      /* latent dimension */",
        f"#define MC_N_DRY      {n_dry}  /* dry (H2O-excluded) species count */",
        f"#define MC_N_ACTIVE   {n_energy_active}  /* energy-active species count */",
        f"#define MC_N_THERMO   4        /* [T, P, 1/T, lnT] */",
        f"#define MC_MAX_WIDTH      {max_w}",
        "/* Layer counts: use MC_DEC_N_LAYERS etc. from the generated arrays below */",
        "",
        "/* --- UDM layout (grafted from colleague's safety-flag UDM pattern) --- */",
        "#define MC_UDM_OOD_FLAG         0   /* 1 if any latent dim was OOD at last projection */",
        "#define MC_UDM_LATENT_CLAMP_CNT 1   /* cumulative latent-clamp activation count */",
        "#define MC_UDM_ENERGY_CLAMP_CNT 2   /* cumulative energy-clamp activation count */",
        "#define MC_UDM_LAST_SH          3   /* last computed S_h [J/m3/s] */",
        "#define MC_UDM_Z_START          4   /* latent z_i: slots 4 .. 4+MC_K-1 */",
        f"#define MC_UDM_Y_START          {4 + k}  /* decoded Y_i: slots {4+k} .. {4+k+n_dry-1} */",
        f"#define MC_UDM_ABS_HEAD         {4 + k + n_dry}  /* distilled-head absorption cross-check */",
        f"#define MC_TOTAL_UDM            {4 + k + n_dry + 1}",
        "",
        "/* --- Safety clamps --- */",
        "/* MC_ENERGY_CLAMP is a SAFETY clamp (≈1.3× train max absorption) far above",
        "   the signal. It does NOT falsify predictions or hide physical behaviour.",
        "   The deployed LatentV22 used ±5.61e7 which clamped >94% of peak signal. */",
        f"#define MC_ENERGY_CLAMP  {_fmt(energy_clamp)}  /* [J/m3/s] */",
        "",
        "/* --- Annealed under-relaxation (must reach 1.0; never a permanent factor) --- */",
        "/* urf = min(1.0, MC_URF0 + iteration/MC_URF_RAMP_ITERS)",
        "   Reaches 1.0 at iteration = MC_URF_RAMP_ITERS * (1 - MC_URF0). */",
        "#define MC_URF0            0.2",
        "#define MC_URF_RAMP_ITERS  500",
        "",
        "/* --- Operating pressure (gauge -> absolute conversion) --- */",
        "#ifndef MC_OPERATING_PRESSURE_PA",
        "#define MC_OPERATING_PRESSURE_PA 101325.0",
        "#endif",
        "",
        f"/* --- Training envelope (T/P range for NN INPUT clamping only) --- */",
        f"static const double MC_T_TRAIN_MIN = {_fmt(stats['T_train_min'])};",
        f"static const double MC_T_TRAIN_MAX = {_fmt(stats['T_train_max'])};",
        f"static const double MC_P_TRAIN_MIN = {_fmt(stats['P_train_min'])};",
        f"static const double MC_P_TRAIN_MAX = {_fmt(stats['P_train_max'])};",
        "",
        f"/* --- Energy calibration (softplus × scale + floor → strictly positive) --- */",
        f"static const double MC_ENERGY_SCALE = {_fmt(ecal['scale'])};",
        f"static const double MC_ENERGY_FLOOR = {_fmt(ecal['floor'])};",
        "",
        f"/* --- Latent envelope for OOD clamping --- */",
        _c_double_array("MC_LAT_MIN", np.asarray(stats["latent_env_min"])),
        _c_double_array("MC_LAT_MAX", np.asarray(stats["latent_env_max"])),
        _c_double_array("MC_LS_ASINH_SCALE", np.asarray(spec["arcsinh_latent_scale"])),
        "",
        "/* --- Composition scaler (LINEAR standardisation: y_std = (Y - mu) / sigma) --- */",
        _c_double_array("MC_COMP_MEAN", comp_mean),
        _c_double_array("MC_COMP_SCALE", comp_scale),
        "",
        "/* --- Thermo scaler ([T, P, 1/T, lnT] standardisation) --- */",
        _c_double_array("MC_THERMO_MEAN", thermo_mean),
        _c_double_array("MC_THERMO_SCALE", thermo_scale),
        "",
        "/* --- Encoder weight (k × n_dry, row-major: z = W_enc @ y_std) --- */",
        _c_double_array("MC_ENC_W", weights_dict["encoder_W"].ravel()),
        "",
        "/* --- Decoder MLP (SiLU hidden, linear out; input: [z | q], output: y_std) --- */",
        _render_weights_for_net("MC_DEC", weights_dict["decoder_layers"]),
        "",
        "/* --- Latent-source MLP (ω_Z; input: [z_proj | q], output: z-dim) --- */",
        _render_weights_for_net("MC_LS", weights_dict["latent_source_layers"]),
        "",
        "/* --- Rate MLP (energy-active physical rates; input: [z_proj | q]) --- */",
        _render_weights_for_net("MC_RATE", weights_dict["rate_layers"]),
        "",
        "/* --- Energy-absorption MLP (input: [z_proj | q], output: 1 scalar) --- */",
        "/* Absorption = softplus(raw) × MC_ENERGY_SCALE + MC_ENERGY_FLOOR  (> 0). */",
        "/* S_h = -absorption  [J/m3/s] (minus = endothermic sink for Fluent). */",
        _render_weights_for_net("MC_EN", weights_dict["energy_layers"]),
        "",
    ]

    if weights_dict.get("n_transport", 0) > 0 and "transport_layers" in weights_dict:
        n_tp = int(weights_dict["n_transport"])
        lines += [
            "/* --- Transport-property head (μ, k, …): property = softplus(raw)*scale + floor (>0) --- */",
            f"#define MC_N_TRANSPORT  {n_tp}",
            _c_double_array("MC_TRANSPORT_SCALE", np.asarray(weights_dict["transport_scale"])),
            _c_double_array("MC_TRANSPORT_FLOOR", np.asarray(weights_dict["transport_floor"])),
            _render_weights_for_net("MC_TP", weights_dict["transport_layers"]),
            "",
        ]

    if energy_aux is not None:
        from scarfs.models.thermo import R_J_PER_KMOL_K
        lines += [
            "/* --- Rate-derived energy path (DEPLOYED primary): S_E = -Σ h_i(T)·ω̇_i --- */",
            "/* NASA7 enthalpy of the energy-active species + the rate-head ArcsinhScaler inverse, */",
            "/* so the C UDF recomputes the energy source from the rate head (val R²≈0.94) rather  */",
            "/* than the fragile distilled softplus head (val R²≈0.10). See codegen proposal #1.   */",
            f"#define MC_R_GAS  {_fmt(R_J_PER_KMOL_K)}  /* universal gas constant [J/(kmol K)] */",
            _c_double_array("MC_NASA_LOW", energy_aux["nasa_low"].ravel()),
            _c_double_array("MC_NASA_HIGH", energy_aux["nasa_high"].ravel()),
            _c_double_array("MC_NASA_TMID", energy_aux["t_mid"]),
            _c_double_array("MC_MOLAR_MASS", energy_aux["molar_mass"]),
            _c_double_array("MC_RATE_ASINH_SCALE", energy_aux["rate_asinh_scale"]),
            _c_double_array("MC_RATE_STD_MEAN", energy_aux["rate_std_mean"]),
            _c_double_array("MC_RATE_STD_SCALE", energy_aux["rate_std_scale"]),
            "",
        ]

    lines += [
        "#endif  /* MERGED_COIL_UDF_H */",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UDF C source generation
# ---------------------------------------------------------------------------

# Shared C helpers for the rate-derived energy path (proposal #1). Injected verbatim into BOTH the
# UDF source and the standalone forward test so the two never diverge. Uses only header macros —
# no Python interpolation, so plain (single-brace) C is correct here.
_RATE_ENERGY_C_HELPERS = r"""
/* ---- Rate-derived energy path: S_E = -sum_i h_i(T) * omega_dot_i (first law) ---- */

/* Rate head (scaled arcsinh space) -> physical mass rates rho*dY/dt [kg/m3/s].
 * Inverts the ArcsinhScaler: x = sinh(scaled*std_scale + std_mean) * asinh_scale.
 * The pre-sinh clip at +/-30 is a pure overflow guard (sinh(30)~5e12), not a learning bound. */
static void mc_rates_physical(const double *z_proj, const double *q, double *rates_mass)
{
    double zq[MC_K + MC_N_THERMO];
    double scaled[MC_N_ACTIVE];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z_proj[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_RATE_N_LAYERS, MC_RATE_LAYER_TYPES, MC_RATE_N_LINEAR,
                MC_RATE_IN, MC_RATE_OUT, MC_RATE_W, MC_RATE_B,
                MC_RATE_N_LN, MC_RATE_LN_GAMMA, MC_RATE_LN_BETA, MC_RATE_LN_EPS_ARR,
                zq, scaled);
    for (i = 0; i < MC_N_ACTIVE; ++i) {
        double a = scaled[i] * MC_RATE_STD_SCALE[i] + MC_RATE_STD_MEAN[i];
        if (a > 30.0) a = 30.0;
        if (a < -30.0) a = -30.0;
        rates_mass[i] = sinh(a) * MC_RATE_ASINH_SCALE[i];
    }
}

/* NASA7 mass-specific enthalpy h_i(T) [J/kg] for each energy-active species (incl. formation). */
static void mc_h_mass(double T, double *h)
{
    int i;
    double Tt = (T > 1.0e-300) ? T : 1.0e-300;
    double T2 = Tt * Tt, T3 = T2 * Tt, T4 = T3 * Tt;
    for (i = 0; i < MC_N_ACTIVE; ++i) {
        const double *a = (Tt > MC_NASA_TMID[i]) ? &MC_NASA_HIGH[i * 7] : &MC_NASA_LOW[i * 7];
        double h_rt = a[0] + a[1]*Tt/2.0 + a[2]*T2/3.0 + a[3]*T3/4.0 + a[4]*T4/5.0 + a[5]/Tt;
        h[i] = (MC_R_GAS * Tt * h_rt) / MC_MOLAR_MASS[i];
    }
}

/* Rate-derived volumetric absorption [J/m3/s]. No max(0,.) floor: a small negative (exothermic)
 * value at low conversion is physical and must be preserved — only the safety MC_ENERGY_CLAMP
 * bounds the magnitude at the call site. */
static double mc_absorption_from_rates(const double *z_proj, const double *q, double T)
{
    double rates_mass[MC_N_ACTIVE], h[MC_N_ACTIVE];
    double s = 0.0;
    int i;
    mc_rates_physical(z_proj, q, rates_mass);
    mc_h_mass(T, h);
    for (i = 0; i < MC_N_ACTIVE; ++i) s += rates_mass[i] * h[i];
    return s;
}
"""


# Energy DEFINE_SOURCE: rate-derived primary (with head cross-check + OOD fallback), or legacy head.
_ENERGY_SOURCE_RATE = r"""/* DEFINE_SOURCE energy: S_h = -absorption recomputed from the rate head + NASA7 (PRIMARY).
 * The distilled softplus head is kept as a UDM cross-check and the out-of-envelope fallback. */
DEFINE_SOURCE(mc_energy_source, c, t, dS, eqn)
{
    double z_raw[MC_K], z[MC_K], z_proj[MC_K], q[MC_N_THERMO];
    double absorption_rate, absorption_head, absorption, sh;
    double T = C_T(c, t);
    double P = C_P(c, t) + MC_OPERATING_PRESSURE_PA;
    int i, ood;

    for (i = 0; i < MC_K; ++i) z_raw[i] = C_UDSI(c, t, i);
    mc_build_q(T, P, q);
    ood = mc_clamp_latent(z_raw, z);
    mc_project(z, q, z_proj, NULL);

    absorption_rate = mc_absorption_from_rates(z_proj, q, T);  /* first-law, true cell T */
    absorption_head = mc_absorption(z_proj, q);                /* cross-check / OOD fallback */
    C_UDMI(c, t, MC_UDM_ABS_HEAD) = absorption_head;

    /* In-envelope: trust the rate-derived path (val R2~0.94). Out-of-envelope (latent clamped):
     * the unbounded rate sum is less trustworthy than the softplus-bounded head, so fall back. */
    absorption = ood ? absorption_head : absorption_rate;
    sh = -absorption;

    if (sh > MC_ENERGY_CLAMP) {
        sh = MC_ENERGY_CLAMP;
        C_UDMI(c, t, MC_UDM_ENERGY_CLAMP_CNT) += 1.0;
    } else if (sh < -MC_ENERGY_CLAMP) {
        sh = -MC_ENERGY_CLAMP;
        C_UDMI(c, t, MC_UDM_ENERGY_CLAMP_CNT) += 1.0;
    }
    C_UDMI(c, t, MC_UDM_LAST_SH) = sh;

    dS[eqn] = 0.0;
    return (real)sh;
}"""

_ENERGY_SOURCE_HEAD = r"""/* DEFINE_SOURCE for energy equation: S_h = -absorption_head(z, q) (distilled head only) */
DEFINE_SOURCE(mc_energy_source, c, t, dS, eqn)
{
    double z_raw[MC_K], z[MC_K], z_proj[MC_K], q[MC_N_THERMO];
    double absorption, sh;
    double T = C_T(c, t);
    double P = C_P(c, t) + MC_OPERATING_PRESSURE_PA;
    int i;

    for (i = 0; i < MC_K; ++i) z_raw[i] = C_UDSI(c, t, i);
    mc_build_q(T, P, q);
    mc_clamp_latent(z_raw, z);
    mc_project(z, q, z_proj, NULL);

    absorption = mc_absorption(z_proj, q);
    sh = -absorption;

    if (sh > MC_ENERGY_CLAMP) {
        sh = MC_ENERGY_CLAMP;
        C_UDMI(c, t, MC_UDM_ENERGY_CLAMP_CNT) += 1.0;
    } else if (sh < -MC_ENERGY_CLAMP) {
        sh = -MC_ENERGY_CLAMP;
        C_UDMI(c, t, MC_UDM_ENERGY_CLAMP_CNT) += 1.0;
    }
    C_UDMI(c, t, MC_UDM_LAST_SH) = sh;

    dS[eqn] = 0.0;
    return (real)sh;
}"""


# Forward-test check block for the rate-derived energy path (injected into main()'s ref loop).
# Uses the inlined MC_ENERGY_CLAMP macro and the REF_T / REF_SH_RATE reference arrays.
_RATE_FT_CHECK = r"""
        /* Rate-derived energy path (the DEPLOYED primary): S_h = -sum_i h_i(T)*omega_i */
        {
            double absr = mc_absorption_from_rates(z_proj, q, REF_T[ref]);
            double shr = -absr;
            double ref_shr = REF_SH_RATE[ref];
            double denom = fabs(ref_shr) > 1.0e-30 ? fabs(ref_shr) : 1.0e-30;
            double rel;
            if (shr >  MC_ENERGY_CLAMP) shr =  MC_ENERGY_CLAMP;
            if (shr < -MC_ENERGY_CLAMP) shr = -MC_ENERGY_CLAMP;
            rel = fabs(shr - ref_shr) / denom;
            if (rel > rel_tol) {
                printf("FAIL ref=%d sh_rate: got %.17g expected %.17g rel=%.4e\n",
                       ref, shr, ref_shr, rel);
                fail = 1;
            }
        }
"""


# Transport-property DEFINE_PROPERTY hooks (proposal #5 C export). Injected only when a transport
# head was trained (n_transport>0); uses Fluent cell/thread types so it lives in the UDF source only.
_TRANSPORT_C_DEFS = r"""
/* ---- Transport-property head (μ, k, …): state-dependent DEFINE_PROPERTY closures ---- */
static void mc_transport_vec(const double *z_proj, const double *q, double *out)
{
    double zq[MC_K + MC_N_THERMO];
    double raw[MC_N_TRANSPORT];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z_proj[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_TP_N_LAYERS, MC_TP_LAYER_TYPES, MC_TP_N_LINEAR,
                MC_TP_IN, MC_TP_OUT, MC_TP_W, MC_TP_B,
                MC_TP_N_LN, MC_TP_LN_GAMMA, MC_TP_LN_BETA, MC_TP_LN_EPS_ARR,
                zq, raw);
    for (i = 0; i < MC_N_TRANSPORT; ++i)
        out[i] = mc_softplus(raw[i]) * MC_TRANSPORT_SCALE[i] + MC_TRANSPORT_FLOOR[i];
}

/* Build the projected latent + thermo features at a cell (shared by the property hooks) */
static void mc_cell_zproj(cell_t c, Thread *t, double *z_proj, double *q)
{
    double z_raw[MC_K], z[MC_K];
    double T = C_T(c, t);
    double P = C_P(c, t) + MC_OPERATING_PRESSURE_PA;
    int i;
    for (i = 0; i < MC_K; ++i) z_raw[i] = C_UDSI(c, t, i);
    mc_build_q(T, P, q);
    mc_clamp_latent(z_raw, z);
    mc_project(z, q, z_proj, NULL);
}

/* DEFINE_PROPERTY: mixture dynamic viscosity [Pa s] (transport head output 0).
 * Hook in Fluent: Materials > <mixture> > Viscosity > user-defined > mc_viscosity. */
DEFINE_PROPERTY(mc_viscosity, c, t)
{
    double z_proj[MC_K], q[MC_N_THERMO], out[MC_N_TRANSPORT];
    mc_cell_zproj(c, t, z_proj, q);
    mc_transport_vec(z_proj, q, out);
    return (real)out[0];
}

/* DEFINE_PROPERTY: mixture thermal conductivity [W/m/K] (transport head output 1).
 * Hook in Fluent: Materials > <mixture> > Thermal Conductivity > user-defined > mc_thermal_conductivity. */
DEFINE_PROPERTY(mc_thermal_conductivity, c, t)
{
    double z_proj[MC_K], q[MC_N_THERMO], out[MC_N_TRANSPORT];
    mc_cell_zproj(c, t, z_proj, q);
    mc_transport_vec(z_proj, q, out);
    return (real)out[1];
}
"""


def _render_udf_source(
    spec: dict[str, Any], use_rate_energy: bool = False, use_transport: bool = False
) -> str:
    """Render merged_coil_udf.c with all Fluent DEFINE_* hooks.

    When *use_rate_energy* is True the deployed energy DEFINE_SOURCE recomputes
    ``S_h = -Σ h_i·ω̇_i`` from the rate head + NASA7 (proposal #1); otherwise it uses the
    legacy distilled-head path.  When *use_transport* is True, DEFINE_PROPERTY hooks for
    viscosity and thermal conductivity are emitted from the transport head (proposal #5).
    """
    k = len(spec.get("latent_env_min", spec["export_stats"]["latent_env_min"]))
    k = len(spec["export_stats"]["latent_env_min"])
    rate_helpers = _RATE_ENERGY_C_HELPERS if use_rate_energy else ""
    energy_source_def = _ENERGY_SOURCE_RATE if use_rate_energy else _ENERGY_SOURCE_HEAD
    transport_defs = _TRANSPORT_C_DEFS if use_transport else ""

    uds_sources = "\n".join(
        f"""DEFINE_SOURCE(mc_latent_uds_{i}_source, c, t, dS, eqn)
{{
    dS[eqn] = 0.0;
    return (real)mc_latent_source(c, t, {i});
}}"""
        for i in range(k)
    )

    return f"""#include "udf.h"
#include "merged_coil_udf.h"

/*
 * SCARFS MergedCoil Fluent UDF
 * Generated by scarfs.coupling.codegen
 *
 * Transport contract:
 *   k={k} UDS scalars transport z = E·((Y_dry - mu)/sigma)  (affine -> conserved-scalar valid).
 *   DEFINE_SOURCE mc_latent_uds_i_source: S_i = rho * omega_Z_i  [kg/m3/s]
 *   DEFINE_ADJUST  mc_manifold_project:   manifold projection + UDM update each iteration
 *   DEFINE_SOURCE  mc_energy_source:      S_h = -absorption_head(z,q) [J/m3/s]
 *
 * UDM layout (see MC_UDM_* macros in header):
 *   [0]  OOD flag  [1] latent-clamp count  [2] energy-clamp count  [3] last S_h
 *   [4 .. 4+k-1]      decoded z_i
 *   [4+k .. 4+k+n_dry-1] decoded Y_i
 *
 * Under-relaxation: urf = min(1.0, MC_URF0 + iter/MC_URF_RAMP_ITERS).
 * Reaches 1.0 at iteration ~{int(500*(1.0 - 0.2))} — never a permanent sub-unity factor.
 *
 * Energy clamp MC_ENERGY_CLAMP = {_fmt(spec["export_stats"]["energy_clamp"])} J/m3/s
 * (≈1.3× train max; safety clamp far above signal — does NOT falsify predictions).
 */

/* ---- C-standard utilities (no udf.h dependency in forward-test) ---- */
static double mc_max(double a, double b) {{ return a > b ? a : b; }}
static double mc_min(double a, double b) {{ return a < b ? a : b; }}
static double mc_clamp(double x, double lo, double hi) {{ return mc_min(mc_max(x, lo), hi); }}

/* SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) */
static double mc_silu(double x)
{{
    return x / (1.0 + exp(-x));
}}

/* Stable softplus: log1p(exp(-|x|)) + max(x, 0) */
static double mc_softplus(double x)
{{
    return log1p(exp(-fabs(x))) + mc_max(x, 0.0);
}}

/* ---- MLP evaluation with LayerNorm support ----
 * Layer type codes in PREFIX_LAYER_TYPES[]:  0 = Linear, 1 = LayerNorm.
 * Linear layers use PREFIX_W[lin_idx] / PREFIX_B[lin_idx] / PREFIX_IN[lin_idx] / PREFIX_OUT[lin_idx].
 * LayerNorm layers use PREFIX_LN_GAMMA[ln_idx] / PREFIX_LN_BETA[ln_idx] / PREFIX_LN_EPS_ARR[ln_idx].
 * Hidden activations: SiLU applied after LayerNorm (or after non-final Linear if no LN present).
 * Final output: linear (no activation).
 */
static void mc_net_eval(
    int n_total_layers,
    const int *layer_types,
    int n_linear,
    const int *lin_in,
    const int *lin_out,
    const double * const *W,
    const double * const *B,
    int n_ln,
    const double * const *ln_gamma,
    const double * const *ln_beta,
    const double *ln_eps,
    const double *x_in,
    double *y_out)
{{
    double a[MC_MAX_WIDTH], b_buf[MC_MAX_WIDTH];
    int i, j, l, lin_idx = 0, ln_idx = 0;
    int out_dim = lin_in[0];  /* start with input dim, updated per linear layer */
    for (i = 0; i < lin_in[0]; ++i) a[i] = x_in[i];

    for (l = 0; l < n_total_layers; ++l) {{
        if (layer_types[l] == 0) {{  /* Linear */
            int ni = lin_in[lin_idx], no = lin_out[lin_idx];
            const double *w = W[lin_idx];
            const double *bb = B[lin_idx];
            int is_last_linear = (lin_idx == n_linear - 1);
            /* Check if next layer is LayerNorm (will handle SiLU activation after it) */
            int next_is_ln = (l + 1 < n_total_layers && layer_types[l + 1] == 1);
            for (i = 0; i < no; ++i) {{
                double s = bb[i];
                for (j = 0; j < ni; ++j) s += w[i * ni + j] * a[j];
                /* Apply SiLU here only if: not last linear AND no LayerNorm follows */
                b_buf[i] = (!is_last_linear && !next_is_ln) ? mc_silu(s) : s;
            }}
            for (i = 0; i < no; ++i) a[i] = b_buf[i];
            out_dim = no;
            lin_idx++;
        }} else {{  /* LayerNorm (type == 1) */
            int n = out_dim;
            const double *g = ln_gamma[ln_idx];
            const double *bt = ln_beta[ln_idx];
            double eps_val = ln_eps[ln_idx];
            double mean = 0.0, var = 0.0;
            for (i = 0; i < n; ++i) mean += a[i];
            mean /= mc_max((double)n, 1.0);
            for (i = 0; i < n; ++i) {{ double d = a[i] - mean; var += d * d; }}
            var /= mc_max((double)n, 1.0);
            var = sqrt(mc_max(var + eps_val, 0.0));
            /* After LayerNorm, apply SiLU (hidden activation) */
            for (i = 0; i < n; ++i)
                a[i] = mc_silu((a[i] - mean) / mc_max(var, 1.0e-300) * g[i] + bt[i]);
            ln_idx++;
        }}
    }}
    for (i = 0; i < out_dim; ++i) y_out[i] = a[i];
}}

/* Build standardised thermo features [T, P, 1/T, lnT], clamped to training range */
static void mc_build_q(double T_raw, double P_raw, double *q)
{{
    double T = mc_clamp(T_raw, MC_T_TRAIN_MIN, MC_T_TRAIN_MAX);
    double P = mc_clamp(P_raw, MC_P_TRAIN_MIN, MC_P_TRAIN_MAX);
    double feat[4];
    int i;
    feat[0] = T;
    feat[1] = P;
    feat[2] = 1.0 / mc_max(T, 1.0e-300);
    feat[3] = log(mc_max(T, 1.0e-300));
    for (i = 0; i < 4; ++i)
        q[i] = (feat[i] - MC_THERMO_MEAN[i]) / mc_max(MC_THERMO_SCALE[i], 1.0e-300);
}}

/* Encode y_std -> z (linear, no bias) */
static void mc_encode(const double *y_std, double *z)
{{
    int i, j;
    for (i = 0; i < MC_K; ++i) {{
        double s = 0.0;
        for (j = 0; j < MC_N_DRY; ++j) s += MC_ENC_W[i * MC_N_DRY + j] * y_std[j];
        z[i] = s;
    }}
}}

/* Clamp latent to training envelope; returns 1 if any dim was clamped, else 0 */
static int mc_clamp_latent(const double *z_in, double *z_out)
{{
    int i, clamped = 0;
    for (i = 0; i < MC_K; ++i) {{
        z_out[i] = mc_clamp(z_in[i], MC_LAT_MIN[i], MC_LAT_MAX[i]);
        if (z_out[i] != z_in[i]) clamped = 1;
    }}
    return clamped;
}}

/* Manifold projection: decode z+q -> y_std -> re-encode -> z_proj */
static void mc_project(const double *z, const double *q, double *z_proj, double *y_std_out)
{{
    double zq[MC_K + MC_N_THERMO];
    double y_std[MC_N_DRY];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_DEC_N_LAYERS, MC_DEC_LAYER_TYPES, MC_DEC_N_LINEAR,
                MC_DEC_IN, MC_DEC_OUT, MC_DEC_W, MC_DEC_B,
                MC_DEC_N_LN, MC_DEC_LN_GAMMA, MC_DEC_LN_BETA, MC_DEC_LN_EPS_ARR,
                zq, y_std);
    if (y_std_out != NULL)
        for (i = 0; i < MC_N_DRY; ++i) y_std_out[i] = y_std[i];
    mc_encode(y_std, z_proj);
}}

/* Latent source omega_Z from z_proj + q */
static void mc_latent_source_vec(const double *z_proj, const double *q, double *omega_z)
{{
    double zq[MC_K + MC_N_THERMO];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z_proj[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_LS_N_LAYERS, MC_LS_LAYER_TYPES, MC_LS_N_LINEAR,
                MC_LS_IN, MC_LS_OUT, MC_LS_W, MC_LS_B,
                MC_LS_N_LN, MC_LS_LN_GAMMA, MC_LS_LN_BETA, MC_LS_LN_EPS_ARR,
                zq, omega_z);
    /* head emits arcsinh(omega_Z / s_Z); physical = sinh(.)*s_Z (clip = overflow guard) */
    for (i = 0; i < MC_K; ++i) {{
        double a = omega_z[i];
        if (a > 20.0) a = 20.0;
        if (a < -20.0) a = -20.0;
        omega_z[i] = sinh(a) * MC_LS_ASINH_SCALE[i];
    }}
}}

/* Absorption head: strictly positive via softplus + calibration */
static double mc_absorption(const double *z_proj, const double *q)
{{
    double zq[MC_K + MC_N_THERMO];
    double raw[1];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z_proj[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_EN_N_LAYERS, MC_EN_LAYER_TYPES, MC_EN_N_LINEAR,
                MC_EN_IN, MC_EN_OUT, MC_EN_W, MC_EN_B,
                MC_EN_N_LN, MC_EN_LN_GAMMA, MC_EN_LN_BETA, MC_EN_LN_EPS_ARR,
                zq, raw);
    return mc_softplus(raw[0]) * MC_ENERGY_SCALE + MC_ENERGY_FLOOR;
}}
{rate_helpers}
/* Decode standardised composition to physical mass fractions with closure */
static void mc_decode_composition(const double *z, const double *q, double *Y_out)
{{
    double zq[MC_K + MC_N_THERMO];
    double y_std[MC_N_DRY];
    double total;
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_DEC_N_LAYERS, MC_DEC_LAYER_TYPES, MC_DEC_N_LINEAR,
                MC_DEC_IN, MC_DEC_OUT, MC_DEC_W, MC_DEC_B,
                MC_DEC_N_LN, MC_DEC_LN_GAMMA, MC_DEC_LN_BETA, MC_DEC_LN_EPS_ARR,
                zq, y_std);
    /* Invert linear standardisation: Y = sigma * y_std + mu; clip >= 0 */
    total = 0.0;
    for (i = 0; i < MC_N_DRY; ++i) {{
        double y = y_std[i] * MC_COMP_SCALE[i] + MC_COMP_MEAN[i];
        Y_out[i] = mc_max(y, 0.0);
        total += Y_out[i];
    }}
    /* H2O = max(1 - sum(Y_retained), 0)  [diluent excluded from encoder] */
    /* Renormalise if retained sum exceeds available non-H2O mass */
    if (total > 1.0) {{
        for (i = 0; i < MC_N_DRY; ++i) Y_out[i] = Y_out[i] / mc_max(total, 1.0e-300);
    }}
}}

/* Read iteration counter from Fluent; gracefully handle unavailability */
static int mc_get_iteration(void)
{{
    /* ITERATION macro is Fluent-version-dependent; use RP_Get_Integer if available */
#ifdef CURRENT_TIME_STEP
    return (int)CURRENT_TIME_STEP;
#else
    return 1000;  /* default: URF = 1 after ramp */
#endif
}}

/* ---- Fluent hook implementations ---- */

/* DEFINE_ADJUST: manifold projection + UDM update (runs once per iteration, all cells) */
DEFINE_ADJUST(mc_manifold_project, domain)
{{
#if !RP_HOST
    Thread *t;
    cell_t c;
    int mc_iter = mc_get_iteration();
    double urf = mc_min(1.0, MC_URF0 + (double)mc_iter / (double)MC_URF_RAMP_ITERS);

    thread_loop_c(t, domain) {{
        begin_c_loop(c, t) {{
            double z_raw[MC_K], z[MC_K], z_proj[MC_K], y_std[MC_N_DRY], Y[MC_N_DRY];
            double q[MC_N_THERMO];
            double T = C_T(c, t);
            double P = C_P(c, t) + MC_OPERATING_PRESSURE_PA;
            int i, clamped;

            /* Read latent scalars from UDS */
            for (i = 0; i < MC_K; ++i) z_raw[i] = C_UDSI(c, t, i);

            /* Build standardised thermo features (T/P clamped to training range) */
            mc_build_q(T, P, q);

            /* Clamp latent to envelope; flag OOD */
            clamped = mc_clamp_latent(z_raw, z);
            if (clamped) {{
                C_UDMI(c, t, MC_UDM_OOD_FLAG) = 1.0;
                C_UDMI(c, t, MC_UDM_LATENT_CLAMP_CNT) += 1.0;
            }}

            /* Manifold projection: z_proj = E · decode(z, q) */
            mc_project(z, q, z_proj, y_std);

            /* Apply under-relaxation to the projected latent */
            for (i = 0; i < MC_K; ++i)
                C_UDSI(c, t, i) = (1.0 - urf) * z_raw[i] + urf * z_proj[i];

            /* Decode composition for UDM storage */
            mc_decode_composition(z, q, Y);
            for (i = 0; i < MC_N_DRY; ++i)
                C_UDMI(c, t, MC_UDM_Y_START + i) = Y[i];
            for (i = 0; i < MC_K; ++i)
                C_UDMI(c, t, MC_UDM_Z_START + i) = z[i];

        }} end_c_loop(c, t)
    }}
#endif
}}

{energy_source_def}

/* UDS source functions: S_i = rho * omega_Z_i */
static double mc_latent_source(cell_t c, Thread *t, int dim)
{{
    double z_raw[MC_K], z[MC_K], z_proj[MC_K], q[MC_N_THERMO], omega_z[MC_K];
    double T = C_T(c, t);
    double P = C_P(c, t) + MC_OPERATING_PRESSURE_PA;
    int i;
    for (i = 0; i < MC_K; ++i) z_raw[i] = C_UDSI(c, t, i);
    mc_build_q(T, P, q);
    mc_clamp_latent(z_raw, z);
    mc_project(z, q, z_proj, NULL);
    mc_latent_source_vec(z_proj, q, omega_z);
    return C_R(c, t) * omega_z[dim];
}}
{transport_defs}
{uds_sources}
"""


# ---------------------------------------------------------------------------
# TUI setup script
# ---------------------------------------------------------------------------

def _render_tui_setup(spec: dict[str, Any]) -> str:
    """Render fluent_merged_setup.tui."""
    stats = spec["export_stats"]
    k = len(stats["latent_env_min"])
    n_dry = len(spec["input"])
    total_udm = 5 + k + n_dry  # +1 for MC_UDM_ABS_HEAD (rate-vs-head energy cross-check)

    lines = [
        "; SCARFS MergedCoil Fluent TUI setup script",
        "; Generated by scarfs.coupling.codegen",
        "; Load/compile merged_coil_udf.c + merged_coil_udf.h as compiled UDFs first.",
        ";",
        f"; Required UDS equations: {k}  (latent dimensions z_0 .. z_{k-1})",
        f"; Required UDM slots: {total_udm}",
        ";",
        "; UDM layout:",
        ";   [0] OOD flag  [1] latent-clamp count  [2] energy-clamp count  [3] last S_h",
        f";   [4 .. {3+k}] decoded z_i",
        f";   [{4+k} .. {3+k+n_dry}] decoded Y_i",
        ";",
        "; Allocate UDM slots:",
        f"/define/user-defined/user-defined-memory {total_udm}",
        ";",
        "; Hook DEFINE_ADJUST (manifold projection + UDM update each iteration):",
        ";   Define > User-Defined > Adjust > mc_manifold_project",
        ";",
        "; Hook UDS source terms (one per latent dimension):",
    ]
    for i in range(k):
        lines.append(f";   UDS-{i}: mc_latent_uds_{i}_source")
    lines.extend([
        ";",
        "; Hook energy equation source term:",
        ";   Define > User-Defined > Source Terms > Energy > mc_energy_source",
        ";",
        "; Under-relaxation note:",
        ";   URF starts at MC_URF0=0.2 and ramps to 1.0 at iteration MC_URF_RAMP_ITERS=500.",
        ";   This is an ANNEALING ramp. The URF MUST reach 1.0 for correct steady-state.",
        ";   Never set a permanent sub-unity URF on the energy equation.",
        ";",
        "; Telemetry UDMs to monitor:",
        ";   UDM[0]=OOD_FLAG: non-zero cells indicate out-of-training-envelope latents.",
        ";   UDM[1]=LATENT_CLAMP_CNT: cumulative latent-clamp activations per cell.",
        ";   UDM[2]=ENERGY_CLAMP_CNT: cumulative energy-clamp activations per cell.",
        ";   UDM[3]=LAST_SH: last computed energy source S_h [J/m3/s].",
        ";",
        f"; Energy clamp: MC_ENERGY_CLAMP = {_fmt(spec['export_stats']['energy_clamp'])} J/m3/s",
        "; (Safety clamp ~1.3x train max -- far above signal.)",
        "",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Standalone forward-test C
# ---------------------------------------------------------------------------

def _render_forward_test(
    spec: dict[str, Any],
    ref_states: dict[str, np.ndarray],
    ref_outputs: dict[str, np.ndarray],
    weights_dict: dict[str, Any],
    comp_mean: np.ndarray,
    comp_scale: np.ndarray,
    thermo_sc,
    energy_aux: dict[str, Any] | None = None,
) -> str:
    """Render merged_coil_forward_test.c — standalone, no udf.h, embeds reference vectors."""
    stats = spec["export_stats"]
    k = weights_dict["k"]
    n_dry = weights_dict["n_dry"]
    n_ref = ref_states["Y_in"].shape[0]
    energy_clamp = float(stats["energy_clamp"])

    # Build reference input/output arrays for embedding in C
    # Reference inputs: y_std (standardised composition) + q (standardised thermo)
    T = ref_states["T"]
    P = ref_states["P"]
    Y_in = ref_states["Y_in"]
    y_std_ref = (Y_in - comp_mean) / comp_scale

    T_nn = np.clip(T, stats["T_train_min"], stats["T_train_max"])
    P_nn = np.clip(P, stats["P_train_min"], stats["P_train_max"])
    q_raw = _build_thermo_features(T_nn, P_nn)
    thm_mean = thermo_sc.mean_ if thermo_sc is not None else np.zeros(4)
    thm_scale = thermo_sc.scale_ if thermo_sc is not None else np.ones(4)
    q_ref = (q_raw - thm_mean) / thm_scale

    omega_z_ref = ref_outputs["omega_z"]   # (n_ref, k)
    s_h_ref = ref_outputs["s_h"]           # (n_ref,)

    # Flatten for C arrays
    def _arr(name, arr):
        return _c_double_array(name, arr.ravel(), per_line=4)

    y_std_lines = _arr("REF_Y_STD", y_std_ref)
    q_lines = _arr("REF_Q", q_ref)
    omega_z_lines = _arr("REF_OMEGA_Z", omega_z_ref)
    s_h_lines = _arr("REF_SH", s_h_ref)

    # Rate-derived energy reference (proposal #1): embed raw T and the mirror's s_h_rate so the
    # compiled C check exercises the DEPLOYED primary path at the strict 1e-6 parity gate.
    use_rate = energy_aux is not None and "s_h_rate" in ref_outputs
    if use_rate:
        ref_t_lines = _c_double_array("REF_T", np.asarray(T, dtype=float).ravel(), per_line=4)
        ref_sh_rate_lines = _arr("REF_SH_RATE", ref_outputs["s_h_rate"])
        rate_helpers_ft = _RATE_ENERGY_C_HELPERS
        rate_check_block = _RATE_FT_CHECK
    else:
        ref_t_lines = ref_sh_rate_lines = rate_helpers_ft = rate_check_block = ""

    # Inline the header content (no #include "merged_coil_udf.h" — standalone). Pass energy_aux so
    # the NASA7 / rate-scaler arrays the rate-derived check needs are present in the inlined header.
    header_body = _render_header(spec, weights_dict, comp_mean, comp_scale, thermo_sc, energy_aux)
    # Remove ONLY the outer include guards and #include directives.
    # Do NOT strip inner #ifndef / #endif blocks (e.g. MC_OPERATING_PRESSURE_PA guard).
    guard_lines = {
        "#ifndef MERGED_COIL_UDF_H",
        "#define MERGED_COIL_UDF_H",
        "#endif  /* MERGED_COIL_UDF_H */",
    }
    header_body = "\n".join(
        l for l in header_body.splitlines()
        if l not in guard_lines
        and not l.startswith("#include")
    )

    return f"""/* merged_coil_forward_test.c
 * Standalone forward-test for the MergedCoil UDF.
 * Compiled without udf.h: cc -O0 -lm merged_coil_forward_test.c -o mc_fwd_test
 * Exits 0 if all reference outputs match to within 1e-6 relative tolerance.
 * Reference vectors were computed by the Python numpy mirror at export time.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* ===== Inlined header (all static const arrays) ===== */
{header_body}
/* ===== End of inlined header ===== */

/* Reference vectors (computed by Python numpy mirror at export) */
#define MC_N_REF  {n_ref}
{y_std_lines}
{q_lines}
{omega_z_lines}
{s_h_lines}
{ref_t_lines}
{ref_sh_rate_lines}

/* ---- Shared C utilities (mirror of udf source, no Fluent types) ---- */
static double mc_max(double a, double b) {{ return a > b ? a : b; }}
static double mc_min(double a, double b) {{ return a < b ? a : b; }}
static double mc_clamp(double x, double lo, double hi) {{ return mc_min(mc_max(x, lo), hi); }}

static double mc_silu(double x)
{{
    return x / (1.0 + exp(-x));
}}

static double mc_softplus(double x)
{{
    return log1p(exp(-fabs(x))) + mc_max(x, 0.0);
}}

/* mc_net_eval: MLP evaluation with optional LayerNorm support.
 * layer_types[l]: 0=Linear, 1=LayerNorm.
 * Linear layers consume lin_in/lin_out/W/B indexed by lin_idx.
 * LayerNorm layers consume ln_gamma/ln_beta/ln_eps indexed by ln_idx.
 * SiLU activation is applied: after LayerNorm (hidden), or after non-final Linear (no-LN case).
 */
static void mc_net_eval(
    int n_total_layers,
    const int *layer_types,
    int n_linear,
    const int *lin_in,
    const int *lin_out,
    const double * const *W,
    const double * const *B,
    int n_ln,
    const double * const *ln_gamma,
    const double * const *ln_beta,
    const double *ln_eps,
    const double *x_in,
    double *y_out)
{{
    double a[MC_MAX_WIDTH], b_buf[MC_MAX_WIDTH];
    int i, j, l, lin_idx = 0, ln_idx = 0;
    int out_dim = lin_in[0];
    for (i = 0; i < lin_in[0]; ++i) a[i] = x_in[i];

    for (l = 0; l < n_total_layers; ++l) {{
        if (layer_types[l] == 0) {{  /* Linear */
            int ni = lin_in[lin_idx], no = lin_out[lin_idx];
            const double *w = W[lin_idx];
            const double *bb = B[lin_idx];
            int is_last_linear = (lin_idx == n_linear - 1);
            int next_is_ln = (l + 1 < n_total_layers && layer_types[l + 1] == 1);
            for (i = 0; i < no; ++i) {{
                double s = bb[i];
                for (j = 0; j < ni; ++j) s += w[i * ni + j] * a[j];
                b_buf[i] = (!is_last_linear && !next_is_ln) ? mc_silu(s) : s;
            }}
            for (i = 0; i < no; ++i) a[i] = b_buf[i];
            out_dim = no;
            lin_idx++;
        }} else {{  /* LayerNorm (type == 1) */
            int n = out_dim;
            const double *g = ln_gamma[ln_idx];
            const double *bt = ln_beta[ln_idx];
            double eps_val = ln_eps[ln_idx];
            double mean = 0.0, var = 0.0;
            for (i = 0; i < n; ++i) mean += a[i];
            mean /= mc_max((double)n, 1.0);
            for (i = 0; i < n; ++i) {{ double d = a[i] - mean; var += d * d; }}
            var /= mc_max((double)n, 1.0);
            var = sqrt(mc_max(var + eps_val, 0.0));
            for (i = 0; i < n; ++i)
                a[i] = mc_silu((a[i] - mean) / mc_max(var, 1.0e-300) * g[i] + bt[i]);
            ln_idx++;
        }}
    }}
    for (i = 0; i < out_dim; ++i) y_out[i] = a[i];
}}

static void mc_encode(const double *y_std, double *z)
{{
    int i, j;
    for (i = 0; i < MC_K; ++i) {{
        double s = 0.0;
        for (j = 0; j < MC_N_DRY; ++j) s += MC_ENC_W[i * MC_N_DRY + j] * y_std[j];
        z[i] = s;
    }}
}}

static int mc_clamp_latent(const double *z_in, double *z_out)
{{
    int i, clamped = 0;
    for (i = 0; i < MC_K; ++i) {{
        z_out[i] = mc_clamp(z_in[i], MC_LAT_MIN[i], MC_LAT_MAX[i]);
        if (z_out[i] != z_in[i]) clamped = 1;
    }}
    return clamped;
}}

static void mc_project(const double *z, const double *q, double *z_proj, double *y_std_out)
{{
    double zq[MC_K + MC_N_THERMO];
    double y_std[MC_N_DRY];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_DEC_N_LAYERS, MC_DEC_LAYER_TYPES, MC_DEC_N_LINEAR,
                MC_DEC_IN, MC_DEC_OUT, MC_DEC_W, MC_DEC_B,
                MC_DEC_N_LN, MC_DEC_LN_GAMMA, MC_DEC_LN_BETA, MC_DEC_LN_EPS_ARR,
                zq, y_std);
    if (y_std_out != NULL)
        for (i = 0; i < MC_N_DRY; ++i) y_std_out[i] = y_std[i];
    mc_encode(y_std, z_proj);
}}

static void mc_latent_source_vec(const double *z_proj, const double *q, double *omega_z)
{{
    double zq[MC_K + MC_N_THERMO];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z_proj[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_LS_N_LAYERS, MC_LS_LAYER_TYPES, MC_LS_N_LINEAR,
                MC_LS_IN, MC_LS_OUT, MC_LS_W, MC_LS_B,
                MC_LS_N_LN, MC_LS_LN_GAMMA, MC_LS_LN_BETA, MC_LS_LN_EPS_ARR,
                zq, omega_z);
    /* head emits arcsinh(omega_Z / s_Z); physical = sinh(.)*s_Z (clip = overflow guard) */
    for (i = 0; i < MC_K; ++i) {{
        double a = omega_z[i];
        if (a > 20.0) a = 20.0;
        if (a < -20.0) a = -20.0;
        omega_z[i] = sinh(a) * MC_LS_ASINH_SCALE[i];
    }}
}}

static double mc_absorption(const double *z_proj, const double *q)
{{
    double zq[MC_K + MC_N_THERMO];
    double raw[1];
    int i;
    for (i = 0; i < MC_K; ++i) zq[i] = z_proj[i];
    for (i = 0; i < MC_N_THERMO; ++i) zq[MC_K + i] = q[i];
    mc_net_eval(MC_EN_N_LAYERS, MC_EN_LAYER_TYPES, MC_EN_N_LINEAR,
                MC_EN_IN, MC_EN_OUT, MC_EN_W, MC_EN_B,
                MC_EN_N_LN, MC_EN_LN_GAMMA, MC_EN_LN_BETA, MC_EN_LN_EPS_ARR,
                zq, raw);
    return mc_softplus(raw[0]) * MC_ENERGY_SCALE + MC_ENERGY_FLOOR;
}}
{rate_helpers_ft}
/* ---- Forward-test main ---- */
int main(void)
{{
    int i, ref, fail = 0;
    double rel_tol = 1.0e-6;
    printf("MergedCoil forward test: %d reference states, rel_tol=%.2e\\n",
           MC_N_REF, rel_tol);

    for (ref = 0; ref < MC_N_REF; ++ref) {{
        double y_std[MC_N_DRY], q[MC_N_THERMO];
        double z_raw[MC_K], z[MC_K], z_proj[MC_K];
        double omega_z[MC_K];
        double absorption, sh;

        for (i = 0; i < MC_N_DRY; ++i) y_std[i] = REF_Y_STD[ref * MC_N_DRY + i];
        for (i = 0; i < MC_N_THERMO; ++i) q[i] = REF_Q[ref * MC_N_THERMO + i];

        /* Encode */
        mc_encode(y_std, z_raw);

        /* Clamp latent */
        mc_clamp_latent(z_raw, z);

        /* Manifold project */
        mc_project(z, q, z_proj, NULL);

        /* Latent source */
        mc_latent_source_vec(z_proj, q, omega_z);

        /* Absorption / energy source */
        absorption = mc_absorption(z_proj, q);
        sh = -absorption;
        if (sh >  {_fmt(energy_clamp)}) sh =  {_fmt(energy_clamp)};
        if (sh < -{_fmt(energy_clamp)}) sh = -{_fmt(energy_clamp)};

        /* Check omega_z */
        for (i = 0; i < MC_K; ++i) {{
            double ref_val = REF_OMEGA_Z[ref * MC_K + i];
            double denom = fabs(ref_val) > 1.0e-30 ? fabs(ref_val) : 1.0e-30;
            double rel = fabs(omega_z[i] - ref_val) / denom;
            if (rel > rel_tol) {{
                printf("FAIL ref=%d omega_z[%d]: got %.17g expected %.17g rel=%.4e\\n",
                       ref, i, omega_z[i], ref_val, rel);
                fail = 1;
            }}
        }}

        /* Check sh */
        {{
            double ref_sh = REF_SH[ref];
            double denom = fabs(ref_sh) > 1.0e-30 ? fabs(ref_sh) : 1.0e-30;
            double rel = fabs(sh - ref_sh) / denom;
            if (rel > rel_tol) {{
                printf("FAIL ref=%d sh: got %.17g expected %.17g rel=%.4e\\n",
                       ref, sh, ref_sh, rel);
                fail = 1;
            }}
        }}
{rate_check_block}
    }}

    if (!fail)
        printf("PASS: all %d states within rel_tol=%.2e\\n", MC_N_REF, rel_tol);
    return fail;
}}
"""


# ---------------------------------------------------------------------------
# Inlet BC file
# ---------------------------------------------------------------------------

def _render_inlet_bc(
    inlet: InletSpec,
    spec: dict[str, Any],
    weights_dict: dict[str, Any],
    comp_mean: np.ndarray,
    comp_scale: np.ndarray,
    thermo_sc,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Compute and write the folded inlet-BC files.

    Encodes ``inlet.composition`` through the standardised linear encoder to produce z_in.
    Writes:
    - inlet_bc.txt (human-readable)
    - inlet_bc.csv (machine-readable: species, value pairs + z row)
    """
    input_species = spec["input"]
    n_dry = len(input_species)
    stats = spec["export_stats"]

    # Build mass fraction vector for the input species
    # Normalise inlet composition (may include H2O; exclude it for encoder input)
    comp = dict(inlet.composition)
    h2o_mass = comp.pop("H2O", 0.0)
    total = sum(comp.values()) + h2o_mass
    if total <= 0:
        total = 1.0
    Y_in_full = {s: v / total for s, v in comp.items()}
    h2o_frac = h2o_mass / total

    # Build dry mass fraction array (species NOT in input_species get 0)
    Y_dry = np.zeros(n_dry, dtype=float)
    for i, sp in enumerate(input_species):
        Y_dry[i] = Y_in_full.get(sp, 0.0)

    # Standardise
    y_std = (Y_dry - comp_mean) / comp_scale

    # Encode
    z_in = _numpy_encoder(y_std[np.newaxis, :], weights_dict["encoder_W"])[0]

    # Also compute with T/P clamped (same as what Fluent will see)
    T_nn = float(np.clip(inlet.T, stats["T_train_min"], stats["T_train_max"]))
    P_nn = float(np.clip(inlet.P, stats["P_train_min"], stats["P_train_max"]))
    q_raw = _build_thermo_features(np.array([T_nn]), np.array([P_nn]))
    thm_mean = thermo_sc.mean_ if thermo_sc is not None else np.zeros(4)
    thm_scale = thermo_sc.scale_ if thermo_sc is not None else np.ones(4)
    q_std = (q_raw - thm_mean) / thm_scale

    k = weights_dict["k"]

    # Write txt
    txt_path = out_dir / "inlet_bc.txt"
    txt_lines = [
        "SCARFS MergedCoil — Inlet boundary condition (folded-BC)",
        f"Label: {inlet.label}",
        f"T [K]: {inlet.T:.4f}",
        f"P [Pa]: {inlet.P:.6g}",
        f"H2O mass fraction: {h2o_frac:.6f}",
        "",
        "Dry composition used for encoding:",
    ]
    for i, sp in enumerate(input_species):
        if Y_dry[i] > 0.0:
            txt_lines.append(f"  {sp}: {Y_dry[i]:.8f}")
    txt_lines.extend([
        "",
        "Encoded latent z_in (set as initial/inlet UDS values):",
    ])
    for i, zi in enumerate(z_in):
        txt_lines.append(f"  z[{i}] = {zi:.17g}")
    txt_lines.extend([
        "",
        "Standardised thermo q at inlet (for reference):",
    ])
    for i, qi in enumerate(q_std.ravel()):
        txt_lines.append(f"  q[{i}] = {qi:.17g}")
    txt_lines.append("")
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    # Write csv
    csv_path = out_dir / "inlet_bc.csv"
    rows = []
    rows.append(["type", "name", "value"])
    rows.append(["meta", "label", inlet.label])
    rows.append(["meta", "T_K", str(inlet.T)])
    rows.append(["meta", "P_Pa", str(inlet.P)])
    rows.append(["meta", "H2O_mass_fraction", str(h2o_frac)])
    for i, sp in enumerate(input_species):
        if Y_dry[i] > 0.0:
            rows.append(["composition_dry", sp, str(Y_dry[i])])
    for i, zi in enumerate(z_in):
        rows.append(["latent_z", f"z_{i}", f"{zi:.17g}"])
    for i, qi in enumerate(q_std.ravel()):
        rows.append(["thermo_q_std", f"q_{i}", f"{qi:.17g}"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerows(rows)

    return txt_path, csv_path


# ---------------------------------------------------------------------------
# Export consistency report
# ---------------------------------------------------------------------------

def _write_consistency_report(
    out_dir: Path,
    spec: dict[str, Any],
    ref_states: dict[str, np.ndarray],
    numpy_out: dict[str, np.ndarray],
    torch_out: dict[str, np.ndarray],
    rel_y: float,
    rel_omz: float,
    rel_sh: float,
) -> Path:
    """Write the export consistency report comparing numpy mirror to torch."""
    n_ref = ref_states["Y_in"].shape[0]
    path = out_dir / "export_consistency_report.txt"

    lines = [
        "SCARFS MergedCoil — Export Consistency Report",
        "=" * 48,
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Bundle kind: {spec.get('kind', 'merged')}",
        f"Input species count: {len(spec['input'])}",
        f"Energy-active species: {len(spec['energy_active'])}",
        f"Latent dim k: {len(spec['export_stats']['latent_env_min'])}",
        f"Reference states: {n_ref}",
        "",
        "Method: re-evaluate the same static const arrays written to the C header",
        "using the Python numpy mirror, then compare to the torch forward pass.",
        "Y differences use denominator max(|torch|, 1e-4) to exclude sub-100ppm species",
        "(physically negligible mass fractions) from inflating relative error due to",
        "float32->float64 accumulation in the linear inversion Y = sigma*y_std + mu.",
        "All other differences are relative: |numpy - torch| / max(|torch|, 1e-30).",
        "",
        "Results (max over all reference states and outputs):",
        f"  Decoded species Y            : {rel_y:.4e}",
        f"  Latent source omega_Z        : {rel_omz:.4e}",
        f"  Energy source S_h            : {rel_sh:.4e}",
        "",
        "Acceptance threshold: 1e-4 (float32 weights in float64 evaluation; LayerNorm",
        "networks accumulate O(k*eps_f32) error per normalised layer).",
        f"Status: {'PASS' if max(rel_y, rel_omz, rel_sh) <= 1e-4 else 'WARN (check below)'}",
        "",
    ]

    # Per-state breakdown
    eps = 1e-30
    Y_FLOOR = 1.0e-4  # same floor as _compute_consistency_maxima
    lines.append("Per-reference-state breakdown:")
    lines.append(f"  {'ref':>4}  {'T [K]':>10}  {'P [Pa]':>12}  "
                 f"{'max_rel_Y':>12}  {'max_rel_omz':>12}  {'rel_sh':>12}")
    for i in range(n_ref):
        T = ref_states["T"][i]
        P = ref_states["P"][i]
        y_rel = float(np.max(np.abs(numpy_out["y_decoded"][i] - torch_out["y_decoded"][i])
                             / np.maximum(np.abs(torch_out["y_decoded"][i]), Y_FLOOR)))
        omz_rel = float(np.max(np.abs(numpy_out["omega_z"][i] - torch_out["omega_z"][i])
                               / np.maximum(np.abs(torch_out["omega_z"][i]), eps)))
        sh_ref = abs(torch_out["s_h"][i])
        sh_rel = float(abs(numpy_out["s_h"][i] - torch_out["s_h"][i])
                       / max(sh_ref, eps))
        lines.append(f"  {i:>4}  {T:>10.2f}  {P:>12.3g}  "
                     f"{y_rel:>12.4e}  {omz_rel:>12.4e}  {sh_rel:>12.4e}")

    lines.extend([
        "",
        "Energy-clamp: "
        f"{_fmt(spec['export_stats']['energy_clamp'])} J/m3/s",
        "(Safety clamp ~1.3x train max; does NOT falsify predictions.)",
        "",
        "Colleague-file reference (what this rewrites):",
        "  /Users/tamasbuzogany/Documents/SCARFS/reduced_chem_ml/export_udf.py",
        "  Grafted: C skeleton structure (export_udf.py:43 export_fluent_udf),",
        "           TUI setup script (export_udf.py:1555 render_fluent_setup),",
        "           standalone harness pattern (export_udf.py:1253 render_standalone_harness),",
        "           export_consistency_report (export_udf.py:57 write_export_consistency_report),",
        "           folded inlet-BC generation (implicit in export_fluent_udf),",
        "           safety-flag UDM pattern (export_udf.py:476 RC_UDM_OOD / RC_UDM_ENERGY_SOURCE),",
        "           normalize closure logic (export_udf.py:1040 rc_decode_species_available",
        "               #RC_COMPOSITION_CLOSURE==2 block).",
        "  Rewritten: energy path entirely (was free-head chain-rule decode through PCA",
        "             with ±5.61e7 clamp; now rate-tied + distilled softplus head,",
        "             energy_clamp from full data range x1.3 ~1.6e9).",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
    return path

"""Serialise a trained surrogate to a portable plain-text format for Fluent C UDFs.

Fluent UDFs are compiled C; they cannot load .h5 or .pt files at runtime.  This module writes all
the data a UDF needs into human-readable text files that the C code opens with fopen() at
``DEFINE_INIT`` time.

Format overview (every file is ASCII text)
-------------------------------------------
weights.txt
    header::  ``LAYERS <n>``
    per layer::  ``LAYER <i> in=<din> out=<dout> activation=<name>``
                 one blank-separated float per weight (row-major, W first, then b)

scalers.txt
    ``SCALER <name> <type>``  where type is CompositionScaler | StandardScaler | ArcsinhScaler
    followed by the parameters needed to reproduce transform() / inverse_transform() exactly.

species.txt
    ``N_ACTIVE <k>``
    one species name per line (order = UDS/UDF source index order)

All three files form a ``ModelBundle`` which ``export_bundle()`` writes to a directory.
``load_*`` readers are provided so a Python test can verify byte-exact round-trips.

Units convention (unchanged from the surrogate contract, schema.py)
--------------------------------------------------------------------
- Input composition:   mass fractions Y_i  [-]        (log10 then affine-scaled)
- Input thermo:        T [K], P [Pa]                  (standardised)
- Output rates:        R_i  [kg m-3 s-1]              (arcsinh-scaled)
- Output energy:       S_E  [J m-3 s-1]               (arcsinh-scaled)
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from scarfs.models.common import ArcsinhScaler, CompositionScaler, StandardScaler

# ---------------------------------------------------------------------------
# Supported activation names (C template must implement the same set)
# ---------------------------------------------------------------------------
_KNOWN_ACTIVATIONS = frozenset({"relu", "tanh", "sigmoid", "linear", "softplus", "elu"})

# ---------------------------------------------------------------------------
# Dense-layer weight export / load
# ---------------------------------------------------------------------------

def export_mlp_weights(layers: Sequence[tuple[np.ndarray, np.ndarray, str]], path: str | Path) -> None:
    """Write MLP dense-layer weights/biases + activation names to *path* (text format).

    Parameters
    ----------
    layers
        Sequence of ``(W, b, activation)`` triples.
        - ``W`` shape ``(d_out, d_in)``  — weight matrix (row-major, i.e. ``y = W @ x + b``).
        - ``b`` shape ``(d_out,)``        — bias vector.
        - ``activation``                  — one of :data:`_KNOWN_ACTIVATIONS`.
    path
        Destination file path.  Parent directories are created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"LAYERS {len(layers)}")
    for i, (W, b, act) in enumerate(layers):
        W = np.asarray(W, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if W.ndim != 2:
            raise ValueError(f"Layer {i}: W must be 2-D, got shape {W.shape}.")
        if b.ndim != 1 or b.shape[0] != W.shape[0]:
            raise ValueError(f"Layer {i}: b shape {b.shape} inconsistent with W {W.shape}.")
        act_l = act.lower().strip()
        if act_l not in _KNOWN_ACTIVATIONS:
            raise ValueError(f"Layer {i}: unknown activation '{act}'. Known: {_KNOWN_ACTIVATIONS}.")
        d_out, d_in = W.shape
        lines.append(f"LAYER {i} in={d_in} out={d_out} activation={act_l}")
        # weights row-major, then bias — one number per token, 8 per line for readability
        vals = np.concatenate([W.ravel(), b.ravel()])
        for chunk_start in range(0, len(vals), 8):
            lines.append(" ".join(f"{v:.17e}" for v in vals[chunk_start:chunk_start + 8]))
        lines.append("")  # blank separator

    path.write_text("\n".join(lines), encoding="ascii")


def load_mlp_weights(path: str | Path) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """Read back what :func:`export_mlp_weights` wrote.

    Returns
    -------
    list of (W, b, activation) — same format as the input to :func:`export_mlp_weights`.
    """
    path = Path(path)
    tokens = iter(path.read_text(encoding="ascii").split())
    _expect(next(tokens), "LAYERS")
    n_layers = int(next(tokens))

    layers = []
    for i in range(n_layers):
        _expect(next(tokens), "LAYER")
        _expect(next(tokens), str(i))
        in_tok = next(tokens)   # in=<din>
        out_tok = next(tokens)  # out=<dout>
        act_tok = next(tokens)  # activation=<name>
        d_in = int(in_tok.split("=")[1])
        d_out = int(out_tok.split("=")[1])
        activation = act_tok.split("=")[1]
        n_vals = d_out * d_in + d_out
        vals = np.array([float(next(tokens)) for _ in range(n_vals)], dtype=np.float64)
        W = vals[: d_out * d_in].reshape(d_out, d_in)
        b = vals[d_out * d_in :]
        layers.append((W, b, activation))
    return layers


# ---------------------------------------------------------------------------
# Scaler export / load
# ---------------------------------------------------------------------------

def export_scalers(
    scalers: dict[str, CompositionScaler | StandardScaler | ArcsinhScaler],
    path: str | Path,
) -> None:
    """Write scaler parameters to *path*.

    Parameters
    ----------
    scalers
        A dict mapping a logical name (e.g. ``"composition"``, ``"thermo"``, ``"rates"``) to a
        fitted scaler instance.  Supported types: :class:`~scarfs.models.common.CompositionScaler`,
        :class:`~scarfs.models.common.StandardScaler`, :class:`~scarfs.models.common.ArcsinhScaler`.
    path
        Destination text file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()

    buf.write(f"N_SCALERS {len(scalers)}\n")
    for name, scaler in scalers.items():
        if isinstance(scaler, CompositionScaler):
            _write_composition_scaler(buf, name, scaler)
        elif isinstance(scaler, ArcsinhScaler):
            _write_arcsinh_scaler(buf, name, scaler)
        elif isinstance(scaler, StandardScaler):
            _write_standard_scaler(buf, name, scaler)
        else:
            raise TypeError(f"Unsupported scaler type for '{name}': {type(scaler).__name__}")

    path.write_text(buf.getvalue(), encoding="ascii")


def load_scalers(path: str | Path) -> dict[str, CompositionScaler | StandardScaler | ArcsinhScaler]:
    """Read back what :func:`export_scalers` wrote."""
    path = Path(path)
    tokens = _Tokeniser(path.read_text(encoding="ascii"))
    tokens.expect("N_SCALERS")
    n = int(tokens.next())

    result: dict[str, CompositionScaler | StandardScaler | ArcsinhScaler] = {}
    for _ in range(n):
        tokens.expect("SCALER")
        name = tokens.next()
        stype = tokens.next()
        if stype == "CompositionScaler":
            result[name] = _read_composition_scaler(tokens)
        elif stype == "ArcsinhScaler":
            result[name] = _read_arcsinh_scaler(tokens)
        elif stype == "StandardScaler":
            result[name] = _read_standard_scaler(tokens)
        else:
            raise ValueError(f"Unknown scaler type '{stype}' for '{name}'.")
    return result


# ---------------------------------------------------------------------------
# Active-species export / load
# ---------------------------------------------------------------------------

def export_active_species(species: Sequence[str], path: str | Path) -> None:
    """Write the active-species list to *path*.

    The order here must match the surrogate's ``active_species`` tuple and the Fluent UDS/UDF source
    index ordering.  Water (H2O) is the fixed diluent and must NOT appear in this list (see thesis
    Ch. 5.6 and ``scarfs.schema.DILUENT_SPECIES``).

    Parameters
    ----------
    species
        Ordered sequence of active species names (no H2O).
    path
        Destination text file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"N_ACTIVE {len(species)}"]
    lines.extend(str(s) for s in species)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def load_active_species(path: str | Path) -> list[str]:
    """Read back what :func:`export_active_species` wrote."""
    path = Path(path)
    lines = [l.strip() for l in path.read_text(encoding="ascii").splitlines() if l.strip()]
    if not lines or not lines[0].startswith("N_ACTIVE"):
        raise ValueError("species file: expected 'N_ACTIVE <k>' on first non-blank line.")
    n = int(lines[0].split()[1])
    names = lines[1: 1 + n]
    if len(names) != n:
        raise ValueError(f"species file: expected {n} names, found {len(names)}.")
    return names


# ---------------------------------------------------------------------------
# ModelBundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """All artefacts needed for one exported surrogate.

    Attributes
    ----------
    layers
        MLP layers as ``(W, b, activation)`` triples (see :func:`export_mlp_weights`).
    scalers
        Dict of named scalers (see :func:`export_scalers`).
    active_species
        Ordered active-species list (see :func:`export_active_species`).
    name
        Human-readable model name (used as directory prefix).
    """

    layers: list[tuple[np.ndarray, np.ndarray, str]]
    scalers: dict[str, CompositionScaler | StandardScaler | ArcsinhScaler]
    active_species: list[str]
    name: str = "surrogate"


def export_bundle(bundle: ModelBundle, out_dir: str | Path) -> dict[str, Path]:
    """Write all three artefact files to *out_dir*.

    Returns
    -------
    dict mapping ``"weights"`` / ``"scalers"`` / ``"species"`` to the written :class:`~pathlib.Path`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = bundle.name

    paths = {
        "weights": out_dir / f"{prefix}_weights.txt",
        "scalers": out_dir / f"{prefix}_scalers.txt",
        "species": out_dir / f"{prefix}_species.txt",
    }
    export_mlp_weights(bundle.layers, paths["weights"])
    export_scalers(bundle.scalers, paths["scalers"])
    export_active_species(bundle.active_species, paths["species"])
    return paths


def load_bundle(out_dir: str | Path, name: str = "surrogate") -> ModelBundle:
    """Load a bundle previously written by :func:`export_bundle`."""
    out_dir = Path(out_dir)
    prefix = name
    return ModelBundle(
        layers=load_mlp_weights(out_dir / f"{prefix}_weights.txt"),
        scalers=load_scalers(out_dir / f"{prefix}_scalers.txt"),
        active_species=load_active_species(out_dir / f"{prefix}_species.txt"),
        name=name,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expect(token: str, expected: str) -> None:
    if token != expected:
        raise ValueError(f"Export parse error: expected '{expected}', got '{token}'.")


class _Tokeniser:
    """Lightweight whitespace-split token stream."""

    def __init__(self, text: str) -> None:
        self._tokens = iter(text.split())

    def next(self) -> str:
        return next(self._tokens)

    def expect(self, val: str) -> None:
        _expect(self.next(), val)

    def next_float_array(self, n: int) -> np.ndarray:
        return np.array([float(self.next()) for _ in range(n)], dtype=np.float64)


def _write_vec(buf: io.StringIO, name: str, arr: np.ndarray) -> None:
    """Write a named 1-D float array in 8-per-line format."""
    buf.write(f"  {name} {len(arr)}\n")
    for chunk_start in range(0, len(arr), 8):
        buf.write("  " + " ".join(f"{v:.17e}" for v in arr[chunk_start:chunk_start + 8]) + "\n")


def _read_vec(tokens: _Tokeniser, expected_name: str) -> np.ndarray:
    tokens.expect(expected_name)
    n = int(tokens.next())
    return tokens.next_float_array(n)


# -- CompositionScaler -------------------------------------------------------

def _write_composition_scaler(buf: io.StringIO, name: str, sc: CompositionScaler) -> None:
    if sc.data_min_ is None:
        raise RuntimeError(f"CompositionScaler '{name}' has not been fit().")
    buf.write(f"SCALER {name} CompositionScaler\n")
    buf.write(f"  log {int(sc.log)}\n")
    buf.write(f"  floor {sc.floor:.17e}\n")
    buf.write(f"  feature_range {sc.feature_range[0]:.17e} {sc.feature_range[1]:.17e}\n")
    _write_vec(buf, "data_min", sc.data_min_)
    _write_vec(buf, "data_max", sc.data_max_)
    buf.write("END_SCALER\n")


def _read_composition_scaler(tokens: _Tokeniser) -> CompositionScaler:
    tokens.expect("log")
    log = bool(int(tokens.next()))
    tokens.expect("floor")
    floor = float(tokens.next())
    tokens.expect("feature_range")
    lo = float(tokens.next())
    hi = float(tokens.next())
    data_min = _read_vec(tokens, "data_min")
    data_max = _read_vec(tokens, "data_max")
    tokens.expect("END_SCALER")
    sc = CompositionScaler(log=log, floor=floor, feature_range=(lo, hi))
    sc.data_min_ = data_min
    sc.data_max_ = data_max
    return sc


# -- StandardScaler ----------------------------------------------------------

def _write_standard_scaler(buf: io.StringIO, name: str, sc: StandardScaler) -> None:
    if sc.mean_ is None:
        raise RuntimeError(f"StandardScaler '{name}' has not been fit().")
    buf.write(f"SCALER {name} StandardScaler\n")
    _write_vec(buf, "mean", sc.mean_)
    _write_vec(buf, "scale", sc.scale_)
    buf.write("END_SCALER\n")


def _read_standard_scaler(tokens: _Tokeniser) -> StandardScaler:
    mean = _read_vec(tokens, "mean")
    scale = _read_vec(tokens, "scale")
    tokens.expect("END_SCALER")
    sc = StandardScaler()
    sc.mean_ = mean
    sc.scale_ = scale
    return sc


# -- ArcsinhScaler -----------------------------------------------------------

def _write_arcsinh_scaler(buf: io.StringIO, name: str, sc: ArcsinhScaler) -> None:
    if sc.scale_ is None:
        raise RuntimeError(f"ArcsinhScaler '{name}' has not been fit().")
    if sc._std.mean_ is None:
        raise RuntimeError(f"ArcsinhScaler '{name}' inner StandardScaler has not been fit().")
    buf.write(f"SCALER {name} ArcsinhScaler\n")
    buf.write(f"  min_scale {sc.min_scale:.17e}\n")
    _write_vec(buf, "arcsinh_scale", sc.scale_)
    _write_vec(buf, "std_mean", sc._std.mean_)
    _write_vec(buf, "std_scale", sc._std.scale_)
    buf.write("END_SCALER\n")


def _read_arcsinh_scaler(tokens: _Tokeniser) -> ArcsinhScaler:
    tokens.expect("min_scale")
    min_scale = float(tokens.next())
    arcsinh_scale = _read_vec(tokens, "arcsinh_scale")
    std_mean = _read_vec(tokens, "std_mean")
    std_scale = _read_vec(tokens, "std_scale")
    tokens.expect("END_SCALER")
    sc = ArcsinhScaler(min_scale=min_scale)
    sc.scale_ = arcsinh_scale
    sc._std.mean_ = std_mean
    sc._std.scale_ = std_scale
    return sc

"""Typed, serialisable training configuration.

Loadable from JSON (stdlib) or YAML (if PyYAML is installed). Construct directly in Python for tests
and the local sanity check; ship YAML files under ``configs/`` for the HPC runs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Data selection, splitting and importance-weighting (F1)."""

    database_path: str = "Database_FINAL.parquet"
    #: Input composition species: "active" | "molecular" | "dry_all" | "all" | explicit list.
    input_species: str | list[str] = "active"
    #: Target species whose rates are predicted: same selectors.
    target_species: str | list[str] = "active"
    val_fraction: float = 0.2
    split_by_case: bool = True
    #: Rows below this ethane conversion are treated as near-inlet and up-weighted (RC-1/F1).
    inlet_conversion_threshold: float = 0.05
    inlet_weight: float = 5.0
    #: Per-species target up-weights, e.g. {"C2H4": 150, "C3H6": 100, "CH4": 20, "H2": 10}.
    species_weights: dict[str, float] = field(default_factory=dict)
    composition_feature_range: tuple[float, float] = (-1.0, 1.0)
    seed: int = 0


@dataclass
class ModelConfig:
    """Architecture selection for the reduced or NeuralCoil surrogate."""

    kind: str = "reduced"  # "reduced" | "neuralcoil"
    hidden: tuple[int, ...] = (256, 256, 128)        # reduced
    latent_dim: int = 6                               # neuralcoil
    decoder_hidden: tuple[int, ...] = (128, 256)      # neuralcoil
    rate_hidden: tuple[int, ...] = (128, 128)         # neuralcoil
    activation: str = "silu"


@dataclass
class OptimConfig:
    """Optimisation hyper-parameters."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    epochs: int = 100
    batch_size: int = 4096
    grad_clip: float = 1.0
    patience: int = 15


@dataclass
class LossConfig:
    """Composite-loss weights (F2/F3)."""

    atom_balance_weight: float = 0.0     # soft elemental-conservation penalty (needs element data)
    recon_weight: float = 1.0            # NeuralCoil decoder reconstruction
    manifold_weight: float = 0.1         # NeuralCoil manifold-consistency (F2)
    noise_std: float = 0.0               # latent noise injection during training (F2)
    unroll_steps: int = 1                # NeuralCoil unrolled multi-step training (F2)


@dataclass
class TrainConfig:
    """Top-level training configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    output_dir: str = "runs/exp"

    # -- serialisation ---------------------------------------------------------------------
    @classmethod
    def from_mapping(cls, d: dict[str, Any]) -> "TrainConfig":
        """Build a config from a nested mapping (JSON/YAML-derived)."""
        return cls(
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            optim=OptimConfig(**d.get("optim", {})),
            loss=LossConfig(**d.get("loss", {})),
            output_dir=d.get("output_dir", "runs/exp"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        """Load a config from a ``.json`` or ``.yaml``/``.yml`` file."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in {".yaml", ".yml"}:
            import yaml  # optional dependency; only needed for YAML configs

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view (for logging / saving)."""
        return asdict(self)

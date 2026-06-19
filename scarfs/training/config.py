"""Typed, serialisable training configuration.

Loadable from JSON (stdlib) or YAML (if PyYAML is installed). Construct directly in Python for tests
and the local sanity check; ship YAML files under ``configs/`` for the HPC runs.

All new fields carry defaults that exactly reproduce current behaviour for ``kind="reduced"`` and
``kind="neuralcoil"`` — new merged-model behaviour is opt-in.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DataConfig:
    """Data selection, splitting and importance-weighting (F1).

    Tail-stratified sampling (merged model)
    ----------------------------------------
    When ``tail_strata > 0`` each row receives an additional weight

        tail_weight = 1 + tail_weight_alpha * (decile_rank / (tail_strata - 1))

    where ``decile_rank ∈ {0, …, tail_strata-1}`` is the rank of the row's
    ``log10(absorption + 1)`` value within *tail_strata* equal-count deciles
    (0 = lowest absorption, tail_strata-1 = highest).  The tail weight is
    multiplied into the per-row importance weight before being passed to the
    loss.  ``tail_strata=0`` disables the feature and preserves the existing
    ``inlet_weight``-only weighting exactly.
    """

    database_path: str = "Database_FINAL.parquet"
    #: Input composition species: "active" | "molecular" | "dry_all" | "all" | explicit list.
    input_species: str | list[str] = "active"
    #: Target species whose rates are predicted:
    #: same selectors as input, plus "energy_active" (resolved to the energy-active subset).
    target_species: str | list[str] = "active"
    val_fraction: float = 0.2
    #: Held-out test fraction for 70/15/15 GroupKFold-style case split.
    #: When > 0, val_fraction applies to the remaining (1-test_fraction) pool.
    #: Default 0.0 preserves the existing 80/20 train/val behaviour.
    test_fraction: float = 0.0
    split_by_case: bool = True
    #: Rows below this ethane conversion are treated as near-inlet and up-weighted (RC-1/F1).
    inlet_conversion_threshold: float = 0.05
    inlet_weight: float = 5.0
    #: Per-species target up-weights, e.g. {"C2H4": 150, "C3H6": 100, "CH4": 20, "H2": 10}.
    species_weights: dict[str, float] = field(default_factory=dict)
    composition_feature_range: tuple[float, float] = (-1.0, 1.0)
    seed: int = 0
    # -- merged-model additions ----------------------------------------------------------
    #: Fraction of total species absorption variance to retain when auto-selecting energy-active
    #: species on the train split.  0.999 captures ~80 species (see §2 design facts).
    energy_active_coverage: float = 0.999
    #: Path to a YAML mechanism file for NASA7 thermo.  Used by B1b SpeciesThermo; optional here
    #: (config round-trip works without it; training will error late if the path is needed).
    mech_yaml: str = "chem_ForTransport.yaml"
    #: Number of tail strata for absorption-decile sample weighting (0 = disabled).
    tail_strata: int = 0
    #: Alpha for tail weight: ``weight = 1 + alpha * (rank / (strata-1))``.
    tail_weight_alpha: float = 2.0
    #: When True, load only the parquet columns needed by the configured species/targets.
    columns_projection: bool = False
    #: Standard-mode composition sigma floor: species with per-column std <= this are
    #: de-activated as encoder inputs (scale_=1.0) instead of standardising numerical noise,
    #: and the latent-source target ż = E·(Ẏ⊘σ) stops amplifying their dYdt solver noise
    #: (overnight diagnosis 2026-06-12: 62/212 stride5 species sit below 1e-10 and made the
    #: ω_Z target noise-bound). 0.0 preserves the legacy behaviour; recommended 1e-10.
    composition_sigma_floor: float = 0.0


@dataclass
class ModelConfig:
    """Architecture selection for the reduced, NeuralCoil, or MergedCoil surrogate."""

    kind: str = "reduced"  # "reduced" | "neuralcoil" | "merged"
    hidden: tuple[int, ...] = (256, 256, 128)          # reduced
    latent_dim: int = 6                                 # neuralcoil / merged
    decoder_hidden: tuple[int, ...] = (128, 256)        # neuralcoil / merged
    rate_hidden: tuple[int, ...] = (128, 128)           # neuralcoil / merged
    activation: str = "silu"
    # -- merged-model additions ----------------------------------------------------------
    #: Hidden widths for the latent-source head (ω_Z(Z,q) -> k).
    latent_source_hidden: tuple[int, ...] = (128, 128)
    #: Hidden widths for the energy-absorption head (→ strictly positive scalar).
    energy_hidden: tuple[int, ...] = (64, 64)
    #: Apply spectral normalisation to all linear layers in the merged model.
    spectral_norm: bool = False
    #: Number of transport-property head outputs (0 disables the head). 2 = (μ, k); the trainer
    #: builds matching DB targets in that column order. Per-species diffusivity D_i is delivered as
    #: data columns (see DataGenConfig) and is a documented UDS-coupling follow-up, not a head output.
    transport_outputs: int = 0
    #: Hidden widths for the transport-property head.
    transport_hidden: tuple[int, ...] = (64, 64)


@dataclass
class OptimConfig:
    """Optimisation hyper-parameters."""

    lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    epochs: int = 100
    batch_size: int = 4096
    grad_clip: float = 1.0
    patience: int = 15
    #: LR schedule for the merged main loop: "none" (constant, default) | "cosine" (cosine
    #: annealing with linear warmup). Cosine helps a deterministic stiff-map regression converge
    #: to a lower final error than a constant LR; opt-in so existing runs are unchanged.
    lr_schedule: str = "none"
    #: Linear warmup length (epochs) before cosine annealing begins.
    warmup_epochs: int = 0
    #: Floor of the cosine schedule as a fraction of the base LR (annealed-to LR = min_lr_frac·lr).
    min_lr_frac: float = 0.05
    #: Merged model only: after the main loop, re-tune the distilled absorption head alone
    #: (trunk + rate/latent heads frozen) for this many epochs, checkpointed on the val
    #: head loss. The main loop's best checkpoint is selected on the TOTAL val loss, which
    #: the latent term dominates — that criterion measurably discards the head's own
    #: optimum (overnight diagnosis E8 vs E9: head R² 0.636 @ 14 ep → 0.096 @ 60 ep).
    head_finetune_epochs: int = 0


@dataclass
class LossConfig:
    """Composite-loss weights (F2/F3).

    For ``kind="merged"`` the total loss is (schematically)::

        L = rate_weight   * arcsinh_rate_loss(ω̂_phys, ω_true; enthalpy_weights, tail_weights)
          + latent_source_weight * latent_source_loss(ω̂_Z, latent_target; arcsinh per-dim)
          + energy_weight        * rate_tied_energy_loss(absorption(ω̂_phys, T), S_E_target)
          + energy_distill_weight * distill_energy_loss(Ŝ_E_head, stopgrad(rate-derived Ŝ_E))
          + energy_target_weight  * direct_energy_loss(Ŝ_E_head, S_E_target)
          + consistency_weight   * split_head_consistency(ω̂_Z, ω̂_phys; σ_comp)
          + qoi_recon_weight     * qoi_recon_loss(ŷ_recon, y_true; species_weights from DataConfig)
          + recon_weight         * decoder_recon_loss(ŷ_recon, y_true)
          + manifold_weight      * manifold_consistency(z, z_proj)
          + atom_balance_weight  * atom_balance_penalty(ω̂_phys; element_matrix)
          + rollout_loss          (manifold or lagrangian, governed by rollout_mode)

    ``rollout_mode="manifold"`` uses the existing unrolled multi-step training (``unroll_steps``).
    ``rollout_mode="lagrangian"`` adds a Lagrangian continuity penalty along same-case τ steps.
    """

    atom_balance_weight: float = 0.0     # soft elemental-conservation penalty (needs element data)
    #: Element conservation via the CONSTANT null-space projector (physics.atom_conservation_projector);
    #: better-conditioned than atom_balance_weight. Penalises ‖r_molar·Q‖² (a-priori/training only —
    #: the deployed UDF exports no per-species rate source). 0.0 disabled; ~5e-3 recommended.
    atom_projection_weight: float = 0.0
    #: Keq equilibrium-consistency penalty for the C2H6<->C2H4+H2 dehydrogenation (drives the step's
    #: net extent rate -> 0 as the reaction quotient approaches Keq(T), using NASA7 a6 entropy).
    #: Scoped, opt-in, OFF by default (0.0); ~1e-2 if enabled. See losses.keq_consistency_penalty.
    keq_weight: float = 0.0
    #: Gaussian width ε (ln-units) of the near-equilibrium weighting for the Keq penalty.
    keq_width: float = 1.0
    #: Realizability floor: penalise consumption rates that would deplete a species within
    #: ``realizability_dt`` (losses.realizability_penalty). 0.0 disabled; ~1e-2 if enabled.
    realizability_weight: float = 0.0
    #: Representative timestep [s] over which the realizability depletion bound is applied.
    realizability_dt: float = 1.0e-3
    #: Transport-property head loss (log-space MSE on μ, k against DB columns). Needs
    #: ``ModelConfig.transport_outputs > 0``. 0.0 disabled; ~0.05 if enabled.
    transport_weight: float = 0.0
    recon_weight: float = 1.0            # NeuralCoil/merged decoder reconstruction
    manifold_weight: float = 0.1         # NeuralCoil/merged manifold-consistency (F2)
    noise_std: float = 0.0               # latent noise injection during training (F2)
    unroll_steps: int = 1                # NeuralCoil unrolled multi-step training (F2)
    # -- merged-model additions ----------------------------------------------------------
    #: Weight on the physical-rate loss (arcsinh-space MSE + enthalpy-aware per-species weights).
    rate_weight: float = 1.0
    #: Weight on the latent-source head loss.
    latent_source_weight: float = 1.0
    #: Weight on the rate-tied energy tie (F3): absorption_from_rates(ω̂_phys, T) vs DB target.
    energy_weight: float = 0.5
    #: Weight on the distillation term: |Ŝ_E_head − stopgrad(rate-derived Ŝ_E)|.
    energy_distill_weight: float = 0.25
    #: Weight on the direct head vs target term: |Ŝ_E_head − S_E_target|.
    energy_target_weight: float = 0.25
    #: Weight on the split-head consistency penalty (MSE between active-species latent projection
    #: and the latent-source head prediction).
    consistency_weight: float = 0.1
    #: QoI-weighted decoder recon loss weight (species weights from ``DataConfig.species_weights``).
    qoi_recon_weight: float = 0.0
    #: Rollout mode: "manifold" (existing multi-step) | "lagrangian" (same-case τ continuity).
    rollout_mode: str = "manifold"


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
        """Build a config from a nested mapping (JSON/YAML-derived).

        Tuple-typed fields (``hidden``, ``decoder_hidden``, ``rate_hidden``,
        ``latent_source_hidden``, ``energy_hidden``, ``composition_feature_range``) are
        coerced from lists (as produced by JSON parsers) to tuples.
        """
        model_d = dict(d.get("model", {}))
        _tuple_fields_model = (
            "hidden", "decoder_hidden", "rate_hidden",
            "latent_source_hidden", "energy_hidden", "transport_hidden",
        )
        for f in _tuple_fields_model:
            if f in model_d and isinstance(model_d[f], list):
                model_d[f] = tuple(model_d[f])

        data_d = dict(d.get("data", {}))
        if "composition_feature_range" in data_d and isinstance(data_d["composition_feature_range"], list):
            data_d["composition_feature_range"] = tuple(data_d["composition_feature_range"])

        return cls(
            data=DataConfig(**data_d),
            model=ModelConfig(**model_d),
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

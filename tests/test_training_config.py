"""Tests for training configuration loading."""

from __future__ import annotations

import json

from scarfs.training.config import TrainConfig


def test_defaults():
    # act
    cfg = TrainConfig()
    # assert
    assert cfg.model.kind == "reduced"
    assert cfg.data.val_fraction == 0.2


def test_from_mapping_overrides():
    # act
    cfg = TrainConfig.from_mapping(
        {"model": {"kind": "neuralcoil", "latent_dim": 8}, "data": {"inlet_weight": 3.0}}
    )
    # assert
    assert cfg.model.kind == "neuralcoil" and cfg.model.latent_dim == 8
    assert cfg.data.inlet_weight == 3.0


def test_load_json_roundtrip(tmp_path):
    # arrange
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model": {"kind": "reduced"}, "optim": {"epochs": 3}}), encoding="utf-8")
    # act
    cfg = TrainConfig.load(path)
    # assert
    assert cfg.optim.epochs == 3
    assert cfg.model.kind == "reduced"

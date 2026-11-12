"""Tests for config loading / validation (config.py)."""

import os

import pytest

from gomoku_ml.config import ConfigError, load_config, resolve_config

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = os.path.join(HERE, "..", "configs")


def test_empty_config_gets_defaults():
    cfg = resolve_config({})
    assert cfg["architecture"]["name"] == "resnet"
    assert cfg["architecture"]["params"] == {"channels": 128, "blocks": 10}
    assert cfg["train"]["augmentation"] is True
    assert cfg["train"]["lr_schedule"] == "cosine"
    assert cfg["eval"]["opponents"] == ["random", "heuristic"]
    assert cfg["base_checkpoint"] is None
    assert cfg["resume"] is None


def test_mobilenet_config_resolves():
    cfg = resolve_config({
        "architecture": {"name": "mobilenet", "params": {"channels": 32, "blocks": 4}},
        "train": {"lr_schedule": "step", "lr_step_size": 5, "lr_gamma": 0.1},
    })
    assert cfg["architecture"]["name"] == "mobilenet"
    assert cfg["architecture"]["params"]["channels"] == 32
    assert cfg["train"]["lr_schedule"] == "step"


def test_unknown_architecture_errors():
    with pytest.raises(ConfigError, match="unknown"):
        resolve_config({"architecture": {"name": "vit"}})


def test_unknown_architecture_param_errors():
    with pytest.raises(ConfigError, match="unknown parameter"):
        resolve_config({"architecture": {"name": "resnet", "params": {"depth": 3}}})


def test_bad_values_error_with_key_name():
    with pytest.raises(ConfigError, match="train.lr"):
        resolve_config({"train": {"lr": 0}})
    with pytest.raises(ConfigError, match="train.lr"):
        resolve_config({"train": {"lr": -0.1}})
    with pytest.raises(ConfigError, match="train.lr_schedule"):
        resolve_config({"train": {"lr_schedule": "exponential"}})
    with pytest.raises(ConfigError, match="train.epochs"):
        resolve_config({"train": {"epochs": 0}})
    with pytest.raises(ConfigError, match="selfplay.sims_per_move"):
        resolve_config({"selfplay": {"sims_per_move": 0}})
    with pytest.raises(ConfigError, match="selfplay.dirichlet_epsilon"):
        resolve_config({"selfplay": {"dirichlet_epsilon": 1.5}})
    with pytest.raises(ConfigError, match="eval.opponents"):
        resolve_config({"eval": {"opponents": ["stockfish"]}})


def test_unknown_keys_error():
    with pytest.raises(ConfigError, match="unknown top-level"):
        resolve_config({"surprise": 1})
    with pytest.raises(ConfigError, match="unknown train key"):
        resolve_config({"train": {"momentum": 0.9}})


def test_legacy_model_key_maps_to_resnet():
    cfg = resolve_config({"model": {"channels": 16, "blocks": 2}})
    assert cfg["architecture"]["name"] == "resnet"
    assert cfg["architecture"]["params"] == {"channels": 16, "blocks": 2}


def test_missing_checkpoint_file_errors():
    with pytest.raises(ConfigError, match="base_checkpoint"):
        resolve_config({"base_checkpoint": "/nonexistent/model.pt"})
    with pytest.raises(ConfigError, match="resume"):
        resolve_config({"resume": "/nonexistent/model.pt"})


def test_shipped_configs_resolve():
    smoke = load_config(os.path.join(CONFIGS, "smoke.yaml"))
    assert smoke["architecture"]["params"]["channels"] == 16
    assert smoke["selfplay"]["games"] == 8
    assert smoke["train"]["augmentation"] is True
    full = load_config(os.path.join(CONFIGS, "full.yaml"))
    assert full["architecture"]["params"]["channels"] == 128
    assert full["selfplay"]["games"] == 5000


def test_missing_config_file_errors():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")

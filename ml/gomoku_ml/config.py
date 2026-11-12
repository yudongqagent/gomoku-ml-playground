"""Config loading, defaults and validation for training/self-play runs.

Config schema (YAML)::

    preset:                        # optional display block for the Trainer UI
      name: Smoke run              # (read by the server's /api/presets;
      description: ...             #  ignored by training itself)
      throughput_note: ...
    run_name: smoke                # short name; becomes part of the run dir
    seed: 0
    architecture:                  # see gomoku_ml.architectures
      name: resnet                 # resnet | mobilenet | resnet-se
      params: {channels: 16, blocks: 2}
    base_checkpoint: null          # or path: load weights only, fresh optimiser
    resume: null                   # or path: resume training incl. optimiser + step
    train:
      epochs: 3
      batch_size: 32
      lr: 0.001
      lr_schedule: cosine          # cosine | step
      lr_step_size: 10             # epochs per decay (step schedule only)
      lr_gamma: 0.5                # decay factor (step schedule only)
      warmup_steps: 5
      weight_decay: 0.0001
      policy_weight: 1.0           # loss weight for policy cross-entropy
      value_weight: 1.0            # loss weight for value MSE
      augmentation: true           # dihedral board symmetries (see augment.py)
      val_fraction: 0.05           # held-out fraction for val loss / best.pt
      checkpoint_every_epochs: 0   # 0 = only final.pt / best.pt
    selfplay:
      games: 8
      sims_per_move: 32
      c_puct: 1.5
      dirichlet_alpha: 0.3
      dirichlet_epsilon: 0.25
      temperature: 1.0
      temperature_moves: 20
      resign_threshold: null       # or float in [-1, 1]
      max_moves: 150
      checkpoint_every_games: 10   # write selfplay_state.json every N games
                                   # (shards + manifest are flushed per game anyway)
    eval:
      games: 4                     # TOTAL games per opponent, split evenly
                                   # across colours (use an even number)
      sims: 32
      opponents: [random, heuristic]   # also: mcts, checkpoint
      mcts_sims: 100               # sims for the heuristic-MCTS opponent
      tactical: true               # run the scripted tactical suite
      tactical_sims: 50            # MCTS sims per tactical position
      auto: false                  # train.py: auto-eval best.pt/final.pt at
                                   # end of training (see ml/README.md)

Every section has defaults, so a minimal config only needs ``run_name``
and ``architecture``. :func:`resolve_config` merges defaults, validates
everything, and raises :class:`ConfigError` with a clear message naming
the offending key — configs fail fast at startup, never mid-run.

Legacy: a top-level ``model: {channels, blocks}`` section (from the
first version of the configs) is accepted as an alias for
``architecture: {name: resnet, params: ...}``.
"""

from __future__ import annotations

import copy
import os

import yaml

from .architectures import ARCHITECTURES

EVAL_OPPONENTS = ("random", "heuristic", "mcts", "checkpoint")
LR_SCHEDULES = ("cosine", "step")


class ConfigError(ValueError):
    """Raised when a config is invalid; the message names the bad key."""


DEFAULTS: dict = {
    "run_name": "run",
    "seed": 0,
    "architecture": {"name": "resnet", "params": {"channels": 128, "blocks": 10}},
    "base_checkpoint": None,
    "resume": None,
    "train": {
        "epochs": 10,
        "batch_size": 64,
        "lr": 1e-3,
        "lr_schedule": "cosine",
        "lr_step_size": 10,
        "lr_gamma": 0.5,
        "warmup_steps": 10,
        "weight_decay": 1e-4,
        "policy_weight": 1.0,
        "value_weight": 1.0,
        "augmentation": True,
        "val_fraction": 0.05,
        "checkpoint_every_epochs": 0,
    },
    "selfplay": {
        "games": 8,
        "sims_per_move": 50,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "temperature": 1.0,
        "temperature_moves": 30,
        "resign_threshold": None,
        "max_moves": 225,
        "checkpoint_every_games": 10,
    },
    "eval": {
        "games": 10,
        "sims": 50,
        "opponents": ["random", "heuristic"],
        "mcts_sims": 100,
        "tactical": True,
        "tactical_sims": 50,
        "auto": False,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _require_number(cfg: dict, section: str, key: str, minimum=None, maximum=None):
    value = cfg[section][key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{section}.{key} must be a number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{section}.{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{section}.{key} must be <= {maximum}, got {value}")
    return value


def _require_int(cfg: dict, section: str, key: str, minimum=None, maximum=None):
    value = cfg[section][key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{section}.{key} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{section}.{key} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{section}.{key} must be <= {maximum}, got {value}")
    return value


def resolve_config(raw: dict, check_files: bool = True) -> dict:
    """Merge ``raw`` over defaults and validate. Returns the resolved
    config (a new dict). Raises ConfigError on any problem."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a YAML mapping at the top level")
    unknown_top = sorted(set(raw) - set(DEFAULTS) - {"model", "preset"})
    if unknown_top:
        raise ConfigError(
            f"unknown top-level config key(s) {unknown_top}; "
            f"known keys: {sorted(DEFAULTS)}"
        )
    cfg = _merge(DEFAULTS, {k: v for k, v in raw.items() if k != "model"})

    # Optional `preset:` display block (Trainer UI; ignored by training).
    if "preset" in cfg and cfg["preset"] is not None:
        preset = cfg["preset"]
        if not isinstance(preset, dict):
            raise ConfigError("preset: must be a mapping with name/description")
        for key in ("name", "description", "throughput_note"):
            if key in preset and not isinstance(preset[key], str):
                raise ConfigError(
                    f"preset.{key} must be a string, got {preset[key]!r}"
                )

    # Legacy `model:` alias -> architecture resnet.
    if "model" in raw and "architecture" not in raw:
        model_cfg = raw["model"]
        if not isinstance(model_cfg, dict):
            raise ConfigError("model: must be a mapping with channels/blocks")
        cfg["architecture"] = {
            "name": "resnet",
            "params": {
                "channels": model_cfg.get("channels", 128),
                "blocks": model_cfg.get("blocks", 10),
            },
        }

    if not isinstance(cfg["run_name"], str) or not cfg["run_name"].strip():
        raise ConfigError("run_name must be a non-empty string")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int) or cfg["seed"] < 0:
        raise ConfigError(f"seed must be a non-negative integer, got {cfg['seed']!r}")

    arch = cfg["architecture"]
    if not isinstance(arch, dict) or "name" not in arch:
        raise ConfigError("architecture must be a mapping with a 'name' key")
    if arch["name"] not in ARCHITECTURES:
        raise ConfigError(
            f"architecture.name '{arch['name']}' is unknown; "
            f"available: {sorted(ARCHITECTURES)}"
        )
    try:
        arch["params"] = ARCHITECTURES[arch["name"]].validate_params(
            arch.get("params") or {}
        )
    except ValueError as e:
        raise ConfigError(str(e)) from e

    for key in ("base_checkpoint", "resume"):
        path = cfg.get(key)
        if path is not None:
            if not isinstance(path, str):
                raise ConfigError(f"{key} must be a path string or null, got {path!r}")
            if check_files and not os.path.exists(path):
                raise ConfigError(f"{key} file not found: {path}")

    tr = cfg["train"]
    unknown = sorted(set(tr) - set(DEFAULTS["train"]))
    if unknown:
        raise ConfigError(f"unknown train key(s) {unknown}; known: {sorted(DEFAULTS['train'])}")
    _require_int(cfg, "train", "epochs", minimum=1)
    _require_int(cfg, "train", "batch_size", minimum=1)
    _require_number(cfg, "train", "lr", minimum=0)
    if tr["lr"] <= 0:
        raise ConfigError(f"train.lr must be > 0, got {tr['lr']}")
    if tr["lr_schedule"] not in LR_SCHEDULES:
        raise ConfigError(
            f"train.lr_schedule must be one of {LR_SCHEDULES}, got {tr['lr_schedule']!r}"
        )
    _require_int(cfg, "train", "lr_step_size", minimum=1)
    _require_number(cfg, "train", "lr_gamma", minimum=0, maximum=1)
    _require_int(cfg, "train", "warmup_steps", minimum=0)
    _require_number(cfg, "train", "weight_decay", minimum=0)
    _require_number(cfg, "train", "policy_weight", minimum=0)
    _require_number(cfg, "train", "value_weight", minimum=0)
    if not isinstance(tr["augmentation"], bool):
        raise ConfigError(f"train.augmentation must be true/false, got {tr['augmentation']!r}")
    _require_number(cfg, "train", "val_fraction", minimum=0, maximum=0.5)
    _require_int(cfg, "train", "checkpoint_every_epochs", minimum=0)

    sp = cfg["selfplay"]
    unknown = sorted(set(sp) - set(DEFAULTS["selfplay"]))
    if unknown:
        raise ConfigError(f"unknown selfplay key(s) {unknown}; known: {sorted(DEFAULTS['selfplay'])}")
    _require_int(cfg, "selfplay", "games", minimum=1)
    _require_int(cfg, "selfplay", "sims_per_move", minimum=1)
    _require_number(cfg, "selfplay", "c_puct", minimum=0)
    _require_number(cfg, "selfplay", "dirichlet_alpha", minimum=0)
    _require_number(cfg, "selfplay", "dirichlet_epsilon", minimum=0, maximum=1)
    _require_number(cfg, "selfplay", "temperature", minimum=0)
    _require_int(cfg, "selfplay", "temperature_moves", minimum=0)
    if sp["resign_threshold"] is not None:
        _require_number(cfg, "selfplay", "resign_threshold", minimum=-1, maximum=1)
    _require_int(cfg, "selfplay", "max_moves", minimum=1, maximum=225)
    _require_int(cfg, "selfplay", "checkpoint_every_games", minimum=0)

    ev = cfg["eval"]
    unknown = sorted(set(ev) - set(DEFAULTS["eval"]))
    if unknown:
        raise ConfigError(f"unknown eval key(s) {unknown}; known: {sorted(DEFAULTS['eval'])}")
    _require_int(cfg, "eval", "games", minimum=1)
    _require_int(cfg, "eval", "sims", minimum=1)
    _require_int(cfg, "eval", "mcts_sims", minimum=1)
    _require_int(cfg, "eval", "tactical_sims", minimum=1)
    if not isinstance(ev["tactical"], bool):
        raise ConfigError(f"eval.tactical must be true/false, got {ev['tactical']!r}")
    if not isinstance(ev["auto"], bool):
        raise ConfigError(f"eval.auto must be true/false, got {ev['auto']!r}")
    if not isinstance(ev["opponents"], list) or not ev["opponents"]:
        raise ConfigError("eval.opponents must be a non-empty list")
    bad = [o for o in ev["opponents"] if o not in EVAL_OPPONENTS]
    if bad:
        raise ConfigError(
            f"eval.opponents contains unknown opponent(s) {bad}; "
            f"available: {list(EVAL_OPPONENTS)}"
        )
    return cfg


def load_config(path: str, check_files: bool = True) -> dict:
    """Load a YAML config file and return the resolved (validated)
    config. Raises ConfigError / FileNotFoundError with clear messages."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config {path} did not parse to a mapping")
    try:
        return resolve_config(raw, check_files=check_files)
    except ConfigError as e:
        raise ConfigError(f"{path}: {e}") from e


def dump_config(cfg: dict) -> str:
    """Serialise a resolved config back to YAML (for run snapshots)."""
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)

"""Shared helpers for ml/scripts/* CLIs: path setup and config lookup."""

from __future__ import annotations

import os
import sys

# Repo layout: ml/scripts/common.py -> ml/ is the parent dir, repo root above.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.dirname(SCRIPTS_DIR)
REPO_ROOT = os.path.dirname(ML_DIR)


def ensure_ml_on_path() -> None:
    if ML_DIR not in sys.path:
        sys.path.insert(0, ML_DIR)


def torch_threads(default: int) -> int:
    """Torch intra-op thread count for this process.

    Overridable with the TORCH_THREADS env var (the training server can
    set it per job). Defaults are deliberately small: MCTS self-play and
    eval run one tiny forward pass at a time, so intra-op parallelism
    across many threads is pure synchronisation overhead — and with
    several jobs on one small VM, oversubscribed OpenMP threads spin
    and starve the actual work (observed: a 4-thread self-play job
    slowed ~10x when a second job started). Training with real batch
    sizes benefits from more threads, so train.py defaults higher.
    """
    env = os.environ.get("TORCH_THREADS")
    if env:
        try:
            value = int(env)
            if value >= 1:
                return value
        except ValueError:
            pass
    return default


def config_path(name: str) -> str:
    """Resolve a config name ('smoke') or path to an absolute path."""
    if os.path.exists(name):
        return os.path.abspath(name)
    candidate = os.path.join(ML_DIR, "configs", f"{name}.yaml")
    if os.path.exists(candidate):
        return candidate
    candidate = os.path.join(ML_DIR, "configs", name)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"config not found: {name}")

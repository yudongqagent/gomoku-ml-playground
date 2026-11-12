"""Settings for the training server.

Precedence: environment variables > server/config.yaml (if present) >
built-in defaults. Copy server/config.example.yaml to server/config.yaml
to set local values; config.yaml is gitignored.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

SERVER_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SERVER_DIR.parent
ML_DIR = REPO_ROOT / "ml"
PUBLIC_MODELS_DIR = REPO_ROOT / "public" / "models"

DEFAULT_DATA_DIR = SERVER_DIR / "data"

# Default GitHub repo that receives model releases. NOTE: this repo does
# NOT exist yet — Yudong creates the empty public repo himself (the Muse
# GitHub App cannot create repos in a personal account). Until it exists,
# dry-runs work fine but a real release upload will fail; the release
# dry-run warns about the missing repo rather than failing.
DEFAULT_MODEL_REPO = "yudongqagent/gomoku-ml-playground"


def _load_yaml_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def find_ml_python() -> str:
    """Interpreter used to run the ML pipeline.

    Prefer the ML workstream's venv (ml/.venv) when it exists — it has
    torch + onnxruntime. Otherwise fall back to the current interpreter.
    This server never installs packages into ml/.venv (read-only use).
    """
    candidate = ML_DIR / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


@dataclass
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    # "owner/repo" for GitHub releases; overridable via GOMOKU_MODEL_REPO
    # or model_repo in server/config.yaml.
    model_repo: str = DEFAULT_MODEL_REPO
    host: str = "127.0.0.1"
    port: int = 8000
    max_concurrent_jobs: int = 1  # v1: strictly one job at a time (FIFO)
    github_token: str = ""  # from GITHUB_TOKEN env only, never from config file
    ml_python: str = field(default_factory=find_ml_python)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def config_file(self) -> Path:
        return SERVER_DIR / "config.yaml"


def load_settings(data_dir: str | Path | None = None) -> Settings:
    """Build settings from config.yaml + env vars.

    ``data_dir`` (used by tests) overrides everything else.
    """
    file_cfg = _load_yaml_config(SERVER_DIR / "config.yaml")
    s = Settings()
    if file_cfg.get("data_dir"):
        s.data_dir = Path(file_cfg["data_dir"]).expanduser()
    if file_cfg.get("model_repo"):
        s.model_repo = str(file_cfg["model_repo"])
    if file_cfg.get("host"):
        s.host = str(file_cfg["host"])
    if file_cfg.get("port"):
        s.port = int(file_cfg["port"])
    if file_cfg.get("max_concurrent_jobs"):
        s.max_concurrent_jobs = int(file_cfg["max_concurrent_jobs"])

    if os.environ.get("GOMOKU_DATA_DIR"):
        s.data_dir = Path(os.environ["GOMOKU_DATA_DIR"]).expanduser()
    if os.environ.get("GOMOKU_MODEL_REPO"):
        s.model_repo = os.environ["GOMOKU_MODEL_REPO"]
    if os.environ.get("GOMOKU_HOST"):
        s.host = os.environ["GOMOKU_HOST"]
    if os.environ.get("GOMOKU_PORT"):
        s.port = int(os.environ["GOMOKU_PORT"])
    if os.environ.get("GITHUB_TOKEN"):
        s.github_token = os.environ["GITHUB_TOKEN"]
    if os.environ.get("GOMOKU_ML_PYTHON"):
        s.ml_python = os.environ["GOMOKU_ML_PYTHON"]

    if data_dir is not None:
        s.data_dir = Path(data_dir).expanduser()

    if not s.data_dir.is_absolute():
        s.data_dir = (SERVER_DIR / s.data_dir).resolve()
    return s

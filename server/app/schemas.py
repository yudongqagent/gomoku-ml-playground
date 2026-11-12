"""Pydantic request/response models for the training-server API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

JobType = Literal["selfplay", "train", "full"]


class Hyperparameters(BaseModel):
    games: int | None = Field(default=None, gt=0, le=1_000_000)
    epochs: int | None = Field(default=None, gt=0, le=10_000)
    mcts_sims: int | None = Field(default=None, gt=0, le=100_000)
    lr: float | None = Field(default=None, gt=0.0, lt=1.0)
    batch_size: int | None = Field(default=None, gt=0, le=65_536)
    temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    # Eval ladder (run for train/full jobs after training).
    eval_games: int | None = Field(default=None, gt=0, le=1_000)
    eval_sims: int | None = Field(default=None, gt=0, le=100_000)


class JobCreate(BaseModel):
    type: JobType
    name: str = Field(default="", max_length=120)
    # None means "use the preset's architecture" (or "resnet" when no
    # preset is given). Previously this defaulted to "resnet", which
    # silently overrode the preset's architecture (e.g. a longrun-cpu
    # job trained a ResNet instead of the preset's MobileNet).
    architecture: str | None = Field(default=None, max_length=64)
    arch_params: dict[str, Any] = Field(default_factory=dict)
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    base_checkpoint: str | None = Field(default=None, max_length=512)
    # Training preset id (ml/configs/<preset>.yaml); overrides below
    # still apply on top of the preset's config. Unknown preset -> 422.
    preset: str | None = Field(default=None, max_length=64)

    @field_validator("arch_params")
    @classmethod
    def _check_arch_params(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Values must be JSON scalars; key names and ranges are validated
        # against the ML registry schema in the create-job endpoint
        # (different architectures accept different params).
        for key, value in v.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"arch_param {key!r} must be a number")
        return v


class EvaluateRequest(BaseModel):
    games: int | None = Field(default=None, gt=0, le=1_000)
    sims: int | None = Field(default=None, gt=0, le=100_000)
    # Only scripted opponents are supported here; "checkpoint" needs a
    # --vs-checkpoint that this endpoint doesn't take (400 if given).
    opponents: list[str] | None = Field(default=None, max_length=8)


class ReleaseRequest(BaseModel):
    # Dry-run by default: the web UI always dry-runs first. dry_run=false
    # performs a real GitHub release upload (requires GOMOKU_MODEL_REPO
    # and gh auth or GITHUB_TOKEN).
    dry_run: bool = True

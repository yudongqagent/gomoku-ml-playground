"""Architecture registry — experiment-first network selection.

Every registered architecture shares the same I/O contract:

* Input:  float32 [N, 8, 15, 15]
* Output: ``policy`` logits [N, 225] and ``value`` [N, 1] in [-1, 1]

Each registry entry exposes a builder, a JSON-serializable parameter
schema (with defaults, types and bounds — this is what the training
server / UI uses to render architecture options), and a short
description.

Add a new architecture by writing the module in model.py (or a new
module), then adding one ``ArchitectureSpec`` entry below. Nothing else
needs to change: configs, the training loop, export and eval all go
through :func:`build_architecture`.

The training server lists the registry with::

    python ml/scripts/list_architectures.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from .model import (
    GomokuMobileNet,
    GomokuNet,
    ResidualBlock,
    SEResidualBlock,
)


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    description: str
    builder: Callable[..., nn.Module]
    # JSON-serializable parameter schema. Each entry:
    #   {"type": "integer"|"number", "default": ..., "minimum": ..., "description": ...}
    params_schema: dict = field(default_factory=dict)

    def default_params(self) -> dict:
        return {k: v["default"] for k, v in self.params_schema.items()}

    def validate_params(self, params: dict) -> dict:
        """Validate user-supplied params against the schema, filling in
        defaults. Raises ValueError with a clear message on any problem.
        Returns the resolved (complete) params dict."""
        if not isinstance(params, dict):
            raise ValueError(
                f"architecture '{self.name}': params must be a mapping, got {type(params).__name__}"
            )
        unknown = sorted(set(params) - set(self.params_schema))
        if unknown:
            raise ValueError(
                f"architecture '{self.name}': unknown parameter(s) {unknown}; "
                f"known parameters: {sorted(self.params_schema)}"
            )
        resolved: dict = {}
        for key, spec in self.params_schema.items():
            value = params.get(key, spec["default"])
            expected = spec["type"]
            if expected == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"architecture '{self.name}': parameter '{key}' must be an integer, got {value!r}"
                    )
            elif expected == "number":
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"architecture '{self.name}': parameter '{key}' must be a number, got {value!r}"
                    )
            if "minimum" in spec and value < spec["minimum"]:
                raise ValueError(
                    f"architecture '{self.name}': parameter '{key}' must be >= {spec['minimum']}, got {value}"
                )
            if "maximum" in spec and value > spec["maximum"]:
                raise ValueError(
                    f"architecture '{self.name}': parameter '{key}' must be <= {spec['maximum']}, got {value}"
                )
            resolved[key] = value
        return resolved

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params_schema,
        }


def _build_resnet(channels: int = 128, blocks: int = 10) -> GomokuNet:
    return GomokuNet(channels=channels, blocks=blocks, block_cls=ResidualBlock)


def _build_resnet_se(channels: int = 128, blocks: int = 10, reduction: int = 8) -> GomokuNet:
    return GomokuNet(
        channels=channels,
        blocks=blocks,
        block_cls=lambda c: SEResidualBlock(c, reduction=reduction),
    )


def _build_mobilenet(channels: int = 128, blocks: int = 10) -> GomokuMobileNet:
    return GomokuMobileNet(channels=channels, blocks=blocks)


_CHANNELS = {
    "type": "integer",
    "default": 128,
    "minimum": 4,
    "description": "Number of feature channels in the trunk.",
}
_BLOCKS = {
    "type": "integer",
    "default": 10,
    "minimum": 1,
    "description": "Number of residual / separable blocks in the trunk.",
}

ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "resnet": ArchitectureSpec(
        name="resnet",
        description=(
            "AlphaZero-style ResNet policy+value network. The default, "
            "well-understood baseline for Gomoku self-play training."
        ),
        builder=_build_resnet,
        params_schema={"channels": dict(_CHANNELS), "blocks": dict(_BLOCKS)},
    ),
    "mobilenet": ArchitectureSpec(
        name="mobilenet",
        description=(
            "MobileNet-style network with depthwise-separable blocks: "
            "~9x fewer parameters per block than ResNet at the same "
            "channel count, aimed at fast on-device inference."
        ),
        builder=_build_mobilenet,
        params_schema={"channels": dict(_CHANNELS), "blocks": dict(_BLOCKS)},
    ),
    "resnet-se": ArchitectureSpec(
        name="resnet-se",
        description=(
            "ResNet with squeeze-and-excitation (SE) blocks: a lightweight "
            "channel-attention mechanism that reweights feature channels "
            "per position."
        ),
        builder=_build_resnet_se,
        params_schema={
            "channels": dict(_CHANNELS),
            "blocks": dict(_BLOCKS),
            "reduction": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "description": "Channel reduction factor in the SE bottleneck.",
            },
        },
    ),
}


def list_architectures() -> dict:
    """Return the registry as a JSON-serializable dict (for the server/UI)."""
    return {"architectures": [spec.to_dict() for spec in ARCHITECTURES.values()]}


def list_architectures_json(indent: int = 2) -> str:
    return json.dumps(list_architectures(), indent=indent)


def build_architecture(name: str, params: dict | None = None) -> nn.Module:
    """Build a network by registry name, validating params against the
    architecture's schema. Raises ValueError for unknown names/params.

    The returned module carries ``architecture_name`` and ``arch_params``
    attributes so checkpoints record exactly what was built.
    """
    if name not in ARCHITECTURES:
        raise ValueError(
            f"unknown architecture '{name}'; available: {sorted(ARCHITECTURES)}"
        )
    spec = ARCHITECTURES[name]
    resolved = spec.validate_params(params or {})
    model = spec.builder(**resolved)
    model.architecture_name = name  # type: ignore[attr-defined]
    model.arch_params = resolved  # type: ignore[attr-defined]
    return model


def load_model_from_checkpoint(path: str, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Load a checkpoint file and rebuild the exact architecture it was
    trained with (from the stored architecture name/params; checkpoints
    written before those fields existed fall back to a ResNet with the
    stored channels/blocks). Returns (model, payload)."""
    import torch

    payload = torch.load(path, map_location=device, weights_only=False)
    name = payload.get("architecture")
    params = payload.get("arch_params")
    if name and name in ARCHITECTURES and isinstance(params, dict):
        model = build_architecture(name, params)
    else:
        model = build_architecture(
            "resnet",
            {
                "channels": int(payload.get("channels", 128)),
                "blocks": int(payload.get("blocks", 10)),
            },
        )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload

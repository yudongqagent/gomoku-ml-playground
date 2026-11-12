"""Policy + value networks for Gomoku (PyTorch).

All networks share the same I/O contract (see architectures.py):

* Input:  float32 [N, 8, 15, 15]  (8 planes, see encoding.py)
* Output: ``policy`` logits [N, 225] (row-major, index y*15+x) and
  ``value`` [N, 1] in [-1, 1] from the current player's perspective.

Architectures available (registered in architectures.py):

* ``resnet`` — AlphaZero-style ResNet (configurable channels/blocks).
* ``mobilenet`` — MobileNet-style depthwise-separable blocks: far fewer
  parameters, faster on-device inference, slightly weaker per-parameter
  capacity. A good candidate when inference speed on a phone matters.
* ``resnet-se`` — ResNet with squeeze-and-excitation (SE) blocks: a
  lightweight channel-attention mechanism that lets the net reweight
  feature channels per position.

Default ("full") size: 128 channels, 10 blocks. This is a reasonable
starting point taken from published Gomoku/AlphaZero-style work — it is
NOT a tuned choice, and the strength of the trained net depends
overwhelmingly on training scale, not on this architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import game


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + x)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-excitation: global average pool -> bottleneck MLP ->
    per-channel sigmoid gates. A lightweight channel-attention block."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        reduced = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))  # [N, C]
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))  # [N, C]
        return x * s[:, :, None, None]


class SEResidualBlock(nn.Module):
    """Residual block with an SE (channel-attention) stage."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + x)
        return out


class DepthwiseSeparableBlock(nn.Module):
    """MobileNet-style residual block: 3x3 depthwise conv (one filter per
    channel) followed by a 1x1 pointwise conv. ~8-9x fewer multiply-adds
    than a standard 3x3 conv at the same channel count."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.depthwise(x)))
        out = self.bn2(self.pointwise(out))
        out = F.relu(out + x)
        return out


class _PolicyValueHeads(nn.Module):
    """Shared policy + value heads used by every architecture."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * game.SIZE * game.SIZE, game.SIZE * game.SIZE)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(game.SIZE * game.SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, trunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p = F.relu(self.policy_bn(self.policy_conv(trunk)))
        p = p.view(p.size(0), -1)
        policy = self.policy_fc(p)
        v = F.relu(self.value_bn(self.value_conv(trunk)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        return policy, value


class GomokuNet(nn.Module):
    """ResNet policy+value network.

    ``block_cls`` selects the residual block type: ResidualBlock for the
    plain ``resnet`` architecture, SEResidualBlock for ``resnet-se``.
    """

    def __init__(
        self,
        channels: int = 128,
        blocks: int = 10,
        block_cls: type[nn.Module] = ResidualBlock,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Conv2d(8, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.Sequential(
            *[block_cls(channels) for _ in range(blocks)]
        )
        self.heads = _PolicyValueHeads(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.stem_bn(self.stem(x)))
        out = self.res_blocks(out)
        return self.heads(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GomokuMobileNet(nn.Module):
    """MobileNet-style policy+value network: depthwise-separable blocks.

    Same I/O contract as GomokuNet, far fewer parameters (~9x fewer per
    block at the same channel count), aimed at fast on-device inference.
    """

    def __init__(self, channels: int = 128, blocks: int = 10) -> None:
        super().__init__()
        self.channels = channels
        self.blocks = blocks
        self.stem = nn.Conv2d(8, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks_seq = nn.Sequential(
            *[DepthwiseSeparableBlock(channels) for _ in range(blocks)]
        )
        self.heads = _PolicyValueHeads(channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.stem_bn(self.stem(x)))
        out = self.blocks_seq(out)
        return self.heads(out)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(config: dict) -> GomokuNet:
    """Legacy helper: build a ResNet from a config dict with
    ``model.channels`` / ``model.blocks``. New code should use
    ``gomoku_ml.architectures.build_architecture`` with the
    ``architecture`` config section instead."""
    model_cfg = config.get("model", {})
    return GomokuNet(
        channels=int(model_cfg.get("channels", 128)),
        blocks=int(model_cfg.get("blocks", 10)),
    )

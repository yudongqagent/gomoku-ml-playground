"""Tests for the per-version value_weight port (ML batch 1 H1's
3-hunk change, carried into ML batch 3 H2 for v37's α sweep).

``ValueMCTS.__init__`` takes ``value_weight`` (None → v8's
``VALUE_WEIGHT`` 0.25, so v8 is byte-identical in behaviour); the
registry passes ``ENGINE_VERSIONS[version].get("value_weight")``
through. The sweep driver sets ``player._mcts.value_weight``
directly per run.
"""

import numpy as np
import torch

from gomoku_ml import game, scoring
from gomoku_ml.architectures import build_architecture
from gomoku_ml.mcts_value import VALUE_WEIGHT, ValueMCTS


def _random_net_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = build_architecture("resnet", {"channels": 16, "blocks": 2})
    path = tmp_path / "value-net.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "resnet",
            "arch_params": {"channels": 16, "blocks": 2},
            "channels": 16,
            "blocks": 2,
        },
        path,
    )
    return str(path)


def _board_with(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def test_value_weight_default_is_v8_weight(tmp_path):
    mcts = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10)
    assert mcts.value_weight == VALUE_WEIGHT == 0.25


def test_value_weight_scales_quiet_leaf(tmp_path):
    # Quiet position: the leaf value is the raw net value times
    # the instance's value_weight (1.0 vs 0.25 differ by 4x).
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    ckpt = _random_net_checkpoint(tmp_path)
    m025 = ValueMCTS(checkpoint_path=ckpt, sims=10, value_weight=0.25)
    m100 = ValueMCTS(checkpoint_path=ckpt, sims=10, value_weight=1.0)
    v025 = m025._leaf_value(board, game.BLACK)
    v100 = m100._leaf_value(board, game.BLACK)
    assert abs(v100 - 4 * v025) < 1e-6


def test_value_weight_registry_passthrough(tmp_path):
    # v37's registry entry carries value_weight 1.0 (placeholder
    # until the α sweep); a player built from the registry must
    # carry it. The checkpoint does not exist yet in a fresh
    # worktree, so point the registry at the temp checkpoint.
    ckpt = _random_net_checkpoint(tmp_path)
    entry = scoring.ENGINE_VERSIONS[37]
    assert entry["value_checkpoint"] == "ml/runs/ml3-h2/value-net.pt"
    assert entry["value_weight"] == 1.0
    saved = entry["value_checkpoint"]
    try:
        entry["value_checkpoint"] = ckpt
        player = scoring.get_player(37, seed=0)
        assert player._mcts.value_weight == 1.0
    finally:
        entry["value_checkpoint"] = saved


def test_v8_registry_has_no_value_weight():
    # v8's entry has no value_weight key, so the dispatch passes
    # None and v8 keeps VALUE_WEIGHT 0.25 exactly.
    assert "value_weight" not in scoring.ENGINE_VERSIONS[8]

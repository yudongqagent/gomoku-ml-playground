"""Regression tests for v8 (fast-loop iteration 1).

Positions from ``gomoku_ml/regression_positions_v8.json``: the five
hard-suite prevention positions the unpatched 4,350-game v8 failed,
plus smoke-4350 game 2 ply 22 (a safe move exists there). The
iteration-1 patch (2-ply threat check in ``ValueMCTS.choose_move``)
must keep passing all of them.

Expect types: ``moves`` (the chosen move must be in the listed set,
checked via :func:`gomoku_ml.evaluate.check_tactical_move`, exactly
as the tactical suite does) or ``safe`` (the chosen move must be
safe against the opponent's creators, via
:func:`gomoku_ml.scoring._move_is_safe`).

These tests load the real value net (``ml/runs/b3-h2/value-net.pt``,
the 4,350-game checkpoint) through the v8 registry entry, so they
exercise the full dispatch path. They are skipped if the checkpoint
is absent (e.g. a fresh checkout without the gitignored data).
"""

from __future__ import annotations

import json
import os

import pytest

from gomoku_ml import evaluate, game, scoring

REGRESSION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "gomoku_ml",
    "regression_positions_v8.json",
)
CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "runs", "b3-h2",
    "value-net.pt",
)


def _positions():
    with open(REGRESSION_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


@pytest.mark.parametrize("pos", _positions(), ids=[p["id"] for p in _positions()])
def test_regression_position(pos):
    if not os.path.exists(CHECKPOINT):
        pytest.skip("v8 value-net checkpoint not present")
    player = scoring.get_player(8, seed=0)
    board = evaluate.board_from_position(pos)
    to_move = pos["to_move"]
    move = player.choose_move(board, to_move)
    expect = pos["expect"]
    if expect["type"] == "moves":
        assert evaluate.check_tactical_move(board, to_move, move, expect), (
            f"{pos['id']}: v8 chose {game.move_to_xy(move)}, "
            f"expected one of {expect['moves']}"
        )
    elif expect["type"] == "safe":
        creators = scoring._opponent_creators(board, to_move)
        assert creators, f"{pos['id']}: no creators — not a threat position"
        assert scoring._move_is_safe(board, move, to_move, creators), (
            f"{pos['id']}: v8 chose {game.move_to_xy(move)}, which is "
            f"not safe against creators {creators}"
        )
    else:
        raise AssertionError(f"unknown expect type: {expect['type']!r}")

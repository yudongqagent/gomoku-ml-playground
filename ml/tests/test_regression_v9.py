"""Regression tests for v9 (batch 4, H1).

Each position in ``gomoku_ml/regression_positions_v9.json`` is a
critical position: the §2.6 position from iter-7 (v7-as-Black loss
to v0, game 16 ply 14 — the loss that motivated the root-decision
change) plus fast-loop critical positions found for v9 (see
``research/iterations/iter-9.md`` §2 for the verification behind
each one). The engine must play an expected move on every
position. A position the current engine does NOT fix is kept as an
xfail with its reason — a documented open weakness — never silently
dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v9.json"
)

# Positions the current engine does not fix (open weaknesses).
XFAIL = {}


def _load_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


@pytest.mark.parametrize("pos", _load_positions(), ids=lambda p: p["id"])
def test_v9_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v9 = scoring.get_player(9, seed=0)
    move = v9.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v9 played {chosen}, expected one of "
        f"{pos['expected']} (played before patch: "
        f"{pos['played_before_patch']})"
    )


def test_regression_positions_are_well_formed():
    positions = _load_positions()
    assert len(positions) >= 1
    for pos in positions:
        assert pos["to_move"] in ("black", "white")
        assert pos["expected"], f"{pos['id']}: no expected move"
        board = _board(pos)
        # No stones overlap; the expected moves are legal (empty).
        assert len(pos["black"]) + len(pos["white"]) == int(
            (board != game.EMPTY).sum()
        )
        for x, y in pos["expected"]:
            assert board[y, x] == game.EMPTY, (
                f"{pos['id']}: expected move {(x, y)} is occupied"
            )


def test_v9_is_tie_prior_gradient():
    v9 = scoring.get_player(9, seed=0)
    assert v9.name == "scoring-v9"
    assert v9._mcts.leaf_mode == "gradient"
    assert v9._mcts.root_decision == "tie-prior"
    assert v9._mcts.sims == 100

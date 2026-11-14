"""Regression tests for v7 (batch 3, fast loop).

Each position in ``gomoku_ml/regression_positions_v7.json`` is a
critical position from a fast-loop loss: the losing move v7 played
and the oracle-verified perfect move instead (see
``research/iterations/iter-7.md`` §2 for the verification behind
each one). The patched engine must play an expected move on every
position. A position the current patch does NOT fix is kept as an
xfail with its reason — a documented open weakness and the next
iteration's target — never silently dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v7.json"
)

# Positions the current patch does not fix (open weaknesses).
# fl1-g3-ply23: the saving move (9,12) is a forcing move ranked 8th
# by v1 (prior 0.04) — starved of visits in 100 sims. No leaf patch
# fixes it; it needs a forcing-move exploration fix (iteration 2).
XFAIL = {
    "fl1-g3-ply23": "forcing move starved by low prior; see iter-7.md §2.1",
}


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
def test_v7_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v7 = scoring.get_player(7, seed=0)
    move = v7.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v7 played {chosen}, expected one of "
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

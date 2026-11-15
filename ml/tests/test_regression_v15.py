"""Regression tests for v15 (batch 6, fast loop).

Each position in ``gomoku_ml/regression_positions_v15.json`` is a
critical position from a fast-loop loss: the losing move v15 played
and the oracle-verified perfect move instead (see
``research/iterations/iter-15.md`` §2 for the verification behind
each one). A position the current engine does NOT fix is kept as an
xfail with its reason — a documented open weakness — never silently
dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v15.json"
)

# Positions the current engine does not fix (open weaknesses).
# fl1-g1-ply35: near-tie at 4 vs 3 visits; correctly-signed q
# prefers the save but visits decide, and the gradient prefers the
# loser. Beyond-horizon value error (iter-13 class), not an
# exploration failure the floor can address; see iter-15.md §2.2.
# fl2-g5-ply19: the classic starvation pattern (save 2 visits vs
# loser 17) — but the save's prior (0.032) is 6x above the floor,
# so the floor cannot reach it, and the search's q prefers the
# loser; see iter-15.md §2.3.
# fl2-g8-ply18: save 6 visits vs loser 11 (roughly
# prior-proportional); q slightly prefers the loser, gradient
# slightly prefers the save — a close call with no structural
# cause; see iter-15.md §2.3.
XFAIL = {
    "fl1-g1-ply35": "beyond-horizon value error; see iter-15.md §2.2",
    "fl2-g5-ply19": "starved save above the floor, q prefers loser; see iter-15.md §2.3",
    "fl2-g8-ply18": "close call, no structural cause; see iter-15.md §2.3",
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
def test_v15_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v15 = scoring.get_player(15, seed=0)
    move = v15.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v15 played {chosen}, expected one of "
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

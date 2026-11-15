"""Regression tests for v14 (batch 5, fast loop).

Each position in ``gomoku_ml/regression_positions_v14.json`` is a
critical position from a fast-loop loss: the losing move v14 played
and the oracle-verified perfect move instead (see
``research/iterations/iter-14.md`` §2 for the verification behind
each one).

Iteration 1 found NO new patchable weakness class, so NO patch was
made and all three positions are xfails — documented open
weaknesses for the next batch, never silently dropped:

* fl1-g1-ply9 / fl1-g3-ply13: the losing move and the saving move
  are Black's two creator squares — both kill all creators, both
  non-forcing, v1 scores nearly tied. No signal in the engine
  separates them, and v7 plays the losing move too.
* fl1-g9-ply15: the v1 gradient prefers the saving move, but the
  search starves it (3 visits, prior 0.044) — an exploration
  weakness, same class as v7's fl1-g3-ply23.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v14.json"
)

# All three positions are open weaknesses (no patch was made in
# iteration 1 — see iter-14.md §2).
XFAIL = {
    "fl1-g1-ply9": "two creator-blocks indistinguishable to every engine signal; see iter-14.md §2.1",
    "fl1-g3-ply13": "same class as fl1-g1-ply9; see iter-14.md §2.1",
    "fl1-g9-ply15": "saving move starved by low prior (exploration weakness); see iter-14.md §2.1",
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
def test_v14_regression_position(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v14 = scoring.get_player(14, seed=0)
    move = v14.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v14 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )

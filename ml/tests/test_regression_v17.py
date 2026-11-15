"""Regression tests for v17 (batch 6, fast loop).

Each position in ``gomoku_ml/regression_positions_v17.json`` is a
critical position from a fast-loop loss: the losing move v17 played
and the oracle-verified perfect move instead (see
``research/iterations/iter-17.md`` §2 for the verification behind
each one).

Iteration 1 found NO new patchable weakness class, so NO patch was
made and the position is an xfail — a documented open weakness for
the next batch, never silently dropped:

* fl1-g1-ply13: the losing move (6,9) and the saving move (6,5) are
  Black's two creator squares — both kill all creators, both safe,
  v1 scores nearly tied. No signal in the engine separates them,
  and v7/v12/v13/v14 ALL play the losing move too (the iter-14
  games 1/3 class).

Iteration 2 (vs v7) found no new patchable class either — all four
worked losses are quiet positions (0 creators, so the threat check
is disengaged) with no forcing move involved (so the forcing
tie-break is not implicated). The classes are the known ones:

* fl2-g1-ply51 / fl2-g4-ply42: visit-tie knife-edges (4-4 and 2-2
  visits, priors separated by 0.0004 and 0.0002) — no tie-break
  signal in the pre-registered set separates the moves.
* fl2-g3-ply9 / fl2-g8-ply30: beyond-horizon value error —
  correctly-signed q prefers the losing move, and the saving move
  is starved of visits (5 vs 6, 4 vs 5). In fl2-g8-ply30 the v1
  gradient points the wrong way where the v0 gradient points
  right (v7/v12/v13 play the save; v14/v17 miss it) — one
  position, not a patch target.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v17.json"
)

# The positions are open weaknesses (no patch was made in either
# fast-loop iteration — see iter-17.md §2.1/§2.2).
XFAIL = {
    "fl1-g1-ply13": "two creator-blocks indistinguishable to every engine signal; v7/v12/v13/v14 all play the loser too; see iter-17.md §2.1",
    "fl2-g1-ply51": "visit-tie knife-edge (4-4, prior gap 0.0004); see iter-17.md §2.2",
    "fl2-g3-ply9": "beyond-horizon value error (q prefers the loser, save starved 5 vs 6 visits); see iter-17.md §2.2",
    "fl2-g4-ply42": "visit-tie knife-edge (2-2, prior gap 0.0002); see iter-17.md §2.2",
    "fl2-g8-ply30": "beyond-horizon value error; v1 gradient points the wrong way where the v0 gradient points right; see iter-17.md §2.2",
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
def test_v17_regression_position(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v17 = scoring.get_player(17, seed=0)
    move = v17.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v17 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )

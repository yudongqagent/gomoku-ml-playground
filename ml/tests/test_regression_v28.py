"""Regression tests for v28 (batch 9, H2) on the v20 critical
positions.

Each position in ``gomoku_ml/regression_positions_v20.json`` is a
critical position from iteration 20: the bl-ply* positions are from
the shared v14/v17 28-ply Black loss to v0 (the expected move is
v7's move, verified by engine-swap continuation — substituting it
flips the loss to a win; see ``research/iterations/iter-20.md``
§2.0), and the fl*-ply* positions are fast-loop losses (the
expected moves are oracle-verified saves; see iter-20.md §2.1-2.2).

v28 plays the save at 0 of 9 positions (iter-28 §1.3) — like v14,
and REGRESSING both of v22's saves (bl-ply14, fl1-g6-ply28). Both
regressions are deeper-tree effects of the theirs-term repair, the
mirror image of v22's bl-ply12 regression: Black's gap-three
breadth counts fully at every White-to-move leaf, so White's
replies look worse for White (better for Black on backup), and the
search prefers the flashier move over the forcing save. All 9 are
xfails — documented open weaknesses, never silently dropped:

* bl-ply8: the tenuki (8,5) is found by neither leaf at 100 sims.
* bl-ply10: v28 avoids the loser (7,9) but plays (8,5), not the
  verified save (5,12) — same as v22/v20.
* bl-ply12: v28 keeps v22's regression — plays (6,11) like v14
  instead of the distant four-maker (5,12).
* bl-ply14: v28 REGRESSES vs v22 — plays (4,9) like v14 instead
  of the forcing open three (6,9). Deeper-tree theirs-term
  effect; see iter-28.md §1.3.
* bl-ply16: the forcing counter-threat (6,9) loses to the direct
  block (9,4) on prior at 100 sims.
* fl1-g6-ply28: v28 REGRESSES vs v22 — plays (7,5) like v14
  instead of either save (v22's winning (4,9) or v20's drawing
  (0,9)). Deeper-tree theirs-term effect; see iter-28.md §1.3.
* fl1-g6-ply34: the save (4,2) is starved by a low prior (0.0214)
  and its value is beyond the horizon.
* fl1-g5-ply27 / fl2-g5-ply65: beyond-horizon value errors — all
  four engines miss the save (v28 plays (11,9) like v22 at the
  first, a new move (11,10) at the second; whether they are
  better is unknown and not claimed).
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v20.json"
)
POSITIONS_V28_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v28.json"
)

# Positions v28 (seed 0) does NOT play the save at — measured in
# iter-28.md §1.3, not assumed. All 9: v28 plays 0 saves.
XFAIL = {
    "bl-ply8": "tenuki (8,5) found by neither leaf at 100 sims; see iter-28.md §1.3",
    "bl-ply10": "v28 avoids the loser but plays (8,5), not the verified save (5,12); see iter-28.md §1.3",
    "bl-ply12": "v28 keeps v22's regression: plays (6,11) like v14, not the save (5,12); see iter-28.md §1.3",
    "bl-ply14": "v28 REGRESSES vs v22: plays (4,9) like v14, not the save (6,9) — deeper-tree theirs-term effect; see iter-28.md §1.3",
    "bl-ply16": "counter-threat (6,9) loses to the direct block on prior at 100 sims; see iter-28.md §1.3",
    "fl1-g6-ply28": "v28 REGRESSES vs v22: plays (7,5) like v14, not either save — deeper-tree theirs-term effect; see iter-28.md §1.3",
    "fl1-g6-ply34": "save (4,2) starved by low prior + beyond-horizon value error; see iter-28.md §1.3",
    "fl1-g5-ply27": "beyond-horizon value error; all engines miss the save; see iter-28.md §1.3",
    "fl2-g5-ply65": "beyond-horizon value error; all engines miss the save; see iter-28.md §1.3",
}


def _load_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _load_v28_positions():
    with open(POSITIONS_V28_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


@pytest.mark.parametrize("pos", _load_positions(), ids=lambda p: p["id"])
def test_v28_regression_position(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v28 = scoring.get_player(28, seed=0)
    move = v28.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v28 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )


# v28's own critical positions (iter-28 §2): fl1-g7-ply15 is an
# engine-swap-verified save that ALL engines miss (shared 100-sim
# search-horizon value error, not a v28-specific defect).
XFAIL_V28 = {
    "fl1-g7-ply15": "engine-swap-verified save (6,4) missed by all engines (v28/v22/v20/v14/v7 all play (6,8)); shared search-horizon value error; see iter-28.md §2.1",
}


@pytest.mark.parametrize(
    "pos", _load_v28_positions(), ids=lambda p: p["id"]
)
def test_v28_own_regression_position(pos):
    if pos["id"] in XFAIL_V28:
        pytest.xfail(XFAIL_V28[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v28 = scoring.get_player(28, seed=0)
    move = v28.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v28 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )

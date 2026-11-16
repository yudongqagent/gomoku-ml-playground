"""Regression tests for v22 (batch 8, H2) on the v20 critical
positions.

Each position in ``gomoku_ml/regression_positions_v20.json`` is a
critical position from iteration 20: the bl-ply* positions are from
the shared v14/v17 28-ply Black loss to v0 (the expected move is
v7's move, verified by engine-swap continuation — substituting it
flips the loss to a win; see ``research/iterations/iter-20.md``
§2.0), and the fl*-ply* positions are fast-loop losses (the
expected moves are oracle-verified saves; see iter-20.md §2.1-2.2).

The colour-conditional demotion (§1) keeps 2 of v20's 3 saves —
bl-ply14 and fl1-g6-ply28 (where v22 plays the WINNING save (4,9),
not v20's drawing save (0,9)). The other 7 are xfails —
documented open weaknesses, never silently dropped:

* bl-ply8: the tenuki (8,5) is found by neither leaf at 100 sims;
  the divergence there is a deeper-tree effect, not a 1-ply
  gradient effect.
* bl-ply10: v22 avoids the loser (7,9) but plays (8,5), not the
  verified save (5,12) — same as v20.
* bl-ply12: v22 REGRESSES vs v20 — plays (6,11) like v14 instead
  of the distant four-maker (5,12). NOT a 1-ply gradient effect
  (the gradients after the save and after (6,11) are identical
  for all three leaves); a deeper-tree effect of the theirs-term:
  White's full gap-three breadth at every White-to-move leaf
  makes the save's lines look worse for Black on backup. See
  iter-22.md §1.3.
* bl-ply16: the forcing counter-threat (6,9) loses to the direct
  block (9,4) on prior at 100 sims.
* fl1-g6-ply34: the save (4,2) is starved by a low prior (0.0214)
  and its value is beyond the horizon (the lines run through a
  must-block before Black's five four-makers win the race).
* fl1-g5-ply27 / fl2-g5-ply65: beyond-horizon value errors — all
  three engines miss the save (v22 plays different moves than
  v20/v14 at both; whether they are better is unknown and not
  claimed).
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v20.json"
)
POSITIONS_V22_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v22.json"
)

# Positions v22 (seed 0) does NOT play the save at — measured in
# iter-22.md §1.3, not assumed.
XFAIL = {
    "bl-ply8": "tenuki (8,5) found by neither leaf at 100 sims; see iter-22.md §1.3",
    "bl-ply10": "v22 avoids the loser but plays (8,5), not the verified save (5,12); see iter-22.md §1.3",
    "bl-ply12": "v22 REGRESSES vs v20: plays (6,11) like v14, not the save (5,12) — deeper-tree theirs-term effect; see iter-22.md §1.3",
    "bl-ply16": "counter-threat (6,9) loses to the direct block on prior at 100 sims; see iter-22.md §1.3",
    "fl1-g6-ply34": "save (4,2) starved by low prior + beyond-horizon value error; see iter-22.md §1.3",
    "fl1-g5-ply27": "beyond-horizon value error; all engines miss the save; see iter-22.md §1.3",
    "fl2-g5-ply65": "beyond-horizon value error; all engines miss the save; see iter-22.md §1.3",
}

# v22's own fast-loop positions (iter-22 §2.2): both are
# engine-swap-verified saves v22 misses — White gap-three
# saturation. Both are xfails (documented open weaknesses).
XFAIL_V22 = {
    "fl2-g7-ply17": "v22 plays (6,10); White holds 18 gap-three makers after it (p99.7) vs 14 after the save (5,9); see iter-22.md §2.2",
    "fl2-g9-ply19": "v22 plays (8,5) 35 visits vs save (8,9) 32; White holds 13 gap-three makers after it (p98.5) vs 10 after the save; see iter-22.md §2.2",
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
def test_v22_regression_position(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v22 = scoring.get_player(22, seed=0)
    move = v22.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v22 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )


def _load_v22_positions():
    with open(POSITIONS_V22_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


@pytest.mark.parametrize("pos", _load_v22_positions(), ids=lambda p: p["id"])
def test_v22_fast_loop_regression_position(pos):
    if pos["id"] in XFAIL_V22:
        pytest.xfail(XFAIL_V22[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v22 = scoring.get_player(22, seed=0)
    move = v22.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v22 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )

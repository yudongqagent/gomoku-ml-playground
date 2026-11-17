"""Regression tests for v20 (batch 7, fast loop + battery forensics).

Each position in ``gomoku_ml/regression_positions_v20.json`` is a
critical position from iteration 20: the bl-ply* positions are from
the shared v14/v17 28-ply Black loss to v0 (the expected move is
v7's move, verified by engine-swap continuation — substituting it
flips the loss to a win; see ``research/iterations/iter-20.md``
§2.0), and the fl*-ply* positions are fast-loop losses (the expected
moves are oracle-verified saves; see iter-20.md §2.1-2.2).

The gap-three demotion (§1) rescues 3 of the 9 positions — exactly
the ones where the gap-three mechanism was the driver (bl-ply12,
bl-ply14, fl1-g6-ply28). The other 6 are xfails — documented open
weaknesses, never silently dropped:

* bl-ply8: the tenuki (5,8) is found by neither leaf at 100 sims;
  the divergence there is a deeper-tree effect, not a 1-ply
  gradient effect.
* bl-ply10: v20 avoids the loser (9,7) but plays (5,8), not the
  verified save (12,5).
* bl-ply16: the forcing counter-threat (9,6) loses to the direct
  block (4,9) on prior (v1 total 200091 vs 20252) at 100 sims.
* fl1-g6-ply34: the save (4,2) is starved by a low prior (0.0214)
  and its value is beyond the horizon (the lines run through a
  must-block before Black's five four-makers win the race).
* fl1-g5-ply27 / fl2-g5-ply65: beyond-horizon value errors — both
  leaves miss the save.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v20.json"
)

# Positions v20 (seed 0) does NOT play the save at — measured in
# iter-20.md §1.3, not assumed.
XFAIL = {
    "bl-ply8": "tenuki (5,8) found by neither leaf at 100 sims; see iter-20.md §1.3",
    "bl-ply10": "v20 avoids the loser but plays (5,8), not the verified save (12,5); see iter-20.md §1.3",
    "bl-ply16": "counter-threat (9,6) loses to the direct block on prior at 100 sims; see iter-20.md §1.3",
    "fl1-g6-ply34": "save (4,2) starved by low prior + beyond-horizon value error; see iter-20.md §2.1",
    "fl1-g5-ply27": "beyond-horizon value error; both leaves miss the save; see iter-20.md §2.1",
    "fl2-g5-ply65": "beyond-horizon value error; both leaves miss the save; see iter-20.md §2.2",
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
def test_v20_regression_position(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v20 = scoring.get_player(20, seed=0)
    move = v20.choose_move(board, player)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]], (
        f"{pos['id']}: v20 played {(move % 15, move // 15)}, "
        f"expected one of {pos['expected']}"
    )

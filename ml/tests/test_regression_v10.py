"""Regression tests for v10 (batch 4, fast loop iteration 1).

Each position in ``gomoku_ml/regression_positions_v10.json`` is a
critical position from a fast-loop loss: the losing move v10 played
and the oracle-verified perfect move instead (see
``research/iterations/iter-10.md`` §2.1 for the verification behind
each one).

Unlike v7/v8, iteration 1 produced NO patch. COORDINATOR'S
CORRECTION (merge, see iter-10.md §6): the original text called
the dominant class the "q-vs-visits root-decision gap" (the
saving move has the HIGHER q but loses on visits), but the
q-values in §2.1 are RAW q from ``analyse_critical.py`` —
the OPPONENT's value under the negamax backup (see iter-9.md
§1.0). Correctly signed, q prefers the LOSING move in 4 of 6
positions (g1/g3/g9/g0); only g5/g7 are a genuine q-vs-visits
gap (q prefers the save, visits decide) — and H1's smoke
independently showed q-based root decisions lose at match level
(pure q 0.20, tie-q 0.45 vs v7). These positions are kept as
xfails — a documented open weakness and the executable
White-weakness suite — never silently dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v10.json"
)

# Positions the current (unpatched) v10 does not fix — all six.
# Coordinator's correction (merge, see iter-10.md §6): the q-values
# below are RAW q (opponent's value); correctly signed (-q), q
# prefers the LOSING move in g1/g3/g9/g0. Only g5/g7 are a genuine
# q-vs-visits gap (q prefers the save, visits decide).
XFAIL = {
    "fl1-g1-ply67": "q (signed) prefers loser: save our-q -0.207 < -0.172, visits 4 < 27",
    "fl1-g3-ply67": "q (signed) prefers loser: save our-q -0.271 < -0.161, visits 11 < 60",
    "fl1-g5-ply71": "genuine q-vs-visits gap: save our-q -0.103 > -0.163, visits decide",
    "fl1-g7-ply31": "genuine q-vs-visits gap: save our-q -0.097 > -0.108, visits decide",
    "fl1-g9-ply19": "q (signed) prefers loser: loser 82 visits at our-q +0.131, save 8 at -0.064",
    "fl1-g0-ply40": "q (signed) prefers loser: save our-q -0.149 < -0.109, visits 2 < 6",
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
def test_v10_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v10 = scoring.get_player(10, seed=0)
    move = v10.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v10 played {chosen}, expected one of "
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

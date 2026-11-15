"""Regression tests for v13 (batch 5, fast loop iteration 1).

Each position in ``gomoku_ml/regression_positions_v13.json`` is a
critical position from a fast-loop loss (v13 base vs v7): the
losing move v13 played and the oracle-verified perfect move
instead (see ``research/iterations/iter-13.md`` §2.2 for the
verification behind each one).

Iteration 1 produced NO patch yet: the dominant class is
beyond-horizon value error (the search's correctly-signed q
prefers the losing move in 3 of the 4 positions), and the
§1b White-prior-temperature probe decides whether a patch
exists. All 4 positions are kept as xfails — documented open
weaknesses — never silently dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v13.json"
)

# Positions the current (unpatched) v13 base does not fix — all four.
# The dominant class is beyond-horizon value error (iter-13 §2.2):
# correctly-signed q prefers the loser in g0/g2/g7; g9-ply15 is the
# exception (q prefers the save, visits override it, and the save is
# threat-check-unsafe). The §1b temperature probe is the next step.
XFAIL = {
    "fl1-g0-ply12": "beyond-horizon value error: our-q prefers the loser",
    "fl1-g2-ply30": "beyond-horizon value error: our-q prefers the loser",
    "fl1-g7-ply25": "beyond-horizon value error: our-q prefers the loser",
    "fl1-g9-ply15": "q prefers the save but visits override it (24 vs 16)",
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
def test_v13_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v13 = scoring.get_player(13, seed=0)
    move = v13.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v13 played {chosen}, expected one of "
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

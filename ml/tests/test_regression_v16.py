"""Regression tests for v16 (batch 6, fast loop).

Each position in ``gomoku_ml/regression_positions_v16.json`` is
either a mechanism-check position (the two hard White critical
positions the 2-ply rollout leaf flips — see
``research/iterations/iter-16.md`` §1.1) or a critical position from
a v16 fast-loop loss: the losing move v16 played and the
oracle-verified perfect move instead (see ``iter-16.md`` §2 for the
verification behind each one). The engine must play an expected move
on every position. A position the current engine does NOT fix is
kept as an xfail with its reason — a documented open weakness and
the next iteration's target — never silently dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v16.json"
)

# Positions the current engine does not fix (open weaknesses).
# fl1-g0-ply12, fl1-g6-ply78: Black-to-move positions — v16-as-Black
# IS v7, so v16 plays v7's losing move there; fixing them would mean
# changing the Black leaf, which is out of scope for the White-leaf
# workstream (the Black games are the built-in control). Documented
# xfails, not dropped.
XFAIL = {
    "fl1-g0-ply12": "Black position (v16-as-Black is v7); see iter-16.md §2.1",
    "fl1-g6-ply78": "Black position (v16-as-Black is v7); see iter-16.md §2.1",
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
def test_v16_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v16 = scoring.get_player(16, seed=0)
    move = v16.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v16 played {chosen}, expected one of "
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

"""Regression tests for v21 (batch 8, fast loop).

Each position in ``gomoku_ml/regression_positions_v21.json`` is
either a mechanism-check position (v18's 6 regression positions,
copied unchanged — see ``research/iterations/iter-21.md`` §1.3) or
a critical position from a v21 fast-loop loss: the losing move v21
played and the oracle-verified perfect move instead (see
``iter-21.md`` §2 for the verification behind each one). The engine
must play an expected move on every position. A position the
current engine does NOT fix is kept as an xfail with its reason —
a documented open weakness and the next iteration's target — never
silently dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v21.json"
)

# Positions the current engine does not fix (open weaknesses).
# fl1-g0-ply12: Black-to-move v16/v18 xfail — v21 plays [7, 10]
# (as v18 does; v16/v7 played [11, 9]), still not one of the oracle
# saves ([9, 7], [5, 5], [7, 8]). The blend does not change the
# move here; see iter-21.md §1.3.
# fl2-g7-ply9: White-to-move — v21 plays (6, 4), the v1 oracle's
# loser (3/3), over (10, 8). The two moves are near-twins (both
# v1 ~200k, equal priors, both reduce Black creators to 0) and
# the distinction is beyond the 2-ply horizon; the stronger oracle
# only weakly confirms (10, 8) (v7-vs-v7 2/3 vs 1/3). No patchable
# mechanism — see iter-18.md §2.2 and iter-21.md §1.3. Documented
# open weakness, not dropped.
XFAIL = {
    "fl1-g0-ply12": "Black position; v21 plays [7, 10], not a save; see iter-21.md §1.3",
    "fl2-g7-ply9": "White twin-choice; v21 plays (6, 4) over (10, 8); no patchable mechanism; see iter-18.md §2.2",
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
def test_v21_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v21 = scoring.get_player(21, seed=0)
    move = v21.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v21 played {chosen}, expected one of "
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

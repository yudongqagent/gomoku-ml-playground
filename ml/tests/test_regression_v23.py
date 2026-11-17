"""Regression tests for v23 (batch 8, fast loop).

Each position in ``gomoku_ml/regression_positions_v23.json`` is
either a v18 regression position (copied unchanged — see
``research/iterations/iter-18.md`` §1.2/§2), one of v18's four
hard-tactical failures (the v23 tie-break targets — see
``research/iterations/iter-23.md`` §1.2), or a critical position
from a v23 fast-loop loss: the losing move v23 played and the
oracle-verified perfect move instead (see ``iter-23.md`` §2 for
the verification behind each one). The engine must play an
expected move on every position. A position the current engine
does NOT fix is kept as an xfail with its reason — a documented
open weakness and the next iteration's target — never silently
dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v23.json"
)

# Positions the current engine does not fix (open weaknesses).
# fl1-g0-ply12: Black-to-move v16 xfail — v18 plays [7, 10]
# (v16/v7 played [11, 9]), still not one of the oracle saves
# ([9, 7], [5, 5], [7, 8]). The Black rollout leaf changes the
# move but does not find the save here; see iter-18.md §1.2.
# fl2-g7-ply9: White-to-move — v18 plays (6, 4), the v1 oracle's
# loser (3/3), over (10, 8). The two moves are near-twins (both
# v1 ~200k, equal priors, both reduce Black creators to 0) and
# the distinction is beyond the 2-ply horizon: the rollout leaf
# prefers (6, 4) too (+0.107 vs +0.039), and the stronger oracle
# only weakly confirms (10, 8) (v7-vs-v7 2/3 vs 1/3; the actual
# matchup 1/2 vs 0/2). No patchable mechanism — see iter-18.md
# §2.2. Documented open weakness, not dropped.
# tact-winning-quiet-setup-2: NOT a tie (79 visits on (9, 7) vs
# 16 on (8, 7)) — v23 plays (9, 7) exactly as v18 did, and (9, 7)
# WINS 3/3 under v7-vs-v7: a suite-list gap, not a defect. See
# iter-18.md §3.1 and iter-23.md §1.2.
# fl2-g6-ply23 / fl2-g8-ply23: verified losing moves (iter-23
# §2.2) — v23 plays (9, 5)/(5, 5), which lose 3/3 under the
# stronger oracles, over the verified saves. Tie-break misfires
# (the losing move wins the visit tie on a 0.016 q gap — gradient
# noise) with no patchable mechanism: the saves hold <=6 visits
# and lower q, and no root signal distinguishes the tied pair
# correctly. Documented open weaknesses, not dropped.
XFAIL = {
    "fl1-g0-ply12": "Black position; v18/v23 play [7, 10], not a save; see iter-18.md §1.2",
    "fl2-g7-ply9": "White twin-choice; v18/v23 play (6, 4) over (10, 8); no patchable mechanism; see iter-18.md §2.2",
    "tact-winning-quiet-setup-2": "Not a tie (79 vs 16 visits); (9, 7) wins 3/3 — suite-list gap, not a defect; see iter-23.md §1.2",
    "fl2-g6-ply23": "Tie-break misfire; v23 plays (9, 5) over verified saves; no patchable mechanism; see iter-23.md §2.2",
    "fl2-g8-ply23": "Tie-break misfire (mirror of fl2-g6-ply23); v23 plays (5, 5) over verified saves; no patchable mechanism; see iter-23.md §2.2",
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
def test_v23_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v23 = scoring.get_player(23, seed=0)
    move = v23.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v23 played {chosen}, expected one of "
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

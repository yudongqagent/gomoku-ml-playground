"""Regression tests for v11 (batch 4 H3, fast loop).

Each position in ``gomoku_ml/regression_positions_v11.json`` is a
critical position from a fast-loop loss: the losing move v11 played
and the oracle-verified perfect move instead (see
``research/iterations/iter-11.md`` §5 for the verification behind
each one). The patched engine must play an expected move on every
position. A position the current patch does NOT fix is kept as an
xfail with its reason — a documented open weakness — never silently
dropped.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v11.json"
)

# Positions the current patch does not fix (open weaknesses).
# fl1-g9-ply11: v11 plays the losing (6,6) (30 visits) over the
# saving (11,6) (4 visits). COORDINATOR'S CORRECTION (merge):
# the original comment read the q-values as our-perspective, but
# they are RAW q (opponent's value under the negamax backup — see
# iter-9.md §1.0); correctly signed, q prefers the LOSING move
# (our-q +0.107 for (6,6) vs -0.015 for (11,6), reproduced by the
# coordinator), so this is NOT a q-vs-visits gap — the search does
# not know the right move here. The position stands as a valid
# failing position. See iter-11.md §9.
# fl2-g0-ply18 / fl2-g8-ply24 / fl2-g9-ply67: same correction
# applies (raw-q reading; see iter-11.md §9). fl2-g7-ply23
# is a near-tie coin-flip (visits 10 vs 12, q 0.141 vs 0.129, net
# neutral) that lands on the losing move with seed 0. See
# iter-11.md §5.2.
XFAIL = {
    "fl1-g9-ply11": "losing move wins on visits; q (correctly signed) prefers it too — see iter-11.md §9",
    "fl2-g0-ply18": "losing move wins on visits; see iter-11.md §5.2 and §9",
    "fl2-g7-ply23": "near-tie coin-flip; no systematic bias, see iter-11.md §5.2",
    "fl2-g8-ply24": "losing move wins on visits; see iter-11.md §5.2 and §9",
    "fl2-g9-ply67": "losing move wins on visits (mild); see iter-11.md §5.2 and §9",
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
def test_v11_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v11 = scoring.get_player(11, seed=0)
    move = v11.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v11 played {chosen}, expected one of "
        f"{pos['expected']} (played before patch: "
        f"{pos['played_before_patch']})"
    )

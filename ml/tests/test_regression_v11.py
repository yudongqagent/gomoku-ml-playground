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
# fl1-g9-ply11: the q-vs-visits gap — v11's own search ranks the
# saving move (11,6) higher in q (+0.024) than the losing (6,6)
# (-0.105), but most-visited picks (6,6) (30 visits vs 4). The
# net's leaf values are near-neutral here, so no H3-scope leaf
# patch fixes it; it needs the q-vs-visits root-decision fix
# (batch-4 H1). See iter-11.md §5.1.
# fl2-g0-ply18 / fl2-g8-ply24 / fl2-g9-ply67: the same q-vs-visits
# gap (saving move has better q but fewer visits). fl2-g7-ply23
# is a near-tie coin-flip (visits 10 vs 12, q 0.141 vs 0.129, net
# neutral) that lands on the losing move with seed 0. See
# iter-11.md §5.2.
XFAIL = {
    "fl1-g9-ply11": "q-vs-visits gap; H1's territory, see iter-11.md §5.1",
    "fl2-g0-ply18": "q-vs-visits gap; H1's territory, see iter-11.md §5.2",
    "fl2-g7-ply23": "near-tie coin-flip; no systematic bias, see iter-11.md §5.2",
    "fl2-g8-ply24": "q-vs-visits gap; H1's territory, see iter-11.md §5.2",
    "fl2-g9-ply67": "q-vs-visits gap (mild); H1's territory, see iter-11.md §5.2",
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

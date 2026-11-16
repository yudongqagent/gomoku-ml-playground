"""Regression tests for v12 (batch 5, H1).

Each position in ``gomoku_ml/regression_positions_v12.json`` is a
critical position from v12's fast loop vs v7 (see
``research/iterations/iter-12.md`` §2 for the verification behind
each one). The engine must play an expected move on every
position. A position the current engine does NOT fix is kept as an
xfail with its reason — a documented open weakness — never silently
dropped.

NOTE: iter-9 §2.6's swap game 41 ply 21 position is deliberately
NOT here — it does not verify as a single losing move (the
preventing moves also lose 3/3 in match-engine playouts), so there
is no verified saving move to assert (iter-12.md §1.0).
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v12.json"
)

# Positions the current engine does not fix (open weaknesses).
# fl2-g2-ply20: no visit tie — the actual move wins visits outright
# (12 vs 9/6); the saving moves are starved by lower priors (the
# exploration class from iter-7's fl1-g3 xfail, not a root-decision
# fix). fl2-g1-ply53: the tie is between two losing forcing moves;
# the saves sit at 3 visits vs 7 — same exploration class.
# fl2-g5-ply37: knife-edge tie at 8 visits between a saver and a
# loser separated by 1 v1 point (24015 vs 24016), neither forcing;
# no tie-break signal in the pre-registered set separates them.
# See iter-12.md §2.2.
XFAIL = {
    "fl2-g2-ply20": "saving moves starved by lower priors; see iter-12.md §2.2",
    "fl2-g1-ply53": "saves at 3 visits vs 7; tie is between two losers; see iter-12.md §2.2",
    "fl2-g5-ply37": "knife-edge tie, 1 v1 point apart, neither forcing; see iter-12.md §2.2",
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
def test_v12_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v12 = scoring.get_player(12, seed=0)
    move = v12.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v12 played {chosen}, expected one of "
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


def test_v12_is_tie_prior_forcing_gradient():
    v12 = scoring.get_player(12, seed=0)
    assert v12.name == "scoring-v12"
    assert v12._mcts.leaf_mode == "gradient"
    assert v12._mcts.root_decision == "tie-prior-forcing"
    assert v12._mcts.sims == 100

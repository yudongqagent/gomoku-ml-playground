"""Regression tests for v40 (batch 11, H2 leaf-blind patch).

Each position in ``gomoku_ml/regression_positions_v40.json`` is one
of the 4 diagnosed missed-prevention instances from batch 10
(iter-34 §3.6 — v27/v34 play an unsafe move when a safe one exists),
plus the prev-v1bl-g2 tactical position, which v40 REGRESSES on
(the must-block upgrade breaks the 8-8 visit tie v27's tie-break
resolves). See ``research/iterations/iter-40.md`` §1.1/§1.3 for the
diagnosis behind each one.

Status:
* b10-g73-ply35 PASSES: v40 plays a safe move ((10, 6)) — the
  must-block upgrade prices Black's VCF-start line at -1 and the
  unsafe (11, 9) drops to 2 visits at our-q -1.0.
* b10-g61-ply21, b10-g43-ply17 XFAIL: blunder-dilution class — the
  1-ply leaf is correct (-1.0 for the unsafe move); the mis-ranking
  is MCTS averaging over opponent blunder lines, which no leaf
  patch addresses. Documented open weaknesses, not dropped.
* b10-g31-ply21 XFAIL: visit-allocation class — q prefers the
  safes but the unsafe move's prior is 3x, so most-visited picks
  it. Fixing it needs the CLOSED q-based root-decisions lever.
* tact-prev-v1bl-g2 XFAIL: the upgrade shifts q values in the
  (7, 5) subtree, breaking the 8-8 tie; v40 plays the documented
  loser (7, 9). Clause 3 is 28/29. Documented regression.

All four prevention instances are outcome-free (the safe moves
lose 3/3 under v27-vs-v27 playouts — iter-40 §1.1), so these
positions pin the leaf behaviour, not game outcomes.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v40.json"
)

# Positions the current engine does not fix (open weaknesses /
# documented regression). See the module docstring and iter-40.md.
XFAIL = {
    "b10-g61-ply21": "blunder dilution: v40 plays (8, 1), not a safe move; see iter-40.md §1.1",
    "b10-g31-ply21": "visit allocation: v40 plays (9, 6), not a safe move; see iter-40.md §1.1",
    "b10-g43-ply17": "blunder dilution: v40 plays (7, 8), not a safe move; see iter-40.md §1.1",
    "tact-prev-v1bl-g2": "REGRESSION: v40 plays (7, 9), the documented loser; the upgrade breaks the 8-8 tie; see iter-40.md §1.3",
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
def test_v40_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v40 = scoring.get_player(40, seed=0)
    move = v40.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v40 played {chosen}, expected one of "
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

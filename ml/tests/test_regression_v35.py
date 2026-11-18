"""Regression tests for v35 (batch 10, H3: v27 loss forensics).

Each position in ``gomoku_ml/regression_positions_v35.json`` is a
verified VCF-prevention save from a v27 battery loss (see
``research/iterations/iter-35.md`` §2 for the verification behind
each one): v27 played the losing move, the VCF check plays a
saving prevention. The engine must play an expected move on every
position. A position the current engine does NOT fix is kept as an
xfail with its reason — a documented open weakness and the next
iteration's target — never silently dropped.

The defensive-only flag is also tested here: on a pure offensive
VCF position (tactical vcf-1) the defensive-only check must NOT
fire (v35 falls through to the search), while the full check (v29)
fires and plays the VCF start. v35 is defensive-only because the
forensics verified missed VCF preventions, not missed forced wins
(iter-35 §3).
"""

import json
import os

import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import HeuristicMCTS

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v35.json"
)
TACTICAL_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tactical_positions.json"
)

# Positions the current engine does not fix (open weaknesses).
# vcf-prev-v27v7-g32-ply24: White holds 2 VCF starts [(10,2),
# (6,6)]; both (10,2) and (6,6) are preventions, but the check
# picks (10,2) by prior noise (0.1381 vs 0.1380) and it loses 3/3
# while (6,6) saves 2W+1D. The search also prefers (10,2) (7
# visits vs 5), so VCF-filtered search would likely pick it too.
# See iter-35.md §2.1 (game 32). Documented open weakness.
XFAIL = {
    "vcf-prev-v27v7-g32-ply24": (
        "check picks the losing prevention (10,2) by prior noise "
        "(0.1381 vs 0.1380); (6,6) saves; see iter-35.md §2.1"
    ),
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
def test_v35_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v35 = scoring.get_player(35, seed=0)
    move = v35.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v35 played {chosen}, expected one of "
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


def test_v35_defensive_only_skips_offensive_vcf():
    """On tactical vcf-1 (a pure offensive VCF: Black holds a VCF
    start, White holds none), v35's defensive-only check returns
    None (falls through to the search) while v29's full check
    returns the VCF start (8, 3)."""
    with open(TACTICAL_PATH, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    vcf1 = next(p for p in positions if p["id"] == "vcf-1")
    board = game.empty_board()
    for x, y in vcf1["black"]:
        board[y, x] = game.BLACK
    for x, y in vcf1["white"]:
        board[y, x] = game.WHITE
    player = int(vcf1["to_move"])
    v35 = scoring.get_player(35, seed=0)
    v29 = scoring.get_player(29, seed=0)
    assert v35._mcts._vcf_root_move(board, player) is None
    full = v29._mcts._vcf_root_move(board, player)
    assert full is not None and (full % 15, full // 15) == (8, 3)


def test_vcf_root_defensive_only_flag_defaults_off():
    """The flag defaults to False: a plain HeuristicMCTS with
    vcf_root=True behaves exactly like v29 (full check)."""
    mcts = HeuristicMCTS(leaf_mode="rollout2_v1_both", vcf_root=True)
    assert mcts.vcf_root_defensive_only is False

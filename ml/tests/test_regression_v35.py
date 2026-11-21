"""Regression tests for v35 (batch 10, H3: v27 loss forensics).

Each position in ``gomoku_ml/regression_positions_v35.json`` is a
verified VCF-prevention save from a v27 battery loss (see
``research/iterations/iter-35.md`` §2 for the verification behind
each one): v27 played the losing move, the VCF check plays a
saving prevention. The engine must play an expected move on every
position. A position the current engine does NOT fix is kept as an
xfail with its reason — a documented open weakness and the next
iteration's target — never silently dropped.

The check is v29's FULL check (offensive step first, then
defensive) with a SEARCH-FIRST defensive step: the search runs
first, and the check overrides ONLY when the search's chosen
move is NOT a prevention (leaves opponent VCF starts live).
When the search already plays a prevention, its visits/q
signal chooses among preventions, not the prior — the plain
check overrode a correct search prevention by prior noise in
3 verified fast-loop instances (iter-35 §4.2). The
``search-first-fast-v0-*`` positions pin that behaviour: v35
must play the search's prevention there, not the check's.
A defensive-only variant was tried and refuted by the
tactical mechanism check (27/29 — in race positions, where both
sides hold a VCF start, the defensive-only check blocked instead
of taking our forced win; iter-35 §3). The race test below pins
the offensive step's priority.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v35.json"
)
TACTICAL_HARD_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tactical_positions_hard.json"
)

# Positions the current engine does not fix (open weaknesses).
# vcf-prev-v27v7-g32-ply24: White holds 2 VCF starts [(10,2),
# (6,6)]; both (10,2) and (6,6) are preventions, but the check
# picks (10,2) by prior noise (0.1381 vs 0.1380) and it loses 3/3
# while (6,6) saves 2W+1D. The search also prefers (10,2) (7
# visits vs 5), so VCF-filtered search would likely pick it too.
# See iter-35.md §2.1 (game 32). Documented open weakness.
# vcf-prev-v27v7-g44-ply40: the search plays (12,10), which IS
# a prevention (removes White's VCF start) but loses 3/3; the
# plain check's prior picked the saving (10,13), but search-first
# v35 does not override a prevention — it plays the search's
# (12,10). The trade for the 3 fast-loop override-harm fixes
# (see iter-35.md §4.2). Documented limitation.
XFAIL = {
    "vcf-prev-v27v7-g32-ply24": (
        "check picks the losing prevention (10,2) by prior noise "
        "(0.1381 vs 0.1380); (6,6) saves; see iter-35.md §2.1"
    ),
    "vcf-prev-v27v7-g44-ply40": (
        "search-first does not override a prevention: the search "
        "plays (12,10), a losing prevention; the plain check's "
        "prior picked the saving (10,13); see iter-35.md §4.2"
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


def test_v35_offensive_vcf_fires_first_in_race():
    """On tactical vcf-race-block-or-win (a RACE: both sides hold
    a VCF start), v35's check must play OUR VCF start (8, 3) —
    the forced win — not a defensive prevention. A
    defensive-only check played the block (8, 11) here and
    failed the position (27/29); the offensive step's priority
    is load-bearing (iter-35 §3)."""
    with open(TACTICAL_HARD_PATH, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    race = next(p for p in positions if p["id"] == "vcf-race-block-or-win")
    board = game.empty_board()
    for x, y in race["black"]:
        board[y, x] = game.BLACK
    for x, y in race["white"]:
        board[y, x] = game.WHITE
    player = int(race["to_move"])
    v35 = scoring.get_player(35, seed=0)
    move = v35.choose_move(board, player)
    assert (move % 15, move // 15) == (8, 3), (
        f"v35 played {(move % 15, move // 15)} in the race position, "
        "expected the VCF start (8, 3)"
    )

"""Regression tests for v45 (batch 12, H1: fix v35's _leaves_opp_vcf hole).

v45 = v35 + the fixed search-first defensive VCF check (see
``research/iterations/iter-45.md`` §1): the gate FULLY re-checks
VCF after the candidate move (winning moves first, then every
four-creator on the resulting board) instead of re-testing only
the opponent's ORIGINAL VCF starts, and the defensive selector
runs UNGATED when the fixed gate fires.

The g0-ply-29 position (``regression_positions_v45.json``) is
batch 11's verified non-monotonicity instance (iter-39 §1.1):
after the actual (0,6) Black holds 0 VCF / 0 creators / 1 setup
[(5,9)]; occupying (5,9) removes the setup but leaves Black with
FIVE VCF starts. The position is a forced block in a real game
(Black holds the immediate win (0,6)), so it tests the GATE
functions directly, not choose_move.

Honesty note (iter-45 §1.3): at g0-ply-29 the old and new gates
AGREE on every candidate move (both True for (5,9) — 4 original
VCF starts survive it), so the instance demonstrates the
PHENOMENON but does not discriminate the two gates. The
discriminating test (old False, new True) is added from the broad
scan's first disagreement (research/iterations/
hole-scan-broad.json) when one is found — see iter-45 §3.0.
"""

import json
import os

import pytest

from gomoku_ml import evaluate, game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v45.json"
)
V35_POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v35.json"
)


def _load_positions(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


def _vcf_starts_after(board, player, move):
    """Exact VCF starts for the opponent after ``move`` (winning
    moves + four-creator starts) — the independent cross-check."""
    b2 = game.apply_move(board, move, player)
    out = list(scoring._winning_moves(b2, -player))
    for m in evaluate.four_creating_moves(b2, -player):
        if evaluate.starts_forced_win(b2, -player, m, depth=3):
            out.append(m)
    return out


def _g0_position():
    return next(
        p for p in _load_positions(POSITIONS_PATH)
        if p["id"] == "vcf-nonmono-b33v27-g0-ply29"
    )


def test_g0_position_is_well_formed():
    pos = _g0_position()
    board = _board(pos)
    assert len(pos["black"]) + len(pos["white"]) == int(
        (board != game.EMPTY).sum()
    )
    assert board[9, 5] == game.EMPTY  # the (5,9) test move is legal
    assert board[6, 0] == game.EMPTY  # the actual (0,6) is legal


def test_g0_actual_move_leaves_no_vcf():
    """After the actual (0,6) — the forced block of Black's
    immediate win — Black holds 0 VCF starts (the mining's
    0/0/1 record: 0 VCF, 0 creators, 1 setup). Both gates agree
    here (False): (0,6) IS a prevention."""
    pos = _g0_position()
    board = _board(pos)
    player = game.WHITE
    move = game.xy_to_move(0, 6)
    assert _vcf_starts_after(board, player, move) == []
    v45 = scoring.get_player(45, seed=0)
    mcts = v45._mcts
    assert mcts._leaves_opp_vcf_full(board, player, move) is False
    assert mcts._leaves_opp_vcf_subset(board, player, move) is False


def test_fixed_gate_catches_g0_setup_square():
    """Occupying the setup square (5,9) leaves Black with FIVE
    VCF starts (4 four-creator starts + the immediate win
    (0,6)) — the fixed gate returns True (not a prevention),
    and the exact count is 5, reconciling the mining record.
    (The old gate also returns True here — 4 original VCF
    starts survive — so this test pins the fixed gate's correct
    answer and the exact count, not a disagreement; the
    discriminating case comes from the broad scan — see the
    module docstring.)"""
    pos = _g0_position()
    board = _board(pos)
    player = game.WHITE
    move = game.xy_to_move(5, 9)
    starts = _vcf_starts_after(board, player, move)
    assert len(starts) == pos["vcf_after_test_move"] == 5
    # The immediate win (0,6) is among them — the winning-move
    # hole: it is a VCF start but NOT a four-creator.
    assert game.xy_to_move(0, 6) in starts
    creators = evaluate.four_creating_moves(
        game.apply_move(board, move, player), game.BLACK
    )
    assert game.xy_to_move(0, 6) not in creators
    v45 = scoring.get_player(45, seed=0)
    mcts = v45._mcts
    assert mcts._leaves_opp_vcf_full(board, player, move) is True
    # The old gate also returns True here (4 original VCF starts
    # survive) — asserted to pin the documented agreement, not a
    # disagreement; the discriminating case comes from the broad
    # scan (see the module docstring).
    assert mcts._leaves_opp_vcf_subset(board, player, move) is True


def test_fixed_gate_checks_winning_moves_first():
    """A position where the opponent's ONLY VCF start after our
    move is an immediate winning move (not a four-creator): the
    fixed gate must return True. g0-ply-29's (5,9) has 4 creator
    starts too, so this test uses a minimal synthetic board: Black
    holds an open four (four in a row, one open end) and White's
    move elsewhere does not block it. After White's move Black
    holds the immediate win and no four-creator VCF start is
    needed — the winning-moves-first check catches it."""
    board = game.empty_board()
    # Black: four in a row on row 7, cols 3-6, with the left end
    # blocked by White at (2,7) — so (7,7) is Black's ONLY
    # winning move (a half-open four; an open four would have
    # two winning squares).
    for x in (3, 4, 5, 6):
        board[7, x] = game.BLACK
    board[7, 2] = game.WHITE
    # White: stones far away so the board is non-empty.
    board[0, 0] = game.WHITE
    board[14, 14] = game.WHITE
    v45 = scoring.get_player(45, seed=0)
    mcts = v45._mcts
    # White plays (0,1) — does not block Black's win at (7,7).
    move = game.xy_to_move(0, 1)
    b2 = game.apply_move(board, move, game.WHITE)
    assert scoring._winning_moves(b2, game.BLACK) == [
        game.xy_to_move(7, 7)
    ]
    assert mcts._leaves_opp_vcf_full(board, game.WHITE, move) is True
    # And blocking the win IS a prevention.
    block = game.xy_to_move(7, 7)
    assert mcts._leaves_opp_vcf_full(board, game.WHITE, block) is False


def test_ungated_selector_matches_gated_when_gate_open():
    """When the opponent holds a VCF start on the current board
    (the gate open), the ungated selector returns the SAME move
    as the gated one (same test set, same first-success rule) —
    on every v35 regression position where the gate is open."""
    v45 = scoring.get_player(45, seed=0)
    mcts = v45._mcts
    n_open = 0
    for pos in _load_positions(V35_POSITIONS_PATH):
        board = _board(pos)
        player = game.BLACK if pos["to_move"] == "black" else game.WHITE
        opp_vcf = [
            m
            for m in evaluate.four_creating_moves(board, -player)
            if evaluate.starts_forced_win(board, -player, m, depth=3)
        ]
        if not opp_vcf:
            continue
        n_open += 1
        gated = mcts._vcf_defensive_move(board, player)
        ungated = mcts._vcf_defensive_move_ungated(board, player)
        assert ungated == gated, (
            f"{pos['id']}: gated={gated}, ungated={ungated}")
    assert n_open >= 1, "no gate-open regression position found"


def test_v45_flag_wiring():
    """v45 has all three flags on; v35 has full_recheck off; v27
    has none (the flag defaults keep every other version
    untouched)."""
    v45 = scoring.get_player(45, seed=0)
    assert v45._mcts.vcf_root is True
    assert v45._mcts.vcf_root_search_first is True
    assert v45._mcts.vcf_full_recheck is True
    v35 = scoring.get_player(35, seed=0)
    assert v35._mcts.vcf_root is True
    assert v35._mcts.vcf_root_search_first is True
    assert v35._mcts.vcf_full_recheck is False
    v27 = scoring.get_player(27, seed=0)
    assert v27._mcts.vcf_root is False
    assert v27._mcts.vcf_root_search_first is False
    assert v27._mcts.vcf_full_recheck is False


@pytest.mark.parametrize(
    "pos", _load_positions(V35_POSITIONS_PATH), ids=lambda p: p["id"]
)
def test_v45_matches_v35_on_regression_positions(pos):
    """Where the old and new gates agree, v45 plays EXACTLY
    v35's move. The scan (research/iterations/hole-scan.json)
    verified old/new agree on the played and expected moves at
    all 8 v35 regression positions, so v45 must reproduce v35's
    move at each (the 2 documented v35 xfails — g32, g44 — are
    inherited limitations, asserted here as move equality with
    v35 rather than as expected-move hits)."""
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v35 = scoring.get_player(35, seed=0)
    v45 = scoring.get_player(45, seed=0)
    assert v45.choose_move(board, player) == v35.choose_move(
        board, player
    ), f"{pos['id']}: v45 diverged from v35"

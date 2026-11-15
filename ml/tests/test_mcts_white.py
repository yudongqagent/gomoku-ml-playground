"""Tests for the colour-asymmetric MCTS (v10, batch 4 H2).

v10 is v7's PUCT MCTS (capped attack-gradient leaf) + v3's 2-ply
threat check at the root + colour-asymmetric sims: 200 as White,
100 as Black (pre-registered, research/iterations/iter-10.md §1).
"""

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import (
    _gradient_leaf_value,
    _terminal_leaf_value,
    HeuristicMCTS,
)
from gomoku_ml.mcts_white import WHITE_SIMS, WhiteMCTS


def _board_with(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def _scattered_board(seed=0, stones=24):
    rng = np.random.default_rng(seed)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:stones]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    return board


def test_white_gets_more_sims_than_black():
    # The pre-registered contract (iter-10 §1): the sims count is
    # the ONLY thing that varies by colour — 200 for White, 100
    # for Black. Verified via the last_sims_used diagnostic on a
    # position that reaches the search (no immediate win/block).
    board = _scattered_board()
    assert not scoring._winning_moves(board, game.BLACK)
    assert not scoring._winning_moves(board, game.WHITE)
    mcts = WhiteMCTS(sims=10, white_sims=20, rng=np.random.default_rng(0))
    assert mcts.last_sims_used is None
    mcts.choose_move(board, game.BLACK)
    assert mcts.last_sims_used == 10
    mcts.choose_move(board, game.WHITE)
    assert mcts.last_sims_used == 20


def test_default_sims_are_100_black_200_white():
    mcts = WhiteMCTS()
    assert mcts.sims == 100
    assert mcts.white_sims == WHITE_SIMS == 200
    assert mcts.leaf_mode == "gradient"


def test_leaf_is_v7_gradient_leaf():
    # v10's leaf is v7's gradient leaf exactly: terminal cases
    # identical to v6's, quiet positions the capped gradient.
    mcts = WhiteMCTS()
    # Quiet position: the leaf value is the gradient, in (-0.2, 0.2).
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    assert _terminal_leaf_value(board, game.BLACK) == 0.0
    assert mcts._leaf_value(board, game.BLACK) == pytest.approx(
        _gradient_leaf_value(board, game.BLACK)
    )
    assert abs(mcts._leaf_value(board, game.BLACK)) <= 0.2
    # Terminal cases are exact, as in v6/v7.
    win = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert mcts._leaf_value(win, game.BLACK) == 1.0
    open_four = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert mcts._leaf_value(open_four, game.BLACK) == -1.0
    broken_four = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert mcts._leaf_value(broken_four, game.BLACK) == -0.25


def test_choose_move_takes_immediate_win():
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = WhiteMCTS(sims=10, white_sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) in [(7, 6), (7, 11)]
    # The win scan decides before the search: no sims recorded.
    assert mcts.last_sims_used is None


def test_choose_move_blocks_single_winning_move():
    # White's broken four: only (5,7) wins; Black must block it.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = WhiteMCTS(sims=10, white_sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) == (5, 7)


def test_choose_move_searches_and_does_not_mutate_board():
    board = _scattered_board()
    before = board.copy()
    mcts = WhiteMCTS(sims=20, white_sims=40)
    move = mcts.choose_move(board, game.BLACK)
    assert board[move // 15, move % 15] == game.EMPTY  # legal move
    assert np.array_equal(board, before)
    assert mcts.last_sims_used == 20
    move = mcts.choose_move(board, game.WHITE)
    assert board[move // 15, move % 15] == game.EMPTY
    assert np.array_equal(board, before)
    assert mcts.last_sims_used == 40


def test_choose_move_deterministic_for_same_seed():
    board = _scattered_board(seed=1, stones=20)
    a = WhiteMCTS(sims=20, white_sims=40, rng=np.random.default_rng(0))
    b = WhiteMCTS(sims=20, white_sims=40, rng=np.random.default_rng(0))
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)
    assert a.choose_move(board, game.WHITE) == b.choose_move(board, game.WHITE)


def test_choose_move_threat_check_picks_safe_move():
    # Ported from v8 (test_mcts_value.py): the opponent holds an
    # open three (White (5,5),(5,6),(5,7) with both ends open —
    # either end creates an open four), so Black must play a safe
    # move: a forcing move, or one leaving White with no live
    # creator.
    board = _board_with(
        [(7, 7), (8, 8), (3, 3)], [(5, 5), (5, 6), (5, 7)]
    )
    creators = scoring._opponent_creators(board, game.BLACK)
    assert creators  # White can create an open four next move
    mcts = WhiteMCTS(sims=20, white_sims=40)
    move = mcts.choose_move(board, game.BLACK)
    assert scoring._move_is_safe(board, move, game.BLACK, creators), (
        f"chose {(move % 15, move // 15)}, not safe against {creators}"
    )
    assert mcts.threat_filter_used == 1


def test_choose_move_threat_check_noop_without_creators():
    # No opponent creators: the filter must not engage.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    assert scoring._opponent_creators(board, game.BLACK) == []
    mcts = WhiteMCTS(sims=20, white_sims=40)
    move = mcts.choose_move(board, game.BLACK)
    assert mcts.threat_filter_used == 0
    assert move != -1


def test_v7_v6_dispatch_unchanged_on_this_branch():
    # NOTE (batch 5, H2): the batch-4-h2 original of this test
    # asserted v10's registration (ENGINE_VERSIONS[10]). v10 was
    # never merged to master, and this branch (from master
    # bf866f7) registers only v13 — so the v10 assertions are
    # dropped here. What must still hold: v7/v6 dispatch to the
    # plain heuristic MCTS (gradient / terminal leaf, no threat
    # check, no colour asymmetry), and WhiteMCTS is exercised
    # directly by the other tests in this file.
    v7 = scoring.get_player(7, seed=0)
    assert type(v7._mcts) is HeuristicMCTS
    assert v7._mcts.leaf_mode == "gradient"
    assert v7._mcts.sims == 100
    # v6 still dispatches to the plain heuristic MCTS (terminal leaf).
    v6 = scoring.get_player(6, seed=0)
    assert type(v6._mcts) is HeuristicMCTS
    assert v6._mcts.leaf_mode == "terminal"


def test_v10_black_matches_v7_when_no_creators():
    # With no opponent creators (threat check a no-op) and the
    # same sims (100) and RNG seed, v10-as-Black must play the
    # same move as v7 — the Black game differs from v7 ONLY by
    # the threat check (iter-10 §1).
    board = _scattered_board(seed=2, stones=20)
    assert scoring._opponent_creators(board, game.BLACK) == []
    v7 = HeuristicMCTS(
        sims=100, rng=np.random.default_rng(0), leaf_mode="gradient"
    )
    v10 = WhiteMCTS(
        sims=100, white_sims=200, rng=np.random.default_rng(0)
    )
    assert v10.choose_move(board, game.BLACK) == v7.choose_move(
        board, game.BLACK
    )


@pytest.mark.xfail(
    reason="pre-registered mechanism check (iter-10 §1): 200 White "
    "sims close the visit gap (7->22 vs 15->26) but do not flip "
    "the move — the search's q does not value the forcing save "
    "(0.016 vs 0.045), and at 400 sims it converges on the block "
    "(113 vs 29 visits). A value problem, not pure visit "
    "starvation; the design stands and the evals decide.",
    strict=False,
)
def test_v10_white_plays_forcing_save_on_fl1_g3_ply23():
    # v7's open xfail (fl1-g3-ply23): White to move, Black holds
    # 2 creators; the saving move (9,12) is FORCING (creates a
    # four, v1 rank 8, prior 0.04) and wins 3/3 in the oracle,
    # while v7 plays the losing block (4,9). v10-as-White with
    # 200 sims should play (9,12) if the sims increase fixes
    # visit starvation of low-prior forcing moves.
    import json
    import os

    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "gomoku_ml",
        "regression_positions_v7.json",
    )
    with open(path, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    pos = next(p for p in positions if p["id"] == "fl1-g3-ply23")
    board = _board_with(pos["black"], pos["white"])
    v10 = WhiteMCTS(sims=100, white_sims=200, rng=np.random.default_rng(0))
    move = v10.choose_move(board, game.WHITE)
    assert (move % 15, move // 15) in [tuple(e) for e in pos["expected"]]

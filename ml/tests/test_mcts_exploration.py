"""Tests for the v15 exploration parameters (batch 6, H1).

The prior floor (unnormalized — see iter-15.md §1.0 for why
renormalization is excluded) and the c_puct passthrough. The
default parameters (floor 0.0, c_puct 1.5) must reproduce v7
exactly: the sweep control cell runs through the same driver as
the sweep cells, so a default-parameter HeuristicMCTS has to play
v7's move.
"""

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import HeuristicMCTS, _priors


def _board_with(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def _midgame_board():
    # A quiet midgame position with many candidates (from a real
    # v7-vs-v1 game shape: centre fight, no immediate wins).
    return _board_with(
        [(7, 7), (6, 6), (8, 7), (9, 7), (5, 7), (7, 6)],
        [(8, 6), (8, 8), (6, 7), (7, 8), (9, 6), (5, 5)],
    )


def test_priors_default_sum_to_one():
    priors = _priors(_midgame_board(), game.BLACK)
    assert abs(sum(priors.values()) - 1.0) < 1e-9


def test_prior_floor_zero_is_identity():
    board = _midgame_board()
    base = _priors(board, game.BLACK)
    floored = _priors(board, game.BLACK, prior_floor=0.0)
    assert floored == base


def test_prior_floor_lifts_tail_without_renormalizing():
    board = _midgame_board()
    base = _priors(board, game.BLACK)
    floored = _priors(board, game.BLACK, prior_floor=0.01)
    # Every below-floor prior is raised to exactly the floor.
    for m, p in base.items():
        assert floored[m] == max(p, 0.01)
    # Above-floor priors are UNCHANGED (no renormalization): the
    # sharp prior is preserved where it matters.
    for m, p in base.items():
        if p > 0.01:
            assert floored[m] == p
    # The sum exceeds 1 (unnormalized by design) — and at least
    # one move was actually lifted.
    assert sum(floored.values()) > 1.0
    assert any(p < 0.01 for p in base.values())


def test_prior_floor_preserves_above_floor_ratios():
    board = _midgame_board()
    base = _priors(board, game.BLACK)
    floored = _priors(board, game.BLACK, prior_floor=0.005)
    above = [m for m, p in base.items() if p > 0.005]
    assert len(above) >= 2
    a, b = above[0], above[1]
    assert floored[a] / floored[b] == pytest.approx(base[a] / base[b])


def test_prior_floor_rejects_out_of_range():
    with pytest.raises(ValueError):
        HeuristicMCTS(prior_floor=-0.1)
    with pytest.raises(ValueError):
        HeuristicMCTS(prior_floor=1.0)


def test_c_puct_and_floor_thread_through_scoring_dispatch():
    # ENGINE_VERSIONS entries carry c_puct / prior_floor; the
    # dispatch must pass them to HeuristicMCTS. v7 carries neither
    # key, so it keeps the defaults.
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.c_puct == 1.5
    assert v7._mcts.prior_floor == 0.0
    v6 = scoring.get_player(6, seed=0)
    assert v6._mcts.c_puct == 1.5
    assert v6._mcts.prior_floor == 0.0


def test_default_sweep_engine_matches_v7():
    # The sweep control cell (HeuristicMCTS at default params)
    # must play v7's move on a fixed position set — otherwise the
    # sweep cells and the control would differ by more than the
    # exploration parameter.
    boards = [
        (_midgame_board(), game.BLACK),
        (_midgame_board(), game.WHITE),
        (_board_with([(7, 7)], [(7, 5)]), game.BLACK),
        (_board_with([(7, 7), (6, 6), (8, 8)], [(7, 9), (9, 9), (5, 5)]),
         game.WHITE),
    ]
    for board, player in boards:
        v7 = scoring.get_player(7, seed=0)
        sweep = HeuristicMCTS(sims=100, leaf_mode="gradient",
                              rng=np.random.default_rng(0))
        assert sweep.choose_move(board.copy(), player) == (
            v7.choose_move(board.copy(), player))


def test_floor_changes_search_but_keeps_win_scan():
    # The win/block pre-search scans run before the MCTS in every
    # version: a floor must not change an immediate win.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = HeuristicMCTS(sims=100, leaf_mode="gradient",
                         prior_floor=0.01,
                         rng=np.random.default_rng(0))
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) in [(7, 6), (7, 11)]

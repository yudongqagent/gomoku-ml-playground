"""Tests for v17: the combination (batch 6, H3).

v17 = v7's PUCT MCTS + v12's tie-prior-forcing root rule + v14's
v1-kernel gradient leaf + v13's 2-ply threat check at the root
(pre-registered, research/iterations/iter-17.md §1). All three
components are ported from research/batch-5, not re-derived.
"""

import numpy as np

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import (
    _gradient_v1_leaf_value,
    HeuristicMCTS,
)
from gomoku_ml.mcts_v17 import V17MCTS


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


def test_v17_defaults():
    mcts = V17MCTS()
    assert mcts.sims == 100
    assert mcts.c_puct == 1.5
    assert mcts.leaf_mode == "gradient_v1"
    assert mcts.root_decision == "tie-prior-forcing"
    assert mcts.threat_filter_used == 0


def test_v17_dispatch():
    # ENGINE_VERSIONS[17] dispatches to V17MCTS (the combination),
    # not to plain HeuristicMCTS.
    player = scoring.get_player(17, seed=0)
    assert isinstance(player._mcts, V17MCTS)
    assert player._mcts.sims == 100
    assert player._mcts.leaf_mode == "gradient_v1"
    assert player._mcts.root_decision == "tie-prior-forcing"


def test_v17_leaf_is_gradient_v1():
    # v17's leaf is v14's v1-kernel gradient leaf exactly (same
    # function object), so terminal values and the capped
    # gradient are v14's by construction.
    mcts = V17MCTS()
    assert mcts._leaf_value is _gradient_v1_leaf_value
    board = _scattered_board(seed=3, stones=20)
    assert (
        mcts._leaf_value(board, game.BLACK)
        == _gradient_v1_leaf_value(board, game.BLACK)
    )


def test_v17_matches_leaf_plus_root_rule_without_creators():
    # With no opponent creators (threat check a no-op) and the
    # same sims and RNG seed, v17 must play the same move as a
    # plain HeuristicMCTS with the v14 leaf and the v12 root
    # rule — the combination differs from leaf+root-rule ONLY
    # by the threat check (mirrors the v10/v7 test).
    board = _scattered_board(seed=2, stones=20)
    assert scoring._opponent_creators(board, game.BLACK) == []
    plain = HeuristicMCTS(
        sims=100,
        rng=np.random.default_rng(0),
        leaf_mode="gradient_v1",
        root_decision="tie-prior-forcing",
    )
    v17 = V17MCTS(sims=100, rng=np.random.default_rng(0))
    assert v17.choose_move(board, game.BLACK) == plain.choose_move(
        board, game.BLACK
    )
    assert v17.threat_filter_used == 0


def test_v17_threat_check_picks_safe_move():
    # Ported from v10 (test_mcts_white.py): the opponent holds an
    # open three (White (5,5),(5,6),(5,7) with both ends open —
    # either end creates an open four), so Black must play a safe
    # move: a forcing move, or one leaving White with no live
    # creator.
    board = _board_with(
        [(7, 7), (8, 8), (3, 3)], [(5, 5), (5, 6), (5, 7)]
    )
    creators = scoring._opponent_creators(board, game.BLACK)
    assert creators  # White can create an open four next move
    mcts = V17MCTS(sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert scoring._move_is_safe(board, move, game.BLACK, creators), (
        f"chose {(move % 15, move // 15)}, not safe against {creators}"
    )
    assert mcts.threat_filter_used == 1


def test_v17_threat_check_noop_without_creators():
    # No opponent creators: the filter must not engage.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    assert scoring._opponent_creators(board, game.BLACK) == []
    mcts = V17MCTS(sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert mcts.threat_filter_used == 0
    assert move != -1


def test_v17_plays_forcing_on_iter9_s26_position():
    # The iter-9 §2.6 diagnostic (batch-v9-vs-v7-swap.json game
    # 41, ply 21, White to move; stones hard-coded from the
    # record): a three-way 5-visit tie where v9/v7 played the
    # losing creator-block (7,8) and v12 plays the forcing (3,11).
    # v17 must play a forcing move too — here via the threat
    # filter ((7,8) is unsafe: it occupies one of two
    # diagonal-adjacent creators, leaving the other live) as well
    # as via the tie-prior-forcing rule. NOTE (iter-12 §1.0):
    # ply 21 does NOT verify as a single losing move, so this
    # tests only that the rule does what it claims, not that the
    # move saves the game.
    board = _board_with(
        [(7, 7), (11, 11), (5, 13), (10, 8), (6, 10), (6, 9),
         (7, 10), (5, 10), (5, 7), (8, 7), (10, 12)],
        [(9, 9), (7, 11), (11, 7), (5, 9), (7, 9), (6, 8),
         (5, 8), (8, 10), (4, 10), (9, 11)],
    )
    mcts = V17MCTS(sims=100, rng=np.random.default_rng(0))
    move = mcts.choose_move(board, game.WHITE)
    assert (move % 15, move // 15) in [(3, 11), (2, 12)], (
        f"v17 played {(move % 15, move // 15)}, expected a forcing "
        "move (3,11) or (2,12)"
    )
    assert scoring._is_forcing(board, move, game.WHITE)


def test_v17_choose_move_deterministic_for_same_seed():
    board = _scattered_board(seed=1, stones=20)
    a = V17MCTS(sims=50, rng=np.random.default_rng(0))
    b = V17MCTS(sims=50, rng=np.random.default_rng(0))
    assert a.choose_move(board, game.BLACK) == b.choose_move(
        board, game.BLACK
    )


def test_v17_choose_move_does_not_mutate_board():
    board = _scattered_board(seed=1, stones=20)
    before = board.copy()
    V17MCTS(sims=20).choose_move(board, game.BLACK)
    assert np.array_equal(board, before)

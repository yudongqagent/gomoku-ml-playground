"""Tests for the heuristic-prior MCTS (v6, batch 2 H2; v7, batch 3 H1)."""

import math

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import (
    HeuristicMCTS,
    _Node,
    _attack_gradient,
    _gradient_leaf_value,
    _priors,
    _terminal_leaf_value,
)


def _board_with(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def test_priors_sum_to_one_and_favour_winning_moves():
    # Black has a four in a row: (7,6) and (7,11) both win.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    priors = _priors(board, game.BLACK)
    assert abs(sum(priors.values()) - 1.0) < 1e-9
    best = max(priors, key=priors.get)
    assert (best % 15, best // 15) in [(7, 6), (7, 11)]


def test_priors_empty_on_full_board():
    board = np.ones((15, 15), dtype=np.int8)
    assert _priors(board, game.BLACK) == {}


def test_terminal_leaf_value_player_to_move_wins():
    # Black to move has a winning move (four in a row).
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _terminal_leaf_value(board, game.BLACK) == 1.0


def test_terminal_leaf_value_opponent_two_winning_moves():
    # White holds an open four (two winning moves); Black has none.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _terminal_leaf_value(board, game.BLACK) == -1.0


def test_terminal_leaf_value_opponent_one_winning_move():
    # White holds a broken four: only the gap (5,7) wins.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _terminal_leaf_value(board, game.BLACK) == -0.25


def test_terminal_leaf_value_quiet_is_zero():
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    assert _terminal_leaf_value(board, game.BLACK) == 0.0


def test_choose_move_takes_immediate_win():
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = HeuristicMCTS(sims=10)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) in [(7, 6), (7, 11)]


def test_choose_move_blocks_single_winning_move():
    # White's broken four: only (5,7) wins; Black must block it.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = HeuristicMCTS(sims=10)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) == (5, 7)


def test_choose_move_does_not_mutate_board():
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = HeuristicMCTS(sims=20, rng=np.random.default_rng(0))
    mcts.choose_move(board, game.BLACK)
    assert np.array_equal(board, before)


def test_choose_move_deterministic_for_same_seed():
    rng = np.random.default_rng(1)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:20]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    a = scoring.get_player(6, seed=0)
    b = scoring.get_player(6, seed=0)
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)


def test_v6_registered_and_dispatches_to_mcts():
    v6 = scoring.get_player(6, seed=0)
    assert v6.name == "scoring-v6"
    assert v6.mcts is True
    assert v6._mcts is not None
    assert v6._mcts.sims == 100
    assert scoring.get_player(1, seed=0).mcts is False


def test_v6_finds_iter5_game1_saving_move():
    # Regression (iter-5 §2.2): v5-vs-v0 game 1, White to move at
    # move 30. v5 played the losing (4,9) — ranked #1 by 1-ply
    # score. The oracle-verified best moves are (7,11) and (6,12)
    # (both lead to a White win in the playout; (4,9) loses in 26
    # plies). v6's MCTS must find one of them — the top-8 minimax
    # could not (they rank 11th-12th by 1-ply score).
    moves = [(7, 7), (7, 9), (8, 8), (9, 9), (8, 9), (8, 10),
             (6, 8), (10, 8), (11, 7), (9, 7), (8, 6), (9, 5),
             (7, 8), (5, 8), (6, 7), (5, 6), (5, 7), (8, 7),
             (6, 6), (6, 5), (4, 7), (3, 7), (9, 8), (5, 9),
             (4, 8), (7, 5), (8, 5), (4, 4), (4, 6)]
    board = game.empty_board()
    for i, (x, y) in enumerate(moves):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    v6 = scoring.get_player(6, seed=0)
    move = v6.choose_move(board, game.WHITE)
    assert (move % 15, move // 15) in [(7, 11), (6, 12)]


# --- v7 (batch 3, H1): attack-gradient leaf -------------------------------


def test_gradient_leaf_matches_terminal_leaf_on_terminal_positions():
    # The gradient leaf is IDENTICAL to the terminal leaf on every
    # non-quiet position, by construction.
    # Player to move wins.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _gradient_leaf_value(board, game.BLACK) == 1.0
    # Opponent holds an open four (two winning moves).
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _gradient_leaf_value(board, game.BLACK) == -1.0
    # Opponent holds a broken four (one winning move).
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _gradient_leaf_value(board, game.BLACK) == -0.25


def test_attack_gradient_zero_on_empty_board():
    # Both sides have identical attack scores on an empty board.
    board = game.empty_board()
    assert _attack_gradient(board, game.BLACK) == 0.0
    assert _attack_gradient(board, game.WHITE) == 0.0
    assert _gradient_leaf_value(board, game.BLACK) == 0.0


def test_attack_gradient_antisymmetric_and_bounded():
    # A mid-game board: the gradient for Black is exactly the
    # negative of the gradient for White, and strictly inside
    # (-0.2, 0.2) — the fast-loop iteration-1 cap, below the
    # must-block value 0.25.
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    gb = _attack_gradient(board, game.BLACK)
    gw = _attack_gradient(board, game.WHITE)
    assert gb == -gw
    assert -0.2 < gb < 0.2


def test_attack_gradient_ordering_open_three_vs_nothing():
    # Black holds an open three, White a lone stone: the gradient
    # is positive for Black (more threats) and negative for White.
    # Quiet position (no winning move for either side).
    board = _board_with([(7, 7), (8, 7), (9, 7)], [(3, 3)])
    assert scoring._winning_moves(board, game.BLACK) == []
    assert scoring._winning_moves(board, game.WHITE) == []
    gb = _attack_gradient(board, game.BLACK)
    gw = _attack_gradient(board, game.WHITE)
    assert gb > 0.0
    assert gw < 0.0
    # The gradient leaf takes the gradient branch on quiet boards.
    assert _gradient_leaf_value(board, game.BLACK) == gb
    # Mirrored: White holds the open three, Black the lone stone.
    board2 = _board_with([(3, 3)], [(7, 7), (8, 7), (9, 7)])
    assert _attack_gradient(board2, game.BLACK) < 0.0
    assert _attack_gradient(board2, game.WHITE) > 0.0


def test_attack_gradient_counts_breadth_not_best():
    # Two open threes beat one: Black with two separate open
    # threes has a strictly larger gradient than Black with one
    # (same best attack in both cases — an open-three maker — so a
    # best-attack gradient could not separate them).
    one = _board_with([(7, 7), (8, 7), (9, 7)], [(3, 3)])
    two = _board_with([(7, 7), (8, 7), (9, 7), (11, 11), (12, 11), (13, 11)],
                      [(3, 3)])
    assert scoring._winning_moves(two, game.BLACK) == []
    assert _attack_gradient(two, game.BLACK) > _attack_gradient(one, game.BLACK)
    # And the value is the capped tanh of the breadth difference:
    # two open threes give Black 4 strong moves vs White's 0
    # (each open three has 2 four-makers), White's lone stone gives
    # only developing moves; check the exact value via the formula.
    from gomoku_ml import kernel_numpy
    from gomoku_ml.mcts_heuristic import _GRADIENT_CAP, _breadth

    black, white = kernel_numpy.attack_both(two)
    mask = two == game.EMPTY
    expected = _GRADIENT_CAP * math.tanh(
        (_breadth(black, mask) - _breadth(white, mask)) / 4.0
    )
    assert _attack_gradient(two, game.BLACK) == expected


def test_v7_registered_and_dispatches_to_gradient_mcts():
    v7 = scoring.get_player(7, seed=0)
    assert v7.name == "scoring-v7"
    assert v7.mcts is True
    assert v7._mcts is not None
    assert v7._mcts.sims == 100
    assert v7._mcts.leaf_mode == "gradient"
    # v6 is unchanged (terminal leaf).
    v6 = scoring.get_player(6, seed=0)
    assert v6._mcts.leaf_mode == "terminal"


def test_invalid_leaf_mode_raises():
    with pytest.raises(ValueError):
        HeuristicMCTS(leaf_mode="bogus")


def test_v7_choose_move_deterministic_for_same_seed():
    rng = np.random.default_rng(1)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:20]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    a = scoring.get_player(7, seed=0)
    b = scoring.get_player(7, seed=0)
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)


def test_v7_choose_move_does_not_mutate_board():
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = HeuristicMCTS(sims=20, leaf_mode="gradient",
                         rng=np.random.default_rng(0))
    mcts.choose_move(board, game.BLACK)
    assert np.array_equal(board, before)


@pytest.mark.xfail(
    reason="Known v7 behaviour, recorded in iter-7.md §1: the "
    "two-sided breadth gradient rewards the losing block (4,9) "
    "over the forcing saving moves (7,11)/(6,12) — 1-ply myopia "
    "reintroduced into the leaf. The evals decide whether it "
    "matters in matches.",
    strict=False,
)
def test_v7_iter5_game1_regression_xfail():
    # Same position as test_v6_finds_iter5_game1_saving_move: v6
    # finds (7,11)/(6,12); v7 plays (4,9). Documented as xfail.
    moves = [(7, 7), (7, 9), (8, 8), (9, 9), (8, 9), (8, 10),
             (6, 8), (10, 8), (11, 7), (9, 7), (8, 6), (9, 5),
             (7, 8), (5, 8), (6, 7), (5, 6), (5, 7), (8, 7),
             (6, 6), (6, 5), (4, 7), (3, 7), (9, 8), (5, 9),
             (4, 8), (7, 5), (8, 5), (4, 4), (4, 6)]
    board = game.empty_board()
    for i, (x, y) in enumerate(moves):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    v7 = scoring.get_player(7, seed=0)
    move = v7.choose_move(board, game.WHITE)
    assert (move % 15, move // 15) in [(7, 11), (6, 12)]


# --- v9 root-decision rules (batch 4, H1) ---------------------------------
# Tests build synthetic roots (no search): a child's raw q is the
# OPPONENT's value under the negamax backup, so "q" in every rule
# means -child.q (iter-9.md §1.0 sign correction).


def _synthetic_root(specs):
    """Root whose children are (move, visits, value_sum, prior).

    Raw q = value_sum / visits (opponent's value); our-q = -q.
    """
    root = _Node(1.0)
    root.children = {}
    for move, visits, value_sum, prior in specs:
        child = _Node(prior)
        child.visit_count = visits
        child.value_sum = value_sum
        root.children[move] = child
    return root


def test_root_decision_default_is_visits():
    mcts = HeuristicMCTS()
    assert mcts.root_decision == "visits"
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.root_decision == "visits"
    v6 = scoring.get_player(6, seed=0)
    assert v6._mcts.root_decision == "visits"


def test_invalid_root_decision_raises():
    with pytest.raises(ValueError):
        HeuristicMCTS(root_decision="bogus")


def test_root_decision_visits_picks_most_visited():
    # Most-visited wins even with the worst our-q.
    root = _synthetic_root([
        (10, 50, 25.0, 0.1),   # our-q -0.5, most visited
        (20, 10, -4.0, 0.1),   # our-q +0.4
        (30, 5, -2.0, 0.1),    # our-q +0.4
    ])
    mcts = HeuristicMCTS(root_decision="visits")
    assert mcts._choose_root_child(root) == 10


def test_root_decision_q_uses_our_perspective_value():
    # SIGN TEST: raw q is the opponent's value, so the child with
    # the LOWEST raw q (worst for the opponent) is the best move
    # for us — even with fewer visits. A rule that argmaxed raw q
    # would pick move 10 (raw q +0.5, great for the opponent).
    root = _synthetic_root([
        (10, 50, 25.0, 0.1),   # raw q +0.5 -> our-q -0.5
        (20, 10, -4.0, 0.1),   # raw q -0.4 -> our-q +0.4 (best)
        (30, 5, -1.0, 0.1),    # raw q -0.2 -> our-q +0.2
    ])
    mcts = HeuristicMCTS(root_decision="q")
    assert mcts._choose_root_child(root) == 20


def test_root_decision_q_ignores_unvisited_children():
    # An unvisited child's q is 0.0 by convention — not a signal.
    # It must never beat a visited child with negative our-q.
    root = _synthetic_root([
        (10, 10, 2.0, 0.1),    # raw q +0.2 -> our-q -0.2
        (20, 0, 0.0, 0.1),     # unvisited: q 0.0 by convention
    ])
    mcts = HeuristicMCTS(root_decision="q")
    assert mcts._choose_root_child(root) == 10


def test_root_decision_q_ties_broken_by_visits():
    root = _synthetic_root([
        (10, 5, -2.0, 0.1),    # our-q +0.4, 5 visits
        (20, 8, -3.2, 0.1),    # our-q +0.4, 8 visits (wins tie)
    ])
    mcts = HeuristicMCTS(root_decision="q")
    assert mcts._choose_root_child(root) == 20


def test_root_decision_tie_q_breaks_visit_ties_by_our_q():
    root = _synthetic_root([
        (10, 10, 4.0, 0.1),    # 10 visits, our-q -0.4
        (20, 10, -1.0, 0.1),   # 10 visits, our-q +0.1 (wins tie)
        (30, 9, -8.1, 0.1),    # 9 visits, our-q +0.9 (not tied)
    ])
    mcts = HeuristicMCTS(root_decision="tie-q")
    assert mcts._choose_root_child(root) == 20


def test_root_decision_blend_formula():
    # score = our-q + 0.1 * (visits / max_visits).
    # Move 10: our-q -0.10, 10/10 visits -> -0.10 + 0.10 = 0.00.
    # Move 20: our-q -0.01, 5/10 visits  -> -0.01 + 0.05 = 0.04.
    # Move 30: our-q +0.20, 1/10 visits  -> +0.20 + 0.01 = 0.21.
    # Pure q would pick 30 too, but visits would pick 10: the blend
    # follows q here because the q gap (0.3) exceeds the 0.1 scale.
    root = _synthetic_root([
        (10, 10, 1.0, 0.1),
        (20, 5, 0.05, 0.1),
        (30, 1, -0.2, 0.1),
    ])
    mcts = HeuristicMCTS(root_decision="blend")
    assert mcts._choose_root_child(root) == 30


def test_root_decision_blend_visits_override_small_q_gap():
    # q gap 0.05 (< 0.1 scale): visits decide. Move 10: our-q
    # -0.05, 10/10 visits -> 0.05. Move 20: our-q 0.00, 1/10 visits
    # -> 0.01. Pure q would pick 20; the blend picks 10.
    root = _synthetic_root([
        (10, 10, 0.5, 0.1),
        (20, 1, 0.0, 0.1),
    ])
    mcts = HeuristicMCTS(root_decision="blend")
    assert mcts._choose_root_child(root) == 10


def test_root_decision_tie_prior_breaks_visit_ties_by_prior():
    root = _synthetic_root([
        (10, 10, 0.0, 0.05),
        (20, 10, 0.0, 0.09),   # same visits, higher prior
        (30, 9, 0.0, 0.50),    # higher prior but fewer visits
    ])
    mcts = HeuristicMCTS(root_decision="tie-prior")
    assert mcts._choose_root_child(root) == 20


def test_root_decision_section26_stats():
    # iter-7 §2.6 root stats (reproduced exactly in iter-9.md
    # §1.0): (7,4) flat 67 and (8,4) flat 68 tie at 10 visits with
    # raw q 0.042 / 0.032; (9,8) flat 129 has 7 visits, raw q
    # 0.071. Correctly signed, q prefers the LOSING (8,4) — so
    # tie-q and q both pick it, and the blend does too (0.068 vs
    # 0.058). Only tie-prior can save the game, and only if (7,4)'s
    # prior exceeds (8,4)'s (v1 ranks (7,4) 3rd, (8,4) 6th).
    root = _synthetic_root([
        (67, 10, 0.42, 0.077),    # (7,4) saving, our-q -0.042
        (68, 10, 0.32, 0.076),    # (8,4) losing, our-q -0.032
        (129, 7, 0.497, 0.078),   # (9,8) saving, our-q -0.071
    ])
    assert HeuristicMCTS(root_decision="tie-q")._choose_root_child(root) == 68
    assert HeuristicMCTS(root_decision="q")._choose_root_child(root) == 68
    assert HeuristicMCTS(root_decision="blend")._choose_root_child(root) == 68
    assert (
        HeuristicMCTS(root_decision="tie-prior")._choose_root_child(root)
        == 67
    )

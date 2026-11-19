"""Tests for the heuristic-prior MCTS (v6, batch 2 H2; v7, batch 3 H1;
v14, batch 5 H3; v16, batch 6 H2; v18, batch 7 H1; v23, batch 8 H3; v27, batch 9 H1; v29, batch 9 H3; v40, batch 11 H2)."""

import json
import math
import os

import numpy as np
import pytest

from gomoku_ml import evaluate, game, scoring
from gomoku_ml.mcts_heuristic import (
    HeuristicMCTS,
    _Node,
    _attack_gradient,
    _attack_gradient_v1,
    _attack_gradient_v1_gap,
    _attack_gradient_v1_gap_black,
    _attack_gradient_v1_gap_ours,
    _breadth_v1,
    _breadth_v1_gap,
    _gradient_leaf_value,
    _gradient_v1_gap_black_leaf_value,
    _gradient_v1_gap_leaf_value,
    _gradient_v1_gap_ours_leaf_value,
    _gradient_v1_leaf_value,
    _priors,
    _rollout_leaf_value,
    _rollout_leaf_value_mbvcf,
    _rollout_move,
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


# --- v16 (batch 6, H2): 2-ply rollout leaf --------------------------------

def test_rollout_leaf_matches_terminal_leaf_on_terminal_positions():
    # The rollout leaf is IDENTICAL to the terminal leaf on every
    # terminal position — the rollout runs only from quiet ones.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _rollout_leaf_value(board, game.BLACK) == 1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _rollout_leaf_value(board, game.BLACK) == -1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _rollout_leaf_value(board, game.BLACK) == -0.25


def test_rollout_leaf_bounded_and_deterministic():
    rng = np.random.default_rng(2)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    v1 = _rollout_leaf_value(board, game.BLACK)
    v2 = _rollout_leaf_value(board, game.BLACK)
    assert -1.0 <= v1 <= 1.0
    assert v1 == v2  # deterministic: no RNG in the leaf


def test_rollout_leaf_does_not_mutate_board():
    rng = np.random.default_rng(3)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    _rollout_leaf_value(board, game.WHITE)
    assert np.array_equal(board, before)


def test_rollout_move_wins_blocks_and_loses():
    # Winning move: taken, terminal +1.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    move, term = _rollout_move(board, game.BLACK)
    assert (move % 15, move // 15) in [(7, 6), (7, 11)]
    assert term == 1.0
    # Single opponent winning move: blocked (game continues).
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    move, term = _rollout_move(board, game.BLACK)
    assert (move % 15, move // 15) == (5, 7)
    assert term is None
    # Two opponent winning moves: terminal -1.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    move, term = _rollout_move(board, game.BLACK)
    assert term == -1.0


def test_rollout_move_prefers_open_four_over_defensive_block():
    # FAST-LOOP PATCH (iter-16 §2.2): the rollout policy's argmax
    # of (own + opp) v1 attack picked a defensive block over an own
    # open-four creation — a forced win. Fast-loop game 9, after
    # White's (5,5): Black's (6,3) creates an open four (own v1
    # attack 200080, two winning moves) but lost the argmax to
    # (6,8) (total 200422: it blocks one White four) by 332 points;
    # (6,8) lets White create a double four and the leaf
    # hallucinated a White win (+1.000) for a losing move. The
    # patched policy plays (6,3) and the leaf is +1.0 for Black.
    # Board: fastloop2-v16-vs-v7.json game 9 after ply 45.
    black = [(7, 7), (7, 6), (7, 5), (8, 5), (6, 7), (4, 9), (8, 4),
             (5, 6), (9, 2), (10, 5), (9, 7), (8, 6), (5, 3), (6, 2),
             (9, 9), (7, 8), (7, 10), (8, 8), (8, 13), (5, 2),
             (4, 5), (7, 2), (5, 4)]
    white = [(6, 6), (7, 9), (7, 4), (6, 5), (9, 4), (5, 8), (8, 3),
             (10, 1), (8, 9), (9, 5), (8, 7), (10, 8), (6, 4),
             (6, 9), (5, 10), (8, 11), (8, 10), (8, 12), (11, 11),
             (9, 11), (3, 4), (8, 2), (5, 5)]
    board = _board_with(black, white)
    move, term = _rollout_move(board, game.BLACK)
    assert (move % 15, move // 15) == (6, 3)
    assert term is None
    assert _rollout_leaf_value(board, game.BLACK) == 1.0


# --- v14 (batch 5, H3): v1-kernel attack-gradient leaf ---------------------


def test_gradient_v1_leaf_matches_terminal_leaf_on_terminal_positions():
    # The v14 leaf is IDENTICAL to the terminal leaf on every
    # non-quiet position, by construction — same as v7.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _gradient_v1_leaf_value(board, game.BLACK) == 1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _gradient_v1_leaf_value(board, game.BLACK) == -1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _gradient_v1_leaf_value(board, game.BLACK) == -0.25


def test_attack_gradient_v1_zero_on_empty_board():
    board = game.empty_board()
    assert _attack_gradient_v1(board, game.BLACK) == 0.0
    assert _attack_gradient_v1(board, game.WHITE) == 0.0
    assert _gradient_v1_leaf_value(board, game.BLACK) == 0.0


def test_attack_gradient_v1_antisymmetric_and_bounded():
    # Same mid-game board as the v7 test: antisymmetric under
    # colour swap, strictly inside (-0.2, 0.2) — the cap is
    # unchanged from v7.
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    gb = _attack_gradient_v1(board, game.BLACK)
    gw = _attack_gradient_v1(board, game.WHITE)
    assert gb == -gw
    assert -0.2 < gb < 0.2
    assert -0.2 < gw < 0.2


def test_attack_gradient_v1_sees_broken_four_creation():
    # The point of v14: Black holds XX_X (a broken-four setup);
    # the v1 gradient counts the broken-four creation as a strong
    # move, the v0 gradient does not. On this board the v1
    # gradient for Black is strictly larger than the v0 gradient.
    board = _board_with([(7, 7), (8, 7), (10, 7)], [(3, 3)])
    assert scoring._winning_moves(board, game.BLACK) == []
    assert scoring._winning_moves(board, game.WHITE) == []
    g_v1 = _attack_gradient_v1(board, game.BLACK)
    g_v0 = _attack_gradient(board, game.BLACK)
    assert g_v1 > g_v0
    assert g_v1 > 0.0
    # And the value is the capped tanh of the v1 breadth
    # difference at the pre-registered scale 7.5.
    from gomoku_ml import kernel_v1
    from gomoku_ml.mcts_heuristic import (
        _GRADIENT_CAP,
        _GRADIENT_V1_SCALE,
        _breadth_v1,
    )

    black, white = kernel_v1.attack_v1_both(board)
    mask = board == game.EMPTY
    expected = _GRADIENT_CAP * math.tanh(
        (_breadth_v1(black, mask) - _breadth_v1(white, mask))
        / _GRADIENT_V1_SCALE
    )
    assert g_v1 == expected
    assert _GRADIENT_V1_SCALE == 7.5


def test_attack_gradient_v1_counts_breadth_not_best():
    # Two open threes beat one, as in v7's test.
    one = _board_with([(7, 7), (8, 7), (9, 7)], [(3, 3)])
    two = _board_with([(7, 7), (8, 7), (9, 7), (11, 11), (12, 11), (13, 11)],
                      [(3, 3)])
    assert scoring._winning_moves(two, game.BLACK) == []
    assert _attack_gradient_v1(two, game.BLACK) > _attack_gradient_v1(one, game.BLACK)


def test_v14_registered_and_dispatches_to_gradient_v1_mcts():
    v14 = scoring.get_player(14, seed=0)
    assert v14.name == "scoring-v14"
    assert v14.mcts is True
    assert v14._mcts is not None
    assert v14._mcts.sims == 100
    assert v14._mcts.leaf_mode == "gradient_v1"
    # v7 is unchanged (v0-kernel gradient leaf).
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"


def test_v16_registered_and_dispatches_to_rollout_mcts():
    v16 = scoring.get_player(16, seed=0)
    assert v16.name == "scoring-v16"
    assert v16.mcts is True
    assert v16._mcts is not None
    assert v16._mcts.sims == 100
    assert v16._mcts.leaf_mode == "rollout2_v1"


# --- v20 (batch 7, H3): gap-three demotion in the v1-kernel leaf -------


def test_gradient_v1_gap_leaf_matches_terminal_leaf_on_terminal_positions():
    # The v20 leaf is IDENTICAL to the terminal leaf on every
    # non-quiet position, by construction — same as v7/v14.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _gradient_v1_gap_leaf_value(board, game.BLACK) == 1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _gradient_v1_gap_leaf_value(board, game.BLACK) == -1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _gradient_v1_gap_leaf_value(board, game.BLACK) == -0.25


def test_attack_gradient_v1_gap_zero_on_empty_board():
    board = game.empty_board()
    assert _attack_gradient_v1_gap(board, game.BLACK) == 0.0
    assert _attack_gradient_v1_gap(board, game.WHITE) == 0.0
    assert _gradient_v1_gap_leaf_value(board, game.BLACK) == 0.0


def test_attack_gradient_v1_gap_antisymmetric_and_bounded():
    # Same mid-game board as the v7/v14 tests: antisymmetric under
    # colour swap, strictly inside (-0.2, 0.2) — the cap is
    # unchanged from v7/v14.
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    gb = _attack_gradient_v1_gap(board, game.BLACK)
    gw = _attack_gradient_v1_gap(board, game.WHITE)
    assert gb == -gw
    assert -0.2 < gb < 0.2
    assert -0.2 < gw < 0.2


def _battery_line_ply10_board():
    """The shared v14/v17 28-ply Black loss to v0 (iter-20 §2.0),
    replayed to ply 10 (Black to move). Stones as (x, y)."""
    black = [(7, 7), (7, 8), (9, 8), (8, 9), (7, 10)]
    white = [(8, 7), (7, 6), (8, 8), (6, 7), (10, 7)]
    return _board_with(black, white)


def test_breadth_v1_gap_demotes_gap_three():
    # The mechanism position (iter-20 §2.0, ply 10): after Black's
    # cluster four (9,7) [row, col] = (x7, y9), Black holds 6
    # three-tier moves — 4 gap-three makers (v0 attack 40) and 2
    # true open-three makers (v0 attack >= 1000). v14's breadth
    # counts all 6 at full weight (18.25); v20's counts the 4 gap
    # makers at 0.25 (15.25). After the distant four-maker (12,5)
    # [row, col] = (x5, y12) there are no three-tier moves, so the
    # breadth is unchanged (15.5) — the demotion only bites where
    # gap-three makers exist.
    from gomoku_ml import kernel_numpy, kernel_v1

    board = _battery_line_ply10_board()
    assert scoring._winning_moves(board, game.BLACK) == []
    assert scoring._winning_moves(board, game.WHITE) == []

    def _breadths(after_board):
        b1, _w1 = kernel_v1.attack_v1_both(after_board)
        b0, _w0 = kernel_numpy.attack_both(after_board)
        mask = after_board == game.EMPTY
        return (
            _breadth_v1(b1, mask),
            _breadth_v1_gap(b1, b0, mask),
            b1,
            b0,
        )

    b_cluster = game.apply_move(board.copy(), 9 * 15 + 7, game.BLACK)
    b14, b20, atk1, atk0 = _breadths(b_cluster)
    assert b14 == 18.25
    assert b20 == 15.25
    # The filter itself: (9,4) [row, col] is a gap-three maker,
    # (9,6) [row, col] a true open-three maker.
    assert 4000 <= atk1[9, 4] < 20000
    assert atk0[9, 4] < 1000
    assert 4000 <= atk1[9, 6] < 20000
    assert atk0[9, 6] >= 1000

    b_distant = game.apply_move(board.copy(), 12 * 15 + 5, game.BLACK)
    b14d, b20d, _a1, _a0 = _breadths(b_distant)
    assert b14d == 15.5
    assert b20d == 15.5


def test_attack_gradient_v1_gap_scale_is_5_5():
    # The value is the capped tanh of the demoted breadth
    # difference at the pre-registered Step-0 scale 5.5
    # (iter-20 §1.1: 1.8 x std 3.02 = 5.44, rounded to 0.5).
    from gomoku_ml import kernel_numpy, kernel_v1
    from gomoku_ml.mcts_heuristic import (
        _GRADIENT_CAP,
        _GRADIENT_V1_GAP_SCALE,
    )

    assert _GRADIENT_V1_GAP_SCALE == 5.5
    board = _battery_line_ply10_board()
    board = game.apply_move(board, 9 * 15 + 7, game.BLACK)
    b1, w1 = kernel_v1.attack_v1_both(board)
    b0, w0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    expected = _GRADIENT_CAP * math.tanh(
        (_breadth_v1_gap(w1, w0, mask) - _breadth_v1_gap(b1, b0, mask))
        / _GRADIENT_V1_GAP_SCALE
    )
    assert _attack_gradient_v1_gap(board, game.WHITE) == expected


def test_v20_registered_and_dispatches_to_gap_mcts():
    v20 = scoring.get_player(20, seed=0)
    assert v20.name == "scoring-v20"
    assert v20.mcts is True
    assert v20._mcts is not None
    assert v20._mcts.sims == 100
    assert v20._mcts.leaf_mode == "gradient_v1_gap"
    # v14 is unchanged (v1-kernel gradient leaf, scale 7.5).
    v14 = scoring.get_player(14, seed=0)
    assert v14._mcts.leaf_mode == "gradient_v1"
    # v7 is unchanged (v0-kernel gradient leaf).
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"
    # v7 is unchanged (gradient leaf).
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"


def test_gradient_v1_gap_black_leaf_matches_terminal_leaf_on_terminal_positions():
    # The v22 leaf is IDENTICAL to the terminal leaf on every
    # non-quiet position, by construction — same as v7/v14/v20.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _gradient_v1_gap_black_leaf_value(board, game.BLACK) == 1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _gradient_v1_gap_black_leaf_value(board, game.BLACK) == -1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _gradient_v1_gap_black_leaf_value(board, game.BLACK) == -0.25


def test_attack_gradient_v1_gap_black_zero_on_empty_board():
    board = game.empty_board()
    assert _attack_gradient_v1_gap_black(board, game.BLACK) == 0.0
    assert _attack_gradient_v1_gap_black(board, game.WHITE) == 0.0
    assert _gradient_v1_gap_black_leaf_value(board, game.BLACK) == 0.0


def test_attack_gradient_v1_gap_black_antisymmetric_player_swap_and_bounded():
    # Antisymmetric under PLAYER swap (Black to move vs White to
    # move on the same board) — the property MCTS needs — and
    # strictly inside (-0.2, 0.2). Deliberately NOT antisymmetric
    # under colour swap: Black's breadth is demoted, White's is
    # not (iter-22 §1).
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    gb = _attack_gradient_v1_gap_black(board, game.BLACK)
    gw = _attack_gradient_v1_gap_black(board, game.WHITE)
    assert gb == -gw
    assert -0.2 < gb < 0.2
    assert -0.2 < gw < 0.2


def test_attack_gradient_v1_gap_black_demotes_only_black():
    # The mechanism position (iter-20 §2.0, ply 10): after Black's
    # cluster four (9,7) [row, col] = (x7, y9), Black holds 6
    # three-tier moves — 4 gap-three makers and 2 true open-three
    # makers. v22's gradient for White to move must use Black's
    # DEMOTED breadth (15.25, as in v20) and White's UNDEMOTED
    # breadth — while v20's uses the demoted breadth for both
    # sides. The two gradients therefore differ exactly by the
    # demotion of White's gap-three makers (at v20's scale 5.5
    # vs v22's scale 7.0 the tanh inputs differ too, so the test
    # compares the breadth inputs, not the final values).
    from gomoku_ml import kernel_numpy, kernel_v1
    from gomoku_ml.mcts_heuristic import _GRADIENT_V1_GAP_BLACK_SCALE

    assert _GRADIENT_V1_GAP_BLACK_SCALE == 7.0
    board = _battery_line_ply10_board()
    board = game.apply_move(board, 9 * 15 + 7, game.BLACK)
    b1, w1 = kernel_v1.attack_v1_both(board)
    b0, w0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    # Black's breadth: demoted (15.25, the v20 value — the
    # cluster four leaves 4 gap-three makers).
    assert _breadth_v1_gap(b1, b0, mask) == 15.25
    # White's breadth: v22 uses the UNDEMOTED v1 breadth.
    white_full = _breadth_v1(w1, mask)
    white_demoted = _breadth_v1_gap(w1, w0, mask)
    assert white_full >= white_demoted
    # v22's gradient for White to move is the capped tanh of
    # (White full - Black demoted) at scale 7.0.
    expected = 0.2 * math.tanh(
        (white_full - 15.25) / _GRADIENT_V1_GAP_BLACK_SCALE
    )
    assert _attack_gradient_v1_gap_black(board, game.WHITE) == expected
    # And for Black to move it is the negation (player-swap
    # antisymmetry), NOT the colour-swapped value: swapping the
    # colours of the stones would change which side is demoted.
    assert _attack_gradient_v1_gap_black(board, game.BLACK) == -expected


def test_v22_breadth_diff_is_v20_diff_plus_white_gap_threes():
    # Identity: v22's breadth difference equals v20's PLUS 0.75
    # per White gap-three maker (White's breadth is undemoted in
    # v22, demoted in v20; Black's is demoted in both). Checked on
    # the ply-10 board after the cluster four (White to move) and
    # on a random mid-game board, for both players to move. The
    # gradient values differ by the tanh scale too (7.0 vs 5.5),
    # so this test compares the breadth differences directly.
    from gomoku_ml import kernel_numpy, kernel_v1

    def _gap3(a1, a0, mask):
        s1 = a1[mask]
        s0 = a0[mask]
        three = (s1 >= 4000) & (s1 < 20000)
        return int(np.count_nonzero(three & (s0 < 1000)))

    def _diffs(board, player):
        b1, w1 = kernel_v1.attack_v1_both(board)
        b0, w0 = kernel_numpy.attack_both(board)
        mask = board == game.EMPTY
        if player == game.BLACK:
            o1, t1, o0, t0 = b1, w1, b0, w0
        else:
            o1, t1, o0, t0 = w1, b1, w0, b0
        d20 = _breadth_v1_gap(o1, o0, mask) - _breadth_v1_gap(t1, t0, mask)
        bb = _breadth_v1_gap(b1, b0, mask)
        wb = _breadth_v1(w1, mask)
        d22 = (bb - wb) if player == game.BLACK else (wb - bb)
        return d20, d22, _gap3(w1, w0, mask)

    board = _battery_line_ply10_board()
    board = game.apply_move(board, 9 * 15 + 7, game.BLACK)
    for player in (game.BLACK, game.WHITE):
        d20, d22, gap3_white = _diffs(board, player)
        # White is "theirs" for Black to move (sign -) and "ours"
        # for White to move (sign +).
        sign = -1 if player == game.BLACK else 1
        assert d22 == pytest.approx(
            d20 + sign * 0.75 * gap3_white
        )

    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    for player in (game.BLACK, game.WHITE):
        d20, d22, gap3_white = _diffs(board, player)
        sign = -1 if player == game.BLACK else 1
        assert d22 == pytest.approx(
            d20 + sign * 0.75 * gap3_white
        )


def test_v22_registered_and_dispatches_to_gap_black_mcts():
    v22 = scoring.get_player(22, seed=0)
    assert v22.name == "scoring-v22"
    assert v22.mcts is True
    assert v22._mcts is not None
    assert v22._mcts.sims == 100
    assert v22._mcts.leaf_mode == "gradient_v1_gap_black"
    # v20 is unchanged (both-sides gap-demoted leaf, scale 5.5).
    v20 = scoring.get_player(20, seed=0)
    assert v20._mcts.leaf_mode == "gradient_v1_gap"
    # v14 is unchanged (v1-kernel gradient leaf, scale 7.5).
    v14 = scoring.get_player(14, seed=0)
    assert v14._mcts.leaf_mode == "gradient_v1"
    # v7 is unchanged (v0-kernel gradient leaf).
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"


def test_gradient_v1_gap_ours_leaf_matches_terminal_leaf_on_terminal_positions():
    # The v28 leaf is IDENTICAL to the terminal leaf on every
    # non-quiet position, by construction — same as v7/v14/v20/v22.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _gradient_v1_gap_ours_leaf_value(board, game.BLACK) == 1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _gradient_v1_gap_ours_leaf_value(board, game.BLACK) == -1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _gradient_v1_gap_ours_leaf_value(board, game.BLACK) == -0.25


def test_attack_gradient_v1_gap_ours_zero_on_empty_board():
    board = game.empty_board()
    assert _attack_gradient_v1_gap_ours(board, game.BLACK) == 0.0
    assert _attack_gradient_v1_gap_ours(board, game.WHITE) == 0.0
    assert _gradient_v1_gap_ours_leaf_value(board, game.BLACK) == 0.0


def test_attack_gradient_v1_gap_ours_breaks_antisymmetry_in_the_pre_registered_direction():
    # v28 DELIBERATELY breaks player-swap antisymmetry (iter-28
    # §1): -value(board, BLACK) >= value(board, WHITE), with
    # strict inequality whenever Black holds gap-three makers —
    # White-to-move leaves are valued LOWER than the mirror of
    # Black-to-move leaves, because Black's full undemoted breadth
    # counts against White in the theirs term. Both values stay
    # strictly inside (-0.2, 0.2).
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    gb = _attack_gradient_v1_gap_ours(board, game.BLACK)
    gw = _attack_gradient_v1_gap_ours(board, game.WHITE)
    assert -gb >= gw
    assert -0.2 < gb < 0.2
    assert -0.2 < gw < 0.2
    # Strict break on the ply-10 board, where Black holds 4
    # gap-three makers after the cluster four (iter-20 §2.0).
    board = _battery_line_ply10_board()
    board = game.apply_move(board, 9 * 15 + 7, game.BLACK)
    gb = _attack_gradient_v1_gap_ours(board, game.BLACK)
    gw = _attack_gradient_v1_gap_ours(board, game.WHITE)
    assert -gb > gw
    assert gb != -gw


def test_attack_gradient_v1_gap_ours_matches_v22_at_black_to_move_and_v14_at_white():
    # The mechanism position (iter-20 §2.0, ply 10): after Black's
    # cluster four (9,7) [row, col] = (x7, y9), Black holds 6
    # three-tier moves — 4 gap-three makers and 2 true open-three
    # makers. v28's gradient for Black to move must use Black's
    # DEMOTED breadth (15.25, as in v20/v22) and White's UNDEMOTED
    # breadth (v22's rule); for White to move it must use BOTH
    # breadths UNDEMOTED (v14's rule) — Black's theirs-breadth
    # counted FULLY, so White fears Black's gap threes. The test
    # compares the capped tanh of the breadth inputs at v28's
    # scale, not the final values of the other versions (whose
    # tanh scales differ).
    from gomoku_ml import kernel_numpy, kernel_v1
    from gomoku_ml.mcts_heuristic import _GRADIENT_V1_GAP_OURS_SCALE

    assert _GRADIENT_V1_GAP_OURS_SCALE == 8.0
    board = _battery_line_ply10_board()
    board = game.apply_move(board, 9 * 15 + 7, game.BLACK)
    b1, w1 = kernel_v1.attack_v1_both(board)
    b0, w0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    # Black's breadth: demoted 15.25 (the v20/v22 value), full
    # 18.25 (15.25 + 0.75 x 4 gap-three makers).
    assert _breadth_v1_gap(b1, b0, mask) == 15.25
    black_full = _breadth_v1(b1, mask)
    assert black_full == pytest.approx(18.25)
    white_full = _breadth_v1(w1, mask)
    # Black to move: v22's rule (Black demoted - White full).
    expected_black = 0.2 * math.tanh(
        (15.25 - white_full) / _GRADIENT_V1_GAP_OURS_SCALE
    )
    assert _attack_gradient_v1_gap_ours(board, game.BLACK) == expected_black
    # White to move: v14's rule (White full - Black FULL).
    expected_white = 0.2 * math.tanh(
        (white_full - black_full) / _GRADIENT_V1_GAP_OURS_SCALE
    )
    assert _attack_gradient_v1_gap_ours(board, game.WHITE) == expected_white
    # Cross-version identity via the inverted tanh (the scales
    # differ, so the breadth differences are compared, not the
    # gradient values): v28's Black-to-move breadth difference
    # equals v22's, and v28's White-to-move breadth difference
    # equals v14's.
    from gomoku_ml.mcts_heuristic import (
        _GRADIENT_V1_GAP_BLACK_SCALE,
        _GRADIENT_V1_SCALE,
    )

    v28_black_diff = (
        math.atanh(_attack_gradient_v1_gap_ours(board, game.BLACK) / 0.2)
        * _GRADIENT_V1_GAP_OURS_SCALE
    )
    v22_black_diff = (
        math.atanh(_attack_gradient_v1_gap_black(board, game.BLACK) / 0.2)
        * _GRADIENT_V1_GAP_BLACK_SCALE
    )
    assert v28_black_diff == pytest.approx(v22_black_diff)
    v28_white_diff = (
        math.atanh(_attack_gradient_v1_gap_ours(board, game.WHITE) / 0.2)
        * _GRADIENT_V1_GAP_OURS_SCALE
    )
    v14_white_diff = (
        math.atanh(_attack_gradient_v1(board, game.WHITE) / 0.2)
        * _GRADIENT_V1_SCALE
    )
    assert v28_white_diff == pytest.approx(v14_white_diff)


def test_v28_breadth_diff_is_v22_diff_minus_black_gap_threes_at_white_to_move():
    # Identity: v28's breadth difference equals v22's at
    # Black-to-move leaves, and equals v22's MINUS 0.75 per Black
    # gap-three maker at White-to-move leaves (Black's
    # theirs-breadth is undemoted in v28, demoted in v22).
    # Checked on the ply-10 board after the cluster four and on a
    # random mid-game board, for both players to move. The
    # gradient values differ by the tanh scale too, so this test
    # compares the breadth differences directly.
    from gomoku_ml import kernel_numpy, kernel_v1

    def _gap3(a1, a0, mask):
        s1 = a1[mask]
        s0 = a0[mask]
        three = (s1 >= 4000) & (s1 < 20000)
        return int(np.count_nonzero(three & (s0 < 1000)))

    def _diffs(board, player):
        b1, w1 = kernel_v1.attack_v1_both(board)
        b0, w0 = kernel_numpy.attack_both(board)
        mask = board == game.EMPTY
        bb_gap = _breadth_v1_gap(b1, b0, mask)
        bb_full = _breadth_v1(b1, mask)
        wb = _breadth_v1(w1, mask)
        d22 = (bb_gap - wb) if player == game.BLACK else (wb - bb_gap)
        d28 = (bb_gap - wb) if player == game.BLACK else (wb - bb_full)
        return d22, d28, _gap3(b1, b0, mask)

    board = _battery_line_ply10_board()
    board = game.apply_move(board, 9 * 15 + 7, game.BLACK)
    for player in (game.BLACK, game.WHITE):
        d22, d28, gap3_black = _diffs(board, player)
        if player == game.BLACK:
            assert d28 == pytest.approx(d22)
        else:
            assert d28 == pytest.approx(d22 - 0.75 * gap3_black)

    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    for player in (game.BLACK, game.WHITE):
        d22, d28, gap3_black = _diffs(board, player)
        if player == game.BLACK:
            assert d28 == pytest.approx(d22)
        else:
            assert d28 == pytest.approx(d22 - 0.75 * gap3_black)


def test_v28_registered_and_dispatches_to_gap_ours_mcts():
    v28 = scoring.get_player(28, seed=0)
    assert v28.name == "scoring-v28"
    assert v28.mcts is True
    assert v28._mcts is not None
    assert v28._mcts.sims == 100
    assert v28._mcts.leaf_mode == "gradient_v1_gap_ours"
    # v22 is unchanged (Black-only gap-demoted leaf, scale 7.0).
    v22 = scoring.get_player(22, seed=0)
    assert v22._mcts.leaf_mode == "gradient_v1_gap_black"
    # v20 is unchanged (both-sides gap-demoted leaf, scale 5.5).
    v20 = scoring.get_player(20, seed=0)
    assert v20._mcts.leaf_mode == "gradient_v1_gap"
    # v14 is unchanged (v1-kernel gradient leaf, scale 7.5).
    v14 = scoring.get_player(14, seed=0)
    assert v14._mcts.leaf_mode == "gradient_v1"
    # v7 is unchanged (v0-kernel gradient leaf).
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"


def test_v16_as_black_matches_v7():
    # v16-as-Black IS v7 exactly (same leaf, same priors, same
    # search) — the Black games are a built-in control. Same seed,
    # same positions, same moves.
    rng = np.random.default_rng(5)
    for trial in range(3):
        board = game.empty_board()
        cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
        rng.shuffle(cells)
        for i, (x, y) in enumerate(cells[:20]):
            board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
        v7 = scoring.get_player(7, seed=0)
        v16 = scoring.get_player(16, seed=0)
        assert v16.choose_move(board, game.BLACK) == v7.choose_move(
            board, game.BLACK
        ), f"trial {trial}: v16-as-Black differs from v7"


def test_v16_leaf_value_matches_v7_when_root_is_black():
    # Leaf-level version of the Black-control claim: with the root
    # player Black, the v16 leaf equals v7's gradient leaf on every
    # sampled board (terminal and quiet alike).
    rng = np.random.default_rng(6)
    mcts = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1")
    mcts._root_player = game.BLACK
    for _ in range(5):
        board = game.empty_board()
        cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
        rng.shuffle(cells)
        for i, (x, y) in enumerate(cells[:20]):
            board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
        for player in (game.BLACK, game.WHITE):
            assert mcts._leaf_value(board, player) == _gradient_leaf_value(
                board, player
            )


def test_v16_choose_move_deterministic_for_same_seed():
    rng = np.random.default_rng(1)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:20]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    a = scoring.get_player(16, seed=0)
    b = scoring.get_player(16, seed=0)
    assert a.choose_move(board, game.WHITE) == b.choose_move(board, game.WHITE)


def test_v16_choose_move_does_not_mutate_board():
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = HeuristicMCTS(sims=20, leaf_mode="rollout2_v1",
                         rng=np.random.default_rng(0))
    mcts.choose_move(board, game.WHITE)
    assert np.array_equal(board, before)


def test_v14_choose_move_does_not_mutate_board():
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = HeuristicMCTS(sims=20, leaf_mode="gradient_v1",
                         rng=np.random.default_rng(0))
    mcts.choose_move(board, game.BLACK)
    assert np.array_equal(board, before)


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


# --- v18 (batch 7, H1): rollout leaf for both colours -----------------------

def test_v18_registered_and_dispatches_to_rollout_mcts():
    v18 = scoring.get_player(18, seed=0)
    assert v18.name == "scoring-v18"
    assert v18.mcts is True
    assert v18._mcts is not None
    assert v18._mcts.sims == 100
    assert v18._mcts.leaf_mode == "rollout2_v1_both"
    # v16 and v7 are unchanged.
    v16 = scoring.get_player(16, seed=0)
    assert v16._mcts.leaf_mode == "rollout2_v1"
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"


def test_v18_as_white_matches_v16():
    # v18-as-White IS v16 exactly (same leaf, same priors, same
    # search) — the White games are a built-in control. Same seed,
    # same positions, same moves.
    rng = np.random.default_rng(5)
    for trial in range(3):
        board = game.empty_board()
        cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
        rng.shuffle(cells)
        for i, (x, y) in enumerate(cells[:20]):
            board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
        v16 = scoring.get_player(16, seed=0)
        v18 = scoring.get_player(18, seed=0)
        assert v18.choose_move(board, game.WHITE) == v16.choose_move(
            board, game.WHITE
        ), f"trial {trial}: v18-as-White differs from v16"


def test_v18_leaf_value_is_rollout_for_both_colours():
    # Leaf-level version of the dispatch claim: with the root
    # player Black OR White (or None — leaf called outside a
    # search), the v18 leaf equals the rollout leaf on every
    # sampled board (terminal and quiet alike). No colour gate.
    rng = np.random.default_rng(6)
    mcts = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both")
    for root in (game.BLACK, game.WHITE, None):
        mcts._root_player = root
        for _ in range(5):
            board = game.empty_board()
            cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
            rng.shuffle(cells)
            for i, (x, y) in enumerate(cells[:20]):
                board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
            for player in (game.BLACK, game.WHITE):
                assert mcts._leaf_value(board, player) == _rollout_leaf_value(
                    board, player
                )


def test_v18_as_black_uses_rollout_leaf_not_gradient():
    # The v18 Black leaf is the rollout leaf, not v7's gradient:
    # on sampled boards the two leaf functions disagree on most
    # quiet positions (8/10 with this seed — verified when the
    # test was written), so a v18 instance with a Black root must
    # differ from the gradient leaf somewhere in the sample.
    rng = np.random.default_rng(7)
    mcts = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both")
    mcts._root_player = game.BLACK
    diffs = 0
    for _ in range(5):
        board = game.empty_board()
        cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
        rng.shuffle(cells)
        for i, (x, y) in enumerate(cells[:20]):
            board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
        for player in (game.BLACK, game.WHITE):
            if mcts._leaf_value(board, player) != _gradient_leaf_value(
                board, player
            ):
                diffs += 1
    assert diffs > 0


def test_rollout_both_leaf_mode_rejects_unknown_mode():
    # The conditional dispatch must still reject unknown modes
    # (and the error message lists both rollout modes).
    with pytest.raises(ValueError, match="rollout2_v1_both"):
        HeuristicMCTS(sims=10, leaf_mode="rollout3_v1")


# --- v23 (batch 8, H3): q-aware visit-tie break ---------------------------
# Pre-registered rule (iter-23.md §1.1): on an exact top-visit tie
# at the root, break it by correctly-signed q (our-q = -child.q);
# if q also ties exactly, fall back to v18's RNG/first choice on
# the q-tied subset. A unique visit leader is untouched.


def _root_with_children(specs):
    """Fabricate a root node: ``specs`` is a list of
    (move, visit_count, value_sum) in dict-insertion order."""
    root = _Node(1.0)
    root.children = {}
    for move, visits, value_sum in specs:
        child = _Node(0.1)
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


# --- v12 "tie-prior-forcing" root decision (batch 5, H1) ---------------
# iter-12.md §1.1: most-visited; ties broken by forcing-ness first
# (moves creating a four), then the v1 prior. Targets iter-9 §2.6's
# failure mode, where the prior preferred occupying a creator
# (~204k) over a forcing counter-threat (~20k).


def _forcing_board():
    """White has three in a row at (7,7)-(7,9): both (7,6) and
    (7,10) create a four (forcing); (0,0) and (1,1) do not."""
    return _board_with([(3, 3)], [(7, 7), (7, 8), (7, 9)])


def test_root_decision_tie_prior_forcing_prefers_forcing_over_prior():
    # The §2.6 shape: the forcing move has the LOWER prior, the
    # quiet move the higher — forcing must win the tie anyway.
    # (7,10) flat 157 is forcing; (0,0) flat 0 is not.
    root = _synthetic_root([
        (157, 5, 0.0, 0.04),   # forcing, lower prior
        (0, 5, 0.0, 0.09),     # not forcing, higher prior
    ])
    board = _forcing_board()
    assert scoring._is_forcing(board, 157, game.WHITE)
    assert not scoring._is_forcing(board, 0, game.WHITE)
    mcts = HeuristicMCTS(root_decision="tie-prior-forcing")
    assert mcts._choose_root_child(root, board, game.WHITE) == 157
    # Control: plain tie-prior picks the higher-prior quiet move.
    assert (
        HeuristicMCTS(root_decision="tie-prior")._choose_root_child(
            root, board, game.WHITE
        )
        == 0
    )


def test_root_decision_tie_prior_forcing_prior_breaks_forcing_ties():
    # Two forcing moves tied: the prior decides between them.
    # (7,6) flat 97 and (7,10) flat 157 are both forcing; (0,0)
    # has the highest prior but is not forcing, so it loses.
    root = _synthetic_root([
        (97, 5, 0.0, 0.03),   # forcing, lower prior
        (157, 5, 0.0, 0.05),   # forcing, higher prior
        (0, 5, 0.0, 0.09),     # not forcing, highest prior
    ])
    board = _forcing_board()
    mcts = HeuristicMCTS(root_decision="tie-prior-forcing")
    assert mcts._choose_root_child(root, board, game.WHITE) == 157


def test_root_decision_tie_prior_forcing_falls_back_to_prior():
    # No tied child is forcing: the rule is plain tie-prior.
    root = _synthetic_root([
        (0, 5, 0.0, 0.04),
        (1, 5, 0.0, 0.09),     # higher prior wins
    ])
    board = _forcing_board()
    assert not scoring._is_forcing(board, 0, game.WHITE)
    assert not scoring._is_forcing(board, 1, game.WHITE)
    mcts = HeuristicMCTS(root_decision="tie-prior-forcing")
    assert mcts._choose_root_child(root, board, game.WHITE) == 1


def test_root_decision_tie_prior_forcing_without_board_is_tie_prior():
    # Synthetic-root callers pass no board: the forcing filter is
    # skipped and the rule degrades to plain tie-prior (the
    # higher-prior child wins even if it would not be forcing).
    root = _synthetic_root([
        (157, 5, 0.0, 0.04),
        (0, 5, 0.0, 0.09),
    ])
    mcts = HeuristicMCTS(root_decision="tie-prior-forcing")
    assert mcts._choose_root_child(root) == 0


def test_root_decision_tie_prior_forcing_ignores_non_tied_forcing():
    # Forcing only breaks TIES: a forcing child with fewer visits
    # must not beat the most-visited child.
    root = _synthetic_root([
        (157, 4, 0.0, 0.04),   # forcing, fewer visits
        (0, 5, 0.0, 0.09),     # not forcing, most visited
    ])
    board = _forcing_board()
    mcts = HeuristicMCTS(root_decision="tie-prior-forcing")
    assert mcts._choose_root_child(root, board, game.WHITE) == 0


def test_v12_registration_is_tie_prior_forcing_gradient():
    v12 = scoring.get_player(12, seed=0)
    assert v12._mcts.root_decision == "tie-prior-forcing"
    assert v12._mcts.leaf_mode == "gradient"
    assert v12._mcts.sims == 100
    # v7/v9 untouched by the v12 registration.
    assert scoring.get_player(7, seed=0)._mcts.root_decision == "visits"
    assert scoring.get_player(9, seed=0)._mcts.root_decision == "tie-prior"


def test_v12_choose_move_deterministic_for_same_seed():
    board = _board_with(
        [(7, 7), (8, 8), (9, 9)], [(7, 8), (8, 7), (6, 6)]
    )
    a = scoring.get_player(12, seed=42)
    b = scoring.get_player(12, seed=42)
    assert a.choose_move(board, game.BLACK) == b.choose_move(
        board, game.BLACK
    )
def test_v23_registered_and_dispatches_to_rollout_mcts_with_tie_break():
    v23 = scoring.get_player(23, seed=0)
    assert v23.name == "scoring-v23"
    assert v23.mcts is True
    assert v23._mcts is not None
    assert v23._mcts.sims == 100
    assert v23._mcts.leaf_mode == "rollout2_v1_both"
    assert v23._mcts.tie_break_q is True
    # v18 and v7 are unchanged (tie-break off by default).
    v18 = scoring.get_player(18, seed=0)
    assert v18._mcts.leaf_mode == "rollout2_v1_both"
    assert v18._mcts.tie_break_q is False
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.tie_break_q is False


def test_v23_tie_break_picks_highest_our_q():
    # Three children tie at 3 visits. child.q is the OPPONENT's
    # perspective, so our-q = -child.q: move 101 (value_sum -3.0,
    # q -1.0, our-q +1.0) beats 100 (our-q +0.583) and 102
    # (our-q +0.667) — even though 101 is not first in dict order.
    root = _root_with_children([(100, 3, -1.75), (101, 3, -3.0), (102, 3, -2.0)])
    mcts = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True)
    assert mcts._pick_root_move(root) == 101
    # Without the flag (v18 behaviour, no RNG) the same root
    # returns the first tied child — the flag is the only change.
    mcts18 = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both")
    assert mcts18._pick_root_move(root) == 100


def test_v23_q_tie_falls_back_to_rng_subset():
    # Moves 100 and 102 tie at 3 visits AND at our-q +0.583
    # (value_sum -1.75 each); move 101 ties on visits but holds
    # worse our-q (+0.333). The fallback must choose between 100
    # and 102 only — never 101.
    root = _root_with_children([(100, 3, -1.75), (101, 3, -1.0), (102, 3, -1.75)])
    mcts = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        rng=np.random.default_rng(0),
    )
    for _ in range(20):
        assert mcts._pick_root_move(root) in (100, 102)
    # With no RNG the fallback is the first q-tied child in dict
    # order (100), not the first visit-tied child overall — here
    # they coincide, so rebuild with the worse-q move first.
    root2 = _root_with_children([(101, 3, -1.0), (100, 3, -1.75), (102, 3, -1.75)])
    mcts_no_rng = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True)
    assert mcts_no_rng._pick_root_move(root2) == 100


def test_v23_unique_visit_leader_untouched():
    # The visit leader (4 visits) holds the WORST our-q — v23
    # must still play it: the rule fires only on exact ties.
    root = _root_with_children([(100, 4, -0.5), (101, 3, -3.0), (102, 3, -2.0)])
    mcts = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True)
    assert mcts._pick_root_move(root) == 100
    # A 1-visit gap is NOT a tie either.
    root2 = _root_with_children([(100, 4, -0.5), (101, 3, -3.0)])
    assert mcts._pick_root_move(root2) == 100


def test_v23_choose_move_deterministic_for_same_seed():
    # Same seed, same board, same move — the tie-break consumes
    # RNG only on exact q ties, so determinism holds.
    board = _board_with(
        [(7, 7), (6, 7), (8, 7), (7, 6)], [(7, 8), (6, 8), (8, 8), (7, 5)]
    )
    v23a = scoring.get_player(23, seed=0)
    v23b = scoring.get_player(23, seed=0)
    assert v23a.choose_move(board, game.BLACK) == v23b.choose_move(board, game.BLACK)


# --- v27 (batch 9, H1): q-gap threshold tie-break --------------------------
# Pre-registered rule (iter-27.md §1.1): v23's tie-break fires ONLY
# when the q gap — the highest our-q minus the second-highest
# DISTINCT our-q among the visit-tied children — strictly exceeds
# the threshold. Below the threshold the tie-break does not fire
# at all (v18's RNG/first over ALL tied children). tau=0.0
# reproduces v23 exactly; tau -> infinity reproduces v18.


def test_v27_threshold_zero_reproduces_v23():
    # Same root as test_v23_tie_break_picks_highest_our_q: with
    # tau=0.0 the threshold must not change v23's choice.
    root = _root_with_children([(100, 3, -1.75), (101, 3, -3.0), (102, 3, -2.0)])
    mcts = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.0,
    )
    assert mcts._pick_root_move(root) == 101
    # All-q-equal root: gap 0, v23's rule-4 subset is all tied
    # children, so tau=0.0 with no RNG returns the first — same
    # as v23.
    root2 = _root_with_children([(100, 3, -1.75), (101, 3, -1.75), (102, 3, -1.75)])
    assert mcts._pick_root_move(root2) == 100
    # Registry: v23 dispatches with threshold 0.0 (v23 unchanged).
    v23 = scoring.get_player(23, seed=0)
    assert v23._mcts.tie_break_q is True
    assert v23._mcts.tie_break_q_threshold == 0.0
    v18 = scoring.get_player(18, seed=0)
    assert v18._mcts.tie_break_q is False
    assert v18._mcts.tie_break_q_threshold == 0.0


def test_v27_threshold_blocks_small_gap_misfire():
    # The documented fl2-g6/g8 misfire (iter-23 §2.2): a 2-way
    # tie at 8 visits, q gap 0.016 (loser our-q +0.076 over saver
    # +0.060). v23 picks the loser (101); v27 at tau=0.05 must
    # NOT fire — with no RNG it returns the first in dict order
    # (100), v18's behaviour. value_sum = -our_q * visits.
    root = _root_with_children([(100, 8, -0.48), (101, 8, -0.608)])
    mcts27 = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05,
    )
    assert mcts27._pick_root_move(root) == 100
    mcts27b = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.10,
    )
    assert mcts27b._pick_root_move(root) == 100
    # Control: v23 (tau=0.0) fires on the same root and picks
    # the higher-q (losing) move.
    mcts23 = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True)
    assert mcts23._pick_root_move(root) == 101


def test_v27_threshold_fires_on_large_gap_signal():
    # The v1bl-g2 tactical fix (iter-23 §1.2): a 2-way tie at 8
    # visits, q gap 0.130 (expected (7,5) our-q -0.594 over loser
    # (7,9) -0.724). Both sweep thresholds must fire and pick the
    # higher-q move (101). value_sum = -our_q * visits.
    root = _root_with_children([(100, 8, 5.792), (101, 8, 4.752)])
    for tau in (0.05, 0.10):
        mcts = HeuristicMCTS(
            sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
            tie_break_q_threshold=tau,
        )
        assert mcts._pick_root_move(root) == 101
    # A threshold ABOVE the gap (0.20) must not fire: first in
    # dict order (100), v18's behaviour. (This is why 0.20 is not
    # swept — it would break the v1bl-g2 fix.)
    mcts20 = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.20,
    )
    assert mcts20._pick_root_move(root) == 100


def test_v27_threshold_boundary_is_strict():
    # Dyadic q values so the gap is exact: our-q 0.625 vs 0.5,
    # gap exactly 0.125. The rule is gap > tau (strict): at
    # tau=0.125 it must NOT fire; at tau=0.10 it must fire.
    root = _root_with_children([(100, 8, -4.0), (101, 8, -5.0)])
    mcts_eq = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.125,
    )
    assert mcts_eq._pick_root_move(root) == 100
    mcts_lt = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.10,
    )
    assert mcts_lt._pick_root_move(root) == 101


def test_v27_below_threshold_rng_uses_all_tied_children():
    # Three children tie at 3 visits with our-q 0.60 / 0.59 /
    # 0.583: the gap between the top two DISTINCT values is 0.01,
    # below tau=0.05, so the tie-break must not fire and the RNG
    # must choose among ALL THREE — including 101, the worst-q
    # child v23's rule would exclude. 50 draws: P(never drawing
    # 101) = (2/3)^50 ~ 1e-9 if the choice is uniform over all.
    root = _root_with_children([(100, 3, -1.8), (101, 3, -1.77), (102, 3, -1.749)])
    mcts = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05, rng=np.random.default_rng(0),
    )
    picks = {mcts._pick_root_move(root) for _ in range(50)}
    assert picks == {100, 101, 102}
    # Control: v23 (tau=0.0) fires on the same root (gap 0.01 >
    # 0) and always picks the top-q child 100.
    mcts23 = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        rng=np.random.default_rng(0),
    )
    assert {mcts23._pick_root_move(root) for _ in range(10)} == {100}


def test_v27_top_q_group_falls_back_to_rng_subset_when_firing():
    # Gap above threshold, but two children share q_max exactly:
    # v23 rule 4 applies within the firing branch — RNG over the
    # q-tied subset only, never the worse-q child. our-q: 100 and
    # 102 at +0.583, 101 at +0.333; gap 0.25 > tau=0.05.
    root = _root_with_children([(100, 3, -1.75), (101, 3, -1.0), (102, 3, -1.75)])
    mcts = HeuristicMCTS(
        sims=10, leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05, rng=np.random.default_rng(0),
    )
    for _ in range(20):
        assert mcts._pick_root_move(root) in (100, 102)


def test_v27_registered_with_selected_threshold():
    # v27 = v23 + threshold 0.05, selected by the pre-registered
    # rule (iter-27 §1.2/§1.5: both sweep values tied at
    # 39W-0L-1D vs v0, so the tie rule picked the smaller one).
    # The temporary sweep entries (90/91) are gone.
    v27 = scoring.get_player(27, seed=0)
    assert v27.name == "scoring-v27"
    assert v27._mcts.leaf_mode == "rollout2_v1_both"
    assert v27._mcts.tie_break_q is True
    assert v27._mcts.tie_break_q_threshold == 0.05
    assert v27._mcts.sims == 100
    for temp in (90, 91):
        with pytest.raises(ValueError):
            scoring.get_player(temp, seed=0)
    # Same seed, same board, same move — the threshold consumes
    # RNG only where v18/v23 would, so determinism holds.
    board = _board_with(
        [(7, 7), (6, 7), (8, 7), (7, 6)], [(7, 8), (6, 8), (8, 8), (7, 5)]
    )
    a = scoring.get_player(27, seed=0)
    b = scoring.get_player(27, seed=0)
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)


# --- v18 (batch 7, H1): rollout leaf in both colours ---------------------
# Ported from research/batch-7 for batch 9 H3 (the v29 base). Leaf
# values verified bit-for-bit on the 401-position
# verify_rollout_port.py set (iter-29 §3).


def test_v18_registered_and_dispatches_to_rollout_both_mcts():
    v18 = scoring.get_player(18, seed=0)
    assert v18.name == "scoring-v18"
    assert v18.mcts is True
    assert v18._mcts is not None
    assert v18._mcts.sims == 100
    assert v18._mcts.leaf_mode == "rollout2_v1_both"
    assert v18._mcts.vcf_root is False




# --- v29 (batch 9, H3): VCF-at-root on the v18 base ----------------------
# See research/iterations/iter-29.md §1.2 (pre-registered design) and
# §3 (implementation + cost guard).


def _tactical_position(suite_path, pid):
    import os

    if suite_path == "hard":
        suite_path = os.path.join(
            os.path.dirname(evaluate.__file__),
            "tactical_positions_hard.json",
        )
    suite = evaluate.load_tactical_suite(suite_path)
    pos = next(p for p in suite["positions"] if p["id"] == pid)
    return evaluate.board_from_position(pos), pos["to_move"], pos


def test_v29_registered_and_vcf_root_enabled():
    v29 = scoring.get_player(29, seed=0)
    assert v29.name == "scoring-v29"
    assert v29.mcts is True
    assert v29._mcts is not None
    assert v29._mcts.sims == 100
    assert v29._mcts.leaf_mode == "rollout2_v1_both"
    assert v29._mcts.vcf_root is True
    # v18 is unchanged (no VCF check).
    v18 = scoring.get_player(18, seed=0)
    assert v18._mcts.vcf_root is False


def test_vcf_root_offensive_fires_on_vcf_positions():
    # Old-suite VCF positions: the offensive check plays the
    # forced-win start directly (no search).
    mcts = HeuristicMCTS(
        sims=100, leaf_mode="rollout2_v1_both", vcf_root=True,
        rng=np.random.default_rng(0),
    )
    for pid, expected_xy in (("vcf-1", (8, 3)), ("vcf-2", (8, 5))):
        board, to_move, _pos = _tactical_position(None, pid)
        move = mcts._vcf_root_move(board, to_move)
        assert move is not None, f"{pid}: offensive VCF did not fire"
        assert game.move_to_xy(move) == expected_xy, pid


def test_vcf_root_defensive_fixes_v18_tactical_failures():
    # v18's three hard-suite failures (iter-18 §3: q-compression
    # visit-ties — the leaf knows the answer, the visit count
    # picks wrong). The defensive VCF check plays a prevention
    # deterministically, so v29 passes all three.
    v29 = scoring.get_player(29, seed=100)
    for pid, expected in (
        ("prev-v1v0s-g26", {(8, 4), (9, 4)}),
        ("prev-v1v1s-g79", {(6, 7), (8, 7), (7, 8), (7, 9)}),
        ("prev-v1bl-g2", {(6, 5), (7, 5)}),
    ):
        board, to_move, _pos = _tactical_position("hard", pid)
        move = v29.choose_move(board, to_move)
        assert game.move_to_xy(move) in expected, (
            f"{pid}: v29 chose {game.move_to_xy(move)}, "
            f"expected one of {sorted(expected)}"
        )


def test_vcf_root_quiet_position_falls_through_to_v18():
    # Where the check does not fire, v29 IS v18 exactly (same
    # leaf, same priors, same search). Low sims to keep the test
    # fast; the comparison is at the MCTS level. Sparse boards
    # (8 stones over the full board): dense random boards are
    # creator-rich and the check fires on them, so they are not
    # fall-through positions.
    rng = np.random.default_rng(11)
    checked = 0
    for _trial in range(8):
        board = game.empty_board()
        cells = [(x, y) for y in range(15) for x in range(15)]
        rng.shuffle(cells)
        for i, (x, y) in enumerate(cells[:8]):
            board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
        v29 = HeuristicMCTS(
            sims=20, leaf_mode="rollout2_v1_both", vcf_root=True,
            rng=np.random.default_rng(0),
        )
        if v29._vcf_root_move(board, game.WHITE) is not None:
            continue  # check fired — not a fall-through position
        v18 = HeuristicMCTS(
            sims=20, leaf_mode="rollout2_v1_both", vcf_root=False,
            rng=np.random.default_rng(0),
        )
        assert v29.choose_move(board, game.WHITE) == v18.choose_move(
            board, game.WHITE
        )
        checked += 1
    assert checked >= 2, "too few fall-through positions sampled"


def test_vcf_root_does_not_mutate_board():
    board, to_move, _pos = _tactical_position(None, "vcf-1")
    before = board.copy()
    mcts = HeuristicMCTS(
        sims=20, leaf_mode="rollout2_v1_both", vcf_root=True,
        rng=np.random.default_rng(0),
    )
    mcts.choose_move(board, to_move)
    assert np.array_equal(board, before)


def test_v29_choose_move_deterministic_for_same_seed():
    board, to_move, _pos = _tactical_position(None, "vcf-1")
    a = scoring.get_player(29, seed=42)
    b = scoring.get_player(29, seed=42)
    assert a.choose_move(board, to_move) == b.choose_move(board, to_move)


# --- v40 (batch 11, H2): must-block VCF upgrade at the leaf start ---


def _v40_regression_position(pid):
    """Load one v40 regression position (black/white stones, to_move)."""
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "gomoku_ml",
        "regression_positions_v40.json",
    )
    with open(path, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    pos = next(p for p in positions if p["id"] == pid)
    board = _board_with(pos["black"], pos["white"])
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    return board, player


def test_v40_registered_and_dispatches_to_mbvcf_leaf():
    v40 = scoring.get_player(40, seed=0)
    assert v40.name == "scoring-v40"
    assert v40.mcts is True
    assert v40._mcts is not None
    assert v40._mcts.sims == 100
    assert v40._mcts.leaf_mode == "rollout2_v1_both_mbvcf"
    assert v40._mcts.tie_break_q is True
    assert v40._mcts.tie_break_q_threshold == 0.05
    assert v40._mcts.vcf_root is False
    # v27 is unchanged (same leaf, same threshold, no upgrade).
    v27 = scoring.get_player(27, seed=0)
    assert v27._mcts.leaf_mode == "rollout2_v1_both"
    assert v27._mcts.tie_break_q_threshold == 0.05


def test_v40_leaf_upgrades_must_block_with_vcf_after_block():
    # g73 line: White plays the unsafe (11, 9); Black plays the VCF
    # start (8, 12). White to move faces a must-block (Black's only
    # winning move is (12, 12)); after the forced block Black holds
    # 3 VCF starts — a forced win. v27 values the leaf -0.25; v40
    # values it -1.0. See iter-40.md §1.1/§1.3.
    board, player = _v40_regression_position("b10-g73-ply35")
    assert player == game.WHITE
    b2 = game.apply_move(board, 9 * 15 + 11, game.WHITE)
    b3 = game.apply_move(b2, 12 * 15 + 8, game.BLACK)
    assert _terminal_leaf_value(b3, game.WHITE) == -0.25
    assert _rollout_leaf_value(b3, game.WHITE) == -0.25
    assert _rollout_leaf_value_mbvcf(b3, game.WHITE) == -1.0


def test_v40_leaf_keeps_must_block_without_vcf_after_block():
    # g73 safe line: White plays the forcing (11, 5); Black to move
    # faces a must-block (White's only winning move). After Black's
    # forced block White holds no VCF start, so the leaf stays -0.25
    # under both v27 and v40 — the upgrade does not fire.
    board, player = _v40_regression_position("b10-g73-ply35")
    b2 = game.apply_move(board, 5 * 15 + 11, game.WHITE)
    assert _terminal_leaf_value(b2, game.BLACK) == -0.25
    assert _rollout_leaf_value(b2, game.BLACK) == -0.25
    assert _rollout_leaf_value_mbvcf(b2, game.BLACK) == -0.25


def test_v40_leaf_identical_to_v27_on_quiet_positions():
    # On quiet positions (no winning move either side) the wrapper
    # delegates to the v18 rollout leaf unchanged.
    for pid in ("b10-g73-ply35", "b10-g61-ply21", "b10-g43-ply17"):
        board, player = _v40_regression_position(pid)
        assert _terminal_leaf_value(board, player) == 0.0
        assert _rollout_leaf_value_mbvcf(board, player) == (
            _rollout_leaf_value(board, player)
        )


def test_v40_leaf_identical_to_v27_on_terminal_positions():
    # A position where the player to move has a winning move (+1)
    # and one where the opponent has >= 2 (-1): the upgrade only
    # touches the must-block case, so both agree exactly.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _rollout_leaf_value_mbvcf(board, game.BLACK) == 1.0
    assert _rollout_leaf_value(board, game.BLACK) == 1.0
    # Opponent holds an open four: Black to move loses (-1).
    board = _board_with(
        [(3, 3)],
        [(7, 7), (7, 8), (7, 9), (7, 10)],
    )
    assert _rollout_leaf_value_mbvcf(board, game.BLACK) == -1.0
    assert _rollout_leaf_value(board, game.BLACK) == -1.0


def test_v40_leaf_does_not_mutate_board():
    board, player = _v40_regression_position("b10-g73-ply35")
    b2 = game.apply_move(board, 9 * 15 + 11, game.WHITE)
    b3 = game.apply_move(b2, 12 * 15 + 8, game.BLACK)
    before = b3.copy()
    _rollout_leaf_value_mbvcf(b3, game.WHITE)
    assert np.array_equal(b3, before)


def test_v40_unknown_leaf_mode_rejected():
    with pytest.raises(ValueError, match="unknown leaf_mode"):
        HeuristicMCTS(leaf_mode="rollout2_v1_both_bogus")

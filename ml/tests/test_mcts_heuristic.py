"""Tests for the heuristic-prior MCTS (v6, batch 2 H2; v7, batch 3 H1;
v16, batch 6 H2)."""

import math

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import (
    HeuristicMCTS,
    _attack_gradient,
    _attack_gradient_v1,
    _gradient_leaf_value,
    _priors,
    _rollout_leaf_value,
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


def test_attack_gradient_v1_antisymmetric_and_bounded():
    rng = np.random.default_rng(4)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    gb = _attack_gradient_v1(board, game.BLACK)
    gw = _attack_gradient_v1(board, game.WHITE)
    assert gb == pytest.approx(-gw)
    assert -0.2 < gb < 0.2
    assert -0.2 < gw < 0.2


def test_v16_registered_and_dispatches_to_rollout_mcts():
    v16 = scoring.get_player(16, seed=0)
    assert v16.name == "scoring-v16"
    assert v16.mcts is True
    assert v16._mcts is not None
    assert v16._mcts.sims == 100
    assert v16._mcts.leaf_mode == "rollout2_v1"
    # v7 is unchanged (gradient leaf).
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

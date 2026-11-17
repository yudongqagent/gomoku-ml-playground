"""Tests for the heuristic-prior MCTS (v6, batch 2 H2; v7, batch 3 H1;
v16, batch 6 H2; v18, batch 7 H1; v23, batch 8 H3)."""

import math

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import (
    HeuristicMCTS,
    _Node,
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


def test_v27_sweep_candidates_registered():
    # TEMPORARY sweep entries (iter-27 §1.2): removed when v27 is
    # registered with the selected threshold.
    v90 = scoring.get_player(90, seed=0)
    assert v90.name == "scoring-v27-tau005"
    assert v90._mcts.leaf_mode == "rollout2_v1_both"
    assert v90._mcts.tie_break_q is True
    assert v90._mcts.tie_break_q_threshold == 0.05
    v91 = scoring.get_player(91, seed=0)
    assert v91.name == "scoring-v27-tau010"
    assert v91._mcts.tie_break_q_threshold == 0.10
    # Same seed, same board, same move — the threshold consumes
    # RNG only where v18/v23 would, so determinism holds.
    board = _board_with(
        [(7, 7), (6, 7), (8, 7), (7, 6)], [(7, 8), (6, 8), (8, 8), (7, 5)]
    )
    a = scoring.get_player(90, seed=0)
    b = scoring.get_player(90, seed=0)
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)

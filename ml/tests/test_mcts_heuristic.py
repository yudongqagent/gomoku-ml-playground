"""Tests for the heuristic-prior MCTS (v6, batch 2 H2; v7, batch 3 H1;
v16, batch 6 H2; v18, batch 7 H1; v19, batch 7 H2)."""

import math

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import (
    HeuristicMCTS,
    _attack_gradient,
    _attack_gradient_v1,
    _blend_rollout_terminal,
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


# --- v19 (batch 7, H2): blended rollout terminal ----------------------------

def test_blend_rollout_terminal_zero_gradient_is_identity():
    # Empty board: both breadths are 0, so the gradient is 0 and the
    # blend returns the terminal unchanged (a draw is a fact).
    board = game.empty_board()
    assert _blend_rollout_terminal(1.0, board, game.BLACK) == 1.0
    assert _blend_rollout_terminal(-1.0, board, game.BLACK) == -1.0
    assert _blend_rollout_terminal(0.0, board, game.BLACK) == 0.0


def test_blend_rollout_terminal_shifts_and_clamps():
    # Black holds an open three (two four-makers); White has a lone
    # stone — Black's breadth dominates, so the gradient for Black
    # is positive and for White negative.
    board = _board_with([(7, 7), (8, 7), (9, 7)], [(3, 3)])
    assert _attack_gradient_v1(board, game.BLACK) > 0
    # A loss softened by counterplay breadth, a win pinned by the
    # clamp (predictions never exceed facts).
    softened_loss = _blend_rollout_terminal(-1.0, board, game.BLACK)
    assert -1.0 < softened_loss <= -0.8
    assert _blend_rollout_terminal(1.0, board, game.BLACK) == 1.0
    # Same board from White's perspective: the signs flip.
    weak_win = _blend_rollout_terminal(1.0, board, game.WHITE)
    assert 0.8 <= weak_win < 1.0
    assert _blend_rollout_terminal(-1.0, board, game.WHITE) == -1.0


def test_blended_rollout_leaf_matches_v16_on_quiet_rollout_ends():
    # From the empty board the 2-ply rollout ends quietly (no
    # terminal in 2 plies), so the blend is a no-op: v19's leaf
    # equals v16's exactly. The blend ONLY affects rollout
    # terminals.
    board = game.empty_board()
    assert _rollout_leaf_value(board, game.BLACK, blend=True) == (
        _rollout_leaf_value(board, game.BLACK, blend=False)
    )
    assert _rollout_leaf_value(board, game.WHITE, blend=True) == (
        _rollout_leaf_value(board, game.WHITE, blend=False)
    )


def test_blended_rollout_leaf_matches_terminal_leaf_on_terminal_positions():
    # Leaf-start terminals are FACTS: the blend never touches them.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert _rollout_leaf_value(board, game.BLACK, blend=True) == 1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert _rollout_leaf_value(board, game.BLACK, blend=True) == -1.0
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert _rollout_leaf_value(board, game.BLACK, blend=True) == -0.25


def test_blended_rollout_leaf_bounded_and_deterministic():
    rng = np.random.default_rng(2)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    v1 = _rollout_leaf_value(board, game.BLACK, blend=True)
    v2 = _rollout_leaf_value(board, game.BLACK, blend=True)
    assert -1.0 <= v1 <= 1.0
    assert v1 == v2  # deterministic: no RNG in the leaf


def test_v19_registered_and_dispatches_to_blend_mcts():
    v19 = scoring.get_player(19, seed=0)
    assert v19.name == "scoring-v19"
    assert v19.mcts is True
    assert v19._mcts is not None
    assert v19._mcts.sims == 100
    assert v19._mcts.leaf_mode == "rollout2_v1_blend"
    # v16 and v7 are unchanged.
    v16 = scoring.get_player(16, seed=0)
    assert v16._mcts.leaf_mode == "rollout2_v1"
    v7 = scoring.get_player(7, seed=0)
    assert v7._mcts.leaf_mode == "gradient"


def test_v19_as_black_matches_v7():
    # v19-as-Black IS v7 exactly (same leaf, same priors, same
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
        v19 = scoring.get_player(19, seed=0)
        assert v19.choose_move(board, game.BLACK) == v7.choose_move(
            board, game.BLACK
        ), f"trial {trial}: v19-as-Black differs from v7"


def test_v19_leaf_value_matches_v7_when_root_is_black():
    # Leaf-level version of the Black-control claim: with the root
    # player Black, the v19 leaf equals v7's gradient leaf on every
    # sampled board (terminal and quiet alike).
    rng = np.random.default_rng(6)
    mcts = HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_blend")
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


def test_v19_choose_move_deterministic_for_same_seed():
    rng = np.random.default_rng(1)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:20]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    a = scoring.get_player(19, seed=0)
    b = scoring.get_player(19, seed=0)
    assert a.choose_move(board, game.WHITE) == b.choose_move(board, game.WHITE)


def test_v19_choose_move_does_not_mutate_board():
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = HeuristicMCTS(sims=20, leaf_mode="rollout2_v1_blend",
                         rng=np.random.default_rng(0))
    mcts.choose_move(board, game.WHITE)
    assert np.array_equal(board, before)


def test_unknown_rollout_leaf_mode_raises():
    with pytest.raises(ValueError, match="unknown leaf_mode"):
        HeuristicMCTS(sims=10, leaf_mode="rollout2_v1_blended")


def test_blended_rollout_leaf_spreads_terminal_lines():
    # The blend's whole point (iter-19 §1): lines the rollout calls
    # terminal get spread by the gradient at the terminal position
    # instead of all collapsing to exactly ±1.0. Two concrete boards
    # (found by probing random boards): a rollout win weakened by an
    # unfavourable end gradient, and a rollout loss softened by
    # counterplay breadth — both stay inside the terminal band
    # ([0.8, 1.0] / [-1.0, -0.8]), clearly separated from quiet
    # leaves (±0.2).
    rng = np.random.default_rng(2)
    board = game.empty_board()
    cells = [(x, y) for y in range(3, 12) for x in range(3, 12)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    assert _rollout_leaf_value(board, game.BLACK) == 1.0
    blended = _rollout_leaf_value(board, game.BLACK, blend=True)
    assert 0.8 <= blended < 1.0

    rng = np.random.default_rng(8)
    board = game.empty_board()
    cells = [(x, y) for y in range(3, 12) for x in range(3, 12)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    assert _rollout_leaf_value(board, game.BLACK) == -1.0
    blended = _rollout_leaf_value(board, game.BLACK, blend=True)
    assert -1.0 < blended <= -0.8

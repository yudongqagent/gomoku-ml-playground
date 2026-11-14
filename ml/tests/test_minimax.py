"""Tests for scoring-v4 (2-ply minimax over v1's top candidates)."""

import numpy as np

from gomoku_ml import game, scoring


def _board_from(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


# Batch-2 game 1 (v4-vs-v0), before White's move 12. Black holds an
# open three (4,7),(5,6),(6,5) with creators (3,8) and (7,4). v4
# plays (8,4) (its own open three) and loses by force; the blocks
# (3,8)/(7,4) avoid the forced loss (VCF-solver verified — see
# research/iterations/iter-4.md §2.3). v5 must play a block.
GAME1_V4V0_BLACK = [[7, 7], [7, 6], [6, 5], [5, 6], [6, 4], [4, 7]]
GAME1_V4V0_WHITE = [[5, 5], [7, 5], [5, 4], [8, 6], [6, 6]]


def test_v5_registered():
    assert 5 in scoring.ENGINE_VERSIONS
    p = scoring.get_player(5, seed=0)
    assert p.name == "scoring-v5"
    assert p.minimax is True
    assert p.minimax_topk == 8
    assert p.terminal_aware is True
    assert p.defence_weight == 1.0
    assert p.threat_check is False


def test_leaf_value_open_four_is_terminal_loss():
    # Black holds an open four; White to move is lost no matter what
    # White's attack is. The value must be exactly -WIN_SCORE (no
    # compensation from our attack score).
    black = [[7, 7], [8, 7], [9, 7], [10, 7]]
    white = [[5, 5], [6, 6]]  # White's own diagonal two (some attack)
    board = _board_from(black, white)
    assert len(scoring._winning_moves(board, game.BLACK)) == 2
    assert scoring._leaf_value(board, game.WHITE) == -scoring.WIN_SCORE


def test_leaf_value_open_four_is_terminal_win():
    white = [[7, 7], [8, 7], [9, 7], [10, 7]]
    black = [[5, 5], [6, 6]]
    board = _board_from(black, white)
    assert len(scoring._winning_moves(board, game.WHITE)) == 2
    assert scoring._leaf_value(board, game.WHITE) == scoring.WIN_SCORE


def test_leaf_value_single_winning_move_capped():
    # Black has a four with exactly one winning square: (6,7) is
    # blocked by White, so only (11,7) wins. White to move blocks
    # it — the position is bad for White but NOT terminal: the value
    # must be far from -WIN_SCORE (a single four capped at FOUR
    # plus Black's best non-winning attack).
    black = [[7, 7], [8, 7], [9, 7], [10, 7]]
    white = [[6, 7], [5, 5]]
    board = _board_from(black, white)
    assert scoring._winning_moves(board, game.BLACK) == [game.xy_to_move(11, 7)]
    value = scoring._leaf_value(board, game.WHITE)
    assert value < 0
    assert value > -scoring.WIN_SCORE // 2


def test_v5_blocks_on_game1_regression():
    # Pre-registered check (iter-5 §1): v5 must play a block on the
    # game-1 position where v4 played the losing attack (8,4).
    board = _board_from(GAME1_V4V0_BLACK, GAME1_V4V0_WHITE)
    assert sorted(game.move_to_xy(c)
                  for c in scoring._opponent_creators(board, game.WHITE)) == [
        (3, 8), (7, 4)]
    choice = scoring._choose_move(board, game.WHITE, None, 1.0,
                                  threat_check=False, minimax=True,
                                  minimax_topk=8, terminal_aware=True)
    assert game.move_to_xy(choice) in [(3, 8), (7, 4)]
    # Same via the versioned player.
    p5 = scoring.get_player(5, seed=0)
    assert game.move_to_xy(p5.choose_move(board, game.WHITE)) in [(3, 8), (7, 4)]


def test_v4_still_plays_attack_on_game1():
    # Guards v4's behaviour: the terminal-aware change must not alter
    # v4 (terminal_aware=False is the default path).
    board = _board_from(GAME1_V4V0_BLACK, GAME1_V4V0_WHITE)
    choice = scoring._choose_move(board, game.WHITE, None, 1.0,
                                  threat_check=False, minimax=True,
                                  minimax_topk=8, terminal_aware=False)
    assert game.move_to_xy(choice) == (8, 4)
    p4 = scoring.get_player(4, seed=0)
    assert game.move_to_xy(p4.choose_move(board, game.WHITE)) == (8, 4)


def test_v5_takes_win_and_blocks():
    # Win/block scans run before minimax for v5 too.
    board = _board_from([[7, 7], [8, 7], [9, 7], [10, 7]], [[5, 5]])
    p5 = scoring.get_player(5, seed=0)
    assert game.move_to_xy(p5.choose_move(board, game.BLACK)) in [(6, 7), (11, 7)]
    board = _board_from([[5, 5]], [[7, 7], [8, 7], [9, 7], [10, 7]])
    assert game.move_to_xy(p5.choose_move(board, game.BLACK)) in [(6, 7), (11, 7)]
# Batch-1 game 13 (v1-vs-v0 swap), after 22 plies, Black to move.
# White holds four open-four creators; v1 plays (5, 4) (double open
# three, 1-ply best) and loses the race. The preventing move is
# (10, 7), a forcing four (see research/iterations/iter-3.md §1).
GAME13_BLACK = [[7, 2], [4, 3], [5, 3], [6, 3], [8, 5], [5, 6],
                [10, 6], [7, 7], [8, 7], [9, 7], [6, 8]]
GAME13_WHITE = [[7, 3], [6, 4], [7, 4], [8, 4], [6, 5], [7, 5],
                [6, 6], [7, 6], [8, 6], [9, 6], [6, 7]]


def test_v4_registered():
    assert 4 in scoring.ENGINE_VERSIONS
    p = scoring.get_player(4, seed=0)
    assert p.name == "scoring-v4"
    assert p.minimax is True
    assert p.minimax_topk == 8
    assert p.defence_weight == 1.0
    assert p.threat_check is False


def test_v4_prevents_where_v1_does_not():
    board = _board_from(GAME13_BLACK, GAME13_WHITE)
    creators = scoring._opponent_creators(board, game.BLACK)
    assert sorted(game.move_to_xy(c) for c in creators) == [
        (5, 4), (5, 7), (9, 3), (9, 4),
    ]
    v1 = scoring.choose_move(board, game.BLACK, None)
    assert v1 == game.xy_to_move(5, 4)
    assert not scoring._move_is_safe(board, v1, game.BLACK, creators)
    v4 = scoring.get_player(4, seed=0).choose_move(board, game.BLACK)
    assert v4 == game.xy_to_move(10, 7)  # the forcing four
    assert scoring._move_is_safe(board, v4, game.BLACK, creators)


def test_v4_takes_immediate_win():
    board = _board_from(
        [[7, 7], [8, 7], [9, 7], [10, 7]],
        [[1, 1], [13, 13]],
    )
    move = scoring.get_player(4, seed=0).choose_move(board, game.BLACK)
    assert game.is_winning_move(board, move, game.BLACK)


def test_v4_blocks_immediate_win():
    board = _board_from(
        [[1, 1], [13, 13]],
        [[7, 7], [8, 7], [9, 7], [10, 7]],
    )
    move = scoring.get_player(4, seed=0).choose_move(board, game.BLACK)
    assert move in (game.xy_to_move(6, 7), game.xy_to_move(11, 7))


def test_minimax_topk_1_equals_v1():
    # With a single candidate the minimax has nothing to choose
    # between, so v4 must return v1's move.
    board = _board_from(
        [[7, 7], [8, 7], [7, 8], [3, 3], [11, 11]],
        [[7, 6], [6, 7], [10, 10], [4, 4]],
    )
    v1 = scoring._choose_move(board, game.BLACK, None, 1.0)
    v4 = scoring._choose_move(board, game.BLACK, None, 1.0,
                              minimax=True, minimax_topk=1)
    assert v4 == v1


def test_v4_deterministic_with_seed():
    board = _board_from(GAME13_BLACK, GAME13_WHITE)
    a = scoring.get_player(4, seed=123).choose_move(board, game.BLACK)
    b = scoring.get_player(4, seed=123).choose_move(board, game.BLACK)
    assert a == b


def test_v4_moves_are_legal():
    rng = np.random.default_rng(11)
    for _ in range(5):
        board = game.empty_board()
        stones = rng.choice(225, size=20, replace=False)
        for i, m in enumerate(stones):
            board = game.apply_move(
                board, int(m), game.BLACK if i % 2 == 0 else game.WHITE)
        move = scoring.get_player(4, seed=0).choose_move(board, game.BLACK)
        assert game.is_legal(board, move)


def test_v4_choose_move_completes_quickly():
    # Loose bound (actual ~80 ms on this VM): catches pathological
    # slowdowns, not small regressions.
    import time

    rng = np.random.default_rng(7)
    board = game.empty_board()
    stones = rng.choice(225, size=24, replace=False)
    for i, m in enumerate(stones):
        board = game.apply_move(
            board, int(m), game.BLACK if i % 2 == 0 else game.WHITE)
    start = time.perf_counter()
    scoring.get_player(4, seed=0).choose_move(board, game.BLACK)
    assert time.perf_counter() - start < 2.0

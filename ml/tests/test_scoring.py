"""Unit tests for the v1 scoring engine (gomoku_ml/scoring.py).

Every behaviour change vs the frozen baseline gets a test here:
the broken-four fix, four valuation, double-threat bonuses,
max-per-direction (no double-counting), defence scoring, and the
choose_move contract (win/block/candidates).
"""

import numpy as np
import pytest

from gomoku_ml import baseline, game, scoring


def _board(black=(), white=()):
    b = game.empty_board()
    for x, y in black:
        b = game.apply_move(b, game.xy_to_move(x, y), game.BLACK)
    for x, y in white:
        b = game.apply_move(b, game.xy_to_move(x, y), game.WHITE)
    return b


# --- the broken-four fix --------------------------------------------------

def test_broken_four_creation_scores_as_four():
    # Black XX_X at (3,7),(4,7),(6,7); playing (7,7) makes XX_XX.
    b = _board(black=[(3, 7), (4, 7), (6, 7)])
    move = game.xy_to_move(7, 7)
    assert scoring.score_move(b, move, game.BLACK) >= scoring.FOUR
    # The frozen baseline is blind to it (documents the bug fixed here).
    assert baseline.score_move(b, move, game.BLACK) < 1000


def test_broken_four_gap_fill_scores_as_win():
    # Black XX_XX at (3,7),(4,7),(6,7),(7,7); the gap (5,7) wins.
    b = _board(black=[(3, 7), (4, 7), (6, 7), (7, 7)])
    move = game.xy_to_move(5, 7)
    assert scoring.score_move(b, move, game.BLACK) >= scoring.WIN_SCORE
    assert scoring.choose_move(b, game.BLACK) == move


def test_broken_four_on_diagonal():
    b = _board(black=[(3, 3), (4, 4), (6, 6)])
    move = game.xy_to_move(7, 7)  # completes the diagonal XX_XX
    assert scoring.score_move(b, move, game.BLACK) >= scoring.FOUR


# --- four valuation -------------------------------------------------------

def test_open_four_beats_closed_four():
    # Open four: Black XXX at (3,7),(4,7),(5,7), both ends open.
    b_open = _board(black=[(3, 7), (4, 7), (5, 7)])
    s_open = scoring.score_move(b_open, game.xy_to_move(6, 7), game.BLACK)
    # Closed four: same, but White blocks (2,7) — only (7,7) wins.
    b_closed = _board(black=[(3, 7), (4, 7), (5, 7)], white=[(2, 7)])
    s_closed = scoring.score_move(b_closed, game.xy_to_move(6, 7), game.BLACK)
    assert s_open >= scoring.OPEN_FOUR
    assert scoring.FOUR <= s_closed < scoring.OPEN_FOUR
    assert s_open > s_closed


def test_closed_four_with_winning_square_still_a_four():
    # Black XXX at (3,7),(4,7),(5,7); White at (1,7) and (7,7).
    # Playing (6,7) makes a four at 3-6 whose only winning square is
    # (2,7) — a real four, even though both outside cells of its
    # window are blocked (the TS engine scores this window 0).
    b = _board(black=[(3, 7), (4, 7), (5, 7)], white=[(1, 7), (7, 7)])
    s = scoring.score_move(b, game.xy_to_move(6, 7), game.BLACK)
    assert scoring.FOUR <= s < scoring.OPEN_FOUR


def test_dead_four_scores_low():
    # Black XXX at (3,7),(4,7),(5,7); White at (2,7) and (7,7).
    # Playing (6,7) makes a four at 3-6 with NO winning square
    # (both adjacent points are White) — correctly worth almost
    # nothing, not a four.
    b = _board(black=[(3, 7), (4, 7), (5, 7)], white=[(2, 7), (7, 7)])
    s = scoring.score_move(b, game.xy_to_move(6, 7), game.BLACK)
    assert s < 1000


def test_open_four_via_window_start_gap():
    # Black XXX at (3,7),(4,7),(5,7); White blocks (1,7) only.
    # Playing (6,7): window [2..6] is [_,B,B,B,B] with (7,7) empty,
    # so the four has two winning squares ((2,7) and (7,7)).
    b = _board(black=[(3, 7), (4, 7), (5, 7)], white=[(1, 7)])
    s = scoring.score_move(b, game.xy_to_move(6, 7), game.BLACK)
    assert s >= scoring.OPEN_FOUR


# --- max-per-direction (no double-counting) -------------------------------

def test_single_four_counts_once():
    # One open four must score ~200k, not ~400k: the four sits in two
    # overlapping 5-windows, but it is one physical pattern.
    b = _board(black=[(3, 7), (4, 7), (5, 7)])
    s = scoring.score_move(b, game.xy_to_move(6, 7), game.BLACK)
    assert scoring.OPEN_FOUR <= s < 2 * scoring.OPEN_FOUR


def test_single_closed_four_counts_once():
    b = _board(black=[(3, 7), (4, 7), (5, 7)], white=[(2, 7)])
    s = scoring.score_move(b, game.xy_to_move(6, 7), game.BLACK)
    assert scoring.FOUR <= s < 2 * scoring.FOUR


# --- double threats -------------------------------------------------------

def test_double_four_beats_single_open_four():
    # Two threes meeting at (7,7): playing there makes two open fours.
    b = _board(black=[(4, 7), (5, 7), (6, 7), (7, 4), (7, 5), (7, 6)])
    s_double = scoring.score_move(b, game.xy_to_move(7, 7), game.BLACK)
    s_single = scoring.score_move(b, game.xy_to_move(3, 7), game.BLACK)
    assert s_double > s_single
    assert s_double >= 2 * scoring.OPEN_FOUR + scoring.DOUBLE_FOUR_BONUS
    assert scoring.choose_move(b, game.BLACK) == game.xy_to_move(7, 7)


def test_double_open_three_beats_single_closed_four():
    # Two twos meeting at (7,7): playing there makes two open threes.
    b = _board(black=[(5, 7), (6, 7), (7, 5), (7, 6)])
    s_double = scoring.score_move(b, game.xy_to_move(7, 7), game.BLACK)
    assert s_double >= 2 * scoring.OPEN_THREE + scoring.DOUBLE_OPEN_THREE_BONUS
    # ... but a double open three still loses to a single open four
    # (which wins next move outright).
    b_four = _board(black=[(3, 7), (4, 7), (5, 7)])
    s_four = scoring.score_move(b_four, game.xy_to_move(6, 7), game.BLACK)
    assert s_double < s_four


# --- defence --------------------------------------------------------------

def test_blocking_open_three_scores_high():
    # White has an open three at (4,7),(5,7),(6,7). Black playing
    # either end denies White an open four there.
    b = _board(white=[(4, 7), (5, 7), (6, 7)])
    s_block = scoring.score_move(b, game.xy_to_move(7, 7), game.BLACK)
    s_elsewhere = scoring.score_move(b, game.xy_to_move(0, 0), game.BLACK)
    assert s_block >= scoring.OPEN_FOUR
    assert s_block > s_elsewhere


def test_defence_of_double_threat_point():
    # White has two threes meeting at (7,7); Black (7,7) blocks both.
    b = _board(white=[(4, 7), (5, 7), (6, 7), (7, 4), (7, 5), (7, 6)])
    assert scoring.choose_move(b, game.BLACK) == game.xy_to_move(7, 7)


# --- choose_move contract -------------------------------------------------

def test_choose_move_takes_win_over_block():
    # Black has an open four and White has a half-open four; Black
    # must take the win, not block.
    b = _board(
        black=[(3, 7), (4, 7), (5, 7), (6, 7)],
        white=[(7, 3), (7, 4), (7, 5), (7, 6), (7, 2)],
    )
    move = scoring.choose_move(b, game.BLACK)
    assert game.is_winning_move(b, move, game.BLACK)


def test_choose_move_blocks_opponent_win():
    # White has a half-open four at (3,7)-(6,7) (Black stone at (2,7)
    # blocks the other end); only (7,7) stops the five.
    b = _board(black=[(2, 7)], white=[(3, 7), (4, 7), (5, 7), (6, 7)])
    assert scoring.choose_move(b, game.BLACK) == game.xy_to_move(7, 7)


def test_choose_move_blocks_gap_four():
    # White's four has a hole: only the gap square stops the five.
    b = _board(white=[(3, 7), (4, 7), (6, 7), (7, 7)])
    assert scoring.choose_move(b, game.BLACK) == game.xy_to_move(5, 7)


def test_choose_move_empty_board_is_centre():
    assert scoring.choose_move(game.empty_board(), game.BLACK) == game.xy_to_move(7, 7)


def test_choose_move_no_legal_moves_raises():
    b = np.full((15, 15), game.BLACK, dtype=np.int8)
    with pytest.raises(ValueError):
        scoring.choose_move(b, game.BLACK)


def test_choose_move_deterministic_without_rng():
    b = _board(black=[(7, 7), (8, 8)], white=[(7, 8)])
    assert scoring.choose_move(b, game.BLACK) == scoring.choose_move(b, game.BLACK)


def test_choose_move_rng_returns_legal_move():
    rng = np.random.default_rng(0)
    b = _board(black=[(7, 7), (8, 8)], white=[(7, 8)])
    for _ in range(20):
        assert game.is_legal(b, scoring.choose_move(b, game.BLACK, rng))


def test_choose_move_on_near_full_board():
    # Seeded near-full board: the candidate fallback must still find
    # a legal move and never crash.
    rng = np.random.default_rng(123)
    b = game.empty_board()
    cells = rng.permutation(game.SIZE * game.SIZE)[:220]
    for i, m in enumerate(cells):
        b = game.apply_move(b, int(m), game.BLACK if i % 2 == 0 else game.WHITE)
    move = scoring.choose_move(b, game.BLACK)
    assert game.is_legal(b, move)


def test_scoring_player_seeded_reproducible():
    b = _board(black=[(7, 7)], white=[(8, 8)])
    a = scoring.ScoringPlayer(seed=42)
    c = scoring.ScoringPlayer(seed=42)
    assert a.choose_move(b, game.BLACK) == c.choose_move(b, game.BLACK)


# --- attack-only scoring (used by the swap colour choice) ------------------

def test_score_attack_is_colour_asymmetric():
    # Attack scores differ between colours (unlike the total score,
    # which is symmetric) — that is what makes the swap-rule colour
    # comparison meaningful.
    b = _board(black=[(3, 7), (4, 7), (5, 7)])
    move = game.xy_to_move(6, 7)  # completes Black's open four
    assert scoring.score_attack(b, move, game.BLACK) >= scoring.OPEN_FOUR
    assert scoring.score_attack(b, move, game.WHITE) < scoring.FOUR


def test_score_attack_never_exceeds_total():
    b = _board(black=[(7, 7), (8, 8)], white=[(7, 8)])
    for m in game.legal_moves(b):
        assert scoring.score_attack(b, int(m), game.BLACK) <= scoring.score_move(
            b, int(m), game.BLACK
        )


# --- structural properties ------------------------------------------------

def test_window_count():
    # 15 rows x 11 + 15 cols x 11 + 2 diagonals x (11+10+...+1)*2... =
    # 572 five-cell windows on a 15x15 board.
    assert len(scoring._WINDOWS) == 572


def test_score_symmetric_under_rotation():
    # No positional bias: rotating the board 180 degrees must not
    # change the score of the rotated move.
    rng = np.random.default_rng(9)
    b = game.empty_board()
    for i, m in enumerate(rng.choice(225, size=14, replace=False)):
        b = game.apply_move(b, int(m), game.BLACK if i % 2 == 0 else game.WHITE)
    move = game.xy_to_move(10, 3)
    assert game.is_legal(b, move)
    b_rot = np.rot90(b, 2)
    move_rot = game.xy_to_move(14 - 10, 14 - 3)
    assert scoring.score_move(b, move, game.BLACK) == scoring.score_move(
        b_rot, move_rot, game.BLACK
    )


def test_interface_matches_baseline():
    # Same call signatures as the frozen baseline.
    import inspect

    for name in ("score_move", "choose_move"):
        assert inspect.signature(getattr(scoring, name)) == inspect.signature(
            getattr(baseline, name)
        )


def test_v2_defence_discount():
    # v2 (defence_weight 0.9) scores a purely defensive move lower
    # than v1 (1.0) by exactly the per-direction discount, and scores
    # a purely attacking move identically.
    board = game.empty_board()
    # White has an open three at row 7, cols 3-5; Black to move.
    for x in (3, 4, 5):
        board[7, x] = game.WHITE
    # Playing (7, 2) blocks the three: mostly defence.
    block = game.xy_to_move(2, 7)
    v1 = scoring.ScoringPlayer(seed=0, version=1)
    v2 = scoring.get_player(2, seed=0)
    assert v2.defence_weight == 0.9
    assert v2.score_move(board, block, game.BLACK) < v1.score_move(
        board, block, game.BLACK
    )
    # A mostly-attacking move (Black builds alone, White has no
    # stones) differs only by the discount on the small defence value
    # empty windows contribute (1-2 per direction, at most 8 total) —
    # the attack component itself is identical.
    board2 = game.empty_board()
    for x in (3, 4):
        board2[7, x] = game.BLACK
    attack = game.xy_to_move(5, 7)
    diff = v1.score_move(board2, attack, game.BLACK) - v2.score_move(
        board2, attack, game.BLACK
    )
    assert 0 <= diff <= 8


def test_get_player_versions():
    assert scoring.get_player(1).name == "scoring-v1"
    assert scoring.get_player(2).name == "scoring-v2"
    assert scoring.get_player(3).name == "scoring-v3"
    with pytest.raises(ValueError):
        scoring.get_player(99)


# Real missed-prevention position: game 13, ply 22 of the v1-vs-v0
# swap batch (research/iterations/batch-v1-vs-v0-swap.json). Black to
# move; White holds four open-four creators. v1 plays (5, 4) — it
# blocks one creator AND creates a double open three (1-ply score
# 258,002) — but leaves three creators live and loses the race. v3
# plays (10, 7), a forcing four (1-ply score 20,097), which leaves
# White no live creator.
_MISSED_PREVENTION_MOVES = [
    112, 81, 49, 97, 113, 111, 83, 96, 126, 98, 95, 99, 100, 66, 51,
    82, 50, 52, 114, 67, 37, 68,
]


def _board_from_moves(moves):
    board = game.empty_board()
    for i, m in enumerate(moves):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE
        )
    return board


def test_v3_prevents_where_v1_does_not():
    board = _board_from_moves(_MISSED_PREVENTION_MOVES)
    creators = scoring._opponent_creators(board, game.BLACK)
    assert len(creators) == 4
    v1_move = scoring._choose_move(board, game.BLACK, None, 1.0, False)
    v3_move = scoring._choose_move(board, game.BLACK, None, 1.0, True)
    assert v1_move == game.xy_to_move(5, 4)
    assert v3_move == game.xy_to_move(10, 7)
    assert not scoring._move_is_safe(board, v1_move, game.BLACK, creators)
    assert scoring._move_is_safe(board, v3_move, game.BLACK, creators)
    # v1's move scores far higher in 1-ply — the threat check, not the
    # score, is what makes v3 choose differently.
    v1 = scoring.get_player(1, seed=0)
    assert v1.score_move(board, v1_move, game.BLACK) > v1.score_move(
        board, v3_move, game.BLACK
    )


def test_v3_same_as_v1_without_threat():
    # With no opponent creator, the threat check is a no-op: v3
    # chooses exactly like v1. (Empty board + a quiet midgame.)
    board = game.empty_board()
    assert scoring._choose_move(
        board, game.BLACK, None, 1.0, True
    ) == scoring._choose_move(board, game.BLACK, None, 1.0, False)
    board = game.empty_board()
    for x in (3, 4):
        board[7, x] = game.BLACK
    board[8, 8] = game.WHITE
    assert scoring._opponent_creators(board, game.BLACK) == []
    assert scoring._choose_move(
        board, game.BLACK, None, 1.0, True
    ) == scoring._choose_move(board, game.BLACK, None, 1.0, False)


def test_v3_forcing_move_is_safe():
    # A move that makes a four is forcing (the opponent must block),
    # so it counts as safe even with creators on the board.
    board = _board_from_moves(_MISSED_PREVENTION_MOVES)
    creators = scoring._opponent_creators(board, game.BLACK)
    four_move = game.xy_to_move(10, 7)
    assert scoring._is_forcing(board, four_move, game.BLACK)
    assert scoring._move_is_safe(board, four_move, game.BLACK, creators)

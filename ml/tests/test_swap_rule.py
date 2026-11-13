"""Unit tests for the swap-rule helpers in gomoku_ml/game.py."""

import numpy as np
import pytest

from gomoku_ml import game, scoring


def _opening(black1=(7, 7), black2=(8, 8), white1=(7, 8)):
    return game.build_swap_opening(
        game.xy_to_move(*black1), game.xy_to_move(*black2), game.xy_to_move(*white1)
    )


def test_build_swap_opening_places_stones():
    b = _opening()
    assert int(np.count_nonzero(b == game.BLACK)) == 2
    assert int(np.count_nonzero(b == game.WHITE)) == 1
    assert b[7, 7] == game.BLACK
    assert b[8, 8] == game.BLACK
    assert b[8, 7] == game.WHITE
    assert game.is_swap_opening(b)


def test_build_swap_opening_rejects_duplicates():
    with pytest.raises(ValueError):
        game.build_swap_opening(0, 0, 1)
    with pytest.raises(ValueError):
        game.build_swap_opening(0, 1, 1)


def test_build_swap_opening_rejects_out_of_range():
    with pytest.raises(ValueError):
        game.build_swap_opening(0, 1, 225)
    with pytest.raises(ValueError):
        game.build_swap_opening(-1, 1, 2)


def test_is_swap_opening_false_for_other_boards():
    assert not game.is_swap_opening(game.empty_board())
    b = game.apply_move(game.empty_board(), 0, game.BLACK)
    assert not game.is_swap_opening(b)
    # 3 black stones is not an opening either.
    b3 = game.empty_board()
    for m in (0, 1, 2):
        b3 = game.apply_move(b3, m, game.BLACK)
    assert not game.is_swap_opening(b3)


def test_choose_swap_color_picks_higher_best_score():
    # Opener placed Black at (0,0),(14,0) and White at (7,7) — but a
    # single white stone is no threat. Instead verify the mechanism
    # with a stub scorer: the colour whose best move scores higher
    # wins, whatever the board looks like.
    b = _opening()
    stub = lambda board, move, player: 10 if player == game.WHITE else 1
    assert game.choose_swap_color(b, stub) == game.WHITE
    stub2 = lambda board, move, player: 10 if player == game.BLACK else 1
    assert game.choose_swap_color(b, stub2) == game.BLACK


def test_choose_swap_color_tie_goes_to_white():
    # Ties go to White: White plays move 4 (RIF rules Art. 2.4), so
    # the next-move tempo breaks the tie.
    b = _opening()
    stub = lambda board, move, player: 5  # every move scores the same
    assert game.choose_swap_color(b, stub) == game.WHITE


def test_choose_swap_color_uses_real_scorer():
    # Black stones adjacent at the corner, White alone in the centre:
    # Black's best attack extends its pair into a three, White's best
    # attack is a lone two next to its single stone. The corner pair
    # wins the comparison, so the second player takes Black.
    b = game.build_swap_opening(
        game.xy_to_move(0, 0), game.xy_to_move(1, 0), game.xy_to_move(7, 7)
    )
    best_black = max(
        scoring.score_attack(b, int(m), game.BLACK) for m in game.legal_moves(b)
    )
    best_white = max(
        scoring.score_attack(b, int(m), game.WHITE) for m in game.legal_moves(b)
    )
    assert best_black > best_white  # the premise of the test
    assert game.choose_swap_color(b, scoring.score_attack) == game.BLACK


def test_choose_swap_color_white_wins_with_real_scorer():
    # Black's two stones are at opposite corners (no synergy — the
    # best Black attack is an edge two), White's single stone is in
    # the centre (best White attack is an open two). White wins.
    b = game.build_swap_opening(
        game.xy_to_move(0, 0), game.xy_to_move(0, 14), game.xy_to_move(7, 7)
    )
    best_black = max(
        scoring.score_attack(b, int(m), game.BLACK) for m in game.legal_moves(b)
    )
    best_white = max(
        scoring.score_attack(b, int(m), game.WHITE) for m in game.legal_moves(b)
    )
    assert best_white > best_black  # the premise of the test
    assert game.choose_swap_color(b, scoring.score_attack) == game.WHITE


def test_total_score_is_colour_symmetric():
    # The total (attack + defence) score of a square is identical
    # for both colours — which is why the swap choice must compare
    # ATTACK scores, not totals (see choose_swap_color's docstring).
    b = _opening()
    for m in game.legal_moves(b):
        assert scoring.score_move(b, int(m), game.BLACK) == scoring.score_move(
            b, int(m), game.WHITE
        )


def test_choose_swap_color_rejects_non_opening():
    with pytest.raises(ValueError):
        game.choose_swap_color(game.empty_board(), scoring.score_attack)
    b = game.apply_move(game.empty_board(), 0, game.BLACK)
    with pytest.raises(ValueError):
        game.choose_swap_color(b, scoring.score_attack)

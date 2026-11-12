"""Unit tests for game rules (beyond the shared parity vectors)."""

import numpy as np
import pytest

from gomoku_ml import game


def test_empty_board():
    b = game.empty_board()
    assert b.shape == (15, 15)
    assert b.dtype == np.int8
    assert (b == 0).all()
    assert len(game.legal_moves(b)) == 225


def test_apply_move_immutability_and_legality():
    b = game.empty_board()
    b2 = game.apply_move(b, game.xy_to_move(7, 7), game.BLACK)
    assert b[7, 7] == 0  # original untouched
    assert b2[7, 7] == game.BLACK
    with pytest.raises(ValueError):
        game.apply_move(b2, game.xy_to_move(7, 7), game.WHITE)  # occupied
    with pytest.raises(ValueError):
        game.apply_move(b2, 225, game.WHITE)  # off board
    with pytest.raises(ValueError):
        game.apply_move(b2, game.xy_to_move(0, 0), 0)  # bad player


def test_legal_moves_shrink():
    b = game.empty_board()
    b = game.apply_move(b, 0, game.BLACK)
    b = game.apply_move(b, 224, game.WHITE)
    moves = game.legal_moves(b)
    assert len(moves) == 223
    assert 0 not in moves and 224 not in moves


def test_is_winning_move_and_stone_agree():
    # Build an open four for Black, then check the winning fifth move.
    b = game.empty_board()
    for x in (7, 8, 9, 10):
        b = game.apply_move(b, game.xy_to_move(x, 7), game.BLACK)
    assert game.is_winning_move(b, game.xy_to_move(6, 7), game.BLACK)
    assert game.is_winning_move(b, game.xy_to_move(11, 7), game.BLACK)
    assert not game.is_winning_move(b, game.xy_to_move(6, 7), game.WHITE)
    assert not game.is_winning_move(b, game.xy_to_move(0, 0), game.BLACK)
    b2 = game.apply_move(b, game.xy_to_move(11, 7), game.BLACK)
    assert game.is_winning_stone(b2, game.xy_to_move(11, 7), game.BLACK)
    assert not game.is_winning_stone(b2, game.xy_to_move(11, 7), game.WHITE)
    assert game.winner(b2) == game.BLACK


def test_xy_round_trip():
    for y in range(15):
        for x in range(15):
            m = game.xy_to_move(x, y)
            assert game.move_to_xy(m) == (x, y)
            assert m == y * 15 + x


def test_other_player():
    assert game.other_player(game.BLACK) == game.WHITE
    assert game.other_player(game.WHITE) == game.BLACK

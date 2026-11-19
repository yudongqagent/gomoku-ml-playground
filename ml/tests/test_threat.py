"""Tests for gomoku_ml.threat (v38 multi-task auxiliary task).

The threat classes (iter-38 §1.1): 0 quiet (no move creates a
four), 1 threatening (some move creates exactly one four), 2
forcing (some move creates an open four, >= 2 winning moves).
Positions are hand-built and verified against the exact winning-
move enumeration the labels are defined by.
"""

import numpy as np

from gomoku_ml import game, threat


def _board(stones):
    """Board from {(y, x): player} stones."""
    board = game.empty_board()
    for (y, x), p in stones.items():
        board[y, x] = p
    return board


def test_empty_board_is_quiet():
    board = game.empty_board()
    assert threat.threat_class(board, game.BLACK) == threat.QUIET
    assert threat.max_winning_moves_after(board, game.BLACK) == 0


def test_single_stone_is_quiet():
    board = _board({(7, 7): game.BLACK})
    assert threat.threat_class(board, game.BLACK) == threat.QUIET


def test_open_three_is_forcing():
    # Black three (7,7)-(7,9), both ends open: (7,6) or (7,10)
    # creates an OPEN four (7,6)-(7,9) with TWO winning moves
    # ((7,5) and (7,10)) — so an open three is a forcing threat.
    board = _board(
        {(7, 7): game.BLACK, (7, 8): game.BLACK, (7, 9): game.BLACK}
    )
    assert threat.threat_class(board, game.BLACK) == threat.FORCING
    assert threat.max_winning_moves_after(board, game.BLACK) == 2


def test_half_open_three_is_threatening():
    # Same three with one end blocked by White: (7,10) creates a
    # four (7,7)-(7,10) with exactly ONE winning move ((7,11) —
    # the other end is White's stone).
    board = _board(
        {
            (7, 6): game.WHITE,
            (7, 7): game.BLACK,
            (7, 8): game.BLACK,
            (7, 9): game.BLACK,
        }
    )
    assert threat.threat_class(board, game.BLACK) == threat.THREATENING
    assert threat.max_winning_moves_after(board, game.BLACK) == 1


def test_cross_is_forcing():
    # Horizontal three (7,7)-(7,9) + vertical three (5,10),(6,10),
    # (8,10): the move (7,10) completes BOTH fours, leaving Black
    # with >= 2 winning moves (an open four).
    board = _board(
        {
            (7, 7): game.BLACK,
            (7, 8): game.BLACK,
            (7, 9): game.BLACK,
            (5, 10): game.BLACK,
            (6, 10): game.BLACK,
            (8, 10): game.BLACK,
        }
    )
    assert threat.threat_class(board, game.BLACK) == threat.FORCING
    assert threat.max_winning_moves_after(board, game.BLACK) >= 2


def test_blocked_three_is_quiet():
    # Black three with both ends blocked by White: no move creates
    # a four (the ends are occupied).
    board = _board(
        {
            (7, 6): game.WHITE,
            (7, 7): game.BLACK,
            (7, 8): game.BLACK,
            (7, 9): game.BLACK,
            (7, 10): game.WHITE,
        }
    )
    assert threat.threat_class(board, game.BLACK) == threat.QUIET


def test_threat_is_perspective_relative():
    # The same open three is forcing for its owner and quiet
    # for the opponent (who holds no four-creator of their own).
    board = _board(
        {(7, 7): game.BLACK, (7, 8): game.BLACK, (7, 9): game.BLACK}
    )
    assert threat.threat_class(board, game.BLACK) == threat.FORCING
    assert threat.threat_class(board, game.WHITE) == threat.QUIET


def test_class_constants():
    assert threat.NUM_CLASSES == 3
    assert threat.CLASS_NAMES == ("quiet", "threatening", "forcing")
    assert (threat.QUIET, threat.THREATENING, threat.FORCING) == (0, 1, 2)

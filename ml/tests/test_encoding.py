"""Unit tests for the 8-plane board encoding (see encoding.py contract)."""

import numpy as np

from gomoku_ml import encoding, game


def _board_with(black, white):
    b = game.empty_board()
    for x, y in black:
        b[y, x] = game.BLACK
    for x, y in white:
        b[y, x] = game.WHITE
    return b


def test_planes_relative_to_side_to_move():
    # Black stones at (7,7),(8,7); White at (7,8). Black to move.
    b = _board_with([(7, 7), (8, 7)], [(7, 8)])
    planes = encoding.encode(b, game.BLACK)
    assert planes.shape == (8, 15, 15)
    assert planes.dtype == np.float32
    assert planes[0, 7, 7] == 1.0 and planes[0, 7, 8] == 1.0  # own = Black
    assert planes[0].sum() == 2.0
    assert planes[1, 8, 7] == 1.0  # opponent = White
    assert planes[1].sum() == 1.0
    assert planes[3].sum() == 225.0  # Black to move -> all ones

    # Same board, White to move: planes 0/1 swap, plane 3 all zeros.
    planes_w = encoding.encode(b, game.WHITE)
    assert planes_w[0].sum() == 1.0 and planes_w[0, 8, 7] == 1.0
    assert planes_w[1].sum() == 2.0
    assert planes_w[3].sum() == 0.0


def test_last_move_plane():
    b = _board_with([(7, 7)], [])
    planes = encoding.encode(b, game.BLACK, last_move=game.xy_to_move(7, 7))
    assert planes[2].sum() == 1.0
    assert planes[2, 7, 7] == 1.0
    planes_none = encoding.encode(b, game.BLACK, last_move=None)
    assert planes_none[2].sum() == 0.0


def test_history_planes():
    # Play: B(7,7), W(0,0), B(8,7), W(1,0), B(9,7). Current board has
    # Black (7,7),(8,7),(9,7), White (0,0),(1,0). White to move.
    boards = []
    b = game.empty_board()
    for move, player in [
        ((7, 7), game.BLACK),
        ((0, 0), game.WHITE),
        ((8, 7), game.BLACK),
        ((1, 0), game.WHITE),
        ((9, 7), game.BLACK),
    ]:
        boards.append(b)
        b = game.apply_move(b, game.xy_to_move(*move), player)
    # history[-1] = board before B(9,7) (1 ply ago); history[-2] = board
    # before W(1,0) (2 plies ago); history[-4] = board before W(0,0).
    history = boards  # 5 snapshots; [-2] is 2 plies ago, [-4] is 4 plies ago
    planes = encoding.encode(b, game.WHITE, history=history, last_move=game.xy_to_move(9, 7))
    # 2 plies ago: Black (7,7),(8,7); White (0,0). Side to move then was
    # White (same as now, even offset) -> own=White, opp=Black.
    assert planes[4].sum() == 1.0 and planes[4, 0, 0] == 1.0  # own (White)
    assert planes[5].sum() == 2.0  # opponent (Black)
    # 4 plies ago: only Black (7,7). Own (White) = 0, opp (Black) = 1.
    assert planes[6].sum() == 0.0
    assert planes[7].sum() == 1.0 and planes[7, 7, 7] == 1.0


def test_history_planes_missing_are_zero():
    b = _board_with([(7, 7)], [(0, 0)])
    planes = encoding.encode(b, game.BLACK, history=[], last_move=None)
    assert planes[4].sum() == 0.0
    assert planes[5].sum() == 0.0
    assert planes[6].sum() == 0.0
    assert planes[7].sum() == 0.0
    # Only 1 snapshot of history: still no 2-ply plane.
    planes = encoding.encode(b, game.BLACK, history=[game.empty_board()])
    assert planes[4].sum() == 0.0


def test_encode_batch():
    b1 = _board_with([(7, 7)], [])
    b2 = _board_with([], [(3, 3)])
    batch = encoding.encode_batch([b1, b2], [game.BLACK, game.WHITE])
    assert batch.shape == (2, 8, 15, 15)
    assert batch[0, 0, 7, 7] == 1.0
    assert batch[1, 0, 3, 3] == 1.0

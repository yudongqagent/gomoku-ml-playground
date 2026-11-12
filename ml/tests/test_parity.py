"""Parity tests: Python rules vs the shared vectors in parity_vectors.json.

These vectors are SHARED with the TypeScript engine tests — the same
cases live on both sides so a rules drift shows up as a failure. Do not
change a vector here without changing it on the TS side too.
"""

import json
import os

import numpy as np

from gomoku_ml import game

HERE = os.path.dirname(os.path.abspath(__file__))


def load_vectors():
    with open(os.path.join(HERE, "parity_vectors.json"), encoding="utf-8") as f:
        return json.load(f)["cases"]


def _winner_name(w):
    return {game.BLACK: "black", game.WHITE: "white", game.EMPTY: "none"}[w]


def test_parity_vectors():
    for case in load_vectors():
        board = game.empty_board()
        if case["kind"] == "sequence":
            player = game.BLACK
            for i, (x, y) in enumerate(case["moves"]):
                move = game.xy_to_move(x, y)
                assert game.is_legal(board, move), f"{case['id']}: move {i} illegal"
                board = game.apply_move(board, move, player)
                over, win = game.game_over(board)
                if i < len(case["moves"]) - 1:
                    assert not over, (
                        f"{case['id']}: game ended early at move {i} "
                        f"(winner={_winner_name(win)})"
                    )
                player = -player
            over, win = game.game_over(board)
            if case["expected_winner"] == "none":
                assert not over, f"{case['id']}: expected game to continue"
            else:
                assert over, f"{case['id']}: expected game over"
                assert _winner_name(win) == case["expected_winner"], (
                    f"{case['id']}: winner { _winner_name(win)} != "
                    f"{case['expected_winner']}"
                )
        elif case["kind"] == "placement":
            for x, y in case["stones"]["black"]:
                board[y, x] = game.BLACK
            for x, y in case["stones"]["white"]:
                board[y, x] = game.WHITE
            assert _winner_name(game.winner(board)) == case["expected_winner"], (
                f"{case['id']}: winner mismatch"
            )
        else:
            raise AssertionError(f"unknown kind {case['kind']}")
        # If a winning line is given, every point on it must hold the
        # winner's stone.
        if case.get("winning_line"):
            winner = game.BLACK if case["expected_winner"] == "black" else game.WHITE
            for x, y in case["winning_line"]:
                assert board[y, x] == winner, (
                    f"{case['id']}: winning line point {(x, y)} is not the winner's"
                )


def _naive_has_five(board: np.ndarray, player: int) -> bool:
    """Independent win check: enumerate every 5-length window in the four
    directions and test for five equal stones. Deliberately written
    differently from game.py so it cross-checks the implementation."""
    for y in range(15):
        for x in range(15):
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = []
                for k in range(5):
                    yy, xx = y + k * dy, x + k * dx
                    if not (0 <= yy < 15 and 0 <= xx < 15):
                        break
                    cells.append(board[yy, xx])
                if len(cells) == 5 and all(c == player for c in cells):
                    return True
    return False


def test_full_board_draw():
    """A full board with no five in a row is a draw.

    Construction (provable, no search needed): colour (x, y) Black iff
    (x + 2y) mod 5 is 0 or 1, else White. Any 5 consecutive cells in any
    of the four directions cover all five residues mod 5 — horizontally
    (x+2y) steps by 1, vertically by 2, diagonally by 3, anti-diagonally
    by -1 — so every 5-window contains both colours and no five in a row
    exists. The naive window checker confirms it independently.
    """
    board = np.zeros((15, 15), dtype=np.int8)
    for y in range(15):
        for x in range(15):
            board[y, x] = game.BLACK if (x + 2 * y) % 5 in (0, 1) else game.WHITE
    assert game.is_full(board)
    assert not _naive_has_five(board, game.BLACK)
    assert not _naive_has_five(board, game.WHITE)
    over, win = game.game_over(board)
    assert over and win == game.EMPTY, "full board with no winner must be a draw"

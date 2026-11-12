"""Pure-Python Gomoku rules.

These rules mirror the TypeScript engine in the web app exactly:

* 15x15 board, Black (1) moves first, White (-1) second.
* A move wins when it creates a line of **five or more** stones of the
  mover's colour (overline counts as a win — no Renju rules).
* No Renju opening bans, no captures, no pass move.
* The game is a draw only when the board is full with no winner.

Board representation: numpy int8 array of shape (15, 15), indexed
``board[y, x]`` — row-major, y = row (0 = top), x = column (0 = left).
Values: 1 = Black, -1 = White, 0 = empty. Flat move indices are
``y * 15 + x``, matching the network policy head.
"""

from __future__ import annotations

import numpy as np

SIZE = 15
EMPTY = 0
BLACK = 1
WHITE = -1

# The four line directions (dy, dx): horizontal, vertical, both diagonals.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def empty_board() -> np.ndarray:
    """Return a fresh empty 15x15 board (int8)."""
    return np.zeros((SIZE, SIZE), dtype=np.int8)


def legal_moves(board: np.ndarray) -> np.ndarray:
    """Return the flat indices (y*15+x) of all empty points, ascending."""
    return np.flatnonzero(board == EMPTY).astype(np.int64)


def is_legal(board: np.ndarray, move: int) -> bool:
    """True if ``move`` (flat index) is on the board and empty."""
    if not 0 <= move < SIZE * SIZE:
        return False
    return board[move // SIZE, move % SIZE] == EMPTY


def apply_move(board: np.ndarray, move: int, player: int) -> np.ndarray:
    """Return a new board with ``player``'s stone placed at ``move``.

    Raises ValueError on an illegal move. The input board is not modified.
    """
    if player not in (BLACK, WHITE):
        raise ValueError(f"player must be BLACK (1) or WHITE (-1), got {player}")
    if not is_legal(board, move):
        raise ValueError(f"illegal move {move}")
    new_board = board.copy()
    new_board[move // SIZE, move % SIZE] = player
    return new_board


def _line_length(board: np.ndarray, y: int, x: int, dy: int, dx: int, player: int) -> int:
    """Length of the contiguous run of ``player`` stones through (y, x)
    along direction (dy, dx), counting (y, x) itself."""
    count = 1
    yy, xx = y + dy, x + dx
    while 0 <= yy < SIZE and 0 <= xx < SIZE and board[yy, xx] == player:
        count += 1
        yy += dy
        xx += dx
    yy, xx = y - dy, x - dx
    while 0 <= yy < SIZE and 0 <= xx < SIZE and board[yy, xx] == player:
        count += 1
        yy -= dy
        xx -= dx
    return count


def is_winning_move(board: np.ndarray, move: int, player: int) -> bool:
    """True if placing ``player``'s stone at ``move`` creates >= 5 in a row.

    ``board`` must not already contain a stone at ``move``.
    """
    y, x = move // SIZE, move % SIZE
    if board[y, x] != EMPTY:
        return False
    return any(
        _line_length_after(board, y, x, dy, dx, player) >= 5
        for dy, dx in _DIRECTIONS
    )


def _line_length_after(board: np.ndarray, y: int, x: int, dy: int, dx: int, player: int) -> int:
    """Run length through (y, x) along (dy, dx) if (y, x) held ``player``."""
    count = 1
    yy, xx = y + dy, x + dx
    while 0 <= yy < SIZE and 0 <= xx < SIZE and board[yy, xx] == player:
        count += 1
        yy += dy
        xx += dx
    yy, xx = y - dy, x - dx
    while 0 <= yy < SIZE and 0 <= xx < SIZE and board[yy, xx] == player:
        count += 1
        yy -= dy
        xx -= dx
    return count


def is_winning_stone(board: np.ndarray, move: int, player: int) -> bool:
    """True if the stone already placed at ``move`` is part of a run of
    >= 5 stones of ``player``'s colour. O(1): only the four lines
    through the stone are examined. Used by MCTS, where the win check
    runs at every tree node and a full-board scan would be too slow."""
    y, x = move // SIZE, move % SIZE
    if board[y, x] != player:
        return False
    return any(
        _line_length(board, y, x, dy, dx, player) >= 5
        for dy, dx in _DIRECTIONS
    )


def winner(board: np.ndarray) -> int:
    """Return BLACK (1) or WHITE (-1) if that player has >= 5 in a row,
    else EMPTY (0). Scans the whole board; used for terminal checks."""
    for y in range(SIZE):
        for x in range(SIZE):
            player = int(board[y, x])
            if player == EMPTY:
                continue
            for dy, dx in _DIRECTIONS:
                # Only count a run from its first stone to avoid double work.
                py, px = y - dy, x - dx
                if 0 <= py < SIZE and 0 <= px < SIZE and board[py, px] == player:
                    continue
                if _line_length(board, y, x, dy, dx, player) >= 5:
                    return player
    return EMPTY


def is_full(board: np.ndarray) -> bool:
    """True if there are no empty points."""
    return not np.any(board == EMPTY)


def game_over(board: np.ndarray) -> tuple[bool, int]:
    """Return ``(over, winner)``: winner is BLACK/WHITE, or EMPTY for a
    draw (board full, no winner). If the game is not over, returns
    ``(False, 0)``."""
    w = winner(board)
    if w != EMPTY:
        return True, w
    if is_full(board):
        return True, EMPTY
    return False, EMPTY


def other_player(player: int) -> int:
    """The opponent of ``player``."""
    return -player


def move_to_xy(move: int) -> tuple[int, int]:
    """Flat index -> (x, y)."""
    return move % SIZE, move // SIZE


def xy_to_move(x: int, y: int) -> int:
    """(x, y) -> flat index."""
    return y * SIZE + x

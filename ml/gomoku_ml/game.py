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


# --- swap rule (pie rule) -------------------------------------------------
# Standard swap rule (NOT Swap2 — see research/iterations/swap-rule.md
# for the interpretation and the self-play policies):
#
# * the opener places 3 stones on the empty board: 2 black, 1 white;
# * the second player then chooses which colour to play;
# * play continues with White to move (move 4 is White's — the 3
#   opening stones ARE moves 1-3, black/white/black, so the game
#   simply continues alternating; RIF rules Article 2.4).
#
# These helpers are purely additive: no existing function changes
# behaviour. The self-play policies (how the opener places the 3
# stones, how the second player picks a colour) live in
# ``gomoku_ml.research`` and are documented in
# ``research/iterations/swap-rule.md``.

SWAP_OPENING_STONES = 3
SWAP_OPENING_BLACK = 2
SWAP_OPENING_WHITE = 1


def build_swap_opening(black1: int, black2: int, white1: int) -> np.ndarray:
    """Build the board after the swap-rule opening: the opener's 2
    black stones and 1 white stone, on distinct legal points.

    Raises ValueError if the three points are not distinct or not on
    the board. The order of the two black stones does not matter.
    """
    moves = (black1, black2, white1)
    if len(set(moves)) != 3:
        raise ValueError(
            f"swap opening stones must be distinct, got {moves}"
        )
    board = empty_board()
    for move in moves:
        if not 0 <= move < SIZE * SIZE:
            raise ValueError(f"swap opening move out of range: {move}")
    board[black1 // SIZE, black1 % SIZE] = BLACK
    board[black2 // SIZE, black2 % SIZE] = BLACK
    board[white1 // SIZE, white1 % SIZE] = WHITE
    return board


def is_swap_opening(board: np.ndarray) -> bool:
    """True if ``board`` holds exactly a swap opening: 2 black stones
    and 1 white stone, and the game is not already over."""
    n_black = int(np.count_nonzero(board == BLACK))
    n_white = int(np.count_nonzero(board == WHITE))
    if (n_black, n_white) != (SWAP_OPENING_BLACK, SWAP_OPENING_WHITE):
        return False
    over, _ = game_over(board)
    return not over


def choose_swap_color(board: np.ndarray, score_move_fn) -> int:
    """The second player's colour choice on a swap-opening board.

    Policy: compare the best available ATTACK score for each colour
    (max over legal moves of ``score_move_fn(board, move, colour)``)
    and take the colour with the higher best score. Ties go to WHITE:
    White plays move 4 (see below), so the next-move tempo breaks
    the tie. ``score_move_fn`` is injected so this module stays free
    of engine dependencies — the research harness passes the v1
    scoring engine's attack-only scorer (``gomoku_ml.scoring.score_attack``)
    by default.

    Turn order (verified against the official RIF rules, Article 2.4
    — "After this, the player with white stones will place a stone"):
    the 3 opening stones ARE moves 1-3 (black, white, black), so the
    game continues with White to move (move 4). An earlier version
    of this program wrongly had Black to move; that gave Black 3 of
    the first 4 moves and made the swap choice degenerate. See
    ``research/iterations/swap-rule.md``.

    The scorer MUST be colour-asymmetric (attack-only). A symmetric
    attack+defence scorer makes the comparison vacuous: at any square
    the attack and defence components swap roles between colours, so
    both colours score identically everywhere and the choice always
    falls to the BLACK tie-break (verified empirically — see
    ``research/iterations/swap-rule.md``).

    Raises ValueError if ``board`` is not a swap opening (see
    :func:`is_swap_opening`).
    """
    if not is_swap_opening(board):
        raise ValueError(
            "choose_swap_color needs a swap-opening board "
            "(2 black stones, 1 white stone)"
        )
    legal = legal_moves(board)
    best_black = max(int(score_move_fn(board, int(m), BLACK)) for m in legal)
    best_white = max(int(score_move_fn(board, int(m), WHITE)) for m in legal)
    return WHITE if best_white >= best_black else BLACK


def other_player(player: int) -> int:
    """The opponent of ``player``."""
    return -player


def move_to_xy(move: int) -> tuple[int, int]:
    """Flat index -> (x, y)."""
    return move % SIZE, move // SIZE


def xy_to_move(x: int, y: int) -> int:
    """(x, y) -> flat index."""
    return y * SIZE + x

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


def choose_swap_color(board: np.ndarray, score_move_fn, judge: str = "attack") -> int:
    """The second player's colour choice on a swap-opening board.

    ``judge="attack"`` (default, batch-1 policy): compare the best
    available ATTACK score for each colour (max over legal moves of
    ``score_move_fn(board, move, colour)``) and take the colour with
    the higher best score. Ties go to WHITE: White plays move 4 (see
    below), so the next-move tempo breaks the tie.

    ``judge="tempo"`` (batch 2, H3): the tempo-aware judge — see
    :func:`choose_swap_color_tempo`. ``score_move_fn`` is injected so
    this module stays free of engine dependencies — the research
    harness passes the v1 scoring engine's attack-only scorer
    (``gomoku_ml.scoring.score_attack``) by default.

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
    :func:`is_swap_opening`) or if ``judge`` is unknown.
    """
    if not is_swap_opening(board):
        raise ValueError(
            "choose_swap_color needs a swap-opening board "
            "(2 black stones, 1 white stone)"
        )
    if judge == "attack":
        legal = legal_moves(board)
        best_black = max(int(score_move_fn(board, int(m), BLACK)) for m in legal)
        best_white = max(int(score_move_fn(board, int(m), WHITE)) for m in legal)
        return WHITE if best_white >= best_black else BLACK
    if judge == "tempo":
        return choose_swap_color_tempo(board, score_move_fn)
    raise ValueError(f"unknown swap judge: {judge!r} (known: 'attack', 'tempo')")


def _setup_value(board: np.ndarray, score_move_fn, player: int) -> int:
    """Best follow-up attack for ``player`` after its own best move.

    Among ``player``'s best immediate attack moves on ``board``
    (max of ``score_move_fn``), take the one with the best follow-up
    (max attack for ``player`` on the board after the setup move)
    and return that follow-up value. The max-over-ties makes the
    value independent of which tied setup move is picked (the
    follow-up varies by at most 1 point across tied setups — see
    ``research/iterations/swap-judge-v2.md``).
    """
    legal = legal_moves(board)
    scored = [(int(score_move_fn(board, int(m), player)), int(m)) for m in legal]
    top = max(s for s, _ in scored)
    best_followup = None
    for score, move in scored:
        if score != top:
            continue
        after = apply_move(board, move, player)
        followup = max(
            int(score_move_fn(after, int(m2), player))
            for m2 in legal_moves(after)
        )
        if best_followup is None or followup > best_followup:
            best_followup = followup
    assert best_followup is not None
    return best_followup


def choose_swap_color_tempo(board: np.ndarray, score_move_fn) -> int:
    """Tempo-aware colour choice on a swap-opening board (batch 2, H3).

    Each colour's value is its best follow-up attack AFTER its own
    best setup move (see :func:`_setup_value`): the colours are
    compared after symmetric 1-ply development, as in the real game
    where White plays move 4 and Black move 5 before the race
    continues. The colour with the higher value wins; ties go to
    WHITE, who moves next (move 4) — the tempo breaks the tie.

    Why symmetric setup, not the naive 1-ply lookahead (White's best
    move 4, then compare best attacks)? The naive version is
    degenerate in the opposite direction: White's setup move unlocks
    a gapped three-in-a-window worth ~4006 while Black's
    non-interacting stones cap at ~164, so it picks White 100/100
    (measured on 50 balanced openings). Any SEQUENTIAL lookahead has
    this last-move-wins artifact — a 2-ply minimax version picks
    Black 20/20 by ~3900. Only the symmetric comparison produces a
    mixed split (Black 76 / White 24 on 100 balanced openings, with
    margins of exactly 0 or 78). A tempo handicap on the plain
    attack gap cannot mix the split either: the gap is 77-78 in
    every balanced opening, so any fixed handicap picks one colour
    always. Full design analysis and numbers:
    ``research/iterations/swap-judge-v2.md``.

    Raises ValueError if ``board`` is not a swap opening.
    """
    if not is_swap_opening(board):
        raise ValueError(
            "choose_swap_color_tempo needs a swap-opening board "
            "(2 black stones, 1 white stone)"
        )
    value_black = _setup_value(board, score_move_fn, BLACK)
    value_white = _setup_value(board, score_move_fn, WHITE)
    return WHITE if value_white >= value_black else BLACK


def other_player(player: int) -> int:
    """The opponent of ``player``."""
    return -player


def move_to_xy(move: int) -> tuple[int, int]:
    """Flat index -> (x, y)."""
    return move % SIZE, move // SIZE


def xy_to_move(x: int, y: int) -> int:
    """(x, y) -> flat index."""
    return y * SIZE + x

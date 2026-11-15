"""Vectorized NumPy implementation of the v1 5-window ATTACK scorer.

This is an exact, bit-for-bit port of :func:`gomoku_ml.scoring.score_attack`
applied to every cell of the board at once: for each of the 225 cells,
the attack-only score for the mover (sum over the 4 directions of the
strongest 5-window through the cell, plus the double-threat bonuses),
with NO defence component and NO centre term.

Unlike :mod:`gomoku_ml.kernel_numpy` (which ports v0's contiguous-run
scoring and is blind to broken fours), this kernel ports v1's 5-window
scan: a move completing ``XX_XX`` scores as a four (20k), and a move
creating a broken four is valued as the threat it is. It exists for the
v14 MCTS leaf gradient (batch 5, H3; see ``research/iterations/iter-14.md``):
v7's gradient uses the v0 kernel's attack scores, so it undervalues
broken-four setups relative to v1's scoring — a known, specific,
measurable defect in v7's leaf.

Faithfulness notes (v1 quirks are ported, not fixed):

* Scores are computed for ALL 225 cells, including occupied ones,
  exactly as ``score_attack`` would return for that flat index.
  ``score_attack`` never checks legality — the move cell's own content
  IS read (it counts toward the friendly/opponent stone counts in each
  window through the cell), so an occupied cell's score follows the
  same formulas with the cell counted as a stone. The one place the
  move position matters — the open-four check for a window holding
  3 friendly stones, where ``empty_idx = empty_sum - move_idx`` —
  is replicated exactly, including its degenerate behaviour when the
  move cell is occupied (``empty_sum`` then excludes the move cell,
  so ``empty_idx`` is not a real window position; the check still
  compares it against 0 and 4 as the Python code does).
* A window holding 5 friendly stones (only possible when the move
  cell is itself a friendly stone) scores 0: the Python scorer's
  ``own == 6`` falls through every branch of ``_score_from_counts``.
* Max-per-direction, not sum-over-windows: each direction contributes
  only its strongest window, exactly as in ``scoring._score_parts_flat``.
* Double-threat bonuses are derived from the per-direction best
  scores, which determine the window kind uniquely (10M -> five,
  200k/20k -> four, 4k -> open three; no two pattern classes share a
  score, so the Python code's first-max-wins tie order cannot change
  the kind): fours in >= 2 directions add 200k, open threes in >= 2
  directions add 50k.

Vectorization strategy: for each direction, the per-window quantities
are computed on the grid of window starts (a window starting at ``s``
holds cells ``s .. s+4d``): ``p`` = friendly count, ``o`` = opponent
count, ``empty_sum`` = sum of the positions of the empty cells in the
window, ``before``/``after`` = emptiness of the cells just outside the
window — each a sum of shifted views of once-padded masks, so no shift
buffer is allocated. The window score for a move at position ``i``
within the window depends on ``i`` only through the open-four check
(``empty_idx = empty_sum - i``), so the position-independent base
scores are computed once per direction and only the open-four upgrade
is recomputed per ``i``; each per-``i`` array is maxed into the
per-direction best with slice views (a cell ``c`` reads the window
starting at ``c - i*d`` — no padding, no intermediate array).
Invalid window starts (the window would run off-board) score 0, which
never wins a max against a real window (all real scores are >= 0, and
an all-empty window scores >= 1).

``attack_v1_all`` works on a single (15, 15) board; ``attack_v1_both``
scores both colours in one pass (the two colours' friendly counts are
each other's opponent counts, and the empty-window quantities are
colour-independent), which is the call the v14 leaf uses.
"""

from __future__ import annotations

import numpy as np

from . import game
from .scoring import (
    DOUBLE_FOUR_BONUS,
    DOUBLE_OPEN_THREE_BONUS,
    FOUR,
    ONE,
    ONE_OPEN,
    OPEN_FOUR,
    OPEN_THREE,
    OPEN_TWO,
    THREE,
    TWO,
    WIN_SCORE,
)

SIZE = game.SIZE

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

# Maximum ray distance the sweep needs: the `after` cell sits 5 steps
# past a window start, the `before` cell 1 step before it.
_MAX_K = 5
_PAD = _MAX_K


def _pad(mask: np.ndarray) -> np.ndarray:
    """Pad the last two axes with False by _PAD on every side."""
    return np.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(_PAD, _PAD), (_PAD, _PAD)])


def _shifted_view(padded: np.ndarray, dy: int, dx: int, k: int) -> np.ndarray:
    """View ``A`` with ``A[..., y, x] = padded[..., y + _PAD + k*dy,
    x + _PAD + k*dx]`` — i.e. the array shifted by (k*dy, k*dx) with
    off-board reads False/0. No copy: this is a slice of ``padded``."""
    y0 = _PAD + k * dy
    x0 = _PAD + k * dx
    return padded[..., y0 : y0 + SIZE, x0 : x0 + SIZE]


# Valid window starts per direction: start s is valid iff s + 4d is
# on the board (s itself is always on the 15x15 grid).
_ys, _xs = np.mgrid[0:SIZE, 0:SIZE]
_VALID_START = {
    (dy, dx): ((0 <= _ys + 4 * dy) & (_ys + 4 * dy < SIZE)
               & (0 <= _xs + 4 * dx) & (_xs + 4 * dx < SIZE))
    for dy, dx in _DIRECTIONS
}


def _base_window_scores(
    p: np.ndarray,
    o: np.ndarray,
    open_ends: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Window scores on the start grid that do NOT depend on the
    move's position within the window — the vectorized
    :func:`gomoku_ml.scoring._score_from_counts` for the attack side
    (``own = p + 1``, ``opp = o``) with every p == 3 window scored
    as a (closed) four. The per-position open-four upgrade is
    applied by :func:`_open_four_mask` for each move position.
    Windows with an opponent stone, and invalid window starts,
    score 0."""
    out = np.zeros(p.shape, dtype=np.int64)
    live = valid & (o == 0)
    # own == 5 (p == 4): the move completes five. p == 5 (own == 6)
    # scores 0 by the Python fall-through — left as 0 here.
    out[live & (p == 4)] = WIN_SCORE
    # own == 4 (p == 3): a four; the open-four upgrade is per-position.
    out[live & (p == 3)] = FOUR
    # own == 3 (p == 2): open three / three / 0 by open ends.
    three = live & (p == 2)
    out[three & (open_ends == 2)] = OPEN_THREE
    out[three & (open_ends == 1)] = THREE
    # own == 2 (p == 1): open two / two / 0.
    two = live & (p == 1)
    out[two & (open_ends == 2)] = OPEN_TWO
    out[two & (open_ends == 1)] = TWO
    # own == 1 (p == 0): 2 with both ends open, else 1.
    one = live & (p == 0)
    out[one & (open_ends == 2)] = ONE_OPEN
    out[one & (open_ends != 2)] = ONE
    return out


def _open_four_mask(
    p: np.ndarray,
    o: np.ndarray,
    empty_sum: np.ndarray,
    before_empty: np.ndarray,
    after_empty: np.ndarray,
    valid: np.ndarray,
    move_idx: int,
) -> np.ndarray:
    """Start-grid mask of the windows that are OPEN fours for a move
    at position ``move_idx`` within the window: p == 3, no opponent
    stone, and the remaining empty cell (``empty_sum - move_idx``,
    exactly as the Python scorer computes it) at an end of the
    window with the outside cell beyond the far end empty."""
    empty_idx = empty_sum - move_idx
    return (
        valid
        & (o == 0)
        & (p == 3)
        & (((empty_idx == 0) & after_empty) | ((empty_idx == 4) & before_empty))
    )


def _max_shifted_into(best: np.ndarray, w: np.ndarray,
                      dy: int, dx: int, i: int) -> None:
    """``best[c] = max(best[c], w[c - i*d])`` for every cell ``c``,
    in place — the window scores on the start grid, shifted so a
    cell reads the window in which it sits at position ``i``.
    Implemented with slice views (no padding, no intermediate
    array): only the cells whose window start is on the grid
    participate."""
    y0 = max(0, i * dy)
    y1 = min(SIZE, SIZE + i * dy)
    x0 = max(0, i * dx)
    x1 = min(SIZE, SIZE + i * dx)
    np.maximum(
        best[y0:y1, x0:x1],
        w[y0 - i * dy : y1 - i * dy, x0 - i * dx : x1 - i * dx],
        out=best[y0:y1, x0:x1],
    )


def _attack_both_from_masks(
    black: np.ndarray, white: np.ndarray, empty: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Attack-only v1 scores for both colours from padded boolean
    masks. Returns ``(black_attack, white_attack)`` int64 arrays of
    shape (15, 15)."""
    black_total = np.zeros((SIZE, SIZE), dtype=np.int64)
    white_total = np.zeros((SIZE, SIZE), dtype=np.int64)
    # Per-direction best scores, kept for the double-threat bonuses
    # (the kind is a function of the best score — see the module
    # docstring).
    black_best: list[np.ndarray] = []
    white_best: list[np.ndarray] = []
    for dy, dx in _DIRECTIONS:
        # Per-window quantities on the start grid. p_black is the
        # opponent count for White's attack and vice versa.
        p_black = np.zeros((SIZE, SIZE), dtype=np.int64)
        p_white = np.zeros((SIZE, SIZE), dtype=np.int64)
        empty_sum = np.zeros((SIZE, SIZE), dtype=np.int64)
        for j in range(5):
            p_black += _shifted_view(black, dy, dx, j)
            p_white += _shifted_view(white, dy, dx, j)
            if j:
                empty_sum += j * _shifted_view(empty, dy, dx, j)
        before_empty = _shifted_view(empty, dy, dx, -1)
        after_empty = _shifted_view(empty, dy, dx, 5)
        open_ends = before_empty.astype(np.int64) + after_empty.astype(np.int64)
        valid = _VALID_START[(dy, dx)]
        # The base window scores are the same for every move
        # position; only the open-four upgrade depends on it.
        base_black = _base_window_scores(p_black, p_white, open_ends, valid)
        base_white = _base_window_scores(p_white, p_black, open_ends, valid)
        best_black = np.zeros((SIZE, SIZE), dtype=np.int64)
        best_white = np.zeros((SIZE, SIZE), dtype=np.int64)
        for i in range(5):
            # A cell c is at position i in the window starting at
            # c - i*d: shift the start-grid scores by (-i*d) to
            # align them with the cells, maxing into the
            # per-direction best.
            open_black = _open_four_mask(
                p_black, p_white, empty_sum, before_empty, after_empty,
                valid, i,
            )
            open_white = _open_four_mask(
                p_white, p_black, empty_sum, before_empty, after_empty,
                valid, i,
            )
            _max_shifted_into(
                best_black,
                np.where(open_black, OPEN_FOUR, base_black), dy, dx, i,
            )
            _max_shifted_into(
                best_white,
                np.where(open_white, OPEN_FOUR, base_white), dy, dx, i,
            )
        black_best.append(best_black)
        white_best.append(best_white)
        black_total += best_black
        white_total += best_white
    # Double-threat bonuses: fours (incl. fives) in >= 2 directions,
    # open threes in >= 2 directions.
    for best, total in ((black_best, black_total), (white_best, white_total)):
        fours = np.zeros((SIZE, SIZE), dtype=np.int64)
        open_threes = np.zeros((SIZE, SIZE), dtype=np.int64)
        for b in best:
            fours += (b == FOUR) | (b == OPEN_FOUR) | (b == WIN_SCORE)
            open_threes += b == OPEN_THREE
        total[fours >= 2] += DOUBLE_FOUR_BONUS
        total[open_threes >= 2] += DOUBLE_OPEN_THREE_BONUS
    return black_total, white_total


def attack_v1_all(board: np.ndarray, player: int) -> np.ndarray:
    """Attack-only v1 scores for all 225 cells of a single (15, 15)
    board for ``player``: ``out[y, x]`` equals
    ``scoring.score_attack(board, y * 15 + x, player)`` exactly,
    including on occupied cells (see the module docstring)."""
    board = np.asarray(board)
    if board.shape != (SIZE, SIZE):
        raise ValueError(f"expected board of shape {(SIZE, SIZE)}, got {board.shape}")
    if player not in (game.BLACK, game.WHITE):
        raise ValueError(f"player must be BLACK (1) or WHITE (-1), got {player}")
    black, white = attack_v1_both(board)
    return black if player == game.BLACK else white


def attack_v1_both(board: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Attack-only v1 scores for BOTH colours on a single (15, 15)
    board, in one pass (the empty-window quantities are shared; the
    two colours' friendly counts are each other's opponent counts).

    Returns ``(black_attack, white_attack)`` — two int64 arrays of
    shape (15, 15), each equal to ``attack_v1_all(board, colour)``.
    This is the call the v14 leaf uses: it needs both sides' threat
    breadth, and one call costs about the same as one
    :func:`attack_v1_all`, not two.
    """
    board = np.asarray(board)
    if board.shape != (SIZE, SIZE):
        raise ValueError(f"expected board of shape {(SIZE, SIZE)}, got {board.shape}")
    return _attack_both_from_masks(
        _pad(board == game.BLACK),
        _pad(board == game.WHITE),
        _pad(board == game.EMPTY),
    )

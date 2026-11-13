"""Vectorized NumPy implementation of the v0 scoring kernel.

This is an exact, bit-for-bit port of :func:`gomoku_ml.baseline.score_move`
applied to every cell of the board at once: for each of the 225 cells,
sum over the 4 directions of ``_WEIGHTS[(min(run, 5), open_ends)]`` for the
mover plus ``int(0.9 * _WEIGHTS[...])`` for the opponent at the same point,
plus the centre preference ``7 - (|x-7| + |y-7|)``.

Faithfulness notes (v0 quirks are ported, not fixed):

* Runs are *contiguous* only: a broken four such as ``XX_XX`` scores as two
  separate runs, exactly as in ``baseline.py``. A v1 kernel may change this;
  this module must match v0 on every cell.
* Scores are computed for ALL 225 cells, including occupied ones, exactly as
  ``score_move`` would return for that flat index (``score_move`` never
  checks legality — the run starts at 1 unconditionally and the cell's own
  content is never read). Callers that only care about legal moves should
  mask with ``board == 0``.
* The opponent weight table is built with the same ``int(0.9 * w)`` float
  arithmetic as ``baseline.score_move`` (float64), so the 90% defence
  factor truncates identically (e.g. weight 1 -> 0, weight 10 -> 9).
  The lookup tables are derived from ``baseline._WEIGHTS`` at import time,
  so a future weight change propagates here automatically.

Vectorization strategy: for each direction and each colour, the run length
and open-end count at every cell are computed with shifted boolean masks.
``run`` at a cell is the number of consecutive friendly stones in both
directions from the cell (plus 1 for the played stone); ``open ends`` counts
the empty points immediately beyond each end of the run. Both are built in
a single sweep over distances k = 1..4, tracking ``cur`` = "all stones at
distances 1..k-1 along this ray are friendly". Off-board shifts read False
for both the friendly and the empty mask, matching the bounds checks in
``baseline._run_and_open_ends``.

Why k stops at 4: the weight table only depends on ``min(run, 5)`` and
``open_ends``, and the weight for run 5 is 100000 for every end count.
A run needs at most 4 stones on one side to reach 5 (1 + 4 + 0), so
per-side runs are capped at 4 without changing ``min(run, 5)``; and an
open end beyond distance 4 on a side means that side's run is >= 4,
hence the total run is >= 5 and the end count cannot affect the weight.
So ends only need to be exact when the total run is < 5, which is
exactly when every end lies within distance 4. Shifts are slice views
into a once-padded mask, so the sweep allocates no shift buffers.

``score_all`` works on a single (15, 15) board; ``score_batch`` stacks N
boards and scores them all with the same vectorized ops (leading batch
dimension), which is the throughput path self-play wants.
"""

from __future__ import annotations

import numpy as np

from . import game
from .baseline import _WEIGHTS

SIZE = game.SIZE

# Lookup tables indexed by [min(run, 5), open_ends]. Row 0 is unused
# (run is always >= 1) but kept so indexing is direct.
_W: np.ndarray = np.zeros((6, 3), dtype=np.int64)
_W_OPP: np.ndarray = np.zeros((6, 3), dtype=np.int64)
for (run, ends), w in _WEIGHTS.items():
    _W[run, ends] = w
    _W_OPP[run, ends] = int(0.9 * w)

# Centre preference 7 - (|x-7| + |y-7|), shape (15, 15).
_ys, _xs = np.mgrid[0:SIZE, 0:SIZE]
_CENTRE: np.ndarray = 7 - (np.abs(_xs - 7) + np.abs(_ys - 7))

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


# Maximum ray distance the sweep needs (see module docstring).
_MAX_K = 4
_PAD = _MAX_K


def _pad(mask: np.ndarray) -> np.ndarray:
    """Pad the last two axes with False by _PAD on every side."""
    return np.pad(mask, [(0, 0)] * (mask.ndim - 2) + [(_PAD, _PAD), (_PAD, _PAD)])


def _shifted_view(padded: np.ndarray, dy: int, dx: int, k: int) -> np.ndarray:
    """View ``A`` with ``A[..., y, x] = padded[..., y + _PAD + k*dy,
    x + _PAD + k*dx]`` — i.e. the mask shifted by (k*dy, k*dx) with
    off-board reads False. No copy: this is a slice of ``padded``."""
    y0 = _PAD + k * dy
    x0 = _PAD + k * dx
    return padded[..., y0 : y0 + SIZE, x0 : x0 + SIZE]


def _run_and_ends(
    friendly: np.ndarray, empty: np.ndarray, dy: int, dx: int
) -> tuple[np.ndarray, np.ndarray]:
    """For every cell, the run length and open-end count along direction
    (dy, dx) if a friendly stone were played there.

    ``friendly`` / ``empty`` are boolean arrays of shape (..., 15, 15),
    already padded by :func:`_pad`. Returns ``(run, ends)`` int arrays of
    shape (..., 15, 15). Run counts the played stone itself (so run >= 1
    everywhere) and is capped per side at 4, which leaves ``min(run, 5)``
    exact; ``ends`` is exact whenever the total run is < 5, which is the
    only case where it affects the weight (see module docstring)."""
    shape = friendly.shape[:-2] + (SIZE, SIZE)
    run = np.ones(shape, dtype=np.int64)
    ends = np.zeros(shape, dtype=np.int64)
    for sgn in (1, -1):
        cur = np.ones(shape, dtype=bool)
        for k in range(1, _MAX_K + 1):
            shifted_friendly = _shifted_view(friendly, sgn * dy, sgn * dx, k)
            shifted_empty = _shifted_view(empty, sgn * dy, sgn * dx, k)
            # The first non-friendly cell along this ray is at distance k
            # exactly when all closer cells are friendly (cur); it is an
            # open end iff it is empty. Exactly one k can satisfy this.
            ends += cur & shifted_empty
            run += cur & shifted_friendly
            cur &= shifted_friendly
    return run, ends


def _score_boards(boards: np.ndarray, player: int) -> np.ndarray:
    """Core kernel. ``boards`` is (..., 15, 15) int8; returns int64 scores
    of the same shape for ``player`` to move."""
    if player not in (game.BLACK, game.WHITE):
        raise ValueError(f"player must be BLACK (1) or WHITE (-1), got {player}")
    friendly = _pad(boards == player)
    opponent = _pad(boards == -player)
    empty = _pad(boards == game.EMPTY)
    total = np.zeros(boards.shape, dtype=np.int64)
    for dy, dx in _DIRECTIONS:
        run, ends = _run_and_ends(friendly, empty, dy, dx)
        total += _W[np.minimum(run, 5), ends]
        run_o, ends_o = _run_and_ends(opponent, empty, dy, dx)
        total += _W_OPP[np.minimum(run_o, 5), ends_o]
    total += _CENTRE
    return total


def score_all(board: np.ndarray, player: int) -> np.ndarray:
    """Score all 225 cells of a single (15, 15) board for ``player``.

    Returns an int64 array of shape (15, 15) where ``out[y, x]`` equals
    ``baseline.score_move(board, y * 15 + x, player)`` exactly, including
    on occupied cells (see module docstring).
    """
    board = np.asarray(board)
    if board.shape != (SIZE, SIZE):
        raise ValueError(f"expected board of shape {(SIZE, SIZE)}, got {board.shape}")
    return _score_boards(board, player)


def score_batch(boards: np.ndarray, player: int) -> np.ndarray:
    """Score a batch of boards of shape (N, 15, 15) for ``player``.

    Returns int64 scores of shape (N, 15, 15); row ``i`` equals
    ``score_all(boards[i], player)`` exactly.
    """
    boards = np.asarray(boards)
    if boards.ndim != 3 or boards.shape[1:] != (SIZE, SIZE):
        raise ValueError(
            f"expected boards of shape (N, {SIZE}, {SIZE}), got {boards.shape}"
        )
    return _score_boards(boards, player)

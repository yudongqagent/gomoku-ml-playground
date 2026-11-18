"""Threat classification for the v38 multi-task value net (ML batch 3 H3).

The auxiliary task (pre-registered in
``research/iterations/iter-38.md`` §1.1): given a QUIET position +
side to move, classify the forcing threat the side to move holds,
by exact 1-ply enumeration:

* class 2 "forcing": some move leaves the side to move with >= 2
  winning moves (an open four — unstoppable);
* class 1 "threatening": some move leaves exactly 1 winning move
  (a single four — forces a block);
* class 0 "quiet": no move leaves a winning move.

The labels are computable from the board alone via the existing,
tested pattern functions (``scoring._candidate_moves`` /
``scoring._winning_moves``) — no teacher, no rollout. The classes
are defined on quiet positions (the net is only ever consulted in
quiet positions at inference, and the training data is quiet-only),
but the function itself is well-defined on any position: on a
non-quiet position where the side to move already holds winning
moves, M counts those too (a move that keeps them is still a move
that leaves a winning move).

Pre-registered limitation (iter-38 §1.1): a move creating two open
threes creates no four in one move, so it is class 0/1 here — the
auxiliary task teaches FOUR-based threat structure; double-threat
setups must still be learned from the value labels.
"""

from __future__ import annotations

import numpy as np

from . import game, scoring

# Class ids (see the module docstring).
QUIET = 0
THREATENING = 1
FORCING = 2
NUM_CLASSES = 3
CLASS_NAMES = ("quiet", "threatening", "forcing")


def max_winning_moves_after(board: np.ndarray, player: int) -> int:
    """Max over candidate moves of the winning moves ``player`` holds
    after playing that move (0 when no candidate creates a four)."""
    best = 0
    for m in scoring._candidate_moves(board):
        after = game.apply_move(board, m, player)
        n = len(scoring._winning_moves(after, player))
        if n > best:
            best = n
            if best >= 2:
                return best  # cannot exceed the forcing threshold
    return best


def threat_class(board: np.ndarray, player: int) -> int:
    """Threat class of the position for ``player`` to move (0/1/2)."""
    m = max_winning_moves_after(board, player)
    if m >= 2:
        return FORCING
    if m == 1:
        return THREATENING
    return QUIET

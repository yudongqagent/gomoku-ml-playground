"""Board + history -> network input tensor.

Contract (shared with the TypeScript AI workstream; see ml/README.md):

The network input is a float32 tensor of shape [8, 15, 15], all planes
relative to the player to move at the current position:

* plane 0: current player's stones
* plane 1: opponent's stones
* plane 2: last move, one-hot (all zeros if no move has been played)
* plane 3: side-to-move plane — all 1.0 if Black is to move, else all 0.0
* plane 4: own stones 2 plies ago (relative to the player who was to move
  at that earlier ply)
* plane 5: opponent stones 2 plies ago
* plane 6: own stones 4 plies ago
* plane 7: opponent stones 4 plies ago

History planes are zeros when the requested earlier position does not
exist (game shorter than 2 / 4 plies). History snapshots are full board
states, so planes 4-7 are computed from the board as it was 2 / 4 plies
ago, with "own"/"opponent" judged from the side to move *at that ply* —
which is the same side as the current player for even-ply offsets (the
side to move alternates every ply, so 2 and 4 plies back it is the same
player again).
"""

from __future__ import annotations

import numpy as np

from . import game

NUM_PLANES = 8


def encode(
    board: np.ndarray,
    player_to_move: int,
    history: list[np.ndarray] | None = None,
    last_move: int | None = None,
) -> np.ndarray:
    """Encode a position as an [8, 15, 15] float32 tensor.

    Args:
        board: current board, int8 [15, 15] (1 Black, -1 White, 0 empty).
        player_to_move: BLACK (1) or WHITE (-1).
        history: list of board snapshots, most recent last. ``history[-1]``
            is the board 1 ply ago, ``history[-2]`` 2 plies ago, etc.
            May be shorter than 4 entries or None.
        last_move: flat index of the most recent move, or None.
    """
    planes = np.zeros((NUM_PLANES, game.SIZE, game.SIZE), dtype=np.float32)
    planes[0] = (board == player_to_move).astype(np.float32)
    planes[1] = (board == -player_to_move).astype(np.float32)
    if last_move is not None:
        planes[2, last_move // game.SIZE, last_move % game.SIZE] = 1.0
    if player_to_move == game.BLACK:
        planes[3] = 1.0
    if history:
        # history[-1] = 1 ply ago, history[-2] = 2 plies ago, ...
        # 2 plies ago the side to move was the same as now (even offset).
        if len(history) >= 2:
            past = history[-2]
            planes[4] = (past == player_to_move).astype(np.float32)
            planes[5] = (past == -player_to_move).astype(np.float32)
        if len(history) >= 4:
            past = history[-4]
            planes[6] = (past == player_to_move).astype(np.float32)
            planes[7] = (past == -player_to_move).astype(np.float32)
    return planes


def encode_batch(
    boards: list[np.ndarray],
    players: list[int],
    histories: list[list[np.ndarray] | None] | None = None,
    last_moves: list[int | None] | None = None,
) -> np.ndarray:
    """Encode a batch of positions -> float32 [N, 8, 15, 15]."""
    n = len(boards)
    histories = histories or [None] * n
    last_moves = last_moves or [None] * n
    out = np.zeros((n, NUM_PLANES, game.SIZE, game.SIZE), dtype=np.float32)
    for i in range(n):
        out[i] = encode(boards[i], players[i], histories[i], last_moves[i])
    return out

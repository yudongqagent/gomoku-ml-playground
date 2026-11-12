#!/usr/bin/env python3
"""JSON-lines checkpoint player (for the TypeScript cross-check).

Loads a checkpoint once, then reads one JSON object per line on stdin
and writes one JSON object per line on stdout:

Request (one line)::

    {"board": [225 ints], "to_move": 1|-1,
     "moves": [[x, y], ...],        # optional, full move history in order
     "sims": 100}                   # optional, overrides --sims

``board`` is row-major (y*15+x), 1 = Black, -1 = White, 0 = empty.
``moves`` is used only to reconstruct the history planes (positions
2 and 4 plies ago); the current position comes from ``board``.

Response (one line)::

    {"move": [x, y], "win_rate": 0.62, "visits": 87}

``move`` is the most-visited MCTS move (temperature 0, no Dirichlet
noise — deterministic). ``win_rate`` is the root value estimate for
the side to move, mapped from [-1, 1] to [0, 1]. ``visits`` is the
chosen move's MCTS visit count. On a bad request the response is
``{"error": "..."}`` and the player keeps serving.

On startup (after the model is loaded) the player prints one line::

    {"event": "ready", "checkpoint": "<path>"}

so the driver knows it can start sending positions. Human-readable
log lines go to stderr; stdout is JSON-lines only.

Used by scripts/play_match.mjs (the TS cross-check driver). Also
handy for manual probing:

    echo '{"board": [0, ... 225 zeros ...], "to_move": 1, "sims": 10}' \\
      | ml/.venv/bin/python ml/scripts/play_position.py \\
          --checkpoint ml/runs/<id>/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import game  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402
from gomoku_ml.mcts import MCTS  # noqa: E402


def _replay_history(moves: list[list[int]]) -> list[np.ndarray]:
    """Replay a move list from the empty board, returning a board
    snapshot after each move (most recent last)."""
    board = game.empty_board()
    snapshots: list[np.ndarray] = []
    player = game.BLACK
    for x, y in moves:
        board = game.apply_move(board, game.xy_to_move(int(x), int(y)), player)
        snapshots.append(board.copy())
        player = -player
    return snapshots


def serve(mcts: MCTS, default_sims: int) -> None:
    """Serve JSON-lines requests from stdin until EOF."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            board_flat = req["board"]
            to_move = int(req["to_move"])
            if len(board_flat) != game.SIZE * game.SIZE:
                raise ValueError(
                    f"board must have {game.SIZE * game.SIZE} entries, "
                    f"got {len(board_flat)}"
                )
            if to_move not in (game.BLACK, game.WHITE):
                raise ValueError(f"to_move must be 1 or -1, got {to_move}")
            board = np.array(board_flat, dtype=np.int8).reshape(
                game.SIZE, game.SIZE
            )
            sims = int(req.get("sims", default_sims))
            if sims < 1:
                raise ValueError(f"sims must be >= 1, got {sims}")
            moves = req.get("moves") or []
            snapshots = _replay_history(moves)
            # history[-1] is the board 1 ply ago: all snapshots except
            # the last (which is the current position).
            history = snapshots[:-1] if len(snapshots) >= 2 else []
            last_move = None
            if moves:
                last_move = game.xy_to_move(int(moves[-1][0]), int(moves[-1][1]))
            legal = game.legal_moves(board)
            if len(legal) == 0:
                print(json.dumps({"move": None, "win_rate": 0.5, "visits": 0}),
                      flush=True)
                continue
            root, root_value = mcts.search(
                board, to_move, sims, history=history,
                last_move=last_move, add_noise=False,
            )
            move = MCTS.choose_move(root, temperature=0)
            visits = root.children[move].visit_count if move in root.children else 0
            x, y = game.move_to_xy(move)
            print(json.dumps({
                "move": [x, y],
                "win_rate": round((float(root_value) + 1.0) / 2.0, 4),
                "visits": visits,
            }), flush=True)
        except Exception as e:  # keep serving after a bad request
            print(json.dumps({"error": str(e)}), flush=True)
            print(f"play_position: {e}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt")
    parser.add_argument("--sims", type=int, default=100,
                        help="default MCTS simulations per move")
    parser.add_argument("--device", default="cpu", help="torch device")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"play_position: checkpoint not found: {args.checkpoint}",
              file=sys.stderr)
        return 2
    torch.set_num_threads(common.torch_threads(2))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set (e.g. embedded use); harmless
    model, _ = load_model_from_checkpoint(args.checkpoint, device=args.device)
    mcts = MCTS(model, device=args.device)
    print(json.dumps({"event": "ready", "checkpoint": args.checkpoint}),
          flush=True)
    serve(mcts, args.sims)
    return 0


if __name__ == "__main__":
    sys.exit(main())

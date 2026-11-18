#!/usr/bin/env python3
"""Add threat-class labels to v24's distill shards (ML batch 3 H3, v38).

Copies v24's labelled shards (``gomoku-ml1-h1/ml/runs/ml1-h1/data/``)
into this worktree's ``ml/runs/ml3-h3/data/`` and adds a ``threat``
array to every shard, computed from the board reconstructed from
the stored planes (planes 0/1 = own/opponent stones, plane 3 =
side to move; see ``gomoku_ml.encoding``). The value labels are
reused VERBATIM — this script never re-runs the rollout leaf, so
there is zero label drift vs v24 (pre-registered in
``research/iterations/iter-38.md`` §1.4).

Verification built in (also pre-registered, §1.4a): every copied
position must satisfy the quiet filter (``_terminal_leaf_value ==
0.0``) after board reconstruction — a wrong reconstruction would
fail this on many positions. Any position failing the check is
reported and the script exits non-zero.

Resume safe: existing output shards are skipped (the counts in the
summary then cover only the newly written shards — re-run from a
clean output dir for the authoritative totals).

Usage:
    python ml/scripts/label_threat.py \\
        --src /home/hatch/workspace/projects/gomoku-ml1-h1/ml/runs/ml1-h1/data \\
        --out ml/runs/ml3-h3/data
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, mcts_heuristic, threat  # noqa: E402


def board_from_planes(planes: np.ndarray) -> tuple[np.ndarray, int]:
    """Reconstruct (board, player_to_move) from one 8-plane encoding."""
    player = game.BLACK if planes[3, 0, 0] > 0.5 else game.WHITE
    board = np.zeros((game.SIZE, game.SIZE), dtype=np.int8)
    board[planes[0] > 0.5] = player
    board[planes[1] > 0.5] = -player
    return board, player


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True,
                        help="v24 data dir with {v7,v18,random}[-holdout] subdirs")
    parser.add_argument("--out", required=True, help="output data dir")
    parser.add_argument("--limit", type=int, default=0,
                        help="label at most N shards per subdir (0 = all; debugging)")
    args = parser.parse_args()

    t0 = time.time()
    summary: dict = {"sources": {}, "total_positions": 0, "class_counts": [0, 0, 0]}
    n_nonquiet = 0
    for sub in ("v7", "v7-holdout", "v18", "v18-holdout", "random", "random-holdout"):
        src_dir = os.path.join(args.src, sub)
        out_dir = os.path.join(args.out, sub)
        os.makedirs(out_dir, exist_ok=True)
        paths = sorted(glob.glob(os.path.join(src_dir, "*.npz")))
        if args.limit:
            paths = paths[: args.limit]
        counts = [0, 0, 0]
        n_pos = n_games = 0
        for path in paths:
            out_path = os.path.join(out_dir, os.path.basename(path))
            if os.path.exists(out_path):
                continue
            with np.load(path) as z:
                planes = z["planes"]
                policy = z["policy"]
                values = z["value"]
                moves = z["moves"]
                winner = z["winner"]
            labels = np.empty(len(planes), dtype=np.int64)
            for i in range(len(planes)):
                board, player = board_from_planes(planes[i])
                if mcts_heuristic._terminal_leaf_value(board, player) != 0.0:
                    n_nonquiet += 1
                    print(f"NON-QUIET position in {path} index {i} "
                          f"(reconstruction suspect)", file=sys.stderr)
                labels[i] = threat.threat_class(board, player)
            np.savez_compressed(
                out_path,
                planes=planes, policy=policy, value=values,
                moves=moves, winner=winner, threat=labels,
            )
            for c in labels:
                counts[int(c)] += 1
            n_pos += len(planes)
            n_games += 1
        summary["sources"][sub] = {
            "games": n_games, "positions": n_pos, "class_counts": counts,
        }
        summary["total_positions"] += n_pos
        for k in range(3):
            summary["class_counts"][k] += counts[k]
        print(f"  {sub}: {n_games} games, {n_pos} positions, "
              f"classes {counts} ({time.time() - t0:.0f}s)",
              file=sys.stderr, flush=True)
    summary["non_quiet_positions"] = n_nonquiet
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    print("LABEL_JSON " + json.dumps(summary))
    if n_nonquiet:
        print(f"ERROR: {n_nonquiet} non-quiet positions — the board "
              f"reconstruction is wrong; refusing to proceed",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

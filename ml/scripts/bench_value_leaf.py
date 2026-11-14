#!/usr/bin/env python3
"""Measure the v8 learned-leaf cost on CPU: per-leaf net eval and per-move cost.

Runs the value net's leaf evaluation on quiet mid-game positions
(encode + batch=1 forward pass) and times full 100-sim
:meth:`ValueMCTS.choose_move` calls, reporting mean/median/p95 and
the number of net leaf evals per move. Prints a JSON summary plus
human-readable lines.

Usage:
    python ml/scripts/bench_value_leaf.py --checkpoint ml/runs/b3-h2/value-net.pt
    python ml/scripts/bench_value_leaf.py --checkpoint <ckpt> --positions 20 --leaf-samples 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import game, scoring  # noqa: E402
from gomoku_ml.mcts_value import ValueMCTS  # noqa: E402


def random_positions(rng: np.random.Generator, n: int, max_plies: int = 40):
    """N random mid-game positions (board, player). Quiet positions
    only (no winning move for either side) — the leaf net only runs
    on quiet positions, so terminal positions would understate the
    per-leaf cost."""
    positions = []
    while len(positions) < n:
        board = game.empty_board()
        player = game.BLACK
        for _ in range(int(rng.integers(10, max_plies + 1))):
            legal = game.legal_moves(board)
            move = int(rng.choice(legal))
            board = game.apply_move(board, move, player)
            player = -player
        over, _ = game.game_over(board)
        if over:
            continue
        if scoring._winning_moves(board, player):
            continue
        if scoring._winning_moves(board, -player):
            continue
        positions.append((board, player))
    return positions


def _summary(times_ms: list[float]) -> dict:
    times_ms = sorted(times_ms)
    return {
        "n": len(times_ms),
        "mean_ms": round(statistics.mean(times_ms), 3),
        "median_ms": round(statistics.median(times_ms), 3),
        "p95_ms": round(times_ms[int(0.95 * (len(times_ms) - 1))], 3),
        "min_ms": round(times_ms[0], 3),
        "max_ms": round(times_ms[-1], 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="value-net .pt checkpoint")
    parser.add_argument("--positions", type=int, default=20,
                        help="mid-game positions for the per-move timing")
    parser.add_argument("--leaf-samples", type=int, default=200,
                        help="leaf evaluations for the per-leaf timing")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS sims per move (default 100, as in v8)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(common.torch_threads(2))
    torch.set_num_interop_threads(1)
    rng = np.random.default_rng(args.seed)
    mcts = ValueMCTS(checkpoint_path=args.checkpoint, sims=args.sims,
                     rng=np.random.default_rng(args.seed))
    positions = random_positions(rng, max(args.positions, 1))

    # (a) Per-leaf net eval: encode + batch=1 forward pass on quiet
    # positions (cycle through the sampled positions).
    leaf_times: list[float] = []
    for i in range(args.leaf_samples):
        board, player = positions[i % len(positions)]
        t0 = time.perf_counter()
        value = mcts._leaf_value(board, player)
        leaf_times.append((time.perf_counter() - t0) * 1000)
        assert -1.0 <= value <= 1.0
    leaf = _summary(leaf_times)
    print(f"per-leaf net eval ({leaf['n']} samples): mean {leaf['mean_ms']} ms, "
          f"median {leaf['median_ms']} ms, p95 {leaf['p95_ms']} ms, "
          f"range [{leaf['min_ms']}, {leaf['max_ms']}] ms")

    # (b) Per-move cost at N sims, with net-leaf counts.
    move_times: list[float] = []
    leaf_counts: list[int] = []
    for board, player in positions[: args.positions]:
        before = mcts.leaf_net_evals
        t0 = time.perf_counter()
        move = mcts.choose_move(board, player)
        move_times.append((time.perf_counter() - t0) * 1000)
        leaf_counts.append(mcts.leaf_net_evals - before)
        assert game.is_legal(board, move)
    moves = _summary(move_times)
    print(f"per-move cost at {args.sims} sims ({moves['n']} positions): "
          f"mean {moves['mean_ms']} ms, median {moves['median_ms']} ms, "
          f"p95 {moves['p95_ms']} ms")
    print(f"net leaf evals per move: mean "
          f"{statistics.mean(leaf_counts):.1f}, min {min(leaf_counts)}, "
          f"max {max(leaf_counts)}")

    summary = {
        "checkpoint": args.checkpoint,
        "sims": args.sims,
        "torch_threads": torch.get_num_threads(),
        "per_leaf_ms": leaf,
        "per_move_ms": moves,
        "leaf_net_evals_per_move": {
            "mean": round(statistics.mean(leaf_counts), 1),
            "min": min(leaf_counts),
            "max": max(leaf_counts),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

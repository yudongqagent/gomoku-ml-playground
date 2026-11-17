#!/usr/bin/env python3
"""Cost guard for the v29 VCF-at-root check (iter-29 §1.2).

Generates 100 quiet positions (v7 self-play at 30 sims + random
playouts, deterministic seed) and times
``HeuristicMCTS._vcf_root_move`` on each with a v29 engine
(``leaf_mode="rollout2_v1_both"``, ``vcf_root=True``). Reports
median / p90 / max root-check time and the fire rate.

``--engine v32`` (iter-32 §1.4) is the same instrument for v32's
OFFENSIVE-ONLY check on the value engine: positions come from v8
self-play (the distribution v32 actually plays on) and the check
is timed on a ``ValueMCTS`` with ``vcf_root=True`` and
``vcf_defensive=False``.

Usage:
    PYTHONPATH=ml python ml/scripts/bench_vcf_root.py [--seed 777] [--n 100]
    PYTHONPATH=ml python ml/scripts/bench_vcf_root.py --engine v32 [--seed 777] [--n 100]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np


def generate_positions(seed: int, n: int, version: int = 7) -> list[tuple[np.ndarray, int]]:
    from gomoku_ml import game, scoring

    rng = np.random.default_rng(seed)
    positions: list[tuple[np.ndarray, int]] = []

    # v7 (or v8, for --engine v32) self-play positions: games at
    # 30 sims, every ply.
    g = 0
    while len(positions) < n:
        pa = scoring.get_player(version, seed=seed + g)
        pb = scoring.get_player(version, seed=seed + 100 + g)
        board = game.empty_board()
        player = game.BLACK
        while len(positions) < n:
            over, _win = game.game_over(board)
            if over:
                break
            positions.append((board.copy(), player))
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        g += 1
    # Top up with random-playout positions if self-play games were short.
    while len(positions) < n:
        board = game.empty_board()
        player = game.BLACK
        ply = 0
        while len(positions) < n:
            over, _win = game.game_over(board)
            if over:
                break
            if ply % 2 == 0:
                positions.append((board.copy(), player))
            empties = np.flatnonzero(board.reshape(-1) == game.EMPTY)
            board = game.apply_move(board, int(rng.choice(empties)), player)
            player = -player
            ply += 1
    return positions[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--engine", choices=("v29", "v32"), default="v29",
                    help="v29: heuristic engine, v7 positions "
                    "(default); v32: value engine, v8 positions, "
                    "offensive-only check")
    args = ap.parse_args()

    if args.engine == "v32":
        import os

        from gomoku_ml.mcts_value import ValueMCTS

        positions = generate_positions(args.seed, args.n, version=8)
        print(f"generated {len(positions)} positions (seed {args.seed})",
              file=sys.stderr)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        mcts = ValueMCTS(
            checkpoint_path=os.path.join(
                repo_root, "ml", "runs", "b3-h2", "value-net.pt"
            ),
            sims=100, vcf_root=True, vcf_defensive=False,
            rng=np.random.default_rng(0),
        )
        defensive = False
    else:
        from gomoku_ml.mcts_heuristic import HeuristicMCTS

        positions = generate_positions(args.seed, args.n)
        print(f"generated {len(positions)} positions (seed {args.seed})",
              file=sys.stderr)
        mcts = HeuristicMCTS(
            sims=100, leaf_mode="rollout2_v1_both", vcf_root=True,
            rng=np.random.default_rng(0),
        )
        defensive = True
    times: list[float] = []
    fired = 0
    for board, player in positions:
        t0 = time.perf_counter()
        move = mcts._vcf_root_move(board, player, defensive=defensive)
        times.append(time.perf_counter() - t0)
        if move is not None:
            fired += 1
    times_sorted = sorted(times)
    p90 = times_sorted[int(0.9 * len(times_sorted)) - 1]
    print(f"n={len(times)} fired={fired} ({fired / len(times):.0%})")
    print(f"median={statistics.median(times) * 1000:.1f}ms "
          f"p90={p90 * 1000:.1f}ms max={max(times) * 1000:.1f}ms "
          f"mean={statistics.mean(times) * 1000:.1f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

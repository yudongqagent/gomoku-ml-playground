#!/usr/bin/env python3
"""Cost guard for the v29 VCF-at-root check (iter-29 §1.2).

Generates 100 quiet positions (v7 self-play at 30 sims + random
playouts, deterministic seed) and times
``HeuristicMCTS._vcf_root_move`` on each with a v29 engine
(``leaf_mode="rollout2_v1_both"``, ``vcf_root=True``). Reports
median / p90 / max root-check time and the fire rate.

Usage:
    PYTHONPATH=ml python ml/scripts/bench_vcf_root.py [--seed 777] [--n 100]
    PYTHONPATH=ml python ml/scripts/bench_vcf_root.py --engine v33

``--engine v33`` (iter-33 §1.3) times the v33 check path instead:
the offensive step plus the full prevention-pool scan (what
``choose_move`` pays before the search). The pool scan has no
first-success early exit, so v33's check is >= v29's wherever
the gate opens; "filter engages" counts positions with a
non-empty pool (the search runs restricted), "gate open" counts
positions where the opponent holds a VCF start at all (pool
computed, possibly empty).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np


def generate_positions(seed: int, n: int) -> list[tuple[np.ndarray, int]]:
    from gomoku_ml import game, scoring

    rng = np.random.default_rng(seed)
    positions: list[tuple[np.ndarray, int]] = []

    # v7 self-play positions: games at 30 sims, every ply.
    g = 0
    while len(positions) < n:
        pa = scoring.get_player(7, seed=seed + g)
        pb = scoring.get_player(7, seed=seed + 100 + g)
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
    ap.add_argument("--engine", default="v29", choices=("v29", "v33"))
    args = ap.parse_args()

    from gomoku_ml.mcts_heuristic import HeuristicMCTS

    positions = generate_positions(args.seed, args.n)
    print(f"generated {len(positions)} positions (seed {args.seed})",
          file=sys.stderr)
    if args.engine == "v33":
        mcts = HeuristicMCTS(
            sims=100, leaf_mode="rollout2_v1_both",
            vcf_filtered_search=True,
            rng=np.random.default_rng(0),
        )
        times: list[float] = []
        offensive = 0
        gate_open = 0
        engaged = 0
        for board, player in positions:
            t0 = time.perf_counter()
            off = mcts._vcf_offensive_move(board, player)
            if off is not None:
                offensive += 1
            else:
                pool = mcts._vcf_prevention_pool(board, player)
                if pool is not None:
                    gate_open += 1
                    if pool:
                        engaged += 1
            times.append(time.perf_counter() - t0)
        times_sorted = sorted(times)
        p90 = times_sorted[int(0.9 * len(times_sorted)) - 1]
        print(f"n={len(times)} offensive={offensive} "
              f"gate_open={gate_open} filter_engaged={engaged}")
        print(f"median={statistics.median(times) * 1000:.1f}ms "
              f"p90={p90 * 1000:.1f}ms max={max(times) * 1000:.1f}ms "
              f"mean={statistics.mean(times) * 1000:.1f}ms")
        return 0
    mcts = HeuristicMCTS(
        sims=100, leaf_mode="rollout2_v1_both", vcf_root=True,
        rng=np.random.default_rng(0),
    )
    times: list[float] = []
    fired = 0
    for board, player in positions:
        t0 = time.perf_counter()
        move = mcts._vcf_root_move(board, player)
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

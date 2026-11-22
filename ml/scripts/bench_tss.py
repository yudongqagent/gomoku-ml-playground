#!/usr/bin/env python3
"""Cost guard for the v52 threat-space search (iter-52 §1.4).

Times ``mcts_heuristic.tss_winning_move`` on two position sets and
reports the fire rate, the time distribution, and the NODE
distribution (the budget is a node count, so the node distribution
is what sets it):

* quiet positions: 100 v7 self-play positions (seed 777) — the same
  generator as ``bench_vcf_root.py``, for comparability with v29's
  numbers (median 90ms, p90 2.9s, max 36s, mean 1.65s);
* creator-rich positions: 100 midgame plies (20-60) of v27-vs-v27
  self-play (seed 777) — the binding case, since the TSS cost
  concentrates where forcing moves are dense (v29's check cost ~7s
  per move in creator-rich games vs 1.65s on quiet positions).

Runs with a HIGH provisional node budget (100000) so the bench sees
the unconstrained distribution; the production budget is set from
the node percentiles afterwards (see iter-52 §1.4).

Usage:
    PYTHONPATH=ml python ml/scripts/bench_tss.py [--seed 777] [--n 100]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np


def generate_quiet_positions(seed: int, n: int) -> list[tuple[np.ndarray, int]]:
    """v7 self-play positions (same generator as bench_vcf_root.py)."""
    from gomoku_ml import game, scoring

    rng = np.random.default_rng(seed)
    positions: list[tuple[np.ndarray, int]] = []
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


def generate_creator_rich_positions(
    seed: int, n: int
) -> list[tuple[np.ndarray, int]]:
    """Midgame plies (20-60) of v27-vs-v27 self-play — positions
    where forcing moves are dense and the TSS cost concentrates."""
    from gomoku_ml import game, scoring

    positions: list[tuple[np.ndarray, int]] = []
    g = 0
    while len(positions) < n:
        pa = scoring.get_player(27, seed=seed + g)
        pb = scoring.get_player(27, seed=seed + 100 + g)
        board = game.empty_board()
        player = game.BLACK
        ply = 0
        while len(positions) < n:
            over, _win = game.game_over(board)
            if over:
                break
            if 20 <= ply <= 60:
                positions.append((board.copy(), player))
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
            ply += 1
        g += 1
    return positions[:n]


def _pct(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def bench(
    name: str, positions: list[tuple[np.ndarray, int]], node_budget: int
) -> None:
    from gomoku_ml.mcts_heuristic import tss_winning_move

    times: list[float] = []
    nodes: list[int] = []
    fired = 0
    exceeded = 0
    for board, player in positions:
        stats: dict = {}
        t0 = time.perf_counter()
        move = tss_winning_move(
            board, player, node_budget=node_budget, stats=stats
        )
        times.append(time.perf_counter() - t0)
        nodes.append(stats["nodes"])
        if move is not None:
            fired += 1
        if stats["budget_exceeded"]:
            exceeded += 1
    ts = sorted(times)
    ns = sorted(nodes)
    print(f"--- {name} (n={len(times)}, budget={node_budget}) ---")
    print(f"fired={fired} ({fired / len(times):.0%}) "
          f"budget_exceeded={exceeded}")
    print(f"time: median={statistics.median(times) * 1000:.1f}ms "
          f"p90={_pct(ts, 0.9) * 1000:.1f}ms "
          f"p99={_pct(ts, 0.99) * 1000:.1f}ms "
          f"max={max(times) * 1000:.1f}ms "
          f"mean={statistics.mean(times) * 1000:.1f}ms")
    print(f"nodes: median={statistics.median(nodes):.0f} "
          f"p90={_pct(ns, 0.9):.0f} p99={_pct(ns, 0.99):.0f} "
          f"max={max(nodes)} mean={statistics.mean(nodes):.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--node-budget", type=int, default=100000)
    args = ap.parse_args()

    quiet = generate_quiet_positions(args.seed, args.n)
    print(f"generated {len(quiet)} quiet positions (seed {args.seed})",
          file=sys.stderr)
    bench("quiet (v7 self-play)", quiet, args.node_budget)

    rich = generate_creator_rich_positions(args.seed, args.n)
    print(f"generated {len(rich)} creator-rich positions (seed {args.seed})",
          file=sys.stderr)
    bench("creator-rich (v27 midgame)", rich, args.node_budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())

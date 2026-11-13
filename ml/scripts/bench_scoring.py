#!/usr/bin/env python3
"""Benchmark heuristic move scoring throughput and choose_move latency.

Measures, on a fixed deterministic midgame board (24 stones, built by
seeded heuristic self-play so the stone cluster is realistic):

* ``score_all`` — scoring every legal move once: moves/second and
  ms per full pass;
* ``choose_move`` — mean latency of one full move choice (win scan,
  block scan, then scoring), in ms.

Benchmarks ``gomoku_ml.baseline`` always, and ``gomoku_ml.scoring``
too when it is importable, so one run gives a before/after comparison.

Timing method: warmup, then several timed repetitions of a batch;
the reported figure is the best (min) batch rate, which is the least
noisy on a shared VM. Also reports the mean for transparency.

Usage:
    python ml/scripts/bench_scoring.py [--reps N] [--batch N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import baseline, game  # noqa: E402

try:
    from gomoku_ml import scoring  # noqa: F401
except ImportError:
    scoring = None


def midgame_board() -> np.ndarray:
    """Deterministic midgame board: 24 stones from seeded heuristic
    self-play (two HeuristicPlayers with different seeds)."""
    board = game.empty_board()
    black = baseline.HeuristicPlayer(seed=7)
    white = baseline.HeuristicPlayer(seed=8)
    player = game.BLACK
    for _ in range(24):
        chooser = black if player == game.BLACK else white
        board = game.apply_move(board, chooser.choose_move(board, player), player)
        player = -player
    return board


def _time_batch(fn, batch: int) -> float:
    """Seconds to run ``fn`` ``batch`` times (min over the batch loop
    is handled by the caller; this is one batch timing)."""
    start = time.perf_counter()
    for _ in range(batch):
        fn()
    return time.perf_counter() - start


def bench_module(name: str, score_move, choose_move, board, reps: int, batch: int) -> dict:
    legal = [int(m) for m in game.legal_moves(board)]
    player = game.BLACK

    def score_all() -> None:
        for m in legal:
            score_move(board, m, player)

    def choose() -> None:
        choose_move(board, player)

    # Warmup (JIT-free, but warms caches / numpy paths).
    score_all()
    choose()

    score_times = [_time_batch(score_all, batch) for _ in range(reps)]
    choose_times = [_time_batch(choose, batch) for _ in range(reps)]

    best_score = min(score_times)
    mean_score = sum(score_times) / len(score_times)
    best_choose = min(choose_times)
    mean_choose = sum(choose_times) / len(choose_times)

    n_moves = len(legal)
    result = {
        "module": name,
        "legal_moves": n_moves,
        "score_all_best_ms": best_score * 1000 / batch,
        "score_all_mean_ms": mean_score * 1000 / batch,
        "score_moves_per_sec_best": n_moves * batch / best_score,
        "score_moves_per_sec_mean": n_moves * batch / mean_score,
        "choose_move_best_ms": best_choose * 1000 / batch,
        "choose_move_mean_ms": mean_choose * 1000 / batch,
    }
    print(f"[{name}] legal moves on midgame board: {n_moves}")
    print(
        f"[{name}] score_all: "
        f"{result['score_moves_per_sec_best']:.1f} moves/s best "
        f"({result['score_all_best_ms']:.2f} ms/pass), "
        f"{result['score_moves_per_sec_mean']:.1f} moves/s mean "
        f"({result['score_all_mean_ms']:.2f} ms/pass)"
    )
    print(
        f"[{name}] choose_move: {result['choose_move_best_ms']:.2f} ms best, "
        f"{result['choose_move_mean_ms']:.2f} ms mean"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5,
                        help="timed repetitions per measurement (default 5)")
    parser.add_argument("--batch", type=int, default=20,
                        help="calls per timed batch (default 20)")
    parser.add_argument("--modules", nargs="+", default=["baseline", "scoring"],
                        choices=["baseline", "scoring"],
                        help="which engines to benchmark (default: both)")
    args = parser.parse_args()

    board = midgame_board()
    stones = int(np.count_nonzero(board))
    print(f"midgame board: {stones} stones, "
          f"{len(game.legal_moves(board))} legal moves "
          f"(reps={args.reps}, batch={args.batch})")
    if "baseline" in args.modules:
        bench_module("baseline", baseline.score_move, baseline.choose_move,
                     board, args.reps, args.batch)
    if "scoring" in args.modules:
        if scoring is None:
            print("[scoring] module not importable — skipping")
        else:
            bench_module("scoring", scoring.score_move, scoring.choose_move,
                         board, args.reps, args.batch)
    return 0


if __name__ == "__main__":
    sys.exit(main())

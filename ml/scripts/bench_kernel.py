#!/usr/bin/env python3
"""Benchmark the v0 scoring kernel (``gomoku_ml.kernel_numpy``) against the
pure-Python reference (``gomoku_ml.baseline.score_move`` over all 225 cells).

Position set: the 60 fixed fixtures in ``ml/tests/kernel_fixtures.json``
(20 tactical, 3 structured, 37 random midgame boards of 8–80 plies),
cycled in order. Each measurement scores a full 225-cell board per
position; throughput is positions/second, latency is ms/position.

Modes measured:

* ``reference`` — pure-Python ``score_move`` called 225 times per position.
* ``numpy-single`` — ``kernel_numpy.score_all`` on one board at a time.
* ``numpy-batch-N`` — ``kernel_numpy.score_batch`` on batches of N boards
  (N = 8, 32, 128), the throughput path self-play wants.

Timing method (same convention as ``bench_scoring.py``): warmup, then
several timed repetitions; the reported figure is the best (min) rate,
which is the least noisy on a shared VM, with the mean alongside.

Why no C extension: on this VM the NumPy kernel is already several times
faster than the pure-Python reference single-position and an order of
magnitude faster batched (see the numbers this script prints). A compiled
extension would add build complexity for a gain the workload does not
need — revisit only if profiling shows scoring dominating a real
self-play run.

Usage (from the repo root):
    ml/.venv/bin/python ml/scripts/bench_kernel.py [--reps N] [--batch N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import baseline, game  # noqa: E402
from gomoku_ml.kernel_numpy import score_all, score_batch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(HERE, "..", "tests", "kernel_fixtures.json")


def load_fixture_boards() -> list[tuple[np.ndarray, int]]:
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        fixtures = json.load(f)["fixtures"]
    out = []
    for fx in fixtures:
        board = np.array(fx["board"], dtype=np.int8).reshape(game.SIZE, game.SIZE)
        out.append((board, fx["to_move"]))
    return out


def _time_calls(fn, calls: int) -> float:
    start = time.perf_counter()
    for _ in range(calls):
        fn()
    return time.perf_counter() - start


def bench(name: str, fn, positions_per_call: int, reps: int, calls: int) -> dict:
    fn()  # warmup
    times = [_time_calls(fn, calls) for _ in range(reps)]
    best = min(times)
    mean = sum(times) / len(times)
    n_pos = positions_per_call * calls
    result = {
        "name": name,
        "best_ms_per_pos": best * 1000 / n_pos,
        "mean_ms_per_pos": mean * 1000 / n_pos,
        "best_pos_per_sec": n_pos / best,
        "mean_pos_per_sec": n_pos / mean,
    }
    print(
        f"[{name}] {result['best_pos_per_sec']:.1f} pos/s best "
        f"({result['best_ms_per_pos']:.3f} ms/pos), "
        f"{result['mean_pos_per_sec']:.1f} pos/s mean "
        f"({result['mean_ms_per_pos']:.3f} ms/pos)"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5,
                        help="timed repetitions per measurement (default 5)")
    parser.add_argument("--calls", type=int, default=60,
                        help="calls per timed repetition (default 60 = one pass "
                        "over the fixture set for single-position modes)")
    args = parser.parse_args()

    positions = load_fixture_boards()
    n = len(positions)
    stones = [int(np.count_nonzero(b)) for b, _ in positions]
    print(f"position set: {n} fixtures "
          f"(stones per board: min {min(stones)}, max {max(stones)}, "
          f"mean {sum(stones) / n:.1f}); reps={args.reps}, calls={args.calls}")

    # Cycle through the fixtures in order so every mode sees the same mix.
    state = {"i": 0}

    def next_single():
        board, player = positions[state["i"] % n]
        state["i"] += 1
        return board, player

    def reference_once():
        board, player = next_single()
        for m in range(game.SIZE * game.SIZE):
            baseline.score_move(board, m, player)

    def numpy_single_once():
        board, player = next_single()
        score_all(board, player)

    bench("reference (pure Python, 225x score_move)", reference_once, 1,
          args.reps, args.calls)
    bench("numpy-single (score_all)", numpy_single_once, 1,
          args.reps, args.calls)

    for batch_size in (8, 32, 128):
        # Fixed batches built from the fixture set (black to move boards
        # only for batch purity — batch takes one player per call; use
        # the black-to-move subset cycled to fill the batch).
        black_boards = [b for b, p in positions if p == game.BLACK]
        assert black_boards, "no black-to-move fixtures"
        stacked = np.stack(
            [black_boards[i % len(black_boards)] for i in range(batch_size)]
        )

        def batch_once(bs=stacked):
            score_batch(bs, game.BLACK)

        bench(f"numpy-batch-{batch_size} (score_batch)", batch_once, batch_size,
              args.reps, max(1, args.calls // batch_size * 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())

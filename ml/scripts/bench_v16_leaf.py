#!/usr/bin/env python3
"""Step 0 measurements for batch 6 H2 (v16 2-ply rollout leaf).

Measures, on real mid-game boards replayed from the batch-3 v7-vs-v1
match records (NOT quoted from earlier notes):

(a) per-call cost of the v16 rollout leaf
    (``mcts_heuristic._rollout_leaf_value`` — terminal check + up to
    2 rollout plies + the symmetric v1 gradient),
(b) for comparison, v7's gradient leaf
    (``mcts_heuristic._gradient_leaf_value``), the symmetric v1
    gradient (``_attack_gradient_v1``), and the terminal-only leaf,
(c) how often the rollout ends early (a terminal reached within the
    2 rollout plies) vs runs the full 2 plies + gradient — early
    termination is the cheap path,
(d) the distribution of rollout-leaf values on quiet positions (the
    leaf's effective range — it should stay inside [-1, 1] with most
    mass in the capped-gradient band).

Usage:
    python ml/scripts/bench_v16_leaf.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    _attack_gradient_v1,
    _gradient_leaf_value,
    _rollout_leaf_value,
    _terminal_leaf_value,
)

REPO_ROOT = common.REPO_ROOT


def _sample_boards() -> list[tuple[np.ndarray, int, int]]:
    """Replay the first 20 games of batch-v7-vs-v1.json; sample the
    board every 4th ply from ply 8 onward. Returns (board, player
    to move, ply) triples."""
    path = os.path.join(REPO_ROOT, "research", "iterations",
                        "batch-v7-vs-v1.json")
    with open(path, encoding="utf-8") as f:
        batch = json.load(f)
    out = []
    for rec in batch["records"][:20]:
        board = game.empty_board()
        for ply, move in enumerate(rec["moves"]):
            board = game.apply_move(board, move,
                                    game.BLACK if ply % 2 == 0 else game.WHITE)
            if ply >= 8 and ply % 4 == 0:
                player = game.BLACK if ply % 2 == 1 else game.WHITE
                out.append((board.copy(), player, ply + 1))
    return out


def _timeit(fn, boards, reps: int = 3, rounds: int = 50) -> dict:
    """Median per-call ms over ``reps`` rounds of ``rounds`` calls
    each (round-robin over boards). Returns median/min/max ms."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for i in range(rounds):
            board, player, _ply = boards[i % len(boards)]
            fn(board, player)
        times.append((time.perf_counter() - t0) / rounds * 1000)
    return {
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
    }


def main() -> int:
    boards = _sample_boards()
    print(f"sampled {len(boards)} boards from batch-v7-vs-v1.json "
          f"(20 games, every 4th ply from ply 8)")
    stones = [int(np.count_nonzero(b)) for b, _, _ in boards]
    print(f"stones per board: min {min(stones)}, median "
          f"{int(statistics.median(stones))}, max {max(stones)}")

    print("\n== per-call cost (ms; median of 3 rounds x 50 calls) ==")
    results = {
        "(a) v16 _rollout_leaf_value (2-ply + v1 gradient)": _timeit(
            lambda b, p: _rollout_leaf_value(b, p), boards),
        "(b) v7 _gradient_leaf_value": _timeit(
            lambda b, p: _gradient_leaf_value(b, p), boards),
        "(c) symmetric v1 gradient _attack_gradient_v1": _timeit(
            lambda b, p: _attack_gradient_v1(b, p), boards),
        "(d) v6 _terminal_leaf_value": _timeit(
            lambda b, p: _terminal_leaf_value(b, p), boards),
    }
    for name, r in results.items():
        print(f"  {name}: median {r['median_ms']} ms "
              f"(min {r['min_ms']}, max {r['max_ms']})")

    print("\n== rollout-leaf values on quiet positions ==")
    values = []
    quiet = 0
    for board, player, _ply in boards:
        if _terminal_leaf_value(board, player) != 0.0:
            continue
        quiet += 1
        values.append(_rollout_leaf_value(board, player))
    values = np.array(values, dtype=float)
    print(f"  quiet positions: {quiet}/{len(boards)}")
    print(f"  rollout-leaf value: mean {values.mean():.3f}, "
          f"std {values.std():.3f}, min {values.min():.3f}, "
          f"p10 {np.percentile(values, 10):.3f}, "
          f"median {np.median(values):.3f}, "
          f"p90 {np.percentile(values, 90):.3f}, max {values.max():.3f}")
    print(f"  terminal-valued (|v| >= 0.25): "
          f"{float((np.abs(values) >= 0.25).mean()):.2%}")
    print(f"  gradient-band (|v| <= 0.2): "
          f"{float((np.abs(values) <= 0.2).mean()):.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

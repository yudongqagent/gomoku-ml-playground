#!/usr/bin/env python3
"""Step 0 measurements for batch 3 H1 (v7 leaf design).

Measures, on real mid-game boards replayed from the batch-2 v6
match records (NOT quoted from batch-2 notes):

(a) per-call cost of ``scoring._leaf_value`` (the Python static leaf),
(b) per-call cost of ``kernel_numpy.score_all`` (single position),
(c) per-call cost of the new attack-only accessors
    (``attack_all`` / ``attack_both``) and of the full breadth
    gradient built on them,
(d) for comparison, the v6 terminal-only leaf
    (``mcts_heuristic._terminal_leaf_value``),
(e) the DISTRIBUTION of the breadth difference
    (ours - theirs, player to move) on quiet positions from real
    games — this calibrates the tanh squash scale for the v7 leaf
    (a design parameter fixed in the pre-registration, not tuned
    on v7 eval results).

Usage:
    python ml/scripts/bench_v7_leaf.py
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

from gomoku_ml import game, kernel_numpy, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import _terminal_leaf_value  # noqa: E402

REPO_ROOT = common.REPO_ROOT

# Breadth definition (candidate for the v7 leaf; fixed in the
# pre-registration): strong = moves creating a four or an open
# three (v0 attack >= 1000 — a (4,1)/(3,2) pattern in at least one
# direction); developing = moves creating a half-open three or an
# open two (100 <= attack < 1000). Breadth = strong + 0.25 * developing.
_STRONG = 1000
_DEVELOPING = 100
_DEVELOPING_WEIGHT = 0.25


def _breadth(attack: np.ndarray, empty_mask: np.ndarray) -> float:
    s = attack[empty_mask]
    strong = int(np.count_nonzero(s >= _STRONG))
    developing = int(np.count_nonzero((s >= _DEVELOPING) & (s < _STRONG)))
    return strong + _DEVELOPING_WEIGHT * developing


def _gradient(board: np.ndarray, player: int) -> float:
    black, white = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    ours = _breadth(black if player == game.BLACK else white, mask)
    theirs = _breadth(white if player == game.BLACK else black, mask)
    return ours - theirs  # raw difference; squash applied by the leaf


def _sample_boards() -> list[tuple[np.ndarray, int, int]]:
    """Replay the first 20 games of batch-v6-vs-v1.json; sample the
    board every 4th ply from ply 8 onward. Returns (board, player
    to move, ply) triples."""
    path = os.path.join(REPO_ROOT, "research", "iterations",
                        "batch-v6-vs-v1.json")
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
    print(f"sampled {len(boards)} boards from batch-v6-vs-v1.json "
          f"(20 games, every 4th ply from ply 8)")
    stones = [int(np.count_nonzero(b)) for b, _, _ in boards]
    print(f"stones per board: min {min(stones)}, median "
          f"{int(statistics.median(stones))}, max {max(stones)}")

    print("\n== per-call cost (ms; median of 3 rounds x 50 calls) ==")
    results = {
        "(a) scoring._leaf_value": _timeit(
            lambda b, p: scoring._leaf_value(b, p), boards),
        "(b) kernel score_all": _timeit(
            lambda b, p: kernel_numpy.score_all(b, p), boards),
        "(c1) kernel attack_all (one colour)": _timeit(
            lambda b, p: kernel_numpy.attack_all(b, p), boards),
        "(c2) kernel attack_both": _timeit(
            lambda b, p: kernel_numpy.attack_both(b), boards),
        "(c3) full breadth gradient (attack_both + counts)": _timeit(
            _gradient, boards),
        "(d) v6 _terminal_leaf_value": _timeit(
            lambda b, p: _terminal_leaf_value(b, p), boards),
    }
    for name, r in results.items():
        print(f"  {name}: median {r['median_ms']} ms "
              f"(min {r['min_ms']}, max {r['max_ms']})")

    print("\n== breadth-difference distribution (quiet positions only) ==")
    diffs = []
    strong_diffs = []
    quiet = 0
    for board, player, _ply in boards:
        if scoring._winning_moves(board, player):
            continue
        if scoring._winning_moves(board, -player):
            continue
        quiet += 1
        black, white = kernel_numpy.attack_both(board)
        mask = board == game.EMPTY
        ours_a = black if player == game.BLACK else white
        theirs_a = white if player == game.BLACK else black
        diffs.append(_breadth(ours_a, mask) - _breadth(theirs_a, mask))
        strong_diffs.append(
            int(np.count_nonzero(ours_a[mask] >= _STRONG))
            - int(np.count_nonzero(theirs_a[mask] >= _STRONG))
        )
    diffs = np.array(diffs, dtype=float)
    strong_diffs = np.array(strong_diffs, dtype=float)
    print(f"  quiet positions: {quiet}/{len(boards)}")
    for label, arr in (("breadth diff (strong + 0.25*developing)", diffs),
                       ("strong-count diff only", strong_diffs)):
        print(f"  {label}: mean {arr.mean():.2f}, std {arr.std():.2f}, "
              f"min {arr.min():.2f}, p10 {np.percentile(arr, 10):.2f}, "
              f"median {np.median(arr):.2f}, p90 {np.percentile(arr, 90):.2f}, "
              f"max {arr.max():.2f}")
        for t in (1, 2, 4, 8):
            print(f"    |diff| >= {t}: "
                  f"{float((np.abs(arr) >= t).mean()):.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

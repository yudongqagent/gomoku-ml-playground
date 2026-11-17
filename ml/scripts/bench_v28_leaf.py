#!/usr/bin/env python3
"""Step 0 measurements for batch 9 H2 (v28 ours-only Black
demotion).

Measures, on real mid-game boards replayed from the batch-3 v7-vs-v1
match records (the SAME replay source as iter-14 §1.1 / iter-20
§1.1 / iter-22 §1.1: first 20 games — but EVERY ply from ply 8
onward, not every 4th ply, because v28's rule is PARITY-dependent
and the every-4th-ply sample is all White to move; see
research/iterations/iter-28.md §1):

(a) per-call cost of the v28 ours-only gap-demoted gradient
    (``mcts_heuristic._attack_gradient_v1_gap_ours`` — one v1
    kernel call plus one v0 kernel call) and of the full leaf
    (``_gradient_v1_gap_ours_leaf_value``), against v22's
    (``_attack_gradient_v1_gap_black``), v20's
    (``_attack_gradient_v1_gap``), v14's
    (``_attack_gradient_v1``), v7's (``_attack_gradient``) and
    the terminal-only leaf,
(b) the distribution of the OURS-ONLY breadth difference on
    quiet positions (mean, std, percentiles, per parity and
    pooled) — the tanh scale is re-derived from the POOLED std
    by v7's rule (scale ~= 1.8 x std),
(c) how much the ours-only demotion changes the breadth: mean
    gap-three makers per colour per position, and the mean
    absolute breadth change vs v14's, vs v20's, and vs v22's
    breadth.

Usage:
    python ml/scripts/bench_v28_leaf.py
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import game, kernel_numpy, kernel_v1  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    _attack_gradient,
    _attack_gradient_v1,
    _attack_gradient_v1_gap,
    _attack_gradient_v1_gap_black,
    _attack_gradient_v1_gap_ours,
    _breadth_v1,
    _breadth_v1_gap,
    _gradient_v1_gap_ours_leaf_value,
    _terminal_leaf_value,
)

REPO_ROOT = common.REPO_ROOT


def _sample_boards() -> list[tuple[np.ndarray, int, int]]:
    """Replay the first 20 games of batch-v7-vs-v1.json; sample the
    board EVERY ply from ply 8 onward. Returns (board, player
    to move, ply) triples. Unlike bench_v22_leaf.py's every-4th-
    ply sample (all White to move), this covers BOTH parities —
    required because v28's rule is parity-dependent (iter-28
    §1)."""
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
            if ply >= 8:
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
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def _gap(a1, a0, mask):
    """Gap-three maker count (three-tier with v0 < 1000)."""
    s1 = a1[mask]
    s0 = a0[mask]
    three = (s1 >= 4000) & (s1 < 20000)
    return int(np.count_nonzero(three & (s0 < 1000)))


def _breadth_diffs(board, player):
    """(v14 breadth diff, v20 breadth diff, v22 breadth diff,
    v28 breadth diff, gap3 black, gap3 white) for the player to
    move."""
    b1, w1 = kernel_v1.attack_v1_both(board)
    b0, w0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        o1, t1, o0, t0 = b1, w1, b0, w0
    else:
        o1, t1, o0, t0 = w1, b1, w0, b0
    d14 = _breadth_v1(o1, mask) - _breadth_v1(t1, mask)
    d20 = _breadth_v1_gap(o1, o0, mask) - _breadth_v1_gap(t1, t0, mask)
    # v22: Black's breadth demoted, White's undemoted, in both
    # terms of the difference.
    bb = _breadth_v1_gap(b1, b0, mask)
    wb = _breadth_v1(w1, mask)
    d22 = (bb - wb) if player == game.BLACK else (wb - bb)
    # v28: Black's breadth demoted ONLY in the ours term — i.e.
    # only at Black-to-move leaves. At White-to-move leaves v28
    # IS v14 (no demotion anywhere).
    d28 = d22 if player == game.BLACK else d14
    return d14, d20, d22, d28, _gap(b1, b0, mask), _gap(w1, w0, mask)


def _summarise(label, ds):
    qs = statistics.quantiles(ds, n=10)
    print(f"  {label}: n={len(ds)}, mean {statistics.mean(ds):+.2f}, "
          f"std {statistics.pstdev(ds):.2f}, "
          f"p10 {qs[0]:+.2f}, p50 {statistics.median(ds):+.2f}, "
          f"p90 {qs[8]:+.2f}")
    print(f"    |diff| >= 2 in "
          f"{sum(1 for d in ds if abs(d) >= 2) / len(ds):.1%}, "
          f">= 4 in {sum(1 for d in ds if abs(d) >= 4) / len(ds):.1%}, "
          f">= 6 in {sum(1 for d in ds if abs(d) >= 6) / len(ds):.1%}")


def main() -> int:
    boards = _sample_boards()
    quiet = [(b, p, pl) for b, p, pl in boards
             if _terminal_leaf_value(b, p) == 0.0]
    n_black = sum(1 for _b, p, _pl in quiet if p == game.BLACK)
    n_white = len(quiet) - n_black
    print(f"sample: {len(boards)} boards, {len(quiet)} quiet "
          f"({n_black} Black to move, {n_white} White to move)")
    print("== cost (median of 3 rounds x 50 calls) ==")
    for name, fn in (
        ("v28 _attack_gradient_v1_gap_ours", _attack_gradient_v1_gap_ours),
        ("v28 _gradient_v1_gap_ours_leaf_value",
         _gradient_v1_gap_ours_leaf_value),
        ("v22 _attack_gradient_v1_gap_black", _attack_gradient_v1_gap_black),
        ("v20 _attack_gradient_v1_gap", _attack_gradient_v1_gap),
        ("v14 _attack_gradient_v1", _attack_gradient_v1),
        ("v7 _attack_gradient", _attack_gradient),
        ("v6 _terminal_leaf_value", _terminal_leaf_value),
    ):
        t = _timeit(fn, quiet)
        print(f"  {name}: {t['median_ms']:.2f} ms "
              f"(min {t['min_ms']:.2f}, max {t['max_ms']:.2f})")
    print("== ours-only breadth-difference distribution (quiet) ==")
    d14s, d20s, d22s, d28s, g_black, g_white = [], [], [], [], [], []
    d28_black, d28_white = [], []
    for b, p, _pl in quiet:
        d14, d20, d22, d28, gb, gw = _breadth_diffs(b, p)
        d14s.append(d14)
        d20s.append(d20)
        d22s.append(d22)
        d28s.append(d28)
        g_black.append(gb)
        g_white.append(gw)
        (d28_black if p == game.BLACK else d28_white).append(d28)
    for label, ds in (("v14 breadth diff", d14s),
                      ("v20 breadth diff", d20s),
                      ("v22 breadth diff", d22s),
                      ("v28 breadth diff (pooled)", d28s),
                      ("v28 breadth diff (Black to move)", d28_black),
                      ("v28 breadth diff (White to move)", d28_white)):
        _summarise(label, ds)
    print(f"  1.8 x v28 pooled std = "
          f"{1.8 * statistics.pstdev(d28s):.2f} "
          f"-> scale {round(1.8 * statistics.pstdev(d28s) * 2) / 2}")
    print(f"  mean gap-three makers per position: black "
          f"{statistics.mean(g_black):.2f}, white "
          f"{statistics.mean(g_white):.2f}")
    for label, ref in (("v14", d14s), ("v20", d20s), ("v22", d22s)):
        chg = [abs(a - b) for a, b in zip(ref, d28s)]
        print(f"  mean |{label} diff - v28 diff| = "
              f"{statistics.mean(chg):.2f}; differs in "
              f"{sum(1 for c in chg if c > 0) / len(chg):.1%} "
              f"of quiet positions")
    print(f"  mean v28 gradient magnitude at scale 7.5: "
          f"{statistics.mean(abs(0.2 * math.tanh(d / 7.5)) for d in d28s):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

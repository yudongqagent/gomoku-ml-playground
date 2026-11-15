"""Step 0 measurements for batch 5 H3 (v14 leaf design).

Measures, on real mid-game boards replayed from the batch-3 v7-vs-v1
match records (the reference engine's games; same sampling rule as
``bench_v7_leaf.py``: first 20 games, every 4th ply from ply 8):

(a) per-call cost of ``kernel_v1.attack_v1_both`` and of the full v1
    breadth gradient built on it, vs ``kernel_numpy.attack_both``
    (the v7 leaf's input) and the terminal-only leaf;
(b) the DISTRIBUTION of the v1 breadth difference (ours - theirs,
    player to move) on quiet positions from real games — this
    calibrates the tanh squash scale for the v14 leaf (a design
    parameter fixed in the pre-registration, not tuned on v14 eval
    results). The scale rule is the same as v7's: scale ~= 1.8 x
    the measured std (v7: scale 4 ~= 1.8 x std 2.26);
(c) how often the v1 breadth differs from the v0 breadth on the
    same positions — a direct measure of the broken-four blindness
    this workstream removes.

Thresholds are derived from the v1 pattern classes, NOT measured:
strong = a move creating a four or an open three (v1 attack >=
4000 — no combination of sub-4000 per-direction values reaches
4000: the strongest sub-open-three direction is a three at 400,
4 x 400 = 1600, and the double-threat bonuses require a four or
an open three); developing = a move creating a half-open three
(400) or an open two (80) (80 <= attack < 4000 — the strongest
sub-open-two combination is 4 x two = 40 < 80). Breadth = strong
+ 0.25 * developing, as in v7.

Usage:
    python ml/scripts/bench_v14_leaf.py
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

from gomoku_ml import game, kernel_numpy, kernel_v1  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    _attack_gradient,
    _terminal_leaf_value,
)

REPO_ROOT = common.REPO_ROOT

# v1 breadth thresholds (pattern-class derived — see module docstring).
_V1_STRONG = 4000
_V1_DEVELOPING = 80
_DEVELOPING_WEIGHT = 0.25

# v0 breadth thresholds (v7's, for the comparison in (c)).
_V0_STRONG = 1000
_V0_DEVELOPING = 100


def _breadth(attack: np.ndarray, empty_mask: np.ndarray,
             strong: int, developing: int) -> float:
    s = attack[empty_mask]
    n_strong = int(np.count_nonzero(s >= strong))
    n_dev = int(np.count_nonzero((s >= developing) & (s < strong)))
    return n_strong + _DEVELOPING_WEIGHT * n_dev


def _v1_gradient(board: np.ndarray, player: int) -> float:
    black, white = kernel_v1.attack_v1_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours, theirs = black, white
    else:
        ours, theirs = white, black
    return (
        _breadth(ours, mask, _V1_STRONG, _V1_DEVELOPING)
        - _breadth(theirs, mask, _V1_STRONG, _V1_DEVELOPING)
    )


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
            if ply >= 8 and ply % 4 == 0:
                player = game.BLACK if ply % 2 == 0 else game.WHITE
                out.append((board.copy(), player, ply))
            board = game.apply_move(board, move,
                                    game.BLACK if ply % 2 == 0 else game.WHITE)
    return out


def _timeit(fn, boards, rounds=3, calls=50):
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for b, p, _ in boards[:calls]:
            fn(b, p)
        times.append((time.perf_counter() - t0) / calls * 1000)
    return min(times), statistics.median(times), max(times)


def main() -> None:
    boards = _sample_boards()
    print(f"sampled {len(boards)} positions from batch-v7-vs-v1.json")
    quiet = [(b, p, ply) for b, p, ply in boards
             if _terminal_leaf_value(b, p) == 0.0]
    print(f"quiet positions (terminal leaf == 0): {len(quiet)}")

    print("\n== cost (ms/call, min / median / max over 3 rounds x 50 calls) ==")
    for label, fn in [
        ("kernel_v1.attack_v1_both",
         lambda b, p: kernel_v1.attack_v1_both(b)),
        ("v1 breadth gradient (full)",
         lambda b, p: _v1_gradient(b, p)),
        ("kernel_numpy.attack_both (v7)",
         lambda b, p: kernel_numpy.attack_both(b)),
        ("v7 _attack_gradient (full)",
         lambda b, p: _attack_gradient(b, p)),
        ("v6 _terminal_leaf_value",
         lambda b, p: _terminal_leaf_value(b, p)),
    ]:
        lo, med, hi = _timeit(fn, boards)
        print(f"  {label:36s} {lo:6.2f} / {med:6.2f} / {hi:6.2f}")

    print("\n== v1 breadth-difference distribution (quiet positions) ==")
    diffs = [_v1_gradient(b, p) for b, p, _ in quiet]
    diffs_sorted = sorted(diffs)
    n = len(diffs)
    print(f"  n={n} mean={statistics.mean(diffs):.2f} "
          f"std={statistics.pstdev(diffs):.2f}")
    print(f"  p10={diffs_sorted[int(0.10 * n)]:.2f} "
          f"p50={diffs_sorted[int(0.50 * n)]:.2f} "
          f"p90={diffs_sorted[int(0.90 * n)]:.2f}")
    for k in (2, 4, 6):
        frac = sum(1 for d in diffs if abs(d) >= k) / n
        print(f"  |diff| >= {k}: {frac:.1%}")
    print(f"  1.8 x std = {1.8 * statistics.pstdev(diffs):.2f}")

    print("\n== v1 breadth vs v0 breadth (same quiet positions) ==")
    n_diff = 0
    n_strong_diff = 0
    v1_strong_total = 0
    v0_strong_total = 0
    for b, p, _ in quiet:
        mask = b == game.EMPTY
        vb, vw = kernel_v1.attack_v1_both(b)
        ob, ow = kernel_numpy.attack_both(b)
        if p == game.BLACK:
            ours_v1, theirs_v1 = vb, vw
            ours_v0, theirs_v0 = ob, ow
        else:
            ours_v1, theirs_v1 = vw, vb
            ours_v0, theirs_v0 = ow, ob
        d1 = (_breadth(ours_v1, mask, _V1_STRONG, _V1_DEVELOPING)
              - _breadth(theirs_v1, mask, _V1_STRONG, _V1_DEVELOPING))
        d0 = (_breadth(ours_v0, mask, _V0_STRONG, _V0_DEVELOPING)
              - _breadth(theirs_v0, mask, _V0_STRONG, _V0_DEVELOPING))
        if d1 != d0:
            n_diff += 1
        s1 = (int(np.count_nonzero(ours_v1[mask] >= _V1_STRONG))
              + int(np.count_nonzero(theirs_v1[mask] >= _V1_STRONG)))
        s0 = (int(np.count_nonzero(ours_v0[mask] >= _V0_STRONG))
              + int(np.count_nonzero(theirs_v0[mask] >= _V0_STRONG)))
        v1_strong_total += s1
        v0_strong_total += s0
        if s1 != s0:
            n_strong_diff += 1
    print(f"  positions where v1 breadth diff != v0 breadth diff: "
          f"{n_diff}/{n} ({n_diff / n:.1%})")
    print(f"  positions where total strong count differs: "
          f"{n_strong_diff}/{n} ({n_strong_diff / n:.1%})")
    print(f"  mean total strong moves per position: v1 "
          f"{v1_strong_total / n:.2f} vs v0 {v0_strong_total / n:.2f}")


if __name__ == "__main__":
    main()

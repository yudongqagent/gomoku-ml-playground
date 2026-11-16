#!/usr/bin/env python3
"""Step 0 for batch 6 H1 (v15, iter-15 §1): measure the v1 prior
distribution on real game positions, so the prior-floor sweep
values are picked from the distribution — NOT tuned on any eval.

Replays boards from batch-v7-vs-v1.json (first 20 games, every
4th ply from ply 8 — iter-7's Step-0 sampling) and reports, for
each candidate floor value:
- how many candidates sit below the floor (median / mean),
- the total prior mass the floor adds before renormalization,
- the renormalization factor (how much above-floor priors shrink),
- the top prior after flooring (sharpness check),
- what a move at prior 0.027 / 0.040 (the documented starved-save
  range, iter-12 §2.2 / iter-14 §2.1) becomes after flooring.

A floor preserves the ratios among above-floor moves exactly
(renormalization is a common divisor); only below-floor moves are
lifted. That is the property that makes a floor NOT flattening
(v13's temperature probe: flattening is catastrophic).

Usage:
    python ml/scripts/measure_prior_floor.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import game  # noqa: E402
from gomoku_ml.mcts_heuristic import _priors  # noqa: E402

FLOORS = (0.01, 0.02, 0.03, 0.04, 0.05)
SAVE_PRIORS = (0.027, 0.040)


def _floored(priors: dict[int, float], floor: float) -> dict[int, float]:
    lifted = {m: max(p, floor) for m, p in priors.items()}
    total = sum(lifted.values())
    return {m: p / total for m, p in lifted.items()}


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = os.path.join(
        repo_root, "research", "iterations", "batch-v7-vs-v1.json")
    with open(path, encoding="utf-8") as f:
        batch = json.load(f)
    positions = []
    for rec in batch["records"][:20]:
        board = game.empty_board()
        player = game.BLACK
        for ply, move in enumerate(rec["moves"]):
            if ply >= 8 and ply % 4 == 0:
                positions.append((board.copy(), player))
            board = game.apply_move(board, move, player)
            player = -player
    print(f"positions: {len(positions)}")
    n_cands, top_prior = [], []
    below = {f: [] for f in FLOORS}
    added_mass = {f: [] for f in FLOORS}
    renorm = {f: [] for f in FLOORS}
    top_after = {f: [] for f in FLOORS}
    save_after = {f: {s: [] for s in SAVE_PRIORS} for f in FLOORS}
    for board, player in positions:
        priors = _priors(board, player)
        n_cands.append(len(priors))
        top_prior.append(max(priors.values()))
        for f in FLOORS:
            below[f].append(sum(1 for p in priors.values() if p < f))
            added = sum(max(0.0, f - p) for p in priors.values())
            added_mass[f].append(added)
            renorm[f].append(1.0 + added)
            fl = _floored(priors, f)
            top_after[f].append(max(fl.values()))
            for s in SAVE_PRIORS:
                save_after[f][s].append(max(s, f) / (1.0 + added))
    def med(xs):
        return round(statistics.median(xs), 4)
    def mean(xs):
        return round(statistics.mean(xs), 4)
    print(f"candidates: median {med(n_cands)} mean {mean(n_cands)} "
          f"p10 {sorted(n_cands)[len(n_cands)//10]} "
          f"p90 {sorted(n_cands)[9*len(n_cands)//10]}")
    print(f"top prior: median {med(top_prior)} mean {mean(top_prior)}")
    print()
    print("floor | below (med/mean) | mass added (med) | renorm (med) | "
          "top after (med) | 0.027-> (med) | 0.040-> (med)")
    for f in FLOORS:
        print(f"{f:.2f}  | {med(below[f]):>5}/{mean(below[f]):<5} | "
              f"{med(added_mass[f]):>6} | {med(renorm[f]):>6} | "
              f"{med(top_after[f]):>6} | {med(save_after[f][0.027]):>6} | "
              f"{med(save_after[f][0.040]):>6}")
    # Prior percentiles across all positions (pooled moves).
    pooled = []
    for board, player in positions:
        pooled.extend(_priors(board, player).values())
    pooled.sort()
    n = len(pooled)
    print()
    print(f"pooled moves: {n}")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
        print(f"  p{int(q*100):>2}: {pooled[int(q*n)]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

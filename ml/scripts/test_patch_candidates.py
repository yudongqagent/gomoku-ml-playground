#!/usr/bin/env python3
"""Patch experiment for fast-loop iteration 1: replay the 5 critical
positions under candidate leaf patches and report which move v7
plays in each. The patch that fixes the most positions (ideally
all 5) is the iteration-1 patch; the experiment itself is NOT an
eval — it is the fast loop's verification step.

Candidates:
- base: the pre-registered gradient (tanh(diff/4), uncapped).
- cap02: gradient capped to +/-0.2 (below the must-block value
  0.25, so a static heuristic can never outweigh an exact
  terminal signal).
- suppress: gradient suppressed (0.0) when either side holds an
  open-four creator (max v0 attack >= 10000 on an empty cell) —
  creator positions are forcing-adjacent, not quiet.
- cap02+suppress: both.

Usage:
    python ml/scripts/test_patch_candidates.py
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import game, kernel_numpy, scoring  # noqa: E402
import gomoku_ml.mcts_heuristic as mh  # noqa: E402

BATCH = os.path.join(common.REPO_ROOT, "research", "iterations",
                     "fastloop1-v7-vs-v0.json")

# (game, ply, actual move, saving move)
CASES = [
    (1, 13, (4, 6), (8, 6)),
    (3, 23, (4, 9), (9, 12)),
    (5, 21, (9, 11), (5, 7)),
    (7, 15, (7, 6), (7, 10)),
    (9, 15, (8, 7), (4, 7)),
]
# Game 5 has a second saving move, (8, 12) — v6's choice.
ALT_SAVE = {(5, 21): (8, 12)}


def _board_at(rec, ply):
    board = game.empty_board()
    for i, m in enumerate(rec["moves"][:ply]):
        board = game.apply_move(
            board, int(m), game.BLACK if i % 2 == 0 else game.WHITE
        )
    return board


def _choose(board, player, leaf_fn):
    """Run v7's MCTS with a patched leaf function (monkey-patched
    into the module's leaf table)."""
    orig = mh._LEAF_FUNCTIONS["gradient"]
    mh._LEAF_FUNCTIONS["gradient"] = leaf_fn
    try:
        mcts = mh.HeuristicMCTS(sims=100, rng=np.random.default_rng(0),
                                 leaf_mode="gradient")
        return mcts.choose_move(board, player)
    finally:
        mh._LEAF_FUNCTIONS["gradient"] = orig


def _leaf_cap02(board, player):
    value = mh._terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    return 0.2 * mh._attack_gradient(board, player)


def _leaf_suppress(board, player):
    value = mh._terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    black, white = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    if not mask.any():
        return 0.0
    if (int(black[mask].max()) >= 10000
            or int(white[mask].max()) >= 10000):
        return 0.0
    return mh._attack_gradient(board, player)


def _leaf_cap02_suppress(board, player):
    value = mh._terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    black, white = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    if not mask.any():
        return 0.0
    if (int(black[mask].max()) >= 10000
            or int(white[mask].max()) >= 10000):
        return 0.0
    return 0.2 * mh._attack_gradient(board, player)


def main() -> int:
    with open(BATCH, encoding="utf-8") as f:
        batch = json.load(f)
    patches = [
        ("base (uncapped)", mh._gradient_leaf_value),
        ("cap02", _leaf_cap02),
        ("suppress", _leaf_suppress),
        ("cap02+suppress", _leaf_cap02_suppress),
    ]
    results: dict[str, list[str]] = {name: [] for name, _ in patches}
    for g, ply, actual, save in CASES:
        rec = next(r for r in batch["records"] if r["game"] == g)
        board = _board_at(rec, ply)
        player = game.BLACK if ply % 2 == 0 else game.WHITE
        alt = ALT_SAVE.get((g, ply))
        print(f"game {g} ply {ply}: actual {actual} (loses), "
              f"saving {save}" + (f" or {alt}" if alt else ""))
        for name, leaf_fn in patches:
            move = _choose(board, player, leaf_fn)
            xy = (move % 15, move // 15)
            ok = xy == save or (alt is not None and xy == alt)
            tag = "FIXED" if ok else ("actual" if xy == actual else "other")
            results[name].append(tag)
            print(f"  {name:16s} plays {xy} [{tag}]")
    print("\nsummary (FIXED / 5):")
    for name, tags in results.items():
        fixed = sum(1 for t in tags if t == "FIXED")
        print(f"  {name:16s} {fixed}/5  {tags}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

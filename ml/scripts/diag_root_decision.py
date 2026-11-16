#!/usr/bin/env python3
"""Pre-eval diagnostics for the v9 root-decision variants (batch 4, H1).

Diagnostic only — the design stands as pre-registered in
``research/iterations/iter-9.md`` §1; the evals decide.

1. The §2.6 position (batch-v7-vs-v0.json game 16, ply 14, Black to
   move): what does each variant play? Full-precision root stats
   for the tied pair are printed, so the tie-prior behaviour is
   checkable. Oracle-verified saving moves: (7,4), (9,8), (6,9);
   the losing move v7 played: (8,4).
2. The 5 v7 fast-loop regression positions: does each variant keep
   them passing?
3. (with --tactical) Old + hard tactical suites per variant, with
   the full-dispatch driver.

Usage:
    python ml/scripts/diag_root_decision.py [--tactical]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from gomoku_ml import evaluate, game, research  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS, _Node  # noqa: E402

RULES = ["visits", "q", "tie-q", "blend", "tie-prior"]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BATCH_V7_V0 = os.path.join(
    REPO, "research", "iterations", "batch-v7-vs-v0.json")
REG_V7 = os.path.join(
    REPO, "ml", "gomoku_ml", "regression_positions_v7.json")
HARD_PATH = os.path.join(HERE, "..", "gomoku_ml",
                         "tactical_positions_hard.json")


def _xy(move: int) -> tuple[int, int]:
    return (move % 15, move // 15)


def _section26_board():
    """Game 16 vs v0 replayed to ply 14 (Black to move)."""
    with open(BATCH_V7_V0, encoding="utf-8") as f:
        rec = json.load(f)["records"][16]
    board = game.empty_board()
    player = game.BLACK
    for m in rec["moves"][:14]:
        board = game.apply_move(board, m, player)
        player = -player
    return board, player


def _root_stats(board, player):
    """Full-precision root stats under v7's search (gradient leaf)."""
    mcts = HeuristicMCTS(sims=100, leaf_mode="gradient", rng=None)
    root = _Node(1.0)
    for _ in range(100):
        mcts._run_one(root, board.copy(), player, None)
    return root


def _board_from_pos(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tactical", action="store_true",
                        help="also run old + hard tactical per variant")
    args = parser.parse_args()

    print("== §2.6 position (game 16 vs v0, ply 14, Black to move) ==")
    board, player = _section26_board()
    root = _root_stats(board, player)
    for label, xy in [("saving (7,4)", (7, 4)),
                      ("losing (8,4)", (8, 4)),
                      ("saving (9,8)", (9, 8)),
                      ("saving (6,9)", (6, 9))]:
        m = xy[1] * 15 + xy[0]
        c = root.children[m]
        print(f"  {label}: visits={c.visit_count} raw_q={c.q:.6f} "
              f"our_q={-c.q:.6f} prior={c.prior:.6f}")
    print("  argmax our-q over ALL visited children:",
          _xy(max(
              (m for m, c in root.children.items() if c.visit_count >= 1),
              key=lambda m: -root.children[m].q,
          )))
    for rule in RULES:
        mcts = HeuristicMCTS(sims=100, leaf_mode="gradient",
                             root_decision=rule, rng=None)
        move = mcts.choose_move(board.copy(), player)
        mark = "SAVING" if _xy(move) in [(7, 4), (9, 8), (6, 9)] else (
            "LOSING" if _xy(move) == (8, 4) else "other")
        print(f"  {rule:9s} plays {_xy(move)} [{mark}]")

    print("== v7 regression positions (5 fast-loop critical positions) ==")
    with open(REG_V7, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    for rule in RULES:
        passed = []
        for pos in positions:
            b = _board_from_pos(pos)
            p = game.BLACK if pos["to_move"] == "black" else game.WHITE
            mcts = HeuristicMCTS(sims=100, leaf_mode="gradient",
                                 root_decision=rule, rng=None)
            move = mcts.choose_move(b, p)
            passed.append(list(_xy(move)) in pos["expected"])
        print(f"  {rule:9s}: {sum(passed)}/{len(passed)} "
              f"({', '.join(p['id'] for p, ok in zip(positions, passed) if not ok) or 'all pass'})")

    if args.tactical:
        print("== tactical suites per variant (full dispatch) ==")
        old = evaluate.load_tactical_suite()
        hard = evaluate.load_tactical_suite(HARD_PATH)
        for rule in RULES:
            mcts = HeuristicMCTS(sims=100, leaf_mode="gradient",
                                 root_decision=rule,
                                 rng=np.random.default_rng(0))
            r_old = research.run_tactical(mcts.choose_move, old)
            r_hard = research.run_tactical(mcts.choose_move, hard)
            fails = [r["id"] for r in r_hard["results"] if not r["passed"]]
            print(f"  {rule:9s}: old {r_old['passed']}/{r_old['total']}, "
                  f"hard {r_hard['passed']}/{r_hard['total']}"
                  + (f" (hard FAIL: {', '.join(fails)})" if fails else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Pre-eval diagnostic for the v12 forcing-first tie-break (batch 5, H1).

Diagnostic only — the design stands as pre-registered in
``research/iterations/iter-12.md`` §1; the evals decide.

1. The iter-9 §2.6 position (batch-v9-vs-v7-swap.json game 41, ply
   21, White to move): what does each root-decision rule play?
   Full-precision root stats for the tied triple are printed, so
   the forcing-first behaviour is checkable. v9 played (7,8)
   (occupying one creator); the forcing counter-threats are (3,11)
   and (2,12). NOTE (iter-12 §1.0): ply 21 does NOT verify as a
   single losing move — the preventing moves also lose 3/3 in
   match-engine playouts — so this diagnostic tests only that the
   rule does what it claims, not that the move saves the game.
2. The v9 regression positions: does tie-prior-forcing keep them
   passing (same pass/fail pattern as tie-prior)?

Usage:
    python ml/scripts/diag_v12_forcing.py [--swap-json PATH]

The v9 swap JSON lives on branch research/batch-4-h1; extract it
with ``git show research/batch-4-h1:research/iterations/
batch-v9-vs-v7-swap.json`` and pass the path, or check it into
research/iterations/ (the default location).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS, _Node  # noqa: E402

RULES = ["visits", "tie-prior", "tie-prior-forcing"]

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SWAP = os.path.join(
    REPO, "research", "iterations", "batch-v9-vs-v7-swap.json")
REG_V9 = os.path.join(
    REPO, "ml", "gomoku_ml", "regression_positions_v9.json")


def _xy(move: int) -> tuple[int, int]:
    return (move % 15, move // 15)


def _swap_g41_board(path: str):
    """Swap game 41 replayed to ply 21 (White to move). The moves
    list includes the 3 opening stones; plies alternate Black,
    White from ply 0, so ply 21 is White to move."""
    with open(path, encoding="utf-8") as f:
        rec = json.load(f)["records"][41]
    assert rec["plies"] == 25 and rec["moves"][21] == 127, (
        "unexpected game 41 record — is this batch-v9-vs-v7-swap?")
    board = game.empty_board()
    player = game.BLACK
    for m in rec["moves"][:21]:
        board = game.apply_move(board, m, player)
        player = -player
    assert player == game.WHITE
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
    parser.add_argument("--swap-json", default=DEFAULT_SWAP,
                        help="path to batch-v9-vs-v7-swap.json")
    args = parser.parse_args()

    print("== §2.6 position (swap game 41, ply 21, White to move) ==")
    board, player = _swap_g41_board(args.swap_json)
    root = _root_stats(board, player)
    flat = board.reshape(-1).tolist()
    for label, xy in [("forcing (3,11)", (3, 11)),
                      ("forcing (2,12)", (2, 12)),
                      ("played (7,8)", (7, 8))]:
        m = xy[1] * 15 + xy[0]
        c = root.children[m]
        score = scoring._score_move_flat(flat, m, player, 1.0)
        print(f"  {label}: visits={c.visit_count} raw_q={c.q:.6f} "
              f"our_q={-c.q:.6f} prior={c.prior:.6f} "
              f"v1_score={score:.0f} "
              f"forcing={scoring._is_forcing(board, m, player)}")
    for rule in RULES:
        mcts = HeuristicMCTS(sims=100, leaf_mode="gradient",
                             root_decision=rule, rng=None)
        move = mcts.choose_move(board.copy(), player)
        mark = "FORCING" if _xy(move) in [(3, 11), (2, 12)] else (
            "PLAYED (7,8)" if _xy(move) == (7, 8) else "other")
        print(f"  {rule:17s} plays {_xy(move)} [{mark}]")

    print("== v9 regression positions ==")
    with open(REG_V9, encoding="utf-8") as f:
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
        print(f"  {rule:17s}: {sum(passed)}/{len(passed)} "
              f"({', '.join(p['id'] for p, ok in zip(positions, passed) if not ok) or 'all pass'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Compare _vcf_prevention_pool on 2 regression positions between
the batch-10 tree and the port tree (the full-pool scan is ~60s
per position, so it is verified separately from ab_v35_port.py,
not on all 58 positions).

Usage (from the worktree root):
    python research/iterations/pool_ab.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml"))

import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, scoring  # noqa: E402


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.join(here, "..", "..", "ml", "gomoku_ml")
    with open(os.path.join(ml_dir, "regression_positions_v35.json"),
              encoding="utf-8") as f:
        positions = json.load(f)["positions"][:2]
    v35 = scoring.get_player(35, seed=0)
    mcts = v35._mcts
    for pos in positions:
        board = game.empty_board()
        for x, y in pos["black"]:
            board[y, x] = game.BLACK
        for x, y in pos["white"]:
            board[y, x] = game.WHITE
        player = (game.BLACK if pos["to_move"] == "black"
                  else game.WHITE)
        pool = mcts._vcf_prevention_pool(board, player)
        print(f"{pos['id']}: pool="
              f"{None if pool is None else [[m % 15, m // 15] for m in pool]}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnostic: run a WhiteMCTS engine on the v10 White-weakness
regression positions (ml/gomoku_ml/regression_positions_v10.json)
and report, per position, the engine's chosen move, whether it is
one of the oracle-verified saves, and the root statistics
(visits / prior / correctly-signed q) for the chosen move and the
best save.

q SIGN (iter-9 §1.0, iter-10 §6): ``child.q`` is the OPPONENT's
value under the negamax backup; our-perspective q is ``-child.q``.
This script reports OUR q (``-child.q``) and labels it as such —
never the raw value.

This is a DIAGNOSTIC, not an eval: it informs the v13 design
(iter-13 §1b), it does not decide the bar.

Usage:
    python ml/scripts/diag_white_positions.py --white-sims 100
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

from gomoku_ml import game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import _Node  # noqa: E402
from gomoku_ml.mcts_white import WhiteMCTS  # noqa: E402

POSITIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "gomoku_ml", "regression_positions_v10.json",
)


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


def _flat(xy):
    return xy[1] * game.SIZE + xy[0]


def _xy(m):
    return (m % game.SIZE, m // game.SIZE)


def _root_stats(mcts, board, player, n_sims):
    """Run the PUCT search manually and return root children stats."""
    root = _Node(1.0)
    for _ in range(n_sims):
        mcts._run_one(root, board, player, None)
    return root.children


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--white-sims", type=int, default=100)
    parser.add_argument("--black-sims", type=int, default=100)
    parser.add_argument("--positions", default=POSITIONS_PATH)
    args = parser.parse_args()

    with open(args.positions, encoding="utf-8") as f:
        positions = json.load(f)["positions"]

    mcts = WhiteMCTS(
        sims=args.black_sims, white_sims=args.white_sims,
        rng=np.random.default_rng(0),
    )
    n_saved = 0
    for pos in positions:
        board = _board(pos)
        player = game.WHITE if pos["to_move"] == "white" else game.BLACK
        n_sims = args.white_sims if player == game.WHITE else args.black_sims
        chosen = mcts.choose_move(board.copy(), player)
        saves = {_flat(s) for s in pos["expected"]}
        is_save = chosen in saves
        n_saved += int(is_save)
        children = _root_stats(mcts, board.copy(), player, n_sims)
        # Threat-check note: choose_move may filter the pool; the
        # raw root stats below are pre-filter.
        creators = scoring._opponent_creators(board, player)
        c_chosen = children.get(chosen)
        best_save = None
        for m in saves:
            c = children.get(m)
            if c is not None and (best_save is None
                                  or c.visit_count > best_save[1].visit_count):
                best_save = (m, c)
        print(f"{pos['id']} ({pos['to_move']}, creators={len(creators)}): "
              f"chose {_xy(chosen)} {'SAVE' if is_save else 'not a save'}; "
              f"expected {[tuple(s) for s in pos['expected']]}")
        if c_chosen is not None:
            print(f"    chosen: visits={c_chosen.visit_count} "
                  f"prior={c_chosen.prior:.3f} our-q={-c_chosen.q:+.3f}")
        if best_save is not None:
            m, c = best_save
            print(f"    best save {_xy(m)}: visits={c.visit_count} "
                  f"prior={c.prior:.3f} our-q={-c.q:+.3f}")
        else:
            print("    no save move among root children")
    print(f"TOTAL: {n_saved}/{len(positions)} saves played "
          f"(white_sims={args.white_sims})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

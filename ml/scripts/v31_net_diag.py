#!/usr/bin/env python3
"""Net-value diagnostic for one fast-loop blunder position (v31).

Replays a match JSON game to a ply, then for the actual move and
each --alt move prints, for a list of --version engines:
  * the raw net value of the resulting board (side to move = the
    opponent), with search-path history (1 ply) and with history
    zeroed (the no-history view);
  * the _value_leaf_value actually used by the search (terminal
    short-circuits included), ours-perspective.

Usage:
    python ml/scripts/v31_net_diag.py research/iterations/fast-v31-vs-v0-iter1.json \\
        --game 7 --ply 43 --version 31 --version 8 \\
        --alt 0,5 --alt 4,9 --alt 2,10 --alt 3,10 --alt 2,11
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
import torch  # noqa: E402
from gomoku_ml import encoding, game, scoring  # noqa: E402


def _xy(m: int) -> str:
    return f"({m % 15},{m // 15})"


def _move(x: int, y: int) -> int:
    return y * 15 + x


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--ply", type=int, required=True,
                        help="moves already played; the position "
                        "BEFORE the move at index --ply")
    parser.add_argument("--version", type=int, action="append",
                        required=True)
    parser.add_argument("--alt", action="append", default=[])
    args = parser.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        batch = json.load(f)
    rec = batch["records"][args.game]
    moves = rec["moves"]
    board = game.empty_board()
    player = game.BLACK
    for m in moves[: args.ply]:
        board = game.apply_move(board, m, player)
        player = -player
    actual = moves[args.ply]
    alts = []
    for a in args.alt:
        x, y = a.split(",")
        alts.append(_move(int(x), int(y)))

    print(f"== {os.path.basename(args.batch)} game {args.game}, "
          f"ply {args.ply} ({'Black' if player == game.BLACK else 'White'} "
          f"to move) ==\nactual move: {_xy(actual)}")
    for version in args.version:
        p = scoring.get_player(version, seed=0)
        mcts = p._mcts
        print(f"-- v{version} (value_weight={mcts.value_weight}) --")
        for label, mv in [("ACTUAL", actual)] + [
            (f"ALT {_xy(a)}", a) for a in alts
        ]:
            b2 = game.apply_move(board, mv, player)
            # Search-path history at a depth-1 leaf: the board
            # before our move (1 ply of history).
            hist = [board.copy()]
            planes = encoding.encode(b2, -player, hist, mv)
            with torch.no_grad():
                x = torch.from_numpy(planes).unsqueeze(0)
                _logits, value = mcts.net(x)
            raw_hist = float(value[0, 0].item())
            planes_nh = planes.copy()
            planes_nh[(2, 4, 5, 6, 7), :, :] = 0.0
            with torch.no_grad():
                x = torch.from_numpy(planes_nh).unsqueeze(0)
                _logits, value = mcts.net(x)
            raw_nohist = float(value[0, 0].item())
            leaf = mcts._value_leaf_value(b2, -player, hist, mv)
            print(f"  {label:>12} {_xy(mv)}: raw net (hist) "
                  f"{raw_hist:+.3f}, raw net (no-hist) {raw_nohist:+.3f}, "
                  f"leaf value (opp-persp) {leaf:+.3f}, "
                  f"ours-persp {-leaf:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

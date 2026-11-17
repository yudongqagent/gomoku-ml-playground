"""Root stats of v25's own ValueMCTS at a critical position.

Reconstructs the board at a given ply of a recorded game and runs
v25's ValueMCTS (100 sims) manually, printing visits/q/prior for
the actual (losing) move and the saving move, plus the top-5 by
visits. Answers: does v25's value leaf systematically prefer the
losing move, or was the game choice search noise in a near-tie?

Usage:
    python ml/scripts/v25_root_stats.py BATCH_JSON --game G --ply P \
        --actual X,Y --save X,Y [--seed 0] [--sims 100]
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
from gomoku_ml.mcts_value import ValueMCTS, _Node  # noqa: E402


def _xy(m: int) -> tuple[int, int]:
    return (m % 15, m // 15)


def _move(xy) -> int:
    return xy[1] * 15 + xy[0]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("batch_json")
    p.add_argument("--game", type=int, required=True)
    p.add_argument("--ply", type=int, required=True)
    p.add_argument("--actual", required=True, help="X,Y")
    p.add_argument("--save", required=True, help="X,Y")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sims", type=int, default=100)
    args = p.parse_args()

    d = json.load(open(args.batch_json, encoding="utf-8"))
    rec = next(r for r in d["records"] if r["game"] == args.game)
    board = game.empty_board()
    for i, m in enumerate(rec["moves"][: args.ply]):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE
        )
    player = game.BLACK if args.ply % 2 == 0 else game.WHITE
    actual = _move(tuple(int(v) for v in args.actual.split(",")))
    save = _move(tuple(int(v) for v in args.save.split(",")))

    player25 = scoring.get_player(25, seed=args.seed)
    mcts: ValueMCTS = player25._mcts
    mcts.sims = args.sims
    root = _Node(1.0)
    for _ in range(args.sims):
        mcts._run_one(root, board.copy(), player, None)
    stats = {}
    for m, c in root.children.items():
        stats[m] = (c.visit_count, round(c.q, 3), round(c.prior, 3))
    ranked = sorted(stats.items(), key=lambda kv: -kv[1][0])
    print(f"== v25 ValueMCTS root stats, game {args.game} ply {args.ply} "
          f"({'Black' if player == game.BLACK else 'White'} to move, "
          f"seed {args.seed}, {args.sims} sims) ==")
    print("  top-5 by visits:",
          ", ".join(f"{_xy(m)} v={v} q={q} p={pr}"
                    for m, (v, q, pr) in ranked[:5]))
    for label, m in (("actual", actual), ("saving", save)):
        if m in stats:
            v, q, pr = stats[m]
            print(f"  {label} {_xy(m)}: visits={v}, q={q}, prior={pr}")
        else:
            print(f"  {label} {_xy(m)}: NOT VISITED (not in root children)")
    best_v = max(v for v, _, _ in stats.values())
    best = [_xy(m) for m, (v, _, _) in stats.items() if v == best_v]
    print(f"  chosen (most visited): {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

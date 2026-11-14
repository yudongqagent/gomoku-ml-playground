#!/usr/bin/env python3
"""Deep analysis of one critical position from a fast-loop loss.

Reconstructs the board at a given ply of a recorded game and reports,
for the actual move and a saving alternative:
- whether each move is forcing (win / four), a creator (open four),
  or a VCF first move;
- each move's v1 1-ply score and rank among candidates;
- the attack gradient after each move (player-to-move perspective);
- v7's and v6's MCTS root stats (visits, q, prior) for each move —
  whether the search even considers the saving move, and whether
  the gradient leaf (v7) or the terminal leaf (v6) ranks it higher.

Usage:
    python ml/scripts/analyse_critical.py BATCH_JSON GAME PLY \\
        --actual X,Y --save X,Y [--sims 100]
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

from gomoku_ml import evaluate, game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    HeuristicMCTS,
    _attack_gradient,
    _Node,
    _priors,
    _terminal_leaf_value,
)


def _xy(m: int) -> tuple[int, int]:
    return (m % 15, m // 15)


def _move(xy: tuple[int, int]) -> int:
    return xy[1] * 15 + xy[0]


def _describe(board, move, player) -> dict:
    b2 = game.apply_move(board, move, player)
    wins = scoring._winning_moves(b2, player)
    creators = len(scoring._opponent_creators(b2, -player))
    # ^ creators for `player` after the move: opponent-creators of
    # the side to move (-player) are `player`'s creators.
    return {
        "wins_now": game.is_winning_stone(b2, move, player),
        "forcing": len(wins) >= 1,
        "winning_moves_after": len(wins),
        "creators_after": creators,
        "vcf": evaluate.starts_forced_win(board, player, move),
    }


def _root_stats(board, player, leaf_mode, sims, seed=0):
    """Run the MCTS search manually and return root children stats:
    {move: (visits, q, prior)} plus the chosen move."""
    mcts = HeuristicMCTS(sims=sims, rng=np.random.default_rng(seed),
                         leaf_mode=leaf_mode)
    root = _Node(1.0)
    for _ in range(sims):
        mcts._run_one(root, board.copy(), player, None)
    stats = {}
    for m, c in root.children.items():
        stats[m] = (c.visit_count, round(c.q, 3), round(c.prior, 3))
    best_visits = max(c.visit_count for c in root.children.values())
    best = [m for m, c in root.children.items()
            if c.visit_count == best_visits]
    return stats, best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_json")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--ply", type=int, required=True)
    parser.add_argument("--actual", required=True,
                        help="actual move as X,Y")
    parser.add_argument("--save", required=True,
                        help="saving move as X,Y")
    parser.add_argument("--sims", type=int, default=100)
    args = parser.parse_args()

    with open(args.batch_json, encoding="utf-8") as f:
        batch = json.load(f)
    rec = next(r for r in batch["records"] if r["game"] == args.game)
    moves = [int(m) for m in rec["moves"]]
    player = game.BLACK if args.ply % 2 == 0 else game.WHITE
    board = game.empty_board()
    for i, m in enumerate(moves[:args.ply]):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE
        )
    actual = _move(tuple(int(v) for v in args.actual.split(",")))
    save = _move(tuple(int(v) for v in args.save.split(",")))
    pname = "Black" if player == game.BLACK else "White"
    print(f"== game {args.game} ply {args.ply}: {pname} to move ==")
    print(f"  Black creators: "
          f"{len(scoring._opponent_creators(board, game.WHITE))}, "
          f"White creators: "
          f"{len(scoring._opponent_creators(board, game.BLACK))}")
    print(f"  Black winning moves: "
          f"{len(scoring._winning_moves(board, game.BLACK))}, White: "
          f"{len(scoring._winning_moves(board, game.WHITE))}")

    # v1 1-ply ranks.
    flat = board.reshape(-1).tolist()
    scored = sorted(
        ((scoring._score_move_flat(flat, m, player, 1.0), m)
         for m in scoring._candidate_moves(board)),
        key=lambda sm: (-sm[0], sm[1]),
    )
    rank = {m: i + 1 for i, (_s, m) in enumerate(scored)}
    score = {m: s for s, m in scored}
    for label, m in (("actual", actual), ("saving", save)):
        d = _describe(board, m, player)
        b2 = game.apply_move(board, m, player)
        grad_after = _attack_gradient(b2, -player)  # opponent to move
        print(f"  {label} {_xy(m)}: v1 score {score[m]}, rank "
              f"{rank[m]}/{len(scored)}; wins_now={d['wins_now']}, "
              f"forcing={d['forcing']}, creators_after="
              f"{d['creators_after']}, vcf={d['vcf']}; "
              f"gradient for opponent after = {grad_after:.3f} "
              f"(i.e. {(-grad_after):.3f} for {pname})")

    # MCTS root stats under v7 (gradient) and v6 (terminal).
    for leaf_mode in ("gradient", "terminal"):
        stats, best = _root_stats(board, player, leaf_mode, args.sims)
        top = sorted(stats.items(), key=lambda kv: -kv[1][0])[:5]
        print(f"  MCTS[{leaf_mode}] top-5 by visits: "
              + ", ".join(f"{_xy(m)} v={v} q={q} p={p}"
                          for m, (v, q, p) in top))
        for label, m in (("actual", actual), ("saving", save)):
            v, q, p = stats[m]
            print(f"    {label} {_xy(m)}: visits={v}, q={q}, prior={p}")
        print(f"    chosen (most visited): "
              f"{[_xy(m) for m in best]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

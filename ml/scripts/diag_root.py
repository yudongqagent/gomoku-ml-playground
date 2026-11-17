#!/usr/bin/env python3
"""Root dump for one position of a recorded game (batch-9 forensics).

Replays a batch JSON game to a given ply, runs the version's MCTS
search (default 100 sims) from that position, and prints:
  * the board (ASCII) and the player to move;
  * the top-10 root children by visits (move, visits, our-q,
    prior), with the actual move played and any --alt moves marked;
  * for the actual move and each --alt: the 1-ply leaf value of the
    resulting board (ours-perspective) under the version's leaf and
    under the v14 (undemoted) and v20 (globally demoted) leaves,
    plus the breadth decomposition (four / true-three / gap-three /
    developing) for both colours after the move.

The leaf comparison is the v22 forensics instrument: it shows
whether the Black-only demotion (vs v14/v20) is what ranks the
loser above the save at a verified losing position.

Usage:
    python ml/scripts/diag_root.py research/iterations/forensics-v22-vs-mcts-ladder.json \\
        --game 3 --ply 20 --version 22 --alt 8,9 --alt 4,5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from gomoku_ml import game, kernel_numpy, kernel_v1, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    _GRADIENT_DEVELOPING_WEIGHT,
    _GRADIENT_V1_DEVELOPING,
    _GRADIENT_V1_STRONG,
    _GRADIENT_STRONG,
    _Node,
    _gradient_v1_gap_black_leaf_value,
    _gradient_v1_gap_leaf_value,
    _gradient_v1_leaf_value,
    _terminal_leaf_value,
)


def _xy(move: int) -> tuple[int, int]:
    return (move % 15, move // 15)


def _move(x: int, y: int) -> int:
    return y * 15 + x


def _ascii(board: np.ndarray) -> str:
    glyph = {0: ".", 1: "X", -1: "O"}
    rows = []
    for y in range(15):
        rows.append("".join(glyph[int(board[y, x])] for x in range(15)))
    return "\n".join(rows)


def _breadth_parts(attack_v1, attack_v0, mask):
    """(four, true_three, gap_three, developing) counts for one side."""
    s1 = attack_v1[mask]
    s0 = attack_v0[mask]
    four = int(np.count_nonzero(s1 >= scoring.FOUR))
    three = (s1 >= _GRADIENT_V1_STRONG) & (s1 < scoring.FOUR)
    true_three = int(np.count_nonzero(three & (s0 >= _GRADIENT_STRONG)))
    gap_three = int(np.count_nonzero(three & (s0 < _GRADIENT_STRONG)))
    developing = int(
        np.count_nonzero(
            (s1 >= _GRADIENT_V1_DEVELOPING) & (s1 < _GRADIENT_V1_STRONG)
        )
    )
    return four, true_three, gap_three, developing


def _leaf_report(board: np.ndarray, player: int, move: int, label: str):
    """1-ply leaf values + breadth decomposition after ``move``.

    Leaf values are reported from the MOVER's perspective: the leaf
    is evaluated at the resulting board with the opponent to move,
    then negated. Terminal values are exact (±1/±0.25/0); quiet
    values are the gradient.
    """
    b2 = game.apply_move(board, move, player)
    opp = -player
    term = _terminal_leaf_value(b2, opp)
    v22 = _gradient_v1_gap_black_leaf_value(b2, opp)
    v20 = _gradient_v1_gap_leaf_value(b2, opp)
    v14 = _gradient_v1_leaf_value(b2, opp)
    black1, white1 = kernel_v1.attack_v1_both(b2)
    black0, white0 = kernel_numpy.attack_both(b2)
    mask = b2 == game.EMPTY
    ours1, theirs1 = (
        (black1, white1) if player == game.BLACK else (white1, black1)
    )
    ours0, theirs0 = (
        (black0, white0) if player == game.BLACK else (white0, black0)
    )
    o = _breadth_parts(ours1, ours0, mask)
    t = _breadth_parts(theirs1, theirs0, mask)
    print(
        f"  {label} {_xy(move)}: terminal(ours)={-term:+.3f} "
        f"v22(ours)={-v22:+.3f} v20(ours)={-v20:+.3f} "
        f"v14(ours)={-v14:+.3f}"
    )
    print(
        f"    breadth ours   four={o[0]} true3={o[1]} gap3={o[2]} "
        f"dev={o[3]}"
    )
    print(
        f"    breadth theirs four={t[0]} true3={t[1]} gap3={t[2]} "
        f"dev={t[3]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--ply", type=int, required=True,
                        help="moves already played; the position "
                        "BEFORE the move at index --ply")
    parser.add_argument("--version", type=int, default=22)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alt", action="append", default=[],
                        help="alternative move as x,y (repeatable)")
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
    actual = moves[args.ply] if args.ply < len(moves) else None
    alts = []
    for a in args.alt:
        x, y = a.split(",")
        alts.append(_move(int(x), int(y)))

    print(f"== {os.path.basename(args.batch)} game {args.game}, "
          f"ply {args.ply} ({'Black' if player == game.BLACK else 'White'} "
          f"to move), v{args.version} search, {args.sims} sims ==")
    print(_ascii(board))
    print(f"actual move played: {_xy(actual) if actual is not None else None}")

    player_obj = scoring.get_player(args.version, seed=args.seed)
    mcts = player_obj._mcts
    if mcts is None:
        print("version has no MCTS search; cannot dump a root")
        return 2
    root = _Node(1.0)
    for _ in range(args.sims):
        mcts._run_one(root, board.copy(), player, None)
    children = sorted(
        root.children.items(), key=lambda kv: -kv[1].visit_count
    )
    print(f"root children: {len(children)}; top 10 by visits "
          f"(our-q = -child.q):")
    marks = {actual: "ACTUAL"}
    for a in alts:
        marks[a] = "ALT"
    shown = set()
    for rank, (move, child) in enumerate(children[:10], start=1):
        mark = marks.get(move, "")
        print(f"  {rank:2d}. {_xy(move)} visits={child.visit_count:3d} "
              f"our-q={-child.q:+.3f} prior={child.prior:.4f} {mark}")
        shown.add(move)
    for move, label in marks.items():
        if move is not None and move not in shown:
            child = root.children.get(move)
            if child is None:
                print(f"  -- {_xy(move)} not a root child ({label})")
            else:
                print(f"  -- {_xy(move)} visits={child.visit_count:3d} "
                      f"our-q={-child.q:+.3f} prior={child.prior:.4f} "
                      f"({label})")

    print("1-ply leaf report (ours-perspective, after the move):")
    if actual is not None:
        _leaf_report(board, player, actual, "actual ")
    for a in alts:
        _leaf_report(board, player, a, "alt    ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

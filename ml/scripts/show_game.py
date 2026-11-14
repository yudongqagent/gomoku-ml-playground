#!/usr/bin/env python3
"""Print a recorded game as ASCII boards for fast-loop reading.

Shows the board every ``--every`` plies plus the final position,
with v8's (side A) moves marked, so the losing move can be read by
eye before it is verified with workup_loss.py.

Usage:
    python ml/scripts/show_game.py <batch.json> --game 2 [--every 10]
    python ml/scripts/show_game.py <batch.json> --game 2 --from-ply 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game as gomoku_game  # noqa: E402


def render(board, moves, upto, a_colour):
    grid = [["." for _ in range(15)] for _ in range(15)]
    for i, m in enumerate(moves[:upto]):
        r, c = divmod(m, 15)
        stone = "X" if i % 2 == 0 else "O"  # X = black (first), O = white
        grid[r][c] = stone
    lines = ["    " + " ".join(f"{c:2}" for c in range(15))]
    for r in range(15):
        lines.append(f"{r:2}  " + " ".join(f"{grid[r][c]:2}" for c in range(15)))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--from-ply", type=int, default=0)
    args = parser.parse_args()

    d = json.load(open(args.batch))
    rec = d["records"][args.game]
    moves = rec["moves"]
    a_colour = rec["a_colour"]
    print(f"game {args.game}: {d['engine_a']} ({a_colour}) vs "
          f"{d['engine_b']}, winner {rec['winner']}, {rec['plies']} plies")
    print("moves (r,c), X=black first: "
          + " ".join(f"({m // 15},{m % 15})" for m in moves))
    plies = list(range(args.from_ply, len(moves), args.every))
    if not plies or plies[-1] != len(moves):
        plies.append(len(moves))
    for upto in plies:
        print(f"\n--- after {upto} plies "
              f"(last move ({moves[upto - 1] // 15},{moves[upto - 1] % 15}) "
              f"by {'X' if (upto - 1) % 2 == 0 else 'O'}) ---")
        print(render(None, moves, upto, a_colour))
    return 0


if __name__ == "__main__":
    sys.exit(main())

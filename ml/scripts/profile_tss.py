#!/usr/bin/env python3
"""Profile TSS on creator-rich positions to find the per-node hot spot
(bench follow-up: ~2-4ms/node is slower than the kernel calls alone
suggest). Regenerates 20 creator-rich positions (seed 777, same
generator as bench_tss.py) and cProfiles tss_winning_move on them.

Usage: PYTHONPATH=ml python ml/scripts/profile_tss.py
"""

from __future__ import annotations

import cProfile
import pstats
import sys

sys.path.insert(0, "ml/scripts")

from bench_tss import generate_creator_rich_positions


def main() -> int:
    from gomoku_ml.mcts_heuristic import tss_winning_move

    positions = generate_creator_rich_positions(777, 20)
    print(f"profiling TSS on {len(positions)} creator-rich positions",
          file=sys.stderr)
    pr = cProfile.Profile()
    pr.enable()
    for board, player in positions:
        tss_winning_move(board, player, node_budget=100000)
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumulative")
    stats.print_stats(18)
    return 0


if __name__ == "__main__":
    sys.exit(main())

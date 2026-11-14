#!/usr/bin/env python3
"""Tactical suites with FULL move dispatch for a versioned engine.

Unlike ``ml/scripts/eval_version.py`` (whose tactical probe calls
``scoring._choose_move`` directly and therefore bypasses the MCTS
dispatch for v6/v7/v8), this driver runs
``research.run_tactical(player.choose_move, suite)`` — the version's
real move choice, including win/block scans, search, and leaf.

Suites: ``old`` (20 positions, ``tactical_positions.json``),
``hard`` (29 positions, ``tactical_positions_hard.json``), or
``both``. Prints a JSON summary (line starts ``TACTICAL_JSON ``)
plus human-readable lines.

Usage:
    python ml/scripts/eval_tactical_full.py --version 8 --suite both
    python ml/scripts/eval_tactical_full.py --version 8 --suite hard --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import evaluate, research, scoring  # noqa: E402

HARD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "gomoku_ml",
    "tactical_positions_hard.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--suite", default="both",
                        choices=["old", "hard", "both"])
    parser.add_argument("--seed", type=int, default=0,
                        help="player seed (tactical measurement is "
                        "deterministic for a fixed seed)")
    args = parser.parse_args()

    player = scoring.get_player(args.version, seed=args.seed)
    out: dict = {"engine": player.name, "version": args.version,
                 "seed": args.seed}
    suites = ["old", "hard"] if args.suite == "both" else [args.suite]
    for name in suites:
        suite = (
            evaluate.load_tactical_suite(HARD_PATH)
            if name == "hard"
            else evaluate.load_tactical_suite()
        )
        print(f"== tactical suite ({name}, {len(suite['positions'])} "
              f"positions): {player.name} ==", file=sys.stderr)
        result = research.run_tactical(player.choose_move, suite)
        out[name] = result
        print(f"  {player.name}: {result['passed']}/{result['total']} "
              f"({result['pass_rate']:.2%})", file=sys.stderr)
        for r in result["results"]:
            if not r["passed"]:
                print(f"    FAIL {r['id']} ({r['name']}): chose "
                      f"{r['chosen']}, expected {r['expect']}",
                      file=sys.stderr)
    print("TACTICAL_JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

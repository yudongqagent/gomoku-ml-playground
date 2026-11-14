#!/usr/bin/env python3
"""Tactical suites for a versioned engine with FULL move dispatch.

Unlike ``eval_version.py``'s tactical probe (which calls
``scoring._choose_move`` directly and therefore BYPASSES the MCTS
dispatch for versions 6+), this driver uses the player's
``choose_move`` — the same call the match harness uses — for both
the old 20-position suite and the hard 29-position suite
(``tactical_positions_hard.json``).

Usage:
    python ml/scripts/eval_tactical_full.py --version 7 [--seed 0]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    player = scoring.get_player(args.version, seed=args.seed)
    hard_path = os.path.join(
        os.path.dirname(evaluate.__file__), "tactical_positions_hard.json"
    )
    out: dict = {"engine": player.name, "version": args.version,
                 "seed": args.seed}

    old = research.run_tactical(player.choose_move)
    out["old_suite"] = old
    print(f"old suite (20): {old['passed']}/{old['total']} "
          f"({old['pass_rate']:.2%})", file=sys.stderr)
    for r in old["results"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']} ({r['name']}): chose {r['chosen']}, "
                  f"expected {r['expect']}", file=sys.stderr)

    hard = research.run_tactical(
        player.choose_move, evaluate.load_tactical_suite(hard_path)
    )
    out["hard_suite"] = hard
    print(f"hard suite (29): {hard['passed']}/{hard['total']} "
          f"({hard['pass_rate']:.2%})", file=sys.stderr)
    for r in hard["results"]:
        if not r["passed"]:
            print(f"  FAIL {r['id']} ({r['name']}): chose {r['chosen']}, "
                  f"expected {r['expect']}", file=sys.stderr)

    print("TACTICAL_JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

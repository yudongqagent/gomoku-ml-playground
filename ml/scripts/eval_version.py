#!/usr/bin/env python3
"""Ladder + tactical eval for a versioned scoring engine.

Runs the eval ladder (random / frozen baseline heuristic / heuristic
MCTS) and the tactical suite for ``--version`` (default 2), printing
a human-readable summary plus a JSON blob (line starts ``EVAL_JSON ``).
Use the same ``--ladder-games`` / ``--mcts-sims`` for every version
being compared — the "better" bar requires identical game counts.

Usage:
    python ml/scripts/eval_version.py --version 2 --ladder-games 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import baseline, research, scoring  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--ladder-games", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    player = scoring.get_player(args.version, seed=args.seed)
    out: dict = {"engine": player.name, "version": args.version}

    print(f"== tactical suite (20 positions): {player.name} ==",
          file=sys.stderr)
    tac = research.run_tactical(
        lambda board, p: scoring._choose_move(
            board, p, None, player.defence_weight
        )
    )
    out["tactical"] = tac
    print(f"  {player.name}: {tac['passed']}/{tac['total']} "
          f"({tac['pass_rate']:.2%})", file=sys.stderr)
    for r in tac["results"]:
        if not r["passed"]:
            print(f"    FAIL {r['id']} ({r['name']}): chose {r['chosen']}, "
                  f"expected {r['expect']}", file=sys.stderr)

    print(f"== ladder for {player.name} ({args.ladder_games} "
          f"games/opponent, mcts_sims={args.mcts_sims}) ==", file=sys.stderr)
    ladder = research.ladder_for_player(
        player.choose_move,
        games_per_opponent=args.ladder_games,
        mcts_sims=args.mcts_sims,
        seed=args.seed + 300,
    )
    out["ladder"] = ladder
    for name, r in ladder["matchups"].items():
        lo, hi = r["win_rate_ci_95"]
        b, w = r["as_black"], r["as_white"]
        print(f"  vs {name}: {r['wins']}W-{r['losses']}L-{r['draws']}D "
              f"(win rate {r['win_rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]; "
              f"as Black {b['wins']}W-{b['losses']}L-{b['draws']}D, "
              f"as White {w['wins']}W-{w['losses']}L-{w['draws']}D)",
              file=sys.stderr)
    elo = ladder["ladder_elo"]
    print(f"  ladder Elo: estimate {elo['estimate']}, interval "
          f"{elo['interval']} (anchors: random=-800, heuristic=0; "
          f"relative to THIS ladder only)", file=sys.stderr)

    print("EVAL_JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

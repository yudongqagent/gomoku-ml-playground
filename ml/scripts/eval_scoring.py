#!/usr/bin/env python3
"""Evaluate the v1 scoring engine against the frozen baseline.

Runs, with fixed seeds for reproducibility:

1. head-to-head: scoring vs the original baseline heuristic, both
   colours, ``--games`` games (default 100);
2. tactical suite for both engines (same 20 positions);
3. the eval ladder for the scoring engine (random / heuristic / mcts),
   ``--ladder-games`` per opponent (default 10);
4. a swap-rule match, scoring vs baseline, ``--swap-games`` games
   (default 20, 0 to skip).

Prints a human-readable summary plus a JSON blob on stdout (the JSON
line starts with ``EVAL_JSON ``). The head-to-head and ladder numbers
are what the research program uses to decide whether an iteration
improves on the baseline.

Usage:
    python ml/scripts/eval_scoring.py [--games N] [--ladder-games N]
        [--mcts-sims N] [--swap-games N] [--seed N]
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


def _fmt_match(r: dict) -> str:
    lo, hi = r["win_rate_ci_95"]
    b, w = r["as_black"], r["as_white"]
    return (
        f"{r['wins']}W-{r['losses']}L-{r['draws']}D "
        f"(win rate {r['win_rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]; "
        f"as Black {b['wins']}W-{b['losses']}L-{b['draws']}D, "
        f"as White {w['wins']}W-{w['losses']}L-{w['draws']}D)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="head-to-head games, scoring vs baseline (default 100)")
    parser.add_argument("--ladder-games", type=int, default=10,
                        help="ladder games per opponent (default 10)")
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="MCTS simulations for the ladder mcts opponent (default 100)")
    parser.add_argument("--swap-games", type=int, default=20,
                        help="swap-rule match games (default 20, 0 to skip)")
    parser.add_argument("--seed", type=int, default=100,
                        help="base seed (scoring uses seed, baseline seed+100)")
    args = parser.parse_args()

    out: dict = {"engine": scoring.ENGINE_NAME, "seed": args.seed}

    print(f"== head-to-head: {scoring.ENGINE_NAME} vs baseline heuristic "
          f"({args.games} games) ==", file=sys.stderr)
    a = scoring.ScoringPlayer(seed=args.seed)
    b = baseline.HeuristicPlayer(seed=args.seed + 100)
    h2h = research.play_match(a.choose_move, b.choose_move, args.games)
    out["head_to_head_vs_baseline"] = h2h
    print(f"  scoring: {_fmt_match(h2h)}", file=sys.stderr)

    print("== tactical suite (20 positions) ==", file=sys.stderr)
    tac_new = research.run_tactical(
        lambda board, player: scoring.choose_move(board, player)
    )
    tac_base = research.run_tactical(
        lambda board, player: baseline.choose_move(board, player)
    )
    out["tactical_scoring"] = tac_new
    out["tactical_baseline"] = tac_base
    print(f"  scoring:  {tac_new['passed']}/{tac_new['total']} "
          f"({tac_new['pass_rate']:.2%})", file=sys.stderr)
    print(f"  baseline: {tac_base['passed']}/{tac_base['total']} "
          f"({tac_base['pass_rate']:.2%})", file=sys.stderr)
    for r in tac_new["results"]:
        if not r["passed"]:
            print(f"    scoring FAIL {r['id']} ({r['name']}): chose "
                  f"{r['chosen']}, expected {r['expect']}", file=sys.stderr)
    for r in tac_base["results"]:
        if not r["passed"]:
            print(f"    baseline FAIL {r['id']} ({r['name']}): chose "
                  f"{r['chosen']}, expected {r['expect']}", file=sys.stderr)

    print(f"== ladder for {scoring.ENGINE_NAME} "
          f"({args.ladder_games} games/opponent, mcts_sims={args.mcts_sims}) ==",
          file=sys.stderr)
    a2 = scoring.ScoringPlayer(seed=args.seed + 200)
    ladder = research.ladder_for_player(
        a2.choose_move,
        games_per_opponent=args.ladder_games,
        mcts_sims=args.mcts_sims,
        seed=args.seed + 300,
    )
    out["ladder"] = ladder
    for name, r in ladder["matchups"].items():
        print(f"  vs {name}: {_fmt_match(r)}", file=sys.stderr)
    elo = ladder["ladder_elo"]
    print(f"  ladder Elo: estimate {elo['estimate']}, interval "
          f"{elo['interval']} (anchors: random=-800, heuristic=0; "
          f"relative to THIS ladder only)", file=sys.stderr)

    if args.swap_games > 0:
        print(f"== swap-rule match: {scoring.ENGINE_NAME} vs baseline "
              f"({args.swap_games} games) ==", file=sys.stderr)
        a3 = scoring.ScoringPlayer(seed=args.seed + 400)
        b3 = baseline.HeuristicPlayer(seed=args.seed + 500)
        swap = research.play_match(
            a3.choose_move, b3.choose_move, args.swap_games, swap_rule=True
        )
        out["swap_match_vs_baseline"] = swap
        print(f"  scoring: {_fmt_match(swap)}", file=sys.stderr)
        print(f"  swap choices (second player): {swap['swap_choices']}",
              file=sys.stderr)

    print("EVAL_JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

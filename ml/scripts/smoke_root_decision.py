#!/usr/bin/env python3
"""Smoke matches for the v9 root-decision variants (batch 4, H1).

Plays N recorded games between one root-decision variant of v7's
MCTS (``--root-decision``: q / tie-q / blend / tie-prior, all with
the gradient leaf) and v7 itself, alternating colours. Output JSON
has the same shape as ``selfplay_batch.py`` output (``records`` with
full move lists) plus ``root_decision`` / ``variant`` fields, and
is flushed after every game so a VM replacement loses at most the
in-progress game.

The smoke picks which variant gets the full battery (see
``research/iterations/iter-9.md`` §1.2); the battery is the verdict,
never the smoke. Smoke seeds (300/400) are fresh — NOT the
battery's 100/200 — so selection is independent of the verdict.

Usage:
    python ml/scripts/smoke_root_decision.py --root-decision q \\
        --games 20 --seed-a 300 --seed-b 400 \\
        --out research/iterations/smoke-v9q-vs-v7.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from gomoku_ml import evaluate, game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS  # noqa: E402
from selfplay_batch import play_recorded_game  # noqa: E402

VARIANT_NAMES = {
    "q": "v9-A (pure q)",
    "tie-q": "v9-B (tie-q)",
    "blend": "v9-C (blend)",
    "tie-prior": "v9-D (tie-prior)",
}


def _write_result(path: str, result: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-decision", required=True,
                        choices=sorted(VARIANT_NAMES))
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed-a", type=int, default=300)
    parser.add_argument("--seed-b", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    variant = HeuristicMCTS(
        sims=100, leaf_mode="gradient", root_decision=args.root_decision,
        rng=np.random.default_rng(args.seed_a),
    )
    v7 = scoring.get_player(7, seed=args.seed_b)
    name_a = VARIANT_NAMES[args.root_decision]
    name_b = v7.name
    print(f"== smoke: {name_a} vs {name_b}, {args.games} games "
          f"(seeds {args.seed_a}/{args.seed_b}) ==", file=sys.stderr)
    games: list[dict] = []
    wins = losses = draws = 0
    t0 = time.time()
    for i in range(args.games):
        a_is_black = (i % 2 == 0)
        a_colour = game.BLACK if a_is_black else game.WHITE
        if a_is_black:
            winner, moves = play_recorded_game(
                variant.choose_move, v7.choose_move,
                game.empty_board(), game.BLACK,
            )
        else:
            winner, moves = play_recorded_game(
                v7.choose_move, variant.choose_move,
                game.empty_board(), game.BLACK,
            )
        rec = {
            "game": i,
            "moves": moves,
            "a_colour": "black" if a_colour == game.BLACK else "white",
            "winner": (
                "black" if winner == game.BLACK
                else "white" if winner == game.WHITE else "draw"
            ),
            "winner_is_a": (
                (winner == a_colour) if winner != game.EMPTY else None
            ),
            "plies": len(moves),
        }
        if winner == game.EMPTY:
            draws += 1
        elif winner == a_colour:
            wins += 1
        else:
            losses += 1
        games.append(rec)
        lo, hi = evaluate.wilson_ci(wins, losses, draws)
        _write_result(args.out, {
            "engine_a": name_a,
            "engine_b": name_b,
            "variant": args.root_decision,
            "root_decision": args.root_decision,
            "games": args.games,
            "games_played": len(games),
            "complete": (i + 1 == args.games),
            "swap_rule": False,
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
            "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
            "elapsed_seconds": round(time.time() - t0, 1),
            "records": games,
        })
        print(f"  {i + 1}/{args.games} games ({time.time() - t0:.0f}s)",
              file=sys.stderr, flush=True)
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    print(f"{name_a} vs {name_b}: {wins}W-{losses}L-{draws}D, "
          f"win rate {evaluate.win_rate(wins, losses, draws):.4f}, "
          f"95% CI [{lo:.4f}, {hi:.4f}]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

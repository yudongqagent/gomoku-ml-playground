#!/usr/bin/env python3
"""White sims sweep for batch 5 H2 (v13, iter-13 §1.2).

Plays a batch of games between the sweep engine — WhiteMCTS (v7 +
2-ply threat check + colour-asymmetric sims, ported from
research/batch-4-h2) with a chosen White sims count, Black fixed
at 100 — and an opponent (v0 or v7), and saves every game as JSON
in the same shape as selfplay_batch.py output, plus per-colour
splits (mandatory: White is the target).

Usage:
    python ml/scripts/sweep_white_sims.py --white-sims 50 \\
        --opponent v0 --games 40 --seed-a 300 --seed-b 400 \\
        --out research/iterations/sweep-s50-vs-v0.json

Seeds 300/400 are the pre-registered sweep seeds (iter-13 §1.2),
distinct from the bar battery's 100/200 and the fast-loop dev
seeds (1001/2001, ...). Same seeds in every cell, so cells differ
ONLY in the White sims count.
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

from gomoku_ml import evaluate, game  # noqa: E402
from gomoku_ml.mcts_white import WhiteMCTS  # noqa: E402
from selfplay_batch import (  # noqa: E402
    _write_result,
    play_recorded_game,
    resolve_engine,
)


def _splits(games: list[dict]) -> dict:
    """Per-colour W/L/D for engine A from the game records."""
    out = {}
    for colour in ("black", "white"):
        sub = [g for g in games if g["a_colour"] == colour]
        w = sum(1 for g in sub if g["winner_is_a"] is True)
        l = sum(1 for g in sub if g["winner_is_a"] is False)
        d = sum(1 for g in sub if g["winner_is_a"] is None)
        lo, hi = evaluate.wilson_ci(w, l, d)
        out[colour] = {
            "games": len(sub),
            "wins": w,
            "losses": l,
            "draws": d,
            "win_rate": round(evaluate.win_rate(w, l, d), 4),
            "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
        }
    return out


def run_sweep(args) -> dict:
    rng_a = np.random.default_rng(args.seed_a)
    sweep = WhiteMCTS(sims=100, white_sims=args.white_sims, rng=rng_a)
    choose_a = sweep.choose_move
    name_a = f"sweep-ws{args.white_sims}"
    choose_b, name_b = resolve_engine(args.opponent, args.seed_b)
    games: list[dict] = []
    wins = losses = draws = 0
    t0 = time.time()
    for i in range(args.games):
        a_is_black = (i % 2 == 0)
        a_colour = game.BLACK if a_is_black else game.WHITE
        if a_is_black:
            winner, moves = play_recorded_game(
                choose_a, choose_b, game.empty_board(), game.BLACK
            )
        else:
            winner, moves = play_recorded_game(
                choose_b, choose_a, game.empty_board(), game.BLACK
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
        result = {
            "engine_a": name_a,
            "engine_b": name_b,
            "white_sims": args.white_sims,
            "black_sims": 100,
            "games": args.games,
            "games_played": len(games),
            "complete": (i + 1 == args.games),
            "swap_rule": False,
            "placement": None,
            "placement_seed": None,
            "judge": None,
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
            "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
            "as_black": _splits(games)["black"],
            "as_white": _splits(games)["white"],
            "elapsed_seconds": round(time.time() - t0, 1),
            "records": games,
        }
        _write_result(args.out, result)
        if (i + 1) % 10 == 0 or i + 1 == args.games:
            print(f"  {i + 1}/{args.games} games ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--white-sims", type=int, required=True)
    parser.add_argument("--opponent", default="v0", choices=["v0", "v7"])
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-a", type=int, default=300)
    parser.add_argument("--seed-b", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"== sweep: WhiteMCTS white_sims={args.white_sims} "
          f"(Black 100) vs {args.opponent}, {args.games} games "
          f"(seeds {args.seed_a}/{args.seed_b}) ==",
          file=sys.stderr)
    result = run_sweep(args)
    lo, hi = result["win_rate_ci_95"]
    w = result["as_white"]
    b = result["as_black"]
    print(f"  sweep-ws{args.white_sims}: {result['wins']}W-"
          f"{result['losses']}L-{result['draws']}D (win rate "
          f"{result['win_rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]) "
          f"in {result['elapsed_seconds']}s", file=sys.stderr)
    print(f"    as Black: {b['wins']}W-{b['losses']}L-{b['draws']}D "
          f"({b['win_rate']:.4f}); as White: {w['wins']}W-"
          f"{w['losses']}L-{w['draws']}D ({w['win_rate']:.4f})",
          file=sys.stderr)
    print(f"  records written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

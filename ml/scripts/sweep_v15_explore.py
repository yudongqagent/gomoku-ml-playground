#!/usr/bin/env python3
"""Exploration sweep for batch 6 H1 (v15, iter-15 §1.2).

Plays a batch of games between the sweep engine — v7's PUCT MCTS
(capped attack-gradient leaf, 100 sims) with a chosen exploration
parameter — and an opponent (v0 or v7), and saves every game as
JSON in the same shape as selfplay_batch.py output, plus
per-colour splits (mandatory).

One of --prior-floor or --c-puct may be set (not both — the floor
and c_puct are NOT combined; combination is H3's workstream).
Neither set = the control cell (v7 defaults: c_puct 1.5, no
floor), run through this same driver so all cells share one code
path and differ ONLY in the exploration parameter.

Usage:
    python ml/scripts/sweep_v15_explore.py --prior-floor 0.01 \
        --opponent v0 --games 40 --seed-a 300 --seed-b 400 \
        --out research/iterations/sweep-f001-vs-v0.json

Seeds 300/400 are the pre-registered sweep seeds (iter-15 §1.2),
distinct from the bar battery's 100/200 and the fast-loop dev
seeds (1001/2001, ...). Same seeds in every cell, so cells differ
ONLY in the exploration parameter.
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
from gomoku_ml.mcts_heuristic import HeuristicMCTS  # noqa: E402
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


def _summary(args, name_a, name_b, games, wins, losses, draws, t0,
             complete):
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    return {
        "engine_a": name_a,
        "engine_b": name_b,
        "games": args.games,
        "games_played": len(games),
        "complete": complete,
        "swap_rule": False,
        "placement": None,
        "placement_seed": None,
        "judge": None,
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "prior_floor": args.prior_floor,
        "c_puct": args.c_puct,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
        "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
        "splits": _splits(games),
        "elapsed_seconds": round(time.time() - t0, 1),
        "records": games,
    }


def run_sweep(args) -> dict:
    rng_a = np.random.default_rng(args.seed_a)
    sweep = HeuristicMCTS(
        sims=100, c_puct=args.c_puct, prior_floor=args.prior_floor,
        leaf_mode="gradient", rng=rng_a)
    if args.prior_floor > 0.0:
        name_a = f"sweep-floor{args.prior_floor}"
    elif args.c_puct != 1.5:
        name_a = f"sweep-cpuct{args.c_puct}"
    else:
        name_a = "sweep-control-v7"
    choose_a = sweep.choose_move
    _, choose_b, name_b = resolve_engine(args.opponent, args.seed_b)
    games: list[dict] = []
    wins = losses = draws = 0
    t0 = time.time()
    for i in range(args.games):
        a_is_black = (i % 2 == 0)
        a_colour = game.BLACK if a_is_black else game.WHITE
        if a_is_black:
            winner, moves = play_recorded_game(
                choose_a, choose_b, game.empty_board(), game.BLACK)
        else:
            winner, moves = play_recorded_game(
                choose_b, choose_a, game.empty_board(), game.BLACK)
        rec: dict = {
            "game": i,
            "moves": moves,
            "a_colour": "black" if a_colour == game.BLACK else "white",
            "winner": ("black" if winner == game.BLACK
                       else "white" if winner == game.WHITE else "draw"),
            "winner_is_a": (winner == a_colour)
            if winner != game.EMPTY else None,
            "plies": len(moves),
        }
        if winner == game.EMPTY:
            draws += 1
        elif winner == a_colour:
            wins += 1
        else:
            losses += 1
        games.append(rec)
        _write_result(args.out, _summary(
            args, name_a, name_b, games, wins, losses, draws, t0,
            complete=(i + 1 == args.games)))
        if (i + 1) % 10 == 0 or i + 1 == args.games:
            print(f"  {i + 1}/{args.games} games ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    return _summary(args, name_a, name_b, games, wins, losses, draws,
                    t0, complete=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-floor", type=float, default=0.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--opponent", default="v0")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-a", type=int, default=300)
    parser.add_argument("--seed-b", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.prior_floor > 0.0 and args.c_puct != 1.5:
        parser.error("set at most one of --prior-floor and --c-puct "
                     "(no combination in v15; see iter-15 §1.1)")
    print(f"== sweep: prior_floor={args.prior_floor} "
          f"c_puct={args.c_puct} vs {args.opponent}, "
          f"{args.games} games ==", file=sys.stderr)
    result = run_sweep(args)
    lo, hi = result["win_rate_ci_95"]
    print(f"  {result['engine_a']}: {result['wins']}W-"
          f"{result['losses']}L-{result['draws']}D (win rate "
          f"{result['win_rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]) in "
          f"{result['elapsed_seconds']}s", file=sys.stderr)
    print(f"  records written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

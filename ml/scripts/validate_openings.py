#!/usr/bin/env python3
"""Screen opening candidates IN PLAY (openings v2 validation).

For each candidate 4-move opening, plays ``--games-per-opening``
games between two engines with alternating colours (game 0 of the
pair: A Black; game 1: A White — the same paired rotation as the
battery drivers), with per-game reseeding (``seed_a + game_index``,
where ``game_index`` is the absolute index across the whole run,
so a resumed run reproduces the uninterrupted run exactly, as in
``selfplay_batch.py``). Results flush to ``--out`` after every
candidate, and a partial ``--out`` is auto-resumed.

The per-opening White win rate is the bias meter: with A the
stronger engine, White 1/2 over a pair means A won both (neutral
opening); White 0/2 means Black won both (Black-favoured);
White 2/2 means White won both (White-favoured). The screen keeps
candidates near 1/2; confirmation pairs (see
``research/openings-v2-validation.md``) decide the final set.

Usage:
    python ml/scripts/validate_openings.py \\
        --candidates research/openings-v2-candidates.json \\
        --engine-a v52 --engine-b v44 --games-per-opening 2 \\
        --seed-a 41001 --seed-b 42001 --start 0 --end 40 \\
        --out research/iterations/openings-v2-screen-v52-v44-a.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game  # noqa: E402
from selfplay_batch import (apply_opening, load_openings,  # noqa: E402
                            play_recorded_game, reseed_player,
                            resolve_engine)


def _write(out, doc):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    os.replace(tmp, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--engine-a", default="v52")
    parser.add_argument("--engine-b", default="v44")
    parser.add_argument("--games-per-opening", type=int, default=2)
    parser.add_argument("--seed-a", type=int, default=41001)
    parser.add_argument("--seed-b", type=int, default=42001)
    parser.add_argument("--start", type=int, default=0,
                        help="first candidate index (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="last candidate index (exclusive)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    openings, cand_hash = load_openings(args.candidates)
    end = args.end if args.end is not None else len(openings)
    assert 0 <= args.start < end <= len(openings)
    gpo = args.games_per_opening
    assert gpo % 2 == 0, "games-per-opening must be even (paired colours)"

    player_a, choose_a, name_a = resolve_engine(args.engine_a, args.seed_a)
    player_b, choose_b, name_b = resolve_engine(args.engine_b, args.seed_b)

    results: list[dict] = []
    resumed_from = args.start
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            prior = json.load(f)
        if (prior.get("engine_a") == name_a
                and prior.get("engine_b") == name_b
                and prior.get("candidates_hash") == cand_hash
                and prior.get("seed_a") == args.seed_a
                and prior.get("seed_b") == args.seed_b
                and prior.get("games_per_opening") == gpo
                and isinstance(prior.get("results"), list)):
            done = {r["candidate"] for r in prior["results"]}
            results = [r for r in prior["results"]
                       if args.start <= r["candidate"] < end]
            resumed_from = (max(done) + 1) if done else args.start
            print(f"== resuming {args.out} from candidate "
                  f"{resumed_from} ({len(results)} done) ==",
                  file=sys.stderr, flush=True)
        else:
            print(f"== prior {args.out} does not match this run — "
                  f"starting fresh ==", file=sys.stderr, flush=True)

    t0 = time.time()
    for cand in range(max(args.start, resumed_from), end):
        opening = openings[cand]
        board, to_move = apply_opening(opening)
        rec_games = []
        white_wins = 0
        for g in range(gpo):
            gi = cand * gpo + g  # absolute game index (per-game seed)
            reseed_player(player_a, args.seed_a + gi)
            reseed_player(player_b, args.seed_b + gi)
            a_is_black = (g % 2 == 0)
            a_colour = game.BLACK if a_is_black else game.WHITE
            if a_is_black:
                winner, moves = play_recorded_game(
                    choose_a, choose_b, board, to_move)
            else:
                winner, moves = play_recorded_game(
                    choose_b, choose_a, board, to_move)
            if winner == game.WHITE:
                white_wins += 1
            rec_games.append({
                "game": gi,
                "a_colour": "black" if a_colour == game.BLACK else "white",
                "winner": ("black" if winner == game.BLACK
                           else "white" if winner == game.WHITE else "draw"),
                "winner_is_a": ((winner == a_colour)
                                if winner != game.EMPTY else None),
                "plies": len(opening) + len(moves),
                "moves": [int(m) for m in opening] + moves,
            })
        results.append({
            "candidate": cand,
            "opening": [int(m) for m in opening],
            "games": rec_games,
            "white_wins": white_wins,
            "white_win_rate": white_wins / gpo,
        })
        doc = {
            "engine_a": name_a, "engine_b": name_b,
            "candidates": args.candidates,
            "candidates_hash": cand_hash,
            "seed_a": args.seed_a, "seed_b": args.seed_b,
            "games_per_opening": gpo,
            "start": args.start, "end": end,
            "candidates_done": len(results),
            "complete": cand + 1 == end,
            "elapsed_seconds": round(time.time() - t0, 1),
            "results": results,
        }
        _write(args.out, doc)
        print(f"  cand {cand}: white {white_wins}/{gpo} "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(f"== done: {len(results)} candidates, "
          f"{sum(r['white_wins'] for r in results)}/"
          f"{len(results) * gpo} White wins "
          f"({time.time() - t0:.0f}s) ==", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

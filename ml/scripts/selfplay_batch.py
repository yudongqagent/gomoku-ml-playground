#!/usr/bin/env python3
"""Swap-rule / normal self-play batches with full game records.

Plays a batch of games between two engines and saves every game
(moves, winner, colours, opening, swap choice) as JSON for loss
analysis, plus a summary (W/L/D from A's perspective with a Wilson
95% CI).

Engines: ``v0`` = the frozen baseline heuristic,
``v1``/``v2``/... = the versioned scoring engine
(:func:`gomoku_ml.scoring.get_player`).

Usage:
    python ml/scripts/selfplay_batch.py --engine-a v1 --engine-b v0 \\
        --games 100 --swap-rule --placement balanced \\
        --out research/iterations/batch-v1-vs-v0-swap.json

Swap-rule batches use a colour judge (see
``research/iterations/swap-judge-v2.md``): ``--judge tempo`` (the
harness default since batch 2, H3) or ``--judge attack`` (the
batch-1 judge). The judge is recorded in the output JSON.

The output JSON is flushed after every game, and a partial
``--out`` from a killed run is auto-resumed (same engine pair,
game count and swap flag; the placement RNG restarts, so resumed
openings may repeat earlier ones — recorded as
``resumed_from_game`` in the JSON).
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

from gomoku_ml import baseline, evaluate, game, research, scoring  # noqa: E402


def resolve_engine(name: str, seed: int):
    """Build a stateful player for ``v0``/``v1``/``v2``/..."""
    if name == "v0":
        player = baseline.HeuristicPlayer(seed=seed)
        return player.choose_move, "v0"
    if name.startswith("v") and name[1:].isdigit():
        player = scoring.get_player(int(name[1:]), seed=seed)
        return player.choose_move, player.name
    raise ValueError(f"unknown engine: {name!r} (want v0, v1, v2, ...)")


def play_recorded_game(choose_black, choose_white, board, player_to_move):
    """Play to completion from ``board``. Returns (winner, moves)."""
    moves: list[int] = []
    player = player_to_move
    while True:
        over, win = game.game_over(board)
        if over:
            return win, moves
        chooser = choose_black if player == game.BLACK else choose_white
        move = chooser(board, player)
        moves.append(int(move))
        board = game.apply_move(board, move, player)
        player = -player


def _write_result(path: str, result: dict) -> None:
    """Atomically write ``result`` to ``path`` (temp file + rename).

    Matches flush after EVERY game: this VM is replaced without
    warning (4 replacements on 2026-11-14 alone), and a killed run
    must lose at most the in-progress game. The file carries
    ``complete: false`` until the batch finishes; a re-run with the
    same seeds is deterministic, so the flushed prefix also serves
    as a replay check.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _summary(args, name_a, name_b, games, wins, losses, draws, t0,
             complete, resumed=False, resumed_from_game=None):
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    return {
        "engine_a": name_a,
        "engine_b": name_b,
        "games": args.games,
        "games_played": len(games),
        "complete": complete,
        "resumed": resumed,
        "resumed_from_game": resumed_from_game,
        "swap_rule": args.swap_rule,
        "placement": args.placement if args.swap_rule else None,
        "placement_seed": args.placement_seed if args.swap_rule else None,
        "judge": args.judge if args.swap_rule else None,
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
        "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
        "elapsed_seconds": round(time.time() - t0, 1),
        "records": games,
    }


def run_batch(args) -> dict:
    choose_a, name_a = resolve_engine(args.engine_a, args.seed_a)
    choose_b, name_b = resolve_engine(args.engine_b, args.seed_b)
    placement_rng = np_random(args.placement_seed)
    scorer = scoring.score_attack
    games: list[dict] = []
    wins = losses = draws = 0
    resumed = False
    resumed_from_game = None
    # Auto-resume: this VM is replaced without warning, so a partial
    # --out from a killed run is continued, not restarted. The
    # placement RNG restarts (openings may repeat earlier ones);
    # alternation is preserved because the loop index is absolute.
    if os.path.exists(args.out):
        try:
            with open(args.out, encoding="utf-8") as f:
                prior = json.load(f)
            if (prior.get("engine_a") == name_a
                    and prior.get("engine_b") == name_b
                    and prior.get("games") == args.games
                    and prior.get("swap_rule") == args.swap_rule
                    and isinstance(prior.get("records"), list)):
                if prior.get("complete"):
                    print(f"== {args.out} already complete "
                          f"({prior['games_played']} games) — nothing to do ==",
                          file=sys.stderr)
                    return prior
                games = prior["records"]
                wins = prior["wins"]
                losses = prior["losses"]
                draws = prior["draws"]
                resumed = True
                resumed_from_game = len(games)
                print(f"== resuming from game {resumed_from_game} "
                      f"({wins}W-{losses}L-{draws}D so far; RNG restarts) ==",
                      file=sys.stderr)
        except (OSError, ValueError, KeyError) as e:
            print(f"== ignoring unreadable prior {args.out}: {e} ==",
                  file=sys.stderr)
    t0 = time.time()
    for i in range(len(games), args.games):
        rec: dict = {"game": i}
        if args.swap_rule:
            a_opens = (i % 2 == 0)
            opener, second = (choose_a, choose_b) if a_opens else (choose_b, choose_a)
            board, opening = research.place_swap_opening(
                opener, args.placement, scorer, placement_rng
            )
            gap, bb, bw = research.opening_gap(board, scorer)
            second_colour = game.choose_swap_color(
                board, scorer, judge=args.judge)
            if second_colour == game.BLACK:
                choose_black, choose_white = second, opener
            else:
                choose_black, choose_white = opener, second
            winner, moves = play_recorded_game(
                choose_black, choose_white, board, game.WHITE
            )
            a_colour = (
                game.BLACK
                if (a_opens and second_colour == game.WHITE)
                or ((not a_opens) and second_colour == game.BLACK)
                else game.WHITE
            )
            rec.update({
                "opener": "A" if a_opens else "B",
                "opening": [int(m) for m in opening],
                "opening_gap": gap,
                "opening_best_black": bb,
                "opening_best_white": bw,
                "second_colour": "black" if second_colour == game.BLACK else "white",
                "moves": [int(m) for m in opening] + moves,
            })
        else:
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
            rec["moves"] = moves
        rec["a_colour"] = "black" if a_colour == game.BLACK else "white"
        rec["winner"] = (
            "black" if winner == game.BLACK
            else "white" if winner == game.WHITE else "draw"
        )
        rec["winner_is_a"] = (winner == a_colour) if winner != game.EMPTY else None
        rec["plies"] = len(rec["moves"])
        if winner == game.EMPTY:
            draws += 1
        elif winner == a_colour:
            wins += 1
        else:
            losses += 1
        games.append(rec)
        _write_result(args.out, _summary(
            args, name_a, name_b, games, wins, losses, draws, t0,
            complete=(i + 1 == args.games),
            resumed=resumed, resumed_from_game=resumed_from_game))
        if (i + 1) % 10 == 0 or i + 1 == args.games:
            print(f"  {i + 1}/{args.games} games ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    return _summary(args, name_a, name_b, games, wins, losses, draws,
                    t0, complete=True,
                    resumed=resumed, resumed_from_game=resumed_from_game)


def np_random(seed: int):
    import numpy as np

    return np.random.default_rng(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-a", default="v1")
    parser.add_argument("--engine-b", default="v0")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--swap-rule", action="store_true")
    parser.add_argument("--placement", default="balanced",
                        choices=["chooser", "balanced"])
    parser.add_argument("--placement-seed", type=int, default=0)
    parser.add_argument("--judge", default="tempo",
                        choices=["attack", "tempo"],
                        help="swap colour judge (swap-rule batches only)")
    parser.add_argument("--seed-a", type=int, default=100)
    parser.add_argument("--seed-b", type=int, default=200)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"== batch: {args.engine_a} vs {args.engine_b}, "
          f"{args.games} games"
          + (f", swap rule ({args.placement} placement, {args.judge} judge)"
             if args.swap_rule else ", normal rules") + " ==",
          file=sys.stderr)
    result = run_batch(args)  # flushes the output JSON after every game
    lo, hi = result["win_rate_ci_95"]
    print(f"  {args.engine_a}: {result['wins']}W-{result['losses']}L-"
          f"{result['draws']}D (win rate {result['win_rate']:.4f}, "
          f"95% CI [{lo:.4f}, {hi:.4f}]) in {result['elapsed_seconds']}s",
          file=sys.stderr)
    print(f"  records written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

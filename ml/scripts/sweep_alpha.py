#!/usr/bin/env python3
"""Alpha sweep for v31 (ML batch 2 H2): pick the value_weight shrink.

Plays N games vs an opponent (default v8, the ML bar reference) for
each alpha in the sweep set, writing one fast_loop_match-shaped JSON
per alpha. The winner is selected by win rate (tie-break: smaller
alpha — less optimizer's-curse exposure); the 100-game battery is
the verdict, not this sweep.

The sweep sets ``player._mcts.value_weight`` directly per run, so
the registry entry is touched only once (to the winner) afterwards.
Ported from ML batch 1 H1's sweep_alpha.py (v24), with the engine
version parameterised (``--version``).

Usage:
    python ml/scripts/sweep_alpha.py --version 31 \\
        --alphas 0.5 1.0 2.0 \\
        --games 20 --opponent v8 --seed-a 1001 --seed-b 2001 \\
        --out-dir research/iterations
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

from gomoku_ml import evaluate, game, scoring  # noqa: E402
from fast_loop_match import resolve_opponent  # noqa: E402
from selfplay_batch import play_recorded_game  # noqa: E402


def _write_result(alpha, args, name_b, games, wins, losses, draws,
                  t0, complete, resumed=False, resumed_from_game=None):
    """Atomically write the current sweep state for one alpha.

    Flushed after every game (like fast_loop_match) so a VM
    replacement loses at most the in-progress game; the final write
    (complete=True) has exactly the pre-registered shape.
    """
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    result = {
        "engine_a": f"scoring-v{args.version}",
        "engine_b": name_b,
        "alpha": alpha,
        "games": args.games,
        "games_played": len(games),
        "complete": complete,
        "swap_rule": False,
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
        "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
        "elapsed_seconds": round(time.time() - t0, 1),
        "resumed": resumed,
        "resumed_from_game": resumed_from_game,
        "records": games,
    }
    out = os.path.join(
        args.out_dir, f"alpha-v{args.version}-{alpha}-vs-{args.opponent}.json")
    os.makedirs(args.out_dir, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, out)
    return result, out


def run_match(alpha: float, args) -> dict:
    out = os.path.join(
        args.out_dir, f"alpha-v{args.version}-{alpha}-vs-{args.opponent}.json")
    games: list[dict] = []
    wins = losses = draws = 0
    start_game = 0
    resumed = False
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("complete"):
            print(f"== alpha {alpha}: already complete "
                  f"({prev['wins']}W-{prev['losses']}L-{prev['draws']}D), "
                  f"skipping ==", file=sys.stderr)
            return prev
        # Resume a partial alpha: the saved games stand (played under
        # the pre-registered protocol); the remaining games continue
        # with fresh players, so the MCTS tie-break RNG restarts
        # instead of continuing the pre-kill stream. The 20 games
        # remain 20 fair games vs the opponent at this alpha with
        # alternating colours; the restart is recorded in the JSON
        # (resumed_from_game) and the iteration log.
        games = prev["records"]
        wins, losses, draws = prev["wins"], prev["losses"], prev["draws"]
        start_game = len(games)
        resumed = True
        print(f"== alpha {alpha}: resuming from game {start_game} "
              f"({wins}W-{losses}L-{draws}D so far; RNG restarts) ==",
              file=sys.stderr)
    player_a = scoring.get_player(args.version, seed=args.seed_a)
    player_a._mcts.value_weight = alpha
    choose_a = player_a.choose_move
    # NB: resolve_opponent returns (player, choose_move, name) since
    # the batch-3 per-game-seeding change; the player object is
    # unused here (the sweep keeps v24/v31's fixed-seed protocol —
    # it only SELECTS alpha, and all arms share the protocol, so the
    # comparison is fair within the sweep; the battery drivers carry
    # the per-game seeding).
    _player_b, choose_b, name_b = resolve_opponent(args.opponent, args.seed_b)
    print(f"== alpha {alpha}: scoring-v{args.version} vs {name_b}, {args.games} games "
          f"(seeds {args.seed_a}/{args.seed_b}) ==", file=sys.stderr)
    t0 = time.time()
    for i in range(start_game, args.games):
        a_is_black = (i % 2 == 0)
        a_colour = game.BLACK if a_is_black else game.WHITE
        if a_is_black:
            winner, moves = play_recorded_game(
                choose_a, choose_b, game.empty_board(), game.BLACK)
        else:
            winner, moves = play_recorded_game(
                choose_b, choose_a, game.empty_board(), game.BLACK)
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
        print(f"  game {i}: {rec['winner']} ({rec['plies']} plies, "
              f"A {rec['a_colour']}) ({time.time() - t0:.0f}s)",
              file=sys.stderr, flush=True)
        _write_result(alpha, args, name_b, games, wins, losses, draws,
                      t0, complete=False, resumed=resumed,
                      resumed_from_game=start_game if resumed else None)
    result, out = _write_result(alpha, args, name_b, games, wins,
                                losses, draws, t0, complete=True,
                                resumed=resumed,
                                resumed_from_game=start_game if resumed else None)
    lo, hi = result["win_rate_ci_95"]
    print(f"  alpha {alpha}: {wins}W-{losses}L-{draws}D (win rate "
          f"{result['win_rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]) -> {out}",
          file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=31,
                        help="engine version to sweep (default 31)")
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", default="v8")
    parser.add_argument("--seed-a", type=int, default=1001)
    parser.add_argument("--seed-b", type=int, default=2001)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    results = [run_match(a, args) for a in args.alphas]
    best = max(results, key=lambda r: (r["win_rate"], -r["alpha"]))
    print("SWEEP_JSON " + json.dumps({
        "opponent": args.opponent, "games": args.games,
        "results": [
            {"alpha": r["alpha"], "win_rate": r["win_rate"],
             "wins": r["wins"], "losses": r["losses"], "draws": r["draws"]}
            for r in results
        ],
        "winner": best["alpha"],
    }))
    print(f"sweep winner: alpha {best['alpha']} "
          f"(win rate {best['win_rate']:.4f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Fast-loop matches: N recorded games between v8 and a weakness-exposing opponent.

Like :mod:`ml.scripts.selfplay_batch` but the opponent may also be the
heuristic-MCTS ladder opponent (``mcts`` — not addressable through
``selfplay_batch.py``'s ``vN`` engine names). Output JSON has the same
shape as ``selfplay_batch.py`` output (``records`` with full move
lists), so ``classify_losses.py`` and ``workup_loss.py`` work on it
unchanged. Resumable: if ``--out`` already holds a partial (incomplete)
match, the saved games stand and the remaining games continue — and
per-game seeding (see ``selfplay_batch.py``) makes the resumed run
reproduce the uninterrupted run exactly. A complete ``--out`` is
left alone.

Fast-loop seed policy (recorded in research/iterations/iter-8.md §4):
iteration k uses seed-a = 1000 + k, seed-b = 2000 + k — fresh games,
never the validation battery's seeds (100/200).

Usage:
    python ml/scripts/fast_loop_match.py --opponent v0 --games 10 \\
        --seed-a 1001 --seed-b 2001 --out research/iterations/fast-v8-vs-v0-iter1.json
    python ml/scripts/fast_loop_match.py --opponent mcts --games 10 \\
        --seed-a 1001 --seed-b 2001 --out research/iterations/fast-v8-vs-mcts-iter1.json
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

from gomoku_ml import baseline, evaluate, game, scoring  # noqa: E402
from selfplay_batch import play_recorded_game, reseed_player  # noqa: E402


def resolve_opponent(name: str, seed: int):
    """Build a stateful opponent player: ``v0``/``v1``/... or ``mcts``
    (the heuristic-MCTS ladder opponent, 100 sims). Returns
    ``(player, choose_move, name)`` — the player object is kept so
    the driver can reseed it before every game (per-game seeding;
    see ``selfplay_batch.py``)."""
    if name == "mcts":
        player = baseline.HeuristicMCTSPlayer(sims=100, seed=seed)
        return player, player.choose_move, "mcts"
    if name == "v0":
        player = baseline.HeuristicPlayer(seed=seed)
        return player, player.choose_move, "v0"
    if name.startswith("v") and name[1:].isdigit():
        player = scoring.get_player(int(name[1:]), seed=seed)
        return player, player.choose_move, player.name
    raise ValueError(f"unknown opponent: {name!r} (want v0, v1, ..., mcts)")


def _write_result(args, name_a, name_b, games, wins, losses, draws,
                  t0, complete, resumed=False, resumed_from_game=None):
    """Atomically write the current batch state to ``args.out``.

    Same shape as ``selfplay_batch.py`` output, plus ``games_played``
    and ``complete``. Flushed after every game so a VM replacement
    loses at most the in-progress game.
    """
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    result = {
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
        "seeding": "per-game",
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
        "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
        "uniqueness": evaluate.uniqueness_audit(games),
        "elapsed_seconds": round(time.time() - t0, 1),
        "resumed": resumed,
        "resumed_from_game": resumed_from_game,
        "records": games,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, args.out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-a", type=int, default=8,
                        help="engine version for side A (default 8)")
    parser.add_argument("--opponent", default="v0",
                        help="opponent: v0, v1, ..., or mcts")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed-a", type=int, default=1001)
    parser.add_argument("--seed-b", type=int, default=2001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    player_a = scoring.get_player(args.version_a, seed=args.seed_a)
    choose_a = player_a.choose_move
    player_b, choose_b, name_b = resolve_opponent(args.opponent, args.seed_b)
    print(f"== fast loop: {player_a.name} vs {name_b}, {args.games} games "
          f"(seeds {args.seed_a}/{args.seed_b}) ==", file=sys.stderr)
    games: list[dict] = []
    wins = losses = draws = 0
    start_game = 0
    resumed = False
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("complete"):
            print(f"== already complete ({prev['wins']}W-{prev['losses']}L-"
                  f"{prev['draws']}D), skipping ==", file=sys.stderr)
            return 0
        # Resume a partial match: the saved games stand (played under
        # the registered protocol); the remaining games continue with
        # per-game seeding, so the resumed run reproduces the
        # uninterrupted run exactly (see selfplay_batch.py). The
        # restart is recorded in the JSON (resumed_from_game) and
        # the iteration log.
        games = prev["records"]
        wins, losses, draws = prev["wins"], prev["losses"], prev["draws"]
        start_game = len(games)
        resumed = True
        print(f"== resuming from game {start_game} "
              f"({wins}W-{losses}L-{draws}D so far; per-game seeding "
              f"reproduces the uninterrupted run) ==",
              file=sys.stderr)
    t0 = time.time()
    for i in range(start_game, args.games):
        # Per-game seeding: each game is determined by its index
        # alone (base_seed + i) — see selfplay_batch.py.
        reseed_player(player_a, args.seed_a + i)
        reseed_player(player_b, args.seed_b + i)
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
        print(f"  game {i}: {rec['winner']} "
              f"({rec['plies']} plies, A {rec['a_colour']}) "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
        # Flush after EVERY game: a VM replacement must lose at most
        # the in-progress game (see selfplay_batch.py).
        _write_result(args, player_a.name, name_b, games, wins, losses,
                      draws, t0, complete=(i + 1 == args.games),
                      resumed=resumed,
                      resumed_from_game=start_game if resumed else None)
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    print(f"  {player_a.name}: {wins}W-{losses}L-{draws}D (win rate "
          f"{evaluate.win_rate(wins, losses, draws):.4f}, "
          f"95% CI [{lo:.4f}, {hi:.4f}]) in "
          f"{round(time.time() - t0, 1)}s", file=sys.stderr)
    print(f"  records written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

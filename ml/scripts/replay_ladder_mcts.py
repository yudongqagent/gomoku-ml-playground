#!/usr/bin/env python3
"""Replay a ladder matchup with full game records.

The ladder (``research.ladder_for_player`` via ``eval_version.py``)
does not save game records, so its games cannot be worked up. This
script reproduces ONE ladder matchup EXACTLY — same player seeds,
same warm-up legs (the version player's RNG advances through the
random and heuristic legs before the mcts leg, exactly as in the
ladder) — and records every game of the chosen leg with full move
lists, in the ``fast_loop_match.py`` output shape, so
``classify_losses.py`` and ``workup_loss.py`` work on it unchanged.

Exactness: the game loop mirrors ``research.play_game_from``
(call order and chooser arguments identical), the warm-up legs
use ``research.play_match`` directly, and seeds follow
``eval_version.py`` (version player ``seed``; opponents
``seed + 300``). A replay of the v22 ladder mcts leg must
reproduce the recorded 5W-15L exactly — that is the port check.

Resumable: RNG state (``bit_generator.state``) of both players
is snapshotted to ``--out + \".rng.json\"`` after every recorded
game. On relaunch with an existing partial ``--out``, the
snapshotted RNG states are restored and the remaining games
continue — no warm-up re-run needed, no RNG restart (unlike the
other drivers, which restart the RNG on resume). The warm-up
legs run only on a fresh start.

Usage:
    python ml/scripts/replay_ladder_mcts.py --version 22 --seed 100 \\
        --out research/iterations/forensics-v22-vs-mcts-ladder.json
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
from gomoku_ml import baseline, evaluate, game, research, scoring  # noqa: E402


def _play_recorded(choose_black, choose_white) -> tuple[list[int], int]:
    """Play one game, mirroring research.play_game_from exactly
    (same call order) while recording moves. Returns (moves,
    winner)."""
    board = game.empty_board()
    moves: list[int] = []
    player = game.BLACK
    for _ in range(game.SIZE * game.SIZE):
        over, win = game.game_over(board)
        if over:
            return moves, win
        chooser = choose_black if player == game.BLACK else choose_white
        move = chooser(board, player)
        moves.append(int(move))
        board = game.apply_move(board, move, player)
        player = -player
    _, win = game.game_over(board)
    return moves, win


def _rng_state(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def _restore_rng(rng: np.random.Generator, state: dict) -> None:
    rng.bit_generator.state = state


def _flush(args, games, wins, losses, draws, t0, name_a, name_b,
           resumed, resumed_from_game, player_a, mcts_player) -> None:
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    result = {
        "engine_a": name_a,
        "engine_b": name_b,
        "games": args.ladder_games,
        "games_played": len(games),
        "complete": len(games) >= args.ladder_games,
        "swap_rule": False,
        "placement": None,
        "placement_seed": None,
        "judge": None,
        "seed_a": args.seed,
        "seed_b": args.seed + 300,
        "ladder_replay": True,
        "ladder_seed": args.seed,
        "opponent_leg": "mcts",
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
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, args.out)
    # RNG snapshot sidecar (NOT part of the committed artefact shape).
    rng_sidecar = {
        "games_played": len(games),
        "rng_a": _rng_state(player_a.rng),
        "rng_b": _rng_state(mcts_player.rng),
    }
    with open(args.out + ".rng.json.tmp", "w", encoding="utf-8") as f:
        json.dump(rng_sidecar, f)
    os.replace(args.out + ".rng.json.tmp", args.out + ".rng.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=22)
    parser.add_argument("--seed", type=int, default=100,
                        help="eval_version.py --seed (ladder seed)")
    parser.add_argument("--ladder-games", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    opp_seed = args.seed + 300
    player_a = scoring.get_player(args.version, seed=args.seed)
    mcts_player = baseline.HeuristicMCTSPlayer(
        sims=args.mcts_sims, seed=opp_seed)
    name_a = player_a.name
    name_b = "mcts"

    games: list[dict] = []
    wins = losses = draws = 0
    start_game = 0
    resumed = False
    resumed_from_game = None
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            prev = json.load(f)
        if prev.get("complete"):
            print(f"== already complete ({prev['wins']}W-"
                  f"{prev['losses']}L-{prev['draws']}D), skipping ==",
                  file=sys.stderr)
            return 0
        sidecar_path = args.out + ".rng.json"
        if not os.path.exists(sidecar_path):
            print("== partial --out without RNG sidecar; refusing to "
                  "resume (delete the partial to restart) ==",
                  file=sys.stderr)
            return 2
        with open(sidecar_path, encoding="utf-8") as f:
            sidecar = json.load(f)
        if sidecar["games_played"] != len(prev["records"]):
            print("== RNG sidecar disagrees with partial --out; "
                  "refusing to resume ==", file=sys.stderr)
            return 2
        games = prev["records"]
        wins, losses, draws = prev["wins"], prev["losses"], prev["draws"]
        start_game = len(games)
        resumed = True
        resumed_from_game = start_game
        _restore_rng(player_a.rng, sidecar["rng_a"])
        _restore_rng(mcts_player.rng, sidecar["rng_b"])
        print(f"== resuming from game {start_game} "
              f"({wins}W-{losses}L-{draws}D so far; RNG state restored "
              f"exactly — no warm-up re-run) ==", file=sys.stderr)
    else:
        # Fresh start: warm-up legs, exactly as ladder_for_player
        # runs them (random then heuristic, same seeds), so the
        # version player's RNG is in the same state as at the start
        # of the ladder's mcts leg.
        t_warm = time.time()
        random_opp = baseline.RandomPlayer(seed=opp_seed)
        research.play_match(player_a.choose_move,
                            random_opp.choose_move,
                            args.ladder_games, seed=opp_seed)
        print(f"== warm-up leg random done "
              f"({time.time() - t_warm:.0f}s) ==", file=sys.stderr)
        heuristic_opp = baseline.HeuristicPlayer(seed=opp_seed)
        research.play_match(player_a.choose_move,
                            heuristic_opp.choose_move,
                            args.ladder_games, seed=opp_seed)
        print(f"== warm-up leg heuristic done "
              f"({time.time() - t_warm:.0f}s) ==", file=sys.stderr)

    print(f"== ladder replay: {name_a} vs {name_b}, "
          f"{args.ladder_games} games (ladder seed {args.seed}) ==",
          file=sys.stderr)
    t0 = time.time()
    for i in range(start_game, args.ladder_games):
        t_game = time.time()
        a_is_black = (i % 2 == 0)
        if a_is_black:
            moves, winner = _play_recorded(
                player_a.choose_move, mcts_player.choose_move)
        else:
            moves, winner = _play_recorded(
                mcts_player.choose_move, player_a.choose_move)
        a_colour = game.BLACK if a_is_black else game.WHITE
        if winner == game.EMPTY:
            draws += 1
            winner_is_a = None
        elif winner == a_colour:
            wins += 1
            winner_is_a = True
        else:
            losses += 1
            winner_is_a = False
        games.append({
            "game": i,
            "moves": moves,
            "a_colour": "black" if a_is_black else "white",
            "winner": ("black" if winner == game.BLACK else
                       "white" if winner == game.WHITE else "draw"),
            "winner_is_a": winner_is_a,
            "plies": len(moves),
        })
        _flush(args, games, wins, losses, draws, t0, name_a, name_b,
               resumed, resumed_from_game, player_a, mcts_player)
        print(f"  game {i}: {games[-1]['winner']} "
              f"({len(moves)} plies, A "
              f"{'black' if a_is_black else 'white'}) "
              f"({time.time() - t_game:.0f}s)", file=sys.stderr)
    print(f"== done: {wins}W-{losses}L-{draws}D, win rate "
          f"{evaluate.win_rate(wins, losses, draws):.4f} ==",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""M2 for v58 (iter-58 §1.3): replay the recorded v52-vs-v27 battery
games (normal + swap) and run production TSS at every turn — TSS fire
/ differ rates in the ACTUAL battery games, by (engine, colour).

Procedure (pre-registered in iter-58 §1.3 BEFORE this script ran):

* Replay the recorded move lists of batch-v52-vs-v27.json (normal)
  and batch-v52-vs-v27-swap.json (swap). Swap games: moves[0:3] are
  the opening stones (placed, not chosen); the replay starts at ply
  3 with White to move.
* At every turn run production TSS (depth 4, budget 3000) for the
  side to move and record, by (engine, colour): turns, fires,
  confirms (TSS move == recorded move), differs (TSS move !=
  recorded move), budget-exceeded count, node sum.
* CONSISTENCY CHECK: for v52's turns, a TSS fire MUST coincide with
  v52's recorded move (v52 plays the TSS move when it fires). Any
  violation is counted separately and must be investigated as a
  replay bug before the numbers are used (expect 0).

Deterministic (TSS is deterministic, the games are recorded), so it
checkpoints after every game and resumes exactly.

Usage:
    PYTHONPATH=ml python ml/scripts/replay_tss_battery.py
        --in research/iterations/batch-v52-vs-v27.json
        --out research/iterations/tss-replay-v58-normal.json
    PYTHONPATH=ml python ml/scripts/replay_tss_battery.py
        --in research/iterations/batch-v52-vs-v27-swap.json
        --out research/iterations/tss-replay-v58-swap.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def _new_stats():
    return {
        "turns": 0,
        "fires": 0,
        "confirms": 0,
        "differs": 0,
        "budget_exceeded": 0,
        "nodes": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--white-only", action="store_true",
                    help="run TSS only on White turns (see iter-58 "
                    "§2.2: M2 scope amendment — both decision "
                    "quantities are White-turn quantities)")
    ap.add_argument("--per-game", action="store_true",
                    help="record per-game White-turn TSS stats plus "
                    "the game outcome (M2b attribution analysis)")
    ap.add_argument("--only-a-colour", choices=["black", "white"],
                    default=None,
                    help="replay only games where engine A plays this "
                    "colour (M2b: v52-White games only)")
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic

    in_path = args.inp
    if not os.path.isabs(in_path):
        in_path = os.path.join(REPO_ROOT, in_path)
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    ckpt_path = args.checkpoint or (out_path + ".checkpoint.json")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    with open(in_path, encoding="utf-8") as f:
        batch = json.load(f)
    records = batch["records"]
    swap_rule = bool(batch.get("swap_rule"))
    engine_a = batch.get("engine_a")
    engine_b = batch.get("engine_b")

    stats = {
        "A": {"black": _new_stats(), "white": _new_stats()},
        "B": {"black": _new_stats(), "white": _new_stats()},
    }
    violations = []  # v52-turn fires where the recorded move != TSS move
    per_game = []
    start_gi = 0
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if (ckpt.get("input") == in_path and ckpt.get(
                    "n_records") == len(records)
                    and ckpt.get("white_only") == args.white_only
                    and ckpt.get("per_game") == args.per_game
                    and ckpt.get("only_a_colour") == args.only_a_colour):
                stats = ckpt["stats"]
                violations = ckpt["violations"]
                per_game = ckpt.get("per_game_records", [])
                start_gi = ckpt["games_done"]
                print(f"resuming from checkpoint: {start_gi}/"
                      f"{len(records)} games done", file=sys.stderr)
        except (OSError, ValueError, KeyError):
            print("checkpoint unreadable — starting from scratch",
                  file=sys.stderr)

    for gi in range(start_gi, len(records)):
        rec = records[gi]
        if (args.only_a_colour is not None
                and rec["a_colour"] != args.only_a_colour):
            continue
        moves = rec["moves"]
        a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
        board = game.empty_board()
        start = 0
        if swap_rule:
            # moves[0:3] are the placed opening stones.
            for i in range(3):
                board = game.apply_move(
                    board, moves[i],
                    game.BLACK if i % 2 == 0 else game.WHITE)
            start = 3
        game_stats = {
            "A": {"black": _new_stats(), "white": _new_stats()},
            "B": {"black": _new_stats(), "white": _new_stats()},
        }
        for i in range(start, len(moves)):
            player = game.BLACK if i % 2 == 0 else game.WHITE
            if args.white_only and player != game.WHITE:
                board = game.apply_move(board, moves[i], player)
                continue
            engine = "A" if player == a_colour else "B"
            key = "black" if player == game.BLACK else "white"
            tss_stats: dict = {}
            tss_move = mcts_heuristic.tss_winning_move(
                board, player, stats=tss_stats)
            cs = stats[engine][key]
            gs = game_stats[engine][key]
            for s in (cs, gs):
                s["turns"] += 1
                s["nodes"] += tss_stats.get("nodes", 0)
                if tss_stats.get("budget_exceeded"):
                    s["budget_exceeded"] += 1
            if tss_move is not None:
                for s in (cs, gs):
                    s["fires"] += 1
                if tss_move == moves[i]:
                    for s in (cs, gs):
                        s["confirms"] += 1
                else:
                    for s in (cs, gs):
                        s["differs"] += 1
                    if engine == "A":
                        # v52 plays the TSS move when it fires — a
                        # mismatch is a replay bug, not a differ.
                        violations.append({
                            "game": gi,
                            "ply": i,
                            "colour": key,
                            "tss_move": tss_move,
                            "recorded_move": moves[i],
                        })
            board = game.apply_move(board, moves[i], player)
        if args.per_game:
            per_game.append({
                "game": gi,
                "a_colour": rec["a_colour"],
                "winner": rec["winner"],
                "winner_is_a": rec["winner_is_a"],
                "plies": rec["plies"],
                "A_white_turns": game_stats["A"]["white"]["turns"],
                "A_white_fires": game_stats["A"]["white"]["fires"],
                "B_white_turns": game_stats["B"]["white"]["turns"],
                "B_white_fires": game_stats["B"]["white"]["fires"],
                "B_white_differs": game_stats["B"]["white"]["differs"],
            })
        print(f"game {gi + 1}/{len(records)}: A-white fires="
              f"{stats['A']['white']['fires']}/"
              f"{stats['A']['white']['turns']} A-black fires="
              f"{stats['A']['black']['fires']}/"
              f"{stats['A']['black']['turns']} violations="
              f"{len(violations)}", file=sys.stderr)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "input": in_path,
                "n_records": len(records),
                "white_only": args.white_only,
                "per_game": args.per_game,
                "only_a_colour": args.only_a_colour,
                "games_done": gi + 1,
                "stats": stats,
                "violations": violations,
                "per_game_records": per_game,
            }, f)

    def _rates(cs):
        t = cs["turns"]
        return {
            **cs,
            "fire_rate": cs["fires"] / t if t else 0.0,
            "differ_rate": cs["differs"] / t if t else 0.0,
            "mean_nodes": cs["nodes"] / t if t else 0.0,
        }

    summary = {
        "input": in_path,
        "engine_a": engine_a,
        "engine_b": engine_b,
        "swap_rule": swap_rule,
        "scope": "white-only" if args.white_only else "all-turns",
        "only_a_colour": args.only_a_colour,
        "games": len(records),
        "games_replayed": len(per_game) if args.per_game else None,
        "by_engine_colour": {
            eng: {col: _rates(stats[eng][col]) for col in ("black", "white")}
            for eng in ("A", "B")
        },
        "consistency_violations": violations,
        "n_violations": len(violations),
    }
    if args.per_game:
        summary["per_game"] = per_game
    print(json.dumps(
        {k: v for k, v in summary.items()
         if k != "consistency_violations"},
        indent=2,
    ))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

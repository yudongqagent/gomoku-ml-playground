#!/usr/bin/env python3
"""Replay a recorded v32 match and measure the VCF check's behaviour.

Walks each recorded game, rebuilds the board at every v32 turn,
and re-runs the offensive-only check exactly as ``choose_move``
would (after the win/block scans, before the search). Reports
how often the check fired, the time/move distribution of the
check in real games, and verifies the recorded move equals the
check's move whenever the check fired (the check is
deterministic, so a mismatch would indicate a bug).

The check itself is engine-independent (same
``HeuristicMCTS._vcf_root_move`` code, same priors), so a plain
``HeuristicMCTS`` is used — no value net needed.

Usage:
    PYTHONPATH=ml python ml/scripts/replay_vcf_fires.py \
        research/iterations/batch-v32-vs-v8.json [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from gomoku_ml import game, scoring
    from gomoku_ml.mcts_heuristic import HeuristicMCTS

    with open(args.match) as f:
        data = json.load(f)
    records = data["records"]
    mcts = HeuristicMCTS(sims=100, vcf_root=True, rng=np.random.default_rng(0))

    n_turns = 0
    n_scans = 0  # win/block scan fired first — the check never ran
    n_fired = 0
    times: list[float] = []
    fires: list[dict] = []
    mismatches: list[dict] = []
    for rec in records:
        board = game.empty_board()
        a_colour = rec["a_colour"]
        # a_colour is "black"/"white" (or 1/-1 in older files).
        a_player = game.BLACK if a_colour in ("black", 1) else game.WHITE
        player = game.BLACK
        for ply, move in enumerate(rec["moves"]):
            if player == a_player:
                wins = scoring._winning_moves(board, player)
                opp_wins = scoring._winning_moves(board, -player)
                cands = scoring._candidate_moves(board)
                if wins or len(opp_wins) == 1 or len(cands) <= 1:
                    n_scans += 1
                else:
                    n_turns += 1
                    t0 = time.perf_counter()
                    check = mcts._vcf_root_move(
                        board, player, defensive=False
                    )
                    times.append(time.perf_counter() - t0)
                    if check is not None:
                        n_fired += 1
                        fires.append(
                            {
                                "game": rec["game"],
                                "ply": ply,
                                "move": game.move_to_xy(check),
                                "colour": (
                                    "black" if player == game.BLACK
                                    else "white"
                                ),
                            }
                        )
                        if check != move:
                            mismatches.append(
                                {
                                    "game": rec["game"],
                                    "ply": ply,
                                    "check": game.move_to_xy(check),
                                    "recorded": game.move_to_xy(move),
                                }
                            )
            board = game.apply_move(board, move, player)
            player = -player
    times_sorted = sorted(times)
    summary = {
        "match": args.match,
        "games": len(records),
        "v32_turns": n_turns,
        "scan_turns": n_scans,
        "check_fired": n_fired,
        "fire_rate": (n_fired / n_turns) if n_turns else 0.0,
        "check_time_ms": {
            "median": statistics.median(times) * 1000 if times else 0.0,
            "p90": (
                times_sorted[int(0.9 * len(times_sorted)) - 1] * 1000
                if times else 0.0
            ),
            "max": max(times) * 1000 if times else 0.0,
            "mean": statistics.mean(times) * 1000 if times else 0.0,
        },
        "fires": fires,
        "mismatches": mismatches,
    }
    print(f"games={len(records)} v32_turns={n_turns} "
          f"scan_turns={n_scans} fired={n_fired} "
          f"({summary['fire_rate']:.1%} of checked turns)")
    ct = summary["check_time_ms"]
    print(f"check time: median={ct['median']:.1f}ms p90={ct['p90']:.1f}ms "
          f"max={ct['max']:.1f}ms mean={ct['mean']:.1f}ms")
    print(f"mismatches (check move != recorded move): {len(mismatches)}")
    for f in fires:
        print(f"  fire: game {f['game']} ply {f['ply']} "
              f"{f['colour']} {f['move']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

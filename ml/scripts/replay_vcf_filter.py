#!/usr/bin/env python3
"""Replay a recorded v33 match and measure the VCF filter's behaviour.

Walks each recorded game, rebuilds the board at every v33 turn,
and re-runs the offensive step + the prevention-pool scan exactly
as ``choose_move`` would (after the win/block scans, before the
search). Reports how often the gate opened, how often the filter
engaged (non-empty pool), the pool-size distribution, the time/move
distribution of the check in real games, and verifies the recorded
move against the filter: when the offensive step fires the recorded
move must equal it; when the filter engages the recorded move must
be a pool member (pool of 1: exactly that move). The check and the
pool are deterministic (no RNG), so a mismatch would indicate a bug
— the replay does NOT re-run the search, so it is RNG-independent.

Usage:
    PYTHONPATH=ml python ml/scripts/replay_vcf_filter.py \
        research/iterations/fast-v33-vs-v0.json [--out out.json]
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
    mcts = HeuristicMCTS(
        sims=100, leaf_mode="rollout2_v1_both",
        vcf_filtered_search=True, rng=np.random.default_rng(0),
    )

    n_turns = 0
    n_scans = 0  # win/block scan fired first — the check never ran
    n_offensive = 0
    n_gate_open = 0
    n_engaged = 0
    n_singleton = 0
    times: list[float] = []
    pool_sizes: list[int] = []
    engagements: list[dict] = []
    mismatches: list[dict] = []
    for rec in records:
        board = game.empty_board()
        a_colour = rec["a_colour"]
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
                    off = mcts._vcf_offensive_move(board, player)
                    pool = None
                    if off is None:
                        pool = mcts._vcf_prevention_pool(board, player)
                    times.append(time.perf_counter() - t0)
                    if off is not None:
                        n_offensive += 1
                        if off != move:
                            mismatches.append(
                                {
                                    "game": rec["game"], "ply": ply,
                                    "kind": "offensive",
                                    "check": game.move_to_xy(off),
                                    "recorded": game.move_to_xy(move),
                                }
                            )
                    elif pool is not None:
                        n_gate_open += 1
                        pool_sizes.append(len(pool))
                        if pool:
                            n_engaged += 1
                            if len(pool) == 1:
                                n_singleton += 1
                            engagements.append(
                                {
                                    "game": rec["game"],
                                    "ply": ply,
                                    "colour": (
                                        "black" if player == game.BLACK
                                        else "white"
                                    ),
                                    "pool": [
                                        game.move_to_xy(m) for m in pool
                                    ],
                                    "recorded": game.move_to_xy(move),
                                    "game_result": (
                                        "win" if rec.get("winner_is_a")
                                        else (
                                            "loss"
                                            if rec.get("winner_is_a") is False
                                            else "draw"
                                        )
                                    ),
                                }
                            )
                            if move not in pool:
                                mismatches.append(
                                    {
                                        "game": rec["game"], "ply": ply,
                                        "kind": "pool",
                                        "pool": [
                                            game.move_to_xy(m) for m in pool
                                        ],
                                        "recorded": game.move_to_xy(move),
                                    }
                                )
            board = game.apply_move(board, move, player)
            player = -player
    times_sorted = sorted(times)
    summary = {
        "match": args.match,
        "games": len(records),
        "v33_turns": n_turns,
        "scan_turns": n_scans,
        "offensive_fired": n_offensive,
        "gate_open": n_gate_open,
        "filter_engaged": n_engaged,
        "singleton_pools": n_singleton,
        "pool_sizes": pool_sizes,
        "check_time_ms": {
            "median": statistics.median(times) * 1000 if times else 0.0,
            "p90": (
                times_sorted[int(0.9 * len(times_sorted)) - 1] * 1000
                if times else 0.0
            ),
            "max": max(times) * 1000 if times else 0.0,
            "mean": statistics.mean(times) * 1000 if times else 0.0,
        },
        "engagements": engagements,
        "mismatches": mismatches,
    }
    print(f"games={len(records)} v33_turns={n_turns} "
          f"scan_turns={n_scans} offensive={n_offensive} "
          f"gate_open={n_gate_open} engaged={n_engaged} "
          f"(singleton {n_singleton})")
    if pool_sizes:
        print(f"pool sizes: min={min(pool_sizes)} "
              f"median={statistics.median(pool_sizes)} "
              f"max={max(pool_sizes)} "
              f"mean={statistics.mean(pool_sizes):.2f}")
    ct = summary["check_time_ms"]
    print(f"check time: median={ct['median']:.1f}ms p90={ct['p90']:.1f}ms "
          f"max={ct['max']:.1f}ms mean={ct['mean']:.1f}ms")
    print(f"mismatches: {len(mismatches)}")
    for e in engagements:
        print(f"  engaged: game {e['game']} ply {e['ply']} {e['colour']} "
              f"pool={e['pool']} recorded={e['recorded']} "
              f"({e['game_result']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Large-scale move-choice-identity A/B for v59 (iter-59 §1.3).

Compares the fast TSS path against v52's kernel path on:

* the 200 bench positions (``bench_tss.py`` generators, seed 777:
  100 quiet + 100 creator-rich);
* 2000 mined real-game positions (v27-vs-v27 self-play turns,
  seeds 777/1777 — Check B's dev seeds, distinct from the
  battery's 100/200).

For every position: same move (or both None) AND same node count
AND same budget-exceeded flag — 0 mismatches required. Node-count
identity proves the search explored the same nodes in the same
order. Results are written as JSON (``--out``); the run is
resumable in the sense that positions are processed in a fixed
order and the JSON is written at the end — re-run from scratch if
interrupted (it is measurement, not a battery leg).

``--replay N`` additionally replays N recorded v27-vs-v27 games
and checks v59 vs v52 ``choose_move`` on every turn (same seed).

Usage:
    PYTHONPATH=ml python ml/scripts/ab_tss_v59.py
        [--mined 2000] [--out research/iterations/ab-v59.json]
        [--replay 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from bench_tss import (  # noqa: E402
    generate_creator_rich_positions,
    generate_quiet_positions,
)


def _ab_positions(positions, label, node_budget):
    from gomoku_ml.mcts_heuristic import tss_winning_move

    mismatches = []
    fired = 0
    exceeded = 0
    for i, (board, player) in enumerate(positions):
        s_slow: dict = {}
        s_fast: dict = {}
        slow = tss_winning_move(board, player, node_budget=node_budget,
                                stats=s_slow, fast=False)
        fast = tss_winning_move(board, player, node_budget=node_budget,
                                stats=s_fast, fast=True)
        if slow is not None:
            fired += 1
        if s_slow["budget_exceeded"]:
            exceeded += 1
        if (
            fast != slow
            or s_fast["nodes"] != s_slow["nodes"]
            or s_fast["budget_exceeded"] != s_slow["budget_exceeded"]
        ):
            mismatches.append(
                {
                    "index": i,
                    "slow_move": slow,
                    "fast_move": fast,
                    "slow_nodes": s_slow["nodes"],
                    "fast_nodes": s_fast["nodes"],
                    "slow_exceeded": s_slow["budget_exceeded"],
                    "fast_exceeded": s_fast["budget_exceeded"],
                }
            )
        if (i + 1) % 200 == 0:
            print(f"  {label}: {i + 1}/{len(positions)} "
                  f"({len(mismatches)} mismatches)", flush=True)
    print(f"{label}: n={len(positions)} fired={fired} "
          f"budget_exceeded={exceeded} mismatches={len(mismatches)}")
    return {
        "label": label,
        "n": len(positions),
        "fired": fired,
        "budget_exceeded": exceeded,
        "mismatches": mismatches,
    }


def _mine_positions(n, seed_a=777, seed_b=1777):
    from gomoku_ml import game, scoring

    positions = []
    g = 0
    while len(positions) < n:
        pa = scoring.get_player(27, seed=seed_a + g)
        pb = scoring.get_player(27, seed=seed_b + g)
        board = game.empty_board()
        player = game.BLACK
        while len(positions) < n:
            over, _ = game.game_over(board)
            if over:
                break
            positions.append((board.copy(), player))
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        g += 1
    return positions


def _replay(n_games):
    from gomoku_ml import game, scoring

    mismatches = []
    turns = 0
    for g in range(n_games):
        pa = scoring.get_player(27, seed=777 + g)
        pb = scoring.get_player(27, seed=1777 + g)
        p52 = scoring.get_player(52, seed=100)
        p59 = scoring.get_player(59, seed=100)
        board = game.empty_board()
        player = game.BLACK
        while True:
            over, _ = game.game_over(board)
            if over:
                break
            m52 = p52.choose_move(board, player)
            m59 = p59.choose_move(board, player)
            turns += 1
            if m59 != m52:
                mismatches.append(
                    {"game": g, "turn": turns, "v52": m52, "v59": m59}
                )
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        print(f"  replay: game {g + 1}/{n_games} done "
              f"({len(mismatches)} mismatches)", flush=True)
    print(f"replay: games={n_games} turns={turns} "
          f"mismatches={len(mismatches)}")
    return {"games": n_games, "turns": turns, "mismatches": mismatches}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=int, default=2000)
    ap.add_argument("--node-budget", type=int, default=3000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--replay", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    results: dict = {"node_budget": args.node_budget}

    quiet = generate_quiet_positions(777, 100)
    print(f"generated {len(quiet)} quiet positions", file=sys.stderr)
    results["bench_quiet"] = _ab_positions(
        quiet, "bench quiet", args.node_budget
    )
    rich = generate_creator_rich_positions(777, 100)
    print(f"generated {len(rich)} creator-rich positions", file=sys.stderr)
    results["bench_rich"] = _ab_positions(
        rich, "bench creator-rich", args.node_budget
    )

    mined = _mine_positions(args.mined)
    print(f"mined {len(mined)} real-game positions", file=sys.stderr)
    results["mined"] = _ab_positions(mined, "mined", args.node_budget)

    if args.replay:
        results["replay"] = _replay(args.replay)

    total_mm = sum(
        len(results[k]["mismatches"])
        for k in ("bench_quiet", "bench_rich", "mined")
    )
    if args.replay:
        total_mm += len(results["replay"]["mismatches"])
    results["total_mismatches"] = total_mm
    results["elapsed_s"] = time.time() - t0
    print(f"TOTAL mismatches: {total_mm} "
          f"(elapsed {results['elapsed_s']:.0f}s)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0 if total_mm == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cost bench for the v59 TSS speed optimisation (iter-59 §1.2).

Times ``mcts_heuristic.tss_winning_move`` on the SAME positions as
v52's bench (``bench_tss.py`` generators, seed 777, n=100 per set:
quiet v7 self-play + creator-rich v27 midgame plies 20-60), at the
PRODUCTION node budget (3000), and reports the fire rate, the time
distribution, and the NODE distribution.

Two clocks are recorded per call: wall time (``perf_counter``, as
the original bench) and CPU time (``process_time``). The VM is
shared (other batch-14 workers run concurrently), so wall time is
contention-inflated; CPU time is the uncontended proxy (on the
user's device the TSS runs single-threaded and uncontended, so
wall time there ≈ CPU time here). The p99 < 1s verdict rests on
the CPU-time p99; both are reported, with the load average noted.

``--fast`` selects the v59 fast path (``tss_winning_move(...,
fast=True)``); without it the v52 path is measured. ``--out`` writes
the per-position records (wall/cpu ms, nodes, fired, budget
exceeded) as JSON, so the before/after comparison is exact and the
node-count identity check (fast vs slow on the same positions) is
auditable.

Usage:
    PYTHONPATH=ml python ml/scripts/bench_tss_v59.py [--fast]
        [--node-budget 3000] [--seed 777] [--n 100]
        [--out research/iterations/bench-v59.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from bench_tss import (  # noqa: E402
    generate_creator_rich_positions,
    generate_quiet_positions,
)


def _pct(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def bench(
    name: str,
    positions: list,
    node_budget: int,
    fast: bool,
) -> dict:
    from gomoku_ml.mcts_heuristic import tss_winning_move

    records: list[dict] = []
    for board, player in positions:
        stats: dict = {}
        t0_wall = time.perf_counter()
        t0_cpu = time.process_time()
        move = tss_winning_move(
            board, player, node_budget=node_budget, stats=stats, fast=fast
        )
        wall_ms = (time.perf_counter() - t0_wall) * 1000
        cpu_ms = (time.process_time() - t0_cpu) * 1000
        records.append(
            {
                "wall_ms": wall_ms,
                "cpu_ms": cpu_ms,
                "nodes": stats["nodes"],
                "fired": move is not None,
                "budget_exceeded": stats["budget_exceeded"],
            }
        )
    walls = sorted(r["wall_ms"] for r in records)
    cpus = sorted(r["cpu_ms"] for r in records)
    nodes = sorted(r["nodes"] for r in records)
    fired = sum(1 for r in records if r["fired"])
    exceeded = sum(1 for r in records if r["budget_exceeded"])
    print(f"--- {name} (n={len(records)}, budget={node_budget}, "
          f"fast={fast}) ---")
    print(f"fired={fired} ({fired / len(records):.0%}) "
          f"budget_exceeded={exceeded}")
    print(f"wall: median={statistics.median(walls):.1f}ms "
          f"p90={_pct(walls, 0.9):.1f}ms p99={_pct(walls, 0.99):.1f}ms "
          f"max={max(walls):.1f}ms mean={statistics.mean(walls):.1f}ms")
    print(f"cpu:  median={statistics.median(cpus):.1f}ms "
          f"p90={_pct(cpus, 0.9):.1f}ms p99={_pct(cpus, 0.99):.1f}ms "
          f"max={max(cpus):.1f}ms mean={statistics.mean(cpus):.1f}ms")
    print(f"nodes: median={statistics.median(nodes):.0f} "
          f"p90={_pct(nodes, 0.9):.0f} p99={_pct(nodes, 0.99):.0f} "
          f"max={max(nodes)} mean={statistics.mean(nodes):.0f}")
    return {
        "name": name,
        "n": len(records),
        "node_budget": node_budget,
        "fast": fast,
        "fired": fired,
        "budget_exceeded": exceeded,
        "wall_ms": {
            "median": statistics.median(walls),
            "p90": _pct(walls, 0.9),
            "p99": _pct(walls, 0.99),
            "max": max(walls),
            "mean": statistics.mean(walls),
        },
        "cpu_ms": {
            "median": statistics.median(cpus),
            "p90": _pct(cpus, 0.9),
            "p99": _pct(cpus, 0.99),
            "max": max(cpus),
            "mean": statistics.mean(cpus),
        },
        "nodes": {
            "median": statistics.median(nodes),
            "p90": _pct(nodes, 0.9),
            "p99": _pct(nodes, 0.99),
            "max": max(nodes),
            "mean": statistics.mean(nodes),
        },
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--node-budget", type=int, default=3000)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    try:
        load = os.getloadavg()
    except OSError:
        load = None
    print(f"load average at start: {load}", file=sys.stderr)

    quiet = generate_quiet_positions(args.seed, args.n)
    print(f"generated {len(quiet)} quiet positions (seed {args.seed})",
          file=sys.stderr)
    q = bench("quiet (v7 self-play)", quiet, args.node_budget, args.fast)

    rich = generate_creator_rich_positions(args.seed, args.n)
    print(f"generated {len(rich)} creator-rich positions (seed {args.seed})",
          file=sys.stderr)
    r = bench("creator-rich (v27 midgame)", rich, args.node_budget, args.fast)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "n": args.n,
                    "node_budget": args.node_budget,
                    "fast": args.fast,
                    "load_avg_start": load,
                    "quiet": q,
                    "creator_rich": r,
                },
                f,
                indent=1,
            )
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Cost bench for the v57 TSS depth/budget sweep (iter-57 §1.4).

Benches all 6 sweep cells (depth x budget) on the SAME 100 quiet +
100 creator-rich positions as iter-52's bench (same generators, same
seed 777 — so the (4, 3000) cell reproduces iter-52 §1.4's numbers
and the other cells are directly comparable). Each cell is benched
at its OWN production budget (3000/6000) — NOT the 100000-node
provisional budget iter-52 used: at depth 5-6 the unconstrained
tree may explode, the budget is the termination guarantee, and the
budget_exceeded rate is the decision-relevant measurement (it
answers "does the budget bind first?" directly).

Results accumulate in the --out JSON, one entry per cell, flushed
after each cell — a killed run resumes by skipping cells already
present (position generation is deterministic from the seed, so a
re-run reproduces the same positions).

Usage:
    PYTHONPATH=ml python ml/scripts/bench_tss_sweep.py
        --out research/iterations/tss-sweep-bench.json [--seed 777] [--n 100]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_tss  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

# The pre-registered sweep grid (iter-57 §1.2): (depth, node_budget).
CELLS = [(4, 3000), (4, 6000), (5, 3000), (5, 6000), (6, 3000), (6, 6000)]


def _pct(sorted_values: list[float], q: float) -> float:
    idx = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values)) - 1))
    return sorted_values[idx]


def bench_cell(
    name: str,
    positions: list[tuple[np.ndarray, int]],
    depth: int,
    node_budget: int,
) -> dict:
    from gomoku_ml.mcts_heuristic import tss_winning_move

    times: list[float] = []
    nodes: list[int] = []
    fired = 0
    exceeded = 0
    for board, player in positions:
        stats: dict = {}
        t0 = time.perf_counter()
        move = tss_winning_move(
            board, player, depth=depth, node_budget=node_budget, stats=stats
        )
        times.append(time.perf_counter() - t0)
        nodes.append(stats["nodes"])
        if move is not None:
            fired += 1
        if stats["budget_exceeded"]:
            exceeded += 1
    ts = sorted(times)
    ns = sorted(nodes)
    result = {
        "name": name,
        "n": len(times),
        "depth": depth,
        "node_budget": node_budget,
        "fired": fired,
        "fire_rate": fired / len(times) if times else 0.0,
        "budget_exceeded": exceeded,
        "budget_exceeded_rate": exceeded / len(times) if times else 0.0,
        "time_ms": {
            "median": statistics.median(times) * 1000,
            "p90": _pct(ts, 0.9) * 1000,
            "p99": _pct(ts, 0.99) * 1000,
            "max": max(times) * 1000,
            "mean": statistics.mean(times) * 1000,
        },
        "nodes": {
            "median": statistics.median(nodes),
            "p90": _pct(ns, 0.9),
            "p99": _pct(ns, 0.99),
            "max": max(nodes),
            "mean": statistics.mean(nodes),
        },
    }
    print(
        f"--- {name} depth={depth} budget={node_budget} (n={len(times)}) ---",
        flush=True,
    )
    print(
        f"fired={fired} ({fired / len(times):.0%}) "
        f"budget_exceeded={exceeded} ({exceeded / len(times):.0%})",
        flush=True,
    )
    print(
        f"time: median={result['time_ms']['median']:.1f}ms "
        f"p90={result['time_ms']['p90']:.1f}ms "
        f"p99={result['time_ms']['p99']:.1f}ms "
        f"max={result['time_ms']['max']:.1f}ms "
        f"mean={result['time_ms']['mean']:.1f}ms",
        flush=True,
    )
    print(
        f"nodes: median={result['nodes']['median']:.0f} "
        f"p90={result['nodes']['p90']:.0f} "
        f"p99={result['nodes']['p99']:.0f} "
        f"max={result['nodes']['max']} "
        f"mean={result['nodes']['mean']:.0f}",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    done: dict[str, dict] = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf-8") as f:
                prior = json.load(f)
            if prior.get("seed") == args.seed and prior.get("n") == args.n:
                done = prior.get("cells", {})
                print(f"resuming: {len(done)} cells already done",
                      file=sys.stderr)
        except (OSError, ValueError, KeyError):
            print("prior bench unreadable — starting from scratch",
                  file=sys.stderr)

    quiet = bench_tss.generate_quiet_positions(args.seed, args.n)
    print(f"generated {len(quiet)} quiet positions (seed {args.seed})",
          file=sys.stderr)
    rich = bench_tss.generate_creator_rich_positions(args.seed, args.n)
    print(f"generated {len(rich)} creator-rich positions (seed {args.seed})",
          file=sys.stderr)

    cells: dict[str, dict] = dict(done)
    for depth, budget in CELLS:
        key = f"d{depth}-b{budget}"
        if key in cells:
            continue
        cells[key] = {
            "quiet": bench_cell("quiet (v7 self-play)", quiet, depth, budget),
            "creator_rich": bench_cell(
                "creator-rich (v27 midgame)", rich, depth, budget
            ),
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {"seed": args.seed, "n": args.n, "cells": cells}, f, indent=2
            )
    print(f"bench complete: {len(cells)} cells -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

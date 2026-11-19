#!/usr/bin/env python3
"""Mechanism check for v41 (batch 11 H3, iter-41 §3.2).

Compares v29's _vcf_root_move (vcf_root=True, vcf_root_cheap=False)
with v41's (both True) on all 49 tactical positions + the
100-position bench set (bench_vcf_root.py, seed 777).

Classification of every position:
  - agree: both return the same move (or both None).
  - disagree_limit_binds: v29 returns a prevention whose rank in
    the (prior desc, flat index asc) candidate order is >= 8, so
    v41's K=8 limit legitimately excludes it (v41 returns None or
    a different prevention). EXPECTED divergence.
  - disagree_no_bind: the moves differ AND v29's prevention (if
    any) ranks in the top 8. This REFUTES the monotonicity claim
    (change 2) — per iter-41 §3.2, change (2) must then be
    withdrawn and the candidate limit alone is the fallback.
  - disagree_offensive: the offensive step differs (should be
    impossible — the offensive step is unchanged; a bug if seen).

Also reports per-engine timing (median/p90/max/mean) and fire
rates on the bench set (cost guard).

Usage:
    PYTHONPATH=ml python ml/scripts/mechanism_v41.py [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from gomoku_ml import evaluate, game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    HeuristicMCTS,
    _VCF_CHEAP_CANDIDATE_LIMIT,
    _priors,
)

HARD_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "gomoku_ml",
    "tactical_positions_hard.json",
)


def rank_of_prevention(board: np.ndarray, player: int, move: int) -> int:
    """Rank (0-based) of ``move`` in the defensive candidate order."""
    priors = _priors(board, player, 0.0)
    cands = sorted(
        scoring._candidate_moves(board),
        key=lambda m: (-priors.get(m, 0.0), m),
    )
    return cands.index(move) if move in cands else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # --- positions ---------------------------------------------------
    positions: list[tuple[str, np.ndarray, int]] = []
    for name, path in (("old", None), ("hard", HARD_PATH)):
        suite = evaluate.load_tactical_suite(path)
        for pos in suite["positions"]:
            board = evaluate.board_from_position(pos)
            player = pos["to_move"]
            positions.append((f"{name}:{pos['id']}", board, player))

    # Bench positions (same generator as bench_vcf_root.py).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bench_vcf_root import generate_positions  # noqa: E402

    for i, (board, player) in enumerate(generate_positions(777, 100)):
        positions.append((f"bench:{i}", board, player))

    print(f"{len(positions)} positions", file=sys.stderr)

    m29 = HeuristicMCTS(
        sims=100, leaf_mode="rollout2_v1_both", vcf_root=True,
        rng=np.random.default_rng(0),
    )
    m41 = HeuristicMCTS(
        sims=100, leaf_mode="rollout2_v1_both", vcf_root=True,
        vcf_root_cheap=True, rng=np.random.default_rng(0),
    )

    results: list[dict] = []
    times29: list[float] = []
    times41: list[float] = []
    fired29 = fired41 = 0
    for label, board, player in positions:
        t0 = time.perf_counter()
        move29 = m29._vcf_root_move(board, player)
        t29 = time.perf_counter() - t0
        t0 = time.perf_counter()
        move41 = m41._vcf_root_move(board, player)
        t41 = time.perf_counter() - t0
        times29.append(t29)
        times41.append(t41)
        if move29 is not None:
            fired29 += 1
        if move41 is not None:
            fired41 += 1
        if move29 == move41:
            cls = "agree"
            rank = rank_of_prevention(board, player, move29) if move29 is not None else None
        else:
            # Which step produced v29's move? Offensive if it is one
            # of our four-creators that starts a forced win.
            ours = evaluate.four_creating_moves(board, player)
            offensive = (
                move29 is not None
                and move29 in ours
                and evaluate.starts_forced_win(board, player, move29, depth=3)
            )
            if offensive:
                cls = "disagree_offensive"
                rank = None
            else:
                rank = rank_of_prevention(board, player, move29) if move29 is not None else None
                if rank is not None and rank >= _VCF_CHEAP_CANDIDATE_LIMIT:
                    cls = "disagree_limit_binds"
                else:
                    cls = "disagree_no_bind"
        results.append({
            "position": label,
            "move_v29": move29,
            "move_v41": move41,
            "class": cls,
            "v29_rank": rank,
            "t29_ms": round(t29 * 1000, 1),
            "t41_ms": round(t41 * 1000, 1),
        })
        if cls != "agree":
            print(f"  {label}: v29={move29} v41={move41} {cls} rank={rank}",
                  file=sys.stderr)
        print(f"  done {len(results)}/{len(positions)}", file=sys.stderr, end="\r")

    print(file=sys.stderr)

    def stats(times: list[float]) -> dict:
        s = sorted(times)
        return {
            "median_ms": round(statistics.median(times) * 1000, 1),
            "p90_ms": round(s[int(0.9 * len(s)) - 1] * 1000, 1),
            "max_ms": round(max(times) * 1000, 1),
            "mean_ms": round(statistics.mean(times) * 1000, 1),
        }

    counts = {}
    for r in results:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    out = {
        "n": len(results),
        "counts": counts,
        "bench_n": 100,
        "v29": {"fired": fired29, **stats(times29)},
        "v41": {"fired": fired41, **stats(times41)},
        "results": results,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", file=sys.stderr)
    # Exit code: 1 if the monotonicity claim is refuted (any
    # disagree_no_bind or disagree_offensive).
    bad = counts.get("disagree_no_bind", 0) + counts.get("disagree_offensive", 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""M3 for v58 (iter-58 §1.3): depth-6 spot-check — does a deeper TSS
find White forcing wins that production TSS (depth 4) misses?

Procedure (pre-registered in iter-58 §1.3 BEFORE this script ran):

* Input: an M1 mining JSON with ``qualifying`` White turns (first 30
  per leg where production TSS returns None AND the forcing move set
  is non-empty AND Black holds no winning move AND the search
  completed, i.e. budget NOT exceeded).
* On the first N qualifying turns (default 20, normal-rules leg
  FIRST — that is where a patch would apply), run
  ``tss_winning_move(board, WHITE, depth=6, node_budget=10000)`` and
  record: found a win / None-completed / None-budget-exceeded.
* A depth-6 find on a turn where depth-4 found nothing is a
  DETECTION gap (patchable in principle); no finds means TSS
  detection is complete for White and the deficit is opportunity
  (structural, not patchable).

Decision rule (iter-58 §1.4): a detection gap is CONFIRMED if
depth-6 finds a win in >=3/20 sampled normal-rules White turns
(15%); NO GAP if 0/20; 1-2/20 is AMBIGUOUS and counts as NO PATCH.

Deterministic; checkpoints after every position (depth-6 calls can
be slow) and resumes exactly.

Usage:
    PYTHONPATH=ml python ml/scripts/spotcheck_tss_depth.py
        --in research/iterations/tss-mine-v58-normal.json
        --out research/iterations/tss-spotcheck-v58-normal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--budget", type=int, default=10000)
    ap.add_argument("--checkpoint", default=None)
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
        mining = json.load(f)
    qualifying = mining["qualifying"][: args.n]

    results = []
    start_i = 0
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if (
                ckpt.get("input") == in_path
                and ckpt.get("n") == args.n
                and ckpt.get("depth") == args.depth
                and ckpt.get("budget") == args.budget
            ):
                results = ckpt["results"]
                start_i = ckpt["done"]
                print(f"resuming from checkpoint: {start_i}/"
                      f"{len(qualifying)} positions done", file=sys.stderr)
        except (OSError, ValueError, KeyError):
            print("checkpoint unreadable — starting from scratch",
                  file=sys.stderr)

    for i in range(start_i, len(qualifying)):
        q = qualifying[i]
        board = game.empty_board()
        for x, y in q["black"]:
            board[y, x] = game.BLACK
        for x, y in q["white"]:
            board[y, x] = game.WHITE
        tss_stats: dict = {}
        move = mcts_heuristic.tss_winning_move(
            board, game.WHITE, depth=args.depth,
            node_budget=args.budget, stats=tss_stats)
        results.append({
            "game": q["game"],
            "ply": q["ply"],
            "found": move is not None,
            "move": [move % 15, move // 15] if move is not None else None,
            "nodes": tss_stats.get("nodes", 0),
            "budget_exceeded": bool(tss_stats.get("budget_exceeded")),
        })
        print(f"position {i + 1}/{len(qualifying)}: game {q['game']} "
              f"ply {q['ply']} found={move is not None} "
              f"nodes={tss_stats.get('nodes', 0)} "
              f"exceeded={tss_stats.get('budget_exceeded')}",
              file=sys.stderr)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "input": in_path,
                "n": args.n,
                "depth": args.depth,
                "budget": args.budget,
                "done": i + 1,
                "results": results,
            }, f)

    found = sum(1 for r in results if r["found"])
    completed_none = sum(
        1 for r in results if not r["found"] and not r["budget_exceeded"])
    aborted = sum(1 for r in results if r["budget_exceeded"])
    summary = {
        "input": in_path,
        "rule": mining.get("rule"),
        "n": len(results),
        "depth": args.depth,
        "budget": args.budget,
        "found": found,
        "none_completed": completed_none,
        "none_budget_exceeded": aborted,
        "find_rate": found / len(results) if results else 0.0,
        "find_rate_completed": (
            found / (found + completed_none)
            if (found + completed_none) else 0.0
        ),
        "results": results,
    }
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "results"}, indent=2))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

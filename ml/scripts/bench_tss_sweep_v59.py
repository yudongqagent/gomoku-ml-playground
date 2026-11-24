#!/usr/bin/env python3
"""Phase-2 (L7) node-budget sweep for v59 (iter-59 §1.4, §5.4).

Phase 1 (fast TSS at the production budget 3000) missed the
p99 < 1s target (creator-rich CPU p99 1780ms), so the
pre-registered L7 rule applies: sweep budgets {2000, 1500,
1000} on the bench positions and pick the LARGEST budget
whose CPU-time p99 < 1s AND abort rate <= 5%. If none
satisfies both, keep 3000 and report honestly.

The bench positions are generated ONCE (seed 777, n=100/set)
and reused for all budgets, so the sweep is cheap and the
per-budget numbers are directly comparable. Per-position
records (cpu_ms, nodes, fired, budget_exceeded) are saved
for each budget, so the abort rate and the p99 are auditable.

A budget cut CHANGES move choice (searches that would have
found a win abort early), so the chosen budget is arbitrated
by the battery (must not lose vs v52; tie acceptable) — see
iter-59 §5.5.

Usage:
    PYTHONPATH=ml python ml/scripts/bench_tss_sweep_v59.py
        [--budgets 2000 1500 1000]
        [--out research/iterations/bench-v59-sweep.json]
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
from bench_tss_v59 import _pct, bench  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+",
                    default=[2000, 1500, 1000])
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=100)
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
    rich = generate_creator_rich_positions(args.seed, args.n)
    print(f"generated {len(rich)} creator-rich positions "
          f"(seed {args.seed})", file=sys.stderr)

    doc: dict = {
        "seed": args.seed,
        "n": args.n,
        "budgets": args.budgets,
        "fast": True,
        "load_avg_start": load,
        "results": {},
    }
    for budget in args.budgets:
        q = bench("quiet (v7 self-play)", quiet, budget, True)
        r = bench("creator-rich (v27 midgame)", rich, budget, True)
        # Abort rate: fraction of positions that hit the budget.
        # (budget_exceeded is recorded per position.)
        for name, res in (("quiet", q), ("creator_rich", r)):
            exceeded = sum(
                1 for rec in res["records"]
                if rec["budget_exceeded"]
            )
            res["abort_rate"] = exceeded / len(res["records"])
        doc["results"][str(budget)] = {"quiet": q, "creator_rich": r}
        print(f"--- budget {budget} summary ---")
        for name, res in (("quiet", q), ("creator_rich", r)):
            print(f"  {name}: cpu p99={res['cpu_ms']['p99']:.1f}ms "
                  f"abort_rate={res['abort_rate']:.1%} "
                  f"fired={res['fired']}")
        if args.out:
            tmp = args.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=1)
            os.replace(tmp, args.out)

    # Pre-registered rule: LARGEST budget with creator-rich CPU
    # p99 < 1000ms AND abort rate <= 5% (both sets).
    pick = None
    for budget in sorted(args.budgets, reverse=True):
        res = doc["results"][str(budget)]
        rich_p99 = res["creator_rich"]["cpu_ms"]["p99"]
        abort = max(res["quiet"]["abort_rate"],
                    res["creator_rich"]["abort_rate"])
        if rich_p99 < 1000 and abort <= 0.05:
            pick = budget
            break
    doc["pick"] = pick
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, args.out)
    if pick is None:
        print("RULE: no budget satisfies p99 < 1s AND abort <= 5%; "
              "keep 3000 and report honestly")
    else:
        print(f"RULE: pick budget {pick}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

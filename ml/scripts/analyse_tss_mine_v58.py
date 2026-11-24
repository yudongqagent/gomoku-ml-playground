#!/usr/bin/env python3
"""Analyse M1 (v58): compare the normal-rules and swap-rules TSS
mining legs and apply the pre-registered decision rule (iter-58
§1.4) for mechanism (b).

Prints, per colour and leg: turns, fires, fire rate, differs, differ
rate, budget-exceeded rate, mean nodes; the White differ-rate ratio
(swap/normal) with binomial SEs and the 2-SE test; the Check B
replication check (normal leg must reproduce 2481 turns / 288 fires
/ 121 differs); Black win rates in v27 self-play per leg; and the
per-opening White differ counts for the swap leg.

Usage:
    python ml/scripts/analyse_tss_mine_v58.py
        research/iterations/tss-mine-v58-normal.json
        research/iterations/tss-mine-v58-swap.json
"""

from __future__ import annotations

import json
import math
import sys


def _se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else 0.0


def _show(label, cs):
    print(f"  {label}: turns={cs['turns']} fires={cs['fires']} "
          f"fire_rate={cs['fire_rate']:.4f} differs={cs['differs']} "
          f"differ_rate={cs['differ_rate']:.4f} confirms={cs['confirms']} "
          f"budget_exceeded={cs['budget_exceeded']} "
          f"mean_nodes={cs['mean_nodes']:.1f}")


def main() -> int:
    normal = json.load(open(sys.argv[1], encoding="utf-8"))
    swap = json.load(open(sys.argv[2], encoding="utf-8"))

    print("=== Check B replication (normal leg) ===")
    print(f"  turns={normal['total_turns']} (expect 2481) "
          f"fires={normal['total_fires']} (expect 288) "
          f"differs={normal['total_differs']} (expect 121)")
    ok = (normal["total_turns"] == 2481
          and normal["total_fires"] == 288
          and normal["total_differs"] == 121)
    print(f"  replication exact: {ok}")

    print("=== by colour ===")
    for name, leg in (("normal", normal), ("swap", swap)):
        print(f"  -- {name} --")
        _show("black", leg["by_colour"]["black"])
        _show("white", leg["by_colour"]["white"])
        print(f"  black wins={leg['black_wins']} white wins="
              f"{leg['white_wins']} draws={leg['draws']} "
              f"mean_plies={leg['mean_plies']:.1f}")

    print("=== decision quantities (mechanism b) ===")
    wn, ws = normal["by_colour"]["white"], swap["by_colour"]["white"]
    bn, bs = normal["by_colour"]["black"], swap["by_colour"]["black"]
    for label, a, b in (("white differ rate", wn, ws),
                        ("white fire rate", wn, ws),
                        ("black differ rate", bn, bs),
                        ("black fire rate", bn, bs)):
        key = "differ_rate" if "differ" in label else "fire_rate"
        count = "differs" if "differ" in label else "fires"
        rn, rs = a[key], b[key]
        ratio = rs / rn if rn else float("inf")
        se = math.sqrt(_se(rn, a["turns"]) ** 2 + _se(rs, b["turns"]) ** 2)
        z = (rs - rn) / se if se else 0.0
        print(f"  {label}: normal {rn:.4f} ({a[count]}/{a['turns']}) "
              f"swap {rs:.4f} ({b[count]}/{b['turns']}) ratio={ratio:.2f} "
              f"diff={rs - rn:+.4f} z={z:+.2f} (2-SE threshold: |z|>2)")
    wr = ws["differ_rate"] / wn["differ_rate"] if wn["differ_rate"] else 0
    print(f"  WHITE DIFFER-RATE RATIO = {wr:.2f}")
    if wr >= 1.5:
        print("  -> (b) CONFIRMED on the ratio cut (check z > 2 above)")
    elif 0.8 <= wr <= 1.25:
        print("  -> (b) REFUTED on the ratio cut (needs M2 agreement)")
    else:
        print("  -> INCONCLUSIVE on the ratio cut (M2 is the tiebreaker)")

    print("=== budget asymmetry (mechanism d) ===")
    for name, leg in (("normal", normal), ("swap", swap)):
        for col in ("black", "white"):
            cs = leg["by_colour"][col]
            print(f"  {name} {col}: budget_exceeded "
                  f"{cs['budget_exceeded']}/{cs['turns']} "
                  f"({cs['budget_exceeded'] / cs['turns']:.4f})")

    print("=== swap per-opening White differs ===")
    pod = swap.get("per_opening_differs", {})
    for okey, counts in sorted(pod.items(),
                               key=lambda kv: -kv[1]["white"]):
        print(f"  opening {okey}: black differs={counts['black']} "
              f"white differs={counts['white']}")
    print(f"  qualifying White turns recorded: normal="
          f"{normal.get('qualifying_count')} swap="
          f"{swap.get('qualifying_count')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

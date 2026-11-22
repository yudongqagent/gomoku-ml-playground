#!/usr/bin/env python3
"""Recompute match aggregates from raw per-game records (never trust
the script summary alone): W/L/D, win rate, Wilson 95% CI,
uniqueness / de-dup rate, and per-colour breakdown. A win = the
record's ``winner_is_a`` is True; a draw = ``winner`` == 'draw'.

Usage:
    python ml/scripts/recompute_match.py <batch.json> [<batch.json> ...]
"""

from __future__ import annotations

import json
import math
import sys


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI. ``k`` may be non-integer: the program's win
    rate counts a draw as half a win, so k = W + 0.5*D."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def main() -> int:
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        recs = d["records"]
        n = len(recs)
        wins = sum(1 for r in recs if r.get("winner_is_a") is True)
        draws = sum(1 for r in recs if r.get("winner") == "draw")
        losses = n - wins - draws
        # Program convention: a draw counts as half a win.
        wr = (wins + 0.5 * draws) / n if n else 0.0
        lo, hi = wilson(wins + 0.5 * draws, n)
        # Uniqueness: distinct move sequences.
        seen: set[tuple[int, ...]] = set()
        dw = dl = dd = 0
        for r in recs:
            key = tuple(r["moves"])
            if key in seen:
                continue
            seen.add(key)
            if r.get("winner_is_a") is True:
                dw += 1
            elif r.get("winner") == "draw":
                dd += 1
            else:
                dl += 1
        dn = len(seen)
        dwr = (dw + 0.5 * dd) / dn if dn else 0.0
        dlo, dhi = wilson(dw + 0.5 * dd, dn)
        # Per-colour (A's colour).
        by_colour: dict[str, list[int]] = {}
        for r in recs:
            c = r.get("a_colour", "?")
            w, l, dr = by_colour.get(c, [0, 0, 0])
            if r.get("winner_is_a") is True:
                w += 1
            elif r.get("winner") == "draw":
                dr += 1
            else:
                l += 1
            by_colour[c] = [w, l, dr]
        print(f"== {path} ==")
        print(f"  games: {n} (complete: {d.get('complete')}, "
              f"resumed: {d.get('resumed')})")
        print(f"  RECOMPUTED: {wins}W-{losses}L-{draws}D "
              f"win_rate={wr:.4f} Wilson95=[{lo:.4f}, {hi:.4f}]")
        print(f"  script summary: {d.get('wins')}W-{d.get('losses')}L-"
              f"{d.get('draws')}D win_rate={d.get('win_rate')} "
              f"CI={d.get('win_rate_ci_95')}")
        print(f"  unique: {dn}/{n} games "
              f"(de-dup {dw}W-{dl}L-{dd}D rate={dwr:.4f} "
              f"CI=[{dlo:.4f}, {dhi:.4f}])")
        for c, (w, l, dr) in sorted(by_colour.items()):
            print(f"  A as {c}: {w}W-{l}L-{dr}D")
    return 0


if __name__ == "__main__":
    sys.exit(main())

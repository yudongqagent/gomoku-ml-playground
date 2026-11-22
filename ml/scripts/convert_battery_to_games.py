#!/usr/bin/env python3
"""Convert a selfplay_batch battery JSON to distill games format (ML batch 6, v55).

v48's patch positions came from v44's four battery legs
(research/iterations/batch-v44-vs-{v0,v7,v8,v37}.json). Every record
carries the full move list (the openings legs' ``moves`` include the
opening plies — iter-48 §6 verified 100/100 records in both legs), so
positions are recovered by replay through
``generate_distill_data.py label``. This script converts one battery
JSON to the games format that command reads: game index, full moves,
winner mapped black->1 / white->-1 / draw->0 (the int64 shard field).

Usage:
    python ml/scripts/convert_battery_to_games.py \\
        research/iterations/batch-v44-vs-v0.json \\
        --out ml/runs/ml6-h2/games-vs-v0.json --source vs-v0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

WINNER = {"black": 1, "white": -1, "draw": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("battery", help="selfplay_batch JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source", required=True,
                        help="game source label (e.g. vs-v0)")
    args = parser.parse_args()

    with open(args.battery, encoding="utf-8") as f:
        payload = json.load(f)
    records = payload["records"]
    games = []
    for rec in sorted(records, key=lambda r: r["game"]):
        winner = rec["winner"]
        if winner not in WINNER:
            raise SystemExit(f"game {rec['game']}: unknown winner "
                             f"{winner!r}")
        moves = [int(m) for m in rec["moves"]]
        games.append({
            "game": int(rec["game"]),
            "source": args.source,
            "sims": 100,
            "seed": 0,
            "moves": moves,
            "winner": WINNER[winner],
            "plies": len(moves),
        })
    out_payload = {"games": games, "complete": bool(payload.get("complete"))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)
        f.write("\n")
    os.replace(tmp, args.out)
    total_plies = sum(g["plies"] for g in games)
    print(f"converted {len(games)} games ({total_plies} plies) "
          f"from {args.battery} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

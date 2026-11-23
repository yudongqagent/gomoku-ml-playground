#!/usr/bin/env python3
"""Convert a battery / fast-loop match JSON to a distill games file.

The labelling step (``generate_distill_data.py label``) takes a
games file: ``{"games": [{"game", "moves", "winner", "plies"}]}``
with ``winner`` as an int (black = 1, white = -1, draw = 0).
Battery JSONs (``selfplay_batch.py`` / ``fast_loop_match.py``)
carry the same information in ``records`` with ``winner`` as a
string ("black"/"white"/"draw"). This converter bridges the two
so patch data can be labelled from committed battery legs or
fresh fast-loop legs with reproducible provenance (ML batch 6,
v54 — see research/iterations/iter-54.md §2; v48's conversion
was uncommitted).

Records are re-indexed 0..N-1 in file order (the label step's
game-level holdout uses ``(game + 1) % holdout_every``), and the
output is marked complete. Opening plies are included in the
recorded moves by the drivers, so the replay is exact.

Usage:
    python ml/scripts/battery_to_games.py \\
        --battery research/iterations/batch-v44-vs-v0.json \\
        --out ml/runs/ml6-h1/games-vs-v0.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

WINNER_TO_INT = {"black": 1, "white": -1, "draw": 0}


def convert(battery: dict) -> dict:
    """Build the games-file payload from a battery JSON object."""
    records = battery["records"]
    games = []
    for i, rec in enumerate(records):
        moves = [int(m) for m in rec["moves"]]
        games.append({
            "game": i,
            "moves": moves,
            "winner": WINNER_TO_INT[rec["winner"]],
            "plies": len(moves),
        })
    return {"games": games, "complete": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battery", required=True,
                        help="battery / fast-loop JSON with records")
    parser.add_argument("--out", required=True,
                        help="games file to write")
    args = parser.parse_args()
    with open(args.battery, encoding="utf-8") as f:
        battery = json.load(f)
    payload = convert(battery)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    os.replace(tmp, args.out)
    print(f"converted {len(payload['games'])} games: "
          f"{args.battery} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

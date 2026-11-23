#!/usr/bin/env python3
"""Reconstruct distill games JSONs from labelled .npz shards (ML batch 6, v55).

v24/v37's games JSON files are gone, but every distill shard stores the
FULL game move list (``moves``) and winner — so the game records are
recoverable from the shards. This script rebuilds the
``generate_distill_data.py`` games format from a shard tree
(train + holdout dirs), preserving the original game indices (from the
shard filenames) so the game-level holdout rule
(``(game + 1) % holdout_every == 0``) reproduces the original split.

Verification (see research/iterations/iter-55.md §1.1): label the
reconstructed games with the ORIGINAL label mode (rollout2 for v37's
shards) and compare all values against the stored shard values — 0
mismatches required before the reconstruction is trusted for a new
labelling.

Usage:
    python ml/scripts/reconstruct_distill_games.py \\
        --shards ~/workspace/projects/gomoku-ml3-h2/ml/runs/ml3-h2/data \\
        --source v7 --seed-base 50000 \\
        --out ml/runs/ml6-h2/games-v7.json

The ``seed`` field is provenance only (the label command does not use
it): v7 games used seeds 50000+game, v18 60000+game, random 70000+game
(iter-44 §1.2).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True,
                        help="shard tree root (contains <source>/ and "
                        "<source>-holdout/ dirs)")
    parser.add_argument("--source", required=True,
                        help="game source (v7, v18, random, ...)")
    parser.add_argument("--seed-base", type=int, default=0,
                        help="seed base for provenance (seed = base + game)")
    parser.add_argument("--sims", type=int, default=50,
                        help="sims per move for provenance")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import numpy as np  # noqa: E402  (heavy import after arg parsing)

    games: dict[int, dict] = {}
    for sub in (args.source, f"{args.source}-holdout"):
        for path in sorted(glob.glob(os.path.join(args.shards, sub, "*.npz"))):
            m = re.search(r"_game(\d+)\.npz$", os.path.basename(path))
            if not m:
                print(f"skip {path}: no game index in filename",
                      file=sys.stderr)
                continue
            idx = int(m.group(1))
            z = np.load(path)
            moves = [int(x) for x in z["moves"]]
            winner = int(z["winner"])
            if idx in games:
                raise SystemExit(f"duplicate game {idx} in {sub}")
            games[idx] = {
                "game": idx,
                "source": args.source,
                "sims": args.sims,
                "seed": args.seed_base + idx,
                "moves": moves,
                "winner": winner,
                "plies": len(moves),
            }
    if not games:
        raise SystemExit(f"no shards found for source {args.source!r} "
                         f"under {args.shards}")
    payload = {"games": [games[i] for i in sorted(games)],
               "complete": True}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    os.replace(tmp, args.out)
    total_plies = sum(g["plies"] for g in payload["games"])
    print(f"reconstructed {len(payload['games'])} {args.source} games "
          f"({total_plies} plies) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

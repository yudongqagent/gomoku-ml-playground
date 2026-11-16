#!/usr/bin/env python3
"""Build a games JSON from B1 10k shards for the H1 diagnostic.

The B1 shards carry full move lists (``moves``) and winners, so
their positions can be reconstructed and labelled with the rollout
leaf exactly like H1's own games — WITHOUT replaying them. This is
the distribution-generalisation diagnostic (iter-24 §1.1/§4): the
trained net is scored on B1 positions (random-init net self-play,
a different distribution) to see whether the distillation
generalises off-distribution. No decision hinges on it.

Usage:
    python ml/scripts/b1_games_json.py --shards-dir <b3-b1-10k/data> \\
        --games 50 --out ml/runs/ml1-h1/games-b1-diag.json
    python ml/scripts/generate_distill_data.py label \\
        --games ml/runs/ml1-h1/games-b1-diag.json \\
        --out-dir ml/runs/ml1-h1/data/b1-diag --prefix b1diag
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.shards_dir, "*.npz")))
    if len(paths) < args.games:
        print(f"only {len(paths)} shards under {args.shards_dir}",
              file=sys.stderr)
        return 1
    import numpy as np  # noqa: E402

    games = []
    for i, path in enumerate(paths[: args.games]):
        with np.load(path) as z:
            moves = [int(m) for m in z["moves"]]
            winner = int(z["winner"])
        games.append({
            "game": i,
            "source": "b1-diag",
            "sims": 50,
            "seed": -1,
            "moves": moves,
            "winner": winner,
            "plies": len(moves),
        })
    payload = {"games": games, "complete": True}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    total_plies = sum(g["plies"] for g in games)
    print(f"wrote {len(games)} games ({total_plies} plies) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

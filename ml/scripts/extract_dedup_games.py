#!/usr/bin/env python3
"""Extract the DEDUPED B1 10k dataset into a games JSON for distillation.

The B1 10k dataset (``gomoku-b3-b1/ml/runs/b3-b1-10k/data``) holds
10,000 shards but only 3,244 unique games — same-seed resume legs
regenerated identical shards (see ``research/iterations/iter-25.md``
§2.1/§2.2). This script deduplicates by MD5 of the move sequence
(reusing ``split_holdout_unique.split_games``), orders the unique
games by first-occurrence shard index, and writes one record per
unique game in the ``generate_distill_data.py`` games-JSON format
(``{"game": i, "source", "sims", "seed", "moves", "winner",
"plies"}``), so the ``label`` phase can label every quiet position
with the rollout leaf.

The ``game`` index is the unique-order position (0-based), so the
label phase's ``--holdout-every 10`` reproduces v25's dedup-aware
holdout exactly (unique-order positions 9, 19, 29, ...). The
``seed`` field records the first-occurrence shard index for
traceability. Verification (exit non-zero on failure): the unique
count must be 3,244 and the implied holdout first-shard indices
must match the v25 split manifest's
``holdout_first_shard_indices`` exactly.

Usage:
    python ml/scripts/extract_dedup_games.py \\
        --data /home/hatch/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data \\
        --out ml/runs/ml2-h2/games-b1-dedup.json \\
        --manifest /home/hatch/workspace/projects/gomoku-ml1-h2/ml/runs/ml1-h2-10k/split-manifest.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from split_holdout_unique import split_games  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="source dir with B1 .npz shards")
    parser.add_argument("--out", required=True,
                        help="output games JSON path")
    parser.add_argument("--manifest", default=None,
                        help="v25 split manifest to verify the "
                        "holdout indices against")
    parser.add_argument("--prefix", default="b3-b1-10k",
                        help="shard filename prefix")
    parser.add_argument("--holdout-every", type=int, default=10,
                        help="holdout cadence to verify (default 10)")
    args = parser.parse_args()

    paths = sorted(glob.glob(
        os.path.join(args.data, f"{args.prefix}_game*.npz")))
    if not paths:
        print(f"no shards found under {args.data}", file=sys.stderr)
        return 2
    print(f"found {len(paths)} shards", file=sys.stderr)

    grouped = split_games(paths)
    unique_order = grouped["unique_order"]
    print(f"unique games: {len(unique_order)} "
          f"({len(paths) - len(unique_order)} duplicate shards)",
          file=sys.stderr)

    index_of = {p: i for i, p in enumerate(paths)}
    games = []
    for i, first_path in enumerate(unique_order):
        with np.load(first_path) as z:
            moves = [int(m) for m in z["moves"]]
            winner = int(z["winner"])
        games.append({
            "game": i,
            "source": "b1-dedup",
            "sims": 50,
            "seed": index_of[first_path],
            "moves": moves,
            "winner": winner,
            "plies": len(moves),
        })
    payload = {"games": games, "complete": True}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    os.replace(tmp, args.out)
    print(f"wrote {len(games)} unique games -> {args.out}",
          file=sys.stderr)

    # Verification: unique count + holdout indices vs the v25 manifest.
    errors = []
    if len(unique_order) != 3244:
        errors.append(f"unique count {len(unique_order)} != 3244")
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        mine = sorted(
            index_of[unique_order[i]]
            for i in range(len(unique_order))
            if (i + 1) % args.holdout_every == 0
        )
        theirs = sorted(manifest["holdout_first_shard_indices"])
        if mine != theirs:
            errors.append(
                f"holdout indices mismatch: {len(mine)} mine vs "
                f"{len(theirs)} manifest; "
                f"first diffs: mine-only {sorted(set(mine) - set(theirs))[:5]}, "
                f"manifest-only {sorted(set(theirs) - set(mine))[:5]}"
            )
        else:
            print(f"holdout indices match the v25 manifest exactly "
                  f"({len(mine)} games)", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
        return 1
    print("VERIFIED: 3,244 unique games; holdout split == v25's",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

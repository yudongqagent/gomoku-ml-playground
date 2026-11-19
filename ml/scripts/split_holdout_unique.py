#!/usr/bin/env python3
"""Split a self-play shard dir into train / holdout by UNIQUE game.

The B1 10k dataset contains duplicate games: same-seed resume legs
regenerated identical shards (the first 4,350 shards hold only 1,449
unique move-sequences — see ``research/iterations/iter-25.md`` §2.1).
A holdout split by shard index leaks: a holdout shard's exact
duplicate usually sits in the training set. This script splits by
unique game instead, so no holdout game (in any copy) appears in
training.

Method (pre-registered, iter-25 §1.2):
* deduplicate shards by MD5 of the ``moves`` array (verified:
  duplicates are byte-identical shards, not just identical move
  lists);
* order unique games by first-occurrence shard index;
* every 10th unique game (positions 9, 19, 29, ... in unique order)
  is a HOLDOUT game;
* ``train/`` gets every shard EXCEPT holdout games' shards (ALL
  copies of a holdout game are excluded, not just the first);
* ``val/`` gets one shard per holdout game (its first occurrence);
* ``val-clean/`` gets the holdout games whose first occurrence is at
  shard index >= ``--new-leg-start`` (default 4350) — games from the
  completion leg only, which NEITHER v8's net (trained on the first
  4,350 shards) nor the v25 net trained on, so both nets can be
  compared honestly on them;
* ``val-v8-leaky/`` gets v8's ORIGINAL holdout shards (every 10th
  shard of the first 4,350, indices 9, 19, ..., 4349) — labelled
  LEAKY, for a like-for-like comparison with v8's published 0.7165
  only.

Outputs a ``split-manifest.json`` with the counts and the holdout
game indices, so the split is reproducible and auditable. Copies
(not moves) — the source dir is untouched.

Usage:
    python ml/scripts/split_holdout_unique.py \\
        --data /home/hatch/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data \\
        --out ml/runs/ml1-h2-10k --prefix b3-b1-10k --new-leg-start 4350
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import glob

import numpy as np  # noqa: E402


def shard_game_hash(path: str) -> str:
    """MD5 of the shard's move sequence (the game identity)."""
    with np.load(path) as z:
        return hashlib.md5(z["moves"].tobytes()).hexdigest()


def split_games(paths: list[str]) -> dict:
    """Group sorted shard paths by game hash.

    Returns {"unique_order": [first-occurrence path per unique game,
    in first-occurrence order], "copies": {hash: [paths...]}}.
    """
    copies: dict[str, list[str]] = {}
    for path in paths:
        copies.setdefault(shard_game_hash(path), []).append(path)
    index_of = {p: i for i, p in enumerate(paths)}
    unique_order = sorted(
        (group[0] for group in copies.values()),
        key=lambda p: index_of[p],
    )
    return {"unique_order": unique_order, "copies": copies}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="source dir with .npz shards")
    parser.add_argument("--out", required=True,
                        help="output dir (train/val/val-clean/val-v8-leaky "
                        "subdirs + split-manifest.json)")
    parser.add_argument("--prefix", default="b3-b1-10k",
                        help="shard filename prefix (glob '<prefix>_game*.npz')")
    parser.add_argument("--new-leg-start", type=int, default=4350,
                        help="first shard index of the completion leg "
                        "(val-clean boundary)")
    parser.add_argument("--holdout-every", type=int, default=10,
                        help="every Nth unique game is holdout (default 10)")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, f"{args.prefix}_game*.npz")))
    if not paths:
        print(f"no shards found under {args.data}", file=sys.stderr)
        return 2
    print(f"found {len(paths)} shards", file=sys.stderr)

    grouped = split_games(paths)
    unique_order = grouped["unique_order"]
    copies = grouped["copies"]
    print(f"unique games: {len(unique_order)} "
          f"({len(paths) - len(unique_order)} duplicate shards)",
          file=sys.stderr)

    # Every Nth unique game (1-based positions N, 2N, ...) is holdout.
    holdout_hashes = set()
    holdout_first_paths = []
    for pos, first_path in enumerate(unique_order, start=1):
        if pos % args.holdout_every == 0:
            h = shard_game_hash(first_path)
            holdout_hashes.add(h)
            holdout_first_paths.append(first_path)
    print(f"holdout unique games: {len(holdout_hashes)}", file=sys.stderr)

    train_dir = os.path.join(args.out, "data")
    val_dir = os.path.join(args.out, "val-data")
    clean_dir = os.path.join(args.out, "val-clean")
    leaky_dir = os.path.join(args.out, "val-v8-leaky")
    for d in (train_dir, val_dir, clean_dir, leaky_dir):
        os.makedirs(d, exist_ok=True)

    index_of = {p: i for i, p in enumerate(paths)}
    n_train = n_excluded = 0
    for path in paths:
        h = shard_game_hash(path)
        if h in holdout_hashes:
            n_excluded += 1
            continue
        shutil.copy2(path, os.path.join(train_dir, os.path.basename(path)))
        n_train += 1
    n_val_clean = 0
    for first_path in holdout_first_paths:
        name = os.path.basename(first_path)
        shutil.copy2(first_path, os.path.join(val_dir, name))
        if index_of[first_path] >= args.new_leg_start:
            shutil.copy2(first_path, os.path.join(clean_dir, name))
            n_val_clean += 1
    # v8's original (leaky) holdout: every 10th shard of the first 4350.
    n_leaky = 0
    for i in range(9, min(4350, len(paths)), 10):
        shutil.copy2(paths[i],
                     os.path.join(leaky_dir, os.path.basename(paths[i])))
        n_leaky += 1

    manifest = {
        "source_data": os.path.abspath(args.data),
        "prefix": args.prefix,
        "total_shards": len(paths),
        "unique_games": len(unique_order),
        "duplicate_shards": len(paths) - len(unique_order),
        "holdout_every": args.holdout_every,
        "holdout_unique_games": len(holdout_hashes),
        "train_shards": n_train,
        "excluded_shards_all_copies_of_holdout": n_excluded,
        "val_shards": len(holdout_first_paths),
        "val_clean_shards_new_leg_only": n_val_clean,
        "new_leg_start": args.new_leg_start,
        "val_v8_leaky_shards": n_leaky,
        "holdout_first_shard_indices": sorted(
            index_of[p] for p in holdout_first_paths
        ),
    }
    with open(os.path.join(args.out, "split-manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"train: {n_train} shards -> {train_dir}", file=sys.stderr)
    print(f"val: {len(holdout_first_paths)} shards -> {val_dir}",
          file=sys.stderr)
    print(f"val-clean (new leg only): {n_val_clean} shards -> {clean_dir}",
          file=sys.stderr)
    print(f"val-v8-leaky: {n_leaky} shards -> {leaky_dir}", file=sys.stderr)
    print(f"excluded (holdout copies): {n_excluded} shards", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

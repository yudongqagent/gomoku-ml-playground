#!/usr/bin/env python3
"""Build the v30 training set: DEDUPED + COLOUR-BALANCED unique games.

The B1 10k dataset holds 10,000 shards but only 3,244 unique games
(same-seed resume replays — see ``research/iterations/iter-25.md``
§2.1/§2.2), and the outcomes are Black-skewed (Black wins 53.1%).
v25 trained on all non-holdout shards (duplicates up to 7x) and its
White play collapsed (vs v8 Black 0.73 / White 0.15 — iter-25 §5.5).
v30 tests whether dedup + colour-balancing fixes it.

Method (pre-registered, iter-30 §1.1):
* deduplicate the 10,000 shards by MD5 of the ``moves`` array
  (``split_holdout_unique.split_games``);
* REUSE v25's dedup-aware holdout exactly (every 10th unique game
  in first-occurrence order) — verified against v25's
  ``split-manifest.json``: the recomputed holdout set must match
  the manifest's ``holdout_first_shard_indices`` exactly, or the
  script fails loudly (the identical holdout makes the v8/v25/v30
  before/after comparison exact);
* train unique games = unique games NOT in the holdout;
* COLOUR-BALANCE by DOWNSAMPLING the majority winner class
  (expected: Black-win games) to the minority count, with a fixed
  RNG seed (default 0) — downsampling, not upweighting, because it
  keeps v8's training recipe byte-identical (no loss-weight code
  change) and does not reintroduce duplicates (upweighting by
  duplication would defeat the dedup); the data loss is small
  (~6% of unique train games, the skew is mild);
* copy ONE shard per selected unique game (its first occurrence)
  into the output dir, and write a manifest with the counts, the
  dropped games, and the position-level balance check.

Draws (none in this dataset) are kept unconditionally — they do
not skew the colour balance.

Usage:
    python ml/scripts/build_dedup_balanced_train.py \\
        --data /home/hatch/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k/data \\
        --split-manifest /home/hatch/workspace/projects/gomoku-ml1-h2/ml/runs/ml1-h2-10k/split-manifest.json \\
        --out ml/runs/ml2-h1-10k/data --balance-seed 0
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from gomoku_ml import game  # noqa: E402
import split_holdout_unique  # noqa: E402


def balance_winners(winners: list[int], seed: int) -> tuple[list[int], list[int]]:
    """Downsample the majority winner class to the minority count.

    ``winners`` is one entry per unique train game (game.BLACK,
    game.WHITE, or game.EMPTY for a draw). Returns ``(kept, dropped)``
    as index lists into ``winners``. Draws are always kept. The
    majority class is shuffled with ``np.random.default_rng(seed)``
    and truncated to the minority count; the minority class is kept
    whole. Deterministic for a fixed seed.
    """
    black = [i for i, w in enumerate(winners) if w == game.BLACK]
    white = [i for i, w in enumerate(winners) if w == game.WHITE]
    draws = [i for i, w in enumerate(winners) if w == game.EMPTY]
    if len(black) == len(white):
        return sorted(black + white + draws), []
    majority, minority = (black, white) if len(black) > len(white) else (white, black)
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(majority))
    keep_majority = shuffled[: len(minority)]
    dropped = shuffled[len(minority):]
    kept = sorted(keep_majority + minority + draws)
    return kept, sorted(dropped)


def shard_winner(path: str) -> int:
    """Winner stored in a shard (game.BLACK / game.WHITE / game.EMPTY)."""
    with np.load(path) as z:
        return int(z["winner"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="source dir with .npz shards")
    parser.add_argument("--split-manifest", required=True,
                        help="v25 split-manifest.json (holdout to reuse)")
    parser.add_argument("--out", required=True,
                        help="output dir for the deduped+balanced train shards")
    parser.add_argument("--prefix", default="b3-b1-10k",
                        help="shard filename prefix (glob '<prefix>_game*.npz')")
    parser.add_argument("--balance-seed", type=int, default=0,
                        help="RNG seed for the downsampling shuffle")
    parser.add_argument("--manifest-out", default=None,
                        help="manifest path (default: <out>/../dedup-balance-manifest.json)")
    args = parser.parse_args()

    with open(args.split_manifest, encoding="utf-8") as f:
        split = json.load(f)

    paths = sorted(glob.glob(os.path.join(args.data, f"{args.prefix}_game*.npz")))
    if not paths:
        print(f"no shards found under {args.data}", file=sys.stderr)
        return 2
    print(f"found {len(paths)} shards", file=sys.stderr)

    grouped = split_holdout_unique.split_games(paths)
    unique_order = grouped["unique_order"]
    print(f"unique games: {len(unique_order)} "
          f"({len(paths) - len(unique_order)} duplicate shards)", file=sys.stderr)
    if len(unique_order) != split["unique_games"]:
        print(f"ERROR: recomputed unique count {len(unique_order)} != "
              f"manifest {split['unique_games']}", file=sys.stderr)
        return 2

    # Recompute v25's holdout (every 10th unique game, 1-based) and
    # verify it matches the manifest exactly.
    index_of = {p: i for i, p in enumerate(paths)}
    holdout_every = split["holdout_every"]
    holdout_first_paths = [
        p for pos, p in enumerate(unique_order, start=1)
        if pos % holdout_every == 0
    ]
    holdout_indices = sorted(index_of[p] for p in holdout_first_paths)
    if holdout_indices != sorted(split["holdout_first_shard_indices"]):
        print("ERROR: recomputed holdout does not match the manifest "
              "(dedup ordering changed?)", file=sys.stderr)
        return 2
    print(f"holdout verified: {len(holdout_indices)} unique games "
          f"(manifest match)", file=sys.stderr)
    holdout_hashes = {
        split_holdout_unique.shard_game_hash(p) for p in holdout_first_paths
    }

    # Train unique games: unique games not in the holdout.
    train_unique = [
        p for p in unique_order
        if split_holdout_unique.shard_game_hash(p) not in holdout_hashes
    ]
    print(f"train unique games: {len(train_unique)}", file=sys.stderr)

    winners = [shard_winner(p) for p in train_unique]
    n_black = sum(1 for w in winners if w == game.BLACK)
    n_white = sum(1 for w in winners if w == game.WHITE)
    n_draw = sum(1 for w in winners if w == game.EMPTY)
    print(f"train unique winners: Black {n_black}, White {n_white}, "
          f"draws {n_draw}", file=sys.stderr)

    kept, dropped = balance_winners(winners, args.balance_seed)
    print(f"balanced: kept {len(kept)} games, dropped {len(dropped)} "
          f"(majority downsample, seed {args.balance_seed})", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)
    total_samples = 0
    # Position-level balance check: side-to-move x target counts.
    pos_black_win = pos_black_loss = pos_white_win = pos_white_loss = 0
    for i in kept:
        src = train_unique[i]
        shutil.copy2(src, os.path.join(args.out, os.path.basename(src)))
        with np.load(src) as z:
            values = z["value"]
            total_samples += len(values)
            side = np.arange(len(values)) % 2  # 0 = Black to move
            pos_black_win += int(((side == 0) & (values > 0)).sum())
            pos_black_loss += int(((side == 0) & (values < 0)).sum())
            pos_white_win += int(((side == 1) & (values > 0)).sum())
            pos_white_loss += int(((side == 1) & (values < 0)).sum())

    manifest = {
        "source_data": os.path.abspath(args.data),
        "split_manifest": os.path.abspath(args.split_manifest),
        "total_shards": len(paths),
        "unique_games": len(unique_order),
        "holdout_unique_games": len(holdout_indices),
        "holdout_verified_against_manifest": True,
        "train_unique_games": len(train_unique),
        "train_unique_winners": {"black": n_black, "white": n_white, "draws": n_draw},
        "balance_method": "downsample majority winner class to minority count",
        "balance_seed": args.balance_seed,
        "kept_games": len(kept),
        "dropped_games": len(dropped),
        "dropped_first_shard_indices": sorted(index_of[train_unique[i]] for i in dropped),
        "kept_first_shard_indices": sorted(index_of[train_unique[i]] for i in kept),
        "total_samples": total_samples,
        "position_balance": {
            "black_to_move_win": pos_black_win,
            "black_to_move_loss": pos_black_loss,
            "white_to_move_win": pos_white_win,
            "white_to_move_loss": pos_white_loss,
        },
    }
    manifest_path = args.manifest_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out)), "dedup-balance-manifest.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"train shards: {len(kept)} -> {args.out} "
          f"({total_samples} samples)", file=sys.stderr)
    print(f"position balance: Black-to-move win {pos_black_win} / loss "
          f"{pos_black_loss}; White-to-move win {pos_white_win} / loss "
          f"{pos_white_loss}", file=sys.stderr)
    print(f"manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

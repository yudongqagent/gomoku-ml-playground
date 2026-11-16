"""Tests for the unique-game holdout split (ML batch 1 H2, v25)."""

import json
import os
import subprocess
import sys

import numpy as np


def _shard(path, moves, winner=1):
    """A minimal valid shard: planes/policy/value/moves/winner."""
    t = len(moves)
    policy = np.zeros((t, 225), dtype=np.float32)
    policy[np.arange(t), np.asarray(moves) % 225] = 1.0
    np.savez_compressed(
        path,
        planes=np.zeros((t, 8, 15, 15), dtype=np.float32),
        policy=policy,
        value=np.ones(t, dtype=np.float32) * winner,
        moves=np.asarray(moves, dtype=np.int64),
        winner=np.int64(winner),
    )


def _build_source(tmp_path):
    """25 shards, 12 unique games: game k appears at indices k and
    k+12 (duplicates), plus game 12 alone at index 24.

    Unique order by first occurrence: games 0..11, then 12.
    Holdout (every 10th unique, 1-based positions 10, 20, ...):
    only unique position 10 -> game 9.
    """
    src = tmp_path / "src"
    src.mkdir()
    for i in range(25):
        g = i % 12 if i < 24 else 12
        moves = [10 + g, 20 + g, 30 + g, 40 + g]
        _shard(src / f"b3-b1-10k_game{i:04d}.npz", moves)
    return src


def _run_split(src, out, new_leg_start=24):
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "split_holdout_unique.py",
    )
    r = subprocess.run(
        [sys.executable, script, "--data", str(src), "--out", str(out),
         "--prefix", "b3-b1-10k", "--new-leg-start", str(new_leg_start)],
        capture_output=True, text=True, check=True,
    )
    return r


def test_split_by_unique_game(tmp_path):
    src = _build_source(tmp_path)
    out = tmp_path / "out"
    _run_split(src, out)
    with open(out / "split-manifest.json", encoding="utf-8") as f:
        m = json.load(f)
    assert m["total_shards"] == 25
    assert m["unique_games"] == 13
    assert m["duplicate_shards"] == 12
    assert m["holdout_unique_games"] == 1
    # Holdout game 9 has TWO copies (indices 9 and 21); both are
    # excluded from train, one lands in val.
    assert m["train_shards"] == 23
    assert m["excluded_shards_all_copies_of_holdout"] == 2
    assert m["val_shards"] == 1
    assert m["holdout_first_shard_indices"] == [9]
    # No holdout move-sequence appears anywhere in train/.
    train_moves = set()
    for name in os.listdir(out / "data"):
        with np.load(out / "data" / name) as z:
            train_moves.add(tuple(z["moves"].tolist()))
    assert (19, 29, 39, 49) not in train_moves  # game 9's moves
    assert len(train_moves) == 12  # 13 unique minus the holdout game
    # Val holds game 9's first occurrence only.
    assert sorted(os.listdir(out / "val-data")) == [
        "b3-b1-10k_game0009.npz"
    ]


def test_val_clean_uses_new_leg_boundary(tmp_path):
    # new_leg_start=24: the holdout game (first occurrence index 9)
    # is an OLD-leg game, so val-clean is empty.
    src = _build_source(tmp_path)
    out = tmp_path / "out"
    _run_split(src, out, new_leg_start=24)
    with open(out / "split-manifest.json", encoding="utf-8") as f:
        m = json.load(f)
    assert m["val_clean_shards_new_leg_only"] == 0
    assert os.listdir(out / "val-clean") == []
    # new_leg_start=0: every holdout game counts as new-leg.
    out2 = tmp_path / "out2"
    _run_split(src, out2, new_leg_start=0)
    with open(out2 / "split-manifest.json", encoding="utf-8") as f:
        m2 = json.load(f)
    assert m2["val_clean_shards_new_leg_only"] == 1
    assert sorted(os.listdir(out2 / "val-clean")) == [
        "b3-b1-10k_game0009.npz"
    ]


def test_val_v8_leaky_is_every_10th_shard(tmp_path):
    src = _build_source(tmp_path)
    out = tmp_path / "out"
    _run_split(src, out)
    # Every 10th shard of the first 4350: indices 9 and 19 here.
    assert sorted(os.listdir(out / "val-v8-leaky")) == [
        "b3-b1-10k_game0009.npz",
        "b3-b1-10k_game0019.npz",
    ]

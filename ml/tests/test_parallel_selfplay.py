"""Tests for B1 parallel self-play (selfplay.generate_games_parallel,
merge_worker_shards, recover_parallel_shards).

The merge is the part that has to be right: several writers, one run
dir, dense resume-safe numbering, a manifest consistent with the
shards on disk, and merged data exactly equal to the union of the
workers' shards. The parallel-run tests use a tiny random-init net
(4 sims, 30 max moves) so they stay fast; the real-net throughput
numbers live in research/iterations/b1-parallel-selfplay.md, not here.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gomoku_ml import game, selfplay
from gomoku_ml.mcts import MCTS
from gomoku_ml.model import GomokuNet

# Pin torch to 1 thread for these tests: the tiny nets here run one
# small forward pass at a time, so more threads is pure OpenMP
# synchronisation overhead — and under contention with sibling jobs on
# this 4-CPU VM the default (4) threads slowed a batched forward pass
# ~290x (579 ms/call vs 2.0 ms at 1 thread, measured 2026-11-13 on
# the 16ch/2block net), turning a 1-minute test file into a 10-minute
# one. The B1 workers pin themselves to 1 thread for the same reason.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _tiny_net():
    torch.manual_seed(0)
    return GomokuNet(channels=16, blocks=2)


def _tiny_spec():
    return selfplay.NetSpec(
        arch_name="resnet",
        arch_params={"channels": 16, "blocks": 2},
        checkpoint=None,
    )


def _make_shards(out_dir, n, run_name="test"):
    """Generate n real shards with the batched path (tiny net)."""
    mcts = MCTS(_tiny_net())
    return selfplay.generate_games_batched(
        mcts, num_games=n, num_simulations=4, out_dir=str(out_dir),
        run_name=run_name, concurrency=2,
        temperature_moves=2, max_moves=30,
    )


def _shard_moves(path):
    with np.load(path) as z:
        return tuple(int(m) for m in z["moves"])


def _assert_valid_shard(path):
    with np.load(path) as z:
        n = len(z["moves"])
        assert n > 0
        assert z["planes"].shape == (n, 8, 15, 15)
        assert z["policy"].shape == (n, 225)
        assert z["value"].shape == (n,)
        assert np.allclose(z["policy"].sum(axis=1), 1.0)
        assert set(np.unique(z["value"])).issubset({-1.0, 0.0, 1.0})
        assert int(z["winner"]) in (-1, 0, 1)


def test_merge_worker_shards_merges_union(tmp_path):
    """Merged data equals the union of the workers' shards: same
    games, dense numbering from 0, manifest consistent."""
    w0 = tmp_path / "worker0"
    w1 = tmp_path / "worker1"
    paths0 = _make_shards(w0, 2)
    paths1 = _make_shards(w1, 3)
    before = sorted(_shard_moves(p) for p in paths0 + paths1)
    out = tmp_path / "merged"
    result = selfplay.merge_worker_shards(str(out), "test", [str(w0), str(w1)])
    assert len(result["merged"]) == 5
    assert result["skipped"] == []
    merged_paths = [new for _, new in result["merged"]]
    assert sorted(_shard_moves(p) for p in merged_paths) == before
    # Dense numbering, no overwrites.
    assert sorted(Path(p).name for p in merged_paths) == [
        f"test_game{i:04d}.npz" for i in range(5)
    ]
    # Manifest consistent with the shards on disk.
    manifest = selfplay.load_manifest(str(out))
    assert manifest["games"] == 5
    assert [s["file"] for s in manifest["shards"]] == [
        f"test_game{i:04d}.npz" for i in range(5)
    ]
    assert manifest["total_samples"] == sum(
        len(_shard_moves(p)) for p in merged_paths
    )
    for p in merged_paths:
        _assert_valid_shard(p)


def test_merge_resume_safe_numbering(tmp_path):
    """Two merges into one dir append, never overwrite: numbering
    continues after the existing shards and the manifest stays
    consistent."""
    out = tmp_path / "merged"
    w0 = tmp_path / "w0"
    w1 = tmp_path / "w1"
    _make_shards(w0, 2)
    _make_shards(w1, 2)
    first = selfplay.merge_worker_shards(str(out), "test", [str(w0)])
    second = selfplay.merge_worker_shards(str(out), "test", [str(w1)])
    assert len(first["merged"]) == 2
    assert len(second["merged"]) == 2
    names = sorted(Path(p).name for _, p in first["merged"] + second["merged"])
    assert names == [f"test_game{i:04d}.npz" for i in range(4)]
    assert selfplay.existing_game_count(str(out), "test") == 4
    assert selfplay.load_manifest(str(out))["games"] == 4


def test_merge_skips_corrupt_shard(tmp_path):
    """A shard truncated mid-write (worker killed) is dropped, not
    merged — and does not punch a hole in the numbering."""
    w0 = tmp_path / "worker0"
    paths = _make_shards(w0, 2)
    # Corrupt shard with a higher index than the valid ones.
    corrupt = w0 / "test_game0002.npz"
    corrupt.write_bytes(b"this is not a zip file")
    assert len(paths) == 2
    out = tmp_path / "merged"
    result = selfplay.merge_worker_shards(str(out), "test", [str(w0)])
    assert len(result["merged"]) == 2
    assert len(result["skipped"]) == 1
    assert not corrupt.exists()  # dropped, not left behind
    names = sorted(Path(p).name for _, p in result["merged"])
    assert names == ["test_game0000.npz", "test_game0001.npz"]
    assert selfplay.load_manifest(str(out))["games"] == 2


def test_recover_parallel_shards(tmp_path):
    """Orphaned shards from an interrupted parallel run (left under
    .parallel-*/) are merged into the run dir and the leftover dirs
    removed; a second call is a no-op."""
    out = tmp_path / "data"
    out.mkdir()
    orphan_root = out / ".parallel-123-456"
    w0 = orphan_root / "worker0"
    w1 = orphan_root / "worker1"
    _make_shards(w0, 2)
    _make_shards(w1, 1)
    # A pre-existing merged shard: recovery must append after it.
    _make_shards(out, 1)
    assert selfplay.existing_game_count(str(out), "test") == 1
    result = selfplay.recover_parallel_shards(str(out), "test")
    assert len(result["merged"]) == 3
    assert not orphan_root.exists()
    assert selfplay.existing_game_count(str(out), "test") == 4
    assert selfplay.load_manifest(str(out))["games"] == 4
    again = selfplay.recover_parallel_shards(str(out), "test")
    assert again["merged"] == [] and again["skipped"] == []


def test_generate_games_parallel_completes(tmp_path):
    """A short 2-worker run completes with valid shards, a consistent
    manifest, and distinct games across workers (different seeds)."""
    out = tmp_path / "data"
    paths = selfplay.generate_games_parallel(
        _tiny_spec(), num_games=4, num_simulations=4, out_dir=str(out),
        run_name="test", workers=2, concurrency=2, seed=0,
        temperature_moves=2, max_moves=30,
    )
    assert len(paths) == 4
    assert len(set(paths)) == 4
    assert sorted(Path(p).name for p in paths) == [
        f"test_game{i:04d}.npz" for i in range(4)
    ]
    manifest = selfplay.load_manifest(str(out))
    assert manifest is not None
    assert manifest["games"] == 4
    assert manifest["total_samples"] == sum(
        len(_shard_moves(p)) for p in paths
    )
    for p in paths:
        _assert_valid_shard(p)
    # Workers seeded differently: the games must not all be identical.
    assert len(set(_shard_moves(p) for p in paths)) >= 2
    # No leftover worker dirs.
    assert list(out.glob(".parallel-*")) == []
    # Merged selfplay_state.json written by the parent.
    state = json.loads((out / "selfplay_state.json").read_text())
    assert state["games_completed_this_call"] == 4
    assert state["games_in_manifest"] == 4
    assert state["workers"] == 2


def test_generate_games_parallel_resume_two_runs(tmp_path):
    """Two parallel calls into one dir append with dense numbering —
    the resume story for an interrupted 10k run."""
    out = tmp_path / "data"
    kwargs = dict(
        net_spec=_tiny_spec(), num_simulations=4, out_dir=str(out),
        run_name="test", workers=2, concurrency=2, seed=0,
        temperature_moves=2, max_moves=30,
    )
    first = selfplay.generate_games_parallel(num_games=2, **kwargs)
    second = selfplay.generate_games_parallel(num_games=2, **kwargs)
    assert len(first) == 2 and len(second) == 2
    assert not set(first) & set(second)
    assert selfplay.existing_game_count(str(out), "test") == 4
    assert selfplay.load_manifest(str(out))["games"] == 4


def test_generate_games_parallel_events(tmp_path):
    """on_event receives one selfplay_game event per merged game,
    with final merged game numbers in order."""
    out = tmp_path / "data"
    events: list[dict] = []
    selfplay.generate_games_parallel(
        _tiny_spec(), num_games=4, num_simulations=4, out_dir=str(out),
        run_name="test", workers=2, concurrency=2, seed=0,
        on_event=events.append,
        temperature_moves=2, max_moves=30,
    )
    assert len(events) == 4
    assert [e["game"] for e in events] == [1, 2, 3, 4]
    assert all(e["event"] == "selfplay_game" for e in events)
    assert all(e["games_total"] == 4 for e in events)
    assert all(e["moves"] > 0 for e in events)

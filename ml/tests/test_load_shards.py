"""Tests for gomoku_ml.train.load_shards (pre-allocated loader)."""

import numpy as np
import pytest

from gomoku_ml.train import load_shards


def _shard(path, n, fill):
    np.savez_compressed(
        path,
        planes=np.full((n, 8, 15, 15), fill, dtype=np.float32),
        policy=np.full((n, 225), fill, dtype=np.float32),
        value=np.full(n, fill, dtype=np.float32),
    )


def test_load_shards_concatenates_in_sorted_order(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    _shard(d / "b_game0001.npz", 3, 1.0)
    _shard(d / "a_game0000.npz", 2, 0.0)
    _shard(d / "c_game0002.npz", 4, 2.0)
    planes, policy, values = load_shards([str(d)])
    # Sorted path order: a (0.0 x2), b (1.0 x3), c (2.0 x4).
    assert planes.shape == (9, 8, 15, 15)
    assert policy.shape == (9, 225)
    assert values.shape == (9,)
    assert planes.dtype == policy.dtype == values.dtype == np.float32
    assert list(values) == [0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
    assert list(policy[:, 0]) == list(values)
    assert list(planes[:, 0, 0, 0]) == list(values)


def test_load_shards_matches_concatenate(tmp_path):
    # The pre-allocated loader returns exactly what the old
    # build-lists-then-concatenate implementation returned.
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    d.mkdir()
    expected_planes, expected_policy, expected_values = [], [], []
    for i, n in enumerate((5, 1, 7)):
        planes = rng.random((n, 8, 15, 15), dtype=np.float32)
        policy = rng.random((n, 225), dtype=np.float32)
        values = rng.random(n, dtype=np.float32)
        np.savez_compressed(
            d / f"game{i:04d}.npz",
            planes=planes, policy=policy, value=values,
        )
        expected_planes.append(planes)
        expected_policy.append(policy)
        expected_values.append(values)
    planes, policy, values = load_shards([str(d)])
    assert np.array_equal(planes, np.concatenate(expected_planes))
    assert np.array_equal(policy, np.concatenate(expected_policy))
    assert np.array_equal(values, np.concatenate(expected_values))


def test_load_shards_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_shards([str(tmp_path)])

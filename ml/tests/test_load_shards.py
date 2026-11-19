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


def test_load_shards_mmap_bundle_matches_npz(tmp_path):
    # A memmap bundle built from shards loads the same arrays as the
    # shards themselves (same content, same order) — as memmaps.
    import subprocess
    import sys

    rng = np.random.default_rng(1)
    src = tmp_path / "shards"
    src.mkdir()
    for i, n in enumerate((4, 6)):
        np.savez_compressed(
            src / f"game{i:04d}.npz",
            planes=rng.random((n, 8, 15, 15), dtype=np.float32),
            policy=rng.random((n, 225), dtype=np.float32),
            value=rng.random(n, dtype=np.float32),
        )
    bundle = tmp_path / "bundle"
    script = (
        __file__.rsplit("/tests/", 1)[0] + "/scripts/build_mmap_bundle.py"
    )
    result = subprocess.run(
        [sys.executable, script, "--data", str(src), "--out", str(bundle)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    p_npz, pol_npz, v_npz = load_shards([str(src)])
    p_mm, pol_mm, v_mm = load_shards([str(bundle)])
    assert isinstance(p_mm, np.memmap)
    assert np.array_equal(p_mm, p_npz)
    assert np.array_equal(pol_mm, pol_npz)
    assert np.array_equal(v_mm, v_npz)


def test_load_shards_partial_bundle_raises(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(str(bundle / "planes.npy"), np.zeros((2, 8, 15, 15), dtype=np.float32))
    with pytest.raises(ValueError, match="partial memmap bundle"):
        load_shards([str(bundle)])


def test_train_memmap_matches_ram(tmp_path):
    # Training on memmap-backed arrays produces the identical loss
    # history as training on the same arrays in RAM (the lazy-index
    # refactor must not change the math).
    import torch

    from gomoku_ml.architectures import build_architecture
    from gomoku_ml.train import train

    rng = np.random.default_rng(2)
    n = 128
    planes = rng.random((n, 8, 15, 15), dtype=np.float32)
    policy = rng.random((n, 225), dtype=np.float32)
    policy /= policy.sum(axis=1, keepdims=True)
    values = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    np.save(str(bundle / "planes.npy"), planes)
    np.save(str(bundle / "policy.npy"), policy)
    np.save(str(bundle / "values.npy"), values)
    p_mm, pol_mm, v_mm = load_shards([str(bundle)])

    kwargs = dict(
        epochs=1,
        batch_size=32,
        lr=0.001,
        val_fraction=0.1,
        seed=0,
        augmentation=True,
        log_every=1000,
    )
    torch.manual_seed(0)
    model_ram = build_architecture("resnet", {"channels": 8, "blocks": 1})
    summary_ram = train(model_ram, planes, policy, values, **kwargs)
    torch.manual_seed(0)
    model_mm = build_architecture("resnet", {"channels": 8, "blocks": 1})
    summary_mm = train(model_mm, p_mm, pol_mm, v_mm, **kwargs)
    # Identical math — compare everything except wall-clock seconds.
    hist_ram = [{k: v for k, v in e.items() if k != "seconds"} for e in summary_ram["history"]]
    hist_mm = [{k: v for k, v in e.items() if k != "seconds"} for e in summary_mm["history"]]
    assert hist_ram == hist_mm

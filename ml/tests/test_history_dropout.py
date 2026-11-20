"""Tests for history-plane dropout (ML batch 4, v43).

``gomoku_ml.train.drop_history_planes`` zeroes the history planes
(2 = last move, 4-7 = boards 2/4 plies ago) on a per-sample
Bernoulli draw, so the value net cannot route through history
features (v37's failure: v7 holdout 0.7395 -> 0.5163 without
history planes). The dropped planes must match the holdout
gate's no-history view exactly, or training and the gate would
measure different interventions.
"""

import importlib.util
import os

import numpy as np

from gomoku_ml.train import HISTORY_PLANES, drop_history_planes

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")


def _batch(n=16, fill=1.0):
    # Every plane nonzero, so zeroing is detectable on any plane.
    return np.full((n, 8, 15, 15), fill, dtype=np.float32)


def test_history_planes_match_holdout_view():
    # The gate (eval_value_holdout.py) zeroes _HISTORY_PLANES; the
    # training dropout must zero exactly the same planes.
    spec = importlib.util.spec_from_file_location(
        "eval_value_holdout",
        os.path.join(SCRIPTS, "eval_value_holdout.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(HISTORY_PLANES) == tuple(module._HISTORY_PLANES)
    assert HISTORY_PLANES == (2, 4, 5, 6, 7)


def test_drop_rate_zero_is_noop():
    batch = _batch()
    before = batch.copy()
    rng = np.random.default_rng(0)
    out = drop_history_planes(batch, rng, 0.0)
    assert out is batch
    np.testing.assert_array_equal(batch, before)


def test_drop_rate_one_zeroes_history_only():
    batch = _batch()
    rng = np.random.default_rng(0)
    drop_history_planes(batch, rng, 1.0)
    for p in HISTORY_PLANES:
        assert not batch[:, p].any(), f"plane {p} not zeroed"
    for p in (0, 1, 3):
        assert batch[:, p].all(), f"current plane {p} wrongly zeroed"


def test_drop_rate_half_zeroes_a_subset_jointly():
    batch = _batch()
    rng = np.random.default_rng(0)
    drop_history_planes(batch, rng, 0.5)
    dropped = [i for i in range(len(batch)) if not batch[i, 2].any()]
    kept = [i for i in range(len(batch)) if batch[i, 2].any()]
    # Both outcomes occur at p=0.5 with this seed (deterministic).
    assert dropped and kept
    for i in dropped:
        # ALL history planes zeroed jointly on a dropped sample.
        for p in HISTORY_PLANES:
            assert not batch[i, p].any()
        # Current planes untouched even on dropped samples.
        for p in (0, 1, 3):
            assert batch[i, p].all()
    for i in kept:
        # Kept samples are entirely untouched.
        assert batch[i].all()


def test_drop_is_fresh_each_call():
    # Two calls with the same rng state draw different masks (the
    # training loop calls once per batch per epoch, so each epoch
    # sees a different subset).
    rng = np.random.default_rng(0)
    b1 = _batch()
    b2 = _batch()
    drop_history_planes(b1, rng, 0.5)
    drop_history_planes(b2, rng, 0.5)
    d1 = tuple(not b1[i, 2].any() for i in range(len(b1)))
    d2 = tuple(not b2[i, 2].any() for i in range(len(b2)))
    assert d1 != d2


def test_drop_rate_above_one_errors():
    import pytest

    with pytest.raises(ValueError, match="history_dropout"):
        drop_history_planes(_batch(), np.random.default_rng(0), 1.5)


def test_train_accepts_history_dropout(tmp_path):
    # A tiny end-to-end run through train() with dropout on: the
    # loop completes, writes checkpoints, and returns history.
    import torch

    from gomoku_ml.architectures import build_architecture
    from gomoku_ml.train import train

    torch.manual_seed(0)
    n = 32
    planes = np.random.default_rng(0).random((n, 8, 15, 15)).astype(np.float32)
    policy = np.zeros((n, 225), dtype=np.float32)
    policy[:, 0] = 1.0
    values = np.random.default_rng(1).uniform(-1, 1, n).astype(np.float32)
    model = build_architecture("resnet", {"channels": 4, "blocks": 1})
    summary = train(
        model,
        planes,
        policy,
        values,
        epochs=1,
        batch_size=8,
        history_dropout=0.5,
        val_fraction=0.25,
        checkpoint_dir=str(tmp_path),
        checkpoint_every_epochs=1,
        seed=0,
    )
    assert len(summary["history"]) == 1
    assert summary["history"][0]["val_loss"] is not None
    assert os.path.exists(os.path.join(str(tmp_path), "best.pt"))

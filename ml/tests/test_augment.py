"""Tests for dihedral board-symmetry augmentation (augment.py)."""

import numpy as np

from gomoku_ml import augment, encoding, game


def _sample_with_single_stone():
    """A position with one Black stone at (3, 5) and a one-hot policy on
    that same point — so planes/policy alignment can be checked after a
    transform."""
    board = game.empty_board()
    board[5, 3] = game.BLACK
    planes = encoding.encode(board, game.BLACK)
    policy = np.zeros(225, dtype=np.float32)
    policy[game.xy_to_move(3, 5)] = 1.0
    return planes, policy


def test_transform_then_inverse_restores():
    planes, policy = _sample_with_single_stone()
    for k in range(8):
        p2, pol2 = augment.apply_dihedral(planes, policy, k)
        inv = augment.inverse_transform_index(k)
        p3, pol3 = augment.apply_dihedral(p2, pol2, inv)
        assert np.array_equal(p3, planes), f"k={k}: planes not restored"
        assert np.array_equal(pol3, policy), f"k={k}: policy not restored"


def test_planes_and_policy_stay_aligned():
    """After any transform, the stone in plane 0 and the one-hot policy
    point must still be the same board point."""
    planes, policy = _sample_with_single_stone()
    for k in range(8):
        p2, pol2 = augment.apply_dihedral(planes, policy, k)
        stone_idx = int(np.argmax(p2[0]))
        assert pol2[stone_idx] == 1.0, f"k={k}: policy no longer on the stone"
        assert pol2.sum() == 1.0
        assert p2[0].sum() == 1.0


def test_transforms_actually_move_the_stone():
    planes, policy = _sample_with_single_stone()
    positions = set()
    for k in range(8):
        p2, _ = augment.apply_dihedral(planes, policy, k)
        positions.add(int(np.argmax(p2[0])))
    # A stone at (3,5) must land on several distinct points across the
    # 8 symmetries (not all transforms are the identity for it).
    assert len(positions) >= 4


def test_random_augment_batch_matches_per_sample():
    rng = np.random.default_rng(123)
    planes = rng.random((16, 8, 15, 15)).astype(np.float32)
    policy = rng.random((16, 225)).astype(np.float32)
    # Draw the same transform indices the batch function will draw.
    ks = np.random.default_rng(123).integers(0, 8, size=16)
    out_planes, out_policy = augment.random_augment_batch(
        planes, policy, np.random.default_rng(123)
    )
    assert out_planes.shape == planes.shape
    assert out_policy.shape == policy.shape
    for i, k in enumerate(ks):
        p_i, pol_i = augment.apply_dihedral(planes[i], policy[i], int(k))
        assert np.allclose(out_planes[i], p_i), f"sample {i} planes mismatch"
        assert np.allclose(out_policy[i], pol_i), f"sample {i} policy mismatch"
    # Inputs are not modified in place.
    assert planes.shape == (16, 8, 15, 15)


def test_random_augment_batch_preserves_sums():
    rng = np.random.default_rng(7)
    planes = rng.random((8, 8, 15, 15)).astype(np.float32)
    policy = rng.random((8, 225)).astype(np.float32)
    out_planes, out_policy = augment.random_augment_batch(planes, policy, rng)
    assert np.allclose(out_planes.sum(axis=(1, 2, 3)), planes.sum(axis=(1, 2, 3)))
    assert np.allclose(out_policy.sum(axis=1), policy.sum(axis=1))


# --- history-plane dropout (ML batch 4, v42) ----------------------------


def _planes_with_history(n=64, seed=0):
    """Random planes with all 8 planes non-zero, so a dropped history
    plane is distinguishable from an untouched one."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 8, 15, 15)).astype(np.float32)


def test_drop_history_planes_zero_is_identity():
    planes = _planes_with_history()
    out = augment.drop_history_planes_batch(planes, np.random.default_rng(0), 0.0)
    assert np.array_equal(out, planes)


def test_drop_history_planes_one_drops_all_history():
    planes = _planes_with_history()
    out = augment.drop_history_planes_batch(planes, np.random.default_rng(0), 1.0)
    for p in augment.HISTORY_PLANES:
        assert np.all(out[:, p] == 0.0), f"plane {p} not zeroed"
    for p in (0, 1, 3):
        assert np.array_equal(out[:, p], planes[:, p]), f"plane {p} changed"


def test_drop_history_planes_rate_and_selectivity():
    planes = _planes_with_history(n=2000, seed=1)
    out = augment.drop_history_planes_batch(planes, np.random.default_rng(2), 0.5)
    # A sample is dropped iff ALL its history planes are zero.
    dropped = np.all(out[:, list(augment.HISTORY_PLANES)] == 0.0, axis=(1, 2, 3))
    assert 0.45 < dropped.mean() < 0.55, f"drop rate {dropped.mean():.3f} != 0.5"
    # Dropped samples: history exactly zero, current planes untouched.
    assert np.array_equal(out[dropped][:, [0, 1, 3]], planes[dropped][:, [0, 1, 3]])
    # Undropped samples: identical to the input, history included.
    assert np.array_equal(out[~dropped], planes[~dropped])
    # Whole-history: no sample is partially dropped (each sample's
    # history planes are either all zero or all untouched).
    hist = out[:, list(augment.HISTORY_PLANES)]
    per_sample_zero_frac = (hist == 0.0).mean(axis=(1, 2, 3))
    assert np.all((per_sample_zero_frac == 0.0) | (per_sample_zero_frac == 1.0))


def test_drop_history_planes_input_not_modified():
    planes = _planes_with_history()
    before = planes.copy()
    augment.drop_history_planes_batch(planes, np.random.default_rng(0), 0.5)
    assert np.array_equal(planes, before)


def test_drop_history_planes_invalid_p_raises():
    import pytest

    planes = _planes_with_history(n=4)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        augment.drop_history_planes_batch(planes, np.random.default_rng(0), 1.5)
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        augment.drop_history_planes_batch(planes, np.random.default_rng(0), -0.1)

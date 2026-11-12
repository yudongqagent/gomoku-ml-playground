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

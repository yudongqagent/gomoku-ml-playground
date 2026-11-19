"""Board-symmetry data augmentation (dihedral group D4).

Gomoku is invariant under the 8 symmetries of the square: 4 rotations
x (with / without a mirror flip). Augmenting training samples with a
random symmetry is cheap and valuable: it multiplies the effective data
8x and teaches the net the board's symmetries instead of letting it
memorise orientations.

Transform index k in 0..7: ``k % 4`` is the number of 90-degree
counter-clockwise rotations, ``k >= 4`` adds a left-right flip after
the rotation. Planes [8, 15, 15] and the policy target [225] are
transformed with the *same* k, so the (position -> move) mapping stays
consistent. The value target is unchanged (the outcome does not depend
on orientation).
"""

from __future__ import annotations

import numpy as np

NUM_TRANSFORMS = 8

# History planes (see gomoku_ml/encoding.py): 2 = last-move one-hot,
# 4-5 = own/opponent stones 2 plies ago, 6-7 = own/opponent stones
# 4 plies ago. Plane 3 (side-to-move) is NOT history and is never
# dropped. The same set eval_value_holdout.py zeroes in its
# no-history view.
HISTORY_PLANES = (2, 4, 5, 6, 7)


def inverse_transform_index(k: int) -> int:
    """Index of the transform that undoes transform ``k``.

    Pure rotations invert to rotation (4 - k); flip+rotation transforms
    are involutions (applying one twice restores the original).
    """
    if k < 4:
        return (4 - k) % 4
    return k


def _apply_to_spatial(arr: np.ndarray, k: int, axes: tuple[int, int], flip_axis: int) -> np.ndarray:
    out = np.rot90(arr, k % 4, axes=axes)
    if k >= 4:
        out = np.flip(out, axis=flip_axis)
    return np.ascontiguousarray(out)


def apply_dihedral(
    planes: np.ndarray, policy: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply transform ``k`` to one sample.

    planes: [8, 15, 15]; policy: [225] (row-major). Returns transformed
    copies with the same shapes.
    """
    new_planes = _apply_to_spatial(planes, k, axes=(1, 2), flip_axis=2)
    new_policy = _apply_to_spatial(
        policy.reshape(15, 15), k, axes=(0, 1), flip_axis=1
    ).reshape(225)
    return new_planes, new_policy


def random_augment_batch(
    planes: np.ndarray, policy: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an independent random dihedral transform to each sample in
    a batch. planes: [B, 8, 15, 15]; policy: [B, 225]. Returns new arrays
    (inputs are not modified). Samples are grouped by transform so the
    work is 8 vectorised transforms, not B individual ones."""
    batch = planes.shape[0]
    ks = rng.integers(0, NUM_TRANSFORMS, size=batch)
    out_planes = planes.copy()
    out_policy = policy.copy()
    for k in range(NUM_TRANSFORMS):
        idx = np.flatnonzero(ks == k)
        if len(idx) == 0 or k == 0:
            continue  # k == 0 is the identity
        out_planes[idx] = _apply_to_spatial(planes[idx], k, axes=(2, 3), flip_axis=3)
        reshaped = policy[idx].reshape(-1, 15, 15)
        out_policy[idx] = _apply_to_spatial(
            reshaped, k, axes=(1, 2), flip_axis=2
        ).reshape(len(idx), 225)
    return out_planes, out_policy


def drop_history_planes_batch(
    planes: np.ndarray, rng: np.random.Generator, p: float
) -> np.ndarray:
    """Per-sample whole-history dropout (ML batch 4, v42 — see
    research/iterations/iter-42.md §1.1).

    For each sample independently, with probability ``p`` zero ALL
    history planes (``HISTORY_PLANES``: last-move one-hot + stones
    2/4 plies ago); otherwise leave the sample intact. The two views
    of a position carry the same label, so the net is trained to
    predict the same value from the board alone — it cannot reduce
    the loss by routing through history features that are absent
    half the time. ``p=0.0`` is the identity (v24's recipe exactly);
    ``p=1.0`` drops every sample's history. Returns a new array
    (the input is not modified)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"history dropout p must be in [0, 1], got {p}")
    if p == 0.0:
        return planes
    out = planes.copy()
    drop = rng.random(planes.shape[0]) < p
    # Single __setitem__ per plane: out[drop][...] = 0 would assign
    # to a boolean-index copy and silently do nothing.
    for plane in HISTORY_PLANES:
        out[drop, plane, :, :] = 0.0
    return out

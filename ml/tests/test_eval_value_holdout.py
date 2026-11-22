"""Tests for eval_value_holdout.py's side-to-move inference (ML batch 6 tooling).

ML batch 5 (iter-48 §0) found that ``load_shards`` inferred the side
to move as ``arange(len(planes)) % 2`` — correct only when the shard
stores EVERY ply of the game. Distill shards store only QUIET
positions (``generate_distill_data.py`` skips plies where
``_terminal_leaf_value != 0.0``), so after the first skipped ply the
index-inferred side FLIPS, and every further skip flips it again —
attenuating any true by-side gap toward zero (up to ~60%;
iter-37/iter-44 by-side numbers are unreliable for this reason).

The fix reads the side from plane 3 of the stored encoding — the
side-to-move plane (all 1.0 iff Black is to move, else all 0.0; see
``gomoku_ml/encoding.py``) — which is exact for every stored position
regardless of which plies the shard kept.
"""

import os
import sys

import numpy as np
import pytest

from gomoku_ml import encoding, game

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "ml", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import eval_value_holdout  # noqa: E402

# A fixed 12-move game around the centre (flat indices, row*15+col).
_MOVES = [112, 113, 97, 98, 127, 128, 82, 83, 142, 143, 67, 68]


def _encode_game(moves):
    """Encode every ply exactly as generate_distill_data.py does.

    Returns (planes, true_side) with true_side[t] = t % 2 (0 Black,
    1 White).
    """
    boards = [game.empty_board()]
    player = game.BLACK
    for m in moves:
        boards.append(game.apply_move(boards[-1], m, player))
        player = -player
    planes, sides = [], []
    for t in range(len(moves)):
        p = game.BLACK if t % 2 == 0 else game.WHITE
        planes.append(
            encoding.encode(
                boards[t], p, boards[:t], moves[t - 1] if t > 0 else None
            )
        )
        sides.append(t % 2)
    return np.stack(planes).astype(np.float32), np.array(sides, dtype=np.int8)


def _write_shard(path, planes):
    np.savez_compressed(
        path,
        planes=planes,
        value=np.zeros(len(planes), dtype=np.float32),
    )


def test_load_shards_reads_side_from_plane3_on_quiet_subset(tmp_path):
    """On a distill-style shard (plies dropped), side is the TRUE ply parity."""
    planes, true_side = _encode_game(_MOVES)
    # Drop plies 3 and 7, simulating the quiet-position filter.
    keep = [t for t in range(len(_MOVES)) if t not in (3, 7)]
    _write_shard(tmp_path / "g_game0000.npz", planes[keep])
    _, _, _, side = eval_value_holdout.load_shards(str(tmp_path))
    assert list(side) == list(true_side[keep])
    # The old index inference would get this shard WRONG after the
    # first dropped ply — proving the test exercises the flaw.
    index_side = np.arange(len(keep), dtype=np.int8) % 2
    assert list(index_side) != list(true_side[keep])


def test_load_shards_side_matches_index_on_full_game(tmp_path):
    """On a full-game shard (no plies dropped) the fix changes nothing."""
    planes, true_side = _encode_game(_MOVES)
    _write_shard(tmp_path / "g_game0000.npz", planes)
    _, _, _, side = eval_value_holdout.load_shards(str(tmp_path))
    assert list(side) == list(true_side)
    assert list(side) == list(np.arange(len(_MOVES), dtype=np.int8) % 2)


def test_load_shards_rejects_non_binary_plane3(tmp_path):
    """A shard whose plane 3 is not 0/1 fails loudly, not silently."""
    planes, _ = _encode_game(_MOVES)
    planes[:, 3, :, :] = 0.5
    _write_shard(tmp_path / "g_game0000.npz", planes)
    with pytest.raises(ValueError, match="plane 3 is not 0/1"):
        eval_value_holdout.load_shards(str(tmp_path))


def test_load_shards_rejects_non_uniform_plane3(tmp_path):
    """A shard whose plane 3 is not uniform per position fails loudly."""
    planes, _ = _encode_game(_MOVES)
    planes[0, 3, 0, 0] = 1.0 - planes[0, 3, 0, 0]
    _write_shard(tmp_path / "g_game0000.npz", planes)
    with pytest.raises(ValueError, match="plane 3 is not uniform"):
        eval_value_holdout.load_shards(str(tmp_path))

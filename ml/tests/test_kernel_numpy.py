"""Cross-check: the NumPy scoring kernel must produce IDENTICAL scores to
``gomoku_ml.baseline.score_move`` on every fixture position and cell.

Two layers of protection:

1. Frozen reference — kernel scores vs the stored ``scores`` in
   ``kernel_fixtures.json`` (computed once from ``baseline.score_move``).
   This fails if the kernel drifts OR if ``baseline.py`` changed without
   regenerating the fixtures.
2. Live reference — kernel scores vs ``baseline.score_move`` computed
   fresh in this test run, for BOTH players, on every fixture. This
   fails if the kernel drifts from the current baseline.

Fixtures: 60 positions (20 tactical, 3 structured, 37 random midgame),
225 cells each = 13,500 cells per player per layer.
"""

import json
import os

import numpy as np
import pytest

from gomoku_ml import baseline, game
from gomoku_ml.kernel_numpy import score_all, score_batch

HERE = os.path.dirname(os.path.abspath(__file__))


def load_fixtures():
    with open(os.path.join(HERE, "kernel_fixtures.json"), encoding="utf-8") as f:
        return json.load(f)["fixtures"]


def _board(fx):
    return np.array(fx["board"], dtype=np.int8).reshape(game.SIZE, game.SIZE)


def test_fixture_count():
    fixtures = load_fixtures()
    assert len(fixtures) >= 50, f"expected >= 50 fixtures, got {len(fixtures)}"
    sources = {fx["source"] for fx in fixtures}
    assert "tactical" in sources and "random" in sources


def test_kernel_matches_frozen_reference():
    """Kernel == stored Python reference scores, side to move, all cells."""
    total_mismatches = 0
    for fx in load_fixtures():
        board = _board(fx)
        expected = np.array(fx["scores"], dtype=np.int64).reshape(game.SIZE, game.SIZE)
        got = score_all(board, fx["to_move"])
        mismatches = int((got != expected).sum())
        if mismatches:
            ys, xs = np.nonzero(got != expected)
            first = [(int(y), int(x), int(expected[y, x]), int(got[y, x]))
                     for y, x in list(zip(ys, xs))[:3]]
            pytest.fail(f"{fx['id']}: {mismatches} mismatched cells, first {first}")
        total_mismatches += mismatches
    assert total_mismatches == 0


def test_kernel_matches_live_reference_both_players():
    """Kernel == baseline.score_move computed fresh, both players, all cells."""
    for fx in load_fixtures():
        board = _board(fx)
        for player in (game.BLACK, game.WHITE):
            expected = np.array(
                [baseline.score_move(board, m, player) for m in range(game.SIZE * game.SIZE)],
                dtype=np.int64,
            ).reshape(game.SIZE, game.SIZE)
            got = score_all(board, player)
            assert (got == expected).all(), (
                f"{fx['id']} player {player}: "
                f"{int((got != expected).sum())} mismatched cells"
            )


def test_batch_matches_single():
    fixtures = load_fixtures()
    boards = np.stack([_board(fx) for fx in fixtures])
    for player in (game.BLACK, game.WHITE):
        batched = score_batch(boards, player)
        for i, fx in enumerate(fixtures):
            assert (batched[i] == score_all(boards[i], player)).all(), (
                f"batch row {i} ({fx['id']}) differs from score_all"
            )


def test_batch_matches_frozen_reference():
    fixtures = load_fixtures()
    # Batch only supports one player per call; split by side to move.
    for player in (game.BLACK, game.WHITE):
        subset = [fx for fx in fixtures if fx["to_move"] == player]
        boards = np.stack([_board(fx) for fx in subset])
        batched = score_batch(boards, player)
        for i, fx in enumerate(subset):
            expected = np.array(fx["scores"], dtype=np.int64).reshape(game.SIZE, game.SIZE)
            assert (batched[i] == expected).all(), f"batch {fx['id']} mismatch"


def test_invalid_inputs_raise():
    board = game.empty_board()
    with pytest.raises(ValueError):
        score_all(board, 0)
    with pytest.raises(ValueError):
        score_all(np.zeros((14, 15), dtype=np.int8), game.BLACK)
    with pytest.raises(ValueError):
        score_batch(np.zeros((3, 15, 14), dtype=np.int8), game.BLACK)

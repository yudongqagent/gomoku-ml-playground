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
from gomoku_ml.kernel_numpy import attack_all, attack_both, score_all, score_batch

HERE = os.path.dirname(os.path.abspath(__file__))


def _ref_attack(board, player):
    """Pure-Python attack-only reference, built from baseline
    internals: for each cell, the sum over the 4 directions of the
    mover's weight only — no opponent part, no centre term."""
    out = np.zeros((game.SIZE, game.SIZE), dtype=np.int64)
    for y in range(game.SIZE):
        for x in range(game.SIZE):
            total = 0
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                run, ends = baseline._run_and_open_ends(
                    board, y, x, dy, dx, player
                )
                total += baseline._WEIGHTS.get((min(run, 5), ends), 0)
            out[y, x] = total
    return out


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
    with pytest.raises(ValueError):
        attack_all(board, 0)
    with pytest.raises(ValueError):
        attack_all(np.zeros((14, 15), dtype=np.int8), game.BLACK)
    with pytest.raises(ValueError):
        attack_both(np.zeros((14, 15), dtype=np.int8))


def test_attack_all_matches_python_reference_both_players():
    """attack_all == the pure-Python attack-only reference built from
    baseline internals, both players, all 225 cells, every fixture
    (0 mismatches required — this is the v7 leaf's input)."""
    for fx in load_fixtures():
        board = _board(fx)
        for player in (game.BLACK, game.WHITE):
            expected = _ref_attack(board, player)
            got = attack_all(board, player)
            assert (got == expected).all(), (
                f"{fx['id']} player {player}: "
                f"{int((got != expected).sum())} mismatched cells"
            )


def test_attack_both_matches_attack_all():
    """attack_both returns exactly attack_all for each colour."""
    for fx in load_fixtures():
        board = _board(fx)
        black, white = attack_both(board)
        assert (black == attack_all(board, game.BLACK)).all()
        assert (white == attack_all(board, game.WHITE)).all()


def test_attack_all_is_friendly_part_only():
    """attack_all excludes the opponent part and the centre term.
    On an empty board every interior cell's attack is exactly 40
    (four directions of (1, 2) = 10 each); edge cells have fewer
    open ends (13 on a non-corner edge, 3 in a corner). On real
    boards: attack_all is bounded above by score_all minus the
    centre term (the opponent part is >= 0), and the two colours'
    attacks sum to at least that (the opponent part in score_all
    is the 0.9-discounted attack, so the undiscounted sum is >= it).
    """
    board = game.empty_board()
    got = attack_all(board, game.BLACK)
    assert (got[1:-1, 1:-1] == 40).all(), "empty board interior: 4 x (1,2)=10"
    assert (got[0, 1:-1] == 13).all() and (got[-1, 1:-1] == 13).all()
    assert (got[1:-1, 0] == 13).all() and (got[1:-1, -1] == 13).all()
    assert got[0, 0] == 3 and got[0, -1] == 3 and got[-1, 0] == 3 and got[-1, -1] == 3
    ys, xs = np.mgrid[0:15, 0:15]
    centre = 7 - (np.abs(xs - 7) + np.abs(ys - 7))
    for fx in load_fixtures():
        board = _board(fx)
        for player in (game.BLACK, game.WHITE):
            ours = attack_all(board, player)
            theirs = attack_all(board, -player)
            total = score_all(board, player)
            assert (ours <= total - centre).all()
            assert (ours + theirs >= total - centre).all()

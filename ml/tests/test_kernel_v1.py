"""Cross-check: the NumPy v1 kernel must produce IDENTICAL attack scores
to ``gomoku_ml.scoring.score_attack`` on every fixture position and cell.

Two layers of protection (mirroring ``test_kernel_numpy.py``):

1. Frozen reference — v1 kernel scores vs the stored ``attack_black`` /
   ``attack_white`` in ``kernel_v1_fixtures.json`` (computed once from
   ``scoring.score_attack``). This fails if the kernel drifts OR if
   ``scoring.py`` changed without regenerating the fixtures.
2. Live reference — v1 kernel scores vs ``scoring.score_attack``
   computed fresh in this test run, for BOTH players, on every fixture.
   This fails if the kernel drifts from the current scorer.

Fixtures: the same 60 positions as the v0 kernel fixtures (20
tactical, 3 structured, 37 random midgame), 225 cells each = 13,500
cells per player per layer. A third layer cross-checks the live
reference on hand-built broken-four boards, the pattern class this
kernel exists to see.
"""

import json
import os

import numpy as np
import pytest

from gomoku_ml import game, scoring
from gomoku_ml.kernel_v1 import attack_v1_all, attack_v1_both

HERE = os.path.dirname(os.path.abspath(__file__))


def load_fixtures():
    with open(os.path.join(HERE, "kernel_v1_fixtures.json"), encoding="utf-8") as f:
        return json.load(f)["fixtures"]


def _board(fx):
    return np.array(fx["board"], dtype=np.int8).reshape(game.SIZE, game.SIZE)


def _board_from_stones(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def test_fixture_count_and_positions_match_v0():
    """60 fixtures, and the same boards as the v0 fixture set."""
    fixtures = load_fixtures()
    assert len(fixtures) == 60, f"expected 60 fixtures, got {len(fixtures)}"
    with open(os.path.join(HERE, "kernel_fixtures.json"), encoding="utf-8") as f:
        v0 = json.load(f)["fixtures"]
    assert [fx["id"] for fx in fixtures] == [fx["id"] for fx in v0]
    for fx, fx0 in zip(fixtures, v0):
        assert fx["board"] == fx0["board"], f"{fx['id']}: board drifted from v0 fixtures"


def test_kernel_matches_frozen_reference_both_players():
    """Kernel == stored Python reference attack scores, both players,
    all cells."""
    for fx in load_fixtures():
        board = _board(fx)
        for player, key in ((game.BLACK, "attack_black"),
                            (game.WHITE, "attack_white")):
            expected = np.array(fx[key], dtype=np.int64).reshape(game.SIZE, game.SIZE)
            got = attack_v1_all(board, player)
            mismatches = int((got != expected).sum())
            if mismatches:
                ys, xs = np.nonzero(got != expected)
                first = [(int(y), int(x), int(expected[y, x]), int(got[y, x]))
                         for y, x in list(zip(ys, xs))[:3]]
                pytest.fail(f"{fx['id']} player {player}: {mismatches} "
                            f"mismatched cells, first {first}")


def test_kernel_matches_live_reference_both_players():
    """Kernel == scoring.score_attack computed fresh, both players,
    all cells, every fixture (0 mismatches required — this is the
    v14 leaf's input)."""
    for fx in load_fixtures():
        board = _board(fx)
        for player in (game.BLACK, game.WHITE):
            expected = np.array(
                [scoring.score_attack(board, m, player)
                 for m in range(game.SIZE * game.SIZE)],
                dtype=np.int64,
            ).reshape(game.SIZE, game.SIZE)
            got = attack_v1_all(board, player)
            assert (got == expected).all(), (
                f"{fx['id']} player {player}: "
                f"{int((got != expected).sum())} mismatched cells"
            )


def test_kernel_matches_live_reference_broken_fours():
    """Live cross-check on hand-built boards rich in broken fours,
    gap threes, and edge patterns — the classes the v0 kernel is
    blind to and this kernel exists to see."""
    boards = {
        # Broken four XX_XX (gap at (7,4)) plus a broken three.
        "broken-four": ([(2, 7), (3, 7), (5, 7), (6, 7), (10, 2), (12, 2)],
                        [(3, 3), (11, 2)]),
        # Two broken fours crossing at the gap cell (double threat).
        "double-broken-four": ([(2, 7), (3, 7), (5, 7), (6, 7),
                                (4, 5), (4, 6), (4, 8), (4, 9)],
                               [(0, 0)]),
        # Gap three X_XX and X_X_X, on the edge and in the centre.
        "gap-threes": ([(0, 0), (2, 0), (3, 0), (7, 7), (9, 7), (11, 7)],
                       [(5, 5), (6, 6)]),
        # Broken four against the edge (only one outside cell).
        "edge-broken-four": ([(0, 14), (1, 14), (3, 14), (4, 14)],
                             [(7, 7)]),
        # Diagonal broken four.
        "diag-broken-four": ([(2, 2), (3, 3), (5, 5), (6, 6)],
                             [(10, 10)]),
        # Four with a blocked far end (closed four, not open).
        "closed-four": ([(2, 7), (3, 7), (4, 7), (5, 7)],
                        [(1, 7), (6, 7)]),
    }
    for name, (black, white) in boards.items():
        board = _board_from_stones(black, white)
        for player in (game.BLACK, game.WHITE):
            expected = np.array(
                [scoring.score_attack(board, m, player)
                 for m in range(game.SIZE * game.SIZE)],
                dtype=np.int64,
            ).reshape(game.SIZE, game.SIZE)
            got = attack_v1_all(board, player)
            assert (got == expected).all(), (
                f"{name} player {player}: "
                f"{int((got != expected).sum())} mismatched cells"
            )


def test_attack_v1_both_matches_attack_v1_all():
    """attack_v1_both returns exactly attack_v1_all for each colour."""
    for fx in load_fixtures():
        board = _board(fx)
        black, white = attack_v1_both(board)
        assert (black == attack_v1_all(board, game.BLACK)).all()
        assert (white == attack_v1_all(board, game.WHITE)).all()


def test_invalid_inputs_raise():
    board = game.empty_board()
    with pytest.raises(ValueError):
        attack_v1_all(board, 0)
    with pytest.raises(ValueError):
        attack_v1_all(np.zeros((14, 15), dtype=np.int8), game.BLACK)
    with pytest.raises(ValueError):
        attack_v1_both(np.zeros((14, 15), dtype=np.int8))


def test_broken_four_scores_as_four_not_two_runs():
    """The point of this kernel: a move CREATING a broken four
    (XX_XX) scores as a four (20k) in the v1 kernel, where the v0
    kernel scores it as two separate runs (~100-200). (Completing a
    broken four makes five, which v0 sees fine — the blindness is
    in creation, not completion.)"""
    from gomoku_ml.kernel_numpy import attack_all

    board = _board_from_stones(
        [(2, 7), (3, 7), (5, 7)], [(3, 3)],
    )
    # (7, 6) creates XX_XX on row 7 (gap at (7, 4)).
    create = (7, 6)
    v1 = attack_v1_all(board, game.BLACK)
    v0 = attack_all(board, game.BLACK)
    assert v1[create] >= 20_000, f"v1 kernel scores the creation at {v1[create]}, not a four"
    assert v0[create] < 1_000, f"v0 kernel unexpectedly scores the creation at {v0[create]}"
    # Same for a gap-three creation (X_X -> X_XX is an open three
    # in v1, a (2,2) run in v0).
    board2 = _board_from_stones([(2, 7), (4, 7)], [(3, 3)])
    v1b = attack_v1_all(board2, game.BLACK)
    v0b = attack_all(board2, game.BLACK)
    assert v1b[7, 5] >= 4_000, f"v1 kernel scores the gap-three at {v1b[7, 5]}"
    assert v0b[7, 5] < 1_000, f"v0 kernel unexpectedly scores the gap-three at {v0b[7, 5]}"


def test_empty_board_centre_attack_value():
    """On an empty board the centre cell's v1 attack is 8 (four
    directions of a lone stone with both ends open = 2 each).
    Edge-adjacent cells vary (windows run off-board); the
    frozen/live cross-checks already cover every cell of the
    empty-board fixture, so only the clean hand-derivable value
    is asserted here."""
    board = game.empty_board()
    got = attack_v1_all(board, game.BLACK)
    assert got[7, 7] == 8, "empty board centre: 4 x 2"
    assert got[7, 7] == attack_v1_all(board, game.WHITE)[7, 7]

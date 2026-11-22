"""Tests for the v56 parity-corrected 3-ply label (ML batch 6 H3).

``_rollout3sym_leaf_value`` is the colour-swap-symmetrised 3-ply
value ``(L3(B, p) - L3(-B, p)) / 2`` (see
``research/iterations/iter-56.md`` §1.1): it cancels the
odd-horizon parity shift per position while keeping v44's 5-ply
horizon. These tests pin the definition: the mirror identity the
derivation depends on, terminal preservation for first-player-
robust wins, the empty-board identity (a colour-symmetric
position symmetrises to exactly 0), non-mutation, determinism,
the setup-v1v0s-g92 regression (the forced win must still read
as a win after symmetrisation), and the labelling script's
``--label-mode rollout3sym`` wiring.
"""

import json
import os

import numpy as np

from gomoku_ml import game, mcts_heuristic as mh


def _board_with(black, white):
    board = game.empty_board()
    for r, c in black:
        board[r, c] = game.BLACK
    for r, c in white:
        board[r, c] = game.WHITE
    return board


def _load_hard_position(pos_id):
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "gomoku_ml",
        "tactical_positions_hard.json",
    )
    with open(path, encoding="utf-8") as f:
        suite = json.load(f)
    pos = next(p for p in suite["positions"] if p["id"] == pos_id)
    # Stone lists are (x, y) pairs (see evaluate.board_from_position).
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board, pos["to_move"]


def _manual_rollout3sym(board, player):
    """Step-by-step recomputation of the §1.1 definition
    (independent of ``_rollout3sym_leaf_value``'s body)."""
    return (
        mh._rollout3_leaf_value(board, player)
        - mh._rollout3_leaf_value(-board, player)
    ) / 2.0


def test_rollout3sym_empty_board_is_exactly_zero():
    # The empty board is colour-symmetric (-B == B), so the two
    # 3-ply terms are identical and the symmetric label is exactly
    # 0 — the parity shift cancels exactly on a symmetric position.
    # (The raw 3-ply label here is +0.1365, iter-44 §0.)
    board = game.empty_board()
    assert mh._rollout3_leaf_value(board, game.BLACK) > 0.0
    assert mh._rollout3sym_leaf_value(board, game.BLACK) == 0.0
    assert mh._rollout3sym_leaf_value(board, game.WHITE) == 0.0


def test_rollout3sym_open_four_returns_plus_one():
    # Black holds an open four (>= 2 winning moves): a forced win
    # robust to who moves first (White cannot block both ends), so
    # the symmetric label preserves the +1 exactly (iter-56 §1.1).
    board = _board_with(
        [(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert mh._rollout3sym_leaf_value(board, game.BLACK) == 1.0


def test_rollout3sym_opponent_open_four_returns_minus_one():
    # White holds an open four; Black has none: a forced loss
    # robust to who moves first, preserved exactly at -1.
    board = _board_with(
        [(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert mh._rollout3sym_leaf_value(board, game.BLACK) == -1.0


def test_rollout3sym_must_block_plays_through():
    # White holds a broken four (exactly one winning move): the
    # rollout policy plays the forced block on ply 1 and the label
    # is the symmetrised 3-ply-ahead value, NOT the -0.25
    # start-position value (the interior-must-block convention,
    # iter-16 §2.1).
    board = _board_with(
        [(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert mh._terminal_leaf_value(board, game.BLACK) == -0.25
    value = mh._rollout3sym_leaf_value(board, game.BLACK)
    assert -1.0 <= value <= 1.0
    assert value != -0.25


def test_rollout3sym_does_not_mutate_board():
    board = _board_with(
        [(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)])
    before = board.copy()
    mh._rollout3sym_leaf_value(board, game.BLACK)
    assert np.array_equal(board, before)


def test_rollout3sym_deterministic():
    board = _board_with(
        [(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)])
    v1 = mh._rollout3sym_leaf_value(board, game.BLACK)
    v2 = mh._rollout3sym_leaf_value(board, game.BLACK)
    assert v1 == v2


def test_rollout3sym_mirror_identity():
    # The identity the §1.1 derivation depends on: the colour-
    # swapped board with the SAME side to move is the mirror image
    # of the original board with the OPPONENT to move, so the two
    # 3-ply labels must agree exactly. If this fails, the
    # symmetrisation does not cancel the shift as derived
    # (iter-56 §1.5 stop rule).
    boards = [
        _board_with([(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)]),
        _board_with([(7, 7), (8, 8), (6, 9)], [(7, 8), (9, 7), (5, 5)]),
        _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)]),
        _load_hard_position("setup-v1v0s-g92")[0],
        game.empty_board(),
    ]
    for board in boards:
        for player in (game.BLACK, game.WHITE):
            assert (
                mh._rollout3_leaf_value(-board, player)
                == mh._rollout3_leaf_value(board, -player)
            )


def test_rollout3sym_matches_manual_definition():
    # The §1.1 definition recomputed step by step must equal the
    # function on quiet positions, terminal positions, and the g92
    # root — for both sides to move.
    boards = [
        _board_with([(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)]),
        _board_with([(7, 7), (8, 8), (6, 9)], [(7, 8), (9, 7), (5, 5)]),
        _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)]),
        _load_hard_position("setup-v1v0s-g92")[0],
        game.empty_board(),
    ]
    for board in boards:
        for player in (game.BLACK, game.WHITE):
            assert (
                mh._rollout3sym_leaf_value(board, player)
                == _manual_rollout3sym(board, player)
            )


def test_rollout3sym_g92_root_still_reads_as_a_win():
    # Regression from iter-56 §1.1: at setup-v1v0s-g92's root
    # (Black to move) the raw 3-ply label is +1.0 (the forced win
    # is inside the 5-ply horizon). After symmetrisation the win
    # may be halved (if it depends on moving first) but must still
    # read as a clear win — above the gradient cap (0.2), so the
    # net can still learn "winning > threatening > neutral".
    board, p = _load_hard_position("setup-v1v0s-g92")
    assert mh._terminal_leaf_value(board, p) == 0.0
    assert mh._rollout3_leaf_value(board, p) == 1.0
    value = mh._rollout3sym_leaf_value(board, p)
    assert 0.2 < value <= 1.0


def test_rollout3sym_label_mode_matches_function(tmp_path):
    # The labelling script's --label-mode rollout3sym path calls
    # _rollout3sym_leaf_value: label one tiny games file and check
    # the shard values equal direct function calls.
    import subprocess
    import sys

    games = {
        "games": [
            {
                "game": 0, "source": "v7", "sims": 50, "seed": 1,
                "moves": [112, 113, 97, 127, 82, 142, 67, 157],
                "winner": 0, "plies": 8,
            }
        ],
        "complete": True,
    }
    games_path = tmp_path / "games.json"
    games_path.write_text(json.dumps(games), encoding="utf-8")
    script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "generate_distill_data.py",
    )
    out_dir = tmp_path / "data"
    res = subprocess.run(
        [sys.executable, script, "label", "--games", str(games_path),
         "--out-dir", str(out_dir), "--prefix", "t",
         "--label-mode", "rollout3sym"],
        capture_output=True, text=True, check=True)
    assert "LABEL_JSON" in res.stdout
    shard = np.load(out_dir / "t_game0000.npz")
    # Reconstruct the quiet positions and compare values directly.
    moves = games["games"][0]["moves"]
    boards = [game.empty_board()]
    player = game.BLACK
    for m in moves:
        boards.append(game.apply_move(boards[-1], m, player))
        player = -player
    expected = []
    for t in range(len(moves)):
        b = boards[t]
        p = game.BLACK if t % 2 == 0 else game.WHITE
        if mh._terminal_leaf_value(b, p) != 0.0:
            continue
        expected.append(mh._rollout3sym_leaf_value(b, p))
    assert len(expected) == len(shard["value"])
    np.testing.assert_allclose(
        shard["value"], np.array(expected, dtype=np.float32), rtol=0,
        atol=1e-6)

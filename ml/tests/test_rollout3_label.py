"""Tests for the v44 3-ply label (ML batch 4 H3).

``_rollout3_leaf_value`` is the rollout-leaf value evaluated 3 plies
ahead under the teacher's own rollout policy (see
``research/iterations/iter-44.md`` §1). These tests pin the
definition: terminal handling, sign conventions (including the
odd-ply negation — the 3-ply form ends with the OPPONENT to move,
unlike the 2-ply form), non-mutation, determinism, and the
setup-v1v0s-g92 regression vectors measured in iter-44 §0.
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


def _manual_rollout3(board, player):
    """Step-by-step recomputation of the §1.1 recursion (independent
    of ``_rollout3_leaf_value``'s body)."""
    m1, t1 = mh._rollout_move(board, player)
    if t1 is not None:
        return t1
    b1 = game.apply_move(board, m1, player)
    m2, t2 = mh._rollout_move(b1, -player)
    if t2 is not None:
        return -t2
    b2 = game.apply_move(b1, m2, -player)
    m3, t3 = mh._rollout_move(b2, player)
    if t3 is not None:
        return t3
    b3 = game.apply_move(b2, m3, player)
    return -mh._rollout_leaf_value(b3, -player)


def test_rollout3_winning_move_returns_plus_one():
    # Black to move has a winning move (four in a row): ply 1 is
    # terminal (+1 from Black's perspective).
    board = _board_with(
        [(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert mh._rollout3_leaf_value(board, game.BLACK) == 1.0


def test_rollout3_opponent_open_four_returns_minus_one():
    # White holds an open four (>= 2 winning moves); Black has
    # none: ply 1 is terminal (-1 from Black's perspective).
    board = _board_with(
        [(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert mh._rollout3_leaf_value(board, game.BLACK) == -1.0


def test_rollout3_must_block_plays_through():
    # White holds a broken four (exactly one winning move): the
    # rollout policy plays the forced block on ply 1 and the label
    # is the 3-ply-ahead value, NOT the -0.25 start-position value
    # (the interior-must-block convention, iter-16 §2.1).
    board = _board_with(
        [(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert mh._terminal_leaf_value(board, game.BLACK) == -0.25
    value = mh._rollout3_leaf_value(board, game.BLACK)
    assert -1.0 <= value <= 1.0
    assert value != -0.25


def test_rollout3_does_not_mutate_board():
    board = _board_with(
        [(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)])
    before = board.copy()
    mh._rollout3_leaf_value(board, game.BLACK)
    assert np.array_equal(board, before)


def test_rollout3_deterministic():
    board = _board_with(
        [(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)])
    v1 = mh._rollout3_leaf_value(board, game.BLACK)
    v2 = mh._rollout3_leaf_value(board, game.BLACK)
    assert v1 == v2


def test_rollout3_matches_manual_recursion():
    # The §1.1 recursion recomputed step by step (including the
    # odd-ply negation at the leaf) must equal the function on
    # quiet positions, terminal positions, and the g92 root.
    boards = [
        _board_with([(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)]),
        _board_with([(7, 7), (8, 8), (6, 9)], [(7, 8), (9, 7), (5, 5)]),
        _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)]),
        _load_hard_position("setup-v1v0s-g92")[0],
    ]
    for board in boards:
        for player in (game.BLACK, game.WHITE):
            assert (
                mh._rollout3_leaf_value(board, player)
                == _manual_rollout3(board, player)
            )


def test_rollout3_g92_root_sees_the_forced_win():
    # Regression vectors from iter-44 §0: at setup-v1v0s-g92's
    # root (Black to move), the 1-ply label only hints at the win
    # (+0.1290), the 2-ply label reaches it (+1.0), and the 3-ply
    # label also reaches it (+1.0) — the forced win is inside the
    # 5-ply horizon too.
    board, p = _load_hard_position("setup-v1v0s-g92")
    assert mh._terminal_leaf_value(board, p) == 0.0
    assert mh._rollout_move(board, p)[0] == game.xy_to_move(9, 6)
    assert abs(mh._rollout_leaf_value(board, p) - 0.1290) < 1e-3
    assert mh._rollout2_leaf_value(board, p) == 1.0
    assert mh._rollout3_leaf_value(board, p) == 1.0


def test_rollout3_g92_children_both_win():
    # iter-44 §0's honest consequence: the 3-ply labels do NOT
    # improve child-level discrimination at g92 over the 2-ply
    # labels — the slow win after the wrong (5,4) is still inside
    # the 5-ply horizon, so both children are +1.0 (from Black's
    # perspective) at both depths.
    board, p = _load_hard_position("setup-v1v0s-g92")
    for xy in ((9, 6), (5, 4)):
        child = game.apply_move(board, game.xy_to_move(*xy), p)
        assert -mh._rollout2_leaf_value(child, -p) == 1.0
        assert -mh._rollout3_leaf_value(child, -p) == 1.0


def test_rollout3_label_mode_matches_function(tmp_path):
    # The labelling script's --label-mode rollout3 path calls
    # _rollout3_leaf_value: label one tiny games file and check
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
         "--label-mode", "rollout3"],
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
        expected.append(mh._rollout3_leaf_value(b, p))
    assert len(expected) == len(shard["value"])
    np.testing.assert_allclose(
        shard["value"], np.array(expected, dtype=np.float32), rtol=0,
        atol=1e-6)

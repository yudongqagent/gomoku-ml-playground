"""Tests for the v37 2-ply label (ML batch 3 H2).

``_rollout2_leaf_value`` is the rollout-leaf value evaluated 2 plies
ahead under the teacher's own rollout policy (see
``research/iterations/iter-37.md`` §1). These tests pin the
definition: terminal handling, sign conventions, non-mutation,
determinism, and the setup-v1v0s-g92 regression vectors measured in
iter-37 §0 (the motivating position for this workstream).
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
    board = game.empty_board()
    for r, c in pos["black"]:
        board[r, c] = game.BLACK
    for r, c in pos["white"]:
        board[r, c] = game.WHITE
    return board, pos["to_move"]


def test_rollout2_winning_move_returns_plus_one():
    # Black to move has a winning move (four in a row): ply 1 is
    # terminal (+1 from Black's perspective).
    board = _board_with(
        [(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert mh._rollout2_leaf_value(board, game.BLACK) == 1.0


def test_rollout2_opponent_open_four_returns_minus_one():
    # White holds an open four (>= 2 winning moves); Black has
    # none: ply 1 is terminal (-1 from Black's perspective).
    board = _board_with(
        [(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert mh._rollout2_leaf_value(board, game.BLACK) == -1.0


def test_rollout2_must_block_plays_through():
    # White holds a broken four (exactly one winning move): the
    # rollout policy plays the forced block on ply 1 and the label
    # is the 2-ply-ahead value, NOT the -0.25 start-position value
    # (the interior-must-block convention, iter-16 §2.1).
    board = _board_with(
        [(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert mh._terminal_leaf_value(board, game.BLACK) == -0.25
    value = mh._rollout2_leaf_value(board, game.BLACK)
    assert -1.0 <= value <= 1.0
    assert value != -0.25


def test_rollout2_does_not_mutate_board():
    board = _board_with(
        [(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)])
    before = board.copy()
    mh._rollout2_leaf_value(board, game.BLACK)
    assert np.array_equal(board, before)


def test_rollout2_deterministic():
    board = _board_with(
        [(7, 7), (7, 8), (8, 7)], [(7, 9), (9, 9), (6, 6)])
    v1 = mh._rollout2_leaf_value(board, game.BLACK)
    v2 = mh._rollout2_leaf_value(board, game.BLACK)
    assert v1 == v2


def test_rollout2_bounded_and_quiet_agrees_with_rollout_on_forced_win():
    # A quiet position where the rollout already finds the forced
    # win: the 2-ply label agrees (both +1). The setup position
    # below is setup-v1v0s-g92's root after the setup move (9,6):
    # Black has already set up the open four, White to move loses.
    board, p = _load_hard_position("setup-v1v0s-g92")
    child = game.apply_move(board, 9 * 15 + 6, p)
    assert mh._terminal_leaf_value(child, -p) == 0.0
    assert mh._rollout_leaf_value(child, -p) == -1.0
    assert mh._rollout2_leaf_value(child, -p) == -1.0


def test_rollout2_g92_root_sees_the_forced_win():
    # THE motivating regression vector (iter-37 §0): at
    # setup-v1v0s-g92's root (Black to move), the 1-ply label only
    # hints at the win (+0.1290 gradient — the payoff is 4 plies
    # out, beyond the 2-ply rollout), while the 2-ply label's
    # 4-ply horizon reaches it (+1.0). The teacher plays the setup
    # (9,6) on ply 1 (verified in §0).
    board, p = _load_hard_position("setup-v1v0s-g92")
    assert mh._terminal_leaf_value(board, p) == 0.0
    assert mh._rollout_move(board, p)[0] == 9 * 15 + 6
    assert abs(mh._rollout_leaf_value(board, p) - 0.1290) < 1e-3
    assert mh._rollout2_leaf_value(board, p) == 1.0


def test_rollout2_label_mode_matches_function(tmp_path):
    # The labelling script's --label-mode rollout2 path calls
    # _rollout2_leaf_value: label one tiny games file both ways and
    # check the rollout2 shard values equal direct function calls.
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
         "--label-mode", "rollout2"],
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
        expected.append(mh._rollout2_leaf_value(b, p))
    assert len(expected) == len(shard["value"])
    np.testing.assert_allclose(
        shard["value"], np.array(expected, dtype=np.float32), rtol=0,
        atol=1e-6)

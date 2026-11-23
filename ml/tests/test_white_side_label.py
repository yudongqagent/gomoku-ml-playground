"""Tests for v54's White-only labelling (ML batch 6 H1).

``generate_distill_data.py label --side {black,white}`` keeps
only positions with that side to move (see
``research/iterations/iter-54.md`` §2), and
``battery_to_games.py`` converts battery JSONs to the games
format the label step takes. These tests pin both: the side
filter selects exactly the right positions with the right
labels, and the converter preserves moves and winners.
"""

import json
import os
import subprocess
import sys

import numpy as np

from gomoku_ml import game, mcts_heuristic as mh

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
_LABEL_SCRIPT = os.path.join(_SCRIPTS, "generate_distill_data.py")
_CONVERT_SCRIPT = os.path.join(_SCRIPTS, "battery_to_games.py")

# Eight scattered plies: Black to move on even plies (t = 0, 2, 4,
# 6), White on odd plies (t = 1, 3, 5, 7). Seven of the eight
# positions are quiet — the last (t = 7, White to move) is
# terminal: Black holds (4,7)-(7,7), four in a column, so White
# faces an immediate loss and _terminal_leaf_value != 0.
_MOVES = [112, 113, 97, 127, 82, 142, 67, 157]


def _tiny_games_file(tmp_path):
    games = {
        "games": [
            {
                "game": 0, "source": "v7", "sims": 50, "seed": 1,
                "moves": _MOVES, "winner": 0, "plies": len(_MOVES),
            }
        ],
        "complete": True,
    }
    path = tmp_path / "games.json"
    path.write_text(json.dumps(games), encoding="utf-8")
    return path


def _label(tmp_path, games_path, side):
    out_dir = tmp_path / f"data-{side}"
    res = subprocess.run(
        [sys.executable, _LABEL_SCRIPT, "label",
         "--games", str(games_path), "--out-dir", str(out_dir),
         "--prefix", "t", "--label-mode", "rollout2",
         "--side", side],
        capture_output=True, text=True, check=True)
    assert "LABEL_JSON" in res.stdout
    return np.load(out_dir / "t_game0000.npz")


def _expected_values(side):
    boards = [game.empty_board()]
    player = game.BLACK
    for m in _MOVES:
        boards.append(game.apply_move(boards[-1], m, player))
        player = -player
    expected = []
    for t in range(len(_MOVES)):
        b = boards[t]
        p = game.BLACK if t % 2 == 0 else game.WHITE
        if mh._terminal_leaf_value(b, p) != 0.0:
            continue  # non-quiet positions are never labelled
        if side == "white" and p != game.WHITE:
            continue
        if side == "black" and p != game.BLACK:
            continue
        expected.append(mh._rollout2_leaf_value(b, p))
    return np.array(expected, dtype=np.float32)


def test_side_white_labels_only_white_positions(tmp_path):
    shard = _label(tmp_path, _tiny_games_file(tmp_path), "white")
    # Plane 3 is the side-to-move plane: all 0.0 iff White to move.
    # Three White quiet positions (t = 1, 3, 5; t = 7 is terminal).
    assert shard["planes"].shape[0] == 3
    assert np.all(shard["planes"][:, 3] == 0.0)
    np.testing.assert_allclose(
        shard["value"], _expected_values("white"), rtol=0, atol=1e-6)


def test_side_black_labels_only_black_positions(tmp_path):
    shard = _label(tmp_path, _tiny_games_file(tmp_path), "black")
    # Plane 3 all 1.0 iff Black to move.
    assert shard["planes"].shape[0] == 4
    assert np.all(shard["planes"][:, 3] == 1.0)
    np.testing.assert_allclose(
        shard["value"], _expected_values("black"), rtol=0, atol=1e-6)


def test_side_both_is_the_unfiltered_default(tmp_path):
    shard = _label(tmp_path, _tiny_games_file(tmp_path), "both")
    assert shard["planes"].shape[0] == 7  # 8 plies, 1 terminal
    np.testing.assert_allclose(
        shard["value"], _expected_values("both"), rtol=0, atol=1e-6)


def test_battery_to_games_converter(tmp_path):
    battery = {
        "records": [
            {"game": 0, "moves": [112, 113], "winner": "black"},
            {"game": 1, "moves": [112, 113, 97], "winner": "white"},
            {"game": 2, "moves": [112], "winner": "draw"},
        ]
    }
    battery_path = tmp_path / "battery.json"
    battery_path.write_text(json.dumps(battery), encoding="utf-8")
    out_path = tmp_path / "games.json"
    res = subprocess.run(
        [sys.executable, _CONVERT_SCRIPT, "--battery", str(battery_path),
         "--out", str(out_path)],
        capture_output=True, text=True, check=True)
    assert "converted 3 games" in res.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["games"] == [
        {"game": 0, "moves": [112, 113], "winner": 1, "plies": 2},
        {"game": 1, "moves": [112, 113, 97], "winner": -1, "plies": 3},
        {"game": 2, "moves": [112], "winner": 0, "plies": 1},
    ]


def test_converter_preserves_committed_v44_leg(tmp_path):
    # The committed v44-vs-v0 battery leg converts to 100 games
    # with moves and winners preserved exactly (the v54 patch
    # source, iter-54.md §2 leg (a)).
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    battery_path = os.path.join(
        repo, "..", "research", "iterations", "batch-v44-vs-v0.json")
    with open(battery_path, encoding="utf-8") as f:
        battery = json.load(f)
    out_path = tmp_path / "games.json"
    subprocess.run(
        [sys.executable, _CONVERT_SCRIPT, "--battery", battery_path,
         "--out", str(out_path)],
        capture_output=True, text=True, check=True)
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(payload["games"]) == 100
    mapping = {"black": 1, "white": -1, "draw": 0}
    for rec, g in zip(battery["records"], payload["games"]):
        assert g["moves"] == rec["moves"]
        assert g["winner"] == mapping[rec["winner"]]
        assert g["plies"] == len(rec["moves"])

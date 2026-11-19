"""Regression tests for v39 (batch 11, H1).

Each position in ``gomoku_ml/regression_positions_v39.json`` is a
v27 turn from the batch-10 legs, worked up in
``research/iterations/iter-39.md`` §2 (turn workups, 9/9): the
actual move loses 3/3 in playouts and the position is a quiet
setup-rich one (0 VCF starts / 0 creators for the opponent after
the actual move). v39 = v27 + the setup-at-root check
(``mcts_heuristic._setup_root_move``): after the search, a full
setup removal (0 VCF / 0 creators / 0 setups after it) overrides
a non-removal search move, ranked by the search.

* g46-ply24, g87-ply36: v39 plays a playout-verified SAVE
  ((10,7) or (8,9) at g46, both 3/3 wins; (4,2) at g87, 3/3
  draws) — the two verified setup-signature instances.
* g58-ply19, g26-ply25: v39 plays the full removal ((3,8),
  (6,7)) — the check fires and picks a removal, although the
  removal is NOT the verified save at these turns ((5,6),
  (5,8)). This pins the check's behaviour, including its
  documented limitation (iter-39 §1.3).
* g49-ply72: the actual move is a FORCED BLOCK (the opponent's
  only win) — the win/block shortcut fires before the search,
  so v39 plays (0,12) exactly like v27 and the check never runs.

Plus a unit test of the removal predicate on the VCF
non-monotonicity case: at g49, (1,9) "removes" the setups but
leaves the opponent an immediate win (0,12) — a VCF start —
so it is NOT a full removal. The check must full-re-check VCF
after the candidate move, not trust the original-starts subset
(iter-39 §2).
"""

import json
import os

import pytest

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import HeuristicMCTS

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v39.json"
)

# Expected v39 moves at seed 0 (verified by the mechanism check,
# iter-39 §3.1). g46 accepts either verified save (both are full
# removals; the search ranks (10,7) first at seed 0 by visits).
EXPECTED = {
    "g46-ply24": {(10, 7), (8, 9)},
    "g87-ply36": {(4, 2)},
    "g58-ply19": {(3, 8)},
    "g26-ply25": {(6, 7)},
    "g49-ply72": {(0, 12)},
}


def _load_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


@pytest.mark.parametrize("pos", _load_positions(), ids=lambda p: p["id"])
def test_v39_plays_expected_move(pos):
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v39 = scoring.get_player(39, seed=0)
    move = v39.choose_move(_board(pos), player)
    assert (move % 15, move // 15) in EXPECTED[pos["id"]]


def test_v39_setup_removal_rejects_vcf_creating_move():
    """(1,9) at g49-ply72 removes the setups but leaves White the
    immediate win (0,12) — a VCF start — so it is NOT a full
    removal. The predicate must catch this (full VCF re-check)."""
    pos = next(p for p in _load_positions() if p["id"] == "g49-ply72")
    board = _board(pos)
    mcts = HeuristicMCTS(setup_root=True)
    assert not mcts._is_setup_removal(board, game.BLACK, 9 * 15 + 1)
    # And the actual block (0,12) is not a removal either (it leaves
    # 3 setups) — the check would not treat it as one.
    assert not mcts._is_setup_removal(board, game.BLACK, 12 * 15 + 0)


def test_v39_setup_root_flag_off_is_v27():
    """With setup_root off, the engine is v27 exactly (the
    choose_move restructure must not change the search path)."""
    pos = next(p for p in _load_positions() if p["id"] == "g46-ply24")
    board = _board(pos)
    v27 = scoring.get_player(27, seed=0)
    m27 = v27.choose_move(board, game.BLACK)
    mcts = HeuristicMCTS(
        sims=100, leaf_mode="rollout2_v1_both",
        tie_break_q=True, tie_break_q_threshold=0.05,
        setup_root=False,
    )
    import numpy as np

    mcts.rng = np.random.default_rng(0)
    assert mcts.choose_move(board, game.BLACK) == m27

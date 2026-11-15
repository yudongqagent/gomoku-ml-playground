"""Tests for v13 (batch 5, H2 — White focus).

v13's base (iter-13 §1b) is v7's PUCT MCTS (capped
attack-gradient leaf) + v3's 2-ply threat check at the root,
with the White sims count chosen by the §1.2 sweep (100: the
peak of the non-monotonicity — White 0.775 vs v0 at 100 sims,
0.525 at 50, 0.55 at 150, 0.00 at 200). The White-specific
priors/book design lands in §1b and gets its own tests here.
"""

import numpy as np

from gomoku_ml import game, scoring
from gomoku_ml.mcts_white import WhiteMCTS


def _scattered_board(seed=0, stones=24):
    rng = np.random.default_rng(seed)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:stones]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    return board


def test_v13_registered_and_dispatches_to_white_mcts():
    v13 = scoring.get_player(13, seed=0)
    assert v13.name == "scoring-v13"
    assert v13.mcts is True
    assert isinstance(v13._mcts, WhiteMCTS)
    assert v13._mcts.sims == 100
    assert v13._mcts.white_sims == 100  # the sweep winner (iter-13 §2.1)
    assert v13._mcts.leaf_mode == "gradient"


def test_v13_uses_100_sims_in_both_colours():
    # At the sweep winner the sims count is symmetric (100/100);
    # only the threat check distinguishes v13's base from v7.
    v13 = scoring.get_player(13, seed=0)
    board = _scattered_board()
    assert not scoring._winning_moves(board, game.BLACK)
    assert not scoring._winning_moves(board, game.WHITE)
    v13.choose_move(board, game.BLACK)
    assert v13._mcts.last_sims_used == 100
    v13.choose_move(board, game.WHITE)
    assert v13._mcts.last_sims_used == 100

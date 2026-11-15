"""Tests for v13 (batch 5, H2 — White focus).

v13's base (iter-13 §1b) is v7's PUCT MCTS (capped
attack-gradient leaf) + v3's 2-ply threat check at the root,
with the White sims count chosen by the §1.2 sweep (100: the
peak of the non-monotonicity — White 0.775 vs v0 at 100 sims,
0.525 at 50, 0.55 at 150, 0.00 at 200). The White-specific
change is a White-only prior temperature (``V13MCTS``): the
prior softmax temperature is ``white_temp`` when the engine
plays White, 2.0 (v7's default) when it plays Black.
"""

import math

import numpy as np

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import _PRIOR_TEMPERATURE, _priors
from gomoku_ml.mcts_v13 import (
    _BLACK_TEMPERATURE,
    _priors_temp,
    V13MCTS,
)
from gomoku_ml.mcts_white import WhiteMCTS


def _scattered_board(seed=0, stones=24):
    rng = np.random.default_rng(seed)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:stones]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    return board


def test_v13_registered_and_dispatches_to_v13_mcts():
    v13 = scoring.get_player(13, seed=0)
    assert v13.name == "scoring-v13"
    assert v13.mcts is True
    assert isinstance(v13._mcts, V13MCTS)
    assert isinstance(v13._mcts, WhiteMCTS)  # V13MCTS subclasses it
    assert v13._mcts.sims == 100
    assert v13._mcts.white_sims == 100  # the sweep winner (iter-13 §2.1)
    # The §1b temperature sweep REFUTED flattening (White 0.775
    # at 2.0, 0.05 at 3.0, 0.00 at 4.0 vs v0): v13 ships at 2.0.
    assert v13._mcts.white_temp == 2.0
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


def test_black_temperature_matches_shared_module():
    # v13's Black temperature must track mcts_heuristic's
    # constant exactly; a drift would silently change the Black
    # game. (mcts_heuristic.py is not edited by this workstream.)
    assert _BLACK_TEMPERATURE == _PRIOR_TEMPERATURE == 2.0


def test_priors_temp_matches_priors_at_default_temperature():
    # The temperature-aware priors function is a copy of
    # mcts_heuristic._priors: at 2.0 it must agree exactly.
    board = _scattered_board()
    for player in (game.BLACK, game.WHITE):
        assert _priors_temp(board, player, 2.0) == _priors(board, player)


def test_higher_temperature_flattens_the_prior():
    board = _scattered_board()
    p2 = _priors_temp(board, game.WHITE, 2.0)
    p4 = _priors_temp(board, game.WHITE, 4.0)
    assert max(p4.values()) < max(p2.values())
    assert min(p4.values()) > min(p2.values())
    # Entropy increases with temperature.
    h = lambda p: -sum(v * math.log(v) for v in p.values())
    assert h(p4) > h(p2)


def test_v13_matches_white_mcts_at_default_temperature():
    # At white_temp=2.0 the _run_one replica must behave exactly
    # like the parent: same move, same RNG seed, same position.
    board = _scattered_board()
    assert not scoring._winning_moves(board, game.BLACK)
    assert not scoring._winning_moves(board, game.WHITE)
    a = WhiteMCTS(sims=30, white_sims=30, rng=np.random.default_rng(0))
    b = V13MCTS(
        sims=30, white_sims=30, white_temp=2.0,
        rng=np.random.default_rng(0),
    )
    for player in (game.BLACK, game.WHITE):
        assert a.choose_move(board.copy(), player) == b.choose_move(
            board.copy(), player
        )


def test_temperature_selected_by_engine_colour():
    mcts = V13MCTS(sims=10, white_sims=10, white_temp=4.0,
                   rng=np.random.default_rng(0))
    board = _scattered_board()
    mcts.choose_move(board, game.BLACK)
    assert mcts._search_temp == 2.0
    mcts.choose_move(board, game.WHITE)
    assert mcts._search_temp == 4.0

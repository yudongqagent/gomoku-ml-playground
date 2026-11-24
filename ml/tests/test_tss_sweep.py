"""Tests for v57 TSS depth/budget sweep (batch 14, H1).

Covers the pre-registered sweep design (``research/iterations/
iter-57.md`` §1):

* the sweep cells use the SAME forcing-move enumeration as v52
  (the cells differ only in depth/budget, never in which moves are
  enumerated — the batch-13 kernel-blindness finding: v1 for
  four-creators, v0 for open-three creators);
* in the unbounded case depth-5 is a superset of depth-4 (every
  depth-4 win is found at depth 5) — so a cell can only LOSE a
  win vs the control via a budget abort, which the mining script
  counts separately;
* a budget abort loses a win the control finds (the loss
  mechanism, pinned with a tiny budget);
* the ``tss_depth`` / ``tss_node_budget`` plumbing (defaults are
  v52's values; invalid values rejected).
"""

import json
import os

import numpy as np
import pytest

from gomoku_ml import game, mcts_heuristic, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tss_mechanism_positions.json"
)


def _load_mechanism_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


def _player(pos):
    return game.BLACK if pos["to_move"] == "black" else game.WHITE


def _xy(move):
    return [move % 15, move // 15]


# --- enumeration: identical across cells --------------------------------

def test_sweep_cells_share_v52_enumeration():
    """The forcing-move enumeration takes no depth/budget: every
    sweep cell enumerates exactly v52's moves. Spot-checked on the
    mechanism positions (incl. the broken-four-sensitive T3)."""
    for pos in _load_mechanism_positions():
        board = _board(pos)
        player = _player(pos)
        moves = mcts_heuristic.tss_forcing_moves(board, player)
        # The enumeration is a pure function of (board, player):
        # calling it again must give the identical list (and it is
        # what every depth/budget cell searches over).
        assert mcts_heuristic.tss_forcing_moves(board, player) == moves
        fours, threes, _a1 = mcts_heuristic._forcing_sets(board, player)
        assert sorted(fours | set(threes)) == moves


# --- depth superset (unbounded) ------------------------------------------

def test_depth5_superset_of_depth4_at_high_budget():
    """In the unbounded case depth-5 finds every depth-4 win: on
    every mechanism position where depth-4 fires, depth-5 (budget
    100000, so nothing binds) fires with a move in the expected
    set; on T7 (counter-four refutation) both return None."""
    for pos in _load_mechanism_positions():
        board = _board(pos)
        player = _player(pos)
        expected = [tuple(e) for e in pos["expected"]]
        d4 = mcts_heuristic.tss_winning_move(
            board, player, depth=4, node_budget=100000
        )
        d5 = mcts_heuristic.tss_winning_move(
            board, player, depth=5, node_budget=100000
        )
        d6 = mcts_heuristic.tss_winning_move(
            board, player, depth=6, node_budget=100000
        )
        if not expected:
            assert d4 is None and d5 is None and d6 is None, pos["id"]
        else:
            assert d4 is not None and tuple(_xy(d4)) in expected, pos["id"]
            assert d5 is not None and tuple(_xy(d5)) in expected, pos["id"]
            assert d6 is not None and tuple(_xy(d6)) in expected, pos["id"]


# --- the loss mechanism: budget abort loses a win -------------------------

def test_budget_abort_loses_a_win_the_control_finds():
    """The sweep's loss mechanism, pinned: on T3 the depth-4/3000
    control finds the forcing win, but depth-5 at a tiny budget
    aborts (budget_exceeded) and returns None — the deeper search
    burns its budget on the longer tree before reaching the win.
    At a production budget depth-5 finds it (superset, above)."""
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-three-sequence-black"
    )
    board = _board(pos)
    player = _player(pos)
    control = mcts_heuristic.tss_winning_move(
        board, player, depth=4, node_budget=3000
    )
    assert control is not None
    assert _xy(control) in pos["expected"]
    stats: dict = {}
    assert (
        mcts_heuristic.tss_winning_move(
            board, player, depth=5, node_budget=1, stats=stats
        )
        is None
    )
    assert stats["budget_exceeded"] is True


# --- plumbing ---------------------------------------------------------------

def test_tss_depth_budget_plumbing_defaults():
    """v52's engine carries the v52 defaults (depth 4, budget 3000)
    — the plumbing changes nothing for v52."""
    v52 = scoring.get_player(52, seed=0)
    assert v52._mcts.tss_depth == mcts_heuristic.TSS_DEPTH == 4
    assert v52._mcts.tss_node_budget == mcts_heuristic.TSS_NODE_BUDGET == 3000


def test_tss_depth_budget_plumbing_overrides():
    """Explicit depth/budget reach the hook: a v52-config engine
    with depth 5 / budget 6000 plays the TSS win on T1 (and the
    override is visible on the constructed MCTS)."""
    mcts = mcts_heuristic.HeuristicMCTS(
        sims=100,
        leaf_mode="rollout2_v1_both",
        tie_break_q=True,
        tie_break_q_threshold=0.05,
        tss_root=True,
        tss_depth=5,
        tss_node_budget=6000,
    )
    assert mcts.tss_depth == 5
    assert mcts.tss_node_budget == 6000
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    assert _xy(mcts.choose_move(_board(pos), game.BLACK)) == [6, 6]


def test_tss_depth_budget_validation():
    with pytest.raises(ValueError, match="tss_depth"):
        mcts_heuristic.HeuristicMCTS(tss_root=True, tss_depth=0)
    with pytest.raises(ValueError, match="tss_depth"):
        mcts_heuristic.HeuristicMCTS(tss_root=True, tss_depth=2.5)
    with pytest.raises(ValueError, match="tss_node_budget"):
        mcts_heuristic.HeuristicMCTS(tss_root=True, tss_node_budget=0)
    with pytest.raises(ValueError, match="tss_node_budget"):
        mcts_heuristic.HeuristicMCTS(tss_root=True, tss_node_budget=-10)


def test_depth6_finds_t1_win_at_6000():
    """The deepest cell still finds the trivial double-three win
    (sanity: depth 6 does not break the search itself)."""
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    move = mcts_heuristic.tss_winning_move(
        _board(pos), game.BLACK, depth=6, node_budget=6000
    )
    assert move is not None and _xy(move) == [6, 6]


# --- v57 engine (sweep winner: depth 5, budget 6000) ----------------------

# A verified (5, 6000) gain mined in Check M (iter-57 §3): game 6,
# Black to move — depth-5/6000 finds the forcing win (5,8) that
# depth-4/3000 misses (verified 3/3 in playouts vs v52's defence).
# Pinned here as the regression test for the sweep's mechanism.
_MINED_DEEP_WIN_BLACK = [(5, 4), (5, 5), (6, 5), (5, 6), (7, 7), (6, 9)]
_MINED_DEEP_WIN_WHITE = [(4, 3), (5, 3), (7, 4), (6, 6), (8, 6), (6, 7)]


def _mined_deep_win_board():
    board = game.empty_board()
    for x, y in _MINED_DEEP_WIN_BLACK:
        board[y, x] = game.BLACK
    for x, y in _MINED_DEEP_WIN_WHITE:
        board[y, x] = game.WHITE
    return board


def test_v57_registration():
    v57 = scoring.get_player(57, seed=0)
    assert v57.name == "scoring-v57"
    assert v57.version == 57
    mcts = v57._mcts
    assert isinstance(mcts, mcts_heuristic.HeuristicMCTS)
    assert mcts.tss_root is True
    assert mcts.tss_depth == 5
    assert mcts.tss_node_budget == 6000
    # Everything else is v52 exactly.
    assert mcts.leaf_mode == "rollout2_v1_both"
    assert mcts.tie_break_q is True
    assert mcts.tie_break_q_threshold == 0.05
    assert mcts.sims == 100
    assert mcts.c_puct == 1.5


def test_v57_plays_tss_win_on_double_three():
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    board = _board(pos)
    v57 = scoring.get_player(57, seed=0)
    move = v57.choose_move(board, game.BLACK)
    assert _xy(move) == [6, 6]


def test_v57_finds_mined_deep_win_depth4_misses():
    """The sweep's mechanism, pinned: on the mined game-6 position,
    depth-4/3000 (v52) finds no forcing win, depth-5/6000 (v57)
    finds (5,8), and v57 plays it."""
    board = _mined_deep_win_board()
    assert (
        mcts_heuristic.tss_winning_move(
            board, game.BLACK, depth=4, node_budget=3000
        )
        is None
    )
    move = mcts_heuristic.tss_winning_move(
        board, game.BLACK, depth=5, node_budget=6000
    )
    assert move is not None and _xy(move) == [5, 8]
    v57 = scoring.get_player(57, seed=0)
    assert _xy(v57.choose_move(board, game.BLACK)) == [5, 8]


def test_v57_falls_through_to_v52_when_tss_silent():
    """On positions where TSS finds nothing at EITHER depth (both
    the (4,3000) control and the (5,6000) cell silent — a position
    silent at depth 5 could still fire at depth 4 via a depth-5
    budget abort, so both must be checked), v57 IS v52 exactly:
    same move at the same seed (TSS consumes no RNG, so the MCTS
    streams are identical). Positions are early v7 self-play plies
    filtered to doubly-silent ones, deterministic seed."""
    positions = []
    rng = np.random.default_rng(777)
    while len(positions) < 5:
        board = game.empty_board()
        player = game.BLACK
        for _ in range(int(rng.integers(4, 12))):
            empties = np.flatnonzero(board.reshape(-1) == game.EMPTY)
            board = game.apply_move(
                board, int(rng.choice(empties)), player
            )
            player = -player
        if (
            mcts_heuristic.tss_winning_move(
                board, player, depth=4, node_budget=3000
            )
            is None
            and mcts_heuristic.tss_winning_move(
                board, player, depth=5, node_budget=6000
            )
            is None
        ):
            positions.append((board.copy(), player))
    assert len(positions) == 5
    for board, player in positions:
        v57 = scoring.get_player(57, seed=0)
        v52 = scoring.get_player(52, seed=0)
        assert v57.choose_move(board, player) == v52.choose_move(
            board, player
        )

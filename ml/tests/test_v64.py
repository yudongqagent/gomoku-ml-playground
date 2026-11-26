"""Tests for v64 (iteration 64): the fix for the pre-existing v52 TSS
defect found by v59 in batch 14 (research/iterations/iter-59.md §2).

The defect: in ``mcts_heuristic._tss_after_attacker_move``, if the
attacker's just-played forcing move is itself an immediate winning
move, the recursion misjudged the line — with exactly 1 extension
threat the defender's "block" left ``game_over`` True (the five
persists) and the function returned False, although the attacker had
already won; with 0 extension threats it fell through to the
open-three check and also returned False. So TSS could miss a forced
win whose converting move is an immediate win with < 2 extension
cells. (With >= 2 extension threats it already returned True.)

The fix (``fix_won_move``) is an entry check in
``_tss_after_attacker_move``: ``game.is_winning_stone(board,
last_move, attacker)`` — O(1) and exact, since a five through
``last_move`` must include it (the cell was empty before the move).
It is flag-gated (default False) so v52's behaviour is preserved
exactly; v64 (``ENGINE_VERSIONS[64]``) sets it True. The tests below
call the flag explicitly, so they fail on v52 code (no such
parameter) and pass on v64.

Reachability note (iter-64 §2): the defect is NOT reachable through
``tss_winning_move`` on real-game positions — every caller checks
for a winning move before playing a forcing move, and
``scoring._winning_moves`` is oracle-complete — so the regression
tests are at the ``_tss_after_attacker_move`` level, the function
that carries the defect, plus engine-level tests that the flag
changes nothing where the defect cannot bite.
"""

import json
import os

from gomoku_ml import game, mcts_heuristic, scoring
from gomoku_ml.mcts_heuristic import _TSSBudget

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tss_mechanism_positions.json"
)


def _board(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def _after_attacker_move(board, last_move, fix):
    return mcts_heuristic._tss_after_attacker_move(
        board, game.BLACK, 3, _TSSBudget(3000), last_move, fix_won_move=fix
    )


# Five on row 7, columns 5-9; White blocks the left end (7,4).
FIVE_A = [(x, 7) for x in range(5, 10)]
# Five on row 7 at the board edge, columns 10-14.
FIVE_B = [(x, 7) for x in range(10, 15)]


def test_already_won_one_extension_threat():
    """The recorded defect: exactly 1 extension threat. v52 returns
    False (the defender's "block" leaves game_over True); v64
    returns True — the attacker already won."""
    board = _board(FIVE_A, [(4, 7)])
    last = game.xy_to_move(9, 7)
    assert game.is_winning_stone(board, last, game.BLACK)
    assert mcts_heuristic._winning_moves_through(board, game.BLACK, last) == [
        game.xy_to_move(10, 7)
    ]
    assert _after_attacker_move(board, last, fix=False) is False
    assert _after_attacker_move(board, last, fix=True) is True


def test_already_won_zero_extension_threats_edge():
    """0 extension threats: the five sits at the board edge and the
    just-played stone is the edge stone. v52 falls through to the
    open-three check and returns False; v64 returns True."""
    board = _board(FIVE_B, [])
    last = game.xy_to_move(14, 7)
    assert game.is_winning_stone(board, last, game.BLACK)
    assert mcts_heuristic._winning_moves_through(board, game.BLACK, last) == []
    assert _after_attacker_move(board, last, fix=False) is False
    assert _after_attacker_move(board, last, fix=True) is True


def test_already_won_zero_extension_threats_blocked():
    """0 extension threats: both ends of the five are blocked by
    White stones. v52 returns False; v64 returns True."""
    board = _board(FIVE_A, [(4, 7), (10, 7)])
    last = game.xy_to_move(9, 7)
    assert game.is_winning_stone(board, last, game.BLACK)
    assert mcts_heuristic._winning_moves_through(board, game.BLACK, last) == []
    assert _after_attacker_move(board, last, fix=False) is False
    assert _after_attacker_move(board, last, fix=True) is True


def test_two_extension_threats_unchanged():
    """Control: with 2 extension threats v52 already returned True
    (open four: unblockable). The fix must not change that."""
    board = _board(FIVE_A, [])
    last = game.xy_to_move(7, 7)
    assert game.is_winning_stone(board, last, game.BLACK)
    assert mcts_heuristic._winning_moves_through(board, game.BLACK, last) == [
        game.xy_to_move(4, 7),
        game.xy_to_move(10, 7),
    ]
    assert _after_attacker_move(board, last, fix=False) is True
    assert _after_attacker_move(board, last, fix=True) is True


def test_fix_is_flag_gated_v52_unchanged():
    """v52's registration has no fix flag (defaults False); v64 sets
    it. The default call path is v52 behaviour exactly."""
    assert not scoring.ENGINE_VERSIONS[52].get("tss_fix_won_move", False)
    assert scoring.ENGINE_VERSIONS[64]["tss_fix_won_move"] is True
    assert scoring.ENGINE_VERSIONS[64]["tss_root"] is True
    # Same base engine as v52 in every other respect.
    for key in ("mcts_sims", "leaf_mode", "tie_break_q",
                "tie_break_q_threshold"):
        assert scoring.ENGINE_VERSIONS[64][key] == scoring.ENGINE_VERSIONS[52][key]


def _load_mechanism_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _pos_board(pos):
    return _board(pos["black"], pos["white"])


def _pos_player(pos):
    return game.BLACK if pos["to_move"] == "black" else game.WHITE


def test_fix_changes_nothing_on_mechanism_positions():
    """On the v52 mechanism positions (where the defect cannot bite),
    the fix must not change any TSS result — same move, same node
    count. Guards against the flag changing the search elsewhere."""
    for pos in _load_mechanism_positions():
        board = _pos_board(pos)
        player = _pos_player(pos)
        stats_off, stats_on = {}, {}
        move_off = mcts_heuristic.tss_winning_move(
            board, player, stats=stats_off
        )
        move_on = mcts_heuristic.tss_winning_move(
            board, player, stats=stats_on, fix_won_move=True
        )
        assert move_on == move_off, pos["id"]
        assert stats_on == stats_off, pos["id"]


def test_v64_engine_plays_tss_win_on_t1():
    """v64 (via the registered engine) plays the TSS win on the
    double-three mechanism position T1, exactly like v52."""
    pos = next(p for p in _load_mechanism_positions() if p["id"] == "tss-double-three-black")
    board = _pos_board(pos)
    player = _pos_player(pos)
    p64 = scoring.get_player(64, seed=100)
    p52 = scoring.get_player(52, seed=100)
    assert p64.choose_move(board, player) == game.xy_to_move(6, 6)
    assert p52.choose_move(board, player) == game.xy_to_move(6, 6)

"""Tests for v52 threat-space search (batch 13, H2).

Covers the TSS building blocks and the pre-registered mechanism
check (``research/iterations/iter-52.md`` §1.3, Check A):

* the kernel-based forcing-move enumeration matches brute-force
  oracles on random positions (0 mismatches) — the fast four-
  creator enumeration matches ``evaluate.four_creating_moves``,
  and the open-three enumeration matches a run-scan oracle;
* TSS finds the forcing win on the constructed positions
  (``tss_mechanism_positions.json``) and on the tactical suite's
  VCF positions, returns None where no forcing win exists, and
  never returns a refuted move;
* the depth and node-budget accounting behave as pre-registered;
* v52 falls through to v27 exactly on positions where TSS does
  not fire (same move, same seed — TSS consumes no RNG).
"""

import json
import os

import numpy as np
import pytest

from gomoku_ml import evaluate, game, mcts_heuristic, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tss_mechanism_positions.json"
)
TACTICAL_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tactical_positions.json"
)
TACTICAL_HARD_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tactical_positions_hard.json"
)


def _load_mechanism_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _load_suite(path):
    with open(path, encoding="utf-8") as f:
        return {p["id"]: p for p in json.load(f)["positions"]}


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


# --- enumeration oracles -------------------------------------------------

def _brute_open_three_creators(board, player):
    """Brute-force oracle: for each near move, apply it and run-scan
    the 4 directions through the move for a contiguous run of exactly
    3 friendly stones with both ends empty. Four-creators excluded
    (same convention as the kernel version)."""
    out = []
    fours = set(evaluate.four_creating_moves(board, player))
    for m in scoring._candidate_moves(board):
        if m in fours:
            continue
        b2 = game.apply_move(board, m, player)
        y, x = m // 15, m % 15
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            # Run containing the move.
            lo_y, lo_x = y, x
            while (
                0 <= lo_y - dy < 15
                and 0 <= lo_x - dx < 15
                and b2[lo_y - dy, lo_x - dx] == player
            ):
                lo_y -= dy
                lo_x -= dx
            hi_y, hi_x = y, x
            while (
                0 <= hi_y + dy < 15
                and 0 <= hi_x + dx < 15
                and b2[hi_y + dy, hi_x + dx] == player
            ):
                hi_y += dy
                hi_x += dx
            length = abs(hi_y - lo_y) + abs(hi_x - lo_x) + 1
            if dy != 0 and dx != 0:
                length = abs(hi_y - lo_y) + 1
            if length != 3:
                continue
            by, bx = lo_y - dy, lo_x - dx
            ay, ax = hi_y + dy, hi_x + dx
            if not (0 <= by < 15 and 0 <= bx < 15):
                continue
            if not (0 <= ay < 15 and 0 <= ax < 15):
                continue
            if b2[by, bx] == game.EMPTY and b2[ay, ax] == game.EMPTY:
                out.append(m)
                break
    return sorted(out)


_POSITIONS_CACHE: dict = {}


def _random_positions(seed=4242, n=80):
    """Quiet + midgame positions: 40 v7 self-play (100 sims) plies
    plus random playouts (capped at 40 plies per game, so the boards
    stay sparse) to fill, deterministic seed. Random playouts create
    dense forcing structure (random clusters), which is exactly the
    coverage the enumeration oracles need. Returns (board, player).
    Cached per (seed, n): position generation is the slow part."""
    key = (seed, n)
    if key in _POSITIONS_CACHE:
        return _POSITIONS_CACHE[key]
    rng = np.random.default_rng(seed)
    positions = []
    g = 0
    while len(positions) < min(40, n):
        pa = scoring.get_player(7, seed=seed + g)
        pb = scoring.get_player(7, seed=seed + 100 + g)
        board = game.empty_board()
        player = game.BLACK
        while len(positions) < min(40, n):
            over, _ = game.game_over(board)
            if over:
                break
            positions.append((board.copy(), player))
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        g += 1
    while len(positions) < n:
        board = game.empty_board()
        player = game.BLACK
        ply = 0
        while len(positions) < n and ply < 40:
            over, _ = game.game_over(board)
            if over:
                break
            positions.append((board.copy(), player))
            empties = np.flatnonzero(board.reshape(-1) == game.EMPTY)
            board = game.apply_move(board, int(rng.choice(empties)), player)
            player = -player
            ply += 1
    positions = positions[:n]
    _POSITIONS_CACHE[key] = positions
    return positions


def _no_winning_move_positions(seed=4242, n=80):
    """Positions where the player to move holds NO winning move —
    the only case TSS enumerates forcing moves in (every caller
    checks for a winning move first). ``evaluate.four_creating_moves``
    counts pre-existing fours (its test is 'a winning move exists
    after the move'), so the equivalence checks run only here."""
    out = [
        (b, p)
        for b, p in _random_positions(seed, n)
        if not scoring._winning_moves(b, p)
    ]
    assert len(out) >= 40, f"only {len(out)} usable positions"
    return out


def test_fast_four_creators_match_evaluate():
    """The kernel-prefiltered four-creator enumeration is exactly
    evaluate.four_creating_moves (0 mismatches) — load-bearing for
    the defender counter-four set: a missed counter-four would be a
    false forced win."""
    mismatches = 0
    for board, player in _no_winning_move_positions():
        fast = sorted(mcts_heuristic._four_creating_moves_fast(board, player))
        exact = sorted(evaluate.four_creating_moves(board, player))
        if fast != exact:
            mismatches += 1
    assert mismatches == 0


def test_open_three_creators_match_brute_force():
    """The kernel-based open-three enumeration matches the run-scan
    oracle (0 mismatches) — the v0>=1000 test is exact for 'creates
    a true open three'. (An earlier v1>=4000 version of this test
    FAILED: v1's window-based open_ends misses genuine open threes
    whose two-out neighbourhood is occupied — v1 attack 405 for a
    real open-three creator. The v0 kernel is the instrument.)"""
    mismatches = 0
    for board, player in _no_winning_move_positions():
        kernel = sorted(
            mcts_heuristic.open_three_creating_moves(board, player)
        )
        brute = _brute_open_three_creators(board, player)
        if kernel != brute:
            mismatches += 1
    assert mismatches == 0


def test_winning_moves_through_matches_full_scan():
    """The local winning-move check (lines through the just-played
    stone) equals the full scan whenever no winning move pre-existed
    — the TSS precondition, and the property the recursion's local
    checks rest on. On random positions, for a random legal move m
    by a player with no winning move, the winning moves after m are
    exactly the local ones through m."""
    rng = np.random.default_rng(99)
    checked = 0
    for board, player in _no_winning_move_positions():
        cands = scoring._candidate_moves(board)
        if not cands:
            continue
        m = int(rng.choice(cands))
        b2 = game.apply_move(board, m, player)
        full = sorted(scoring._winning_moves(b2, player))
        local = sorted(mcts_heuristic._winning_moves_through(b2, player, m))
        assert local == full, (
            f"move {(m % 15, m // 15)}: local {local} != full {full}"
        )
        checked += 1
    assert checked >= 40


def test_all_open_threes():
    # T1 after the double-three move: two open threes.
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    board = _board(pos)
    b2 = game.apply_move(board, game.xy_to_move(6, 6), game.BLACK)
    threes = mcts_heuristic._all_open_threes(b2, game.BLACK)
    assert len(threes) == 2
    ends = {frozenset(t) for t in threes}
    assert ends == {
        frozenset({game.xy_to_move(5, 6), game.xy_to_move(9, 6)}),
        frozenset({game.xy_to_move(6, 5), game.xy_to_move(6, 9)}),
    }
    # A gap three is NOT an open three (run length 2, not 3).
    board = game.empty_board()
    for x, y in [(5, 5), (6, 5), (8, 5)]:
        board[y, x] = game.BLACK
    assert mcts_heuristic._all_open_threes(board, game.BLACK) == []
    # A closed three (one end blocked) is NOT an open three.
    board = game.empty_board()
    for x, y in [(5, 5), (6, 5), (7, 5)]:
        board[y, x] = game.BLACK
    board[5, 8] = game.WHITE
    assert mcts_heuristic._all_open_threes(board, game.BLACK) == []
    # A three touching the edge is NOT open.
    board = game.empty_board()
    for x, y in [(0, 5), (1, 5), (2, 5)]:
        board[y, x] = game.BLACK
    assert mcts_heuristic._all_open_threes(board, game.BLACK) == []


# --- mechanism check (Check A, pre-registered positions) ------------------

@pytest.mark.parametrize(
    "pos", _load_mechanism_positions(), ids=lambda p: p["id"]
)
def test_tss_mechanism_positions(pos):
    board = _board(pos)
    player = _player(pos)
    move = mcts_heuristic.tss_winning_move(board, player)
    if not pos["expected"]:
        assert move is None, (
            f"{pos['id']}: TSS returned {_xy(move)} but no forcing "
            "win exists (the refuted move must not be returned)"
        )
        return
    assert move is not None, f"{pos['id']}: TSS found no forcing win"
    assert _xy(move) in pos["expected"], (
        f"{pos['id']}: TSS returned {_xy(move)}, expected one of "
        f"{pos['expected']}"
    )
    assert _xy(move) not in pos["refuted"], (
        f"{pos['id']}: TSS returned the refuted move {_xy(move)}"
    )


@pytest.mark.parametrize(
    "suite_path,pos_id",
    [
        (TACTICAL_PATH, "vcf-1"),
        (TACTICAL_PATH, "vcf-2"),
        (TACTICAL_HARD_PATH, "vcf-two-four-sequence"),
        (TACTICAL_HARD_PATH, "vcf-two-four-sequence-rot"),
        (TACTICAL_HARD_PATH, "vcf-race-block-or-win"),
    ],
)
def test_tss_finds_suite_vcf_positions(suite_path, pos_id):
    """TSS subsumes the offensive VCF check: on every suite VCF
    position it returns a move that starts a forced win by
    consecutive fours (the suite's own VCF definition)."""
    suite = _load_suite(suite_path)
    pos = suite[pos_id]
    board = evaluate.board_from_position(pos)
    player = pos["to_move"]
    move = mcts_heuristic.tss_winning_move(board, player)
    assert move is not None, f"{pos_id}: TSS found no forcing win"
    assert evaluate.starts_forced_win(board, player, move, depth=3), (
        f"{pos_id}: TSS returned {_xy(move)}, not a VCF start"
    )


def test_tss_returns_none_on_quiet_positions():
    board = game.empty_board()
    assert mcts_heuristic.tss_winning_move(board, game.BLACK) is None
    # A few scattered stones, no forcing structure at all.
    board = game.empty_board()
    for x, y in [(7, 7), (3, 3), (11, 11)]:
        board[y, x] = game.BLACK
    for x, y in [(7, 8), (12, 3)]:
        board[y, x] = game.WHITE
    assert mcts_heuristic.tss_winning_move(board, game.BLACK) is None


def test_tss_returns_immediate_win_first():
    """An immediate winning move IS a forcing win (standing fact)
    and is returned first — it is a VCF start but not necessarily
    a four-creator, so the fast test checks winning moves first."""
    board = game.empty_board()
    for x, y in [(7, 7), (8, 7), (9, 7), (10, 7)]:
        board[y, x] = game.BLACK
    move = mcts_heuristic.tss_winning_move(board, game.BLACK)
    assert move is not None
    assert game.is_winning_move(board, move, game.BLACK)


def test_tss_depth_accounting():
    """The T3 sequence needs 2 attacker moves ((7,9) then (7,7),
    or the equivalent lines): depth 1 finds nothing, depth 2 finds
    the win. Pins the depth accounting (the root move consumes 1)."""
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-three-sequence-black"
    )
    board = _board(pos)
    player = _player(pos)
    assert mcts_heuristic.tss_winning_move(board, player, depth=1) is None
    move = mcts_heuristic.tss_winning_move(board, player, depth=2)
    assert move is not None
    assert _xy(move) in pos["expected"]


def test_tss_node_budget_aborts_deterministically():
    """On budget exhaustion TSS returns None (fall through to the
    MCTS) — never a partial answer. A zero budget aborts even the
    trivial double-three win."""
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    board = _board(pos)
    player = _player(pos)
    stats: dict = {}
    assert (
        mcts_heuristic.tss_winning_move(
            board, player, node_budget=0, stats=stats
        )
        is None
    )
    assert stats["budget_exceeded"] is True
    stats = {}
    move = mcts_heuristic.tss_winning_move(
        board, player, node_budget=100000, stats=stats
    )
    assert move is not None and _xy(move) == [6, 6]
    assert stats["budget_exceeded"] is False
    assert stats["nodes"] > 0


# --- v52 engine ------------------------------------------------------------

def test_v52_registration():
    v52 = scoring.get_player(52, seed=0)
    assert v52.name == "scoring-v52"
    assert v52.version == 52
    mcts = v52._mcts
    assert isinstance(mcts, mcts_heuristic.HeuristicMCTS)
    assert mcts.tss_root is True
    assert mcts.leaf_mode == "rollout2_v1_both"
    assert mcts.tie_break_q is True
    assert mcts.tie_break_q_threshold == 0.05
    assert mcts.sims == 100


def test_v52_plays_tss_win_on_double_three():
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    board = _board(pos)
    v52 = scoring.get_player(52, seed=0)
    move = v52.choose_move(board, game.BLACK)
    assert _xy(move) == [6, 6]


def test_v52_falls_through_to_v27_when_tss_silent():
    """On positions where TSS finds nothing, v52 IS v27 exactly:
    same move at the same seed (TSS consumes no RNG, so the MCTS
    streams are identical). Positions are early v7 self-play plies
    filtered to TSS-silent ones, deterministic seed."""
    positions = []
    for board, player in _random_positions(seed=777, n=40):
        if mcts_heuristic.tss_winning_move(board, player) is None:
            positions.append((board, player))
        if len(positions) >= 5:
            break
    assert len(positions) == 5
    for board, player in positions:
        v52 = scoring.get_player(52, seed=0)
        v27 = scoring.get_player(27, seed=0)
        assert v52.choose_move(board, player) == v27.choose_move(
            board, player
        )

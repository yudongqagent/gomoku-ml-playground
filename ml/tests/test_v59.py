"""Tests for v59 TSS speed optimisation (batch 14, H3).

Every test is a move-choice-identity check: the fast path must agree
EXACTLY with v52's kernel path (0 mismatches — the batch-10 A/B
standard, iter-59 §1.3):

* the window-scan forcing enumeration matches the kernel enumeration
  (fours set, threes set, and the v1 ordering scores on forcing
  moves) on the no-winning-move positions, and matches the same
  oracles (``evaluate.four_creating_moves``, the brute-force
  run-scan oracle for open threes);
* the run-based winning-move test matches the full scan wherever
  no winning move pre-existed (the TSS precondition);
* the fast TSS returns the same move with the same node count and
  the same budget-exceeded flag as the slow TSS, on the mechanism
  positions and on mined real-game positions;
* the two kernel-blind-spot positions (broken-four creator, v1-blind
  open three) are found by the window scan;
* v59 is registered, plays the TSS win on T1, and falls through to
  v27 exactly when TSS is silent (same test as v52's).

The large-scale A/B (200 bench positions + 2000 mined positions)
lives in ``ml/scripts/ab_tss_v59.py`` — too slow for the unit suite.
"""

import json
import os

import numpy as np
import pytest

from gomoku_ml import evaluate, game, kernel_v1, mcts_heuristic, scoring

from test_tss import (
    _brute_open_three_creators,
    _load_mechanism_positions,
    _no_winning_move_positions,
    _random_positions,
)

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "tss_mechanism_positions.json"
)


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


def _player(pos):
    return game.BLACK if pos["to_move"] == "black" else game.WHITE


# --- enumeration A/B -------------------------------------------------------


def test_fast_forcing_sets_match_kernel():
    """Window-scan enumeration == kernel enumeration (fours, threes,
    and the v1 ordering scores on forcing moves) on every
    no-winning-move position — 0 mismatches."""
    mismatches = 0
    for board, player in _no_winning_move_positions():
        fours_k, threes_k, a1 = mcts_heuristic._forcing_sets(board, player)
        fours_f, threes_f, scores = mcts_heuristic._forcing_sets_fast(
            board, player, need_ordering=True
        )
        if fours_f != fours_k or sorted(threes_f) != sorted(threes_k):
            mismatches += 1
            continue
        for m in sorted(fours_k | set(threes_k)):
            # Scores are computed only for members of a class with
            # >= 2 members; singletons read as 0 and cannot change
            # the within-class order — but the VALUE must still match
            # wherever it is computed.
            if m in scores and scores[m] != int(a1.reshape(-1)[m]):
                mismatches += 1
    assert mismatches == 0


def test_fast_four_creators_match_evaluate():
    """The window-scan four-creator enumeration is exactly
    evaluate.four_creating_moves (0 mismatches) — same oracle as
    the kernel version, now on the fast path (and on the L1-ported
    evaluate.py)."""
    mismatches = 0
    for board, player in _no_winning_move_positions():
        fast = sorted(
            mcts_heuristic._four_creating_moves_fast(board, player, fast=True)
        )
        exact = sorted(evaluate.four_creating_moves(board, player))
        if fast != exact:
            mismatches += 1
    assert mismatches == 0


def test_fast_open_three_creators_match_brute_force():
    """The window-scan open-three enumeration matches the run-scan
    oracle (0 mismatches) — same oracle as the kernel version."""
    mismatches = 0
    for board, player in _no_winning_move_positions():
        fast = sorted(
            mcts_heuristic.open_three_creating_moves(board, player, fast=True)
        )
        brute = _brute_open_three_creators(board, player)
        if fast != brute:
            mismatches += 1
    assert mismatches == 0


def test_scalar_ordering_scores_match_kernel():
    """The scalar v1 attack scores (``_score_parts_flat`` on the
    shared flat list) equal ``attack_v1_all`` exactly, on every
    forcing move of every no-winning-move position."""
    mismatches = 0
    checked = 0
    for board, player in _no_winning_move_positions():
        a1 = kernel_v1.attack_v1_all(board, player).reshape(-1)
        flat = board.reshape(-1).tolist()
        moves = mcts_heuristic.tss_forcing_moves(board, player)
        for m in moves:
            scalar = scoring._score_parts_flat(flat, m, player)[0]
            if scalar != int(a1[m]):
                mismatches += 1
            checked += 1
    assert checked > 0
    assert mismatches == 0


# --- run-based winning-move test ------------------------------------------


def test_winning_moves_through_run_matches_full_scan():
    """The run-based winning-move check equals the full scan (and
    the scan-based local check) whenever no winning move pre-existed
    — the TSS precondition, same positions as the existing
    local-vs-full test."""
    rng = np.random.default_rng(99)
    checked = 0
    for board, player in _no_winning_move_positions():
        cands = scoring._candidate_moves(board)
        if not cands:
            continue
        m = int(rng.choice(cands))
        b2 = game.apply_move(board, m, player)
        flat = b2.reshape(-1).tolist()
        full = sorted(scoring._winning_moves(b2, player))
        run = mcts_heuristic._winning_moves_through_run(flat, m, player)
        scan = sorted(mcts_heuristic._winning_moves_through(b2, player, m))
        assert run == full, f"move {(m % 15, m // 15)}: run {run} != full {full}"
        assert run == scan, f"move {(m % 15, m // 15)}: run {run} != scan {scan}"
        # The boolean form agrees with the list form.
        assert mcts_heuristic._creates_four_run(flat, m, player) == bool(full)
        checked += 1
    assert checked >= 40


# --- kernel-blind-spot regressions ----------------------------------------


def test_broken_four_creator_found():
    """v0 is blind to broken-four creators (batch-13: 22 mismatches
    on the first v0-only attempt). Black holds XX_XX with the gap
    empty: playing the gap creates a four. Both paths must find it."""
    board = game.empty_board()
    # Horizontal broken four: (5,7),(6,7),(8,7),(9,7); gap (7,7).
    for x, y in [(5, 7), (6, 7), (8, 7), (9, 7)]:
        board[y, x] = game.BLACK
    board[0, 0] = game.WHITE
    board[14, 14] = game.WHITE
    gap = game.xy_to_move(7, 7)
    assert gap in mcts_heuristic._four_creating_moves_fast(board, game.BLACK)
    assert gap in mcts_heuristic._four_creating_moves_fast(
        board, game.BLACK, fast=True
    )
    assert gap in evaluate.four_creating_moves(board, game.BLACK)


def test_v1_blind_open_three_found():
    """v1 is blind to some open threes (batch-13: an open-three
    creator with v1 attack 405, v0 1021 — the cells two out from
    the run are occupied, so v1's window-based open_ends scores it
    as a mere THREE). The window scan reads the run off the board
    and must find it, as the v0 kernel does."""
    board = game.empty_board()
    # Black run (6,7),(7,7); the creator (8,7) makes _XXX_ with ends
    # (5,7),(9,7) — but the cells two out, (4,7) and (10,7), are
    # occupied by White, which is what blinds v1.
    for x, y in [(6, 7), (7, 7)]:
        board[y, x] = game.BLACK
    for x, y in [(4, 7), (10, 7)]:
        board[y, x] = game.WHITE
    creator = game.xy_to_move(8, 7)
    assert creator in mcts_heuristic.open_three_creating_moves(
        board, game.BLACK
    )
    assert creator in mcts_heuristic.open_three_creating_moves(
        board, game.BLACK, fast=True
    )
    # And v1 really is blind here (the regression's premise).
    assert int(kernel_v1.attack_v1_all(board, game.BLACK).reshape(-1)[creator]) < 1000


# --- TSS A/B ---------------------------------------------------------------


@pytest.mark.parametrize("pos", _load_mechanism_positions())
def test_fast_tss_matches_slow_on_mechanism_positions(pos):
    """Fast and slow TSS return the same move, the same node count,
    and the same budget-exceeded flag on every mechanism position
    (T1–T7) — including the counter-four refutation (None)."""
    board = _board(pos)
    player = _player(pos)
    stats_slow: dict = {}
    stats_fast: dict = {}
    slow = mcts_heuristic.tss_winning_move(
        board, player, stats=stats_slow, fast=False
    )
    fast = mcts_heuristic.tss_winning_move(
        board, player, stats=stats_fast, fast=True
    )
    assert fast == slow, f"{pos['id']}: fast {fast} != slow {slow}"
    assert stats_fast["nodes"] == stats_slow["nodes"], (
        f"{pos['id']}: fast nodes {stats_fast['nodes']} != "
        f"slow nodes {stats_slow['nodes']}"
    )
    assert stats_fast["budget_exceeded"] == stats_slow["budget_exceeded"]


def test_fast_tss_matches_slow_on_mined_positions():
    """Fast and slow TSS agree (move + nodes + budget flag) on
    mined real-game positions — v27-vs-v27 turns, seeds 777/1777
    (Check B's dev seeds). 120 turns keeps the unit suite fast;
    the full 2000-position A/B is ml/scripts/ab_tss_v59.py."""
    from gomoku_ml.mcts_heuristic import tss_winning_move

    positions = []
    g = 0
    while len(positions) < 120:
        pa = scoring.get_player(27, seed=777 + g)
        pb = scoring.get_player(27, seed=1777 + g)
        board = game.empty_board()
        player = game.BLACK
        while len(positions) < 120:
            over, _ = game.game_over(board)
            if over:
                break
            positions.append((board.copy(), player))
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        g += 1
    mismatches = 0
    for board, player in positions:
        s_slow: dict = {}
        s_fast: dict = {}
        slow = tss_winning_move(board, player, stats=s_slow, fast=False)
        fast = tss_winning_move(board, player, stats=s_fast, fast=True)
        if (
            fast != slow
            or s_fast["nodes"] != s_slow["nodes"]
            or s_fast["budget_exceeded"] != s_slow["budget_exceeded"]
        ):
            mismatches += 1
    assert mismatches == 0


# --- v59 registration and engine behaviour --------------------------------


def test_v59_registration():
    """v59 is v52 + tss_fast, nothing else."""
    v52 = scoring.ENGINE_VERSIONS[52]
    v59 = scoring.ENGINE_VERSIONS[59]
    assert v59["tss_root"] is True
    assert v59["tss_fast"] is True
    for key in v52:
        if key == "name":
            continue  # 'scoring-v59' vs 'scoring-v52', by design
        assert v59[key] == v52[key], f"v59[{key!r}] differs from v52"
    p59 = scoring.get_player(59, seed=100)
    assert p59._mcts.tss_root is True
    assert p59._mcts.tss_fast is True


def test_v59_plays_tss_win_on_double_three():
    """v59 plays the TSS win on T1 (same as v52)."""
    pos = next(
        p for p in _load_mechanism_positions()
        if p["id"] == "tss-double-three-black"
    )
    board = _board(pos)
    v59 = scoring.get_player(59, seed=0)
    move = v59.choose_move(board, game.BLACK)
    assert [move % 15, move // 15] == [6, 6]


def test_v59_falls_through_to_v27_when_tss_silent():
    """v59 == v27 on positions where TSS does not fire (same test as
    v52's — TSS consumes no RNG, so the MCTS sees the same seed)."""
    positions = []
    for board, player in _random_positions(seed=777, n=40):
        if mcts_heuristic.tss_winning_move(board, player, fast=True) is None:
            positions.append((board, player))
        if len(positions) >= 5:
            break
    assert len(positions) == 5
    for board, player in positions:
        v59 = scoring.get_player(59, seed=0)
        v27 = scoring.get_player(27, seed=0)
        assert v59.choose_move(board, player) == v27.choose_move(
            board, player
        )


def test_v59_matches_v52_move_for_move():
    """v59 and v52 choose the same move on every turn of two
    recorded v27-vs-v27 games (same seed) — the engine-level
    move-choice-identity check (the full 20-game replay is
    ml/scripts/ab_tss_v59.py --replay)."""
    for g in range(2):
        pa = scoring.get_player(27, seed=777 + g)
        pb = scoring.get_player(27, seed=1777 + g)
        p52 = scoring.get_player(52, seed=100)
        p59 = scoring.get_player(59, seed=100)
        board = game.empty_board()
        player = game.BLACK
        while True:
            over, _ = game.game_over(board)
            if over:
                break
            m52 = p52.choose_move(board, player)
            m59 = p59.choose_move(board, player)
            assert m59 == m52, (
                f"game {g}: v59 {m59} != v52 {m52} "
                f"at {(m52 % 15, m52 // 15)}"
            )
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player

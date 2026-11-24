"""Tests for v58 (batch 14, H2, re-scoped): the threat-potential
term in the leaf evaluation.

Covers the pre-registered design (``research/iterations/iter-58.md``
§11):

* the forcing-move counts match the TSS enumeration
  (:func:`_forcing_sets`) and the brute-force oracles on sampled
  positions (0 mismatches) — the right kernel for each move class
  (batch 13's kernel-blindness lesson: v0 is blind to broken-four
  creators, v1 is blind to some open threes);
* the v1-blind open-three creator is counted by the term but NOT
  by the gradient's strong tier (the term's genuinely new signal);
* the term's values, antisymmetry, and +/-0.05 bound;
* the leaf: terminal identity with v52 (leaf-start terminals and
  rollout terminals), the quiet-path decomposition (v58 == clamp
  (v52 + term)), the +/-0.2 clamp, antisymmetry under colour swap,
  and the must-block skip (rollout-end must-block returns the
  gradient alone);
* the difference gate: the v58 leaf differs from the v52 leaf on
  > 20% of sampled positions (bench: 58.1% — the gate is set well
  below the measured rate);
* the cost ceiling: mean leaf time <= 1.5x v52's (bench: 1.21x);
* v58's registration and TSS-hook precedence (v58 plays the TSS
  move when TSS fires).
"""

import ast
import json
import math
import os
import time

import numpy as np
import pytest

from gomoku_ml import evaluate, game, kernel_numpy, kernel_v1, mcts_heuristic, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BATCH_V52 = os.path.join(
    REPO_ROOT, "research", "iterations", "batch-v52-vs-v27.json"
)

# Pre-registered (§11.4): mean leaf-time ratio ceiling 1.5x (bench
# 1.21x); difference gate > 20% (bench 58.1%).
COST_CEILING = 1.5
MIN_DIFFERENCE_FRACTION = 0.20


def _battery_positions(n_games=10):
    """Every ply position of the first ``n_games`` committed
    v52-vs-v27 battery games (both colours to move as they occur)."""
    with open(BATCH_V52, encoding="utf-8") as f:
        batch = json.load(f)
    out = []
    for rec in batch["records"][:n_games]:
        moves = rec["moves"]
        if isinstance(moves, str):
            moves = ast.literal_eval(moves)
        board = game.empty_board()
        for ply, mv in enumerate(moves):
            player = game.BLACK if ply % 2 == 0 else game.WHITE
            out.append((board.copy(), player))
            board = game.apply_move(board, mv, player)
    return out


def _no_winning_move_positions(n_games=10):
    """Sampled positions where NEITHER side holds a winning move
    (the forcing-count precondition)."""
    out = []
    for board, player in _battery_positions(n_games):
        if scoring._winning_moves(board, game.BLACK):
            continue
        if scoring._winning_moves(board, game.WHITE):
            continue
        out.append((board, player))
    return out


def _brute_open_three_creators(board, player):
    """Brute-force oracle (copied from test_tss.py): for each near
    move, apply it and run-scan the 4 directions through the move
    for a contiguous run of exactly 3 friendly stones with both
    ends empty. Four-creators excluded (same convention as the
    kernel version)."""
    out = []
    fours = set(evaluate.four_creating_moves(board, player))
    for m in scoring._candidate_moves(board):
        if m in fours:
            continue
        b2 = game.apply_move(board, m, player)
        y, x = m // 15, m % 15
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
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


def _rollout_end(board, player):
    """Replay the 2-ply rollout; return (end_board, end_player,
    terminal_or_None). Mirrors _rollout_leaf_value's rollout."""
    value = mcts_heuristic._terminal_leaf_value(board, player)
    if value != 0.0:
        return board, player, value
    cur = board.copy()
    p = player
    for i in range(mcts_heuristic._ROLLOUT_PLIES):
        move, term = mcts_heuristic._rollout_move(cur, p)
        if term is not None:
            return cur, p, (term if i % 2 == 0 else -term)
        cur = game.apply_move(cur, move, p)
        p = -p
    return cur, p, None


# --- enumeration oracles -------------------------------------------------

def test_forcing_counts_match_forcing_sets():
    """_forcing_counts_both (shared v1 call, one v0 call per colour)
    agrees with _forcing_sets' per-colour enumeration on every
    sampled no-winning-move position — the shared-call refactor
    changes cost, not the counts."""
    checked = 0
    for board, _player in _no_winning_move_positions():
        b1, w1 = kernel_v1.attack_v1_both(board)
        fb, fw = mcts_heuristic._forcing_counts_both(board, b1, w1)
        fours_b, threes_b, _ = mcts_heuristic._forcing_sets(board, game.BLACK)
        fours_w, threes_w, _ = mcts_heuristic._forcing_sets(board, game.WHITE)
        assert fb == len(fours_b) + len(threes_b)
        assert fw == len(fours_w) + len(threes_w)
        checked += 1
    assert checked >= 200


def test_forcing_counts_match_brute_force_oracles():
    """The forcing counts match the brute-force oracles directly:
    four-creators == evaluate.four_creating_moves, open-three
    creators == the run-scan oracle — 0 mismatches on sampled
    positions. (The counts are sums, so this test checks the
    per-class split via _forcing_sets against the oracles.)"""
    mismatches = 0
    checked = 0
    for board, _player in _no_winning_move_positions():
        for player in (game.BLACK, game.WHITE):
            fours, threes, _ = mcts_heuristic._forcing_sets(board, player)
            exact_fours = set(evaluate.four_creating_moves(board, player))
            brute_threes = set(_brute_open_three_creators(board, player))
            if fours != exact_fours or set(threes) != brute_threes:
                mismatches += 1
        checked += 1
    assert checked >= 200
    assert mismatches == 0


def test_v1_blind_open_three_counted_but_not_in_gradient():
    """Kernel blindness (batch 13): Black (7,7),(8,7) with White at
    BOTH two-out cells (5,7),(11,7). The move (9,7) creates a TRUE
    open three (ends (6,7),(10,7) empty), but v1's window-based
    open_ends scores it 6 — below even the developing threshold —
    so the gradient's strong tier (v1 >= 4000) and the priors miss
    it entirely. The v0 kernel scores it 1030 (>= 1000), and the
    forcing count includes it: the term is the ONLY signal valuing
    this move."""
    board = game.empty_board()
    for x, y in [(7, 7), (8, 7)]:
        board[y, x] = game.BLACK
    for x, y in [(5, 7), (11, 7), (3, 3), (12, 12)]:
        board[y, x] = game.WHITE
    m = 7 * 15 + 9  # (9,7)
    b1, w1 = kernel_v1.attack_v1_both(board)
    a0 = kernel_numpy.attack_all(board, game.BLACK)
    assert int(b1.flat[m]) < 80  # below the developing threshold
    assert int(a0.flat[m]) >= 1000
    assert m in mcts_heuristic.tss_forcing_moves(board, game.BLACK)
    fb, fw = mcts_heuristic._forcing_counts_both(board, b1, w1)
    assert fb == 1 and fw == 0
    # The gradient's strong tier does NOT count (9,7): Black's
    # breadth counts only v1 >= 4000 moves, and (9,7) scores 6.
    mask = board == game.EMPTY
    strong = int(np.count_nonzero(b1[mask] >= mcts_heuristic._GRADIENT_V1_STRONG))
    assert strong == 0


def test_broken_four_creator_counted():
    """The v0 kernel is blind to broken-four creators (XX_XX) —
    the count must use the v1 prefilter + exact test for fours.
    Black (7,5),(7,6),(7,8): the move (7,9) makes (7,5),(7,6),
    (7,8),(7,9) — four stones with a gap at (7,7), a winning move
    at (7,7) (completing five (7,5)-(7,9)) — so (7,9) IS a
    four-creator, and its v0 score is ~100-200 (the run is split
    by the gap), below the >= 1000 tier."""
    board = game.empty_board()
    for x, y in [(7, 5), (7, 6), (7, 8)]:
        board[y, x] = game.BLACK
    for x, y in [(3, 3), (12, 12)]:
        board[y, x] = game.WHITE
    m = 9 * 15 + 7  # (7,9)
    a0 = kernel_numpy.attack_all(board, game.BLACK)
    assert int(a0.flat[m]) < 1000  # v0-blind: below the >= 1000 tier
    assert m in mcts_heuristic.tss_forcing_moves(board, game.BLACK)
    b1, w1 = kernel_v1.attack_v1_both(board)
    fb, fw = mcts_heuristic._forcing_counts_both(board, b1, w1)
    assert fb >= 1


# --- term unit tests -----------------------------------------------------

def test_threat_term_values():
    """Term values, antisymmetry, and the +/-0.05 bound."""
    term = mcts_heuristic._threat_potential_term
    assert term(0, 0) == 0.0
    assert term(1, 0) == pytest.approx(0.05 * math.tanh(1.0))
    assert term(2, 0) == pytest.approx(0.05 * math.tanh(2.0))
    assert term(0, 1) == pytest.approx(-term(1, 0))
    assert term(3, 1) == pytest.approx(-term(1, 3))
    assert abs(term(100, 0)) <= 0.05
    assert abs(term(0, 100)) <= 0.05
    # Saturation: a difference of 2 is already 96% of the cap.
    assert term(2, 0) > 0.96 * 0.05


# --- leaf tests ----------------------------------------------------------

def test_v58_leaf_terminal_identity_with_v52():
    """On leaf-start terminals (win / loss / must-block) the v58
    leaf is EXACTLY the v52 leaf — terminals are exact, the term
    never touches them."""
    checked = {"win": 0, "loss": 0, "must_block": 0}
    for board, player in _battery_positions():
        v52 = mcts_heuristic._terminal_leaf_value(board, player)
        if v52 == 0.0:
            continue
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        assert v58 == v52
        if v52 == 1.0:
            checked["win"] += 1
        elif v52 == -1.0:
            checked["loss"] += 1
        else:
            checked["must_block"] += 1
    assert checked["win"] >= 1
    assert checked["must_block"] >= 1


def test_v58_leaf_rollout_terminal_identity_with_v52():
    """On positions where the rollout reaches a terminal (|v52 leaf|
    == 1.0 via the rollout, not the leaf start), the v58 leaf is
    EXACTLY the v52 leaf."""
    checked = 0
    for board, player in _battery_positions():
        if mcts_heuristic._terminal_leaf_value(board, player) != 0.0:
            continue
        v52 = mcts_heuristic._rollout_leaf_value(board, player)
        if abs(v52) != 1.0:
            continue
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        assert v58 == v52
        checked += 1
    assert checked >= 1


def test_v58_leaf_quiet_path_is_gradient_plus_term():
    """On quiet rollout ends, the v58 leaf is exactly clamp(v52's
    gradient + the independently computed term, +/-0.2) — the
    spec's decomposition, verified against a from-scratch replay
    of the rollout and the forcing counts."""
    checked = 0
    for board, player in _battery_positions():
        end_board, end_player, term = _rollout_end(board, player)
        if term is not None:
            continue
        if scoring._winning_moves(end_board, end_player):
            continue
        opp_wins = scoring._winning_moves(end_board, -end_player)
        if len(opp_wins) >= 1:
            continue  # terminal or must-block: covered elsewhere
        v52 = mcts_heuristic._rollout_leaf_value(board, player)
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        grad = mcts_heuristic._attack_gradient_v1(end_board, end_player)
        b1, w1 = kernel_v1.attack_v1_both(end_board)
        fb, fw = mcts_heuristic._forcing_counts_both(end_board, b1, w1)
        f_ours, f_theirs = (
            (fb, fw) if end_player == game.BLACK else (fw, fb)
        )
        t = mcts_heuristic._threat_potential_term(f_ours, f_theirs)
        expected = max(-0.2, min(0.2, grad + t))
        assert v52 == pytest.approx(grad), "v52 leaf is the gradient here"
        assert v58 == pytest.approx(expected)
        checked += 1
    assert checked >= 100


def test_v58_leaf_must_block_skips_term():
    """On rollout-end must-blocks (opponent holds exactly 1 winning
    move), the v58 leaf is EXACTLY the v52 leaf — the term is
    skipped there (the opponent's forcing count is degenerate)."""
    checked = 0
    for board, player in _battery_positions():
        end_board, end_player, term = _rollout_end(board, player)
        if term is not None:
            continue
        if scoring._winning_moves(end_board, end_player):
            continue
        opp_wins = scoring._winning_moves(end_board, -end_player)
        if len(opp_wins) != 1:
            continue
        v52 = mcts_heuristic._rollout_leaf_value(board, player)
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        assert v58 == v52
        checked += 1
    assert checked >= 5


def test_v58_leaf_clamped_and_bounded():
    """Quiet v58 leaves stay in [-0.2, 0.2] (the iter-7 cap
    invariant — below the must-block 0.25); all leaves in [-1, 1]."""
    n_quiet = 0
    for board, player in _battery_positions():
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        assert -1.0 <= v58 <= 1.0
        if mcts_heuristic._terminal_leaf_value(board, player) == 0.0:
            end_board, end_player, term = _rollout_end(board, player)
            if term is None and not scoring._winning_moves(
                end_board, end_player
            ):
                if len(scoring._winning_moves(end_board, -end_player)) < 2:
                    assert -0.2 <= v58 <= 0.2
                    n_quiet += 1
    assert n_quiet >= 100


def test_v58_leaf_colour_swap_symmetric():
    """Leaf symmetry under simultaneous board+player colour swap:
    leaf(board, BLACK) == leaf(-board, WHITE) on sampled positions
    — White on the colour-swapped board is in exactly Black's
    position (the rollout mirrors, and the gradient and the term
    are both antisymmetric under a player swap on the SAME board,
    so the mirrored rollout end evaluates identically from the
    respective side's perspective). NOTE: leaf(board, BLACK) ==
    -leaf(board, WHITE) does NOT hold for a rollout leaf — the
    rollout path depends on who moves first — so the colour-swap
    symmetry is the correct invariant, not player-swap
    antisymmetry on one board."""
    checked = 0
    for board, player in _battery_positions():
        if player != game.BLACK:
            continue
        v_black = mcts_heuristic._rollout_threat_leaf_value(board, game.BLACK)
        v_white = mcts_heuristic._rollout_threat_leaf_value(
            -board, game.WHITE
        )
        assert v_black == pytest.approx(v_white)
        checked += 1
        if checked >= 50:
            break
    assert checked >= 50


def test_v58_leaf_differs_from_v52():
    """Difference gate: the v58 leaf differs from the v52 leaf on
    > 20% of sampled positions (bench: 58.1%). If the term never
    changed the leaf value, the mechanism would be dead on
    arrival."""
    positions = _battery_positions()
    n_diff = 0
    for board, player in positions:
        v52 = mcts_heuristic._rollout_leaf_value(board, player)
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        if abs(v52 - v58) > 1e-9:
            n_diff += 1
    assert n_diff / len(positions) > MIN_DIFFERENCE_FRACTION, (
        f"difference rate {n_diff / len(positions):.3f} below gate "
        f"{MIN_DIFFERENCE_FRACTION}"
    )


def test_v58_leaf_cost_within_ceiling():
    """Cost ceiling: mean v58 leaf time <= 1.5x v52's (bench:
    1.21x). Timed on the same sampled positions, same process."""
    positions = _battery_positions()
    t52, t58 = [], []
    for board, player in positions:
        t0 = time.perf_counter()
        mcts_heuristic._rollout_leaf_value(board, player)
        t52.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        mcts_heuristic._rollout_threat_leaf_value(board, player)
        t58.append(time.perf_counter() - t0)
    ratio = float(np.mean(t58)) / float(np.mean(t52))
    assert ratio <= COST_CEILING, f"leaf cost ratio {ratio:.3f}x > {COST_CEILING}x"


# --- engine tests --------------------------------------------------------

def test_v58_registration():
    """v58 is v52 with the threat leaf: same sims, c_puct, priors,
    tie-break, TSS root hook — only leaf_mode differs."""
    cfg52 = scoring.ENGINE_VERSIONS[52]
    cfg58 = scoring.ENGINE_VERSIONS[58]
    assert cfg58["name"] == "scoring-v58"
    for key in ("mcts", "mcts_sims", "tie_break_q",
                "tie_break_q_threshold", "tss_root"):
        assert cfg58[key] == cfg52[key], key
    assert cfg58["leaf_mode"] == "rollout2_threat_v58"
    assert cfg52["leaf_mode"] == "rollout2_v1_both"
    player = scoring.get_player(58, seed=100)
    assert player._mcts.leaf_mode == "rollout2_threat_v58"
    assert player._mcts.tss_root is True


def test_v58_plays_tss_win():
    """TSS-hook precedence: on the double-three mechanism position
    (T1), v58 plays the TSS winning move (6,6), exactly as v52
    does — the leaf term never overrides the root hook."""
    board = game.empty_board()
    for x, y in [(7, 6), (8, 6), (6, 7), (6, 8)]:
        board[y, x] = game.BLACK
    for x, y in [(3, 3), (11, 11), (12, 3)]:
        board[y, x] = game.WHITE
    assert mcts_heuristic.tss_winning_move(board, game.BLACK) == 6 * 15 + 6
    e58 = scoring.get_player(58, seed=100)
    assert e58.choose_move(board.copy(), game.BLACK) == 6 * 15 + 6
    e52 = scoring.get_player(52, seed=100)
    assert e52.choose_move(board.copy(), game.BLACK) == 6 * 15 + 6

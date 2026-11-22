"""Tests for v53 (batch 13, H3).

Two parts, matching the two halves of ``research/iterations/iter-53.md``:

* **2-ply differential tests (§§0–5, regression).** v53 was
  originally specified as a 2-ply rollout leaf; the premise failed
  (v18/v27's leaf is ALREADY 2-ply) and the fresh-from-spec
  implementation coincides with v27's leaf everywhere sampled.
  These tests pin that identity finding: leaf-value identity on
  >= 2000 sampled positions, move-choice identity on >= 30 full
  100-sim searches, cost ratio under the 1.5x ceiling. They run
  against ``_rollout2ply_search_leaf_value`` / ``leaf_mode=
  "rollout2ply_v53"`` directly — NOT through the v53 registry
  entry, which now points at the 3-ply leaf.
* **3-ply tests (§6, the re-scoped v53).** The pre-registered
  mechanism check for the 3-ply rollout leaf: leaf-value
  distribution vs the 2-ply leaf (sanity + difference gates),
  rollout-end parity handling (spec replay + win-rule grounding),
  and the cost ceiling (2.0x, expected ~1.5x). The move-choice
  agreement check (§6.3b) is here too, as characterisation.
"""

import ast
import json
import math
import os
import time

import numpy as np
import pytest

from gomoku_ml import evaluate, game, mcts_heuristic, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BATCH_V27 = os.path.join(
    REPO_ROOT, "research", "iterations", "batch-v27-vs-v7.json"
)

# Pre-registered (§1.3): exact leaf-value equality on >= 2000
# sampled positions, move-choice identity on >= 30 searches,
# cost ratio under the 1.5x ceiling.
MIN_LEAF_POSITIONS = 2000
MIN_SEARCH_POSITIONS = 30
COST_CEILING_2PLY = 1.5
# Pre-registered (§6.2/§6.3): 3-ply cost ceiling 2.0x, and the
# difference gate — the 3-ply leaf must differ from the 2-ply
# leaf on > 5% of sampled positions.
COST_CEILING_3PLY = 2.0
MIN_DIFFERENCE_FRACTION = 0.05


def _to_move(pos) -> int:
    tm = pos["to_move"]
    if isinstance(tm, str):
        if tm in ("1", "-1"):
            return int(tm)
        return game.BLACK if tm == "black" else game.WHITE
    return int(tm)


def _board_from_stones(pos):
    return evaluate.board_from_position(pos)


def _battery_positions(n_games=10):
    """Every ply position of the first ``n_games`` committed
    v27-vs-v7 battery games (both colours to move as they occur)."""
    with open(BATCH_V27, encoding="utf-8") as f:
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


def _tactical_positions():
    """All old-suite (20) + hard-suite (29) tactical positions."""
    out = []
    for path in (
        os.path.join(HERE, "..", "gomoku_ml", "tactical_positions.json"),
        os.path.join(
            HERE, "..", "gomoku_ml", "tactical_positions_hard.json"
        ),
    ):
        with open(path, encoding="utf-8") as f:
            suite = json.load(f)
        for pos in suite["positions"]:
            out.append((_board_from_stones(pos), _to_move(pos)))
    return out


def _random_playout_positions(n_games=30, seed=53):
    """Seeded random-playout positions at every 2nd ply, from the
    empty board — quiet, terminal, must-block and creator-rich
    positions alike. Random play ends games early only rarely;
    playouts stop at a win or a full board."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_games):
        board = game.empty_board()
        player = game.BLACK
        ply = 0
        while True:
            if ply % 2 == 0:
                out.append((board.copy(), player))
            empty = np.flatnonzero(board == game.EMPTY)
            if len(empty) == 0:
                break
            mv = int(rng.choice(empty))
            board = game.apply_move(board, mv, player)
            if game.is_winning_stone(board, mv, player):
                break
            player = -player
            ply += 1
    return out


def _all_leaf_positions():
    positions = (
        _battery_positions()
        + _tactical_positions()
        + _random_playout_positions()
    )
    assert len(positions) >= MIN_LEAF_POSITIONS, (
        f"sample too small: {len(positions)} < {MIN_LEAF_POSITIONS}"
    )
    return positions


def _time_leaf(fn, board, player):
    t0 = time.perf_counter()
    fn(board, player)
    return time.perf_counter() - t0


# --- 2-ply differential tests (§§0–5): regression --------------------------
# These pin the identity finding. They run against the 2-ply leaf
# function / leaf mode directly, NOT through ENGINE_VERSIONS[53],
# which now points at the 3-ply leaf (§6).


def test_v53_leaf_value_identical_to_v27_leaf():
    """Pre-registered check (§1.3a): exact leaf-value equality on
    >= 2000 sampled positions (battery + tactical + random
    playouts, both colours to move)."""
    positions = _all_leaf_positions()
    mismatches = []
    for board, player in positions:
        a = mcts_heuristic._rollout2ply_search_leaf_value(board, player)
        b = mcts_heuristic._rollout_leaf_value(board, player)
        if a != b:
            mismatches.append((player, a, b))
    assert not mismatches, (
        f"{len(mismatches)}/{len(positions)} leaf mismatches "
        f"(first 5: {mismatches[:5]})"
    )


def test_v53_move_choice_identical_to_v27():
    """Pre-registered check (§1.3b): same chosen move on >= 30 full
    100-sim searches at identical seeds (tactical + sampled
    midgames, both colours). Built on direct HeuristicMCTS
    construction with the two leaf modes — the v53 registry entry
    now points at the 3-ply leaf, so get_player(53) no longer
    builds the 2-ply engine."""
    tactical = _tactical_positions()
    battery = _battery_positions()
    # 10 old + 10 hard tactical, 10 midgames (ply >= 10) — 30 total.
    midgames = [
        (b, p) for (b, p) in battery if int((b != game.EMPTY).sum()) >= 10
    ]
    rng = np.random.default_rng(53)
    picks = tactical[:10] + tactical[20:30]
    picks += [midgames[i] for i in rng.choice(len(midgames), 10, replace=False)]
    assert len(picks) >= MIN_SEARCH_POSITIONS
    mismatches = []
    for i, (board, player) in enumerate(picks):
        seed = 100 + i
        m2ply = mcts_heuristic.HeuristicMCTS(
            sims=100, c_puct=1.5, leaf_mode="rollout2ply_v53",
            tie_break_q=True, tie_break_q_threshold=0.05,
            rng=np.random.default_rng(seed),
        ).choose_move(board, player)
        m27 = mcts_heuristic.HeuristicMCTS(
            sims=100, c_puct=1.5, leaf_mode="rollout2_v1_both",
            tie_break_q=True, tie_break_q_threshold=0.05,
            rng=np.random.default_rng(seed),
        ).choose_move(board, player)
        if m2ply != m27:
            mismatches.append((i, player, m2ply, m27))
    assert not mismatches, f"move-choice mismatches: {mismatches}"


def test_v53_leaf_cost_within_ceiling():
    """Pre-registered check (§1.3c): mean leaf cost <= 1.5x v27's on
    200 quiet midgame positions (same positions, both leaves,
    each timed individually; min-of-3 per position to cut VM
    noise). Identical work is expected to land ~1.0x."""
    battery = _battery_positions()
    quiet = []
    for board, player in battery:
        if mcts_heuristic._terminal_leaf_value(board, player) == 0.0:
            quiet.append((board, player))
        if len(quiet) >= 200:
            break
    assert len(quiet) == 200
    t53 = t27 = 0.0
    for board, player in quiet:
        best53 = min(
            (
                _time_leaf(
                    mcts_heuristic._rollout2ply_search_leaf_value,
                    board,
                    player,
                )
                for _ in range(3)
            )
        )
        best27 = min(
            (
                _time_leaf(mcts_heuristic._rollout_leaf_value, board, player)
                for _ in range(3)
            )
        )
        t53 += best53
        t27 += best27
    ratio = t53 / t27
    assert ratio < COST_CEILING_2PLY, (
        f"leaf cost ratio {ratio:.3f} >= {COST_CEILING_2PLY}"
    )
    # Recorded for the iteration log; the assertion above is the gate.
    print(f"\nv53 2-ply leaf cost: {t53:.4f}s vs v27 {t27:.4f}s over "
          f"200 positions (ratio {ratio:.3f})")


# --- 3-ply tests (§6): the re-scoped v53 ------------------------------------


def test_v53_registered_like_v27():
    """v53 (re-scoped, §6) is v27 + the 3-ply rollout leaf:
    everything but the leaf mode is v27 unchanged."""
    assert 53 in scoring.ENGINE_VERSIONS
    v53 = scoring.ENGINE_VERSIONS[53]
    v27 = scoring.ENGINE_VERSIONS[27]
    assert v53["name"] == "scoring-v53"
    assert v53["mcts"] is True
    assert v53["mcts_sims"] == 100
    assert v53["leaf_mode"] == "rollout3ply_v53"
    # Everything but the leaf mode is v27 unchanged.
    for key in (
        "defence_weight",
        "threat_check",
        "tie_break_q",
        "tie_break_q_threshold",
    ):
        assert v53[key] == v27[key], key
    player = scoring.get_player(53, seed=0)
    assert player._mcts.leaf_mode == "rollout3ply_v53"
    assert player._mcts.tie_break_q is True
    assert player._mcts.tie_break_q_threshold == 0.05


def _replay_3ply(board, player):
    """Independent transcription of the §6.1 spec, for the
    parity test: replay the 3-ply rollout (using the shared
    ``_rollout_move`` policy) and return ``(outcome, value, b3)``
    where ``outcome`` is one of 'start-terminal', 'ply1', 'ply2',
    'ply3', 'end-loss', 'end-win', 'end-gradient', ``value`` is
    the expected leaf value from the spec, and ``b3`` is the
    rollout-end board (None unless the rollout reaches its end).
    The terminal branches are grounded in the game's win rule by
    the caller (``game.is_winning_stone``), not just by
    ``scoring._winning_moves``."""
    value = mcts_heuristic._terminal_leaf_value(board, player)
    if value != 0.0:
        return "start-terminal", value, None
    m1, term1 = mcts_heuristic._rollout_move(board, player)
    if term1 is not None:
        return "ply1", term1, None
    b1 = game.apply_move(board, m1, player)
    m2, term2 = mcts_heuristic._rollout_move(b1, -player)
    if term2 is not None:
        return "ply2", -term2, None
    b2 = game.apply_move(b1, m2, -player)
    m3, term3 = mcts_heuristic._rollout_move(b2, player)
    if term3 is not None:
        return "ply3", term3, None
    b3 = game.apply_move(b2, m3, player)
    if scoring._winning_moves(b3, -player):
        return "end-loss", -1.0, b3
    if len(scoring._winning_moves(b3, player)) >= 2:
        return "end-win", 1.0, b3
    return (
        "end-gradient",
        -mcts_heuristic._attack_gradient_v1(b3, -player),
        b3,
    )


def test_v53_3ply_leaf_matches_spec_replay():
    """§6.1 parity handling: the implementation agrees with an
    independent transcription of the pre-registered spec on every
    sampled position, every REACHABLE rollout branch is exercised
    by the sample (a branch with zero instances would leave the
    test vacuous for that branch), and the terminal branches are
    grounded in the game's win rule (``is_winning_stone``), not
    just in ``_winning_moves``. The gradient antisymmetry the
    spec relies on (``-_attack_gradient_v1(b3, -player)`` equals
    ``_attack_gradient_v1(b3, player)``) is asserted on every
    rollout-end position.

    The 'end-loss' branch (opponent has a winning move at the
    rollout end) is NOT required: it is UNREACHABLE under the
    rollout policy, and the test asserts the property that makes
    it so. Proof: at b2 (before ply 3), ``_rollout_move`` ends the
    rollout if the opponent has >= 2 winning moves (term3 = -1.0)
    and blocks the single winning move if the opponent has exactly
    1; and our ply-3 stone cannot CREATE an opponent winning move
    (it adds no opponent stones and occupies one square — a
    winning move needs an empty square completing five opponent
    stones). So at b3 the opponent has no winning move whenever
    the rollout reaches its end. The check stays in the
    implementation (it is in the pre-registered spec, mirrors the
    2-ply leaf's rollout-end checks, and guards against future
    policy changes) — as dead code, documented here. (The same
    argument shows the 2-ply leaf's rollout-end +1.0 check is
    dead too: the opponent's ply-2 move cannot create OUR winning
    moves, and our single threat is blocked by ply 2.)"""
    positions = _all_leaf_positions()
    counts = {}
    mismatches = []
    for board, player in positions:
        outcome, expected, b3 = _replay_3ply(board, player)
        counts[outcome] = counts.get(outcome, 0) + 1
        got = mcts_heuristic._rollout3ply_search_leaf_value(board, player)
        if got != expected:
            mismatches.append((player, outcome, got, expected))
        if outcome in ("end-gradient", "end-win"):
            # Unreachability of 'end-loss': the opponent has no
            # winning move at the rollout end (see docstring).
            assert scoring._winning_moves(b3, -player) == []
        if outcome == "end-gradient":
            # Antisymmetry: the negated opponent-perspective
            # gradient equals the player-perspective gradient
            # (tolerance, not exact equality: math.tanh is
            # odd up to a possible last-ulp difference).
            assert abs(
                got - mcts_heuristic._attack_gradient_v1(b3, player)
            ) < 1e-12
        elif outcome == "end-loss":
            # Unreachable (see docstring) — but if the policy ever
            # changes so it becomes reachable, ground it in the
            # win rule like the other terminal branches.
            w = scoring._winning_moves(b3, -player)[0]
            assert game.is_winning_stone(
                game.apply_move(b3, w, -player), w, -player
            )
        elif outcome == "end-win":
            # Each of our two winning moves on b3 really wins.
            for w in scoring._winning_moves(b3, player)[:2]:
                assert game.is_winning_stone(
                    game.apply_move(b3, w, player), w, player
                )
    assert not mismatches, (
        f"{len(mismatches)}/{len(positions)} spec-replay mismatches "
        f"(first 5: {mismatches[:5]})"
    )
    # Every REACHABLE branch of the spec must be exercised, or the
    # test is vacuous for the unexercised branch. 'end-loss' is
    # excluded: unreachable (see docstring), and its
    # unreachability is asserted above on every rollout-end
    # position instead.
    for branch in ("ply3", "end-win", "end-gradient"):
        assert counts.get(branch, 0) > 0, (
            f"no {branch} instances in the sample — test vacuous "
            f"for that branch (counts: {counts})"
        )
    assert counts.get("end-loss", 0) == 0, (
        f"end-loss reached {counts['end-loss']} times — the "
        "unreachability proof is wrong (or the policy changed)"
    )
    print(f"\n3-ply rollout outcome counts over {len(positions)} "
          f"positions: {counts}")


def test_v53_3ply_leaf_distribution_vs_2ply():
    """Pre-registered check (§6.3a): leaf-value distribution of
    the 3-ply leaf vs the 2-ply leaf on the same >= 2000 sampled
    positions. Sanity gate (blocks eval): every 3-ply value finite
    and in [-1, 1]. Difference gate (blocks eval): the 3-ply leaf
    differs from the 2-ply leaf on > 5% of positions. The
    parity-shift question (mean difference vs +0.05) is reported
    for the iteration log — it is a finding, not a gate."""
    positions = _all_leaf_positions()
    v3, v2 = [], []
    diff = 0
    for board, player in positions:
        a = mcts_heuristic._rollout3ply_search_leaf_value(board, player)
        b = mcts_heuristic._rollout_leaf_value(board, player)
        assert math.isfinite(a) and -1.0 <= a <= 1.0, (
            f"3-ply leaf value out of range: {a} (player {player})"
        )
        v3.append(a)
        v2.append(b)
        if a != b:
            diff += 1
    n = len(positions)
    frac_diff = diff / n
    assert frac_diff > MIN_DIFFERENCE_FRACTION, (
        f"3-ply leaf differs from 2-ply on only {diff}/{n} "
        f"({frac_diff:.4f} <= {MIN_DIFFERENCE_FRACTION}) — the "
        "implementation may not actually roll out 3 plies"
    )

    def _stats(vals):
        mean = sum(vals) / len(vals)
        frac_pos = sum(1 for v in vals if v > 0) / len(vals)
        frac_p1 = sum(1 for v in vals if v == 1.0) / len(vals)
        frac_m1 = sum(1 for v in vals if v == -1.0) / len(vals)
        frac_mb = sum(1 for v in vals if v == -0.25) / len(vals)
        frac_grad = sum(
            1 for v in vals if v not in (1.0, -1.0, -0.25, 0.0)
        ) / len(vals)
        return mean, frac_pos, frac_p1, frac_m1, frac_mb, frac_grad

    s3, s2 = _stats(v3), _stats(v2)
    print(
        "\n3-ply vs 2-ply leaf distribution over "
        f"{n} positions (mean, frac>0, frac+1, frac-1, "
        "frac-0.25, frac-gradient):\n"
        f"  3-ply: {tuple(round(x, 4) for x in s3)}\n"
        f"  2-ply: {tuple(round(x, 4) for x in s2)}\n"
        f"  mean difference (3-ply minus 2-ply): {s3[0] - s2[0]:+.4f} "
        f"(parity-shift threshold +0.05)\n"
        f"  positions differing: {diff}/{n} ({frac_diff:.4f})"
    )


def test_v53_3ply_leaf_cost_within_ceiling():
    """Pre-registered check (§6.3c / §6.2): mean leaf cost <= 2.0x
    v27's on 200 quiet midgame positions (same bench as the 2-ply
    test; min-of-3 per position). Expected ~1.5x (3 rollout plies
    vs 2)."""
    battery = _battery_positions()
    quiet = []
    for board, player in battery:
        if mcts_heuristic._terminal_leaf_value(board, player) == 0.0:
            quiet.append((board, player))
        if len(quiet) >= 200:
            break
    assert len(quiet) == 200
    t53 = t27 = 0.0
    for board, player in quiet:
        best53 = min(
            (
                _time_leaf(
                    mcts_heuristic._rollout3ply_search_leaf_value,
                    board,
                    player,
                )
                for _ in range(3)
            )
        )
        best27 = min(
            (
                _time_leaf(mcts_heuristic._rollout_leaf_value, board, player)
                for _ in range(3)
            )
        )
        t53 += best53
        t27 += best27
    ratio = t53 / t27
    assert ratio < COST_CEILING_3PLY, (
        f"3-ply leaf cost ratio {ratio:.3f} >= {COST_CEILING_3PLY}"
    )
    print(f"\nv53 3-ply leaf cost: {t53:.4f}s vs v27 {t27:.4f}s over "
          f"200 positions (ratio {ratio:.3f}, expected ~1.5)")


def test_v53_3ply_move_choice_agreement_vs_v27():
    """Pre-registered check (§6.3b): move-choice agreement vs v27
    on all 49 tactical positions — full 100-sim searches at
    identical seeds (100–148). Characterisation, not a gate:
    >=45/49 means tactical behaviour is preserved; <45/49 means
    it changed and clause 3 is at risk. The agreement count is
    reported for the iteration log."""
    tactical = _tactical_positions()
    assert len(tactical) == 49
    agree = 0
    disagreements = []
    for i, (board, player) in enumerate(tactical):
        seed = 100 + i
        m53 = scoring.get_player(53, seed=seed).choose_move(board, player)
        m27 = scoring.get_player(27, seed=seed).choose_move(board, player)
        if m53 == m27:
            agree += 1
        else:
            disagreements.append((i, player, m53, m27))
    print(
        f"\nv53 (3-ply) vs v27 move-choice agreement on tactical "
        f"positions: {agree}/49 "
        f"(>=45/49 = behaviour preserved; disagreements: "
        f"{disagreements})"
    )

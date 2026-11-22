"""Differential tests for v53 (batch 13, H3) — v27 + a 2-ply rollout
leaf implemented fresh from the pre-registered spec
(``research/iterations/iter-53.md`` §1.1).

Premise (iter-53 §0): v18/v27's leaf is ALREADY a 2-ply rollout
leaf (``_ROLLOUT_PLIES = 2``), so the specified leaf is expected
to coincide with v27's. These tests are the pre-registered
mechanism check (§1.3): leaf-value identity on >= 2000 sampled
positions, move-choice identity on >= 30 full 100-sim searches
at identical seeds, and the cost ratio under the 1.5x ceiling.
If all pass, v53 is behaviourally identical to v27 and the
battery is vacated (a v27-vs-v27 battery measures 0.50 by
construction). If any fails, the difference is the version's
actual content and the battery runs.
"""

import ast
import json
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
COST_CEILING = 1.5


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


def test_v53_registered_like_v27():
    assert 53 in scoring.ENGINE_VERSIONS
    v53 = scoring.ENGINE_VERSIONS[53]
    v27 = scoring.ENGINE_VERSIONS[27]
    assert v53["name"] == "scoring-v53"
    assert v53["mcts"] is True
    assert v53["mcts_sims"] == 100
    assert v53["leaf_mode"] == "rollout2ply_v53"
    # Everything but the leaf mode is v27 unchanged.
    for key in (
        "defence_weight",
        "threat_check",
        "tie_break_q",
        "tie_break_q_threshold",
    ):
        assert v53[key] == v27[key], key
    player = scoring.get_player(53, seed=0)
    assert player._mcts.leaf_mode == "rollout2ply_v53"
    assert player._mcts.tie_break_q is True
    assert player._mcts.tie_break_q_threshold == 0.05


def test_v53_leaf_value_identical_to_v27_leaf():
    """Pre-registered check (a): exact leaf-value equality on
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
    """Pre-registered check (b): same chosen move on >= 30 full
    100-sim searches at identical seeds (tactical + sampled
    midgames, both colours)."""
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
        m53 = scoring.get_player(53, seed=seed).choose_move(board, player)
        m27 = scoring.get_player(27, seed=seed).choose_move(board, player)
        if m53 != m27:
            mismatches.append((i, player, m53, m27))
    assert not mismatches, f"move-choice mismatches: {mismatches}"


def test_v53_leaf_cost_within_ceiling():
    """Pre-registered check (c): mean leaf cost <= 1.5x v27's on
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
    assert ratio < COST_CEILING, f"leaf cost ratio {ratio:.3f} >= {COST_CEILING}"
    # Recorded for the iteration log; the assertion above is the gate.
    print(f"\nv53 leaf cost: {t53:.4f}s vs v27 {t27:.4f}s over 200 positions "
          f"(ratio {ratio:.3f})")


def _time_leaf(fn, board, player):
    t0 = time.perf_counter()
    fn(board, player)
    return time.perf_counter() - t0

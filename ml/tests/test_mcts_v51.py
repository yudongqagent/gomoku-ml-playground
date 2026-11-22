"""Tests for v51 (batch 13, H1): soft-min backup at opponent nodes.

The mechanism is iter-40 §1.1a's blunder dilution, as v47
attempted it: the MCTS mean backup mixes the opponent's best
reply with their blunders, so a move that loses at once can rank
above safe moves. v51 switches opponent nodes (odd depth from the
root) to a SOFT-MIN backup — q is the softmax-weighted mean of the
opponent-perspective values of the node's visited children
(``-child.q``), weights ``exp(beta * value)`` with beta = 2.0 —
while our nodes keep the mean. The soft-min weights the worst
reply heavily but keeps the other replies' contribution, so it
should price the opponent's best reply without collapsing the
"how it loses" gradient the way v47's pure minimax did. See
research/iterations/iter-51.md §1.2 for the pre-registered rule.

The mechanism check (iter-51.md §1.4) FAILED: the soft-min
collapses exactly like minimax on three of the four positions,
because at 100 sims a root child has 0-1 visited children and a
softmax over one child IS that child's value — and the soft-min
discards the node's own expansion leaf, which is where the mean
backup's prevention gradient actually lives. The tests below
encode the OBSERVED behaviour (as iter-47's did for v47), not the
pre-registered prediction.

Fixtures: the two diagnosed dilution instances, reconstructed
from the v34 battery JSONs on research/batch-10-h2 (move lists
below; coordinates in iter-40 are (col, row), flat = row * 15 +
col), plus the two prevention positions from the hard tactical
suite (ml/gomoku_ml/tactical_positions_hard.json).
"""

import math

import numpy as np

from gomoku_ml import evaluate, game, scoring
from gomoku_ml.mcts_heuristic import (
    HeuristicMCTS,
    _Node,
    _softmax_mean,
)

# g61: batch-v34-vs-v27.json game 61, ply 21 (White to move; the
# played move, flat 23 = (8,1), is NOT in the list — it is the move
# under test). v34-as-White.
_G61_MOVES = [
    112, 96, 81, 84, 67, 127, 95, 109, 53, 39,
    54, 50, 52, 55, 82, 97, 37, 22, 69, 21, 80,
]
_G61_UNSAFE = 23  # (8,1)
_G61_SAFES = [78, 79, 83]  # (3,5), (4,5), (8,5)

# g43: batch-v34-vs-v27-swap.json game 43, ply 17 (White to move;
# the played move, flat 127 = (7,8), is NOT in the list). The
# opening is (black1, white1, black2); White moves first after it.
_G43_MOVES = [
    112, 144, 176, 84, 129, 85, 83, 143, 145, 113,
    173, 174, 131, 159, 130, 189, 204,
]
_G43_UNSAFE = 127  # (7,8)
_G43_SAFES = [128, 132]  # (8,8), (12,8)

# prev-v1v0s-g13: Black to move; expected preventions (2,3), (3,3),
# (10,7), (11,7). prev-v1bl-g2: Black to move; expected preventions
# (6,5), (7,5). Loaded from the hard suite JSON (the corrected
# suite), so the fixtures track the suite.
_HARD_PATH = evaluate.__file__.replace(
    "evaluate.py", "tactical_positions_hard.json"
)


def _board_normal(moves):
    board = game.empty_board()
    for i, m in enumerate(moves):
        player = game.BLACK if i % 2 == 0 else game.WHITE
        board = game.apply_move(board, m, player)
    return board


def _board_swap(moves):
    board = game.empty_board()
    for i, m in enumerate(moves):
        if i == 0 or i == 2:
            player = game.BLACK
        elif i == 1:
            player = game.WHITE
        else:
            player = game.WHITE if i % 2 == 1 else game.BLACK
        board = game.apply_move(board, m, player)
    return board


def _tactical_position(pid):
    suite = evaluate.load_tactical_suite(_HARD_PATH)
    pos = next(p for p in suite["positions"] if p["id"] == pid)
    board = evaluate.board_from_position(pos)
    expect = [game.xy_to_move(x, y) for x, y in pos["expect"]["moves"]]
    return board, pos["to_move"], expect


def _v27_engine():
    return HeuristicMCTS(
        sims=100, c_puct=1.5, rng=np.random.default_rng(0),
        leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05,
    )


def _v51_engine():
    return HeuristicMCTS(
        sims=100, c_puct=1.5, rng=np.random.default_rng(0),
        leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05, backup_rule="softmax-opp",
        softmax_beta=2.0,
    )


def _run_search(engine, board, player):
    root = _Node(1.0)
    for _ in range(engine.sims):
        engine._run_one(root, board, player, None)
    return root


def test_v51_registered():
    player = scoring.get_player(51, seed=0)
    assert player.name == "scoring-v51"
    assert player._mcts.backup_rule == "softmax-opp"
    assert player._mcts.softmax_beta == 2.0
    assert player._mcts.leaf_mode == "rollout2_v1_both"
    assert player._mcts.tie_break_q is True
    assert player._mcts.tie_break_q_threshold == 0.05


def test_softmax_mean_formula():
    # Analytic values: beta = 2, values {+1, -1, -1} (the g61
    # unsafe's subtree shape: one creator child, two blunder
    # children) -> (e^2 - 2 e^-2) / (e^2 + 2 e^-2).
    got = _softmax_mean([1.0, -1.0, -1.0], 2.0)
    expect = (math.exp(2.0) - 2 * math.exp(-2.0)) / (
        math.exp(2.0) + 2 * math.exp(-2.0)
    )
    assert got == expect
    assert abs(got - 0.9293263) < 1e-6
    # One value: no smoothing, returns the value exactly.
    assert _softmax_mean([-0.5], 2.0) == -0.5
    # beta -> 0 tends to the plain mean.
    assert abs(_softmax_mean([1.0, 0.0], 1e-9) - 0.5) < 1e-6
    # Large beta tends to the max.
    assert abs(_softmax_mean([1.0, 0.0], 50.0) - 1.0) < 1e-9
    # Stays inside [min, max]; no overflow at extreme inputs.
    assert _softmax_mean([1000.0, -1000.0], 2.0) == 1000.0


def test_node_q_softmax_semantics():
    # Mean node: q is the mean.
    mean = _Node(0.5)
    mean.visit_count = 4
    mean.value_sum = 1.2
    assert mean.q == 0.3
    # Soft-min node with no visited children yet: falls back to
    # the mean (its leaf value right after expansion).
    soft = _Node(0.5, softmax=True)
    soft.visit_count = 1
    soft.value_sum = -0.25
    assert soft.softmax_value is None
    assert soft.q == -0.25
    # Soft-min node with a cached value: q IS the cached value.
    soft.softmax_value = 0.8
    assert soft.q == 0.8
    # Default construction is a mean node.
    assert _Node(0.5).softmax is False


def test_backup_rule_validation():
    try:
        HeuristicMCTS(backup_rule="max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backup_rule must raise ValueError")
    try:
        HeuristicMCTS(backup_rule="softmax-opp", softmax_beta=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive softmax_beta must raise")
    assert HeuristicMCTS().backup_rule == "mean"
    assert HeuristicMCTS().softmax_beta == 2.0


def test_flag_propagation():
    # Root children are soft-min nodes (opponent to move, odd
    # depth); depth-2 nodes are mean nodes; the root is a mean
    # node.
    board = _board_normal(_G61_MOVES)
    root = _run_search(_v51_engine(), board, game.WHITE)
    assert root.softmax is False
    for child in root.children.values():
        assert child.softmax is True
        if child.children:
            for grandchild in child.children.values():
                assert grandchild.softmax is False
    # Default (mean) backup: no node is soft-min.
    root_mean = _run_search(_v27_engine(), board, game.WHITE)
    assert root_mean.softmax is False
    assert all(not c.softmax for c in root_mean.children.values())


def test_backup_invariants_on_real_tree():
    # After a v51 search, every soft-min node with visited
    # children reports the softmax (beta=2.0) over visited
    # children of -child.q, and every mean node reports
    # value_sum / visit_count.
    for board, player in (
        (_board_normal(_G61_MOVES), game.WHITE),
        (_board_swap(_G43_MOVES), game.WHITE),
    ):
        root = _run_search(_v51_engine(), board, player)
        stack = [root]
        n_soft = n_mean = 0
        while stack:
            node = stack.pop()
            if node.children is None:
                continue
            visited = [c for c in node.children.values() if c.visit_count]
            if node.softmax and node.softmax_value is not None:
                n_soft += 1
                expect = _softmax_mean([-c.q for c in visited], 2.0)
                assert node.q == expect
            elif not node.softmax and node.visit_count:
                n_mean += 1
                assert node.q == node.value_sum / node.visit_count
            stack.extend(node.children.values())
        assert n_soft > 0 and n_mean > 0


def test_v27_reproduces_blunder_dilution():
    # CONTROL: v27 mis-ranks on both instances (iter-40 §1.1a) —
    # the unsafe move leads on visits with the best our-q even
    # though its 1-ply leaf is -1.0. If this fails, the fixture
    # (not the engine) is wrong.
    for board, player, unsafe, safes in (
        (_board_normal(_G61_MOVES), game.WHITE, _G61_UNSAFE, _G61_SAFES),
        (_board_swap(_G43_MOVES), game.WHITE, _G43_UNSAFE, _G43_SAFES),
    ):
        root = _run_search(_v27_engine(), board, player)
        u = root.children[unsafe]
        best_safe_q = max(-root.children[s].q for s in safes)
        assert u.visit_count == max(
            c.visit_count for c in root.children.values()
        )
        assert -u.q > best_safe_q


def test_v51_mechanism_outcome():
    # OBSERVED mechanism-check outcome (iter-51.md §1.4) — the
    # pre-registered prediction FAILED on three of four
    # positions. The soft-min collapses like minimax wherever
    # the root child has ~1 visited child (the softmax over one
    # child IS that child's value, and the expansion leaf — where
    # the mean's gradient lives — is discarded).
    #
    # g13/g2: preventions at exactly -1.0, the pick is a losing
    # move — same as v47.
    for pid, losing_pick in (
        ("prev-v1v0s-g13", game.xy_to_move(5, 4)),
        ("prev-v1bl-g2", game.xy_to_move(7, 9)),
    ):
        board, player, expect = _tactical_position(pid)
        engine = _v51_engine()
        root = _run_search(engine, board, player)
        assert all(
            -root.children[m].q == -1.0 for m in expect
        ), f"{pid}: preventions should sit at exactly -1.0 under v51"
        pick = engine._choose_root_child(root, board, player)
        assert pick == losing_pick, (
            f"{pid}: v51 should pick the losing move "
            f"{(losing_pick % 15, losing_pick // 15)}, picked "
            f"{(pick % 15, pick // 15)}"
        )
    # g61: the ranking INVERTS — the unsafe keeps a gradient
    # value (-0.9293: the beta=2 softmax of its creator child
    # +1.0 and its two blunder children -1.0) while the safes
    # collapse to -1.0, so the unsafe ranks BEST.
    board = _board_normal(_G61_MOVES)
    engine = _v51_engine()
    root = _run_search(engine, board, game.WHITE)
    u = root.children[_G61_UNSAFE]
    assert -0.93 < -u.q < -0.92
    assert all(-root.children[s].q == -1.0 for s in _G61_SAFES)
    assert -u.q > max(-root.children[s].q for s in _G61_SAFES)
    # g43: the ONE position where the mechanism works as
    # predicted — the safes have high priors (0.1342), so they
    # get 9-11 visits and several visited children, and the
    # softmax mixes: unsafe strictly worst, best safe > -1.0,
    # pick is a safe move.
    board = _board_swap(_G43_MOVES)
    engine = _v51_engine()
    root = _run_search(engine, board, game.WHITE)
    u = root.children[_G43_UNSAFE]
    best_safe_q = max(-root.children[s].q for s in _G43_SAFES)
    assert -u.q < min(-root.children[s].q for s in _G43_SAFES)
    assert best_safe_q > -1.0
    pick = engine._choose_root_child(root, board, game.WHITE)
    assert pick in _G43_SAFES


def test_mean_backup_unchanged_by_flag():
    # With backup_rule="mean" (the default), the search is the v27
    # search: same root pick and same visit distribution as an
    # engine constructed without the parameter.
    board = _board_normal(_G61_MOVES)
    a = HeuristicMCTS(
        sims=100, c_puct=1.5, rng=np.random.default_rng(0),
        leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05,
    )
    b = HeuristicMCTS(
        sims=100, c_puct=1.5, rng=np.random.default_rng(0),
        leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05, backup_rule="mean",
    )
    ra = _run_search(a, board, game.WHITE)
    rb = _run_search(b, board, game.WHITE)
    assert a._choose_root_child(ra, board, game.WHITE) == (
        b._choose_root_child(rb, board, game.WHITE)
    )
    for m in ra.children:
        assert ra.children[m].visit_count == rb.children[m].visit_count
        assert ra.children[m].q == rb.children[m].q

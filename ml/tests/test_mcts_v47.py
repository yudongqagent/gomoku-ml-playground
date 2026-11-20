"""Tests for v47 (batch 12, H3): minimax backup at opponent nodes.

The mechanism is iter-40 §1.1a's blunder dilution: the MCTS mean
backup mixes the opponent's best reply with their blunders, so a
move that loses at once (the opponent holds a creator) can rank
above safe moves. v47 switches opponent nodes (odd depth from the
root) to a minimax backup — q is the opponent's best reply (max
over visited children of -child.q) — while our nodes keep the mean.
See research/iterations/iter-47.md §1.2 for the pre-registered rule.

Fixtures: the two diagnosed instances, reconstructed from the v34
battery JSONs on research/batch-10-h2 (move lists below; coordinates
in iter-40 are (col, row), flat = row * 15 + col).
"""

import numpy as np

from gomoku_ml import game, scoring
from gomoku_ml.mcts_heuristic import HeuristicMCTS, _Node

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


def _v27_engine():
    return HeuristicMCTS(
        sims=100, c_puct=1.5, rng=np.random.default_rng(0),
        leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05,
    )


def _v47_engine():
    return HeuristicMCTS(
        sims=100, c_puct=1.5, rng=np.random.default_rng(0),
        leaf_mode="rollout2_v1_both", tie_break_q=True,
        tie_break_q_threshold=0.05, backup_rule="minimax-opp",
    )


def _run_search(engine, board, player):
    root = _Node(1.0)
    for _ in range(engine.sims):
        engine._run_one(root, board, player, None)
    return root


def test_v47_registered():
    player = scoring.get_player(47, seed=0)
    assert player.name == "scoring-v47"
    assert player._mcts.backup_rule == "minimax-opp"
    assert player._mcts.leaf_mode == "rollout2_v1_both"
    assert player._mcts.tie_break_q is True
    assert player._mcts.tie_break_q_threshold == 0.05


def test_node_q_minimax_semantics():
    # Mean node: q is the mean.
    mean = _Node(0.5)
    mean.visit_count = 4
    mean.value_sum = 1.2
    assert mean.q == 0.3
    # Minimax node with no visited children yet: falls back to the
    # mean (its leaf value right after expansion).
    mini = _Node(0.5, minimax=True)
    mini.visit_count = 1
    mini.value_sum = -0.25
    assert mini.minimax_value is None
    assert mini.q == -0.25
    # Minimax node with a cached value: q IS the cached value.
    mini.minimax_value = 0.8
    assert mini.q == 0.8
    # Default construction is a mean node.
    assert _Node(0.5).minimax is False


def test_backup_rule_validation():
    try:
        HeuristicMCTS(backup_rule="max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown backup_rule must raise ValueError")
    assert HeuristicMCTS().backup_rule == "mean"


def test_flag_propagation():
    # Root children are minimax nodes (opponent to move, odd depth);
    # depth-2 nodes are mean nodes; the root is a mean node.
    board = _board_normal(_G61_MOVES)
    root = _run_search(_v47_engine(), board, game.WHITE)
    assert root.minimax is False
    for child in root.children.values():
        assert child.minimax is True
        if child.children:
            for grandchild in child.children.values():
                assert grandchild.minimax is False
    # Default (mean) backup: no node is minimax.
    root_mean = _run_search(_v27_engine(), board, game.WHITE)
    assert root_mean.minimax is False
    assert all(not c.minimax for c in root_mean.children.values())


def test_backup_invariants_on_real_tree():
    # After a v47 search, every minimax node with visited children
    # reports max over visited children of -child.q, and every mean
    # node reports value_sum / visit_count.
    for board, player in (
        (_board_normal(_G61_MOVES), game.WHITE),
        (_board_swap(_G43_MOVES), game.WHITE),
    ):
        root = _run_search(_v47_engine(), board, player)
        stack = [root]
        n_minimax = n_mean = 0
        while stack:
            node = stack.pop()
            if node.children is None:
                continue
            visited = [c for c in node.children.values() if c.visit_count]
            if node.minimax and node.minimax_value is not None:
                n_minimax += 1
                expect = max(-c.q for c in visited)
                assert node.q == expect
            elif not node.minimax and node.visit_count:
                n_mean += 1
                assert node.q == node.value_sum / node.visit_count
            stack.extend(node.children.values())
        assert n_minimax > 0 and n_mean > 0


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


def test_v47_prices_opponent_best_reply():
    # MECHANISM: under v47 the unsafe move's our-q is exactly -1.0
    # on both instances — the opponent's best reply (the creator)
    # is priced, not diluted by blunder leaves. The safes are ALSO
    # -1.0: the positions are lost (every safe's subtree contains
    # a visited -1.0 line — Black's (6,4) in g61, (10,7) in g43 —
    # so minimax correctly values every move at -1.0 and the q
    # ranking TIES; the mean's "how slowly do we lose" gradient is
    # gone. The pick nevertheless flips to a safe move on both
    # instances, via visit reallocation (with all q's tied at -1.0,
    # UCB is prior-driven and the highest-prior safe moves win
    # visits). See iter-47.md §1.2 / §2.
    for board, player, unsafe, safes in (
        (_board_normal(_G61_MOVES), game.WHITE, _G61_UNSAFE, _G61_SAFES),
        (_board_swap(_G43_MOVES), game.WHITE, _G43_UNSAFE, _G43_SAFES),
    ):
        engine = _v47_engine()
        root = _run_search(engine, board, player)
        u = root.children[unsafe]
        assert -u.q == -1.0
        assert all(-root.children[s].q == -1.0 for s in safes)
        assert u.visit_count < max(
            root.children[s].visit_count for s in safes
        )
        pick = engine._choose_root_child(root, board, player)
        assert pick in safes, (
            f"v47 should pick a safe move, picked {(pick % 15, pick // 15)}"
        )


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

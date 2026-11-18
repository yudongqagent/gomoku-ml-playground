"""Tests for the learned-value MCTS leaf (v8, batch 3 H2)."""

import numpy as np
import pytest
import torch

from gomoku_ml import game, scoring
from gomoku_ml.architectures import build_architecture
from gomoku_ml.mcts_heuristic import _MUST_BLOCK_VALUE as _V6_MUST_BLOCK
from gomoku_ml.mcts_heuristic import _Node
from gomoku_ml.mcts_value import _MUST_BLOCK_VALUE, ValueMCTS


def _pool_with_children(specs):
    """Fabricate a root pool: ``specs`` is a list of
    (move, visit_count, value_sum) in dict-insertion order."""
    pool = {}
    for move, visits, value_sum in specs:
        child = _Node(0.1)
        child.visit_count = visits
        child.value_sum = value_sum
        pool[move] = child
    return pool


def _board_with(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def _random_net_checkpoint(tmp_path):
    """A tiny random-init resnet 16/2 checkpoint on disk."""
    torch.manual_seed(0)
    model = build_architecture("resnet", {"channels": 16, "blocks": 2})
    path = tmp_path / "value-net.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "resnet",
            "arch_params": {"channels": 16, "blocks": 2},
            "channels": 16,
            "blocks": 2,
        },
        path,
    )
    return str(path)


def _value_mcts(tmp_path, sims=10):
    return ValueMCTS(checkpoint_path=_random_net_checkpoint(tmp_path), sims=sims)


def test_must_block_value_matches_v6():
    # The terminal leaf cases are shared with v6 by design; a drift
    # here would silently change the defence behaviour.
    assert _MUST_BLOCK_VALUE == _V6_MUST_BLOCK == -0.25


def test_leaf_value_player_to_move_wins(tmp_path):
    # Black to move has a winning move (four in a row) -> +1, no net call.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = _value_mcts(tmp_path)
    assert mcts._leaf_value(board, game.BLACK) == 1.0
    assert mcts.leaf_net_evals == 0


def test_leaf_value_opponent_two_winning_moves(tmp_path):
    # White holds an open four (two winning moves); Black has none -> -1.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    mcts = _value_mcts(tmp_path)
    assert mcts._leaf_value(board, game.BLACK) == -1.0
    assert mcts.leaf_net_evals == 0


def test_leaf_value_opponent_one_winning_move(tmp_path):
    # White holds a broken four: only the gap (5,7) wins -> -0.25.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = _value_mcts(tmp_path)
    assert mcts._leaf_value(board, game.BLACK) == -0.25
    assert mcts.leaf_net_evals == 0


def test_leaf_value_quiet_uses_net(tmp_path):
    # Quiet position: the net value SHRUNK by VALUE_WEIGHT
    # (fast-loop iteration 2), in [-1, 1], matching a direct
    # forward pass on the same encoding times the weight.
    from gomoku_ml.mcts_value import VALUE_WEIGHT

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _value_mcts(tmp_path)
    value = mcts._leaf_value(board, game.BLACK)
    assert -1.0 <= value <= 1.0
    assert mcts.leaf_net_evals == 1
    from gomoku_ml import encoding

    planes = encoding.encode(board, game.BLACK, None, None)
    with torch.no_grad():
        _logits, direct = mcts.net(torch.from_numpy(planes).unsqueeze(0))
    assert value == pytest.approx(float(direct[0, 0].item()) * VALUE_WEIGHT)
    assert 0.0 < VALUE_WEIGHT < 1.0


def test_leaf_value_shrink_bound(tmp_path):
    # The shrink bounds the quiet leaf value by VALUE_WEIGHT in
    # magnitude, whatever the net outputs.
    from gomoku_ml.mcts_value import VALUE_WEIGHT

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _value_mcts(tmp_path)
    assert abs(mcts._leaf_value(board, game.BLACK)) <= VALUE_WEIGHT


def test_leaf_value_with_history_and_last_move(tmp_path):
    # History threading: a leaf deep in the search gets the path
    # boards as history and the move that led to it as last_move.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    prev = _board_with([(7, 7)], [(7, 9)])
    mcts = _value_mcts(tmp_path)
    value = mcts._leaf_value(board, game.BLACK, history=[prev, board], last_move=8 * 15 + 8)
    assert -1.0 <= value <= 1.0
    assert mcts.leaf_net_evals == 1


def test_choose_move_takes_immediate_win(tmp_path):
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = _value_mcts(tmp_path)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) in [(7, 6), (7, 11)]


def test_choose_move_blocks_single_winning_move(tmp_path):
    # White's broken four: only (5,7) wins; Black must block it.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = _value_mcts(tmp_path)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) == (5, 7)


def test_choose_move_searches_and_does_not_mutate_board(tmp_path):
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = _value_mcts(tmp_path, sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert board[move // 15, move % 15] == game.EMPTY  # legal move
    assert np.array_equal(board, before)
    assert mcts.leaf_net_evals > 0  # the search actually evaluated leaves


def test_choose_move_deterministic_for_same_seed(tmp_path):
    rng = np.random.default_rng(1)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:20]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    ckpt = _random_net_checkpoint(tmp_path)
    a = ValueMCTS(checkpoint_path=ckpt, sims=20, rng=np.random.default_rng(0))
    b = ValueMCTS(checkpoint_path=ckpt, sims=20, rng=np.random.default_rng(0))
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)


def test_choose_move_threat_check_picks_safe_move(tmp_path):
    # Fast-loop iteration 1: the opponent holds an open three
    # (White (5,5),(5,6),(5,7) with both ends open — either end
    # creates an open four), so Black must play a safe move: a
    # forcing move, or one leaving White with no live creator.
    # The unpatched search plays its highest-visited (usually
    # attacking) move; the patched choice must be safe.
    board = _board_with(
        [(7, 7), (8, 8), (3, 3)], [(5, 5), (5, 6), (5, 7)]
    )
    creators = scoring._opponent_creators(board, game.BLACK)
    assert creators  # White can create an open four next move
    mcts = _value_mcts(tmp_path, sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert scoring._move_is_safe(board, move, game.BLACK, creators), (
        f"chose {(move % 15, move // 15)}, not safe against {creators}"
    )
    assert mcts.threat_filter_used == 1


def test_choose_move_threat_check_noop_without_creators(tmp_path):
    # No opponent creators: the filter must not engage and the
    # choice must be the plain most-visited move.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    assert scoring._opponent_creators(board, game.BLACK) == []
    mcts = _value_mcts(tmp_path, sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert mcts.threat_filter_used == 0
    assert move != -1


def test_value_mcts_needs_a_net():
    import pytest

    with pytest.raises(ValueError, match="needs a value net"):
        ValueMCTS()


def test_v8_registered_and_dispatches_to_value_mcts(tmp_path, monkeypatch):
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[8], "value_checkpoint", ckpt)
    v8 = scoring.get_player(8, seed=0)
    assert v8.name == "scoring-v8"
    assert v8.mcts is True
    assert isinstance(v8._mcts, ValueMCTS)
    assert v8._mcts.sims == 100
    assert v8._mcts.checkpoint_path == ckpt
    # v6 still dispatches to the plain heuristic MCTS.
    from gomoku_ml.mcts_heuristic import HeuristicMCTS

    v6 = scoring.get_player(6, seed=0)
    assert type(v6._mcts) is HeuristicMCTS


def test_v8_env_override_and_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GOMOKU_V8_CHECKPOINT", str(tmp_path / "nope.pt"))
    import pytest

    with pytest.raises(FileNotFoundError):
        scoring.get_player(8, seed=0)


# --- v36: v27's threshold tie-break ported to the value engine ---------
# (ML batch 3, H1; research/iterations/iter-36.md §1.1). The rule lives
# in ValueMCTS._pick_pool_move, mirroring HeuristicMCTS._pick_root_move
# on the effective pool (after the threat filter).


def test_v36_tie_break_picks_highest_our_q_when_gap_exceeds_threshold(tmp_path):
    # Three children tie at 3 visits. child.q is the OPPONENT's
    # perspective, so our-q = -child.q: move 101 (value_sum -3.0,
    # q -1.0, our-q +1.0) beats 100 (our-q +0.583) and 102
    # (our-q +0.667) — even though 101 is not first in dict order.
    # Gap 1.0 - 0.667 = 0.333 > τ=0.05, so the tie-break fires.
    pool = _pool_with_children([(100, 3, -1.75), (101, 3, -3.0), (102, 3, -2.0)])
    mcts = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10,
        tie_break_q=True, tie_break_q_threshold=0.05,
    )
    assert mcts._pick_pool_move(pool) == 101


def test_v36_threshold_blocks_small_gap(tmp_path):
    # The gap is 0.0167 (the documented rollout-leaf noise gap):
    # our-q +0.5833 (move 101) vs +0.5667 (move 100) vs +0.3333
    # (move 102). Below τ=0.05 the tie-break must NOT fire — with
    # no RNG the choice is the FIRST tied child in dict order
    # (100), not the q-max child (101).
    pool = _pool_with_children([(100, 3, -1.70), (101, 3, -1.75), (102, 3, -1.0)])
    mcts = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10,
        tie_break_q=True, tie_break_q_threshold=0.05,
    )
    assert mcts._pick_pool_move(pool) == 100
    # At τ=0.0 (v23's rule) the same gap fires and picks 101.
    mcts23 = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10,
        tie_break_q=True, tie_break_q_threshold=0.0,
    )
    assert mcts23._pick_pool_move(pool) == 101


def test_v36_q_tie_falls_back_to_rng_subset(tmp_path):
    # Moves 100 and 102 tie at 3 visits AND at our-q +0.583
    # (value_sum -1.75 each); move 101 ties on visits but holds
    # worse our-q (+0.333). Gap 0.583 - 0.333 = 0.25 > τ, so the
    # tie-break fires, then falls back to the q-tied subset: the
    # choice must be 100 or 102 only — never 101.
    pool = _pool_with_children([(100, 3, -1.75), (101, 3, -1.0), (102, 3, -1.75)])
    mcts = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10,
        tie_break_q=True, tie_break_q_threshold=0.05,
        rng=np.random.default_rng(0),
    )
    for _ in range(20):
        assert mcts._pick_pool_move(pool) in (100, 102)
    # With no RNG the fallback is the first q-tied child in dict
    # order (100), not the first visit-tied child overall — here
    # they coincide, so rebuild with the worse-q move first.
    pool2 = _pool_with_children([(101, 3, -1.0), (100, 3, -1.75), (102, 3, -1.75)])
    mcts_no_rng = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10,
        tie_break_q=True, tie_break_q_threshold=0.05,
    )
    assert mcts_no_rng._pick_pool_move(pool2) == 100


def test_v36_unique_visit_leader_untouched(tmp_path):
    # The visit leader (4 visits) holds the WORST our-q — v36
    # must still play it: the rule fires only on exact ties.
    pool = _pool_with_children([(100, 4, -0.5), (101, 3, -3.0), (102, 3, -2.0)])
    mcts = ValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=10,
        tie_break_q=True, tie_break_q_threshold=0.05,
    )
    assert mcts._pick_pool_move(pool) == 100
    # A 1-visit gap is NOT a tie either.
    pool2 = _pool_with_children([(100, 4, -0.5), (101, 3, -3.0)])
    assert mcts._pick_pool_move(pool2) == 100


def test_v36_flag_off_is_v8_behaviour(tmp_path):
    # Without the flag the same tied root returns the first tied
    # child (no RNG) regardless of q — the flag is the only change.
    pool = _pool_with_children([(100, 3, -1.75), (101, 3, -3.0), (102, 3, -2.0)])
    mcts = ValueMCTS(checkpoint_path=_random_net_checkpoint(tmp_path), sims=10)
    assert mcts.tie_break_q is False
    assert mcts.tie_break_q_threshold == 0.0
    assert mcts._pick_pool_move(pool) == 100


def test_v36_sweep_versions_registered_and_dispatch(tmp_path, monkeypatch):
    # Temporary sweep candidates 92 (τ=0.05) and 93 (τ=0.10) —
    # removed when v36 is registered with the selected threshold.
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[92], "value_checkpoint", ckpt)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[93], "value_checkpoint", ckpt)
    v92 = scoring.get_player(92, seed=0)
    assert v92.name == "scoring-v92"
    assert isinstance(v92._mcts, ValueMCTS)
    assert v92._mcts.sims == 100
    assert v92._mcts.tie_break_q is True
    assert v92._mcts.tie_break_q_threshold == 0.05
    v93 = scoring.get_player(93, seed=0)
    assert v93._mcts.tie_break_q is True
    assert v93._mcts.tie_break_q_threshold == 0.10
    # v8 is unchanged (tie-break off by default).
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[8], "value_checkpoint", ckpt)
    v8 = scoring.get_player(8, seed=0)
    assert v8._mcts.tie_break_q is False
    assert v8._mcts.tie_break_q_threshold == 0.0

"""Tests for the learned-prior MCTS (v26, ML batch 1 H3)."""

import numpy as np
import pytest
import torch

from gomoku_ml import game, scoring
from gomoku_ml.architectures import build_architecture
from gomoku_ml.mcts_heuristic import _MUST_BLOCK_VALUE as _V6_MUST_BLOCK
from gomoku_ml.mcts_policy import _MUST_BLOCK_VALUE, PolicyValueMCTS
from gomoku_ml.mcts_value import VALUE_WEIGHT


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
    path = tmp_path / "policy-value-net.pt"
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


def _policy_mcts(tmp_path, sims=10, **kwargs):
    return PolicyValueMCTS(
        checkpoint_path=_random_net_checkpoint(tmp_path), sims=sims, **kwargs
    )


def test_must_block_value_matches_v6():
    # The terminal leaf cases are shared with v6 by design; a drift
    # here would silently change the defence behaviour.
    assert _MUST_BLOCK_VALUE == _V6_MUST_BLOCK == -0.25


def test_leaf_value_terminal_cases_exact_no_net_call(tmp_path):
    # Terminal cases never touch the net (same as v8).
    mcts = _policy_mcts(tmp_path)
    # Black to move has a winning move -> +1.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert mcts._leaf_value(board, game.BLACK) == 1.0
    # White holds an open four (two winning moves) -> -1.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    assert mcts._leaf_value(board, game.BLACK) == -1.0
    # White holds a broken four: only the gap wins -> -0.25.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    assert mcts._leaf_value(board, game.BLACK) == -0.25
    assert mcts.leaf_net_evals == 0


def test_leaf_value_quiet_uses_net_shrunk(tmp_path):
    # Quiet position: the net value SHRUNK by value_weight (0.25),
    # matching a direct forward pass on the same encoding.
    from gomoku_ml import encoding

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _policy_mcts(tmp_path)
    value = mcts._leaf_value(board, game.BLACK)
    assert mcts.leaf_net_evals == 1
    planes = encoding.encode(board, game.BLACK, None, None)
    with torch.no_grad():
        _logits, direct = mcts.net(
            torch.from_numpy(planes).unsqueeze(0).to(mcts.device)
        )
    assert value == pytest.approx(float(direct[0, 0].item()) * VALUE_WEIGHT)


def test_priors_sum_to_one_over_candidates(tmp_path):
    # Priors cover exactly the candidate moves and sum to 1.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _policy_mcts(tmp_path)
    priors, _value = mcts._net_eval(board, game.BLACK)
    cands = scoring._candidate_moves(board)
    assert set(priors.keys()) == set(cands)
    assert sum(priors.values()) == pytest.approx(1.0)
    assert all(p > 0 for p in priors.values())


def test_priors_match_softmax_of_logits(tmp_path):
    # Priors are the temperature softmax of the policy logits over
    # candidates — no v1 score anywhere in the computation.
    import math

    from gomoku_ml import encoding

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _policy_mcts(tmp_path, prior_temperature=0.5)
    priors, _value = mcts._net_eval(board, game.BLACK)
    planes = encoding.encode(board, game.BLACK, None, None)
    with torch.no_grad():
        logits, _v = mcts.net(torch.from_numpy(planes).unsqueeze(0).to(mcts.device))
    flat = logits[0].detach().cpu().numpy()
    cands = scoring._candidate_moves(board)
    scaled = {m: float(flat[m]) / 0.5 for m in cands}
    top = max(scaled.values())
    weights = {m: math.exp(v - top) for m, v in scaled.items()}
    total = sum(weights.values())
    for m in cands:
        assert priors[m] == pytest.approx(weights[m] / total)


def test_lower_temperature_sharpens_priors(tmp_path):
    # T=0.25 concentrates more mass on the top move than T=1.0.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts_hot = _policy_mcts(tmp_path, prior_temperature=1.0)
    mcts_cold = _policy_mcts(tmp_path, prior_temperature=0.25)
    priors_hot, _ = mcts_hot._net_eval(board, game.BLACK)
    priors_cold, _ = mcts_cold._net_eval(board, game.BLACK)
    assert max(priors_cold.values()) > max(priors_hot.values())


def test_invalid_temperature_rejected(tmp_path):
    with pytest.raises(ValueError, match="prior_temperature"):
        _policy_mcts(tmp_path, prior_temperature=0.0)
    with pytest.raises(ValueError, match="prior_temperature"):
        _policy_mcts(tmp_path, prior_temperature=-1.0)


def test_one_forward_per_expansion(tmp_path):
    # The search runs exactly one forward pass per expansion (root
    # + one per simulated leaf): sims forwards for a sims-sim search
    # on a fresh board (each sim expands exactly one node).
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _policy_mcts(tmp_path, sims=10)
    mcts.choose_move(board, game.BLACK)
    assert mcts.leaf_net_evals == 10


def test_choose_move_plays_winning_move(tmp_path):
    # Win scan before the search: Black has a vertical four in
    # column 7 (rows 7-10); either end (7,6) or (7,11) wins.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = _policy_mcts(tmp_path, sims=10)
    assert mcts.choose_move(board, game.BLACK) in (6 * 15 + 7, 11 * 15 + 7)


def test_choose_move_blocks_single_win(tmp_path):
    # Block scan: White has one winning move; Black blocks it.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = _policy_mcts(tmp_path, sims=10)
    assert mcts.choose_move(board, game.BLACK) == 7 * 15 + 5


def test_v26_registered_in_scoring(tmp_path, monkeypatch):
    # scoring.get_player(26) dispatches to PolicyValueMCTS with the
    # checkpoint from the env override (the registry path does not
    # exist in a fresh checkout).
    monkeypatch.setenv("GOMOKU_V26_CHECKPOINT", _random_net_checkpoint(tmp_path))
    monkeypatch.setenv("GOMOKU_V26_TEMPERATURE", "0.5")
    player = scoring.get_player(26, seed=0)
    assert player.name == "scoring-v26"
    assert isinstance(player._mcts, PolicyValueMCTS)
    assert player._mcts.prior_temperature == 0.5

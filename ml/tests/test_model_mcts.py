"""Tests for the network, MCTS, baseline player, and self-play plumbing."""

import numpy as np
import pytest
import torch

from gomoku_ml import baseline, encoding, game, selfplay
from gomoku_ml.mcts import MCTS
from gomoku_ml.model import GomokuNet, build_model


def _tiny_net():
    torch.manual_seed(0)
    return GomokuNet(channels=16, blocks=2)


def test_model_forward_shapes():
    net = _tiny_net()
    x = torch.randn(3, 8, 15, 15)
    policy, value = net(x)
    assert policy.shape == (3, 225)
    assert value.shape == (3, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)  # tanh
    assert net.num_parameters() > 0


def test_model_batch_one_matches_contract():
    net = _tiny_net()
    net.eval()
    x = torch.zeros(1, 8, 15, 15)
    with torch.no_grad():
        policy, value = net(x)
    assert policy.shape == (1, 225)
    assert value.shape == (1, 1)


def test_build_model_from_config():
    net = build_model({"model": {"channels": 8, "blocks": 1}})
    assert net.channels == 8 and net.blocks == 1
    net_default = build_model({})
    assert net_default.channels == 128 and net_default.blocks == 10


def test_mcts_returns_legal_move():
    net = _tiny_net()
    mcts = MCTS(net)
    board = game.empty_board()
    board = game.apply_move(board, game.xy_to_move(7, 7), game.BLACK)
    root, value = mcts.search(board, game.WHITE, num_simulations=16)
    assert -1.0 <= value <= 1.0
    move = MCTS.choose_move(root, temperature=0.0)
    assert game.is_legal(board, move)
    dist = MCTS.visit_distribution(root, temperature=1.0)
    assert dist.shape == (225,)
    assert abs(dist.sum() - 1.0) < 1e-6
    assert dist[game.xy_to_move(7, 7)] == 0.0  # occupied point gets no visits


def test_mcts_visits_concentrate_on_winning_move():
    """With enough sims, MCTS should find an immediate win: the winning
    move is a terminal child (value +1 for the mover), so PUCT must
    prefer it over non-terminal moves once explored."""
    net = _tiny_net()
    mcts = MCTS(net, c_puct=1.0)
    board = game.empty_board()
    for x in (7, 8, 9, 10):
        board = game.apply_move(board, game.xy_to_move(x, 7), game.BLACK)
    root, _ = mcts.search(board, game.BLACK, num_simulations=100)
    move = MCTS.choose_move(root, temperature=0.0)
    assert move in (game.xy_to_move(6, 7), game.xy_to_move(11, 7))


def test_baseline_takes_win_and_blocks():
    board = game.empty_board()
    for x in (7, 8, 9, 10):
        board = game.apply_move(board, game.xy_to_move(x, 7), game.BLACK)
    assert baseline.choose_move(board, game.BLACK) in (
        game.xy_to_move(6, 7), game.xy_to_move(11, 7))
    # White must block Black's open four.
    assert baseline.choose_move(board, game.WHITE) in (
        game.xy_to_move(6, 7), game.xy_to_move(11, 7))


def test_selfplay_game_produces_consistent_samples():
    net = _tiny_net()
    mcts = MCTS(net)
    samples = selfplay.play_game(mcts, num_simulations=4, temperature_moves=2,
                                 max_moves=30)
    n = len(samples["moves"])
    assert n > 0
    assert samples["planes"].shape == (n, 8, 15, 15)
    assert samples["policy"].shape == (n, 225)
    assert samples["value"].shape == (n,)
    # Policy targets are distributions; values are in {-1, 0, 1}.
    assert np.allclose(samples["policy"].sum(axis=1), 1.0)
    assert set(np.unique(samples["value"])).issubset({-1.0, 0.0, 1.0})
    # The winner (if any) is consistent with the value signs.
    if samples["winner"] != game.EMPTY:
        assert samples["value"][0] in (-1.0, 1.0)

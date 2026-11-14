"""Tests for batched leaf evaluation (MCTS.search_many / _evaluate_batch)
and batched self-play (selfplay.generate_games_batched).

Thresholds are set from measured behaviour, not guessed: on CPU the
batched forward pass matches single-leaf eval to ~1e-8, and
search_many builds visit distributions identical (TV 0.0) to search
in the checks run for research/iterations/batched-leaf-eval.md — the
assertions below leave 3+ orders of magnitude of headroom for the
occasional float-noise selection flip on other hardware.
"""

import numpy as np
import pytest
import torch

from gomoku_ml import game, selfplay
from gomoku_ml.mcts import MCTS, LeafRequest, SearchRequest
from gomoku_ml.model import GomokuNet


def _tiny_net():
    torch.manual_seed(0)
    return GomokuNet(channels=16, blocks=2)


def _random_positions(rng, n, max_plies=20):
    """n random mid-game positions: (board, player, history, last_move)."""
    positions = []
    for _ in range(n):
        board = game.empty_board()
        history: list[np.ndarray] = []
        last_move: int | None = None
        player = game.BLACK
        for _ in range(int(rng.integers(0, max_plies + 1))):
            legal = game.legal_moves(board)
            move = int(rng.choice(legal))
            history.append(board)
            board = game.apply_move(board, move, player)
            last_move = move
            player = -player
        positions.append((board, player, history, last_move))
    return positions


def test_evaluate_batch_matches_single():
    net = _tiny_net()
    mcts = MCTS(net)
    rng = np.random.default_rng(123)
    requests = [
        LeafRequest(b, p, h, lm)
        for b, p, h, lm in _random_positions(rng, 8)
    ]
    batched = mcts._evaluate_batch(requests)
    assert len(batched) == len(requests)
    max_policy_diff = 0.0
    max_value_diff = 0.0
    for req, (probs_b, value_b) in zip(requests, batched):
        probs_s, value_s = mcts._evaluate(
            req.board, req.player, req.history, req.last_move
        )
        assert probs_b.shape == (225,)
        assert abs(probs_b.sum() - 1.0) < 1e-9
        assert -1.0 <= value_b <= 1.0
        max_policy_diff = max(
            max_policy_diff, float(np.abs(probs_b - probs_s).max())
        )
        max_value_diff = max(max_value_diff, abs(value_b - value_s))
    assert max_policy_diff < 1e-5
    assert max_value_diff < 1e-5


def test_evaluate_batch_empty():
    mcts = MCTS(_tiny_net())
    assert mcts._evaluate_batch([]) == []


def test_search_many_matches_search():
    net = _tiny_net()
    mcts = MCTS(net)
    rng = np.random.default_rng(7)
    requests = [
        SearchRequest(board=b, player=p, num_simulations=32,
                      history=h, last_move=lm, add_noise=False)
        for b, p, h, lm in _random_positions(rng, 4, max_plies=10)
    ]
    batched = mcts.search_many(requests)
    assert len(batched) == len(requests)
    for req, (root_b, value_b) in zip(requests, batched):
        # Every simulation is accounted for: the root is visited once
        # per simulation, and the child visits sum to one less — the
        # first simulation expands the root without selecting a child.
        assert root_b.visit_count == req.num_simulations
        assert sum(MCTS.visit_counts(root_b).values()) == req.num_simulations - 1
        root_s, value_s = mcts.search(
            req.board, req.player, req.num_simulations,
            history=req.history, last_move=req.last_move, add_noise=False,
        )
        dist_b = MCTS.visit_distribution(root_b, temperature=1.0)
        dist_s = MCTS.visit_distribution(root_s, temperature=1.0)
        tv = 0.5 * float(np.abs(dist_b - dist_s).sum())
        assert tv < 0.05
        assert abs(value_b - value_s) < 0.01


def test_search_many_max_batch_size_chunks():
    """Chunking the batch (even down to 1) must not change the search."""
    net = _tiny_net()
    mcts = MCTS(net)
    rng = np.random.default_rng(11)
    requests = [
        SearchRequest(board=b, player=p, num_simulations=24,
                      history=h, last_move=lm, add_noise=False)
        for b, p, h, lm in _random_positions(rng, 4, max_plies=10)
    ]
    wide = mcts.search_many(requests, max_batch_size=32)
    narrow = mcts.search_many(requests, max_batch_size=1)
    for (root_w, _), (root_n, _) in zip(wide, narrow):
        dist_w = MCTS.visit_distribution(root_w, temperature=1.0)
        dist_n = MCTS.visit_distribution(root_n, temperature=1.0)
        assert 0.5 * float(np.abs(dist_w - dist_n).sum()) < 0.05


def test_search_many_terminal_and_empty():
    net = _tiny_net()
    mcts = MCTS(net)
    assert mcts.search_many([]) == []
    # Black has just completed five; White to move has already lost.
    board = game.empty_board()
    for x in (7, 8, 9, 10, 11):
        board = game.apply_move(board, game.xy_to_move(x, 7), game.BLACK)
    req = SearchRequest(
        board=board, player=game.WHITE, num_simulations=8,
        last_move=game.xy_to_move(11, 7), add_noise=False,
    )
    (root, value), = mcts.search_many([req])
    assert value == -1.0
    assert root.children == {}
    # Same result from the single-leaf path.
    root_s, value_s = mcts.search(
        board, game.WHITE, 8, last_move=game.xy_to_move(11, 7),
        add_noise=False,
    )
    assert value_s == -1.0
    assert root_s.children == {}


def test_generate_games_batched_completes(tmp_path):
    net = _tiny_net()
    mcts = MCTS(net)
    out = tmp_path / "data"
    paths = selfplay.generate_games_batched(
        mcts, num_games=4, num_simulations=4, out_dir=str(out),
        run_name="test", concurrency=3, max_batch_size=2,
        temperature_moves=2, max_moves=30,
    )
    assert len(paths) == 4
    assert len(set(paths)) == 4  # distinct shards, no overwrites
    manifest = selfplay.load_manifest(str(out))
    assert manifest is not None
    assert manifest["games"] == 4
    for path in paths:
        with np.load(path) as z:
            n = len(z["moves"])
            assert n > 0
            assert z["planes"].shape == (n, 8, 15, 15)
            assert z["policy"].shape == (n, 225)
            assert z["value"].shape == (n,)
            assert np.allclose(z["policy"].sum(axis=1), 1.0)
            assert set(np.unique(z["value"])).issubset({-1.0, 0.0, 1.0})
            assert int(z["winner"]) in (-1, 0, 1)


def test_generate_games_batched_resume_safe_numbering(tmp_path):
    """A second batched call into the same dir appends, never overwrites."""
    net = _tiny_net()
    mcts = MCTS(net)
    out = tmp_path / "data"
    kwargs = dict(num_simulations=4, out_dir=str(out), run_name="test",
                  concurrency=2, temperature_moves=2, max_moves=30)
    first = selfplay.generate_games_batched(mcts, num_games=2, **kwargs)
    second = selfplay.generate_games_batched(mcts, num_games=2, **kwargs)
    assert len(first) == 2 and len(second) == 2
    assert not set(first) & set(second)
    assert selfplay.load_manifest(str(out))["games"] == 4


def test_generate_games_batched_concurrency_one(tmp_path):
    """Degenerate case: concurrency=1 batches one leaf at a time and
    must still complete with valid shards."""
    net = _tiny_net()
    mcts = MCTS(net)
    out = tmp_path / "data"
    paths = selfplay.generate_games_batched(
        mcts, num_games=2, num_simulations=4, out_dir=str(out),
        run_name="test", concurrency=1, temperature_moves=2, max_moves=30,
    )
    assert len(paths) == 2
    with np.load(paths[0]) as z:
        assert len(z["moves"]) > 0
        assert z["planes"].shape[0] == len(z["moves"])

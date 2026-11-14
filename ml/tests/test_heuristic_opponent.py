"""Tests for heuristic-opponent data generation (batch 4 H3, v11).

``generate_games_batched(..., opponent="heuristic")`` plays one
colour per game with the frozen baseline heuristic instead of the
neural MCTS (odd-numbered games the net plays Black, even-numbered
the net plays White). The tests verify the shard contract and the
heuristic signature:

* shards are valid (same schema as self-play);
* heuristic plies carry one-hot policies (the heuristic has no
  visit distribution), neural plies carry visit distributions;
* on heuristic plies the recorded move obeys the heuristic's two
  hard rules — take an immediate win, block the opponent's single
  immediate win — verified by replaying each game from its shard.

The parallel test checks the merge path (2 workers) with the
opponent option threaded through ``game_kwargs``.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from gomoku_ml import game, selfplay
from gomoku_ml.mcts import MCTS
from gomoku_ml.model import GomokuNet

# Same 1-thread pinning as test_parallel_selfplay.py (see its
# comment: the default 4 threads slowed a batched forward pass
# ~290x under contention on this VM).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _tiny_net():
    torch.manual_seed(0)
    return GomokuNet(channels=16, blocks=2)


def _tiny_spec():
    return selfplay.NetSpec(
        arch_name="resnet",
        arch_params={"channels": 16, "blocks": 2},
        checkpoint=None,
    )


def _is_one_hot(row: np.ndarray) -> bool:
    return bool(np.count_nonzero(row) == 1 and np.isclose(row.max(), 1.0))


def _game_number(path) -> int:
    """1-based game number from a shard filename (``test_game0000.npz``
    is game 1). Paths are returned in finish/merge order, NOT game
    order, so the number must come from the name."""
    stem = Path(path).stem  # e.g. test_game0003
    return int(stem.rsplit("_game", 1)[1]) + 1


def _check_heuristic_signature(path, game_number):
    """Replay one shard and verify the heuristic side's signature.

    Returns (heuristic_plies, win_take_checks, block_checks).
    ``game_number`` is 1-based: odd games the heuristic plays White,
    even games the heuristic plays Black.
    """
    heuristic_player = game.WHITE if game_number % 2 == 1 else game.BLACK
    with np.load(path) as z:
        moves = [int(m) for m in z["moves"]]
        policies = z["policy"]
        players = [game.BLACK if i % 2 == 0 else game.WHITE
                   for i in range(len(moves))]
    board = game.empty_board()
    heuristic_plies = win_takes = blocks = 0
    for ply, move in enumerate(moves):
        player = players[ply]
        if player == heuristic_player:
            heuristic_plies += 1
            # One-hot policy on the played move.
            assert _is_one_hot(policies[ply]), (
                f"game {game_number} ply {ply}: heuristic policy not "
                "one-hot"
            )
            assert int(np.argmax(policies[ply])) == move
            # Hard rule 1: take an immediate win.
            wins = [m for m in game.legal_moves(board)
                    if game.is_winning_move(board, int(m), player)]
            if wins:
                win_takes += 1
                assert move in wins, (
                    f"game {game_number} ply {ply}: heuristic did not "
                    f"take the win (played {move}, wins {wins})"
                )
            # Hard rule 2: block the opponent's single immediate win.
            opp_wins = [m for m in game.legal_moves(board)
                        if game.is_winning_move(board, int(m), -player)]
            if len(opp_wins) == 1:
                blocks += 1
                assert move == opp_wins[0], (
                    f"game {game_number} ply {ply}: heuristic did not "
                    f"block the win (played {move}, must block "
                    f"{opp_wins[0]})"
                )
        board = game.apply_move(board, move, player)
    return heuristic_plies, win_takes, blocks


def _assert_valid_shard(path):
    with np.load(path) as z:
        n = len(z["moves"])
        assert n > 0
        assert z["planes"].shape == (n, 8, 15, 15)
        assert z["policy"].shape == (n, 225)
        assert z["value"].shape == (n,)
        assert np.allclose(z["policy"].sum(axis=1), 1.0)
        assert set(np.unique(z["value"])).issubset({-1.0, 0.0, 1.0})
        assert int(z["winner"]) in (-1, 0, 1)


def test_heuristic_opponent_batched(tmp_path):
    """Batched run with opponent='heuristic': valid shards, the
    heuristic signature on the right plies, both colours covered."""
    np.random.seed(0)
    torch.manual_seed(0)
    mcts = MCTS(_tiny_net())
    paths = selfplay.generate_games_batched(
        mcts, num_games=6, num_simulations=4, out_dir=str(tmp_path),
        run_name="test", concurrency=2,
        temperature_moves=2, max_moves=60, opponent="heuristic",
    )
    assert len(paths) == 6
    total_heuristic = total_wins = total_blocks = 0
    for p in paths:
        _assert_valid_shard(p)
        h, w, b = _check_heuristic_signature(p, game_number=_game_number(p))
        assert h > 0, f"{p}: no heuristic plies recorded"
        total_heuristic += h
        total_wins += w
        total_blocks += b
    assert total_heuristic > 0
    # At least one hard-rule situation must have arisen across the
    # run, or the win/block checks above are vacuous.
    assert total_wins + total_blocks >= 1, (
        "no win-take or block situation arose in 6 games — "
        "the heuristic signature checks are vacuous"
    )


def test_heuristic_opponent_parallel(tmp_path):
    """Parallel run (2 workers) with opponent='heuristic': merged
    shards valid, dense numbering, heuristic signature intact."""
    out = tmp_path / "merged"
    paths = selfplay.generate_games_parallel(
        _tiny_spec(), num_games=4, num_simulations=4,
        out_dir=str(out), run_name="test", workers=2, concurrency=2,
        max_batch_size=32, seed=0,
        temperature_moves=2, max_moves=60, opponent="heuristic",
    )
    assert len(paths) == 4
    names = sorted(Path(p).name for p in paths)
    assert names == [f"test_game{i:04d}.npz" for i in range(4)]
    manifest = selfplay.load_manifest(str(out))
    assert manifest["games"] == 4
    for p in paths:
        _assert_valid_shard(p)
        h, _, _ = _check_heuristic_signature(p, game_number=_game_number(p))
        assert h > 0


def test_heuristic_opponent_rejects_unknown(tmp_path):
    """An unknown opponent value fails fast, before any game."""
    mcts = MCTS(_tiny_net())
    with pytest.raises(ValueError, match="opponent must be"):
        selfplay.generate_games_batched(
            mcts, num_games=1, num_simulations=4, out_dir=str(tmp_path),
            run_name="test", concurrency=1, opponent="v1",
        )

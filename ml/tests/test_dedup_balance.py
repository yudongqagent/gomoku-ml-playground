"""Tests for the v30 dedup + colour-balance train-set builder."""

import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from gomoku_ml import game  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_dedup_balanced_train",
    os.path.join(os.path.dirname(__file__), "..", "scripts",
                 "build_dedup_balanced_train.py"),
)
build_dedup_balanced_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_dedup_balanced_train)
balance_winners = build_dedup_balanced_train.balance_winners


def test_balance_downsamples_majority_to_minority():
    # 53 Black wins, 47 White wins: keep all 47 White + 47 Black.
    winners = [game.BLACK] * 53 + [game.WHITE] * 47
    kept, dropped = balance_winners(winners, seed=0)
    assert len(kept) == 94
    assert len(dropped) == 6
    kept_winners = [winners[i] for i in kept]
    assert kept_winners.count(game.BLACK) == 47
    assert kept_winners.count(game.WHITE) == 47
    # Dropped games are all majority-class.
    assert all(winners[i] == game.BLACK for i in dropped)
    # Kept + dropped partition the input.
    assert sorted(kept + dropped) == list(range(100))


def test_balance_deterministic_for_seed():
    winners = [game.BLACK] * 60 + [game.WHITE] * 40
    kept_a, dropped_a = balance_winners(winners, seed=0)
    kept_b, dropped_b = balance_winners(winners, seed=0)
    assert kept_a == kept_b
    assert dropped_a == dropped_b
    # A different seed (almost surely) picks a different subset.
    kept_c, _ = balance_winners(winners, seed=1)
    assert kept_c != kept_a


def test_balance_balanced_input_keeps_everything():
    winners = [game.BLACK] * 50 + [game.WHITE] * 50
    kept, dropped = balance_winners(winners, seed=0)
    assert len(kept) == 100
    assert dropped == []


def test_balance_keeps_draws_and_handles_white_majority():
    winners = [game.BLACK] * 30 + [game.WHITE] * 50 + [game.EMPTY] * 5
    kept, dropped = balance_winners(winners, seed=0)
    kept_winners = [winners[i] for i in kept]
    assert kept_winners.count(game.BLACK) == 30
    assert kept_winners.count(game.WHITE) == 30
    assert kept_winners.count(game.EMPTY) == 5
    assert len(dropped) == 20
    assert all(winners[i] == game.WHITE for i in dropped)


def test_shard_winner_reads_stored_winner(tmp_path):
    import build_dedup_balanced_train as mod

    path = tmp_path / "g.npz"
    np.savez_compressed(path, winner=np.int64(game.WHITE))
    assert mod.shard_winner(str(path)) == game.WHITE

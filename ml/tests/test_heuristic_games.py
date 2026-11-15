"""Tests for heuristic-vs-heuristic game generation (batch 4 H3, v11).

Covers: one game's shard arrays (shapes, one-hot policies, value
targets, legal moves, winner consistency); the parallel driver
(merge, dense numbering, dedup, manifest); determinism (a generated
shard replays exactly from its game index and seeds); and the
MCTS-version rejection.
"""

from pathlib import Path

import numpy as np

from gomoku_ml import game, heuristic_games, selfplay


def _assert_valid_shard(path):
    with np.load(path) as z:
        n = len(z["moves"])
        assert n > 0
        assert z["planes"].shape == (n, 8, 15, 15)
        assert z["policy"].shape == (n, 225)
        assert z["value"].shape == (n,)
        # One-hot policies: each row sums to 1 with a single 1.0.
        assert np.allclose(z["policy"].sum(axis=1), 1.0)
        assert np.all(np.count_nonzero(z["policy"], axis=1) == 1)
        assert np.all(z["policy"][np.arange(n), z["moves"]] == 1.0)
        assert set(np.unique(z["value"])).issubset({-1.0, 0.0, 1.0})
        assert int(z["winner"]) in (-1, 0, 1)
        # Moves are legal in sequence; the winner matches the board.
        board = game.empty_board()
        for i, move in enumerate(z["moves"]):
            player = game.BLACK if i % 2 == 0 else game.WHITE
            assert board[move // 15, move % 15] == 0, f"ply {i}: occupied"
            board = game.apply_move(board, int(move), player)
        over, win = game.game_over(board)
        if over:
            assert int(z["winner"]) == win
        # Value targets are the outcome from the player-to-move view.
        winner = int(z["winner"])
        for i, v in enumerate(z["value"]):
            player = game.BLACK if i % 2 == 0 else game.WHITE
            expected = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
            assert float(v) == expected


def test_play_heuristic_game_valid():
    arrays = heuristic_games.play_heuristic_game(1, 0, 0, 1)
    n = len(arrays["moves"])
    assert n > 0
    assert arrays["planes"].shape == (n, 8, 15, 15)
    assert arrays["policy"].shape == (n, 225)
    assert arrays["value"].shape == (n,)
    assert np.allclose(arrays["policy"].sum(axis=1), 1.0)
    assert np.all(np.count_nonzero(arrays["policy"], axis=1) == 1)
    # Deterministic: same seeds replay the same game.
    again = heuristic_games.play_heuristic_game(1, 0, 0, 1)
    assert [int(m) for m in again["moves"]] == [int(m) for m in arrays["moves"]]
    assert int(again["winner"]) == int(arrays["winner"])


def test_generate_heuristic_games_small(tmp_path):
    """8 games, 2 workers: merged, valid, densely numbered, deduped
    (no duplicate move sequence survives), manifest consistent."""
    out = tmp_path / "data"
    stats = heuristic_games.generate_heuristic_games(
        out_dir=str(out), run_name="test", num_games=8,
        version_a=1, version_b=0, workers=2, seed=0,
    )
    assert stats["games_generated"] == 8
    assert stats["games_total"] <= 8
    assert stats["games_total"] >= 1
    assert stats["total_samples"] > 0
    assert "dedup" in stats
    assert stats["dedup"]["games_before"] == 8
    assert stats["dedup"]["games_after"] == stats["games_total"]
    names = sorted(p.name for p in out.glob("test_game*.npz"))
    assert names == [
        f"test_game{i:04d}.npz" for i in range(stats["games_total"])
    ]
    seqs = []
    for p in sorted(out.glob("test_game*.npz")):
        _assert_valid_shard(p)
        with np.load(p) as z:
            seqs.append(tuple(int(m) for m in z["moves"]))
    assert len(set(seqs)) == len(seqs), "duplicate sequences survived dedup"
    manifest = selfplay.load_manifest(str(out))
    assert manifest["games"] == stats["games_total"]
    assert manifest["total_samples"] == stats["total_samples"]


def test_generate_heuristic_games_deterministic(tmp_path):
    """A generated shard replays exactly from its game index and
    seeds (game 0: version A Black, seeds from _game_seeds)."""
    out = tmp_path / "data"
    heuristic_games.generate_heuristic_games(
        out_dir=str(out), run_name="test", num_games=2,
        version_a=1, version_b=0, workers=1, seed=0, dedup=False,
    )
    seed_a, seed_b = heuristic_games._game_seeds(0, 0)
    expected = heuristic_games.play_heuristic_game(1, 0, seed_a, seed_b)
    with np.load(out / "test_game0000.npz") as z:
        assert [int(m) for m in z["moves"]] == [
            int(m) for m in expected["moves"]
        ]
        assert int(z["winner"]) == int(expected["winner"])
    # Game 1: colours swapped (version A White).
    seed_a, seed_b = heuristic_games._game_seeds(0, 1)
    expected1 = heuristic_games.play_heuristic_game(0, 1, seed_b, seed_a)
    with np.load(out / "test_game0001.npz") as z:
        assert [int(m) for m in z["moves"]] == [
            int(m) for m in expected1["moves"]
        ]


def test_heuristic_games_rejects_mcts_version(tmp_path):
    """MCTS versions are too slow for this generator — fail fast."""
    import pytest

    with pytest.raises(ValueError, match="MCTS engine"):
        heuristic_games.generate_heuristic_games(
            out_dir=str(tmp_path), run_name="test", num_games=1,
            version_a=6, version_b=0, workers=1,
        )
    with pytest.raises(ValueError, match="MCTS engine"):
        heuristic_games.play_heuristic_game(7, 0, 0, 1)


def test_sampled_choose_move_never_skips_win_or_block():
    """Even at eps=1.0 (always sample), the hard rules hold: take
    an immediate win, block the opponent's single immediate win."""
    import pytest  # noqa: F401  (kept local, as in the MCTS test)

    from gomoku_ml import scoring

    rng = np.random.default_rng(0)
    v1 = scoring.get_player(1, seed=0)
    # Black has a winning move (four in a row) -> must take it.
    board = game.empty_board()
    for x, y in [(7, 7), (7, 8), (7, 9), (7, 10)]:
        board[y, x] = game.BLACK
    board[3, 3] = game.WHITE
    for _ in range(20):
        move = heuristic_games._sampled_choose_move(
            v1, board, game.BLACK, rng, eps=1.0, topk=8
        )
        assert (move % 15, move // 15) in [(7, 6), (7, 11)]
    # White's broken four: only (5,7) wins; Black must block it.
    board = game.empty_board()
    board[3, 3] = game.BLACK
    for x, y in [(5, 5), (5, 6), (5, 8), (5, 9)]:
        board[y, x] = game.WHITE
    for _ in range(20):
        move = heuristic_games._sampled_choose_move(
            v1, board, game.BLACK, rng, eps=1.0, topk=8
        )
        assert (move % 15, move // 15) == (5, 7)


def test_sampled_games_are_unique(tmp_path):
    """100 sampled games (eps=0.2, topk=8) are (near-)all unique —
    the fix for the pure-play pool collapse (75/100 unique at n=100,
    621/4000 at n=4000)."""
    out = tmp_path / "data"
    stats = heuristic_games.generate_heuristic_games(
        out_dir=str(out), run_name="test", num_games=100,
        version_a=1, version_b=0, workers=2, seed=0,
        sample_eps=0.2, sample_topk=8,
    )
    assert stats["games_generated"] == 100
    assert stats["games_total"] >= 95, (
        f"sampled games still collapsing: {stats['games_total']}/100 unique"
    )
    assert stats["sample_eps"] == 0.2
    assert stats["sample_topk"] == 8


def test_sampled_games_deterministic(tmp_path):
    """Same seeds + same sampling params replay the same game."""
    a = heuristic_games.play_heuristic_game(
        1, 0, 0, 1, sample_eps=0.2, sample_topk=8, sample_version=1
    )
    b = heuristic_games.play_heuristic_game(
        1, 0, 0, 1, sample_eps=0.2, sample_topk=8, sample_version=1
    )
    assert [int(m) for m in a["moves"]] == [int(m) for m in b["moves"]]
    assert int(a["winner"]) == int(b["winner"])
    # Sampling changes the game (vs pure play with the same seeds).
    pure = heuristic_games.play_heuristic_game(1, 0, 0, 1)
    assert [int(m) for m in a["moves"]] != [int(m) for m in pure["moves"]]


def test_sampling_validation():
    import pytest

    with pytest.raises(ValueError, match="scoring engine"):
        heuristic_games.play_heuristic_game(
            1, 0, 0, 1, sample_eps=0.2, sample_topk=8, sample_version=0
        )
    with pytest.raises(ValueError, match="sample_eps"):
        heuristic_games.play_heuristic_game(
            1, 0, 0, 1, sample_eps=1.5, sample_topk=8, sample_version=1
        )
    with pytest.raises(ValueError, match="sample_topk"):
        heuristic_games.play_heuristic_game(
            1, 0, 0, 1, sample_eps=0.2, sample_topk=0, sample_version=1
        )


def test_deduplicate_shards(tmp_path):
    """Dedup keeps the first occurrence, renumbers densely, rebuilds
    the manifest."""
    out = tmp_path / "data"
    out.mkdir()
    a = heuristic_games.play_heuristic_game(1, 0, 0, 1)
    b = heuristic_games.play_heuristic_game(1, 0, 2, 3)
    assert [int(m) for m in a["moves"]] != [int(m) for m in b["moves"]]
    np.savez_compressed(out / "test_game0000.npz", **a)
    np.savez_compressed(out / "test_game0001.npz", **b)
    np.savez_compressed(out / "test_game0002.npz", **a)  # duplicate of 0
    np.savez_compressed(out / "test_game0003.npz", **b)  # duplicate of 1
    stats = heuristic_games.deduplicate_shards(str(out), "test")
    assert stats == {
        "games_before": 4,
        "duplicates_removed": 2,
        "games_after": 2,
        "total_samples": len(a["moves"]) + len(b["moves"]),
    }
    names = sorted(p.name for p in out.glob("test_game*.npz"))
    assert names == ["test_game0000.npz", "test_game0001.npz"]
    manifest = selfplay.load_manifest(str(out))
    assert manifest["games"] == 2
    assert manifest["total_samples"] == len(a["moves"]) + len(b["moves"])
    # Survivors are the FIRST occurrences (game 0 = a, game 1 = b).
    with np.load(out / "test_game0000.npz") as z:
        assert [int(m) for m in z["moves"]] == [int(m) for m in a["moves"]]
    with np.load(out / "test_game0001.npz") as z:
        assert [int(m) for m in z["moves"]] == [int(m) for m in b["moves"]]

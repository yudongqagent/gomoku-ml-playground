"""Unit tests for the research harness (gomoku_ml/research.py).

Match counts are kept small so the suite stays fast; the real
head-to-head numbers come from dedicated runs, not these tests.
"""

import numpy as np

from gomoku_ml import baseline, evaluate, game, research, scoring


def test_play_match_shape_and_totals():
    a = scoring.ScoringPlayer(seed=1)
    b = baseline.HeuristicPlayer(seed=2)
    r = research.play_match(a.choose_move, b.choose_move, games=4)
    assert r["games"] == 4
    assert r["wins"] + r["losses"] + r["draws"] == 4
    assert r["as_black"]["wins"] + r["as_black"]["losses"] + r["as_black"]["draws"] == 2
    assert r["as_white"]["wins"] + r["as_white"]["losses"] + r["as_white"]["draws"] == 2
    assert r["swap_rule"] is False
    lo, hi = r["win_rate_ci_95"]
    assert 0.0 <= lo <= r["win_rate"] <= hi <= 1.0


def test_play_match_deterministic_with_seeded_players():
    def run():
        a = scoring.ScoringPlayer(seed=11)
        b = baseline.HeuristicPlayer(seed=12)
        return research.play_match(a.choose_move, b.choose_move, games=4)

    assert run() == run()


def test_play_match_heuristic_beats_random():
    # Sanity: the 1-ply heuristic should crush random play.
    a = baseline.HeuristicPlayer(seed=3)
    b = baseline.RandomPlayer(seed=4)
    r = research.play_match(a.choose_move, b.choose_move, games=6)
    assert r["wins"] >= 5


def test_play_match_swap_rule_shape_and_totals():
    a = scoring.ScoringPlayer(seed=5)
    b = baseline.HeuristicPlayer(seed=6)
    r = research.play_match(a.choose_move, b.choose_move, games=4, swap_rule=True)
    assert r["games"] == 4
    assert r["wins"] + r["losses"] + r["draws"] == 4
    assert r["swap_rule"] is True
    assert r["swap_choices"]["black"] + r["swap_choices"]["white"] == 4


def test_play_match_swap_rule_deterministic():
    def run():
        a = scoring.ScoringPlayer(seed=13)
        b = baseline.HeuristicPlayer(seed=14)
        return research.play_match(a.choose_move, b.choose_move, games=4, swap_rule=True)

    assert run() == run()


def test_place_swap_opening_uses_chooser():
    # The opener's chooser places all 3 stones: first move is the
    # centre for both engines on an empty board.
    board, moves = research.place_swap_opening(
        lambda b, p: scoring.choose_move(b, p)
    )
    assert game.is_swap_opening(board)
    assert moves[0] == game.xy_to_move(7, 7)  # first black stone: centre
    assert len(set(moves)) == 3


def test_play_swap_game_white_to_move_after_opening():
    # After the swap, move 4 is White's (RIF rules Art. 2.4): the
    # first chooser call on a board holding the 3 opening stones must
    # be for White. (The opener's chooser is also called during
    # placement, on boards with < 3 stones — those calls are not
    # game moves, so filter them out by stone count.)
    calls: list[tuple[int, int]] = []

    def recording_chooser(board, player):
        calls.append((player, int(np.count_nonzero(board))))
        return scoring.choose_move(board, player)

    winner, second_colour = research.play_swap_game(
        recording_chooser,
        lambda b, p: baseline.choose_move(b, p),
    )
    assert winner in (game.BLACK, game.WHITE, game.EMPTY)
    assert second_colour in (game.BLACK, game.WHITE)
    game_calls = [player for player, stones in calls if stones >= 3]
    assert game_calls, "no moves were played after the opening"
    assert game_calls[0] == game.WHITE


def test_ladder_for_player_shape():
    a = scoring.ScoringPlayer(seed=7)
    result = research.ladder_for_player(
        a.choose_move, games_per_opponent=2, mcts_sims=5, seed=8
    )
    assert set(result["matchups"]) == {"random", "heuristic", "mcts"}
    for name, r in result["matchups"].items():
        assert r["games"] == 2
        assert r["wins"] + r["losses"] + r["draws"] == 2
    elo = result["ladder_elo"]
    assert "estimate" in elo and "interval" in elo
    assert elo["anchors"] == evaluate.LADDER_ANCHORS
    # The mcts matchup must not enter the Elo estimate (unanchored).
    assert "mcts" not in elo["per_opponent"]


def test_run_tactical_shape():
    result = research.run_tactical(
        lambda b, p: scoring.choose_move(b, p)
    )
    assert result["total"] == len(evaluate.load_tactical_suite()["positions"])
    assert result["passed"] + sum(
        1 for r in result["results"] if not r["passed"]
    ) == result["total"]
    assert 0.0 <= result["pass_rate"] <= 1.0


def test_best_attack_candidates_exact():
    # best_attack (Chebyshev-4 scan) must equal the max over ALL legal
    # moves on boards with stones present (see _attack_candidates).
    rng = np.random.default_rng(42)
    for _ in range(5):
        board = game.empty_board()
        for _ in range(rng.integers(5, 40)):
            legal = game.legal_moves(board)
            m = int(rng.choice(legal))
            board = game.apply_move(board, m, int(rng.choice([-1, 1])))
        for player in (game.BLACK, game.WHITE):
            full = max(
                scoring.score_attack(board, int(m), player)
                for m in game.legal_moves(board)
            )
            assert research.best_attack(board, player) == full


def test_balanced_opening_valid_and_deterministic():
    def run():
        return research.place_swap_opening(
            lambda b, p: scoring.choose_move(b, p), placement="balanced"
        )

    board, moves = run()
    assert game.is_swap_opening(board)
    assert moves[0] == game.xy_to_move(7, 7)  # first black stone: centre
    assert len(set(moves)) == 3
    board2, moves2 = run()
    assert moves2 == moves
    assert np.array_equal(board2, board)


def test_balanced_opening_black2_minimises_gap():
    # Balanced placement picks the black2 candidate minimising the
    # attack-score gap given its own white1 stone: no other nearby
    # candidate may give a strictly smaller gap.
    rng = np.random.default_rng(7)
    board, moves = research.place_swap_opening(
        lambda b, p: scoring.choose_move(b, p),
        placement="balanced", rng=rng,
    )
    black1, white1, black2 = moves
    gap, _, _ = research.opening_gap(board)
    pre = game.empty_board()
    pre = game.apply_move(pre, black1, game.BLACK)
    pre = game.apply_move(pre, white1, game.WHITE)
    for cand in research._attack_candidates(pre, radius=2):
        trial = game.apply_move(pre, cand, game.BLACK)
        g, _, _ = research.opening_gap(trial)
        assert g >= gap


def test_balanced_swap_match_shape():
    a = scoring.ScoringPlayer(seed=21)
    b = baseline.HeuristicPlayer(seed=22)
    r = research.play_match(
        a.choose_move, b.choose_move, games=4, swap_rule=True,
        placement="balanced",
    )
    assert r["games"] == 4
    assert r["wins"] + r["losses"] + r["draws"] == 4
    assert r["swap_choices"]["black"] + r["swap_choices"]["white"] == 4

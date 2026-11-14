"""Unit tests for the swap-rule helpers in gomoku_ml/game.py."""

import numpy as np
import pytest

from gomoku_ml import baseline, game, research, scoring


def _opening(black1=(7, 7), black2=(8, 8), white1=(7, 8)):
    return game.build_swap_opening(
        game.xy_to_move(*black1), game.xy_to_move(*black2), game.xy_to_move(*white1)
    )


def test_build_swap_opening_places_stones():
    b = _opening()
    assert int(np.count_nonzero(b == game.BLACK)) == 2
    assert int(np.count_nonzero(b == game.WHITE)) == 1
    assert b[7, 7] == game.BLACK
    assert b[8, 8] == game.BLACK
    assert b[8, 7] == game.WHITE
    assert game.is_swap_opening(b)


def test_build_swap_opening_rejects_duplicates():
    with pytest.raises(ValueError):
        game.build_swap_opening(0, 0, 1)
    with pytest.raises(ValueError):
        game.build_swap_opening(0, 1, 1)


def test_build_swap_opening_rejects_out_of_range():
    with pytest.raises(ValueError):
        game.build_swap_opening(0, 1, 225)
    with pytest.raises(ValueError):
        game.build_swap_opening(-1, 1, 2)


def test_is_swap_opening_false_for_other_boards():
    assert not game.is_swap_opening(game.empty_board())
    b = game.apply_move(game.empty_board(), 0, game.BLACK)
    assert not game.is_swap_opening(b)
    # 3 black stones is not an opening either.
    b3 = game.empty_board()
    for m in (0, 1, 2):
        b3 = game.apply_move(b3, m, game.BLACK)
    assert not game.is_swap_opening(b3)


def test_choose_swap_color_picks_higher_best_score():
    # Opener placed Black at (0,0),(14,0) and White at (7,7) — but a
    # single white stone is no threat. Instead verify the mechanism
    # with a stub scorer: the colour whose best move scores higher
    # wins, whatever the board looks like.
    b = _opening()
    stub = lambda board, move, player: 10 if player == game.WHITE else 1
    assert game.choose_swap_color(b, stub) == game.WHITE
    stub2 = lambda board, move, player: 10 if player == game.BLACK else 1
    assert game.choose_swap_color(b, stub2) == game.BLACK


def test_choose_swap_color_tie_goes_to_white():
    # Ties go to White: White plays move 4 (RIF rules Art. 2.4), so
    # the next-move tempo breaks the tie.
    b = _opening()
    stub = lambda board, move, player: 5  # every move scores the same
    assert game.choose_swap_color(b, stub) == game.WHITE


def test_choose_swap_color_uses_real_scorer():
    # Black stones adjacent at the corner, White alone in the centre:
    # Black's best attack extends its pair into a three, White's best
    # attack is a lone two next to its single stone. The corner pair
    # wins the comparison, so the second player takes Black.
    b = game.build_swap_opening(
        game.xy_to_move(0, 0), game.xy_to_move(1, 0), game.xy_to_move(7, 7)
    )
    best_black = max(
        scoring.score_attack(b, int(m), game.BLACK) for m in game.legal_moves(b)
    )
    best_white = max(
        scoring.score_attack(b, int(m), game.WHITE) for m in game.legal_moves(b)
    )
    assert best_black > best_white  # the premise of the test
    assert game.choose_swap_color(b, scoring.score_attack) == game.BLACK


def test_choose_swap_color_white_wins_with_real_scorer():
    # Black's two stones are at opposite corners (no synergy — the
    # best Black attack is an edge two), White's single stone is in
    # the centre (best White attack is an open two). White wins.
    b = game.build_swap_opening(
        game.xy_to_move(0, 0), game.xy_to_move(0, 14), game.xy_to_move(7, 7)
    )
    best_black = max(
        scoring.score_attack(b, int(m), game.BLACK) for m in game.legal_moves(b)
    )
    best_white = max(
        scoring.score_attack(b, int(m), game.WHITE) for m in game.legal_moves(b)
    )
    assert best_white > best_black  # the premise of the test
    assert game.choose_swap_color(b, scoring.score_attack) == game.WHITE


def test_total_score_is_colour_symmetric():
    # The total (attack + defence) score of a square is identical
    # for both colours — which is why the swap choice must compare
    # ATTACK scores, not totals (see choose_swap_color's docstring).
    b = _opening()
    for m in game.legal_moves(b):
        assert scoring.score_move(b, int(m), game.BLACK) == scoring.score_move(
            b, int(m), game.WHITE
        )


def test_choose_swap_color_rejects_non_opening():
    with pytest.raises(ValueError):
        game.choose_swap_color(game.empty_board(), scoring.score_attack)
    b = game.apply_move(game.empty_board(), 0, game.BLACK)
    with pytest.raises(ValueError):
        game.choose_swap_color(b, scoring.score_attack)


# --- tempo-aware judge (batch 2, H3) --------------------------------------
# value(colour) = best follow-up attack after the colour's own best
# setup move (max over tied best-immediate moves); higher value wins,
# ties to White. See game.choose_swap_color_tempo and
# research/iterations/swap-judge-v2.md.


def _tempo_stub(board, move, player):
    """Stub scorer keyed on stone counts: on the 3-stone opening
    Black's best attack (100) beats White's (50), so the attack judge
    picks Black — but after each colour's own setup move White's
    follow-up (90) beats Black's (40), so the tempo judge picks
    White. Boards are distinguished by (n_black, n_white): (2, 1)
    opening, (2, 2) after White's setup, (3, 1) after Black's."""
    n_b = int(np.count_nonzero(board == game.BLACK))
    n_w = int(np.count_nonzero(board == game.WHITE))
    if (n_b, n_w) == (2, 1):
        return 100 if player == game.BLACK else 50
    if (n_b, n_w) == (2, 2):
        return 90 if player == game.WHITE else 10
    if (n_b, n_w) == (3, 1):
        return 40 if player == game.BLACK else 10
    raise AssertionError(f"unexpected board: {(n_b, n_w)}")


def test_choose_swap_color_tempo_mechanism():
    b = _opening()
    assert game.choose_swap_color(b, _tempo_stub) == game.BLACK  # attack judge
    assert game.choose_swap_color(b, _tempo_stub, judge="tempo") == game.WHITE
    assert game.choose_swap_color_tempo(b, _tempo_stub) == game.WHITE
    assert game._setup_value(b, _tempo_stub, game.BLACK) == 40
    assert game._setup_value(b, _tempo_stub, game.WHITE) == 90


def test_choose_swap_color_tempo_tie_goes_to_white():
    b = _opening()
    stub = lambda board, move, player: 5  # every value ties
    assert game.choose_swap_color(b, stub, judge="tempo") == game.WHITE
    assert game.choose_swap_color_tempo(b, stub) == game.WHITE


def test_choose_swap_color_tempo_rejects_non_opening():
    with pytest.raises(ValueError):
        game.choose_swap_color_tempo(game.empty_board(), scoring.score_attack)
    b = game.apply_move(game.empty_board(), 0, game.BLACK)
    with pytest.raises(ValueError):
        game.choose_swap_color_tempo(b, scoring.score_attack)


def test_choose_swap_color_unknown_judge_raises():
    b = _opening()
    with pytest.raises(ValueError):
        game.choose_swap_color(b, scoring.score_attack, judge="bogus")


def test_choose_swap_color_tempo_real_scorer_black():
    # First balanced opening of the seed-11 stream (black1, white1,
    # black2) = (112, 83, 51): Black's setup value (4084) beats
    # White's (4006), so both judges pick Black.
    b = game.build_swap_opening(112, 51, 83)
    assert game._setup_value(b, scoring.score_attack, game.BLACK) == 4084
    assert game._setup_value(b, scoring.score_attack, game.WHITE) == 4006
    assert game.choose_swap_color(b, scoring.score_attack) == game.BLACK
    assert game.choose_swap_color(b, scoring.score_attack, judge="tempo") == game.BLACK


def test_choose_swap_color_tempo_real_scorer_white():
    # Third balanced opening of the seed-11 stream (black1, white1,
    # black2) = (112, 140, 168): the setup values tie (4006 vs 4006),
    # so the tempo judge picks White on the tempo tie-break while the
    # attack judge still picks Black (gap 78).
    b = game.build_swap_opening(112, 168, 140)
    assert game._setup_value(b, scoring.score_attack, game.BLACK) == 4006
    assert game._setup_value(b, scoring.score_attack, game.WHITE) == 4006
    assert game.choose_swap_color(b, scoring.score_attack) == game.BLACK
    assert game.choose_swap_color(b, scoring.score_attack, judge="tempo") == game.WHITE


def test_play_swap_game_accepts_tempo_judge():
    a = scoring.ScoringPlayer(seed=21)
    b = baseline.HeuristicPlayer(seed=22)
    winner, second_colour = research.play_swap_game(
        a.choose_move, b.choose_move, placement="balanced",
        placement_rng=np.random.default_rng(5), judge="tempo",
    )
    assert winner in (game.BLACK, game.WHITE, game.EMPTY)
    assert second_colour in (game.BLACK, game.WHITE)


def test_play_match_tempo_judge_deterministic():
    def run():
        a = scoring.ScoringPlayer(seed=23)
        b = baseline.HeuristicPlayer(seed=24)
        return research.play_match(
            a.choose_move, b.choose_move, games=2, swap_rule=True,
            placement="balanced", placement_seed=3, judge="tempo",
        )

    r1, r2 = run(), run()
    assert r1 == r2
    assert r1["games"] == 2
    assert r1["wins"] + r1["losses"] + r1["draws"] == 2
    assert r1["swap_choices"]["black"] + r1["swap_choices"]["white"] == 2


def test_play_match_default_judge_is_tempo():
    # H3 decision (2026-11-13): the harness default is the tempo-aware
    # judge — a default swap match must equal an explicit tempo match.
    def run(judge):
        a = scoring.ScoringPlayer(seed=25)
        b = baseline.HeuristicPlayer(seed=26)
        kwargs = dict(games=2, swap_rule=True, placement="balanced",
                      placement_seed=4)
        if judge is not None:
            kwargs["judge"] = judge
        return research.play_match(a.choose_move, b.choose_move, **kwargs)

    assert run(None) == run("tempo")

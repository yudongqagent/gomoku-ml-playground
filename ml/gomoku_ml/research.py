"""Research harness for the Gomoku auto-research program.

Small, dependency-light tools for comparing move choosers (plain
callables ``(board, player) -> flat move``) without a neural network:

* :func:`play_match` — head-to-head matches with alternating colours,
  optionally under the swap rule;
* :func:`ladder_for_player` — the eval ladder (random, the frozen
  baseline heuristic, heuristic MCTS) for any chooser, with a ladder
  Elo estimate;
* :func:`run_tactical` — the scripted tactical suite for any chooser.

Statistics reuse :mod:`gomoku_ml.evaluate` (:func:`win_rate`,
:func:`wilson_ci`, :func:`ladder_elo`, :func:`run_tactical_suite`) so
numbers here are computed exactly the way the checkpoint evals compute
them. The ladder Elo anchors (random = -800, heuristic = 0) are
relative to THIS ladder only — never comparable to human ratings.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import baseline, evaluate, game, scoring

ChooseFn = Callable[[np.ndarray, int], int]
ScoreFn = Callable[[np.ndarray, int, int], int]


# --- single games ---------------------------------------------------------

def play_game_from(
    board: np.ndarray,
    choose_black: ChooseFn,
    choose_white: ChooseFn,
    player_to_move: int = game.BLACK,
    max_moves: int = game.SIZE * game.SIZE,
    moves_out: list[int] | None = None,
) -> int:
    """Play a game from ``board`` to completion. Returns the winner
    (``game.BLACK`` / ``game.WHITE`` / ``game.EMPTY`` for a draw).
    When ``moves_out`` is given, the moves played (flat indices, in
    order, NOT including any stones already on ``board``) are
    appended to it — the caller prepends the opening stones."""
    player = player_to_move
    for _ in range(max_moves):
        over, win = game.game_over(board)
        if over:
            return win
        chooser = choose_black if player == game.BLACK else choose_white
        move = chooser(board, player)
        if moves_out is not None:
            moves_out.append(int(move))
        board = game.apply_move(board, move, player)
        player = -player
    _, win = game.game_over(board)
    return win


def place_swap_opening(choose_fn: ChooseFn, placement: str = "chooser",
                       scorer: ScoreFn | None = None,
                       rng: np.random.Generator | None = None
                       ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Place the swap-rule opening (2 black + 1 white stones).

    ``placement="chooser"`` (default, batch-1 policy): the opener uses
    its own move chooser for all three stones, playing each colour's
    best move as it sees it — Black stone, White stone, Black stone.
    Returns ``(board, (black1, white1, black2))``.

    ``placement="balanced"`` (policy v2, iteration 2+): the opener
    places stones to BALANCE the position, as the pie rule intends —
    the chooser policy makes Black systematically stronger (the second
    player took Black 20/20 in batch 1; see
    ``research/iterations/swap-rule.md``). All three stones stay in a
    compact region (within Chebyshev distance 2 of the centre):
    Black's first stone is the centre; White's stone is drawn
    (seeded) from the cells within Chebyshev distance 2 of the
    centre, giving varied openings; Black's second stone is the
    nearby candidate that minimises the gap between the two colours'
    best attack scores on the finished opening. Ties go to the
    smaller Black best-attack, then the lowest index. The
    ``choose_fn`` is not used in balanced mode. ``scorer`` is the
    colour-asymmetric attack scorer (default:
    ``scoring.score_attack``); ``rng`` supplies the White-stone draw
    (a fresh default_rng(0) when omitted — pass an explicit RNG for
    varied openings across games).
    """
    if placement == "balanced":
        return _place_swap_opening_balanced(scorer, rng)
    if placement != "chooser":
        raise ValueError(f"unknown placement policy: {placement!r}")
    board = game.empty_board()
    black1 = choose_fn(board, game.BLACK)
    board = game.apply_move(board, black1, game.BLACK)
    white1 = choose_fn(board, game.WHITE)
    board = game.apply_move(board, white1, game.WHITE)
    black2 = choose_fn(board, game.BLACK)
    board = game.apply_move(board, black2, game.BLACK)
    return board, (black1, white1, black2)


def _attack_candidates(board: np.ndarray, radius: int = 4) -> list[int]:
    """Empty points within Chebyshev distance ``radius`` of any stone.

    Used to bound best-attack scans: a cell further than 4 from every
    stone shares no 5-window with any stone, so its attack score is
    at most 8 (1-2 per direction from empty windows), while any board
    holding stones has a best attack of at least 10 (playing next to
    a friendly stone makes a two). So the max over these candidates
    equals the max over all legal moves whenever stones are present
    (verified by test: ``test_best_attack_candidates_exact``).
    """
    stones = np.argwhere(board != game.EMPTY)
    if len(stones) == 0:
        return [int(m) for m in game.legal_moves(board)]
    near = np.zeros((game.SIZE, game.SIZE), dtype=bool)
    for y, x in stones:
        y0, y1 = max(0, y - radius), min(game.SIZE, y + radius + 1)
        x0, x1 = max(0, x - radius), min(game.SIZE, x + radius + 1)
        near[y0:y1, x0:x1] = True
    return [int(m) for m in np.flatnonzero((board == game.EMPTY) & near)]


def best_attack(board: np.ndarray, player: int,
                scorer: ScoreFn | None = None) -> int:
    """Best available attack score for ``player`` on ``board`` (max
    over legal moves of the attack-only ``scorer``). Scanned over the
    Chebyshev-4 candidates (exact — see :func:`_attack_candidates`)."""
    fn = scorer or scoring.score_attack
    cands = _attack_candidates(board, radius=4)
    if not cands:
        return 0
    return max(int(fn(board, m, player)) for m in cands)


def opening_gap(board: np.ndarray, scorer: ScoreFn | None = None
                ) -> tuple[int, int, int]:
    """``(gap, best_black, best_white)`` for a swap-opening board:
    the absolute gap between the two colours' best attack scores.
    0 means neither colour has a better next attack than the other."""
    bb = best_attack(board, game.BLACK, scorer)
    bw = best_attack(board, game.WHITE, scorer)
    return abs(bb - bw), bb, bw


def _place_swap_opening_balanced(scorer: ScoreFn | None = None,
                                 rng: np.random.Generator | None = None
                                 ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Balanced swap opening (see :func:`place_swap_opening`).

    Compact region (Chebyshev distance <= 2 of the centre) throughout:
    an unconstrained gap search finds degenerate scattered openings
    (gap 0 by isolation — no stone interacts with any other), which
    are not real Gomoku positions. White's stone is drawn (seeded)
    rather than "White's best attack move": with no White stones on
    the board every move scores ~equally, so argmax would pick an
    arbitrary edge cell.
    """
    fn = scorer or scoring.score_attack
    rng = rng if rng is not None else np.random.default_rng(0)
    board = game.empty_board()
    black1 = game.xy_to_move(7, 7)
    board = game.apply_move(board, black1, game.BLACK)
    # White's stone: seeded draw from the cells within Chebyshev
    # distance 2 of the centre (24 cells) — varied openings.
    region = [
        int(m) for m in np.flatnonzero((board == game.EMPTY))
        if max(abs(m // game.SIZE - 7), abs(m % game.SIZE - 7)) <= 2
    ]
    white1 = int(rng.choice(region))
    board = game.apply_move(board, white1, game.WHITE)
    # Black's second stone: the nearby candidate minimising the
    # attack-score gap on the finished opening.
    best: tuple[int, int, int] | None = None  # (gap, best_black, move)
    for cand in _attack_candidates(board, radius=2):
        trial = game.apply_move(board, cand, game.BLACK)
        gap, bb, _bw = opening_gap(trial, fn)
        key = (gap, bb, cand)
        if best is None or key < best:
            best = key
    assert best is not None
    black2 = best[2]
    board = game.apply_move(board, black2, game.BLACK)
    return board, (black1, white1, black2)


def _apply_opening(opening: list[int]) -> tuple[np.ndarray, int]:
    """Apply a fixed opening (flat moves, Black first, alternating)
    to an empty board. Returns ``(board, player_to_move)``."""
    board = game.empty_board()
    player = game.BLACK
    for move in opening:
        board = game.apply_move(board, move, player)
        player = -player
    return board, player


def play_swap_game(
    choose_opener: ChooseFn,
    choose_second: ChooseFn,
    swap_scorer: ScoreFn | None = None,
    max_moves: int = game.SIZE * game.SIZE,
    placement: str = "chooser",
    placement_rng: np.random.Generator | None = None,
    judge: str = "tempo",
    fixed_opening: tuple[int, int, int] | None = None,
    moves_out: list[int] | None = None,
) -> tuple[int, int]:
    """Play one game under the swap rule.

    The opener places the 3 opening stones (:func:`place_swap_opening`
    with the given ``placement`` policy); the second player chooses a
    colour (:func:`gomoku_ml.game.choose_swap_color` with ``judge`` —
    ``"tempo"`` (default) uses the tempo-aware symmetric-setup judge,
    ``"attack"`` compares best attack scores; see
    ``research/iterations/swap-judge-v2.md``); then White moves (move
    4 is White's — the 3 opening stones are moves 1-3, so the game
    simply continues alternating; RIF rules Art. 2.4). Returns
    ``(winner, second_player_colour)`` — the winner is ``game.BLACK``
    / ``game.WHITE`` / ``game.EMPTY`` (draw), and the second element
    records which colour the second player took.

    ``fixed_opening`` (evaluation audit): a 3-tuple of flat moves
    ``(black1, white1, black2)`` in play order, taken from a curated
    swap-openings set instead of the placement generator — use it to
    rotate through a fixed set of openings so the swap choice is a
    real 50/50 decision (see ``research/openings-swap.json``). When
    given, ``placement``/``placement_rng`` are ignored.
    """
    scorer = swap_scorer or scoring.score_attack
    if fixed_opening is not None:
        black1, white1, black2 = fixed_opening
        board = game.build_swap_opening(black1, black2, white1)
        opening_moves = [black1, white1, black2]
    else:
        board, _moves = place_swap_opening(
            choose_opener, placement, scorer, placement_rng
        )
        opening_moves = list(_moves)
    second_colour = game.choose_swap_color(board, scorer, judge=judge)
    if second_colour == game.BLACK:
        choose_black, choose_white = choose_second, choose_opener
    else:
        choose_black, choose_white = choose_opener, choose_second
    played: list[int] = []
    winner = play_game_from(board, choose_black, choose_white, game.WHITE,
                            max_moves, moves_out=played)
    if moves_out is not None:
        moves_out.extend(opening_moves + played)
    return winner, second_colour


# --- matches --------------------------------------------------------------

def _empty_slot() -> dict:
    return {"wins": 0, "losses": 0, "draws": 0}


def play_match(
    choose_a: ChooseFn,
    choose_b: ChooseFn,
    games: int,
    swap_rule: bool = False,
    seed: int = 0,
    swap_scorer: ScoreFn | None = None,
    max_moves: int = game.SIZE * game.SIZE,
    placement: str = "chooser",
    placement_seed: int = 0,
    judge: str = "tempo",
    openings: list[list[int]] | None = None,
    swap_openings: list[list[int]] | None = None,
    record_games: bool = False,
) -> dict:
    """Play ``games`` games between choosers A and B, alternating
    colours (game ``i`` is Black for A when ``i`` is even; with an
    odd count Black gets the extra game — use an even count for exact
    balance). Results are from A's perspective.

    With ``swap_rule=True`` the opener alternates instead (game ``i``
    is opened by A when ``i`` is even); the second player chooses a
    colour via :func:`gomoku_ml.game.choose_swap_color` with
    ``swap_scorer`` (default: the v1 attack-only scorer
    :func:`gomoku_ml.scoring.score_attack` — see its docstring for
    why the scorer must be colour-asymmetric) and ``judge``
    (``"tempo"`` (default) or ``"attack"``; see :func:`play_swap_game`). The
    ``as_black`` / ``as_white`` breakdown then reflects the colours
    the players actually ended up with, which need not be balanced —
    the swap choice decides them. ``placement`` selects the opening
    policy (see :func:`place_swap_opening`); ``"balanced"`` is the
    recommended policy for research batches, with ``placement_seed``
    seeding its White-stone draw (deterministic per seed).

    ``openings`` (evaluation audit): for normal-rules matches, a
    list of fixed openings (each a list of flat moves, Black first)
    rotated across games in PAIRS — games ``2k`` and ``2k+1`` use
    the SAME opening (``openings[(i // 2) % len(openings)]``), so
    each opening is played once with A as Black and once with A as
    White within a leg. The old scheme (``openings[i %
    len(openings)]``) confounded colour and opening whenever the
    openings count is even: with 24 openings and a 40-game leg,
    opening ``j`` was used in games ``j`` and ``j+24`` — the same
    parity, so A played the same colour in both, and each opening
    was only ever played by A in one colour. Paired rotation
    requires the openings count to divide the leg length evenly by
    2: the balanced set has 20 openings, so each appears exactly
    twice per 40-game leg (once per colour); in a 100-game match
    each appears 4 or 6 times, always colour-balanced (the 40-game
    pairing cycle repeats 2.5 times). Starting every game from the empty board lets
    deterministic engines replay the same lines (v63 clause-1: 31
    unique games in 100), so the nominal CI overstates the
    precision; varied openings are now the default for every
    battery. ``swap_openings`` is the same idea for swap matches: a
    list of 3-move openings ``[black1, white1, black2]`` (flat
    moves, play order) rotated across games — game ``i`` uses
    ``swap_openings[i % len(swap_openings)]`` (UNCHANGED: the swap
    judge, not the rotation, decides the colours, and the curated
    set already yields exact 20/20 colour splits). The curated set
    (``research/openings-swap.json``) mixes tempo-margin-0 and
    tempo-margin-78 openings 50/50, so the judge's pick is a real
    choice instead of the 88%-Black pick the uncurated placement
    produced. When ``swap_openings`` is given, ``placement`` /
    ``placement_seed`` are ignored.

    ``record_games``: when True the result also carries a
    ``"records"`` list (one entry per game: moves, winner, colours,
    opening) so the ladder's variety is auditable — ladder legs
    previously stored only summaries, so their uniqueness was
    unverifiable.

    ``seed`` is accepted for interface symmetry with
    :func:`gomoku_ml.evaluate.run_ladder`; the harness itself uses no
    randomness — reproducibility comes from the choosers' own seeds.

    Returns W/L/D overall and per colour, the win rate (draws count
    as half a win) with its Wilson 95% CI, and — for swap matches —
    how often the second player took each colour.
    """
    wins = losses = draws = 0
    by_colour = {"black": _empty_slot(), "white": _empty_slot()}
    swap_choices = {"black": 0, "white": 0}
    placement_rng = np.random.default_rng(placement_seed)
    records: list[dict] = []
    for i in range(games):
        opening_moves: list[int] = []
        # ``full`` is the complete move list (opening stones first)
        # for the record; ``played`` is the moves made after the
        # opening. play_swap_game's moves_out already includes the
        # opening stones, play_game_from's does not.
        full: list[int] = []
        played: list[int] = []
        second_colour = None
        if swap_rule:
            a_opens = (i % 2 == 0)
            fixed = None
            if swap_openings is not None:
                fixed = tuple(swap_openings[i % len(swap_openings)])
            if a_opens:
                winner, second_colour = play_swap_game(
                    choose_a, choose_b, swap_scorer, max_moves, placement,
                    placement_rng, judge, fixed_opening=fixed,
                    moves_out=played,
                )
                a_colour = game.WHITE if second_colour == game.BLACK else game.BLACK
            else:
                winner, second_colour = play_swap_game(
                    choose_b, choose_a, swap_scorer, max_moves, placement,
                    placement_rng, judge, fixed_opening=fixed,
                    moves_out=played,
                )
                a_colour = second_colour
            swap_choices["black" if second_colour == game.BLACK else "white"] += 1
            full = list(played)
            opening_moves = full[:3]
        else:
            a_is_black = (i % 2 == 0)
            a_colour = game.BLACK if a_is_black else game.WHITE
            if openings is not None:
                # Paired rotation: games 2k/2k+1 share opening k, so
                # each opening is played once per colour per leg (see
                # the docstring — the old i % len scheme confounded
                # colour and opening).
                opening_moves = list(openings[(i // 2) % len(openings)])
                board, player_to_move = _apply_opening(opening_moves)
            else:
                board, player_to_move = game.empty_board(), game.BLACK
            if a_is_black:
                winner = play_game_from(
                    board, choose_a, choose_b, player_to_move, max_moves,
                    moves_out=played,
                )
            else:
                winner = play_game_from(
                    board, choose_b, choose_a, player_to_move, max_moves,
                    moves_out=played,
                )
            full = [int(m) for m in opening_moves] + played
        slot = by_colour["black" if a_colour == game.BLACK else "white"]
        if winner == game.EMPTY:
            draws += 1
            slot["draws"] += 1
        elif winner == a_colour:
            wins += 1
            slot["wins"] += 1
        else:
            losses += 1
            slot["losses"] += 1
        if record_games:
            rec = {
                "game": i,
                "opening": [int(m) for m in opening_moves],
                "moves": [int(m) for m in full],
                "a_colour": "black" if a_colour == game.BLACK else "white",
                "winner": ("black" if winner == game.BLACK
                           else "white" if winner == game.WHITE else "draw"),
                "winner_is_a": ((winner == a_colour)
                                if winner != game.EMPTY else None),
            }
            if swap_rule:
                # The colour the second player picked — the swap
                # fairness audit field (the pick distribution is the
                # judge's behaviour; a_colour is what A ended up
                # playing).
                rec["second_colour"] = (
                    "black" if second_colour == game.BLACK else "white")
            records.append(rec)
    lo, hi = evaluate.wilson_ci(wins, losses, draws)
    result = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
        "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
        "as_black": by_colour["black"],
        "as_white": by_colour["white"],
        "swap_rule": swap_rule,
        "elo_diff_vs_opponent": evaluate.elo_diff(wins, losses, draws),
    }
    if swap_rule:
        result["swap_choices"] = swap_choices
    if record_games:
        result["records"] = records
    return result


# --- ladder ---------------------------------------------------------------

#: Ladder opponent order (the dict order in the original
#: ``ladder_for_player``). Keep this order: the player's seeded RNG
#: advances across legs (tie-breaks only), so leg order is part of
#: the measurement — a leg played after other legs is only identical
#: to the uninterrupted ladder when the player's RNG state is the
#: same (see ``eval_version.py --out``).
LADDER_OPPONENTS = ("random", "heuristic", "mcts")


def ladder_opponent(name: str, mcts_sims: int = 100, seed: int = 0):
    """Build one ladder opponent by name (see
    :func:`ladder_for_player`)."""
    if name == "random":
        return baseline.RandomPlayer(seed=seed)
    if name == "heuristic":
        return baseline.HeuristicPlayer(seed=seed)
    if name == "mcts":
        return baseline.HeuristicMCTSPlayer(sims=mcts_sims, seed=seed)
    raise ValueError(f"unknown ladder opponent: {name!r}")


def ladder_leg_for_player(
    choose_fn: ChooseFn,
    opponent: str,
    games_per_opponent: int = 10,
    mcts_sims: int = 100,
    seed: int = 0,
    openings: list[list[int]] | None = None,
    record_games: bool = True,
) -> dict:
    """Play one ladder leg (``games_per_opponent`` games, alternating
    colours) against the named ladder opponent. The opponent is
    constructed fresh with ``seed`` (as in
    :func:`ladder_for_player`), so a leg is reproducible on its own
    — EXCEPT that the player's own seeded RNG advances across legs,
    so a leg played after other legs only matches the uninterrupted
    ladder when the player's RNG state is restored to the state it
    had after the previous legs (``eval_version.py --out`` saves and
    restores it).

    ``openings`` (evaluation audit): fixed openings rotated across
    the leg's games (see :func:`play_match`) — the ladder used to
    start every game from the empty board, which lets deterministic
    engines replay the same lines. ``record_games`` stores the
    per-game move lists in the leg result so the leg's uniqueness is
    auditable (previously only summaries were stored).
    """
    opp = ladder_opponent(opponent, mcts_sims=mcts_sims, seed=seed)
    return play_match(choose_fn, opp.choose_move, games_per_opponent,
                      seed=seed, openings=openings,
                      record_games=record_games)


def ladder_for_player(
    choose_fn: ChooseFn,
    games_per_opponent: int = 10,
    mcts_sims: int = 100,
    seed: int = 0,
    openings: list[list[int]] | None = None,
    record_games: bool = True,
) -> dict:
    """Run the eval ladder for a chooser: ``games_per_opponent`` games
    (alternating colours) against each of ``random``, the ORIGINAL
    baseline ``heuristic`` (the frozen ladder anchor), and the
    heuristic ``mcts`` searcher (``mcts_sims`` simulations per move).

    Returns per-opponent W/L/D + Wilson 95% CI plus a ladder Elo
    estimate (:func:`gomoku_ml.evaluate.ladder_elo`, anchored on
    random/heuristic only — the mcts matchup is reported but does not
    enter the estimate, since its strength on this scale is unknown).
    ``openings`` rotates fixed openings across every leg's games;
    ``record_games`` stores per-game move lists in each leg result.
    """
    matchups: dict[str, dict] = {}
    for name in LADDER_OPPONENTS:
        matchups[name] = ladder_leg_for_player(
            choose_fn, name, games_per_opponent, mcts_sims, seed,
            openings=openings, record_games=record_games,
        )
    return {
        "matchups": matchups,
        "ladder_elo": evaluate.ladder_elo(matchups),
        "games_per_opponent": games_per_opponent,
        "mcts_sims": mcts_sims,
    }


# --- tactical suite -------------------------------------------------------

def run_tactical(choose_fn: ChooseFn, suite: dict | None = None) -> dict:
    """Run the scripted tactical suite
    (:func:`gomoku_ml.evaluate.run_tactical_suite`) with a move
    chooser. Returns per-position pass/fail plus the total pass rate.
    """
    return evaluate.run_tactical_suite(choose_fn, suite)

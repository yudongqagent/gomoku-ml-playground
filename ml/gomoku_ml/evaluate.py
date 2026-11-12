"""Evaluation harness: ladder matches, tactical suite, ratings, verdicts.

What this module does
---------------------
* **Ladder matches** (:func:`run_ladder`): play a checkpoint (via MCTS)
  against ``random``, the 1-ply ``heuristic`` baseline, a heuristic
  **MCTS** opponent (``mcts`` — PUCT search with heuristic priors, no
  net), and optionally a previous ``checkpoint``. Every matchup plays
  both colours in equal numbers. Each matchup reports W/L/D, the win
  rate (draws count as half a win), and a Wilson-score 95% confidence
  interval — never a bare percentage.
* **Ladder Elo** (:func:`ladder_elo`): a rating estimate anchored to
  the ladder itself (``random = -800``, ``heuristic = 0``). The anchors
  are ARBITRARY reference points, not real Elo: the number is only
  meaningful relative to this ladder, and every report says so. With
  few games the honest answer is an interval, not a point — the
  estimate is reported with its Wilson-derived interval.
* **Tactical suite** (:func:`run_tactical_suite`): ~20 scripted
  positions (``tactical_positions.json``) covering immediate wins,
  blocks, double threats, the only defence, and forced VCF sequences.
  The checkpoint's MCTS and the heuristic baseline both run the suite,
  so reports show a comparison.
* **Verdict** (:func:`verdict_for`): a plain-language tier computed
  from the numbers against the "good level" definition in ml/README.md
  (v1: >= 50% vs the heuristic-MCTS opponent over >= 40 games as both
  colours, AND >= 80% tactical pass rate). Verdicts are computed, not
  hand-written per run.

Honesty note: beating the random and heuristic baselines says almost
nothing about absolute strength. The heuristic player is weak, the
heuristic-MCTS player is a modest searcher, and the ladder Elo is a
relative number on an arbitrary scale. A serious strength claim needs
matches against strong engines or rated human play, which this ladder
does not provide.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from typing import Callable

import numpy as np

from . import baseline, game
from .mcts import MCTS

TACTICAL_PATH = os.path.join(os.path.dirname(__file__), "tactical_positions.json")

# --- Ladder Elo anchors ---------------------------------------------------
# ARBITRARY reference points, chosen so the ladder reads like a rating
# list. They are NOT real Elo ratings and must never be compared with
# human ratings, online ratings, or ratings from any other ladder.
LADDER_ANCHORS: dict[str, int] = {
    "random": -800,
    "heuristic": 0,
}

ELO_CAVEAT = (
    "Ladder Elo is relative to THIS ladder only (anchors: random = -800, "
    "heuristic = 0 — arbitrary reference points, not real Elo). It is not "
    "comparable to human ratings, online ratings, or any other system. "
    "With few games the interval is wide; treat the point estimate as a "
    "rough centre, not a precise rating."
)

# --- "Good level" definition (v1, see ml/README.md) -----------------------
# A checkpoint has reached a "good level" when it meets ALL of these.
GOOD_LEVEL_MIN_MCTS_WIN_RATE = 0.50
GOOD_LEVEL_MIN_MCTS_GAMES = 40
GOOD_LEVEL_MIN_TACTICAL_PASS_RATE = 0.80

# Verdict tier thresholds (win rates are point estimates; the report
# always shows the confidence interval alongside).
VERDICT_BEATS_HEURISTIC = 0.60   # clearly above heuristic
VERDICT_AROUND_HEURISTIC = 0.40  # below this, clearly below heuristic

DISCLAIMER = (
    "This report measures a checkpoint against a small ladder of weak "
    "opponents (random play, a 1-ply heuristic, a heuristic searcher) "
    "and a scripted tactical suite. It shows relative progress, not "
    "absolute strength. " + ELO_CAVEAT
)


# --- statistics -----------------------------------------------------------

def win_rate(wins: int, losses: int, draws: int) -> float:
    """Score rate with draws counting as half a win. 0.0 with no games."""
    n = wins + losses + draws
    if n == 0:
        return 0.0
    return (wins + 0.5 * draws) / n


def wilson_ci(
    wins: int, losses: int, draws: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson-score 95% confidence interval for the win rate (draws
    count as half a win). Returns (lo, hi) in [0, 1]. With no games
    the interval is (0, 1): no information at all."""
    n = wins + losses + draws
    if n == 0:
        return (0.0, 1.0)
    p = (wins + 0.5 * draws) / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def elo_diff(wins: int, losses: int, draws: int) -> float | None:
    """Rough Elo difference from W/L/D. None if either side has no
    decisive games (the estimate would be infinite)."""
    if wins == 0 or losses == 0:
        return None
    score = (wins + 0.5 * draws) / (wins + losses + draws)
    if score <= 0 or score >= 1:
        return None
    return 400 * math.log10(score / (1 - score))


def _elo_from_score(anchor: float, score: float) -> float | None:
    """Rating implied by ``score`` against an opponent rated ``anchor``
    under the logistic formula. None for score 0 or 1 (unbounded)."""
    if score <= 0 or score >= 1:
        return None
    return anchor + 400 * math.log10(score / (1 - score))


def ladder_elo(matchups: dict) -> dict:
    """Estimate the checkpoint's ladder Elo from its matchups against
    ANCHORED opponents (random, heuristic).

    Returns ``{"estimate", "interval": [lo, hi], "per_opponent", ...}``:

    * per anchored matchup, the point estimate from the observed score
      (when the score is strictly between 0 and 1) and the Wilson
      interval translated into a rating interval;
    * the headline estimate is the games-weighted mean of the finite
      per-opponent estimates, or the midpoint of the combined interval
      when no finite estimate exists (e.g. all wins vs random, all
      losses vs heuristic — the honest answer is "between them");
    * the combined interval is [max of lower bounds, min of upper
      bounds] — empty (None ends) when the data does not bound it.
      When the per-opponent bounds do not overlap (e.g. 40-0 vs random
      and 0-40 vs heuristic — the two constraints are each one-sided
      and marginally inconsistent), the interval is reported as the
      span between them with ``interval_inconsistent: True``: the
      single-rating logistic model does not perfectly fit a player
      whose strength has a cliff between the anchors, so the span is
      a rough range, not a strict confidence interval.

    Unanchored opponents (mcts, checkpoint) do not enter the estimate:
    their strength on this scale is unknown, so anchoring them would
    invent information.
    """
    per_opponent: dict[str, dict] = {}
    estimates: list[tuple[float, int]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for name, result in matchups.items():
        if name not in LADDER_ANCHORS:
            continue
        anchor = LADDER_ANCHORS[name]
        wins = int(result.get("wins", 0))
        losses = int(result.get("losses", 0))
        draws = int(result.get("draws", 0))
        n = wins + losses + draws
        if n == 0:
            continue
        score = win_rate(wins, losses, draws)
        lo, hi = wilson_ci(wins, losses, draws)
        entry: dict = {
            "anchor": anchor,
            "games": n,
            "score": round(score, 4),
            "score_ci_95": [round(lo, 4), round(hi, 4)],
            "estimate": _elo_from_score(anchor, score),
            "lower_bound": _elo_from_score(anchor, lo) if lo > 0 else None,
            "upper_bound": _elo_from_score(anchor, hi) if hi < 1 else None,
        }
        if entry["estimate"] is not None:
            estimates.append((entry["estimate"], n))
        if entry["lower_bound"] is not None:
            lower_bounds.append(entry["lower_bound"])
        if entry["upper_bound"] is not None:
            upper_bounds.append(entry["upper_bound"])
        per_opponent[name] = entry
    lo_bound = max(lower_bounds) if lower_bounds else None
    hi_bound = min(upper_bounds) if upper_bounds else None
    if estimates:
        total_games = sum(n for _, n in estimates)
        estimate = sum(e * n for e, n in estimates) / total_games
    elif lo_bound is not None and hi_bound is not None:
        estimate = (lo_bound + hi_bound) / 2
    else:
        estimate = None
    inconsistent = (
        lo_bound is not None and hi_bound is not None and lo_bound > hi_bound
    )
    if inconsistent:
        # The per-opponent bounds do not overlap: report the span
        # between them (sorted) and flag it — see the docstring.
        lo_bound, hi_bound = hi_bound, lo_bound
    return {
        "estimate": round(estimate, 1) if estimate is not None else None,
        "interval": [
            round(lo_bound, 1) if lo_bound is not None else None,
            round(hi_bound, 1) if hi_bound is not None else None,
        ],
        "interval_inconsistent": inconsistent,
        "anchors": dict(LADDER_ANCHORS),
        "per_opponent": per_opponent,
        "caveat": ELO_CAVEAT,
    }


# --- ladder matches -------------------------------------------------------

def play_match_game(
    mcts_player: MCTS,
    opponent,
    mcts_is_black: bool,
    num_simulations: int,
    max_moves: int = game.SIZE * game.SIZE,
) -> int:
    """Play one game. Returns +1 if the MCTS player wins, -1 if it loses,
    0 for a draw."""
    board = game.empty_board()
    history: list[np.ndarray] = []
    last_move: int | None = None
    player = game.BLACK
    mcts_colour = game.BLACK if mcts_is_black else game.WHITE
    for _ in range(max_moves):
        over, win = game.game_over(board)
        if over:
            if win == game.EMPTY:
                return 0
            return 1 if win == mcts_colour else -1
        if player == mcts_colour:
            root, _ = mcts_player.search(
                board, player, num_simulations,
                history=history, last_move=last_move, add_noise=False,
            )
            move = MCTS.choose_move(root, temperature=0.0)
        else:
            move = opponent.choose_move(board, player)
        history.append(board)
        board = game.apply_move(board, move, player)
        last_move = move
        player = -player
    _, win = game.game_over(board)
    if win == game.EMPTY:
        return 0
    return 1 if win == mcts_colour else -1


class _MCTSOpponent:
    """Adapter so an MCTS player can be used as a plain opponent.

    Note: history is not tracked across moves here, so the opponent's
    history planes are empty — acceptable for a checkpoint-vs-checkpoint
    sanity check, not for a rigorous comparison.
    """

    def __init__(self, mcts: MCTS, num_simulations: int) -> None:
        self.mcts = mcts
        self.num_simulations = num_simulations

    def choose_move(self, board: np.ndarray, player: int) -> int:
        root, _ = self.mcts.search(
            board, player, self.num_simulations, add_noise=False
        )
        return MCTS.choose_move(root, temperature=0.0)


def run_ladder(
    mcts_player: MCTS,
    num_simulations: int,
    games_per_opponent: int = 10,
    opponents: tuple[str, ...] = ("random", "heuristic"),
    opponent_mcts: MCTS | None = None,
    opponent_simulations: int = 0,
    mcts_sims: int = 100,
    max_moves: int = game.SIZE * game.SIZE,
    seed: int = 0,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Run the eval ladder. ``games_per_opponent`` is the TOTAL games
    per opponent, split as evenly as possible across colours (game i
    is Black for the evaluated model when i is even; with an odd count
    Black gets the extra game — use an even count for exact balance).

    Returns a dict of per-opponent results, each with W/L/D overall
    and per colour, the win rate with its Wilson 95% CI, and the rough
    Elo difference vs the opponent (kept for backward compatibility
    with the server, which reads ``elo_diff_vs_opponent``).
    Human-readable lines go to stderr; ``on_event`` receives
    ``{"event": "eval_result", ...}`` per opponent.
    """
    results: dict = {}
    for name in opponents:
        if name == "random":
            opp = baseline.RandomPlayer(seed=seed)
        elif name == "heuristic":
            opp = baseline.HeuristicPlayer(seed=seed)
        elif name == "mcts":
            opp = baseline.HeuristicMCTSPlayer(sims=mcts_sims, seed=seed)
        elif name == "checkpoint":
            if opponent_mcts is None:
                raise ValueError("opponent_mcts required for 'checkpoint' opponent")
            opp = _MCTSOpponent(opponent_mcts, opponent_simulations)
        else:
            raise ValueError(f"unknown opponent: {name}")
        wins = losses = draws = 0
        by_colour = {
            "black": {"wins": 0, "losses": 0, "draws": 0},
            "white": {"wins": 0, "losses": 0, "draws": 0},
        }
        for i in range(games_per_opponent):
            as_black = (i % 2 == 0)
            outcome = play_match_game(
                mcts_player, opp,
                mcts_is_black=as_black,
                num_simulations=num_simulations,
                max_moves=max_moves,
            )
            slot = by_colour["black" if as_black else "white"]
            if outcome > 0:
                wins += 1
                slot["wins"] += 1
            elif outcome < 0:
                losses += 1
                slot["losses"] += 1
            else:
                draws += 1
                slot["draws"] += 1
        lo, hi = wilson_ci(wins, losses, draws)
        results[name] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "games": games_per_opponent,
            "win_rate": round(win_rate(wins, losses, draws), 4),
            "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
            "as_black": by_colour["black"],
            "as_white": by_colour["white"],
            "elo_diff_vs_opponent": elo_diff(wins, losses, draws),
        }
        print(
            f"  vs {name}: {wins}W-{losses}L-{draws}D "
            f"(win rate {results[name]['win_rate']:.3f}, "
            f"95% CI [{lo:.3f}, {hi:.3f}])",
            file=sys.stderr, flush=True,
        )
        if on_event is not None:
            on_event({"event": "eval_result", "opponent": name, **results[name]})
    return results


# --- tactical suite -------------------------------------------------------

def load_tactical_suite(path: str | None = None) -> dict:
    """Load the scripted tactical positions (JSON)."""
    with open(path or TACTICAL_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def board_from_position(pos: dict) -> np.ndarray:
    """Build the board for a tactical position from its stone lists."""
    board = game.empty_board()
    for x, y in pos.get("black", []):
        board[y, x] = game.BLACK
    for x, y in pos.get("white", []):
        board[y, x] = game.WHITE
    return board


def winning_moves(board: np.ndarray, player: int) -> list[int]:
    """All immediate winning moves for ``player``. A winning move must
    be adjacent to a friendly stone, so only those points are scanned."""
    out: list[int] = []
    seen: set[int] = set()
    for y, x in zip(*((board == player).nonzero())):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    m = game.xy_to_move(xx, yy)
                    if m not in seen and board[yy, xx] == game.EMPTY:
                        seen.add(m)
                        if game.is_winning_move(board, m, player):
                            out.append(m)
    return out


def _near_moves(board: np.ndarray, player: int) -> list[int]:
    """Empty points within Chebyshev distance 2 of any ``player``
    stone. A move further away cannot create a four or a threat."""
    near: set[int] = set()
    for y, x in zip(*((board == player).nonzero())):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return sorted(near)


def four_creating_moves(board: np.ndarray, player: int) -> list[int]:
    """Moves after which ``player`` has at least one winning move next
    turn (i.e. the move creates a four with an open end)."""
    out: list[int] = []
    for m in _near_moves(board, player):
        b2 = game.apply_move(board, m, player)
        if winning_moves(b2, player):
            out.append(m)
    return out


def forces_win(board: np.ndarray, attacker: int, attacker_moves_left: int) -> bool:
    """True if ``attacker`` (to move) forces a win by consecutive
    fours within ``attacker_moves_left`` of their own moves.

    Each attacker move must create a four; the defender's only replies
    are blocks of the attacker's winning square (anything else loses
    at once). An open four (two winning squares) ends the search: the
    defender cannot block both. Defender immediate wins (including a
    block that completes five for the defender) refute the line.
    """
    if winning_moves(board, attacker):
        return True
    if attacker_moves_left == 0:
        return False
    for move in four_creating_moves(board, attacker):
        b2 = game.apply_move(board, move, attacker)
        if winning_moves(b2, -attacker):
            continue  # defender wins at once instead of blocking
        threats = winning_moves(b2, attacker)
        if len(threats) >= 2:
            return True  # open four: unblockable
        if len(threats) == 1:
            b3 = game.apply_move(b2, threats[0], -attacker)
            over, _ = game.game_over(b3)
            if over:
                continue  # the block itself won for the defender
            if forces_win(b3, attacker, attacker_moves_left - 1):
                return True
    return False


def starts_forced_win(
    board: np.ndarray, player: int, move: int, depth: int = 3
) -> bool:
    """True if ``move`` starts a forced win by consecutive fours: the
    move itself must create a four (an immediate win or an open four
    counts), and after the defender's forced block the attacker still
    forces a win within ``depth`` more of their own moves.

    Deliberately narrow: a move that does not create a four (e.g. a
    subtle double-threat setup) does not count, even if it happens to
    win — the VCF positions test recognition of forcing sequences,
    and the four-creating requirement is what makes the check fast
    and unambiguous.
    """
    if not game.is_legal(board, move):
        return False
    if game.is_winning_move(board, move, player):
        return True
    b2 = game.apply_move(board, move, player)
    if winning_moves(b2, -player):
        return False  # defender wins at once instead of blocking
    threats = winning_moves(b2, player)
    if len(threats) >= 2:
        return True  # open four: unblockable
    if len(threats) == 1:
        b3 = game.apply_move(b2, threats[0], -player)
        over, _ = game.game_over(b3)
        if over:
            return False
        return forces_win(b3, player, depth)
    return False


def check_tactical_move(
    board: np.ndarray, to_move: int, move: int, expect: dict
) -> bool:
    """True if ``move`` satisfies a position's expectation.

    * ``{"type": "win"}`` — the move wins immediately.
    * ``{"type": "moves", "moves": [[x, y], ...]}`` — the move is one
      of the listed squares.
    * ``{"type": "vcf"}`` — the move starts a forced win by
      consecutive fours (VCF search, see :func:`starts_forced_win`).
    """
    if expect["type"] == "win":
        return game.is_winning_move(board, move, to_move)
    if expect["type"] == "moves":
        expected = {game.xy_to_move(x, y) for x, y in expect["moves"]}
        return move in expected
    if expect["type"] == "vcf":
        return starts_forced_win(board, to_move, move)
    raise ValueError(f"unknown tactical expectation type: {expect['type']!r}")


def run_tactical_suite(
    choose_move: Callable[[np.ndarray, int], int],
    suite: dict | None = None,
) -> dict:
    """Run the tactical suite with a move chooser
    (``choose_move(board, player) -> flat move``). Returns per-position
    pass/fail plus the total pass rate."""
    suite = suite or load_tactical_suite()
    positions = suite["positions"]
    results: list[dict] = []
    passed = 0
    for pos in positions:
        board = board_from_position(pos)
        to_move = pos["to_move"]
        move = choose_move(board, to_move)
        ok = check_tactical_move(board, to_move, move, pos["expect"])
        passed += 1 if ok else 0
        x, y = game.move_to_xy(move)
        results.append({
            "id": pos["id"],
            "name": pos["name"],
            "category": pos["category"],
            "passed": ok,
            "chosen": [x, y],
            "expect": pos["expect"],
        })
    return {
        "total": len(positions),
        "passed": passed,
        "pass_rate": round(passed / len(positions), 4) if positions else 0.0,
        "results": results,
    }


def mcts_tactical_chooser(mcts_player: MCTS, num_simulations: int):
    """Build a deterministic tactical-suite chooser from an MCTS player
    (temperature 0, no Dirichlet noise)."""

    def choose(board: np.ndarray, player: int) -> int:
        root, _ = mcts_player.search(
            board, player, num_simulations, add_noise=False
        )
        return MCTS.choose_move(root, temperature=0.0)

    return choose


# --- verdict --------------------------------------------------------------

def verdict_for(ladder: dict, tactical: dict | None) -> dict:
    """Compute the plain-language verdict tier from the numbers.

    Tiers (see the "good level" definition in ml/README.md):

    * ``below heuristic`` — win rate vs heuristic < 40%.
    * ``around heuristic`` — win rate vs heuristic 40–60%.
    * ``beats heuristic`` — win rate vs heuristic >= 60%, but the
      search (mcts) matchup was not run.
    * ``beats heuristic, below search`` — beats the heuristic but wins
      < 50% vs the heuristic-MCTS opponent.
    * ``challenges search`` — beats the heuristic and wins >= 50% vs
      the search opponent, but does not yet meet the full "good level"
      definition (too few games vs search, or tactical pass rate below
      80%).
    * ``good level (v1)`` — meets the full definition: >= 50% vs the
      heuristic-MCTS opponent over >= 40 games as both colours AND
      >= 80% tactical pass rate.
    * ``unevaluated`` — the heuristic matchup was not run, so the
      checkpoint cannot be placed on the ladder at all.
    """
    thresholds = {
        "beats_heuristic_win_rate": VERDICT_BEATS_HEURISTIC,
        "around_heuristic_win_rate": VERDICT_AROUND_HEURISTIC,
        "good_level_min_mcts_win_rate": GOOD_LEVEL_MIN_MCTS_WIN_RATE,
        "good_level_min_mcts_games": GOOD_LEVEL_MIN_MCTS_GAMES,
        "good_level_min_tactical_pass_rate": GOOD_LEVEL_MIN_TACTICAL_PASS_RATE,
    }
    heu = ladder.get("heuristic")
    if heu is None:
        return {
            "tier": "unevaluated",
            "label": "Unevaluated — heuristic matchup not run",
            "meets_good_level": False,
            "explanation": (
                "The heuristic matchup was not run, so this checkpoint "
                "cannot be placed on the ladder. Re-run with 'heuristic' "
                "in --opponents."
            ),
            "thresholds": thresholds,
        }
    wr_h = win_rate(heu["wins"], heu["losses"], heu["draws"])
    if wr_h < VERDICT_AROUND_HEURISTIC:
        return {
            "tier": "below heuristic",
            "label": "Below heuristic",
            "meets_good_level": False,
            "explanation": (
                f"Win rate vs the 1-ply heuristic baseline is {wr_h:.2f} "
                f"({heu['wins']}W-{heu['losses']}L-{heu['draws']}D), below "
                "the 0.40 threshold. The model has not yet learned to "
                "outplay a simple pattern player."
            ),
            "thresholds": thresholds,
        }
    if wr_h < VERDICT_BEATS_HEURISTIC:
        return {
            "tier": "around heuristic",
            "label": "Around heuristic",
            "meets_good_level": False,
            "explanation": (
                f"Win rate vs the heuristic baseline is {wr_h:.2f} "
                f"({heu['wins']}W-{heu['losses']}L-{heu['draws']}D) — "
                "roughly even with a simple pattern player, not clearly "
                "above it."
            ),
            "thresholds": thresholds,
        }
    mcts_result = ladder.get("mcts")
    if mcts_result is None:
        return {
            "tier": "beats heuristic",
            "label": "Beats heuristic (search ladder not run)",
            "meets_good_level": False,
            "explanation": (
                f"Win rate vs the heuristic baseline is {wr_h:.2f}, "
                "clearly above it — but the heuristic-MCTS matchup was "
                "not run, so the checkpoint cannot be placed against "
                "search. Re-run with 'mcts' in --opponents."
            ),
            "thresholds": thresholds,
        }
    wr_m = win_rate(
        mcts_result["wins"], mcts_result["losses"], mcts_result["draws"]
    )
    games_m = int(mcts_result.get("games", 0))
    if wr_m < GOOD_LEVEL_MIN_MCTS_WIN_RATE:
        return {
            "tier": "beats heuristic, below search",
            "label": "Beats heuristic, below search",
            "meets_good_level": False,
            "explanation": (
                f"Beats the heuristic baseline ({wr_h:.2f}) but wins only "
                f"{wr_m:.2f} vs the heuristic-MCTS search opponent "
                f"({mcts_result['wins']}W-{mcts_result['losses']}L-"
                f"{mcts_result['draws']}D over {games_m} games). The "
                "model outplays 1-ply patterns but not lookahead search."
            ),
            "thresholds": thresholds,
        }
    tac_rate = tactical["pass_rate"] if tactical else None
    missing: list[str] = []
    if games_m < GOOD_LEVEL_MIN_MCTS_GAMES:
        missing.append(
            f"only {games_m} games vs the search opponent "
            f"(need >= {GOOD_LEVEL_MIN_MCTS_GAMES} to confirm)"
        )
    if tac_rate is None:
        missing.append("tactical suite not run")
    elif tac_rate < GOOD_LEVEL_MIN_TACTICAL_PASS_RATE:
        missing.append(
            f"tactical pass rate {tac_rate:.2f} "
            f"(need >= {GOOD_LEVEL_MIN_TACTICAL_PASS_RATE:.2f})"
        )
    if missing:
        return {
            "tier": "challenges search",
            "label": "Challenges search — close to a good level",
            "meets_good_level": False,
            "explanation": (
                f"Beats the heuristic ({wr_h:.2f}) and wins {wr_m:.2f} vs "
                f"the search opponent over {games_m} games, but does not "
                "yet meet the full 'good level' definition: "
                + "; ".join(missing) + "."
            ),
            "thresholds": thresholds,
        }
    return {
        "tier": "good level (v1)",
        "label": "Good level (v1 working definition)",
        "meets_good_level": True,
        "explanation": (
            f"Meets the v1 'good level' definition: {wr_m:.2f} vs the "
            f"heuristic-MCTS search opponent over {games_m} games as "
            f"both colours, and a tactical pass rate of {tac_rate:.2f}. "
            "This is a working definition against a weak ladder — it "
            "means 'clearly stronger than the search baseline', not "
            "'strong in absolute terms'."
        ),
        "thresholds": thresholds,
    }


# --- previous checkpoint --------------------------------------------------

def find_previous_checkpoint(run_dir: str, current_path: str) -> str | None:
    """Pick the prior checkpoint in ``run_dir`` for a
    checkpoint-vs-checkpoint comparison.

    Rule: the latest ``checkpoint-<step>.pt`` in the run dir wins; if
    the evaluated checkpoint is itself a ``checkpoint-<step>.pt``, the
    latest one with a SMALLER step wins. If there are no stepped
    checkpoints, ``best.pt`` is the previous of ``final.pt`` and vice
    versa. Returns None when no prior checkpoint exists.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    current = os.path.abspath(current_path)
    current_step: int | None = None
    base = os.path.basename(current)
    if base.startswith("checkpoint-") and base.endswith(".pt"):
        try:
            current_step = int(base[len("checkpoint-"):-len(".pt")])
        except ValueError:
            current_step = None
    stepped: list[tuple[int, str]] = []
    for name in os.listdir(ckpt_dir):
        if name.startswith("checkpoint-") and name.endswith(".pt"):
            try:
                step = int(name[len("checkpoint-"):-len(".pt")])
            except ValueError:
                continue
            path = os.path.join(ckpt_dir, name)
            if os.path.abspath(path) == current:
                continue
            if current_step is not None and step >= current_step:
                continue
            stepped.append((step, path))
    if stepped:
        stepped.sort()
        return stepped[-1][1]
    if base == "final.pt":
        candidate = os.path.join(ckpt_dir, "best.pt")
    elif base == "best.pt":
        candidate = os.path.join(ckpt_dir, "final.pt")
    else:
        return None
    if os.path.exists(candidate) and os.path.abspath(candidate) != current:
        return candidate
    return None


# --- report ---------------------------------------------------------------

def build_eval_report(
    checkpoint: str,
    ladder: dict,
    tactical: dict | None,
    tactical_baseline: dict | None,
    checkpoint_path_opponent: str | None = None,
    ts_match: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the full eval report dict (JSON-serializable)."""
    from .runs import utc_now_iso

    report = {
        "checkpoint": checkpoint,
        "evaluated_at": utc_now_iso(),
        "ladder": ladder,
        "ladder_elo": ladder_elo(ladder),
        "tactical": tactical,
        "tactical_baseline_heuristic": tactical_baseline,
        "opponent_checkpoint": checkpoint_path_opponent,
        "ts_match": ts_match,
        "verdict": verdict_for(ladder, tactical),
        "disclaimer": DISCLAIMER,
    }
    if extra:
        report.update(extra)
    return report


def _fmt_expect(expect: dict) -> str:
    """Human-readable tactical expectation (the raw dict is unreadable
    in a report)."""
    etype = expect.get("type")
    if etype == "win":
        return "any winning move"
    if etype == "moves":
        squares = ", ".join(f"({x},{y})" for x, y in expect["moves"])
        return f"one of {squares}"
    if etype == "vcf":
        return "a move starting a forced win (VCF)"
    return str(expect)


def _fmt_ci(ci: list) -> str:
    return f"[{ci[0]:.2f}, {ci[1]:.2f}]"


def render_eval_report_md(report: dict) -> str:
    """Render the eval report as Markdown."""
    lines = [
        "# Eval report",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Evaluated at: {report['evaluated_at']} (UTC)",
        "",
        "## Verdict",
        "",
        f"**{report['verdict']['label']}**",
        "",
        report["verdict"]["explanation"],
        "",
        "## Ladder matches",
        "",
        "Win rates count draws as half a win. The 95% CI is a Wilson-score "
        "interval — with few games it is wide, and that is the honest "
        "picture.",
        "",
        "| Opponent | W-L-D | Win rate | 95% CI | As Black | As White |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["ladder"].items():
        b, w = r["as_black"], r["as_white"]
        lines.append(
            f"| {name} | {r['wins']}-{r['losses']}-{r['draws']} | "
            f"{r['win_rate']:.3f} | {_fmt_ci(r['win_rate_ci_95'])} | "
            f"{b['wins']}W-{b['losses']}L-{b['draws']}D | "
            f"{w['wins']}W-{w['losses']}L-{w['draws']}D |"
        )
    elo = report["ladder_elo"]
    lo, hi = elo["interval"]
    anchors = ", ".join(
        f"{name} = {value}" for name, value in elo["anchors"].items()
    )
    lines += [
        "",
        "## Ladder Elo (relative to this ladder only)",
        "",
        f"- Estimate: **{elo['estimate']}**" if elo["estimate"] is not None
        else "- Estimate: n/a (no anchored matchup with a finite estimate)",
        f"- 95% interval: [{lo}, {hi}]"
        if lo is not None and hi is not None
        else "- 95% interval: unbounded on at least one side",
        f"- Anchors: {anchors} (arbitrary reference points)",
        "",
        "> " + elo["caveat"],
        "",
    ]
    if elo.get("interval_inconsistent"):
        lines += [
            "> Note: the per-opponent intervals do not overlap — the "
            "model beats one anchor decisively and loses to the other "
            "decisively, so no single rating fits both matchups exactly. "
            "The interval above is the span between the two bounds, a "
            "rough range rather than a strict confidence interval.",
            "",
        ]
    tac = report.get("tactical")
    tac_base = report.get("tactical_baseline_heuristic")
    lines += ["## Tactical suite", ""]
    if tac is None:
        lines.append("Not run.")
    else:
        lines.append(
            f"- Checkpoint: **{tac['passed']}/{tac['total']} "
            f"({tac['pass_rate']:.0%})**"
        )
        if tac_base is not None:
            lines.append(
                f"- Heuristic baseline (same suite): "
                f"{tac_base['passed']}/{tac_base['total']} "
                f"({tac_base['pass_rate']:.0%})"
            )
        failures = [r for r in tac["results"] if not r["passed"]]
        if failures:
            lines += ["", "Failures:"]
            for r in failures:
                lines.append(
                    f"- `{r['id']}` ({r['name']}): chose {r['chosen']}, "
                    f"expected {_fmt_expect(r['expect'])}"
                )
        else:
            lines.append("- No failures.")
    if report.get("opponent_checkpoint"):
        lines += [
            "",
            "## Checkpoint vs checkpoint",
            "",
            f"Opponent checkpoint: `{report['opponent_checkpoint']}` "
            "(see the `checkpoint` row in the ladder table above).",
        ]
    ts = report.get("ts_match")
    if ts:
        lines += [
            "",
            "## TypeScript cross-check",
            "",
            f"- TS level: `{ts.get('level')}`, games: {ts.get('games')}",
            f"- Checkpoint: {ts.get('wins')}W-{ts.get('losses')}L-"
            f"{ts.get('draws')}D (win rate {ts.get('win_rate'):.3f})",
        ]
        if ts.get("note"):
            lines.append(f"- Note: {ts['note']}")
    lines += [
        "",
        "## Disclaimer",
        "",
        report["disclaimer"],
        "",
    ]
    return "\n".join(lines)


def write_eval_report(run_dir: str, report: dict) -> tuple[str, str]:
    """Write ``eval_report.json`` + ``eval_report.md`` into ``run_dir``."""
    json_path = os.path.join(run_dir, "eval_report.json")
    md_path = os.path.join(run_dir, "eval_report.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_eval_report_md(report))
    return json_path, md_path


# --- full evaluation entry point ------------------------------------------

def evaluate_checkpoint(
    checkpoint_path: str,
    games: int = 10,
    sims: int = 50,
    opponents: tuple[str, ...] = ("random", "heuristic"),
    mcts_sims: int = 100,
    opponent_checkpoint: str | None = None,
    opponent_sims: int = 50,
    tactical: bool = True,
    tactical_sims: int = 50,
    device: str = "cpu",
    seed: int = 0,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Evaluate a checkpoint end to end: ladder + tactical suite
    (checkpoint and heuristic baseline) + verdict. Returns the full
    report dict (see :func:`build_eval_report`)."""
    from .architectures import load_model_from_checkpoint

    model, _ = load_model_from_checkpoint(checkpoint_path, device=device)
    mcts_player = MCTS(model, device=device)
    opponent_mcts = None
    if "checkpoint" in opponents:
        if opponent_checkpoint is None:
            raise ValueError(
                "'checkpoint' is in opponents but no opponent_checkpoint given"
            )
        opp_model, _ = load_model_from_checkpoint(opponent_checkpoint, device=device)
        opponent_mcts = MCTS(opp_model, device=device)
    ladder = run_ladder(
        mcts_player,
        num_simulations=sims,
        games_per_opponent=games,
        opponents=opponents,
        opponent_mcts=opponent_mcts,
        opponent_simulations=opponent_sims,
        mcts_sims=mcts_sims,
        seed=seed,
        on_event=on_event,
    )
    tactical_results = None
    tactical_baseline_results = None
    if tactical:
        print("  tactical suite (checkpoint)...", file=sys.stderr, flush=True)
        tactical_results = run_tactical_suite(
            mcts_tactical_chooser(mcts_player, tactical_sims)
        )
        print(
            f"  tactical: checkpoint {tactical_results['passed']}/"
            f"{tactical_results['total']}",
            file=sys.stderr, flush=True,
        )
        tactical_baseline_results = run_tactical_suite(
            lambda board, player: baseline.choose_move(board, player)
        )
        print(
            f"  tactical: heuristic baseline "
            f"{tactical_baseline_results['passed']}/"
            f"{tactical_baseline_results['total']}",
            file=sys.stderr, flush=True,
        )
    return build_eval_report(
        checkpoint=checkpoint_path,
        ladder=ladder,
        tactical=tactical_results,
        tactical_baseline=tactical_baseline_results,
        checkpoint_path_opponent=opponent_checkpoint,
    )


# --- TypeScript cross-check (stretch) -------------------------------------

def run_ts_match(
    checkpoint_path: str,
    level: str,
    games: int = 4,
    sims: int = 50,
    ts_time_ms: int | None = None,
    timeout_s: int = 1800,
) -> dict:
    """Play the checkpoint against a TypeScript AI level via
    ``scripts/play_match.mjs`` (which bundles ``src/ai`` with esbuild
    and drives the Python checkpoint over a JSON-lines protocol).

    Returns the match summary dict. Raises RuntimeError with a clear
    message when the Node driver is unavailable or fails — the caller
    decides whether that is fatal or a documented gap.
    """
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    driver = os.path.join(repo_root, "scripts", "play_match.mjs")
    if not os.path.exists(driver):
        raise RuntimeError(
            f"TS match driver not found: {driver} "
            "(cross-check unavailable; Python ladder only)"
        )
    cmd = [
        "node", driver,
        "--checkpoint", checkpoint_path,
        "--level", level,
        "--games", str(games),
        "--sims", str(sims),
        "--json",
    ]
    if ts_time_ms is not None:
        cmd += ["--ts-time-ms", str(ts_time_ms)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=repo_root,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"TS match timed out after {timeout_s}s "
            f"(level={level}, games={games})"
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"TS match driver failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()[-500:]}"
        )
    for line in proc.stdout.splitlines():
        if line.startswith("MATCH_RESULT "):
            return json.loads(line[len("MATCH_RESULT "):])
    raise RuntimeError(
        "TS match driver produced no MATCH_RESULT line; "
        f"stdout tail: {proc.stdout.strip()[-500:]}"
    )

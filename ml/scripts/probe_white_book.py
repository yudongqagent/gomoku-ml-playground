#!/usr/bin/env python3
"""White opening-book probe (iter-13 §2.5, §1b contingency).

Plays a batch of games vs v0 with White's ply-1 reply FORCED
to (6,8) whenever Black opens (7,7); every other move is the
v13 base (V13MCTS, white_temp=2.0, 100 sims). The control is
the §2.1 s100-vs-v0 cell's White rate (0.775, n=20, same seeds
300/400, same opponent, engine's mixed replies).

Decision rule (pre-registered, iter-13 §2.5): book (6,8) ONLY
if the forced arm's as-White win rate is >= 0.875 (control +
0.10). Otherwise NO book.

This is a PROBE, not an eval: it decides whether a book entry
exists, it does not decide the bar.

Usage:
    python ml/scripts/probe_white_book.py --games 40 \\
        --out research/iterations/probe-book-68-vs-v0.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import evaluate, game  # noqa: E402
from gomoku_ml.mcts_v13 import V13MCTS  # noqa: E402
from selfplay_batch import (  # noqa: E402
    _write_result,
    play_recorded_game,
    resolve_engine,
)
from sweep_white_sims import _splits  # noqa: E402

# Book entry under test: Black opens the centre, White replies
# the diagonal (6,8). Flat indices on the 15x15 board.
CENTRE = 7 * game.SIZE + 7
BOOK_REPLY = 8 * game.SIZE + 6  # (x=6, y=8)


def _booked_choose(mcts):
    """Wrap an MCTS choose_move with the book reply: if the board
    holds exactly Black's centre opening stone and White is to
    move, play (6,8); otherwise delegate to the engine."""
    def choose(board, player):
        if player == game.WHITE and int((board != game.EMPTY).sum()) == 1:
            if board[7, 7] == game.BLACK:
                return BOOK_REPLY
        return mcts.choose_move(board, player)
    return choose


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed-a", type=int, default=300)
    parser.add_argument("--seed-b", type=int, default=400)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    mcts = V13MCTS(sims=100, white_sims=100, white_temp=2.0,
                   rng=np.random.default_rng(args.seed_a))
    choose_a = _booked_choose(mcts)
    choose_b, name_b = resolve_engine("v0", args.seed_b)
    games: list[dict] = []
    wins = losses = draws = 0
    book_used = 0
    t0 = time.time()
    for i in range(args.games):
        a_is_black = (i % 2 == 0)
        a_colour = game.BLACK if a_is_black else game.WHITE
        if a_is_black:
            winner, moves = play_recorded_game(
                choose_a, choose_b, game.empty_board(), game.BLACK
            )
        else:
            winner, moves = play_recorded_game(
                choose_b, choose_a, game.empty_board(), game.BLACK
            )
        if not a_is_black and len(moves) >= 2 and moves[1] == BOOK_REPLY:
            book_used += 1
        rec = {
            "game": i,
            "moves": moves,
            "a_colour": "black" if a_colour == game.BLACK else "white",
            "winner": (
                "black" if winner == game.BLACK
                else "white" if winner == game.WHITE else "draw"
            ),
            "winner_is_a": (
                (winner == a_colour) if winner != game.EMPTY else None
            ),
            "plies": len(moves),
        }
        if winner == game.EMPTY:
            draws += 1
        elif winner == a_colour:
            wins += 1
        else:
            losses += 1
        games.append(rec)
        lo, hi = evaluate.wilson_ci(wins, losses, draws)
        result = {
            "engine_a": "sweep-ws100-book68",
            "engine_b": name_b,
            "white_sims": 100,
            "white_temp": 2.0,
            "book_reply": "(6,8) after Black (7,7)",
            "book_used_games": book_used,
            "games": args.games,
            "games_played": len(games),
            "complete": (i + 1 == args.games),
            "swap_rule": False,
            "placement": None,
            "placement_seed": None,
            "judge": None,
            "seed_a": args.seed_a,
            "seed_b": args.seed_b,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
            "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
            "as_black": _splits(games)["black"],
            "as_white": _splits(games)["white"],
            "elapsed_seconds": round(time.time() - t0, 1),
            "records": games,
        }
        _write_result(args.out, result)
        if (i + 1) % 10 == 0 or i + 1 == args.games:
            print(f"  {i + 1}/{args.games} games ({time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
    w = result["as_white"]
    print(f"  book-(6,8): {wins}W-{losses}L-{draws}D "
          f"(win rate {result['win_rate']:.4f}); as White: "
          f"{w['wins']}W-{w['losses']}L-{w['draws']}D "
          f"({w['win_rate']:.4f}); book used in {book_used}/20 White "
          f"games", file=sys.stderr)
    print(f"  control (engine's mixed replies, s100-vs-v0): "
          f"White 0.7750; book rule: book (6,8) iff forced White "
          f"rate >= 0.875", file=sys.stderr)
    print(f"  records written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Find the losing move in each fast-loop loss (v25, iteration 1).

For every loss in the two iter-1 match files, replay the game and,
after each v25 move, check whether the opponent forces a win
(``evaluate.forces_win``, VCF solver, depth 4 attacker moves).
The losing move is the first v25 move after which the opponent
forces a win AND some alternative reply avoids it (verified by
enumerating v25's near-move replies at that ply). If no reply
avoids it, the position was already lost — reported as such.

Usage:
    python ml/scripts/find_losing_moves.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import evaluate, game  # noqa: E402

DEPTH = 4


def _xy(m: int) -> str:
    return f"({m % 15},{m // 15})"


def _replies_avoid(board, player, opp):
    """Near-move replies for `player` after which the opponent does
    NOT force a win. Returns (saving_moves, n_checked)."""
    saving = []
    checked = 0
    for m in evaluate._near_moves(board, player):
        b2 = game.apply_move(board, m, player)
        if game.is_winning_move(board, m, player):
            saving.append(m)
            checked += 1
            continue
        checked += 1
        if not evaluate.forces_win(b2, opp, DEPTH):
            saving.append(m)
    return saving, checked


def analyse_loss(rec, a_colour):
    moves = rec["moves"]
    opp = -a_colour
    board = game.empty_board()
    # Rebuild the board ply by ply; check after each v25 move.
    for ply, m in enumerate(moves):
        player = game.BLACK if ply % 2 == 0 else game.WHITE
        pre = board.copy()
        board = game.apply_move(board, m, player)
        if player != a_colour:
            continue
        # v25 just moved; opponent to move. Does the opponent force a win?
        if evaluate.forces_win(board, opp, DEPTH):
            saving, checked = _replies_avoid(pre, a_colour, opp)
            return {
                "game": rec["game"],
                "a_colour": "black" if a_colour == game.BLACK else "white",
                "plies": len(moves),
                "losing_ply": ply,
                "losing_move": _xy(m),
                "saving_moves": [_xy(s) for s in saving[:8]],
                "n_saving": len(saving),
                "n_checked": checked,
                "already_lost": len(saving) == 0,
            }
    return {
        "game": rec["game"],
        "a_colour": "black" if a_colour == game.BLACK else "white",
        "plies": len(moves),
        "losing_ply": None,
        "note": "no VCF moment found: opponent never forces a win by "
        "consecutive fours within depth 4 after a v25 move "
        "(positional loss, or the forced win is deeper than 4)",
    }


def main():
    for fname in ("fast-v25-vs-v8-iter1", "fast-v25-vs-v0-iter1"):
        d = json.load(
            open(f"research/iterations/{fname}.json", encoding="utf-8")
        )
        print(f"== {fname}: {d['wins']}W-{d['losses']}L-{d['draws']}D ==")
        for rec in d["records"]:
            if rec["winner_is_a"] is not False:
                continue
            a_colour = (
                game.BLACK if rec["a_colour"] == "black" else game.WHITE
            )
            out = analyse_loss(rec, a_colour)
            print(json.dumps(out))


if __name__ == "__main__":
    main()

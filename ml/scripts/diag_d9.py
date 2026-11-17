#!/usr/bin/env python3
"""d9 diagnostic (iter-32 §1.2): v24's verified horizon miss.

Reconstructs the position from v24's vs-v0 fast-loop game 1
(ML batch 1, ``fast-v24-vs-v0-iter1.json`` game 1, White to
move after 19 plies — the workup's "ply 19" is the 0-based
move index): White played i10 (143) and lost; d9 (123) saves
2/3 in same-engine playouts (``workup-v24-v0-g1.json``).

Asks: (1) does Black hold a VCF start here (would the
defensive check even fire)? (2) what does v32
(offensive-only) play? (3) what does v8 play? (4) what does
the defensive check play, run as a diagnostic with
``defensive=True``?

Diagnostic only — it is NOT a design trigger by itself
(iter-32 §1.2: one position cannot override the
pre-registered defensive-inclusion rule).

Usage:
    PYTHONPATH=ml python ml/scripts/diag_d9.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

# v24-vs-v0 fast-loop game 1, moves 0..18 (19 plies, Black
# first). Source: ~/workspace/projects/gomoku-ml1-h1/
# research/iterations/fast-v24-vs-v0-iter1.json, records[1].
GAME1_PREFIX = [
    112, 97, 111, 113, 81, 126, 96, 66, 80, 128,
    82, 83, 98, 78, 110, 124, 95, 125, 127,
]
ACTUAL_I10 = 143  # (8, 9) — loses 3/3 in playouts
SAVE_D9 = 123  # (3, 8) — saves 2/3 in playouts


def main() -> int:
    from gomoku_ml import evaluate, game, scoring

    board = game.empty_board()
    player = game.BLACK
    for m in GAME1_PREFIX:
        board = game.apply_move(board, m, player)
        player = -player
    assert player == game.WHITE  # White to move at ply 19
    print(f"position: {len(GAME1_PREFIX)} plies, White to move")
    print(f"actual (loses 3/3): i10 = {game.move_to_xy(ACTUAL_I10)}")
    print(f"save (saves 2/3):   d9  = {game.move_to_xy(SAVE_D9)}")

    # (1) VCF structure at the position.
    ours = evaluate.four_creating_moves(board, game.WHITE)
    ours_vcf = [
        m for m in ours
        if evaluate.starts_forced_win(board, game.WHITE, m, depth=3)
    ]
    opp = evaluate.four_creating_moves(board, game.BLACK)
    opp_vcf = [
        m for m in opp
        if evaluate.starts_forced_win(board, game.BLACK, m, depth=3)
    ]
    print(f"White four-creators: {len(ours)}, VCF starts: "
          f"{[game.move_to_xy(m) for m in ours_vcf]}")
    print(f"Black four-creators: {len(opp)}, VCF starts: "
          f"{[game.move_to_xy(m) for m in opp_vcf]}")

    # (2)/(3) What do v32 and v8 play?
    for version in (32, 8):
        t0 = time.perf_counter()
        player_obj = scoring.get_player(version, seed=100)
        move = player_obj.choose_move(board, game.WHITE)
        dt = time.perf_counter() - t0
        tag = "SAVE" if move == SAVE_D9 else (
            "actual-i10" if move == ACTUAL_I10 else "other")
        print(f"v{version}: {game.move_to_xy(move)} [{tag}] ({dt:.1f}s)")

    # (4) Defensive check as a diagnostic (prior-based, v29's
    # design — run here only to see whether it would find d9).
    t0 = time.perf_counter()
    v32 = scoring.get_player(32, seed=100)
    move = v32._mcts._vcf_root_move(board, game.WHITE, defensive=True)
    dt = time.perf_counter() - t0
    if move is None:
        print(f"defensive check (diagnostic): does not fire ({dt:.1f}s)")
    else:
        tag = "SAVE" if move == SAVE_D9 else (
            "actual-i10" if move == ACTUAL_I10 else "other")
        print(f"defensive check (diagnostic): {game.move_to_xy(move)} "
              f"[{tag}] ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

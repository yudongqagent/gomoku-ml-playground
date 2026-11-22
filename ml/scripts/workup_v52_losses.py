#!/usr/bin/env python3
"""Fast-loop loss workup for v52: replay the losses in a batch JSON
and report, at every v52 turn, whether TSS fired and whether the
played move was the TSS move. A TSS fire in a LOST game is the
prime suspect for a false positive (TSS claiming a forced win that
isn't) or a dropped forcing line (the win slipping away after the
TSS move because the continuation needed a quiet move — the
pre-registered limitation).

Usage:
    PYTHONPATH=ml python ml/scripts/workup_v52_losses.py
        research/iterations/fastloop-v52-vs-v27-i1.json
"""

from __future__ import annotations

import json
import sys


def _xy(move):
    return (move % 15, move // 15)


def main() -> int:
    from gomoku_ml import game, mcts_heuristic

    with open(sys.argv[1], encoding="utf-8") as f:
        batch = json.load(f)
    for rec in batch["records"]:
        if rec.get("winner_is_a") is not False:
            continue
        a_colour = rec["a_colour"]
        a_player = game.BLACK if a_colour == "black" else game.WHITE
        board = game.empty_board()
        player = game.BLACK
        print(f"=== game {rec['game']} (A={a_colour}, "
              f"winner={rec['winner']}, {rec['plies']} plies) ===")
        fires = 0
        for ply, move in enumerate(rec["moves"]):
            if player == a_player:
                tss = mcts_heuristic.tss_winning_move(board, player)
                if tss is not None:
                    fires += 1
                    tag = "FOLLOWED" if tss == move else "IGNORED?!"
                    print(f"  ply {ply}: TSS fired {_xy(tss)}, "
                          f"played {_xy(move)} — {tag}")
            board = game.apply_move(board, move, player)
            player = -player
        print(f"  TSS fired on {fires} v52 turn(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

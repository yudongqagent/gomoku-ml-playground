#!/usr/bin/env python3
"""Fast-loop loss workup for v57: replay the losses in a batch JSON
and report, at every v57 turn, whether TSS fired at the winner
cell's (depth, budget) and at the v52 control (4, 3000), and
whether the played move was the TSS move. A TSS fire in a LOST game
is the prime suspect for a false positive (TSS claiming a forced
win that isn't) or a dropped forcing line (the win slipping away
after the TSS move because the continuation needed a quiet move —
the pre-registered limitation). A control fire the cell MISSED (the
cell aborted where v52 found a win) is the loss mechanism from the
sweep, visible in a real game.

Usage:
    PYTHONPATH=ml python ml/scripts/workup_v57_losses.py
        research/iterations/fastloop-v57-vs-v52-i1.json
        --depth 5 --budget 6000
"""

from __future__ import annotations

import argparse
import json
import sys


def _xy(move):
    return (move % 15, move // 15)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_json")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--budget", type=int, required=True)
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic

    with open(args.batch_json, encoding="utf-8") as f:
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
        missed = 0
        for ply, move in enumerate(rec["moves"]):
            if player == a_player:
                cell = mcts_heuristic.tss_winning_move(
                    board, player, depth=args.depth,
                    node_budget=args.budget,
                )
                control = mcts_heuristic.tss_winning_move(
                    board, player, depth=4, node_budget=3000
                )
                if cell is not None:
                    fires += 1
                    tag = "FOLLOWED" if cell == move else "IGNORED?!"
                    print(f"  ply {ply}: cell TSS fired {_xy(cell)}, "
                          f"played {_xy(move)} — {tag}")
                elif control is not None:
                    # The cell aborted (or otherwise missed) a win
                    # the v52 control finds — the sweep's loss
                    # mechanism, in a real game.
                    missed += 1
                    print(f"  ply {ply}: cell MISSED control win "
                          f"{_xy(control)}, played {_xy(move)}")
            board = game.apply_move(board, move, player)
            player = -player
        print(f"  cell TSS fired on {fires} v57 turn(s); "
              f"missed {missed} control win(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

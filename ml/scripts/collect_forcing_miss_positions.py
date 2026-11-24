#!/usr/bin/env python3
"""Collect Check A positions for the v58 mechanism check
(iter-58 §11.3): White turns in v52-White's normal-rules losses
where White held forcing moves but v52 played a non-forcing move.

Procedure (pre-registered in iter-58 §11.3 BEFORE this script ran):

* Replay the recorded move lists of batch-v52-vs-v27.json (normal
  rules), v52-White LOSSES only (27 games; draws excluded).
* At each White turn, before applying White's recorded move:
  White holds no winning move, Black holds no winning move (else
  the win/block scans force the move), White has >= 1 forcing
  move (``tss_forcing_moves``), and v52's recorded move is NOT a
  forcing move.
* Collect the FIRST 20 such turns (game order, then ply order).
  Sanity: TSS must be silent on every collected position (it is
  logically guaranteed — had TSS fired, v52 would have played
  the TSS move, a forcing move — but the check guards against a
  replay bug; a firing candidate is skipped, not collected).
* Positions are stored as black/white stone lists plus the game,
  ply, recorded move, and forcing-move list.

Deterministic (recorded games, deterministic enumeration).

Usage:
    PYTHONPATH=ml python ml/scripts/collect_forcing_miss_positions.py
        --in research/iterations/batch-v52-vs-v27.json
        --out research/iterations/threat-miss-positions-v58.json
        [--count 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=20)
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic, scoring

    in_path = args.inp
    if not os.path.isabs(in_path):
        in_path = os.path.join(REPO_ROOT, in_path)
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)

    with open(in_path, encoding="utf-8") as f:
        batch = json.load(f)

    positions = []
    stats = {
        "games_scanned": 0,
        "white_turns_scanned": 0,
        "white_turns_with_forcing_moves": 0,
        "miss_turns_total": 0,
        "tss_fired_on_candidate": 0,
    }
    for rec in batch["records"]:
        if rec["a_colour"] != "white":
            continue
        if rec["winner_is_a"] is not False:
            continue  # losses only (draws have winner_is_a None)
        stats["games_scanned"] += 1
        board = game.empty_board()
        moves = rec["moves"]
        for ply, mv in enumerate(moves):
            player = game.BLACK if ply % 2 == 0 else game.WHITE
            if player == game.WHITE:
                stats["white_turns_scanned"] += 1
                if scoring._winning_moves(board, game.WHITE):
                    board = game.apply_move(board, mv, player)
                    continue
                if scoring._winning_moves(board, game.BLACK):
                    board = game.apply_move(board, mv, player)
                    continue
                forcing = mcts_heuristic.tss_forcing_moves(board, game.WHITE)
                if not forcing:
                    board = game.apply_move(board, mv, player)
                    continue
                stats["white_turns_with_forcing_moves"] += 1
                if mv in forcing:
                    board = game.apply_move(board, mv, player)
                    continue
                stats["miss_turns_total"] += 1
                if len(positions) >= args.count:
                    board = game.apply_move(board, mv, player)
                    continue
                # Sanity: TSS must be silent (see module docstring).
                if mcts_heuristic.tss_winning_move(board, game.WHITE) is not None:
                    stats["tss_fired_on_candidate"] += 1
                    board = game.apply_move(board, mv, player)
                    continue
                black = [
                    [int(x), int(y)]
                    for y, x in zip(*((board == game.BLACK).nonzero()))
                ]
                white = [
                    [int(x), int(y)]
                    for y, x in zip(*((board == game.WHITE).nonzero()))
                ]
                positions.append(
                    {
                        "game": rec["game"],
                        "ply": ply,
                        "black": black,
                        "white": white,
                        "to_move": "white",
                        "recorded_move": mv,
                        "forcing_moves": sorted(forcing),
                    }
                )
            board = game.apply_move(board, mv, player)

    result = {
        "source": os.path.basename(in_path),
        "count_requested": args.count,
        "count_collected": len(positions),
        "stats": stats,
        "positions": positions,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        f"collected {len(positions)}/{args.count} miss positions "
        f"from {stats['games_scanned']} v52-White losses "
        f"({stats['white_turns_scanned']} White turns, "
        f"{stats['white_turns_with_forcing_moves']} with forcing moves, "
        f"{stats['miss_turns_total']} miss turns, "
        f"{stats['tss_fired_on_candidate']} TSS-fired candidates skipped)"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

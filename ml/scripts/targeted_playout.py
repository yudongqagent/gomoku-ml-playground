#!/usr/bin/env python3
"""Targeted playout + exact analysis for specific moves (batch-9 forensics).

``workup_loss.py`` only tests top-K 1-ply alternatives, so a quiet
saving move (occupying the opponent's setup square) can be missed.
This script tests EXPLICITLY NAMED moves at one turn of one game:

* exact analysis (cheap): after the move, does the opponent hold a
  VCF-starting move (``evaluate.starts_forced_win``)? How many
  open-four creators? How many "setup" moves (a move after which the
  opponent holds >= 2 creators)?
* playout verification (same engines/seeds as workup_loss): the
  move saves iff it draws/wins at >= 2 of 3 seeds AND the actual
  move loses at >= 2 of 3 seeds (the workup's corrected method).

Usage:
    python ml/scripts/targeted_playout.py <batch.json> --game 11 \\
        --ply 23 --move 5,10 --move 5,11 [--no-playout]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from gomoku_ml import evaluate, game, scoring  # noqa: E402
from workup_loss import _PlayerCache, play_out  # noqa: E402

MIN_SEEDS = 2


def _xy(move: int) -> tuple[int, int]:
    return (move % 15, move // 15)


def _creators(board: np.ndarray, player: int) -> list[int]:
    """Open-four creators for ``player`` (a move leaving them >= 2
    winning moves)."""
    out = []
    for c in scoring._candidate_moves(board):
        b2 = game.apply_move(board, c, player)
        if len(scoring._winning_moves(b2, player)) >= 2:
            out.append(c)
    return out


def _vcf_starts(board: np.ndarray, player: int) -> list[int]:
    """Moves starting a forced win by consecutive fours."""
    return [
        m
        for m in scoring._candidate_moves(board)
        if evaluate.starts_forced_win(board, player, m)
    ]


def _setups(board: np.ndarray, player: int) -> list[int]:
    """Quiet setup moves: a move after which ``player`` holds >= 2
    open-four creators (and which is not itself a VCF start — those
    are counted separately)."""
    out = []
    for m in scoring._candidate_moves(board):
        b2 = game.apply_move(board, m, player)
        if scoring._winning_moves(b2, player):
            continue  # an immediate win, not a setup
        if len(_creators(b2, player)) >= 2:
            out.append(m)
    return out


def analyse(board: np.ndarray, player: int, move: int, label: str) -> dict:
    """Exact analysis after ``player`` plays ``move``: the
    opponent's VCF starts / creators / setups on the resulting
    board."""
    b2 = game.apply_move(board, move, player)
    opp = -player
    vcf = _vcf_starts(b2, opp)
    creators = _creators(b2, opp)
    setups = _setups(b2, opp)
    print(
        f"  {label} {_xy(move)}: opp VCF starts={len(vcf)} "
        f"{[_xy(m) for m in vcf][:4]}, creators={len(creators)} "
        f"{[_xy(m) for m in creators][:4]}, setups={len(setups)} "
        f"{[_xy(m) for m in setups][:4]}"
    )
    return {"vcf": len(vcf), "creators": len(creators),
            "setups": len(setups)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--ply", type=int, required=True)
    parser.add_argument("--move", action="append", default=[],
                        help="move to test as x,y (repeatable)")
    parser.add_argument("--version-a", type=int, default=22)
    parser.add_argument("--version-b", default="mcts")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--no-playout", action="store_true",
                        help="exact analysis only, no playouts")
    args = parser.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        batch = json.load(f)
    rec = next(r for r in batch["records"] if r["game"] == args.game)
    a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
    moves = rec["moves"]
    board = game.empty_board()
    for i, m in enumerate(moves[: args.ply]):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE
        )
    actual = moves[args.ply]
    print(f"== {os.path.basename(args.batch)} game {args.game}, ply "
          f"{args.ply} ({'Black' if a_colour == game.BLACK else 'White'} "
          f"to move); actual played {_xy(actual)} ==")
    print("exact analysis (opponent's threats AFTER the move):")
    analyse(board, a_colour, actual, "actual ")
    tested = []
    for spec in args.move:
        x, y = spec.split(",")
        tested.append(int(y) * 15 + int(x))
    for m in tested:
        analyse(board, a_colour, m, "test   ")
    if args.no_playout:
        return 0
    cache = _PlayerCache()
    after_actual = game.apply_move(board, actual, a_colour)
    actual_out = [
        play_out(after_actual, -a_colour, args.version_a,
                 args.version_b, a_colour, seed, cache)
        for seed in args.seeds
    ]
    actual_losses = sum(1 for o in actual_out if o < 0)
    print(f"playout actual {_xy(actual)}: {actual_out} "
          f"({actual_losses}/{len(args.seeds)} losses)"
          + (" — QUALIFIES" if actual_losses >= MIN_SEEDS else ""))
    for m in tested:
        after = game.apply_move(board, m, a_colour)
        out = [
            play_out(after, -a_colour, args.version_a, args.version_b,
                     a_colour, seed, cache)
            for seed in args.seeds
        ]
        non_losses = sum(1 for o in out if o >= 0)
        saves = non_losses >= MIN_SEEDS and actual_losses >= MIN_SEEDS
        print(f"playout test   {_xy(m)}: {out} "
              f"({non_losses}/{len(args.seeds)} non-losses)"
              + (" — SAVES" if saves else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

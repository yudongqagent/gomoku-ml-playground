#!/usr/bin/env python3
"""Walk back a lost game with the CORRECTED playout-oracle method.

For each of the loser's last K turns (walking back from the end):
play the ACTUAL move, then play out the game with a fast oracle
(v1 vs v1 — both sides the strongest fast engine — fresh seeds per
playout) to completion. The actual move's playout must LOSE FIRST
at the same seeds before an alternative counts (iter-6 §2.2(a):
the first version of the v6 walk-back compared the alternative's
playout against the HISTORICAL game outcome and produced a
withdrawn finding — the playout engines have fresh RNG state, so
even the actual move's playout diverges from the real game). Only
then are alternatives tested (top candidates by v1 1-ply score
plus any forcing move): an alternative counts as "the best move
instead" only if its playout draws or wins at seeds where the
actual move's playout loses.

Why v1-vs-v1 and not the match engines: the match engine (v7) is
an MCTS player (~0.4 s/move), so a playout game costs ~40 s and
the required volume (turns x alternatives x seeds) is infeasible.
The v1-vs-v1 oracle measures whether the position is lost/drawn
under strong 1-ply play — a proxy, stated as such. A turn where
the actual move's oracle playout DRAWS is not a blunder: the loss
is distributed drift (the iter-6 v6 result), not a single losing
move.

Usage:
    python ml/scripts/walkback_loss.py \\
        research/iterations/batch-v7-vs-v1.json --game 3 \\
        --turns 8 --seeds 0,1,2 --alternatives 6
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

from gomoku_ml import game, scoring  # noqa: E402


def _playout(board: np.ndarray, player_to_move: int, seed: int) -> int:
    """Play to completion with v1 vs v1 (fresh players per playout).
    Returns the winner (BLACK / WHITE / EMPTY for a draw). The
    terminal check runs BEFORE the first move (iter-5's walk-back
    bug was a missing terminal check)."""
    black = scoring.get_player(1, seed=seed)
    white = scoring.get_player(1, seed=seed + 1000)
    player = player_to_move
    for _ in range(game.SIZE * game.SIZE):
        over, win = game.game_over(board)
        if over:
            return win
        chooser = black if player == game.BLACK else white
        board = game.apply_move(board, chooser.choose_move(board, player),
                                player)
        player = -player
    _, win = game.game_over(board)
    return win


def _candidates(board: np.ndarray, player: int, actual: int, n: int
                ) -> list[int]:
    """Alternative moves to test: the top-``n`` by v1 1-ply score
    (excluding the actual move) plus any forcing move (win or four)
    not already in the top-``n``."""
    flat = board.reshape(-1).tolist()
    scored = [
        (scoring._score_move_flat(flat, m, player, 1.0), m)
        for m in scoring._candidate_moves(board)
    ]
    scored.sort(key=lambda sm: (-sm[0], sm[1]))
    out = [m for _s, m in scored[:n] if m != actual]
    for _s, m in scored:
        if m != actual and m not in out and scoring._is_forcing(board, m, player):
            out.append(m)
    return out


def walkback(rec: dict, turns: int, seeds: list[int], n_alt: int) -> dict:
    moves = [int(m) for m in rec["moves"]]
    a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
    boards = [game.empty_board()]
    board = game.empty_board()
    for i, m in enumerate(moves):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE
        )
        boards.append(board)
    # A's turns, latest first, up to ``turns`` of them (skip the
    # final ply: the game already ended there).
    a_turns = [ply for ply in range(len(moves))
               if (game.BLACK if ply % 2 == 0 else game.WHITE) == a_colour]
    tested = a_turns[-turns:] if turns else a_turns
    results = []
    for ply in reversed(tested):
        board_before = boards[ply]
        actual = moves[ply]
        # (1) The actual move's playout must lose FIRST, same seeds.
        actual_outcomes = []
        for s in seeds:
            b2 = game.apply_move(board_before, actual, a_colour)
            w = _playout(b2, -a_colour, s)
            actual_outcomes.append(
                "win" if w == a_colour else "loss" if w == -a_colour else "draw"
            )
        entry: dict = {
            "ply": ply,
            "actual": (actual % 15, actual // 15),
            "actual_outcomes": actual_outcomes,
            "actual_loses_all_seeds": all(
                o == "loss" for o in actual_outcomes
            ),
            "alternatives": [],
        }
        # (2) Only if it loses at every seed do alternatives count.
        if entry["actual_loses_all_seeds"]:
            for alt in _candidates(board_before, a_colour, actual, n_alt):
                alt_outcomes = []
                for s in seeds:
                    b2 = game.apply_move(board_before, alt, a_colour)
                    w = _playout(b2, -a_colour, s)
                    alt_outcomes.append(
                        "win" if w == a_colour else "loss"
                        if w == -a_colour else "draw"
                    )
                entry["alternatives"].append({
                    "move": (alt % 15, alt // 15),
                    "outcomes": alt_outcomes,
                    "saves": any(o in ("win", "draw") for o in alt_outcomes),
                })
        results.append(entry)
        print(f"  ply {ply}: actual {entry['actual']} -> "
              f"{actual_outcomes}"
              + (f"; {sum(1 for a in entry['alternatives'] if a['saves'])} "
                 f"saving alternative(s) of {len(entry['alternatives'])}"
                 if entry["actual_loses_all_seeds"] else
                 " (actual does not lose at all seeds: no alternative counts)"),
              file=sys.stderr, flush=True)
    return {"game": rec["game"], "a_colour": rec["a_colour"],
            "plies": len(moves), "turns": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_json")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--alternatives", type=int, default=6)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    with open(args.batch_json, encoding="utf-8") as f:
        batch = json.load(f)
    rec = next(r for r in batch["records"] if r["game"] == args.game)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"== walk-back: {os.path.basename(args.batch_json)} game "
          f"{args.game} (A={rec['a_colour']}, winner={rec['winner']}, "
          f"{rec['plies']} plies) ==", file=sys.stderr)
    result = walkback(rec, args.turns, seeds, args.alternatives)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"  results written to {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Classify losses in a selfplay_batch.py JSON: how did engine A lose?

For every game engine A lost (``winner_is_a`` False), replay the
moves and report:

* ``via_open_four`` — the winner held >= 2 winning moves on the
  board before the loser's LAST move (i.e. after the winner's
  second-to-last move): the standard open-four win is winner creates
  the open four, loser blocks one square, winner completes the five
  — so judging on the board before the FINAL move would miss nearly
  every open-four loss (the blocked square is gone by then);
* ``missed_blocks`` — turns where the opponent had exactly ONE
  winning move and A played a different, non-winning move (with
  >= 2 opponent winning moves the game is lost anyway — that is
  not a missed block, it is an open four);
* ``clean_missed_preventions`` — turns where the opponent had no
  immediate winning move but held >= 1 open-four creator (a move
  leaving them >= 2 winning moves), A's move was neither forcing
  nor safe (it left a creator live), AND a safe alternative existed
  among the candidate moves (forcing, or leaving no creator live).
  ``threatened_turns`` counts the turns where the opponent held
  >= 1 creator with no immediate winning move — the denominator
  for the prevention rate.

Definitions match the batch-1/2 loss analyses (see
``research/iterations/iter-6.md`` §2.1, whose vs-v1 aggregates this
script reproduces: 37 losses, 37/37 via open four, 0 missed blocks,
0 clean missed preventions in 221 threatened turns).

Usage:
    python ml/scripts/classify_losses.py research/iterations/batch-v8-vs-v1.json
    python ml/scripts/classify_losses.py batch-a.json batch-b.json --out /tmp/losses.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, scoring  # noqa: E402


def analyse_loss(moves: list[int], loser: int) -> dict:
    """Analyse one lost game. ``moves`` is the full flat move list
    (opening stones included for swap batches); ``loser`` is
    ``game.BLACK`` / ``game.WHITE``. Returns per-game classification."""
    board = game.empty_board()
    missed_blocks: list[int] = []
    missed_preventions: list[dict] = []
    threatened_turns = 0
    # The winner is the colour of the final move.
    winner = game.BLACK if (len(moves) - 1) % 2 == 0 else game.WHITE
    # "Via open four" is judged on the board BEFORE the loser's last
    # move (i.e. after the winner's second-to-last move): the
    # standard open-four win is winner creates the open four, loser
    # blocks one square, winner completes the five — so on the board
    # before the FINAL move the winner has only one winning square
    # left (the other was blocked), and judging there would miss
    # nearly every open-four loss (validated: 1/37 on the v6-vs-v1
    # batch, vs 37/37 with this definition — see iter-6 §2.1).
    board_before_loser_last = None
    for ply, move in enumerate(moves):
        if ply == len(moves) - 2:
            board_before_loser_last = board.copy()
        player = game.BLACK if ply % 2 == 0 else game.WHITE
        if player == loser:
            opp_wins = scoring._winning_moves(board, -loser)
            if len(opp_wins) == 1 and move != opp_wins[0]:
                # Missed block — unless our move wins at once.
                if not game.is_winning_move(board, move, loser):
                    missed_blocks.append(ply)
            elif len(opp_wins) == 0:
                creators = scoring._opponent_creators(board, loser)
                if creators:
                    threatened_turns += 1
                    if not scoring._move_is_safe(board, move, loser, creators):
                        safe_exists = any(
                            scoring._move_is_safe(board, m, loser, creators)
                            for m in scoring._candidate_moves(board)
                        )
                        if safe_exists:
                            missed_preventions.append({
                                "ply": ply,
                                "played": [move % 15, move // 15],
                                "creators": len(creators),
                            })
        board = game.apply_move(board, move, player)
    via_open_four = False
    if board_before_loser_last is not None:
        via_open_four = (
            len(scoring._winning_moves(board_before_loser_last, winner)) >= 2
        )
    return {
        "plies": len(moves),
        "loser_colour": "black" if loser == game.BLACK else "white",
        "via_open_four": via_open_four,
        "missed_blocks": missed_blocks,
        "missed_preventions": missed_preventions,
        "threatened_turns": threatened_turns,
    }


def classify_batch(path: str) -> dict:
    """Classify every loss of engine A in one batch JSON."""
    with open(path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    losses = []
    for rec in batch["records"]:
        if rec["winner"] == "draw" or rec.get("winner_is_a") is not False:
            continue
        loser = (
            game.BLACK if rec["a_colour"] == "black" else game.WHITE
        )
        detail = analyse_loss(rec["moves"], loser)
        detail["game"] = rec["game"]
        losses.append(detail)
    n = len(losses)
    return {
        "batch": os.path.basename(path),
        "engine_a": batch.get("engine_a"),
        "engine_b": batch.get("engine_b"),
        "games": batch.get("games"),
        "losses": n,
        "via_open_four": sum(1 for d in losses if d["via_open_four"]),
        "missed_blocks": sum(len(d["missed_blocks"]) for d in losses),
        "missed_block_games": sum(1 for d in losses if d["missed_blocks"]),
        "clean_missed_preventions": sum(
            len(d["missed_preventions"]) for d in losses
        ),
        "missed_prevention_games": sum(
            1 for d in losses if d["missed_preventions"]
        ),
        "threatened_turns": sum(d["threatened_turns"] for d in losses),
        "details": losses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batches", nargs="+", help="batch JSON path(s)")
    parser.add_argument("--out", default=None, help="JSON summary path")
    args = parser.parse_args()

    summaries = []
    for path in args.batches:
        print(f"classifying {path} ...", file=sys.stderr, flush=True)
        summary = classify_batch(path)
        summaries.append(summary)
        print(
            f"  {summary['batch']}: {summary['losses']} losses, "
            f"{summary['via_open_four']} via open four, "
            f"{summary['missed_blocks']} missed blocks "
            f"({summary['missed_block_games']} games), "
            f"{summary['clean_missed_preventions']} clean missed "
            f"preventions ({summary['missed_prevention_games']} games) "
            f"in {summary['threatened_turns']} threatened turns"
        )
    if len(summaries) > 1:
        pooled = {
            "losses": sum(s["losses"] for s in summaries),
            "via_open_four": sum(s["via_open_four"] for s in summaries),
            "missed_blocks": sum(s["missed_blocks"] for s in summaries),
            "clean_missed_preventions": sum(
                s["clean_missed_preventions"] for s in summaries
            ),
            "threatened_turns": sum(s["threatened_turns"] for s in summaries),
        }
        print(f"POOLED: {pooled['losses']} losses, "
              f"{pooled['via_open_four']} via open four, "
              f"{pooled['missed_blocks']} missed blocks, "
              f"{pooled['clean_missed_preventions']} clean missed "
              f"preventions in {pooled['threatened_turns']} threatened turns")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
            f.write("\n")
        print(f"details written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

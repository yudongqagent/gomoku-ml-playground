#!/usr/bin/env python3
"""Loss classifier for self-play batch records (batch 3).

Replays every game in a ``selfplay_batch.py`` JSON and classifies
engine A's losses:

* **Loss via open four:** the winner held >= 2 winning moves at the
  start of the loser's last turn (2 plies before the end). Checked
  there, NOT on the board before the final move: when the loser
  blocks one of the two winning moves, only 1 remains before the
  final move, and the naive check misses the open four (found in
  the v7 smoke: both losses had the winner holding 2 winning moves
  before v7's last turn; v7 blocked one, the winner took the
  other). The loser's move can only REMOVE the winner's winning
  moves (by occupying one), never add them, so >= 2 before the
  final move implies >= 2 before the loser's last turn — the
  earlier check subsumes the later one.
* **Missed block:** on A's turn the opponent had exactly 1 winning
  move and A did not play it (>= 2 winning moves is unblockable
  and is NOT a missed block). Taking A's OWN immediate win instead
  of blocking is NOT a miss — the game ends with A's win (the
  win scan runs before the block scan in every version).
* **Threatened turn:** on A's turn the opponent held >= 1
  open-four creator and no immediate winning move (prevention was
  relevant). Headline counts follow the batch-2 convention:
  threatened turns and missed preventions are counted on A's
  turns in A's LOSSES only (iter-6's 1,061 threatened turns =
  221 + 510 + 330 across the three v6 matches, all in losses);
  all-games counts are reported alongside as secondary numbers.
* **Clean missed prevention:** on a threatened turn, A's move was
  not preventing (not forcing and left a creator live) while a
  preventing alternative existed. If NO preventing move existed,
  the game was already lost — not counted as a miss.

Definitions match batch 1/2 (see ``research/iterations/iter-5.md``
§2.1 and ``tactical-suite-v2.md``); the batch-2 classifier scripts
lived in /tmp and are gone, so this is a fresh implementation from
those definitions. ``scoring._opponent_creators`` /
``scoring._move_is_safe`` do the creator work.

Usage:
    python ml/scripts/classify_batch_losses.py \\
        research/iterations/batch-v7-vs-v1.json [--json-out OUT]
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


def _replay(moves: list[int]) -> list:
    """Boards before each ply (index i = board before moves[i])."""
    boards = [game.empty_board()]
    board = game.empty_board()
    for i, m in enumerate(moves):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE
        )
        boards.append(board)
    return boards


def _is_preventing(board, move, player, creators) -> bool:
    return scoring._move_is_safe(board, move, player, creators)


def _any_preventing(board, player, creators) -> bool:
    for m in game.legal_moves(board):
        if _is_preventing(board, int(m), player, creators):
            return True
    return False


def classify_game(rec: dict) -> dict:
    """Classify one recorded game from A's perspective."""
    moves = [int(m) for m in rec["moves"]]
    boards = _replay(moves)
    a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
    winner = (
        game.BLACK if rec["winner"] == "black"
        else game.WHITE if rec["winner"] == "white" else game.EMPTY
    )
    a_lost = winner != game.EMPTY and winner != a_colour
    out: dict = {
        "game": rec["game"],
        "a_lost": a_lost,
        "loss_via_open_four": None,
        "missed_blocks": [],
        "threatened_turns": 0,
        "missed_preventions": [],
    }
    # Loss via open four: the winner held >= 2 winning moves at the
    # start of the loser's last turn (2 plies before the end) — see
    # the module docstring for why the board before the final move
    # is the wrong place to check.
    if a_lost and len(moves) >= 2:
        board_before_loser_last = boards[len(moves) - 2]
        wins = scoring._winning_moves(board_before_loser_last, winner)
        out["loss_via_open_four"] = len(wins) >= 2
    # Walk A's turns: missed blocks and clean missed preventions.
    for ply, move in enumerate(moves):
        player = game.BLACK if ply % 2 == 0 else game.WHITE
        if player != a_colour:
            continue
        board = boards[ply]
        opp = -player
        opp_wins = scoring._winning_moves(board, opp)
        if len(opp_wins) == 1 and move != opp_wins[0]:
            # Taking our own immediate win instead of blocking is
            # not a miss — the game ends with our win.
            b2 = game.apply_move(board, move, player)
            if not game.is_winning_stone(b2, move, player):
                out["missed_blocks"].append({"ply": ply, "block": int(opp_wins[0]),
                                             "played": int(move)})
        if opp_wins:
            continue  # prevention is moot when a win is on the board
        creators = scoring._opponent_creators(board, player)
        if not creators:
            continue
        out["threatened_turns"] += 1
        if not _is_preventing(board, move, player, creators):
            if _any_preventing(board, player, creators):
                out["missed_preventions"].append(
                    {"ply": ply, "played": int(move),
                     "creators": [int(c) for c in creators]}
                )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_json")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    with open(args.batch_json, encoding="utf-8") as f:
        batch = json.load(f)
    games = [classify_game(rec) for rec in batch["records"]]
    losses = [g for g in games if g["a_lost"]]
    summary = {
        "batch": os.path.basename(args.batch_json),
        "engine_a": batch.get("engine_a"),
        "engine_b": batch.get("engine_b"),
        "games": batch.get("games"),
        "losses": len(losses),
        "losses_via_open_four": sum(
            1 for g in losses if g["loss_via_open_four"]
        ),
        "missed_blocks": sum(len(g["missed_blocks"]) for g in games),
        "missed_blocks_in_losses": sum(
            len(g["missed_blocks"]) for g in losses
        ),
        "threatened_turns": sum(g["threatened_turns"] for g in games),
        "threatened_turns_in_losses": sum(
            g["threatened_turns"] for g in losses
        ),
        "clean_missed_preventions": sum(
            len(g["missed_preventions"]) for g in games
        ),
        "clean_missed_preventions_in_losses": sum(
            len(g["missed_preventions"]) for g in losses
        ),
        "games_with_missed_prevention": sum(
            1 for g in games if g["missed_preventions"]
        ),
        "per_game": games,
    }
    print(f"== {summary['batch']}: {summary['engine_a']} vs "
          f"{summary['engine_b']} ==")
    print(f"  losses: {summary['losses']}, via open four: "
          f"{summary['losses_via_open_four']}")
    print(f"  missed blocks (all games): {summary['missed_blocks']} "
          f"(in losses: {summary['missed_blocks_in_losses']})")
    print(f"  threatened turns in losses: "
          f"{summary['threatened_turns_in_losses']} "
          f"(all games: {summary['threatened_turns']})")
    print(f"  clean missed preventions in losses: "
          f"{summary['clean_missed_preventions_in_losses']} "
          f"(all games: {summary['clean_missed_preventions']}, "
          f"in {summary['games_with_missed_prevention']} games)")
    if summary["threatened_turns_in_losses"]:
        rate = (summary["clean_missed_preventions_in_losses"]
                / summary["threatened_turns_in_losses"])
        print(f"  missed-prevention rate (in losses): {rate:.1%} "
              f"of threatened turns")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"  details written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

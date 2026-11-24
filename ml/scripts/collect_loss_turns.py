#!/usr/bin/env python3
"""M3b for v58 (iter-58 §2.4, POST-HOC robustness check): collect
qualifying White turns from v52-White's NON-WON games (losses +
draws) in a battery JSON, for a depth-6 spot-check on the
decision-relevant population — the games a deeper TSS would have
had to win.

M3 (pre-registered) sampled qualifying turns from v27-vs-v27 mining
games and found 0/20. M3b asks the same question in v52's own lost
games: at White turns where production TSS (depth 4) found nothing
but forcing moves exist, does depth-6 TSS find a forcing win? If
yes, a deeper TSS would have won some of those losses — a real
patch. If no, the losses are unwinnable by TSS at any depth — a
pure opportunity gap.

A turn qualifies if: production TSS returns None AND the forcing
move set is non-empty AND Black holds no winning move (no forced
block) AND the search completed (budget NOT exceeded). The first
MAX_QUALIFYING turns (default 30) are recorded, in game order.

Deterministic (TSS is deterministic, the games are recorded), so it
checkpoints after every game and resumes exactly. Output JSON has
the same ``qualifying`` + ``rule`` shape as the M1 mining JSONs, so
``spotcheck_tss_depth.py`` runs on it unchanged.

Usage:
    PYTHONPATH=ml python ml/scripts/collect_loss_turns.py
        --in research/iterations/batch-v52-vs-v27.json
        --out research/iterations/tss-loss-turns-v58-normal.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

MAX_QUALIFYING = 30


def _board_to_lists(board):
    black = [[int(x), int(y)] for y, x in zip(*((board == 1).nonzero()))]
    white = [[int(x), int(y)] for y, x in zip(*((board == -1).nonzero()))]
    return black, white


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-qualifying", type=int, default=MAX_QUALIFYING)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic, scoring

    in_path = args.inp
    if not os.path.isabs(in_path):
        in_path = os.path.join(REPO_ROOT, in_path)
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    ckpt_path = args.checkpoint or (out_path + ".checkpoint.json")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    with open(in_path, encoding="utf-8") as f:
        batch = json.load(f)
    records = batch["records"]
    swap_rule = bool(batch.get("swap_rule"))
    # v52-White non-wins (losses + draws): engine A is v52, White.
    target = [r for r in records
              if r["a_colour"] == "white" and not r["winner_is_a"]]

    qualifying = []
    white_turns = 0
    tss_none = 0
    start_i = 0
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if (ckpt.get("input") == in_path
                    and ckpt.get("n_target") == len(target)):
                qualifying = ckpt["qualifying"]
                white_turns = ckpt["white_turns"]
                tss_none = ckpt["tss_none"]
                start_i = ckpt["games_done"]
                print(f"resuming from checkpoint: {start_i}/"
                      f"{len(target)} games done", file=sys.stderr)
        except (OSError, ValueError, KeyError):
            print("checkpoint unreadable — starting from scratch",
                  file=sys.stderr)

    for ti in range(start_i, len(target)):
        rec = target[ti]
        moves = rec["moves"]
        board = game.empty_board()
        start = 0
        if swap_rule:
            for i in range(3):
                board = game.apply_move(
                    board, moves[i],
                    game.BLACK if i % 2 == 0 else game.WHITE)
            start = 3
        for i in range(start, len(moves)):
            player = game.BLACK if i % 2 == 0 else game.WHITE
            if player == game.WHITE:
                white_turns += 1
                tss_stats: dict = {}
                tss_move = mcts_heuristic.tss_winning_move(
                    board, player, stats=tss_stats)
                if tss_move is None:
                    tss_none += 1
                    if (len(qualifying) < args.max_qualifying
                            and not tss_stats.get("budget_exceeded")
                            and not scoring._winning_moves(
                                board, game.BLACK)
                            and mcts_heuristic.tss_forcing_moves(
                                board, player)):
                        black, white = _board_to_lists(board)
                        qualifying.append({
                            "game": rec["game"],
                            "ply": i,
                            "black": black,
                            "white": white,
                        })
            board = game.apply_move(board, moves[i], player)
        print(f"game {ti + 1}/{len(target)} (battery game "
              f"{rec['game']}, {rec['winner']}): white_turns="
              f"{white_turns} tss_none={tss_none} qualifying="
              f"{len(qualifying)}", file=sys.stderr)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "input": in_path,
                "n_target": len(target),
                "games_done": ti + 1,
                "qualifying": qualifying,
                "white_turns": white_turns,
                "tss_none": tss_none,
            }, f)

    summary = {
        "input": in_path,
        "rule": "swap" if swap_rule else "normal",
        "population": "v52-White non-wins (losses + draws)",
        "games": len(target),
        "white_turns": white_turns,
        "tss_none_turns": tss_none,
        "qualifying_count": len(qualifying),
        "qualifying": qualifying,
    }
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "qualifying"},
        indent=2,
    ))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

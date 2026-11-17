#!/usr/bin/env python3
"""Screen lost games for VCF-prevention opportunities (iter-32 §1.1).

For each loss of side A in a recorded match, at each of A's last
``--last-turns`` turns, asks: did the opponent hold a VCF start,
and would the defensive VCF check have fired (playing a
prevention different from the recorded move)? The offensive step
is tested first so an offensive fire is not mistaken for a
prevention opportunity, and win/block-scan turns are skipped
(the check never runs there).

This is the cheap exact screen for the defensive-inclusion rule:
a loss can only motivate a defensive check if a prevention
opportunity existed at one of its turns. A screen hit still needs
a playout-verified save (``workup_loss.py``) before it counts.

Usage:
    PYTHONPATH=ml python ml/scripts/screen_vcf_preventions.py \
        research/iterations/fast-v32-vs-v8-iter1.json [--last-turns 4]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("match")
    ap.add_argument("--last-turns", type=int, default=4)
    args = ap.parse_args()

    from gomoku_ml import game, scoring
    from gomoku_ml.mcts_heuristic import HeuristicMCTS

    with open(args.match) as f:
        data = json.load(f)
    mcts = HeuristicMCTS(sims=100, vcf_root=True, rng=np.random.default_rng(0))

    n_losses = 0
    n_hits = 0
    for rec in data["records"]:
        if rec.get("winner_is_a", None) is not False:
            # winner_is_a False means A lost; draws (None) and
            # wins (True) are skipped.
            continue
        n_losses += 1
        board = game.empty_board()
        a_colour = rec["a_colour"]
        a_player = game.BLACK if a_colour in ("black", 1) else game.WHITE
        # Collect A's turns (ply, board-before-move, recorded move).
        a_turns: list[tuple[int, np.ndarray, int]] = []
        player = game.BLACK
        for ply, move in enumerate(rec["moves"]):
            if player == a_player:
                a_turns.append((ply, board.copy(), move))
            board = game.apply_move(board, move, player)
            player = -player
        hits = []
        for ply, b, move in a_turns[-args.last_turns:]:
            wins = scoring._winning_moves(b, a_player)
            opp_wins = scoring._winning_moves(b, -a_player)
            cands = scoring._candidate_moves(b)
            if wins or len(opp_wins) == 1 or len(cands) <= 1:
                continue  # scan turn — the check never runs
            offensive = mcts._vcf_root_move(b, a_player, defensive=False)
            if offensive is not None:
                continue  # offensive fire, not a prevention
            prevention = mcts._vcf_root_move(b, a_player, defensive=True)
            if prevention is not None and prevention != move:
                hits.append((ply, game.move_to_xy(prevention),
                             game.move_to_xy(move)))
        if hits:
            n_hits += 1
            print(f"game {rec['game']} ({a_colour}, {rec['plies']} plies): "
                  f"prevention opportunities at:")
            for ply, prev, actual in hits:
                print(f"  ply {ply}: check would play {prev}, "
                      f"actual was {actual}")
    print(f"losses screened: {n_losses}, with a prevention "
          f"opportunity in the last {args.last_turns} A-turns: {n_hits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

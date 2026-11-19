#!/usr/bin/env python3
"""Diagnose the setup check's fire rate in fast-loop games (batch 11 H1).

v39 == v27 + the setup check, so at any v39 turn the check fired
iff v39's played move differs from v27's move at the same position
AND the played move is a full setup removal. (The search is
identical; the RNG only breaks exact ties, so a fresh-seed v27
move agrees with the in-game search's move except on ties — and a
tie-break difference would not produce a removal.)

For each sampled v39 turn this records: v27's move, v39's played
move, whether they differ, whether the played move is a full
removal, and the gate state after v27's move (opponent setups).
Sampled turns: every 5th v39 turn in each game, plus all v39
turns in the last 20 plies of each loss (the losing moves).

Usage:
    python research/iterations/diagnose_check.py \
        research/iterations/fast-v39-vs-v27.json \
        --out research/iterations/diag-v39-vs-v27.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml", "scripts"))

import numpy as np  # noqa: E402

from gomoku_ml import game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import (  # noqa: E402
    _creators_windowed, _has_vcf_start, _setups_limited)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", help="fast-loop batch JSON")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample", type=int, default=5,
                        help="sample every Nth v39 turn")
    args = parser.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        batch = json.load(f)

    results: dict = {"games": []}
    done: set[int] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done = {g["game"] for g in results["games"]}
        print(f"resuming: {len(done)} games already diagnosed",
              file=sys.stderr, flush=True)

    mcts27 = scoring.get_player(27, seed=0)._mcts
    for rec in batch["records"]:
        if rec["game"] in done:
            continue
        a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
        moves = rec["moves"]
        result = ("W" if rec["winner_is_a"] is True
                  else "L" if rec["winner_is_a"] is False else "D")
        print(f"== game {rec['game']} ({result}, {len(moves)} plies, "
              f"v39 as {rec['a_colour']})", flush=True)
        entry = {"game": rec["game"], "result": result,
                 "a_colour": rec["a_colour"], "turns": []}
        # v39's turns: plies where the side to move is a_colour.
        v39_plies = [p for p in range(len(moves))
                     if (game.BLACK if p % 2 == 0 else game.WHITE)
                     == a_colour]
        last20 = set(v39_plies[-20:]) if result == "L" else set()
        sampled = [p for i, p in enumerate(v39_plies)
                   if i % args.sample == 0 or p in last20]
        for ply in sampled:
            board = game.empty_board()
            for i, m in enumerate(moves[:ply]):
                board = game.apply_move(
                    board, m, game.BLACK if i % 2 == 0 else game.WHITE)
            mcts27.rng = np.random.default_rng(0)
            m27 = mcts27.choose_move(board, a_colour)
            played = moves[ply]
            # Gate state after v27's move.
            b27 = game.apply_move(board, m27, a_colour)
            opp = -a_colour
            gate_vcf = _has_vcf_start(b27, opp)
            gate_cr = len(_creators_windowed(b27, opp, limit=1))
            gate_st = (len(_setups_limited(b27, opp, 0))
                       if not gate_cr else None)
            is_removal = mcts27._is_setup_removal(board, a_colour,
                                                  played)
            fired = (played != m27) and is_removal
            entry["turns"].append({
                "ply": ply,
                "v27_move": [m27 % 15, m27 // 15],
                "played": [played % 15, played // 15],
                "differs": played != m27,
                "played_is_removal": is_removal,
                "check_fired": fired,
                "gate_after_v27": {"vcf": gate_vcf,
                                   "creators": gate_cr,
                                   "setups": gate_st},
            })
            if fired:
                print(f"   ply {ply}: CHECK FIRED — v27 "
                      f"{[m27 % 15, m27 // 15]} -> v39 "
                      f"{[played % 15, played // 15]} "
                      f"(gate: {gate_st} setups)", flush=True)
        n = len(entry["turns"])
        n_fired = sum(1 for t in entry["turns"] if t["check_fired"])
        n_gate = sum(1 for t in entry["turns"]
                     if t["gate_after_v27"]["setups"])
        print(f"   -> {n_fired}/{n} sampled turns overrode "
              f"({n_gate}/{n} gate-open after v27's move)", flush=True)
        results["games"].append(entry)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

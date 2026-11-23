#!/usr/bin/env python3
"""Check A for the v58 mechanism check (iter-58 §11.3): on the
mined miss positions (White held forcing moves, v52 played a
non-forcing move), does v58 play a forcing move?

Procedure (pre-registered in iter-58 §11.3 BEFORE this script ran):

* Load the collected positions (collect_forcing_miss_positions.py
  output). TSS is silent on all of them by construction, so any
  forcing move v58 plays comes from the MCTS + threat term, not
  the root hook.
* Run v58 (seed 100) on each position; count how often it plays a
  forcing move (recomputed via ``tss_forcing_moves``).
* Control: run v52 (seed 100) on the same positions (informational
  only — v52's isolated-run move may differ from the recorded game
  move because the RNG state differs from the battery's per-game
  seeding; the decision is on v58's count vs the pre-registered
  thresholds).

Decision (pre-registered): PASS if v58 plays a forcing move on
>= 6/20 (30%); FAIL if <= 2/20 (10%); WEAK 3-5/20. Check A is
diagnostic — the battery decision turns on Check B (iter-58
§11.3).

Usage:
    PYTHONPATH=ml python ml/scripts/check_threat_term_mechanism.py
        --in research/iterations/threat-miss-positions-v58.json
        --out research/iterations/threat-mechanism-check-a-v58.json
        [--seed 100]
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
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic, scoring

    in_path = args.inp
    if not os.path.isabs(in_path):
        in_path = os.path.join(REPO_ROOT, in_path)
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)

    with open(in_path, encoding="utf-8") as f:
        data = json.load(f)
    positions = data["positions"]

    results = []
    n58_forcing = 0
    n52_forcing = 0
    for i, pos in enumerate(positions):
        board = game.empty_board()
        for x, y in pos["black"]:
            board[y, x] = game.BLACK
        for x, y in pos["white"]:
            board[y, x] = game.WHITE
        player = game.WHITE
        forcing = set(mcts_heuristic.tss_forcing_moves(board, player))
        assert forcing, f"position {i}: no forcing moves on recompute"
        # TSS silence re-verified (belt and braces).
        assert (
            mcts_heuristic.tss_winning_move(board, player) is None
        ), f"position {i}: TSS fires (should be silent)"
        e58 = scoring.get_player(58, seed=args.seed)
        m58 = e58.choose_move(board.copy(), player)
        e52 = scoring.get_player(52, seed=args.seed)
        m52 = e52.choose_move(board.copy(), player)
        f58 = m58 in forcing
        f52 = m52 in forcing
        n58_forcing += 1 if f58 else 0
        n52_forcing += 1 if f52 else 0
        results.append(
            {
                "index": i,
                "game": pos["game"],
                "ply": pos["ply"],
                "recorded_move": pos["recorded_move"],
                "forcing_moves": sorted(forcing),
                "v58_move": m58,
                "v58_forcing": f58,
                "v52_move": m52,
                "v52_forcing": f52,
            }
        )
        print(
            f"  [{i}] game {pos['game']} ply {pos['ply']}: "
            f"v58 -> {(m58 % 15, m58 // 15)} "
            f"({'FORCING' if f58 else 'quiet'}), "
            f"v52 -> {(m52 % 15, m52 // 15)} "
            f"({'FORCING' if f52 else 'quiet'})",
            flush=True,
        )

    n = len(positions)
    if n58_forcing >= 6:
        verdict = "PASS"
    elif n58_forcing <= 2:
        verdict = "FAIL"
    else:
        verdict = "WEAK"
    print(
        f"Check A: v58 plays a forcing move on {n58_forcing}/{n} "
        f"({n58_forcing / n:.2f}); v52 control {n52_forcing}/{n}. "
        f"Verdict: {verdict} "
        f"(PASS >= 6/20, FAIL <= 2/20, WEAK 3-5/20)"
    )
    result = {
        "source": os.path.basename(in_path),
        "seed": args.seed,
        "n": n,
        "v58_forcing": n58_forcing,
        "v52_forcing_control": n52_forcing,
        "verdict": verdict,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Position probe for the v13 White-prior-temperature design
(iter-13 §1b, step 1).

Runs V13MCTS at each temperature in {2.0, 3.0, 4.0} on the 7
White-to-move diagnostic positions (the 2 White critical
positions from fast-loop iteration 1 + the 5 White v10
regression positions) and counts oracle-verified saves played.
Proceed to the match sweep ONLY if some T > 2.0 plays strictly
more saves than the T = 2.0 control.

Also reports the §1b mechanism check separately (NOT counted in
the 7): the g9-ply-5 opening position, where v13-as-White should
play (9,7) or (4,7) instead of the losing block (6,7).

This is a DIAGNOSTIC, not an eval: it gates the match sweep, it
does not decide the bar.

Usage:
    python ml/scripts/probe_white_temp.py
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

from gomoku_ml import game  # noqa: E402
from gomoku_ml.mcts_v13 import V13MCTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GOMOKU_ML = os.path.join(HERE, "..", "gomoku_ml")
ITERATIONS = os.path.join(HERE, "..", "..", "research", "iterations")

TEMPS = (2.0, 3.0, 4.0)


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


def _load_white_positions():
    """The 7 White-to-move diagnostic positions (iter-13 §1b)."""
    out = []
    for fname in ("regression_positions_v13.json",
                  "regression_positions_v10.json"):
        with open(os.path.join(GOMOKU_ML, fname), encoding="utf-8") as f:
            for pos in json.load(f)["positions"]:
                if pos["to_move"] == "white":
                    out.append(pos)
    return out


def _mechanism_check_position():
    """g9 ply 5 (fastloop1-v13-vs-v7 game 9): the §1b mechanism
    check. Reconstructed from the game record; expected saves
    (9,7)/(4,7) from the walk-back JSON."""
    with open(os.path.join(ITERATIONS, "fastloop1-v13-vs-v7.json"),
              encoding="utf-8") as f:
        batch = json.load(f)
    rec = next(r for r in batch["records"] if r["game"] == 9)
    board = game.empty_board()
    for i, m in enumerate(rec["moves"][:5]):
        board[m // 15, m % 15] = game.BLACK if i % 2 == 0 else game.WHITE
    return {
        "id": "fl1-g9-ply5-mechanism-check",
        "board": board,
        "expected": {(9, 7), (4, 7)},
        "losing": (6, 7),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temps", default="2.0,3.0,4.0")
    args = parser.parse_args()
    temps = [float(t) for t in args.temps.split(",")]

    positions = _load_white_positions()
    assert len(positions) == 7, f"expected 7 White positions, got {len(positions)}"
    mech = _mechanism_check_position()

    totals = {}
    for temp in temps:
        mcts = V13MCTS(sims=100, white_sims=100, white_temp=temp,
                       rng=np.random.default_rng(0))
        n_saved = 0
        print(f"== white_temp={temp} ==")
        for pos in positions:
            board = _board(pos)
            move = mcts.choose_move(board, game.WHITE)
            chosen = (move % 15, move // 15)
            saves = {tuple(e) for e in pos["expected"]}
            is_save = chosen in saves
            n_saved += int(is_save)
            print(f"  {pos['id']}: chose {chosen} "
                  f"{'SAVE' if is_save else 'not a save'}; "
                  f"expected {sorted(saves)}")
        # Mechanism check (reported separately, not counted).
        move = mcts.choose_move(mech["board"].copy(), game.WHITE)
        chosen = (move % 15, move // 15)
        print(f"  MECHANISM CHECK {mech['id']}: chose {chosen} "
              f"({'SAVE' if chosen in mech['expected'] else 'not a save; losing was ' + str(mech['losing'])}); "
              f"expected {sorted(mech['expected'])}")
        totals[temp] = n_saved
        print(f"  TOTAL: {n_saved}/{len(positions)} saves "
              f"(white_temp={temp})")
    control = totals.get(2.0)
    best = max(totals.items(), key=lambda kv: kv[1])
    print(f"CONTROL (T=2.0): {control}/7 saves; "
          f"best: T={best[0]} with {best[1]}/7")
    if control is not None and best[1] > control:
        print(f"PROBE PASSES: T={best[0]} beats the control "
              f"({best[1]} > {control}) — proceed to the match sweep.")
    else:
        print("PROBE FAILS: no temperature beats the control — "
              "the temperature hypothesis is refuted on arrival "
              "(iter-13 §1b step 1).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

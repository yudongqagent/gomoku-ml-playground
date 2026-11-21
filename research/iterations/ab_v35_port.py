#!/usr/bin/env python3
"""A/B port verification for the v35 port (batch 12 H1).

Runs v35.choose_move (seed 0) on a fixed position set — the 8 v35
regression positions, all 49 tactical-suite positions (20 old +
29 hard), and batch-v33-vs-v27 g0 ply 29 — and dumps the chosen
move plus the VCF-check internals (_vcf_root_move,
_vcf_offensive_move, _vcf_defensive_move, _vcf_prevention_pool,
_leaves_opp_vcf on the chosen move) to JSON. The FULL prevention
pool is NOT compared here (its scan has no early exit and takes
~60s per position); it is verified separately on 2 positions by
pool_ab.py. Run this script in
BOTH the batch-10 tree and the port tree with the same Python;
the outputs must be IDENTICAL (any difference is a port defect).

The script resolves all paths relative to itself, so the same
file runs unmodified in either worktree.

Usage (from the worktree root):
    python research/iterations/ab_v35_port.py \
        --out research/iterations/ab-v35-port-<tree>.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml"))

import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, scoring  # noqa: E402

# batch-v33-vs-v27 g0, moves[:29] (White to move at ply 29).
G0_MOVES_29 = [112, 96, 114, 113, 141, 84, 68, 94, 97, 127, 125,
               109, 83, 111, 79, 66, 81, 80, 52, 108, 122, 107,
               110, 82, 106, 23, 154, 170, 138]


def _xy(m: int | None) -> list[int] | None:
    return None if m is None else [m % 15, m // 15]


def _board_from_stones(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.join(here, "..", "..", "ml", "gomoku_ml")

    positions: list[tuple[str, object, int]] = []

    with open(os.path.join(ml_dir, "regression_positions_v35.json"),
              encoding="utf-8") as f:
        for pos in json.load(f)["positions"]:
            board = _board_from_stones(pos["black"], pos["white"])
            player = (game.BLACK if pos["to_move"] == "black"
                      else game.WHITE)
            positions.append((f"v35-{pos['id']}", board, player))

    for fname in ("tactical_positions.json",
                  "tactical_positions_hard.json"):
        with open(os.path.join(ml_dir, fname),
                  encoding="utf-8") as f:
            for pos in json.load(f)["positions"]:
                board = _board_from_stones(pos["black"], pos["white"])
                positions.append(
                    (f"{fname[:-5]}-{pos['id']}", board,
                     int(pos["to_move"])))

    board = game.empty_board()
    for i, m in enumerate(G0_MOVES_29):
        board = game.apply_move(
            board, m, game.BLACK if i % 2 == 0 else game.WHITE)
    positions.append(("g0-ply29", board, game.WHITE))

    v35 = scoring.get_player(35, seed=0)
    mcts = v35._mcts
    assert mcts.vcf_root and mcts.vcf_root_search_first
    # vcf_full_recheck exists only in the port tree (default False
    # there); batch-10 has no such attribute at all.
    assert not getattr(mcts, "vcf_full_recheck", False)

    out: list[dict] = []
    for pid, board, player in positions:
        move = v35.choose_move(board, player)
        rec = {
            "id": pid,
            "choose_move": _xy(move),
            "vcf_root_move": _xy(mcts._vcf_root_move(board, player)),
            "vcf_offensive": _xy(
                mcts._vcf_offensive_move(board, player)),
            "vcf_defensive": _xy(
                mcts._vcf_defensive_move(board, player)),
            "leaves_opp_vcf": mcts._leaves_opp_vcf(
                board, player, move),
        }
        out.append(rec)
        print(f"{pid}: move={_xy(move)} "
              f"root={rec['vcf_root_move']} "
              f"leaves={rec['leaves_opp_vcf']}",
              flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"positions": out}, f, indent=2)
    print(f"DONE: {len(out)} positions", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

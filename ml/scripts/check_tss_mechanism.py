#!/usr/bin/env python3
"""Mechanism check A for v52 (iter-52 §1.3): does TSS find a forcing
win the MCTS misses on a set of pre-registered test positions?

Positions (all pre-registered in iter-52 §1.3 BEFORE this script ran):

* T1-T4: constructed positions in
  ``ml/gomoku_ml/tss_mechanism_positions.json`` (double open three
  x2, open-three-sequence x2) — TSS must return a move in the
  expected set;
* T5-T6: VCF controls from the tactical suites (vcf-1,
  vcf-two-four-sequence) — TSS must return a VCF start;
* T7: the counter-four refutation position — TSS must return None.

For each position the script also runs v27 (seed 100) and reports
whether v27 plays a move in the expected set — the MCTS-miss
sub-criterion: on >=1 of T1-T4, v27 plays outside the expected set
while TSS returns a move inside it.

Pass criterion (pre-registered): TSS correct on >=6/7 positions.

Usage:
    PYTHONPATH=ml python ml/scripts/check_tss_mechanism.py
        --out research/iterations/tss-mechanism-check-a.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

VCF_CONTROLS = [
    ("ml/gomoku_ml/tactical_positions.json", "vcf-1"),
    ("ml/gomoku_ml/tactical_positions_hard.json", "vcf-two-four-sequence"),
]


def _xy(move):
    return [move % 15, move // 15] if move is not None else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    from gomoku_ml import evaluate, game, mcts_heuristic, scoring

    with open(
        os.path.join(
            REPO_ROOT, "ml/gomoku_ml/tss_mechanism_positions.json"
        ),
        encoding="utf-8",
    ) as f:
        constructed = json.load(f)["positions"]

    results = []
    for pos in constructed:
        board = game.empty_board()
        for x, y in pos["black"]:
            board[y, x] = game.BLACK
        for x, y in pos["white"]:
            board[y, x] = game.WHITE
        player = game.BLACK if pos["to_move"] == "black" else game.WHITE
        tss_move = mcts_heuristic.tss_winning_move(board, player)
        v27 = scoring.get_player(27, seed=args.seed)
        v27_move = v27.choose_move(board, player)
        expected = [tuple(e) for e in pos["expected"]]
        tss_ok = (
            (tss_move is None and not expected)
            or (tss_move is not None and tuple(_xy(tss_move)) in expected)
        )
        v27_ok = v27_move is not None and tuple(_xy(v27_move)) in expected
        results.append({
            "id": pos["id"],
            "to_move": pos["to_move"],
            "expected": pos["expected"],
            "tss_move": _xy(tss_move),
            "tss_correct": tss_ok,
            "v27_move": _xy(v27_move),
            "v27_correct": v27_ok,
            "mcts_miss": bool(tss_ok and not v27_ok),
        })
    for suite_rel, pos_id in VCF_CONTROLS:
        with open(os.path.join(REPO_ROOT, suite_rel), encoding="utf-8") as f:
            suite = {p["id"]: p for p in json.load(f)["positions"]}
        pos = suite[pos_id]
        board = evaluate.board_from_position(pos)
        player = pos["to_move"]
        tss_move = mcts_heuristic.tss_winning_move(board, player)
        v27 = scoring.get_player(27, seed=args.seed)
        v27_move = v27.choose_move(board, player)
        tss_ok = tss_move is not None and evaluate.starts_forced_win(
            board, player, tss_move, depth=3
        )
        v27_ok = v27_move is not None and evaluate.starts_forced_win(
            board, player, v27_move, depth=3
        )
        results.append({
            "id": pos_id,
            "to_move": "black" if player == game.BLACK else "white",
            "expected": "vcf",
            "tss_move": _xy(tss_move),
            "tss_correct": tss_ok,
            "v27_move": _xy(v27_move),
            "v27_correct": v27_ok,
            "mcts_miss": bool(tss_ok and not v27_ok),
        })

    n_correct = sum(1 for r in results if r["tss_correct"])
    n_miss = sum(
        1 for r in results
        if r["mcts_miss"]
        and (
            r["id"].startswith("tss-double-three")
            or r["id"].startswith("tss-three-sequence")
        )
    )
    summary = {
        "positions": len(results),
        "tss_correct": n_correct,
        "criterion": "tss_correct >= 6",
        "check_a_pass": n_correct >= 6,
        "mcts_miss_constructed": n_miss,
        "mcts_miss_criterion": ">= 1 on T1-T4",
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Broad scan for live hole instances: v27's ACTUAL (search) move at
its last-4 turns in the three normal-rules batch-10 legs
(batch-v33/v34/v35-vs-v27.json), testing old vs new gate.

A turn counts as LIVE only if choose_move would actually run the
search there (no immediate win for v27, no single opponent win to
block — those short-circuit before the check). Each live turn
evaluates both gates on v27's actual move. A disagreement
(old=False, new=True) is a position where v35's check would have
stayed closed and v45's opens — a discriminating instance for the
regression test and a direct measure of the hole's incidence on
real search moves.

Resumable (per-game JSON writes) and slow (~1s/quiet turn,
~10-30s/VCF-rich turn) — run in the background.

Usage:
    python research/iterations/scan_hole_broad.py \
        --b10dir ~/workspace/projects/gomoku-b10-verify/research/iterations \
        --out research/iterations/hole-scan-broad.json
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

from gomoku_ml import evaluate, game, scoring  # noqa: E402

FILES = ["batch-v33-vs-v27.json", "batch-v34-vs-v27.json",
         "batch-v35-vs-v27.json"]
LAST_TURNS = 4


def _xy(m: int) -> list[int]:
    return [m % 15, m // 15]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b10dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results: dict = {"games": [], "disagreements": []}
    done: set[tuple[str, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done = {(g["file"], g["game"]) for g in results["games"]}
        print(f"resuming: {len(done)} games already scanned",
              file=sys.stderr, flush=True)

    n_live = n_disagree = 0
    for fname in FILES:
        with open(os.path.join(args.b10dir, fname),
                  encoding="utf-8") as f:
            batch = json.load(f)
        v27_is_a = batch["engine_a"].endswith("v27")
        seen_moves: set[tuple[int, ...]] = set()
        for rec in batch["records"]:
            if rec["winner"] == "draw":
                continue
            v27_lost = (not rec["winner_is_a"]) if v27_is_a \
                else rec["winner_is_a"]
            if not v27_lost:
                continue
            key = tuple(rec["moves"])
            if key in seen_moves:
                continue
            seen_moves.add(key)
            if (fname, rec["game"]) in done:
                continue
            a_colour = (game.BLACK if rec["a_colour"] == "black"
                        else game.WHITE)
            v27_colour = a_colour if v27_is_a else -a_colour
            moves = rec["moves"]
            v27_turns = [
                ply for ply in range(len(moves))
                if (game.BLACK if ply % 2 == 0 else game.WHITE)
                == v27_colour
            ][-LAST_TURNS:]
            entry = {"file": fname, "game": rec["game"],
                     "turns": []}
            for ply in v27_turns:
                board = game.empty_board()
                for i, m in enumerate(moves[:ply]):
                    board = game.apply_move(
                        board, m,
                        game.BLACK if i % 2 == 0 else game.WHITE)
                actual = moves[ply]
                opp = -v27_colour
                # Live? choose_move short-circuits on our immediate
                # win and on a single opponent win.
                if game.is_winning_move(board, actual, v27_colour):
                    entry["turns"].append(
                        {"ply": ply, "live": False,
                         "reason": "immediate win"})
                    continue
                if len(scoring._winning_moves(board, opp)) == 1:
                    entry["turns"].append(
                        {"ply": ply, "live": False,
                         "reason": "forced block"})
                    continue
                n_live += 1
                opp_creators = evaluate.four_creating_moves(
                    board, opp)
                opp_vcf = [
                    m for m in opp_creators
                    if evaluate.starts_forced_win(board, opp, m,
                                                  depth=3)
                ]
                b2 = game.apply_move(board, actual, v27_colour)
                old = bool(opp_vcf) and any(
                    evaluate.starts_forced_win(b2, opp, m, depth=3)
                    for m in opp_vcf)
                new = bool(scoring._winning_moves(b2, opp)) or any(
                    evaluate.starts_forced_win(b2, opp, m, depth=3)
                    for m in evaluate.four_creating_moves(b2, opp))
                turn = {"ply": ply, "live": True,
                        "played": _xy(actual),
                        "opp_vcf_current": len(opp_vcf),
                        "old": old, "new": new}
                entry["turns"].append(turn)
                if old != new:
                    n_disagree += 1
                    after = list(scoring._winning_moves(b2, opp))
                    after += [
                        m for m in evaluate.four_creating_moves(
                            b2, opp)
                        if evaluate.starts_forced_win(b2, opp, m,
                                                      depth=3)]
                    print(f"DISAGREE {fname} g{rec['game']} ply "
                          f"{ply} played {_xy(actual)}: old={old} "
                          f"new={new}; opp_vcf_current={len(opp_vcf)}; "
                          f"VCF after: {[_xy(s) for s in after]}",
                          flush=True)
                    results["disagreements"].append({
                        "file": fname, "game": rec["game"],
                        "ply": ply, "played": _xy(actual),
                        "old": old, "new": new,
                        "opp_vcf_current": [_xy(s)
                                            for s in opp_vcf],
                        "vcf_after": [_xy(s) for s in after],
                        "moves_prefix": moves[:ply],
                    })
            results["games"].append(entry)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"{fname} g{rec['game']}: scanned "
                  f"({len(results['games'])} games, {n_live} live "
                  f"turns, {n_disagree} disagreements)", flush=True)
    print(f"DONE: {len(results['games'])} games, {n_live} live "
          f"turns, {n_disagree} disagreements", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

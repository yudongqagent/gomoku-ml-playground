#!/usr/bin/env python3
"""Scan for positions where v35's original-starts subset gate and
v45's full-re-check gate DISAGREE (old=False, new=True) — i.e. the
hole actually bites.

Sources (all pre-existing, no new games):
- mining turns from batch 11's setup-mining.json with small EXACT
  VCF counts after a test move (2-6): few VCF starts left means
  the originals may all be gone while a new one was created;
- the 8 v35 regression positions (played_before_patch + expected
  moves): positions where the check is load-bearing.

For each turn, rebuild the board from the batch-10 batch file,
compute the opponent's VCF starts on the current board ONCE, then
evaluate BOTH gates for every test move (+ the actual move).
Prints every disagreement with full context, and a summary.

Usage:
    python research/iterations/scan_hole_instances.py \
        --mining /tmp/setup-mining.json \
        --b10dir ~/workspace/projects/gomoku-b10-verify/research/iterations \
        --out research/iterations/hole-scan.json
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


def _xy(m: int) -> list[int]:
    return [m % 15, m // 15]


def _old_gate(board, player, move, opp_vcf) -> bool:
    """v35's subset test, with the current-board VCF starts
    precomputed (identical logic to _leaves_opp_vcf_subset)."""
    if not opp_vcf:
        return False
    b2 = game.apply_move(board, move, player)
    return any(
        evaluate.starts_forced_win(b2, -player, m, depth=3)
        for m in opp_vcf
    )


def _new_gate(board, player, move) -> bool:
    """v45's full re-check (identical logic to
    _leaves_opp_vcf_full)."""
    b2 = game.apply_move(board, move, player)
    if scoring._winning_moves(b2, -player):
        return True
    for m in evaluate.four_creating_moves(b2, -player):
        if evaluate.starts_forced_win(b2, -player, m, depth=3):
            return True
    return False


def _vcf_after(board, player, move) -> list[int]:
    """Exact VCF starts for the opponent after ``move`` (winning
    moves + four-creator starts)."""
    b2 = game.apply_move(board, move, player)
    out = list(scoring._winning_moves(b2, -player))
    for m in evaluate.four_creating_moves(b2, -player):
        if evaluate.starts_forced_win(b2, -player, m, depth=3):
            out.append(m)
    return out


def _scan_board(board, player, moves, label, results):
    opp = -player
    opp_creators = evaluate.four_creating_moves(board, opp)
    opp_vcf = [
        m for m in opp_creators
        if evaluate.starts_forced_win(board, opp, m, depth=3)
    ]
    opp_wins = scoring._winning_moves(board, opp)
    print(f"{label}: {len(opp_creators)} creators, {len(opp_vcf)} "
          f"VCF starts, {len(opp_wins)} winning moves on current "
          f"board; {len(moves)} moves to test", flush=True)
    for m in moves:
        o = _old_gate(board, player, m, opp_vcf)
        n = _new_gate(board, player, m)
        if o != n:
            after = _vcf_after(board, player, m)
            new_starts = [s for s in after if s not in opp_vcf
                          and s not in opp_wins]
            print(f"  DISAGREE {label} move {_xy(m)}: old={o} "
                  f"new={n}; VCF after: {[_xy(s) for s in after]}; "
                  f"NEW starts: {[_xy(s) for s in new_starts]}",
                  flush=True)
            results.append({
                "label": label, "move": _xy(m), "old": o,
                "new": n,
                "vcf_after": [_xy(s) for s in after],
                "new_starts": [_xy(s) for s in new_starts],
                "opp_vcf_current": [_xy(s) for s in opp_vcf],
                "opp_wins_current": [_xy(s) for s in opp_wins],
            })
        else:
            print(f"  agree {label} move {_xy(m)}: {o}",
                  flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mining", required=True)
    parser.add_argument("--b10dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.mining, encoding="utf-8") as f:
        mining = json.load(f)

    # Target turns: normal-rules mining turns with an exact small
    # VCF count (2-6) after some test move. g0 ply 29 is excluded
    # (already fully scanned: 0 disagreements / 98 candidates).
    targets: dict[tuple[str, int, int], list[int]] = {}
    for g in mining["games"]:
        if "swap" in g["file"]:
            continue
        for t in g["turns"]:
            tests = t.get("tests", [])
            if not any(te["vcf"] is not None
                       and 2 <= te["vcf"] <= 6 for te in tests):
                continue
            if (g["file"], g["game"], t["ply"]) == (
                    "batch-v33-vs-v27.json", 0, 29):
                continue
            key = (g["file"], g["game"], t["ply"])
            moves = [game.xy_to_move(*te["move"]) for te in tests]
            moves.append(game.xy_to_move(*t["played"]))
            targets[key] = sorted(set(moves))
    print(f"target mining turns: {len(targets)}", flush=True)

    results: list[dict] = []
    batch_cache: dict[str, dict] = {}
    for (fname, gno, ply), moves in sorted(targets.items()):
        if fname not in batch_cache:
            with open(os.path.join(args.b10dir, fname),
                      encoding="utf-8") as f:
                batch_cache[fname] = json.load(f)
        batch = batch_cache[fname]
        rec = next(r for r in batch["records"] if r["game"] == gno)
        a_colour = (game.BLACK if rec["a_colour"] == "black"
                    else game.WHITE)
        v27_is_a = batch["engine_a"].endswith("v27")
        v27_colour = a_colour if v27_is_a else -a_colour
        assert (game.BLACK if ply % 2 == 0 else game.WHITE) \
            == v27_colour, "ply parity / v27 colour mismatch"
        board = game.empty_board()
        for i, m in enumerate(rec["moves"][:ply]):
            board = game.apply_move(
                board, m, game.BLACK if i % 2 == 0 else game.WHITE)
        _scan_board(board, v27_colour, moves,
                    f"{fname} g{gno} ply {ply}", results)

    # v35 regression positions: played_before_patch + expected.
    here = os.path.dirname(os.path.abspath(__file__))
    pos_path = os.path.join(here, "..", "..", "ml", "gomoku_ml",
                            "regression_positions_v35.json")
    with open(pos_path, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    for pos in positions:
        board = game.empty_board()
        for x, y in pos["black"]:
            board[y, x] = game.BLACK
        for x, y in pos["white"]:
            board[y, x] = game.WHITE
        player = (game.BLACK if pos["to_move"] == "black"
                  else game.WHITE)
        moves = [game.xy_to_move(*pos["played_before_patch"])]
        moves += [game.xy_to_move(*e) for e in pos["expected"]]
        _scan_board(board, player, sorted(set(moves)),
                    f"v35-regression {pos['id']}", results)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"disagreements": results}, f, indent=2)
    print(f"DONE: {len(results)} disagreements", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

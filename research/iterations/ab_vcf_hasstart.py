#!/usr/bin/env python3
"""Validate _has_vcf_start against the exact VCF counts recorded in
setup-mining.json (batch 11 H1 mining safety check).

Every recorded turn with a non-null exact ``vcf_after`` (from the
slow _vcf_starts enumeration) and every recorded test with a
non-null exact ``vcf`` is recomputed with the early-exit
_has_vcf_start; the zero / non-zero verdict must agree on all of
them. Appends a "vcf" section to setup-mining-ab.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game  # noqa: E402
import mine_setup_signature as mine  # noqa: E402

B10 = "/home/hatch/workspace/projects/gomoku-b10/research/iterations"
HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    with open(os.path.join(HERE, "setup-mining.json"),
              encoding="utf-8") as f:
        mining = json.load(f)
    batches: dict[str, dict] = {}

    def _batch(fname: str) -> dict:
        if fname not in batches:
            with open(os.path.join(B10, fname), encoding="utf-8") as f:
                batches[fname] = json.load(f)
        return batches[fname]

    compared = mismatches = 0
    cases = []
    for g in mining["games"]:
        batch = _batch(g["file"])
        rec = next(r for r in batch["records"] if r["game"] == g["game"])
        v27_colour = (game.BLACK if g["v27_colour"] == "black"
                      else game.WHITE)
        opp = -v27_colour
        moves = rec["moves"]
        for turn in g["turns"]:
            ply = turn["ply"]
            board = game.empty_board()
            for i, m in enumerate(moves[:ply]):
                board = game.apply_move(
                    board, m, game.BLACK if i % 2 == 0 else game.WHITE)
            actual = moves[ply]
            if turn["vcf_after"] is not None and not turn.get(
                    "vcf_capped"):
                after_actual = game.apply_move(board, actual, v27_colour)
                fast = mine._has_vcf_start(after_actual, opp)
                slow_nonzero = turn["vcf_after"] >= 1
                compared += 1
                if fast != slow_nonzero:
                    mismatches += 1
                    cases.append([g["file"], g["game"], ply, "actual",
                                  turn["vcf_after"], fast])
            for test in turn["tests"]:
                if test["vcf"] is None:
                    continue
                m = test["move"][1] * 15 + test["move"][0]
                after_test = game.apply_move(board, m, v27_colour)
                fast = mine._has_vcf_start(after_test, opp)
                slow_nonzero = test["vcf"] >= 1
                compared += 1
                if fast != slow_nonzero:
                    mismatches += 1
                    cases.append([g["file"], g["game"], ply,
                                  f"test {test['move']}", test["vcf"],
                                  fast])
    res = {"compared": compared, "mismatches": mismatches,
           "cases": cases}
    print(json.dumps(res, indent=2))
    ab_path = os.path.join(HERE, "setup-mining-ab.json")
    with open(ab_path, encoding="utf-8") as f:
        ab = json.load(f)
    ab["vcf"] = res
    with open(ab_path, "w", encoding="utf-8") as f:
        json.dump(ab, f, indent=2)
        f.write("\n")
    print("VCF A/B " + ("PASS" if mismatches == 0 else "FAIL"),
          file=sys.stderr)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

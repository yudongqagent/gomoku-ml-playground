#!/usr/bin/env python3
"""Verify setup-signature mining candidates (batch 11 H1).

For each candidate instance flagged by mine_setup_signature.py, run
the workup protocol (iter-6 §2.2 corrected method, same as
iter-35's forensics):

1. The ACTUAL move's playout must lose at >= 2 of 3 seeds (match
   engines: v27 vs the leg's winner version, seeds 0/1/2) — else
   the turn is not a verified losing move and the instance does
   not count.
2. The REMOVING move's playout must draw/win at >= 2 of the same
   3 seeds — else it is not a verified save.
3. The EXACT audit (iter-35 §2.3 — playouts can produce false
   saves against a weak opponent): after the removing move the
   opponent must hold 0 VCF starts, 0 creators, 0 setups
   (recomputed here, not trusted from the mining filter).

A candidate counts as a verified setup-signature instance iff all
three hold. Distinctness (transposes / symmetries) is checked
afterwards over the verified set.

The match engines live in different trees: v33/v34/v35 exist only
in the batch-10 worktree, v7 in this worktree. Run with
--ml-root pointing at the tree that holds the winner version
(b10's evaluate.py differs from this tree's only by the v33
four-creator optimisation, verified A/B-identical, 0 mismatches
— iter-33 §2.3; game.py, workup_loss.py and targeted_playout.py
are byte-identical, so the verification is faithful in either
tree for the shared engines).

Usage:
    python research/iterations/verify_setup_candidate.py \\
        research/iterations/setup-mining.json \\
        --ml-root ~/workspace/projects/gomoku-b10 \\
        --winner 33 --winner 34 --winner 35 \\
        --out research/iterations/setup-verification.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

B10 = "/home/hatch/workspace/projects/gomoku-b10/research/iterations"
MIN_SEEDS = 2


def _xy(move: int) -> list[int]:
    return [move % 15, move // 15]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mining", help="setup-mining.json path")
    parser.add_argument("--ml-root", required=True,
                        help="repo root whose ml/ holds the engines")
    parser.add_argument("--winner", type=int, action="append",
                        default=[], help="winner versions to verify "
                        "(default: all candidates in the file)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    ml_root = os.path.expanduser(args.ml_root)
    sys.path.insert(0, os.path.join(ml_root, "ml", "scripts"))
    sys.path.insert(0, os.path.join(ml_root, "ml"))
    import common  # noqa: E402

    common.ensure_ml_on_path()
    import numpy as np  # noqa: E402
    from gomoku_ml import game, scoring  # noqa: E402
    from targeted_playout import _creators, _vcf_starts  # noqa: E402
    from workup_loss import _PlayerCache, play_out  # noqa: E402

    def _setups_limited(board: np.ndarray, player: int,
                        limit: int) -> list[int]:
        out = []
        for m in scoring._candidate_moves(board):
            b2 = game.apply_move(board, m, player)
            if scoring._winning_moves(b2, player):
                continue
            if len(_creators(b2, player)) >= 2:
                out.append(m)
                if limit > 0 and len(out) >= limit:
                    break
        return out

    with open(args.mining, encoding="utf-8") as f:
        mining = json.load(f)

    # Resume: load existing verifications, skip completed candidates.
    results: dict = {"candidates": []}
    done: set[tuple[str, int, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done = {(c["file"], c["game"], c["ply"])
                for c in results["candidates"]}
        print(f"resuming: {len(done)} candidates already verified",
              file=sys.stderr, flush=True)

    cache = _PlayerCache()
    batches: dict[str, dict] = {}

    def _batch(fname: str) -> dict:
        if fname not in batches:
            with open(os.path.join(B10, fname), encoding="utf-8") as f:
                batches[fname] = json.load(f)
        return batches[fname]

    n_verified = 0
    for g in mining["games"]:
        if args.winner and g["winner_version"] not in args.winner:
            continue
        for turn in g["turns"]:
            if not turn["removing_moves"]:
                continue
            key = (g["file"], g["game"], turn["ply"])
            if key in done:
                continue
            batch = _batch(g["file"])
            rec = next(r for r in batch["records"]
                       if r["game"] == g["game"])
            v27_colour = (game.BLACK if g["v27_colour"] == "black"
                          else game.WHITE)
            moves = rec["moves"]
            ply = turn["ply"]
            board = game.empty_board()
            for i, m in enumerate(moves[:ply]):
                board = game.apply_move(
                    board, m, game.BLACK if i % 2 == 0 else game.WHITE)
            actual = moves[ply]
            opp = -v27_colour
            after_actual = game.apply_move(board, actual, v27_colour)
            actual_out = [
                play_out(after_actual, opp, 27, g["winner_version"],
                         v27_colour, seed, cache)
                for seed in args.seeds
            ]
            actual_losses = sum(1 for o in actual_out if o < 0)
            qualifies = actual_losses >= MIN_SEEDS
            entry: dict = {
                "file": g["file"], "game": g["game"], "ply": ply,
                "winner_version": g["winner_version"],
                "v27_colour": g["v27_colour"],
                "played": turn["played"],
                "setups_after_actual": turn["setups_after"],
                "setup_squares": turn["setups"],
                "actual_playout": actual_out,
                "actual_losses": actual_losses,
                "qualifies": qualifies,
                "removing_tests": [],
                "verified": False,
            }
            print(f"== {g['file']} g{g['game']} ply {ply}: actual "
                  f"{turn['played']} playout {actual_out} "
                  f"({'QUALIFIES' if qualifies else 'no qualify'})",
                  file=sys.stderr, flush=True)
            if qualifies:
                for rm in turn["removing_moves"]:
                    m = rm["move"][1] * 15 + rm["move"][0]
                    after_rm = game.apply_move(board, m, v27_colour)
                    out = [
                        play_out(after_rm, opp, 27,
                                 g["winner_version"], v27_colour,
                                 seed, cache)
                        for seed in args.seeds
                    ]
                    non_losses = sum(1 for o in out if o >= 0)
                    saves = non_losses >= MIN_SEEDS
                    # Exact audit (recomputed, not trusted).
                    vcf_n = len(_vcf_starts(after_rm, opp))
                    creators_n = len(_creators(after_rm, opp))
                    setups_n = len(_setups_limited(after_rm, opp, 1))
                    audit_clean = (vcf_n == 0 and creators_n == 0
                                   and setups_n == 0)
                    entry["removing_tests"].append({
                        "move": rm["move"], "source": rm["source"],
                        "playout": out, "non_losses": non_losses,
                        "saves": saves,
                        "audit": {"vcf": vcf_n, "creators": creators_n,
                                  "setups": setups_n},
                        "audit_clean": audit_clean,
                        "verified_save": saves and audit_clean,
                    })
                    print(f"   removing {rm['move']} ({rm['source']}): "
                          f"playout {out} "
                          f"({'SAVES' if saves else 'no save'}), audit "
                          f"VCF {vcf_n} / creators {creators_n} / "
                          f"setups {setups_n} "
                          f"({'CLEAN' if audit_clean else 'DIRTY'})",
                          file=sys.stderr, flush=True)
                entry["verified"] = any(
                    t["verified_save"] for t in entry["removing_tests"])
            if entry["verified"]:
                n_verified += 1
            results["candidates"].append(entry)
            done.add(key)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
                f.write("\n")
    results["meta"] = {
        "candidates_tested": len(results["candidates"]),
        "verified_instances": sum(
            1 for c in results["candidates"] if c["verified"]),
        "seeds": args.seeds,
        "ml_root": ml_root,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"DONE: {len(results['candidates'])} candidates tested, "
          f"{results['meta']['verified_instances']} verified",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

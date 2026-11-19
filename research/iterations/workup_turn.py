#!/usr/bin/env python3
"""Work up one candidate turn: save-search + root dump (batch 11 H1).

Fallback forensics for the setup-signature mining: the mining's 9
candidates are verified LOSING turns (the actual move loses >= 2/3
in playouts) in setup-rich positions, but the removing move does
not save any of them. This script asks the next question for each
candidate turn: does ANY move save — and if so, what does the
leaf/search think of it?

For each candidate turn (ordered by ply ascending — early turns are
the most likely still-saveable):

1. Shortcut check: if the opponent holds exactly one winning move,
   v27's engine blocks it deterministically (no search ran) — the
   turn is not a leaf mis-ranking at all. Recorded explicitly.
2. Alternatives: the workup set (top-5 by 1-ply v1 score, the
   400-sim MCTS proposal, any VCF move) PLUS the mining's recorded
   setup-square / creator-after-setup test moves. Each alternative
   gets a 3-seed playout (match engines v27 vs the leg's winner).
   A move SAVES iff it draws/wins at >= 2 of 3 seeds.
3. Exact audit (slow functions) after each saving alternative and
   after the removing move: opponent VCF starts / creators /
   setups. This completes the setup verification for turns the
   verification script has not covered yet (winners 33/34).
4. Root dump (only when no shortcut fired): a fresh 100-sim v27
   search at the turn, dumping visits / our-q (-value_sum/visits,
   raw q is the opponent's perspective) / prior for the top-10
   children by visits, plus the actual move, each saving move and
   the removing move. The dump is the leaf-mis-ranking diagnosis:
   a saving move with a WORSE leaf value than the actual is a leaf
   mis-ranking; a saving move with a BETTER leaf value but fewer
   visits is a search/selection failure (the q-based root-decision
   lever is CLOSED — that flavour is not patchable); a saving
   move with zero visits is a prior/pruning failure.

The actual move's playout is reused from setup-verification.json
when present (the verification already ran it) and re-run
otherwise.

Usage:
    python research/iterations/workup_turn.py \
        research/iterations/setup-mining.json \
        --ml-root ~/workspace/projects/gomoku-b10 \
        --winner 33 --winner 34 --winner 35 \
        --verification research/iterations/setup-verification.json \
        --out research/iterations/turn-workups.json
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
                        default=None, help="winner versions to include")
    parser.add_argument("--verification", default=None,
                        help="setup-verification.json (reuse actual playouts)")
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
    from gomoku_ml.mcts_heuristic import _Node  # noqa: E402
    from targeted_playout import _creators, _setups, _vcf_starts  # noqa: E402
    from workup_loss import (  # noqa: E402
        _PlayerCache, alternatives_for, play_out)

    with open(args.mining, encoding="utf-8") as f:
        mining = json.load(f)

    # Known actual-playout results, reused to avoid re-running them.
    # (Also the removing-move playouts + audits, when the verification
    # already ran them.)
    known_actual: dict[tuple[str, int, int], list[int]] = {}
    known_removing: dict[tuple[str, int, int], dict] = {}
    if args.verification and os.path.exists(args.verification):
        with open(args.verification, encoding="utf-8") as f:
            ver = json.load(f)
        for c in ver["candidates"]:
            k = (c["file"], c["game"], c["ply"])
            known_actual[k] = c["actual_playout"]
            for rt in c.get("removing_tests", []):
                known_removing[(k, rt["move"][1] * 15
                                + rt["move"][0])] = rt

    # Resume: load existing workups, skip completed turns.
    results: dict = {"turns": []}
    done: set[tuple[str, int, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done = {(t["file"], t["game"], t["ply"])
                for t in results["turns"]}
        print(f"resuming: {len(done)} turns already worked up",
              file=sys.stderr, flush=True)

    # Candidate turns, earliest ply first.
    cands: list[tuple[dict, dict]] = []
    for g in mining["games"]:
        if args.winner and g["winner_version"] not in args.winner:
            continue
        for turn in g["turns"]:
            if turn["removing_moves"]:
                cands.append((g, turn))
    cands.sort(key=lambda gt: gt[1]["ply"])
    print(f"{len(cands)} candidate turns to work up",
          file=sys.stderr, flush=True)

    cache = _PlayerCache()
    batches: dict[str, dict] = {}

    def _batch(fname: str) -> dict:
        if fname not in batches:
            with open(os.path.join(B10, fname), encoding="utf-8") as f:
                batches[fname] = json.load(f)
        return batches[fname]

    for g, turn in cands:
        key = (g["file"], g["game"], turn["ply"])
        if key in done:
            continue
        batch = _batch(g["file"])
        rec = next(r for r in batch["records"] if r["game"] == g["game"])
        v27_colour = (game.BLACK if g["v27_colour"] == "black"
                      else game.WHITE)
        winner = g["winner_version"]
        moves = rec["moves"]
        ply = turn["ply"]
        board = game.empty_board()
        for i, m in enumerate(moves[:ply]):
            board = game.apply_move(
                board, m, game.BLACK if i % 2 == 0 else game.WHITE)
        actual = moves[ply]
        print(f"== {g['file']} g{g['game']} ply {ply} "
              f"(v27 vs v{winner}): actual {_xy(actual)}",
              flush=True)

        entry: dict = {
            "file": g["file"], "game": g["game"], "ply": ply,
            "winner_version": winner, "v27_colour": g["v27_colour"],
            "played": _xy(actual),
            "removing_moves": turn["removing_moves"],
            "setups_after_actual": turn["setups_after"],
        }

        # (1) Shortcut check: did the engine even search here?
        our_wins = scoring._winning_moves(board, v27_colour)
        opp_wins = scoring._winning_moves(board, -v27_colour)
        entry["our_winning_moves"] = [_xy(w) for w in our_wins]
        entry["opp_winning_moves"] = [_xy(w) for w in opp_wins]
        forced_block = len(opp_wins) == 1 and actual == opp_wins[0]
        entry["forced_block"] = forced_block
        if forced_block:
            print("   actual is a FORCED BLOCK (opponent's only win) — "
                  "no search ran here", flush=True)

        # (2) Actual playout (reused when known).
        after_actual = game.apply_move(board, actual, v27_colour)
        if key in known_actual:
            actual_outcomes = known_actual[key]
            print(f"   actual playout (from verification): "
                  f"{actual_outcomes}", flush=True)
        else:
            actual_outcomes = [
                play_out(after_actual, -v27_colour, 27, winner,
                         v27_colour, seed, cache)
                for seed in args.seeds]
            print(f"   actual playout: {actual_outcomes}", flush=True)
        entry["actual_playout"] = actual_outcomes
        entry["qualifies"] = sum(1 for o in actual_outcomes
                                 if o < 0) >= MIN_SEEDS
        if not entry["qualifies"]:
            print("   actual does NOT qualify as a losing move — "
                  "skipping alternatives", flush=True)
            results["turns"].append(entry)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            continue

        # (3) Alternatives: workup set + mining's setup test moves.
        alts = alternatives_for(board, v27_colour, actual, 27,
                                topk=5, deep_sims=400,
                                seed=args.seeds[0], cache=cache)
        seen = {a["move"] for a in alts} | {actual}
        for test in turn["tests"]:
            m = test["move"][1] * 15 + test["move"][0]
            if m not in seen:
                seen.add(m)
                alts.append({"move": m, "xy": test["move"],
                             "source": test["source"]})
        print(f"   {len(alts)} alternatives to playout-test",
              flush=True)
        entry["alternatives"] = []
        entry["saving_moves"] = []
        for alt in alts:
            rkey = (key, alt["move"])
            if rkey in known_removing:
                rt = known_removing[rkey]
                outcomes = rt["playout"]
                print(f"    alt {alt['xy']} ({alt['source']}): "
                      f"{outcomes} (from verification)",
                      flush=True)
            else:
                after_alt = game.apply_move(board, alt["move"],
                                            v27_colour)
                outcomes = [
                    play_out(after_alt, -v27_colour, 27, winner,
                             v27_colour, seed, cache)
                    for seed in args.seeds]
            non_losses = sum(1 for o in outcomes if o >= 0)
            alt_entry = {**alt, "playout": outcomes,
                         "non_losses": non_losses,
                         "saves": non_losses >= MIN_SEEDS}
            entry["alternatives"].append(alt_entry)
            if alt_entry["saves"]:
                entry["saving_moves"].append(alt_entry)
            if rkey not in known_removing:
                print(f"    alt {alt['xy']} ({alt['source']}): "
                      f"{outcomes} ({non_losses}/{len(args.seeds)} "
                      f"non-losses)"
                      + (" — SAVES" if alt_entry["saves"] else ""),
                      flush=True)

        # (4) Exact audit after each save and after the removing move.
        audit_moves: dict[int, str] = {}
        for s in entry["saving_moves"]:
            audit_moves[s["move"]] = "save"
        for r in turn["removing_moves"]:
            m = r["move"][1] * 15 + r["move"][0]
            audit_moves.setdefault(m, "removing")
        entry["audits"] = {}
        for m, kind in audit_moves.items():
            b2 = game.apply_move(board, m, v27_colour)
            audit = {
                "kind": kind,
                "vcf": len(_vcf_starts(b2, -v27_colour)),
                "creators": len(_creators(b2, -v27_colour)),
                "setups": len(_setups(b2, -v27_colour)),
            }
            entry["audits"][f"{m % 15},{m // 15}"] = audit
            print(f"    audit after {_xy(m)} ({kind}): VCF "
                  f"{audit['vcf']} / creators {audit['creators']} / "
                  f"setups {audit['setups']}", flush=True)

        # (5) Root dump (only when the engine actually searched).
        if not forced_block and not our_wins:
            player = cache.get(27, args.seeds[0])
            mcts = player._mcts
            root = _Node(1.0)
            for _ in range(mcts.sims):
                mcts._run_one(root, board, v27_colour, None)
            children = []
            for m, node in (root.children or {}).items():
                q_ours = (-node.value_sum / node.visit_count
                          if node.visit_count else None)
                children.append({
                    "move": _xy(m), "visits": node.visit_count,
                    "q_ours": round(q_ours, 4)
                    if q_ours is not None else None,
                    "prior": round(node.prior, 4),
                })
            children.sort(key=lambda c: (-c["visits"], -c["prior"]))
            entry["root_top10"] = children[:10]
            wanted = {actual}
            wanted |= {s["move"] for s in entry["saving_moves"]}
            wanted |= {r["move"][1] * 15 + r["move"][0]
                       for r in turn["removing_moves"]}
            entry["root_wanted"] = [c for c in children
                                    if c["move"][1] * 15
                                    + c["move"][0] in wanted]
            chosen = mcts._choose_root_child(root, board, v27_colour)
            entry["root_choice"] = _xy(chosen)
            entry["root_reproduces_actual"] = chosen == actual
            print(f"    root: top child {children[0]['move']} "
                  f"({children[0]['visits']} visits), choice "
                  f"{_xy(chosen)} "
                  f"({'reproduces' if chosen == actual else 'DIFFERS FROM'} "
                  f"actual)", flush=True)
            for c in entry["root_wanted"]:
                print(f"      wanted {c['move']}: {c['visits']} visits, "
                      f"q_ours {c['q_ours']}, prior {c['prior']}",
                      flush=True)
        else:
            entry["root_top10"] = None
            entry["root_choice"] = None
            print("    root dump skipped (shortcut fired)", flush=True)

        results["turns"].append(entry)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"   -> {len(entry['saving_moves'])} saving move(s)",
              flush=True)

    n_saving = sum(1 for t in results["turns"] if t["saving_moves"])
    print(f"DONE: {len(results['turns'])} turns worked up, "
          f"{n_saving} with a saving move", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

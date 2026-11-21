"""Replay a batch leg with v45's VCF-check instrumentation on.

Re-runs ONE engine (v45) against the RECORDED opponent moves of a
completed batch JSON, with ``vcf_fire_log`` and
``vcf_audit_old_gate`` set on v45's MCTS, and verifies the replayed
moves match the recorded moves exactly. The opponent's moves are
replayed, not re-chosen, so a leg costs one engine, not two.

Why this reproduces v45's moves exactly: v45's MCTS consumes RNG
only inside ``choose_move`` (root tie-breaks), and the driver
reseeds per game (base_seed + game index). Replaying v45's
``choose_move`` calls in ply order on the same boards consumes
the RNG identically. The verification (replayed == recorded at
every v45 turn) proves it per game — any mismatch is a hard
failure, not a warning.

Works for normal-rules and swap-rules legs (balanced placement
does not call the engine — the placement RNG is separate — so
v45's first engine call is its first real turn in both cases).

Output: per-game entries + an aggregate summary:
  - v45_turns: number of v45 choose_move calls (live turns; the
    opening placement moves in swap legs are not turns).
  - gate_fires: turns where the FIXED gate returned True (the
    search move leaves the opponent a VCF start).
  - overrides: gate fires where the defensive selector found a
    prevention (the played move differs from the search move).
  - no_prevention: gate fires where no prevention exists.
  - hole_instances: turns where the OLD gate said prevention
    (False) and the FIXED gate said non-prevention (True),
    split into B1 (opponent held NO VCF start on the current
    board — the old gate's early return) and B2 (opponent held
    VCF starts, but our move removes them all AND creates a
    new one). The B1/B2 split is computed only at hole turns
    (a current-board VCF enumeration per hole turn — cheap
    because hole turns are rare).
  - hole_changed_move: hole instances where the override
    happened (the fix changed the played move).
  - fire_rate / override_rate / hole_rate: the above as
    fractions of v45_turns.
  - replay_mismatches: games where a replayed move differed
    from the recorded move (must be 0).

Resume: per-game JSON writes; re-running skips completed games.

Usage:
  python research/iterations/replay_hole_incidence.py \
      research/iterations/batch-v45-vs-v27.json \
      --engine 45 --side A \
      --out research/iterations/replay-v45-vs-v27.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ml"))

import numpy as np

from gomoku_ml import evaluate, game, scoring


def _vcf_starts_on_board(board, player):
    """All VCF starts the player holds on this board (exact count)."""
    starts = set(scoring._winning_moves(board, player))
    for creator in evaluate.four_creating_moves(board, player):
        if evaluate.starts_forced_win(board, player, creator, depth=3):
            starts.add(creator)
    return starts


def _replay_game(batch, rec, player, seed, v45_is_a, v45_colour):
    """Replay one game; returns (entry, ok)."""
    i = rec["game"]
    player.reseed(seed + i)
    mcts = player._mcts
    mcts.vcf_fire_log = []
    mcts.vcf_audit_old_gate = True
    board = game.empty_board()
    moves = rec["moves"]
    swap = bool(batch.get("swap_rule"))
    # In swap legs the first 3 moves are placement (not engine
    # turns); in normal legs every move is an engine turn.
    first_turn = 3 if swap else 0
    mismatches = []
    hole_turns = []
    for ply, recorded in enumerate(moves):
        to_move = game.BLACK if ply % 2 == 0 else game.WHITE
        if ply >= first_turn and to_move == v45_colour:
            log_before = len(mcts.vcf_fire_log)
            chosen = player.choose_move(board, to_move)
            if int(chosen) != int(recorded):
                mismatches.append(
                    {"ply": ply, "recorded": int(recorded),
                     "replayed": int(chosen)}
                )
            # Classify the fire-log entries this turn added
            # (search-first logs exactly one per live turn).
            for entry in mcts.vcf_fire_log[log_before:]:
                search_move, leaves, old, def_move = entry
                if old is False and leaves is True:
                    # Hole instance: classify B1/B2 by the
                    # opponent's VCF starts on the CURRENT board.
                    opp = -to_move
                    current = _vcf_starts_on_board(board, opp)
                    case = "B1" if not current else "B2"
                    hole_turns.append({
                        "ply": ply,
                        "case": case,
                        "current_vcf_starts": sorted(int(s) for s in current),
                        "search_move": int(search_move),
                        "defensive_move": (int(def_move)
                                           if def_move is not None else None),
                        "override": def_move is not None,
                    })
        board = game.apply_move(board, recorded, to_move)
    log = mcts.vcf_fire_log
    mcts.vcf_fire_log = None
    mcts.vcf_audit_old_gate = False
    turns = len(log)
    fires = sum(1 for e in log if e[1])
    overrides = sum(1 for e in log if e[1] and e[3] is not None)
    no_prev = sum(1 for e in log if e[1] and e[3] is None)
    holes = [h for h in hole_turns]
    entry = {
        "game": i,
        "v45_colour": "black" if v45_colour == game.BLACK else "white",
        "v45_turns": turns,
        "gate_fires": fires,
        "overrides": overrides,
        "no_prevention": no_prev,
        "hole_instances": len(holes),
        "hole_b1": sum(1 for h in holes if h["case"] == "B1"),
        "hole_b2": sum(1 for h in holes if h["case"] == "B2"),
        "hole_changed_move": sum(1 for h in holes if h["override"]),
        "hole_turns": holes,
        "replay_mismatches": mismatches,
    }
    return entry, not mismatches


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("batch", help="completed batch JSON to replay")
    ap.add_argument("--engine", type=int, default=45,
                    help="instrumented engine version (default 45)")
    ap.add_argument("--side", choices=["A", "B"], default="A",
                    help="which side of the batch is the engine")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    batch = json.load(open(args.batch))
    if not batch.get("complete"):
        raise SystemExit(f"{args.batch} is not complete "
                         f"({batch.get('games_played')} games)")
    engine_key = "engine_a" if args.side == "A" else "engine_b"
    if batch[engine_key] not in (f"v{args.engine}",
                                 f"scoring-v{args.engine}"):
        raise SystemExit(f"{engine_key} is {batch[engine_key]}, "
                         f"not v{args.engine}")
    seed = batch["seed_a"] if args.side == "A" else batch["seed_b"]
    v45_is_a = args.side == "A"

    out = {"batch": args.batch, "engine": args.engine, "side": args.side,
           "games": []}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    done = {g["game"] for g in out["games"]}

    player = scoring.get_player(args.engine, seed=seed)
    total = len(batch["records"])
    for rec in batch["records"]:
        i = rec["game"]
        if i in done:
            continue
        a_colour = (game.BLACK if rec["a_colour"] == "black"
                    else game.WHITE)
        v45_colour = a_colour if v45_is_a else -a_colour
        entry, ok = _replay_game(batch, rec, player, seed, v45_is_a,
                                 v45_colour)
        out["games"].append(entry)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        status = "OK" if ok else "MISMATCH"
        print(f"g{i}: turns={entry['v45_turns']} fires={entry['gate_fires']} "
              f"ovr={entry['overrides']} holes={entry['hole_instances']} "
              f"({entry['hole_b1']} B1 / {entry['hole_b2']} B2) {status}",
              flush=True)
        if not ok:
            raise SystemExit(f"replay mismatch in game {i}: "
                             f"{entry['replay_mismatches']}")

    gs = out["games"]
    turns = sum(g["v45_turns"] for g in gs)
    fires = sum(g["gate_fires"] for g in gs)
    overrides = sum(g["overrides"] for g in gs)
    no_prev = sum(g["no_prevention"] for g in gs)
    holes = sum(g["hole_instances"] for g in gs)
    b1 = sum(g["hole_b1"] for g in gs)
    b2 = sum(g["hole_b2"] for g in gs)
    changed = sum(g["hole_changed_move"] for g in gs)
    out["summary"] = {
        "games": len(gs),
        "v45_turns": turns,
        "gate_fires": fires,
        "overrides": overrides,
        "no_prevention": no_prev,
        "hole_instances": holes,
        "hole_b1": b1,
        "hole_b2": b2,
        "hole_changed_move": changed,
        "fire_rate": fires / turns if turns else 0.0,
        "override_rate": overrides / turns if turns else 0.0,
        "hole_rate": holes / turns if turns else 0.0,
        "replay_mismatches": 0,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("SUMMARY:", json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()

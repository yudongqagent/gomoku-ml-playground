#!/usr/bin/env python3
"""Replay a fast-loop batch with the instrumented v39, logging setup-check fires.

Replicates selfplay_batch.py's game loop exactly (per-game seeding,
alternating colours) for a completed batch JSON, and records the
setup check's TRUE fire list per game (from the engine's
setup_fire_log, cleared before each game). Verifies the replay
reproduces the recorded moves exactly — a mismatch means the
instrumentation changed play (a bug).

Usage:
    python research/iterations/replay_with_fires.py \
        research/iterations/fast-v39-vs-v0.json \
        --out research/iterations/fires-v39-vs-v0.json
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

from selfplay_batch import (  # noqa: E402
    play_recorded_game, reseed_player, resolve_engine)
from gomoku_ml import game  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", help="completed batch JSON to replay")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.batch, encoding="utf-8") as f:
        batch = json.load(f)
    assert not batch.get("swap_rule"), "normal-rules batches only"

    def _engine_arg(name: str) -> str:
        # Batch JSONs store the player name ("scoring-v39", "v0");
        # resolve_engine wants "v39" / "v0".
        if name.startswith("scoring-v"):
            return "v" + name[len("scoring-v"):]
        return name

    player_a, choose_a, name_a = resolve_engine(
        _engine_arg(batch["engine_a"]), batch["seed_a"])
    player_b, choose_b, name_b = resolve_engine(
        _engine_arg(batch["engine_b"]), batch["seed_b"])
    mcts_a = getattr(player_a, "_mcts", None)
    assert mcts_a is not None and hasattr(mcts_a, "setup_fire_log"), (
        "engine_a must be a setup_root MCTS engine")

    results: dict = {"games": []}
    done: set[int] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done = {g["game"] for g in results["games"]}
        print(f"resuming: {len(done)} games already replayed",
              file=sys.stderr, flush=True)

    for rec in batch["records"]:
        i = rec["game"]
        if i in done:
            continue
        reseed_player(player_a, batch["seed_a"] + i)
        reseed_player(player_b, batch["seed_b"] + i)
        mcts_a.setup_fire_log.clear()
        a_is_black = (i % 2 == 0)
        if a_is_black:
            winner, moves = play_recorded_game(
                choose_a, choose_b, game.empty_board(), game.BLACK)
        else:
            winner, moves = play_recorded_game(
                choose_b, choose_a, game.empty_board(), game.BLACK)
        replayed = [int(m) for m in moves]
        match = replayed == rec["moves"]
        fires = [{
            "search_move": [s % 15, s // 15],
            "removal": [r % 15, r // 15],
            "n_setups": n,
            "removal_visits": v,
            "removal_q_ours": (round(q, 4) if q is not None else None),
        } for (s, r, n, v, q) in mcts_a.setup_fire_log]
        print(f"game {i}: replay {'MATCHES' if match else 'MISMATCH!!'} "
              f"({len(moves)} plies, {len(fires)} check fires)",
              flush=True)
        if not match:
            print(f"  recorded: {rec['moves'][:20]}...",
                  file=sys.stderr, flush=True)
            print(f"  replayed: {replayed[:20]}...",
                  file=sys.stderr, flush=True)
        results["games"].append({
            "game": i, "result": rec["winner"],
            "winner_is_a": rec["winner_is_a"],
            "a_colour": rec["a_colour"], "plies": len(moves),
            "replay_matches": match, "fires": fires,
        })
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    n_fires = sum(len(g["fires"]) for g in results["games"])
    print(f"DONE: {len(results['games'])} games replayed, "
          f"{n_fires} total check fires", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

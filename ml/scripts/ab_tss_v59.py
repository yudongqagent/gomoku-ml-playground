#!/usr/bin/env python3
"""Large-scale move-choice-identity A/B for v59 (iter-59 §1.3).

Compares the fast TSS path against v52's kernel path on:

* the 200 bench positions (``bench_tss.py`` generators, seed 777:
  100 quiet + 100 creator-rich);
* 2000 mined real-game positions (v27-vs-v27 self-play turns,
  seeds 777/1777 — Check B's dev seeds, distinct from the
  battery's 100/200).

For every position: same move (or both None) AND same node count
AND same budget-exceeded flag — 0 mismatches required. Node-count
identity proves the search explored the same nodes in the same
order. Results are written as JSON (``--out``), checkpointed as the
run proceeds (bench sections per section, mined every 200
positions, replay per game); ``--resume`` continues from the
``--out`` JSON after an interruption (position generation is
deterministic, so chunk boundaries are stable).

``--replay N`` additionally replays N recorded v27-vs-v27 games
and checks v59 vs v52 ``choose_move`` on every turn (same seed).

Usage:
    PYTHONPATH=ml python ml/scripts/ab_tss_v59.py
        [--mined 2000] [--out research/iterations/ab-v59.json]
        [--replay 20]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from bench_tss import (  # noqa: E402
    generate_creator_rich_positions,
    generate_quiet_positions,
)

CHUNK = 200  # checkpoint the mined section every CHUNK positions


def _save(results: dict, out: str | None) -> None:
    if not out:
        return
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, out)


def _load(out: str | None) -> dict:
    if out and os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _ab_positions(positions, label, node_budget, start=0, state=None):
    """A/B ``positions[start:]``; ``state`` (when given) is the
    running result dict to extend (resume). Returns the result
    dict. ``state['done']`` counts completed positions."""
    from gomoku_ml.mcts_heuristic import tss_winning_move

    if state is None:
        state = {}
    state.setdefault("label", label)
    state.setdefault("n", len(positions))
    state.setdefault("fired", 0)
    state.setdefault("budget_exceeded", 0)
    state.setdefault("mismatches", [])
    state.setdefault("done", 0)
    for i in range(start, len(positions)):
        board, player = positions[i]
        s_slow: dict = {}
        s_fast: dict = {}
        slow = tss_winning_move(board, player, node_budget=node_budget,
                                stats=s_slow, fast=False)
        fast = tss_winning_move(board, player, node_budget=node_budget,
                                stats=s_fast, fast=True)
        if slow is not None:
            state["fired"] += 1
        if s_slow["budget_exceeded"]:
            state["budget_exceeded"] += 1
        if (
            fast != slow
            or s_fast["nodes"] != s_slow["nodes"]
            or s_fast["budget_exceeded"] != s_slow["budget_exceeded"]
        ):
            state["mismatches"].append(
                {
                    "index": i,
                    "slow_move": slow,
                    "fast_move": fast,
                    "slow_nodes": s_slow["nodes"],
                    "fast_nodes": s_fast["nodes"],
                    "slow_exceeded": s_slow["budget_exceeded"],
                    "fast_exceeded": s_fast["budget_exceeded"],
                }
            )
        state["done"] = i + 1
        if (i + 1) % CHUNK == 0:
            print(f"  {label}: {i + 1}/{len(positions)} "
                  f"({len(state['mismatches'])} mismatches)", flush=True)
            if state.get("_save"):
                state["_save"]()
    print(f"{label}: n={len(positions)} fired={state['fired']} "
          f"budget_exceeded={state['budget_exceeded']} "
          f"mismatches={len(state['mismatches'])}")
    state.pop("_save", None)
    return state


def _mine_positions(n, seed_a=777, seed_b=1777):
    from gomoku_ml import game, scoring

    positions = []
    g = 0
    while len(positions) < n:
        pa = scoring.get_player(27, seed=seed_a + g)
        pb = scoring.get_player(27, seed=seed_b + g)
        board = game.empty_board()
        player = game.BLACK
        while len(positions) < n:
            over, _ = game.game_over(board)
            if over:
                break
            positions.append((board.copy(), player))
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        g += 1
    return positions


def _replay(n_games, start=0, state=None):
    from gomoku_ml import game, scoring

    if state is None:
        state = {}
    state.setdefault("games", n_games)
    state.setdefault("turns", 0)
    state.setdefault("mismatches", [])
    state.setdefault("done", 0)
    for g in range(start, n_games):
        pa = scoring.get_player(27, seed=777 + g)
        pb = scoring.get_player(27, seed=1777 + g)
        p52 = scoring.get_player(52, seed=100)
        p59 = scoring.get_player(59, seed=100)
        board = game.empty_board()
        player = game.BLACK
        while True:
            over, _ = game.game_over(board)
            if over:
                break
            m52 = p52.choose_move(board, player)
            m59 = p59.choose_move(board, player)
            state["turns"] += 1
            if m59 != m52:
                state["mismatches"].append(
                    {"game": g, "turn": state["turns"],
                     "v52": m52, "v59": m59}
                )
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
        state["done"] = g + 1
        print(f"  replay: game {g + 1}/{n_games} done "
              f"({len(state['mismatches'])} mismatches)", flush=True)
        if state.get("_save"):
            state["_save"]()
    print(f"replay: games={n_games} turns={state['turns']} "
          f"mismatches={len(state['mismatches'])}")
    state.pop("_save", None)
    return state


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", type=int, default=2000)
    ap.add_argument("--node-budget", type=int, default=3000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--replay", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the --out JSON (skips completed "
                    "sections/chunks; position generation is "
                    "deterministic, so chunk boundaries are stable)")
    args = ap.parse_args()

    t0 = time.time()
    results: dict = {"node_budget": args.node_budget}
    if args.resume:
        results = _load(args.out) or results
        results.setdefault("node_budget", args.node_budget)

    def save():
        results["elapsed_s"] = time.time() - t0
        _save(results, args.out)

    # Bench sections (100 positions each — checkpointed per section).
    for key, gen, label in (
        ("bench_quiet", generate_quiet_positions, "bench quiet"),
        ("bench_rich", generate_creator_rich_positions,
         "bench creator-rich"),
    ):
        prev = results.get(key) or {}
        if prev.get("done", 0) >= 100:
            print(f"{label}: already done "
                  f"({len(prev.get('mismatches', []))} mismatches)")
            continue
        positions = gen(777, 100)
        print(f"generated {len(positions)} {label} positions",
              file=sys.stderr)
        state = dict(prev)
        state["_save"] = save
        results[key] = state  # assign BEFORE running: checkpoints
        # save the live (mutated) state, not a stale copy
        _ab_positions(
            positions, label, args.node_budget,
            start=prev.get("done", 0), state=state,
        )
        save()

    # Mined section (checkpointed every CHUNK positions).
    prev = results.get("mined") or {}
    if prev.get("done", 0) >= args.mined:
        print(f"mined: already done "
              f"({len(prev.get('mismatches', []))} mismatches)")
    else:
        mined = _mine_positions(args.mined)
        print(f"mined {len(mined)} real-game positions", file=sys.stderr)
        state = dict(prev)
        state["_save"] = save
        results["mined"] = state  # assign BEFORE running (see above)
        _ab_positions(
            mined, "mined", args.node_budget,
            start=prev.get("done", 0), state=state,
        )
        save()

    # Replay section (checkpointed per game).
    if args.replay:
        prev = results.get("replay") or {}
        if prev.get("done", 0) >= args.replay:
            print(f"replay: already done "
                  f"({len(prev.get('mismatches', []))} mismatches)")
        else:
            state = dict(prev)
            state["_save"] = save
            results["replay"] = state  # assign BEFORE running
            _replay(
                args.replay, start=prev.get("done", 0), state=state
            )
            save()

    total_mm = sum(
        len(results[k]["mismatches"])
        for k in ("bench_quiet", "bench_rich", "mined")
        if k in results
    )
    if args.replay and "replay" in results:
        total_mm += len(results["replay"]["mismatches"])
    results["total_mismatches"] = total_mm
    results["elapsed_s"] = time.time() - t0
    print(f"TOTAL mismatches: {total_mm} "
          f"(elapsed {results['elapsed_s']:.0f}s)")
    _save(results, args.out)
    return 0 if total_mm == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

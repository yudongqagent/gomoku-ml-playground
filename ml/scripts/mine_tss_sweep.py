#!/usr/bin/env python3
"""Mechanism check M for v57 (iter-57 §1.5): differential mining of
TSS depth/budget cells against the (4, 3000) v52 control.

Procedure (pre-registered in iter-57 §1.5 BEFORE this script ran):

* Play 50 v27-vs-v27 games (normal rules, alternating colours,
  seeds 777/1777 — the SAME setup as iter-52's Check B, so the
  positions are directly comparable; v27-vs-v27 games are the probe
  set because they are TSS-naive — richer in missed forcing wins
  than v52-vs-v52 games, where v52 already takes the shallow wins).
* At every turn, run TSS at the control (4, 3000) AND at each of
  the other 5 sweep cells (TSS consumes no RNG, so the probes do
  not perturb the game) and record, per cell:
  - gain: the cell fires where the control returns None;
  - loss: the control fires where the cell returns None — only
    possible via budget abort (in the unbounded case depth-5 is a
    superset of depth-4); every loss must coincide with
    budget_exceeded=True, and losses WITHOUT an abort are counted
    separately as an invariant violation to investigate;
  - abort: the cell's budget_exceeded flag.
* For each cell, verify the first 5 gains with 3 playouts each:
  play the cell's move, then the cell-engine attacking vs v52
  defending (seeds 0/1/2). A gain is VERIFIED if the attacker wins
  3/3. (Each gain is already a TSS-proven forcing win; the playout
  checks the end-to-end follow-through, as in iter-52's Check B.)

Pass criterion (pre-registered): >= 1 verified gain for at least
one cell. If 0 gains occur in 50 games for every cell, the check
is INCONCLUSIVE on mechanism (not failed) and the selection rule's
fallback (iter-57 §1.6) applies.

The mining phase is deterministic (per-game seeds, TSS consumes no
RNG), so it checkpoints after every game (``--checkpoint``, default
``<out>.checkpoint.json``) and resumes from the last completed game
on restart — the resumed run reproduces the uninterrupted run
exactly.

Usage:
    PYTHONPATH=ml python ml/scripts/mine_tss_sweep.py
        --out research/iterations/tss-sweep-mechanism.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

CONTROL = (4, 3000)  # v52
CELLS = [(4, 6000), (5, 3000), (5, 6000), (6, 3000), (6, 6000)]

MAX_VERIFY = 5  # gains verified per cell, to bound runtime
PLAYOUT_SEEDS = (0, 1, 2)


def _xy(move):
    return [move % 15, move // 15] if move is not None else None


def _board_to_lists(board):
    black = [[int(x), int(y)] for y, x in zip(*((board == 1).nonzero()))]
    white = [[int(x), int(y)] for y, x in zip(*((board == -1).nonzero()))]
    return black, white


def _cell_key(depth: int, budget: int) -> str:
    return f"d{depth}-b{budget}"


def _cell_engine(depth: int, budget: int, seed: int):
    """A v52 engine with the TSS depth/budget overridden to the cell
    (the v57 plumbing makes this a plain attribute set on the
    constructed MCTS — the engine is otherwise exactly v52)."""
    from gomoku_ml import scoring

    player = scoring.get_player(52, seed=seed)
    player._mcts.tss_depth = depth
    player._mcts.tss_node_budget = budget
    return player


def _playout(board, attacker, cell_move, depth, budget, seed):
    """Play the cell's move, then the cell-engine (attacker) vs v52
    (defender). Returns the winner (1/-1/0 for draw)."""
    from gomoku_ml import game, scoring

    b = game.apply_move(board, cell_move, attacker)
    over, winner = game.game_over(b)
    if over:
        return winner
    pa = _cell_engine(depth, budget, seed)
    pd = scoring.get_player(52, seed=seed + 1000)
    player = -attacker
    while True:
        over, winner = game.game_over(b)
        if over:
            return winner
        chooser = pa if player == attacker else pd
        move = chooser.choose_move(b, player)
        if move < 0:
            return 0
        b = game.apply_move(b, move, player)
        player = -player


def _new_cell_state() -> dict:
    return {
        "gains": [],
        "losses": 0,
        "losses_without_abort": 0,
        "aborts": 0,
        "fires": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed-a", type=int, default=777)
    ap.add_argument("--seed-b", type=int, default=1777)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic, scoring

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    ckpt_path = args.checkpoint or (out_path + ".checkpoint.json")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    # Resume from the checkpoint if it matches this run's parameters.
    turns = 0
    control_fires = 0
    control_confirms = 0
    control_differs = 0
    cells = {k: _new_cell_state() for k in
             (_cell_key(d, b) for d, b in CELLS)}
    start_gi = 0
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if (
                ckpt.get("games") == args.games
                and ckpt.get("seed_a") == args.seed_a
                and ckpt.get("seed_b") == args.seed_b
            ):
                turns = ckpt["turns"]
                control_fires = ckpt["control_fires"]
                control_confirms = ckpt["control_confirms"]
                control_differs = ckpt["control_differs"]
                cells = ckpt["cells"]
                start_gi = ckpt["games_done"]
                print(f"resuming from checkpoint: {start_gi}/"
                      f"{args.games} games done", file=sys.stderr)
        except (OSError, ValueError, KeyError):
            print("checkpoint unreadable — starting from scratch",
                  file=sys.stderr)

    for gi in range(start_gi, args.games):
        pa = scoring.get_player(27, seed=args.seed_a + gi)
        pb = scoring.get_player(27, seed=args.seed_b + gi)
        board = game.empty_board()
        player = game.BLACK
        # Alternate colours: even games A=Black, odd games A=White.
        black_engine = pa if gi % 2 == 0 else pb
        white_engine = pb if gi % 2 == 0 else pa
        while True:
            over, _winner = game.game_over(board)
            if over:
                break
            chooser = black_engine if player == game.BLACK else white_engine
            v27_move = chooser.choose_move(board, player)
            c_stats: dict = {}
            control_move = mcts_heuristic.tss_winning_move(
                board, player, depth=CONTROL[0], node_budget=CONTROL[1],
                stats=c_stats,
            )
            turns += 1
            if control_move is not None:
                control_fires += 1
                if control_move == v27_move:
                    control_confirms += 1
                else:
                    control_differs += 1
            for depth, budget in CELLS:
                key = _cell_key(depth, budget)
                stats: dict = {}
                move = mcts_heuristic.tss_winning_move(
                    board, player, depth=depth, node_budget=budget,
                    stats=stats,
                )
                st = cells[key]
                if stats.get("budget_exceeded"):
                    st["aborts"] += 1
                if move is not None:
                    st["fires"] += 1
                if move is not None and control_move is None:
                    black, white = _board_to_lists(board)
                    st["gains"].append({
                        "game": gi,
                        "to_move": "black" if player == game.BLACK
                        else "white",
                        "black": black,
                        "white": white,
                        "cell_move": _xy(move),
                        "control_move": None,
                        "v27_move": _xy(v27_move),
                        "nodes": stats["nodes"],
                    })
                elif move is None and control_move is not None:
                    st["losses"] += 1
                    if not stats.get("budget_exceeded"):
                        # Invariant violation: depth-5 is a superset
                        # of depth-4 in the unbounded case, so a loss
                        # without an abort must not happen. Counted,
                        # not asserted — the mining must not crash.
                        st["losses_without_abort"] += 1
            if v27_move < 0:
                break
            board = game.apply_move(board, v27_move, player)
            player = -player
        print(
            f"game {gi + 1}/{args.games}: turns={turns} "
            f"control_fires={control_fires} "
            + " ".join(
                f"{k}:g{len(v['gains'])}/l{v['losses']}/a{v['aborts']}"
                for k, v in cells.items()
            ),
            file=sys.stderr,
        )
        # Checkpoint after every game (the mining is deterministic,
        # so a resume reproduces the uninterrupted run exactly).
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "games": args.games,
                "seed_a": args.seed_a,
                "seed_b": args.seed_b,
                "games_done": gi + 1,
                "turns": turns,
                "control_fires": control_fires,
                "control_confirms": control_confirms,
                "control_differs": control_differs,
                "cells": cells,
            }, f)

    # Verify the first MAX_VERIFY gains per cell with playouts.
    summary_cells = {}
    for depth, budget in CELLS:
        key = _cell_key(depth, budget)
        st = cells[key]
        verified = []
        for g in st["gains"][:MAX_VERIFY]:
            board = game.empty_board()
            for x, y in g["black"]:
                board[y, x] = game.BLACK
            for x, y in g["white"]:
                board[y, x] = game.WHITE
            attacker = game.BLACK if g["to_move"] == "black" else game.WHITE
            cell_move = g["cell_move"][1] * 15 + g["cell_move"][0]
            wins = 0
            outcomes = []
            for seed in PLAYOUT_SEEDS:
                winner = _playout(board, attacker, cell_move,
                                  depth, budget, seed)
                outcomes.append(winner)
                if winner == attacker:
                    wins += 1
            g["playouts"] = outcomes
            g["verified"] = wins == len(PLAYOUT_SEEDS)
            verified.append(g)
            print(f"{key} gain game {g['game']} ({g['to_move']}, cell "
                  f"{g['cell_move']} vs v27 {g['v27_move']}): playouts "
                  f"{outcomes} verified={g['verified']}", file=sys.stderr)
        n_verified = sum(1 for g in verified if g["verified"])
        summary_cells[key] = {
            "depth": depth,
            "node_budget": budget,
            "fires": st["fires"],
            "gains": len(st["gains"]),
            "losses": st["losses"],
            "losses_without_abort": st["losses_without_abort"],
            "aborts": st["aborts"],
            "verified_gains": n_verified,
            "verified_gains_checked": len(verified),
            "net": n_verified - st["losses"],
            "eligible": n_verified >= 1 and (n_verified - st["losses"]) > 0,
        }

    n_verified_total = sum(
        v["verified_gains"] for v in summary_cells.values()
    )
    summary = {
        "games": args.games,
        "seeds": [args.seed_a, args.seed_b],
        "control": {"depth": CONTROL[0], "node_budget": CONTROL[1]},
        "turns": turns,
        "control_fires": control_fires,
        "control_fire_rate": control_fires / turns if turns else 0.0,
        "control_confirms": control_confirms,
        "control_differs": control_differs,
        "cells": summary_cells,
        "verified_gains_total": n_verified_total,
        "criterion": "verified_gains_total >= 1",
        "check_m_pass": n_verified_total >= 1,
        "check_m_inconclusive": all(
            len(cells[_cell_key(d, b)]["gains"]) == 0
            for d, b in CELLS
        ),
        "all_gains": {k: v["gains"] for k, v in cells.items()},
    }
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "all_gains"},
        indent=2,
    ))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

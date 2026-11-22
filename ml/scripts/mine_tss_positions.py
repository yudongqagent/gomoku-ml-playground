#!/usr/bin/env python3
"""Mechanism check B for v52 (iter-52 §1.3): mine realistic positions
where TSS finds a forcing win the MCTS misses.

Procedure (pre-registered in iter-52 §1.3 BEFORE this script ran):

* Play 50 v27-vs-v27 games (normal rules, alternating colours,
  seeds 777/1777 — dev seeds, distinct from the battery's 100/200).
* At every turn, run TSS (production budget) WITHOUT perturbing the
  game (TSS consumes no RNG) and record: fires (TSS returns a move),
  confirms (TSS move == v27's move), differs (TSS move != v27's).
* For each differ (first 5, to bound runtime): verify the TSS move
  with 3 playouts from the position — play the TSS move, then
  continue with v52 on the attacker's side (it follows the forcing
  line via TSS on later turns) vs v27 defending (seeds 0/1/2). The
  differ is VERIFIED if the attacker wins 3/3 against v27's defence.

Pass criterion (pre-registered): >=1 verified differ. If 0 differs
occur in 50 games, the check is INCONCLUSIVE (not failed) — the
mechanism may simply be rare in v27-vs-v27 play — and the fast loop
is the arbiter of the in-game fire rate.

Usage:
    PYTHONPATH=ml python ml/scripts/mine_tss_positions.py
        --out research/iterations/tss-mechanism-check-b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

MAX_VERIFY = 5  # differs verified, to bound runtime
PLAYOUT_SEEDS = (0, 1, 2)


def _xy(move):
    return [move % 15, move // 15] if move is not None else None


def _board_to_lists(board):
    black = [[int(x), int(y)] for y, x in zip(*((board == 1).nonzero()))]
    white = [[int(x), int(y)] for y, x in zip(*((board == -1).nonzero()))]
    return black, white


def _playout(board, attacker, tss_move, seed):
    """Play the TSS move, then v52 (attacker) vs v27 (defender).
    Returns the winner (1/-1/0 for draw)."""
    from gomoku_ml import game, scoring

    b = game.apply_move(board, tss_move, attacker)
    over, winner = game.game_over(b)
    if over:
        return winner
    pa = scoring.get_player(52, seed=seed)
    pd = scoring.get_player(27, seed=seed + 1000)
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed-a", type=int, default=777)
    ap.add_argument("--seed-b", type=int, default=1777)
    args = ap.parse_args()

    from gomoku_ml import game, mcts_heuristic, scoring

    turns = 0
    fires = 0
    confirms = 0
    differs = []
    for gi in range(args.games):
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
            tss_move = mcts_heuristic.tss_winning_move(board, player)
            turns += 1
            if tss_move is not None:
                fires += 1
                if tss_move == v27_move:
                    confirms += 1
                else:
                    black, white = _board_to_lists(board)
                    differs.append({
                        "game": gi,
                        "to_move": "black" if player == game.BLACK else "white",
                        "black": black,
                        "white": white,
                        "tss_move": _xy(tss_move),
                        "v27_move": _xy(v27_move),
                    })
            if v27_move < 0:
                break
            board = game.apply_move(board, v27_move, player)
            player = -player
        print(f"game {gi + 1}/{args.games}: turns={turns} fires={fires} "
              f"differs={len(differs)}", file=sys.stderr)

    verified = []
    for d in differs[:MAX_VERIFY]:
        board = game.empty_board()
        for x, y in d["black"]:
            board[y, x] = game.BLACK
        for x, y in d["white"]:
            board[y, x] = game.WHITE
        attacker = game.BLACK if d["to_move"] == "black" else game.WHITE
        tss_move = d["tss_move"][1] * 15 + d["tss_move"][0]
        wins = 0
        outcomes = []
        for seed in PLAYOUT_SEEDS:
            winner = _playout(board, attacker, tss_move, seed)
            outcomes.append(winner)
            if winner == attacker:
                wins += 1
        d["playouts"] = outcomes
        d["verified"] = wins == len(PLAYOUT_SEEDS)
        verified.append(d)
        print(f"differ game {d['game']} ({d['to_move']}, TSS "
              f"{d['tss_move']} vs v27 {d['v27_move']}): playouts "
              f"{outcomes} verified={d['verified']}", file=sys.stderr)

    n_verified = sum(1 for d in verified if d["verified"])
    summary = {
        "games": args.games,
        "seeds": [args.seed_a, args.seed_b],
        "turns": turns,
        "fires": fires,
        "fire_rate": fires / turns if turns else 0.0,
        "confirms": confirms,
        "differs": len(differs),
        "differ_rate": len(differs) / turns if turns else 0.0,
        "verified_differs": n_verified,
        "criterion": "verified_differs >= 1",
        "check_b_pass": n_verified >= 1,
        "check_b_inconclusive": len(differs) == 0,
        "verified": verified,
        "all_differs": differs,
    }
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "all_differs"},
        indent=2,
    ))
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

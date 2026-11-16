#!/usr/bin/env python3
"""Generate policy+value training data from v18 self-play STARTING
FROM the tactical-suite positions (ML batch 1, H3 / v26, fast-loop
iteration 1 — setup-pattern augmentation).

Why: the v26 net is blind to quiet attacking setups (the tactical
suite's setup positions: the expected move ranks 20th-77th in the
net's policy, ~0 mass), and the value head cannot see that a setup
leads to a forced win 2 plies past the opponent's block (g94: the
net ranks the setup move #1 with 0.42 mass, yet the search picks a
different move because the setup line's q is +0.044 vs +0.555 for
a non-setup move). Normal v18 self-play rarely visits these
contrived configurations (the suite positions come from weak-engine
games), so the net never learns the pattern.

What: v18@50 self-play from ALL 49 tactical-suite positions (old +
hard — not just the 7 failures, so the net learns the general
setup/double-threat pattern rather than 7 answers). The games
diverge from the starting positions after a few moves, so the bulk
of the training positions are FRESH positions, not the suite
positions themselves. Policy targets are the teacher's root visit
distributions (same as gen_policy_data.py); value targets are game
outcomes. The first position of each game IS a suite position with
the teacher's visit distribution — that is the point (the teacher
finds the setup move; the net learns to rank it first).

Output: one ``.npz`` shard per game, same format as
gen_policy_data.py, under ``--out-dir``. Numbering continues after
existing shards (resume-safe).

Usage:
    python ml/scripts/gen_suite_data.py --games-per-position 3 \\
        --sims 50 --seed 6001 --out-dir ml/runs/ml1-h3/data-suite-a \\
        --position-offset 0 --position-count 25
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import encoding, evaluate, game, selfplay  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS  # noqa: E402
from gen_policy_data import _search_move  # noqa: E402


def play_from_position(board: np.ndarray, to_move: int,
                       mcts_black: HeuristicMCTS,
                       mcts_white: HeuristicMCTS,
                       rng: np.random.Generator,
                       temperature_moves: int = 4) -> dict:
    """One v18 self-play game starting from ``board`` with ``to_move``
    to play. Same sample format as
    :func:`gen_policy_data.play_teacher_game`."""
    board = board.copy()
    history: list[np.ndarray] = []
    last_move: int | None = None
    player = to_move
    planes_list, policy_list, players_list, moves_list = [], [], [], []
    winner = game.EMPTY
    for ply in range(game.SIZE * game.SIZE):
        over, win = game.game_over(board)
        if over:
            winner = win
            break
        mcts = mcts_black if player == game.BLACK else mcts_white
        planes_list.append(encoding.encode(board, player, history, last_move))
        move, policy = _search_move(
            mcts, board, player, rng, sample=(ply < temperature_moves))
        policy_list.append(policy)
        players_list.append(player)
        moves_list.append(move)
        history.append(board)
        board = game.apply_move(board, move, player)
        last_move = move
        player = -player
    else:
        _, winner = game.game_over(board)
    values = np.zeros(len(players_list), dtype=np.float32)
    for i, p in enumerate(players_list):
        values[i] = 0.0 if winner == game.EMPTY else (1.0 if winner == p else -1.0)
    return {
        "planes": np.stack(planes_list).astype(np.float32) if planes_list else np.zeros((0, 8, 15, 15), dtype=np.float32),
        "policy": np.stack(policy_list).astype(np.float32) if policy_list else np.zeros((0, 225), dtype=np.float32),
        "value": values,
        "moves": np.array(moves_list, dtype=np.int64),
        "winner": np.array(winner, dtype=np.int64),
        "aborted": np.array(False, dtype=bool),
    }


def load_all_positions() -> list[dict]:
    """All 49 tactical-suite positions (old 20 + hard 29)."""
    old = evaluate.load_tactical_suite()
    hard = evaluate.load_tactical_suite(
        os.path.join(os.path.dirname(evaluate.__file__),
                     "tactical_positions_hard.json"))
    return old["positions"] + hard["positions"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-position", type=int, default=3)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature-moves", type=int, default=4)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="ml1-h3-suite")
    parser.add_argument("--position-offset", type=int, default=0,
                        help="first suite position index (worker split)")
    parser.add_argument("--position-count", type=int, default=None,
                        help="how many suite positions (worker split)")
    args = parser.parse_args()

    positions = load_all_positions()
    if args.position_count is not None:
        positions = positions[args.position_offset:
                              args.position_offset + args.position_count]
    else:
        positions = positions[args.position_offset:]
    print(f"{len(positions)} suite positions, "
          f"{args.games_per_position} games each = "
          f"{len(positions) * args.games_per_position} games",
          file=sys.stderr, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    start_index = selfplay.existing_game_count(args.out_dir, args.run_name)
    rng = np.random.default_rng(args.seed)
    mcts_black = HeuristicMCTS(sims=args.sims, leaf_mode="rollout2_v1_both",
                               rng=np.random.default_rng(args.seed + 1))
    mcts_white = HeuristicMCTS(sims=args.sims, leaf_mode="rollout2_v1_both",
                               rng=np.random.default_rng(args.seed + 2))
    t0 = time.time()
    game_number = start_index
    total = len(positions) * args.games_per_position
    done = 0
    for pos in positions:
        board = evaluate.board_from_position(pos)
        to_move = pos["to_move"]
        for _ in range(args.games_per_position):
            game_number += 1
            done += 1
            g0 = time.time()
            samples = play_from_position(
                board, to_move, mcts_black, mcts_white, rng,
                temperature_moves=args.temperature_moves)
            path = os.path.join(
                args.out_dir, f"{args.run_name}-{game_number:05d}.npz")
            np.savez_compressed(
                path,
                planes=samples["planes"], policy=samples["policy"],
                value=samples["value"], moves=samples["moves"],
                winner=samples["winner"], aborted=samples["aborted"])
            n = len(samples["moves"])
            print(f"game {game_number} ({done}/{total}) [{pos['id']}]: "
                  f"{n} plies, winner={int(samples['winner'])}, "
                  f"{time.time() - g0:.1f}s, elapsed {time.time() - t0:.0f}s",
                  file=sys.stderr, flush=True)
    print(f"done: {total} games in {time.time() - t0:.0f}s -> {args.out_dir}",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate policy+value training data from v18 (rollout-leaf MCTS)
self-play (ML batch 1, H3 / v26).

For every position in a teacher self-play game, the policy target is
the teacher's root VISIT DISTRIBUTION (temperature 1, normalised over
candidates) — the AlphaZero target. Positions decided by the
win/block scans (no search ran) get a 1-hot target on the played
move. The value target is the game outcome from the player-to-move
perspective. The played move is SAMPLED from the visit distribution
for the first ``--temperature-moves`` plies, then greedy
(most-visited, ties by RNG) — AlphaZero-style early exploration.

Output: one ``.npz`` shard per game (``planes``/``policy``/``value``/
``moves``/``winner`` — the same format as
:mod:`gomoku_ml.selfplay`), under ``--out-dir``. Numbering continues
after existing shards (resume-safe).

Usage:
    python ml/scripts/gen_policy_data.py --games 150 --sims 50 \\
        --seed 5001 --out-dir ml/runs/ml1-h3/data-a
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

from gomoku_ml import encoding, game, scoring, selfplay  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS, _Node  # noqa: E402


def _search_move(mcts: HeuristicMCTS, board: np.ndarray, player: int,
                 rng: np.random.Generator, sample: bool):
    """Run the teacher's search on (board, player). Returns
    ``(move, policy)`` where ``policy`` is the [225] visit
    distribution (0 on non-candidates). Win/block scans bypass the
    search and return a 1-hot policy — replicating
    :meth:`HeuristicMCTS.choose_move`'s scan order exactly."""
    wins = scoring._winning_moves(board, player)
    if wins:
        policy = np.zeros(game.SIZE * game.SIZE, dtype=np.float32)
        policy[wins[0]] = 1.0
        return wins[0], policy
    opp_wins = scoring._winning_moves(board, -player)
    if len(opp_wins) == 1:
        policy = np.zeros(game.SIZE * game.SIZE, dtype=np.float32)
        policy[opp_wins[0]] = 1.0
        return opp_wins[0], policy
    cands = scoring._candidate_moves(board)
    if not cands:
        return -1, np.zeros(game.SIZE * game.SIZE, dtype=np.float32)
    if len(cands) == 1:
        policy = np.zeros(game.SIZE * game.SIZE, dtype=np.float32)
        policy[cands[0]] = 1.0
        return cands[0], policy
    root = _Node(1.0)
    for _ in range(mcts.sims):
        mcts._run_one(root, board, player, None)
    # Note: the first sim expands the root without visiting a child,
    # so the child visits sum to sims - 1, not sims — normalise by
    # the actual sum.
    total = sum(c.visit_count for c in root.children.values())
    policy = np.zeros(game.SIZE * game.SIZE, dtype=np.float32)
    for m, c in root.children.items():
        policy[m] = c.visit_count / total
    if sample:
        moves = np.array(sorted(root.children.keys()))
        probs = policy[moves]
        probs = probs / probs.sum()
        move = int(rng.choice(moves, p=probs))
    else:
        best_visits = max(c.visit_count for c in root.children.values())
        best = [m for m, c in root.children.items()
                if c.visit_count == best_visits]
        move = int(rng.choice(best)) if len(best) > 1 else best[0]
    return move, policy


def play_teacher_game(mcts_black: HeuristicMCTS, mcts_white: HeuristicMCTS,
                      rng: np.random.Generator,
                      temperature_moves: int = 10) -> dict:
    """One v18 self-play game. Returns the sample dict in the
    :func:`gomoku_ml.selfplay.play_game` format."""
    board = game.empty_board()
    history: list[np.ndarray] = []
    last_move: int | None = None
    player = game.BLACK
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature-moves", type=int, default=10)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", default="ml1-h3")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    start_index = selfplay.existing_game_count(args.out_dir, args.run_name)
    rng = np.random.default_rng(args.seed)
    mcts_black = HeuristicMCTS(sims=args.sims, leaf_mode="rollout2_v1_both",
                               rng=np.random.default_rng(args.seed + 1))
    mcts_white = HeuristicMCTS(sims=args.sims, leaf_mode="rollout2_v1_both",
                               rng=np.random.default_rng(args.seed + 2))
    t0 = time.time()
    for i in range(args.games):
        game_number = start_index + i + 1
        g0 = time.time()
        samples = play_teacher_game(
            mcts_black, mcts_white, rng,
            temperature_moves=args.temperature_moves)
        path = os.path.join(
            args.out_dir, f"{args.run_name}-{game_number:05d}.npz")
        np.savez_compressed(
            path,
            planes=samples["planes"], policy=samples["policy"],
            value=samples["value"], moves=samples["moves"],
            winner=samples["winner"], aborted=samples["aborted"])
        n = len(samples["moves"])
        print(f"game {game_number} ({i + 1}/{args.games}): {n} plies, "
              f"winner={int(samples['winner'])}, {time.time() - g0:.1f}s, "
              f"elapsed {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    print(f"done: {args.games} games in {time.time() - t0:.0f}s -> {args.out_dir}",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

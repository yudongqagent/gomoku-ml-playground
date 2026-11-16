#!/usr/bin/env python3
"""Convert H1's move-list game JSONs to .npz policy shards (ML batch 1,
H3 / v26).

H1 saves v18/v7 self-play games as JSON with full move lists
(``ml/runs/ml1-h1/games-v18.json`` etc.) but WITHOUT visit
distributions, so the policy target here is 1-hot on the played
move (the teacher's chosen move) and the value target is the game
outcome from the player-to-move perspective. Positions are encoded
with the same 8-plane encoding + game history as
:mod:`gen_policy_data` output, so the shards mix directly with
H3's own data at train time.

Only ``--source`` games are converted (default ``v18`` — the
teacher; H1's v7 games are a weaker teacher and are NOT mixed in).

Usage:
    python ml/scripts/convert_h1_games.py \\
        --games ~/workspace/projects/gomoku-ml1-h1/ml/runs/ml1-h1/games-v18.json \\
        --out-dir ml/runs/ml1-h3/data-h1 --source v18
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import encoding, game  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source", default="v18")
    args = parser.parse_args()

    with open(args.games, encoding="utf-8") as f:
        payload = json.load(f)
    games = payload["games"] if isinstance(payload, dict) else payload
    os.makedirs(args.out_dir, exist_ok=True)
    n_out = 0
    n_pos = 0
    for g in games:
        if g.get("source", args.source) != args.source:
            continue
        moves = [int(m) for m in g["moves"]]
        winner = int(g["winner"])
        board = game.empty_board()
        history: list[np.ndarray] = []
        last_move: int | None = None
        player = game.BLACK
        planes_list, policy_list, players_list = [], [], []
        for move in moves:
            planes_list.append(
                encoding.encode(board, player, history, last_move))
            one_hot = np.zeros(game.SIZE * game.SIZE, dtype=np.float32)
            one_hot[move] = 1.0
            policy_list.append(one_hot)
            players_list.append(player)
            history.append(board)
            board = game.apply_move(board, move, player)
            last_move = move
            player = -player
        values = np.zeros(len(players_list), dtype=np.float32)
        for i, p in enumerate(players_list):
            values[i] = 0.0 if winner == game.EMPTY else (
                1.0 if winner == p else -1.0)
        path = os.path.join(args.out_dir, f"h1-{g.get('game', n_out):05d}.npz")
        np.savez_compressed(
            path,
            planes=np.stack(planes_list).astype(np.float32),
            policy=np.stack(policy_list).astype(np.float32),
            value=values,
            moves=np.array(moves, dtype=np.int64),
            winner=np.array(winner, dtype=np.int64),
            aborted=np.array(False, dtype=bool))
        n_out += 1
        n_pos += len(moves)
    print(f"converted {n_out} {args.source} games, {n_pos} positions "
          f"-> {args.out_dir}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

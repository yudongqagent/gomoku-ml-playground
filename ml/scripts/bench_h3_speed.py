#!/usr/bin/env python3
"""H3 sizing measurements: v18 self-play game time at 50/100 sims,
v8 game time, and 16ch/2-block net forward cost (batch=1)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gomoku_ml import game, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS  # noqa: E402
from gomoku_ml.architectures import build_architecture  # noqa: E402
from selfplay_batch import play_recorded_game  # noqa: E402


def time_games(player_a, player_b, n, label):
    times = []
    for i in range(n):
        t0 = time.time()
        winner, moves = play_recorded_game(
            player_a.choose_move, player_b.choose_move,
            game.empty_board(), game.BLACK,
        )
        dt = time.time() - t0
        times.append(dt)
        print(f"{label} game {i + 1}: {dt:.1f}s, {len(moves)} plies, "
              f"winner={winner}", flush=True)
    print(f"{label}: mean {np.mean(times):.1f}s/game", flush=True)


def main():
    # v18 at 50 sims (teacher setting for data gen).
    a = HeuristicMCTS(sims=50, leaf_mode="rollout2_v1_both",
                      rng=np.random.default_rng(1))
    b = HeuristicMCTS(sims=50, leaf_mode="rollout2_v1_both",
                      rng=np.random.default_rng(2))
    time_games(a, b, 2, "v18@50 vs v18@50")
    # v18 at 100 sims (battery setting).
    a = HeuristicMCTS(sims=100, leaf_mode="rollout2_v1_both",
                      rng=np.random.default_rng(3))
    b = HeuristicMCTS(sims=100, leaf_mode="rollout2_v1_both",
                      rng=np.random.default_rng(4))
    time_games(a, b, 1, "v18@100 vs v18@100")
    # v8 vs v0 (battery cost reference for net-using engines).
    v8 = scoring.get_player(8, seed=5)
    from gomoku_ml import baseline
    v0 = baseline.HeuristicPlayer(seed=6)
    time_games(v8, v0, 2, "v8 vs v0")
    # Net forward cost, 16ch/2 blocks, batch=1, 2 threads.
    torch.set_num_threads(2)
    net = build_architecture("resnet", {"channels": 16, "blocks": 2})
    net.eval()
    x = torch.zeros(1, 8, 15, 15)
    with torch.no_grad():
        for _ in range(5):
            net(x)
        t0 = time.time()
        n = 200
        for _ in range(n):
            net(x)
    print(f"net fwd 16ch/2blk batch=1: "
          f"{(time.time() - t0) / n * 1000:.2f} ms/call", flush=True)
    print(f"net params: {net.num_parameters()}", flush=True)


if __name__ == "__main__":
    main()

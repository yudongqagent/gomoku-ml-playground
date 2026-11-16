#!/usr/bin/env python3
"""Cost measurements for ML batch 1 H1 (distillation sizing).

Measures, in-process and warm:
  * rollout-leaf cost per call (all positions + quiet-only subset),
  * v7-vs-v7 and v18-vs-v18 self-play game time at 50 sims,
  * value-net forward cost (batch=1, and batched training throughput).

Usage:
    python ml/scripts/bench_distill.py --positions /tmp/rollout_pos.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", default="/tmp/rollout_pos.json")
    parser.add_argument("--v18-games", type=int, default=1)
    parser.add_argument("--v7-games", type=int, default=2)
    args = parser.parse_args()

    import numpy as np  # noqa: E402
    from gomoku_ml import game, mcts_heuristic, scoring  # noqa: E402

    out: dict = {}
    with open(args.positions, encoding="utf-8") as f:
        positions = json.load(f)["positions"]
    boards = [(np.array(p["board"], dtype=np.int8), p["player"])
              for p in positions]
    quiet = [(b, p) for b, p in boards
             if mcts_heuristic._terminal_leaf_value(b, p) == 0.0]
    print(f"{len(boards)} positions, {len(quiet)} quiet", flush=True)

    # Leaf cost: 3 warm passes over the quiet subset, per-call times.
    times: list[float] = []
    for _ in range(3):
        for b, p in quiet:
            t0 = time.perf_counter()
            mcts_heuristic._rollout_leaf_value(b, p)
            times.append(time.perf_counter() - t0)
    times_ms = sorted(t * 1000 for t in times)
    out["leaf_quiet_ms"] = {
        "n": len(times_ms),
        "median": round(statistics.median(times_ms), 3),
        "mean": round(statistics.mean(times_ms), 3),
        "p90": round(times_ms[int(0.9 * len(times_ms))], 3),
    }
    print(f"leaf (quiet): median {out['leaf_quiet_ms']['median']} ms/call "
          f"(mean {out['leaf_quiet_ms']['mean']}, p90 "
          f"{out['leaf_quiet_ms']['p90']}, n={len(times_ms)})", flush=True)

    # Leaf cost on ALL positions (game-play path includes terminals).
    times = []
    for _ in range(2):
        for b, p in boards:
            t0 = time.perf_counter()
            mcts_heuristic._rollout_leaf_value(b, p)
            times.append(time.perf_counter() - t0)
    times_ms = sorted(t * 1000 for t in times)
    out["leaf_all_ms"] = {
        "n": len(times_ms),
        "median": round(statistics.median(times_ms), 3),
        "mean": round(statistics.mean(times_ms), 3),
    }
    print(f"leaf (all): median {out['leaf_all_ms']['median']} ms/call "
          f"(mean {out['leaf_all_ms']['mean']})", flush=True)

    # v7 self-play game time at 50 sims.
    v7_times = []
    for g in range(args.v7_games):
        pa = scoring.get_player(7, seed=9000 + g)
        pb = scoring.get_player(7, seed=9100 + g)
        pa._mcts.sims = 50
        pb._mcts.sims = 50
        board = game.empty_board()
        player = game.BLACK
        t0 = time.time()
        plies = 0
        while True:
            over, _win = game.game_over(board)
            if over:
                break
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
            plies += 1
        v7_times.append(round(time.time() - t0, 1))
        print(f"v7 game {g}: {v7_times[-1]}s ({plies} plies)", flush=True)
    out["v7_game_s_50sims"] = v7_times

    # v18 self-play game time at 50 sims.
    v18_times = []
    for g in range(args.v18_games):
        pa = scoring.get_player(18, seed=9200 + g)
        pb = scoring.get_player(18, seed=9300 + g)
        pa._mcts.sims = 50
        pb._mcts.sims = 50
        board = game.empty_board()
        player = game.BLACK
        t0 = time.time()
        plies = 0
        while True:
            over, _win = game.game_over(board)
            if over:
                break
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player
            plies += 1
        v18_times.append(round(time.time() - t0, 1))
        print(f"v18 game {g}: {v18_times[-1]}s ({plies} plies)", flush=True)
    out["v18_game_s_50sims"] = v18_times

    # Net forward cost: v8's checkpoint (16ch/2 blocks), batch=1 and
    # batched (training throughput proxy).
    import torch  # noqa: E402
    from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402

    torch.set_num_threads(2)
    repo_root = common.REPO_ROOT
    net, _ = load_model_from_checkpoint(
        os.path.join(repo_root, "ml", "runs", "b3-h2", "value-net.pt"))
    net.eval()
    x1 = torch.zeros(1, 8, 15, 15)
    with torch.no_grad():
        for _ in range(20):
            net(x1)
        t0 = time.perf_counter()
        for _ in range(200):
            net(x1)
        out["net_forward_batch1_ms"] = round(
            (time.perf_counter() - t0) / 200 * 1000, 3)
        print(f"net forward batch=1: {out['net_forward_batch1_ms']} ms",
              flush=True)
        xb = torch.zeros(64, 8, 15, 15)
        for _ in range(5):
            net(xb)
        t0 = time.perf_counter()
        for _ in range(20):
            net(xb)
        out["net_forward_batch64_ms"] = round(
            (time.perf_counter() - t0) / 20 * 1000, 2)
        print(f"net forward batch=64: {out['net_forward_batch64_ms']} ms "
              f"({out['net_forward_batch64_ms'] / 64:.2f} ms/sample)",
              flush=True)

    print("BENCH_JSON " + json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark + verify batched leaf evaluation for neural MCTS self-play.

Runs three checks with one fixed config (same net/checkpoint, same
sims) and prints a JSON summary plus human-readable lines:

1. **Consistency**: ``_evaluate_batch`` vs ``_evaluate`` on N random
   positions — max abs policy/value diff (expect ~1e-6 float noise).
2. **Agreement**: ``search_many`` vs ``search`` visit distributions on
   N random positions — mean/max total-variation distance and argmax
   agreement (expect near-identical trees: same selection/backprop,
   only float noise + RNG interleaving differ).
3. **Timing**: sequential ``generate_games`` vs batched
   ``generate_games_batched`` for G games at S sims — wall-clock,
   games/hour each way, speedup factor, and outcome stats
   (W/L/D, mean plies) as self-play equivalence evidence.

GPU note: this script only measures what it runs on. On this CPU-only
VM it measures CPU numbers; a GPU speedup claim needs a run on a GPU
(pass ``--device cuda`` there).

Usage:
    python ml/scripts/bench_batched.py --config longrun-cpu \\
        --checkpoint server/data/jobs/<job>/final.pt \\
        --games 20 --sims 100 --concurrency 8
    python ml/scripts/bench_batched.py --config smoke --skip-timing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import game, selfplay  # noqa: E402
from gomoku_ml.architectures import build_architecture  # noqa: E402
from gomoku_ml.config import load_config  # noqa: E402
from gomoku_ml.mcts import MCTS, LeafRequest, SearchRequest  # noqa: E402
from gomoku_ml.train import load_checkpoint  # noqa: E402


def random_positions(rng: np.random.Generator, n: int, max_plies: int = 30):
    """N random mid-game positions (board, player, history, last_move)."""
    positions = []
    for _ in range(n):
        board = game.empty_board()
        history: list[np.ndarray] = []
        last_move: int | None = None
        player = game.BLACK
        for _ in range(int(rng.integers(0, max_plies + 1))):
            legal = game.legal_moves(board)
            move = int(rng.choice(legal))
            history.append(board)
            board = game.apply_move(board, move, player)
            last_move = move
            player = -player
        positions.append((board, player, history, last_move))
    return positions


def check_consistency(mcts: MCTS, positions) -> dict:
    """Batch vs single forward pass on the same positions."""
    requests = [LeafRequest(b, p, h, lm) for b, p, h, lm in positions]
    batched = mcts._evaluate_batch(requests)
    max_policy = 0.0
    max_value = 0.0
    for req, (probs_b, value_b) in zip(requests, batched):
        probs_s, value_s = mcts._evaluate(
            req.board, req.player, req.history, req.last_move
        )
        max_policy = max(max_policy, float(np.abs(probs_b - probs_s).max()))
        max_value = max(max_value, abs(value_b - value_s))
    return {
        "positions": len(requests),
        "max_policy_abs_diff": max_policy,
        "max_value_abs_diff": max_value,
    }


def check_agreement(mcts: MCTS, positions, sims: int) -> dict:
    """search_many vs search visit distributions (no Dirichlet noise)."""
    requests = [
        SearchRequest(board=b, player=p, num_simulations=sims,
                      history=h, last_move=lm, add_noise=False)
        for b, p, h, lm in positions
    ]
    batched = mcts.search_many(requests)
    tvs = []
    argmax_agree = 0
    value_diffs = []
    for req, (root_b, value_b) in zip(requests, batched):
        root_s, value_s = mcts.search(
            req.board, req.player, req.num_simulations,
            history=req.history, last_move=req.last_move, add_noise=False,
        )
        dist_b = MCTS.visit_distribution(root_b, temperature=1.0)
        dist_s = MCTS.visit_distribution(root_s, temperature=1.0)
        tvs.append(0.5 * float(np.abs(dist_b - dist_s).sum()))
        if int(np.argmax(dist_b)) == int(np.argmax(dist_s)):
            argmax_agree += 1
        value_diffs.append(abs(value_b - value_s))
    return {
        "positions": len(requests),
        "sims": sims,
        "mean_tv_distance": float(np.mean(tvs)) if tvs else 0.0,
        "max_tv_distance": float(np.max(tvs)) if tvs else 0.0,
        "argmax_agreement": f"{argmax_agree}/{len(requests)}",
        "max_root_value_diff": float(np.max(value_diffs)) if value_diffs else 0.0,
    }


def shard_stats(paths: list[str]) -> dict:
    """Outcome stats over saved shards (equivalence evidence)."""
    wins = losses = draws = 0
    plies: list[int] = []
    for path in paths:
        with np.load(path) as z:
            winner = int(z["winner"])
            plies.append(int(z["moves"].shape[0]))
        if winner == game.BLACK:
            wins += 1
        elif winner == game.WHITE:
            losses += 1
        else:
            draws += 1
    return {
        "games": len(paths),
        "black_wins": wins,
        "white_wins": losses,
        "draws": draws,
        "mean_plies": round(float(np.mean(plies)), 1) if plies else 0.0,
        "min_plies": int(np.min(plies)) if plies else 0,
        "max_plies": int(np.max(plies)) if plies else 0,
    }


def run_timing_arm(mcts, arm, games, sims, out_dir, run_name, seed,
                   concurrency, max_batch_size, game_kwargs):
    """One timed arm ('sequential' or 'batched'). Returns a result dict."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    if arm == "sequential":
        paths = selfplay.generate_games(
            mcts, num_games=games, num_simulations=sims,
            out_dir=out_dir, run_name=run_name, **game_kwargs,
        )
    else:
        paths = selfplay.generate_games_batched(
            mcts, num_games=games, num_simulations=sims,
            out_dir=out_dir, run_name=run_name,
            concurrency=concurrency, max_batch_size=max_batch_size,
            **game_kwargs,
        )
    elapsed = time.time() - t0
    result = {
        "arm": arm,
        "games": games,
        "sims": sims,
        "wall_seconds": round(elapsed, 1),
        "games_per_hour": round(games / elapsed * 3600, 1) if elapsed > 0 else 0.0,
        "out_dir": out_dir,
    }
    result.update(shard_stats(paths))
    if arm == "batched":
        result["concurrency"] = concurrency
        result["max_batch_size"] = max_batch_size
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="longrun-cpu",
                        help="config name or path (arch + self-play defaults)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint .pt (default: config base_checkpoint)")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--agreement-sims", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work-dir", default=None,
                        help="shard output root (default: /tmp/bench-batched-<ts>)")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--out", default=None, help="JSON summary path")
    args = parser.parse_args()

    cfg = load_config(common.config_path(args.config))
    sp = cfg["selfplay"]
    checkpoint = args.checkpoint or cfg.get("base_checkpoint")
    if not checkpoint:
        print("need --checkpoint (config has no base_checkpoint)",
              file=sys.stderr)
        return 2
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    arch = cfg["architecture"]
    model = build_architecture(arch["name"], arch["params"])
    load_checkpoint(checkpoint, model, device=args.device)
    model.to(args.device)
    model.eval()
    mcts = MCTS(
        model, device=args.device, c_puct=float(sp["c_puct"]),
        dirichlet_alpha=float(sp["dirichlet_alpha"]),
        dirichlet_epsilon=float(sp["dirichlet_epsilon"]),
    )
    summary: dict = {
        "checkpoint": checkpoint,
        "architecture": arch,
        "device": args.device,
        "torch_threads": args.threads,
        "seed": args.seed,
    }

    if not args.skip_checks:
        rng = np.random.default_rng(args.seed + 999)
        positions = random_positions(rng, args.positions)
        consistency = check_consistency(mcts, positions)
        summary["consistency"] = consistency
        print(f"consistency ({consistency['positions']} positions): "
              f"max policy abs diff {consistency['max_policy_abs_diff']:.3e}, "
              f"max value abs diff {consistency['max_value_abs_diff']:.3e}")
        agreement = check_agreement(mcts, positions, args.agreement_sims)
        summary["agreement"] = agreement
        print(f"agreement ({agreement['positions']} positions, "
              f"{agreement['sims']} sims): mean TV "
              f"{agreement['mean_tv_distance']:.4f}, max TV "
              f"{agreement['max_tv_distance']:.4f}, argmax "
              f"{agreement['argmax_agreement']}, max root-value diff "
              f"{agreement['max_root_value_diff']:.4f}")

    if not args.skip_timing:
        work_dir = args.work_dir or os.path.join(
            "/tmp", f"bench-batched-{time.strftime('%Y%m%d-%H%M%S')}")
        game_kwargs = dict(
            temperature_moves=int(sp["temperature_moves"]),
            temperature=float(sp["temperature"]),
            resign_threshold=sp["resign_threshold"],
            max_moves=int(sp["max_moves"]),
        )
        seq = run_timing_arm(
            mcts, "sequential", args.games, args.sims,
            os.path.join(work_dir, "sequential"), "bench",
            args.seed, args.concurrency, args.max_batch_size, game_kwargs,
        )
        print(f"sequential: {seq['games']} games in {seq['wall_seconds']}s "
              f"-> {seq['games_per_hour']} games/hour "
              f"(B {seq['black_wins']}W / W {seq['white_wins']}W / "
              f"{seq['draws']}D, mean {seq['mean_plies']} plies)")
        bat = run_timing_arm(
            mcts, "batched", args.games, args.sims,
            os.path.join(work_dir, "batched"), "bench",
            args.seed, args.concurrency, args.max_batch_size, game_kwargs,
        )
        print(f"batched (concurrency={args.concurrency}, "
              f"max_batch={args.max_batch_size}): {bat['games']} games in "
              f"{bat['wall_seconds']}s -> {bat['games_per_hour']} games/hour "
              f"(B {bat['black_wins']}W / W {bat['white_wins']}W / "
              f"{bat['draws']}D, mean {bat['mean_plies']} plies)")
        speedup = (
            bat["games_per_hour"] / seq["games_per_hour"]
            if seq["games_per_hour"] > 0 else 0.0
        )
        summary["timing"] = {
            "games": args.games, "sims": args.sims,
            "sequential": seq, "batched": bat,
            "speedup": round(speedup, 2),
            "work_dir": work_dir,
        }
        print(f"speedup (batched / sequential): {speedup:.2f}x")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"summary written to {args.out}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Benchmark B1 parallel self-play against the B2-only batched path.

Runs one timed arm per worker count (default: 1, 2, 3, 4) with the
same config, checkpoint, sims, seed scheme and game count, and prints
a JSON summary plus a human-readable table:

* workers=1 arm: the B2-only path, ``generate_games_batched`` in this
  process (torch threads 2, as in ``bench_batched.py``) — the "before".
* workers>=2 arms: ``generate_games_parallel`` (1 torch thread per
  worker) — the "after".

Load average is recorded before/after each arm: sibling jobs on this
shared 4-CPU VM contend with the measurement, and a contended number
must be reported as contended, not clean (batch 2's B2 doc records
exactly this failure mode — a 12.87x "speedup" that was a contention
artifact). Run when the VM is as quiet as possible, and read the
load averages in the summary before quoting any number.

No GPU numbers: this VM has no GPU. The script measures CPU
throughput only; pass ``--device cuda`` on a GPU box to measure
there (the parallel path is device-agnostic — each worker loads its
own copy of the net on the given device).

Usage:
    python ml/scripts/bench_parallel.py --config ml/runs/b3-b1/config.yaml \\
        --checkpoint ml/runs/b3-b1/final.pt --games 40 --sims 50 \\
        --concurrency 8 --workers 1,2,3,4 --out /tmp/bench-b1.json
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
from gomoku_ml.mcts import MCTS  # noqa: E402
from gomoku_ml.train import load_checkpoint  # noqa: E402


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


def run_arm(mcts, net_spec, workers, games, sims, out_dir, run_name,
            seed, concurrency, max_batch_size, game_kwargs, threads):
    """One timed arm. Returns a result dict."""
    load_before = [round(x, 2) for x in os.getloadavg()]
    t0 = time.time()
    if workers == 1:
        # B2-only baseline, in-process, threads as configured (2 by
        # default, matching bench_batched.py).
        torch.set_num_threads(threads)
        np.random.seed(seed)
        torch.manual_seed(seed)
        paths = selfplay.generate_games_batched(
            mcts, num_games=games, num_simulations=sims,
            out_dir=out_dir, run_name=run_name,
            concurrency=concurrency, max_batch_size=max_batch_size,
            **game_kwargs,
        )
    else:
        paths = selfplay.generate_games_parallel(
            net_spec, num_games=games, num_simulations=sims,
            out_dir=out_dir, run_name=run_name,
            workers=workers, concurrency=concurrency,
            max_batch_size=max_batch_size, seed=seed,
            **game_kwargs,
        )
    elapsed = time.time() - t0
    load_after = [round(x, 2) for x in os.getloadavg()]
    result = {
        "workers": workers,
        "games": games,
        "sims": sims,
        "concurrency_per_worker": concurrency,
        "max_batch_size": max_batch_size,
        "wall_seconds": round(elapsed, 1),
        "games_per_hour": round(games / elapsed * 3600, 1) if elapsed > 0 else 0.0,
        "loadavg_before_1_5_15": load_before,
        "loadavg_after_1_5_15": load_after,
        "out_dir": out_dir,
    }
    if workers == 1:
        result["torch_threads"] = threads
    else:
        result["torch_threads_per_worker"] = 1
    result.update(shard_stats(paths))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="longrun-cpu",
                        help="config name or path (arch + self-play defaults)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint .pt (default: config base_checkpoint)")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="games at once per worker (also the workers=1 arm's concurrency)")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--workers", default="1,2,3,4",
                        help="comma-separated worker counts to measure")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2,
                        help="torch threads for the workers=1 arm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--work-dir", default=None,
                        help="shard output root (default: /tmp/bench-parallel-<ts>)")
    parser.add_argument("--out", default=None, help="JSON summary path")
    args = parser.parse_args()

    worker_counts = [int(x) for x in args.workers.split(",") if x.strip()]
    if not worker_counts or any(w < 1 for w in worker_counts):
        print("--workers must be a comma-separated list of ints >= 1",
              file=sys.stderr)
        return 2

    cfg = load_config(common.config_path(args.config))
    sp = cfg["selfplay"]
    checkpoint = args.checkpoint or cfg.get("base_checkpoint")
    if not checkpoint:
        print("need --checkpoint (config has no base_checkpoint)",
              file=sys.stderr)
        return 2

    arch = cfg["architecture"]
    model = build_architecture(arch["name"], arch["params"])
    load_checkpoint(checkpoint, model, device=args.device)
    model.to(args.device)
    model.eval()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    mcts = MCTS(
        model, device=args.device, c_puct=float(sp["c_puct"]),
        dirichlet_alpha=float(sp["dirichlet_alpha"]),
        dirichlet_epsilon=float(sp["dirichlet_epsilon"]),
    )
    net_spec = selfplay.NetSpec(
        arch_name=arch["name"], arch_params=arch["params"],
        checkpoint=checkpoint, device=args.device,
        c_puct=float(sp["c_puct"]),
        dirichlet_alpha=float(sp["dirichlet_alpha"]),
        dirichlet_epsilon=float(sp["dirichlet_epsilon"]),
    )
    summary: dict = {
        "checkpoint": checkpoint,
        "architecture": arch,
        "device": args.device,
        "seed": args.seed,
        "games_per_arm": args.games,
        "sims": args.sims,
        "concurrency_per_worker": args.concurrency,
        "max_batch_size": args.max_batch_size,
        "cpus": os.cpu_count(),
        "arms": {},
    }
    work_dir = args.work_dir or os.path.join(
        "/tmp", f"bench-parallel-{time.strftime('%Y%m%d-%H%M%S')}")
    game_kwargs = dict(
        temperature_moves=int(sp["temperature_moves"]),
        temperature=float(sp["temperature"]),
        resign_threshold=sp["resign_threshold"],
        max_moves=int(sp["max_moves"]),
    )
    baseline_gph = None
    for workers in worker_counts:
        out_dir = os.path.join(work_dir, f"workers-{workers}")
        result = run_arm(
            mcts, net_spec, workers, args.games, args.sims,
            out_dir, "bench", args.seed, args.concurrency,
            args.max_batch_size, game_kwargs, args.threads,
        )
        summary["arms"][f"workers_{workers}"] = result
        if baseline_gph is None:
            baseline_gph = result["games_per_hour"]
        speedup = (
            result["games_per_hour"] / baseline_gph
            if baseline_gph else 0.0
        )
        result["speedup_vs_first_arm"] = round(speedup, 2)
        print(f"workers={workers}: {result['games']} games in "
              f"{result['wall_seconds']}s -> "
              f"{result['games_per_hour']} games/hour "
              f"({speedup:.2f}x vs first arm) "
              f"(B {result['black_wins']}W / W {result['white_wins']}W / "
              f"{result['draws']}D, mean {result['mean_plies']} plies; "
              f"loadavg before {result['loadavg_before_1_5_15']}, "
              f"after {result['loadavg_after_1_5_15']})",
              flush=True)

    summary["work_dir"] = work_dir
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

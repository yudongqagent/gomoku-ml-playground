#!/usr/bin/env python3
"""Generate self-play games with MCTS and save them as .npz shards.

Writes to a run directory (see gomoku_ml.runs): shards land in
``<run-dir>/data/``, the resolved config snapshot in
``<run-dir>/config.yaml``, progress JSON events on stdout (one JSON
object per line, each with an "event" key) and in ``metrics.jsonl``,
human-readable progress on stderr.

Long runs: the current game is written to ``<run-dir>/current_game.json``
after every move (live feed for the Trainer UI), and a data manifest
(``data/manifest.json``) is updated after every game.

**Top-up semantics:** ``--games`` / ``selfplay.games`` is the TOTAL
number of games wanted in the output directory, not the number to add.
Re-running the same command generates only the games still missing
(``max(0, target - existing)``), continuing shard numbering — a
4-game job interrupted after 3 games resumes to 4 shards, not 7.
To generate MORE games than a previous run, raise the target.
(The library function ``generate_games`` itself always appends exactly
N new games; the top-up policy lives in this script.)

SIGINT/SIGTERM stop the run gracefully: the in-progress game is
abandoned (it has no outcome), all completed games are already on
disk, and the exit code is 130 (SIGINT) or 143 (SIGTERM) with the
resume command printed (re-run the same command — top-up handles the
rest).

Usage:
    python ml/scripts/selfplay.py --config ml/configs/smoke.yaml
    python ml/scripts/selfplay.py --config smoke --run-dir ml/runs/20261112-000000-smoke
    python ml/scripts/selfplay.py --config full.yaml --checkpoint ml/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import game, runs, selfplay  # noqa: E402
from gomoku_ml.architectures import build_architecture  # noqa: E402
from gomoku_ml.config import ConfigError, load_config  # noqa: E402
from gomoku_ml.interrupt import InterruptFlag  # noqa: E402
from gomoku_ml.mcts import MCTS  # noqa: E402
from gomoku_ml.train import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke", help="config name or path")
    parser.add_argument("--run-dir", default=None,
                        help="run directory (default: fresh ml/runs/<timestamp>-<name>/)")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint .pt to play with (overrides config base_checkpoint; default: random init)")
    parser.add_argument("--games", type=int, default=None,
                        help="TOTAL games wanted in the output dir (top-up: only the missing games are generated)")
    parser.add_argument("--sims", type=int, default=None, help="override MCTS simulations per move")
    parser.add_argument("--out-dir", default=None, help="override shard output directory")
    parser.add_argument("--run-name", default=None, help="override run name (used for the run-dir suffix)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="games played at once with batched leaf eval "
                        "(default 1 = the original sequential single-leaf "
                        "path; >1 uses generate_games_batched)")
    parser.add_argument("--max-batch-size", type=int, default=32,
                        help="max positions per batched forward pass "
                        "(only used with --concurrency > 1)")
    args = parser.parse_args()

    try:
        cfg = load_config(common.config_path(args.config))
    except (ConfigError, FileNotFoundError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.run_name:
        cfg["run_name"] = args.run_name
    sp = cfg["selfplay"]
    num_games = args.games if args.games is not None else int(sp["games"])
    sims = args.sims if args.sims is not None else int(sp["sims_per_move"])
    run_dir = runs.resolve_run_dir(common.ML_DIR, args.run_dir, cfg["run_name"])
    out_dir = args.out_dir or os.path.join(run_dir, "data")
    flag = InterruptFlag().install()

    with runs.Run(run_dir) as run:
        run.save_config(cfg)
        # Top-up: generate only the games still missing to reach the
        # target total (see module docstring).
        already = selfplay.existing_game_count(out_dir, cfg["run_name"])
        to_generate = max(0, num_games - already)
        run.log_event("run_started", kind="selfplay", run_dir=run_dir,
                      config=cfg["run_name"], games=num_games, sims=sims,
                      games_existing=already, games_to_generate=to_generate,
                      concurrency=args.concurrency,
                      max_batch_size=args.max_batch_size)
        run.info(f"run dir: {run_dir}")
        if to_generate == 0:
            run.info(
                f"already have {already} game(s) (target {num_games}); "
                "nothing to generate"
            )
            run.log_event("run_finished", kind="selfplay", games=0,
                          games_total=already)
            return 0
        if already:
            run.info(
                f"resuming: {already} game(s) already in {out_dir}; "
                f"generating {to_generate} more to reach {num_games}"
            )

        # Seed for reproducibility (Dirichlet noise, move sampling, and
        # the random-init weights all draw from these generators).
        np.random.seed(int(cfg["seed"]))
        torch.manual_seed(int(cfg["seed"]))

        arch = cfg["architecture"]
        model = build_architecture(arch["name"], arch["params"])
        checkpoint = args.checkpoint or cfg.get("base_checkpoint")
        if checkpoint:
            load_checkpoint(checkpoint, model, device=args.device)
            run.info(f"loaded checkpoint {checkpoint}")
        else:
            run.info("no checkpoint given: playing with a randomly initialised net")
        model.to(args.device)
        model.eval()
        torch.set_num_threads(common.torch_threads(2))
        torch.set_num_interop_threads(1)

        mcts = MCTS(
            model,
            device=args.device,
            c_puct=float(sp["c_puct"]),
            dirichlet_alpha=float(sp["dirichlet_alpha"]),
            dirichlet_epsilon=float(sp["dirichlet_epsilon"]),
        )
        run.info(f"generating {to_generate} games x {sims} sims -> {out_dir}")
        game_kwargs = dict(
            temperature_moves=int(sp["temperature_moves"]),
            temperature=float(sp["temperature"]),
            resign_threshold=sp["resign_threshold"],
            max_moves=int(sp["max_moves"]),
        )
        if args.concurrency > 1:
            run.info(
                f"batched mode: concurrency={args.concurrency}, "
                f"max_batch_size={args.max_batch_size}"
            )
            paths = selfplay.generate_games_batched(
                mcts,
                num_games=to_generate,
                num_simulations=sims,
                out_dir=out_dir,
                run_name=cfg["run_name"],
                concurrency=args.concurrency,
                max_batch_size=args.max_batch_size,
                on_event=lambda rec: run.log_event(**rec),
                live_path=os.path.join(run_dir, "current_game.json"),
                stop_flag=flag,
                checkpoint_every_games=int(sp["checkpoint_every_games"]),
                **game_kwargs,
            )
        else:
            paths = selfplay.generate_games(
                mcts,
                num_games=to_generate,
                num_simulations=sims,
                out_dir=out_dir,
                run_name=cfg["run_name"],
                on_event=lambda rec: run.log_event(**rec),
                live_path=os.path.join(run_dir, "current_game.json"),
                stop_flag=flag,
                checkpoint_every_games=int(sp["checkpoint_every_games"]),
                **game_kwargs,
            )
        if flag.requested:
            completed_total = already + len(paths)
            remaining = num_games - completed_total
            # Top-up: re-running the SAME command finishes the job.
            resume_cmd = (
                f"ml/.venv/bin/python ml/scripts/selfplay.py "
                f"--config {common.config_path(args.config)} "
                f"--run-dir {run_dir} --games {num_games}"
            )
            run.log_event(
                "run_interrupted", kind="selfplay",
                signal=flag.signal_name, games_completed=completed_total,
                games_remaining=remaining, resume_command=resume_cmd,
            )
            run.info(
                f"interrupted ({flag.signal_name}) — "
                f"{completed_total}/{num_games} games completed and saved; "
                "re-run the same command to top up the rest:"
            )
            run.info(f"  {resume_cmd}")
            return flag.exit_code()
        run.log_event("run_finished", kind="selfplay", games=len(paths))
        run.info(f"done. {already + len(paths)} shards in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

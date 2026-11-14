#!/usr/bin/env python3
"""Generate heuristic-vs-heuristic games as .npz shards (batch 4 H3, v11).

Plays games between two 1-ply scoring engines (default v1 vs the
frozen baseline v0) across worker processes and records them in the
self-play shard schema (planes/policy/value/moves/winner) so
``train.py`` consumes them unchanged. See
``gomoku_ml.heuristic_games`` and
``research/iterations/iter-11.md`` §1.1a.

Top-up semantics (as in selfplay.py): ``--games`` is the TOTAL
number of games wanted in the output directory; re-running the same
command generates only the games still missing. Duplicates are
removed after the merge (see ``deduplicate_shards``), so the final
unique count may be below the target — the stats JSON reports both.

Usage:
    python ml/scripts/generate_heuristic_games.py --games 4000 \\
        --out-dir ml/runs/b4-h3/data --run-name b4-h3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import heuristic_games, selfplay  # noqa: E402
from gomoku_ml.interrupt import InterruptFlag  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, required=True,
                        help="TOTAL games wanted in the output dir (top-up)")
    parser.add_argument("--out-dir", required=True,
                        help="shard output directory")
    parser.add_argument("--run-name", default="run")
    parser.add_argument("--version-a", type=int, default=1,
                        help="learner engine version (default 1 = v1)")
    parser.add_argument("--version-b", type=int, default=0,
                        help="opponent engine version (default 0 = baseline)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-moves", type=int, default=225)
    parser.add_argument("--sample-eps", type=float, default=0.0,
                        help="learner sampling: with this probability "
                        "version A plays a uniform random move among "
                        "its top --sample-topk moves instead of its "
                        "top move (default 0 = pure play)")
    parser.add_argument("--sample-topk", type=int, default=0,
                        help="top-k pool for learner sampling "
                        "(required when --sample-eps > 0)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="skip duplicate removal (default: dedup)")
    args = parser.parse_args()
    if args.workers < 1:
        print(f"--workers must be >= 1, got {args.workers}", file=sys.stderr)
        return 2
    if args.sample_eps > 0 and args.sample_topk < 1:
        print("--sample-topk must be >= 1 when --sample-eps > 0",
              file=sys.stderr)
        return 2
    if args.sample_eps < 0 or args.sample_eps > 1:
        print(f"--sample-eps must be in [0, 1], got {args.sample_eps}",
              file=sys.stderr)
        return 2

    flag = InterruptFlag().install()
    os.makedirs(args.out_dir, exist_ok=True)
    # Recover orphaned shards before the top-up computation (a no-op
    # when there are no leftovers).
    recovered = selfplay.recover_parallel_shards(args.out_dir, args.run_name)
    if recovered["merged"]:
        print(f"recovered {len(recovered['merged'])} orphaned shard(s)",
              file=sys.stderr, flush=True)
    already = selfplay.existing_game_count(args.out_dir, args.run_name)
    to_generate = max(0, args.games - already)
    print(f"target {args.games} games in {args.out_dir} "
          f"(already {already}, generating {to_generate})",
          file=sys.stderr, flush=True)
    if to_generate == 0:
        print("nothing to generate", file=sys.stderr)
        return 0
    try:
        stats = heuristic_games.generate_heuristic_games(
            out_dir=args.out_dir,
            run_name=args.run_name,
            num_games=to_generate,
            version_a=args.version_a,
            version_b=args.version_b,
            workers=args.workers,
            seed=args.seed,
            max_moves=args.max_moves,
            stop_flag=flag,
            dedup=not args.no_dedup,
            sample_eps=args.sample_eps,
            sample_topk=args.sample_topk,
        )
    except RuntimeError as e:
        print(f"heuristic game generation failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(stats, indent=2))
    if flag.requested:
        print(f"interrupted ({flag.signal_name}) — completed games are "
              "merged and saved; re-run the same command to top up",
              file=sys.stderr)
        return flag.exit_code()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Evaluate a checkpoint: ladder matches + tactical suite + verdict.

Ladder opponents: ``random``, ``heuristic`` (1-ply pattern player),
``mcts`` (heuristic PUCT search, no net), ``checkpoint`` (a previous
checkpoint, via --opponent-checkpoint / --previous). Every matchup
plays both colours in equal numbers (``--games`` is the TOTAL games
per opponent — use an even number for exact balance) and reports W/L/D
plus the win rate with a Wilson-score 95% confidence interval.

The tactical suite (``gomoku_ml/tactical_positions.json``) is run for
the checkpoint and for the heuristic baseline, unless --no-tactical.

With ``--run-dir``, results are written there:

* ``eval.json`` — per-opponent ladder results (the shape the training
  server and ``compare_runs.py`` read; kept backward compatible).
* ``eval_report.json`` + ``eval_report.md`` — the full report: ladder,
  ladder Elo estimate (relative to this ladder only), tactical suite,
  checkpoint-vs-checkpoint result, and the computed verdict.
* ``eval_history.jsonl`` — one line per eval, for tracking improvement.
* ``model_card.md`` — regenerated with the eval results.

Progress JSON events (``{"event": "eval_result", ...}``) go to stdout;
human-readable lines go to stderr.

Usage:
    python ml/scripts/evaluate.py --checkpoint ml/runs/<id>/checkpoints/final.pt
    python ml/scripts/evaluate.py --checkpoint ml/runs/<id>/checkpoints/final.pt --run-dir ml/runs/<id>
    python ml/scripts/evaluate.py --checkpoint a.pt --run-dir ml/runs/<id> --previous
    python ml/scripts/evaluate.py --checkpoint a.pt --opponent-checkpoint b.pt --opponents random,heuristic,mcts,checkpoint
    python ml/scripts/evaluate.py --checkpoint a.pt --ts-opponent advanced --games 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import torch  # noqa: E402
from gomoku_ml import evaluate, runs  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="checkpoint .pt to evaluate")
    parser.add_argument("--run-dir", default=None,
                        help="run directory to write eval.json + report into")
    parser.add_argument("--opponent-checkpoint", "--vs-checkpoint",
                        dest="opponent_checkpoint", default=None,
                        help="opponent checkpoint .pt (for the 'checkpoint' opponent)")
    parser.add_argument("--previous", action="store_true",
                        help="auto-pick the prior checkpoint in --run-dir as the opponent "
                             "(requires --run-dir; adds 'checkpoint' to --opponents)")
    parser.add_argument("--games", type=int, default=10,
                        help="TOTAL games per opponent, split evenly across colours (default: 10)")
    parser.add_argument("--sims", type=int, default=50,
                        help="MCTS sims per move for the evaluated model")
    parser.add_argument("--opponent-sims", type=int, default=50,
                        help="MCTS sims per move for the checkpoint opponent")
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="sims per move for the heuristic-MCTS opponent")
    parser.add_argument("--opponents", default="random,heuristic",
                        help="comma-separated: random,heuristic,mcts,checkpoint")
    parser.add_argument("--tactical", dest="tactical", action="store_true", default=True,
                        help="run the tactical suite (default: on)")
    parser.add_argument("--no-tactical", dest="tactical", action="store_false",
                        help="skip the tactical suite")
    parser.add_argument("--tactical-sims", type=int, default=50,
                        help="MCTS sims per tactical position")
    parser.add_argument("--ts-opponent", default=None, choices=["advanced", "expert"],
                        help="also play the TypeScript AI level (advanced|expert) via "
                             "scripts/play_match.mjs (optional cross-check)")
    parser.add_argument("--ts-time-ms", type=int, default=None,
                        help="time budget (ms) per move for the TS opponent (default: level default)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if args.opponent_checkpoint and not os.path.exists(args.opponent_checkpoint):
        print(f"opponent checkpoint not found: {args.opponent_checkpoint}", file=sys.stderr)
        return 2
    opponents = [o.strip() for o in args.opponents.split(",") if o.strip()]
    if args.previous:
        if not args.run_dir:
            parser.error("--previous requires --run-dir")
        previous = evaluate.find_previous_checkpoint(args.run_dir, args.checkpoint)
        if previous is None:
            print(
                f"no previous checkpoint found in {args.run_dir}/checkpoints "
                f"(need a checkpoint-<step>.pt, or best.pt/final.pt pair)",
                file=sys.stderr,
            )
            return 2
        args.opponent_checkpoint = previous
        if "checkpoint" not in opponents:
            opponents.append("checkpoint")
        print(f"previous checkpoint: {previous}", file=sys.stderr)
    if "checkpoint" in opponents and not args.opponent_checkpoint:
        parser.error(
            "--opponent-checkpoint (or --previous with --run-dir) is required "
            "when 'checkpoint' is in --opponents"
        )
    if args.games < 1:
        parser.error("--games must be >= 1")
    if args.games % 2 == 1:
        print(
            "note: --games is odd, so Black gets one extra game; use an even "
            "number for exactly equal colours",
            file=sys.stderr,
        )

    torch.set_num_threads(common.torch_threads(2))
    torch.set_num_interop_threads(1)
    run = runs.Run(args.run_dir) if args.run_dir else None
    try:
        def emit(rec: dict) -> None:
            if run is not None:
                run.log_event(**rec)
            else:
                print(json.dumps({**rec, "timestamp": runs.utc_now_iso()}), flush=True)

        def info(msg: str) -> None:
            if run is not None:
                run.info(msg)
            else:
                print(msg, file=sys.stderr, flush=True)

        info(
            f"evaluating {args.checkpoint} "
            f"({args.games} games/opponent, {args.sims} sims, "
            f"opponents: {','.join(opponents)})"
        )
        report = evaluate.evaluate_checkpoint(
            args.checkpoint,
            games=args.games,
            sims=args.sims,
            opponents=tuple(opponents),
            mcts_sims=args.mcts_sims,
            opponent_checkpoint=args.opponent_checkpoint,
            opponent_sims=args.opponent_sims,
            tactical=args.tactical,
            tactical_sims=args.tactical_sims,
            device=args.device,
            seed=args.seed,
            on_event=emit,
        )
        if args.ts_opponent:
            info(f"TS cross-check vs {args.ts_opponent} ({args.games} games)...")
            try:
                ts = evaluate.run_ts_match(
                    args.checkpoint,
                    level=args.ts_opponent,
                    games=args.games,
                    sims=args.sims,
                    ts_time_ms=args.ts_time_ms,
                )
                report["ts_match"] = ts
                info(
                    f"  vs TS {args.ts_opponent}: {ts['wins']}W-{ts['losses']}L-"
                    f"{ts['draws']}D (win rate {ts['win_rate']:.3f})"
                )
            except RuntimeError as e:
                info(f"  TS cross-check failed (not fatal): {e}")
                report["ts_match"] = {
                    "level": args.ts_opponent,
                    "error": str(e),
                    "note": "Cross-check unavailable; see error. Python ladder only.",
                }

        # Summary (stderr + a final JSON event).
        verdict = report["verdict"]
        elo = report["ladder_elo"]
        info(f"verdict: {verdict['label']}")
        info(f"  {verdict['explanation']}")
        if elo["estimate"] is not None:
            lo, hi = elo["interval"]
            info(f"ladder Elo: {elo['estimate']} (95% interval [{lo}, {hi}]) — {elo['caveat']}")
        if report["tactical"] is not None:
            tac = report["tactical"]
            base = report["tactical_baseline_heuristic"]
            info(
                f"tactical: checkpoint {tac['passed']}/{tac['total']} "
                f"({tac['pass_rate']:.0%}), heuristic baseline "
                f"{base['passed']}/{base['total']} ({base['pass_rate']:.0%})"
            )
        emit({
            "event": "eval_finished",
            "checkpoint": args.checkpoint,
            "verdict": verdict["tier"],
            "ladder_elo": elo["estimate"],
            "tactical_pass_rate": (report["tactical"] or {}).get("pass_rate"),
        })

        if run is not None:
            run.write_eval(report)
            json_path, md_path = run.write_eval_report(report)
            run.append_eval_history(report)
            card = runs.generate_model_card(args.run_dir)
            info(f"eval results -> {os.path.join(args.run_dir, 'eval.json')}")
            info(f"eval report -> {json_path} + {md_path}")
            info(f"model card -> {card}")
        else:
            print(json.dumps(report, indent=2))
        return 0
    finally:
        if run is not None:
            run.close()


if __name__ == "__main__":
    sys.exit(main())

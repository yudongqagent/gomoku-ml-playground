#!/usr/bin/env python3
"""Compare training runs side by side.

Reads run directories (explicit paths, or every directory under
ml/runs/ when no paths are given) and prints a comparison table:
architecture, self-play games, final/val loss, and eval win-rates vs
each baseline. Also writes the same data as JSON.

Usage:
    python ml/scripts/compare_runs.py
    python ml/scripts/compare_runs.py ml/runs/20261112-000000-smoke ml/runs/20261112-010000-mobilenet
    python ml/scripts/compare_runs.py --out /tmp/comparison.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import runs  # noqa: E402


def summarise_run(run_dir: str) -> dict:
    """Extract the comparable facts of one run directory."""
    run = runs.Run(run_dir)
    try:
        cfg = run.load_config() or {}
        metrics = run.load_metrics()
        eval_raw = run.load_eval() or {}
        # eval.json is either the enriched shape ({ladder, ...}) or a
        # legacy plain ladder dict.
        eval_results = eval_raw.get("ladder", eval_raw)
        arch = cfg.get("architecture", {})
        epochs = [m for m in metrics if m.get("event") == "epoch"]
        games = [m for m in metrics if m.get("event") == "selfplay_game"]
        final = epochs[-1] if epochs else None
        val_losses = [m["val_loss"] for m in epochs if m.get("val_loss") is not None]
        summary = {
            "run": os.path.basename(run_dir.rstrip("/")),
            "run_dir": os.path.abspath(run_dir),
            "architecture": arch.get("name"),
            "arch_params": arch.get("params"),
            "games_generated": len(games),
            "epochs": len(epochs),
            "final_loss": final["loss"] if final else None,
            "final_policy_loss": final["policy_loss"] if final else None,
            "final_value_loss": final["value_loss"] if final else None,
            "best_val_loss": min(val_losses) if val_losses else None,
            "samples": final["num_samples"] if final else None,
            "eval": eval_results,
        }
        return summary
    finally:
        run.close()


def _fmt(value, fmt="{:.4f}"):
    if value is None:
        return "-"
    if isinstance(value, float):
        return fmt.format(value)
    return str(value)


def _fmt_eval(eval_results: dict, opponent: str) -> str:
    r = eval_results.get(opponent)
    if not r:
        return "-"
    return f"{r['wins']}W-{r['losses']}L-{r['draws']}D"


def print_table(summaries: list[dict]) -> None:
    headers = ["run", "arch", "games", "epochs", "final loss", "best val", "vs random", "vs heuristic", "vs checkpoint"]
    rows = []
    for s in summaries:
        arch = s["architecture"] or "-"
        params = s["arch_params"] or {}
        arch_label = f"{arch} ({params.get('channels')}ch/{params.get('blocks')}b)" if params else arch
        rows.append([
            s["run"][:40],
            arch_label,
            str(s["games_generated"]),
            str(s["epochs"]),
            _fmt(s["final_loss"]),
            _fmt(s["best_val_loss"]),
            _fmt_eval(s["eval"], "random"),
            _fmt_eval(s["eval"], "heuristic"),
            _fmt_eval(s["eval"], "checkpoint"),
        ])
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*", help="run directories to compare (default: all under ml/runs/)")
    parser.add_argument("--out", default=None,
                        help="where to write comparison.json (default: ml/runs/comparison.json)")
    args = parser.parse_args()

    if args.run_dirs:
        dirs = args.run_dirs
    else:
        runs_root = os.path.join(common.ML_DIR, "runs")
        if not os.path.isdir(runs_root):
            print(f"no runs found: {runs_root} does not exist", file=sys.stderr)
            return 2
        dirs = sorted(
            os.path.join(runs_root, d) for d in os.listdir(runs_root)
            if os.path.isdir(os.path.join(runs_root, d))
        )
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        print(f"run dir(s) not found: {missing}", file=sys.stderr)
        return 2
    if not dirs:
        print("no runs to compare", file=sys.stderr)
        return 2

    summaries = [summarise_run(d) for d in dirs]
    print_table(summaries)
    out = args.out or os.path.join(common.ML_DIR, "runs", "comparison.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, indent=2)
        f.write("\n")
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

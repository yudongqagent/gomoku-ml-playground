"""Run tracking: run directories, JSON event logging, model cards.

Every training / self-play run writes to a run directory::

    ml/runs/<run-id>/
      config.yaml          exact snapshot of the resolved config
      metrics.jsonl        one JSON object per logged event
      run.log              human-readable log lines
      eval.json            eval ladder results (when an eval is run)
      model_card.md        auto-generated summary (arch, config, metrics, eval)
      data/                self-play .npz shards
      checkpoints/
        checkpoint-<step>.pt
        best.pt            lowest validation loss so far (copy)
        final.pt           end-of-training checkpoint (copy)

``<run-id>`` is ``YYYYMMDD-HHMMSS-<short-name>`` (UTC). Scripts accept
``--run-dir`` to use a specific directory (the training server passes
one); otherwise a fresh timestamped directory is created.

Logging contract (for the training server, which invokes the scripts
as subprocesses):

* stdout carries JSON event lines only — one JSON object per line, each
  with an ``"event"`` key. The server can parse every stdout line as
  JSON.
* Human-readable progress lines go to stderr (and to ``run.log``).
* The same JSON events are appended to ``metrics.jsonl``.

Events emitted:
  run_started, selfplay_game, train_step, epoch, checkpoint,
  eval_result, run_finished, error.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone

from .config import dump_config


def new_run_id(short_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in short_name)
    return f"{stamp}-{safe or 'run'}"


def resolve_run_dir(ml_dir: str, run_dir_arg: str | None, short_name: str) -> str:
    """Return the run directory to use: ``run_dir_arg`` if given,
    otherwise a fresh timestamped directory under ``ml/runs/``."""
    if run_dir_arg:
        return os.path.abspath(run_dir_arg)
    return os.path.join(ml_dir, "runs", new_run_id(short_name))


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Run:
    """A run directory plus its loggers. Create via :func:`open_run`."""

    def __init__(self, run_dir: str) -> None:
        self.dir = os.path.abspath(run_dir)
        self.data_dir = os.path.join(self.dir, "data")
        self.checkpoint_dir = os.path.join(self.dir, "checkpoints")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self._log_path = os.path.join(self.dir, "run.log")
        self._metrics_file = open(self._metrics_path, "a", encoding="utf-8")
        self._log_file = open(self._log_path, "a", encoding="utf-8")

    # -- logging ------------------------------------------------------------

    def log_event(self, event: str, **fields) -> dict:
        """Emit one JSON event: stdout (parseable) + metrics.jsonl."""
        record = {"event": event, "timestamp": utc_now_iso(), **fields}
        line = json.dumps(record)
        print(line, flush=True)
        self._metrics_file.write(line + "\n")
        self._metrics_file.flush()
        return record

    def info(self, msg: str) -> None:
        """Human-readable progress line: stderr + run.log."""
        print(msg, file=sys.stderr, flush=True)
        self._log_file.write(f"[{utc_now_iso()}] {msg}\n")
        self._log_file.flush()

    def error(self, msg: str) -> None:
        self.log_event("error", message=msg)
        self.info(f"ERROR: {msg}")

    # -- artifacts ----------------------------------------------------------

    def save_config(self, cfg: dict) -> str:
        path = os.path.join(self.dir, "config.yaml")
        with open(path, "w", encoding="utf-8") as f:
            f.write(dump_config(cfg))
        return path

    def load_config(self) -> dict | None:
        import yaml

        path = os.path.join(self.dir, "config.yaml")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_metrics(self) -> list[dict]:
        records: list[dict] = []
        if os.path.exists(self._metrics_path):
            with open(self._metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def load_eval(self) -> dict | None:
        path = os.path.join(self.dir, "eval.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_eval(self, results: dict) -> str:
        """Write ``eval.json``. Accepts either:

        * a full eval report (has a ``"ladder"`` key) — written in the
          enriched shape the training server's ``/api/jobs/{id}/eval``
          endpoint reads: ``ladder`` / ``elo_estimate`` / ``tactical``
          / ``verdict`` / ``generated_at`` (+ checkpoint, opponent
          checkpoint, TS cross-check when present); or
        * a plain per-opponent ladder dict (legacy shape) — written
          as-is, so older callers and existing tests keep working.

        Readers must handle both shapes (``eval.get("ladder", eval)``).
        """
        if "ladder" in results and isinstance(results.get("ladder"), dict):
            payload: dict = {
                "checkpoint": results.get("checkpoint"),
                "generated_at": results.get("evaluated_at"),
                "ladder": results["ladder"],
                "elo_estimate": results.get("ladder_elo"),
                "tactical": results.get("tactical"),
                "tactical_baseline_heuristic": results.get(
                    "tactical_baseline_heuristic"
                ),
                "verdict": results.get("verdict"),
                "opponent_checkpoint": results.get("opponent_checkpoint"),
                "ts_match": results.get("ts_match"),
            }
        else:
            payload = results
        path = os.path.join(self.dir, "eval.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        return path

    def load_eval_report(self) -> dict | None:
        path = os.path.join(self.dir, "eval_report.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_eval_report(self, report: dict) -> tuple[str, str]:
        """Write ``eval_report.json`` + ``eval_report.md`` (full report:
        ladder, ladder Elo, tactical suite, verdict)."""
        from .evaluate import write_eval_report as _write

        return _write(self.dir, report)

    def append_eval_history(self, report: dict) -> str:
        """Append one line per eval to ``eval_history.jsonl`` (timestamp,
        checkpoint, per-opponent W/L/D, ladder Elo, tactical pass rate,
        verdict tier) so a run's improvement over time is visible."""
        path = os.path.join(self.dir, "eval_history.jsonl")
        ladder = report.get("ladder", {})
        entry = {
            "evaluated_at": report.get("evaluated_at"),
            "checkpoint": report.get("checkpoint"),
            "ladder": {
                name: {
                    "wins": r.get("wins"),
                    "losses": r.get("losses"),
                    "draws": r.get("draws"),
                    "win_rate": r.get("win_rate"),
                }
                for name, r in ladder.items()
            },
            "ladder_elo": (report.get("ladder_elo") or {}).get("estimate"),
            "ladder_elo_interval": (report.get("ladder_elo") or {}).get("interval"),
            "tactical_pass_rate": (report.get("tactical") or {}).get("pass_rate"),
            "verdict": (report.get("verdict") or {}).get("tier"),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return path

    def close(self) -> None:
        self._metrics_file.close()
        self._log_file.close()

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def generate_model_card(run_dir: str) -> str:
    """(Re)generate ``model_card.md`` for a run directory from its
    config.yaml, metrics.jsonl and eval.json. Safe to call repeatedly;
    missing pieces are reported as 'not recorded' rather than failing."""
    run = Run(run_dir)
    try:
        cfg = run.load_config() or {}
        metrics = run.load_metrics()
        eval_results = run.load_eval()
        eval_report = run.load_eval_report()
        arch = cfg.get("architecture", {})
        tr = cfg.get("train", {})
        sp = cfg.get("selfplay", {})

        epochs = [m for m in metrics if m.get("event") == "epoch"]
        games = [m for m in metrics if m.get("event") == "selfplay_game"]
        final = epochs[-1] if epochs else None
        best_val = None
        for m in epochs:
            v = m.get("val_loss")
            if v is not None and (best_val is None or v < best_val):
                best_val = v

        base = cfg.get("base_checkpoint") or "none (from scratch)"
        resume = cfg.get("resume") or "none (fresh run)"
        lines = [
            f"# Model card — run `{os.path.basename(run_dir.rstrip('/'))}`",
            "",
            "> Auto-generated from the run directory. Strength claims: "
            "none beyond what the eval ladder below measures. Beating the "
            "random and heuristic baselines says almost nothing about "
            "absolute strength — a serious claim needs matches against "
            "strong engines or rated human play.",
            "",
            "## Architecture",
            "",
            f"- Name: `{arch.get('name', 'unknown')}`",
            f"- Params: `{json.dumps(arch.get('params', {}))}`",
            "",
            "## Training configuration",
            "",
            f"- Seed: `{cfg.get('seed')}`",
            f"- Base checkpoint: `{base}`",
            f"- Resume: `{resume}`",
            f"- Epochs: `{tr.get('epochs')}`, batch size: `{tr.get('batch_size')}`, "
            f"lr: `{tr.get('lr')}` ({tr.get('lr_schedule')}), warmup: `{tr.get('warmup_steps')}`",
            f"- Loss weights: policy `{tr.get('policy_weight')}`, value `{tr.get('value_weight')}`, "
            f"weight decay `{tr.get('weight_decay')}`",
            f"- Augmentation (board symmetries): `{tr.get('augmentation')}`",
            "",
            "## Self-play",
            "",
            f"- Games generated in this run: `{len(games)}` "
            f"(config: `{sp.get('games')}` per run)",
            f"- MCTS sims/move: `{sp.get('sims_per_move')}`, c_puct: `{sp.get('c_puct')}`, "
            f"temperature: `{sp.get('temperature')}` for `{sp.get('temperature_moves')}` moves, "
            f"Dirichlet alpha `{sp.get('dirichlet_alpha')}` / eps `{sp.get('dirichlet_epsilon')}`",
            "",
            "## Metrics",
            "",
        ]
        if final:
            lines += [
                f"- Final epoch ({final.get('epoch')}): loss `{final.get('loss'):.4f}`, "
                f"policy `{final.get('policy_loss'):.4f}`, value `{final.get('value_loss'):.4f}`"
                + (f", val `{final.get('val_loss'):.4f}`" if final.get("val_loss") is not None else ""),
                f"- Best val loss: `{best_val:.4f}`" if best_val is not None else "- Best val loss: not recorded",
                f"- Samples: `{final.get('num_samples')}`",
            ]
        else:
            lines.append("- No training epochs recorded yet.")
        # eval.json is either the enriched shape ({ladder, elo_estimate,
        # tactical, verdict, ...}) or a legacy plain ladder dict.
        ladder = (eval_results or {}).get("ladder", eval_results or {})
        lines += ["", "## Eval ladder", ""]
        if ladder:
            for opp, r in ladder.items():
                elo = r.get("elo_diff_vs_opponent")
                wr = r.get("win_rate")
                ci = r.get("win_rate_ci_95")
                lines.append(
                    f"- vs {opp}: {r.get('wins')}W-{r.get('losses')}L-{r.get('draws')}D"
                    + (f" (win rate {wr:.3f}, 95% CI [{ci[0]:.2f}, {ci[1]:.2f}])" if wr is not None and ci else "")
                    + (f" (rough Elo diff {elo:+.0f})" if elo is not None else " (Elo diff n/a — one side won all decisive games)")
                )
        else:
            lines.append("- No eval run recorded for this run yet.")
        # Verdict / Elo / tactical: prefer eval.json's top-level keys,
        # fall back to eval_report.json (same data, older runs).
        verdict = (eval_results or {}).get("verdict") or (eval_report or {}).get("verdict") or {}
        elo = (eval_results or {}).get("elo_estimate") or (eval_report or {}).get("ladder_elo") or {}
        tac = (eval_results or {}).get("tactical")
        if tac is None and eval_report is not None:
            tac = eval_report.get("tactical")
        if verdict or elo.get("estimate") is not None or tac is not None:
            lines += ["", "## Eval verdict", ""]
            if verdict:
                lines.append(f"- Verdict: **{verdict.get('label', 'n/a')}**")
                if verdict.get("explanation"):
                    lines.append(f"- {verdict['explanation']}")
            if elo.get("estimate") is not None:
                lo, hi = elo.get("interval") or (None, None)
                lines.append(
                    f"- Ladder Elo (relative to this ladder only): "
                    f"**{elo['estimate']}**"
                    + (f" (95% interval [{lo}, {hi}])" if lo is not None and hi is not None else "")
                )
            if tac is not None:
                lines.append(
                    f"- Tactical suite: {tac['passed']}/{tac['total']} "
                    f"({tac['pass_rate']:.0%})"
                )
            lines.append("- Full report: `eval_report.md` / `eval_report.json`.")
        lines += [
            "",
            "## Files",
            "",
            "- `config.yaml` — exact resolved config for this run",
            "- `metrics.jsonl` — per-step / per-game metrics",
            "- `eval.json` — eval ladder results (when run)",
            "- `eval_report.json` / `eval_report.md` — full eval report (ladder Elo, tactical suite, verdict)",
            "- `eval_history.jsonl` — one line per eval, for tracking improvement over time",
            "- `checkpoints/final.pt` — end-of-training checkpoint",
            "- `checkpoints/best.pt` — lowest validation-loss checkpoint",
            "",
        ]
        path = os.path.join(run_dir, "model_card.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path
    finally:
        run.close()


def copy_best_or_final(run_dir: str, src: str, name: str) -> str:
    """Copy a checkpoint file into the run's checkpoint dir as ``name``
    (``best.pt`` / ``final.pt``). Returns the destination path."""
    dest = os.path.join(run_dir, "checkpoints", name)
    shutil.copyfile(src, dest)
    return dest

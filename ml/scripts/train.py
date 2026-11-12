#!/usr/bin/env python3
"""Train the policy+value network on self-play shards.

Writes to a run directory (see gomoku_ml.runs): checkpoints in
``<run-dir>/checkpoints/`` (``final.pt``, ``best.pt`` by val loss,
``checkpoint-<step>.pt`` when configured), the resolved config snapshot
in ``<run-dir>/config.yaml``, progress JSON events on stdout (one JSON
object per line, each with an "event" key) and in ``metrics.jsonl``, and
an auto-generated ``model_card.md`` at the end.

Data: ``.npz`` shards. Default source is ``<run-dir>/data/`` (the
self-play output of the same run); pass ``--data`` to train on another
run's shards. Resumable: ``--resume`` (or ``resume:`` in the config)
restores weights + optimiser + step and runs ``epochs`` ADDITIONAL
epochs from the checkpoint (the cosine LR schedule continues exactly
across the resume); ``base_checkpoint:`` loads weights only. After an
interrupt at epoch E of a T-epoch plan, resume with
``--epochs <T-E>`` to finish the plan exactly — the interrupt message
prints that command.

Auto-eval: when the config sets ``eval.auto: true``, the eval harness
runs at the END of training (against ``best.pt`` when it exists, else
``final.pt``) and appends to ``eval.json`` / ``eval_report.*`` /
``eval_history.jsonl``. Mid-training eval is deliberately not done: on
CPU it would roughly double the run time for little signal, and the
end-of-training eval answers the question that matters ("what level did
this run reach?"). Re-run ``evaluate.py`` any time for a fresh eval.

Graceful interrupt: SIGINT/SIGTERM finish the current batch, write
``checkpoints/interrupted.pt`` (weights + optimiser + step), and exit
with code 130 (SIGINT) / 143 (SIGTERM) plus the exact resume command.

Usage:
    python ml/scripts/train.py --config ml/configs/smoke.yaml
    python ml/scripts/train.py --config smoke --smoke
    python ml/scripts/train.py --config smoke --run-dir ml/runs/20261112-000000-smoke
    python ml/scripts/train.py --config full.yaml --data ml/runs/<id>/data --resume ml/runs/<id>/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import torch  # noqa: E402
from gomoku_ml import evaluate, runs  # noqa: E402
from gomoku_ml.architectures import build_architecture  # noqa: E402
from gomoku_ml.config import ConfigError, load_config  # noqa: E402
from gomoku_ml.interrupt import InterruptFlag  # noqa: E402
from gomoku_ml.train import load_checkpoint, load_shards, save_checkpoint, train  # noqa: E402


def _auto_eval(run, run_dir: str, cfg: dict, device: str) -> None:
    """Run the eval harness at end of training (eval.auto: true) and
    record the results in the run dir. Best-effort: an eval failure is
    logged, never fatal to the training run."""
    ev = cfg["eval"]
    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    final_path = os.path.join(run_dir, "checkpoints", "final.pt")
    target = best_path if os.path.exists(best_path) else final_path
    if not os.path.exists(target):
        run.info("auto-eval skipped: no best.pt / final.pt found")
        return
    opponents = list(ev["opponents"])
    opponent_ckpt = None
    if "checkpoint" in opponents:
        opponent_ckpt = evaluate.find_previous_checkpoint(run_dir, target)
        if opponent_ckpt is None:
            run.info("auto-eval: no previous checkpoint; dropping 'checkpoint' opponent")
            opponents = [o for o in opponents if o != "checkpoint"]
    run.info(f"auto-eval: evaluating {target} ({','.join(opponents)})")
    try:
        report = evaluate.evaluate_checkpoint(
            target,
            games=int(ev["games"]),
            sims=int(ev["sims"]),
            opponents=tuple(opponents),
            mcts_sims=int(ev["mcts_sims"]),
            opponent_checkpoint=opponent_ckpt,
            opponent_sims=int(ev["sims"]),
            tactical=bool(ev["tactical"]),
            tactical_sims=int(ev["tactical_sims"]),
            device=device,
            seed=int(cfg["seed"]),
            on_event=lambda rec: run.log_event(**rec),
        )
    except Exception as e:  # noqa: BLE001 — eval must not fail the run
        run.error(f"auto-eval failed (training results are unaffected): {e}")
        return
    run.write_eval(report)
    run.write_eval_report(report)
    run.append_eval_history(report)
    run.info(f"auto-eval verdict: {report['verdict']['label']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="smoke", help="config name or path")
    parser.add_argument("--smoke", action="store_true", help="force smoke-scale settings (3 epochs, batch 32)")
    parser.add_argument("--run-dir", default=None,
                        help="run directory (default: fresh ml/runs/<timestamp>-<name>/)")
    parser.add_argument("--data", nargs="+", default=None, help="data dir(s) with .npz shards")
    parser.add_argument("--resume", default=None, help="checkpoint .pt to resume from (overrides config)")
    parser.add_argument("--out", default=None, help="extra copy of the final checkpoint at this path")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs (on resume: ADDITIONAL epochs from the checkpoint)")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    args = parser.parse_args()

    try:
        cfg = load_config(common.config_path(args.config))
    except (ConfigError, FileNotFoundError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    tr = cfg["train"]
    if args.smoke:
        tr = {**tr, "epochs": 3, "batch_size": 32}
        cfg["train"] = tr
    epochs = args.epochs if args.epochs is not None else int(tr["epochs"])
    run_dir = runs.resolve_run_dir(common.ML_DIR, args.run_dir, cfg["run_name"])
    flag = InterruptFlag().install()

    with runs.Run(run_dir) as run:
        run.save_config(cfg)
        data_dirs = args.data
        if data_dirs is None:
            run_data = os.path.join(run_dir, "data")
            legacy_data = os.path.join(common.ML_DIR, "runs", cfg["run_name"], "data")
            data_dirs = [run_data if os.path.isdir(run_data) else legacy_data]
        try:
            planes, policy, values = load_shards(data_dirs)
        except FileNotFoundError as e:
            run.error(str(e))
            return 2
        run.info(f"loaded {len(planes)} samples from {data_dirs}")

        arch = cfg["architecture"]
        model = build_architecture(arch["name"], arch["params"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"])
        )
        start_step = start_epoch = 0
        resume_path = args.resume or cfg.get("resume")
        base_path = cfg.get("base_checkpoint")
        if resume_path:
            payload = load_checkpoint(resume_path, model, optimizer, device=args.device)
            start_step = int(payload.get("step", 0))
            start_epoch = int(payload.get("epoch", 0))
            run.info(f"resumed from {resume_path} (step {start_step}, epoch {start_epoch})")
        elif base_path:
            load_checkpoint(base_path, model, device=args.device)
            run.info(f"loaded base checkpoint weights from {base_path}")

        # Resume runs `epochs` ADDITIONAL epochs from the checkpoint
        # (this is the long-standing behaviour the training server
        # relies on when it passes --resume with a base checkpoint).
        # To finish an interrupted T-epoch plan exactly, resume with
        # --epochs <T - completed>.
        torch.set_num_threads(
            common.torch_threads(min(4, os.cpu_count() or 1))
        )
        run.log_event("run_started", kind="train", run_dir=run_dir,
                      config=cfg["run_name"], epochs=epochs,
                      start_epoch=start_epoch, samples=len(planes))
        summary = train(
            model,
            planes,
            policy,
            values,
            epochs=epochs,
            batch_size=int(tr["batch_size"]),
            lr=float(tr["lr"]),
            lr_schedule=tr["lr_schedule"],
            lr_step_size=int(tr["lr_step_size"]),
            lr_gamma=float(tr["lr_gamma"]),
            weight_decay=float(tr["weight_decay"]),
            policy_weight=float(tr["policy_weight"]),
            value_weight=float(tr["value_weight"]),
            warmup_steps=int(tr["warmup_steps"]),
            augmentation=bool(tr["augmentation"]),
            val_fraction=float(tr["val_fraction"]),
            device=args.device,
            checkpoint_dir=run.checkpoint_dir,
            checkpoint_every_epochs=int(tr["checkpoint_every_epochs"]),
            optimizer=optimizer,
            start_step=start_step,
            start_epoch=start_epoch,
            seed=int(cfg["seed"]),
            on_event=lambda rec: run.log_event(**rec),
            stop_flag=flag,
            schedule_epochs=start_epoch + epochs,
        )
        if summary["interrupted"]:
            interrupted_path = summary["interrupted_checkpoint"]
            data_args = (
                " --data " + " ".join(data_dirs) if args.data else ""
            )
            # The plan was `epochs` more from `start_epoch`; the
            # interrupt completed len(history) of them. Suggest the
            # exact --epochs that finishes the original plan.
            completed = start_epoch + len(summary["history"])
            planned_end = start_epoch + epochs
            remaining = planned_end - completed
            epochs_arg = (
                f" --epochs {remaining}" if remaining > 0 else ""
            )
            resume_cmd = (
                f"ml/.venv/bin/python ml/scripts/train.py "
                f"--config {common.config_path(args.config)} "
                f"--run-dir {run_dir} --resume {interrupted_path}"
                f"{epochs_arg}{data_args}"
            )
            run.log_event(
                "run_interrupted", kind="train",
                signal=flag.signal_name,
                checkpoint=interrupted_path,
                step=summary["total_steps"],
                resume_command=resume_cmd,
            )
            run.info(
                f"interrupted ({flag.signal_name}) — resumable from "
                f"{interrupted_path}"
            )
            run.info(f"  {resume_cmd}")
            return flag.exit_code()
        final_path = os.path.join(run.checkpoint_dir, "final.pt")
        save_checkpoint(
            final_path, model, summary["optimizer"], summary["total_steps"],
            start_epoch + epochs,
            extra={"config": cfg, "history": summary["history"]},
        )
        if args.out:
            save_checkpoint(
                args.out, model, summary["optimizer"], summary["total_steps"],
                start_epoch + epochs,
                extra={"config": cfg, "history": summary["history"]},
            )
        final_loss = summary["history"][-1]["loss"] if summary["history"] else None
        if cfg["eval"].get("auto"):
            _auto_eval(run, run_dir, cfg, args.device)
        card = runs.generate_model_card(run_dir)
        run.log_event(
            "run_finished", kind="train",
            final_checkpoint=final_path,
            best_checkpoint=summary["best_checkpoint"],
            best_val_loss=summary["best_val_loss"],
            final_loss=final_loss,
        )
        run.info(f"saved final checkpoint -> {final_path}")
        run.info(f"model card -> {card}")
        if final_loss is not None:
            run.info(f"final epoch loss: {final_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

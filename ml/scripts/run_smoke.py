#!/usr/bin/env python3
"""End-to-end smoke run: self-play -> train -> export ONNX -> manifest.

Runs the real CLI scripts in sequence (so the smoke run exercises the
same code paths a real training run uses), all sharing one run directory
under ml/runs/, then writes the model file and manifest the web app
loads at runtime:

    public/models/gomoku-smoke-v1.onnx
    public/models/models.json

The smoke model is a DEMO NET (tiny net, a handful of self-play games)
and is labelled as such in models.json. Do not present it as strong.

This is also the reference command for the training server to launch a
smoke job end-to-end (from the repo root, with the venv Python):

    ml/.venv/bin/python ml/scripts/run_smoke.py

Usage:
    python ml/scripts/run_smoke.py
    python ml/scripts/run_smoke.py --games 4 --epochs 2   # even smaller
    python ml/scripts/run_smoke.py --skip-eval             # skip the quick eval
    python ml/scripts/run_smoke.py --run-dir ml/runs/my-smoke-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import runs  # noqa: E402
from gomoku_ml.config import load_config  # noqa: E402

MODEL_ID = "gomoku-smoke-v1"
MODEL_FILE = "gomoku-smoke-v1.onnx"


def run(cmd: list[str], step: str) -> None:
    print(f"\n=== {step} ===", file=sys.stderr, flush=True)
    print(f"$ {' '.join(cmd)}", file=sys.stderr, flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"FAILED: {step} (exit {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=None, help="override self-play games")
    parser.add_argument("--epochs", type=int, default=None, help="override training epochs")
    parser.add_argument("--skip-eval", action="store_true", help="skip the quick eval ladder")
    parser.add_argument("--run-dir", default=None, help="run directory (default: fresh ml/runs/<timestamp>-smoke/)")
    args = parser.parse_args()

    py = sys.executable
    scripts = common.SCRIPTS_DIR
    cfg_path = common.config_path("smoke")
    config = load_config(cfg_path)
    games = args.games if args.games is not None else int(config["selfplay"]["games"])
    run_dir = runs.resolve_run_dir(common.ML_DIR, args.run_dir, config["run_name"])

    checkpoint = os.path.join(run_dir, "checkpoints", "final.pt")
    onnx_path = os.path.join(common.REPO_ROOT, "public", "models", MODEL_FILE)
    manifest_path = os.path.join(common.REPO_ROOT, "public", "models", "models.json")

    selfplay_cmd = [py, os.path.join(scripts, "selfplay.py"),
                    "--config", cfg_path, "--run-dir", run_dir]
    if args.games is not None:
        selfplay_cmd += ["--games", str(args.games)]
    run(selfplay_cmd, "1/4 self-play")

    train_cmd = [py, os.path.join(scripts, "train.py"),
                 "--config", cfg_path, "--run-dir", run_dir, "--smoke"]
    if args.epochs is not None:
        train_cmd += ["--epochs", str(args.epochs)]
    run(train_cmd, "2/4 train")

    run(
        [py, os.path.join(scripts, "export_onnx.py"),
         "--checkpoint", checkpoint, "--out", onnx_path],
        "3/4 export ONNX + verify",
    )

    if not args.skip_eval:
        ev = config["eval"]
        run(
            [py, os.path.join(scripts, "evaluate.py"),
             "--checkpoint", checkpoint, "--run-dir", run_dir,
             "--games", str(ev["games"]), "--sims", str(ev["sims"]),
             "--opponents", ",".join(ev["opponents"])],
            "4/4 quick eval ladder",
        )

    size_kib = os.path.getsize(onnx_path) / 1024
    manifest = {
        "models": [
            {
                "id": MODEL_ID,
                "name": "Smoke demo net (tiny)",
                "url": f"/models/{MODEL_FILE}",
                "description": (
                    "Demo net from the smoke pipeline: 16 channels, 2 residual "
                    "blocks, trained on a handful of self-play games. Included "
                    "so the app's model loading and on-device inference can be "
                    "tested end to end."
                ),
                "trainedGames": games,
                "notes": (
                    "DEMO NET — not strong. Trained on only a handful of "
                    "self-play games with a tiny network; it plays poorly and "
                    "must not be presented as a strong opponent. Real strength "
                    "needs AlphaZero-scale training (see ml/README.md)."
                ),
            }
        ]
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nrun dir: {run_dir}", file=sys.stderr)
    print(f"wrote manifest -> {manifest_path}", file=sys.stderr)
    print(f"model: {onnx_path} ({size_kib:.1f} KiB)", file=sys.stderr)
    print(json.dumps({"event": "smoke_finished", "run_dir": run_dir,
                      "model": onnx_path, "size_kib": round(size_kib, 1)}))
    print("\nSmoke run complete. The model is a DEMO NET — see models.json notes.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

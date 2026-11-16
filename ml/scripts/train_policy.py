#!/usr/bin/env python3
"""Train v26's joint policy+value net on v18 self-play data
(ML batch 1, H3).

Pre-registered config (research/iterations/iter-26.md §1.2): ResNet
16ch/2 blocks (same size as v8's value net), AdamW lr 1e-3 cosine,
batch 64, weight decay 1e-4, dihedral augmentation on, 5% val split,
best checkpoint by val loss. Epochs are set so total steps ≈ 6,500
(``epochs = round(6500 * 64 / n_train)``) — the run length scales
with the dataset instead of being fixed.

Usage:
    python ml/scripts/train_policy.py \\
        --data ml/runs/ml1-h3/data-a ml/runs/ml1-h3/data-b \\
        --out ml/runs/ml1-h3/policy-value-net.pt
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import torch  # noqa: E402

from gomoku_ml import train  # noqa: E402
from gomoku_ml.architectures import build_architecture  # noqa: E402

TARGET_STEPS = 6500
BATCH_SIZE = 64
VAL_FRACTION = 0.05
SEED = 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True,
                        help=".npz shard directories (recursive)")
    parser.add_argument("--out", required=True,
                        help="destination path for the best checkpoint")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the pre-registered step rule "
                        "(on resume: ADDITIONAL epochs from the checkpoint)")
    parser.add_argument("--resume", default=None,
                        help="checkpoint .pt to resume from (weights + "
                        "optimiser; the cosine schedule still spans the "
                        "full run via schedule_epochs)")
    args = parser.parse_args()

    # Single-threaded torch: measured 0.57 s/step vs 2.73 s/step
    # with 2 threads under this VM's contention (load ~11 from the
    # sibling workstreams' workers) — the OpenMP threads spin and
    # starve the actual work (see ml/scripts/common.py
    # torch_threads() for the same observation on self-play).
    # At 1 thread the ~6,500-step run takes ~1h wall.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    planes, policy, value = train.load_shards(args.data)
    n = len(planes)
    n_train = n - int(round(n * VAL_FRACTION))
    epochs = args.epochs
    if epochs is None:
        epochs = max(1, round(TARGET_STEPS * BATCH_SIZE / n_train))
    print(f"data: {n} positions ({n_train} train, "
          f"{n - n_train} val); epochs={epochs} "
          f"(~{epochs * ((n_train + BATCH_SIZE - 1) // BATCH_SIZE)} steps)",
          file=sys.stderr, flush=True)

    model = build_architecture("resnet", {"channels": 16, "blocks": 2})
    print(f"model: resnet 16ch/2blk, {model.num_parameters()} params",
          file=sys.stderr, flush=True)
    ckpt_dir = os.path.dirname(os.path.abspath(args.out))
    optimizer = None
    start_step = start_epoch = 0
    if args.resume:
        import torch as _torch

        optimizer = _torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-4)
        payload = train.load_checkpoint(
            args.resume, model, optimizer, device="cpu")
        start_step = int(payload.get("step", 0))
        start_epoch = int(payload.get("epoch", 0))
        print(f"resumed from {args.resume} (step {start_step}, "
              f"epoch {start_epoch})", file=sys.stderr, flush=True)
    summary = train.train(
        model, planes, policy, value,
        epochs=epochs, batch_size=BATCH_SIZE, lr=1e-3,
        lr_schedule="cosine", weight_decay=1e-4,
        policy_weight=1.0, value_weight=1.0,
        augmentation=True, val_fraction=VAL_FRACTION,
        device="cpu", checkpoint_dir=ckpt_dir,
        checkpoint_every_epochs=5,
        optimizer=optimizer, start_step=start_step,
        start_epoch=start_epoch, schedule_epochs=start_epoch + epochs,
        seed=SEED,
    )
    print(f"best val loss: {summary['best_val_loss']:.4f}",
          file=sys.stderr, flush=True)
    best = summary["best_checkpoint"]
    if best is None:
        print("no best checkpoint written (no val set?)", file=sys.stderr)
        return 1
    if os.path.abspath(best) != os.path.abspath(args.out):
        shutil.copyfile(best, args.out)
    print(f"checkpoint -> {args.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

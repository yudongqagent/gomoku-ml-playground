#!/usr/bin/env python3
"""Pearson r between a value net's predictions and shard labels (v55).

Cross-depth sign agreement is parity-confounded (iter-44 §0: the +0.12
3-ply parity shift alone drives it to ~0.49), so Pearson r
(shift-invariant) is the primary cross-depth measure. This script loads
.npz shards from a holdout dir, runs the checkpoint's value head over
every position (full planes), and reports Pearson r + sign agreement +
MSE, pooled and per side to move (plane 3 = side-to-move plane, all
1.0 iff Black to move — the fixed side reading, 038232c).

Usage:
    python ml/scripts/holdout_corr.py --data ml/runs/ml6-h2/data/v7-holdout \\
        --checkpoint ml/runs/ml6-h2/value-net.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="directory with held-out .npz shards")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402

    paths = sorted(glob.glob(os.path.join(args.data, "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz shards found under {args.data}")
    planes_list, value_list = [], []
    for path in paths:
        z = np.load(path)
        planes_list.append(z["planes"])
        value_list.append(z["value"])
    planes = np.concatenate(planes_list)
    values = np.concatenate(value_list)
    # Side to move from plane 3 (all 1.0 iff Black to move).
    p3 = planes[:, 3]
    if not np.all((p3 == 0.0) | (p3 == 1.0)):
        raise SystemExit("plane 3 is not 0/1 — cannot read side to move")
    black_to_move = p3[:, 0, 0] > 0.5

    model = load_model_from_checkpoint(args.checkpoint)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(planes), args.batch_size):
            batch = torch.from_numpy(planes[i:i + args.batch_size])
            _, v = model(batch)
            preds.append(v.cpu().numpy().reshape(-1))
    preds = np.concatenate(preds)

    def _report(mask, label):
        p, t = preds[mask], values[mask]
        r = float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 else float("nan")
        sign = float(np.mean((p > 0) == (t > 0)))
        mse = float(np.mean((p - t) ** 2))
        print(f"{label:8s} n={len(p):5d}  Pearson r={r:+.4f}  "
              f"sign={sign:.4f}  MSE={mse:.4f}  mean_pred={p.mean():+.4f}")
        return r

    print(f"{args.checkpoint} on {args.data}")
    _report(np.ones(len(values), dtype=bool), "pooled")
    _report(black_to_move, "Black")
    _report(~black_to_move, "White")
    return 0


if __name__ == "__main__":
    sys.exit(main())

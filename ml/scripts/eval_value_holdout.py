#!/usr/bin/env python3
"""Value-head hold-out accuracy for the v8 value net (batch 3 H2).

Loads .npz shards (the held-out games — NEVER training data), runs
each checkpoint's value head over every position, and reports the
sign agreement between the predicted value and the actual game
outcome on non-draw positions (draw positions carry no sign
signal), plus value MSE over all positions. Two views are computed:

* ``full`` — the planes exactly as stored in the shards (history
  planes filled from the real game history);
* ``no_history`` — planes 2 (last move) and 4–7 (history) zeroed,
  approximating leaf conditions at the search root, where the real
  game history is unknown to ``choose_move`` (deeper leaves get
  search-path history, so this is a lower-bound-style approximation
  of leaf quality, not an exact one).

Usage:
    python ml/scripts/eval_value_holdout.py --data ml/runs/b3-h2/val-data \\
        --checkpoint ml/runs/b3-h2-value/checkpoints/best.pt \\
        --checkpoint ml/runs/b3-h2-value/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402

# Planes zeroed in the no-history view: 2 = last move one-hot,
# 4-7 = stones 2/4 plies ago (see gomoku_ml/encoding.py).
_HISTORY_PLANES = (2, 4, 5, 6, 7)


def load_shards(data_dir: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Concatenate (planes, value) over all shards in ``data_dir``."""
    planes_list, value_list = [], []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    for path in paths:
        with np.load(path) as z:
            planes_list.append(z["planes"])
            value_list.append(z["value"])
    if not planes_list:
        raise FileNotFoundError(f"no .npz shards found under {data_dir}")
    return (
        np.concatenate(planes_list),
        np.concatenate(value_list),
        len(paths),
    )


def predict_values(
    model: torch.nn.Module,
    planes: np.ndarray,
    batch_size: int = 256,
    zero_history: bool = False,
) -> np.ndarray:
    """Value-head output for every position (float array in [-1, 1])."""
    if zero_history:
        planes = planes.copy()
        planes[:, _HISTORY_PLANES, :, :] = 0.0
    out = np.empty(len(planes), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(planes), batch_size):
            x = torch.from_numpy(planes[i : i + batch_size])
            _logits, value = model(x)
            out[i : i + batch_size] = value[:, 0].cpu().numpy()
    return out


def score_predictions(values: np.ndarray, targets: np.ndarray) -> dict:
    """Sign agreement on non-draw positions + MSE over all positions."""
    nonzero = targets != 0
    pred_sign = np.sign(values[nonzero])
    true_sign = np.sign(targets[nonzero])
    agree = (pred_sign == true_sign)
    # A prediction of exactly 0.0 agrees with neither sign; count it
    # as a miss (it is neither right nor wrong — but it is not a
    # correct sign prediction).
    return {
        "positions": int(len(targets)),
        "non_draw_positions": int(nonzero.sum()),
        "sign_agreement": round(float(agree.mean()), 4) if nonzero.sum() else None,
        "correct": int(agree.sum()),
        "value_mse": round(float(np.mean((values - targets) ** 2)), 4),
        "mean_abs_value": round(float(np.mean(np.abs(values))), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="directory with held-out .npz shards")
    parser.add_argument("--checkpoint", required=True, action="append",
                        help="value-net .pt checkpoint (repeatable)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    torch.set_num_threads(common.torch_threads(2))
    torch.set_num_interop_threads(1)
    planes, targets, n_games = load_shards(args.data)
    print(f"hold-out: {n_games} games, {len(planes)} positions "
          f"({int((targets != 0).sum())} non-draw)", file=sys.stderr)

    results: dict = {
        "data": args.data,
        "games": n_games,
        "positions": int(len(planes)),
        "checkpoints": {},
    }
    for ckpt in args.checkpoint:
        model, _payload = load_model_from_checkpoint(ckpt, device="cpu")
        full = predict_values(model, planes, args.batch_size,
                              zero_history=False)
        nohist = predict_values(model, planes, args.batch_size,
                                zero_history=True)
        entry = {
            "full_planes": score_predictions(full, targets),
            "no_history_planes": score_predictions(nohist, targets),
        }
        results["checkpoints"][ckpt] = entry
        f, h = entry["full_planes"], entry["no_history_planes"]
        print(f"{ckpt}:")
        print(f"  full planes:       sign agreement {f['sign_agreement']} "
              f"({f['correct']}/{f['non_draw_positions']}), "
              f"MSE {f['value_mse']}, mean|v| {f['mean_abs_value']}")
        print(f"  no-history planes: sign agreement {h['sign_agreement']} "
              f"({h['correct']}/{h['non_draw_positions']}), "
              f"MSE {h['value_mse']}, mean|v| {h['mean_abs_value']}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

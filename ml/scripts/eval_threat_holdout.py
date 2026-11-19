#!/usr/bin/env python3
"""Threat-head hold-out accuracy for the v38 multi-task net (ML batch 3 H3).

Loads threat-labelled .npz shards (the held-out games — NEVER
training data), runs the multi-task checkpoint's threat head over
every position, and reports classification accuracy: overall, per
true class, and by side to move (plane 3: Black vs White to move).
The majority-class baseline is reported alongside (the classes are
imbalanced — most quiet positions are class 0). Value sign
accuracy on the same positions is reported for reference, so the
value and threat heads are measured on one pass.

Two views, as in eval_value_holdout.py:
* ``full`` — the planes exactly as stored in the shards;
* ``no_history`` — planes 2 (last move) and 4–7 (history) zeroed,
  approximating leaf conditions at the search root.

Usage:
    python ml/scripts/eval_threat_holdout.py \\
        --data ml/runs/ml3-h3/data/v7-holdout \\
        --checkpoint ml/runs/ml3-h3/value-net.pt
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
from gomoku_ml import threat as threat_mod  # noqa: E402

# Planes zeroed in the no-history view: 2 = last move one-hot,
# 4-7 = stones 2/4 plies ago (see gomoku_ml/encoding.py).
_HISTORY_PLANES = (2, 4, 5, 6, 7)


def load_shards(data_dir: str):
    """Concatenate (planes, value, threat, side_to_move) over shards."""
    planes_list, value_list, threat_list = [], [], []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    for path in paths:
        with np.load(path) as z:
            planes_list.append(z["planes"])
            value_list.append(z["value"])
            threat_list.append(z["threat"])
    if not planes_list:
        raise FileNotFoundError(f"no .npz shards found under {data_dir}")
    planes = np.concatenate(planes_list)
    values = np.concatenate(value_list)
    threats = np.concatenate(threat_list)
    # Side to move: plane 3 is all 1.0 iff Black is to move.
    black_to_move = planes[:, 3, 0, 0] > 0.5
    return planes, values, threats, black_to_move, len(paths)


def predict(model, planes, batch_size, zero_history):
    """(threat logits, values) for every position."""
    model.eval()
    all_logits, all_values = [], []
    with torch.no_grad():
        for i in range(0, len(planes), batch_size):
            batch = planes[i : i + batch_size].copy()
            if zero_history:
                batch[:, _HISTORY_PLANES] = 0.0
            x = torch.from_numpy(batch)
            _p, v, t = model(x, return_threat=True)
            all_logits.append(t.numpy())
            all_values.append(v.numpy()[:, 0])
    return np.concatenate(all_logits), np.concatenate(all_values)


def score(threat_logits, values, targets, threats, black_to_move):
    """Accuracy report for one view."""
    pred = threat_logits.argmax(axis=1)
    correct = pred == threats
    n = len(threats)
    per_class = {}
    for c in range(threat_mod.NUM_CLASSES):
        mask = threats == c
        per_class[threat_mod.CLASS_NAMES[c]] = {
            "n": int(mask.sum()),
            "accuracy": round(float(correct[mask].mean()), 4)
            if mask.any() else None,
        }
    majority = int(np.bincount(threats, minlength=3).argmax())
    majority_acc = float((threats == majority).mean())
    # Value sign agreement on non-zero targets (rollout labels).
    nz = targets != 0
    value_sign = (
        round(float(((values[nz] > 0) == (targets[nz] > 0)).mean()), 4)
        if nz.any()
        else None
    )
    out = {
        "positions": n,
        "threat_accuracy": round(float(correct.mean()), 4),
        "majority_class": threat_mod.CLASS_NAMES[majority],
        "majority_baseline": round(majority_acc, 4),
        "per_class": per_class,
        "value_sign_agreement": value_sign,
        "by_side_to_move": {},
    }
    for name, mask in (("black", black_to_move), ("white", ~black_to_move)):
        if not mask.any():
            continue
        out["by_side_to_move"][name] = {
            "n": int(mask.sum()),
            "threat_accuracy": round(float(correct[mask].mean()), 4),
            "value_sign_agreement": round(
                float(
                    ((values[mask & nz] > 0) == (targets[mask & nz] > 0)).mean()
                ),
                4,
            )
            if (mask & nz).any()
            else None,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True,
                        help="directory with held-out .npz shards")
    parser.add_argument("--checkpoint", required=True, action="append",
                        help="multi-task .pt checkpoint (repeatable)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    torch.set_num_threads(common.torch_threads(2))
    torch.set_num_interop_threads(1)
    planes, targets, threats, black_to_move, n_games = load_shards(args.data)
    print(f"hold-out: {n_games} games, {len(planes)} positions "
          f"(classes {np.bincount(threats, minlength=3).tolist()}, "
          f"Black to move {int(black_to_move.sum())})", file=sys.stderr)

    results: dict = {
        "data": args.data,
        "games": n_games,
        "positions": int(len(planes)),
        "checkpoints": {},
    }
    for ckpt in args.checkpoint:
        model, _payload = load_model_from_checkpoint(ckpt, device="cpu")
        if not getattr(model, "has_threat_head", False):
            print(f"{ckpt}: no threat head — skipping", file=sys.stderr)
            continue
        full_logits, full_values = predict(model, planes, args.batch_size, False)
        nohist_logits, nohist_values = predict(model, planes, args.batch_size, True)
        entry = {
            "full_planes": score(full_logits, full_values, targets, threats, black_to_move),
            "no_history_planes": score(nohist_logits, nohist_values, targets, threats, black_to_move),
        }
        results["checkpoints"][ckpt] = entry
        f, h = entry["full_planes"], entry["no_history_planes"]
        print(f"{ckpt}:")
        print(f"  full planes:       threat acc {f['threat_accuracy']} "
              f"(majority baseline {f['majority_baseline']}), "
              f"value sign {f['value_sign_agreement']}")
        print(f"  no-history planes: threat acc {h['threat_accuracy']}, "
              f"value sign {h['value_sign_agreement']}")
        for side, s in f["by_side_to_move"].items():
            print(f"    {side} to move: threat acc {s['threat_accuracy']} "
                  f"(n={s['n']}), value sign {s['value_sign_agreement']}")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

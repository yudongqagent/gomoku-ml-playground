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


def load_shards(data_dir: str) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Concatenate (planes, value) over all shards in ``data_dir``.

    Also returns the side to move per position (0 = Black, 1 =
    White), read from plane 3 of the stored encoding — the
    side-to-move plane (all 1.0 iff Black is to move, else all 0.0;
    see ``gomoku_ml/encoding.py``). Plane 3 is exact for every
    stored position regardless of which plies the shard kept.

    Do NOT infer the side from the within-shard position index
    (``arange(len(planes)) % 2``): that is correct only when the
    shard stores EVERY ply of the game. Distill shards store only
    QUIET positions (``generate_distill_data.py`` skips plies where
    ``_terminal_leaf_value != 0.0``), so after the first skipped
    ply the index-inferred side FLIPS, and every further skip flips
    it again — attenuating any true by-side gap toward zero (ML
    batch 5, iter-48 §0: up to ~60% attenuation; iter-37/iter-44
    by-side numbers are unreliable for this reason).
    """
    planes_list, value_list, side_list = [], [], []
    paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    for path in paths:
        with np.load(path) as z:
            planes = z["planes"]
            planes_list.append(planes)
            value_list.append(z["value"])
            p3 = planes[:, 3]
            # Plane 3 must be a uniform 0/1 plane per position; if a
            # shard was encoded without it, fail loudly rather than
            # silently falling back to the broken index inference.
            if not np.all((p3 == 0.0) | (p3 == 1.0)):
                raise ValueError(
                    f"{path}: plane 3 is not 0/1 — cannot read side to move"
                )
            if not np.all(p3.min(axis=(1, 2)) == p3.max(axis=(1, 2))):
                raise ValueError(
                    f"{path}: plane 3 is not uniform per position — "
                    "cannot read side to move"
                )
            # Plane 3 is all 1.0 iff Black is to move.
            side_list.append((p3[:, 0, 0] < 0.5).astype(np.int8))
    if not planes_list:
        raise FileNotFoundError(f"no .npz shards found under {data_dir}")
    return (
        np.concatenate(planes_list),
        np.concatenate(value_list),
        len(paths),
        np.concatenate(side_list),
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


def score_predictions(values: np.ndarray, targets: np.ndarray,
                      mask: np.ndarray | None = None) -> dict:
    """Sign agreement on non-draw positions + MSE over all positions.

    With ``mask`` given, only the masked positions are scored (used
    for the by-side-to-move split).
    """
    if mask is not None:
        values, targets = values[mask], targets[mask]
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
    planes, targets, n_games, side = load_shards(args.data)
    print(f"hold-out: {n_games} games, {len(planes)} positions "
          f"({int((targets != 0).sum())} non-draw)", file=sys.stderr)
    black_mask = side == 0
    white_mask = side == 1

    results: dict = {
        "data": args.data,
        "games": n_games,
        "positions": int(len(planes)),
        "black_to_move_positions": int(black_mask.sum()),
        "white_to_move_positions": int(white_mask.sum()),
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
            # By side to move (v30's hypothesis test: v25's White
            # collapse predicts a White-to-move accuracy gap that
            # colour-balanced training should close).
            "by_side": {
                "black_to_move": {
                    "full_planes": score_predictions(full, targets, black_mask),
                    "no_history_planes": score_predictions(nohist, targets, black_mask),
                },
                "white_to_move": {
                    "full_planes": score_predictions(full, targets, white_mask),
                    "no_history_planes": score_predictions(nohist, targets, white_mask),
                },
            },
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
        fb = entry["by_side"]["black_to_move"]["full_planes"]
        fw = entry["by_side"]["white_to_move"]["full_planes"]
        hb = entry["by_side"]["black_to_move"]["no_history_planes"]
        hw = entry["by_side"]["white_to_move"]["no_history_planes"]
        print(f"  by side (full):    Black-to-move {fb['sign_agreement']} "
              f"({fb['correct']}/{fb['non_draw_positions']}), "
              f"White-to-move {fw['sign_agreement']} "
              f"({fw['correct']}/{fw['non_draw_positions']})")
        print(f"  by side (no-hist): Black-to-move {hb['sign_agreement']} "
              f"({hb['correct']}/{hb['non_draw_positions']}), "
              f"White-to-move {hw['sign_agreement']} "
              f"({hw['correct']}/{hw['non_draw_positions']})")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

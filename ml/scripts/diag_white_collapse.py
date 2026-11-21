#!/usr/bin/env python3
"""Diagnose the White collapse (ML batch 5, H1 / v48).

Replays v44's battery games against four opponents (v0, v7, v8,
v37), extracts the quiet positions with their TRUE side to move,
labels each with the 2-ply and 3-ply rollout-leaf teachers, and
scores the v44 net (against its own 3-ply target) and the v37 net
(against its own 2-ply target) by opponent x side to move.

Why not ``eval_value_holdout.py``: that script infers the side to
move as ``arange(len(planes)) % 2``, which is correct only when a
shard stores EVERY ply. Distill shards store only quiet positions,
so after the first skipped non-quiet ply the inferred side flips.
This script records the true side at replay time (ply t -> Black
iff t even) and additionally reports the index-inferred split, to
quantify that contamination (see iter-48.md section 0).

Usage:
    python ml/scripts/diag_white_collapse.py \
        --leg v0:research/iterations/batch-v44-vs-v0.json \
        --leg v7:research/iterations/batch-v44-vs-v7.json \
        --leg v8:research/iterations/batch-v44-vs-v8.json \
        --leg v37:research/iterations/batch-v44-vs-v37.json \
        --checkpoint-v44 ml/runs/ml4-h3/value-net.pt \
        --checkpoint-v37 ml/runs/ml3-h2/value-net.pt \
        --out ml/runs/ml5-h1/diag-white-collapse.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import encoding, game, mcts_heuristic  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402

_HISTORY_PLANES = (2, 4, 5, 6, 7)


def load_leg(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["records"]


def replay_positions(records: list[dict], opponent: str) -> list[dict]:
    """Quiet positions from replayed games, with true side + outcome.

    Each entry: planes, label2, label3, side (0 Black / 1 White),
    outcome (+1 side-to-move eventually wins, -1 loses, 0 draw),
    quiet_index (position index among the game's quiet positions,
    for the index-inferred side comparison), opponent, game.
    """
    out: list[dict] = []
    for rec in records:
        moves = rec["moves"]
        winner = rec.get("winner")
        boards: list[np.ndarray] = [game.empty_board()]
        player = game.BLACK
        for m in moves:
            boards.append(game.apply_move(boards[-1], m, player))
            player = -player
        quiet_idx = 0
        for t in range(len(moves)):
            board = boards[t]
            p = game.BLACK if t % 2 == 0 else game.WHITE
            if mcts_heuristic._terminal_leaf_value(board, p) != 0.0:
                continue
            history = boards[:t]
            last_move = moves[t - 1] if t > 0 else None
            planes = encoding.encode(board, p, history, last_move)
            if winner == "draw" or winner is None:
                outcome = 0
            elif (winner == "black") == (p == game.BLACK):
                outcome = 1
            else:
                outcome = -1
            out.append({
                "planes": planes,
                "label2": mcts_heuristic._rollout2_leaf_value(board, p),
                "label3": mcts_heuristic._rollout3_leaf_value(board, p),
                "side": 0 if p == game.BLACK else 1,
                "outcome": outcome,
                "quiet_index": quiet_idx,
                "opponent": opponent,
                "game": rec["game"],
            })
            quiet_idx += 1
    return out


def predict_values(model, planes, batch_size=256, zero_history=False):
    if zero_history:
        planes = planes.copy()
        planes[:, _HISTORY_PLANES, :, :] = 0.0
    out = np.empty(len(planes), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(planes), batch_size):
            x = torch.from_numpy(planes[i:i + batch_size])
            _logits, value = model(x)
            out[i:i + batch_size] = value[:, 0].cpu().numpy()
    return out


def score(values, targets, mask=None):
    if mask is not None:
        values, targets = values[mask], targets[mask]
    nonzero = targets != 0
    agree = (np.sign(values[nonzero]) == np.sign(targets[nonzero]))
    r = float(np.corrcoef(values, targets)[0, 1]) if len(values) > 2 else None
    return {
        "positions": int(len(targets)),
        "non_draw_positions": int(nonzero.sum()),
        "sign_agreement": round(float(agree.mean()), 4) if nonzero.sum() else None,
        "correct": int(agree.sum()),
        "value_mse": round(float(np.mean((values - targets) ** 2)), 4),
        "pearson_r": round(r, 4) if r is not None else None,
        "mean_abs_value": round(float(np.mean(np.abs(values))), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", required=True, action="append",
                        help="opponent:path to a battery JSON (repeatable)")
    parser.add_argument("--checkpoint-v44", required=True)
    parser.add_argument("--checkpoint-v37", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(common.torch_threads(2))
    torch.set_num_interop_threads(1)

    t0 = time.time()
    positions: list[dict] = []
    for leg in args.leg:
        opponent, path = leg.split(":", 1)
        records = load_leg(path)
        before = len(positions)
        positions.extend(replay_positions(records, opponent))
        print(f"{opponent}: {len(records)} games, "
              f"{len(positions) - before} quiet positions "
              f"({time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(f"total: {len(positions)} quiet positions in "
          f"{time.time() - t0:.0f}s", file=sys.stderr)

    planes = np.stack([p["planes"] for p in positions]).astype(np.float32)
    label2 = np.array([p["label2"] for p in positions], dtype=np.float64)
    label3 = np.array([p["label3"] for p in positions], dtype=np.float64)
    side = np.array([p["side"] for p in positions], dtype=np.int8)
    outcome = np.array([p["outcome"] for p in positions], dtype=np.int8)
    quiet_idx = np.array([p["quiet_index"] for p in positions], dtype=np.int64)
    opponents = np.array([p["opponent"] for p in positions])
    # Index-inferred side (eval_value_holdout's assumption): the
    # position's index among the game's QUIET positions, mod 2.
    side_inferred = (quiet_idx % 2).astype(np.int8)

    results: dict = {
        "positions": int(len(positions)),
        "legs": {opp: int((opponents == opp).sum())
                 for opp in sorted(set(opponents))},
        "subjects": {},
    }
    subjects = [
        ("v44", args.checkpoint_v44, label3, "3-ply"),
        ("v37", args.checkpoint_v37, label2, "2-ply"),
    ]
    for name, ckpt, targets, label_name in subjects:
        model, _payload = load_model_from_checkpoint(ckpt, device="cpu")
        full = predict_values(model, planes, zero_history=False)
        nohist = predict_values(model, planes, zero_history=True)
        entry: dict = {"checkpoint": ckpt, "target": label_name,
                       "overall": {"full_planes": score(full, targets),
                                   "no_history_planes": score(nohist, targets)},
                       "by_opponent_side": {},
                       "by_opponent_side_inferred": {},
                       "by_outcome": {}}
        for opp in sorted(set(opponents)):
            for s, sname in ((0, "black"), (1, "white")):
                mask = (opponents == opp) & (side == s)
                entry["by_opponent_side"][f"{opp}:{sname}"] = {
                    "full_planes": score(full, targets, mask),
                    "no_history_planes": score(nohist, targets, mask),
                }
                mask_inf = (opponents == opp) & (side_inferred == s)
                entry["by_opponent_side_inferred"][f"{opp}:{sname}"] = {
                    "full_planes": score(full, targets, mask_inf),
                }
            # Win/loss split (v30's cut): accuracy on positions the
            # side to move eventually wins vs loses, by side.
            for s, sname in ((0, "black"), (1, "white")):
                for oc, ocname in ((1, "side_wins"), (-1, "side_loses")):
                    mask = ((opponents == opp) & (side == s)
                            & (outcome == oc))
                    entry["by_outcome"][f"{opp}:{sname}:{ocname}"] = {
                        "full_planes": score(full, targets, mask),
                    }
        # Pooled heuristic (v0+v7) vs ML (v8+v37) cells for the
        # pre-registered decision rule (iter-48.md section 2.3).
        heur = np.isin(opponents, ["v0", "v7"])
        ml = np.isin(opponents, ["v8", "v37"])
        for fam, fam_mask in (("heuristic", heur), ("ml", ml)):
            for s, sname in ((0, "black"), (1, "white")):
                mask = fam_mask & (side == s)
                entry["by_opponent_side"][f"{fam}-pooled:{sname}"] = {
                    "full_planes": score(full, targets, mask),
                    "no_history_planes": score(nohist, targets, mask),
                }
        results["subjects"][name] = entry

        # Human-readable cell table.
        print(f"\n== {name} net (target: {label_name} labels) ==")
        print(f"overall full: {entry['overall']['full_planes']}")
        print(f"overall no-hist: {entry['overall']['no_history_planes']}")
        print(f"{'cell':22s} {'n':>5s} {'full':>7s} {'no-hist':>7s} {'r':>6s}")
        for cell, scores in entry["by_opponent_side"].items():
            f = scores["full_planes"]
            h = scores["no_history_planes"]
            print(f"{cell:22s} {f['positions']:5d} "
                  f"{f['sign_agreement']!s:>7s} {h['sign_agreement']!s:>7s} "
                  f"{f['pearson_r']!s:>6s}")
        print("-- index-inferred side (eval_value_holdout assumption) --")
        for cell, scores in entry["by_opponent_side_inferred"].items():
            f = scores["full_planes"]
            print(f"{cell:22s} {f['positions']:5d} "
                  f"{f['sign_agreement']!s:>7s}")
        print("-- by eventual outcome (full planes) --")
        for cell, scores in entry["by_outcome"].items():
            f = scores["full_planes"]
            print(f"{cell:28s} {f['positions']:5d} "
                  f"{f['sign_agreement']!s:>7s}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

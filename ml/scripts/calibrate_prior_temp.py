#!/usr/bin/env python3
"""Calibrate v26's prior temperature OFFLINE (ML batch 1, H3).

Pre-registered rule (research/iterations/iter-26.md §1.3): on the
held-out positions (the SAME 5% val split training used — seed 0),
compute the v1 prior's median top-1 mass and the learned prior's
median top-1 mass at each T in {0.25, 0.5, 1.0, 2.0}; pick the T
whose median top-1 mass is CLOSEST to the v1 prior's. Top-1 mass is
measured AFTER masking to candidates and renormalising — the prior
the PUCT search actually sees.

Also reports the net's top-1 agreement with the teacher's top move
(argmax of the visit-distribution target) as a sanity metric.

Usage:
    python ml/scripts/calibrate_prior_temp.py \\
        --checkpoint ml/runs/ml1-h3/policy-value-net.pt \\
        --data ml/runs/ml1-h3/data-a ml/runs/ml1-h3/data-b \\
        --out ml/runs/ml1-h3/calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gomoku_ml import game, scoring, train  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402
from gomoku_ml.mcts_heuristic import _priors as v1_priors  # noqa: E402

TEMPERATURES = (0.25, 0.5, 1.0, 2.0)
VAL_FRACTION = 0.05
SEED = 0


def board_from_planes(planes: np.ndarray) -> tuple[np.ndarray, int]:
    """Reconstruct (board, player_to_move) from the 8-plane encoding.
    Plane 0 = player-to-move stones, plane 1 = opponent stones,
    plane 3 = all 1.0 iff Black to move."""
    player = game.BLACK if planes[3, 0, 0] > 0.5 else game.WHITE
    if player == game.BLACK:
        board = (planes[0] - planes[1]).astype(np.int8)
    else:
        board = (planes[1] - planes[0]).astype(np.int8)
    return board, player


def masked_softmax_top1(logits: np.ndarray, cands: list[int], temp: float) -> float:
    scaled = {m: float(logits[m]) / temp for m in cands}
    top = max(scaled.values())
    weights = {m: math.exp(v - top) for m, v in scaled.items()}
    return max(weights.values()) / sum(weights.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    planes, policy, _value = train.load_shards(args.data)
    n = len(planes)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_val = int(round(n * VAL_FRACTION))
    val_idx = perm[:n_val]
    print(f"held-out positions: {n_val} of {n}", file=sys.stderr, flush=True)

    net, _payload = load_model_from_checkpoint(args.checkpoint, device="cpu")
    net.eval()
    logits_all = np.zeros((n_val, 225), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n_val, 256):
            x = torch.from_numpy(planes[val_idx[i:i + 256]])
            logits, _v = net(x)
            logits_all[i:i + 256] = logits.numpy()

    v1_top1: list[float] = []
    net_top1: dict[float, list[float]] = {t: [] for t in TEMPERATURES}
    teacher_top1: list[float] = []
    agree = 0
    for k, idx in enumerate(val_idx):
        board, player = board_from_planes(planes[idx])
        cands = scoring._candidate_moves(board)
        if not cands:
            continue
        v1 = v1_priors(board, player)
        v1_top1.append(max(v1.values()))
        for t in TEMPERATURES:
            net_top1[t].append(masked_softmax_top1(logits_all[k], cands, t))
        teacher_top1.append(float(policy[idx].max()))
        teacher_best = int(np.argmax(policy[idx]))
        # Net's top candidate (argmax logit among candidates).
        cand_logits = [(logits_all[k][m], m) for m in cands]
        if max(cand_logits)[1] == teacher_best:
            agree += 1
    n_used = len(v1_top1)
    v1_med = float(np.median(v1_top1))
    result = {
        "n_held_out": n_used,
        "v1_prior_median_top1": round(v1_med, 4),
        "v1_prior_mean_top1": round(float(np.mean(v1_top1)), 4),
        "teacher_median_top1": round(float(np.median(teacher_top1)), 4),
        "net_top1_agreement_with_teacher": round(agree / max(1, n_used), 4),
        "temperatures": {},
        "chosen_temperature": None,
    }
    best_t, best_gap = None, float("inf")
    for t in TEMPERATURES:
        med = float(np.median(net_top1[t]))
        gap = abs(med - v1_med)
        result["temperatures"][str(t)] = {
            "median_top1": round(med, 4),
            "mean_top1": round(float(np.mean(net_top1[t])), 4),
            "gap_to_v1_median": round(gap, 4),
        }
        if gap < best_gap:
            best_t, best_gap = t, gap
    result["chosen_temperature"] = best_t
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"chosen T = {best_t} (closest median top-1 to v1's {v1_med:.4f})",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

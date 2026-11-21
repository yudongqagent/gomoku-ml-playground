#!/usr/bin/env python3
"""Score a value net on labelled shards by TRUE side to move (v48).

``eval_value_holdout.py`` infers the side as ``arange % 2``, which
is wrong on distill shards (only quiet positions are stored — see
iter-48.md section 0). This script recovers the true side by
replaying the games JSON and re-running the quiet-position filter
(the terminal check only — no rollout labelling), then pairs the
sides with the shard positions in order and scores a checkpoint
against the shard values by opponent x side, full planes and
no-history planes. Reports both ALL positions and HOLDOUT-only
positions (the honest generalisation check for a net trained on
the train shards).

Usage:
    python ml/scripts/score_shards_by_side.py \
        --leg v0:ml/runs/ml5-h1/games-vs-v0.json:ml/runs/ml5-h1/data-new/vs-v0:ml/runs/ml5-h1/data-new/vs-v0-holdout \
        --checkpoint ml/runs/ml5-h1/value-net.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import game, mcts_heuristic  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402
from eval_value_holdout import predict_values, score_predictions  # noqa: E402

_GAME_RE = re.compile(r"_game(\d+)\.npz$")


def quiet_sides(moves: list[int]) -> list[int]:
    """True side to move (0 Black / 1 White) for each QUIET ply."""
    boards: list[np.ndarray] = [game.empty_board()]
    player = game.BLACK
    for m in moves:
        boards.append(game.apply_move(boards[-1], m, player))
        player = -player
    sides: list[int] = []
    for t in range(len(moves)):
        p = game.BLACK if t % 2 == 0 else game.WHITE
        if mcts_heuristic._terminal_leaf_value(boards[t], p) != 0.0:
            continue
        sides.append(0 if p == game.BLACK else 1)
    return sides


def load_leg(games_path: str, train_dir: str, holdout_dir: str):
    """Return (planes, values, sides, opponents, is_holdout) arrays."""
    with open(games_path, encoding="utf-8") as f:
        games = {rec["game"]: rec["moves"]
                 for rec in json.load(f)["games"]}
    planes_list, values_list, sides_list, hold_list = [], [], [], []
    for directory, is_hold in ((train_dir, False), (holdout_dir, True)):
        for path in sorted(glob.glob(os.path.join(directory, "*.npz"))):
            game_no = int(_GAME_RE.search(path).group(1))
            sides = quiet_sides(games[game_no])
            with np.load(path) as z:
                planes, values = z["planes"], z["value"]
            assert len(planes) == len(sides), (
                f"{path}: {len(planes)} shard positions vs "
                f"{len(sides)} quiet plies — replay mismatch")
            planes_list.append(planes)
            values_list.append(values)
            sides_list.append(np.asarray(sides, dtype=np.int8))
            hold_list.append(np.full(len(sides), is_hold, dtype=bool))
    return (np.concatenate(planes_list), np.concatenate(values_list),
            np.concatenate(sides_list), np.concatenate(hold_list))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", required=True, action="append",
                        help="opponent:games.json:train_dir:holdout_dir")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    torch.set_num_threads(common.torch_threads(2))
    torch.set_num_interop_threads(1)

    all_planes, all_values, all_sides, all_hold, all_opp = [], [], [], [], []
    for leg in args.leg:
        opponent, games_path, train_dir, holdout_dir = leg.split(":")
        planes, values, sides, hold = load_leg(games_path, train_dir,
                                               holdout_dir)
        print(f"{opponent}: {len(planes)} positions "
              f"({int(hold.sum())} holdout)", file=sys.stderr)
        all_planes.append(planes)
        all_values.append(values)
        all_sides.append(sides)
        all_hold.append(hold)
        all_opp.append(np.full(len(planes), opponent))
    planes = np.concatenate(all_planes)
    values = np.concatenate(all_values)
    sides = np.concatenate(all_sides)
    hold = np.concatenate(all_hold)
    opp = np.concatenate(all_opp)

    model, _payload = load_model_from_checkpoint(args.checkpoint,
                                                 device="cpu")
    full = predict_values(model, planes, zero_history=False)
    nohist = predict_values(model, planes, zero_history=True)

    for label, base_mask in (("ALL positions", np.ones(len(planes), bool)),
                             ("HOLDOUT only", hold)):
        print(f"\n== {label} ({args.checkpoint}) ==")
        print(f"{'cell':16s} {'n':>5s} {'full':>7s} {'no-hist':>7s}")
        for opponent in sorted(set(opp)):
            for s, sname in ((0, "black"), (1, "white")):
                mask = base_mask & (opp == opponent) & (sides == s)
                f = score_predictions(full, values, mask)
                h = score_predictions(nohist, values, mask)
                print(f"{opponent}:{sname:5s} {f['positions']:5d} "
                      f"{str(f['sign_agreement']):>7s} "
                      f"{str(h['sign_agreement']):>7s}")
        for s, sname in ((0, "black"), (1, "white")):
            mask = base_mask & (sides == s)
            f = score_predictions(full, values, mask)
            h = score_predictions(nohist, values, mask)
            print(f"pooled:{sname:5s} {f['positions']:5d} "
                  f"{str(f['sign_agreement']):>7s} "
                  f"{str(h['sign_agreement']):>7s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

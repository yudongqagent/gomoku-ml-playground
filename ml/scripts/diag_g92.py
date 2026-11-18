#!/usr/bin/env python3
"""Net-value diagnostic for setup-v1v0s-g92 (v38, ML batch 3 H3).

The v26/v31 failure position (iter-38 §1.9): Black to move, the
expected move is the quiet 2-ply setup (9,6); v31's net values it
-0.377 (1 visit) vs the wrong (5,4)'s +0.788 (75 visits) although
the v1 prior ranks (9,6) 1st. v8's net values (9,6) +0.246 and
passes.

For each --version engine (38/8) and each --checkpoint net (e.g.
v24's, by absolute path — v24 is not registered on this branch),
prints:
  * the 100-sim root ranking: top-10 children by visits (move,
    visits, our-q, prior), with (9,6) and (5,4) marked;
  * the raw net value of the resulting board after (9,6) and (5,4)
    (side to move = White), with 1-ply search-path history and
    with history zeroed, plus the leaf value the search uses
    (terminal short-circuits included), ours-perspective.

Usage:
    python ml/scripts/diag_g92.py --version 38 --version 8 \\
        --checkpoint /home/hatch/workspace/projects/gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml import encoding, game, scoring  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402
from gomoku_ml.mcts_heuristic import _Node  # noqa: E402
from gomoku_ml.mcts_value import ValueMCTS  # noqa: E402

SETUP = (9, 6)  # expected move (x, y)
WRONG = (5, 4)  # v31's chosen move (x, y)


def _xy(m: int) -> str:
    return f"({m % 15},{m // 15})"


def _move(xy: tuple[int, int]) -> int:
    return xy[1] * 15 + xy[0]


def _load_position() -> tuple[np.ndarray, int]:
    path = os.path.join(
        common.ML_DIR, "gomoku_ml", "tactical_positions_hard.json"
    )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    positions = data if isinstance(data, list) else data.get("positions", [])
    for p in positions:
        if p["id"] == "setup-v1v0s-g92":
            board = game.empty_board()
            for x, y in p["black"]:
                board[y, x] = game.BLACK
            for x, y in p["white"]:
                board[y, x] = game.WHITE
            return board, p["to_move"]
    raise ValueError("setup-v1v0s-g92 not found in the hard suite")


def _root_ranking(mcts, board, player, sims=100):
    root = _Node(1.0)
    for _ in range(sims):
        mcts._run_one(root, board.copy(), player, None)
    rows = []
    for m, c in root.children.items():
        # value_sum is accumulated from the perspective of the player
        # to move AT THE CHILD (the opponent), so our-q is negated.
        q = -c.value_sum / c.visit_count if c.visit_count else 0.0
        rows.append((m, c.visit_count, q, c.prior))
    rows.sort(key=lambda r: (-r[1], -r[2]))
    return rows


def _net_values(mcts, board, player, move):
    """Raw net value + leaf value of the board after ``move``."""
    b2 = game.apply_move(board, move, player)
    opp = -player
    out = {}
    for label, history in (("hist1", [board]), ("nohist", None)):
        planes = encoding.encode(b2, opp, history, move)
        with torch.no_grad():
            x = torch.from_numpy(planes).unsqueeze(0).to(mcts.device)
            _logits, value = mcts.net(x)
        out[label] = float(value[0, 0].item())
    # Leaf value as the search uses it (ours-perspective: negated,
    # since the leaf is evaluated for the opponent to move).
    leaf = mcts._value_leaf_value(b2, opp, [board], move)
    out["leaf_ours"] = -leaf
    return out


def _report(name, mcts, board, player):
    print(f"-- {name} (value_weight={mcts.value_weight}) --")
    rows = _root_ranking(mcts, board, player)
    for m, visits, q, prior in rows[:10]:
        mark = ""
        if m == _move(SETUP):
            mark = "  <-- EXPECTED (9,6)"
        elif m == _move(WRONG):
            mark = "  <-- v31's choice (5,4)"
        print(f"  {_xy(m):>7} visits={visits:3d} our-q={q:+.3f} "
              f"prior={prior:.4f}{mark}")
    for label, xy in (("EXPECTED (9,6)", SETUP), ("WRONG (5,4)", WRONG)):
        mv = _move(xy)
        vals = _net_values(mcts, board, player, mv)
        print(f"  after {label}: raw net (White to move) "
              f"hist1={vals['hist1']:+.3f} nohist={vals['nohist']:+.3f} "
              f"leaf_ours={vals['leaf_ours']:+.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="extra .pt net to diagnose (ValueMCTS, alpha 1.0)")
    args = parser.parse_args()
    board, player = _load_position()
    print("== setup-v1v0s-g92 (Black to move, expected (9,6)) ==")
    for version in args.version:
        p = scoring.get_player(version, seed=0)
        _report(f"v{version}", p._mcts, board, player)
    for ckpt in args.checkpoint:
        net, _payload = load_model_from_checkpoint(ckpt, device="cpu")
        mcts = ValueMCTS(model=net, sims=100, rng=np.random.default_rng(0),
                         value_weight=1.0)
        _report(os.path.basename(os.path.dirname(ckpt)) + "/value-net.pt",
                mcts, board, player)
    return 0


if __name__ == "__main__":
    sys.exit(main())

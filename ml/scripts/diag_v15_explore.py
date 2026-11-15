#!/usr/bin/env python3
"""Pre-eval diagnostic for batch 6 H1 (v15, iter-15 §1.5).

Replays the 5 documented starved-save positions (iter-12 §2.2's
three, iter-14 §2.1 g9, iter-7 §2.1 fl1-g3) under each sweep
mechanism (control, floor 0.005, floor 0.01, c_puct 2.0, c_puct
3.0) and reports root visits / our-q / prior for the saving
move(s) vs the losing move, plus which move is played.

DIAGNOSTIC ONLY: it checks each mechanism moves visits in the
predicted direction (the save's visit share rising toward its
prior ratio). It does NOT change the design or the sweep, and
it does not validate a global search change (batch-5 lesson:
single-position diagnostics cannot — v13's temperature probe
passed 1/7 while the sweep refuted it overwhelmingly). The
sweep (§1.2) and the battery (§1.4) decide.

The v12/v14 regression files live on research/batch-5-h1 and
research/batch-5-h3 (not on master); extract them first:
    git show origin/research/batch-5-h1:ml/gomoku_ml/regression_positions_v12.json > /tmp/regression_positions_v12.json
    git show origin/research/batch-5-h3:ml/gomoku_ml/regression_positions_v14.json > /tmp/regression_positions_v14.json

Usage:
    python ml/scripts/diag_v15_explore.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import game  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS, _Node  # noqa: E402

# (file, position id) — the 5 documented starved-save positions.
POSITIONS = [
    ("v12", "fl2-g2-ply20"),
    ("v12", "fl2-g1-ply53"),
    ("v12", "fl2-g5-ply37"),
    ("v14", "fl1-g9-ply15"),
    ("v7", "fl1-g3-ply23"),
]

MECHANISMS = [
    ("control (v7)", {}),
    ("floor 0.005", {"prior_floor": 0.005}),
    ("floor 0.01", {"prior_floor": 0.01}),
    ("c_puct 2.0", {"c_puct": 2.0}),
    ("c_puct 3.0", {"c_puct": 3.0}),
]


def _xy(m: int) -> tuple[int, int]:
    return (m % 15, m // 15)


def _flat(xy) -> int:
    return xy[1] * 15 + xy[0]


def _load(repo_root: str) -> dict[str, dict]:
    files = {
        "v12": "/tmp/regression_positions_v12.json",
        "v14": "/tmp/regression_positions_v14.json",
        "v7": os.path.join(
            repo_root, "ml", "gomoku_ml", "regression_positions_v7.json"),
    }
    out = {}
    for key, path in files.items():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for pos in data["positions"]:
            out[f"{key}:{pos['id']}"] = pos
    return out


def _board(pos: dict) -> tuple[np.ndarray, int]:
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    return board, player


def _root_stats(board, player, sims=100, **kwargs):
    """Run the search manually; return (stats, played). stats maps
    move -> (visits, our-q, prior). our-q = -child.q (child.q is
    the OPPONENT's perspective)."""
    mcts = HeuristicMCTS(sims=sims, rng=np.random.default_rng(0),
                         leaf_mode="gradient", **kwargs)
    root = _Node(1.0)
    for _ in range(sims):
        mcts._run_one(root, board.copy(), player, None)
    stats = {m: (c.visit_count, -c.q, c.prior)
             for m, c in root.children.items()}
    best_visits = max(c.visit_count for c in root.children.values())
    best = [m for m, c in root.children.items()
            if c.visit_count == best_visits]
    return stats, best


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    positions = _load(repo_root)
    for key, pid in POSITIONS:
        pos = positions[f"{key}:{pid}"]
        board, player = _board(pos)
        saves = [_flat(xy) for xy in pos["expected"]]
        loser = _flat(pos["played_before_patch"])
        print(f"== {key}:{pid} ({pos['to_move']} to move) "
              f"loser {_xy(loser)}, saves {[_xy(s) for s in saves]} ==")
        for name, kwargs in MECHANISMS:
            stats, best = _root_stats(board, player, **kwargs)
            lv, lq, lp = stats[loser]
            save_strs = []
            for s in saves:
                v, q, p = stats[s]
                save_strs.append(f"{_xy(s)}: v{v} q{q:+.3f} p{p:.3f}")
            played = _xy(best[0]) if len(best) == 1 else (
                f"tie {[_xy(b) for b in best]}")
            print(f"  {name:>14}: loser v{lv} q{lq:+.3f} p{lp:.3f} | "
                  f"{' ; '.join(save_strs)} | plays {played}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

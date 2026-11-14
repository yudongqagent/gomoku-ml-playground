#!/usr/bin/env python3
"""Generate the frozen Python reference fixtures for the TypeScript v7 port.

Writes ``src/ai/__fixtures__/v7-reference.json``: for each position, the
stone lists, the side to move, the v1 total + attack scores and the
kernel attack-only scores (both colours) on every empty cell, and the
move Python v7 (HeuristicMCTS, 100 sims, c_puct 1.5, leaf_mode
"gradient", rng=None so root ties break to the first move) chooses.

Positions: the old tactical suite (20), the hard tactical suite (29),
the v7 fast-loop regression positions (5), and 30 random mid-game
positions (seeded, reproducible). The vitest suite
(``src/ai/v7.test.ts``) compares the TypeScript port against these
numbers cell by cell — the scorer and the kernel attack must match
exactly, and the TS engine's move must match Python's on every
position (agreement is reported, and any disagreement is a bug to
investigate, not noise: both sides use the same deterministic
first-best tie-break).

Usage (from the repo root; needs the ml venv):
    ml/.venv/bin/python ml/scripts/gen_v7_ts_fixtures.py
    ml/.venv/bin/python ml/scripts/gen_v7_ts_fixtures.py --out <path>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

from gomoku_ml import game, kernel_numpy, scoring  # noqa: E402
from gomoku_ml.mcts_heuristic import HeuristicMCTS  # noqa: E402

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DEFAULT_OUT = os.path.join(
    REPO_ROOT, "src", "ai", "__fixtures__", "v7-reference.json"
)


def _board_from_stones(black: list, white: list) -> np.ndarray:
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def _random_position(rng: random.Random, plies: int) -> tuple[list, list, int] | None:
    """Random playout of ``plies`` moves (uniform over candidates)."""
    board = game.empty_board()
    player = game.BLACK
    for _ in range(plies):
        cands = scoring._candidate_moves(board)
        if not cands:
            return None
        move = rng.choice(cands)
        board[move // 15, move % 15] = player
        if game.is_winning_stone(board, move, player):
            return None  # game already over; skip
        player = -player
    black = [[int(x), int(y)] for y, x in zip(*np.where(board == game.BLACK))]
    white = [[int(x), int(y)] for y, x in zip(*np.where(board == game.WHITE))]
    return black, white, player


def collect_positions() -> list[dict]:
    positions: list[dict] = []

    def _load(path: str, tag: str) -> list[dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data["positions"]

    ml_dir = os.path.join(REPO_ROOT, "ml", "gomoku_ml")
    for pos in _load(os.path.join(ml_dir, "tactical_positions.json"), "old"):
        positions.append(
            {
                "id": f"old-{pos['id']}",
                "black": pos["black"],
                "white": pos["white"],
                "to_move": pos["to_move"],
            }
        )
    for pos in _load(os.path.join(ml_dir, "tactical_positions_hard.json"), "hard"):
        positions.append(
            {
                "id": f"hard-{pos['id']}",
                "black": pos["black"],
                "white": pos["white"],
                "to_move": pos["to_move"],
            }
        )
    for pos in _load(os.path.join(ml_dir, "regression_positions_v7.json"), "reg"):
        positions.append(
            {
                "id": f"reg-{pos['id']}",
                "black": pos["black"],
                "white": pos["white"],
                "to_move": 1 if pos["to_move"] == "black" else -1,
            }
        )
    rng = random.Random(20261114)
    made = 0
    attempts = 0
    while made < 30 and attempts < 300:
        attempts += 1
        plies = [6, 12, 20, 30, 45, 60][made % 6]
        got = _random_position(rng, plies)
        if got is None:
            continue
        black, white, to_move = got
        positions.append(
            {"id": f"rand-{made:02d}-p{plies}", "black": black, "white": white,
             "to_move": to_move}
        )
        made += 1
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    positions = collect_positions()
    print(f"{len(positions)} positions", file=sys.stderr)
    engine = HeuristicMCTS(sims=100, c_puct=1.5, rng=None, leaf_mode="gradient")
    out_positions: list[dict] = []
    for i, pos in enumerate(positions):
        board = _board_from_stones(pos["black"], pos["white"])
        to_move = pos["to_move"]
        flat = board.reshape(-1).tolist()
        empty = [m for m in range(225) if flat[m] == 0]
        v1_total: dict[str, int] = {}
        v1_attack: dict[str, int] = {}
        for m in empty:
            a, d = scoring._score_parts_flat(flat, m, to_move, 1.0)
            v1_total[str(m)] = a + d
            v1_attack[str(m)] = a
        black_atk, white_atk = kernel_numpy.attack_both(board)
        atk_black = {str(m): int(black_atk[m // 15, m % 15]) for m in empty}
        atk_white = {str(m): int(white_atk[m // 15, m % 15]) for m in empty}
        move = engine.choose_move(board, to_move)
        out_positions.append(
            {
                "id": pos["id"],
                "black": pos["black"],
                "white": pos["white"],
                "to_move": to_move,
                "v1_total": v1_total,
                "v1_attack": v1_attack,
                "attack_black": atk_black,
                "attack_white": atk_white,
                "v7_move": [move % 15, move // 15],
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(positions)}", file=sys.stderr)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "_meta": {
                    "description": (
                        "Frozen Python reference for the TypeScript v7 port. "
                        "Generated by ml/scripts/gen_v7_ts_fixtures.py. "
                        "v1_total/v1_attack: scoring._score_parts_flat "
                        "(defence_weight 1.0) for to_move. attack_black/white: "
                        "kernel_numpy.attack_both. v7_move: HeuristicMCTS "
                        "(100 sims, c_puct 1.5, leaf_mode gradient, rng=None) "
                        "choose_move. Scores keyed by flat cell index "
                        "(y*15+x), empty cells only."
                    )
                },
                "positions": out_positions,
            },
            f,
        )
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

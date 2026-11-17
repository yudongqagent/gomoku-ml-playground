#!/usr/bin/env python3
"""Verify the ported v18 rollout leaf against the research/batch-7 original.

Two modes:

* ``--generate positions.json``: build a deterministic position set
  (v7 self-play positions, random-playout positions, both tactical
  suites) and write boards + side-to-move to JSON.
* ``--eval positions.json --out values.json``: evaluate
  ``gomoku_ml.mcts_heuristic._rollout_leaf_value`` on every position
  with the ``gomoku_ml`` importable from ``--ml-dir`` (default: this
  repo's ml/) and write the values.

Run the eval once per worktree (this branch + the research/batch-7
worktree) and diff the outputs — the port is exact iff the values
agree bit-for-bit on every position.

Usage:
    python ml/scripts/verify_rollout_port.py --generate /tmp/pos.json
    python ml/scripts/verify_rollout_port.py --eval /tmp/pos.json --out /tmp/val-h1.json
    python ml/scripts/verify_rollout_port.py --eval /tmp/pos.json --out /tmp/val-b7.json \\
        --ml-dir ~/workspace/projects/gomoku-b7/ml
    python ml/scripts/verify_rollout_port.py --compare /tmp/val-h1.json /tmp/val-b7.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def _load_ml(ml_dir: str | None):
    if ml_dir:
        ml_dir = os.path.abspath(os.path.expanduser(ml_dir))
        if ml_dir not in sys.path:
            sys.path.insert(0, ml_dir)
    else:
        common.ensure_ml_on_path()
    from gomoku_ml import game, mcts_heuristic, scoring  # noqa: E402
    return game, mcts_heuristic, scoring


def generate(out_path: str, seed: int = 777) -> dict:
    common.ensure_ml_on_path()
    from gomoku_ml import evaluate, game, scoring  # noqa: E402
    import numpy as np  # noqa: E402

    rng = np.random.default_rng(seed)
    positions: list[dict] = []

    def add(board, player, source):
        positions.append({
            "board": board.astype(int).tolist(),
            "player": int(player),
            "source": source,
        })

    # v7 self-play positions: 3 games at 30 sims, every ply.
    for g in range(3):
        pa = scoring.get_player(7, seed=seed + g)
        pb = scoring.get_player(7, seed=seed + 100 + g)
        board = game.empty_board()
        player = game.BLACK
        while True:
            over, _win = game.game_over(board)
            if over:
                break
            add(board, player, f"v7-game{g}")
            chooser = pa.choose_move if player == game.BLACK else pb.choose_move
            board = game.apply_move(board, chooser(board, player), player)
            player = -player

    # Random-playout positions: 4 random games, every 2nd ply.
    for g in range(4):
        board = game.empty_board()
        player = game.BLACK
        ply = 0
        while True:
            over, _win = game.game_over(board)
            if over:
                break
            if ply % 2 == 0:
                add(board, player, f"random-game{g}")
            empties = np.flatnonzero(board.reshape(-1) == game.EMPTY)
            board = game.apply_move(
                board, int(rng.choice(empties)), player)
            player = -player
            ply += 1

    # Both tactical suites (player to move as specified).
    here = os.path.dirname(os.path.abspath(__file__))
    for name, path in (
        ("old", os.path.join(here, "..", "gomoku_ml", "tactical_positions.json")),
        ("hard", os.path.join(here, "..", "gomoku_ml", "tactical_positions_hard.json")),
    ):
        suite = evaluate.load_tactical_suite(path)
        for pos in suite["positions"]:
            board = game.empty_board()
            for x, y in pos.get("black", []):
                board[y, x] = game.BLACK
            for x, y in pos.get("white", []):
                board[y, x] = game.WHITE
            add(board, int(pos["to_move"]),
                f"tactical-{name}-{pos['id']}")

    payload = {"seed": seed, "positions": positions}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"wrote {len(positions)} positions to {out_path}", file=sys.stderr)
    return payload


def evaluate_positions(positions_path: str, out_path: str, ml_dir: str | None) -> dict:
    game, mcts_heuristic, _scoring = _load_ml(ml_dir)
    import numpy as np  # noqa: E402

    with open(positions_path, encoding="utf-8") as f:
        payload = json.load(f)
    values = []
    for pos in payload["positions"]:
        board = np.array(pos["board"], dtype=np.int8)
        values.append(mcts_heuristic._rollout_leaf_value(board, pos["player"]))
    result = {"positions": positions_path, "ml_dir": ml_dir or "self",
              "values": values}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(f"wrote {len(values)} values to {out_path}", file=sys.stderr)
    return result


def compare(a_path: str, b_path: str) -> int:
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(b_path, encoding="utf-8") as f:
        b = json.load(f)
    va, vb = a["values"], b["values"]
    assert len(va) == len(vb), f"length mismatch {len(va)} vs {len(vb)}"
    mismatches = [(i, x, y) for i, (x, y) in enumerate(zip(va, vb)) if x != y]
    print(f"{len(va)} positions, {len(mismatches)} mismatches")
    for i, x, y in mismatches[:10]:
        print(f"  pos {i}: h1={x} b7={y}")
    return 0 if not mismatches else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", metavar="PATH",
                        help="generate the position set to PATH")
    parser.add_argument("--eval", metavar="PATH",
                        help="evaluate the position set at PATH")
    parser.add_argument("--out", metavar="PATH", help="output JSON path")
    parser.add_argument("--ml-dir", default=None,
                        help="ml/ dir to import gomoku_ml from")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="compare two value JSONs")
    args = parser.parse_args()
    if args.generate:
        generate(args.generate)
        return 0
    if args.eval:
        if not args.out:
            parser.error("--eval requires --out")
        evaluate_positions(args.eval, args.out, args.ml_dir)
        return 0
    if args.compare:
        return compare(*args.compare)
    parser.error("need one of --generate, --eval, --compare")
    return 2


if __name__ == "__main__":
    sys.exit(main())

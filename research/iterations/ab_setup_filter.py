#!/usr/bin/env python3
"""A/B-validate the window-filtered setups against the slow exact
versions (batch 11 H1 mining safety check).

Two checks, both requiring 0 mismatches:
1. INNER: _setup_creates_two(board, m, player) vs the slow
   definition (no winning move after m AND >= 2 creators after m,
   via targeted_playout._creators) on (board, m) pairs sampled
   from the corpus, restricted to boards where the player holds 0
   creators (the precondition).
2. DIRECT: _setups_limited (window-filtered) vs slow _setups on
   sampled boards (both colours, precondition boards only).

Resumable: results are written per board to the --out JSON; a
relaunch skips completed boards. Writes
research/iterations/setup-mining-ab.json by default.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, scoring  # noqa: E402
from targeted_playout import _creators, _setups  # noqa: E402
import mine_setup_signature as mine  # noqa: E402

B10 = "/home/hatch/workspace/projects/gomoku-b10/research/iterations"
FILES = [
    "batch-v33-vs-v27.json", "batch-v34-vs-v27.json",
    "batch-v35-vs-v27.json", "batch-v33-vs-v27-swap.json",
    "batch-v34-vs-v27-swap.json", "batch-v35-vs-v27-swap.json",
    "batch-v27-vs-v7-swap.json",
]
SEED = 777


def _sample_boards(n: int, rng: random.Random) -> list:
    boards = []
    for fname in FILES:
        with open(os.path.join(B10, fname), encoding="utf-8") as f:
            batch = json.load(f)
        recs = batch["records"]
        for _ in range(n // len(FILES) + 1):
            rec = rng.choice(recs)
            moves = rec["moves"]
            if len(moves) < 12:
                continue
            ply = rng.randrange(10, len(moves))
            board = game.empty_board()
            for i, m in enumerate(moves[:ply]):
                board = game.apply_move(
                    board, m, game.BLACK if i % 2 == 0 else game.WHITE)
            boards.append((fname, rec["game"], ply, board))
    return boards[:n]


def _save(out: str, res: dict) -> None:
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "setup-mining-ab.json"))
    parser.add_argument("--inner-boards", type=int, default=60)
    parser.add_argument("--direct-boards", type=int, default=25)
    args = parser.parse_args()

    res: dict = {
        "inner": {"compared": 0, "mismatches": 0, "cases": [],
                  "boards_done": []},
        "direct": {"compared": 0, "mismatches": 0, "cases": [],
                   "boards_done": []},
        "timing": {},
    }
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            res = json.load(f)
        print(f"resuming: inner {len(res['inner']['boards_done'])} "
              f"boards, direct {len(res['direct']['boards_done'])} "
              f"boards done", file=sys.stderr, flush=True)
    # Separate RNG streams per phase so a resume reproduces the
    # same board lists exactly (a shared stream would diverge when
    # done boards are skipped).
    rng_boards_inner = random.Random(SEED)
    rng_boards_direct = random.Random(SEED + 1)
    rng_moves = random.Random(SEED + 2)

    # 1. INNER.
    t0 = time.time()
    inner_done = {tuple(b) for b in res["inner"]["boards_done"]}
    for fname, game_idx, ply, board in _sample_boards(
            args.inner_boards, rng_boards_inner):
        key = (fname, game_idx, ply)
        if key in inner_done:
            continue
        for player in (game.BLACK, game.WHITE):
            if _creators(board, player):
                continue  # precondition: 0 creators on input
            windowed = mine._window_candidates(board, player, 2)
            others = [m for m in scoring._candidate_moves(board)
                      if m not in set(windowed)]
            test_ms = (rng_moves.sample(windowed,
                                      min(25, len(windowed)))
                       + rng_moves.sample(others,
                                          min(10, len(others))))
            for m in test_ms:
                b2 = game.apply_move(board, m, player)
                slow = (not scoring._winning_moves(b2, player)
                        and len(_creators(b2, player)) >= 2)
                fast = mine._setup_creates_two(board, m, player)
                res["inner"]["compared"] += 1
                if slow != fast:
                    res["inner"]["mismatches"] += 1
                    res["inner"]["cases"].append(
                        [fname, game_idx, ply, player, m % 15, m // 15,
                         slow, fast])
                    print(f"  INNER MISMATCH {fname} g{game_idx} ply "
                          f"{ply} player {player} move "
                          f"({m % 15},{m // 15}): slow {slow} fast "
                          f"{fast}", file=sys.stderr, flush=True)
        res["inner"]["boards_done"].append(list(key))
        _save(args.out, res)
        print(f"  inner board {fname} g{game_idx} ply {ply}: "
              f"{res['inner']['compared']} compared",
              file=sys.stderr, flush=True)
    res["timing"]["inner_s"] = round(time.time() - t0, 1)
    _save(args.out, res)

    # 2. DIRECT.
    t0 = time.time()
    direct_done = {tuple(b) for b in res["direct"]["boards_done"]}
    for fname, game_idx, ply, board in _sample_boards(
            args.direct_boards, rng_boards_direct):
        key = (fname, game_idx, ply)
        if key in direct_done:
            continue
        for player in (game.BLACK, game.WHITE):
            if _creators(board, player):
                continue
            slow = _setups(board, player)
            fast = mine._setups_limited(board, player, 0)
            res["direct"]["compared"] += 1
            if sorted(slow) != sorted(fast):
                res["direct"]["mismatches"] += 1
                res["direct"]["cases"].append(
                    [fname, game_idx, ply, player,
                     sorted(slow), sorted(fast)])
                print(f"  DIRECT MISMATCH {fname} g{game_idx} ply "
                      f"{ply} player {player}: slow {sorted(slow)} "
                      f"fast {sorted(fast)}", file=sys.stderr,
                      flush=True)
        res["direct"]["boards_done"].append(list(key))
        _save(args.out, res)
        print(f"  direct board {fname} g{game_idx} ply {ply}",
              file=sys.stderr, flush=True)
    res["timing"]["direct_s"] = round(time.time() - t0, 1)
    _save(args.out, res)
    print(json.dumps(res, indent=2))
    ok = (res["inner"]["mismatches"] == 0
          and res["direct"]["mismatches"] == 0)
    print("A/B " + ("PASS" if ok else "FAIL"), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

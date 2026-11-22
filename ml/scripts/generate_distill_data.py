#!/usr/bin/env python3
"""Generate + label distillation data for ML batch 1 H1 (v24).

Two phases (run separately so labeling can be re-run without
replaying games):

* ``games``: play self-play games (v7, v18, or random) and append
  full move lists to a JSON file (flushed after every game, resume
  safe: re-running with the same --games target only plays the
  games still missing). The games file is the H3-reusable artifact
  (v18 games carry full move lists for the policy-prior data).
* ``label``: reconstruct every position of every game, keep QUIET
  positions only (``_terminal_leaf_value == 0`` — the net is only
  ever consulted in quiet positions at inference), label each with
  the v18 rollout-leaf value (``_rollout_leaf_value``, blend=False),
  and write one .npz shard per game (planes/policy/value, the
  ``train.load_shards`` format). Resume safe: existing shards are
  skipped. ``--label-mode rollout2`` (ML batch 3 H2, v37) labels
  with the 2-ply value instead (``_rollout2_leaf_value`` — the
  rollout-leaf value 2 plies ahead under the teacher's own rollout
  policy; see ``research/iterations/iter-37.md`` §1); the default
  ``rollout`` reproduces v24's labels exactly. ``--label-mode
  rollout3`` (ML batch 4 H3, v44) labels with the 3-ply value
  (``_rollout3_leaf_value`` — the rollout-leaf value 3 plies
  ahead; see ``research/iterations/iter-44.md`` §1).
  ``--label-mode rollout4`` (ML batch 5 H2, v49) labels with the
  4-ply value (``_rollout4_leaf_value`` — the rollout-leaf value
  4 plies ahead; see ``research/iterations/iter-49.md`` §1).

Sampling: ALL quiet positions of every game (positions per game
~35-45; game generation is the expensive step, labeling is ~6ms
per position, so every quiet position is kept). Policy targets are
one-hot on the move actually played (valid rows summing to 1;
unused at policy_weight=0).

Usage:
    python ml/scripts/generate_distill_data.py games --source v7 \\
        --games 500 --sims 50 --seed 50000 \\
        --out ml/runs/ml1-h1/games-v7.json
    python ml/scripts/generate_distill_data.py label \\
        --games ml/runs/ml1-h1/games-v7.json \\
        --out-dir ml/runs/ml1-h1/data/v7 --prefix v7
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


def _load_games(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"games": [], "complete": False}


def _write_games(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
    os.replace(tmp, path)


def cmd_games(args) -> int:
    import numpy as np  # noqa: E402
    from gomoku_ml import game, scoring  # noqa: E402

    payload = _load_games(args.out)
    existing = len(payload["games"])
    if existing >= args.games:
        print(f"{args.out} already has {existing} games "
              f"(target {args.games}); nothing to do", file=sys.stderr)
        return 0
    print(f"== {args.source} self-play: games {existing}..{args.games} "
          f"({args.sims} sims, seed base {args.seed}) ==",
          file=sys.stderr)
    t0 = time.time()
    for i in range(existing, args.games):
        board = game.empty_board()
        player = game.BLACK
        moves: list[int] = []
        if args.source == "random":
            rng = np.random.default_rng(args.seed + i)
            chooser = None
        else:
            version = 7 if args.source == "v7" else 18
            pa = scoring.get_player(version, seed=args.seed + i)
            pb = scoring.get_player(version, seed=args.seed + 100000 + i)
            pa._mcts.sims = args.sims
            pb._mcts.sims = args.sims
            chooser = (pa, pb)
        while True:
            over, win = game.game_over(board)
            if over:
                break
            if args.source == "random":
                empties = np.flatnonzero(board.reshape(-1) == game.EMPTY)
                move = int(rng.choice(empties))
            else:
                p = chooser[0] if player == game.BLACK else chooser[1]
                move = p.choose_move(board, player)
            moves.append(int(move))
            board = game.apply_move(board, move, player)
            player = -player
        payload["games"].append({
            "game": i,
            "source": args.source,
            "sims": args.sims,
            "seed": args.seed + i,
            "moves": moves,
            "winner": int(win),
            "plies": len(moves),
        })
        payload["complete"] = (i + 1 == args.games)
        _write_games(args.out, payload)
        if (i + 1) % 10 == 0 or i + 1 == args.games:
            el = time.time() - t0
            done = i + 1 - existing
            print(f"  game {i}: {len(moves)} plies, "
                  f"winner {win} ({done} games in {el:.0f}s, "
                  f"{el / done:.1f}s/game)", file=sys.stderr, flush=True)
    print(f"  done: {args.games} games in {time.time() - t0:.0f}s",
          file=sys.stderr)
    return 0


def cmd_label(args) -> int:
    import numpy as np  # noqa: E402
    from gomoku_ml import encoding, game, mcts_heuristic  # noqa: E402

    with open(args.games, encoding="utf-8") as f:
        payload = json.load(f)
    os.makedirs(args.out_dir, exist_ok=True)
    if args.holdout_dir:
        os.makedirs(args.holdout_dir, exist_ok=True)
    label_fn = (
        mcts_heuristic._rollout4_leaf_value
        if args.label_mode == "rollout4"
        else mcts_heuristic._rollout3_leaf_value
        if args.label_mode == "rollout3"
        else mcts_heuristic._rollout2_leaf_value
        if args.label_mode == "rollout2"
        else mcts_heuristic._rollout_leaf_value
    )
    total_pos = total_quiet = 0
    held_out = 0
    label_hist: dict[str, int] = {"terminal_pm1": 0, "gradient": 0}
    t0 = time.time()
    for rec in payload["games"]:
        # Game-level holdout (v8's protocol): every Nth game is
        # held out of training entirely, so the holdout measures
        # generalisation to unseen games, not near-duplicate plies
        # of a seen game (a random sample-level split leaks those).
        is_holdout = (
            args.holdout_dir is not None
            and args.holdout_every > 0
            and (rec["game"] + 1) % args.holdout_every == 0
        )
        dest_dir = args.holdout_dir if is_holdout else args.out_dir
        shard = os.path.join(
            dest_dir, f"{args.prefix}_game{rec['game']:04d}.npz")
        if os.path.exists(shard):
            if is_holdout:
                held_out += 1
            continue
        moves = rec["moves"]
        # Reconstruct the board before every ply + the history list.
        boards: list[np.ndarray] = [game.empty_board()]
        player = game.BLACK
        for m in moves:
            boards.append(game.apply_move(boards[-1], m, player))
            player = -player
        planes_list, policy_list, value_list = [], [], []
        for t in range(len(moves)):
            board = boards[t]
            p = game.BLACK if t % 2 == 0 else game.WHITE
            total_pos += 1
            if mcts_heuristic._terminal_leaf_value(board, p) != 0.0:
                continue
            label = label_fn(board, p)
            history = boards[:t]  # boards before each earlier move
            last_move = moves[t - 1] if t > 0 else None
            planes_list.append(
                encoding.encode(board, p, history, last_move))
            policy = np.zeros(225, dtype=np.float32)
            policy[moves[t]] = 1.0
            policy_list.append(policy)
            value_list.append(label)
            total_quiet += 1
            if label in (1.0, -1.0):
                label_hist["terminal_pm1"] += 1
            else:
                label_hist["gradient"] += 1
        if planes_list:
            np.savez_compressed(
                shard,
                planes=np.stack(planes_list).astype(np.float32),
                policy=np.stack(policy_list).astype(np.float32),
                value=np.array(value_list, dtype=np.float32),
                moves=np.array(moves, dtype=np.int64),
                winner=np.array(rec["winner"], dtype=np.int64),
            )
            if is_holdout:
                held_out += 1
        if (rec["game"] + 1) % 50 == 0:
            print(f"  labeled through game {rec['game']} "
                  f"({total_quiet} quiet / {total_pos} positions, "
                  f"{time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    print(f"  done: {total_quiet} quiet positions from {total_pos} "
          f"total ({label_hist['terminal_pm1']} rollout-terminal ±1, "
          f"{label_hist['gradient']} gradient), {held_out} games held "
          f"out, in {time.time() - t0:.0f}s",
          file=sys.stderr)
    print("LABEL_JSON " + json.dumps({
        "games_file": args.games, "out_dir": args.out_dir,
        "holdout_dir": args.holdout_dir,
        "label_mode": args.label_mode,
        "total_positions": total_pos, "quiet_positions": total_quiet,
        "held_out_games": held_out,
        "label_hist": label_hist,
        "elapsed_seconds": round(time.time() - t0, 1),
    }))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    pg = sub.add_parser("games", help="play self-play games")
    pg.add_argument("--source", choices=["v7", "v18", "random"],
                    required=True)
    pg.add_argument("--games", type=int, required=True)
    pg.add_argument("--sims", type=int, default=50)
    pg.add_argument("--seed", type=int, required=True)
    pg.add_argument("--out", required=True)
    pg.set_defaults(func=cmd_games)
    pl = sub.add_parser("label", help="label games into .npz shards")
    pl.add_argument("--games", required=True)
    pl.add_argument("--out-dir", required=True)
    pl.add_argument("--prefix", required=True)
    pl.add_argument("--holdout-dir", default=None,
                    help="dir for held-out game shards (game-level holdout)")
    pl.add_argument("--holdout-every", type=int, default=0,
                    help="hold out every Nth game (0 = no holdout)")
    pl.add_argument("--label-mode",
                    choices=["rollout", "rollout2", "rollout3", "rollout4"],
                    default="rollout",
                    help="rollout: v24's _rollout_leaf_value labels; "
                    "rollout2: v37's _rollout2_leaf_value labels; "
                    "rollout3: v44's _rollout3_leaf_value labels; "
                    "rollout4: v49's _rollout4_leaf_value labels")
    pl.set_defaults(func=cmd_label)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

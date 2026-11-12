"""Self-play game generation.

Plays full games with MCTS and records training samples:

* ``planes``: [8, 15, 15] float32 input tensor (see encoding.py).
* ``policy``: [T, 225] float32 visit-count distribution from the MCTS root.
* ``value``: scalar outcome z from the perspective of the player to move
  at that position: +1 if that player eventually won, -1 if they lost,
  0 for a draw (or a resigned game counted as a loss for the resigner).

Games are saved as numpy ``.npz`` shards under ``ml/runs/<run>/data/``
(or any directory passed in). Each shard holds one game.

Long-run support (see ml/README.md "Long runs"):

* **Live feed**: with ``live_path`` set, ``current_game.json`` in the
  run dir is rewritten after EVERY move (atomic temp + rename), so the
  Trainer UI can watch the game in progress. Schema:
  ``{"game": n, "moves": [{"x", "y"}], "to_move": 1|-1,
  "status": "playing"|"finished", "winner": 1|-1|0|null}``.
* **Data manifest**: ``data/manifest.json`` lists every shard with its
  sample count and is updated after every game. Shards on disk ARE the
  replay buffer — in-memory buffer contents are not persisted, so a
  resumed run reuses the shards listed in the manifest instead of
  regenerating them.
* **Resume-safe numbering**: shard names continue after the highest
  existing ``<run>_gameNNNN.npz`` in the output directory, so a
  resumed run never overwrites earlier games.
* **Graceful interrupt**: with a ``stop_flag`` (see interrupt.py),
  the in-progress game is abandoned at the next move boundary and
  generation stops after writing the manifest. Completed games are
  never lost: each shard is flushed to disk as soon as its game ends.
* **Throughput**: every ``selfplay_game`` event carries
  ``games_per_hour`` (running rate since generation started).

Single-process v1: games run sequentially. Multiprocessing / batched
MCTS across parallel games is future work — see ml/README.md.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from typing import Callable

import numpy as np

from . import encoding, game
from .mcts import MCTS

_SHARD_RE = re.compile(r"^(?P<run>.+)_game(?P<idx>\d+)\.npz$")


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON atomically (temp file + rename) so a reader never
    sees a half-written file — the Trainer UI polls this file."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_manifest(data_dir: str) -> dict | None:
    """Load ``data/manifest.json`` if it exists, else None."""
    path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def update_manifest(data_dir: str) -> dict:
    """Rebuild ``data/manifest.json`` from the shards on disk: every
    ``.npz`` file with its sample count, plus totals. Returns the
    manifest dict. This is the source of truth for what data a run
    has — a resumed run reads it instead of regenerating games."""
    os.makedirs(data_dir, exist_ok=True)
    shards: list[dict] = []
    total_samples = 0
    for path in sorted(glob.glob(os.path.join(data_dir, "*.npz"))):
        try:
            with np.load(path) as z:
                samples = int(z["planes"].shape[0])
                moves = int(z["moves"].shape[0])
                winner = int(z["winner"])
        except Exception:
            continue
        shards.append({
            "file": os.path.basename(path),
            "samples": samples,
            "moves": moves,
            "winner": winner,
        })
        total_samples += samples
    manifest = {
        "shards": shards,
        "games": len(shards),
        "total_samples": total_samples,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _atomic_write_json(os.path.join(data_dir, "manifest.json"), manifest)
    return manifest


def existing_game_count(out_dir: str, run_name: str) -> int:
    """How many ``<run_name>_gameNNNN.npz`` shards already exist in
    ``out_dir`` (the next game index is this count when numbering is
    dense; the max index + 1 in general)."""
    highest = -1
    for path in glob.glob(os.path.join(out_dir, f"{run_name}_game*.npz")):
        m = _SHARD_RE.match(os.path.basename(path))
        if m and m.group("run") == run_name:
            highest = max(highest, int(m.group("idx")))
    return highest + 1


def play_game(
    mcts: MCTS,
    num_simulations: int,
    temperature_moves: int = 30,
    temperature: float = 1.0,
    resign_threshold: float | None = None,
    max_moves: int = game.SIZE * game.SIZE,
    add_noise: bool = True,
    on_move: Callable[[dict], None] | None = None,
    stop_flag=None,
) -> dict[str, np.ndarray]:
    """Play one self-play game. Returns a dict of sample arrays:

    * ``planes``: [T, 8, 15, 15] float32
    * ``policy``: [T, 225] float32 visit distributions
    * ``value``:  [T] float32 outcomes from the side-to-move perspective
    * ``moves``:  [T] int64 flat move indices actually played
    * ``winner``: scalar int (BLACK/WHITE/EMPTY for draw)
    * ``aborted``: scalar bool — True when ``stop_flag`` was set
      mid-game (the game has no outcome; the caller must discard it).

    ``on_move`` is called after every move with
    ``{"ply", "move", "x", "y", "player"}`` (used for the live feed).
    ``stop_flag`` is an :class:`interrupt.InterruptFlag` (or any object
    with a ``requested`` attribute); when set, the game stops at the
    next move boundary and is marked aborted.
    """
    board = game.empty_board()
    history: list[np.ndarray] = []
    last_move: int | None = None
    player = game.BLACK
    planes_list: list[np.ndarray] = []
    policy_list: list[np.ndarray] = []
    players_list: list[int] = []
    moves_list: list[int] = []
    winner = game.EMPTY
    aborted = False

    for ply in range(max_moves):
        if stop_flag is not None and stop_flag.requested:
            aborted = True
            break
        over, win = game.game_over(board)
        if over:
            winner = win
            break
        temp = temperature if ply < temperature_moves else 0.0
        root, root_value = mcts.search(
            board, player, num_simulations,
            history=history, last_move=last_move, add_noise=add_noise,
        )
        if (
            resign_threshold is not None
            and ply >= 10
            and root_value < resign_threshold
        ):
            # Player to move resigns: the opponent wins.
            winner = -player
            break
        planes_list.append(encoding.encode(board, player, history, last_move))
        policy_list.append(MCTS.visit_distribution(root, temperature=temp))
        players_list.append(player)
        move = MCTS.choose_move(root, temperature=temp)
        moves_list.append(move)
        history.append(board)
        board = game.apply_move(board, move, player)
        last_move = move
        if on_move is not None:
            x, y = game.move_to_xy(move)
            on_move({"ply": ply, "move": int(move), "x": x, "y": y, "player": player})
        player = -player
    else:
        _, winner = game.game_over(board)

    values = np.zeros(len(players_list), dtype=np.float32)
    for i, p in enumerate(players_list):
        if winner == game.EMPTY:
            values[i] = 0.0
        else:
            values[i] = 1.0 if winner == p else -1.0

    return {
        "planes": np.stack(planes_list).astype(np.float32) if planes_list else np.zeros((0, 8, 15, 15), dtype=np.float32),
        "policy": np.stack(policy_list).astype(np.float32) if policy_list else np.zeros((0, 225), dtype=np.float32),
        "value": values,
        "moves": np.array(moves_list, dtype=np.int64),
        "winner": np.array(winner, dtype=np.int64),
        "aborted": np.array(aborted, dtype=bool),
    }


def generate_games(
    mcts: MCTS,
    num_games: int,
    num_simulations: int,
    out_dir: str,
    run_name: str = "run",
    on_event: Callable[[dict], None] | None = None,
    live_path: str | None = None,
    stop_flag=None,
    checkpoint_every_games: int = 0,
    **game_kwargs,
) -> list[str]:
    """Generate ``num_games`` self-play games, saving one ``.npz`` shard
    per game into ``out_dir``. Returns the list of shard paths written
    by this call.

    Numbering continues after existing shards in ``out_dir`` (resume
    safe). After every game the data manifest is updated; every
    ``checkpoint_every_games`` games (0 = every game) a
    ``selfplay_state.json`` snapshot is written next to the manifest.

    ``live_path``: when given, the current game is written there after
    every move (see module docstring). ``stop_flag``: when set, the
    in-progress game is abandoned and generation stops after the
    manifest update — check ``stop_flag.requested`` afterwards to
    tell an interrupted run from a completed one.

    Human-readable progress goes to stderr; machine-readable progress
    (``{"event": "selfplay_game", ...}``) is passed to ``on_event`` when
    given (the CLI forwards it to the run logger: stdout JSON + metrics.jsonl).
    """
    os.makedirs(out_dir, exist_ok=True)
    start_index = existing_game_count(out_dir, run_name)
    total_target = start_index + num_games
    paths: list[str] = []
    started = time.time()
    games_done = 0
    for i in range(num_games):
        game_number = start_index + i + 1  # 1-based, absolute in the run dir
        moves_so_far: list[dict] = []

        def on_move(rec: dict, _game=game_number, _moves=moves_so_far) -> None:
            _moves.append({"x": rec["x"], "y": rec["y"]})
            if live_path is not None:
                _atomic_write_json(live_path, {
                    "game": _game,
                    "moves": list(_moves),
                    "to_move": -rec["player"],
                    "status": "playing",
                    "winner": None,
                })

        t0 = time.time()
        samples = play_game(
            mcts, num_simulations,
            on_move=on_move if live_path is not None else None,
            stop_flag=stop_flag,
            **game_kwargs,
        )
        if bool(samples["aborted"]):
            if live_path is not None:
                _atomic_write_json(live_path, {
                    "game": game_number,
                    "moves": moves_so_far,
                    "to_move": game.BLACK if len(moves_so_far) % 2 == 0 else game.WHITE,
                    "status": "finished",
                    "winner": None,
                })
            print(
                f"  game {game_number} aborted by interrupt "
                f"(partial game discarded)",
                file=sys.stderr, flush=True,
            )
            break
        path = os.path.join(out_dir, f"{run_name}_game{start_index + i:04d}.npz")
        np.savez_compressed(path, **{k: v for k, v in samples.items() if k != "aborted"})
        paths.append(path)
        games_done += 1
        winner_name = (
            "Black" if samples["winner"] == 1
            else "White" if samples["winner"] == -1 else "draw"
        )
        elapsed = time.time() - t0
        total_elapsed = time.time() - started
        games_per_hour = games_done / total_elapsed * 3600 if total_elapsed > 0 else 0.0
        if live_path is not None:
            _atomic_write_json(live_path, {
                "game": game_number,
                "moves": [
                    {"x": int(m) % game.SIZE, "y": int(m) // game.SIZE}
                    for m in samples["moves"]
                ],
                "to_move": game.BLACK if len(samples["moves"]) % 2 == 0 else game.WHITE,
                "status": "finished",
                "winner": int(samples["winner"]),
            })
        manifest = update_manifest(out_dir)
        if checkpoint_every_games and games_done % checkpoint_every_games == 0:
            _atomic_write_json(os.path.join(out_dir, "selfplay_state.json"), {
                "games_completed_this_call": games_done,
                "games_total_target": total_target,
                "games_in_manifest": manifest["games"],
                "total_samples": manifest["total_samples"],
                "games_per_hour": round(games_per_hour, 2),
                "elapsed_seconds": round(total_elapsed, 1),
            })
        print(
            f"  game {game_number}/{total_target}: {len(samples['moves'])} moves, "
            f"winner={winner_name}, {elapsed:.1f}s "
            f"({games_per_hour:.1f} games/hour)",
            file=sys.stderr, flush=True,
        )
        if on_event is not None:
            on_event({
                "event": "selfplay_game",
                "game": game_number,
                "games_total": total_target,
                "moves": len(samples["moves"]),
                "winner": winner_name,
                "seconds": round(elapsed, 2),
                "games_per_hour": round(games_per_hour, 2),
                "path": path,
            })
        if stop_flag is not None and stop_flag.requested:
            break
    return paths

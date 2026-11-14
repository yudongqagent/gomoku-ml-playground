"""Heuristic-vs-heuristic game generation (batch 4 H3, v11).

Plays games between two 1-ply scoring engines (default v1 vs the
frozen baseline v0) and records them as ``.npz`` shards in the SAME
schema as neural self-play (``selfplay.py``): planes / policy /
value / moves / winner — so ``train.load_shards`` and
``eval_value_holdout`` consume them unchanged. The policy target is
one-hot on the played move for both sides (neither side has a visit
distribution); only the value head is used by ValueMCTS, so the
one-hot policy is a harmless training target.

Why this exists: the v11 value net must train on games against
heuristic opponents (iter-8 §7 finding 1 — a net trained on weak
neural self-play is unreliable in the distribution heuristic
opponents actually play). The round-2 neural net cannot generate
those games: it loses to the heuristic in ~10 plies, so its games
contain only opening positions (see
``research/iterations/iter-11.md`` §1.1a). v1-vs-v0 games are
competitive (0.585) and full-length (mean 76 plies), and both
engines are ~ms/move, so thousands of games take about an hour.

Design (mirrors B1's parallel self-play where it applies):

* Workers are spawned, not forked; each writes its shards into its
  own shard dir under ``<out_dir>/.parallel-<pid>-<ts>/worker<N>/``
  and the parent merges them with
  :func:`gomoku_ml.selfplay.merge_worker_shards` (validated merge,
  dense resume-safe numbering). Orphaned dirs from an interrupted
  run are recovered by :func:`selfplay.recover_parallel_shards` on
  the next run — same contract as the neural path.
* Games are independent and reproducible: game ``g`` (0-based,
  global) uses fresh players with seeds derived from the game
  index, and colours alternate (even ``g``: version A plays Black,
  odd ``g``: version A plays White).
* After the merge, :func:`deduplicate_shards` removes duplicate
  move sequences (deterministic engines transpose — ~25% of
  v1-vs-v0 games repeat), keeps the first occurrence, renumbers
  densely and rebuilds the manifest. The holdout split is applied
  to the UNIQUE set.
"""

from __future__ import annotations

import glob
import multiprocessing
import os
import shutil
import sys
import time

import numpy as np

from . import baseline, encoding, game, scoring, selfplay


def _build_player(version: int, seed: int):
    """A 1-ply player for ``version`` (0 = frozen baseline, 1-5 =
    versioned scoring engine). MCTS versions are rejected: they are
    ~0.5 s/move, which defeats the purpose of this generator."""
    if version == 0:
        return baseline.HeuristicPlayer(seed=seed)
    entry = scoring.ENGINE_VERSIONS.get(version)
    if entry is None:
        raise ValueError(f"unknown scoring engine version: {version}")
    if entry.get("mcts"):
        raise ValueError(
            f"version {version} is an MCTS engine — too slow for "
            "heuristic game generation (use a 1-ply version)"
        )
    return scoring.get_player(version, seed=seed)


def _game_seeds(seed: int, game_index: int) -> tuple[int, int]:
    """Per-game seeds for versions A and B (deterministic, unique
    per game, independent across games)."""
    base = int(seed) * 1_000_000 + 2 * int(game_index)
    return base, base + 1


def play_heuristic_game(
    version_black: int,
    version_white: int,
    seed_black: int,
    seed_white: int,
    max_moves: int = 225,
) -> dict[str, np.ndarray]:
    """Play one game between two 1-ply engines; return shard arrays.

    Mirrors :class:`gomoku_ml.selfplay._BatchedGame` exactly: the
    position is encoded (with history) BEFORE the move, the policy
    target is one-hot on the played move, and the value target is
    the game outcome from the player-to-move perspective (+1 win /
    -1 loss / 0 draw). ``history`` holds the boards before each
    move, most recent last — the same convention
    :func:`gomoku_ml.encoding.encode` expects.
    """
    black = _build_player(version_black, seed_black)
    white = _build_player(version_white, seed_white)
    board = game.empty_board()
    history: list[np.ndarray] = []
    last_move: int | None = None
    player = game.BLACK
    planes: list[np.ndarray] = []
    policies: list[np.ndarray] = []
    players: list[int] = []
    moves: list[int] = []
    winner = game.EMPTY
    while True:
        over, win = game.game_over(board)
        if over:
            winner = win
            break
        if len(moves) >= max_moves:
            winner = game.EMPTY
            break
        chooser = black if player == game.BLACK else white
        planes.append(encoding.encode(board, player, history, last_move))
        move = int(chooser.choose_move(board, player))
        policy = np.zeros(225, dtype=np.float32)
        policy[move] = 1.0
        policies.append(policy)
        players.append(player)
        moves.append(move)
        history.append(board)
        board = game.apply_move(board, move, player)
        last_move = move
        player = -player
    values = np.zeros(len(players), dtype=np.float32)
    for i, p in enumerate(players):
        if winner == game.EMPTY:
            values[i] = 0.0
        else:
            values[i] = 1.0 if winner == p else -1.0
    return {
        "planes": (
            np.stack(planes).astype(np.float32)
            if planes
            else np.zeros((0, 8, 15, 15), dtype=np.float32)
        ),
        "policy": (
            np.stack(policies).astype(np.float32)
            if policies
            else np.zeros((0, 225), dtype=np.float32)
        ),
        "value": values,
        "moves": np.array(moves, dtype=np.int64),
        "winner": np.array(winner, dtype=np.int64),
    }


def deduplicate_shards(out_dir: str, run_name: str) -> dict:
    """Remove duplicate move sequences from ``out_dir`` (keep the
    first occurrence by shard index), renumber the survivors densely
    from 0, and rebuild the manifest. Returns stats."""
    paths = sorted(
        glob.glob(os.path.join(out_dir, f"{run_name}_game*.npz"))
    )
    seen: set[tuple[int, ...]] = set()
    keep: list[str] = []
    removed = 0
    for path in paths:
        with np.load(path) as z:
            seq = tuple(int(m) for m in z["moves"])
        if seq in seen:
            removed += 1
            os.remove(path)
        else:
            seen.add(seq)
            keep.append(path)
    # Renumber densely (via temp names — a survivor may already sit
    # at its target index, and os.replace onto itself is fine, but
    # a two-pass rename avoids any collision).
    tmp_paths = []
    for i, path in enumerate(keep):
        tmp = os.path.join(out_dir, f".dedup-tmp-{i:04d}.npz")
        os.replace(path, tmp)
        tmp_paths.append(tmp)
    for i, tmp in enumerate(tmp_paths):
        os.replace(tmp, os.path.join(out_dir, f"{run_name}_game{i:04d}.npz"))
    manifest = selfplay.update_manifest(out_dir)
    return {
        "games_before": len(paths),
        "duplicates_removed": removed,
        "games_after": len(keep),
        "total_samples": manifest["total_samples"],
    }


def _heuristic_worker(spec: dict) -> None:
    """Entry point for one worker process (spawn start method).

    Plays ``spec["game_indices"]`` (global 0-based indices) and
    writes one shard per game into ``spec["out_dir"]``. A worker
    killed mid-write can leave a truncated shard — the merge
    validation drops it (see :func:`selfplay.merge_worker_shards`).
    """
    import traceback

    result_queue = spec["result_queue"]
    try:
        os.makedirs(spec["out_dir"], exist_ok=True)
        for local_i, g in enumerate(spec["game_indices"]):
            seed_a, seed_b = _game_seeds(spec["seed"], g)
            if g % 2 == 0:
                arrays = play_heuristic_game(
                    spec["version_a"], spec["version_b"],
                    seed_a, seed_b, max_moves=spec["max_moves"],
                )
            else:
                arrays = play_heuristic_game(
                    spec["version_b"], spec["version_a"],
                    seed_b, seed_a, max_moves=spec["max_moves"],
                )
            path = os.path.join(
                spec["out_dir"], f"{spec['run_name']}_game{local_i:04d}.npz"
            )
            np.savez_compressed(path, **arrays)
        result_queue.put({"worker": spec["worker"], "games": len(spec["game_indices"])})
    except BaseException:
        result_queue.put({
            "worker": spec["worker"],
            "error": traceback.format_exc(),
        })
        raise


def generate_heuristic_games(
    out_dir: str,
    run_name: str = "run",
    num_games: int = 100,
    version_a: int = 1,
    version_b: int = 0,
    workers: int = 4,
    seed: int = 0,
    max_moves: int = 225,
    stop_flag=None,
    dedup: bool = True,
) -> dict:
    """Generate ``num_games`` heuristic-vs-heuristic games across
    ``workers`` processes and merge them into ``out_dir``.

    ``num_games`` is the number of NEW games to generate (the caller
    owns top-up: pass ``max(0, target - existing)``). Game indices
    are global and dense from the existing count, so a top-up run
    continues the alternation and the seed sequence. Returns stats
    (games merged, dedup counts, samples, games/hour, winner split).
    """
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if num_games < 0:
        raise ValueError(f"num_games must be >= 0, got {num_games}")
    # Validate the versions up front (fail fast, before spawning).
    _build_player(version_a, 0)
    _build_player(version_b, 0)
    os.makedirs(out_dir, exist_ok=True)
    recovered = selfplay.recover_parallel_shards(out_dir, run_name)
    if recovered["merged"]:
        print(
            f"  recovered {len(recovered['merged'])} orphaned shard(s) "
            "from an interrupted heuristic-games run",
            file=sys.stderr, flush=True,
        )
    start_index = selfplay.existing_game_count(out_dir, run_name)
    if num_games == 0:
        manifest = selfplay.load_manifest(out_dir) or {"games": 0, "total_samples": 0}
        return {
            "games_generated": 0,
            "games_total": manifest["games"],
            "total_samples": manifest["total_samples"],
        }
    started = time.time()
    indices = list(range(start_index, start_index + num_games))
    # Split as evenly as possible; the first workers get one extra.
    base, extra = divmod(num_games, workers)
    splits: list[list[int]] = []
    pos = 0
    for w in range(workers):
        n = base + (1 if w < extra else 0)
        splits.append(indices[pos:pos + n])
        pos += n

    shard_root = os.path.join(
        out_dir, f".parallel-{os.getpid()}-{int(time.time())}"
    )
    worker_dirs = [
        os.path.join(shard_root, f"worker{w}") for w in range(workers)
    ]
    for d in worker_dirs:
        os.makedirs(d, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    procs = []
    for w in range(workers):
        spec = {
            "worker": w,
            "out_dir": worker_dirs[w],
            "run_name": run_name,
            "game_indices": splits[w],
            "version_a": version_a,
            "version_b": version_b,
            "seed": int(seed),
            "max_moves": int(max_moves),
            "result_queue": result_queue,
        }
        proc = ctx.Process(target=_heuristic_worker, args=(spec,),
                           name=f"heuristic-worker-{w}")
        proc.start()
        procs.append(proc)

    interrupted = False
    while any(p.is_alive() for p in procs):
        if stop_flag is not None and stop_flag.requested and not interrupted:
            interrupted = True
            print(
                "  interrupt requested — terminating workers "
                "(completed shards are merged)",
                file=sys.stderr, flush=True,
            )
            for p in procs:
                if p.is_alive():
                    p.terminate()
            break
        time.sleep(0.2)
    for p in procs:
        p.join()

    errors: list[str] = []
    while True:
        try:
            item = result_queue.get_nowait()
        except Exception:
            break
        if "error" in item:
            errors.append(f"worker {item['worker']}:\n{item['error']}")
    for w, proc in enumerate(procs):
        if proc.exitcode not in (0, None) and not interrupted:
            errors.append(f"worker {w}: exit code {proc.exitcode}")
    result_queue.close()
    result_queue.join_thread()

    merge = selfplay.merge_worker_shards(out_dir, run_name, worker_dirs)
    shutil.rmtree(shard_root, ignore_errors=True)
    merged = len(merge["merged"])
    if merge["skipped"]:
        print(
            f"  dropped {len(merge['skipped'])} corrupt shard(s) "
            "(worker killed mid-write)",
            file=sys.stderr, flush=True,
        )
    stats: dict = {
        "games_generated": merged,
        "games_total": merge["manifest"]["games"],
        "total_samples": merge["manifest"]["total_samples"],
    }
    if dedup:
        stats["dedup"] = deduplicate_shards(out_dir, run_name)
        stats["games_total"] = stats["dedup"]["games_after"]
        stats["total_samples"] = stats["dedup"]["total_samples"]
    # Winner split over the final shards (cheap: manifest has winners).
    manifest = selfplay.load_manifest(out_dir)
    if manifest:
        winners = [s["winner"] for s in manifest["shards"]]
        stats["black_wins"] = winners.count(1)
        stats["white_wins"] = winners.count(-1)
        stats["draws"] = winners.count(0)
        stats["mean_plies"] = round(
            sum(s["moves"] for s in manifest["shards"]) / max(1, len(winners)), 1
        )
    elapsed = time.time() - started
    stats["elapsed_seconds"] = round(elapsed, 1)
    stats["games_per_hour"] = round(merged / elapsed * 3600, 1) if elapsed > 0 else 0.0
    print(
        f"  heuristic games: {merged}/{num_games} merged "
        f"({workers} workers) in {elapsed:.1f}s "
        f"({stats['games_per_hour']:.1f} games/hour)",
        file=sys.stderr, flush=True,
    )
    if errors:
        raise RuntimeError(
            "heuristic game generation finished with worker failures "
            f"({merged}/{num_games} games merged and saved):\n"
            + "\n".join(errors)
        )
    return stats

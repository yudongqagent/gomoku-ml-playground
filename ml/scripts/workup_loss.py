#!/usr/bin/env python3
"""Work up a lost game: find the losing move (if any) with a playout oracle.

Replays one lost game from a selfplay_batch.py JSON and, at each of
the loser's turns in a chosen ply window, asks: was this the move
that lost the game? The oracle is a playout to completion with the
SAME two engines as the match (fresh RNG seeds per playout).

CORRECTED METHOD (see research/iterations/iter-6.md §2.2(a) — the
first version of the v6 walk-back produced a WITHDRAWN finding by
comparing the alternative's playout against the HISTORICAL game
outcome): the ACTUAL move's playout must lose FIRST, at the same
seeds, before an alternative counts as saving the game. Playout
engines have fresh RNG state, so even the actual move's playout
diverges from the real game — a turn only qualifies when the actual
move loses the playout at >= 2 of the 3 seeds AND an alternative
draws or wins at >= 2 of the same 3 seeds.

Alternatives tested at a qualifying turn: the top-K moves by 1-ply
v1 score (excluding the actual move), the v8 MCTS choice at deeper
sims (``--deep-sims``, if different), and any move starting a
forced win by consecutive fours (VCF solver) if one exists.

Usage:
    python ml/scripts/workup_loss.py research/iterations/batch-v8-vs-v1.json \\
        --game 3 --version-a 8 --version-b 1 --seeds 0 1 2 \\
        --last-turns 8 --topk 5 --out /tmp/workup.json
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
from gomoku_ml import baseline, evaluate, game, research, scoring  # noqa: E402

# Threshold: a move "loses" / "saves" a turn when it does so at >= 2
# of the 3 playout seeds (a 1-2 split is noise, not a finding).
MIN_SEEDS = 2


def _build_player(version, seed: int):
    """A stateful player for engine version ``version`` (0 = the
    frozen baseline heuristic; ``"mcts"`` = the heuristic-MCTS
    ladder opponent at 100 sims)."""
    if version == "mcts":
        return baseline.HeuristicMCTSPlayer(sims=100, seed=seed)
    if version == 0:
        return baseline.HeuristicPlayer(seed=seed)
    return scoring.get_player(version, seed=seed)


class _PlayerCache:
    """Players keyed by (version, seed), re-seeded before every
    playout. Building a v8 player loads the value-net checkpoint
    from disk, so players are built once and reused — the RNG reset
    is what makes each playout fresh, not a rebuild. (MCTS is
    stateless per move: a fresh root is built for every
    ``choose_move`` call, so reusing a player across positions is
    safe.)"""

    def __init__(self) -> None:
        self._players: dict[tuple[object, int], object] = {}

    def get(self, version, seed: int):
        key = (version, seed)
        if key not in self._players:
            self._players[key] = _build_player(version, seed)
        player = self._players[key]
        rng = np.random.default_rng(seed)
        player.rng = rng
        mcts = getattr(player, "_mcts", None)
        if mcts is not None:
            mcts.rng = rng
        return player


def play_out(board: np.ndarray, to_move: int, version_a, version_b,
             a_colour: int, seed: int, cache: _PlayerCache,
             sims_a: int = 100) -> int:
    """Play to completion from ``board`` with fresh engines.
    Returns +1 if the A side (``a_colour``) wins, -1 if it loses, 0
    for a draw. ``sims_a`` overrides the MCTS sims for version A
    (used for the deeper-search alternative proposal, not the
    playouts themselves — playouts use the match settings)."""
    player_a = cache.get(version_a, seed)
    player_b = cache.get(version_b, seed + 1000)
    mcts_a = getattr(player_a, "_mcts", None)
    old_sims = None
    if mcts_a is not None and sims_a != mcts_a.sims:
        old_sims = mcts_a.sims
        mcts_a.sims = sims_a
    try:
        choose_black, choose_white = (
            (player_a.choose_move, player_b.choose_move)
            if a_colour == game.BLACK
            else (player_b.choose_move, player_a.choose_move)
        )
        winner = research.play_game_from(board, choose_black, choose_white, to_move)
    finally:
        if old_sims is not None:
            mcts_a.sims = old_sims
    if winner == game.EMPTY:
        return 0
    return 1 if winner == a_colour else -1


def alternatives_for(board: np.ndarray, player: int, actual: int,
                     version_a: int, topk: int, deep_sims: int,
                     seed: int, cache: _PlayerCache) -> list[dict]:
    """Candidate alternatives at a turn: top-K by 1-ply v1 score,
    the deeper-search MCTS choice, and any VCF move. Each entry has
    ``move`` (flat), ``xy``, and ``source``."""
    flat = board.reshape(-1).tolist()
    scored = sorted(
        (
            (scoring._score_move_flat(flat, m, player, 1.0), m)
            for m in scoring._candidate_moves(board)
        ),
        key=lambda sm: (-sm[0], sm[1]),
    )
    out: list[dict] = []
    seen: set[int] = {actual}
    for _s, m in scored[:topk]:
        if m not in seen:
            seen.add(m)
            out.append({"move": m, "xy": [m % 15, m // 15],
                        "source": "1-ply-topk"})
    # Deeper-search proposal (only meaningful for MCTS versions).
    if deep_sims:
        deep = cache.get(version_a, seed)
        mcts = getattr(deep, "_mcts", None)
        if mcts is not None:
            old_sims = mcts.sims
            mcts.sims = deep_sims
            try:
                m = deep.choose_move(board, player)
            finally:
                mcts.sims = old_sims
            if m not in seen:
                seen.add(m)
                out.append({"move": m, "xy": [m % 15, m // 15],
                            "source": f"mcts-{deep_sims}-sims"})
    # VCF moves (a move starting a forced win by consecutive fours).
    for m in scoring._candidate_moves(board):
        if m not in seen and evaluate.starts_forced_win(board, player, m):
            seen.add(m)
            out.append({"move": m, "xy": [m % 15, m // 15],
                        "source": "vcf"})
    return out


def workup(batch_path: str, game_idx: int, version_a, version_b,
           seeds: list[int], last_turns: int, topk: int, deep_sims: int,
           verbose: bool = True) -> dict:
    """Walk back one lost game. Returns the workup dict."""
    with open(batch_path, "r", encoding="utf-8") as f:
        batch = json.load(f)
    rec = next(r for r in batch["records"] if r["game"] == game_idx)
    if rec["winner"] == "draw" or rec.get("winner_is_a") is not False:
        raise ValueError(f"game {game_idx} is not a loss for engine A")
    a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
    moves = rec["moves"]
    cache = _PlayerCache()
    # The loser's turns, latest first, limited to the last N.
    loser_turns = [
        ply for ply in range(len(moves))
        if (game.BLACK if ply % 2 == 0 else game.WHITE) == a_colour
    ][-last_turns:]
    turns: list[dict] = []
    for ply in loser_turns:
        board = game.empty_board()
        for i, m in enumerate(moves[:ply]):
            board = game.apply_move(
                board, m, game.BLACK if i % 2 == 0 else game.WHITE
            )
        actual = moves[ply]
        # (1) The actual move's playout must lose FIRST.
        after_actual = game.apply_move(board, actual, a_colour)
        actual_outcomes = [
            play_out(after_actual, -a_colour, version_a, version_b,
                     a_colour, seed, cache)
            for seed in seeds
        ]
        actual_losses = sum(1 for o in actual_outcomes if o < 0)
        entry: dict = {
            "ply": ply,
            "played": [actual % 15, actual // 15],
            "actual_playout": actual_outcomes,
            "actual_losses": actual_losses,
            "qualifies": actual_losses >= MIN_SEEDS,
            "alternatives": [],
            "saving_moves": [],
        }
        if verbose:
            print(f"  ply {ply} played {entry['played']}: actual playout "
                  f"{actual_outcomes} ({actual_losses}/{len(seeds)} losses)"
                  + (" — QUALIFIES" if entry["qualifies"] else ""),
                  file=sys.stderr, flush=True)
        if not entry["qualifies"]:
            turns.append(entry)
            continue
        # (2) Only now test alternatives, at the SAME seeds.
        alts = alternatives_for(board, a_colour, actual, version_a,
                                topk, deep_sims, seeds[0], cache)
        for alt in alts:
            after_alt = game.apply_move(board, alt["move"], a_colour)
            outcomes = [
                play_out(after_alt, -a_colour, version_a, version_b,
                         a_colour, seed, cache)
                for seed in seeds
            ]
            non_losses = sum(1 for o in outcomes if o >= 0)
            alt_entry = {**alt, "playout": outcomes,
                         "non_losses": non_losses,
                         "saves": non_losses >= MIN_SEEDS}
            entry["alternatives"].append(alt_entry)
            if alt_entry["saves"]:
                entry["saving_moves"].append(alt_entry)
            if verbose:
                print(f"    alt {alt['xy']} ({alt['source']}): {outcomes} "
                      f"({non_losses}/{len(seeds)} non-losses)"
                      + (" — SAVES" if alt_entry["saves"] else ""),
                      file=sys.stderr, flush=True)
        turns.append(entry)
    qualifying = [t for t in turns if t["qualifies"]]
    saving = [t for t in turns if t["saving_moves"]]
    return {
        "batch": os.path.basename(batch_path),
        "game": game_idx,
        "version_a": version_a,
        "version_b": version_b,
        "a_colour": rec["a_colour"],
        "plies": len(moves),
        "seeds": seeds,
        "last_turns_tested": len(turns),
        "qualifying_turns": len(qualifying),
        "turns_with_saving_move": len(saving),
        "earliest_saving_turn": saving[0]["ply"] if saving else None,
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", help="batch JSON path")
    parser.add_argument("--game", type=int, required=True,
                        help="game index within the batch")
    parser.add_argument("--version-a", type=int, default=8,
                        help="loser's engine version (default 8)")
    parser.add_argument("--version-b", required=True,
                        help="winner's engine version (0, 1, ..., or mcts)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="playout seeds (default 0 1 2)")
    parser.add_argument("--last-turns", type=int, default=8,
                        help="how many of the loser's latest turns to test")
    parser.add_argument("--topk", type=int, default=5,
                        help="1-ply alternatives per qualifying turn")
    parser.add_argument("--deep-sims", type=int, default=400,
                        help="sims for the deeper-search alternative (0 to skip)")
    parser.add_argument("--out", default=None, help="JSON output path")
    args = parser.parse_args()

    print(f"== workup: {args.batch} game {args.game} "
          f"(v{args.version_a} vs v{args.version_b}) ==",
          file=sys.stderr, flush=True)
    version_b = args.version_b
    if version_b != "mcts":
        version_b = int(version_b)
    result = workup(
        args.batch, args.game, args.version_a, version_b,
        args.seeds, args.last_turns, args.topk, args.deep_sims,
    )
    print(f"  tested {result['last_turns_tested']} turns, "
          f"{result['qualifying_turns']} qualifying (actual move loses "
          f"the playout at >= {MIN_SEEDS}/{len(args.seeds)} seeds), "
          f"{result['turns_with_saving_move']} with a saving move",
          file=sys.stderr)
    if result["earliest_saving_turn"] is not None:
        print(f"  earliest saving turn: ply {result['earliest_saving_turn']}",
              file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"  workup written to {args.out}", file=sys.stderr)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

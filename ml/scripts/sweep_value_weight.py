#!/usr/bin/env python3
"""Sweep the ValueMCTS shrink α (VALUE_WEIGHT) for a value-net version.

Plays short matches (default 20 games) of the given version against
an opponent at each α in the grid, using the same seeds for every α
so the comparison is paired. The version's registry α is NOT used:
the sweep overrides ``player._mcts.value_weight`` directly, so the
registry records the winner only AFTER the sweep (see
``research/iterations/iter-11.md`` §1.3 for v11; ``iter-25.md``
§1.4 for v25, which runs this sweep against TWO opponents — v0 and
v8 — and selects on the COMBINED win rate; the per-opponent winner
in each JSON is advisory only for v25).

Selection rule (pre-registered for v11, same shape for v25):
highest win rate; ties broken toward LARGER α (prefer using the
net more when equal).

The output JSON is flushed after EVERY α arm (``complete`` is
false until the last arm), so a VM replacement loses at most the
in-progress arm — matches are deterministic given the seeds, so
a re-run reproduces the lost arm exactly.

Usage:
    python ml/scripts/sweep_value_weight.py --version 11 \\
        --alphas 1.0,0.5,0.25,0.125 --games 20 --opponent v0 \\
        --seed-a 1001 --seed-b 2001 --out research/iterations/sweep-v11.json
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

from gomoku_ml import baseline, evaluate, game, scoring  # noqa: E402
import selfplay_batch  # noqa: E402


def _write_payload(args, results, complete):
    """Write the sweep JSON (flushed after every α arm)."""
    payload = {
        "version": args.version,
        "opponent": args.opponent,
        "games_per_alpha": args.games,
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "alphas": results,
        "selected_alpha": pick_winner(results) if results else None,
        "selection_rule": "highest win rate; ties toward larger alpha",
        "complete": complete,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return payload


def run_sweep(version, alphas, games, opponent, seed_a, seed_b,
              on_arm=None):
    """Play ``games`` games per α; return per-α summaries."""
    results = []
    for alpha in alphas:
        player = scoring.get_player(version, seed=seed_a)
        player._mcts.value_weight = alpha
        if opponent == "v0":
            opp = baseline.HeuristicPlayer(seed=seed_b)
            choose_opp = opp.choose_move
            opp_name = "v0"
        else:
            opp_player = scoring.get_player(int(opponent[1:]), seed=seed_b)
            choose_opp = opp_player.choose_move
            opp_name = opp_player.name
        wins = losses = draws = 0
        t0 = time.time()
        for i in range(games):
            a_is_black = (i % 2 == 0)
            if a_is_black:
                winner, _moves = selfplay_batch.play_recorded_game(
                    player.choose_move, choose_opp,
                    game.empty_board(), game.BLACK,
                )
                a_colour = game.BLACK
            else:
                winner, _moves = selfplay_batch.play_recorded_game(
                    choose_opp, player.choose_move,
                    game.empty_board(), game.BLACK,
                )
                a_colour = game.WHITE
            if winner == game.EMPTY:
                draws += 1
            elif winner == a_colour:
                wins += 1
            else:
                losses += 1
            print(f"  α={alpha}: game {i + 1}/{games} "
                  f"({wins}W-{losses}L-{draws}D, {time.time() - t0:.0f}s)",
                  file=sys.stderr, flush=True)
        lo, hi = evaluate.wilson_ci(wins, losses, draws)
        results.append({
            "alpha": alpha,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(evaluate.win_rate(wins, losses, draws), 4),
            "win_rate_ci_95": [round(lo, 4), round(hi, 4)],
            "elapsed_seconds": round(time.time() - t0, 1),
        })
        print(f"  α={alpha}: {wins}W-{losses}L-{draws}D "
              f"(win rate {results[-1]['win_rate']:.4f}, "
              f"95% CI [{lo:.4f}, {hi:.4f}])",
              file=sys.stderr, flush=True)
        if on_arm is not None:
            on_arm(results)
    return results


def pick_winner(results):
    """Highest win rate; ties broken toward LARGER α."""
    best = None
    for r in sorted(results, key=lambda r: r["alpha"]):
        if best is None or r["win_rate"] >= best["win_rate"]:
            best = r
    return best["alpha"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--alphas", default="1.0,0.5,0.25,0.125",
                        help="comma-separated α grid")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", default="v0")
    parser.add_argument("--seed-a", type=int, default=1001)
    parser.add_argument("--seed-b", type=int, default=2001)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"== α sweep: v{args.version} vs {args.opponent}, "
          f"{args.games} games/α, α ∈ {alphas}, "
          f"seeds {args.seed_a}/{args.seed_b} ==",
          file=sys.stderr)
    results = run_sweep(args.version, alphas, args.games,
                        args.opponent, args.seed_a, args.seed_b,
                        on_arm=lambda r: _write_payload(args, r, False))
    payload = _write_payload(args, results, True)
    winner = payload["selected_alpha"]
    print(f"  selected α = {winner} (written to {args.out})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

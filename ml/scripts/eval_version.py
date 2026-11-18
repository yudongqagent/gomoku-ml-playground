#!/usr/bin/env python3
"""Ladder + tactical eval for a versioned scoring engine.

Runs the eval ladder (random / frozen baseline heuristic / heuristic
MCTS) and the tactical suite for ``--version`` (default 2), printing
a human-readable summary plus a JSON blob (line starts ``EVAL_JSON ``).
Use the same ``--ladder-games`` / ``--mcts-sims`` for every version
being compared — the "better" bar requires identical game counts.

With ``--out PATH`` the result is also written to PATH (atomically,
flushed after every ladder leg), and a partial PATH from a killed run
is auto-resumed: completed legs are kept and the player's seeded RNG
state (saved with the partial) is restored, so a resumed ladder
reproduces the uninterrupted ladder exactly. Re-running with a
complete PATH prints the saved result without recomputing. The RNG
restore matters: the player's RNG advances across legs (tie-breaks
only), so a leg replayed with a fresh RNG would NOT match the
uninterrupted ladder.

Usage:
    python ml/scripts/eval_version.py --version 2 --ladder-games 20
    python ml/scripts/eval_version.py --version 27 --ladder-games 20 \\
        --out research/iterations/ladder-v27.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import baseline, evaluate, research, scoring  # noqa: E402


def _write_result(path: str, result: dict) -> None:
    """Atomically write ``result`` to ``path`` (temp file + rename).

    Same pattern as selfplay_batch.py: this VM is replaced without
    warning, and a killed run must lose at most the in-progress leg.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _print_matchup(name: str, r: dict, resumed: bool = False) -> None:
    lo, hi = r["win_rate_ci_95"]
    b, w = r["as_black"], r["as_white"]
    print(f"  vs {name}: {r['wins']}W-{r['losses']}L-{r['draws']}D "
          f"(win rate {r['win_rate']:.4f}, 95% CI [{lo:.4f}, {hi:.4f}]; "
          f"as Black {b['wins']}W-{b['losses']}L-{b['draws']}D, "
          f"as White {w['wins']}W-{w['losses']}L-{w['draws']}D)"
          + (" (resumed)" if resumed else ""),
          file=sys.stderr)


def _result(player, args, tac, matchups, complete, resumed,
            resumed_from_leg):
    """The full eval object: the EVAL_JSON shape plus resume metadata.

    ``ladder_elo`` is None until all legs are done (the estimate
    needs the anchored random + heuristic legs).
    """
    done = list(matchups) == list(research.LADDER_OPPONENTS)
    return {
        "engine": player.name,
        "version": args.version,
        "ladder_games": args.ladder_games,
        "mcts_sims": args.mcts_sims,
        "seed": args.seed,
        "complete": complete,
        "resumed": resumed,
        "resumed_from_leg": resumed_from_leg,
        # Player RNG state after the last flushed leg (None until the
        # first leg completes). Restoring it on resume is what makes
        # a resumed ladder identical to an uninterrupted one.
        "rng_state": (player.rng.bit_generator.state
                      if matchups else None),
        "tactical": tac,
        "ladder": {
            "matchups": matchups,
            "ladder_elo": (evaluate.ladder_elo(matchups)
                           if done else None),
            "games_per_opponent": args.ladder_games,
            "mcts_sims": args.mcts_sims,
        },
    }


def _load_partial(path: str, player, args):
    """Load a resumable partial ``--out`` file, or None to start fresh.

    Returns (matchups, rng_state, complete). A partial only resumes
    when it matches this run's engine/version/params exactly, its
    completed legs are a prefix of the ladder order, and it carries
    the RNG state (without it a resumed ladder would differ from the
    uninterrupted one — never resume inexactly).
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            prior = json.load(f)
        if (prior.get("engine") != player.name
                or prior.get("version") != args.version
                or prior.get("ladder_games") != args.ladder_games
                or prior.get("mcts_sims") != args.mcts_sims
                or prior.get("seed") != args.seed
                or not isinstance(prior.get("ladder"), dict)
                or not isinstance(prior["ladder"].get("matchups"), dict)):
            print(f"== ignoring non-matching prior {path} "
                  f"(different engine/version/params) ==", file=sys.stderr)
            return None
        matchups = prior["ladder"]["matchups"]
        if list(matchups) != list(
                research.LADDER_OPPONENTS[:len(matchups)]):
            print(f"== ignoring prior {path} (legs are not a ladder "
                  f"prefix) ==", file=sys.stderr)
            return None
        if matchups and prior.get("rng_state") is None:
            print(f"== ignoring prior {path} (no RNG state — cannot "
                  f"resume exactly) ==", file=sys.stderr)
            return None
        return matchups, prior.get("rng_state"), bool(prior.get("complete"))
    except (OSError, ValueError, KeyError) as e:
        print(f"== ignoring unreadable prior {path}: {e} ==",
              file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--ladder-games", type=int, default=20)
    parser.add_argument("--mcts-sims", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--out", default=None,
                        help="write the result JSON here (flushed "
                        "after every ladder leg; a partial file from "
                        "a killed run is auto-resumed)")
    args = parser.parse_args()

    player = scoring.get_player(args.version, seed=args.seed)

    # --out resume: a complete prior file is the answer already; a
    # partial one restores the completed legs and the player's RNG
    # state (the MCTS holds the same Generator object, so restoring
    # the state in place covers it too).
    prior = _load_partial(args.out, player, args) if args.out else None
    if prior is not None and prior[2]:
        with open(args.out, encoding="utf-8") as f:
            saved = json.load(f)
        print(f"== {args.out} already complete — nothing to do ==",
              file=sys.stderr)
        print("EVAL_JSON " + json.dumps(saved))
        return 0
    matchups: dict = {}
    resumed = False
    resumed_from_leg = None
    if prior is not None:
        matchups, rng_state, _ = prior
        if rng_state is not None:
            player.rng.bit_generator.state = rng_state
        resumed = True
        resumed_from_leg = research.LADDER_OPPONENTS[len(matchups)]
        print(f"== resuming {args.out} from leg {resumed_from_leg} "
              f"({len(matchups)}/{len(research.LADDER_OPPONENTS)} legs "
              f"done; RNG state restored) ==", file=sys.stderr)

    print(f"== tactical suite (20 positions): {player.name} ==",
          file=sys.stderr)
    # NOTE (batch 2): the tactical probe must use the version's FULL
    # move choice (threat check / minimax), not just its defence
    # weight — an earlier version of this driver dropped the search
    # flags, so v3's 20/20 was measured as v1. rng stays None so
    # tactical measurement is deterministic and comparable with
    # batch-1 numbers.
    tac = research.run_tactical(
        lambda board, p: scoring._choose_move(
            board, p, None, player.defence_weight, player.threat_check,
            player.minimax, player.minimax_topk,
        )
    )
    print(f"  {player.name}: {tac['passed']}/{tac['total']} "
          f"({tac['pass_rate']:.2%})", file=sys.stderr)
    for r in tac["results"]:
        if not r["passed"]:
            print(f"    FAIL {r['id']} ({r['name']}): chose {r['chosen']}, "
                  f"expected {r['expect']}", file=sys.stderr)

    print(f"== ladder for {player.name} ({args.ladder_games} "
          f"games/opponent, mcts_sims={args.mcts_sims}) ==", file=sys.stderr)
    if args.out is None:
        # Original path, unchanged: one call, one EVAL_JSON.
        ladder = research.ladder_for_player(
            player.choose_move,
            games_per_opponent=args.ladder_games,
            mcts_sims=args.mcts_sims,
            seed=args.seed + 300,
        )
        for name, r in ladder["matchups"].items():
            _print_matchup(name, r)
        elo = ladder["ladder_elo"]
        print(f"  ladder Elo: estimate {elo['estimate']}, interval "
              f"{elo['interval']} (anchors: random=-800, heuristic=0; "
              f"relative to THIS ladder only)", file=sys.stderr)
        out = {"engine": player.name, "version": args.version,
               "tactical": tac, "ladder": ladder}
        print("EVAL_JSON " + json.dumps(out))
        return 0

    # --out path: run the remaining legs one at a time, flushing
    # after each leg so a killed run loses at most the in-progress
    # leg. Leg seeds match ladder_for_player exactly (seed + 300).
    for name in research.LADDER_OPPONENTS:
        if name in matchups:
            _print_matchup(name, matchups[name], resumed=True)
            continue
        matchups[name] = research.ladder_leg_for_player(
            player.choose_move, name,
            games_per_opponent=args.ladder_games,
            mcts_sims=args.mcts_sims,
            seed=args.seed + 300,
        )
        _print_matchup(name, matchups[name])
        _write_result(args.out, _result(
            player, args, tac, matchups, complete=False,
            resumed=resumed, resumed_from_leg=resumed_from_leg))
    out = _result(player, args, tac, matchups, complete=True,
                  resumed=resumed, resumed_from_leg=resumed_from_leg)
    _write_result(args.out, out)
    elo = out["ladder"]["ladder_elo"]
    print(f"  ladder Elo: estimate {elo['estimate']}, interval "
          f"{elo['interval']} (anchors: random=-800, heuristic=0; "
          f"relative to THIS ladder only)", file=sys.stderr)
    print("EVAL_JSON " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

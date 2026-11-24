#!/usr/bin/env python3
"""M1 for v58 (iter-58 §1.3): TSS mining, v27-vs-v27, normal rules vs
swap rules — does White get MORE forcing-win opportunities under swap?

Procedure (pre-registered in iter-58 §1.3 BEFORE this script ran):

* Play 50 v27-vs-v27 games per leg, seeds 777/1777 (SAME as Check B
  — the normal leg replicates Check B: it must reproduce 2481 turns /
  288 fires / 121 differs, and adds by-colour breakdowns).
* Swap leg: balanced placement, placement_seed 11 (the battery's
  placement seed), tempo judge; per-game placement RNG seeded
  ``11 + game_index``, exactly as selfplay_batch.py does.
* At every turn, run production TSS (depth 4, budget 3000) WITHOUT
  perturbing the game (TSS consumes no RNG) and record, BY COLOUR:
  turns, fires, confirms, differs, budget-exceeded count, node sum.
* Record game outcomes (winner colour, plies) and, for swap games,
  the opening and per-opening differ counts.
* Record qualifying White turns for the M3 depth-6 spot-check (the
  first 30 per leg where production TSS returns None AND the forcing
  move set is non-empty AND Black holds no winning move AND the
  search completed, i.e. budget NOT exceeded).
* NO playout verification (Check B already verified differs convert
  5/5; TSS fires are exact by construction). This mining measures
  RATES, not verification.

Decision rule (iter-58 §1.4): (b) CONFIRMED if White's differ rate
is >=1.5x under swap vs normal AND the difference exceeds 2 binomial
SEs; (b) REFUTED if the ratio is in [0.8, 1.25] on both M1 and M2.

The mining is deterministic (per-game seeds, TSS consumes no RNG),
so it checkpoints after every game and resumes from the last
completed game on restart.

Usage:
    PYTHONPATH=ml python ml/scripts/mine_tss_swap.py --rule normal
        --out research/iterations/tss-mine-v58-normal.json
    PYTHONPATH=ml python ml/scripts/mine_tss_swap.py --rule swap
        --out research/iterations/tss-mine-v58-swap.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))

MAX_QUALIFYING = 30  # White turns recorded for the M3 spot-check


def _xy(move):
    return [move % 15, move // 15] if move is not None else None


def _board_to_lists(board):
    black = [[int(x), int(y)] for y, x in zip(*((board == 1).nonzero()))]
    white = [[int(x), int(y)] for y, x in zip(*((board == -1).nonzero()))]
    return black, white


def _new_colour_stats():
    return {
        "turns": 0,
        "fires": 0,
        "confirms": 0,
        "differs": 0,
        "budget_exceeded": 0,
        "nodes": 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["normal", "swap"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--seed-a", type=int, default=777)
    ap.add_argument("--seed-b", type=int, default=1777)
    ap.add_argument("--placement-seed", type=int, default=11)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    import numpy as np

    from gomoku_ml import game, mcts_heuristic, research, scoring

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    ckpt_path = args.checkpoint or (out_path + ".checkpoint.json")
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)

    stats = {"black": _new_colour_stats(), "white": _new_colour_stats()}
    differs = []
    qualifying = []
    outcomes = []  # (winner_colour, plies) per game
    openings = []  # swap only: opening tuple per game
    per_opening_differs = {}  # "b,w,b" -> {"black": n, "white": n}
    start_gi = 0
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                ckpt = json.load(f)
            if (
                ckpt.get("rule") == args.rule
                and ckpt.get("games") == args.games
                and ckpt.get("seed_a") == args.seed_a
                and ckpt.get("seed_b") == args.seed_b
                and ckpt.get("placement_seed") == args.placement_seed
            ):
                stats = ckpt["stats"]
                differs = ckpt["differs"]
                qualifying = ckpt["qualifying"]
                outcomes = ckpt["outcomes"]
                openings = ckpt["openings"]
                per_opening_differs = ckpt["per_opening_differs"]
                start_gi = ckpt["games_done"]
                print(f"resuming from checkpoint: {start_gi}/"
                      f"{args.games} games done", file=sys.stderr)
        except (OSError, ValueError, KeyError):
            print("checkpoint unreadable — starting from scratch",
                  file=sys.stderr)

    scorer = scoring.score_attack
    for gi in range(start_gi, args.games):
        pa = scoring.get_player(27, seed=args.seed_a + gi)
        pb = scoring.get_player(27, seed=args.seed_b + gi)
        opening = None
        if args.rule == "swap":
            placement_rng = np.random.default_rng(
                args.placement_seed + gi)
            a_opens = (gi % 2 == 0)
            # Mirror selfplay_batch.py: the opener is a choose fn
            # (unused by balanced placement, but faithful).
            opener_fn = pa.choose_move if a_opens else pb.choose_move
            opener, second = (pa, pb) if a_opens else (pb, pa)
            board, opening = research.place_swap_opening(
                opener_fn, "balanced", scorer, placement_rng
            )
            second_colour = game.choose_swap_color(
                board, scorer, judge="tempo")
            if second_colour == game.BLACK:
                black_engine, white_engine = second, opener
            else:
                black_engine, white_engine = opener, second
            player = game.WHITE  # move 4, White to move
            openings.append([int(m) for m in opening])
        else:
            board = game.empty_board()
            black_engine = pa if gi % 2 == 0 else pb
            white_engine = pb if gi % 2 == 0 else pa
            player = game.BLACK
        plies = 0
        while True:
            over, _winner = game.game_over(board)
            if over:
                break
            chooser = black_engine if player == game.BLACK else white_engine
            v27_move = chooser.choose_move(board, player)
            tss_stats: dict = {}
            tss_move = mcts_heuristic.tss_winning_move(
                board, player, stats=tss_stats)
            key = "black" if player == game.BLACK else "white"
            cs = stats[key]
            cs["turns"] += 1
            cs["nodes"] += tss_stats.get("nodes", 0)
            if tss_stats.get("budget_exceeded"):
                cs["budget_exceeded"] += 1
            if tss_move is not None:
                cs["fires"] += 1
                if tss_move == v27_move:
                    cs["confirms"] += 1
                else:
                    cs["differs"] += 1
                    black, white = _board_to_lists(board)
                    d = {
                        "game": gi,
                        "to_move": key,
                        "black": black,
                        "white": white,
                        "tss_move": _xy(tss_move),
                        "v27_move": _xy(v27_move),
                    }
                    if opening is not None:
                        d["opening"] = [int(m) for m in opening]
                        okey = ",".join(str(int(m)) for m in opening)
                        pod = per_opening_differs.setdefault(
                            okey, {"black": 0, "white": 0})
                        pod[key] += 1
                    differs.append(d)
            elif (
                player == game.WHITE
                and len(qualifying) < MAX_QUALIFYING
                and not tss_stats.get("budget_exceeded")
                and not scoring._winning_moves(board, game.BLACK)
            ):
                # M3 candidate: forcing moves exist, no depth-4 win,
                # no forced block, search completed.
                if mcts_heuristic.tss_forcing_moves(board, player):
                    black, white = _board_to_lists(board)
                    qualifying.append({
                        "game": gi,
                        "ply": plies,
                        "black": black,
                        "white": white,
                    })
            if v27_move < 0:
                break
            board = game.apply_move(board, v27_move, player)
            player = -player
            plies += 1
        _over, winner = game.game_over(board)
        outcomes.append({
            "game": gi,
            "winner": (
                "black" if winner == game.BLACK
                else "white" if winner == game.WHITE else "draw"
            ),
            "plies": plies,
            "opening": [int(m) for m in opening] if opening else None,
        })
        print(f"game {gi + 1}/{args.games}: black turns="
              f"{stats['black']['turns']} fires={stats['black']['fires']} "
              f"differs={stats['black']['differs']} | white turns="
              f"{stats['white']['turns']} fires={stats['white']['fires']} "
              f"differs={stats['white']['differs']} | qualifying="
              f"{len(qualifying)}", file=sys.stderr)
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump({
                "rule": args.rule,
                "games": args.games,
                "seed_a": args.seed_a,
                "seed_b": args.seed_b,
                "placement_seed": args.placement_seed,
                "games_done": gi + 1,
                "stats": stats,
                "differs": differs,
                "qualifying": qualifying,
                "outcomes": outcomes,
                "openings": openings,
                "per_opening_differs": per_opening_differs,
            }, f)

    def _rates(cs):
        t = cs["turns"]
        return {
            **cs,
            "fire_rate": cs["fires"] / t if t else 0.0,
            "differ_rate": cs["differs"] / t if t else 0.0,
            "mean_nodes": cs["nodes"] / t if t else 0.0,
        }

    summary = {
        "rule": args.rule,
        "games": args.games,
        "seeds": [args.seed_a, args.seed_b],
        "placement_seed": args.placement_seed if args.rule == "swap" else None,
        "by_colour": {
            "black": _rates(stats["black"]),
            "white": _rates(stats["white"]),
        },
        "total_turns": stats["black"]["turns"] + stats["white"]["turns"],
        "total_fires": stats["black"]["fires"] + stats["white"]["fires"],
        "total_differs": stats["black"]["differs"] + stats["white"]["differs"],
        "outcomes": outcomes,
        "black_wins": sum(1 for o in outcomes if o["winner"] == "black"),
        "white_wins": sum(1 for o in outcomes if o["winner"] == "white"),
        "draws": sum(1 for o in outcomes if o["winner"] == "draw"),
        "mean_plies": (
            sum(o["plies"] for o in outcomes) / len(outcomes)
            if outcomes else 0.0
        ),
        "per_opening_differs": per_opening_differs,
        "qualifying_count": len(qualifying),
        "qualifying": qualifying,
        "differs": differs,
    }
    print(json.dumps(
        {k: v for k, v in summary.items()
         if k not in ("differs", "qualifying", "outcomes")},
        indent=2,
    ))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

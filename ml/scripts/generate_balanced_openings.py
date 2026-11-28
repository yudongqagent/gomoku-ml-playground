#!/usr/bin/env python3
"""Generate the balanced openings sets (evaluation-audit fix).

NOTE (2026-11-28, openings v2): the v1 balanced set this script
writes (24 gap-0 openings) was SUPERSEDED — gap 0 does not imply
balanced in play (White collapsed from the v1 set), and the
replacement v2 set (20 openings, selected by played White win
rate) lives in ``research/openings-balanced.json`` with its
validation in ``research/openings-v2-validation.md``. The sampler
below (``gen_balanced`` / ``sample_4move_opening``) is still the
v2 candidate sampler (reused by
``ml/scripts/sample_openings_v2.py``), but do NOT re-run this
script's ``main()`` — it would overwrite the validated v2 file
with the v1 set (and regenerate the swap set). The swap set it
writes is unchanged and still current.

Two files, both in the openings-ml4.json format
(``{"openings": [[[x, y], ...], ...]}``) plus a ``meta`` block
documenting the balance criterion:

1. ``research/openings-balanced.json`` — 24 four-move openings
   (Black, White, Black, White) for normal-rules matches. Each is
   balanced by the attack-gap criterion: best Black attack equals
   best White attack (gap 0) on the finished opening, so neither
   colour starts with a better immediate attack; both sides hold a
   real attack (best attack 164 or 4006, never a trivial 86). No
   opening ends the game and none holds an open four. Stones are
   placed alternately within Chebyshev distance 2 of an existing
   stone (compact, interacting region). The 24 are distinct and
   cover a mix of quiet (164/164) and sharp (4006/4006) positions.

2. ``research/openings-swap.json`` — 24 three-stone swap openings
   (Black centre, White, Black), sampled from the broad compact
   3-stone space and curated to a 50/50 tempo-margin mix: 12 with
   tempo margin 0 (judge ties -> White) and 12 with tempo margin 78
   (judge picks Black). The uncurated placement yields margin 0 only
   ~16-21% of the time, so the judge picked Black ~78-88% of the time
   and the swap choice was nearly deterministic; the curated set makes
   the pick a real 50/50 choice. The two margin classes are arranged
   in pairs [0,0,78,78] x 6, not strictly alternating [0,78,...]:
   strict alternation has period 2, which resonates with the opener
   alternation and gives engine A Black in EVERY game (the pick is
   50/50 but the colour assignment is maximally unfair). Pairs have
   even-length blocks, so each engine plays Black exactly half the
   time while the pick stays 50/50. Each entry records its tempo
   margin and the judge pick. NOTE: the openings are NOT drawn from
   the balanced placement — its reachable set has only 24 distinct
   openings and only 5 of them have tempo margin 0, so 12 distinct
   margin-0 openings cannot be drawn from it (the first version of
   this generator looped forever on exactly that).

Both files are deterministic: fixed RNG seeds, so re-running this
script reproduces the same sets byte-for-byte.
"""
import json
import os
import sys

REPO = "/home/hatch/workspace/projects/gomoku-ml-playground"
sys.path.insert(0, os.path.join(REPO, "ml", "scripts"))
import common
common.ensure_ml_on_path()

import numpy as np
from gomoku_ml import game, research, scoring

scorer = scoring.score_attack


def sample_4move_opening(rng):
    """One random 4-move opening (B,W,B,W), compact region."""
    board = game.empty_board()
    chosen = []
    for ply, player in enumerate(
            [game.BLACK, game.WHITE, game.BLACK, game.WHITE]):
        if ply == 0:
            cands = [(7, 7)]
        else:
            near = set()
            for (cx, cy) in chosen:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < 15 and 0 <= ny < 15 and (nx, ny) not in chosen:
                            near.add((nx, ny))
            cands = sorted(near)
        x, y = cands[int(rng.integers(len(cands)))]
        chosen.append((x, y))
        board = game.apply_move(board, game.xy_to_move(x, y), player)
    return board, chosen


def gen_balanced(n=24, seed=20261125):
    rng = np.random.default_rng(seed)
    picked = []
    seen = set()
    stats = {"sampled": 0, "gap0": 0, "rejected_open_four": 0}
    while len(picked) < n:
        stats["sampled"] += 1
        board, chosen = sample_4move_opening(rng)
        over, _ = game.game_over(board)
        if over:
            continue
        gap, bb, bw = research.opening_gap(board, scorer)
        if gap != 0:
            continue
        stats["gap0"] += 1
        if bb >= 10000 or bw >= 10000:
            stats["rejected_open_four"] += 1
            continue
        key = tuple(chosen)
        if key in seen:
            continue
        seen.add(key)
        picked.append({"moves_xy": [list(m) for m in chosen],
                       "best_black": bb, "best_white": bw, "gap": gap})
    return picked, stats


def sample_swap_opening(rng):
    """One random 3-stone swap opening (B,W,B), compact region.

    Black 1 is the centre. White is drawn from the 24 cells within
    Chebyshev distance 2 of the centre. Black 2 is drawn from the
    cells within Chebyshev distance 2 of White or of the centre
    (a compact, interacting region — but NOT the gap-minimising
    choice: the balanced placement's reachable set has only 24
    distinct openings and only 5 of them have tempo margin 0, so
    12 distinct margin-0 openings cannot be drawn from it at all).
    """
    black1 = (7, 7)
    near_centre = [(x, y) for x in range(5, 10) for y in range(5, 10)
                   if (x, y) != black1]
    white = near_centre[int(rng.integers(len(near_centre)))]
    near = set()
    for (cx, cy) in (black1, white):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 15 and 0 <= ny < 15 and (nx, ny) not in (black1, white):
                    near.add((nx, ny))
    cands = sorted(near)
    black2 = cands[int(rng.integers(len(cands)))]
    board = game.empty_board()
    for (x, y), player in ((black1, game.BLACK), (white, game.WHITE),
                           (black2, game.BLACK)):
        board = game.apply_move(board, game.xy_to_move(x, y), player)
    return board, (game.xy_to_move(*black1), game.xy_to_move(*white),
                   game.xy_to_move(*black2))


def gen_swap(n_each=12, seed=20261125, max_samples=2000):
    """Curated swap openings: n_each with tempo margin 0 and n_each
    with tempo margin 78, all distinct.

    Samples 3-stone openings from the broad compact space (see
    :func:`sample_swap_opening`) and curates by the exact tempo
    margin (``game._setup_value`` for each colour). Margins other
    than 0/78 are skipped. Raises if the sample cap is hit before
    the set is complete (so a bad margin distribution fails loudly
    instead of looping forever — the first version of this function
    drew from the balanced placement, whose reachable set holds
    only 5 distinct margin-0 openings, and never terminated).
    """
    rng = np.random.default_rng(seed)
    by_margin = {0: [], 78: []}
    seen = set()
    sampled = 0
    while len(by_margin[0]) < n_each or len(by_margin[78]) < n_each:
        sampled += 1
        if sampled > max_samples:
            raise RuntimeError(
                f"gen_swap hit the {max_samples}-sample cap with "
                f"margin-0={len(by_margin[0])}, margin-78="
                f"{len(by_margin[78])} (need {n_each} each)")
        board, opening = sample_swap_opening(rng)
        key = tuple(opening)
        if key in seen:
            continue
        vb = game._setup_value(board, scorer, game.BLACK)
        vw = game._setup_value(board, scorer, game.WHITE)
        margin = vb - vw
        if margin not in by_margin:
            continue
        if len(by_margin[margin]) >= n_each:
            continue
        seen.add(key)
        xy = [list(game.move_to_xy(m)) for m in opening]
        pick = game.choose_swap_color(board, scorer, judge="tempo")
        by_margin[margin].append({
            "moves_xy": xy,
            "tempo_margin": margin,
            "tempo_value_black": vb,
            "tempo_value_white": vw,
            "judge_pick": "black" if pick == game.BLACK else "white",
        })
        if sampled % 25 == 0 or (len(by_margin[0]) == n_each
                                 and len(by_margin[78]) == n_each):
            print(f"  swap sampling: {sampled} sampled, "
                  f"margin-0={len(by_margin[0])}, "
                  f"margin-78={len(by_margin[78])}", flush=True)
    # Arrange in pairs [0,0,78,78] x 6 — NOT strictly alternating
    # [0,78,0,78,...]. A strictly alternating margin sequence has
    # period 2, which resonates with the opener alternation (also
    # period 2: game i's opener is A iff i is even) and gives engine
    # A Black in EVERY game: even games are margin-0 (second picks
    # White, so the opener A is Black) and odd games are margin-78
    # (second A picks Black). The pick is 50/50 but the colour
    # assignment is maximally unfair. Pairs have even-length blocks,
    # so the opener alternation averages out within each block and
    # each engine plays Black exactly half the time (for any battery
    # length that is a multiple of 4), while the pick stays 50/50.
    out = []
    for i in range(0, n_each, 2):
        out.extend(by_margin[0][i:i + 2])
        out.extend(by_margin[78][i:i + 2])
    return out, {"sampled": sampled,
                 "margin0": len(by_margin[0]),
                 "margin78": len(by_margin[78])}


def main():
    balanced, stats = gen_balanced()
    balanced_doc = {
        "meta": {
            "purpose": "balanced 4-move openings for normal-rules matches",
            "criterion": "attack gap 0 (best Black attack == best White attack), "
                         "no five, no open four, compact interacting region",
            "generator": "ml/scripts/generate_balanced_openings.py (eval-audit)",
            "seed": 20261125,
            "sampled": stats["sampled"],
            "gap0_found": stats["gap0"],
            "rejected_open_four": stats["rejected_open_four"],
            "count": len(balanced),
            "sharp_4006": sum(1 for o in balanced if o["best_black"] == 4006),
            "quiet_164": sum(1 for o in balanced if o["best_black"] == 164),
        },
        "openings": [o["moves_xy"] for o in balanced],
        "details": balanced,
    }
    # Write the balanced file first: the swap generation below is
    # slow (exact tempo-margin computation per sampled opening), so
    # the balanced set is safely on disk even if the swap part is
    # interrupted.
    path = os.path.join(REPO, "research/openings-balanced.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(balanced_doc, f, indent=2)
        f.write("\n")
    print(f"wrote {path}: {balanced_doc['meta']['count']} openings")
    print("balanced stats:", stats)
    swap, swap_stats = gen_swap()
    swap_doc = {
        "meta": {
            "purpose": "curated 3-stone swap openings, 50/50 tempo-margin mix",
            "criterion": "12 openings with tempo margin 0 (judge ties -> White) "
                         "and 12 with tempo margin 78 (judge picks Black), "
                         "sampled from the broad compact 3-stone space "
                         "and curated by the exact tempo margin",
            "generator": "ml/scripts/generate_balanced_openings.py (eval-audit)",
            "seed": 20261125,
            "sampled": swap_stats["sampled"],
            "count": len(swap),
            "margin0": swap_stats["margin0"],
            "margin78": swap_stats["margin78"],
        },
        "openings": [o["moves_xy"] for o in swap],
        "details": swap,
    }
    path = os.path.join(REPO, "research/openings-swap.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(swap_doc, f, indent=2)
        f.write("\n")
    print(f"wrote {path}: {swap_doc['meta']['count']} openings")
    print("swap stats:", swap_stats)
    # sanity: judge pick distribution over the swap set
    picks = [o["judge_pick"] for o in swap]
    print("swap judge picks over the set:",
          {p: picks.count(p) for p in set(picks)})


if __name__ == "__main__":
    main()

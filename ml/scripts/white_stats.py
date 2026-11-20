#!/usr/bin/env python3
"""White-side play statistics (batch 12, H2 — v46, Phase 1).

Replays recorded v27 games and computes colour-split play
statistics, per the pre-registered protocol in
``research/iterations/iter-46.md`` §1.1. Every statistic is an
exact computation from the game records (VCF solver / creator
enumeration / kernel scores), never a guessed losing move.

Ply bands: opening 1-12, midgame 13-32, endgame 33+ (ply =
1-indexed move number). Statistics (numbered as in iter-46 §1.1):

1. Win rate by colour (per leg and pooled).
2. Threatened-turn rate: opponent holds >= 1 open-four creator
   and no immediate winning move (the classifier's definition).
3. Clean missed prevention rate (creator-based): among threatened
   turns, the played move was not preventing while a preventing
   alternative existed (``classify_batch_losses.py``'s exact
   definition, via ``scoring._move_is_safe``).
4. Forced-block share: opponent holds exactly 1 winning move.
5. Threat-creation rate: among QUIET turns (no opponent winning
   move and no opponent creator before our move), the fraction of
   moves after which the mover holds >= 1 open-four creator, and
   separately >= 1 VCF start.
6. First-threat ply: mean ply at which each colour first holds a
   VCF start / first holds a creator (after its own move), per
   game; games where a colour never holds one are counted
   separately and excluded from that colour's mean.
7. VCF-start exposure: mean opponent VCF starts remaining after
   our move, on turns where the opponent held >= 1 VCF start
   before our move.
8. Clean missed VCF prevention rate (VCF oracle): among
   VCF-threatened turns (opponent holds >= 1 VCF start before our
   move, no immediate winning move), the fraction where the played
   move left >= 1 of the ORIGINAL VCF starts live while a
   preventing alternative existed (a candidate move after which
   the opponent holds 0 of the original VCF starts). The
   alternative enumeration tests candidates in v1-total-score
   order and stops at the first prevention (existence is all the
   statistic needs); candidates are ``scoring._candidate_moves``
   (Chebyshev-2 of stones — the realistic prevention set). A
   candidate that wins immediately for us counts as preventing
   without a VCF test. Caveat (pre-registered): the test covers
   the original VCF starts only (the v35-style check — our move
   can CREATE a new VCF start, and this test does not catch that);
   the rate is therefore an upper bound on true VCF-prevention
   failures, and the same bound applies to both colours, so the
   colour COMPARISON is unbiased.
9. Decisive-phase distribution: for each loss, the ply of the
   first losing move by the VCF-solver method
   (``find_losing_moves.analyse_loss``: first loser move after
   which the opponent forces a win at depth 4 AND some near reply
   avoids it; "already lost" if no reply avoids it), by colour
   of the loser and ply band.
10. v1-score regret (diagnostic only): mean gap between the best
    candidate's v1 total score and the played move's v1 total
    score, by colour x band. NOT an error measure.

The patch rule (iter-46 §1.1) rests on statistics 3 and 8: a
patchable asymmetry needs White's rate to exceed Black's by
> 5.0 percentage points in the midgame or endgame band with the
two-proportion z-test 95% CI for the difference excluding 0.
The script prints those gaps and CIs for every statistic x band
so the decision is a read-off, not a judgment call.

Error instances are recorded in full (game, ply, move, threat
counts, preventing alternatives with their v1 rank, played
move's v1 rank) for the signature inspection (iter-46 §1.1(b)),
together with matched Black controls (threatened turns where
Black prevented successfully).

Usage:
    python ml/scripts/white_stats.py \\
        research/iterations/stats-v27-selfplay-60.json \\
        research/iterations/stats-v27-vs-v0-20.json \\
        --out research/iterations/white-stats-v46.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402

from gomoku_ml import evaluate, game, kernel_v1, scoring  # noqa: E402
import find_losing_moves  # noqa: E402

BANDS = ((1, 12, "opening"), (13, 32, "midgame"), (33, 10 ** 9, "endgame"))
VCF_DEPTH = 3


def _band(ply: int) -> str:
    for lo, hi, name in BANDS:
        if lo <= ply <= hi:
            return name
    raise AssertionError(ply)


def _xy(m: int) -> list[int]:
    return [m % 15, m // 15]


def _vcf_starts(board, player) -> list[int]:
    """VCF starts for ``player`` on ``board`` (depth 3, the
    tactical suite's VCF definition), pre-filtered to
    four-creating moves (``starts_forced_win`` requires creating
    a four)."""
    return [
        m
        for m in evaluate.four_creating_moves(board, player)
        if evaluate.starts_forced_win(board, player, m, depth=VCF_DEPTH)
    ]


def _v1_totals(board, player) -> dict[int, int]:
    """v1 total score (own attack + opponent attack at the square)
    for every candidate move — the same score ``_rollout_move``
    argmaxes over."""
    black, white = kernel_v1.attack_v1_both(board)
    if player == game.BLACK:
        own, opp = black, white
    else:
        own, opp = white, black
    total = own + opp
    total[board != game.EMPTY] = -1
    return {int(m): int(total.flat[m]) for m in scoring._candidate_moves(board)}


def _v1_rank(totals: dict[int, int], move: int) -> int | None:
    """1-based rank of ``move`` by v1 total (ties share the better
    rank); None if the move is not a candidate."""
    if move not in totals:
        return None
    score = totals[move]
    return 1 + sum(1 for s in totals.values() if s > score)


def _vcf_prevention_exists(board, player, orig_vcf, totals) -> dict | None:
    """First candidate (v1-total order) after which the opponent
    holds 0 of ``orig_vcf``. Returns ``{"move": m, "v1_rank": r}``
    or None. An immediate winning move for us counts as
    preventing (the game ends)."""
    ordered = sorted(totals, key=lambda m: (-totals[m], m))
    for c in ordered:
        if game.is_winning_move(board, c, player):
            return {"move": int(c), "v1_rank": _v1_rank(totals, c)}
        b2 = game.apply_move(board, c, player)
        if not any(
            evaluate.starts_forced_win(b2, -player, v, depth=VCF_DEPTH)
            for v in orig_vcf
        ):
            return {"move": int(c), "v1_rank": _v1_rank(totals, c)}
    return None


def _empty_cell() -> dict:
    return {
        "turns": 0,
        "threatened_turns": 0,
        "clean_missed_preventions": 0,
        "forced_blocks": 0,
        "quiet_turns": 0,
        "quiet_creator_created": 0,
        "quiet_vcf_created": 0,
        "vcf_threatened_turns": 0,
        "clean_missed_vcf_preventions": 0,
        "vcf_exposure_sum": 0,
        "vcf_exposure_n": 0,
        "regret_sum": 0.0,
        "regret_n": 0,
    }


def _empty_state() -> dict:
    return {
        "cells": {
            colour: {name: _empty_cell() for _, _, name in BANDS}
            for colour in ("black", "white")
        },
        "wins": {"black": 0, "white": 0, "draw": 0},
        "first_threat": [],  # per game: {game, black: {...}, white: {...}}
        "errors": [],  # White + Black clean-missed preventions (both kinds)
        "controls": [],  # threatened turns where the mover prevented
        "decisive": [],  # find_losing_moves output per loss
        "next_game": 0,
        "elapsed_seconds": 0.0,
    }


def _write_checkpoint(path: str, digest: str, state: dict) -> None:
    """Write the leg checkpoint atomically (temp file + rename, so
    a kill mid-write cannot corrupt it)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"input_digest": digest, "state": state}, f, indent=1)
    os.replace(tmp, path)


def analyse_leg(path: str, checkpoint: str | None = None) -> dict:
    """Analyse one selfplay_batch JSON; returns per-colour x band
    cells, per-game first-threat data, error instances, controls,
    and the decisive-phase table.

    ``checkpoint`` is a part file for restart safety: the leg's
    partial state is written after every 10 games, and a re-run
    resumes from the checkpoint instead of starting over (VM
    replacements kill the process; a 60-game leg takes ~1.5 h, so
    per-leg checkpointing alone would lose up to a whole leg).
    The checkpoint is keyed by the input file's SHA-256 — a
    checkpoint for different input is ignored."""
    data = json.load(open(path, encoding="utf-8"))
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    state = _empty_state()
    if checkpoint and os.path.exists(checkpoint):
        cached = json.load(open(checkpoint, encoding="utf-8"))
        if cached.get("input_digest") == digest:
            state = cached["state"]
            print(
                f"   {os.path.basename(path)}: resuming from game "
                f"{state['next_game']}/{len(data['records'])}",
                flush=True,
            )
    cells = state["cells"]
    wins = state["wins"]
    first_threat = state["first_threat"]
    errors = state["errors"]
    controls = state["controls"]
    decisive = state["decisive"]
    t_start = time.time()
    for gi, rec in enumerate(data["records"]):
        if gi < state["next_game"]:
            continue
        if gi % 10 == 0:
            print(
                f"   {os.path.basename(path)}: game {gi}/{len(data['records'])} "
                f"({time.time() - t_start + state['elapsed_seconds']:.0f}s)",
                flush=True,
            )
        moves = [int(m) for m in rec["moves"]]
        winner = rec["winner"]
        wins[winner if winner in ("black", "white") else "draw"] += 1
        board = game.empty_board()
        first = {
            "black": {"vcf": None, "creator": None},
            "white": {"vcf": None, "creator": None},
        }
        # Opponent VCF starts on the CURRENT board (stats 7/8).
        # Initialized for turn 0 (empty board: no four-makers, so
        # no VCF starts) and refreshed at the end of each turn
        # from the resulting board — the same enumeration serves
        # as this turn's our-VCF-after (stats 5/6) AND the next
        # turn's opp-VCF-before (same board, same player-as-
        # opponent), so the VCF enumeration runs once per turn
        # instead of ~twice. Same boards, same function —
        # identical statistics, ~40% less CPU.
        opp_vcf_before: list[int] = []
        for ply0, move in enumerate(moves):
            ply = ply0 + 1
            player = game.BLACK if ply0 % 2 == 0 else game.WHITE
            colour = "black" if player == game.BLACK else "white"
            cell = cells[colour][_band(ply)]
            cell["turns"] += 1
            opp_wins = scoring._winning_moves(board, -player)
            creators = scoring._opponent_creators(board, player)
            totals = _v1_totals(board, player)
            # Statistic 10 (diagnostic): v1-score regret.
            if totals:
                best = max(totals.values())
                played = totals.get(move)
                if played is not None:
                    cell["regret_sum"] += best - played
                    cell["regret_n"] += 1
            if len(opp_wins) == 1:
                cell["forced_blocks"] += 1
            threatened = bool(creators) and not opp_wins
            if threatened:
                cell["threatened_turns"] += 1
            quiet = not opp_wins and not creators
            if quiet:
                cell["quiet_turns"] += 1
            # VCF state before our move (stats 7/8) — carried
            # forward from the previous turn's end (see above).
            vcf_threatened = bool(opp_vcf_before) and not opp_wins
            if vcf_threatened:
                cell["vcf_threatened_turns"] += 1
            b2 = game.apply_move(board, move, player)
            # Threat creation (statistic 5, quiet turns only) and
            # first-threat tracking (statistic 6): first ply at
            # which the colour holds a VCF start / creator AFTER
            # its own move. Our VCF starts after our move ARE the
            # next turn's opp-VCF-before (same board), so one
            # enumeration serves both (except on the last turn,
            # where there is no next turn: compute only if a
            # statistic needs it).
            our_creators_after = scoring._opponent_creators(b2, -player)
            if quiet and our_creators_after:
                cell["quiet_creator_created"] += 1
            if first[colour]["creator"] is None and our_creators_after:
                first[colour]["creator"] = ply
            is_last = ply0 == len(moves) - 1
            if is_last:
                need = quiet or first[colour]["vcf"] is None
                our_vcf_after = _vcf_starts(b2, player) if need else []
            else:
                our_vcf_after = _vcf_starts(b2, player)
            if quiet and our_vcf_after:
                cell["quiet_vcf_created"] += 1
            if first[colour]["vcf"] is None and our_vcf_after:
                first[colour]["vcf"] = ply
            # Statistic 3: clean missed prevention (creator-based).
            if threatened:
                preventing = scoring._move_is_safe(board, move, player, creators)
                if not preventing:
                    any_prev = any(
                        scoring._move_is_safe(board, c, player, creators)
                        for c in game.legal_moves(board)
                    )
                    if any_prev:
                        cell["clean_missed_preventions"] += 1
                        prev_moves = [
                            int(c)
                            for c in scoring._candidate_moves(board)
                            if scoring._move_is_safe(board, int(c), player, creators)
                        ]
                        prev_ranks = [
                            _v1_rank(totals, c)
                            for c in prev_moves
                            if _v1_rank(totals, c) is not None
                        ]
                        errors.append(
                            {
                                "kind": "creator",
                                "leg": os.path.basename(path),
                                "game": rec["game"],
                                "colour": colour,
                                "ply": ply,
                                "band": _band(ply),
                                "played": _xy(move),
                                "played_v1_rank": _v1_rank(totals, move),
                                "n_creators": len(creators),
                                "n_opp_vcf_before": len(opp_vcf_before),
                                "n_preventing": len(prev_moves),
                                "best_preventing_v1_rank": min(prev_ranks)
                                if prev_ranks
                                else None,
                            }
                        )
                else:
                    controls.append(
                        {
                            "kind": "creator",
                            "leg": os.path.basename(path),
                            "game": rec["game"],
                            "colour": colour,
                            "ply": ply,
                            "band": _band(ply),
                            "played": _xy(move),
                            "played_v1_rank": _v1_rank(totals, move),
                            "n_creators": len(creators),
                            "n_opp_vcf_before": len(opp_vcf_before),
                        }
                    )
            # Statistics 7/8: VCF exposure + clean missed VCF
            # prevention (original VCF starts only — see module
            # docstring for the pre-registered caveat).
            if vcf_threatened:
                still_live = [
                    v
                    for v in opp_vcf_before
                    if evaluate.starts_forced_win(b2, -player, v, depth=VCF_DEPTH)
                ]
                cell["vcf_exposure_sum"] += len(still_live)
                cell["vcf_exposure_n"] += 1
                if still_live:
                    prev = _vcf_prevention_exists(
                        board, player, opp_vcf_before, totals
                    )
                    if prev is not None:
                        cell["clean_missed_vcf_preventions"] += 1
                        errors.append(
                            {
                                "kind": "vcf",
                                "leg": os.path.basename(path),
                                "game": rec["game"],
                                "colour": colour,
                                "ply": ply,
                                "band": _band(ply),
                                "played": _xy(move),
                                "played_v1_rank": _v1_rank(totals, move),
                                "n_opp_vcf_before": len(opp_vcf_before),
                                "n_opp_vcf_still_live": len(still_live),
                                "best_preventing": _xy(prev["move"]),
                                "best_preventing_v1_rank": prev["v1_rank"],
                            }
                        )
                    # else: no prevention existed — already lost,
                    # not an error.
            # Carry forward: our VCF starts after our move are the
            # next turn's opp-VCF-before (same board).
            if not is_last:
                opp_vcf_before = our_vcf_after
            board = b2
        first_threat.append(
            {"game": rec["game"], "black": first["black"], "white": first["white"]}
        )
        # Statistic 9: decisive phase for losses (both colours —
        # the loser is whichever colour lost this game).
        if winner in ("black", "white"):
            loser = game.WHITE if winner == "black" else game.BLACK
            out = find_losing_moves.analyse_loss(rec, loser)
            out["leg"] = os.path.basename(path)
            out["loser_colour"] = "black" if loser == game.BLACK else "white"
            decisive.append(out)
        state["next_game"] = gi + 1
        state["elapsed_seconds"] = round(
            state["elapsed_seconds"] + (time.time() - t_start), 1
        )
        t_start = time.time()
        if checkpoint and (gi + 1) % 10 == 0:
            _write_checkpoint(checkpoint, digest, state)
    if checkpoint:
        _write_checkpoint(checkpoint, digest, state)
    return {
        "leg": os.path.basename(path),
        "games": len(data["records"]),
        "wins": wins,
        "cells": cells,
        "first_threat": first_threat,
        "errors": errors,
        "controls": controls,
        "decisive": decisive,
        "elapsed_seconds": state["elapsed_seconds"],
    }


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 4) if d else None


def _diff_ci(p1: float, n1: int, p2: float, n2: int) -> list[float]:
    """Two-proportion z-test 95% CI for p1 - p2 (normal approx),
    clamped to [-1, 1] (a difference of proportions cannot leave
    that range; the clamp never changes whether the CI excludes
    0, which is what the patch rule reads)."""
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return [
        round(max(-1.0, diff - 1.96 * se), 4),
        round(min(1.0, diff + 1.96 * se), 4),
    ]


def summarize(legs: list[dict]) -> dict:
    """Aggregate cells across legs and compute the pre-registered
    colour gaps (White minus Black) with CIs, per band."""
    pooled = {
        colour: {name: _empty_cell() for _, _, name in BANDS}
        for colour in ("black", "white")
    }
    for leg in legs:
        for colour in ("black", "white"):
            for _, _, name in BANDS:
                for k, v in leg["cells"][colour][name].items():
                    pooled[colour][name][k] += v
    summary: dict = {"pooled_cells": pooled, "gaps": {}}
    # Error-rate gaps (the patch-rule statistics): White minus
    # Black, per band, with two-proportion CIs.
    for band in ("opening", "midgame", "endgame"):
        b, w = pooled["black"][band], pooled["white"][band]
        entry: dict = {}
        for key, num, den in (
            ("clean_missed_prevention_rate", "clean_missed_preventions", "threatened_turns"),
            ("clean_missed_vcf_prevention_rate", "clean_missed_vcf_preventions", "vcf_threatened_turns"),
        ):
            if b[den] and w[den]:
                pb, pw = b[num] / b[den], w[num] / w[den]
                entry[key] = {
                    "black": _rate(b[num], b[den]),
                    "white": _rate(w[num], w[den]),
                    "white_minus_black": round(pw - pb, 4),
                    "ci_95": _diff_ci(pw, w[den], pb, b[den]),
                    "n_black": b[den],
                    "n_white": w[den],
                }
        summary["gaps"][band] = entry
    # Per-leg + pooled rates for every statistic (readable table).
    summary["rates"] = {}
    for scope, cells in [("pooled", pooled)] + [
        (leg["leg"], leg["cells"]) for leg in legs
    ]:
        summary["rates"][scope] = {}
        for colour in ("black", "white"):
            summary["rates"][scope][colour] = {}
            for _, _, name in BANDS:
                c = cells[colour][name]
                summary["rates"][scope][colour][name] = {
                    "turns": c["turns"],
                    "threatened_turn_rate": _rate(c["threatened_turns"], c["turns"]),
                    "clean_missed_prevention_rate": _rate(
                        c["clean_missed_preventions"], c["threatened_turns"]
                    ),
                    "forced_block_share": _rate(c["forced_blocks"], c["turns"]),
                    "quiet_creator_creation_rate": _rate(
                        c["quiet_creator_created"], c["quiet_turns"]
                    ),
                    "quiet_vcf_creation_rate": _rate(
                        c["quiet_vcf_created"], c["quiet_turns"]
                    ),
                    "vcf_exposure_mean": _rate(
                        c["vcf_exposure_sum"], c["vcf_exposure_n"]
                    ),
                    "clean_missed_vcf_prevention_rate": _rate(
                        c["clean_missed_vcf_preventions"], c["vcf_threatened_turns"]
                    ),
                    "v1_regret_mean": round(c["regret_sum"] / c["regret_n"], 1)
                    if c["regret_n"]
                    else None,
                }
    # First-threat means.
    summary["first_threat"] = {}
    for scope, leg_list in [("pooled", legs)] + [
        (leg["leg"], [leg]) for leg in legs
    ]:
        summary["first_threat"][scope] = {}
        for colour in ("black", "white"):
            for kind in ("vcf", "creator"):
                vals = [
                    g[colour][kind]
                    for leg in leg_list
                    for g in leg["first_threat"]
                    if g[colour][kind] is not None
                ]
                summary["first_threat"][scope][f"{colour}_{kind}"] = {
                    "mean_ply": round(sum(vals) / len(vals), 1) if vals else None,
                    "n": len(vals),
                    "never": sum(
                        1
                        for leg in leg_list
                        for g in leg["first_threat"]
                        if g[colour][kind] is None
                    ),
                }
    # Decisive phase: losing-ply band x loser colour.
    summary["decisive"] = {}
    for leg in legs:
        for d in leg["decisive"]:
            key = d["loser_colour"]
            band = _band(d["losing_ply"] + 1) if d.get("losing_ply") is not None else "none"
            summary["decisive"].setdefault(key, {}).setdefault(band, 0)
            summary["decisive"][key][band] += 1
            if d.get("already_lost"):
                summary["decisive"][key].setdefault("already_lost", 0)
                summary["decisive"][key]["already_lost"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="selfplay_batch JSON files")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    legs = []
    parts = []
    for path in args.files:
        # Per-leg checkpoint (restart-safe): the analysis takes
        # ~2-3 h for the full corpus and VM replacements kill the
        # process; the leg's partial state is checkpointed after
        # every 10 games (see analyse_leg), so a re-run resumes
        # mid-leg instead of starting over. Part files are deleted
        # after the final output is written.
        part = f"{args.out}.{os.path.basename(path)}.part.json"
        parts.append(part)
        print(f"== analysing {path} ==", flush=True)
        leg = analyse_leg(path, checkpoint=part)
        legs.append(leg)
        print(
            f"   {leg['games']} games, {leg['elapsed_seconds']}s",
            flush=True,
        )
    out = {"legs": legs, "summary": summarize(legs)}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for part in parts:
        if os.path.exists(part):
            os.remove(part)
    print(f"wrote {args.out}")
    # Print the patch-rule read-off.
    print("== patch-rule gaps (White minus Black) ==")
    for band, entry in out["summary"]["gaps"].items():
        for key, g in entry.items():
            print(
                f"  {band} {key}: black {g['black']} (n={g['n_black']}) "
                f"white {g['white']} (n={g['n_white']}) "
                f"diff {g['white_minus_black']} CI {g['ci_95']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

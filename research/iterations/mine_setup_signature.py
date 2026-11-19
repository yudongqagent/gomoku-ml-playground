#!/usr/bin/env python3
"""Mine setup-signature instances from v27's losses (batch 11 H1).

The setup signature (iter-35 §2.1, v0-g4/g32 ply 52 — ONE distinct
position): at a v27 turn in a lost game, after v27's actual move the
opponent holds >= 1 "setup" moves (a quiet move after which the
opponent holds >= 2 open-four creators) with 0 VCF starts and 0
creators, and some alternative move removes ALL opponent setups
(0 VCF starts / 0 creators / 0 setups after it) while the actual
move leaves them live. The verified saves in the known instance do
exactly that (2 -> 0).

This script is the FILTER pass (exact analysis only, no playouts):
for every v27 loss in the batch-10 legs (v33/v34/v35 vs v27, normal
+ swap) and v27's swap leg vs v7 — all fresh, none worked up in
iter-35 — it scans v27's last 4 turns and flags turns where the
pre-condition holds and a removing move exists. Candidates are then
verified with playouts + the exact audit by verify_setup_candidate.py.

Filter order (cheap checks first): VCF starts after the actual move
(0 required — else the VCF-prevention class, CLOSED), creators after
the actual move (0 required — else a creator-threat class), then
setups after the actual move (>= 1 required). Second pass tests, as
removing moves: the setup squares themselves, the creators the
opponent would hold after each setup (occupying one blocks the
setup), the top-5 1-ply moves, and the 400-sim MCTS proposal.
Validated on the known instance: the net catches both saves,
(9,4) (a setup square) and (5,8) (a creator-after-setup square).

Resumable: results are appended per game to the --out JSON; a
relaunch skips games already recorded. Losses are deduplicated by
move sequence across all files (the batch-10 legs contain
seed-replay duplicates).

Usage:
    python research/iterations/mine_setup_signature.py --out \\
        research/iterations/setup-mining.json
    python research/iterations/mine_setup_signature.py --out /tmp/x.json \\
        --only batch-v27-vs-v0.json --only-game 4   # validate on known instance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "ml"))

import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
from gomoku_ml import evaluate, game, scoring  # noqa: E402
from targeted_playout import _creators  # noqa: E402
from workup_loss import _PlayerCache  # noqa: E402

B10 = "/home/hatch/workspace/projects/gomoku-b10/research/iterations"

# (file, winner version for later verification). v27 is engine-b in
# the v33/v34/v35 legs and engine-a in the v27-vs-v7 swap leg.
FILES = [
    ("batch-v33-vs-v27.json", 33),
    ("batch-v34-vs-v27.json", 34),
    ("batch-v35-vs-v27.json", 35),
    ("batch-v33-vs-v27-swap.json", 33),
    ("batch-v34-vs-v27-swap.json", 34),
    ("batch-v35-vs-v27-swap.json", 35),
    ("batch-v27-vs-v7-swap.json", 7),
]

# A single move is implausible as a full removal when the opponent
# holds more than this many setups; such turns are recorded as
# capped, not tested (revisit if the yield is close to the bar).
MAX_SETUPS_TO_TEST = 6
LAST_TURNS = 4


def _xy(move: int) -> list[int]:
    return [move % 15, move // 15]


def _window_candidates(board: np.ndarray, player: int,
                       min_stones: int) -> list[int]:
    """Empty cells in a 5-window (any of the 4 directions) holding
    >= ``min_stones`` ``player`` stones. Same shape as the v33
    four-creator optimisation (iter-33 §2.3)."""
    cands: set[int] = set()
    for dy, dx, ys, xs in (
        (0, 1, range(15), range(11)),
        (1, 0, range(11), range(15)),
        (1, 1, range(11), range(11)),
        (1, -1, range(11), range(4, 15)),
    ):
        for y in ys:
            for x in xs:
                stones = 0
                empties: list[int] = []
                for i in range(5):
                    v = board[y + i * dy, x + i * dx]
                    if v == player:
                        stones += 1
                    elif v == game.EMPTY:
                        empties.append(
                            game.xy_to_move(x + i * dx, y + i * dy))
                if stones >= min_stones:
                    cands.update(empties)
    return sorted(cands)


def _creators_windowed(board: np.ndarray, player: int,
                       limit: int = 0) -> list[int]:
    """Creators for ``player`` (same definition as
    targeted_playout._creators), via the v33 window filter: a
    creator completes a four, so it sits in a 5-window holding >= 3
    ``player`` stones. Valid only when ``player`` holds NO winning
    move on ``board`` (else nearly every move 'creates' a four —
    the pre-existing win survives); in that case fall back to the
    full enumeration. ``limit`` > 0 stops after that many found."""
    if scoring._winning_moves(board, player):
        cands = scoring._candidate_moves(board)
    else:
        cands = _window_candidates(board, player, 3)
    out = []
    for c in cands:
        b2 = game.apply_move(board, c, player)
        if len(scoring._winning_moves(b2, player)) >= 2:
            out.append(c)
            if limit > 0 and len(out) >= limit:
                break
    return out


def _setup_creates_two(board: np.ndarray, m: int, player: int) -> bool:
    """True iff playing ``m`` (a setup candidate for ``player`` on
    ``board``) leaves ``player`` with >= 2 creators. Tighter than
    enumerating all creators: a creator c after m must complete a
    four that INCLUDES m (else c was already a creator before m —
    ``player`` holds 0 creators on the input board, the
    ``_setups_limited`` precondition — since a four not using m
    exists with or without it). So c sits in a 5-window THROUGH m
    holding >= 2 other ``player`` stones; only those windows'
    empty cells are tested. Returns False when m leaves an
    immediate winning move (not a setup)."""
    b2 = game.apply_move(board, m, player)
    if scoring._winning_moves(b2, player):
        return False
    x, y = m % 15, m // 15
    cands: set[int] = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for i in range(5):
            # 5-windows containing m: start i steps back along the
            # direction (clipped to the board).
            sy, sx = y - i * dy, x - i * dx
            ey, ex = sy + 4 * dy, sx + 4 * dx
            if not (0 <= sy < 15 and 0 <= sx < 15
                    and 0 <= ey < 15 and 0 <= ex < 15):
                continue
            stones = 0
            empties: list[int] = []
            for j in range(5):
                v = board[sy + j * dy, sx + j * dx]
                if v == player:
                    stones += 1
                elif v == game.EMPTY:
                    empties.append(
                        game.xy_to_move(sx + j * dx, sy + j * dy))
            if stones >= 2:
                cands.update(empties)
    cands.discard(m)  # m is occupied on b2
    found = 0
    for c in cands:
        b3 = game.apply_move(b2, c, player)
        if len(scoring._winning_moves(b3, player)) >= 2:
            found += 1
            if found >= 2:
                return True
    return False


def _has_vcf_start(board: np.ndarray, player: int) -> bool:
    """True iff ``player`` holds a VCF start on ``board`` (zero /
    non-zero only — the mining filter never needs the exact count).
    A VCF start must be a four-creating move
    (``starts_forced_win`` requires it), so only the four-creators
    are tested (same restriction as the VCF check in
    mcts_heuristic), with early exit at the first start. This is
    what makes VCF-rich boards affordable: the full
    ``_vcf_starts`` enumeration took minutes on boards with a
    dozen starts (mining g49 ply 78). Validated against the
    recorded exact counts in setup-mining.json (see
    research/iterations/setup-mining-ab.json, "vcf" section).

    BUG FIX (found by the verification audit, v35 g49/g65): an
    IMMEDIATE winning move is itself a VCF start
    (``starts_forced_win`` returns True for it at once) but is
    NOT necessarily a four-creating move (after the win there is
    usually no further winning move), so the four-creator
    restriction misses it. After the test move (1,9) in v35 g49
    ply 72, White holds the immediate win (0,12) — the square
    v27's actual move blocked — and the first version of this
    function wrongly recorded a clean 0/0/0 removal. The
    winning-moves check below closes the hole: the function is
    now (player holds a winning move) OR (some four-creator
    starts a forced win), which is exactly the ``_vcf_starts``
    non-empty condition. Games mined before the fix may contain
    false pre-conditions / false candidates; the verification's
    slow exact audit is the safeguard (it caught both), and the
    verified-instance count is unaffected (the bug only ADDS
    candidates, never removes true ones — a missed VCF start can
    only open the gate wider).
    """
    if scoring._winning_moves(board, player):
        return True  # an immediate win is itself a VCF start
    for m in evaluate.four_creating_moves(board, player):
        if evaluate.starts_forced_win(board, player, m):
            return True
    return False


def _setups_limited(board: np.ndarray, player: int,
                    limit: int) -> list[int]:
    """Setup moves for ``player`` (same definition as
    targeted_playout._setups), stopping after ``limit`` found.
    limit <= 0 means no limit.

    FAST VERSION (window-filtered; A/B-validated against the slow
    exact version — see research/iterations/setup-mining-ab.json):
    a setup move must lie in a 5-window holding >= 2 ``player``
    stones. Proof: a setup leaves no winning move and creates >= 2
    creators; take a creator c — playing c leaves >= 2 winning
    moves, each completing a five that includes c (else it would
    already be a winning move after the setup); since ``player``
    holds 0 creators on the input board, c was not a creator
    before the setup, so at least one of those fives includes the
    setup move itself plus 2 other ``player`` stones. The creator
    test itself is tightened the same way (see
    :func:`_setup_creates_two`): a new creator's four must include
    the setup move. PRECONDITION: ``player`` holds 0 creators on
    ``board`` (both call sites check creators first); if the
    player already held >= 2 creators, every quiet move would be
    a setup and the filter would miss them.
    """
    out = []
    for m in _window_candidates(board, player, 2):
        if _setup_creates_two(board, m, player):
            out.append(m)
            if limit > 0 and len(out) >= limit:
                break
    return out


def _topk_1ply(board: np.ndarray, player: int, actual: int,
               k: int = 5) -> list[int]:
    flat = board.reshape(-1).tolist()
    scored = sorted(
        (
            (scoring._score_move_flat(flat, m, player, 1.0), m)
            for m in scoring._candidate_moves(board)
        ),
        key=lambda sm: (-sm[0], sm[1]),
    )
    return [m for _s, m in scored[:k] if m != actual]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", default=None,
                        help="only scan this batch file (validation)")
    parser.add_argument("--only-game", type=int, default=None)
    parser.add_argument("--include-known", action="store_true",
                        help="also scan v27's own normal battery "
                        "(already worked up in iter-35; validation only)")
    args = parser.parse_args()

    files = FILES
    if args.include_known:
        files = files + [("batch-v27-vs-v0.json", 0),
                         ("batch-v27-vs-v7.json", 7)]
    if args.only:
        files = [(f, w) for f, w in files if f == args.only]
        if not files:
            files = [(args.only, 0)]
    # Resume: load existing results, skip completed games.
    results: dict = {"games": [], "meta": {}}
    done: set[tuple[str, int]] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            results = json.load(f)
        done = {(g["file"], g["game"]) for g in results["games"]}
        print(f"resuming: {len(done)} games already recorded",
              file=sys.stderr, flush=True)

    # Collect v27 losses, deduplicated by move sequence.
    losses = []
    seen_moves: set[tuple[int, ...]] = set()
    dup_count = 0
    for fname, winner_ver in files:
        path = os.path.join(B10, fname)
        with open(path, encoding="utf-8") as f:
            batch = json.load(f)
        v27_is_a = batch["engine_a"].endswith("v27")
        for rec in batch["records"]:
            if rec["winner"] == "draw":
                continue
            v27_lost = (not rec["winner_is_a"]) if v27_is_a \
                else rec["winner_is_a"]
            if not v27_lost:
                continue
            key = tuple(rec["moves"])
            if key in seen_moves:
                dup_count += 1
                continue
            seen_moves.add(key)
            if args.only_game is not None and rec["game"] != args.only_game:
                continue
            losses.append((fname, winner_ver, v27_is_a, rec))
    print(f"v27 losses to scan: {len(losses)} unique "
          f"({dup_count} move-sequence duplicates dropped)",
          file=sys.stderr, flush=True)

    cache = _PlayerCache()
    t_start = time.time()
    n_turns = n_precond = n_capped = n_candidates = 0
    for fname, winner_ver, v27_is_a, rec in losses:
        if (fname, rec["game"]) in done:
            continue
        a_colour = game.BLACK if rec["a_colour"] == "black" else game.WHITE
        v27_colour = a_colour if v27_is_a else -a_colour
        moves = rec["moves"]
        v27_turns = [
            ply for ply in range(len(moves))
            if (game.BLACK if ply % 2 == 0 else game.WHITE) == v27_colour
        ][-LAST_TURNS:]
        entry = {
            "file": fname, "game": rec["game"],
            "winner_version": winner_ver,
            "v27_colour": "black" if v27_colour == game.BLACK else "white",
            "plies": len(moves), "turns": [],
        }
        for ply in v27_turns:
            n_turns += 1
            board = game.empty_board()
            for i, m in enumerate(moves[:ply]):
                board = game.apply_move(
                    board, m, game.BLACK if i % 2 == 0 else game.WHITE)
            actual = moves[ply]
            opp = -v27_colour
            after_actual = game.apply_move(board, actual, v27_colour)
            # Cheap checks first, in cost order: creators (0.07s)
            # before VCF starts (expensive on rich boards — the
            # profile showed _vcf_starts dominating). Either count
            # > 0 excludes the turn, so the VCF number is only
            # computed when creators == 0 (recorded as null when
            # skipped — the excluded-turn class split is not needed
            # for mining).
            creators_a = _creators(after_actual, opp)
            turn = {
                "ply": ply, "played": _xy(actual),
                "vcf_after": None,
                "creators_after": len(creators_a),
                "setups_after": None, "setups": [],
                "preconditioned": False, "capped": False,
                "tests": [], "removing_moves": [],
            }
            if creators_a:
                entry["turns"].append(turn)
                continue
            vcf_a = _has_vcf_start(after_actual, opp)
            # vcf_after is 0/1 (0 vs >= 1) from the early-exit test;
            # the exact count is never needed for the filter.
            turn["vcf_after"] = 1 if vcf_a else 0
            turn["vcf_capped"] = True
            if vcf_a:
                entry["turns"].append(turn)
                continue
            setups_a = _setups_limited(after_actual, opp, 0)
            turn["setups_after"] = len(setups_a)
            turn["setups"] = [_xy(s) for s in setups_a]
            if not setups_a:
                entry["turns"].append(turn)
                continue
            n_precond += 1
            turn["preconditioned"] = True
            if len(setups_a) > MAX_SETUPS_TO_TEST:
                n_capped += 1
                turn["capped"] = True
                entry["turns"].append(turn)
                continue
            # Second pass: candidate removing moves, tested in source
            # priority order (setup squares first — the most likely
            # removers). Stop at the first full removal: for mining,
            # one removing move per turn establishes the instance;
            # the full per-source breakdown is recovered later for
            # the few candidate instances only.
            test_moves: list[tuple[int, str]] = []
            seen_test: set[int] = set()

            def _add_test(m: int, source: str) -> None:
                if m != actual and m not in seen_test:
                    seen_test.add(m)
                    test_moves.append((m, source))

            for s in setups_a:
                _add_test(s, "setup-square")
            for s in setups_a:
                b_setup = game.apply_move(board, s, opp)
                for c in _creators(b_setup, opp):
                    _add_test(c, "creator-after-setup")
            for m in _topk_1ply(board, v27_colour, actual):
                _add_test(m, "1-ply-top5")

            def _test_removal(m: int, source: str) -> bool:
                """True iff playing ``m`` leaves the opponent with
                0 VCF starts / 0 creators / 0 setups. Checks in
                cost order (creators, setups with early exit, VCF
                only when both are 0)."""
                after_test = game.apply_move(board, m, v27_colour)
                creators_t = _creators(after_test, opp)
                setups_t = [] if creators_t else _setups_limited(
                    after_test, opp, 1)
                vcf_n = None
                if not creators_t and not setups_t:
                    vcf_n = 1 if _has_vcf_start(after_test, opp) else 0
                removes = (not creators_t and not setups_t
                           and vcf_n == 0)
                turn["tests"].append({
                    "move": _xy(m), "source": source,
                    "vcf": vcf_n, "creators": len(creators_t),
                    "setups": len(setups_t), "removes_all": removes,
                })
                if removes:
                    turn["removing_moves"].append(
                        {"move": _xy(m), "source": source})
                    print(f"  CANDIDATE {fname} g{rec['game']} ply "
                          f"{ply}: removing {_xy(m)} ({source})",
                          file=sys.stderr, flush=True)
                return removes

            found = False
            for m, source in test_moves:
                if game.is_winning_move(board, m, v27_colour):
                    continue  # an immediate win ends the game; not a prevention
                if _test_removal(m, source):
                    n_candidates += 1
                    found = True
                    break
            # The 400-sim proposal is computed LAZILY (it costs
            # ~7s): only if no earlier source found a removal.
            if not found:
                deep = cache.get(27, 0)
                mcts = getattr(deep, "_mcts", None)
                if mcts is not None:
                    old_sims = mcts.sims
                    mcts.sims = 400
                    try:
                        m400 = deep.choose_move(board, v27_colour)
                    finally:
                        mcts.sims = old_sims
                    if (m400 != actual and m400 not in seen_test
                            and not game.is_winning_move(
                                board, m400, v27_colour)):
                        if _test_removal(m400, "mcts-400"):
                            n_candidates += 1
            entry["turns"].append(turn)
        results["games"].append(entry)
        done.add((fname, rec["game"]))
        print(f"  {fname} g{rec['game']}: {len(entry['turns'])} turns, "
              f"{sum(1 for t in entry['turns'] if t['preconditioned'])} "
              f"pre-conditioned, elapsed {time.time() - t_start:.0f}s",
              file=sys.stderr, flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            f.write("\n")
    results["meta"] = {
        "files": [f for f, _ in files],
        "unique_v27_losses": len(losses),
        "duplicates_dropped": dup_count,
        "turns_scanned": n_turns,
        "preconditioned_turns": n_precond,
        "capped_turns": n_capped,
        "candidate_instances": n_candidates,
        "elapsed_seconds": time.time() - t_start,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"DONE: {n_turns} turns, {n_precond} pre-conditioned, "
          f"{n_capped} capped, {n_candidates} candidate instances, "
          f"{time.time() - t_start:.0f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

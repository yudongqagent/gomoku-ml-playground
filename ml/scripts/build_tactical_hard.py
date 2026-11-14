#!/usr/bin/env python3
"""Build the harder tactical suite (v2) for the auto-research program.

Batch 2, Step 0. Extracts positions from batch-1 lost games (the
decisive open-four positions) and adds hand-constructed VCF/VCT
positions, then writes ``gomoku_ml/tactical_positions_hard.json``.

Verification properties (all computed independently of the engines,
from ``gomoku_ml.game`` / ``gomoku_ml.evaluate`` only):

* **prevention** — the opponent holds >= 1 open-four creator (a move
  leaving them >= 2 winning moves) and no immediate winning move.
  A move is *preventing* iff it wins immediately, or it is forcing
  (leaves the mover >= 1 winning move while the opponent has no
  immediate win), or it leaves the opponent with zero live creators.
  The expected list is EXACTLY the set of preventing moves (scanned
  over all legal moves).
* **setup** (VCT-style attack) — the side to move holds NO creator
  yet. A move is a *setup* iff after it the opponent has no immediate
  win, and for EVERY opponent reply there is a creator move left
  that leaves us >= 2 winning moves with the opponent holding no
  immediate win. The expected list is EXACTLY the set of setup moves
  (scanned over empty points within Chebyshev distance 4 of the
  mover's stones — a setup move further away cannot interact with
  any of the mover's stones in a 5-window).
* **vcf** — expect type ``vcf``: the move must start a forced win by
  consecutive fours (``evaluate.starts_forced_win``). Documented
  forcing first moves and traps are checked here.

Extraction (batch-1 games only — no batch-2 version is judged
against a suite it was tuned on):

* decisive games only; find the winner's LAST move before the final
  move that left the winner >= 2 winning moves (the decisive open
  four). Games won without an open four are skipped.
* prevention: the board 1 ply before the creator (loser to move),
  kept only if the loser's actual game move was NOT preventing.
* setup: the board 2 plies before the creator (winner to move),
  kept only if the winner's actual game move IS a setup.

Usage:
    python ml/scripts/build_tactical_hard.py [--extract-only]
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import evaluate, game  # noqa: E402

REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ITER_DIR = os.path.join(REPO_ROOT, "research", "iterations")
BATCH_FILES = [
    "batch-v1-vs-v0-swap.json",
    "batch-v1-vs-v1-swap.json",
    "batch-v2-vs-v0.json",
    "batch-v2-vs-v1-swap.json",
    "batch-v2-vs-v1.json",
    "batch-v3-vs-v0.json",
    "batch-v3-vs-v1-swap.json",
    "batch-v3-vs-v1.json",
]

# ---------------------------------------------------------------------------
# verification properties (independent of the engines)
# ---------------------------------------------------------------------------


def _near_radius1(board: np.ndarray, player: int) -> list[int]:
    """Empty points adjacent (Chebyshev distance 1) to a ``player``
    stone. A creator must be adjacent to a friendly stone: it
    completes a four in a 5-window, and in a 5-window holding 4
    stones every cell is adjacent to a stone."""
    near: set[int] = set()
    for y, x in zip(*((board == player).nonzero())):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return sorted(near)


def creators(board: np.ndarray, player: int) -> list[int]:
    """Moves after which ``player`` has >= 2 winning moves (open-four
    creators). A creator must be adjacent to a friendly stone, so
    only radius-1 points are scanned (exact — see
    :func:`_near_radius1`)."""
    out: list[int] = []
    for m in _near_radius1(board, player):
        b2 = game.apply_move(board, m, player)
        if len(evaluate.winning_moves(b2, player)) >= 2:
            out.append(m)
    return out


def creator_winning_squares(board: np.ndarray, player: int,
                            creator_list: list[int] | None = None
                            ) -> dict[int, set[int]]:
    """For each creator ``c`` of ``player``, the set of winning moves
    ``player`` would hold after playing ``c``.

    Key fact for the prevention check: our move ``m`` removes creator
    ``c`` iff ``m == c`` or ``m`` occupies one of ``c``'s winning
    squares. (Our stone can only affect ``c``'s four by occupying
    the creator square itself or a winning square — every other cell
    of the four's window already holds an opponent stone.)
    """
    out: dict[int, set[int]] = {}
    for c in creator_list if creator_list is not None else creators(board, player):
        b2 = game.apply_move(board, c, player)
        out[c] = set(evaluate.winning_moves(b2, player))
    return out


def _near_of(board: np.ndarray, player: int, radius: int) -> set[int]:
    """Empty points within Chebyshev ``radius`` of a ``player`` stone."""
    near: set[int] = set()
    for y, x in zip(*((board == player).nonzero())):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return near


def is_preventing(board: np.ndarray, move: int, player: int,
                  opp_creators: list[int] | None = None,
                  opp_creator_wins: dict[int, set[int]] | None = None,
                  near4: set[int] | None = None) -> bool:
    """True if ``move`` prevents the opponent's open-four threat: it
    wins at once, or it is forcing (leaves us >= 1 winning move and
    the opponent no immediate win), or it leaves the opponent with
    zero live creators.

    ``near4`` (optional) is the set of empty points within Chebyshev
    distance 4 of the mover's stones: a forcing move must lie in it
    (a four's window holds the move plus 3 friendly stones), so
    moves outside it skip the forcing check."""
    if game.is_winning_move(board, move, player):
        return True
    if opp_creators is None:
        opp_creators = creators(board, -player)
    if opp_creator_wins is None:
        opp_creator_wins = creator_winning_squares(
            board, -player, opp_creators)
    # Cheap check first: does the move remove every creator?
    # (Set lookups only — no board copies.)
    removes_all = True
    for c in opp_creators:
        # The creator survives unless our move occupies it or one of
        # its winning squares AND that leaves it with < 2 wins.
        # (A double-four creator has 3 winning squares: blocking one
        # still leaves 2, so it survives.)
        wins = opp_creator_wins[c]
        if move == c:
            continue
        if move in wins and len(wins) <= 2:
            continue
        removes_all = False
        break
    if removes_all:
        return True
    if near4 is not None and move not in near4:
        return False  # cannot be forcing: too far from our stones
    b2 = game.apply_move(board, move, player)
    if evaluate.winning_moves(b2, player):
        # Forcing four: the opponent must answer the threat instead
        # of playing a creator — unless the opponent wins at once.
        return not evaluate.winning_moves(b2, -player)
    return False


def preventing_moves(board: np.ndarray, player: int) -> list[int]:
    """EXACTLY the preventing moves (all legal moves scanned)."""
    opp_creators = creators(board, -player)
    opp_creator_wins = creator_winning_squares(board, -player, opp_creators)
    near4 = _near_of(board, player, 4)
    return [
        int(m) for m in game.legal_moves(board)
        if is_preventing(board, int(m), player, opp_creators,
                         opp_creator_wins, near4)
    ]


def _near_union(board: np.ndarray, radius: int = 2) -> list[int]:
    """Empty points within Chebyshev ``radius`` of any stone."""
    near: set[int] = set()
    for y, x in zip(*((board != game.EMPTY).nonzero())):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return sorted(near)


def _creator_survives_after_reply(board_after_m: np.ndarray, reply: int,
                                  creator_list: list[int],
                                  player: int) -> bool:
    """True if, after the opponent's ``reply``, some creator in
    ``creator_list`` still leaves us >= 2 winning moves with the
    opponent holding no immediate win."""
    b_r = game.apply_move(board_after_m, reply, -player)
    for c in creator_list:
        if c == reply:
            continue
        b_c = game.apply_move(b_r, c, player)
        if len(evaluate.winning_moves(b_c, player)) >= 2:
            if not evaluate.winning_moves(b_c, -player):
                return True
    return False


def is_setup(board: np.ndarray, move: int, player: int) -> bool:
    """True if ``move`` is a 2-ply setup: after it the opponent has no
    immediate win, and for EVERY opponent reply we still hold a
    creator (a move leaving us >= 2 winning moves with the opponent
    holding no immediate win).

    Only "dangerous" replies need checking — a reply can only break
    a setup by (a) occupying a creator square, (b) occupying one of
    a creator's winning squares, or (c) creating a four for the
    opponent (leaving them a winning move). Any other reply leaves
    every creator intact and the opponent with no winning move, so
    any creator survives it. This reduction is exact (see the
    proof in the module docstring of the test file).
    """
    if game.is_winning_move(board, move, player):
        return False  # an immediate win is not a setup
    b2 = game.apply_move(board, move, player)
    if evaluate.winning_moves(b2, -player):
        return False  # opponent wins at once instead of replying
    creator_list = creators(b2, player)
    if not creator_list:
        return False
    creator_wins = creator_winning_squares(b2, player, creator_list)
    dangerous: set[int] = set(creator_list)
    for wins in creator_wins.values():
        dangerous |= wins
    dangerous |= set(evaluate.four_creating_moves(b2, -player))
    dangerous &= set(int(m) for m in game.legal_moves(b2))
    for r in sorted(dangerous):
        if game.is_winning_move(b2, r, -player):
            return False  # opponent wins at once with this reply
        if not _creator_survives_after_reply(b2, r, creator_list, player):
            return False
    return True


def setup_moves(board: np.ndarray, player: int) -> list[int]:
    """EXACTLY the setup moves. Candidates are empty points within
    Chebyshev distance 4 of the mover's stones (a setup move further
    away shares no 5-window with any friendly stone and cannot
    create a creator)."""
    out: list[int] = []
    for m in _near_union(board, radius=4):
        # Cheap pre-filter: a setup needs >= 1 creator after the move.
        b2 = game.apply_move(board, m, player)
        if evaluate.winning_moves(b2, -player):
            continue
        if not creators(b2, player):
            continue
        if is_setup(board, m, player):
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# game replay / extraction
# ---------------------------------------------------------------------------


def replay(moves: list[int]) -> list[np.ndarray]:
    """Boards after each prefix: boards[i] is the position before
    move i (boards[0] is empty)."""
    boards = [game.empty_board()]
    board = boards[0]
    for i, m in enumerate(moves):
        board = game.apply_move(board, int(m), game.BLACK if i % 2 == 0 else game.WHITE)
        boards.append(board)
    return boards


def find_decisive_creator(moves: list[int], winner: int) -> int | None:
    """Ply index of the winner's LAST move before the final move that
    left the winner >= 2 winning moves (the decisive open four).
    None if the win did not come via an open four."""
    boards = replay(moves)
    last: int | None = None
    for i in range(len(moves) - 1):  # exclude the final (winning) move
        player = game.BLACK if i % 2 == 0 else game.WHITE
        if player != winner:
            continue
        if len(evaluate.winning_moves(boards[i + 1], winner)) >= 2:
            last = i
    return last


def _stones(board: np.ndarray) -> tuple[list[list[int]], list[list[int]]]:
    black = [[int(x), int(y)] for y, x in zip(*((board == game.BLACK).nonzero()))]
    white = [[int(x), int(y)] for y, x in zip(*((board == game.WHITE).nonzero()))]
    return black, white


def _board_key(board: np.ndarray, to_move: int) -> tuple:
    return (board.tobytes(), to_move)


def extract_from_record(moves: list[int], winner: int, source: str,
                        game_no: int) -> tuple[dict | None, dict | None, str]:
    """Extract (prevention, setup) candidate positions from one
    decisive game. Returns (prevention_pos, setup_pos, skip_reason)
    where each pos is a candidate dict or None.

    Candidates carry the board and the game move only — the
    EXHAUSTIVE expected-move lists are computed later, at assembly
    time, and only for selected positions (the exhaustive scans are
    the expensive part: ~1-4 s per position).
    """
    creator_ply = find_decisive_creator(moves, winner)
    if creator_ply is None:
        return None, None, "no open-four creator before the win"
    if creator_ply < 2:
        return None, None, "creator too early (ply < 2)"
    boards = replay(moves)
    loser = -winner
    prevention = None
    setup = None

    # --- prevention: board 1 ply before the creator, loser to move ---
    # boards[k] is the position BEFORE move k. The creator is move
    # creator_ply (winner); the loser's turn before it is move
    # creator_ply - 1, played on boards[creator_ply - 1].
    prev_board = boards[creator_ply - 1]
    game_move = moves[creator_ply - 1]
    if not evaluate.winning_moves(prev_board, loser) and \
            not evaluate.winning_moves(prev_board, winner):
        opp_creators = creators(prev_board, winner)
        if opp_creators:
            opp_wins = creator_winning_squares(
                prev_board, winner, opp_creators)
            if not is_preventing(prev_board, game_move, loser,
                                 opp_creators, opp_wins):
                prevention = {
                    "board": prev_board,
                    "to_move": loser,
                    "creators": len(opp_creators),
                    "game_move": game.move_to_xy(game_move),
                    "source": source,
                    "game_no": game_no,
                    "creator_ply": creator_ply,
                    "plies": len(moves),
                }

    # --- setup: board 2 plies before the creator, winner to move ---
    setup_board = boards[creator_ply - 2]
    setup_game_move = moves[creator_ply - 2]
    if not evaluate.winning_moves(setup_board, winner) and \
            not evaluate.winning_moves(setup_board, loser):
        if not creators(setup_board, winner):
            if is_setup(setup_board, setup_game_move, winner):
                setup = {
                    "board": setup_board,
                    "to_move": winner,
                    "game_move": game.move_to_xy(setup_game_move),
                    "source": source,
                    "game_no": game_no,
                    "creator_ply": creator_ply,
                    "plies": len(moves),
                }
    return prevention, setup, ""


def load_batch_games() -> list[tuple[list[int], int, str, int]]:
    """All decisive games from the batch-1 record files.

    Returns (moves, winner, source, game_no). Winner is BLACK/WHITE.
    """
    games: list[tuple[list[int], int, str, int]] = []
    for fname in BATCH_FILES:
        path = os.path.join(ITER_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data["records"]:
            if rec["winner"] in ("black", "white"):
                winner = game.BLACK if rec["winner"] == "black" else game.WHITE
                games.append((rec["moves"], winner, fname, rec["game"]))
    # v1-black-losses.json: moves_xy, Black first, all White wins.
    path = os.path.join(ITER_DIR, "v1-black-losses.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for rec in data["losses"]:
        moves = [game.xy_to_move(x, y) for x, y in rec["moves_xy"]]
        games.append((moves, game.WHITE, "v1-black-losses.json", rec["game"]))
    return games


def extract_all() -> tuple[list[dict], list[dict], dict]:
    """Run the extraction over all batch-1 decisive games."""
    games = load_batch_games()
    preventions: list[dict] = []
    setups: list[dict] = []
    seen: set[tuple] = set()
    stats: dict[str, int] = {"games": len(games), "skipped": 0}
    skip_reasons: dict[str, int] = {}
    for idx, (moves, winner, source, game_no) in enumerate(games):
        if idx % 100 == 0:
            print(f"  ... {idx}/{len(games)} games scanned "
                  f"({len(preventions)} prevention, {len(setups)} setup)",
                  file=sys.stderr, flush=True)
        prevention, setup, reason = extract_from_record(
            moves, winner, source, game_no
        )
        if reason:
            stats["skipped"] += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        for cand, bucket in ((prevention, preventions), (setup, setups)):
            if cand is None:
                continue
            key = _board_key(cand["board"], cand["to_move"])
            if key in seen:
                skip_reasons["duplicate board"] = skip_reasons.get(
                    "duplicate board", 0) + 1
                continue
            seen.add(key)
            bucket.append(cand)
    stats["skip_reasons"] = skip_reasons
    return preventions, setups, stats


def loser_engines() -> dict[tuple[str, int], str]:
    """Map (source file, game no) -> the engine that LOST the game."""
    out: dict[tuple[str, int], str] = {}
    for fname in BATCH_FILES:
        path = os.path.join(ITER_DIR, fname)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for rec in data["records"]:
            if rec["winner"] in ("black", "white"):
                loser = (data["engine_b"] if rec["winner_is_a"]
                         else data["engine_a"])
                out[(fname, rec["game"])] = loser
    path = os.path.join(ITER_DIR, "v1-black-losses.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for rec in data["losses"]:
        out[("v1-black-losses.json", rec["game"])] = "scoring-v1"
    return out


FILE_TAGS = {
    "batch-v1-vs-v0-swap.json": "v1v0s",
    "batch-v1-vs-v1-swap.json": "v1v1s",
    "batch-v2-vs-v0.json": "v2v0",
    "batch-v2-vs-v1-swap.json": "v2v1s",
    "batch-v2-vs-v1.json": "v2v1",
    "batch-v3-vs-v0.json": "v3v0",
    "batch-v3-vs-v1-swap.json": "v3v1s",
    "batch-v3-vs-v1.json": "v3v1",
    "v1-black-losses.json": "v1bl",
}


def select_preventions(preventions: list[dict]) -> list[dict]:
    """Select 16 prevention positions by fixed quotas (no tuning on
    suite-run outcomes — selection uses only provenance + list size):

    * the iter-3 canonical case (v1-vs-v0-swap game 13) is always in;
    * quotas by loser engine and colour: v1 8 (4B/4W), v0 4 (2B/2W),
      v2 4 (2B/2W) — scarcer engines first, per-file cap 3;
    * usable only (a preventing move exists), plies <= 70;
    * within a quota, prefer sharp lists (2-4 moves), short games,
      and positions with >= 3 opponent creators.
    """
    losers = loser_engines()
    usable = []
    for c in preventions:
        if c["plies"] > 70:
            continue
        safe = preventing_moves(c["board"], c["to_move"])
        if not safe:
            continue  # no preventing move existed: already lost
        c["safe_moves"] = safe
        c["loser_engine"] = losers.get((c["source"], c["game_no"]), "?")
        usable.append(c)
    by_key = {(c["source"], c["game_no"]): c for c in usable}
    selected = [by_key[("batch-v1-vs-v0-swap.json", 13)]]
    selected_keys = {("batch-v1-vs-v0-swap.json", 13)}
    per_file: dict[str, int] = {"batch-v1-vs-v0-swap.json": 1}
    quotas = [
        ("scoring-v2", 1, 2), ("scoring-v2", -1, 2),
        ("v0", 1, 2), ("v0", -1, 2),
        ("scoring-v1", 1, 3), ("scoring-v1", -1, 4),
    ]

    def score(c: dict) -> int:
        s = 0
        if 2 <= len(c["safe_moves"]) <= 4:
            s += 2
        elif len(c["safe_moves"]) <= 6:
            s += 1
        if c["plies"] <= 40:
            s += 1
        if c["creators"] >= 3:
            s += 1
        return s

    for engine, tm, n in quotas:
        pool = [c for c in usable if c["loser_engine"] == engine
                and c["to_move"] == tm
                and (c["source"], c["game_no"]) not in selected_keys]
        pool.sort(key=lambda c: (-score(c), c["source"], c["game_no"]))
        took = 0
        for c in pool:
            if took >= n:
                break
            if per_file.get(c["source"], 0) >= 3:
                continue
            selected.append(c)
            selected_keys.add((c["source"], c["game_no"]))
            per_file[c["source"]] = per_file.get(c["source"], 0) + 1
            took += 1
        assert took == n, f"quota {engine} to_move={tm}: took {took}/{n}"
    assert len(selected) == 16
    return selected


# Game-setup picks: clean positions (no VCF first move outside the
# setup list), spread across 6 batch files, 3 Black / 3 White to move.
# Selected from the 3-per-file shortlist (first/middle/last by game
# number, plies <= 70) by those rules; the assembly re-verifies each
# pick (setup list recomputed, VCF extras asserted empty).
GAME_SETUP_PICKS = [
    ("batch-v1-vs-v0-swap.json", 92),
    ("batch-v1-vs-v1-swap.json", 33),
    ("batch-v2-vs-v1-swap.json", 94),
    ("batch-v2-vs-v1.json", 2),
    ("batch-v3-vs-v1-swap.json", 35),
    ("batch-v3-vs-v1.json", 31),
]


def select_game_setups(setups: list[dict]) -> list[dict]:
    """Pick the 6 game-setup positions (see GAME_SETUP_PICKS) and
    compute their exhaustive setup lists."""
    by_key = {(c["source"], c["game_no"]): c for c in setups}
    picked = []
    for key in GAME_SETUP_PICKS:
        c = by_key[key]
        c["setup_moves"] = setup_moves(c["board"], c["to_move"])
        picked.append(c)
    return picked


def vcf_first_moves(board: np.ndarray, player: int) -> list[int]:
    """Four-creating moves that start a forced win (VCF search)."""
    return [m for m in evaluate.four_creating_moves(board, player)
            if evaluate.starts_forced_win(board, player, m)]


# --- hand-constructed positions -------------------------------------------
# Each entry: stones, side to move, category, id, name, description.
# Expected moves are COMPUTED at assembly (never hardcoded):
#  * "setup"   -> exactly setup_moves(); must be clean (no VCF move
#                 outside the list, no immediate win on the board);
#  * "winning" -> exactly setup_moves() ∪ vcf_first_moves() (a move
#                 that wins by a 2-ply setup OR by consecutive fours
#                 passes — listing only one kind would mark a genuine
#                 winning move wrong);
#  * "vcf"     -> expect type "vcf"; must have 1-10 forcing first
#                 moves and no setup move (a VCT win outside the VCF
#                 check would make the position unfair).
HAND_POSITIONS: list[dict] = [
    {
        "id": "setup-double-open-three",
        "name": "Create a double open three",
        "category": "setup",
        "to_move": 1,
        "black": [[5, 7], [6, 7], [7, 5], [7, 6], [13, 1], [1, 13]],
        "white": [[13, 13], [1, 1], [11, 2]],
        "description": (
            "Hand-constructed control: Black has a horizontal two "
            "(5,7)-(6,7) and a vertical two (7,5)-(7,6). Only (7,7) "
            "completes both into open threes — White can block one, "
            "and Black extends the other into an open four. Every "
            "1-ply engine finds this one (double-open-three bonus); "
            "it guards against regressions on easy setups."
        ),
    },
    {
        "id": "winning-quiet-setup",
        "name": "Quiet setup — open three plus double-four threat",
        "category": "winning",
        "to_move": 1,
        "black": [[5, 7], [6, 7], [6, 8], [8, 6], [9, 6], [9, 7], [9, 8],
                  [13, 1], [1, 13]],
        "white": [[5, 9], [9, 9], [13, 13], [1, 1]],
        "description": (
            "Hand-constructed: (7,7) looks quiet (one open three plus "
            "a closed three) but wins in two — it completes the "
            "horizontal open three and arms the double-four square "
            "(9,5), so White cannot stop both. (6,6) also wins (it "
            "creates a second open three). The tempting four-creating "
            "moves (9,4) and (9,5) score far higher in 1-ply: (9,5) "
            "does force a win by consecutive fours (so it passes "
            "too), but (9,4) does not — White blocks and the attack "
            "dies. Expected moves are exactly the verified winning "
            "moves (2-ply setups plus VCF first moves)."
        ),
    },
    {
        "id": "winning-quiet-setup-2",
        "name": "Quiet setup — the 58k double three is a decoy",
        "category": "winning",
        "to_move": 1,
        "black": [[5, 7], [6, 7], [8, 8], [9, 9], [8, 10], [9, 10],
                  [11, 10], [13, 1], [1, 13]],
        "white": [[6, 6], [12, 10], [13, 13], [1, 1]],
        "description": (
            "Hand-constructed: the highest 1-ply moves here are "
            "double-open-three decoys — (9,7) scores 58,167 but its "
            "threats are gap-threes whose extensions give only "
            "single-ended fours, so White stops them. The winning "
            "moves are the quiet (7,7) (open three plus the armed "
            "double-four square (10,10)), (8,7) and (8,9) (2-ply "
            "setups), and (10,10) itself, which forces a win by "
            "consecutive fours. Expected moves are exactly the "
            "verified winning moves (2-ply setups plus VCF first "
            "moves)."
        ),
    },
    {
        "id": "vcf-race-block-or-win",
        "name": "Race — take your open four instead of blocking",
        "category": "vcf",
        "to_move": 1,
        "black": [[6, 6], [7, 6], [8, 6], [8, 4], [8, 5], [1, 13]],
        "white": [[5, 6], [8, 8], [5, 11], [6, 11], [7, 11], [13, 13],
                  [1, 1]],
        "description": (
            "Hand-constructed: Black holds an open-four creator "
            "(8,3) and a VCF sequence (9,6)/(10,6), but White holds "
            "an open three on row 11. The 1-ply engines block White's "
            "threat (8,11) — defence scores one point above the open "
            "four — and miss the forced win. Only a move starting a "
            "forced win by consecutive fours passes. The four-"
            "creating traps (8,7) and (8,2) do not force a win."
        ),
    },
    {
        "id": "vcf-race-block-or-win-white",
        "name": "Race (White to move) — take your open four",
        "category": "vcf",
        "to_move": -1,
        "black": [[5, 6], [8, 8], [5, 11], [6, 11], [7, 11], [13, 13],
                  [1, 1]],
        "white": [[6, 6], [7, 6], [8, 6], [8, 4], [8, 5], [1, 13]],
        "description": (
            "Hand-constructed: the colour-swap of "
            "vcf-race-block-or-win — White holds the open-four "
            "creator (8,3) and the VCF sequence, Black holds the open "
            "three on row 11. Same trap: blocking scores above "
            "winning in 1-ply. Only a move starting a forced win by "
            "consecutive fours passes."
        ),
    },
    {
        "id": "vcf-two-four-sequence",
        "name": "Forced win — two-four sequence, quiet second four",
        "category": "vcf",
        "to_move": 1,
        "black": [[6, 6], [7, 6], [8, 6], [7, 4], [8, 5], [13, 1],
                  [1, 13]],
        "white": [[5, 6], [5, 2], [13, 13], [1, 1]],
        "description": (
            "Hand-constructed: the forcing sequence needs two fours. "
            "(9,6) makes a four White must block at (10,6); only "
            "then does (10,7) make the open four on the diagonal "
            "(7,4)-(10,7) — (10,7) is not four-creating on move 1. "
            "The trap (10,6) also makes a four, but White's block "
            "at (9,6) occupies the square the sequence needs, so the "
            "attack dies. Only (9,6) starts a forced win."
        ),
    },
    {
        "id": "vcf-two-four-sequence-rot",
        "name": "Forced win — two-four sequence, rotated",
        "category": "vcf",
        "to_move": 1,
        "black": [[6, 8], [6, 7], [6, 6], [4, 7], [5, 6], [1, 1],
                  [13, 13]],
        "white": [[6, 9], [2, 9], [1, 13], [13, 1]],
        "description": (
            "Hand-constructed: vcf-two-four-sequence rotated 90 "
            "degrees. (6,5) makes a four White must block at (6,4); "
            "only then does (7,4) make the open four. The trap "
            "(6,4) also makes a four, but White's block at (6,5) "
            "occupies the square the sequence needs. Only (6,5) "
            "starts a forced win."
        ),
    },
]


def _fmt_xy(move: int) -> list[int]:
    x, y = game.move_to_xy(move)
    return [x, y]


def _position_dict(pid: str, name: str, category: str, description: str,
                   board: np.ndarray, to_move: int,
                   expect: dict) -> dict:
    black, white = _stones(board)
    return {
        "id": pid,
        "name": name,
        "category": category,
        "description": description,
        "black": black,
        "white": white,
        "to_move": to_move,
        "expect": expect,
    }


def assemble_suite(preventions: list[dict], game_setups: list[dict]
                   ) -> dict:
    """Build + verify the hard suite dict (same schema as
    tactical_positions.json). Raises AssertionError if any position
    fails verification — nothing unverified enters the suite."""
    positions: list[dict] = []

    for c in preventions:
        board = c["board"]
        to_move = c["to_move"]
        assert not game.game_over(board)[0]
        assert not evaluate.winning_moves(board, to_move)
        assert not evaluate.winning_moves(board, -to_move)
        safe = preventing_moves(board, to_move)
        assert safe, f"{c['source']} g{c['game_no']}: empty safe list"
        assert game.xy_to_move(*c["game_move"]) not in safe
        assert sorted(safe) == sorted(c["safe_moves"])
        tag = FILE_TAGS[c["source"]]
        colour = "Black" if to_move == 1 else "White"
        pid = f"prev-{tag}-g{c['game_no']}"
        positions.append(_position_dict(
            pid,
            f"Prevent the open four — {tag} game {c['game_no']}",
            "prevention",
            (f"From batch-1 {c['source']} game {c['game_no']} "
             f"(ply {c['creator_ply']}, {c['plies']} plies; "
             f"{c['loser_engine']} to move as {colour}, lost). The "
             f"opponent holds {c['creators']} open-four creator(s) "
             f"and no immediate win. In the game {colour} played "
             f"{tuple(c['game_move'])} and lost to the open four "
             f"next move. The listed moves are exactly the moves "
             f"that prevent it: an immediate win, a forcing four, "
             f"or a move leaving the opponent with zero live "
             f"creators (verified exhaustively)."),
            board, to_move,
            {"type": "moves", "moves": [_fmt_xy(m) for m in safe]},
        ))

    for c in game_setups:
        board = c["board"]
        to_move = c["to_move"]
        assert not game.game_over(board)[0]
        assert not evaluate.winning_moves(board, to_move)
        assert not evaluate.winning_moves(board, -to_move)
        assert not creators(board, to_move), "pre-existing creator"
        setups = setup_moves(board, to_move)
        assert setups, f"{c['source']} g{c['game_no']}: empty setup list"
        assert game.xy_to_move(*c["game_move"]) in setups
        vcf = vcf_first_moves(board, to_move)
        assert not vcf, (
            f"{c['source']} g{c['game_no']}: VCF moves {vcf} outside "
            "the setup list — position is not clean"
        )
        tag = FILE_TAGS[c["source"]]
        colour = "Black" if to_move == 1 else "White"
        pid = f"setup-{tag}-g{c['game_no']}"
        positions.append(_position_dict(
            pid,
            f"Set up the open four — {tag} game {c['game_no']}",
            "setup",
            (f"From batch-1 {c['source']} game {c['game_no']} "
             f"(ply {c['creator_ply']}, {c['plies']} plies; {colour} "
             f"to move, won). In the game {colour} played "
             f"{tuple(c['game_move'])} and created the decisive open "
             f"four two plies later. The listed moves are exactly "
             f"the 2-ply setups: after the move the opponent has no "
             f"immediate win, and for every reply an open-four "
             f"creator survives (verified exhaustively)."),
            board, to_move,
            {"type": "moves", "moves": [_fmt_xy(m) for m in setups]},
        ))

    for h in HAND_POSITIONS:
        board = evaluate.board_from_position(h)
        to_move = h["to_move"]
        assert not game.game_over(board)[0], h["id"]
        assert not evaluate.winning_moves(board, to_move), h["id"]
        assert not evaluate.winning_moves(board, -to_move), h["id"]
        if h["category"] == "setup":
            assert not creators(board, to_move), h["id"]
            setups = setup_moves(board, to_move)
            assert setups, h["id"]
            assert not vcf_first_moves(board, to_move), h["id"]
            expect = {"type": "moves",
                      "moves": [_fmt_xy(m) for m in setups]}
        elif h["category"] == "winning":
            assert not creators(board, to_move), h["id"]
            moves = sorted(set(setup_moves(board, to_move))
                           | set(vcf_first_moves(board, to_move)))
            assert moves, h["id"]
            expect = {"type": "moves",
                      "moves": [_fmt_xy(m) for m in moves]}
        elif h["category"] == "vcf":
            forcing = vcf_first_moves(board, to_move)
            assert 1 <= len(forcing) <= 10, (
                f"{h['id']}: {len(forcing)} forcing first moves"
            )
            assert not setup_moves(board, to_move), (
                f"{h['id']}: setup moves exist outside the VCF check"
            )
            expect = {"type": "vcf"}
        else:
            raise ValueError(f"unknown hand category {h['category']!r}")
        positions.append(_position_dict(
            h["id"], h["name"], h["category"], h["description"],
            board, to_move, expect,
        ))

    ids = [p["id"] for p in positions]
    assert len(ids) == len(set(ids)), "duplicate position ids"
    return {
        "_meta": {
            "version": 2,
            "description": (
                "Harder tactical suite (v2) for the auto-research "
                "program, batch 2. The v1 suite (20 positions) did "
                "not discriminate: scoring v1, v2 and v3 all scored "
                "20/20. These positions come from batch-1 lost games "
                "(prevention: the side to move missed the only "
                "prevention 1 ply before the decisive open four; "
                "setup: the winner's 2-ply setup 2 plies before it) "
                "plus hand-constructed VCF/VCT positions. Every "
                "expected-move list was verified exhaustively "
                "(ml/scripts/build_tactical_hard.py): prevention "
                "lists are exactly the moves that win, force a four, "
                "or leave the opponent with zero live open-four "
                "creators; setup lists are exactly the 2-ply setups; "
                "winning lists are exactly the 2-ply setups plus the "
                "VCF first moves; vcf positions pass iff the move "
                "starts a forced win by consecutive fours. v1's "
                "score on this suite is the clause-3 reference for "
                "batch-2 versions (see "
                "research/iterations/tactical-suite-v2.md). No "
                "batch-2 version was tuned on these positions."
            ),
        },
        "positions": positions,
    }


def main() -> int:
    import time
    if "--assemble" in sys.argv:
        t0 = time.time()
        preventions, setups, stats = extract_all()
        print(f"extraction took {time.time() - t0:.1f}s", file=sys.stderr)
        print(f"prevention candidates: {len(preventions)}, "
              f"setup candidates: {len(setups)}", file=sys.stderr)
        selected_prev = select_preventions(preventions)
        print(f"selected {len(selected_prev)} preventions",
              file=sys.stderr)
        selected_setups = select_game_setups(setups)
        print(f"selected {len(selected_setups)} game setups",
              file=sys.stderr)
        suite = assemble_suite(selected_prev, selected_setups)
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "gomoku_ml",
            "tactical_positions_hard.json",
        )
        out_path = os.path.normpath(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(suite, f, indent=2)
            f.write("\n")
        print(f"wrote {len(suite['positions'])} positions to {out_path}")
        print(f"total took {time.time() - t0:.1f}s", file=sys.stderr)
        return 0
    t0 = time.time()
    preventions, setups, stats = extract_all()
    print(f"extraction took {time.time() - t0:.1f}s", file=sys.stderr)
    print(f"decisive games scanned: {stats['games']}")
    print(f"games skipped (no creator): {stats['skipped']}")
    print(f"skip reasons: {stats['skip_reasons']}")
    print(f"prevention candidates: {len(preventions)}")
    print(f"setup candidates: {len(setups)}")
    print("\nprevention candidates (creators, source):")
    for c in preventions:
        print(f"  {c['source']} g{c['game_no']} ply{c['creator_ply']}: "
              f"creators={c['creators']} "
              f"to_move={'B' if c['to_move'] == 1 else 'W'} "
              f"game_move={c['game_move']}")
    print("\nsetup candidates (source):")
    for c in setups:
        print(f"  {c['source']} g{c['game_no']} ply{c['creator_ply']}: "
              f"to_move={'B' if c['to_move'] == 1 else 'W'} "
              f"game_move={c['game_move']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Improved heuristic Gomoku engine (v1) — 5-window pattern scoring.

Same interface as :mod:`gomoku_ml.baseline` — ``score_move(board, move,
player) -> int`` and ``choose_move(board, player, rng=None) -> int`` —
so it drops into any harness that takes a move chooser. It is a NEW
module: :mod:`gomoku_ml.baseline` is the frozen ladder anchor and is
not modified.

Differences from the baseline (each small, each tested in
``ml/tests/test_scoring.py``):

1. **5-window scan (the broken-four fix).** The baseline counts only
   contiguous runs through a candidate point, so a broken four
   (``XX_XX``) is invisible to it. This engine scores every 5-cell
   window through the point (mirroring the web engine in
   ``src/ai/patterns.ts``): a move completing ``XX_XX`` scores as a
   four, and a move creating a broken four is valued as the threat it
   is.
2. **Correct four valuation.** A 5-window holding four stones of one
   colour always has a winning square — the empty cell inside the
   window — so a four scores as a four (20k) even when both outside
   cells are blocked. Only a *contiguous* four whose far outside cell
   is empty is an open four (200k, two winning squares). (The TS
   engine scores a fully-closed four-window as 0; that undervalues a
   real threat, so this engine deliberately deviates there.)
3. **Max-per-direction, not sum-over-windows.** A single contiguous
   four sits in two overlapping 5-windows; summing all windows (as the
   TS engine does) double-counts one physical pattern and would let a
   lone four masquerade as a double threat. Each direction contributes
   only its strongest window, and the four directions are summed.
4. **Double-threat bonuses.** A move creating fours in two or more
   directions is a forced win next move (the opponent can block only
   one); a move creating open threes in two or more directions forces
   a win in two. Both get explicit bonuses so they beat single
   threats of the same class.
5. **Cheaper win/block scans.** An immediate winning move must be
   adjacent to a friendly stone, so the win and block scans only check
   radius-1 candidates of the relevant colour instead of all ~200
   legal moves.
6. **Candidate restriction for scoring.** Only empty points within
   Chebyshev distance 2 of a stone are scored (centre on an empty
   board; all legal moves as a fallback if the candidate set is
   empty). A move far from all stones is never the best move in
   Gomoku, and this is what makes the engine fast enough for
   large-scale self-play.
7. **No centre preference in the score.** The baseline adds a
   Manhattan-distance term; the window scores carry the signal, and
   the empty-board centre move is handled directly in
   ``choose_move``.

Weights mirror the web engine (orders of magnitude apart so pattern
classes dominate each other): five 10,000,000; open four 200,000;
four 20,000; open three 4,000; three 400; open two 80; two 10; one
1–2. Attack and defence are summed at full value (the baseline
discounts defence to 90%).

Version: v1. Later research iterations add versions to
:data:`ENGINE_VERSIONS` (each version is a parameter set) and use
:func:`get_player` / :class:`ScoringPlayer` with ``version=N``; the
module-level :func:`score_move` / :func:`choose_move` / :class:`ScoringPlayer`
defaults stay v1 forever so old results remain reproducible and old
versions stay importable for A/B comparisons. v2 tests a defence
discount (0.9); v3 adds a 2-ply threat check (prefer a move that
leaves the opponent with no live open-four creator).
"""

from __future__ import annotations

import numpy as np

from . import game

VERSION = 1
ENGINE_NAME = "scoring-v1"

# --- version registry -----------------------------------------------------
# Each version is a parameter set for the same engine code. v1 is the
# 5-window engine (full-value defence). Later versions change ONE
# parameter at a time so A/B comparisons stay interpretable.
ENGINE_VERSIONS: dict[int, dict] = {
    1: {"name": "scoring-v1", "defence_weight": 1.0, "threat_check": False},
    # v2 (iteration 2): attack bias — defence discounted to 0.9, the
    # value the frozen baseline uses. Tests the iteration-1
    # hypothesis (see research/iterations/iter-1.md and iter-2.md):
    # v1's full-value defence makes it a strong defender but a
    # passive attacker — it loses the race to the first open four.
    # Every decisive game in the iteration-1 batches was decided by
    # an open four, and v1's losses came fast (mean 27.5 plies) while
    # its wins ground out slowly (mean 69.5 plies).
    # REJECTED (iter-2.md): lost head-to-head to v1 (0.4050) and
    # lost ladder Elo (70.4 -> -52.5).
    2: {"name": "scoring-v2", "defence_weight": 0.9, "threat_check": False},
    # v3 (iteration 3): v1 + a 2-ply threat check. v1 never misses a
    # block and never ignores a critical threat, but it MISSES
    # PREVENTIONS: in 35 swap losses, on 13 turns (in 9 games) its
    # non-forcing move left an opponent open-four creator live when
    # a preventing alternative existed — 8.2% of the 159 threatened
    # turns where the opponent had no immediate winning move (the
    # baseline misses at the same rate, 8.6%, so the gap is real but
    # small and v3's expected gain is modest). The 1-ply scorer
    # cannot see prevention: a forcing four (20k) or a line-block
    # that kills a creator scores far less than a double open three
    # (58k), so the attacking move wins and the creator stays live.
    # v3 prefers a move that leaves the opponent with zero live
    # open-four creators (or a forcing move) whenever the 1-ply best
    # does not.
    3: {"name": "scoring-v3", "defence_weight": 1.0, "threat_check": True},
}

# --- weights --------------------------------------------------------------
# Orders of magnitude apart so pattern classes dominate each other.
WIN_SCORE = 10_000_000
OPEN_FOUR = 200_000
FOUR = 20_000
OPEN_THREE = 4_000
THREE = 400
OPEN_TWO = 80
TWO = 10
ONE_OPEN = 2
ONE = 1

# A move creating fours in >= 2 directions wins next move outright;
# a move creating open threes in >= 2 directions forces a win in two.
DOUBLE_FOUR_BONUS = 200_000
DOUBLE_OPEN_THREE_BONUS = 50_000

# Pattern kinds (per direction, from the strongest window in it).
_KIND_NONE = 0
_KIND_FOUR = 1
_KIND_OPEN_THREE = 2
_KIND_FIVE = 3

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


# --- precomputed windows --------------------------------------------------
# All 572 five-cell windows, each as (cells, before, after, dir_index):
# ``cells`` is the 5 flat indices of the window, ``before``/``after``
# the flat index of the cell just outside each end (-1 if off-board),
# ``dir_index`` the index into _DIRECTIONS.

def _build_windows() -> list[tuple[tuple[int, ...], int, int, int]]:
    wins: list[tuple[tuple[int, ...], int, int, int]] = []
    for di, (dy, dx) in enumerate(_DIRECTIONS):
        for y in range(game.SIZE):
            for x in range(game.SIZE):
                ey, ex = y + 4 * dy, x + 4 * dx
                if not (0 <= ey < game.SIZE and 0 <= ex < game.SIZE):
                    continue
                cells = tuple(
                    (y + i * dy) * game.SIZE + (x + i * dx) for i in range(5)
                )
                by, bx = y - dy, x - dx
                ay, ax = ey + dy, ex + dx
                before = (
                    by * game.SIZE + bx
                    if 0 <= by < game.SIZE and 0 <= bx < game.SIZE
                    else -1
                )
                after = (
                    ay * game.SIZE + ax
                    if 0 <= ay < game.SIZE and 0 <= ax < game.SIZE
                    else -1
                )
                wins.append((cells, before, after, di))
    return wins


_WINDOWS = _build_windows()

# For each cell, the windows containing it (at most 20), as
# (window index, position of the cell within the window). The position
# lets the scorer count all five cells without a per-cell
# "is this the move?" check: the move cell is empty on the board, so
# it contributes to neither side's count, and for a four the other
# empty cell's position is (sum of empty positions) - (move position).
_WINDOWS_BY_CELL: list[list[tuple[int, int]]] = [
    [] for _ in range(game.SIZE * game.SIZE)
]
for _wi, (_cells, _b, _a, _d) in enumerate(_WINDOWS):
    for _i, _c in enumerate(_cells):
        _WINDOWS_BY_CELL[_c].append((_wi, _i))


# --- window evaluation ----------------------------------------------------

def _score_from_counts(
    own: int, opp: int, empty_idx: int, open_ends: int,
    flat: list[int], before: int, after: int,
) -> tuple[int, int]:
    """``(score, kind)`` for a window holding ``own`` stones of the
    side that played the candidate move and ``opp`` opponent stones.

    ``empty_idx`` is the position (0-4) of the window's empty cell
    (only meaningful when ``own == 4``, when there is exactly one);
    ``open_ends`` counts the empty cells just outside the window.
    A window containing an opponent stone scores 0 (blocked).
    """
    if opp > 0:
        return 0, _KIND_NONE
    if own == 5:
        return WIN_SCORE, _KIND_FIVE
    if own == 4:
        # The empty cell inside the window is always a winning square,
        # so this is a four whatever the outside looks like. It is an
        # OPEN four (two winning squares) only when the four is
        # contiguous and the outside cell beyond its far end is empty:
        # the gap of a broken four is its only winning square.
        if empty_idx == 0:
            # Four occupies cells 1-4; the far outside cell is `after`.
            if after != -1 and flat[after] == game.EMPTY:
                return OPEN_FOUR, _KIND_FOUR
        elif empty_idx == 4:
            # Four occupies cells 0-3; the far outside cell is `before`.
            if before != -1 and flat[before] == game.EMPTY:
                return OPEN_FOUR, _KIND_FOUR
        return FOUR, _KIND_FOUR
    if own == 3:
        if open_ends == 2:
            return OPEN_THREE, _KIND_OPEN_THREE
        if open_ends == 1:
            return THREE, _KIND_NONE
        return 0, _KIND_NONE
    if own == 2:
        if open_ends == 2:
            return OPEN_TWO, _KIND_NONE
        if open_ends == 1:
            return TWO, _KIND_NONE
        return 0, _KIND_NONE
    if own == 1:
        return (ONE_OPEN if open_ends == 2 else ONE), _KIND_NONE
    return 0, _KIND_NONE


def _score_parts_flat(flat: list[int], move: int, player: int,
                      defence_weight: float = 1.0) -> tuple[int, int]:
    """Core scorer on a flat board (list of 225 ints).

    Returns ``(attack, defence)``: the attack value (what the move
    builds for ``player``, including the double-threat bonuses) and
    the defence value (what it denies the opponent by occupying the
    point), with each direction's defence discounted by
    ``defence_weight`` (per-direction truncation, mirroring the
    baseline's ``int(0.9 * w)``). ``defence_weight=1.0`` is v1
    (full-value defence); later versions test other weights.

    One pass over the ~20 windows through ``move`` counts the
    mover's and the opponent's stones in each window; attack and
    defence follow from the same counts (attack: the move is the
    mover's stone, so own = p+1, opp = o; defence: the move is the
    opponent's stone, so own = o+1, opp = p). Each direction
    contributes only its strongest window (see the module docstring,
    point 3).

    Note the symmetry: at ``defence_weight=1.0``,
    attack(move, player) + defence(move, player) equals
    attack(move, -player) + defence(move, -player) — the total score
    of a square is the same for both colours (attack and defence
    swap roles). With a defence discount the totals differ between
    colours. Either way, anything that must distinguish the colours
    (e.g. the swap-rule colour choice) compares the ATTACK parts,
    not the totals.
    """
    opp = -player
    best_a = [0, 0, 0, 0]
    kinds_a = [_KIND_NONE, _KIND_NONE, _KIND_NONE, _KIND_NONE]
    best_d = [0, 0, 0, 0]
    for wi, move_idx in _WINDOWS_BY_CELL[move]:
        cells, before, after, di = _WINDOWS[wi]
        p = 0
        o = 0
        empty_sum = 0
        # The move cell is empty on the board, so it counts as empty
        # here and contributes to neither side — no per-cell check
        # needed. For a four, the other empty cell's position is
        # empty_sum - move_idx (the empties are the move cell and one
        # other cell).
        for i, c in enumerate(cells):
            v = flat[c]
            if v == player:
                p += 1
            elif v == opp:
                o += 1
            else:
                empty_sum += i
        if p > 0 and o > 0:
            continue  # blocked for both sides
        # Open ends: the cells just outside the window (edge = closed).
        open_ends = 0
        if before != -1 and flat[before] == game.EMPTY:
            open_ends += 1
        if after != -1 and flat[after] == game.EMPTY:
            open_ends += 1
        if o == 0:
            s, kind = _score_from_counts(
                p + 1, 0, empty_sum - move_idx, open_ends,
                flat, before, after,
            )
            if s > best_a[di]:
                best_a[di] = s
                kinds_a[di] = kind
        if p == 0:
            s, _kind = _score_from_counts(
                o + 1, 0, empty_sum - move_idx, open_ends,
                flat, before, after,
            )
            if s > best_d[di]:
                best_d[di] = s
    attack = sum(best_a)
    fours = sum(1 for k in kinds_a if k in (_KIND_FOUR, _KIND_FIVE))
    open_threes = sum(1 for k in kinds_a if k == _KIND_OPEN_THREE)
    if fours >= 2:
        attack += DOUBLE_FOUR_BONUS
    if open_threes >= 2:
        attack += DOUBLE_OPEN_THREE_BONUS
    defence = sum(int(defence_weight * d) for d in best_d)
    return attack, defence


def _score_move_flat(flat: list[int], move: int, player: int,
                     defence_weight: float = 1.0) -> int:
    """Total score (attack + discounted defence) on a flat board."""
    attack, defence = _score_parts_flat(flat, move, player, defence_weight)
    return attack + defence


# --- public interface (mirrors baseline) ----------------------------------

def score_move(board: np.ndarray, move: int, player: int) -> int:
    """Heuristic score for ``player`` playing ``move`` on ``board``
    (v1: attack plus full-value defence, plus double-threat bonuses —
    see the module docstring). Versioned scoring lives on
    :class:`ScoringPlayer` (``.score_move``); this module function
    stays v1 so the interface matches the frozen baseline exactly.
    """
    flat = board.reshape(-1).tolist()
    return _score_move_flat(flat, move, player)


def score_attack(board: np.ndarray, move: int, player: int) -> int:
    """Attack-only score for ``player`` playing ``move`` on ``board``:
    what the move builds for ``player``, including the double-threat
    bonuses, WITHOUT the defence component.

    This is the colour-asymmetric half of :func:`score_move` — the
    total score is identical for both colours at any square (attack
    and defence swap roles), so any comparison between colours (the
    swap-rule colour choice in :func:`gomoku_ml.game.choose_swap_color`)
    must use this, not the total.
    """
    flat = board.reshape(-1).tolist()
    attack, _defence = _score_parts_flat(flat, move, player)
    return attack


def _near_empty(board: np.ndarray, color: int | None, radius: int) -> list[int]:
    """Empty points within Chebyshev distance ``radius`` of any stone
    of ``color`` (``None`` = any stone). Sorted ascending."""
    if color is None:
        stones = np.argwhere(board != game.EMPTY)
    else:
        stones = np.argwhere(board == color)
    if len(stones) == 0:
        return []
    near = np.zeros((game.SIZE, game.SIZE), dtype=bool)
    for y, x in stones:
        y0, y1 = max(0, y - radius), min(game.SIZE, y + radius + 1)
        x0, x1 = max(0, x - radius), min(game.SIZE, x + radius + 1)
        near[y0:y1, x0:x1] = True
    return [int(m) for m in np.flatnonzero((board == game.EMPTY) & near)]


def _candidate_moves(board: np.ndarray) -> list[int]:
    """Moves worth scoring: empty points within Chebyshev distance 2
    of any stone (the centre on an empty board). Falls back to all
    legal moves if the candidate set is empty (near-full board)."""
    stones = int(np.count_nonzero(board))
    if stones == 0:
        return [game.xy_to_move(7, 7)]
    cands = _near_empty(board, None, 2)
    if cands:
        return cands
    return [int(m) for m in game.legal_moves(board)]


# --- 2-ply threat check (v3) ------------------------------------------------
# An "open-four creator" for a player is a move that leaves them with
# >= 2 winning moves (an open four). If the opponent has a creator,
# they make the open four next move and win in two — unless our move
# wins, makes a four (forcing a block), or leaves them with no live
# creator. A 1-ply scorer cannot see this: a line-block that kills a
# creator scores only the opponent's attack at that square, so a
# higher-scoring attacking move beats it and the creator stays live.

def _winning_moves(board: np.ndarray, player: int) -> list[int]:
    """All immediate winning moves for ``player`` (same semantics as
    :func:`gomoku_ml.evaluate.winning_moves`, implemented locally to
    keep this module free of the evaluate dependency)."""
    return [
        m for m in _near_empty(board, player, 1)
        if game.is_winning_move(board, m, player)
    ]


def _opponent_creators(board: np.ndarray, player: int) -> list[int]:
    """Moves by which ``player``'s opponent would create an open four
    (leave themselves >= 2 winning moves). A creator must be adjacent
    to an opponent stone, so it is always in ``_candidate_moves``."""
    opp = -player
    return [
        m for m in _candidate_moves(board)
        if len(_winning_moves(game.apply_move(board, m, opp), opp)) >= 2
    ]


def _is_forcing(board: np.ndarray, move: int, player: int) -> bool:
    """Does ``player``'s ``move`` win immediately or leave them with a
    four (>= 1 winning move)? Either forces the opponent to answer
    instead of playing a creator."""
    b2 = game.apply_move(board, move, player)
    return (
        game.is_winning_stone(b2, move, player)
        or len(_winning_moves(b2, player)) >= 1
    )


def _move_is_safe(board: np.ndarray, move: int, player: int,
                  creators: list[int]) -> bool:
    """Is ``move`` safe against the opponent's ``creators`` (computed
    on the current board)? Safe = forcing (win or four), or no creator
    remains live after the move. A move can only REMOVE creators (it
    adds one of our stones), so the creators after the move are a
    subset of ``creators`` — only those need re-checking."""
    if _is_forcing(board, move, player):
        return True
    b2 = game.apply_move(board, move, player)
    for c in creators:
        if c == move:
            continue  # occupied by our move
        b3 = game.apply_move(b2, c, -player)
        if len(_winning_moves(b3, -player)) >= 2:
            return False
    return True


def _choose_move(board: np.ndarray, player: int,
                 rng: np.random.Generator | None,
                 defence_weight: float,
                 threat_check: bool = False) -> int:
    """Versioned move choice (see :func:`choose_move`).

    With ``threat_check`` (v3): after the win/block scans, if the
    opponent has an open-four creator, prefer the best-scoring move
    that leaves them with no live creator (or a forcing move) over
    the 1-ply best move when the 1-ply best is unsafe. If no safe
    move exists, fall back to the 1-ply best. With no opponent
    creator, v3 chooses exactly like v1.
    """
    legal = game.legal_moves(board)
    if len(legal) == 0:
        raise ValueError("no legal moves")
    # 1. Take the win.
    for move in _near_empty(board, player, 1):
        if game.is_winning_move(board, move, player):
            return move
    # 2. Block the opponent's win.
    for move in _near_empty(board, -player, 1):
        if game.is_winning_move(board, move, -player):
            return move
    # 3. Best heuristic score among candidates.
    flat = board.reshape(-1).tolist()
    scored: list[tuple[int, int]] = [
        (_score_move_flat(flat, move, player, defence_weight), move)
        for move in _candidate_moves(board)
    ]
    best_score = max(s for s, _m in scored)
    best_moves = [m for s, m in scored if s == best_score]

    def _pick(moves: list[int]) -> int:
        if rng is not None:
            return int(rng.choice(moves))
        return moves[0]

    if not threat_check:
        return _pick(best_moves)
    # 3b. 2-ply threat check: if the opponent can create an open four
    # next move, the 1-ply best is only good if it stops them.
    creators = _opponent_creators(board, player)
    if not creators:
        return _pick(best_moves)
    safe_best = [m for m in best_moves
                 if _move_is_safe(board, m, player, creators)]
    if safe_best:
        return _pick(safe_best)
    safe_scored = [(s, m) for s, m in scored
                   if _move_is_safe(board, m, player, creators)]
    if safe_scored:
        best_safe = max(s for s, _m in safe_scored)
        return _pick([m for s, m in safe_scored if s == best_safe])
    return _pick(best_moves)


def choose_move(board: np.ndarray, player: int, rng: np.random.Generator | None = None) -> int:
    """Pick a move for ``player`` on ``board`` (v1).

    1. Immediate win (scanned only among radius-1 candidates of the
       mover's stones — a winning move must be adjacent to a friendly
       stone). 2. Block the opponent's immediate win (same, around the
       opponent's stones). 3. Best 5-window heuristic score among the
       candidate moves (random tie-break with the given rng, or the
       first best move if no rng is given). Versioned move choice
    lives on :class:`ScoringPlayer`; this module function stays v1
    so the interface matches the frozen baseline exactly.
    """
    return _choose_move(board, player, rng, 1.0)


class ScoringPlayer:
    """Stateful wrapper with a seeded RNG for reproducible eval games
    (mirrors :class:`gomoku_ml.baseline.HeuristicPlayer`).

    ``version`` selects the parameter set from
    :data:`ENGINE_VERSIONS` (default 1 = v1 behaviour).
    """

    def __init__(self, seed: int = 0, version: int = 1) -> None:
        if version not in ENGINE_VERSIONS:
            raise ValueError(
                f"unknown scoring engine version: {version} "
                f"(known: {sorted(ENGINE_VERSIONS)})"
            )
        self.version = version
        self.name = ENGINE_VERSIONS[version]["name"]
        self.defence_weight = ENGINE_VERSIONS[version]["defence_weight"]
        self.threat_check = ENGINE_VERSIONS[version].get("threat_check", False)
        self.rng = np.random.default_rng(seed)

    def score_move(self, board: np.ndarray, move: int, player: int) -> int:
        """This version's heuristic score for ``player`` playing
        ``move`` on ``board``."""
        flat = board.reshape(-1).tolist()
        return _score_move_flat(flat, move, player, self.defence_weight)

    def choose_move(self, board: np.ndarray, player: int) -> int:
        return _choose_move(
            board, player, self.rng, self.defence_weight, self.threat_check
        )


def get_player(version: int, seed: int = 0) -> ScoringPlayer:
    """Build a :class:`ScoringPlayer` for engine ``version``."""
    return ScoringPlayer(seed=seed, version=version)

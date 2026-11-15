"""PUCT MCTS with the v1 heuristic as prior and leaf value (v6, batch 2 H2;
v7, batch 3 H1, adds an attack gradient to the leaf; v16, batch 6 H2,
adds a White-only 2-ply rollout leaf — see below).

Standalone module: it does NOT import :mod:`gomoku_ml.mcts` (the
neural AlphaZero MCTS, owned by the B2 batched-leaf workstream).
It reuses only the heuristic scorer (:mod:`gomoku_ml.scoring`),
the board rules (:mod:`gomoku_ml.game`), and — for the v7 leaf —
the portable kernel's attack-only scores
(:mod:`gomoku_ml.kernel_numpy`).

Design (see ``research/iterations/iter-6.md`` for v6 and
``research/iterations/iter-7.md`` for v7):

* Priors come from v1's 1-ply scores (5-window, broken-four-aware):
  softmax over ``log1p(score) / 2`` over ALL candidate moves — no
  top-K cutoff (iter-5 §2.2: the saving move in v5-vs-v0 game 1
  ranked 11th-12th by 1-ply score, outside v4/v5's top-8).
* Leaf values are terminal-aware and cheap: the player to move
  with any winning move is +1 (wins next move); the opponent
  with >= 2 winning moves is -1 (cannot block all); the
  opponent with exactly 1 is -0.25 (must block — the search
  resolves which block). v6 values quiet positions at 0; v7
  (``leaf_mode="gradient"``) values them by the BREADTH of each
  side's attack — the count of high-attack moves from the
  kernel's attack-only scores, squashed with tanh — so the
  search has a gradient toward building threats in quiet
  positions (iter-6 §2: v6 defends perfectly and loses the
  open-four race because its leaf gives no attack signal). The
  gradient is capped at +/-0.2 (fast-loop iteration 1): a static
  heuristic must never outweigh an exact terminal signal, and the
  uncapped gradient masked the -0.25 forcing penalties (see
  ``_GRADIENT_CAP`` and iter-7.md §2.1).
* Win/block scans run before the search (as in every version).
* Move choice: the most-visited root child, ties by the seeded
  RNG. Sims: 100, c_puct: 1.5 — the SAME settings as the
  ladder's :class:`gomoku_ml.baseline.HeuristicMCTSPlayer`, so
  any edge over it comes purely from the better priors (v1 vs v0
  scoring) and the better leaf value (terminal-aware vs crude
  local attack).
* v16 (``leaf_mode="rollout2_v1"``, batch 6 H2): when the engine
  plays White, quiet leaves are valued by a 2-PLY ROLLOUT ending in
  the symmetric v1-kernel gradient — the leaf horizon extends past
  the forcing exchange, where White's counterplay value lives; see
  ``_rollout_leaf_value`` and ``research/iterations/iter-16.md``.
  When the engine plays Black, v16 IS v7 exactly, so the Black
  games are a built-in control.

Backup accounting follows the AlphaZero convention: every node
on the simulated path — including the newly expanded leaf — has
its visit count incremented and the (sign-adjusted) value added
to its value sum. Note this differs from
:class:`gomoku_ml.baseline.HeuristicMCTSPlayer`, which does not
update the leaf itself; with that accounting the first selection
from a newly expanded node has ``sqrt(parent_visits) == 0`` and
ignores the priors entirely.
"""

from __future__ import annotations

import math

import numpy as np

from . import game
from . import kernel_numpy
from . import kernel_v1
from . import scoring

# Temperature for the prior softmax over log1p(v1 scores). A
# winning move (10M) maps to log1p ~ 16.1, an open four (200k) to
# ~12.2, an open three (4k) to ~8.3, a quiet move (~10-100) to
# ~2.4-4.6; dividing by 2 gives a ~7x prior ratio between
# successive pattern classes.
_PRIOR_TEMPERATURE = 2.0

# Leaf value when the opponent has exactly one winning move and we
# have none: we must block, but the game is not lost — the search
# resolves which block. A mild penalty, not a terminal value.
_MUST_BLOCK_VALUE = -0.25

# --- v7 attack-gradient leaf (batch 3, H1) --------------------------------
# A side's attack BREADTH is the count of its high-attack moves,
# from the kernel's attack-only scores (v0 contiguous-run scoring —
# see the caveat in iter-7.md §1: the kernel ports v0, NOT v1, so it
# is blind to broken fours). Thresholds in v0 weight units: a move
# scoring >= 1000 creates a four or an open three ((4,1)/(3,2) in at
# least one direction — no combination of sub-1000 per-direction
# values reaches 1000); a move scoring in [100, 1000) creates a
# half-open three or an open two. The developing tier (weight 0.25)
# keeps the gradient informative in early quiet positions where
# neither side has a strong move yet. The tanh scale (4) is fixed in
# the pre-registration from the measured breadth-difference
# distribution (std 2.26 on batch-2 game positions).
#
# The gradient CAP (0.2) is fast-loop iteration 1's patch (see
# iter-7.md §2.1): the uncapped gradient (range +/-1) overwhelmed
# the exact terminal values — in all 5 fast-loop losses the
# gradient preferred the losing move, and its dense positive values
# on the blocking move's lines masked the sparse -0.25 penalties
# from the opponent's forcing sequences, so the search's visit
# counts followed the gradient while its own q-values preferred the
# saving move. A static heuristic must never outweigh an exact
# terminal signal: the cap (0.2) is strictly below the smallest
# terminal value (the must-block 0.25), so forcing moves always beat
# quiet gradient differences in leaf value, and the gradient acts as
# a tie-breaker between quiet lines — which is all a static
# heuristic can honestly claim to be. Capping fixed 4/5 critical
# positions (the 5th, a forcing move starved by a low prior, is a
# different weakness class — see the regression file).
_GRADIENT_STRONG = 1000
_GRADIENT_DEVELOPING = 100
_GRADIENT_DEVELOPING_WEIGHT = 0.25
_GRADIENT_SCALE = 4.0
_GRADIENT_CAP = 0.2


class _Node:
    """One MCTS node. ``children`` is None until expansion."""

    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "_Node"] | None = None

    @property
    def q(self) -> float:
        """Mean backed-up value from this node's player perspective."""
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def _priors(board: np.ndarray, player: int) -> dict[int, float]:
    """Softmax over log1p(v1 scores) / temperature, over all
    candidate moves. Returns a move -> prior dict summing to 1
    (empty when there are no candidates)."""
    flat = board.reshape(-1).tolist()
    cands = scoring._candidate_moves(board)
    if not cands:
        return {}
    logits = {
        m: math.log1p(max(0.0, scoring._score_move_flat(flat, m, player, 1.0)))
        / _PRIOR_TEMPERATURE
        for m in cands
    }
    top = max(logits.values())
    weights = {m: math.exp(v - top) for m, v in logits.items()}
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


def _terminal_leaf_value(board: np.ndarray, player: int) -> float:
    """Terminal-aware leaf value in [-1, 1] for ``player`` to move.

    +1: ``player`` has a winning move (wins next move — the player
    to move plays first, so this dominates any opponent threat).
    -1: the opponent has >= 2 winning moves (only one can be
    blocked). -0.25: the opponent has exactly 1 (must block).
    0: quiet — no immediate winning move for either side.
    """
    if scoring._winning_moves(board, player):
        return 1.0
    opp_wins = scoring._winning_moves(board, -player)
    if len(opp_wins) >= 2:
        return -1.0
    if len(opp_wins) == 1:
        return _MUST_BLOCK_VALUE
    return 0.0


def _breadth(attack: np.ndarray, empty_mask: np.ndarray) -> float:
    """Attack breadth of one side: the count of its strong moves
    (attack >= 1000: four/open-three makers) plus 0.25 per
    developing move (attack in [100, 1000): half-open-three /
    open-two makers), over empty cells only. A count, not a max —
    iter-5 §2.2 showed the best attack misleads where the breadth
    of threats decides (six single-four creators beat one
    open-four threat)."""
    s = attack[empty_mask]
    strong = int(np.count_nonzero(s >= _GRADIENT_STRONG))
    developing = int(
        np.count_nonzero((s >= _GRADIENT_DEVELOPING) & (s < _GRADIENT_STRONG))
    )
    return strong + _GRADIENT_DEVELOPING_WEIGHT * developing


def _attack_gradient(board: np.ndarray, player: int) -> float:
    """Breadth-gradient leaf value in (-0.2, 0.2) for ``player`` to
    move in a quiet position: ``0.2 * tanh((breadth(ours) -
    breadth(theirs)) / 4)`` from the kernel's attack-only scores for
    both colours (one ``attack_both`` call). Antisymmetric under
    colour swap. The 0.2 cap (fast-loop iteration 1) keeps the
    static gradient strictly below the smallest exact terminal
    value (the must-block 0.25) — see ``_GRADIENT_CAP``."""
    black, white = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours, theirs = black, white
    else:
        ours, theirs = white, black
    diff = _breadth(ours, mask) - _breadth(theirs, mask)
    return _GRADIENT_CAP * math.tanh(diff / _GRADIENT_SCALE)


def _gradient_leaf_value(board: np.ndarray, player: int) -> float:
    """v7 leaf value: v6's terminal-aware values for terminal
    positions, the attack gradient for quiet ones. The terminal
    cases are identical to :func:`_terminal_leaf_value` BY
    CONSTRUCTION (it returns 0.0 exactly for quiet positions), so
    the ONLY difference between v6 and v7 is the quiet-leaf value
    — a clean A/B of the gradient."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    return _attack_gradient(board, player)


_LEAF_FUNCTIONS = {
    "terminal": _terminal_leaf_value,
    "gradient": _gradient_leaf_value,
}

# --- v1-kernel breadth (ported from v14, batch 5 H3) -----------------------
# Same gradient FORM as v7 (breadth count, tanh squash, +/-0.2 cap),
# but the breadth counts come from the v1 kernel's attack scores
# (5-window, broken-four-aware) instead of the v0 kernel's
# (contiguous-run, blind to broken fours). Thresholds are the v1
# analogues of v7's, derived from the v1 pattern classes (see
# iter-14.md §1): strong = a move creating a four or an open three
# (v1 attack >= 4000 — no combination of sub-4000 per-direction
# values reaches 4000); developing = a move creating a half-open
# three (400) or an open two (80) (80 <= attack < 4000 — the
# strongest sub-open-two combination is 4 x two = 40).
_GRADIENT_V1_STRONG = 4000
_GRADIENT_V1_DEVELOPING = 80


def _breadth_v1(attack: np.ndarray, empty_mask: np.ndarray) -> float:
    """Attack breadth of one side from v1 attack scores: the count
    of its strong moves (v1 attack >= 4000: four/open-three makers)
    plus 0.25 per developing move (v1 attack in [80, 4000):
    half-open-three / open-two makers), over empty cells only.
    Same form as :func:`_breadth`, with the v1 pattern-class
    thresholds (see ``_GRADIENT_V1_STRONG``). Ported from v14
    (batch 5, H3); the kernel itself (:mod:`gomoku_ml.kernel_v1`)
    is v14's exact port (0 mismatches vs frozen + live references)."""
    s = attack[empty_mask]
    strong = int(np.count_nonzero(s >= _GRADIENT_V1_STRONG))
    developing = int(
        np.count_nonzero((s >= _GRADIENT_V1_DEVELOPING) & (s < _GRADIENT_V1_STRONG))
    )
    return strong + _GRADIENT_DEVELOPING_WEIGHT * developing


# --- v16 2-ply rollout leaf (batch 6, H2) ---------------------------------
# The diagnosed White failure (iter-13 §2.2): in 4 of 5 critical
# positions the search's correctly-signed q prefers the LOSING move —
# a top-v1-score block or four-maker — over the oracle-verified save,
# which is quiet counterplay whose value materialises several plies
# out (the v1-vs-v1 oracle sees it only via deep playouts). No STATIC
# breadth signal separates the save from the block: the symmetric,
# asymmetric and one-sided gradients (v0 and v1 kernels) all prefer
# the losing move on 7-8 of the 10 White critical positions, because
# the block raises White's breadth too (iter-7 §2.1's one-sided
# refutation, reproduced on the v1 kernel — iter-16 §1.1). The fix is
# to extend the leaf's HORIZON: value the position AFTER a short
# rollout, where the counterplay has materialised. v16's White leaf
# rolls out exactly 2 plies (opponent reply + our reply) under a
# win/block/attack-argmax policy and evaluates the symmetric v1
# gradient at the resulting position:
#
# * rollout policy per ply (player P to move): a winning move wins
#   (terminal +1); >= 2 opponent winning moves loses (terminal -1);
#   exactly 1 opponent winning move is blocked (forced); else the
#   argmax over empty cells of P's v1 total score (own v1 attack +
#   opponent's v1 attack at the square — v1's full-value defence,
#   broken-four aware — from one ``attack_v1_both`` call, up to the
#   double-threat bonus, which v1's defence term omits). FAST-LOOP
#   PATCH (iter-16 §2.1): the pre-registered policy used the v0
#   kernel's attack-only argmax, whose opponent never prevents
#   broken-four creations (v0 is blind to them) — in fast-loop game
#   7 it replied (13,4) (v1 score 81) where v1's best is the
#   prevention (12,0) (20004), letting the quiet move borrow the
#   forcing move's value. The v1 policy is v1's 1-ply best move for
#   both sides: a real engine's reply, not a greedy attacker.
# * 2 plies, not 1: a 1-ply (opponent-reply-only) leaf measures
#   robustness to the opponent's best attack, which favours the
#   block — the save's value comes from OUR next move, so the
#   rollout must include it. 4 plies was measured and rejected
#   (iter-16 §1.1): the crude attack-argmax policy blunders at that
#   depth, returning false terminal losses (-1.000) for 3 of the 10
#   saving moves.
# * the final value is v14's symmetric v1 gradient (scale 7.5, cap
#   +/-0.2) — the v1 kernel feeding the value estimate, broken-four
#   aware. Terminal values are exact throughout (a terminal reached
#   inside the rollout returns the terminal value, not the gradient).
#
# The rollout leaf is White-only: when the engine plays Black, v16 IS
# v7 exactly (same leaf, same priors, same search), so the Black
# games are a built-in control. Two earlier v16 candidates were
# measured and refuted before pre-registration (iter-16 §1.1): the
# White asymmetric gradient (0/5 v13/v14 saves at the search level,
# saturates) and a symmetric must-block gradient term (regressed v7
# from 4/5 to 1/5 saves on its own regression positions — it softens
# the -0.25 penalties the cap protects). The one-sided must-block
# variant kept 4/5 + 0/5 (no regression, no signal). The rollout is
# the ONLY candidate with positive search-level signal: 2/5 v13/v14
# saves (v13-g9-ply15, v14-g3-ply13) at the cost of 1 v7 position
# (g5, plays a non-save non-loser).
_ROLLOUT_PLIES = 2
_GRADIENT_V1_SCALE = 7.5


def _attack_gradient_v1(board: np.ndarray, player: int) -> float:
    """Symmetric v1-gradient leaf value in (-0.2, 0.2) for ``player``
    to move in a quiet position, from the v1 kernel's attack-only
    scores for both colours (one ``attack_v1_both`` call):
    ``0.2 * tanh((breadth_v1(ours) - breadth_v1(theirs)) / 7.5)``.
    Antisymmetric under colour swap. v14's form (batch 5, H3), with
    the v1 thresholds and the re-derived scale (7.5); the 0.2 cap is
    unchanged (see ``_GRADIENT_CAP``)."""
    black, white = kernel_v1.attack_v1_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours, theirs = black, white
    else:
        ours, theirs = white, black
    diff = _breadth_v1(ours, mask) - _breadth_v1(theirs, mask)
    return _GRADIENT_CAP * math.tanh(diff / _GRADIENT_V1_SCALE)


def _rollout_move(board: np.ndarray, player: int) -> tuple[int, float | None]:
    """One rollout ply for ``player`` to move. Returns ``(move,
    terminal)`` where ``terminal`` is the value for ``player`` when
    the ply ends the game (+1: ``player`` has a winning move; -1:
    the opponent has >= 2 winning moves; 0.0: the board is full)
    and None when the game continues after ``move``. Policy: win if
    possible, block a single opponent winning move (forced), else
    the argmax over empty cells of ``player``'s v1 total score (own
    v1 attack + opponent's v1 attack at the square — v1's
    full-value defence, broken-four aware — from one
    ``attack_v1_both`` call, up to the double-threat bonus, which
    v1's defence term omits; ties by lowest flat index —
    deterministic). FAST-LOOP PATCH (iter-16 §2.1): the
    pre-registered policy used the v0 kernel's attack-only argmax,
    whose opponent never prevents broken-four creations."""
    wins = scoring._winning_moves(board, player)
    if wins:
        return wins[0], 1.0
    opp_wins = scoring._winning_moves(board, -player)
    if len(opp_wins) >= 2:
        return -1, -1.0
    if len(opp_wins) == 1:
        return opp_wins[0], None
    black, white = kernel_v1.attack_v1_both(board)
    if player == game.BLACK:
        own, opp = black, white
    else:
        own, opp = white, black
    empty = board == game.EMPTY
    if not empty.any():
        return -1, 0.0
    # FAST-LOOP PATCH (iter-16 §2.2): an own open-four creation
    # (own v1 attack >= OPEN_FOUR = 200_000) leaves >= 2 winning
    # moves — a forced win — and must not lose the argmax to a
    # defensive block. In fast-loop game 9, Black's (6,3) (own
    # 200080, an open four) lost to (6,8) (total 200422: a block of
    # one White four) by 332 points; (6,8) loses the game, (6,3)
    # wins it outright, and the rollout leaf hallucinated a White
    # win (+1.000) for a losing move. This sits after the forced
    # block above (an opponent four on the board must be answered
    # first — otherwise the opponent wins immediately) and before
    # the argmax. No non-forcing combination reaches 200_000 (the
    # next-largest bonus is DOUBLE_OPEN_THREE_BONUS = 50_000), so
    # the threshold is exact.
    masked_own = own.copy()
    masked_own[~empty] = -1
    if masked_own.max() >= scoring.OPEN_FOUR:
        return int(np.argmax(masked_own)), None
    total = own + opp
    total[~empty] = -1
    return int(np.argmax(total)), None


def _rollout_leaf_value(board: np.ndarray, player: int) -> float:
    """v16 White leaf value: v7's terminal-aware values for terminal
    positions; for quiet positions, roll out exactly 2 plies
    (opponent reply + our reply, :func:`_rollout_move`) and evaluate
    the symmetric v1 gradient at the resulting position. A win/loss
    reached inside the rollout returns the exact terminal value
    (sign-adjusted to ``player``'s perspective: after an even number
    of plies the player to move is ``player`` again). At the
    rollout's END, only wins/losses (+/-1) are terminal: a must-block
    (-0.25 — the opponent holds exactly one winning move) is a
    position to evaluate with the gradient, not a game end.
    FAST-LOOP PATCH (iter-16 §2.1): treating a rollout-found
    must-block as an exact -0.25 lets a quiet move borrow a forcing
    move's value (in fast-loop game 7, (11,9) scored +0.25 because
    the rollout plays White's forcing (12,0) on ply 2, tying the
    real forcing move (12,0) whose +0.25 is a FACT about the
    position, not a prediction under the 1-ply rollout policy). The
    must-block at the START (the real position) keeps v7's exact
    -0.25 — only the rollout's end is gradient-evaluated. The board
    is not mutated (the rollout plays on a copy)."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    cur = board.copy()
    p = player
    for i in range(_ROLLOUT_PLIES):
        move, term = _rollout_move(cur, p)
        if term is not None:
            return term if i % 2 == 0 else -term
        cur = game.apply_move(cur, move, p)
        p = -p
    # 2 plies: p == player again. Only a win/loss ends the rollout
    # game; a must-block here is gradient-evaluated (see above).
    if scoring._winning_moves(cur, p):
        return 1.0
    if len(scoring._winning_moves(cur, -p)) >= 2:
        return -1.0
    return _attack_gradient_v1(cur, p)


class HeuristicMCTS:
    """PUCT search with v1 priors and a terminal-aware leaf value.

    ``leaf_mode="terminal"`` (default) is v6: quiet leaves are 0.
    ``leaf_mode="gradient"`` is v7: quiet leaves get the attack
    breadth gradient (:func:`_gradient_leaf_value`); terminal
    values are identical in both modes.
    ``leaf_mode="rollout2_v1"`` is v16: when the engine plays White,
    quiet leaves get the 2-ply rollout leaf
    (:func:`_rollout_leaf_value`); when it plays Black, the leaf is
    v7's exactly. The mode is instance-bound (the leaf needs the
    root player's colour, recorded by :meth:`_run_one` on the root
    call).
    """

    def __init__(
        self,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
        leaf_mode: str = "terminal",
    ) -> None:
        if leaf_mode not in _LEAF_FUNCTIONS and (
            leaf_mode != "rollout2_v1"
        ):
            raise ValueError(
                f"unknown leaf_mode: {leaf_mode!r} "
                f"(known: {sorted(_LEAF_FUNCTIONS) + ['rollout2_v1']})"
            )
        self.sims = sims
        self.c_puct = c_puct
        self.rng = rng
        self.leaf_mode = leaf_mode
        if leaf_mode == "rollout2_v1":
            self._leaf_value = self._rollout_v1_leaf_value
        else:
            self._leaf_value = _LEAF_FUNCTIONS[leaf_mode]
        # Colour to move at the root of the current search (the
        # engine's colour for this move). Set by _run_one on every
        # root call — the same instance is reused across games in
        # both colours, so it must be re-recorded per move, never
        # set once. Read only by the v16 leaf.
        self._root_player: int | None = None

    def _rollout_v1_leaf_value(
        self, board: np.ndarray, player: int
    ) -> float:
        """v16 leaf value: v7's leaf (:func:`_gradient_leaf_value`)
        when the engine plays Black — making v16-as-Black EXACTLY
        v7 (same leaf function, same priors, same search), so the
        Black games are a built-in control — and the 2-ply rollout
        leaf (:func:`_rollout_leaf_value`) when it plays White.
        ``_root_player`` is set by :meth:`_run_one` on the root call
        before any leaf is evaluated; ``None`` (leaf called outside
        a search) falls back to the White branch."""
        if self._root_player == game.BLACK:
            return _gradient_leaf_value(board, player)
        return _rollout_leaf_value(board, player)

    def choose_move(self, board: np.ndarray, player: int) -> int:
        # Immediate win (any winning move wins).
        wins = scoring._winning_moves(board, player)
        if wins:
            return wins[0]
        # Block the opponent's immediate win (if exactly one; with
        # 2+ the game is lost and the search picks the best effort).
        opp_wins = scoring._winning_moves(board, -player)
        if len(opp_wins) == 1:
            return opp_wins[0]
        cands = scoring._candidate_moves(board)
        if not cands:
            return -1
        if len(cands) == 1:
            return cands[0]
        root = _Node(1.0)
        for _ in range(self.sims):
            self._run_one(root, board, player, None)
        best_visits = max(c.visit_count for c in root.children.values())
        best = [
            m for m, c in root.children.items() if c.visit_count == best_visits
        ]
        if len(best) == 1:
            return best[0]
        if self.rng is not None:
            return int(self.rng.choice(best))
        return best[0]

    def _select_child(self, node: _Node) -> int:
        """PUCT: ``-child.q`` (child values are from the opponent's
        perspective) plus the exploration term."""
        best_move: int | None = None
        best_score = -float("inf")
        parent_visits = node.visit_count
        for move, child in node.children.items():
            score = (
                -child.q
                + self.c_puct
                * child.prior
                * math.sqrt(parent_visits)
                / (1 + child.visit_count)
            )
            if score > best_score:
                best_score, best_move = score, move
        return best_move

    def _run_one(
        self,
        node: _Node,
        board: np.ndarray,
        player: int,
        last_move: int | None,
    ) -> float:
        """One simulation. Returns the value from ``player``'s
        perspective and updates every node on the path (including
        the newly expanded leaf). Mutates ``board`` in place but
        restores it before returning."""
        # Terminal check: the move that led here just ended the game.
        if last_move is not None:
            if game.is_winning_move(board, last_move, -player):
                node.visit_count += 1
                node.value_sum += -1.0
                return -1.0
            if game.is_full(board):
                node.visit_count += 1
                return 0.0
        # Expansion: first visit — compute priors, evaluate the leaf.
        if node.children is None:
            priors = _priors(board, player)
            node.children = {m: _Node(p) for m, p in priors.items()}
            if last_move is None:  # root: no leaf value
                # Record the root player (the engine's colour for
                # this move) — the v16 leaf branches on it. Set on
                # every root call: the instance is reused across
                # games in both colours.
                self._root_player = player
                node.visit_count += 1
                return 0.0
            value = self._leaf_value(board, player)
            node.visit_count += 1
            node.value_sum += value
            return value
        # Selection: descend to the best child by PUCT.
        move = self._select_child(node)
        board[move // 15, move % 15] = player
        value = -self._run_one(node.children[move], board, -player, move)
        board[move // 15, move % 15] = 0
        node.visit_count += 1
        node.value_sum += value
        return value

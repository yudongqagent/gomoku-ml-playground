"""PUCT MCTS with the v1 heuristic as prior and leaf value (v6, batch 2 H2;
v7, batch 3 H1, adds an attack gradient to the leaf; v16, batch 6 H2,
adds a White-only 2-ply rollout leaf — see below; v18, batch 7 H1,
applies the same rollout leaf as Black too).

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
* Move choice: by default the most-visited root child, ties by
  the seeded RNG (v6/v7). v9 (batch 4, H1) varies the ROOT
  DECISION rule via ``root_decision``: ``"q"`` (argmax
  our-perspective q), ``"tie-q"`` (most-visited, ties by q),
  ``"blend"`` (q + a visit term), ``"tie-prior"`` (most-visited,
  ties by v1 prior) — see :meth:`HeuristicMCTS._choose_root_child`
  and ``research/iterations/iter-9.md`` §1, including the SIGN
  CORRECTION there: a root child's raw ``q`` is the OPPONENT's
  value (negamax backup), so "q" in every rule means ``-child.q``.
  v12 (batch 5, H1) adds ``"tie-prior-forcing"`` (most-visited,
  ties by forcing-ness first, then v1 prior) — see
  ``research/iterations/iter-12.md`` §1.1.
  Sims: 100, c_puct: 1.5 — the SAME settings as the
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
* v18 (``leaf_mode="rollout2_v1_both"``, batch 7 H1): the SAME
  2-ply rollout leaf as v16, applied as Black too — the dispatch
  returns ``_rollout_leaf_value`` for both colours. When the
  engine plays White, v18 IS v16 exactly, so the White games are
  a built-in control; see ``research/iterations/iter-18.md``.

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
from . import evaluate
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

# --- v14 attack-gradient leaf (batch 5, H3) -------------------------------
# Same gradient FORM as v7 (breadth count, tanh squash, +/-0.2 cap),
# but the breadth counts come from the v1 kernel's attack scores
# (5-window, broken-four-aware) instead of the v0 kernel's
# (contiguous-run, blind to broken fours). Thresholds are the v1
# analogues of v7's, derived from the v1 pattern classes (see
# iter-14.md §1): strong = a move creating a four or an open three
# (v1 attack >= 4000 — no combination of sub-4000 per-direction
# values reaches 4000); developing = a move creating a half-open
# three (400) or an open two (80) (80 <= attack < 4000 — the
# strongest sub-open-two combination is 4 x two = 40). The tanh
# scale (7.5) is re-derived from the measured v1 breadth-difference
# distribution (std 4.09 on batch-3 game positions) by v7's rule
# (scale ~= 1.8 x std); the cap (0.2) is UNCHANGED — a static
# heuristic must never outweigh an exact terminal signal, and the
# cap is what made v7 pass.
_GRADIENT_V1_STRONG = 4000
_GRADIENT_V1_DEVELOPING = 80
_GRADIENT_V1_SCALE = 7.5

# --- v9 root-decision rules (batch 4, H1) ----------------------------------
# ``root_decision`` selects how the root child is picked after the
# search (see HeuristicMCTS._choose_root_child). "visits" is the
# v6/v7 default. The blend weight (0.1) is fixed in the
# pre-registration (iter-9.md §1.1): q gaps below 0.1 are within
# one gradient step (the gradient's full range is +/-0.2), so the
# visit term overrides only sub-gradient-scale q differences.
# v12 (batch 5, H1) adds "tie-prior-forcing": most-visited, ties
# broken by forcing-ness first (moves creating a four), then the v1
# prior — see iter-12.md §1.1.
_ROOT_DECISIONS = (
    "visits", "q", "tie-q", "blend", "tie-prior", "tie-prior-forcing",
)
_BLEND_VISIT_WEIGHT = 0.1


class _Node:
    """One MCTS node. ``children`` is None until expansion.

    ``softmax`` (v51, batch 13 H1) marks an opponent node under
    the soft-min backup rule: its q is the softmax-weighted mean
    of the opponent-perspective values of its visited children
    (``-child.q``), weighted by ``exp(beta * value)``, instead of
    the mean. ``softmax_value`` caches that value, maintained by
    :meth:`HeuristicMCTS._run_one` after every descent through
    the node; it is None until the first descent (no visited
    children yet), and :attr:`q` falls back to the mean until
    then — which is the node's leaf value right after expansion.
    """

    __slots__ = (
        "prior", "visit_count", "value_sum", "children",
        "softmax", "softmax_value",
    )

    def __init__(self, prior: float, softmax: bool = False) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "_Node"] | None = None
        self.softmax = softmax
        self.softmax_value: float | None = None

    @property
    def q(self) -> float:
        """Backed-up value from this node's player perspective.

        The mean (``value_sum / visit_count``) for mean nodes;
        the cached soft-min value for soft-min nodes once any
        child has been visited (before that, the mean — the
        node's leaf value)."""
        if self.softmax and self.softmax_value is not None:
            return self.softmax_value
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def _softmax_mean(values: list[float], beta: float) -> float:
    """Softmax-weighted mean of ``values`` with temperature ``beta``:
    ``sum(w_i * v_i)`` with ``w_i`` proportional to
    ``exp(beta * v_i)``. The v51 soft-min backup
    (:meth:`HeuristicMCTS._run_one`) applies it to the
    opponent-perspective values of a soft-min node's visited
    children. Max-subtracted weights, so no overflow is possible;
    the result stays in ``[min(values), max(values)]``. A single
    value returns that value exactly (no smoothing with one
    child — see iter-51.md §1.4 for why that matters at 100
    sims)."""
    top = max(values)
    weights = [math.exp(beta * (v - top)) for v in values]
    total = sum(weights)
    return sum(w * v for w, v in zip(weights, values)) / total


def _priors(
    board: np.ndarray, player: int, prior_floor: float = 0.0,
) -> dict[int, float]:
    """Softmax over log1p(v1 scores) / temperature, over all
    candidate moves. Returns a move -> prior dict summing to 1
    (empty when there are no candidates) — UNLESS ``prior_floor``
    is set (v15, batch 6 H1; see iter-15.md §1.0): priors below
    the floor are raised to the floor and the sum is NOT
    re-divided. PUCT uses the prior only as an exploration weight,
    so the weights need not sum to 1; and Step 0 measured that
    renormalization shrinks the very moves the floor is meant to
    lift (a 0.027 save becomes 0.020 at floor 0.01, 0.0127 at
    0.02 — the tail's added mass dominates with ~93 candidates).
    The unnormalized floor is a pure exploration bonus for the
    tail: above-floor moves keep their EXACT v7 priors, so the
    sharp prior is preserved where it matters. A floor is not
    flattening (v13's temperature probe: flattening is
    catastrophic)."""
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
    priors = {m: w / total for m, w in weights.items()}
    if prior_floor > 0.0:
        priors = {m: max(p, prior_floor) for m, p in priors.items()}
    return priors


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


def _breadth_v1(attack: np.ndarray, empty_mask: np.ndarray) -> float:
    """Attack breadth of one side from v1 attack scores: the count
    of its strong moves (v1 attack >= 4000: four/open-three makers)
    plus 0.25 per developing move (v1 attack in [80, 4000):
    half-open-three / open-two makers), over empty cells only.
    Same form as :func:`_breadth`, with the v1 pattern-class
    thresholds (see ``_GRADIENT_V1_STRONG``)."""
    s = attack[empty_mask]
    strong = int(np.count_nonzero(s >= _GRADIENT_V1_STRONG))
    developing = int(
        np.count_nonzero((s >= _GRADIENT_V1_DEVELOPING) & (s < _GRADIENT_V1_STRONG))
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


def _attack_gradient_v1(board: np.ndarray, player: int) -> float:
    """Breadth-gradient leaf value in (-0.2, 0.2) for ``player`` to
    move in a quiet position, from the v1 kernel's attack-only
    scores for both colours (one ``attack_v1_both`` call):
    ``0.2 * tanh((breadth_v1(ours) - breadth_v1(theirs)) / 7.5)``.
    Antisymmetric under colour swap. Same form as
    :func:`_attack_gradient`, with the v1 thresholds and the
    re-derived scale (7.5); the 0.2 cap is unchanged (see
    ``_GRADIENT_CAP``)."""
    black, white = kernel_v1.attack_v1_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours, theirs = black, white
    else:
        ours, theirs = white, black
    diff = _breadth_v1(ours, mask) - _breadth_v1(theirs, mask)
    return _GRADIENT_CAP * math.tanh(diff / _GRADIENT_V1_SCALE)


def _gradient_v1_leaf_value(board: np.ndarray, player: int) -> float:
    """v14 leaf value: v7's terminal-aware values for terminal
    positions, the v1-kernel attack gradient for quiet ones. The
    terminal cases are identical to :func:`_terminal_leaf_value`
    BY CONSTRUCTION (it returns 0.0 exactly for quiet positions),
    so the ONLY difference between v7 and v14 is the quiet-leaf
    gradient's input scores — a clean A/B of the v1 kernel."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    return _attack_gradient_v1(board, player)


# --- v20 gap-three demotion (batch 7, H3) ---------------------------------
# The v1 breadth counts gap-three makers as STRONG (a gap three
# XX_X scores 4006 in v1 — see scoring._score_from_counts, which
# awards OPEN_THREE whenever a window holds 3 friendly stones with
# both outside cells empty, contiguous or gapped). But a gap three
# is killable by a single opponent move occupying the gap (XXOX is
# dead), while a true open three (_XXX_) cannot be killed in one
# move — and greedy 1-ply play finds the gap block automatically
# (its defence term scores the gap-fill as blocking a 4-run). Gap-
# three breadth is therefore PHANTOM against greedy play: it
# evaporates on contact while the opponent builds its own race.
# Iter-20 §2.0 verified the mechanism causally (6/6 engine-swap
# continuations flip a v14 Black loss to v0 into a win when v7's
# move replaces v14's; the ply-10 decomposition shows the v1
# gradient's +4.0 preference for the cluster four comes entirely
# from the three tier, 4 of whose 6 makers are gap-three makers)
# and §2.1 confirmed it at a fresh seed (game 6 ply 28: v14 plays
# the gap-heavy loser, v7 plays the save).
#
# The fix: a three-tier move (v1 attack in [4000, 20000)) counts
# as STRONG only if it is a TRUE open-three maker — detected with
# the v0 kernel as the filter (a true open three scores >= 1000 in
# v0's contiguous-run scoring; a gap-three maker scores ~40-200,
# because v0 cannot see the gap pattern at all). A three-tier move
# with v0 attack < 1000 is a GAP-three maker and counts as
# DEVELOPING (weight 0.25, the existing developing weight — no new
# constant). Four-tier moves (v1 >= 20000, INCLUDING broken-four
# makers) are unaffected: a broken four has exactly one winning
# square, the same as a straight four, so both force a block
# identically. The developing tier [80, 4000) is unaffected. The
# tanh scale is re-derived from the demoted breadth distribution
# by v7's rule (scale ~= 1.8 x std) in iter-20 §1.1; the cap
# (+/-0.2) is UNCHANGED.
_GRADIENT_V1_GAP_SCALE = 5.5  # Step-0 value (iter-20 §1.1): 1.8 x
# the demoted breadth-difference std (3.02), rounded to the nearest
# 0.5 by v7's rule — the same rule v7 (4.0) and v14 (7.5) used.


def _breadth_v1_gap(
    attack_v1: np.ndarray, attack_v0: np.ndarray, empty_mask: np.ndarray
) -> float:
    """Attack breadth of one side from v1 attack scores with
    gap-three makers DEMOTED to the developing tier (v20, batch 7
    H3): strong moves are four-makers (v1 attack >= 20000,
    including broken fours) plus TRUE open-three makers (v1 attack
    in [4000, 20000) AND v0 attack >= 1000); developing moves
    (weight 0.25) are the [80, 4000) tier plus GAP-three makers
    (v1 attack in [4000, 20000) AND v0 attack < 1000). Over empty
    cells only. The v0 attack array is the filter: it is 0-mismatch
    verified (kernel_numpy) and distinguishes the two three kinds
    exactly where the v1 score cannot (gap-three 4006 vs
    open-three 4000+)."""
    s1 = attack_v1[empty_mask]
    s0 = attack_v0[empty_mask]
    four = int(np.count_nonzero(s1 >= scoring.FOUR))
    three = (s1 >= _GRADIENT_V1_STRONG) & (s1 < scoring.FOUR)
    true_three = int(
        np.count_nonzero(three & (s0 >= _GRADIENT_STRONG))
    )
    gap_three = int(
        np.count_nonzero(three & (s0 < _GRADIENT_STRONG))
    )
    developing = int(
        np.count_nonzero(
            (s1 >= _GRADIENT_V1_DEVELOPING) & (s1 < _GRADIENT_V1_STRONG)
        )
    )
    return (
        four
        + true_three
        + _GRADIENT_DEVELOPING_WEIGHT * (developing + gap_three)
    )


def _attack_gradient_v1_gap(board: np.ndarray, player: int) -> float:
    """Breadth-gradient leaf value in (-0.2, 0.2) for ``player`` to
    move in a quiet position, from the v1 kernel's attack scores
    with gap-three makers demoted (see :func:`_breadth_v1_gap`),
    using the v0 kernel's attack scores as the true/gap filter (one
    ``attack_v1_both`` call plus one ``attack_both`` call):
    ``0.2 * tanh((breadth_gap(ours) - breadth_gap(theirs)) /
    scale)``. Antisymmetric under colour swap. Same form as
    :func:`_attack_gradient_v1`, with the demoted breadth and the
    re-derived scale; the 0.2 cap is unchanged (see
    ``_GRADIENT_CAP``)."""
    black1, white1 = kernel_v1.attack_v1_both(board)
    black0, white0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours1, theirs1 = black1, white1
        ours0, theirs0 = black0, white0
    else:
        ours1, theirs1 = white1, black1
        ours0, theirs0 = white0, black0
    diff = _breadth_v1_gap(ours1, ours0, mask) - _breadth_v1_gap(
        theirs1, theirs0, mask
    )
    return _GRADIENT_CAP * math.tanh(diff / _GRADIENT_V1_GAP_SCALE)


def _gradient_v1_gap_leaf_value(board: np.ndarray, player: int) -> float:
    """v20 leaf value: v7's terminal-aware values for terminal
    positions, the gap-demoted v1-kernel attack gradient for quiet
    ones. The terminal cases are identical to
    :func:`_terminal_leaf_value` BY CONSTRUCTION (it returns 0.0
    exactly for quiet positions), so the ONLY difference between
    v14 and v20 is the quiet-leaf gradient's three-tier weighting
    — a clean A/B of the gap-three demotion."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    return _attack_gradient_v1_gap(board, player)


# --- v22 colour-conditional gap-three demotion (batch 8, H2) ---------------
# v20's GLOBAL demotion traded Black for White: against greedy v0
# it recovered Black (losses 20 -> 9, win rate 0.72) but broke
# White (losses 20 -> 27, win rate 0.45) — iter-20 §3.4. The
# lesson (iter-20 §5): gap-three breadth is phantom for the
# initiative-holder (Black moves first and holds the tempo; Black
# builds clusters with a free hand and the defender kills the gap
# threes for free) but signal for White's counterplay (White
# creates threats in reply, and Black — who must answer them —
# cannot spend the move killing the gap). v22 therefore demotes
# gap-three makers ONLY in BLACK's breadth, in both terms of the
# gradient difference: at Black-to-move leaves Black's OWN breadth
# is demoted (the ply-10 fix — the v1 gradient's +4.0 preference
# for the losing cluster move came entirely from Black's gap-three
# makers); at White-to-move leaves Black's THEIRS breadth is
# demoted (White does not over-fear Black's phantom threats — the
# term that made v20's White BETTER against v7, 0.48 -> 0.53).
# White's breadth is never demoted. The leaf stays antisymmetric
# under PLAYER swap (value(board, WHITE) == -value(board, BLACK),
# the property MCTS needs) but is deliberately NOT antisymmetric
# under colour swap. Alternatives considered and rejected are in
# research/iterations/iter-22.md §1 (position-based initiative,
# per-game switch, ours-only, theirs-only). The tanh scale is
# re-derived from the conditional breadth-difference distribution
# by v7's rule (scale ~= 1.8 x std) in iter-22 §1.1; the cap
# (+/-0.2) is UNCHANGED.
_GRADIENT_V1_GAP_BLACK_SCALE = 7.0  # Step-0 value (iter-22 §1.1):
# 1.8 x the conditional breadth-difference std (3.75), rounded to
# the nearest 0.5 by v7's rule — the same rule v7 (4.0), v14 (7.5),
# and v20 (5.5) used.


def _attack_gradient_v1_gap_black(board: np.ndarray, player: int) -> float:
    """Breadth-gradient leaf value in (-0.2, 0.2) for ``player`` to
    move in a quiet position, from the v1 kernel's attack scores
    with gap-three makers demoted ONLY in BLACK's breadth (see the
    v22 comment block above): Black's breadth via
    :func:`_breadth_v1_gap`, White's via :func:`_breadth_v1`, one
    ``attack_v1_both`` call plus one ``attack_both`` call (the v0
    scores are the true/gap filter for Black's breadth):
    ``0.2 * tanh((breadth(ours) - breadth(theirs)) / scale)``.
    Antisymmetric under PLAYER swap (``player`` BLACK vs WHITE),
    NOT under colour swap — by design. Same form as
    :func:`_attack_gradient_v1_gap`, with the conditional breadth
    and the re-derived scale; the 0.2 cap is unchanged (see
    ``_GRADIENT_CAP``)."""
    black1, white1 = kernel_v1.attack_v1_both(board)
    black0, _white0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    black_breadth = _breadth_v1_gap(black1, black0, mask)
    white_breadth = _breadth_v1(white1, mask)
    if player == game.BLACK:
        diff = black_breadth - white_breadth
    else:
        diff = white_breadth - black_breadth
    return _GRADIENT_CAP * math.tanh(diff / _GRADIENT_V1_GAP_BLACK_SCALE)


def _gradient_v1_gap_black_leaf_value(board: np.ndarray, player: int) -> float:
    """v22 leaf value: v7's terminal-aware values for terminal
    positions, the Black-only gap-demoted v1-kernel attack gradient
    for quiet ones. The terminal cases are identical to
    :func:`_terminal_leaf_value` BY CONSTRUCTION (it returns 0.0
    exactly for quiet positions), so the ONLY difference between
    v14 and v22 is the quiet-leaf gradient's three-tier weighting
    for Black's breadth — a clean A/B of the colour-conditional
    gap-three demotion."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    return _attack_gradient_v1_gap_black(board, player)


# --- v28 ours-only Black demotion (batch 9, H2) ---------------------------
# v22's Black-only demotion (both terms) traded Black for White in
# the opposite direction to its prediction: vs v0, Black 0.93 (ZERO
# losses) and White 0.38 (30 losses) — iter-22 §3.4. The
# decomposition prices demoting Black's breadth at +0.33 for Black
# and -0.22 for White, and the two terms are separable: the +0.33
# is an OURS-term effect (the ply-10 fix — the v1 gradient's +4.0
# preference for the losing cluster move came entirely from Black's
# OWN gap-three makers at a Black-to-move leaf), while the -0.22 is
# a THEIRS-term effect (at White-to-move leaves Black's gap-three
# breadth is demoted, so White UNDER-FEARS Black's threats — the
# fast-loop games 7/9 mechanism, measured at battery scale). v28
# therefore demotes Black's gap-three makers ONLY in the OURS term
# — i.e. only at Black-to-move leaves, where Black's breadth IS the
# ours term — and never in the theirs term: at White-to-move leaves
# NOTHING is demoted, so White fears Black's gap threes fully.
# White's breadth is never demoted (as in v22). Equivalently: v28
# IS v22 at Black-to-move leaves and v14 at White-to-move leaves
# (modulo the re-derived tanh scale, which applies to both
# parities). This is NOT iter-22 §1's rejected option 3 (colour-
# blind ours-only demotion, which would demote White's own breadth
# at White-to-move leaves): v28 is colour-conditional AND
# term-conditional.
#
# The condition DELIBERATELY breaks player-swap antisymmetry
# (value(board, WHITE) != -value(board, BLACK)): -value(board,
# BLACK) = 0.2*tanh((v1(W) - gap(B))/s) >= value(board, WHITE) =
# 0.2*tanh((v1(W) - v1(B))/s), since gap(B) <= v1(B). The break has
# a definite direction — White-to-move leaves are valued LOWER than
# the mirror of Black-to-move leaves, because Black's full
# undemoted breadth counts against White in the theirs term. That
# IS the intended effect. MCTS backup remains valid: each board has
# a fixed player to move by stone-count parity, so in a real game
# tree a node is only ever evaluated with its correct player, and
# the backup (negate along the path) is correct as long as each
# leaf returns the value for the player to move at that leaf.
# Precedent: v16's rollout leaf is colour-gated (when the engine
# plays Black, v16 IS v7 exactly). The tanh scale is re-derived
# from the v28 breadth-difference distribution (BOTH parities
# pooled — the rule is parity-dependent) by v7's rule (scale ~=
# 1.8 x std) in iter-28 §1.1; the cap (+/-0.2) is UNCHANGED.
_GRADIENT_V1_GAP_OURS_SCALE = 8.0  # Step-0 value (iter-28 §1.1):
# 1.8 x the pooled v28 breadth-difference std (4.54), rounded to
# the nearest 0.5 by v7's rule — the same rule v7 (4.0), v14 (7.5),
# v20 (5.5), and v22 (7.0) used. Both parities pooled: the rule is
# parity-dependent, so the Step-0 sample covers both (577 Black
# to move, 574 White to move).


def _attack_gradient_v1_gap_ours(board: np.ndarray, player: int) -> float:
    """Breadth-gradient leaf value in (-0.2, 0.2) for ``player`` to
    move in a quiet position, from the v1 kernel's attack scores
    with gap-three makers demoted ONLY in BLACK's breadth and ONLY
    in the OURS term (see the v28 comment block above): at
    Black-to-move leaves Black's breadth via
    :func:`_breadth_v1_gap` and White's via :func:`_breadth_v1`
    (v22's rule); at White-to-move leaves both breadths via
    :func:`_breadth_v1` (v14's rule — Black's theirs-breadth
    counted FULLY). One ``attack_v1_both`` call plus one
    ``attack_both`` call (the v0 scores are the true/gap filter
    for Black's breadth at Black-to-move leaves):
    ``0.2 * tanh((breadth(ours) - breadth(theirs)) / scale)``.
    Deliberately NOT antisymmetric under player swap (see the
    comment block above) — the inequality direction is
    ``-value(board, BLACK) >= value(board, WHITE)``. Same form as
    :func:`_attack_gradient_v1_gap_black`, with the ours-only
    breadth and the re-derived scale; the 0.2 cap is unchanged
    (see ``_GRADIENT_CAP``)."""
    black1, white1 = kernel_v1.attack_v1_both(board)
    black0, _white0 = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    white_breadth = _breadth_v1(white1, mask)
    if player == game.BLACK:
        # Black's breadth is the ours term: demoted (v22's rule).
        diff = _breadth_v1_gap(black1, black0, mask) - white_breadth
    else:
        # Black's breadth is the theirs term: counted FULLY
        # (v14's rule) — White fears Black's gap threes.
        diff = white_breadth - _breadth_v1(black1, mask)
    return _GRADIENT_CAP * math.tanh(diff / _GRADIENT_V1_GAP_OURS_SCALE)


def _gradient_v1_gap_ours_leaf_value(board: np.ndarray, player: int) -> float:
    """v28 leaf value: v7's terminal-aware values for terminal
    positions, the ours-only Black-demoted v1-kernel attack
    gradient for quiet ones. The terminal cases are identical to
    :func:`_terminal_leaf_value` BY CONSTRUCTION (it returns 0.0
    exactly for quiet positions), so the ONLY difference between
    v14 and v28 is the quiet-leaf gradient's three-tier weighting
    for Black's breadth in the ours term — a clean A/B of the
    ours-only Black demotion."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    return _attack_gradient_v1_gap_ours(board, player)


_LEAF_FUNCTIONS = {
    "terminal": _terminal_leaf_value,
    "gradient": _gradient_leaf_value,
    "gradient_v1": _gradient_v1_leaf_value,
    "gradient_v1_gap": _gradient_v1_gap_leaf_value,
    "gradient_v1_gap_black": _gradient_v1_gap_black_leaf_value,
    "gradient_v1_gap_ours": _gradient_v1_gap_ours_leaf_value,
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


def _blend_rollout_terminal(
    term: float, board: np.ndarray, player: int
) -> float:
    """Blend a rollout-found terminal value with the symmetric v1
    gradient at the terminal position (v19, batch 7 H2 — see
    ``research/iterations/iter-19.md`` §1).

    ``term`` is ±1 from ``player``'s perspective at ``board`` (the
    position where the rollout reached the terminal — the position
    BEFORE the winning move is played; the terminal is a prediction
    that the player to move there wins, not a finished game). The
    blend is ``clamp(term + g, -1, 1)`` where ``g`` is the v1
    gradient at that position for ``player``. The gradient's ±0.2
    cap bounds the shift, so a blended rollout loss lands in
    [-1.0, -0.8] and a blended rollout win in [0.8, 1.0]: the
    terminal's dominance over quiet leaves (±0.2) and the
    must-block (-0.25) is preserved, and the blend reorders ONLY
    among lines that share a terminal outcome — precisely the set
    v16's raw terminals flatten (iter-16 §3.1: in creator-rich
    positions nearly every quiet leaf returns -1.000, so the search
    cannot rank defences it knows are all losing). The clamp is the
    predictions-never-exceed-facts rule: a rollout win must not
    outrank an exact win at the leaf start (+1.0), nor a rollout
    loss sink below an exact loss (-1.0). A draw (term 0.0, board
    full) is returned unchanged — a full board has no empty cells,
    so its gradient is 0 anyway.
    """
    if term == 0.0:
        return 0.0
    return max(-1.0, min(1.0, term + _attack_gradient_v1(board, player)))


def _rollout_leaf_value(
    board: np.ndarray, player: int, blend: bool = False
) -> float:
    """v16 White leaf value: v7's terminal-aware values for terminal
    positions; for quiet positions, roll out exactly 2 plies
    (opponent reply + our reply, :func:`_rollout_move`) and evaluate
    the symmetric v1 gradient at the resulting position. A win/loss
    reached inside the rollout returns the exact terminal value
    (sign-adjusted to ``player``'s perspective: after an even number
    of plies the player to move is ``player`` again) — or, with
    ``blend=True`` (v19), the terminal blended with the v1 gradient
    at the terminal position (:func:`_blend_rollout_terminal`),
    sign-adjusted the same way. At the rollout's END, only
    wins/losses (+/-1) are terminal: a must-block (-0.25 — the
    opponent holds exactly one winning move) is a position to
    evaluate with the gradient, not a game end.
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
            if blend:
                term = _blend_rollout_terminal(term, cur, p)
            return term if i % 2 == 0 else -term
        cur = game.apply_move(cur, move, p)
        p = -p
    # 2 plies: p == player again. Only a win/loss ends the rollout
    # game; a must-block here is gradient-evaluated (see above).
    if scoring._winning_moves(cur, p):
        return _blend_rollout_terminal(1.0, cur, p) if blend else 1.0
    if len(scoring._winning_moves(cur, -p)) >= 2:
        return _blend_rollout_terminal(-1.0, cur, p) if blend else -1.0
    return _attack_gradient_v1(cur, p)


# --- v53 2-ply rollout search leaf (batch 13, H3) --------------------------
# Written FRESH from the pre-registered spec in
# ``research/iterations/iter-53.md`` §1.1 — deliberately NOT an
# alias of :func:`_rollout_leaf_value`, so the differential test
# (``tests/test_regression_v53.py``) tests the SPEC against v27's
# leaf rather than an alias against itself. The spec: at a leaf
# with ``player`` to move, play exactly 2 rollout plies under the
# teacher's own policy (:func:`_rollout_move` — win if possible,
# block a single opponent winning move, open-four priority, else
# the v1-total argmax) and evaluate the symmetric v1 gradient at
# the resulting position. Terminal handling follows v16/v18's
# conventions exactly: the leaf-start terminal (including the
# -0.25 must-block, a fact about the real position) is exact;
# terminals reached INSIDE the rollout return the exact terminal
# value, sign-adjusted to ``player``'s perspective; at the
# rollout's END only wins/losses are terminal — a must-block
# there is gradient-evaluated (iter-16 §2.1: a rollout-end
# must-block is a prediction under the rollout policy, not a
# fact). No blend (v19's blend was rejected and is not part of
# v27). The board is not mutated (the rollout plays on copies).
#
# Premise note (iter-53 §0): this spec, implemented faithfully,
# IS v18/v27's leaf — ``_rollout_leaf_value`` already rolls out
# exactly 2 plies (``_ROLLOUT_PLIES = 2``) under this same
# policy. The batch-13 plan's "1-ply (one random move)" premise
# conflated the ML label-depth naming (plies BEFORE the teacher
# leaf: ``_rollout2_leaf_value`` is 2 plies + the v18 leaf = 4
# plies effective) with the search leaf's own depth. The
# differential test is the verdict on whether the spec's natural
# implementation coincides with v27's leaf.
def _rollout2ply_search_leaf_value(board: np.ndarray, player: int) -> float:
    """v53 search leaf: v7's terminal-aware values for terminal
    positions; for quiet positions, roll out exactly 2 plies under
    the teacher's own rollout policy (:func:`_rollout_move`) and
    evaluate the symmetric v1 gradient at the resulting position.
    See ``research/iterations/iter-53.md`` §1.1 for the
    pre-registered spec. Blend is NOT applied (v19 rejected, not
    part of v27). The board is not mutated."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    # Ply 1: player to move.
    m1, term1 = _rollout_move(board, player)
    if term1 is not None:
        return term1
    b1 = game.apply_move(board, m1, player)
    # Ply 2: opponent's reply.
    m2, term2 = _rollout_move(b1, -player)
    if term2 is not None:
        return -term2
    b2 = game.apply_move(b1, m2, -player)
    # Rollout end: player to move again. Only wins/losses are
    # terminal here; a must-block is gradient-evaluated.
    if scoring._winning_moves(b2, player):
        return 1.0
    if len(scoring._winning_moves(b2, -player)) >= 2:
        return -1.0
    return _attack_gradient_v1(b2, player)


# --- v53 3-ply rollout search leaf (batch 13, H3, re-scoped) -----------------
# The re-scoped v53 (``research/iterations/iter-53.md`` §6): v27 +
# a 3-ply rollout leaf — the only unpriced rung of the heuristic
# depth ladder (1 ply rejected by v16, 2 plies = v18/v27, 4 plies
# rejected twice by v16/v40). Written fresh from the pre-registered
# spec in iter-53 §6.1, reusing the SAME deterministic rollout
# policy (:func:`_rollout_move`) as the 2-ply leaf, so depth is the
# only change. The parity handling at the rollout end is a design
# decision, not a detail: after an ODD number of rollout plies the
# side to move is the opponent, so the rollout-end value is
# computed from the opponent's perspective and negated to the leaf
# player's perspective — for BOTH the terminal checks (an opponent
# winning move is a loss, −1.0; our double threat is a win, +1.0)
# and the gradient (``-_attack_gradient_v1(b3, -player)``, which
# equals ``_attack_gradient_v1(b3, player)`` by the gradient's
# antisymmetry under colour swap — the explicit negated form is
# used so the parity handling is visible in the code). Terminal
# handling otherwise follows v16/v18's conventions exactly: the
# leaf-start terminal (including the −0.25 must-block, a fact
# about the real position) is exact; terminals reached INSIDE the
# rollout return the exact terminal value, sign-adjusted to
# ``player``'s perspective (ply 1 and ply 3 are ``player``'s plies,
# ply 2 the opponent's); at the rollout's END only wins/losses are
# terminal — a must-block there is gradient-evaluated (iter-16
# §2.1). No blend (v19 rejected, not part of v27). The board is
# not mutated (the rollout plays on copies).
def _rollout3ply_search_leaf_value(board: np.ndarray, player: int) -> float:
    """v53 (re-scoped) search leaf: v7's terminal-aware values for
    terminal positions; for quiet positions, roll out exactly 3
    plies under the teacher's own rollout policy
    (:func:`_rollout_move`) and evaluate the symmetric v1 gradient
    at the resulting position, negated from the opponent-to-move
    perspective. See ``research/iterations/iter-53.md`` §6.1 for
    the pre-registered spec. Blend is NOT applied (v19 rejected,
    not part of v27). The board is not mutated."""
    value = _terminal_leaf_value(board, player)
    if value != 0.0:
        return value
    # Ply 1: player to move.
    m1, term1 = _rollout_move(board, player)
    if term1 is not None:
        return term1
    b1 = game.apply_move(board, m1, player)
    # Ply 2: opponent's reply.
    m2, term2 = _rollout_move(b1, -player)
    if term2 is not None:
        return -term2
    b2 = game.apply_move(b1, m2, -player)
    # Ply 3: player to move again.
    m3, term3 = _rollout_move(b2, player)
    if term3 is not None:
        return term3
    b3 = game.apply_move(b2, m3, player)
    # Rollout end: OPPONENT to move. Only wins/losses are
    # terminal here; a must-block is gradient-evaluated. The
    # terminal checks are the 2-ply leaf's checks with the sides
    # swapped (the side to move is the opponent): an opponent
    # winning move is a loss for player, our double threat a win.
    # NOTE: the opponent-winning-move check below is DEAD CODE
    # under the current rollout policy — ply 3 blocks a single
    # opponent threat (and a double threat ends the rollout at
    # ply 3), and our ply-3 stone cannot create an opponent
    # winning move. It stays because it is in the pre-registered
    # spec (iter-53 §6.1 step 5), mirrors the 2-ply leaf's
    # rollout-end checks, and guards against future policy
    # changes; tests/test_regression_v53.py asserts the
    # unreachability property on every sampled rollout-end
    # position.
    if scoring._winning_moves(b3, -player):
        return -1.0
    if len(scoring._winning_moves(b3, player)) >= 2:
        return 1.0
    # Gradient from the side-to-move's (opponent's) perspective,
    # negated to player's perspective. Equals
    # _attack_gradient_v1(b3, player) by antisymmetry; the
    # explicit negated form keeps the parity handling visible.
    return -_attack_gradient_v1(b3, -player)


def _rollout2_leaf_value(board: np.ndarray, player: int) -> float:
    """v37 2-ply label: v18's rollout-leaf value evaluated 2 plies
    ahead under the teacher's own rollout policy (ML batch 3 H2 —
    see ``research/iterations/iter-37.md`` §1).

    The teacher is v18's rollout leaf — the SAME teacher as v24,
    for comparability. The label for a quiet position (``board``,
    ``player`` to move) is:

    * ply 1 (teacher's move): ``m1 = _rollout_move(board, player)``
      — the teacher's own move policy (win if possible, block a
      single opponent winning move, else the v1-argmax with
      open-four priority). If the ply is terminal (``term1`` is
      not None), the label is ``term1`` — from ``player``'s
      perspective, as in :func:`_rollout_leaf_value` at ply 0.
    * ply 2 (opponent's reply): ``m2 = _rollout_move(board+m1,
      -player)``. If terminal, the label is ``-term2`` (``term2``
      is from the opponent's perspective; negated to ``player``'s,
      as in :func:`_rollout_leaf_value` at ply 1).
    * otherwise the label is ``_rollout_leaf_value(board+m1+m2,
      player)`` — ``player`` to move again at the resulting
      position, blend=False (v24's setting).

    The label is in [-1, 1] from the side-to-move perspective,
    exactly like v24's. The function is defined on quiet positions
    (the labelling pipeline only calls it where
    ``_terminal_leaf_value == 0.0``); on non-quiet positions it
    still answers, with the rollout's own convention: a winning
    move or a double opponent threat returns the exact terminal
    (±1), while a must-block (-0.25) plays the forced block and
    continues — the same treatment :func:`_rollout_leaf_value`
    gives must-blocks found INSIDE its rollout (iter-16 §2.1:
    only the must-block at the START of a leaf evaluation is an
    exact -0.25; interior ones are played through). The board is
    not mutated (the rollout plays on copies via
    :func:`gomoku_ml.game.apply_move`).

    Effective horizon: 2 plies of rollout-policy play + the 2
    plies inside :func:`_rollout_leaf_value` = 4 plies, vs v24's
    2. The move-selection rule is deliberately the teacher's own
    policy rather than a leaf-value argmax: maximising the noisy
    leaf over ~40 candidates would inject optimizer's-curse bias
    into the LABELS, and the rollout policy already plays the
    setup move at the motivating position (it plays (9,6) at
    setup-v1v0s-g92's root, and the unique refuting reply (4,3)
    after the wrong (5,4) — iter-37 §0). A minimax reply was
    measured and rejected: ~1.5 s/position (50x the cost) with NO
    better discrimination at setup-v1v0s-g92 (both children
    +1.000 either way — (5,4) does not lose the win within the
    teacher's horizon, it only wins slower)."""
    m1, term1 = _rollout_move(board, player)
    if term1 is not None:
        return term1
    b1 = game.apply_move(board, m1, player)
    m2, term2 = _rollout_move(b1, -player)
    if term2 is not None:
        return -term2
    b2 = game.apply_move(b1, m2, -player)
    return _rollout_leaf_value(b2, player)


def _rollout3_leaf_value(board: np.ndarray, player: int) -> float:
    """v44 3-ply label: v18's rollout-leaf value evaluated 3 plies
    ahead under the teacher's own rollout policy (ML batch 4 H3 —
    see ``research/iterations/iter-44.md`` §1).

    The teacher is v18's rollout leaf — the SAME teacher as v24
    (1-ply) and v37 (2-ply), for comparability. The label for a
    quiet position (``board``, ``player`` to move) is:

    * ply 1 (teacher's move): ``m1 = _rollout_move(board, player)``.
      If terminal (``term1`` is not None), the label is ``term1``
      — from ``player``'s perspective, as in
      :func:`_rollout_leaf_value` at ply 0.
    * ply 2 (opponent's reply): ``m2 = _rollout_move(board+m1,
      -player)``. If terminal, the label is ``-term2`` (``term2``
      is from the opponent's perspective; negated to ``player``'s,
      as in :func:`_rollout_leaf_value` at ply 1).
    * ply 3 (teacher's move): ``m3 = _rollout_move(board+m1+m2,
      player)``. If terminal, the label is ``term3`` — from
      ``player``'s perspective, as at ply 1.
    * otherwise the label is ``-_rollout_leaf_value(board+m1+m2+m3,
      -player)`` — the OPPONENT to move at the resulting position,
      so the leaf value (from the opponent's perspective) is
      negated to ``player``'s. (This is the one place the 3-ply
      form differs structurally from the 2-ply form, which ends
      with ``player`` to move again and needs no negation.)

    The label is in [-1, 1] from the side-to-move perspective,
    exactly like v24's and v37's. The function is defined on quiet
    positions (the labelling pipeline only calls it where
    ``_terminal_leaf_value == 0.0``); on non-quiet positions it
    still answers, with the rollout's own convention (see
    :func:`_rollout2_leaf_value`). The board is not mutated (the
    rollout plays on copies via :func:`gomoku_ml.game.apply_move`).

    Effective horizon: 3 plies of rollout-policy play + the 2
    plies inside :func:`_rollout_leaf_value` = 5 plies, vs v37's
    4 and v24's 2. The move-selection rule is the teacher's own
    policy for the same reasons as v37's (no optimizer's-curse
    bias in the labels; the minimax reply was measured and
    rejected at 2-ply, ~50x the cost with no better
    discrimination — iter-37 §0)."""
    m1, term1 = _rollout_move(board, player)
    if term1 is not None:
        return term1
    b1 = game.apply_move(board, m1, player)
    m2, term2 = _rollout_move(b1, -player)
    if term2 is not None:
        return -term2
    b2 = game.apply_move(b1, m2, -player)
    m3, term3 = _rollout_move(b2, player)
    if term3 is not None:
        return term3
    b3 = game.apply_move(b2, m3, player)
    return -_rollout_leaf_value(b3, -player)


# --- v52 threat-space search (batch 13, H2) -------------------------------
# Threat-space search (Allis's TSS): search ONLY forcing moves to a
# deeper horizon than the 100-sim MCTS, and play the winning forcing
# sequence if one exists. A genuinely new mechanism, distinct from
# v29's VCF-at-root (which searches only fours — a VCF sequence is a
# TSS sequence in which every attacker move is a four-creator) and
# from the MCTS (which searches all moves). By restricting the move
# set to forcing moves, TSS searches much deeper than the MCTS and
# finds deep forcing wins the MCTS can't see. Pre-registered design:
# research/iterations/iter-52.md §1.
#
# ATTACKER MOVE SET (forcing moves only), enumerated by
# :func:`_forcing_sets` — two kernel calls, one per class, because
# each kernel is blind to the other's patterns (v0's contiguous-run
# scoring is blind to broken fours; v1's window-based open_ends
# misses some true open threes):
# * four-creating moves — v1 prefilter (>= 20000) + a local exact
#   test, verified 0-mismatch against
#   :func:`gomoku_ml.evaluate.four_creating_moves` in test_tss.py;
# * true-open-three-creating moves — v0 attack >= 1000, minus
#   four-creators, verified against a brute-force run-scan oracle
#   in test_tss.py. Gap-three creators are
#   EXCLUDED (pre-registered): a gap three is killed by occupying
#   the gap, so it rarely forces; including them would multiply the
#   branching for near-zero win rate.
# Double threats are NOT a separate move class: a move creating >= 2
# threats is in the union (each threat is a four or an open three);
# the search detects them by counting threats after the move.
#
# DEFENDER REPLIES (exact — every reply must lose for the attacker
# to win):
# * after a four (exactly 1 winning square): the block is the ONLY
#   reply (anything else loses at once);
# * after open three(s): replies are the squares blocking ALL of the
#   attacker's open threes (recomputed from the resulting board —
#   the batch-11 non-monotonicity lesson: full re-check after every
#   move) UNION the defender's four-creating moves (counter-threats:
#   a counter-four forces the attacker to block instead of
#   converting). If the reply set is empty, the attacker wins
#   (double threat, no defence). Defender open-three creators are
#   NOT replies: they do not stop the attacker converting the
#   standing threat next move (open three -> open four -> win), so
#   they cannot refute a line.
# * after a defender counter-four: the attacker's reply is the forced
#   block (>= 2 defender winning moves refutes the line).
# An immediate winning move IS a forcing win and is checked first
# (standing fact: it is a VCF start but not necessarily a
# four-creator — the fast test checks winning moves first).
#
# DEPTH: TSS_DEPTH attacker forcing moves (pre-registered). Forced
# blocks after defender counter-fours do not consume depth (they are
# replies, not forcing moves) — so the NODE BUDGET is the termination
# guarantee for counter-four loops (each iteration adds 2 stones, but
# the budget caps the total work deterministically). On budget
# exhaustion the search aborts and returns None (fall through to the
# MCTS) — never a partial answer.
#
# LIMITATION (pre-registered): every attacker move in a sequence must
# be forcing, except forced blocks after defender counter-fours.
# Sequences needing a quiet defensive interlude are out of scope —
# that is what the MCTS is for. TSS is OFFENSIVE only: there is no
# defensive prevention check (VCF-prevention for the v27 family is
# CLOSED — v29/v33/v35/v45); if the opponent holds a forcing win,
# the MCTS handles it as v27 does.

_TSS_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _build_tss_windows() -> list[tuple[int, ...]]:
    """All 484 5-windows of the board as flat-index tuples (v59,
    iter-59 §1.1 L3): the window-scan forcing enumeration reads
    windows off this table instead of calling the v1/v0 kernels."""
    windows: list[tuple[int, ...]] = []
    for dy, dx, ys, xs in (
        (0, 1, range(15), range(11)),
        (1, 0, range(11), range(15)),
        (1, 1, range(11), range(11)),
        (1, -1, range(11), range(4, 15)),
    ):
        for y in ys:
            for x in xs:
                windows.append(
                    tuple((y + i * dy) * game.SIZE + (x + i * dx) for i in range(5))
                )
    return windows


_TSS_WINDOWS = _build_tss_windows()

# Max attacker forcing moves in a TSS sequence (pre-registered,
# iter-52 §1). VCF uses depth 3 (after the first move); TSS at 4
# attacker moves with open-three threats reaches ~7-8 plies,
# genuinely beyond the MCTS + 2-ply-rollout horizon (~4-6 plies).
TSS_DEPTH = 4

# Max TSS nodes per root call (pre-registered, iter-52 §1.4 — set
# from the cost bench: the creator-rich p99 node count is 2763, so
# 3000 binds only on ~1% of positions (the extreme tail, max 12916
# nodes) and caps the worst-case move time at ~6s uncontended
# (~10s contended) instead of the observed 53s max. The plan's
# p99-time-under-1s criterion is NOT met (p99 time is ~5s
# uncontended — the per-node cost is ~2ms, dominated by kernel
# calls): a budget low enough for p99 < 1s (~500 nodes) would abort
# a quarter of all searches and gut the mechanism, so the budget
# bounds the tail instead of the median. Deterministic (a node
# count, not a wall clock), so resumed runs reproduce exactly.
TSS_NODE_BUDGET = 3000


class _TSSBudgetExceeded(Exception):
    """Internal: the TSS node budget is exhausted — abort the search."""


class _TSSBudget:
    """Deterministic node budget for one TSS root call."""

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.used = 0

    def tick(self) -> None:
        self.used += 1
        if self.used > self.limit:
            raise _TSSBudgetExceeded


def _winning_moves_through(
    board: np.ndarray, player: int, move: int
) -> list[int]:
    """Winning moves for ``player`` on the 4 lines through ``move``
    (within distance 4 — a winning move completing a five that
    includes ``move`` is one of the five's cells). The local version
    of :func:`gomoku_ml.scoring._winning_moves`: TSS uses it wherever
    the winning moves must involve the just-played stone (no
    pre-existing winning move — every caller establishes that), which
    is ~3x cheaper than the full near-empty scan (and skips the
    ``_near_empty`` call entirely). Verified against the full scan in
    ``ml/tests/test_tss.py``."""
    out: list[int] = []
    y, x = move // game.SIZE, move % game.SIZE
    for dy, dx in _TSS_DIRECTIONS:
        for k in range(-4, 5):
            yy, xx = y + k * dy, x + k * dx
            if (
                0 <= yy < game.SIZE
                and 0 <= xx < game.SIZE
                and board[yy, xx] == game.EMPTY
            ):
                m = yy * game.SIZE + xx
                if game.is_winning_move(board, m, player):
                    out.append(m)
    return out


def _winning_moves_through_run(
    flat: list[int], move: int, player: int
) -> list[int]:
    """Window-based version of :func:`_winning_moves_through` (v59,
    iter-59 §1.1 L4): the winning moves for ``player`` involving the
    stone at ``move``. ``flat`` is the board as a flat list of 225
    ints; the cell at ``move`` is TREATED as a ``player`` stone
    (every caller passes the just-played stone, or a candidate move
    being tested as-if-played — the loop counts ``move`` as a stone
    without reading ``flat[move]``). For each of the 5 windows
    through ``move`` in each of the 4 directions, a window holding
    exactly 4 ``player`` stones (``move`` included) and 1 empty cell
    contributes that empty cell — playing it completes a five. This
    covers gap-fill winning moves ([X,X,_,X,X] — the five's stones
    need not be contiguous with ``move``), which a run-extension
    test misses. Returns the same SET as
    :func:`_winning_moves_through` whenever no winning move
    pre-existed the stone at ``move`` (the TSS precondition — a
    winning move whose five does not include ``move`` would have
    pre-existed). Sorted, for a canonical order (call sites use
    only the length and the single element, so order is
    immaterial). ~10x cheaper than the scan version (no
    ``is_winning_move`` calls)."""
    out: set[int] = set()
    y, x = move // game.SIZE, move % game.SIZE
    for dy, dx in _TSS_DIRECTIONS:
        for i in range(5):
            # The window with `move` at position i: it starts at
            # move - i*d and runs 5 cells along d.
            sy, sx = y - i * dy, x - i * dx
            ey, ex = sy + 4 * dy, sx + 4 * dx
            if not (
                0 <= sy < game.SIZE
                and 0 <= sx < game.SIZE
                and 0 <= ey < game.SIZE
                and 0 <= ex < game.SIZE
            ):
                continue
            stones = 0
            empty = -1
            for k in range(5):
                c = (sy + k * dy) * game.SIZE + (sx + k * dx)
                if c == move or flat[c] == player:
                    stones += 1
                elif flat[c] == game.EMPTY:
                    empty = c
            if stones == 4 and empty != -1:
                out.add(empty)
    return sorted(out)


def _creates_four_run(flat: list[int], move: int, player: int) -> bool:
    """Boolean form of :func:`_winning_moves_through_run` (v59,
    iter-59 §1.1 L3/L4): does ``player`` playing ``move`` (treated
    as-if-played) leave them with a winning move? True iff some
    window through ``move`` holds exactly 4 ``player`` stones
    (``move`` included) and 1 empty cell. Early-exits on the first
    such window. Same precondition as
    :func:`_winning_moves_through_run` (no pre-existing winning
    move)."""
    y, x = move // game.SIZE, move % game.SIZE
    for dy, dx in _TSS_DIRECTIONS:
        for i in range(5):
            sy, sx = y - i * dy, x - i * dx
            ey, ex = sy + 4 * dy, sx + 4 * dx
            if not (
                0 <= sy < game.SIZE
                and 0 <= sx < game.SIZE
                and 0 <= ey < game.SIZE
                and 0 <= ex < game.SIZE
            ):
                continue
            stones = 0
            empty = False
            for k in range(5):
                c = (sy + k * dy) * game.SIZE + (sx + k * dx)
                if c == move or flat[c] == player:
                    stones += 1
                elif flat[c] == game.EMPTY:
                    empty = True
            if stones == 4 and empty:
                return True
    return False


def _forcing_sets(
    board: np.ndarray, player: int
) -> tuple[set[int], list[int], np.ndarray]:
    """The TSS forcing-move enumeration: ``(fours, threes, a1)`` —
    the four-creator set, the true-open-three-creator list, and the
    flat v1 attack array (for move ordering in :func:`_tss_win`).
    Two kernel calls, one per class — each kernel is blind to the
    other's patterns, so neither alone suffices:

    * fours via the V1 kernel: a four-creator scores v1 attack >=
      20000 in the direction of the four (a window with 4 friendly
      stones scores FOUR or OPEN_FOUR — see
      :func:`gomoku_ml.scoring._score_from_counts`), INCLUDING
      broken-four creators (XX_XX), which the v0 kernel's
      contiguous-run scoring is BLIND to (a broken-four creator's
      v0 score is ~100-200 — the run is split by the gap). The
      prefilter can only ADD non-creators (e.g. double-open-three
      creators score 58000) — it cannot drop a creator — so the
      local exact test (:func:`_winning_moves_through` — the
      winning moves after the move must involve it) makes the set
      exact.
    * threes via the V0 kernel: a move's v0 attack score is >= 1000
      iff it creates a four, a true open three, or a win (the (3,2)
      weight is 1000; no combination of sub-1000 per-direction
      values reaches 1000 — the largest is 100; see
      :data:`gomoku_ml.baseline._WEIGHTS`). Minus the four-creators
      (exact set above), the >= 1000 cells are exactly the
      open-three creators. The v0 kernel — NOT v1 — is the right
      instrument here: v1's window-based ``open_ends`` counts the
      cells just OUTSIDE the 5-window, so v1 scores a genuine open
      three as a mere THREE (400) when the cells two out from the
      run are occupied (found: an open-three creator with v1 attack
      405, v0 1021).

    Precondition: ``player`` holds NO winning move on ``board``
    (every TSS caller checks first — a winning move is returned
    before any enumeration). Without it the local test would miss
    pre-existing winning moves.
    """
    black1, white1 = kernel_v1.attack_v1_both(board)
    a1 = black1 if player == game.BLACK else white1
    empty = board == game.EMPTY
    fours: set[int] = set()
    for m in np.flatnonzero(empty & (a1 >= scoring.FOUR)):
        m = int(m)
        b2 = game.apply_move(board, m, player)
        if _winning_moves_through(b2, player, m):
            fours.add(m)
    a0 = kernel_numpy.attack_all(board, player)
    threes = [
        int(m)
        for m in np.flatnonzero(empty & (a0 >= _GRADIENT_STRONG))
        if int(m) not in fours
    ]
    return fours, threes, a1.reshape(-1)


def _forcing_sets_fast(
    board: np.ndarray, player: int, need_ordering: bool
) -> tuple[set[int], list[int], dict[int, int]]:
    """The v59 fast forcing-move enumeration (iter-59 §1.1 L3/L5/L6):
    ``(fours, threes, scores)`` — the same sets
    :func:`_forcing_sets` returns, computed by ONE pass over all
    5-windows (``_TSS_WINDOWS``) instead of the v1 + v0 kernel
    calls, plus scalar per-move v1 attack scores (``scores``, only
    when ``need_ordering``) for the move ordering in
    :func:`_tss_win`.

    * fours: empties in windows with >= 3 ``player`` stones (the
      batch-10 filter — every four-creator is in this set, broken
      fours included, since the filter counts stones in the window
      rather than contiguous runs), exact-tested by
      :func:`_creates_four_run` on the flat board (no
      ``apply_move`` needed — the test treats the candidate as
      played).
    * threes: open-three creators from the window pattern — a
      window with empties at both ends (cells 0 and 4) and exactly
      2 ``player`` stones among the middle three cells, whose empty
      middle cell creates _XXX_ exactly — UNION the winning moves
      (windows with 4 stones + 1 empty), minus the fours. The
      winning-move component replicates the current v0>=1000
      enumeration exactly (v0 scores a winning move >= 100_000);
      under the TSS precondition (no winning move on ``board``)
      it is vacuous.
    * scores: ``scoring._score_parts_flat`` attack values for the
      forcing moves only, on the flat list built once here. These
      equal ``attack_v1_all`` exactly (kernel_v1 docstring), so
      the ``_tss_win`` ordering — and therefore the node
      exploration order and node counts — are identical to the
      kernel path by construction. Scores are computed only for
      members of a class with >= 2 members (a singleton class has
      no order to determine); absent entries read as 0 via
      ``dict.get`` at the sort site.

    Kernel blindness is sidestepped by construction: no kernel is
    consulted, so the v0 blind spot (broken-four creators) and the
    v1 blind spot (open threes whose two-out neighbourhood is
    occupied) cannot occur. Verified against the same oracles as
    the kernel version in ``ml/tests/test_v59.py``.

    Precondition: ``player`` holds NO winning move on ``board``
    (same as :func:`_forcing_sets` — every TSS caller checks
    first). Without it the four-creator test would miss
    pre-existing winning moves (as would the kernel version's
    local test).
    """
    flat = board.reshape(-1).tolist()
    four_cands: set[int] = set()
    three_cells: set[int] = set()
    win_cells: set[int] = set()
    for cells in _TSS_WINDOWS:
        stones = 0
        empties: list[int] = []
        for c in cells:
            v = flat[c]
            if v == player:
                stones += 1
            elif v == game.EMPTY:
                empties.append(c)
        if stones >= 3:
            four_cands.update(empties)
        if stones == 4 and len(empties) == 1:
            win_cells.add(empties[0])
        elif (
            stones == 2
            and len(empties) == 3
            and flat[cells[0]] == game.EMPTY
            and flat[cells[4]] == game.EMPTY
        ):
            # Two stones among the middle three cells, both ends
            # empty: the empty middle cell creates _XXX_ exactly.
            for c in cells[1:4]:
                if flat[c] == game.EMPTY:
                    three_cells.add(c)
    fours = {m for m in four_cands if _creates_four_run(flat, m, player)}
    threes = sorted((three_cells | win_cells) - fours)
    scores: dict[int, int] = {}
    if need_ordering:
        if len(fours) >= 2:
            for m in fours:
                scores[m] = scoring._score_parts_flat(flat, m, player)[0]
        if len(threes) >= 2:
            for m in threes:
                scores[m] = scoring._score_parts_flat(flat, m, player)[0]
    return fours, threes, scores


def _four_creating_moves_fast(
    board: np.ndarray, player: int, fast: bool = False
) -> list[int]:
    """Moves by which ``player`` CREATES a four (after the move they
    have a winning move, and they had none before). A v1 kernel
    prefilter (>= 20000 — catches broken fours, which v0 is blind
    to) plus a local exact test per candidate (see
    :func:`_forcing_sets`). Verified 0-mismatch against
    :func:`gomoku_ml.evaluate.four_creating_moves` in
    ``ml/tests/test_tss.py``.

    With ``fast=True`` (v59, iter-59 §1.1 L3/L5) the window-scan
    enumeration (:func:`_forcing_sets_fast`, fours only — the
    threes work and the ordering scores are skipped) is used
    instead; the result is identical (verified in
    ``ml/tests/test_v59.py``).

    Contract note vs :func:`gomoku_ml.evaluate.four_creating_moves`:
    that function returns every near move when ``player`` ALREADY
    holds a winning move (its test is "a winning move exists after
    the move", which a pre-existing four satisfies for any move).
    This function returns only genuine creators. TSS never
    enumerates in the pre-existing case — every caller checks for a
    winning move first — so the two agree everywhere TSS looks."""
    if fast:
        fours, _threes, _scores = _forcing_sets_fast(
            board, player, need_ordering=False
        )
        return sorted(fours)
    fours, _threes, _a1 = _forcing_sets(board, player)
    return sorted(fours)


def open_three_creating_moves(
    board: np.ndarray, player: int, fast: bool = False
) -> list[int]:
    """Moves creating a TRUE open three (_XXX_, contiguous, both ends
    empty) for ``player``. The v0 kernel's >= 1000 cells minus the
    four-creators (see :func:`_forcing_sets`). Verified against a
    brute-force run-scan oracle in ``ml/tests/test_tss.py``.

    With ``fast=True`` (v59) the window-scan enumeration
    (:func:`_forcing_sets_fast`) is used instead; the result is
    identical (verified in ``ml/tests/test_v59.py``).

    The v0 kernel — NOT v1 — is the right instrument here, and the
    oracle test caught why: v1's window-based ``open_ends`` counts
    the cells just OUTSIDE the 5-window, so v1 scores a genuine open
    three as a mere THREE (400) when the cells two out from the run
    are occupied (found: an open-three creator with v1 attack 405,
    v0 1021). v1 is blind to such threes (a pre-existing v1 property
    — its priors undervalue those moves); v0's run-based scoring sees
    the run directly."""
    if fast:
        _fours, threes, _scores = _forcing_sets_fast(
            board, player, need_ordering=False
        )
        return sorted(threes)
    _fours, threes, _a1 = _forcing_sets(board, player)
    return sorted(threes)


def _all_open_threes(board: np.ndarray, player: int) -> list[frozenset[int]]:
    """Every true open three ``player`` holds on ``board``, each as
    the frozenset of its two open-end squares. A maximal contiguous
    run of exactly 3 friendly stones with both flanking cells
    on-board and empty. Each run is found once (from its start
    cell)."""
    threes: list[frozenset[int]] = []
    for y, x in zip(*((board == player).nonzero())):
        y, x = int(y), int(x)
        for dy, dx in _TSS_DIRECTIONS:
            py, px = y - dy, x - dx
            if (
                0 <= py < game.SIZE
                and 0 <= px < game.SIZE
                and board[py, px] == player
            ):
                continue  # not the run start
            length = 1
            ey, ex = y + dy, x + dx
            while (
                0 <= ey < game.SIZE
                and 0 <= ex < game.SIZE
                and board[ey, ex] == player
            ):
                length += 1
                ey += dy
                ex += dx
            if length != 3:
                continue
            # (ey, ex) is the cell just past the run's far end;
            # (py, px) the cell just before its start. Both must be
            # on-board and empty for the three to be open.
            if not (0 <= py < game.SIZE and 0 <= px < game.SIZE):
                continue
            if not (0 <= ey < game.SIZE and 0 <= ex < game.SIZE):
                continue
            if board[py, px] != game.EMPTY or board[ey, ex] != game.EMPTY:
                continue
            threes.append(
                frozenset({game.xy_to_move(px, py), game.xy_to_move(ex, ey)})
            )
    return threes


def _forcing_moves_and_fours(
    board: np.ndarray,
    player: int,
    fast: bool = False,
    need_ordering: bool = True,
) -> tuple[list[int], set[int], np.ndarray | dict[int, int]]:
    """The TSS attacker move set (four-creators ∪ true-open-three-
    creators, sorted by flat index) plus the four-creator set and
    the ordering scores (the flat v1 attack array on the kernel
    path, a ``{move: v1 attack}`` dict on the v59 fast path — both
    for move ordering in :func:`_tss_win`). See
    :func:`_forcing_sets` for the enumeration. Gap-three creators
    are excluded by design (see the v52 comment block above).
    ``need_ordering=False`` skips the ordering scores on the fast
    path (callers that re-order by prior, or only test membership,
    do not need them); it is ignored on the kernel path, where the
    array comes free with the kernel call."""
    if fast:
        fours, threes, scores = _forcing_sets_fast(
            board, player, need_ordering=need_ordering
        )
        return sorted(fours | set(threes)), fours, scores
    fours, threes, a1 = _forcing_sets(board, player)
    return sorted(fours | set(threes)), fours, a1


def tss_forcing_moves(
    board: np.ndarray, player: int, fast: bool = False
) -> list[int]:
    """The TSS attacker move set: four-creators ∪ true-open-three-
    creators (sorted by flat index; callers order by prior/score)."""
    moves, _, _ = _forcing_moves_and_fours(
        board, player, fast=fast, need_ordering=False
    )
    return moves


def _tss_after_attacker_move(
    board: np.ndarray,
    attacker: int,
    depth_left: int,
    budget: _TSSBudget,
    last_move: int,
    fast: bool = False,
) -> bool:
    """Defender to move on ``board``; the attacker has just played
    ``last_move`` (a forcing move, or a forced block after a defender
    counter-four) and holds a threat. True if the attacker forces a
    win against ALL defender replies. ``depth_left`` counts further
    attacker forcing moves (the move just played already consumed
    one).

    No defender-winning-move check at entry: it is provably
    impossible (the attacker's move cannot create a defender winning
    move, and every caller establishes the defender has none —
    :func:`_tss_win` checks at entry, and the counter-four sub-case
    below blocks the single defender win it creates). The threats
    check is LOCAL to ``last_move`` (the attacker's winning moves
    must involve it — none pre-existed).

    With ``fast=True`` (v59) the threats checks use the run-based
    :func:`_winning_moves_through_run` and the counter-four
    enumeration uses the fours-only window scan — same sets, same
    result (verified in ``ml/tests/test_v59.py``)."""
    budget.tick()
    if fast:
        flat = board.reshape(-1).tolist()
        threats = _winning_moves_through_run(flat, last_move, attacker)
    else:
        threats = _winning_moves_through(board, attacker, last_move)
    if len(threats) >= 2:
        return True  # open four: unblockable
    if len(threats) == 1:
        # The block is the defender's ONLY reply.
        b3 = game.apply_move(board, threats[0], -attacker)
        over, _winner = game.game_over(b3)
        if over:
            return False  # the block won for the defender (or a draw)
        return _tss_win(b3, attacker, depth_left, budget, threats[0], fast)
    # No immediate winning move: the attacker's threat is one or more
    # open threes (recomputed from the board — the attacker may hold
    # a standing three from an earlier move in the sequence, e.g.
    # after a counter-four exchange).
    threes = _all_open_threes(board, attacker)
    if not threes:
        return False  # no standing threat: the forcing sequence ended
    # Defender replies: squares blocking ALL open threes, plus
    # counter-fours (defender four-creators — a counter-four forces
    # the attacker to block instead of converting). Recomputed on
    # the resulting board (batch-11 lesson: full re-check).
    block_all = threes[0].intersection(*threes[1:])
    replies = block_all | set(
        _four_creating_moves_fast(board, -attacker, fast=fast)
    )
    if not replies:
        return True  # double threat, no defence
    for r in sorted(replies):
        b3 = game.apply_move(board, r, -attacker)
        over, _winner = game.game_over(b3)
        if over:
            return False  # the reply won for the defender (or a draw)
        # The defender's winning moves on b3 must involve the reply
        # (none pre-existed on the board before it) — local check.
        if fast:
            flat3 = b3.reshape(-1).tolist()
            dthreats = _winning_moves_through_run(flat3, r, -attacker)
        else:
            dthreats = _winning_moves_through(b3, -attacker, r)
        if len(dthreats) >= 2:
            return False  # counter open four: the line is refuted
        if len(dthreats) == 1:
            # Counter-four: the attacker MUST block.
            b4 = game.apply_move(b3, dthreats[0], attacker)
            over, winner = game.game_over(b4)
            if over:
                if winner == attacker:
                    continue  # the block itself won — this reply loses
                return False  # draw (board full): no forced win
            # The standing threat persists (or the block created a
            # new one): the defender is still to move under attack.
            if not _tss_after_attacker_move(
                b4, attacker, depth_left, budget, dthreats[0], fast
            ):
                return False
        else:
            # Plain block: the attacker's next move must be forcing.
            if not _tss_win(b3, attacker, depth_left, budget, r, fast):
                return False
    return True


def _tss_win(
    board: np.ndarray,
    attacker: int,
    depth_left: int,
    budget: _TSSBudget,
    last_defender_move: int,
    fast: bool = False,
) -> bool:
    """Attacker to move; the defender has just played
    ``last_defender_move``. True if the attacker forces a win via
    forcing moves only, within ``depth_left`` more attacker forcing
    moves.

    The attacker's entry check is the FULL scan (a safety net — it
    is provably redundant, since the defender's stone cannot create
    an attacker winning move and none pre-existed, but it converts
    any proof error into a correct True rather than a missed win).
    The defender's entry check is LOCAL to ``last_defender_move``
    (the defender's winning moves must involve it — none pre-existed
    on the board before it).

    With ``fast=True`` (v59, iter-59 §1.1 L2) the provably-redundant
    full scan is DROPPED (the proof: both call sites in
    :func:`_tss_after_attacker_move` pass a non-terminal board, the
    attacker held no winning move before the defender's reply, and
    a defender stone cannot create an attacker winning move — any
    hole in the proof shows up as an A/B mismatch, see
    ``ml/tests/test_v59.py``), and the defender's entry check uses
    the run-based :func:`_winning_moves_through_run`."""
    budget.tick()
    if not fast:
        if scoring._winning_moves(board, attacker):
            return True
    if fast:
        flat = board.reshape(-1).tolist()
        opp_wins = _winning_moves_through_run(flat, last_defender_move, -attacker)
    else:
        opp_wins = _winning_moves_through(board, -attacker, last_defender_move)
    if len(opp_wins) >= 2:
        return False  # already lost: two unblockable wins
    if len(opp_wins) == 1:
        # The attacker MUST block (the defender's forced block may
        # itself have counter-attacked — a real case); the block must
        # itself be forcing for the sequence to continue (a quiet
        # block breaks the forcing chain — pre-registered limitation).
        block = opp_wins[0]
        forcing, _, _ = _forcing_moves_and_fours(
            board, attacker, fast=fast, need_ordering=False
        )
        if block not in forcing:
            return False
        b2 = game.apply_move(board, block, attacker)
        return _tss_after_attacker_move(
            b2, attacker, depth_left - 1, budget, block, fast
        )
    if depth_left <= 0:
        return False
    forcing, fours, a1 = _forcing_moves_and_fours(board, attacker, fast=fast)
    if not forcing:
        return False
    # Try four-creators first (most forcing), then by v1 attack score
    # desc (ties: lowest flat index — deterministic). Order affects
    # only speed here (the root's selection rule is the prior order
    # in :func:`tss_winning_move`). On the fast path ``a1`` is the
    # scalar-score dict (singleton classes absent — ``dict.get``
    # reads them as 0, which cannot change the order within a
    # singleton class).
    if fast:
        forcing.sort(key=lambda m: (0 if m in fours else 1, -a1.get(m, 0), m))
    else:
        forcing.sort(key=lambda m: (0 if m in fours else 1, -int(a1[m]), m))
    for m in forcing:
        b2 = game.apply_move(board, m, attacker)
        if _tss_after_attacker_move(
            b2, attacker, depth_left - 1, budget, m, fast
        ):
            return True
    return False


def tss_winning_move(
    board: np.ndarray,
    player: int,
    depth: int = TSS_DEPTH,
    node_budget: int = TSS_NODE_BUDGET,
    stats: dict | None = None,
    fast: bool = False,
) -> int | None:
    """Threat-space search at the root (v52): the first forcing move
    (highest v1 prior first, ties lowest flat index) that starts a
    forced win by consecutive forcing moves, or None. An immediate
    winning move is returned first (it IS a forcing win). On node-
    budget exhaustion returns None (fall through to the MCTS).
    Deterministic (no RNG). ``stats``, when given, is filled with
    ``{"nodes": <used>, "budget_exceeded": <bool>}``.

    With ``fast=True`` (v59, iter-59 §1.1) the search runs the
    speed-optimised path (window-scan enumeration, run-based
    winning-move tests, no redundant entry scan) — the same search,
    the same moves, the same node counts (verified in
    ``ml/tests/test_v59.py``)."""
    wins = scoring._winning_moves(board, player)
    if wins:
        if stats is not None:
            stats.update({"nodes": 0, "budget_exceeded": False})
        return wins[0]
    opp_wins = scoring._winning_moves(board, -player)
    if len(opp_wins) >= 2:
        if stats is not None:
            stats.update({"nodes": 0, "budget_exceeded": False})
        return None  # already lost
    budget = _TSSBudget(node_budget)
    exceeded = False
    try:
        forcing, _, _ = _forcing_moves_and_fours(
            board, player, fast=fast, need_ordering=False
        )
        if len(opp_wins) == 1:
            # Must block: only the block can start a forcing win.
            forcing = [opp_wins[0]] if opp_wins[0] in forcing else []
        if not forcing:
            return None
        priors = _priors(board, player, 0.0)
        forcing.sort(key=lambda m: (-priors.get(m, 0.0), m))
        for m in forcing:
            b2 = game.apply_move(board, m, player)
            if _tss_after_attacker_move(b2, player, depth - 1, budget, m, fast):
                return m
    except _TSSBudgetExceeded:
        exceeded = True
        return None
    finally:
        if stats is not None:
            stats.update({"nodes": budget.used, "budget_exceeded": exceeded})
    return None


# --- v58 threat-potential leaf term (batch 14, H2, re-scoped) -------------
# The diagnosis (research/iterations/iter-58.md §§2-3) found v52-White
# wins ONLY by reaching a forcing position (47/47 wins involve a White
# TSS fire) but reaches one in only 42% of normal-rules games — and
# that TSS detection is complete for White at depth 4 (0 finds in 55
# completed depth-6 searches), so no detection patch exists. The
# re-scoped v58 (iter-58.md §11) steers the MCTS TOWARD forcing
# positions instead of waiting for them: a term in the leaf
# evaluation valuing threat potential (future forcing positions).
#
# The term counts FORCING MOVES — the TSS move set (four-creators
# ∪ true-open-three-creators, enumerated exactly by
# :func:`_forcing_sets`' machinery: the v1 kernel + a local exact
# test for fours, the v0 kernel >= 1000 minus fours for open-three
# creators — each kernel enumerates one class, batch 13's kernel-
# blindness lesson) — held by each side at the rollout-end position,
# and adds ``W * tanh((F(ours) - F(theirs)) / S)`` to the v1
# gradient, with the sum clamped to +/-0.2 (the iter-7 cap: a static
# heuristic must never outweigh the smallest exact terminal value,
# the must-block 0.25). W = 0.05, S = 1.0 (pre-registered, iter-58
# §11): a one-forcing-move advantage shifts the leaf by ~0.038 —
# about 40% of the gradient's typical midgame magnitude — a
# meaningful nudge that cannot dominate the gradient, and the tanh
# saturates at a difference of 2 (holding 2+ more forcing moves
# than the opponent is already "clearly ahead in threat potential").
#
# The term is evaluated at the ROLLOUT END (the same position as
# the gradient), not the leaf start: forcing potential that
# survives the opponent's best reply (the rollout's ply 2) is what
# matters — a forcing move the opponent kills on the reply is not
# potential. It applies ONLY on the quiet rollout-end path:
# terminals reached inside the rollout return exact values, and on
# a rollout-end must-block (the opponent holds exactly 1 winning
# move) the term is SKIPPED and the gradient alone is returned —
# the opponent's forcing count is degenerate when the opponent
# already holds a winning move (the local four-creator test would
# count cells near the pre-existing winning square as creators;
# see :func:`_four_creating_moves_fast`'s contract note), and the
# must-block dominates the position anyway.
#
# Distinction from CLOSED levers (pre-registered, iter-58 §11):
# * v40's leaf-blind direction is closed, but v40 was about the leaf
#   MIS-RANKING on specific prevention instances (it ranks the
#   unsafe move worst already) — this adds a NEW POSITIVE term for
#   threat creation, a different mechanism (the leaf's ranking of
#   preventions is untouched: terminals and the must-block are
#   exact, as in v52).
# * The prior/book levers (v13, v16) are about the PRIOR, not the
#   leaf value — the priors are v52's exactly.
# * The White gap as a single-flaw framing (v41, v46, v58
#   diagnosis) is closed — this term is NOT a White patch: it
#   applies to BOTH colours (the leaf is colour-symmetric), and
#   the mechanism check measures White's firing rate only because
#   that is the diagnosis's quantity.
#
# Cost: the term adds two v0 ``attack_all`` calls (one per colour)
# plus local exact tests for the v1 >= 20000 candidates, reusing
# the gradient's single ``attack_v1_both`` call (the v1 arrays are
# passed into :func:`_forcing_counts_both`). The per-leaf budget
# is pre-registered from the bench in iter-58 §11.4.

# Threat-potential term weight and scale (pre-registered, iter-58
# §11.1): W = 0.05 (half the gradient's cap, a quarter of the
# must-block), S = 1.0 (one forcing-move difference is one scale
# unit — the term saturates at a difference of 2).
_THREAT_TERM_WEIGHT = 0.05
_THREAT_TERM_SCALE = 1.0


def _forcing_counts_both(
    board: np.ndarray, black1: np.ndarray, white1: np.ndarray
) -> tuple[int, int]:
    """Exact forcing-move counts ``(black, white)`` on ``board``:
    the number of four-creators ∪ true-open-three-creators for
    each side — the TSS move set (see :func:`_forcing_sets`).

    The v1 attack arrays (``black1``, ``white1`` — from one
    ``attack_v1_both`` call, shared with the gradient) prefilter
    the four-creators (>= 20000, catches broken fours, which the
    v0 kernel is blind to); each candidate is exact-tested
    locally (:func:`_winning_moves_through`). The open-three
    creators come from the v0 kernel (>= 1000, minus fours — v1
    is blind to some open threes). The two counts share the v1
    call but need one v0 ``attack_all`` call per colour.

    Precondition: NEITHER side holds a winning move on ``board``
    (the only caller, :func:`_gradient_plus_threat`, is reached
    only after the rollout-end terminal checks, and the
    must-block case skips the term — see
    :func:`_rollout_threat_leaf_value`). Without it the local
    four-creator test would count cells near a pre-existing
    winning square as creators. Verified against
    :func:`_forcing_sets` (differential) and the brute-force
    oracles in ``ml/tests/test_threat_term_v58.py``."""
    empty = board == game.EMPTY
    counts = []
    for player, a1 in ((game.BLACK, black1), (game.WHITE, white1)):
        fours: set[int] = set()
        for m in np.flatnonzero(empty & (a1 >= scoring.FOUR)):
            m = int(m)
            b2 = game.apply_move(board, m, player)
            if _winning_moves_through(b2, player, m):
                fours.add(m)
        a0 = kernel_numpy.attack_all(board, player)
        threes = [
            int(m)
            for m in np.flatnonzero(empty & (a0 >= _GRADIENT_STRONG))
            if int(m) not in fours
        ]
        counts.append(len(fours) + len(threes))
    return counts[0], counts[1]


def _threat_potential_term(f_ours: int, f_theirs: int) -> float:
    """The threat-potential term: ``W * tanh((F(ours) - F(theirs)) /
    S)`` where ``F`` is the exact forcing-move count for a side.
    Positive when we hold more forcing moves than the opponent,
    negative when fewer; bounded by +/-W (+/-0.05); antisymmetric
    under colour swap, like the gradient. See the v58 comment
    block above for the pre-registered weight and scale."""
    return _THREAT_TERM_WEIGHT * math.tanh(
        (f_ours - f_theirs) / _THREAT_TERM_SCALE
    )


def _gradient_plus_threat(board: np.ndarray, player: int) -> float:
    """Rollout-end value for the v58 leaf: v14's symmetric v1
    gradient (scale 7.5, cap +/-0.2) PLUS the threat-potential
    term, the sum clamped to +/-0.2 (the iter-7 cap invariant:
    quiet leaves never outweigh the must-block 0.25). One
    ``attack_v1_both`` call serves both the gradient breadth and
    the four-creator prefilter; the forcing counts add two v0
    ``attack_all`` calls (one per colour) plus local exact tests.

    Precondition: neither side holds a winning move on ``board``
    (see :func:`_forcing_counts_both`) — the caller,
    :func:`_rollout_threat_leaf_value`, establishes it via the
    rollout-end terminal checks and skips this function on a
    must-block."""
    black1, white1 = kernel_v1.attack_v1_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours1, theirs1 = black1, white1
    else:
        ours1, theirs1 = white1, black1
    diff = _breadth_v1(ours1, mask) - _breadth_v1(theirs1, mask)
    grad = _GRADIENT_CAP * math.tanh(diff / _GRADIENT_V1_SCALE)
    f_black, f_white = _forcing_counts_both(board, black1, white1)
    if player == game.BLACK:
        f_ours, f_theirs = f_black, f_white
    else:
        f_ours, f_theirs = f_white, f_black
    term = _threat_potential_term(f_ours, f_theirs)
    return max(-_GRADIENT_CAP, min(_GRADIENT_CAP, grad + term))


def _rollout_threat_leaf_value(board: np.ndarray, player: int) -> float:
    """v58 search leaf: v18/v27's 2-ply rollout leaf
    (:func:`_rollout_leaf_value`) with the threat-potential term
    added to the rollout-end gradient. Terminal handling follows
    v16/v18's conventions exactly: the leaf-start terminal
    (including the -0.25 must-block, a fact about the real
    position) is exact; terminals reached INSIDE the rollout
    return the exact terminal value, sign-adjusted to ``player``'s
    perspective; at the rollout's END only wins/losses are
    terminal — a must-block there is gradient-evaluated WITHOUT
    the term (the opponent's forcing count is degenerate when the
    opponent already holds a winning move). No blend (v19
    rejected, not part of v27). The board is not mutated (the
    rollout plays on copies). Written fresh from the pre-
    registered spec in ``research/iterations/iter-58.md`` §11 —
    deliberately NOT a wrapper around :func:`_rollout_leaf_value`,
    so the differential test (``tests/test_threat_term_v58.py``)
    tests the SPEC against v52's leaf rather than an alias
    against itself (the v53 precedent)."""
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
    opp_wins = scoring._winning_moves(cur, -p)
    if len(opp_wins) >= 2:
        return -1.0
    if len(opp_wins) == 1:
        # Must-block at the rollout end: the gradient alone, no
        # term — the opponent's forcing count is degenerate when
        # the opponent already holds a winning move (see
        # _forcing_counts_both's precondition).
        return _attack_gradient_v1(cur, p)
    return _gradient_plus_threat(cur, p)


class HeuristicMCTS:
    """PUCT search with v1 priors and a terminal-aware leaf value.

    ``leaf_mode="terminal"`` (default) is v6: quiet leaves are 0.
    ``leaf_mode="gradient"`` is v7: quiet leaves get the attack
    breadth gradient (:func:`_gradient_leaf_value`); terminal
    values are identical in both modes.

    ``prior_floor`` (v15, batch 6 H1) is an exploration floor on
    the priors, applied at every expansion: priors below the
    floor are raised to it, WITHOUT renormalization (see
    :func:`_priors` and iter-15.md §1.0 for why renormalization
    is excluded). 0.0 (the default) is v6/v7 behaviour exactly.
    ``c_puct`` is the PUCT exploration constant (v6/v7: 1.5);
    v15 sweeps higher values (see iter-15.md §1.2).
    ``leaf_mode="rollout2_v1"`` is v16: when the engine plays White,
    quiet leaves get the 2-ply rollout leaf
    (:func:`_rollout_leaf_value`); when it plays Black, the leaf is
    v7's exactly. The mode is instance-bound (the leaf needs the
    root player's colour, recorded by :meth:`_run_one` on the root
    call).
    ``leaf_mode="gradient_v1"`` is v14: quiet leaves get the attack
    breadth gradient from the v1 kernel's scores
    (:func:`_gradient_v1_leaf_value`); terminal values are
    identical to v7's, so the only difference between v7 and v14
    is the gradient's input scores.
    ``leaf_mode="gradient_v1_gap"`` is v20: v14's leaf with
    gap-three makers demoted from the strong tier to the developing
    tier (:func:`_gradient_v1_gap_leaf_value`); terminal values
    identical, so the only difference between v14 and v20 is the
    three-tier weighting.
    ``leaf_mode="gradient_v1_gap_black"`` is v22: v14's leaf with
    gap-three makers demoted ONLY in Black's breadth
    (:func:`_gradient_v1_gap_black_leaf_value`); terminal values
    identical, so the only difference between v14 and v22 is the
    three-tier weighting of Black's breadth.
    ``leaf_mode="gradient_v1_gap_ours"`` is v28: v14's leaf with
    gap-three makers demoted ONLY in Black's breadth and ONLY in
    the ours term — at Black-to-move leaves Black's breadth is
    demoted (v22's rule), at White-to-move leaves nothing is
    demoted (v14's rule), so White fears Black's gap threes
    fully (:func:`_gradient_v1_gap_ours_leaf_value`); terminal
    values identical. Deliberately not antisymmetric under player
    swap (see the v28 comment block above the leaf function).


    ``leaf_mode="rollout2_v1_both"`` is v18: quiet leaves get the
    2-ply rollout leaf for BOTH colours — the same function v16
    uses as White, with no colour gate. v18-as-White IS v16.

    ``leaf_mode="rollout2ply_v53"`` is the ORIGINAL v53 spec
    (batch 13, H3): a fresh-from-spec 2-ply rollout leaf
    (:func:`_rollout2ply_search_leaf_value`) for both colours.
    The differential test proved it coincides with v18/v27's
    leaf everywhere sampled, so v53-as-specified was NOT A NEW
    VERSION (iter-53 §§0–5); the mode stays as the regression
    record of that finding.
    ``leaf_mode="rollout3ply_v53"`` is the RE-SCOPED v53
    (iter-53 §6): a 3-ply rollout leaf
    (:func:`_rollout3ply_search_leaf_value`) for both colours —
    the only unpriced rung of the depth ladder. After 3 rollout
    plies the opponent is to move at the rollout end, so the
    gradient is negated from the opponent's perspective.

    ``leaf_mode="rollout2_threat_v58"`` is the RE-SCOPED v58
    (iter-58 §11): v18/v27's 2-ply rollout leaf with a
    threat-potential term added to the rollout-end gradient
    (:func:`_rollout_threat_leaf_value`) for both colours — the
    exact forcing-move count difference (four-creators ∪
    true-open-three-creators, both kernels) enters as
    ``0.05 * tanh(diff)``, the sum clamped to +/-0.2. Terminals
    are exact, as in v52; on a rollout-end must-block the term
    is skipped (the opponent's forcing count is degenerate
    there) and the gradient alone is returned.

    ``tie_break_q=True`` is v23 (batch 8, H3): when two or more
    root children share the maximum visit count (an exact tie at
    the top), break the tie by correctly-signed q (our-q =
    ``-child.q`` — ``child.q`` is the OPPONENT's perspective, see
    :meth:`_run_one`'s backup and :meth:`_select_child`). If q
    also ties exactly, fall back to the RNG/first choice on the
    q-tied subset. A unique visit leader is untouched. Default
    False: v18 and all earlier versions are unaffected. See
    ``research/iterations/iter-23.md`` §1.1 for the
    pre-registered rule and why this narrow retest does not reopen
    the CLOSED q-based root-decisions lever.

    ``tie_break_q_threshold`` is v27 (batch 9, H1): the tie-break
    fires ONLY when the q gap — the highest our-q minus the
    second-highest DISTINCT our-q among the visit-tied children —
    strictly exceeds the threshold. Below the threshold the
    tie-break does not fire at all and the choice is v18's (RNG
    over ALL tied children, else first). Default 0.0 reproduces
    v23 exactly (any positive gap fires; a zero gap means all
    tied children share q_max, so v23's rule-4 subset IS all tied
    children). See ``research/iterations/iter-27.md`` §1.1 for
    the pre-registered rule: the threshold separates terminal
    signal (documented gaps >= 0.130) from gradient noise (the
    documented misfire gap is 0.016).

    ``root_decision`` (v9, batch 4 H1; v12, batch 5 H1) selects the
    root move choice: ``"visits"`` (default, v6/v7 behaviour —
    most-visited, ties by RNG), ``"q"``, ``"tie-q"``, ``"blend"``,
    ``"tie-prior"``, ``"tie-prior-forcing"`` — see
    :meth:`_choose_root_child`.
    ``vcf_root`` (v29, batch 9 H3) enables the VCF-at-root check
    (:meth:`_vcf_root_move`): before the search, an offensive VCF
    start is played immediately, and a defensive VCF prevention
    is played when the opponent holds a VCF start. False (the
    default) is v18 behaviour exactly.

    ``backup_rule`` (v51, batch 13 H1) selects the backup rule:
    ``"mean"`` (default) is the v6/v7 mean backup exactly;
    ``"softmax-opp"`` switches opponent nodes (odd depth from the
    root) to a SOFT-MIN backup — q is the softmax-weighted mean
    of the opponent-perspective values of the node's visited
    children (``-child.q``), weights ``exp(beta * value)`` with
    ``beta = softmax_beta`` — while our nodes (even depth,
    including the root) keep the mean (of the soft-min values
    returned by their children). The soft-min weights the
    opponent's worst reply heavily but keeps the other replies'
    contribution, so it prices the opponent's best reply (undoing
    iter-40 §1.1a's blunder dilution) without collapsing the
    "how it loses" gradient the way v47's pure minimax did
    (iter-47 §3.1 — q-collapse: ALL 74 visited children at −1.0
    on g13). ``softmax_beta`` (default 2.0; v51 sets it
    explicitly) is the temperature: at β=2 a 1.0 value gap gets
    weight ratio e² ≈ 7.4. See
    ``research/iterations/iter-51.md`` §1.2 for the pre-registered
    rule and the worst-case analysis against the τ=0.05 tie-break
    threshold.

    ``tss_root`` (v52, batch 13 H2) enables threat-space search at
    the root (:func:`tss_winning_move`): before the search, a
    forcing win by consecutive forcing moves (fours and true open
    threes) is played immediately; otherwise the engine is exactly
    the base version. False (the default) leaves the base engine
    unchanged. Offensive only — no defensive prevention check.
    """

    def __init__(
        self,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
        leaf_mode: str = "terminal",
        prior_floor: float = 0.0,
        root_decision: str = "visits",
        tie_break_q: bool = False,
        tie_break_q_threshold: float = 0.0,
        vcf_root: bool = False,
        backup_rule: str = "mean",
        softmax_beta: float = 2.0,
        tss_root: bool = False,
        tss_fast: bool = False,
    ) -> None:
        if leaf_mode not in _LEAF_FUNCTIONS and (
            leaf_mode != "rollout2_v1"
            and leaf_mode != "rollout2_v1_both"
            and leaf_mode != "rollout2ply_v53"
            and leaf_mode != "rollout3ply_v53"
            and leaf_mode != "rollout2_threat_v58"
        ):
            raise ValueError(
                f"unknown leaf_mode: {leaf_mode!r} "
                f"(known: {sorted(_LEAF_FUNCTIONS) + ['rollout2_v1', 'rollout2_v1_both', 'rollout2ply_v53', 'rollout3ply_v53', 'rollout2_threat_v58']})"
            )
        if not 0.0 <= prior_floor < 1.0:
            raise ValueError(
                f"prior_floor must be in [0, 1), got {prior_floor!r}"
            )
        if root_decision not in _ROOT_DECISIONS:
            raise ValueError(
                f"unknown root_decision: {root_decision!r} "
                f"(known: {list(_ROOT_DECISIONS)})"
            )
        if backup_rule not in ("mean", "softmax-opp"):
            raise ValueError(
                f"unknown backup_rule: {backup_rule!r} "
                "(known: 'mean', 'softmax-opp')"
            )
        if not (softmax_beta > 0.0):
            raise ValueError(
                f"softmax_beta must be positive, got {softmax_beta!r}"
            )
        self.sims = sims
        self.c_puct = c_puct
        self.rng = rng
        self.leaf_mode = leaf_mode
        self.prior_floor = prior_floor
        self.root_decision = root_decision
        self.tie_break_q = tie_break_q
        self.tie_break_q_threshold = tie_break_q_threshold
        self.vcf_root = vcf_root
        self.backup_rule = backup_rule
        self.softmax_beta = softmax_beta
        self.tss_root = tss_root
        # v59: run the TSS on the speed-optimised path (window-scan
        # enumeration, run-based winning-move tests — same search,
        # same moves, same node counts; see tss_winning_move).
        self.tss_fast = tss_fast
        # Rollout modes are NOT in _LEAF_FUNCTIONS (they are
        # instance-bound methods): the dispatch MUST use this
        # conditional form — the naive ``_LEAF_FUNCTIONS[leaf_mode]``
        # raises KeyError on "rollout2_v1" / "rollout2_v1_both"
        # (batch 6 caught this as a silent merge breakage).
        if leaf_mode == "rollout2_v1":
            self._leaf_value = self._rollout_v1_leaf_value
        elif leaf_mode == "rollout2_v1_both":
            self._leaf_value = self._rollout_v1_both_leaf_value
        elif leaf_mode == "rollout2ply_v53":
            self._leaf_value = self._rollout2ply_v53_leaf_value
        elif leaf_mode == "rollout3ply_v53":
            self._leaf_value = self._rollout3ply_v53_leaf_value
        elif leaf_mode == "rollout2_threat_v58":
            self._leaf_value = self._rollout_threat_v58_leaf_value
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

    def _rollout_v1_both_leaf_value(
        self, board: np.ndarray, player: int
    ) -> float:
        """v18 leaf value: the 2-ply rollout leaf
        (:func:`_rollout_leaf_value`) for BOTH colours — no colour
        gate. The rollout function, policy, gradient and both
        fast-loop patches are v16's, byte-identical; only the
        dispatch changed (v16 gates on ``_root_player``, v18 does
        not). v18-as-White IS v16 exactly."""
        return _rollout_leaf_value(board, player)

    def _rollout2ply_v53_leaf_value(
        self, board: np.ndarray, player: int
    ) -> float:
        """v53 leaf value: the fresh-from-spec 2-ply rollout leaf
        (:func:`_rollout2ply_search_leaf_value`) for BOTH colours.
        See ``research/iterations/iter-53.md`` §1.1. The
        differential test (``tests/test_regression_v53.py``) is
        the verdict on whether this coincides with v18/v27's
        ``_rollout_leaf_value``."""
        return _rollout2ply_search_leaf_value(board, player)

    def _rollout3ply_v53_leaf_value(
        self, board: np.ndarray, player: int
    ) -> float:
        """v53 (re-scoped) leaf value: the 3-ply rollout leaf
        (:func:`_rollout3ply_search_leaf_value`) for BOTH colours.
        See ``research/iterations/iter-53.md`` §6.1. After 3
        rollout plies the opponent is to move at the rollout end,
        so the gradient is negated from the opponent's
        perspective."""
        return _rollout3ply_search_leaf_value(board, player)

    def _rollout_threat_v58_leaf_value(
        self, board: np.ndarray, player: int
    ) -> float:
        """v58 (re-scoped) leaf value: the 2-ply rollout leaf with
        the threat-potential term added to the rollout-end
        gradient (:func:`_rollout_threat_leaf_value`) for BOTH
        colours. See ``research/iterations/iter-58.md`` §11."""
        return _rollout_threat_leaf_value(board, player)

    def _vcf_root_move(
        self, board: np.ndarray, player: int
    ) -> int | None:
        """VCF-at-root check (v29, batch 9 H3 — see
        ``research/iterations/iter-29.md`` §1.2). Returns the move
        to play, or None to fall through to the MCTS search.

        1. **Offensive VCF.** Among our four-creating moves, any
           that starts a forced win (``evaluate.starts_forced_win``,
           depth 3 — the tactical suite's own VCF definition) is
           played immediately, highest v1 prior first (ties: lowest
           flat move index — deterministic, no RNG).
        2. **Defensive VCF prevention.** If any opponent
           four-creating move starts a forced win for the opponent,
           play the highest-prior candidate move after which the
           opponent has NO VCF-starting move (re-checked on the
           resulting board); if none exists, fall through.
        The depth-3 VCF definition is exact, not heuristic — the
        check fires only on a genuinely forced win (by consecutive
        fours). Note the forensics caveat (iter-29 §2.3):
        ``starts_forced_win`` does not model defender multi-move
        counter-threats, so a reported VCF start can fail to
        convert in real play — the battery is the verdict on
        whether the check helps or misfires.

        Implementation note (NOT a design change): both steps test
        moves in (prior desc, flat index asc) order and return the
        FIRST success, which is exactly the pre-registered
        selection rule (highest prior, ties lowest index). The
        defensive step additionally tests only the opponent's
        ORIGINAL four-creators on each resulting board (our move
        can only remove the opponent's four-creators — it adds
        our stone, never an opponent stone — so the four-creators
        after our move are a subset of the original set, and
        ``starts_forced_win`` returns False for a move that no
        longer creates a four there), VCF starts first (a VCF
        start on the original board usually still forces a win
        after our move, so ``any`` short-circuits on the first
        test for non-preventions). Both are pure reordering /
        subset reasoning — the selected move is identical to
        testing every four-creator on every resulting board
        (verified: identical moves on all 7 smoke-test positions
        and all 14 White turns of fast-loop game 1 vs v0; the
        only exception is the already-lost 2+-opponent-wins case,
        where the game is lost whatever the check returns). The
        naive test-all implementation took **38.7s** for one
        defensive check on hard-suite position prev-v1v1s-g79
        (profiled: 51,109 ``winning_moves`` calls via the
        ``forces_win`` recursion); the early exit + subset +
        ordering is 8–13x faster on the three prevention
        positions (g79: 38.68s → 4.98s; g26: → 2.01s; v1bl-g2:
        → 2.21s) with IDENTICAL moves. These change only how
        many moves are tested and in what order, never which move
        is selected.
        """
        # 1. Offensive VCF — test in selection order, first VCF
        # start wins (identical to testing all and taking the max).
        ours = evaluate.four_creating_moves(board, player)
        if ours:
            priors = _priors(board, player, self.prior_floor)
            ours.sort(key=lambda m: (-priors.get(m, 0.0), m))
            for m in ours:
                if evaluate.starts_forced_win(board, player, m, depth=3):
                    return m
        # 2. Defensive VCF prevention — only when the opponent
        # holds a VCF start on the current board (the gate that
        # keeps this cheap in quiet positions: the per-candidate
        # loop below runs only under a live VCF threat).
        opp = evaluate.four_creating_moves(board, -player)
        opp_vcf = [
            m
            for m in opp
            if evaluate.starts_forced_win(board, -player, m, depth=3)
        ]
        if not opp_vcf:
            return None
        priors = _priors(board, player, self.prior_floor)
        cands = sorted(
            scoring._candidate_moves(board),
            key=lambda m: (-priors.get(m, 0.0), m),
        )
        # Test only the opponent's ORIGINAL four-creators (subset
        # property, see the docstring), VCF starts first.
        vcf_set = set(opp_vcf)
        ordered = opp_vcf + [m for m in opp if m not in vcf_set]
        for c in cands:
            b2 = game.apply_move(board, c, player)
            if not any(
                evaluate.starts_forced_win(b2, -player, m, depth=3)
                for m in ordered
            ):
                return c
        return None

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
        # VCF-at-root (v29): forced wins the 100-sim search misses.
        if self.vcf_root:
            vcf_move = self._vcf_root_move(board, player)
            if vcf_move is not None:
                return vcf_move
        # Threat-space search at the root (v52): a forcing win by
        # consecutive forcing moves (fours and true open threes)
        # that the 100-sim search misses. Offensive only; the
        # win/block scans above already ran, so an immediate win or
        # a forced block always takes precedence over the TSS.
        if self.tss_root:
            tss_move = tss_winning_move(board, player, fast=self.tss_fast)
            if tss_move is not None:
                return tss_move
        root = _Node(1.0)
        for _ in range(self.sims):
            self._run_one(root, board, player, None)
        return self._choose_root_child(root, board, player)

    def _rng_or_first(self, moves: list[int]) -> int:
        """Break a final tie: the seeded RNG's choice, or the first
        move when there is no RNG (v6/v7 behaviour)."""
        if len(moves) == 1:
            return moves[0]
        if self.rng is not None:
            return int(self.rng.choice(moves))
        return moves[0]

    def _pick_root_move(self, root: _Node) -> int:
        """Final root selection: the most-visited child. On an exact
        top-visit tie, v23 (``tie_break_q=True``) breaks it by
        correctly-signed q (our-q = ``-child.q`` — ``child.q`` is
        the OPPONENT's perspective; see :meth:`_run_one`'s backup
        and :meth:`_select_child`); if q also ties exactly, fall
        back to v18's behaviour (RNG choice, else first) restricted
        to the q-tied subset. v27 gates that rule on the q gap:
        the tie-break fires ONLY when the highest our-q exceeds
        the second-highest DISTINCT our-q among the tied children
        by strictly more than ``tie_break_q_threshold``; below the
        threshold the tie-break does not fire at all and the choice
        is v18's (RNG over ALL tied children, else first). With
        ``tie_break_q=False`` (v18 and earlier) a tie is broken by
        RNG choice among ALL tied children, else first. A unique
        visit leader is untouched either way. Pre-registered rules:
        iter-23.md §1.1, iter-27.md §1.1."""
        best_visits = max(c.visit_count for c in root.children.values())
        best = [
            m for m, c in root.children.items() if c.visit_count == best_visits
        ]
        if len(best) == 1:
            return best[0]
        if self.tie_break_q:
            our_q = {m: -root.children[m].q for m in best}
            q_max = max(our_q.values())
            distinct = sorted(set(our_q.values()), reverse=True)
            q_second = distinct[1] if len(distinct) > 1 else q_max
            if q_max - q_second > self.tie_break_q_threshold:
                best = [m for m in best if our_q[m] == q_max]
                if len(best) == 1:
                    return best[0]
                # Else fall through to RNG/first on the q-tied
                # subset (v23 rule 4).
            # Else (gap <= threshold): fall through with `best`
            # unchanged — v18's RNG/first over ALL tied children.
        if self.rng is not None:
            return int(self.rng.choice(best))
        return best[0]

    def _choose_root_child(
        self,
        root: _Node,
        board: np.ndarray | None = None,
        player: int | None = None,
    ) -> int:
        """Pick the root move by this instance's root-decision rule
        (v9, batch 4 H1; v12, batch 5 H1; see
        ``research/iterations/iter-9.md`` §1 and
        ``research/iterations/iter-12.md`` §1).

        Every rule considers only VISITED children (visit_count >=
        1): an unvisited child's q is undefined (0.0 by convention),
        not a signal, and must never win an argmax over q. And "q"
        throughout is OUR-perspective value, ``-child.q`` — a root
        child's raw ``q`` is the OPPONENT's value under the negamax
        backup (iter-9.md §1.0 sign correction; iter-7 §2.6 read
        raw q as our value and got the finding backwards).

        ``board``/``player`` are needed only by
        ``"tie-prior-forcing"`` (to test forcing-ness); when they
        are absent that rule degrades to plain ``"tie-prior"``.
        ``choose_move`` always passes them.

        * ``"visits"`` (v6/v7): most-visited; ties by RNG/first.
          Unchanged behaviour — with sims >= 1 the max-visit child
          always has >= 1 visit, so the visited filter is a no-op.
        * ``"q"`` (variant A): argmax our-q; ties by visits, then
          RNG/first.
        * ``"tie-q"`` (variant B): most-visited; ties by our-q,
          then RNG/first.
        * ``"blend"`` (variant C): argmax ``our-q + 0.1 *
          (visits / max_visits)``; ties by visits, then RNG/first.
        * ``"tie-prior"`` (variant D, v9): most-visited; ties by
          v1 prior (a monotone transform of the v1 1-ply score),
          then RNG/first.
        * ``"tie-prior-forcing"`` (v12): most-visited; ties by
          forcing-ness first (moves creating a four —
          ``scoring._is_forcing``), then v1 prior, then RNG/first.
          Targets iter-9 §2.6's failure mode: v1's 1-ply score
          undervalues forcing counter-threats (~20k) against
          occupying a creator (~204k), so the prior tie-break
          systematically prefers blocking one of two creators over
          forcing — and with two creators, blocking one is useless.
        """
        visited = [
            (m, c) for m, c in root.children.items() if c.visit_count >= 1
        ]
        if not visited:
            # No simulation ran (sims == 0): fall back to the
            # highest prior. Unused in practice (sims >= 1
            # everywhere); the sane behaviour when there is no
            # search to decide by.
            return max(root.children, key=lambda m: root.children[m].prior)
        if self.root_decision == "q":
            best_q = max(-c.q for _, c in visited)
            cands = [m for m, c in visited if -c.q == best_q]
            best_v = max(root.children[m].visit_count for m in cands)
            cands = [
                m for m in cands if root.children[m].visit_count == best_v
            ]
            return self._rng_or_first(cands)
        if self.root_decision == "blend":
            max_v = max(c.visit_count for _, c in visited)

            def _score(child: _Node) -> float:
                return -child.q + _BLEND_VISIT_WEIGHT * (
                    child.visit_count / max_v
                )

            best_s = max(_score(c) for _, c in visited)
            cands = [m for m, c in visited if _score(c) == best_s]
            best_v = max(root.children[m].visit_count for m in cands)
            cands = [
                m for m in cands if root.children[m].visit_count == best_v
            ]
            return self._rng_or_first(cands)
        # "visits" / "tie-q" / "tie-prior" / "tie-prior-forcing":
        # most-visited first.
        best_v = max(c.visit_count for _, c in visited)
        cands = [m for m, c in visited if c.visit_count == best_v]
        if len(cands) > 1:
            if self.root_decision == "tie-q":
                best_q = max(-root.children[m].q for m in cands)
                cands = [
                    m for m in cands if -root.children[m].q == best_q
                ]
            elif self.root_decision in ("tie-prior", "tie-prior-forcing"):
                if (
                    self.root_decision == "tie-prior-forcing"
                    and board is not None
                    and player is not None
                ):
                    # Forcing-first (v12, iter-12.md §1.1): among
                    # the tied children, prefer moves that create
                    # a four. Only the tied children are tested
                    # (ties are ~24% of searched moves, tie sizes
                    # small), so the scan is cheap. If none is
                    # forcing, the tie-break falls through to the
                    # prior below. Without a board (synthetic-root
                    # tests) the rule degrades to plain tie-prior.
                    forcing = [
                        m
                        for m in cands
                        if scoring._is_forcing(board, m, player)
                    ]
                    if forcing:
                        cands = forcing
                best_p = max(root.children[m].prior for m in cands)
                cands = [
                    m for m in cands
                    if root.children[m].prior == best_p
                ]
        return self._rng_or_first(cands)

    def _choose_root_child(
        self,
        root: _Node,
        board: np.ndarray | None = None,
        player: int | None = None,
    ) -> int:
        """Pick the root move by this instance's root-decision rule
        (v9, batch 4 H1; v12, batch 5 H1; see
        ``research/iterations/iter-9.md`` §1 and
        ``research/iterations/iter-12.md`` §1).

        Every rule considers only VISITED children (visit_count >=
        1): an unvisited child's q is undefined (0.0 by convention),
        not a signal, and must never win an argmax over q. And "q"
        throughout is OUR-perspective value, ``-child.q`` — a root
        child's raw ``q`` is the OPPONENT's value under the negamax
        backup (iter-9.md §1.0 sign correction; iter-7 §2.6 read
        raw q as our value and got the finding backwards).

        ``board``/``player`` are needed only by
        ``"tie-prior-forcing"`` (to test forcing-ness); when they
        are absent that rule degrades to plain ``"tie-prior"``.
        ``choose_move`` always passes them.

        * ``"visits"`` (v6/v7): most-visited; ties by RNG/first.
          Unchanged behaviour — with sims >= 1 the max-visit child
          always has >= 1 visit, so the visited filter is a no-op.
          With ``tie_break_q=True`` (v23/v27 only) exact visit ties
          are broken by our-q first, gated on the q gap by
          ``tie_break_q_threshold`` (see :meth:`_pick_root_move` —
          same rule, same candidate set); the gate is the only
          difference between v23 (threshold 0.0) and v27 (0.05).
        * ``"q"`` (variant A): argmax our-q; ties by visits, then
          RNG/first.
        * ``"tie-q"`` (variant B): most-visited; ties by our-q,
          then RNG/first.
        * ``"blend"`` (variant C): argmax ``our-q + 0.1 *
          (visits / max_visits)``; ties by visits, then RNG/first.
        * ``"tie-prior"`` (variant D, v9): most-visited; ties by
          v1 prior (a monotone transform of the v1 1-ply score),
          then RNG/first.
        * ``"tie-prior-forcing"`` (v12): most-visited; ties by
          forcing-ness first (moves creating a four —
          ``scoring._is_forcing``), then v1 prior, then RNG/first.
          Targets iter-9 §2.6's failure mode: v1's 1-ply score
          undervalues forcing counter-threats (~20k) against
          occupying a creator (~204k), so the prior tie-break
          systematically prefers blocking one of two creators over
          forcing — and with two creators, blocking one is useless.
        """
        visited = [
            (m, c) for m, c in root.children.items() if c.visit_count >= 1
        ]
        if not visited:
            # No simulation ran (sims == 0): fall back to the
            # highest prior. Unused in practice (sims >= 1
            # everywhere); the sane behaviour when there is no
            # search to decide by.
            return max(root.children, key=lambda m: root.children[m].prior)
        if self.root_decision == "q":
            best_q = max(-c.q for _, c in visited)
            cands = [m for m, c in visited if -c.q == best_q]
            best_v = max(root.children[m].visit_count for m in cands)
            cands = [
                m for m in cands if root.children[m].visit_count == best_v
            ]
            return self._rng_or_first(cands)
        if self.root_decision == "blend":
            max_v = max(c.visit_count for _, c in visited)

            def _score(child: _Node) -> float:
                return -child.q + _BLEND_VISIT_WEIGHT * (
                    child.visit_count / max_v
                )

            best_s = max(_score(c) for _, c in visited)
            cands = [m for m, c in visited if _score(c) == best_s]
            best_v = max(root.children[m].visit_count for m in cands)
            cands = [
                m for m in cands if root.children[m].visit_count == best_v
            ]
            return self._rng_or_first(cands)
        # "visits" / "tie-q" / "tie-prior" / "tie-prior-forcing":
        # most-visited first.
        best_v = max(c.visit_count for _, c in visited)
        cands = [m for m, c in visited if c.visit_count == best_v]
        if len(cands) > 1:
            if self.root_decision == "tie-q":
                best_q = max(-root.children[m].q for m in cands)
                cands = [
                    m for m in cands if -root.children[m].q == best_q
                ]
            elif self.root_decision in ("tie-prior", "tie-prior-forcing"):
                if (
                    self.root_decision == "tie-prior-forcing"
                    and board is not None
                    and player is not None
                ):
                    # Forcing-first (v12, iter-12.md §1.1): among
                    # the tied children, prefer moves that create
                    # a four. Only the tied children are tested
                    # (ties are ~24% of searched moves, tie sizes
                    # small), so the scan is cheap. If none is
                    # forcing, the tie-break falls through to the
                    # prior below. Without a board (synthetic-root
                    # tests) the rule degrades to plain tie-prior.
                    forcing = [
                        m
                        for m in cands
                        if scoring._is_forcing(board, m, player)
                    ]
                    if forcing:
                        cands = forcing
                best_p = max(root.children[m].prior for m in cands)
                cands = [
                    m for m in cands
                    if root.children[m].prior == best_p
                ]
            elif self.tie_break_q:
                # v23/v27 (batch 8 H3 / batch 9 H1): q-aware visit-tie
                # break, gated on the q gap by tie_break_q_threshold.
                # Only reachable when root_decision is "visits" (the
                # other rules branch above) and tie_break_q is True —
                # v23 and v27 are the only versions that set it, so
                # every other version's behaviour is unchanged. Same
                # rule as _pick_root_move, same candidate set (with
                # sims >= 1 the visited filter is a no-op here).
                # Default threshold 0.0 reproduces v23 exactly (any
                # positive gap fires); v27 sets 0.05.
                our_q = {m: -root.children[m].q for m in cands}
                q_max = max(our_q.values())
                distinct = sorted(set(our_q.values()), reverse=True)
                q_second = distinct[1] if len(distinct) > 1 else q_max
                if q_max - q_second > self.tie_break_q_threshold:
                    cands = [m for m in cands if our_q[m] == q_max]
                # Else (gap <= threshold): cands unchanged — v18's
                # RNG/first over ALL tied children.
        return self._rng_or_first(cands)

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
        restores it before returning. Under
        ``backup_rule="softmax-opp"`` (v51) a soft-min node returns
        its updated soft-min q rather than the single-simulation
        value, so its mean parent accumulates soft-min values
        (see the descent branch below)."""
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
            if last_move is None:  # root: record the root player
                # FIRST, before the children are created: the
                # children's softmax flag (v51) is decided by
                # whether this node's player is the root player,
                # and the v16 leaf branches on it. Set on every
                # root call: the instance is reused across games
                # in both colours. (Behaviour-neutral for every
                # other version: the assignment is the same, only
                # earlier — nothing between here and the old site
                # reads _root_player.)
                self._root_player = player
            priors = _priors(board, player, self.prior_floor)
            # v51: children of a node whose player-to-move is the
            # root player are OPPONENT nodes (odd depth) — soft-min
            # backup under backup_rule="softmax-opp"; all other
            # children are our nodes (mean backup). Default
            # backup_rule="mean" leaves every flag False.
            child_softmax = (
                self.backup_rule == "softmax-opp"
                and player == self._root_player
            )
            node.children = {
                m: _Node(p, softmax=child_softmax)
                for m, p in priors.items()
            }
            if last_move is None:  # root: no leaf value
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
        if node.softmax:
            # v51 soft-min backup: this node's q is the
            # softmax-weighted mean of the opponent-perspective
            # values of its VISITED children (-child.q), weights
            # exp(beta * value), recomputed after every descent —
            # every simulation that changes a child's q passes
            # through here, so the cached value is always fresh.
            # Unvisited children are excluded (their q is
            # undefined, not a signal). The node's returned value
            # is its soft-min q, so its (mean) parent accumulates
            # soft-min values.
            node.softmax_value = _softmax_mean(
                [
                    -c.q
                    for c in node.children.values()
                    if c.visit_count >= 1
                ],
                self.softmax_beta,
            )
            return node.softmax_value
        return value

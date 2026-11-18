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

    ``tie_break_q_tactical_only`` is v34 (batch 10, H2): the
    tie-break fires ONLY when the root position is tactical —
    either side holds an open three or better on the board
    (:func:`gomoku_ml.scoring._has_forcing_threat`) — in
    addition to v27's gap condition. In a quiet position the
    tie-break does not fire at all and the choice is v18's (RNG
    over ALL tied children, else first). Without a board (the
    synthetic-root tests) the position cannot be shown tactical,
    so the gate blocks the fire. Default False: v27 and all
    earlier versions are unaffected. See
    ``research/iterations/iter-34.md`` §1.1 for the
    pre-registered rule: a large gap in a murky midgame is not
    the same evidence as in a tactical position (the documented
    0.1481-gap fire separated two verified losers).

    ``root_decision`` (v9, batch 4 H1; v12, batch 5 H1) selects the
    root move choice: ``"visits"`` (default, v6/v7 behaviour —
    most-visited, ties by RNG), ``"q"``, ``"tie-q"``, ``"blend"``,
    ``"tie-prior"``, ``"tie-prior-forcing"`` — see
    :meth:`_choose_root_child`.
    ``vcf_root`` (v29, batch 9 H3) enables the VCF-at-root check
    (:meth:`_vcf_root_move`): before the search, an offensive VCF
    start is played immediately, and a defensive VCF prevention
    is played when the opponent holds a VCF start. False (the
    default) is v18 behaviour exactly. The offensive step runs
    FIRST and is load-bearing: in race positions (both sides hold
    a VCF start) it plays our forced win instead of letting the
    defensive step block — a defensive-only variant was tried for
    v35 and refuted by the tactical mechanism check (27/29: it
    blocked instead of winning both race positions; see
    ``research/iterations/iter-35.md`` §3).
    ``vcf_filtered_search`` (v33, batch 10 H1) enables VCF-filtered
    search (:meth:`_vcf_prevention_pool` + :meth:`_search_restricted`):
    the offensive VCF step is v29's unchanged, but the defensive
    step does NOT pick a prevention by prior — it computes the FULL
    prevention pool and runs the normal 100-sim MCTS with the root
    restricted to the pool (priors renormalised over the pool),
    letting the search choose among preventions by visits. Pool of
    1 plays directly; empty pool falls through to the unrestricted
    search. Mutually exclusive with ``vcf_root`` (the two defensive
    rules conflict); setting both raises ValueError. See
    ``research/iterations/iter-33.md`` §1.1.
    ``vcf_root_search_first`` (v35, batch 10 H3) changes the
    DEFENSIVE step to search-first: the search runs first, and
    the defensive check overrides ONLY when the search's chosen
    move is NOT a prevention (leaves opponent VCF starts live).
    When the search already plays a prevention, its visits/q
    signal — built from 100 sims — chooses among preventions,
    not the prior's static score. The fast loop verified 3
    override-harm instances for the plain check (it overrode a
    correct search prevention by prior noise: fast-loop g2 ply
    12, g3 ply 9, g5 ply 9) against 1 case where the prior
    picked the right prevention over a wrong search prevention
    (forensics g73; g44 is the second). Search-first is weakly
    dominant over the no-check baseline: it overrides only a
    move that loses by force (opponent VCF), so it can only
    help, never harm, relative to v27. The offensive step is
    unaffected (still short-circuits). Only meaningful with
    ``vcf_root=True``; False (the default) is v29's behaviour
    exactly. See ``research/iterations/iter-35.md`` §4.2.
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
        tie_break_q_tactical_only: bool = False,
        vcf_root: bool = False,
        vcf_filtered_search: bool = False,
        vcf_root_search_first: bool = False,
    ) -> None:
        if leaf_mode not in _LEAF_FUNCTIONS and (
            leaf_mode != "rollout2_v1" and leaf_mode != "rollout2_v1_both"
        ):
            raise ValueError(
                f"unknown leaf_mode: {leaf_mode!r} "
                f"(known: {sorted(_LEAF_FUNCTIONS) + ['rollout2_v1', 'rollout2_v1_both']})"
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
        if vcf_root and vcf_filtered_search:
            raise ValueError(
                "vcf_root and vcf_filtered_search are mutually "
                "exclusive (conflicting defensive rules)"
            )
        self.sims = sims
        self.c_puct = c_puct
        self.rng = rng
        self.leaf_mode = leaf_mode
        self.prior_floor = prior_floor
        self.root_decision = root_decision
        self.tie_break_q = tie_break_q
        self.tie_break_q_threshold = tie_break_q_threshold
        self.tie_break_q_tactical_only = tie_break_q_tactical_only
        self.vcf_root = vcf_root
        self.vcf_filtered_search = vcf_filtered_search
        # Root-pool restriction for VCF-filtered search (v33): when
        # set (a frozenset of flat moves), the ROOT expansion in
        # _run_one keeps only those children, priors renormalised
        # over the pool. Set by _search_restricted for the duration
        # of one search and cleared in a finally block; None means
        # unrestricted (every other version, and v33 outside the
        # filtered search). Initialised here (not lazily) so
        # subclasses (ValueMCTS) that inherit _run_one always have
        # the attribute.
        self._root_pool: frozenset[int] | None = None
        self.vcf_root_search_first = vcf_root_search_first
        # Rollout modes are NOT in _LEAF_FUNCTIONS (they are
        # instance-bound methods): the dispatch MUST use this
        # conditional form — the naive ``_LEAF_FUNCTIONS[leaf_mode]``
        # raises KeyError on "rollout2_v1" / "rollout2_v1_both"
        # (batch 6 caught this as a silent merge breakage).
        if leaf_mode == "rollout2_v1":
            self._leaf_value = self._rollout_v1_leaf_value
        elif leaf_mode == "rollout2_v1_both":
            self._leaf_value = self._rollout_v1_both_leaf_value
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

    def _vcf_offensive_move(
        self, board: np.ndarray, player: int
    ) -> int | None:
        """Step 1 of the VCF-at-root check (v29, batch 9 H3),
        factored out for reuse by v33's filtered search and v35's
        search-first defensive (both keep the offensive step
        unchanged): among our four-creating moves, the
        highest-v1-prior one that starts a forced win
        (``evaluate.starts_forced_win``, depth 3), ties by lowest
        flat move index — or None. Tests in selection order and
        returns the FIRST VCF start, which is exactly the
        pre-registered selection rule (identical to testing all
        and taking the max)."""
        ours = evaluate.four_creating_moves(board, player)
        if ours:
            priors = _priors(board, player, self.prior_floor)
            ours.sort(key=lambda m: (-priors.get(m, 0.0), m))
            for m in ours:
                if evaluate.starts_forced_win(board, player, m, depth=3):
                    return m
        return None

    def _vcf_prevention_pool(
        self, board: np.ndarray, player: int
    ) -> list[int] | None:
        """The FULL VCF-prevention pool (v33, batch 10 H1 — see
        ``research/iterations/iter-33.md`` §1.1). Returns None
        when the opponent holds NO VCF start on the current board
        (the gate is closed — the same gate as v29's defensive
        step, so quiet positions pay only the gate check);
        otherwise the list of candidate moves (in prior-desc,
        flat-index-asc order) after which the opponent has NO
        VCF-starting move — possibly empty, when no prevention
        exists. The definition is EXACTLY v29's step 2, with the
        same subset + ordering optimisations (test only the
        opponent's ORIGINAL four-creators — our move can only
        remove them — VCF starts first, so ``any`` short-circuits
        on the first surviving VCF start for non-preventions),
        but WITHOUT v29's first-success early exit: the search
        needs the whole pool. The board is not mutated."""
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
        vcf_set = set(opp_vcf)
        ordered = opp_vcf + [m for m in opp if m not in vcf_set]
        pool: list[int] = []
        for c in cands:
            b2 = game.apply_move(board, c, player)
            if not any(
                evaluate.starts_forced_win(b2, -player, m, depth=3)
                for m in ordered
            ):
                pool.append(c)
        return pool

    def _search_restricted(
        self, board: np.ndarray, player: int, pool: list[int]
    ) -> int:
        """Run the normal MCTS with the root restricted to ``pool``
        (v33, batch 10 H1): root children = pool moves only, priors
        renormalised over the pool (applied in :meth:`_run_one`'s
        root expansion via ``self._root_pool``), ``self.sims``
        simulations, most-visited root choice by this instance's
        root-decision rule (``"visits"`` for v33 — v29/v18's rule).
        All deeper nodes expand normally over all candidates. The
        pool restriction is cleared in a finally block so a reused
        instance never leaks it into the next move. Callers
        guarantee ``len(pool) >= 2`` (a 1-move pool plays directly
        — identical to what a 1-child search would pick)."""
        root = _Node(1.0)
        self._root_pool = frozenset(pool)
        try:
            for _ in range(self.sims):
                self._run_one(root, board, player, None)
        finally:
            self._root_pool = None
        return self._choose_root_child(root, board, player)

    def _vcf_defensive_move(
        self, board: np.ndarray, player: int
    ) -> int | None:
        """Defensive VCF step (v29): the highest-prior prevention
        when the opponent holds a VCF start, or None (no threat,
        or no prevention exists)."""
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
        # property, see _vcf_root_move's docstring), VCF starts
        # first.
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

    def _leaves_opp_vcf(
        self, board: np.ndarray, player: int, move: int
    ) -> bool:
        """True if the opponent still holds a VCF start AFTER
        ``move`` — i.e. ``move`` is NOT a prevention. False when
        the opponent holds no VCF start on the current board (our
        move cannot create one — monotonicity) or when ``move``
        removes them all. Only the opponent's ORIGINAL VCF starts
        need re-testing on the resulting board (a non-VCF-start
        cannot become one by adding our stone)."""
        opp = evaluate.four_creating_moves(board, -player)
        opp_vcf = [
            m
            for m in opp
            if evaluate.starts_forced_win(board, -player, m, depth=3)
        ]
        if not opp_vcf:
            return False
        b2 = game.apply_move(board, move, player)
        return any(
            evaluate.starts_forced_win(b2, -player, m, depth=3)
            for m in opp_vcf
        )

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
        # Both steps factored out (see _vcf_offensive_move /
        # _vcf_defensive_move); v29's selection unchanged.
        off = self._vcf_offensive_move(board, player)
        if off is not None:
            return off
        return self._vcf_defensive_move(board, player)

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
        # VCF-filtered search (v33): offensive VCF played
        # immediately (v29's step 1, unchanged); a defensive
        # prevention pool is SEARCHED, not prior-picked. Pool of
        # 1 plays directly (identical to a 1-child search); empty
        # pool or closed gate falls through to the search below.
        if self.vcf_filtered_search:
            offensive = self._vcf_offensive_move(board, player)
            if offensive is not None:
                return offensive
            pool = self._vcf_prevention_pool(board, player)
            if pool:
                if len(pool) == 1:
                    return pool[0]
                return self._search_restricted(board, player, pool)
        # VCF-at-root (v29): forced wins the 100-sim search misses.
        # Search-first (v35): only the OFFENSIVE step short-circuits
        # here; the defensive step runs after the search below and
        # overrides only a non-prevention (see _leaves_opp_vcf).
        if self.vcf_root and not self.vcf_root_search_first:
            vcf_move = self._vcf_root_move(board, player)
            if vcf_move is not None:
                return vcf_move
        if self.vcf_root and self.vcf_root_search_first:
            off = self._vcf_offensive_move(board, player)
            if off is not None:
                return off
        root = _Node(1.0)
        for _ in range(self.sims):
            self._run_one(root, board, player, None)
        search_move = self._choose_root_child(root, board, player)
        if (
            self.vcf_root
            and self.vcf_root_search_first
            and self._leaves_opp_vcf(board, player, search_move)
        ):
            def_move = self._vcf_defensive_move(board, player)
            if def_move is not None:
                return def_move
        return search_move

    def _rng_or_first(self, moves: list[int]) -> int:
        """Break a final tie: the seeded RNG's choice, or the first
        move when there is no RNG (v6/v7 behaviour)."""
        if len(moves) == 1:
            return moves[0]
        if self.rng is not None:
            return int(self.rng.choice(moves))
        return moves[0]

    def _pick_root_move(
        self, root: _Node, board: np.ndarray | None = None
    ) -> int:
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
        is v18's (RNG over ALL tied children, else first). v34
        (``tie_break_q_tactical_only=True``) gates it further on
        the position: the tie-break fires ONLY when the root
        position is tactical (either side holds an open three or
        better — ``scoring._has_forcing_threat``); without a board
        the position cannot be shown tactical, so the gate blocks
        the fire. With ``tie_break_q=False`` (v18 and earlier) a
        tie is broken by RNG choice among ALL tied children, else
        first. A unique visit leader is untouched either way.
        Pre-registered rules: iter-23.md §1.1, iter-27.md §1.1,
        iter-34.md §1.1. NOTE: this method is test-only —
        ``choose_move`` dispatches through :meth:`_choose_root_child`
        (the live definition); keep the two in sync."""
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
                # v34's tactical gate: the fire needs a tactical
                # position too (checked only when the tie-break
                # would otherwise fire, so the board scan costs
                # nothing on the ~91% of searched moves with no
                # tie). Without a board the position cannot be
                # shown tactical, so the gate blocks the fire.
                if not self.tie_break_q_tactical_only or (
                    board is not None
                    and scoring._has_forcing_threat(board)
                ):
                    best = [m for m in best if our_q[m] == q_max]
                    if len(best) == 1:
                        return best[0]
                    # Else fall through to RNG/first on the q-tied
                    # subset (v23 rule 4).
            # Else (gap <= threshold, or v34's tactical gate blocks
            # the fire): fall through with `best` unchanged — v18's
            # RNG/first over ALL tied children.
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

        ``board``/``player`` are needed by
        ``"tie-prior-forcing"`` (to test forcing-ness) and by
        v34's tactical gate on the q-gap tie-break (to test
        whether either side holds an open three or better); when
        they are absent the former degrades to plain
        ``"tie-prior"`` and the latter blocks the fire.
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
                # v23/v27/v34 (batch 8 H3 / batch 9 H1 / batch 10
                # H2): q-aware visit-tie break, gated on the q gap
                # by tie_break_q_threshold and (v34 only) on the
                # position being tactical. Only reachable when
                # root_decision is "visits" (the other rules branch
                # above) and tie_break_q is True — v23, v27 and v34
                # are the only versions that set it, so every other
                # version's behaviour is unchanged. Same rule as
                # _pick_root_move, same candidate set (with sims >=
                # 1 the visited filter is a no-op here). Default
                # threshold 0.0 reproduces v23 exactly (any positive
                # gap fires); v27 sets 0.05. v34 additionally
                # requires a tactical root position (either side
                # holds an open three or better —
                # scoring._has_forcing_threat); without a board the
                # position cannot be shown tactical, so the gate
                # blocks the fire. The board scan runs only when
                # the tie-break would otherwise fire.
                our_q = {m: -root.children[m].q for m in cands}
                q_max = max(our_q.values())
                distinct = sorted(set(our_q.values()), reverse=True)
                q_second = distinct[1] if len(distinct) > 1 else q_max
                if q_max - q_second > self.tie_break_q_threshold:
                    if not self.tie_break_q_tactical_only or (
                        board is not None
                        and scoring._has_forcing_threat(board)
                    ):
                        cands = [m for m in cands if our_q[m] == q_max]
                # Else (gap <= threshold, or v34's tactical gate
                # blocks the fire): cands unchanged — v18's
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
            priors = _priors(board, player, self.prior_floor)
            if last_move is None and self._root_pool is not None:
                # Restricted root (v33, VCF-filtered search): the
                # root's children are the prevention pool only,
                # priors renormalised over the pool (a proper prior
                # distribution over the restricted action set).
                # Only the root call has last_move None; deeper
                # nodes always expand over all candidates.
                total = sum(
                    p for m, p in priors.items() if m in self._root_pool
                )
                priors = {
                    m: p / total
                    for m, p in priors.items()
                    if m in self._root_pool
                }
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

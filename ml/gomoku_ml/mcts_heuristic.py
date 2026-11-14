"""PUCT MCTS with the v1 heuristic as prior and leaf value (v6, batch 2 H2;
v7, batch 3 H1, adds an attack gradient to the leaf).

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
  open-four race because its leaf gives no attack signal).
* Win/block scans run before the search (as in every version).
* Move choice: the most-visited root child, ties by the seeded
  RNG. Sims: 100, c_puct: 1.5 — the SAME settings as the
  ladder's :class:`gomoku_ml.baseline.HeuristicMCTSPlayer`, so
  any edge over it comes purely from the better priors (v1 vs v0
  scoring) and the better leaf value (terminal-aware vs crude
  local attack).

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
_GRADIENT_STRONG = 1000
_GRADIENT_DEVELOPING = 100
_GRADIENT_DEVELOPING_WEIGHT = 0.25
_GRADIENT_SCALE = 4.0


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
    """Breadth-gradient leaf value in (-1, 1) for ``player`` to move
    in a quiet position: ``tanh((breadth(ours) - breadth(theirs))
    / 4)`` from the kernel's attack-only scores for both colours
    (one ``attack_both`` call). Antisymmetric under colour swap."""
    black, white = kernel_numpy.attack_both(board)
    mask = board == game.EMPTY
    if player == game.BLACK:
        ours, theirs = black, white
    else:
        ours, theirs = white, black
    diff = _breadth(ours, mask) - _breadth(theirs, mask)
    return math.tanh(diff / _GRADIENT_SCALE)


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


class HeuristicMCTS:
    """PUCT search with v1 priors and a terminal-aware leaf value.

    ``leaf_mode="terminal"`` (default) is v6: quiet leaves are 0.
    ``leaf_mode="gradient"`` is v7: quiet leaves get the attack
    breadth gradient (:func:`_gradient_leaf_value`); terminal
    values are identical in both modes.
    """

    def __init__(
        self,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
        leaf_mode: str = "terminal",
    ) -> None:
        if leaf_mode not in _LEAF_FUNCTIONS:
            raise ValueError(
                f"unknown leaf_mode: {leaf_mode!r} "
                f"(known: {sorted(_LEAF_FUNCTIONS)})"
            )
        self.sims = sims
        self.c_puct = c_puct
        self.rng = rng
        self.leaf_mode = leaf_mode
        self._leaf_value = _LEAF_FUNCTIONS[leaf_mode]

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

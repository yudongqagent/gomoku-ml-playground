"""v17: the combination (batch 6, H3; iter-17 §1).

v7's PUCT MCTS + v12's tie-prior-forcing root rule + v14's
v1-kernel gradient leaf + v13's 2-ply threat check at the root.
Each component passed 3/4 in batch 5 and failed a different
clause (v12: clause 1, 0.4950; v13: clause 1, 0.5250; v14:
clause 2, Elo -34.9); this version tests whether they combine
to 4/4. All three components are PORTED, not re-derived:
the root rule and the leaf live in
:mod:`gomoku_ml.mcts_heuristic` (``root_decision`` /
``leaf_mode``), the v1 kernel in :mod:`gomoku_ml.kernel_v1`,
and the threat check below is verbatim from
:class:`gomoku_ml.mcts_white.WhiteMCTS.choose_move`.

Why a new subclass instead of reusing ``WhiteMCTS``:
``WhiteMCTS`` hard-codes ``leaf_mode="gradient"`` (v7's v0-kernel
leaf) and replicates the root decision with plain most-visited +
RNG tie-break, bypassing ``_choose_root_child`` — so neither
the v12 root rule nor the v14 leaf can be expressed through it.
``V17MCTS`` subclasses :class:`HeuristicMCTS` directly with
``leaf_mode="gradient_v1"`` and
``root_decision="tie-prior-forcing"``, and replicates
``WhiteMCTS.choose_move``'s structure with exactly two changes:
the sims count is 100 in both colours (no colour-asymmetric
sims — v10's 200-White-sims is refuted, iter-13 §2.1), and the
final decision line applies ``_choose_root_child`` to the
(threat-filtered) root instead of plain most-visited.

Composition order (pre-registered, iter-17 §1.0): win scan ->
block scan -> 100-sim search -> threat filter restricts the
pool -> the tie-prior-forcing rule decides within the pool.
The filter itself (creators via
``scoring._opponent_creators``, safety via
``scoring._move_is_safe``, the ``safe and len(safe) <
len(root.children)`` condition, the ``threat_filter_used``
counter) is ported verbatim from ``WhiteMCTS``.

Edge case, noted honestly: if the safe pool contains only
UNVISITED children, ``_choose_root_child`` falls back to the
highest prior among the pool, where ``WhiteMCTS``'s plain
most-visited would pick by RNG/first among the 0-visit tie.
That requires creators on the board AND every safe move
unvisited after 100 sims; safe moves (forcing or
creator-killing) carry high priors and are visited in
practice, so the case is not expected to occur. The fallback
is part of the ported v12 rule, not a new invention.

q SIGN (restated so batch 4's error does not recur): a root
child's raw ``q`` is the OPPONENT's value (negamax backup);
our-perspective q is ``-child.q``. q is NOT used anywhere in
v17's root decision (q-based root decisions are CLOSED).
"""

from __future__ import annotations

import numpy as np

from . import game, scoring
from .mcts_heuristic import HeuristicMCTS, _Node


class V17MCTS(HeuristicMCTS):
    """v7's PUCT search with the v1-kernel gradient leaf
    (``leaf_mode="gradient_v1"``), the tie-prior-forcing root
    rule (``root_decision="tie-prior-forcing"``), and v3's
    2-ply threat check at the root.

    Args:
        sims: simulations per move in both colours (100, v7's
            setting — the sims optimum mapped by iter-13 §2.1).
        c_puct / rng: as :class:`HeuristicMCTS` (1.5 by default,
            matching v6/v7 and the ladder MCTS).
    """

    def __init__(
        self,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            sims=sims,
            c_puct=c_puct,
            rng=rng,
            leaf_mode="gradient_v1",
            root_decision="tie-prior-forcing",
        )
        # Fast-loop diagnostic (as in v8/v10/v13): moves where
        # the 2-ply threat check restricted the root pool to
        # safe moves.
        self.threat_filter_used = 0

    def choose_move(self, board: np.ndarray, player: int) -> int:
        """v7's move choice plus the 2-ply threat check at the
        root.

        Win scan, block scan and the PUCT search are identical
        to :meth:`HeuristicMCTS.choose_move` (the search loop
        is replicated here because the parent returns only the
        chosen move, and the safety filter needs the root
        children's visit counts to pick the best SAFE move).
        After the search: if the opponent has creators, the
        choice is restricted to safe moves; if no root child is
        safe, the root rule's choice on the full pool is played
        anyway (best effort, as in v3/v8/v13). With no creators
        the choice is exactly the parent's (the tie-prior-forcing
        rule on the full root).
        """
        wins = scoring._winning_moves(board, player)
        if wins:
            return wins[0]
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
        pool = root.children
        creators = scoring._opponent_creators(board, player)
        if creators:
            safe = {
                m: c for m, c in root.children.items()
                if scoring._move_is_safe(board, m, player, creators)
            }
            if safe and len(safe) < len(root.children):
                pool = safe
                self.threat_filter_used += 1
        if pool is root.children:
            return self._choose_root_child(root, board, player)
        filtered = _Node(1.0)
        filtered.children = pool
        return self._choose_root_child(filtered, board, player)

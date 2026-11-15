"""PUCT MCTS with colour-asymmetric search depth (v10, batch 4 H2).

Subclasses :class:`gomoku_ml.mcts_heuristic.HeuristicMCTS` (v7:
``leaf_mode="gradient"``) and overrides only the root move choice.
Everything else — priors, selection, backup, leaf value, win/block
scans, most-visited decision — is inherited unchanged, so any
difference from v7 is attributable to the two changes below.

Design (pre-registered, ``research/iterations/iter-10.md`` §1):

* **Colour-asymmetric sims (the H2 hypothesis).** The root search
  runs ``white_sims`` (200) simulations when the engine plays
  White and ``sims`` (100) when it plays Black. NOTHING else
  varies by colour. The White gap is an exploration/tempo
  problem, not a value-estimation problem: iter-7 §2.1's fast
  loop found a White saving move (forcing, ranked 8th by v1,
  prior 0.04) starved of visits in 100 sims (the
  ``fl1-g3-ply23`` regression xfail). Doubling White sims
  directly targets visit starvation of low-prior forcing
  moves. The colour is known at ``choose_move`` time: the
  ``player`` argument is the engine's own colour for the game.
* **v3's 2-ply threat check at the root** (ported exactly as in
  v8, ``gomoku_ml.mcts_value.ValueMCTS.choose_move``): when
  the opponent has creators (moves leaving them >= 2 winning
  moves), the choice is restricted to safe moves (forcing, or
  no creator remains live); if no root child is safe, the
  most-visited move is played (best effort, as in v3). With
  no creators the choice is unchanged. v7 does NOT have the
  threat check; iter-8 §7 finding 3 calls it a durable
  portable gain (fixed v8's hard tactical 24/29 -> 29/29).

No net value is used (see iter-10 §1 for why: v8's lesson is
that a value net trained on weak self-play is worse than no
value function unless shrunk hard, and the net leaf lands at
ladder Elo 0.0 against v7's gradient leaf at 190.8 — grafting
a net onto v7's leaf would contaminate the best leaf in the
program with the weakest).

Diagnostics: ``last_sims_used`` records the sims count of the
most recent ``choose_move`` that reached the search (None when
the win/block scan decided, or before any move); the tests
assert it is ``white_sims`` for White and ``sims`` for Black.
``threat_filter_used`` counts moves where the threat check
restricted the root pool to safe moves (as in v8).
"""

from __future__ import annotations

import numpy as np

from . import game, scoring
from .mcts_heuristic import HeuristicMCTS, _Node

# Colour-asymmetric search depth (pre-registered, iter-10 §1):
# White gets twice the search of Black. Black keeps v7's 100 sims
# exactly, so the Black game differs from v7 only by the threat
# check; White's extra 100 sims target the visit starvation of
# low-prior forcing moves (iter-7 §2.1, fl1-g3-ply23).
WHITE_SIMS = 200


class WhiteMCTS(HeuristicMCTS):
    """v7's PUCT search with colour-asymmetric sims and the 2-ply
    threat check at the root.

    Args:
        sims: simulations per move as Black (100, v7's setting).
        white_sims: simulations per move as White (200).
        c_puct / rng: as :class:`HeuristicMCTS` (1.5 by default,
            matching v6/v7 and the ladder MCTS).
    """

    def __init__(
        self,
        sims: int = 100,
        white_sims: int = WHITE_SIMS,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        # v7's leaf: the capped attack gradient in quiet positions.
        super().__init__(
            sims=sims, c_puct=c_puct, rng=rng, leaf_mode="gradient"
        )
        self.white_sims = white_sims
        self.last_sims_used: int | None = None
        # Fast-loop diagnostic (as in v8): moves where the 2-ply
        # threat check restricted the root pool to safe moves.
        self.threat_filter_used = 0

    def choose_move(self, board: np.ndarray, player: int) -> int:
        """v7's move choice plus the colour-asymmetric sims and
        v3's 2-ply threat check.

        Win scan, block scan and the PUCT search are identical to
        :meth:`HeuristicMCTS.choose_move` except the sims count,
        which is ``white_sims`` for White and ``sims`` for Black
        (the search loop is replicated here because the parent
        returns only the chosen move, and the safety filter needs
        the root children's visit counts to pick the best SAFE
        move). After the search: if the opponent has creators,
        the choice is restricted to safe moves; if no root child
        is safe, the most-visited move is played anyway (best
        effort, as in v3/v8). With no creators the choice is
        exactly the parent's (modulo the sims count).
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
        n_sims = self.white_sims if player == game.WHITE else self.sims
        self.last_sims_used = n_sims
        root = _Node(1.0)
        for _ in range(n_sims):
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
        best_visits = max(c.visit_count for c in pool.values())
        best = [
            m for m, c in pool.items() if c.visit_count == best_visits
        ]
        if len(best) == 1:
            return best[0]
        if self.rng is not None:
            return int(self.rng.choice(best))
        return best[0]

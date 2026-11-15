"""v13: WhiteMCTS (v7 + threat check + colour-asymmetric sims) with a
White-only prior temperature (batch 5, H2; iter-13 §1b).

Subclasses :class:`gomoku_ml.mcts_white.WhiteMCTS` and changes ONE
thing: the temperature of the prior softmax, and only when the
engine plays White. Everything else — priors, selection, backup,
leaf value, win/block scans, threat check, most-visited decision —
is inherited unchanged.

Design (pre-registered, ``research/iterations/iter-13.md`` §1b):

* ``_priors`` softmaxes ``log1p(v1 score) / 2.0`` over all
  candidates; the /2.0 temperature gives a ~7x prior ratio between
  successive pattern classes, so a 200k move (an open-four maker
  or its block) takes a huge prior share. The fast loop (§2.2)
  diagnosed the mechanism on White's losses: in g9 ply 5 the
  losing block (6,7) holds prior 0.34 (38 of 100 visits) while
  the oracle-verified saves (9,7)/(4,7) hold 0.108 each; in g7
  ply 25 the two 20k four-makers take 0.087 each (10 visits each)
  while the save (8,9) takes 0.04 (4 visits). White's saving
  moves are quiet counterplay the v1 scorer ranks below the big
  single-pattern moves, so the prior starves them of visits.
* v13 flattens the prior ONLY when the engine plays White:
  temperature ``white_temp`` (Black keeps 2.0), applied to every
  node of the search when the engine is White. The temperature is
  selected by the engine's colour (the root player), not by the
  colour to move at each node — the engine's colour is the
  White-specific variable; the opponent model inside the search
  is unchanged.

Implementation note: ``mcts_heuristic._priors`` hard-codes the
temperature, and ``mcts_heuristic.py`` is NOT edited (parallel
H1/H3 workstreams). The temperature-aware priors function lives
here (a copy of ``_priors`` with a temperature parameter), and
``_run_one`` is overridden to use it — the override replicates
:class:`gomoku_ml.mcts_heuristic.HeuristicMCTS._run_one`
exactly except for the priors call. A test asserts the replica
agrees with the parent's search at the default temperature
(``test_v13_matches_white_mcts_at_default_temperature``).
"""

from __future__ import annotations

import math

import numpy as np

from . import game, scoring
from .mcts_heuristic import _Node
from .mcts_white import WhiteMCTS

# Black's prior temperature — matches
# ``gomoku_ml.mcts_heuristic._PRIOR_TEMPERATURE`` (2.0) exactly;
# asserted in the tests so a drift in the shared module is caught.
_BLACK_TEMPERATURE = 2.0


def _priors_temp(board: np.ndarray, player: int,
                 temperature: float) -> dict[int, float]:
    """Softmax over log1p(v1 scores) / ``temperature``, over all
    candidate moves. Identical to
    :func:`gomoku_ml.mcts_heuristic._priors` at temperature 2.0.
    A higher temperature flattens the prior: low-score moves get
    relatively more prior mass (and hence more visits)."""
    flat = board.reshape(-1).tolist()
    cands = scoring._candidate_moves(board)
    if not cands:
        return {}
    logits = {
        m: math.log1p(max(0.0, scoring._score_move_flat(flat, m, player, 1.0)))
        / temperature
        for m in cands
    }
    top = max(logits.values())
    weights = {m: math.exp(v - top) for m, v in logits.items()}
    total = sum(weights.values())
    return {m: w / total for m, w in weights.items()}


class V13MCTS(WhiteMCTS):
    """v7 + threat check + White prior temperature.

    Args:
        sims: simulations per move as Black (100, v7's setting).
        white_sims: simulations per move as White (100 — the §1.2
            sweep winner).
        white_temp: prior-softmax temperature when the engine
            plays White (2.0 = the v7 default, no change).
        c_puct / rng: as :class:`HeuristicMCTS` (1.5 by default).
    """

    def __init__(
        self,
        sims: int = 100,
        white_sims: int = 100,
        white_temp: float = _BLACK_TEMPERATURE,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(
            sims=sims, white_sims=white_sims, c_puct=c_puct, rng=rng
        )
        self.white_temp = white_temp
        # Temperature of the search in progress, set by
        # choose_move from the engine's colour.
        self._search_temp = _BLACK_TEMPERATURE

    def choose_move(self, board: np.ndarray, player: int) -> int:
        """WhiteMCTS's move choice with the White prior
        temperature selected by the engine's colour."""
        self._search_temp = (
            self.white_temp if player == game.WHITE else _BLACK_TEMPERATURE
        )
        return super().choose_move(board, player)

    def _run_one(
        self,
        node: _Node,
        board: np.ndarray,
        player: int,
        last_move: int | None,
    ) -> float:
        """One simulation — a replica of
        :meth:`gomoku_ml.mcts_heuristic.HeuristicMCTS._run_one`
        except the expansion priors use ``self._search_temp``.
        Returns the value from ``player``'s perspective and
        updates every node on the path (including the newly
        expanded leaf). Mutates ``board`` in place but restores
        it before returning."""
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
            priors = _priors_temp(board, player, self._search_temp)
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

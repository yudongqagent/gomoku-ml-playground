"""PUCT MCTS with a LEARNED policy prior and a learned value leaf (v26,
ML batch 1 H3).

Subclasses :class:`gomoku_ml.mcts_value.ValueMCTS` (v8) and replaces
the ONE thing v8 takes from the heuristic: the prior. v8's priors come
from v1's 1-ply scores (``softmax(log1p(score) / 2)``); v26's come
from a policy net trained on v18 self-play visit distributions (see
``research/iterations/iter-26.md`` §1 and
``ml/scripts/gen_policy_data.py``). The value leaf comes from the
SAME net's value head (quiet positions only, shrunk by
``value_weight`` = 0.25 — v8's measured-best weight), replacing v8's
value net (trained on weak neural self-play) with a value head
trained on v18 self-play outcomes.

One forward pass per expansion yields BOTH the priors and the leaf
value (root: priors only), so v26's inference cost per move is ≈
v8's — and v26 SKIPS the v1-prior computation at every expansion (the
net forward yields the priors directly), so in practice it is
somewhat cheaper per move than v8 (see iter-26.md §2.0).

Prior construction (pre-registered, iter-26.md §1.3): policy logits
over the candidate moves, softmaxed with temperature
``prior_temperature`` (calibrated OFFLINE by matching the v1 prior's
median top-1 mass on held-out v18 positions — the v1 prior is sharp
and load-bearing, v13: flattening was catastrophic). NO blending
with the v1 prior, NO prior floor, NO fallback: the battery measures
the learned prior, not a hybrid.

Everything else is v8 unchanged: 100 sims, c_puct 1.5, win/block
scans before the search, exact terminal leaf cases (+1/-1/-0.25),
v3's 2-ply threat check at the root (inherited from ValueMCTS),
most-visited root child with ties by the seeded RNG, and the same
history threading for the 8-plane encoding.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import encoding, game, scoring
from .mcts_heuristic import _Node
from .mcts_value import VALUE_WEIGHT, ValueMCTS

# Leaf value when the opponent has exactly one winning move and we
# have none — identical to v6's _MUST_BLOCK_VALUE (kept in sync by
# test: test_mcts_value.py asserts the two agree).
_MUST_BLOCK_VALUE = -0.25


class PolicyValueMCTS(ValueMCTS):
    """v8's PUCT search with a learned policy prior and a joint
    policy+value net as the leaf.

    Args:
        checkpoint_path: ``.pt`` checkpoint to load the joint
            policy+value net from (architecture rebuilt from the
            checkpoint payload).
        model: an already-built net to use directly (takes precedence
            over ``checkpoint_path``; used by tests).
        sims / c_puct / rng: as :class:`HeuristicMCTS` (100 / 1.5 by
            default, matching v6/v7/v8 and the ladder MCTS).
        device: torch device string (CPU here; there is no GPU).
        prior_temperature: softmax temperature for the policy logits
            over candidates (pre-registered calibration, see
            iter-26.md §1.3). Must be > 0.
        value_weight: shrink factor for the net's value in quiet
            positions (pre-registered fixed at v8's 0.25).
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model: torch.nn.Module | None = None,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
        prior_temperature: float = 1.0,
        value_weight: float = VALUE_WEIGHT,
    ) -> None:
        super().__init__(
            checkpoint_path=checkpoint_path,
            model=model,
            sims=sims,
            c_puct=c_puct,
            rng=rng,
            device=device,
        )
        if prior_temperature <= 0:
            raise ValueError(
                f"prior_temperature must be > 0, got {prior_temperature!r}"
            )
        self.prior_temperature = prior_temperature
        self.value_weight = value_weight
        # ValueMCTS.__init__ binds self._leaf_value to its own leaf
        # method; this class's _run_one never calls it (one forward
        # pass per expansion yields priors AND leaf value together),
        # so the binding is inert here. Rebind to the combined
        # evaluator's value half for direct callers.
        self._leaf_value = self._policy_value_leaf_value

    def choose_move(self, board: np.ndarray, player: int) -> int:
        """v8's move choice (inherited: win/block scans, PUCT search,
        v3's 2-ply threat check at the root), with single-threaded
        torch inference: a batch=1 forward measures 5.9 ms at 1
        thread vs 36.0 ms at 2 threads under this VM's contention
        (load ~11 from the sibling workstreams), so the 2 threads
        :class:`ValueMCTS` sets would make every move ~6x slower.
        The setting is process-global and persists, so in a match
        process it also speeds up the opponent's net inference
        (v8) after v26's first move — a performance setting only,
        not a game decision (see iter-26.md §2.1)."""
        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        return super().choose_move(board, player)

    def _net_eval(
        self,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray] | None = None,
        last_move: int | None = None,
    ) -> tuple[dict[int, float], float]:
        """One forward pass: the learned priors over the candidate
        moves (policy logits, temperature softmax, renormalised)
        and the raw value-head output in [-1, 1] from ``player``'s
        perspective. Terminal cases are NOT applied here — the
        caller overrides the value with the exact terminal value
        when one applies (the priors are still needed: the node may
        be revisited)."""
        cands = scoring._candidate_moves(board)
        if not cands:
            return {}, 0.0
        planes = encoding.encode(board, player, history, last_move)
        with torch.no_grad():
            x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
            logits, value = self.net(x)
        self.leaf_net_evals += 1
        flat_logits = logits[0].detach().cpu().numpy()
        scaled = {
            m: float(flat_logits[m]) / self.prior_temperature for m in cands
        }
        top = max(scaled.values())
        weights = {m: math.exp(v - top) for m, v in scaled.items()}
        total = sum(weights.values())
        priors = {m: w / total for m, w in weights.items()}
        return priors, float(value[0, 0].item())

    def _policy_value_leaf_value(
        self,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray] | None = None,
        last_move: int | None = None,
    ) -> float:
        """Learned leaf value in [-1, 1] for ``player`` to move:
        terminal cases exact (as in v6/v8); quiet positions get the
        net's value head shrunk by ``value_weight``. Direct-call
        helper (runs its own forward pass); the search path uses
        :meth:`_net_eval` instead so the priors share the pass."""
        if scoring._winning_moves(board, player):
            return 1.0
        opp_wins = scoring._winning_moves(board, -player)
        if len(opp_wins) >= 2:
            return -1.0
        if len(opp_wins) == 1:
            return _MUST_BLOCK_VALUE
        _priors, value = self._net_eval(board, player, history, last_move)
        return value * self.value_weight

    def _run_one(
        self,
        node: "_Node",
        board: np.ndarray,
        player: int,
        last_move: int | None,
        history: list[np.ndarray] | None = None,
    ) -> float:
        """One simulation — identical to
        :meth:`ValueMCTS._run_one` except the expansion step: a
        single forward pass yields the priors (policy head) AND the
        leaf value (value head), instead of v1 priors plus a
        separate value-leaf forward. Terminal leaf cases stay exact;
        quiet leaves get the value head shrunk by ``value_weight``.
        """
        if history is None:
            history = []
        # Terminal check: the move that led here just ended the game.
        if last_move is not None:
            if game.is_winning_move(board, last_move, -player):
                node.visit_count += 1
                node.value_sum += -1.0
                return -1.0
            if game.is_full(board):
                node.visit_count += 1
                return 0.0
        # Expansion: first visit — one forward pass for priors +
        # leaf value; evaluate the leaf.
        if node.children is None:
            priors, raw_value = self._net_eval(board, player, history, last_move)
            node.children = {m: _Node(p) for m, p in priors.items()}
            if last_move is None:  # root: no leaf value
                node.visit_count += 1
                return 0.0
            if scoring._winning_moves(board, player):
                value = 1.0
            elif len(scoring._winning_moves(board, -player)) >= 2:
                value = -1.0
            elif scoring._winning_moves(board, -player):
                value = _MUST_BLOCK_VALUE
            else:
                value = raw_value * self.value_weight
            node.visit_count += 1
            node.value_sum += value
            return value
        # Selection: descend to the best child by PUCT.
        move = self._select_child(node)
        prev = board.copy()
        board[move // 15, move % 15] = player
        value = -self._run_one(
            node.children[move], board, -player, move, history + [prev]
        )
        board[move // 15, move % 15] = 0
        node.visit_count += 1
        node.value_sum += value
        return value

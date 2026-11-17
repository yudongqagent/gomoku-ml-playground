"""PUCT MCTS with v1 heuristic priors and a LEARNED value leaf (v8, batch 3 H2).

Subclasses :class:`gomoku_ml.mcts_heuristic.HeuristicMCTS` (v6) and
overrides the leaf evaluation and the root move choice. v6's
terminal-only leaf values quiet positions at 0, so the search has
no attack gradient in quiet positions and move choice falls back to
priors plus noise (see ``research/iterations/iter-6.md`` §2.2(c)).
v8 keeps the terminal cases exact and asks a trained value net for
quiet positions — the one leaf design that can value
attack-building without a hand-built heuristic (the hand-built
leaf was explored to its limit in v4/v5/v6).

Root move choice adds v3's 2-ply threat check (fast-loop
iteration 1 patch, ``research/iterations/iter-8.md`` §4.1): the
MCTS leaf only scores positions where a winning move already
exists, so the search cannot see the opponent CREATING an open
four next move — v8 played locally-optimal moves that let v1 build
multiple threats over several moves (13/13 smoke losses vs v1,
no single losing move in any of them). When the opponent has
creators (moves leaving them >= 2 winning moves), the chosen move
must be safe against them (forcing, or no creator remains live);
otherwise the best safe move by visit count is played. With no
creators the choice is unchanged.

Leaf value (pre-registered, ``research/iterations/iter-8.md`` §1):
terminal check first, net value otherwise:

* +1.0 — the player to move has any winning move (wins next move);
* -1.0 — the opponent has >= 2 winning moves (cannot block all);
* -0.25 — the opponent has exactly 1 winning move (must block — the
  search resolves which block; same value as v6's
  ``_MUST_BLOCK_VALUE``);
* otherwise the trained net's value head (tanh in [-1, 1], from the
  player-to-move perspective) on the 8-plane encoding of the leaf
  position.

The terminal cases stay exact: a net cannot be trusted to recognise
a forced win/loss as reliably as a scan, and v6's defence is already
perfect — the leaf must not regress it. The net's job is exactly the
gap v6 exposed: valuing QUIET positions.

Net input: the full 8-plane encoding (:func:`gomoku_ml.encoding.encode`).
History planes are threaded through the search path — ``_run_one``
carries the boards before each move along the path (copied, because
the search mutates the board in place), so planes 4–7 are the
position 2/4 plies back along the search path, exactly as in
training (where they are the real game history). The root's history
is empty: the real game history before the current position is
unknown to ``choose_move``, which receives only the board — so
leaves at depth < 2 have zeroed history planes, the same as
early-game training positions. ``last_move`` is the move that led
to the leaf (plane 2).

Inference is CPU, batch=1, 2 torch threads (the harness shares this
VM with the parallel H1 workstream). ``leaf_net_evals`` counts net
forward passes (diagnostics for the cost measurement in iter-8 §3).
"""

from __future__ import annotations

import numpy as np
import torch

from . import encoding, game, scoring
from .architectures import load_model_from_checkpoint
from .mcts_heuristic import HeuristicMCTS, _Node, _priors

# Leaf value when the opponent has exactly one winning move and we
# have none — identical to v6's _MUST_BLOCK_VALUE (kept in sync by
# test: test_mcts_value.py asserts the two agree).
_MUST_BLOCK_VALUE = -0.25

# Fast-loop iteration 2: the learned value is SHRUNK toward the
# terminal value (0 in quiet positions). The net was trained on
# weak neural self-play, and its values are unreliable against
# heuristic opponents — the unpatched v8 lost 0-10 to v0 while
# v6 (terminal-only leaf, value 0 in quiet positions) scored
# 4W-4L-2D on the same seeds (iter-8.md §4.2). MCTS exploits the
# net's overvaluations (the optimizer's curse): with the full net
# value the search overrides the priors with noise, starting in
# the opening (v8 replies (7,9) to (7,7) where v6 plays (5,5) /
# (5,7) / (5,9)). Shrinking by VALUE_WEIGHT keeps the attack
# gradient as a weak signal while letting the priors and the
# terminal cases do most of the work. Terminal cases are exact
# and unaffected (the shrink applies only to quiet positions).
VALUE_WEIGHT = 0.25


class ValueMCTS(HeuristicMCTS):
    """v6's PUCT search with a learned value function as the leaf.

    Args:
        checkpoint_path: ``.pt`` checkpoint to load the value net
            from (architecture rebuilt from the checkpoint payload).
        model: an already-built net to use directly (takes precedence
            over ``checkpoint_path``; used by tests).
        sims / c_puct / rng: as :class:`HeuristicMCTS` (100 / 1.5 by
            default, matching v6 and the ladder MCTS).
        device: torch device string (CPU here; there is no GPU).
        value_weight: shrink applied to the net value in quiet
            positions (batch 4 H3, v11 — per-version α; default
            ``VALUE_WEIGHT`` = 0.25, v8's swept value, so v8 is
            unchanged). Terminal cases are exact and unaffected.
        vcf_root: enable the VCF-at-root check (v32, ML batch 2
            H3): an offensive VCF start is played immediately,
            before the search — the same hook as v29's in
            :class:`HeuristicMCTS`, which this class's overridden
            :meth:`choose_move` would otherwise bypass. False (the
            default) is v8 behaviour exactly.
        vcf_defensive: include the defensive VCF-prevention step
            of :meth:`HeuristicMCTS._vcf_root_move`. False (the
            default) is v32's pre-registered offensive-only design
            (see ``research/iterations/iter-32.md`` §1.1): v29's
            defensive check is a verified flaw (it picks
            preventions by prior noise — iter-29 §3.4), so it is
            OFF unless the fast loop validates a VCF-filtered
            search. Only meaningful with ``vcf_root=True``.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model: torch.nn.Module | None = None,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
        value_weight: float = VALUE_WEIGHT,
        vcf_root: bool = False,
        vcf_defensive: bool = False,
    ) -> None:
        super().__init__(sims=sims, c_puct=c_puct, rng=rng, vcf_root=vcf_root)
        self.value_weight = float(value_weight)
        # The parent's vcf_root flag is stored but the parent's
        # choose_move is overridden below, so the hook is applied
        # explicitly there (with the defensive step gated by
        # vcf_defensive — False is v32's offensive-only design).
        self.vcf_defensive = vcf_defensive
        # Batch-3 merge fix: H1's HeuristicMCTS sets ``self._leaf_value``
        # to a module-level leaf function in __init__; that instance
        # attribute shadows this class's leaf method, so the net leaf
        # never ran (and _run_one's 4-arg call raised TypeError).
        # Rebind the dispatch point to the learned leaf so this class
        # behaves exactly as it did on the H2 branch, where
        # ``_leaf_value`` was a plain overridable method.
        self._leaf_value = self._value_leaf_value
        if model is not None:
            self.net = model
            self.net.eval()
        elif checkpoint_path is not None:
            self.net, _payload = load_model_from_checkpoint(
                checkpoint_path, device=device
            )
        else:
            raise ValueError(
                "ValueMCTS needs a value net: pass checkpoint_path or model"
            )
        self.device = torch.device(device)
        self.net.to(self.device)
        # Two intra-op threads: the harness shares this VM with the
        # parallel H1 workstream, and a batch=1 forward pass gains
        # nothing from more threads (see ml/README.md "Threading").
        # Interop threads can only be set once per process, before
        # parallel work starts — hence the guard.
        if torch.get_num_threads() != 2:
            torch.set_num_threads(2)
        try:
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # parallel work already started; the default stands
        self.checkpoint_path = checkpoint_path
        self.leaf_net_evals = 0
        # Fast-loop iteration 1 diagnostic: moves where the 2-ply
        # threat check restricted the root pool to safe moves.
        self.threat_filter_used = 0

    def choose_move(self, board: np.ndarray, player: int) -> int:
        """Parent's move choice plus v3's 2-ply threat check.

        Win scan, block scan and the PUCT search are identical to
        :meth:`HeuristicMCTS.choose_move` (the search loop is
        replicated here because the parent returns only the chosen
        move, and the safety filter needs the root children's visit
        counts to pick the best SAFE move). With ``vcf_root`` (v32)
        an offensive VCF start is played immediately after the
        win/block scans, before the search — the parent's hook does
        not apply because this method overrides the parent's. After
        the search: if the opponent has creators (moves leaving them
        >= 2 winning moves — see
        :func:`gomoku_ml.scoring._opponent_creators`), the choice
        is restricted to safe moves
        (:func:`gomoku_ml.scoring._move_is_safe`); if no root
        child is safe, the most-visited move is played anyway
        (best effort, as in v3). With no creators the choice is
        exactly the parent's.
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
        # VCF-at-root (v32, offensive-only by default): a forced
        # win the 100-sim search misses is played immediately.
        # Same position as v29's hook in the parent's choose_move
        # (after the win/block scans, before the search).
        if self.vcf_root:
            vcf_move = self._vcf_root_move(
                board, player, defensive=self.vcf_defensive
            )
            if vcf_move is not None:
                return vcf_move
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
        best_visits = max(c.visit_count for c in pool.values())
        best = [
            m for m, c in pool.items() if c.visit_count == best_visits
        ]
        if len(best) == 1:
            return best[0]
        if self.rng is not None:
            return int(self.rng.choice(best))
        return best[0]

    def _value_leaf_value(
        self,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray] | None = None,
        last_move: int | None = None,
    ) -> float:
        """Learned leaf value in [-1, 1] for ``player`` to move.

        Terminal cases first (exact, as in v6); quiet positions get
        the net's value head on the 8-plane encoding. Bound to
        ``self._leaf_value`` in __init__ (see the merge-fix comment
        there), so ``_run_one`` and direct callers use this method.
        """
        if scoring._winning_moves(board, player):
            return 1.0
        opp_wins = scoring._winning_moves(board, -player)
        if len(opp_wins) >= 2:
            return -1.0
        if len(opp_wins) == 1:
            return _MUST_BLOCK_VALUE
        planes = encoding.encode(board, player, history, last_move)
        with torch.no_grad():
            x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
            _logits, value = self.net(x)
        self.leaf_net_evals += 1
        # Fast-loop iteration 2: shrink the net value toward the
        # terminal value (0 here) — see VALUE_WEIGHT; the instance
        # value_weight lets each version sweep its own α (v11).
        return float(value[0, 0].item()) * self.value_weight

    def _run_one(
        self,
        node: "_Node",
        board: np.ndarray,
        player: int,
        last_move: int | None,
        history: list[np.ndarray] | None = None,
    ) -> float:
        """One simulation — identical to :meth:`HeuristicMCTS._run_one`
        except the leaf evaluation (see :meth:`_value_leaf_value`,
        bound to ``self._leaf_value`` in __init__) and the
        history threading (the boards before each move along the
        search path, copied, so the leaf encoding's history planes
        match the training distribution).

        Signature is a compatible extension of the parent's
        (``history`` defaults to None), so the inherited
        ``choose_move`` drives it unchanged.
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
        # Expansion: first visit — compute priors, evaluate the leaf.
        if node.children is None:
            priors = _priors(board, player)
            node.children = {m: _Node(p) for m, p in priors.items()}
            if last_move is None:  # root: no leaf value
                node.visit_count += 1
                return 0.0
            value = self._leaf_value(board, player, history, last_move)
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

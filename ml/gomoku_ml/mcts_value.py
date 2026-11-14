"""PUCT MCTS with v1 heuristic priors and a LEARNED value leaf (v8, batch 3 H2).

Subclasses :class:`gomoku_ml.mcts_heuristic.HeuristicMCTS` (v6) and
overrides ONLY the leaf evaluation. v6's terminal-only leaf values
quiet positions at 0, so the search has no attack gradient in quiet
positions and move choice falls back to priors plus noise (see
``research/iterations/iter-6.md`` §2.2(c)). v8 keeps the terminal
cases exact and asks a trained value net for quiet positions — the
one leaf design that can value attack-building without a hand-built
heuristic (the hand-built leaf was explored to its limit in v4/v5/v6).

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
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        model: torch.nn.Module | None = None,
        sims: int = 100,
        c_puct: float = 1.5,
        rng: np.random.Generator | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(sims=sims, c_puct=c_puct, rng=rng)
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

    def _leaf_value(
        self,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray] | None = None,
        last_move: int | None = None,
    ) -> float:
        """Learned leaf value in [-1, 1] for ``player`` to move.

        Terminal cases first (exact, as in v6); quiet positions get
        the net's value head on the 8-plane encoding.
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
        return float(value[0, 0].item())

    def _run_one(
        self,
        node: "_Node",
        board: np.ndarray,
        player: int,
        last_move: int | None,
        history: list[np.ndarray] | None = None,
    ) -> float:
        """One simulation — identical to :meth:`HeuristicMCTS._run_one`
        except the leaf evaluation (see :meth:`_leaf_value`) and the
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

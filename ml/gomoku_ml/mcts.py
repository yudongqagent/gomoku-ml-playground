"""PUCT Monte Carlo Tree Search using the neural network.

Standard AlphaZero-style search:

* Each node stores visit count N, total value W, mean value Q, and the
  prior probability P from the network policy.
* Selection maximises Q + U with U = c_puct * P * sqrt(N_parent) / (1 + N).
* Leaves are expanded with a single network evaluation (batch=1 for v1;
  batched leaf evaluation is noted as future work in ml/README.md).
* Batched leaf evaluation is available via :meth:`MCTS.search_many`:
  several independent searches run concurrently and their leaf
  positions are evaluated in one batched forward pass. Simulations
  within one search still run strictly sequentially, so selection and
  backprop semantics are identical to :meth:`MCTS.search` — no virtual
  loss, no search-order change. The only differences from the
  single-leaf path are float noise in batched vs single forward passes
  (~1e-6) and RNG interleaving order for Dirichlet noise. See
  research/iterations/batched-leaf-eval.md.
* Values are stored from the perspective of the player to move at the
  *parent* node and negated as they propagate up the tree, matching the
  network's current-player value convention.
* Root Dirichlet noise (for self-play exploration) and temperature-based
  move sampling are applied by the caller / self-play driver, not here.
* A resign threshold is supported by the self-play driver: if the root
  value falls below it for the player to move, the game is resigned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generator, NamedTuple

import numpy as np
import torch

from . import encoding, game


class LeafRequest(NamedTuple):
    """One leaf position awaiting network evaluation.

    Yielded by :meth:`MCTS._simulate`; the driver answers it with a
    ``(policy_probs, value)`` tuple, where ``policy_probs`` is a
    225-vector of legal-move probabilities and ``value`` is the scalar
    value from ``player``'s perspective.
    """

    board: np.ndarray
    player: int
    history: list[np.ndarray]
    last_move: int | None


@dataclass
class SearchRequest:
    """One independent search for :meth:`MCTS.search_many`."""

    board: np.ndarray
    player: int
    num_simulations: int
    history: list[np.ndarray] | None = None
    last_move: int | None = None
    add_noise: bool = False


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "expanded")

    def __init__(self, prior: float = 1.0) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.expanded = False

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    """PUCT search over a single root position.

    Args:
        net: GomokuNet (eval mode is set by the caller or here).
        device: torch device string.
        c_puct: exploration constant.
        dirichlet_alpha: Dirichlet noise concentration for root exploration
            (0 disables noise).
        dirichlet_epsilon: weight of the Dirichlet noise at the root.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        device: str = "cpu",
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self.net = net
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.net.eval()

    # -- network evaluation -------------------------------------------------

    def _evaluate(
        self, board: np.ndarray, player: int, history: list[np.ndarray], last_move: int | None
    ) -> tuple[np.ndarray, float]:
        """Run the net once; return (policy probs over 225, scalar value).

        Policy is masked to legal moves and renormalised. The value is
        from ``player``'s perspective.
        """
        planes = encoding.encode(board, player, history, last_move)
        with torch.no_grad():
            x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
            logits, value = self.net(x)
        probs = torch.softmax(logits[0], dim=0).cpu().numpy().astype(np.float64)
        mask = (board == game.EMPTY).reshape(-1)
        probs = np.where(mask, probs, 0.0)
        total = probs.sum()
        if total <= 0:
            probs = mask.astype(np.float64) / mask.sum()
        else:
            probs = probs / total
        return probs, float(value[0, 0].item())

    def _evaluate_batch(
        self, requests: list[LeafRequest]
    ) -> list[tuple[np.ndarray, float]]:
        """Evaluate several leaf positions in ONE batched forward pass.

        Returns a list of ``(policy probs over 225, scalar value)``
        pairs in the same order as ``requests``, with the same
        legal-move masking / renormalisation as :meth:`_evaluate`.
        An empty request list returns an empty list (no forward pass).

        Per-position results match single-leaf evaluation up to float
        noise (batched kernels reduce in a different order; observed
        max abs diff ~1e-6 on CPU — see
        research/iterations/batched-leaf-eval.md). The net is in eval
        mode, so batch-norm uses running stats and positions in a batch
        do not interact.
        """
        if not requests:
            return []
        planes = encoding.encode_batch(
            [r.board for r in requests],
            [r.player for r in requests],
            [r.history for r in requests],
            [r.last_move for r in requests],
        )
        with torch.no_grad():
            x = torch.from_numpy(planes).to(self.device)
            logits, values = self.net(x)
        probs_all = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
        values_all = values[:, 0].cpu().numpy()
        out: list[tuple[np.ndarray, float]] = []
        for i, req in enumerate(requests):
            probs = probs_all[i]
            mask = (req.board == game.EMPTY).reshape(-1)
            probs = np.where(mask, probs, 0.0)
            total = probs.sum()
            if total <= 0:
                probs = mask.astype(np.float64) / mask.sum()
            else:
                probs = probs / total
            out.append((probs, float(values_all[i])))
        return out

    # -- search -------------------------------------------------------------

    def _apply_root_noise(self, root: Node) -> None:
        """Mix Dirichlet noise into the root priors (self-play
        exploration). No-op when noise is disabled or the root has no
        children (e.g. a terminal root)."""
        if self.dirichlet_epsilon <= 0 or not root.children:
            return
        moves = list(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, n in zip(moves, noise):
            child = root.children[move]
            child.prior = (
                (1 - self.dirichlet_epsilon) * child.prior
                + self.dirichlet_epsilon * float(n)
            )

    def _select_child(self, node: Node) -> tuple[int, Node]:
        best_move, best_child, best_score = -1, None, -math.inf
        sqrt_parent = math.sqrt(max(1, node.visit_count))
        for move, child in node.children.items():
            # child.q is from the perspective of the player to move at the
            # child (the opponent), so negate it: the parent maximises its
            # own expected outcome.
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = -child.q + u
            if score > best_score:
                best_move, best_child, best_score = move, child, score
        assert best_child is not None
        return best_move, best_child

    def _simulate(
        self,
        node: Node,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray],
        last_move: int | None,
    ) -> Generator[LeafRequest, tuple[np.ndarray, float], float]:
        """One simulation as a generator.

        Selection and backprop run synchronously; the single leaf
        evaluation is a yield point: the driver answers the yielded
        :class:`LeafRequest` with ``(policy_probs, value)`` via
        ``send()``. A simulation yields at most once (terminal
        simulations yield nothing — they return a value directly).
        Returns the value from the perspective of the player to move
        at ``node``. Selection/backprop semantics are exactly those of
        the old recursive ``_run_one``; only the net call is deferred
        to the driver, which is what makes batched evaluation possible.
        """
        # Terminal check: only the last move can have created a win, so
        # check the four lines through it (O(1)) instead of scanning the
        # whole board. A draw is only possible right after a move fills
        # the board, so the full-board check is only needed then.
        if last_move is not None:
            if game.is_winning_stone(board, last_move, -player):
                # The player who just moved (the opponent of the player
                # to move here) won.
                return -1.0
            if game.is_full(board):
                return 0.0
        if not node.expanded:
            probs, value = yield LeafRequest(board, player, history, last_move)
            for move in game.legal_moves(board):
                node.children[int(move)] = Node(prior=float(probs[move]))
            node.expanded = True
            return value
        move, child = self._select_child(node)
        next_board = game.apply_move(board, move, player)
        next_history = history + [board]
        # Value from the child's perspective is negated back to this node.
        value = -(yield from self._simulate(child, next_board, -player, next_history, move))
        child.visit_count += 1
        child.value_sum += -value  # value from the child's player's view
        return value

    def _run_one(
        self,
        node: Node,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray],
        last_move: int | None,
    ) -> float:
        """Run one simulation from ``node``; return the value from the
        perspective of the player to move at ``node``.

        Single-leaf path: drives :meth:`_simulate`, evaluating its
        (at most one) leaf request immediately with :meth:`_evaluate`.
        This is the original behaviour, unchanged."""
        gen = self._simulate(node, board, player, history, last_move)
        try:
            request = next(gen)
        except StopIteration as done:
            return done.value
        probs, value = self._evaluate(
            request.board, request.player, request.history, request.last_move
        )
        try:
            gen.send((probs, value))
        except StopIteration as done:
            return done.value
        raise AssertionError("a simulation yields at most one leaf request")

    def search(
        self,
        board: np.ndarray,
        player: int,
        num_simulations: int,
        history: list[np.ndarray] | None = None,
        last_move: int | None = None,
        add_noise: bool = False,
    ) -> tuple[Node, float]:
        """Run ``num_simulations`` simulations from the root.

        Returns (root node, root value estimate from ``player``'s
        perspective). With ``add_noise``, Dirichlet noise is mixed into
        the root priors after the first expansion (self-play exploration).
        """
        history = history or []
        root = Node()
        # First simulation expands the root.
        root_value = self._run_one(root, board, player, history, last_move)
        root.visit_count += 1
        root.value_sum += root_value
        if add_noise:
            self._apply_root_noise(root)
        for _ in range(max(0, num_simulations - 1)):
            value = self._run_one(root, board, player, history, last_move)
            root.visit_count += 1
            root.value_sum += value
        return root, root.value_sum / max(1, root.visit_count)

    def search_many(
        self,
        requests: list[SearchRequest],
        max_batch_size: int = 32,
    ) -> list[tuple[Node, float]]:
        """Run several independent searches concurrently, batching leaf
        evaluations across them.

        Returns one ``(root node, root value)`` pair per request, in
        request order, with the same meaning as :meth:`search`.

        How it works: the searches are driven round by round — one
        simulation per active search per round. Each simulation yields
        at most one leaf request; all requests from a round are
        evaluated in a single batched forward pass (chunked to
        ``max_batch_size``), then each simulation is resumed with its
        result. Simulations *within* one search still run strictly
        sequentially, so each search builds the same tree
        :meth:`search` would, up to float noise in batched vs single
        forward passes and RNG interleaving order for Dirichlet noise
        (noise is drawn per search after its first simulation, exactly
        as in :meth:`search`).

        ``max_batch_size`` caps the forward-pass batch (GPU memory /
        latency tuning); a round with fewer active searches simply
        uses a smaller batch. An empty request list returns ``[]``.
        """
        if not requests:
            return []
        max_batch_size = max(1, int(max_batch_size))
        # Per-search state: root, sims done, sims total, request, history.
        states = [
            {
                "root": Node(),
                "done": 0,
                "total": max(1, req.num_simulations),
                "req": req,
                "history": req.history or [],
            }
            for req in requests
        ]
        while True:
            active = [s for s in states if s["done"] < s["total"]]
            if not active:
                break
            pending: list[tuple[dict, Generator]] = []
            leaves: list[LeafRequest] = []
            for s in active:
                req = s["req"]
                gen = self._simulate(
                    s["root"], req.board, req.player, s["history"], req.last_move
                )
                try:
                    leaf = next(gen)
                except StopIteration as finished:
                    # Terminal simulation: no leaf eval needed.
                    self._finish_simulation(s, finished.value)
                    continue
                pending.append((s, gen))
                leaves.append(leaf)
            results: list[tuple[np.ndarray, float]] = []
            for i in range(0, len(leaves), max_batch_size):
                results.extend(self._evaluate_batch(leaves[i : i + max_batch_size]))
            for (s, gen), (probs, value) in zip(pending, results):
                try:
                    gen.send((probs, value))
                except StopIteration as finished:
                    self._finish_simulation(s, finished.value)
                else:
                    raise AssertionError(
                        "a simulation yields at most one leaf request"
                    )
        return [
            (s["root"], s["root"].value_sum / max(1, s["root"].visit_count))
            for s in states
        ]

    def _finish_simulation(self, state: dict, value: float) -> None:
        """Book-keep one completed simulation: root visit/value update,
        plus Dirichlet noise after the first simulation when requested
        (mirrors :meth:`search`)."""
        root = state["root"]
        root.visit_count += 1
        root.value_sum += value
        state["done"] += 1
        if state["done"] == 1 and state["req"].add_noise:
            self._apply_root_noise(root)

    # -- move selection -----------------------------------------------------

    @staticmethod
    def visit_counts(root: Node) -> dict[int, int]:
        return {m: c.visit_count for m, c in root.children.items()}

    @staticmethod
    def visit_distribution(root: Node, temperature: float = 1.0) -> np.ndarray:
        """Visit-count distribution over all 225 points (zeros elsewhere).

        temperature=0 -> argmax (one-hot on the most visited move).
        """
        counts = np.zeros(game.SIZE * game.SIZE, dtype=np.float64)
        for move, child in root.children.items():
            counts[move] = child.visit_count
        if temperature <= 0:
            out = np.zeros_like(counts)
            out[int(np.argmax(counts))] = 1.0
            return out
        counts = counts ** (1.0 / temperature)
        total = counts.sum()
        if total <= 0:
            return np.ones_like(counts) / counts.size
        return counts / total

    @staticmethod
    def choose_move(root: Node, temperature: float = 1.0) -> int:
        """Sample (or argmax, temperature=0) a move from visit counts."""
        dist = MCTS.visit_distribution(root, temperature)
        if temperature <= 0:
            return int(np.argmax(dist))
        return int(np.random.choice(len(dist), p=dist))

"""PUCT Monte Carlo Tree Search using the neural network.

Standard AlphaZero-style search:

* Each node stores visit count N, total value W, mean value Q, and the
  prior probability P from the network policy.
* Selection maximises Q + U with U = c_puct * P * sqrt(N_parent) / (1 + N).
* Leaves are expanded with a single network evaluation (batch=1 for v1;
  batched leaf evaluation is noted as future work in ml/README.md).
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

import numpy as np
import torch

from . import encoding, game


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

    # -- search -------------------------------------------------------------

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

    def _run_one(
        self,
        node: Node,
        board: np.ndarray,
        player: int,
        history: list[np.ndarray],
        last_move: int | None,
    ) -> float:
        """Run one simulation from ``node``; return the value from the
        perspective of the player to move at ``node``."""
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
            probs, value = self._evaluate(board, player, history, last_move)
            for move in game.legal_moves(board):
                node.children[int(move)] = Node(prior=float(probs[move]))
            node.expanded = True
            return value
        move, child = self._select_child(node)
        next_board = game.apply_move(board, move, player)
        next_history = history + [board]
        # Value from the child's perspective is negated back to this node.
        value = -self._run_one(child, next_board, -player, next_history, move)
        child.visit_count += 1
        child.value_sum += -value  # value from the child's player's view
        return value

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
        if add_noise and self.dirichlet_epsilon > 0 and root.children:
            moves = list(root.children)
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            for move, n in zip(moves, noise):
                child = root.children[move]
                child.prior = (
                    (1 - self.dirichlet_epsilon) * child.prior
                    + self.dirichlet_epsilon * float(n)
                )
        for _ in range(max(0, num_simulations - 1)):
            value = self._run_one(root, board, player, history, last_move)
            root.visit_count += 1
            root.value_sum += value
        return root, root.value_sum / max(1, root.visit_count)

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

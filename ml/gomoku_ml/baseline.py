"""Simple heuristic Gomoku player, used as an evaluation baseline.

Mirrors the web app's heuristic AI conceptually (not move-for-move):

1. Take an immediate win if one exists.
2. Block an opponent immediate win if one exists.
3. Otherwise score every legal move by local pattern strength and pick
   the best (ties broken by a fixed pseudo-random choice with a seed).

Pattern scoring counts, for each direction through a candidate point,
the longest run the move would create for the mover and for the
opponent, weighting open ends. This is a deliberately weak baseline:
any trained net should beat it easily once it has seen real training.
It exists so the eval ladder has a non-random opponent that still loses
to competent play.
"""

from __future__ import annotations

import math

import numpy as np

from . import game
from .mcts import Node

# Weights indexed by (run length after the move, number of open ends).
# An open end is an adjacent empty point along the line direction.
_WEIGHTS = {
    (5, 0): 100_000,
    (5, 1): 100_000,
    (5, 2): 100_000,
    (4, 2): 10_000,  # open four: wins next move either way
    (4, 1): 1_000,   # half-open four
    (4, 0): 100,
    (3, 2): 1_000,   # open three
    (3, 1): 100,
    (3, 0): 10,
    (2, 2): 100,
    (2, 1): 10,
    (2, 0): 1,
    (1, 2): 10,
    (1, 1): 1,
    (1, 0): 0,
}


def _run_and_open_ends(
    board: np.ndarray, y: int, x: int, dy: int, dx: int, player: int
) -> tuple[int, int]:
    """If ``player`` played at (y, x), the run length and open-end count
    along direction (dy, dx)."""
    run = 1
    open_ends = 0
    yy, xx = y + dy, x + dx
    while 0 <= yy < game.SIZE and 0 <= xx < game.SIZE and board[yy, xx] == player:
        run += 1
        yy += dy
        xx += dx
    if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE and board[yy, xx] == game.EMPTY:
        open_ends += 1
    yy, xx = y - dy, x - dx
    while 0 <= yy < game.SIZE and 0 <= xx < game.SIZE and board[yy, xx] == player:
        run += 1
        yy -= dy
        xx -= dx
    if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE and board[yy, xx] == game.EMPTY:
        open_ends += 1
    return run, open_ends


def score_move(board: np.ndarray, move: int, player: int) -> int:
    """Heuristic score for ``player`` playing ``move`` on ``board``.

    Own patterns count fully; opponent patterns at the same point count
    at 90% (defence matters slightly less than attack at equal shape —
    wins and blocks are handled separately before scoring anyway).
    """
    y, x = move // game.SIZE, move % game.SIZE
    total = 0
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run, ends = _run_and_open_ends(board, y, x, dy, dx, player)
        total += _WEIGHTS.get((min(run, 5), ends), 0)
        run_o, ends_o = _run_and_open_ends(board, y, x, dy, dx, -player)
        total += int(0.9 * _WEIGHTS.get((min(run_o, 5), ends_o), 0))
    # Small centre preference to keep early play sensible.
    total += 7 - (abs(x - 7) + abs(y - 7))
    return total


def choose_move(board: np.ndarray, player: int, rng: np.random.Generator | None = None) -> int:
    """Pick a move for ``player`` on ``board``.

    1. Immediate win. 2. Block opponent's immediate win. 3. Best
    heuristic score (random tie-break with the given rng, or the first
    best move if no rng is given).
    """
    legal = game.legal_moves(board)
    if len(legal) == 0:
        raise ValueError("no legal moves")
    # 1. Take the win.
    for move in legal:
        if game.is_winning_move(board, int(move), player):
            return int(move)
    # 2. Block the opponent's win.
    for move in legal:
        if game.is_winning_move(board, int(move), -player):
            return int(move)
    # 3. Best heuristic score.
    best_score = None
    best_moves: list[int] = []
    for move in legal:
        s = score_move(board, int(move), player)
        if best_score is None or s > best_score:
            best_score = s
            best_moves = [int(move)]
        elif s == best_score:
            best_moves.append(int(move))
    if rng is not None:
        return int(rng.choice(best_moves))
    return best_moves[0]


class HeuristicPlayer:
    """Stateful wrapper with a seeded RNG for reproducible eval games."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def choose_move(self, board: np.ndarray, player: int) -> int:
        return choose_move(board, player, self.rng)


class RandomPlayer:
    """Uniformly random legal move. The weakest eval opponent."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def choose_move(self, board: np.ndarray, player: int) -> int:
        legal = game.legal_moves(board)
        return int(self.rng.choice(legal))


def _candidate_moves(board: np.ndarray) -> list[int]:
    """Legal moves worth considering: empty points within Chebyshev
    distance 2 of any stone (plus the centre 3x3 on a near-empty
    board). Gomoku moves far from all stones are almost never good,
    and restricting the branching factor is what makes this search
    affordable in pure Python."""
    stones = np.argwhere(board != game.EMPTY)
    if len(stones) == 0:
        return [game.xy_to_move(x, y) for y in range(6, 9) for x in range(6, 9)]
    near = np.zeros((game.SIZE, game.SIZE), dtype=bool)
    for y, x in stones:
        y0, y1 = max(0, y - 2), min(game.SIZE, y + 3)
        x0, x1 = max(0, x - 2), min(game.SIZE, x + 3)
        near[y0:y1, x0:x1] = True
    out = [
        int(m) for m in np.flatnonzero((board == game.EMPTY) & near)
    ]
    return out or [int(m) for m in game.legal_moves(board)]


def _local_attack_value(board: np.ndarray, last_move: int, mover: int) -> float:
    """Crude leaf evaluation: how strong is the attack ``mover`` just
    built through ``last_move``? Returns a value in [0, 1) from the
    mover's perspective (an open four is ~0.67, an open three ~0.17).
    Used only as a tie-breaker between non-terminal lines — terminal
    wins/losses dominate through the search itself."""
    y, x = last_move // game.SIZE, last_move % game.SIZE
    best = 0
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        run, ends = _run_and_open_ends(board, y, x, dy, dx, mover)
        # _run_and_open_ends counts the stone at (y, x) as if played;
        # it is already on the board here, so the run is exact.
        best = max(best, _WEIGHTS.get((min(run, 5), ends), 0))
    return best / (best + 5000.0)


class HeuristicMCTSPlayer:
    """PUCT search with heuristic priors and a heuristic leaf value —
    no neural network. The middle rung of the eval ladder: strictly
    stronger than the 1-ply heuristic player (it looks ahead and finds
    forced wins within its search horizon) but far weaker than a real
    trained net with search, because its leaf evaluation is a crude
    local pattern score, not learned judgement.

    Priors come from :func:`score_move` (softmax over log-scores);
    leaf values are terminal win/loss (exact) plus the local attack
    score above. Immediate wins are taken and immediate losses blocked
    before searching, exactly like :class:`HeuristicPlayer`.

    Determinism: move choice is the most-visited root child, with
    ties broken by the seeded RNG. The RNG stream advances across
    games, so repeated games against a deterministic checkpoint still
    diverge — without it, a deterministic-vs-deterministic ladder
    would replay the same game every time.
    """

    def __init__(self, sims: int = 100, c_puct: float = 1.5, seed: int = 0) -> None:
        self.sims = sims
        self.c_puct = c_puct
        self.rng = np.random.default_rng(seed)

    # -- priors ---------------------------------------------------------

    def _priors(self, board: np.ndarray, player: int) -> dict[int, float]:
        moves = _candidate_moves(board)
        scores = np.array(
            [math.log1p(max(0, score_move(board, m, player))) for m in moves],
            dtype=np.float64,
        )
        # Softmax over log-scores with temperature 2: strong pattern
        # moves get most of the prior mass, weak moves keep a little.
        scaled = scores / 2.0
        scaled -= scaled.max()
        weights = np.exp(scaled)
        weights /= weights.sum()
        return {m: float(w) for m, w in zip(moves, weights)}

    # -- search ---------------------------------------------------------

    def _select_child(self, node: Node) -> tuple[int, Node]:
        best_move, best_child, best_score = -1, None, -math.inf
        sqrt_parent = math.sqrt(max(1, node.visit_count))
        for move, child in node.children.items():
            u = self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            score = -child.q + u
            if score > best_score:
                best_move, best_child, best_score = move, child, score
        assert best_child is not None
        return best_move, best_child

    def _run_one(
        self, node: Node, board: np.ndarray, player: int, last_move: int | None
    ) -> float:
        """One simulation. Returns the value from the perspective of
        the player to move at ``node``."""
        if last_move is not None:
            if game.is_winning_stone(board, last_move, -player):
                return -1.0  # the player who just moved won
            if game.is_full(board):
                return 0.0
        if not node.expanded:
            priors = self._priors(board, player)
            for move, prior in priors.items():
                node.children[move] = Node(prior=prior)
            node.expanded = True
            if last_move is None:
                return 0.0
            # Leaf value: the attack the opponent just built, from this
            # player's (negative) perspective.
            return -_local_attack_value(board, last_move, -player)
        move, child = self._select_child(node)
        next_board = game.apply_move(board, move, player)
        value = -self._run_one(child, next_board, -player, move)
        child.visit_count += 1
        child.value_sum += -value
        return value

    def choose_move(self, board: np.ndarray, player: int) -> int:
        legal = game.legal_moves(board)
        if len(legal) == 0:
            raise ValueError("no legal moves")
        # 1. Take an immediate win. 2. Block an immediate loss.
        for move in legal:
            if game.is_winning_move(board, int(move), player):
                return int(move)
        for move in legal:
            if game.is_winning_move(board, int(move), -player):
                return int(move)
        root = Node()
        value = self._run_one(root, board, player, None)
        root.visit_count += 1
        root.value_sum += value
        for _ in range(max(0, self.sims - 1)):
            value = self._run_one(root, board, player, None)
            root.visit_count += 1
            root.value_sum += value
        best_visits = max(c.visit_count for c in root.children.values())
        best = [m for m, c in root.children.items() if c.visit_count == best_visits]
        if len(best) == 1:
            return best[0]
        return int(self.rng.choice(best))

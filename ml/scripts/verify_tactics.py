#!/usr/bin/env python3
"""Verify the tactical test suite is sound.

For every position in ``gomoku_ml/tactical_positions.json`` this checks,
exhaustively (not by trusting the author):

* the board is legal: stones on distinct empty points, no side has
  already won, every expected move is legal;
* ``win`` positions: at least one winning move exists for the side to
  move;
* ``block`` / ``trap-avoid`` positions: the listed squares are EXACTLY
  the moves after which the opponent has no immediate winning move
  (every other move lets the opponent win at once);
* ``double-threat`` (create) positions: the listed squares are EXACTLY
  the moves after which the mover has >= 2 winning moves (two
  independent threats the opponent cannot block with one move);
* ``defence`` positions: the listed squares are EXACTLY the moves after
  which the opponent has no immediate win and no reply that creates
  >= 2 winning moves;
* ``vcf`` positions: the documented forcing first moves really do
  start a forced win (VCF search), the documented trap moves do not,
  and the total number of forcing first moves is small (1-10) so the
  position discriminates — a random move almost never passes.

Exit 0 when every position verifies, 1 otherwise (with details).

Usage:
    python ml/scripts/verify_tactics.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game  # noqa: E402
from gomoku_ml.evaluate import (  # noqa: E402
    board_from_position,
    four_creating_moves,
    load_tactical_suite,
    starts_forced_win,
    winning_moves,
)

# Documented forcing first moves and traps per VCF position (kept in
# sync with the position descriptions in tactical_positions.json).
VCF_MAIN_LINES = {
    "vcf-1": [(9, 6), (10, 6), (8, 3)],
    "vcf-2": [(9, 9), (10, 10), (8, 5)],
}
VCF_TRAPS = {
    "vcf-1": [(8, 7), (8, 2)],
    "vcf-2": [(8, 9), (8, 4)],
}


def _fmt(move: int) -> str:
    x, y = game.move_to_xy(move)
    return f"({x},{y})"


def _near_moves_opponent(board, player: int) -> list[int]:
    """Empty points within Chebyshev distance 2 of any ``player``
    stone — the only replies that can create a threat."""
    near: set[int] = set()
    for y, x in zip(*((board == player).nonzero())):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return sorted(near)


def verify_position(pos: dict) -> list[str]:
    """Return a list of problems (empty = sound)."""
    problems: list[str] = []
    pid = pos["id"]
    board = board_from_position(pos)
    to_move = pos["to_move"]
    over, win = game.game_over(board)
    if over:
        problems.append(f"{pid}: position is already over (winner={win})")
        return problems
    expect = pos["expect"]
    if expect["type"] == "moves":
        expected = {game.xy_to_move(x, y) for x, y in expect["moves"]}
        for m in expected:
            if not game.is_legal(board, m):
                problems.append(f"{pid}: expected move {_fmt(m)} is illegal")
    else:
        expected = set()

    category = pos["category"]
    if expect["type"] == "win":
        wins = winning_moves(board, to_move)
        if not wins:
            problems.append(f"{pid}: 'win' position has no winning move")
    elif category in ("block", "trap-avoid"):
        # Exactly the moves after which the opponent has no winning move.
        good = set()
        for m in game.legal_moves(board):
            b2 = game.apply_move(board, int(m), to_move)
            if not winning_moves(b2, -to_move):
                good.add(int(m))
        if good != expected:
            problems.append(
                f"{pid}: moves leaving the opponent without a win are "
                f"{sorted(_fmt(m) for m in good)}, expected "
                f"{sorted(_fmt(m) for m in expected)}"
            )
    elif category == "double-threat":
        good = set()
        for m in game.legal_moves(board):
            b2 = game.apply_move(board, int(m), to_move)
            if len(winning_moves(b2, to_move)) >= 2:
                good.add(int(m))
        if good != expected:
            problems.append(
                f"{pid}: moves creating >=2 winning moves are "
                f"{sorted(_fmt(m) for m in good)}, expected "
                f"{sorted(_fmt(m) for m in expected)}"
            )
    elif category == "defence":
        good = set()
        for m in game.legal_moves(board):
            b2 = game.apply_move(board, int(m), to_move)
            if winning_moves(b2, -to_move):
                continue  # opponent wins at once
            defended = True
            for w in _near_moves_opponent(b2, -to_move):
                b3 = game.apply_move(b2, w, -to_move)
                if len(winning_moves(b3, -to_move)) >= 2:
                    defended = False
                    break
            if defended:
                good.add(int(m))
        if good != expected:
            problems.append(
                f"{pid}: moves that defend both threats are "
                f"{sorted(_fmt(m) for m in good)}, expected "
                f"{sorted(_fmt(m) for m in expected)}"
            )
    elif category == "vcf":
        for x, y in VCF_MAIN_LINES.get(pid, []):
            m = game.xy_to_move(x, y)
            if not starts_forced_win(board, to_move, m):
                problems.append(
                    f"{pid}: documented forcing move {_fmt(m)} does not "
                    "start a forced win"
                )
        for x, y in VCF_TRAPS.get(pid, []):
            m = game.xy_to_move(x, y)
            if starts_forced_win(board, to_move, m):
                problems.append(
                    f"{pid}: documented trap move {_fmt(m)} unexpectedly "
                    "starts a forced win"
                )
        forcing = [
            m for m in four_creating_moves(board, to_move)
            if starts_forced_win(board, to_move, m)
        ]
        if not (1 <= len(forcing) <= 10):
            problems.append(
                f"{pid}: {len(forcing)} forcing first moves "
                f"{sorted(_fmt(m) for m in forcing)} — expected a small "
                "number (1-10) so the position discriminates"
            )
    else:
        problems.append(f"{pid}: unknown category {category!r}")
    return problems


def main() -> int:
    suite = load_tactical_suite()
    positions = suite["positions"]
    print(f"verifying {len(positions)} tactical positions...", file=sys.stderr)
    all_problems: list[str] = []
    for pos in positions:
        problems = verify_position(pos)
        status = "OK  " if not problems else "FAIL"
        print(f"  {status} {pos['id']}", file=sys.stderr)
        all_problems.extend(problems)
    if all_problems:
        print(f"\n{len(all_problems)} problem(s):", file=sys.stderr)
        for p in all_problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"\nall {len(positions)} positions verified sound.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

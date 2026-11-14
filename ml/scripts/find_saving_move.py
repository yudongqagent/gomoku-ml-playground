#!/usr/bin/env python3
"""Find saving moves at a critical position by exhaustive enumeration.

For a position (reached by replaying a recorded game to a ply),
enumerate every candidate move for the side to move and report,
for each, the opponent's winning moves and *threat moves* (moves
that would give the opponent >= 2 winning moves, i.e. create an
open four or a double four) AFTER the candidate. A move that leaves
the opponent with 0 winning moves AND 0 threat moves is a clean
save; a move that leaves <= 1 winning move and 0 threat moves
holds with a forced block next turn. A move that gives the side to
move its own >= 2 winning moves is a counter-win.

Threat moves are only counted when the opponent does NOT already
have >= 2 winning moves on the board (in that case the threat
already exists and every move trivially "creates" it).

Usage:
    python ml/scripts/find_saving_move.py <batch.json> --game 2 --ply 28
    python ml/scripts/find_saving_move.py <batch.json> --game 5 --ply 19 --topk 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, scoring  # noqa: E402


def threat_moves(board, player):
    """Moves after which ``player`` has >= 2 winning moves.

    Returns [] if ``player`` already has >= 2 winning moves (the
    threat already exists; nothing is being created).
    """
    if len(scoring._winning_moves(board, player)) >= 2:
        return []
    out = []
    for m in game.legal_moves(board):
        b2 = game.apply_move(board, int(m), player)
        if len(scoring._winning_moves(b2, player)) >= 2:
            out.append(int(m))
    return out


def candidates(board):
    """Plausible moves: empty squares within Chebyshev distance 2
    of an existing stone (the MCTS candidate set)."""
    stones = [(int(y), int(x)) for y in range(15) for x in range(15)
              if board[y, x] != game.EMPTY]
    out = set()
    for y, x in stones:
        for dy in (-2, -1, 0, 1, 2):
            for dx in (-2, -1, 0, 1, 2):
                ny, nx = y + dy, x + dx
                if 0 <= ny < 15 and 0 <= nx < 15 and board[ny, nx] == game.EMPTY:
                    out.add(ny * 15 + nx)
    return sorted(out)


def analyse(board, player):
    """For every candidate move, the opponent's threats afterwards.

    Returns a list of (move, opp_wins, opp_threats, own_wins,
    own_threats) sorted by (opp_threats, opp_wins) ascending.
    """
    opp = -player
    rows = []
    for m in candidates(board):
        b2 = game.apply_move(board, m, player)
        ow = scoring._winning_moves(b2, opp)
        ot = threat_moves(b2, opp)
        yw = scoring._winning_moves(b2, player)
        yt = threat_moves(b2, player)
        rows.append((m, len(ow), len(ot), len(yw), len(yt)))
    rows.sort(key=lambda r: (r[2], r[1], -r[4], -r[3]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch")
    parser.add_argument("--game", type=int, required=True)
    parser.add_argument("--ply", type=int, required=True,
                        help="ply index of the side-to-move position "
                        "(the board BEFORE ply ``ply`` is played)")
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    d = json.load(open(args.batch))
    rec = d["records"][args.game]
    moves = rec["moves"]
    board = game.empty_board()
    player = game.BLACK
    for m in moves[:args.ply]:
        board = game.apply_move(board, m, player)
        player = -player
    a_black = (rec["a_colour"] == "black")
    v8 = game.BLACK if a_black else game.WHITE
    assert player == v8, "ply is not a v8 turn"
    actual = moves[args.ply]
    print(f"game {args.game} ply {args.ply}: v8 ({rec['a_colour']}) to "
          f"move, actually played ({actual // 15},{actual % 15})")
    print(f"  opponent winning moves now: "
          f"{len(scoring._winning_moves(board, -v8))}, threat moves now: "
          f"{len(threat_moves(board, -v8))}")
    rows = analyse(board, v8)
    print(f"  best {args.topk} moves by (opp threats, opp wins) after:")
    for m, ow, ot, yw, yt in rows[:args.topk]:
        mark = " <-- ACTUAL" if m == actual else ""
        print(f"    ({m // 15},{m % 15}): opp wins {ow}, opp threats {ot}, "
              f"own wins {yw}, own threats {yt}{mark}")
    actual_row = [r for r in rows if r[0] == actual]
    if actual_row and actual_row[0] not in rows[:args.topk]:
        m, ow, ot, yw, yt = actual_row[0]
        print(f"    ({m // 15},{m % 15}): opp wins {ow}, opp threats {ot}, "
              f"own wins {yw}, own threats {yt} <-- ACTUAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())

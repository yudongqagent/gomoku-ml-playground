"""Find the losing move in v58's losses to v0 (battery games 53, 79).

For each White turn, in order:
  - Does the PLAYED move allow Black a TSS win on the next turn?
  - If yes, does ANY White move prevent it (Black has no TSS win after
    that move)? The first turn where the played move allows a Black
    TSS win while a preventing move exists is the losing move.
  - At that turn, compare v52's and v58's choices: does v52 play a
    preventing move while v58 does not? That is the term-causation test
    (not confounded by the symmetric ply-1 tie-break).

Usage:
    PYTHONPATH=ml python ml/scripts/workup_v58_v0_prevention.py \
        research/iterations/batch-v58-vs-v0.json
"""
import json
import sys

from gomoku_ml import game, mcts_heuristic, scoring


def black_tss_after(board, white_move):
    """Black's TSS winning move after White plays white_move (or None)."""
    after = game.apply_move(board, int(white_move), game.WHITE)
    return mcts_heuristic.tss_winning_move(after, game.BLACK)


def engine_choice(version, board, player, seed=100):
    player_obj = scoring.get_player(version, seed=seed)
    return player_obj.choose_move(board, player)


def main(path):
    with open(path) as f:
        battery = json.load(f)
    losses = [r for r in battery["records"] if r["winner_is_a"] is False]
    print(f"losses: {len(losses)}")
    for rec in losses:
        moves = rec["moves"]
        print(f"\n== game {rec['game']} (v58 {rec['a_colour']}, "
              f"{rec['plies']} plies) ==")
        board = game.empty_board()
        losing_turn = None
        for ply, mv in enumerate(moves):
            player = game.BLACK if ply % 2 == 0 else game.WHITE
            if player == game.WHITE:
                tss_after_played = black_tss_after(board, mv)
                if tss_after_played is not None:
                    # Played move allows a Black TSS win. Any preventer?
                    legal = game.legal_moves(board)
                    preventing = [
                        int(c) for c in legal
                        if black_tss_after(board, int(c)) is None
                    ]
                    print(f"  ply {ply}: played {mv} allows Black TSS "
                          f"win ({tss_after_played}); preventing moves: "
                          f"{len(preventing)}/{len(legal)}")
                    if preventing:
                        losing_turn = (ply, board.copy(), mv, preventing)
                        break
                    else:
                        print("    (no preventing move exists — "
                              "position already lost)")
            board = game.apply_move(board, mv, player)
        if losing_turn is None:
            print("  no losing turn found by this test (White never had "
                  "a move that prevents a Black TSS win while playing "
                  "one that allows it)")
            continue
        ply, board_at, played, preventing = losing_turn
        c52 = engine_choice(52, board_at, game.WHITE)
        c58 = engine_choice(58, board_at, game.WHITE)
        print(f"  LOSING TURN ply {ply}: played {played}; v52 would "
              f"play {c52} ({'PREVENTS' if c52 in preventing else 'does not prevent'}); "
              f"v58 replays {c58} "
              f"({'PREVENTS' if c58 in preventing else 'does not prevent'})")
        print(f"  preventing moves: {preventing[:12]}")
        if c52 in preventing and c58 not in preventing:
            print("  VERDICT: TERM-CAUSED — v52 prevents, v58 does not")
        elif c52 not in preventing and c58 not in preventing:
            print("  VERDICT: NOT term-caused — neither engine prevents "
                  "(v52 would lose this game too)")
        else:
            print("  VERDICT: v58 prevents on replay — the loss is a "
                  "seed/tie-break artifact, not a systematic failure")


if __name__ == "__main__":
    main(sys.argv[1])

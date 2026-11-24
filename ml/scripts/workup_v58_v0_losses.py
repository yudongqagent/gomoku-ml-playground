"""Work up v58's losses to v0 in the battery (games 53 and 79).

v0 is a greedy heuristic player: it does not run TSS, so a Black TSS
fire is not the losing moment. The causal test is move substitution:
  1. Scan every White turn of each loss game; run v52 and v58 (fresh,
     seed 100) on the position before White's move. Record divergences.
  2. At the FIRST divergence, substitute v52's move and play out the
     rest of the game with fresh v58 (White) vs v0 (Black), seed 100.
     If v58 wins or draws the continuation, the divergence was the
     losing move and the term caused it.
  3. Also record, for each loss game, the first ply where Black holds
     a TSS win (informational: v0 does not play it).

Usage:
    PYTHONPATH=ml python ml/scripts/workup_v58_v0_losses.py \
        research/iterations/batch-v58-vs-v0.json
"""
import json
import sys

from gomoku_ml import baseline, game, mcts_heuristic, scoring


def engine_choice(version, board, player, seed=100):
    """The engine's move on the position (fresh engine, seeded)."""
    player_obj = scoring.get_player(version, seed=seed)
    return player_obj.choose_move(board, player)


def first_black_tss_fire(moves):
    """Replay; return the first ply where Black holds a TSS win."""
    board = game.empty_board()
    for ply, mv in enumerate(moves):
        player = game.BLACK if ply % 2 == 0 else game.WHITE
        if player == game.BLACK:
            tss = mcts_heuristic.tss_winning_move(board, player)
            if tss is not None:
                return ply, tss, moves[ply] == tss
        board = game.apply_move(board, mv, player)
    return None, None, None


def play_out(board, next_player, max_plies=225):
    """Play out from a position: v58 as White, v0 as Black (seed 100).

    Returns the winner ('black', 'white', or 'draw').
    """
    v58 = scoring.get_player(58, seed=100)
    v0 = baseline.HeuristicPlayer(seed=100)
    player = next_player
    plies = 0
    while plies < max_plies:
        legal = game.legal_moves(board)
        if len(legal) == 0:
            return "draw"
        if player == game.WHITE:
            mv = v58.choose_move(board, player)
        else:
            mv = v0.choose_move(board, player)
        board = game.apply_move(board, int(mv), player)
        if game.is_winning_move(board, int(mv), player):
            return "black" if player == game.BLACK else "white"
        player = game.WHITE if player == game.BLACK else game.BLACK
        plies += 1
    return "draw"


def main(path):
    with open(path) as f:
        battery = json.load(f)
    losses = [r for r in battery["records"] if r["winner_is_a"] is False]
    print(f"losses: {len(losses)}")
    for rec in losses:
        moves = rec["moves"]
        print(f"\n== game {rec['game']} (v58 {rec['a_colour']}, "
              f"{rec['plies']} plies) ==")
        fire_ply, fire_move, played_it = first_black_tss_fire(moves)
        print(f"  Black's TSS first fires at ply {fire_ply} "
              f"(move {fire_move}); v0 played it: {played_it}")
        # Scan White turns for the first v52/v58 divergence.
        board = game.empty_board()
        first_div = None
        for ply, mv in enumerate(moves):
            player = game.BLACK if ply % 2 == 0 else game.WHITE
            if player == game.WHITE:
                c52 = engine_choice(52, board, player)
                c58 = engine_choice(58, board, player)
                if c52 != c58:
                    first_div = (ply, board.copy(), c52, c58, mv)
                    print(f"  first v52/v58 divergence at ply {ply}: "
                          f"v52 {c52}, v58 {c58}, played {mv}")
                    break
            board = game.apply_move(board, mv, player)
        if first_div is None:
            print("  v52 and v58 agree on every White turn — the loss "
                  "is NOT term-caused (v52 would lose it too)")
            continue
        ply, board_at, c52, c58, played = first_div
        # Substitute v52's move and play out with v58 vs v0.
        after = game.apply_move(board_at, int(c52), game.WHITE)
        outcome = play_out(after, game.BLACK)
        print(f"  substitution: v52's move {c52} at ply {ply}, then "
              f"v58-vs-v0 continuation -> {outcome}")
        if outcome in ("white", "draw"):
            print("  VERDICT: the divergence at ply "
                  f"{ply} IS the losing move (term-caused)")
        else:
            print("  VERDICT: substituting v52's move does not save "
                  "the game — the loss is not (only) term-caused at "
                  "this turn")


if __name__ == "__main__":
    main(sys.argv[1])

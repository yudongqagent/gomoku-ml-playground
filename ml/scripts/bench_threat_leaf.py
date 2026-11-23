#!/usr/bin/env python3
"""Cost + difference bench for the v58 threat-potential leaf term.

Measures, on real-game positions (every ply of the first N committed
v52-vs-v27 battery games):

* per-leaf-call time of ``_rollout_leaf_value`` (v52) vs
  ``_rollout_threat_leaf_value`` (v58) — mean / median / p90 and the
  ratio (the pre-registered cost budget in iter-58 §11.4 rests on
  this bench, the iter-52 §1.4 precedent: the bench runs BEFORE the
  pre-registration is committed);
* the leaf-value difference rate: fraction of sampled positions
  where the v58 leaf differs from the v52 leaf (and the mean
  |difference|) — if the term never changes the leaf value, the
  mechanism is dead on arrival;
* the forcing-count difference distribution at rollout ends (how
  often the term is nonzero);
* move-choice agreement: full 100-sim searches with the v52 and
  v58 engines on sampled midgame positions (TSS-silent ones only,
  so the difference is the leaf, not the hook).

Usage:
    PYTHONPATH=ml python ml/scripts/bench_threat_leaf.py [--games N]
        [--searches M] [--out PATH]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

from gomoku_ml import game, mcts_heuristic, scoring  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCH_V52 = os.path.join(
    REPO_ROOT, "research", "iterations", "batch-v52-vs-v27.json"
)


def _battery_positions(n_games=10):
    """Every ply position of the first ``n_games`` committed
    v52-vs-v27 battery games (both colours to move as they occur)."""
    with open(BATCH_V52, encoding="utf-8") as f:
        batch = json.load(f)
    out = []
    for rec in batch["records"][:n_games]:
        moves = rec["moves"]
        if isinstance(moves, str):
            moves = ast.literal_eval(moves)
        board = game.empty_board()
        for ply, mv in enumerate(moves):
            player = game.BLACK if ply % 2 == 0 else game.WHITE
            out.append((board.copy(), player))
            board = game.apply_move(board, mv, player)
    return out


def _rollout_end(board, player):
    """Replay the 2-ply rollout; return (end_board, end_player,
    terminal_or_None). Mirrors _rollout_leaf_value's rollout."""
    value = mcts_heuristic._terminal_leaf_value(board, player)
    if value != 0.0:
        return board, player, value
    cur = board.copy()
    p = player
    for i in range(mcts_heuristic._ROLLOUT_PLIES):
        move, term = mcts_heuristic._rollout_move(cur, p)
        if term is not None:
            return cur, p, (term if i % 2 == 0 else -term)
        cur = game.apply_move(cur, move, p)
        p = -p
    return cur, p, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--searches", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    positions = _battery_positions(args.games)
    print(f"positions: {len(positions)} (first {args.games} battery games)")

    # --- leaf cost + difference rate -------------------------------------
    t52, t58 = [], []
    diffs = []
    n_diff = 0
    n_quiet_end = 0
    n_term_nonzero = 0
    fdiff_hist: dict[int, int] = {}
    for board, player in positions:
        t0 = time.perf_counter()
        v52 = mcts_heuristic._rollout_leaf_value(board, player)
        t52.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        v58 = mcts_heuristic._rollout_threat_leaf_value(board, player)
        t58.append(time.perf_counter() - t0)
        if abs(v52 - v58) > 1e-9:
            n_diff += 1
            diffs.append(abs(v52 - v58))
        # Rollout-end forcing-count difference (the term's input).
        end_board, end_player, term = _rollout_end(board, player)
        if term is None:
            if scoring._winning_moves(end_board, end_player):
                continue
            opp_wins = scoring._winning_moves(end_board, -end_player)
            if len(opp_wins) >= 2:
                continue
            if len(opp_wins) == 1:
                continue  # must-block: term skipped
            n_quiet_end += 1
            b1, w1 = mcts_heuristic.kernel_v1.attack_v1_both(end_board)
            fb, fw = mcts_heuristic._forcing_counts_both(end_board, b1, w1)
            fd = (fb - fw) if end_player == game.BLACK else (fw - fb)
            fdiff_hist[fd] = fdiff_hist.get(fd, 0) + 1
            if fd != 0:
                n_term_nonzero += 1

    t52 = np.array(t52) * 1000
    t58 = np.array(t58) * 1000
    print(f"v52 leaf: mean {t52.mean():.3f}ms median {np.median(t52):.3f}ms "
          f"p90 {np.percentile(t52, 90):.3f}ms max {t52.max():.3f}ms")
    print(f"v58 leaf: mean {t58.mean():.3f}ms median {np.median(t58):.3f}ms "
          f"p90 {np.percentile(t58, 90):.3f}ms max {t58.max():.3f}ms")
    print(f"cost ratio (mean): {t58.mean() / t52.mean():.3f}x")
    print(f"leaf difference rate: {n_diff}/{len(positions)} = "
          f"{n_diff / len(positions):.3f} "
          f"(mean |diff| where differing: "
          f"{float(np.mean(diffs)) if diffs else 0.0:.4f})")
    print(f"quiet rollout ends: {n_quiet_end}/{len(positions)}; "
          f"term nonzero on {n_term_nonzero}/{n_quiet_end} "
          f"= {n_term_nonzero / max(n_quiet_end, 1):.3f}")
    print(f"forcing-diff histogram at quiet rollout ends: "
          f"{dict(sorted(fdiff_hist.items()))}")

    # --- move-choice agreement (full 100-sim searches) -------------------
    rng = np.random.default_rng(777)
    midgame = [
        (b, p) for (b, p) in positions
        if 10 <= int((b != game.EMPTY).sum()) <= 40
    ]
    idx = rng.choice(len(midgame), size=min(args.searches, len(midgame)),
                     replace=False)
    n_tss_silent = 0
    n_agree = 0
    n_diff_moves = 0
    for i in idx:
        board, player = midgame[int(i)]
        if mcts_heuristic.tss_winning_move(board, player) is not None:
            continue  # TSS fires: both engines play the TSS move
        n_tss_silent += 1
        e52 = scoring.get_player(52, seed=100)
        e58 = scoring.get_player(58, seed=100)
        m52 = e52.choose_move(board.copy(), player)
        m58 = e58.choose_move(board.copy(), player)
        if m52 == m58:
            n_agree += 1
        else:
            n_diff_moves += 1
    print(f"move choice on {n_tss_silent} TSS-silent midgame positions "
          f"(100 sims, seed 100): agree {n_agree}, differ {n_diff_moves}")

    result = {
        "positions": len(positions),
        "v52_leaf_ms": {"mean": float(t52.mean()),
                        "median": float(np.median(t52)),
                        "p90": float(np.percentile(t52, 90)),
                        "max": float(t52.max())},
        "v58_leaf_ms": {"mean": float(t58.mean()),
                        "median": float(np.median(t58)),
                        "p90": float(np.percentile(t58, 90)),
                        "max": float(t58.max())},
        "cost_ratio_mean": float(t58.mean() / t52.mean()),
        "leaf_difference_rate": n_diff / len(positions),
        "leaf_difference_n": n_diff,
        "mean_abs_diff_where_differing": float(np.mean(diffs)) if diffs else 0.0,
        "quiet_rollout_ends": n_quiet_end,
        "term_nonzero": n_term_nonzero,
        "forcing_diff_hist": {str(k): v for k, v in sorted(fdiff_hist.items())},
        "move_choice": {"tss_silent": n_tss_silent, "agree": n_agree,
                        "differ": n_diff_moves},
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

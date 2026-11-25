"""Tests for per-game seeding + uniqueness audit (ML batch 3 tooling).

Batch 10 found a resume seed-replay bug: the old drivers created the
engines once with fixed seeds, so on auto-resume their RNG streams
restarted and the resumed segment replayed the prefix when
segment-start parity aligned (v34: games 0-33 = 66-99, 63 unique in
100). Worse, engine determinism alone collapses vs-v0 legs to 41-45
unique games even with NO resume.

Per-game seeding fixes both: before every game ``i`` each engine is
reseeded with ``base_seed + i`` and the swap placement RNG is rebuilt
from ``placement_seed + i``, so each game is determined by its index
alone. The key property under test: a resumed run reproduces the
uninterrupted run EXACTLY (full move lists, not just the score).
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from gomoku_ml import baseline, evaluate, game, research, scoring

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "ml", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import selfplay_batch  # noqa: E402


def test_scoring_player_reseed_matches_fresh_rng():
    """reseed(seed) makes the next draw equal a fresh default_rng(seed)."""
    player = scoring.get_player(1, seed=999)
    player.reseed(42)
    assert int(player.rng.integers(0, 1000)) == int(
        np.random.default_rng(42).integers(0, 1000)
    )
    # Reseeding again to the same seed reproduces the same stream.
    player.reseed(42)
    assert int(player.rng.integers(0, 1000)) == int(
        np.random.default_rng(42).integers(0, 1000)
    )


def test_scoring_player_reseed_updates_mcts_rng():
    """For an MCTS version the inner engine shares the reset RNG."""
    player = scoring.get_player(7, seed=999)
    assert player._mcts is not None
    player.reseed(7)
    assert player._mcts.rng is player.rng
    assert int(player._mcts.rng.integers(0, 1000)) == int(
        np.random.default_rng(7).integers(0, 1000)
    )


def test_reseed_player_baseline():
    """Baseline players (frozen module, no reseed method) reset via
    their public rng attribute."""
    player = baseline.HeuristicPlayer(seed=999)
    selfplay_batch.reseed_player(player, 123)
    assert int(player.rng.integers(0, 1000)) == int(
        np.random.default_rng(123).integers(0, 1000)
    )
    mcts = baseline.HeuristicMCTSPlayer(sims=5, seed=999)
    selfplay_batch.reseed_player(mcts, 5)
    assert int(mcts.rng.integers(0, 1000)) == int(
        np.random.default_rng(5).integers(0, 1000)
    )


def test_uniqueness_audit():
    """Distinct-game count and de-dup W/L/D on synthetic records."""
    records = [
        {"moves": [1, 2, 3], "winner_is_a": True},
        {"moves": [1, 2, 3], "winner_is_a": True},  # duplicate line
        {"moves": [1, 2, 4], "winner_is_a": False},
        {"moves": [9, 9], "winner_is_a": None},  # draw
        {"moves": [9, 9], "winner_is_a": None},  # duplicate draw
    ]
    audit = evaluate.uniqueness_audit(records)
    assert audit["unique_games"] == 3
    assert audit["total_games"] == 5
    assert audit["dedup_wins"] == 1
    assert audit["dedup_losses"] == 1
    assert audit["dedup_draws"] == 1
    assert audit["dedup_win_rate"] == 0.5
    assert audit["unique_games"] == 0 or audit["dedup_win_rate_ci_95"][0] <= 0.5 <= (
        audit["dedup_win_rate_ci_95"][1]
    )


def test_uniqueness_gate_passes_at_threshold():
    """Evaluation audit gate: 60% distinct games passes, below fails.

    The threshold is 0.6 (see UNIQUENESS_GATE_THRESHOLD): below 60%
    distinct the effective sample size is <60% of nominal, so the CI
    width is understated by more than sqrt(1/0.6) ~= 1.29x — at v63's
    31% it was understated 1.7x.
    """
    # Exactly 60% distinct (3/5) passes; 40% (2/5) fails.
    passing = [{"moves": [1]}, {"moves": [2]}, {"moves": [3]},
               {"moves": [1]}, {"moves": [2]}]
    gate = evaluate.uniqueness_gate(passing)
    assert gate["unique_games"] == 3
    assert gate["total_games"] == 5
    assert gate["required_unique"] == 3
    assert gate["pass"] is True
    failing = [{"moves": [1]}, {"moves": [2]},
               {"moves": [1]}, {"moves": [2]}, {"moves": [1]}]
    gate = evaluate.uniqueness_gate(failing)
    assert gate["unique_games"] == 2
    assert gate["pass"] is False
    assert "INVALID" in gate["note"]
    # Empty record list is not a pass (nothing was measured).
    assert evaluate.uniqueness_gate([])["pass"] is False
    # A custom threshold is honoured.
    assert evaluate.uniqueness_gate(passing, threshold=0.8)["pass"] is False


def _args(**over):
    kw = dict(
        engine_a="v1",
        engine_b="v0",
        games=6,
        swap_rule=False,
        placement="balanced",
        placement_seed=11,
        judge="tempo",
        seed_a=100,
        seed_b=200,
        out="",
    )
    kw.update(over)
    return argparse.Namespace(**kw)


def _play_prefix(args, n_games):
    """Play the first ``n_games`` of a batch in-process, using the
    driver's own functions, and return the flushed-partial summary
    dict (complete=False) — exactly what the driver writes after
    game ``n_games``.

    Mirrors the driver's defaults (evaluation audit): normal-rules
    games start from the balanced openings set and swap games from
    the curated swap openings set — the partial's openings hashes
    must match the CLI's, or the resume check rejects it.
    """
    import time

    player_a, choose_a, name_a = selfplay_batch.resolve_engine(
        args.engine_a, args.seed_a
    )
    player_b, choose_b, name_b = selfplay_batch.resolve_engine(
        args.engine_b, args.seed_b
    )
    openings = None
    openings_hash = None
    swap_openings = None
    swap_openings_hash = None
    if args.swap_rule:
        swap_openings, swap_openings_hash = selfplay_batch.load_openings(
            selfplay_batch.common.default_swap_openings_path())
    else:
        openings, openings_hash = selfplay_batch.load_openings(
            selfplay_batch.common.default_openings_path())
    games: list[dict] = []
    wins = losses = draws = 0
    t0 = time.time()
    for i in range(n_games):
        selfplay_batch.reseed_player(player_a, args.seed_a + i)
        selfplay_batch.reseed_player(player_b, args.seed_b + i)
        rec: dict = {"game": i}
        if args.swap_rule:
            a_opens = (i % 2 == 0)
            opener, second = (
                (choose_a, choose_b) if a_opens else (choose_b, choose_a)
            )
            opening = swap_openings[i % len(swap_openings)]
            board, _ = selfplay_batch.apply_opening(opening)
            gap, bb, bw = research.opening_gap(
                board, scoring.score_attack
            )
            second_colour = game.choose_swap_color(
                board, scoring.score_attack, judge=args.judge
            )
            if second_colour == game.BLACK:
                choose_black, choose_white = second, opener
            else:
                choose_black, choose_white = opener, second
            winner, moves = selfplay_batch.play_recorded_game(
                choose_black, choose_white, board, game.WHITE
            )
            a_colour = (
                game.BLACK
                if (a_opens and second_colour == game.WHITE)
                or ((not a_opens) and second_colour == game.BLACK)
                else game.WHITE
            )
            rec.update({
                "opener": "A" if a_opens else "B",
                "opening": [int(m) for m in opening],
                "opening_gap": gap,
                "opening_best_black": bb,
                "opening_best_white": bw,
                "second_colour": (
                    "black" if second_colour == game.BLACK else "white"
                ),
                "moves": [int(m) for m in opening] + moves,
            })
        else:
            a_is_black = (i % 2 == 0)
            a_colour = game.BLACK if a_is_black else game.WHITE
            opening = openings[i % len(openings)]
            board, turn = selfplay_batch.apply_opening(opening)
            if a_is_black:
                winner, moves = selfplay_batch.play_recorded_game(
                    choose_a, choose_b, board, turn
                )
            else:
                winner, moves = selfplay_batch.play_recorded_game(
                    choose_b, choose_a, board, turn
                )
            # Mirror the driver's record shape: every non-swap
            # record carries its opening (the balanced set, by
            # default) and the full move list (opening first).
            rec["opening"] = [int(m) for m in opening]
            rec["moves"] = [int(m) for m in opening] + moves
        rec["a_colour"] = "black" if a_colour == game.BLACK else "white"
        rec["winner"] = (
            "black" if winner == game.BLACK
            else "white" if winner == game.WHITE else "draw"
        )
        rec["winner_is_a"] = (
            (winner == a_colour) if winner != game.EMPTY else None
        )
        rec["plies"] = len(rec["moves"])
        if winner == game.EMPTY:
            draws += 1
        elif winner == a_colour:
            wins += 1
        else:
            losses += 1
        games.append(rec)
    return selfplay_batch._summary(
        args, name_a, name_b, games, wins, losses, draws, t0, complete=False,
        openings_hash=openings_hash,
        openings_count=len(openings) if openings else None,
        swap_openings_hash=swap_openings_hash,
        swap_openings_count=(len(swap_openings)
                             if swap_openings else None),
    )


def _run_cli(out, swap=False):
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_DIR, "selfplay_batch.py"),
        "--engine-a", "v1",
        "--engine-b", "v0",
        "--games", "6",
        "--seed-a", "100",
        "--seed-b", "200",
        "--out", out,
    ]
    if swap:
        cmd += ["--swap-rule", "--placement", "balanced",
                "--placement-seed", "11", "--judge", "tempo"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def test_resume_reproduces_uninterrupted(tmp_path):
    """A resumed batch equals the uninterrupted batch exactly — full
    move lists, not just the score. This is the property the old
    scheme violated (the resumed segment replayed the prefix)."""
    ref = _run_cli(str(tmp_path / "ref.json"))
    partial = _play_prefix(_args(), 3)
    out = str(tmp_path / "resumed.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(partial, f)
    resumed = _run_cli(out)
    assert resumed["resumed"] is True
    assert resumed["resumed_from_game"] == 3
    assert resumed["complete"] is True
    assert resumed["records"] == ref["records"]
    assert (resumed["wins"], resumed["losses"], resumed["draws"]) == (
        ref["wins"],
        ref["losses"],
        ref["draws"],
    )
    assert resumed["seeding"] == "per-game"


def test_resume_reproduces_uninterrupted_swap(tmp_path):
    """Same property under swap rules (covers the per-game placement
    RNG: resumed openings must match the uninterrupted run's)."""
    ref = _run_cli(str(tmp_path / "ref-swap.json"), swap=True)
    partial = _play_prefix(_args(swap_rule=True), 3)
    out = str(tmp_path / "resumed-swap.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(partial, f)
    resumed = _run_cli(out, swap=True)
    assert resumed["resumed"] is True
    assert resumed["records"] == ref["records"]
    assert [r["opening"] for r in resumed["records"]] == [
        r["opening"] for r in ref["records"]
    ]


def test_per_game_seeding_diversifies(tmp_path):
    """A 10-game v1-vs-v0 batch has (nearly) all distinct games.
    Under the old scheme the vs-v0 legs collapsed to 41-45 unique
    in 100; per-game seeding gives each game an independent tie-break
    stream, so lines diverge from the first tie."""
    out = str(tmp_path / "div.json")
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "selfplay_batch.py"),
         "--engine-a", "v1", "--engine-b", "v0", "--games", "10",
         "--seed-a", "100", "--seed-b", "200", "--out", out],
        capture_output=True, text=True, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    with open(out, encoding="utf-8") as f:
        result = json.load(f)
    assert result["uniqueness"]["unique_games"] >= 8

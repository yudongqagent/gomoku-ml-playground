"""Tests for the evaluation harness: Wilson CIs, ladder Elo, verdicts,
tactical suite, checkpoint-vs-checkpoint lookup, and report rendering."""

import json
import os
import subprocess
import sys

import pytest

from gomoku_ml import baseline, evaluate, game
from gomoku_ml.architectures import build_architecture
from gomoku_ml.mcts import MCTS

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SMOKE_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "runs",
    "20261112-080423-smoke", "checkpoints", "final.pt",
)


# --- Wilson CI ------------------------------------------------------------

def test_wilson_ci_known_values():
    # 0/10: upper bound ~0.278; 10/10 mirrors it; 5/10 is symmetric ~0.5.
    lo, hi = evaluate.wilson_ci(0, 10, 0)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.2776, abs=1e-3)
    lo, hi = evaluate.wilson_ci(10, 0, 0)
    assert lo == pytest.approx(0.7224, abs=1e-3)
    assert hi == pytest.approx(1.0, abs=1e-9)
    lo, hi = evaluate.wilson_ci(5, 5, 0)
    assert lo == pytest.approx(0.2365, abs=1e-3)
    assert hi == pytest.approx(0.7635, abs=1e-3)


def test_wilson_ci_invariants():
    # Interval always contains the point estimate and narrows with n.
    for wins, losses, draws in [(3, 1, 0), (30, 10, 0), (1, 1, 8), (0, 0, 4)]:
        lo, hi = evaluate.wilson_ci(wins, losses, draws)
        p = evaluate.win_rate(wins, losses, draws)
        assert 0.0 <= lo <= p <= hi <= 1.0
    lo4, hi4 = evaluate.wilson_ci(3, 1, 0)
    lo40, hi40 = evaluate.wilson_ci(30, 10, 0)
    assert hi40 - lo40 < hi4 - lo4  # more games -> tighter interval
    # No games: total ignorance.
    assert evaluate.wilson_ci(0, 0, 0) == (0.0, 1.0)


def test_win_rate_draws_count_half():
    assert evaluate.win_rate(1, 1, 2) == pytest.approx(0.5)
    assert evaluate.win_rate(0, 0, 0) == 0.0


# --- ladder Elo -----------------------------------------------------------

def test_ladder_elo_smoke_like_case():
    # All wins vs random, all losses vs heuristic: no finite estimate,
    # so the answer is the midpoint of the Wilson interval — between
    # the two anchors, with a wide interval (n=4 each).
    elo = evaluate.ladder_elo({
        "random": {"wins": 4, "losses": 0, "draws": 0},
        "heuristic": {"wins": 0, "losses": 4, "draws": 0},
    })
    assert elo["estimate"] == pytest.approx(-400, abs=10)
    lo, hi = elo["interval"]
    # 4/4 vs random -> Wilson lower bound 0.51 -> rating >= -793;
    # 0/4 vs heuristic -> Wilson upper bound 0.49 -> rating <= -7.
    assert lo == pytest.approx(-793, abs=10)
    assert hi == pytest.approx(-7, abs=10)
    assert elo["anchors"] == {"random": -800, "heuristic": 0}
    assert "arbitrary" in elo["caveat"]


def test_ladder_elo_mixed_case():
    # 6-4 vs heuristic (score 0.6) -> 400*log10(1.5) ~= +70.
    elo = evaluate.ladder_elo({
        "heuristic": {"wins": 6, "losses": 4, "draws": 0},
    })
    assert elo["estimate"] == pytest.approx(70.4, abs=1.0)


def test_ladder_elo_inconsistent_bounds_report_span():
    """40-0 vs random and 0-40 vs heuristic: the per-opponent bounds
    do not overlap (lower -393 > upper -407), so the interval is
    reported as the sorted span with interval_inconsistent flagged —
    never as an inverted [lo, hi] with lo > hi."""
    elo = evaluate.ladder_elo({
        "random": {"wins": 40, "losses": 0, "draws": 0},
        "heuristic": {"wins": 0, "losses": 40, "draws": 0},
    })
    assert elo["estimate"] == -400.0
    lo, hi = elo["interval"]
    assert lo <= hi
    assert lo == pytest.approx(-407.0, abs=1.0)
    assert hi == pytest.approx(-393.0, abs=1.0)
    assert elo["interval_inconsistent"] is True


def test_ladder_elo_ignores_unanchored_opponents():
    base = evaluate.ladder_elo({"heuristic": {"wins": 6, "losses": 4, "draws": 0}})
    with_mcts = evaluate.ladder_elo({
        "heuristic": {"wins": 6, "losses": 4, "draws": 0},
        "mcts": {"wins": 10, "losses": 0, "draws": 0},
        "checkpoint": {"wins": 0, "losses": 10, "draws": 0},
    })
    assert with_mcts["estimate"] == base["estimate"]


# --- verdict --------------------------------------------------------------

def _ladder(heu, mcts=None):
    out = {"heuristic": {"wins": heu[0], "losses": heu[1], "draws": heu[2],
                         "games": sum(heu)}}
    if mcts is not None:
        out["mcts"] = {"wins": mcts[0], "losses": mcts[1], "draws": mcts[2],
                       "games": sum(mcts)}
    return out


def test_verdict_tiers():
    assert evaluate.verdict_for(_ladder((1, 9, 0)), None)["tier"] == "below heuristic"
    assert evaluate.verdict_for(_ladder((5, 5, 0)), None)["tier"] == "around heuristic"
    assert evaluate.verdict_for(_ladder((8, 2, 0)), None)["tier"] == "beats heuristic"
    assert evaluate.verdict_for(
        _ladder((8, 2, 0), (3, 7, 0)), None
    )["tier"] == "beats heuristic, below search"
    # 50% vs search but too few games -> challenges, not good level.
    assert evaluate.verdict_for(
        _ladder((8, 2, 0), (5, 5, 0)), {"pass_rate": 0.9}
    )["tier"] == "challenges search"
    # 50% vs search over 40 games but tactical too low -> challenges.
    assert evaluate.verdict_for(
        _ladder((32, 8, 0), (20, 20, 0)), {"pass_rate": 0.5}
    )["tier"] == "challenges search"
    # Full definition met.
    v = evaluate.verdict_for(
        _ladder((32, 8, 0), (20, 20, 0)), {"pass_rate": 0.85}
    )
    assert v["tier"] == "good level (v1)"
    assert v["meets_good_level"] is True
    # No heuristic matchup at all.
    assert evaluate.verdict_for({"random": {"wins": 1, "losses": 0, "draws": 0}}, None)["tier"] == "unevaluated"


# --- tactical suite -------------------------------------------------------

def test_tactical_suite_loads_and_is_structurally_sound():
    suite = evaluate.load_tactical_suite()
    positions = suite["positions"]
    assert len(positions) == 20
    ids = set()
    for pos in positions:
        assert pos["id"] not in ids
        ids.add(pos["id"])
        assert pos["to_move"] in (1, -1)
        assert pos["expect"]["type"] in ("win", "moves", "vcf")
        board = evaluate.board_from_position(pos)
        over, _ = game.game_over(board)
        assert not over, f"{pos['id']}: position is already over"
        if pos["expect"]["type"] == "moves":
            for x, y in pos["expect"]["moves"]:
                assert game.is_legal(board, game.xy_to_move(x, y)), (
                    f"{pos['id']}: expected move ({x},{y}) is illegal"
                )


def _winning_moves(board, player):
    return [int(m) for m in game.legal_moves(board)
            if game.is_winning_move(board, int(m), player)]


def test_tactical_expected_moves_satisfy_their_property():
    """Each expected move does what the position claims (the exhaustive
    'exactly these moves' proof is ml/scripts/verify_tactics.py)."""
    suite = evaluate.load_tactical_suite()
    for pos in suite["positions"]:
        board = evaluate.board_from_position(pos)
        to_move = pos["to_move"]
        expect = pos["expect"]
        if expect["type"] == "win":
            assert _winning_moves(board, to_move), (
                f"{pos['id']}: no winning move exists"
            )
            continue
        if expect["type"] == "vcf":
            continue  # covered by test_tactical_vcf_positions_force_a_win
        for x, y in expect["moves"]:
            move = game.xy_to_move(x, y)
            b2 = game.apply_move(board, move, to_move)
            if pos["category"] in ("block", "trap-avoid"):
                assert not _winning_moves(b2, -to_move), (
                    f"{pos['id']}: after ({x},{y}) the opponent still wins"
                )
            elif pos["category"] == "double-threat":
                assert len(_winning_moves(b2, to_move)) >= 2, (
                    f"{pos['id']}: ({x},{y}) does not create two threats"
                )
            elif pos["category"] == "defence":
                assert not _winning_moves(b2, -to_move), (
                    f"{pos['id']}: after ({x},{y}) the opponent wins at once"
                )


def test_tactical_vcf_positions_force_a_win():
    """The documented VCF first moves start a forced win; the tempting
    four-creating traps do not (the defender blocks and the attack
    dies). The full soundness proof is ml/scripts/verify_tactics.py."""
    suite = evaluate.load_tactical_suite()
    by_id = {p["id"]: p for p in suite["positions"]}
    cases = {
        "vcf-1": {"good": [(9, 6), (10, 6), (8, 3)], "traps": [(8, 7), (8, 2)]},
        "vcf-2": {"good": [(9, 9), (10, 10), (8, 5)], "traps": [(8, 9), (8, 4)]},
    }
    for pid, moves in cases.items():
        pos = by_id[pid]
        board = evaluate.board_from_position(pos)
        to_move = pos["to_move"]
        assert pos["expect"]["type"] == "vcf"
        for x, y in moves["good"]:
            assert evaluate.starts_forced_win(
                board, to_move, game.xy_to_move(x, y)
            ), f"{pid}: ({x},{y}) does not force a win"
            assert evaluate.check_tactical_move(
                board, to_move, game.xy_to_move(x, y), pos["expect"]
            )
        for x, y in moves["traps"]:
            assert not evaluate.starts_forced_win(
                board, to_move, game.xy_to_move(x, y)
            ), f"{pid}: trap ({x},{y}) unexpectedly forces a win"
            assert not evaluate.check_tactical_move(
                board, to_move, game.xy_to_move(x, y), pos["expect"]
            )


def test_tactical_suite_heuristic_passes_and_random_fails():
    """Known pass/fail: the heuristic takes wins and blocks by
    construction, so it should pass nearly all positions; random play
    should fail nearly all of them."""
    suite = evaluate.load_tactical_suite()
    heu = evaluate.run_tactical_suite(
        lambda board, player: baseline.choose_move(board, player)
    )
    assert heu["total"] == 20
    assert heu["passed"] >= 15, (
        f"heuristic passed only {heu['passed']}/20: "
        f"{[r['id'] for r in heu['results'] if not r['passed']]}"
    )
    rng_player = baseline.RandomPlayer(seed=0)
    rnd = evaluate.run_tactical_suite(
        lambda board, player: rng_player.choose_move(board, player)
    )
    assert rnd["passed"] <= 8, f"random passed {rnd['passed']}/20?!"


def test_tactical_suite_mcts_chooser_runs():
    """The MCTS chooser runs on a small subset (tiny net, few sims) and
    returns well-formed results — an exercise of the code path, not a
    strength assertion."""
    model = build_architecture("resnet", {"channels": 8, "blocks": 1})
    mcts = MCTS(model)
    suite = evaluate.load_tactical_suite()
    subset = {"positions": suite["positions"][:2]}
    result = evaluate.run_tactical_suite(
        evaluate.mcts_tactical_chooser(mcts, 8), suite=subset
    )
    assert result["total"] == 2
    assert result["passed"] in (0, 1, 2)
    assert len(result["results"]) == 2
    assert all("chosen" in r and "passed" in r for r in result["results"])


# --- ladder ---------------------------------------------------------------

def test_run_ladder_shape_and_colour_split():
    model = build_architecture("resnet", {"channels": 8, "blocks": 1})
    mcts = MCTS(model)
    results = evaluate.run_ladder(
        mcts, num_simulations=4, games_per_opponent=4,
        opponents=("random",), seed=0,
    )
    r = results["random"]
    assert r["wins"] + r["losses"] + r["draws"] == 4
    assert r["games"] == 4
    assert 0.0 <= r["win_rate"] <= 1.0
    lo, hi = r["win_rate_ci_95"]
    assert 0.0 <= lo <= r["win_rate"] <= hi <= 1.0
    # Per-colour breakdown sums to the totals; 4 games = 2 as Black.
    assert r["as_black"]["wins"] + r["as_white"]["wins"] == r["wins"]
    black_games = sum(r["as_black"].values())
    white_games = sum(r["as_white"].values())
    assert black_games == 2 and white_games == 2


def test_heuristic_mcts_player_takes_wins_and_blocks():
    player = baseline.HeuristicMCTSPlayer(sims=10, seed=0)
    board = game.empty_board()
    for x in (3, 4, 5, 6):
        board[7, x] = game.BLACK
    assert player.choose_move(board, game.BLACK) in (
        game.xy_to_move(2, 7), game.xy_to_move(7, 7),
    )
    board = game.empty_board()
    for x in (3, 4, 5, 6):
        board[7, x] = game.WHITE
    board[7, 2] = game.BLACK
    assert player.choose_move(board, game.BLACK) == game.xy_to_move(7, 7)


def test_heuristic_mcts_player_finds_a_forced_win():
    """The search opponent should find the VCF-style forced win that
    the 1-ply heuristic only stumbles on by tie-break luck: with
    enough sims it must pick a forcing first move."""
    suite = evaluate.load_tactical_suite()
    pos = next(p for p in suite["positions"] if p["id"] == "vcf-1")
    board = evaluate.board_from_position(pos)
    player = baseline.HeuristicMCTSPlayer(sims=200, seed=0)
    move = player.choose_move(board, pos["to_move"])
    assert evaluate.starts_forced_win(board, pos["to_move"], move), (
        f"heuristic-MCTS chose {game.move_to_xy(move)}, which does not "
        "start a forced win"
    )


# --- previous checkpoint --------------------------------------------------

def test_find_previous_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ("checkpoint-100.pt", "checkpoint-200.pt", "best.pt", "final.pt"):
        (ckpt_dir / name).write_bytes(b"x")
    run_dir = str(tmp_path)
    # final.pt -> latest stepped checkpoint.
    assert evaluate.find_previous_checkpoint(
        run_dir, str(ckpt_dir / "final.pt")
    ) == str(ckpt_dir / "checkpoint-200.pt")
    # A stepped checkpoint -> the previous step.
    assert evaluate.find_previous_checkpoint(
        run_dir, str(ckpt_dir / "checkpoint-200.pt")
    ) == str(ckpt_dir / "checkpoint-100.pt")
    # The earliest stepped checkpoint has no previous.
    assert evaluate.find_previous_checkpoint(
        run_dir, str(ckpt_dir / "checkpoint-100.pt")
    ) is None
    # Without stepped checkpoints, best/final pair off.
    (ckpt_dir / "checkpoint-100.pt").unlink()
    (ckpt_dir / "checkpoint-200.pt").unlink()
    assert evaluate.find_previous_checkpoint(
        run_dir, str(ckpt_dir / "final.pt")
    ) == str(ckpt_dir / "best.pt")
    assert evaluate.find_previous_checkpoint(
        run_dir, str(ckpt_dir / "best.pt")
    ) == str(ckpt_dir / "final.pt")


# --- report ---------------------------------------------------------------

def test_build_and_render_report():
    ladder = {
        "random": {"wins": 4, "losses": 0, "draws": 0, "games": 4,
                   "win_rate": 1.0, "win_rate_ci_95": [0.4, 1.0],
                   "as_black": {"wins": 2, "losses": 0, "draws": 0},
                   "as_white": {"wins": 2, "losses": 0, "draws": 0},
                   "elo_diff_vs_opponent": None},
        "heuristic": {"wins": 0, "losses": 4, "draws": 0, "games": 4,
                      "win_rate": 0.0, "win_rate_ci_95": [0.0, 0.6],
                      "as_black": {"wins": 0, "losses": 2, "draws": 0},
                      "as_white": {"wins": 0, "losses": 2, "draws": 0},
                      "elo_diff_vs_opponent": None},
    }
    report = evaluate.build_eval_report(
        checkpoint="/tmp/ckpt.pt",
        ladder=ladder,
        tactical={"total": 20, "passed": 5, "pass_rate": 0.25,
                  "results": [{"id": "win-h-open4", "name": "x",
                               "category": "take-win", "passed": False,
                               "chosen": [0, 0],
                               "expect": {"type": "win"}}]},
        tactical_baseline={"total": 20, "passed": 19, "pass_rate": 0.95,
                           "results": []},
    )
    assert report["verdict"]["tier"] == "below heuristic"
    assert report["ladder_elo"]["estimate"] is not None
    assert "relative to THIS ladder only" in report["ladder_elo"]["caveat"]
    md = evaluate.render_eval_report_md(report)
    assert "Below heuristic" in md
    assert "Ladder Elo" in md
    assert "Tactical suite" in md
    assert "win-h-open4" in md  # failures are listed
    assert "Disclaimer" in md
    # JSON-serializable.
    json.dumps(report)


# --- TS cross-check plumbing ----------------------------------------------

def test_play_position_serves_one_request():
    """play_position.py loads the smoke checkpoint, sends a ready event,
    and answers a position request with a legal move + win rate."""
    if not os.path.exists(SMOKE_CHECKPOINT):
        pytest.skip("smoke checkpoint not present")
    req = json.dumps({"board": [0] * 225, "to_move": 1, "sims": 4})
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "play_position.py"),
         "--checkpoint", SMOKE_CHECKPOINT],
        input=req + "\n", capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = [json.loads(line) for line in proc.stdout.strip().splitlines()]
    assert lines[0]["event"] == "ready"
    resp = lines[1]
    assert "error" not in resp, resp
    x, y = resp["move"]
    assert 0 <= x < 15 and 0 <= y < 15
    assert 0.0 <= resp["win_rate"] <= 1.0
    assert resp["visits"] >= 1


def test_play_match_driver_help():
    """scripts/play_match.mjs exists and parses --help (the full match
    is exercised manually, not in pytest — it needs esbuild + a game)."""
    driver = os.path.join(REPO_ROOT, "scripts", "play_match.mjs")
    assert os.path.exists(driver)
    proc = subprocess.run(
        ["node", driver, "--help"], capture_output=True, text=True,
        timeout=60,
    )
    assert proc.returncode == 0
    assert "--checkpoint" in proc.stdout

"""Tests for eval_version.py --out auto-resume (batch 10 tooling).

A killed ladder run must resume exactly: the driver's partial file
carries the completed legs plus the player's RNG state, and a resumed
ladder must equal the uninterrupted ladder exactly — the player's
seeded RNG advances across legs (tie-breaks only), so replaying a
leg with a fresh RNG would NOT match.
"""

import json
import os
import subprocess
import sys

from gomoku_ml import research, scoring

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "ml", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import selfplay_batch  # noqa: E402  (openings loader + default paths)


def _eval_json(proc):
    for line in proc.stdout.splitlines():
        if line.startswith("EVAL_JSON "):
            return json.loads(line[len("EVAL_JSON "):])
    raise AssertionError(f"no EVAL_JSON in stdout:\n{proc.stdout[-2000:]}")


def _run_cli(out):
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, "eval_version.py"),
         "--version", "1", "--ladder-games", "2", "--mcts-sims", "5",
         "--seed", "100", "--out", out],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc


def test_ladder_resume_reproduces_uninterrupted(tmp_path):
    """Build a partial (leg 1 done, with the RNG state as of then),
    resume via the real CLI, and require the final ladder to equal
    the uninterrupted ladder exactly."""
    out = str(tmp_path / "ladder-v1.json")
    # Ladder legs use the balanced openings set by default
    # (evaluation audit), so the reference and the partial must use
    # it too — and the partial must carry its hash, or the resume
    # check rejects it.
    openings, openings_hash = selfplay_batch.load_openings(
        selfplay_batch.common.default_openings_path())
    # Uninterrupted reference, in-process: same args the CLI uses
    # (player seed 100, ladder seed 100 + 300).
    ref_player = scoring.get_player(1, seed=100)
    ref = research.ladder_for_player(
        ref_player.choose_move, games_per_opponent=2, mcts_sims=5,
        seed=400, openings=openings,
    )
    # Partial: play only the first leg in-process and capture the
    # RNG state as of then — exactly what the driver flushes after
    # each leg.
    part_player = scoring.get_player(1, seed=100)
    first_leg = research.ladder_leg_for_player(
        part_player.choose_move, "random", games_per_opponent=2,
        mcts_sims=5, seed=400, openings=openings,
    )
    tac = research.run_tactical(
        lambda board, p: scoring._choose_move(
            board, p, None, part_player.defence_weight,
            part_player.threat_check, part_player.minimax,
            part_player.minimax_topk,
        )
    )
    partial = {
        "engine": part_player.name,
        "version": 1,
        "ladder_games": 2,
        "mcts_sims": 5,
        "seed": 100,
        "complete": False,
        "resumed": False,
        "resumed_from_leg": None,
        "rng_state": part_player.rng.bit_generator.state,
        "openings": None,
        "openings_hash": openings_hash,
        "openings_count": len(openings),
        "tactical": tac,
        "ladder": {
            "matchups": {"random": first_leg},
            "ladder_elo": None,
            "games_per_opponent": 2,
            "mcts_sims": 5,
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(partial, f, indent=2)

    proc = _run_cli(out)
    result = _eval_json(proc)
    assert result["resumed"] is True
    assert result["resumed_from_leg"] == "heuristic"
    assert result["complete"] is True
    assert result["ladder"]["matchups"] == ref["matchups"]
    assert result["ladder"]["ladder_elo"] == ref["ladder_elo"]

    # Re-running with the complete file is a no-op returning the
    # same result.
    proc2 = _run_cli(out)
    assert _eval_json(proc2) == result
    assert "already complete" in proc2.stderr

"""Tests for opening variety in selfplay_batch (ML batch 4 tooling).

The mirror-match determinism collapse persists despite per-game
seeding: v36-vs-v8 produced only 21 unique games in 100 — engine
similarity, not RNG sharing. ``--openings <file>`` starts every
normal-rules game from a fixed opening position instead of the
empty board: game ``i`` uses ``openings[i % len(openings)]``.
The opening is colour-agnostic (move 1 is Black's stone regardless
of which engine is Black that game) and the usual colour
alternation is preserved, so fairness is unchanged.
"""

import argparse
import json
import os
import sys

import pytest

from gomoku_ml import game, research, scoring

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SCRIPTS_DIR = os.path.join(REPO_ROOT, "ml", "scripts")
sys.path.insert(0, SCRIPTS_DIR)

import selfplay_batch  # noqa: E402

OPENINGS_FILE = os.path.join(REPO_ROOT, "research", "openings-ml4.json")
BALANCED_FILE = os.path.join(REPO_ROOT, "research", "openings-balanced.json")
SWAP_FILE = os.path.join(REPO_ROOT, "research", "openings-swap.json")


def _write_openings(tmp_path, openings):
    path = tmp_path / "openings.json"
    path.write_text(json.dumps({"openings": openings}))
    return str(path)


def test_shipped_openings_file_is_valid():
    """The pre-registered ML batch 4 set: 10 openings x 4 moves,
    all legal, no duplicates within an opening, no opening ends
    the game."""
    openings, digest = selfplay_batch.load_openings(OPENINGS_FILE)
    assert len(openings) == 10
    assert len(digest) == 64  # sha256 hex
    for opening in openings:
        assert len(opening) == 4
        assert len(set(opening)) == 4
        board, to_move = selfplay_batch.apply_opening(opening)
        assert to_move == game.BLACK  # even length -> Black to move
        over, _ = game.game_over(board)
        assert not over
        # Exactly 2 black and 2 white stones on the board.
        assert int((board == game.BLACK).sum()) == 2
        assert int((board == game.WHITE).sum()) == 2
    # The 10 openings are all distinct positions.
    assert len({tuple(o) for o in openings}) == 10


def test_apply_opening_odd_length_gives_white_to_move():
    board, to_move = selfplay_batch.apply_opening(
        [game.xy_to_move(7, 7)])
    assert to_move == game.WHITE
    assert board[7, 7] == game.BLACK


def test_load_openings_rejects_duplicate_square(tmp_path):
    path = _write_openings(tmp_path, [[[7, 7], [7, 8], [7, 7]]])
    with pytest.raises(ValueError, match="repeats square"):
        selfplay_batch.load_openings(path)


def test_load_openings_rejects_out_of_range(tmp_path):
    path = _write_openings(tmp_path, [[[7, 7], [15, 0]]])
    with pytest.raises(ValueError, match="out-of-range"):
        selfplay_batch.load_openings(path)


def test_load_openings_rejects_empty(tmp_path):
    path = _write_openings(tmp_path, [])
    with pytest.raises(ValueError, match="non-empty"):
        selfplay_batch.load_openings(path)
    path = _write_openings(tmp_path, [[]])
    with pytest.raises(ValueError, match="empty"):
        selfplay_batch.load_openings(path)


def _batch_args(out, openings=None, games=2, swap_rule=False):
    return argparse.Namespace(
        engine_a="v0", engine_b="v0", games=games,
        swap_rule=swap_rule, placement="balanced", placement_seed=0,
        judge="tempo", seed_a=100, seed_b=200,
        openings=openings, out=out)


def test_openings_batch_records_openings_and_uses_rotation(tmp_path):
    """Two v0-vs-v0 games with openings: each record carries its
    opening, moves start with the opening, and games 0/1 use
    openings[0]/openings[1] (rotation by game index)."""
    out = str(tmp_path / "batch.json")
    result = selfplay_batch.run_batch(
        _batch_args(out, openings=OPENINGS_FILE, games=2))
    assert result["complete"] is True
    assert result["openings_count"] == 10
    assert result["openings_hash"] is not None
    openings, _ = selfplay_batch.load_openings(OPENINGS_FILE)
    recs = result["records"]
    assert len(recs) == 2
    for i, rec in enumerate(recs):
        assert rec["opening"] == openings[i]
        assert rec["moves"][:4] == openings[i]
        assert rec["plies"] == len(rec["moves"])
    # Colour alternation is preserved with openings.
    assert recs[0]["a_colour"] == "black"
    assert recs[1]["a_colour"] == "white"


def test_openings_batch_resume_reproduces_uninterrupted(tmp_path):
    """A 4-game run interrupted after 2 games (simulated by
    truncating the flushed JSON to 2 records, complete=false)
    resumes to 4 games and reproduces the uninterrupted 4-game
    run exactly (full move lists), with games 2/3 using
    openings[2]/openings[3]."""
    out = str(tmp_path / "batch.json")
    out2 = str(tmp_path / "batch2.json")
    fresh = selfplay_batch.run_batch(
        _batch_args(out2, openings=OPENINGS_FILE, games=4))
    partial = dict(fresh)
    partial["records"] = fresh["records"][:2]
    partial["games_played"] = 2
    partial["complete"] = False
    partial["wins"] = sum(
        1 for r in partial["records"] if r["winner_is_a"] is True)
    partial["losses"] = sum(
        1 for r in partial["records"] if r["winner_is_a"] is False)
    partial["draws"] = 2 - partial["wins"] - partial["losses"]
    with open(out, "w", encoding="utf-8") as f:
        json.dump(partial, f)
    resumed = selfplay_batch.run_batch(
        _batch_args(out, openings=OPENINGS_FILE, games=4))
    assert resumed["resumed"] is True
    assert resumed["resumed_from_game"] == 2
    assert len(resumed["records"]) == 4
    assert [r["moves"] for r in resumed["records"]] == [
        r["moves"] for r in fresh["records"]]
    openings, _ = selfplay_batch.load_openings(OPENINGS_FILE)
    assert resumed["records"][2]["opening"] == openings[2]
    assert resumed["records"][3]["opening"] == openings[3]


def test_openings_rejected_with_swap_rule(tmp_path):
    out = str(tmp_path / "batch.json")
    with pytest.raises(ValueError, match="cannot be combined"):
        selfplay_batch.run_batch(_batch_args(
            out, openings=OPENINGS_FILE, games=1, swap_rule=True))


# --- Evaluation audit (2026-11-25): balanced + swap openings sets ---

def test_shipped_balanced_openings_file_is_valid():
    """The balanced set (default for normal-rules legs): 24 four-move
    openings, all distinct, all with attack gap 0 (best Black attack
    == best White attack), no opening ends the game, none holds an
    open four. The empty-board start let deterministic heuristic
    engines replay the same lines (v63 clause-1: 31 unique games in
    100), so the nominal CI overstated the precision."""
    openings, digest = selfplay_batch.load_openings(BALANCED_FILE)
    assert len(openings) == 24
    assert len(digest) == 64  # sha256 hex
    assert len({tuple(o) for o in openings}) == 24
    scorer = scoring.score_attack
    for opening in openings:
        assert len(opening) == 4
        assert len(set(opening)) == 4
        board, to_move = selfplay_batch.apply_opening(opening)
        assert to_move == game.BLACK  # even length -> Black to move
        over, _ = game.game_over(board)
        assert not over
        assert int((board == game.BLACK).sum()) == 2
        assert int((board == game.WHITE).sum()) == 2
        gap, bb, bw = research.opening_gap(board, scorer)
        assert gap == 0
        assert bb == bw
        assert bb < 10000 and bw < 10000  # no open four


def test_shipped_swap_openings_file_is_valid():
    """The curated swap set (default for swap-rule legs): 24
    three-stone openings, all distinct, 2 black + 1 white stones,
    12 with tempo margin 0 (judge ties -> White) and 12 with tempo
    margin 78 (judge picks Black), so the judge's pick is 12 Black /
    12 White over the set — versus ~88% Black for the uncurated
    placement. The recorded margins/picks are spot-checked against
    the exact judge on one opening of each margin (the full exact
    check is slow: _setup_value is seconds per opening)."""
    openings, digest = selfplay_batch.load_openings(SWAP_FILE)
    assert len(openings) == 24
    assert len(digest) == 64
    assert len({tuple(o) for o in openings}) == 24
    doc = json.loads(open(SWAP_FILE, encoding="utf-8").read())
    details = doc["details"]
    assert len(details) == 24
    margins = [d["tempo_margin"] for d in details]
    picks = [d["judge_pick"] for d in details]
    assert margins.count(0) == 12
    assert margins.count(78) == 12
    assert picks.count("black") == 12
    assert picks.count("white") == 12
    scorer = scoring.score_attack
    for opening in openings:
        assert len(opening) == 3
        assert len(set(opening)) == 3
        board, to_move = selfplay_batch.apply_opening(opening)
        assert to_move == game.WHITE  # odd length -> White to move
        over, _ = game.game_over(board)
        assert not over
        assert int((board == game.BLACK).sum()) == 2
        assert int((board == game.WHITE).sum()) == 1
    # Spot-check the recorded margin/pick against the exact judge.
    for want_margin, want_pick in ((0, "white"), (78, "black")):
        d = next(x for x in details if x["tempo_margin"] == want_margin)
        opening = [game.xy_to_move(x, y) for x, y in d["moves_xy"]]
        board, _ = selfplay_batch.apply_opening(opening)
        vb = game._setup_value(board, scorer, game.BLACK)
        vw = game._setup_value(board, scorer, game.WHITE)
        assert vb - vw == want_margin
        pick = game.choose_swap_color(board, scorer, judge="tempo")
        assert ("black" if pick == game.BLACK else "white") == want_pick


def _batch_args_default(out, games=2, swap_rule=False):
    """Batch args with NO openings specified: the eval-audit defaults
    (balanced set / curated swap set) must be used."""
    return argparse.Namespace(
        engine_a="v0", engine_b="v0", games=games,
        swap_rule=swap_rule, placement="balanced", placement_seed=0,
        judge="tempo", seed_a=100, seed_b=200,
        openings=None, out=out)


def test_default_openings_used_when_none_given(tmp_path):
    """A normal-rules batch with no --openings uses the balanced set
    (24 openings) and records its hash; the uniqueness gate is
    present in the summary."""
    out = str(tmp_path / "batch.json")
    result = selfplay_batch.run_batch(
        _batch_args_default(out, games=2, swap_rule=False))
    assert result["complete"] is True
    assert result["openings_count"] == 24
    assert result["openings_hash"] is not None
    assert result["uniqueness_gate"]["total_games"] == 2
    openings, _ = selfplay_batch.load_openings(BALANCED_FILE)
    assert result["records"][0]["opening"] == openings[0]
    assert result["records"][1]["opening"] == openings[1]


def test_default_swap_openings_used_and_picks_balanced(tmp_path):
    """A swap-rule batch with no --swap-openings uses the curated set
    (24 openings) and records the pick distribution and colours. The
    set interleaves margin-0/margin-78, so the first 4 games pick
    White, Black, White, Black — 2/2, not the ~88% Black of the
    uncurated placement."""
    out = str(tmp_path / "batch.json")
    result = selfplay_batch.run_batch(
        _batch_args_default(out, games=4, swap_rule=True))
    assert result["complete"] is True
    assert result["swap_openings_count"] == 24
    assert result["swap_openings_hash"] is not None
    assert result["swap_picks"] == {"black": 2, "white": 2}
    assert result["a_colours"] == {"black": 2, "white": 2}
    openings, _ = selfplay_batch.load_openings(SWAP_FILE)
    for i, rec in enumerate(result["records"]):
        assert rec["opening"] == openings[i]
        assert rec["moves"][:3] == openings[i]

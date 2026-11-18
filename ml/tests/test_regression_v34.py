"""Regression tests for v34 (batch 10, fast loop).

Each position in ``gomoku_ml/regression_positions_v34.json`` is a
v27 regression position (copied unchanged from the final batch-9
state — see ``research/iterations/iter-27.md`` §1.2/§2), or a
critical position from a v34 fast-loop loss: the losing move v34
played and the oracle-verified perfect move instead (see
``iter-34.md`` §2 for the verification behind each one). The engine
must play an expected move on every position. A position the
current engine does NOT fix is kept as an xfail with its reason —
a documented open weakness and the next iteration's target — never
silently dropped.

v34's only change from v27 is the tactical gate on the tie-break,
which fires only on exact top-visit ties with a q gap above
tau=0.05 in a tactical position — so v34 is expected to reproduce
v27's behaviour on every position here:

* tact-prev-v1v0s-g26 / tact-prev-v1v1s-g79 / tact-prev-v1bl-g2:
  the three tie fixes — all three positions are tactical
  (iter-34 §1.2 diagnostic 1), so the gate lets the tie-break
  fire and v34 plays the expected move exactly as v27 does.
* fl2-g6-ply23 / fl2-g8-ply23: v27 passes these at seed 0 ONLY
  because the threshold converts v23's deterministic-wrong
  tie-break (q gap 0.016) into an RNG coin flip over the tied
  pair, and the seed-0 flip lands on the saving move. The gap is
  below the threshold, so v34's gate is irrelevant here — v34
  behaves exactly like v27 (coin flip, seed-0 save). Across
  seeds 0-9 the save rate is 7/10 (iter-27 §1.3) — a coin flip,
  NOT a fix. The positions stay documented open weaknesses in
  iter-27.md; the regression test pins the deterministic seed-0
  behaviour and would catch a regression to v23's
  deterministic-wrong choice.
"""

import json
import os

import pytest

from gomoku_ml import game, scoring

HERE = os.path.dirname(os.path.abspath(__file__))
POSITIONS_PATH = os.path.join(
    HERE, "..", "gomoku_ml", "regression_positions_v34.json"
)

# Positions the current engine does not fix (open weaknesses).
# fl1-g0-ply12: Black-to-move v16 xfail — v18/v23 play [7, 10]
# (v16/v7 played [11, 9]), still not one of the oracle saves
# ([9, 7], [5, 5], [7, 8]). The Black rollout leaf changes the
# move but does not find the save here; see iter-18.md §1.2.
# fl2-g7-ply9: White-to-move — v18/v23 play (6, 4), the v1 oracle's
# loser (3/3), over (10, 8). The two moves are near-twins (both
# v1 ~200k, equal priors, both reduce Black creators to 0) and
# the distinction is beyond the 2-ply horizon: the rollout leaf
# prefers (6, 4) too (+0.107 vs +0.039), and the stronger oracle
# only weakly confirms (10, 8) (v7-vs-v7 2/3 vs 1/3; the actual
# matchup 1/2 vs 0/2). No patchable mechanism — see iter-18.md
# §2.2. Documented open weakness, not dropped.
XFAIL = {
    "fl1-g0-ply12": "Black position; v18/v23 play [7, 10], not a save; see iter-18.md §1.2",
    "fl2-g7-ply9": "White twin-choice; v18/v23 play (6, 4) over (10, 8); no patchable mechanism; see iter-18.md §2.2",
}


def _load_positions():
    with open(POSITIONS_PATH, encoding="utf-8") as f:
        return json.load(f)["positions"]


def _board(pos):
    board = game.empty_board()
    for x, y in pos["black"]:
        board[y, x] = game.BLACK
    for x, y in pos["white"]:
        board[y, x] = game.WHITE
    return board


@pytest.mark.parametrize("pos", _load_positions(), ids=lambda p: p["id"])
def test_v34_plays_expected_move(pos):
    if pos["id"] in XFAIL:
        pytest.xfail(XFAIL[pos["id"]])
    board = _board(pos)
    player = game.BLACK if pos["to_move"] == "black" else game.WHITE
    v34 = scoring.get_player(34, seed=0)
    move = v34.choose_move(board, player)
    chosen = [move % 15, move // 15]
    assert chosen in pos["expected"], (
        f"{pos['id']}: v34 played {chosen}, expected one of "
        f"{pos['expected']} (played before patch: "
        f"{pos['played_before_patch']})"
    )


def test_regression_positions_are_well_formed():
    positions = _load_positions()
    assert len(positions) >= 1
    for pos in positions:
        assert pos["to_move"] in ("black", "white")
        assert pos["expected"], f"{pos['id']}: no expected move"
        board = _board(pos)
        # No stones overlap; the expected moves are legal (empty).
        assert len(pos["black"]) + len(pos["white"]) == int(
            (board != game.EMPTY).sum()
        )
        for x, y in pos["expected"]:
            assert board[y, x] == game.EMPTY, (
                f"{pos['id']}: expected move {(x, y)} is occupied"
            )

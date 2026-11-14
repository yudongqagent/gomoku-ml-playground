"""Tests for the harder tactical suite (v2, batch 2).

The suite lives in ``gomoku_ml/tactical_positions_hard.json`` (29
positions). The v1 suite (20 positions) did not discriminate — scoring
v1, v2 and v3 all scored 20/20 — so batch 2 built this harder suite
from batch-1 lost games plus hand-constructed VCF/VCT positions (see
``research/iterations/tactical-suite-v2.md`` and the build script
``ml/scripts/build_tactical_hard.py``).

Verification properties (re-implemented here INDEPENDENTLY of the
build script, from ``gomoku_ml.game`` / ``gomoku_ml.evaluate`` only —
the test does not import the build script, so a bug there cannot
silently pass here):

* **prevention** — the opponent holds >= 1 open-four creator (a move
  leaving them >= 2 winning moves) and no immediate winning move. A
  move is *preventing* iff it wins immediately, or it is forcing
  (leaves the mover >= 1 winning move while the opponent has no
  immediate win), or it leaves the opponent with zero live creators.
  A move removes a creator iff it occupies the creator square or one
  of the creator's winning squares AND that leaves the creator with
  < 2 winning moves (a double-four creator has 3 winning squares:
  blocking one still leaves 2). The listed moves must be EXACTLY the
  preventing moves (all legal moves scanned).
* **setup** — the side to move holds NO creator yet. A move is a
  *setup* iff after it the opponent has no immediate win, and for
  EVERY opponent reply some creator survives (a move leaving us
  >= 2 winning moves with the opponent holding no immediate win).
  Only "dangerous" replies need checking: a reply can only break a
  setup by occupying a creator square, occupying one of a creator's
  winning squares, or creating a four for the opponent (leaving them
  a winning move). Any other reply leaves every creator intact and
  the opponent with no winning move, so any creator survives it —
  the reduction is exact. Candidates are empty points within
  Chebyshev distance 4 of the mover's stones: a setup move further
  away shares no 5-window with any friendly stone, and every creator
  it could create needs a four-window holding the move plus friendly
  stones, so it cannot create a creator. The listed moves must be
  EXACTLY the setup moves, and no VCF first move may exist outside
  the list (else a winning move would be marked wrong).
* **winning** — the listed moves are EXACTLY the 2-ply setups plus
  the VCF first moves (moves starting a forced win by consecutive
  fours). Used where a position has genuine winning moves of both
  kinds; listing only one kind would mark a winning move wrong.
* **vcf** — expect type ``vcf``: the move must start a forced win
  by consecutive fours (``evaluate.starts_forced_win``). The number
  of forcing first moves must be small (1-10) so the position
  discriminates, and no setup move may exist (a VCT win outside the
  VCF check would make the position unfair).

Known limitation (same narrowness as the v1 suite's VCF positions):
the checks verify wins by consecutive fours (depth 3) and by 2-ply
setup. A move winning by a longer, non-VCF route would not be
listed. No such move is known on any suite position.
"""

import os

import numpy as np

from gomoku_ml import baseline, evaluate, game, scoring

HARD_SUITE_PATH = os.path.join(
    os.path.dirname(evaluate.__file__), "tactical_positions_hard.json"
)

# v1's recorded score on this suite (ScoringPlayer version=1, seed=0),
# the clause-3 reference for batch-2 versions. See
# research/iterations/tactical-suite-v2.md. Stable across seeds 0, 1, 2.
V1_PASSED = 12
SUITE_TOTAL = 29


def _load():
    return evaluate.load_tactical_suite(HARD_SUITE_PATH)


# --- independent verification helpers -------------------------------------


def _near_radius1(board, player):
    """Empty points adjacent to a ``player`` stone. A creator must be
    adjacent to a friendly stone: it completes a four in a 5-window,
    and in a 5-window holding 4 stones every cell is adjacent to a
    stone."""
    near = set()
    for y, x in zip(*((board == player).nonzero())):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return sorted(near)


def _creators(board, player):
    """Moves after which ``player`` has >= 2 winning moves."""
    out = []
    for m in _near_radius1(board, player):
        b2 = game.apply_move(board, m, player)
        if len(evaluate.winning_moves(b2, player)) >= 2:
            out.append(m)
    return out


def _creator_winning_squares(board, player, creator_list=None):
    out = {}
    for c in (creator_list if creator_list is not None
              else _creators(board, player)):
        b2 = game.apply_move(board, c, player)
        out[c] = set(evaluate.winning_moves(b2, player))
    return out


def _is_preventing(board, move, player, opp_creators=None,
                   opp_creator_wins=None):
    if game.is_winning_move(board, move, player):
        return True
    if opp_creators is None:
        opp_creators = _creators(board, -player)
    if opp_creator_wins is None:
        opp_creator_wins = _creator_winning_squares(
            board, -player, opp_creators)
    removes_all = True
    for c in opp_creators:
        wins = opp_creator_wins[c]
        if move == c:
            continue
        if move in wins and len(wins) <= 2:
            continue
        removes_all = False
        break
    if removes_all:
        return True
    b2 = game.apply_move(board, move, player)
    if evaluate.winning_moves(b2, player):
        return not evaluate.winning_moves(b2, -player)
    return False


def _preventing_moves(board, player):
    opp_creators = _creators(board, -player)
    opp_wins = _creator_winning_squares(board, -player, opp_creators)
    return [int(m) for m in game.legal_moves(board)
            if _is_preventing(board, int(m), player, opp_creators,
                              opp_wins)]


def _near_union(board, radius):
    near = set()
    for y, x in zip(*((board != game.EMPTY).nonzero())):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                yy, xx = int(y) + dy, int(x) + dx
                if 0 <= yy < game.SIZE and 0 <= xx < game.SIZE:
                    if board[yy, xx] == game.EMPTY:
                        near.add(game.xy_to_move(xx, yy))
    return sorted(near)


def _creator_survives_after_reply(board_after_m, reply, creator_list,
                                  player):
    b_r = game.apply_move(board_after_m, reply, -player)
    for c in creator_list:
        if c == reply:
            continue
        b_c = game.apply_move(b_r, c, player)
        if len(evaluate.winning_moves(b_c, player)) >= 2:
            if not evaluate.winning_moves(b_c, -player):
                return True
    return False


def _is_setup(board, move, player):
    if game.is_winning_move(board, move, player):
        return False
    b2 = game.apply_move(board, move, player)
    if evaluate.winning_moves(b2, -player):
        return False
    creator_list = _creators(b2, player)
    if not creator_list:
        return False
    creator_wins = _creator_winning_squares(b2, player, creator_list)
    dangerous = set(creator_list)
    for wins in creator_wins.values():
        dangerous |= wins
    dangerous |= set(evaluate.four_creating_moves(b2, -player))
    dangerous &= set(int(m) for m in game.legal_moves(b2))
    for r in sorted(dangerous):
        if game.is_winning_move(b2, r, -player):
            return False
        if not _creator_survives_after_reply(b2, r, creator_list, player):
            return False
    return True


def _creators_involving(board_after_m, player, move):
    """Creators after ``move`` that involve ``move`` (their
    four-window contains it). A setup move's creators must all
    involve it — any other creator pre-existed the move, and setup
    positions have no pre-existing creator. Used as a cheap
    pre-filter for _setup_moves: candidates whose only creators
    would have to involve the move, but none does, cannot be setups.
    """
    out = []
    mx, my = game.move_to_xy(move)
    for c in _near_radius1(board_after_m, player):
        cx, cy = game.move_to_xy(c)
        if max(abs(cx - mx), abs(cy - my)) > 4:
            continue  # shares no 5-window with the move
        b2 = game.apply_move(board_after_m, c, player)
        if len(evaluate.winning_moves(b2, player)) >= 2:
            out.append(c)
    return out


def _setup_moves(board, player):
    # The involving pre-filter is valid only when no creator
    # pre-exists on the board: otherwise a move could be a setup via
    # a pre-existing creator that does not involve it. Either way, a
    # setup needs >= 2 creators after the move: the opponent can
    # reply on any one creator square, and another creator must
    # survive that reply (a reply can only remove our wins, never
    # add them, so the survivor was already a creator). Counting the
    # creators after the move — pre-existing ones that survive it
    # plus new ones involving it — is an exact, cheap pre-filter.
    pre_creators = _creators(board, player)
    pre_set = set(pre_creators)
    out = []
    for m in _near_union(board, radius=4):
        b2 = game.apply_move(board, m, player)
        if evaluate.winning_moves(b2, -player):
            continue
        n = 0
        for c in pre_creators:
            if c == m:
                continue
            b_c = game.apply_move(b2, c, player)
            if len(evaluate.winning_moves(b_c, player)) >= 2:
                n += 1
        if n < 2:
            n += len([c for c in _creators_involving(b2, player, m)
                      if c not in pre_set])
        if n < 2:
            continue
        if _is_setup(board, m, player):
            out.append(m)
    return out


def _vcf_first_moves(board, player):
    return [m for m in evaluate.four_creating_moves(board, player)
            if evaluate.starts_forced_win(board, player, m)]


# --- tests ----------------------------------------------------------------


def test_hard_suite_loads_and_is_structurally_sound():
    suite = _load()
    assert suite["_meta"]["version"] == 2
    positions = suite["positions"]
    assert len(positions) == SUITE_TOTAL
    ids = set()
    for pos in positions:
        assert pos["id"] not in ids
        ids.add(pos["id"])
        assert pos["to_move"] in (1, -1)
        assert pos["category"] in ("prevention", "setup", "winning", "vcf")
        assert pos["expect"]["type"] in ("moves", "vcf")
        board = evaluate.board_from_position(pos)
        over, _ = game.game_over(board)
        assert not over, f"{pos['id']}: position is already over"
        if pos["expect"]["type"] == "moves":
            assert pos["expect"]["moves"], f"{pos['id']}: empty list"
            for x, y in pos["expect"]["moves"]:
                assert game.is_legal(board, game.xy_to_move(x, y)), (
                    f"{pos['id']}: expected move ({x},{y}) is illegal"
                )


def test_hard_suite_expected_moves_are_exact():
    """Every position's expected moves are EXACTLY the moves
    satisfying its property (recomputed here, not trusted)."""
    suite = _load()
    for pos in suite["positions"]:
        board = evaluate.board_from_position(pos)
        to_move = pos["to_move"]
        pid = pos["id"]
        # No immediate win on the board: a winning move would be a
        # trivial pass the property lists do not capture (for
        # prevention positions a winning move IS preventing, so the
        # assertion below only applies to the attacking categories —
        # prevention positions are asserted win-free by extraction).
        if pos["category"] == "prevention":
            assert _creators(board, -to_move), (
                f"{pid}: opponent holds no creator"
            )
            assert not evaluate.winning_moves(board, to_move), (
                f"{pid}: side to move has an immediate win"
            )
            assert not evaluate.winning_moves(board, -to_move), (
                f"{pid}: opponent has an immediate win"
            )
            expected = {game.xy_to_move(x, y)
                        for x, y in pos["expect"]["moves"]}
            good = set(_preventing_moves(board, to_move))
            assert good == expected, (
                f"{pid}: preventing moves are "
                f"{sorted(game.move_to_xy(m) for m in good)}, listed "
                f"{sorted(game.move_to_xy(m) for m in expected)}"
            )
        elif pos["category"] == "setup":
            assert not _creators(board, to_move), (
                f"{pid}: side to move already holds a creator"
            )
            assert not evaluate.winning_moves(board, to_move), (
                f"{pid}: side to move has an immediate win"
            )
            assert not evaluate.winning_moves(board, -to_move), (
                f"{pid}: opponent has an immediate win"
            )
            assert not _vcf_first_moves(board, to_move), (
                f"{pid}: VCF moves exist outside the setup list"
            )
            expected = {game.xy_to_move(x, y)
                        for x, y in pos["expect"]["moves"]}
            good = set(_setup_moves(board, to_move))
            assert good == expected, (
                f"{pid}: setup moves are "
                f"{sorted(game.move_to_xy(m) for m in good)}, listed "
                f"{sorted(game.move_to_xy(m) for m in expected)}"
            )
        elif pos["category"] == "winning":
            assert not _creators(board, to_move), (
                f"{pid}: side to move already holds a creator"
            )
            assert not evaluate.winning_moves(board, to_move), (
                f"{pid}: side to move has an immediate win"
            )
            assert not evaluate.winning_moves(board, -to_move), (
                f"{pid}: opponent has an immediate win"
            )
            expected = {game.xy_to_move(x, y)
                        for x, y in pos["expect"]["moves"]}
            good = (set(_setup_moves(board, to_move))
                    | set(_vcf_first_moves(board, to_move)))
            assert good == expected, (
                f"{pid}: winning moves (setup + VCF) are "
                f"{sorted(game.move_to_xy(m) for m in good)}, listed "
                f"{sorted(game.move_to_xy(m) for m in expected)}"
            )
        elif pos["category"] == "vcf":
            assert pos["expect"]["type"] == "vcf"
            forcing = _vcf_first_moves(board, to_move)
            assert 1 <= len(forcing) <= 10, (
                f"{pid}: {len(forcing)} forcing first moves "
                f"{sorted(game.move_to_xy(m) for m in forcing)}"
            )
            assert not _setup_moves(board, to_move), (
                f"{pid}: setup moves exist outside the VCF check"
            )
            for m in forcing:
                assert evaluate.check_tactical_move(
                    board, to_move, m, pos["expect"]
                ), f"{pid}: forcing move {game.move_to_xy(m)} fails check"


def test_hard_suite_v1_baseline_reproduces():
    """v1's recorded score (ScoringPlayer version=1, seed=0) is the
    clause-3 reference for batch-2 versions. It must reproduce
    exactly — and it must be below the total, or the suite does not
    discriminate (the v1 suite's failure)."""
    assert V1_PASSED >= 0, "V1_PASSED placeholder not filled in"
    suite = _load()
    v1 = scoring.ScoringPlayer(seed=0, version=1)
    result = evaluate.run_tactical_suite(v1.choose_move, suite=suite)
    assert result["total"] == SUITE_TOTAL
    assert result["passed"] == V1_PASSED, (
        f"v1 passed {result['passed']}/{result['total']}, recorded "
        f"{V1_PASSED}/{SUITE_TOTAL}"
    )
    assert result["passed"] < result["total"], (
        "v1 passes the whole hard suite — it does not discriminate"
    )


def test_hard_suite_known_discrimination_cases():
    """Two pinned cases (deterministic, fresh seed-0 players):
    the iter-3 canonical missed prevention (v1 fails it, v3 passes
    it) and the block-or-win race (the frozen baseline takes the win,
    v1 blocks instead)."""
    suite = _load()
    by_id = {p["id"]: p for p in suite["positions"]}

    pos = by_id["prev-v1v0s-g13"]
    board = evaluate.board_from_position(pos)
    v1 = scoring.ScoringPlayer(seed=0, version=1)
    v3 = scoring.ScoringPlayer(seed=0, version=3)
    assert not evaluate.check_tactical_move(
        board, pos["to_move"], v1.choose_move(board, pos["to_move"]),
        pos["expect"],
    ), "v1 unexpectedly prevents on the iter-3 canonical case"
    assert evaluate.check_tactical_move(
        board, pos["to_move"], v3.choose_move(board, pos["to_move"]),
        pos["expect"],
    ), "v3 does not prevent on the iter-3 canonical case"

    pos = by_id["vcf-race-block-or-win"]
    board = evaluate.board_from_position(pos)
    v0 = baseline.HeuristicPlayer(seed=0)
    v1 = scoring.ScoringPlayer(seed=0, version=1)
    assert evaluate.check_tactical_move(
        board, pos["to_move"], v0.choose_move(board, pos["to_move"]),
        pos["expect"],
    ), "baseline does not take the win on the race position"
    assert not evaluate.check_tactical_move(
        board, pos["to_move"], v1.choose_move(board, pos["to_move"]),
        pos["expect"],
    ), "v1 unexpectedly takes the win on the race position"

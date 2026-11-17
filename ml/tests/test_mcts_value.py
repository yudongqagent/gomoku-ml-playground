"""Tests for the learned-value MCTS leaf (v8, batch 3 H2)."""

import numpy as np
import pytest
import torch

from gomoku_ml import game, scoring
from gomoku_ml.architectures import build_architecture
from gomoku_ml.mcts_heuristic import _MUST_BLOCK_VALUE as _V6_MUST_BLOCK
from gomoku_ml.mcts_value import _MUST_BLOCK_VALUE, ValueMCTS


def _board_with(black, white):
    board = game.empty_board()
    for x, y in black:
        board[y, x] = game.BLACK
    for x, y in white:
        board[y, x] = game.WHITE
    return board


def _random_net_checkpoint(tmp_path):
    """A tiny random-init resnet 16/2 checkpoint on disk."""
    torch.manual_seed(0)
    model = build_architecture("resnet", {"channels": 16, "blocks": 2})
    path = tmp_path / "value-net.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "architecture": "resnet",
            "arch_params": {"channels": 16, "blocks": 2},
            "channels": 16,
            "blocks": 2,
        },
        path,
    )
    return str(path)


def _value_mcts(tmp_path, sims=10):
    return ValueMCTS(checkpoint_path=_random_net_checkpoint(tmp_path), sims=sims)


def test_must_block_value_matches_v6():
    # The terminal leaf cases are shared with v6 by design; a drift
    # here would silently change the defence behaviour.
    assert _MUST_BLOCK_VALUE == _V6_MUST_BLOCK == -0.25


def test_leaf_value_player_to_move_wins(tmp_path):
    # Black to move has a winning move (four in a row) -> +1, no net call.
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = _value_mcts(tmp_path)
    assert mcts._leaf_value(board, game.BLACK) == 1.0
    assert mcts.leaf_net_evals == 0


def test_leaf_value_opponent_two_winning_moves(tmp_path):
    # White holds an open four (two winning moves); Black has none -> -1.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 7), (5, 8)])
    mcts = _value_mcts(tmp_path)
    assert mcts._leaf_value(board, game.BLACK) == -1.0
    assert mcts.leaf_net_evals == 0


def test_leaf_value_opponent_one_winning_move(tmp_path):
    # White holds a broken four: only the gap (5,7) wins -> -0.25.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = _value_mcts(tmp_path)
    assert mcts._leaf_value(board, game.BLACK) == -0.25
    assert mcts.leaf_net_evals == 0


def test_leaf_value_quiet_uses_net(tmp_path):
    # Quiet position: the net value SHRUNK by VALUE_WEIGHT
    # (fast-loop iteration 2), in [-1, 1], matching a direct
    # forward pass on the same encoding times the weight.
    from gomoku_ml.mcts_value import VALUE_WEIGHT

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _value_mcts(tmp_path)
    value = mcts._leaf_value(board, game.BLACK)
    assert -1.0 <= value <= 1.0
    assert mcts.leaf_net_evals == 1
    from gomoku_ml import encoding

    planes = encoding.encode(board, game.BLACK, None, None)
    with torch.no_grad():
        _logits, direct = mcts.net(torch.from_numpy(planes).unsqueeze(0))
    assert value == pytest.approx(float(direct[0, 0].item()) * VALUE_WEIGHT)
    assert 0.0 < VALUE_WEIGHT < 1.0


def test_leaf_value_shrink_bound(tmp_path):
    # The shrink bounds the quiet leaf value by VALUE_WEIGHT in
    # magnitude, whatever the net outputs.
    from gomoku_ml.mcts_value import VALUE_WEIGHT

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    mcts = _value_mcts(tmp_path)
    assert abs(mcts._leaf_value(board, game.BLACK)) <= VALUE_WEIGHT


def test_leaf_value_with_history_and_last_move(tmp_path):
    # History threading: a leaf deep in the search gets the path
    # boards as history and the move that led to it as last_move.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    prev = _board_with([(7, 7)], [(7, 9)])
    mcts = _value_mcts(tmp_path)
    value = mcts._leaf_value(board, game.BLACK, history=[prev, board], last_move=8 * 15 + 8)
    assert -1.0 <= value <= 1.0
    assert mcts.leaf_net_evals == 1


def test_choose_move_takes_immediate_win(tmp_path):
    board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    mcts = _value_mcts(tmp_path)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) in [(7, 6), (7, 11)]


def test_choose_move_blocks_single_winning_move(tmp_path):
    # White's broken four: only (5,7) wins; Black must block it.
    board = _board_with([(3, 3)], [(5, 5), (5, 6), (5, 8), (5, 9)])
    mcts = _value_mcts(tmp_path)
    move = mcts.choose_move(board, game.BLACK)
    assert (move % 15, move // 15) == (5, 7)


def test_choose_move_searches_and_does_not_mutate_board(tmp_path):
    rng = np.random.default_rng(0)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:24]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    before = board.copy()
    mcts = _value_mcts(tmp_path, sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert board[move // 15, move % 15] == game.EMPTY  # legal move
    assert np.array_equal(board, before)
    assert mcts.leaf_net_evals > 0  # the search actually evaluated leaves


def test_choose_move_deterministic_for_same_seed(tmp_path):
    rng = np.random.default_rng(1)
    board = game.empty_board()
    cells = [(x, y) for y in range(4, 11) for x in range(4, 11)]
    rng.shuffle(cells)
    for i, (x, y) in enumerate(cells[:20]):
        board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
    ckpt = _random_net_checkpoint(tmp_path)
    a = ValueMCTS(checkpoint_path=ckpt, sims=20, rng=np.random.default_rng(0))
    b = ValueMCTS(checkpoint_path=ckpt, sims=20, rng=np.random.default_rng(0))
    assert a.choose_move(board, game.BLACK) == b.choose_move(board, game.BLACK)


def test_choose_move_threat_check_picks_safe_move(tmp_path):
    # Fast-loop iteration 1: the opponent holds an open three
    # (White (5,5),(5,6),(5,7) with both ends open — either end
    # creates an open four), so Black must play a safe move: a
    # forcing move, or one leaving White with no live creator.
    # The unpatched search plays its highest-visited (usually
    # attacking) move; the patched choice must be safe.
    board = _board_with(
        [(7, 7), (8, 8), (3, 3)], [(5, 5), (5, 6), (5, 7)]
    )
    creators = scoring._opponent_creators(board, game.BLACK)
    assert creators  # White can create an open four next move
    mcts = _value_mcts(tmp_path, sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert scoring._move_is_safe(board, move, game.BLACK, creators), (
        f"chose {(move % 15, move // 15)}, not safe against {creators}"
    )
    assert mcts.threat_filter_used == 1


def test_choose_move_threat_check_noop_without_creators(tmp_path):
    # No opponent creators: the filter must not engage and the
    # choice must be the plain most-visited move.
    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    assert scoring._opponent_creators(board, game.BLACK) == []
    mcts = _value_mcts(tmp_path, sims=20)
    move = mcts.choose_move(board, game.BLACK)
    assert mcts.threat_filter_used == 0
    assert move != -1


def test_value_mcts_needs_a_net():
    import pytest

    with pytest.raises(ValueError, match="needs a value net"):
        ValueMCTS()


def test_v8_registered_and_dispatches_to_value_mcts(tmp_path, monkeypatch):
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[8], "value_checkpoint", ckpt)
    v8 = scoring.get_player(8, seed=0)
    assert v8.name == "scoring-v8"
    assert v8.mcts is True
    assert isinstance(v8._mcts, ValueMCTS)
    assert v8._mcts.sims == 100
    assert v8._mcts.checkpoint_path == ckpt
    # v6 still dispatches to the plain heuristic MCTS.
    from gomoku_ml.mcts_heuristic import HeuristicMCTS

    v6 = scoring.get_player(6, seed=0)
    assert type(v6._mcts) is HeuristicMCTS


def test_v8_env_override_and_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GOMOKU_V8_CHECKPOINT", str(tmp_path / "nope.pt"))
    import pytest

    with pytest.raises(FileNotFoundError):
        scoring.get_player(8, seed=0)


def test_value_weight_default_is_v8_shrink(tmp_path):
    # Default value_weight is VALUE_WEIGHT (0.25): v8's behaviour is
    # unchanged by the per-version α (batch 4 H3, v11; reused by
    # v25 and v30).
    from gomoku_ml.mcts_value import VALUE_WEIGHT

    mcts = _value_mcts(tmp_path)
    assert mcts.value_weight == VALUE_WEIGHT == 0.25


def test_value_weight_scales_quiet_leaf(tmp_path):
    # The instance value_weight scales the quiet leaf value; the
    # terminal cases stay exact whatever α is.
    from gomoku_ml import encoding

    board = _board_with([(7, 7), (8, 8)], [(7, 9), (9, 9)])
    ckpt = _random_net_checkpoint(tmp_path)
    full = ValueMCTS(checkpoint_path=ckpt, sims=10, value_weight=1.0)
    half = ValueMCTS(checkpoint_path=ckpt, sims=10, value_weight=0.5)
    v_full = full._leaf_value(board, game.BLACK)
    v_half = half._leaf_value(board, game.BLACK)
    assert v_half == pytest.approx(v_full * 0.5)
    planes = encoding.encode(board, game.BLACK, None, None)
    with torch.no_grad():
        _logits, direct = full.net(torch.from_numpy(planes).unsqueeze(0))
    assert v_full == pytest.approx(float(direct[0, 0].item()))
    # Terminal cases ignore α.
    win_board = _board_with([(7, 7), (7, 8), (7, 9), (7, 10)], [(3, 3)])
    assert full._leaf_value(win_board, game.BLACK) == 1.0
    assert half._leaf_value(win_board, game.BLACK) == 1.0


def test_v30_registered_and_dispatches_with_own_alpha(tmp_path, monkeypatch):
    # v30 (ML batch 2 H1) dispatches to ValueMCTS with its own
    # checkpoint path, env override and α from the registry.
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[30], "value_checkpoint", ckpt)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[30], "value_weight", 0.5)
    v30 = scoring.get_player(30, seed=0)
    assert v30.name == "scoring-v30"
    assert v30.mcts is True
    assert isinstance(v30._mcts, ValueMCTS)
    assert v30._mcts.sims == 100
    assert v30._mcts.checkpoint_path == ckpt
    assert v30._mcts.value_weight == 0.5
    # v8 still gets the default α (0.25) — the registry change does
    # not leak across versions.
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[8], "value_checkpoint", ckpt)
    v8 = scoring.get_player(8, seed=0)
    assert v8._mcts.value_weight == 0.25


def test_v30_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("GOMOKU_V30_CHECKPOINT", str(tmp_path / "nope.pt"))
    import pytest

    with pytest.raises(FileNotFoundError):
        scoring.get_player(30, seed=0)


# --- v32 (ML batch 2, H3): offensive-only VCF at root on the v8 base ---
# See research/iterations/iter-32.md §1.1 (pre-registered design).


def _tactical_position(suite, pid):
    import os

    from gomoku_ml import evaluate

    if suite == "hard":
        suite = os.path.join(
            os.path.dirname(evaluate.__file__),
            "tactical_positions_hard.json",
        )
    data = evaluate.load_tactical_suite(suite)
    pos = next(p for p in data["positions"] if p["id"] == pid)
    return evaluate.board_from_position(pos), pos["to_move"], pos


def test_v32_registered_and_offensive_only(tmp_path, monkeypatch):
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[32], "value_checkpoint", ckpt)
    v32 = scoring.get_player(32, seed=0)
    assert v32.name == "scoring-v32"
    assert v32.mcts is True
    assert isinstance(v32._mcts, ValueMCTS)
    assert v32._mcts.sims == 100
    assert v32._mcts.checkpoint_path == ckpt
    assert v32._mcts.vcf_root is True
    assert v32._mcts.vcf_defensive is False
    # v8 is unchanged (no VCF check).
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[8], "value_checkpoint", ckpt)
    v8 = scoring.get_player(8, seed=0)
    assert v8._mcts.vcf_root is False
    assert v8._mcts.vcf_defensive is False


def test_v32_offensive_vcf_fires_on_vcf_positions(tmp_path):
    # Old-suite VCF positions: the offensive check plays the
    # forced-win start directly (no search) — the same moves
    # v29's check plays (iter-29 §3.1 smoke test).
    mcts = _value_mcts(tmp_path, sims=100)
    mcts.vcf_root = True
    for pid, expected_xy in (("vcf-1", (8, 3)), ("vcf-2", (8, 5))):
        board, to_move, _pos = _tactical_position(None, pid)
        move = mcts._vcf_root_move(board, to_move, defensive=False)
        assert move is not None, f"{pid}: offensive VCF did not fire"
        assert game.move_to_xy(move) == expected_xy, pid


def test_v32_offensive_only_defensive_step_absent(tmp_path):
    # v18's three hard-suite prevention positions: the defensive
    # check fires there (v29 plays a prevention), but v32's
    # offensive-only check must NOT fire — those are defensive
    # preventions, which the offensive step cannot play. This
    # pins the pre-registered offensive-only design (§1.1): the
    # defensive step is absent, not merely unused.
    mcts = _value_mcts(tmp_path, sims=100)
    for pid in ("prev-v1v0s-g26", "prev-v1v1s-g79", "prev-v1bl-g2"):
        board, to_move, _pos = _tactical_position("hard", pid)
        assert mcts._vcf_root_move(board, to_move, defensive=False) is None, pid
        # Control: the defensive step DOES fire on these positions.
        assert mcts._vcf_root_move(board, to_move, defensive=True) is not None, pid


def test_v32_falls_through_to_v8_where_check_does_not_fire(tmp_path):
    # Where the check does not fire, v32 IS v8 exactly (same
    # leaf, same priors, same search, same threat check). Low
    # sims to keep the test fast; both engines share one random
    # checkpoint and seed. Sparse boards (8 stones over the full
    # board): dense random boards are creator-rich and the check
    # fires on them, so they are not fall-through positions.
    ckpt = _random_net_checkpoint(tmp_path)
    rng = np.random.default_rng(11)
    checked = 0
    for _trial in range(8):
        board = game.empty_board()
        cells = [(x, y) for y in range(15) for x in range(15)]
        rng.shuffle(cells)
        for i, (x, y) in enumerate(cells[:8]):
            board[y, x] = game.BLACK if i % 2 == 0 else game.WHITE
        v32 = ValueMCTS(
            checkpoint_path=ckpt, sims=20, vcf_root=True,
            rng=np.random.default_rng(0),
        )
        if v32._vcf_root_move(board, game.WHITE, defensive=False) is not None:
            continue  # check fired — not a fall-through position
        v8 = ValueMCTS(
            checkpoint_path=ckpt, sims=20, rng=np.random.default_rng(0)
        )
        assert v32.choose_move(board, game.WHITE) == v8.choose_move(
            board, game.WHITE
        )
        checked += 1
    assert checked >= 2, "too few fall-through positions sampled"


def test_v32_choose_move_does_not_mutate_board(tmp_path):
    board, to_move, _pos = _tactical_position(None, "vcf-1")
    before = board.copy()
    mcts = _value_mcts(tmp_path, sims=20)
    mcts.vcf_root = True
    mcts.choose_move(board, to_move)
    assert np.array_equal(board, before)

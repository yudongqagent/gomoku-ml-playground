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


def test_v43_registered_and_dispatches_to_value_mcts(tmp_path, monkeypatch):
    # ML batch 4 H2: v43 = v8's search + history-dropout 2-ply net.
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[43], "value_checkpoint", ckpt)
    v43 = scoring.get_player(43, seed=0)
    assert v43.name == "scoring-v43"
    assert v43.mcts is True
    assert isinstance(v43._mcts, ValueMCTS)
    assert v43._mcts.sims == 100
    assert v43._mcts.checkpoint_path == ckpt
    # The registry's value_weight (α) passes through to the MCTS.
    assert v43._mcts.value_weight == scoring.ENGINE_VERSIONS[43]["value_weight"]


def test_v37_registered_as_v43_comparison_engine(tmp_path, monkeypatch):
    # ML batch 4 H2: v37 is registered ONLY as v43's comparison
    # engine (same positions/labels, only the training differs).
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[37], "value_checkpoint", ckpt)
    v37 = scoring.get_player(37, seed=0)
    assert v37.name == "scoring-v37"
    assert isinstance(v37._mcts, ValueMCTS)
    assert v37._mcts.value_weight == 0.25  # v37's swept α


def test_value_mcts_value_weight_default_and_override(tmp_path):
    ckpt = _random_net_checkpoint(tmp_path)
    default = ValueMCTS(checkpoint_path=ckpt, sims=10)
    assert default.value_weight == 0.25  # v8's VALUE_WEIGHT
    override = ValueMCTS(checkpoint_path=ckpt, sims=10, value_weight=0.5)
    assert override.value_weight == 0.5


def test_v8_env_override_and_missing_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("GOMOKU_V8_CHECKPOINT", str(tmp_path / "nope.pt"))
    import pytest

    with pytest.raises(FileNotFoundError):
        scoring.get_player(8, seed=0)


def test_v42_registered_and_dispatches_to_value_mcts(tmp_path, monkeypatch):
    """v42 (ML batch 4 H1): v8's search + the history-dropout net.
    The registry entry's value_weight (the swept α) passes through
    to ValueMCTS; the test reads it from the registry so it stays
    green when the sweep updates the entry."""
    ckpt = _random_net_checkpoint(tmp_path)
    monkeypatch.setitem(scoring.ENGINE_VERSIONS[42], "value_checkpoint", ckpt)
    v42 = scoring.get_player(42, seed=0)
    assert v42.name == "scoring-v42"
    assert v42.mcts is True
    assert isinstance(v42._mcts, ValueMCTS)
    assert v42._mcts.sims == 100
    assert v42._mcts.checkpoint_path == ckpt
    assert v42._mcts.value_weight == scoring.ENGINE_VERSIONS[42]["value_weight"]


def test_value_mcts_value_weight_default_is_v8s(tmp_path):
    """No value_weight given -> v8's VALUE_WEIGHT 0.25, so v8 is
    byte-identical in behaviour (the port must not change v8)."""
    ckpt = _random_net_checkpoint(tmp_path)
    mcts = ValueMCTS(checkpoint_path=str(ckpt), sims=10)
    from gomoku_ml.mcts_value import VALUE_WEIGHT

    assert mcts.value_weight == VALUE_WEIGHT == 0.25

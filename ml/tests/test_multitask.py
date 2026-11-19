"""Tests for the v38 multi-task net (ML batch 3 H3).

Covers: the resnet-multitask architecture (registry build, I/O
contract, threat head shapes), checkpoint round-trip through the
architecture registry, ValueMCTS with a multi-task model (the
inference path uses only the value head — the 2-tuple unpack),
the multi-task training loop on a tiny synthetic dataset, the
threat-label loader, and the v38 registry entry.
"""

import numpy as np
import torch

from gomoku_ml import game, scoring
from gomoku_ml.architectures import (
    build_architecture,
    load_model_from_checkpoint,
)
from gomoku_ml.model import GomokuMultiTaskNet
from gomoku_ml.train import load_threat_labels, train


def _tiny_data(n=128, seed=0):
    rng = np.random.default_rng(seed)
    planes = rng.integers(0, 2, size=(n, 8, 15, 15)).astype(np.float32)
    policy = np.zeros((n, 225), dtype=np.float32)
    policy[np.arange(n), rng.integers(0, 225, size=n)] = 1.0
    values = rng.uniform(-1, 1, size=(n,)).astype(np.float32)
    threats = rng.integers(0, 3, size=(n,)).astype(np.int64)
    return planes, policy, values, threats


def test_multitask_builds_from_registry():
    model = build_architecture(
        "resnet-multitask",
        {"channels": 16, "blocks": 2, "threat_classes": 3},
    )
    assert isinstance(model, GomokuMultiTaskNet)
    assert model.channels == 16 and model.blocks == 2
    assert model.threat_classes == 3
    # v8-size trunk (126,586) + one value-head-shaped threat head.
    assert 126_586 < model.num_parameters() < 150_000


def test_multitask_forward_contract():
    model = build_architecture(
        "resnet-multitask",
        {"channels": 16, "blocks": 2, "threat_classes": 3},
    )
    model.eval()
    x = torch.zeros(2, 8, 15, 15)
    # Default: the existing (policy, value) contract — ValueMCTS
    # unpacks a 2-tuple unchanged.
    out = model(x)
    assert len(out) == 2
    policy, value = out
    assert policy.shape == (2, 225)
    assert value.shape == (2, 1)
    assert bool(((value >= -1) & (value <= 1)).all())  # tanh
    # Explicit: threat logits as the third return.
    policy2, value2, threat_logits = model(x, return_threat=True)
    assert threat_logits.shape == (2, 3)
    assert torch.equal(policy, policy2)
    assert torch.equal(value, value2)


def test_multitask_value_head_matches_plain_resnet():
    """The value head is unchanged in form: same layer names/shapes
    as the plain ResNet's value head (iter-38 §1.3)."""
    from gomoku_ml.architectures import build_architecture as build

    plain = build("resnet", {"channels": 16, "blocks": 2})
    multi = build(
        "resnet-multitask",
        {"channels": 16, "blocks": 2, "threat_classes": 3},
    )
    plain_sd = plain.state_dict()
    multi_sd = multi.state_dict()
    for key in (
        "heads.value_conv.weight",
        "heads.value_bn.weight",
        "heads.value_bn.bias",
        "heads.value_fc1.weight",
        "heads.value_fc1.bias",
        "heads.value_fc2.weight",
        "heads.value_fc2.bias",
        "stem.weight",
        "stem_bn.weight",
    ):
        assert key in multi_sd, key
        assert multi_sd[key].shape == plain_sd[key].shape, key
    # And the threat head exists with the value head's shape.
    assert multi_sd["heads.threat_fc1.weight"].shape == (64, 225)
    assert multi_sd["heads.threat_fc2.weight"].shape == (3, 64)


def test_multitask_checkpoint_round_trip(tmp_path):
    from gomoku_ml.train import save_checkpoint

    model = build_architecture(
        "resnet-multitask",
        {"channels": 16, "blocks": 2, "threat_classes": 3},
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = str(tmp_path / "mt.pt")
    save_checkpoint(path, model, opt, step=0, epoch=0)
    loaded, payload = load_model_from_checkpoint(path, device="cpu")
    assert isinstance(loaded, GomokuMultiTaskNet)
    assert payload["architecture"] == "resnet-multitask"
    assert payload["arch_params"] == {
        "channels": 16,
        "blocks": 2,
        "threat_classes": 3,
    }
    # Weights identical.
    for k, v in model.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k]), k


def test_value_mcts_with_multitask_model():
    """ValueMCTS drives a multi-task model through the value head
    only (the 2-tuple unpack in _value_leaf_value)."""
    from gomoku_ml.mcts_value import ValueMCTS

    model = build_architecture(
        "resnet-multitask",
        {"channels": 16, "blocks": 2, "threat_classes": 3},
    )
    engine = ValueMCTS(model=model, sims=4, rng=np.random.default_rng(0))
    board = game.empty_board()
    move = engine.choose_move(board, game.BLACK)
    assert move == 7 * 15 + 7  # centre on an empty board
    # A non-empty board exercises the net leaf (quiet position).
    board[7, 7] = game.BLACK
    board[7, 8] = game.WHITE
    move = engine.choose_move(board, game.BLACK)
    assert 0 <= move < 225
    assert engine.leaf_net_evals > 0


def test_multitask_train_loop_tiny(tmp_path):
    """train() with threat labels runs, reports threat fields, and
    selects best by VALUE val loss (iter-38 §1.2)."""
    torch.set_num_threads(1)
    planes, policy, values, threats = _tiny_data()
    model = build_architecture(
        "resnet-multitask",
        {"channels": 16, "blocks": 2, "threat_classes": 3},
    )
    summary = train(
        model,
        planes,
        policy,
        values,
        epochs=1,
        batch_size=32,
        lr=1e-3,
        policy_weight=0.0,
        value_weight=1.0,
        threats=threats,
        threat_weight=0.5,
        augmentation=False,
        val_fraction=0.1,
        checkpoint_dir=str(tmp_path),
    )
    assert summary["best_selection"] == "value"
    assert summary["best_checkpoint"] is not None
    entry = summary["history"][0]
    assert "threat_loss" in entry and "threat_acc" in entry
    assert "val_threat_loss" in entry and "val_threat_acc" in entry
    assert 0.0 <= entry["threat_acc"] <= 1.0
    assert 0.0 <= entry["val_threat_acc"] <= 1.0
    # best_val_loss is the VALUE val loss, not the total.
    assert summary["best_val_loss"] == entry["val_value_loss"]


def test_train_rejects_threats_without_threat_head():
    planes, policy, values, threats = _tiny_data()
    plain = build_architecture("resnet", {"channels": 16, "blocks": 2})
    try:
        train(
            plain,
            planes,
            policy,
            values,
            epochs=1,
            batch_size=32,
            threats=threats,
            threat_weight=0.5,
            augmentation=False,
        )
    except ValueError as e:
        assert "threat head" in str(e)
    else:
        raise AssertionError("expected ValueError for a plain net")


def test_load_threat_labels_from_shards(tmp_path):
    planes, policy, values, threats = _tiny_data(n=10)
    np.savez(
        tmp_path / "shard0.npz",
        planes=planes, policy=policy, value=values, threat=threats,
    )
    loaded = load_threat_labels([str(tmp_path)])
    assert loaded.dtype == np.int64
    assert np.array_equal(loaded, threats)


def test_v38_registry_entry():
    entry = scoring.ENGINE_VERSIONS[38]
    assert entry["name"] == "scoring-v38"
    assert entry["mcts"] is True
    assert entry["mcts_sims"] == 100
    assert entry["value_checkpoint"] == "ml/runs/ml3-h3/value-net.pt"
    assert "value_weight" in entry

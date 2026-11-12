"""Tests for the architecture registry (architectures.py)."""

import json

import pytest
import torch

from gomoku_ml.architectures import (
    ARCHITECTURES,
    build_architecture,
    list_architectures,
    load_model_from_checkpoint,
)
from gomoku_ml.train import save_checkpoint


def test_registry_contents():
    assert set(ARCHITECTURES) == {"resnet", "mobilenet", "resnet-se"}
    for spec in ARCHITECTURES.values():
        assert spec.description
        assert spec.params_schema
        # Schema must be JSON-serializable (the server/UI consumes it).
        json.dumps(spec.to_dict())


def test_list_architectures_json_serializable():
    payload = list_architectures()
    text = json.dumps(payload)
    assert json.loads(text)["architectures"]
    names = {a["name"] for a in payload["architectures"]}
    assert names == {"resnet", "mobilenet", "resnet-se"}
    for entry in payload["architectures"]:
        assert "channels" in entry["params"]
        assert "blocks" in entry["params"]


@pytest.mark.parametrize("name", ["resnet", "mobilenet", "resnet-se"])
def test_each_arch_builds_and_forward_shapes(name):
    net = build_architecture(name, {"channels": 8, "blocks": 1})
    net.eval()
    x = torch.randn(2, 8, 15, 15)
    with torch.no_grad():
        policy, value = net(x)
    assert policy.shape == (2, 225)
    assert value.shape == (2, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)
    assert net.architecture_name == name


def test_each_arch_default_params():
    for name, spec in ARCHITECTURES.items():
        defaults = spec.default_params()
        assert defaults["channels"] == 128
        assert defaults["blocks"] == 10
        net = build_architecture(name)  # defaults
        assert net.channels == 128 and net.blocks == 10


def test_resnet_se_reduction_param():
    net = build_architecture("resnet-se", {"channels": 16, "blocks": 1, "reduction": 4})
    assert net.arch_params["reduction"] == 4


def test_unknown_architecture_and_params_raise():
    with pytest.raises(ValueError, match="unknown architecture"):
        build_architecture("transformer", {})
    with pytest.raises(ValueError, match="unknown parameter"):
        build_architecture("resnet", {"channels": 16, "blocks": 1, "nope": 3})
    with pytest.raises(ValueError, match="must be an integer"):
        build_architecture("resnet", {"channels": "wide", "blocks": 1})
    with pytest.raises(ValueError, match="must be >="):
        build_architecture("resnet", {"channels": 2, "blocks": 1})


def test_mobilenet_has_fewer_params_than_resnet():
    resnet = build_architecture("resnet", {"channels": 32, "blocks": 4})
    mobile = build_architecture("mobilenet", {"channels": 32, "blocks": 4})
    assert mobile.num_parameters() < resnet.num_parameters()


def test_checkpoint_round_trip_preserves_architecture(tmp_path):
    torch.manual_seed(0)
    net = build_architecture("mobilenet", {"channels": 8, "blocks": 1})
    optimizer = torch.optim.AdamW(net.parameters())
    ckpt = str(tmp_path / "m.pt")
    save_checkpoint(ckpt, net, optimizer, step=5, epoch=1)
    loaded, payload = load_model_from_checkpoint(ckpt)
    assert payload["architecture"] == "mobilenet"
    assert loaded.architecture_name == "mobilenet"
    x = torch.randn(1, 8, 15, 15)
    net.eval()
    with torch.no_grad():
        p1, v1 = net(x)
        p2, v2 = loaded(x)
    assert torch.allclose(p1, p2)
    assert torch.allclose(v1, v2)

"""ONNX export round-trip test (skipped if onnxruntime is unavailable)."""

import os
import sys

import pytest
import torch

torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from gomoku_ml.model import GomokuNet  # noqa: E402
from gomoku_ml.train import save_checkpoint  # noqa: E402
import export_onnx  # noqa: E402


def test_export_round_trip(tmp_path):
    torch.manual_seed(0)
    net = GomokuNet(channels=8, blocks=1)
    optimizer = torch.optim.AdamW(net.parameters())
    ckpt = str(tmp_path / "tiny.pt")
    save_checkpoint(ckpt, net, optimizer, step=0, epoch=0)
    out = str(tmp_path / "tiny.onnx")
    model = export_onnx.export_onnx(ckpt, out)
    assert os.path.exists(out) and os.path.getsize(out) > 0
    # Verify the graph's I/O names and shapes match the TS contract.
    session = onnxruntime.InferenceSession(out, providers=["CPUExecutionProvider"])
    inputs = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    outputs = [(o.name, o.shape) for o in session.get_outputs()]
    assert inputs == [("input", [1, 8, 15, 15], "tensor(float)")]
    assert outputs == [("policy", [1, 225]), ("value", [1, 1])]
    result = export_onnx.verify_onnx(model, out, tolerance=1e-4)
    assert result["ok"], f"ONNX verification failed: {result}"

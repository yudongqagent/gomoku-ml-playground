#!/usr/bin/env python3
"""Export a checkpoint to ONNX (opset 13, onnxruntime-web compatible).

Contract with the TypeScript AI workstream:
  * input  ``input``  float32 [1, 8, 15, 15]  (fixed batch 1, fixed 15x15)
  * output ``policy`` float32 [1, 225] logits, row-major (index y*15+x)
  * output ``value``  float32 [1, 1] tanh in [-1, 1], current player's
    perspective

Usage:
    python ml/scripts/export_onnx.py --checkpoint ml/checkpoints/smoke_final.pt --out public/models/gomoku-smoke-v1.onnx
    python ml/scripts/export_onnx.py --checkpoint ml/checkpoints/x.pt --out out.onnx --quantize fp16

``--quantize fp16`` additionally writes a ``<name>.fp16.onnx`` file with
16-bit float weights (about half the size; onnxruntime-web can run it
via WebGPU, but the fp32 file is the safe default — the web app loads
whichever file models.json points at).

Dynamic int8 quantisation: not applied here. onnxruntime's
``quantize_dynamic`` targets CPU MatMul-heavy models and gives little on
this conv net, and int8 weights can shift the policy slightly. If you
want to try it, the exact command is:

    python -c "from onnxruntime.quantization import quantize_dynamic, QuantType; \\
        quantize_dynamic('model.onnx', 'model.int8.onnx', weight_type=QuantType.QInt8)"

and then re-run ``--verify`` against the quantised file before shipping
it. (TODO: automate this comparison in this script.)

Verification: after export the script runs the ONNX model in
onnxruntime on a fixed random input and compares policy/value against
the PyTorch model, printing the max absolute difference. It fails if
the difference exceeds the tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402
from gomoku_ml.model import GomokuNet  # noqa: E402


def export_onnx(checkpoint: str, out_path: str, opset: int = 13) -> GomokuNet:
    model, payload = load_model_from_checkpoint(checkpoint, device="cpu")
    arch = payload.get("architecture") or "resnet"
    print(f"loaded checkpoint {checkpoint} (architecture: {arch}, "
          f"params: {payload.get('arch_params') or {'channels': payload.get('channels'), 'blocks': payload.get('blocks')}})")
    dummy = torch.zeros(1, 8, 15, 15, dtype=torch.float32)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["input"],
            output_names=["policy", "value"],
            opset_version=opset,
            dynamo=False,
        )
    print(f"exported ONNX (opset {opset}) -> {out_path} "
          f"({os.path.getsize(out_path) / 1024:.1f} KiB)")
    return model


def verify_onnx(model: torch.nn.Module, onnx_path: str, tolerance: float = 1e-4) -> dict:
    """Compare PyTorch vs onnxruntime outputs on a fixed random input."""
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 15, 15)).astype(np.float32)
    with torch.no_grad():
        torch_policy, torch_value = model(torch.from_numpy(x))
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outs = session.run(None, {"input": x})
    ort_policy, ort_value = ort_outs[0], ort_outs[1]
    policy_diff = float(np.max(np.abs(torch_policy.numpy() - ort_policy)))
    value_diff = float(np.max(np.abs(torch_value.numpy() - ort_value)))
    print(f"  verify {onnx_path}:")
    print(f"    policy max abs diff: {policy_diff:.3e}")
    print(f"    value  max abs diff: {value_diff:.3e}")
    print(f"    policy shape {ort_policy.shape}, value shape {ort_value.shape}")
    ok = policy_diff <= tolerance and value_diff <= tolerance
    print(f"    {'PASS' if ok else 'FAIL'} (tolerance {tolerance})")
    return {"policy_diff": policy_diff, "value_diff": value_diff, "ok": ok}


def quantize_fp16(onnx_path: str) -> str:
    """Write a fp16-weight copy next to ``onnx_path``; return its path."""
    from onnxconverter_common import float16  # type: ignore[import-unavailable]
    import onnx

    fp16_path = onnx_path.replace(".onnx", ".fp16.onnx")
    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, fp16_path)
    print(f"fp16 quantised -> {fp16_path} ({os.path.getsize(fp16_path) / 1024:.1f} KiB)")
    return fp16_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="output .onnx path")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--quantize", choices=["fp16"], default=None,
                        help="also write a fp16-weight variant")
    parser.add_argument("--verify", action="store_true", default=True,
                        help="verify with onnxruntime (default: on)")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    model = export_onnx(args.checkpoint, args.out, opset=args.opset)
    ok = True
    verify_result = None
    if args.verify:
        try:
            verify_result = verify_onnx(model, args.out, tolerance=args.tolerance)
            ok = verify_result["ok"]
        except ImportError:
            print("WARNING: onnxruntime is not installed; verification skipped. "
                  "The .onnx file was still produced by torch.onnx.export.")
            ok = False
    if args.quantize == "fp16":
        try:
            fp16_path = quantize_fp16(args.out)
            if args.verify:
                result = verify_onnx(model, fp16_path, tolerance=1e-2)
                ok = ok and result["ok"]
        except ImportError as e:
            print(f"WARNING: fp16 quantisation needs onnxconverter-common: {e}")
            print("  install with: pip install onnxconverter-common")
    # Machine-readable summary line for the training server (filter stdout
    # for lines starting with '{').
    print(json.dumps({
        "event": "export_verified" if verify_result else "export_finished",
        "path": args.out,
        "size_bytes": os.path.getsize(args.out),
        "ok": ok,
        **(verify_result or {}),
    }))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()

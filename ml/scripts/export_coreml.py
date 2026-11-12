#!/usr/bin/env python3
"""Export a checkpoint to Core ML (.mlpackage) for native iOS use.

Input:  ``input`` multi-array float32 [1, 8, 15, 15]
Outputs: ``policy`` [1, 225] logits, ``value`` [1, 1] tanh — the same
contract as the ONNX export.

The exported model is configured for compute units ALL (CPU + GPU +
Neural Engine) at *load* time in the app — see docs/ios-coreml.md. The
.mlpackage itself carries no compute-unit setting; the app chooses it
via MLModelConfiguration.

Requirements: coremltools. It installs on Linux for *conversion* in many
cases, but the supported, tested path is macOS (Apple Silicon). If
coremltools is missing, this script exits with install instructions
instead of faking an export.

Usage (on a Mac):
    pip install coremltools
    python ml/scripts/export_coreml.py --checkpoint ml/runs/<run-id>/checkpoints/final.pt --out GomokuNet.mlpackage

Usage (conversion via ONNX, alternative if the torch converter has issues):
    python ml/scripts/export_coreml.py --onnx public/models/gomoku-smoke-v1.onnx --out GomokuNet.mlpackage
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import torch  # noqa: E402
from gomoku_ml.architectures import load_model_from_checkpoint  # noqa: E402

COREMLTOOLS_INSTALL_HINT = (
    "coremltools is not installed.\n"
    "  Install it with:  pip install coremltools\n"
    "  Notes: conversion is supported on macOS (Apple Silicon recommended)\n"
    "  and usually works on Linux too, but on-device validation and the\n"
    "  Neural Engine path need a Mac / iOS device. See docs/ios-coreml.md\n"
    "  for the exact command to run on Yudong's Mac."
)


def _load_model(checkpoint: str):
    model, _ = load_model_from_checkpoint(checkpoint, device="cpu")
    return model


def export_from_torch(checkpoint: str, out_path: str) -> str:
    try:
        import coremltools as ct
    except ImportError:
        print(COREMLTOOLS_INSTALL_HINT, file=sys.stderr)
        sys.exit(2)
    model = _load_model(checkpoint)
    traced = torch.jit.trace(model, torch.zeros(1, 8, 15, 15))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 8, 15, 15))],
        outputs=["policy", "value"],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(out_path)
    print(f"exported Core ML package -> {out_path}")
    print("  inputs:  input  float32 [1, 8, 15, 15]")
    print("  outputs: policy float32 [1, 225], value float32 [1, 1]")
    print("  (compute units are chosen at load time in the app: .all = CPU+GPU+Neural Engine)")
    return out_path


def export_from_onnx(onnx_path: str, out_path: str) -> str:
    try:
        import coremltools as ct
    except ImportError:
        print(COREMLTOOLS_INSTALL_HINT, file=sys.stderr)
        sys.exit(2)
    mlmodel = ct.convert(
        onnx_path,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(out_path)
    print(f"exported Core ML package from ONNX -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="PyTorch checkpoint .pt")
    group.add_argument("--onnx", help="ONNX file to convert instead")
    parser.add_argument("--out", required=True, help="output .mlpackage path")
    args = parser.parse_args()

    if args.checkpoint:
        export_from_torch(args.checkpoint, args.out)
    else:
        export_from_onnx(args.onnx, args.out)


if __name__ == "__main__":
    main()

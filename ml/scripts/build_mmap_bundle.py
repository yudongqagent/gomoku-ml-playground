"""Build a memory-mapped training bundle from ``.npz`` self-play shards.

Streams the shards in ``<data-dir>`` into three ``.npy`` files
(``planes.npy``, ``policy.npy``, ``values.npy``) in ``<out-dir>``,
written chunk by chunk via ``np.lib.format.open_memmap`` — peak RAM
is ~one shard, not the whole dataset. The training loader
(``gomoku_ml.train.load_shards``) loads a directory containing these
three files with ``mmap_mode='r'``, so training at 10k-game scale
(~3.1 GB) runs with ~200 MB of anonymous memory instead of ~3.2 GB
and survives this VM's host memory pressure (the in-RAM load was
OOM-killed 2026-11-16 — see ``research/iterations/iter-25.md``
§2.4). The bundle is byte-identical in content to loading the
shards with ``load_shards`` (same order, same dtypes); training
math is unchanged.

Usage:
    python ml/scripts/build_mmap_bundle.py \
        --data ml/runs/ml1-h2-10k/data --out ml/runs/ml1-h2-10k/data-mmap
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="directory of .npz shards")
    parser.add_argument("--out", required=True, help="output directory for the bundle")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "**", "*.npz"), recursive=True))
    if not paths:
        print(f"no .npz shards found under {args.data}", file=sys.stderr)
        return 2

    # Shapes-only first pass (np.load is lazy — .shape reads no data).
    total = 0
    shapes, dtypes = {}, {}
    for path in paths:
        with np.load(path) as z:
            total += z["planes"].shape[0]
            for key in ("planes", "policy", "value"):
                if key not in shapes:
                    shapes[key] = z[key].shape[1:]
                    dtypes[key] = z[key].dtype

    os.makedirs(args.out, exist_ok=True)
    out_names = {"planes": "planes.npy", "policy": "policy.npy", "value": "values.npy"}
    mmaps = {
        key: np.lib.format.open_memmap(
            os.path.join(args.out, out_names[key]),
            mode="w+",
            dtype=dtypes[key],
            shape=(total,) + shapes[key],
        )
        for key in ("planes", "policy", "value")
    }

    offset = 0
    for i, path in enumerate(paths):
        with np.load(path) as z:
            n = z["planes"].shape[0]
            mmaps["planes"][offset : offset + n] = z["planes"]
            mmaps["policy"][offset : offset + n] = z["policy"]
            mmaps["value"][offset : offset + n] = z["value"]
            offset += n
        if (i + 1) % 1000 == 0:
            print(f"... {i + 1}/{len(paths)} shards, {offset} samples", flush=True)
    for mm in mmaps.values():
        mm.flush()

    print(f"wrote bundle: {len(paths)} shards, {total} samples -> {args.out}")
    print(
        "shapes: planes",
        mmaps["planes"].shape,
        "policy",
        mmaps["policy"].shape,
        "values",
        mmaps["value"].shape,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

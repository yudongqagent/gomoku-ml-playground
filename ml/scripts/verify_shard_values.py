#!/usr/bin/env python3
"""Compare re-labelled shards against original shards, value by value (v55).

Verification gate for the base reconstruction (iter-55 §1.1): label the
reconstructed games with the ORIGINAL label mode and compare every stored
value against the original shards — 0 mismatches required. Also reports
quiet-position counts per source (the patch gate: counts must match
iter-48 §7.1 exactly).

Usage:
    python ml/scripts/verify_shard_values.py \\
        --orig ~/workspace/projects/gomoku-ml3-h2/ml/runs/ml3-h2/data \\
        --new ml/runs/ml6-h2/verify-rollout2 --sources v7 v18 random
    python ml/scripts/verify_shard_values.py --counts-only \\
        --new ml/runs/ml6-h2/data --sources vs-v0 vs-v7 vs-v8 vs-v37
"""

from __future__ import annotations

import argparse
import glob
import os
import sys


def _shard_maps(root: str, sources: list[str]) -> dict[str, dict[int, str]]:
    import re
    out: dict[str, dict[int, str]] = {}
    for src in sources:
        m: dict[int, str] = {}
        for sub in (src, f"{src}-holdout"):
            for path in glob.glob(os.path.join(root, sub, "*.npz")):
                mm = re.search(r"_game(\d+)\.npz$", os.path.basename(path))
                if mm:
                    m[int(mm.group(1))] = path
        out[src] = m
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", default=None,
                        help="original shard tree root")
    parser.add_argument("--new", required=True,
                        help="new shard tree root")
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--counts-only", action="store_true",
                        help="only report quiet-position counts, no comparison")
    args = parser.parse_args()

    import numpy as np  # noqa: E402

    new_maps = _shard_maps(args.new, args.sources)
    total_mismatch = 0
    total_pos = 0
    for src in args.sources:
        new = new_maps[src]
        npos = 0
        for idx, path in sorted(new.items()):
            z = np.load(path)
            npos += len(z["value"])
        total_pos += npos
        print(f"{src}: {len(new)} shards, {npos} quiet positions")
        if args.counts_only:
            continue
        orig_maps = _shard_maps(args.orig, args.sources)
        orig = orig_maps[src]
        if set(orig) != set(new):
            missing = sorted(set(orig) - set(new))
            extra = sorted(set(new) - set(orig))
            print(f"  GAME SET MISMATCH: missing {missing[:10]} "
                  f"({len(missing)}), extra {extra[:10]} ({len(extra)})")
            total_mismatch += len(missing) + len(extra)
            continue
        mism = 0
        max_abs = 0.0
        for idx in sorted(orig):
            zo = np.load(orig[idx])
            zn = np.load(new[idx])
            if zo["value"].shape != zn["value"].shape:
                print(f"  game {idx}: shape {zo['value'].shape} vs "
                      f"{zn['value'].shape}")
                mism += 1
                continue
            d = np.abs(zo["value"].astype(np.float64)
                       - zn["value"].astype(np.float64))
            max_abs = max(max_abs, float(d.max()) if d.size else 0.0)
            mism += int(np.count_nonzero(d > 0.0))
        print(f"  value mismatches: {mism} / {npos} "
              f"(max |delta| {max_abs:.2e})")
        total_mismatch += mism
    print(f"TOTAL: {total_pos} quiet positions, "
          f"{total_mismatch} mismatches")
    return 1 if total_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())

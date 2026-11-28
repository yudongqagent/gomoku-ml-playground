#!/usr/bin/env python3
"""Sample the openings-v2 candidate pool.

Reuses the eval-audit sampler from
``ml/scripts/generate_balanced_openings.py`` (same sanity
constraints: attack gap 0 by the v1 score, no five, no open four,
compact interacting region, distinct openings) with a fresh seed,
and writes the pool to ``research/openings-v2-candidates.json``.
The pool is then screened IN PLAY by
``ml/scripts/validate_openings.py`` — gap 0 alone does not make an
opening balanced in play (the v1 set proved that), so this file is
a candidate pool, not a set.

Usage:
    python ml/scripts/sample_openings_v2.py --n 120 --seed 20261127
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

common.ensure_ml_on_path()

import generate_balanced_openings as gen  # noqa: E402

REPO = "/home/hatch/workspace/projects/gomoku-ml-playground"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261127)
    parser.add_argument(
        "--out", default=os.path.join(REPO, "research",
                                      "openings-v2-candidates.json"))
    args = parser.parse_args()
    picked, stats = gen.gen_balanced(n=args.n, seed=args.seed)
    doc = {
        "meta": {
            "purpose": "candidate pool for openings v2 (in-play screening)",
            "criterion": "attack gap 0 (best Black attack == best White "
                         "attack), no five, no open four, compact "
                         "interacting region — SAME sampler as the "
                         "eval-audit v1 set; play balance is decided "
                         "by screening, not by the gap",
            "generator": "ml/scripts/generate_balanced_openings.py "
                         "gen_balanced (reused, not rewritten)",
            "seed": args.seed,
            "sampled": stats["sampled"],
            "gap0_found": stats["gap0"],
            "rejected_open_four": stats["rejected_open_four"],
            "count": len(picked),
        },
        "openings": [o["moves_xy"] for o in picked],
        "details": picked,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}: {len(picked)} candidates "
          f"(sampled {stats['sampled']}, gap0 {stats['gap0']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

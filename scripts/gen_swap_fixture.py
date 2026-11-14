"""Generate the TypeScript swap-rule test fixture from the Python code.

Regenerates the 200 balanced swap openings (placement seed 23) with the
research harness, verifies they reproduce the recorded experiment in
``research/iterations/swap-judge-split-200.json`` exactly (openings,
tempo setup values, choices), and emits ``src/ai/swap-fixtures.ts``:

* all 200 openings with their tempo setup values, tempo choice, and
  best-attack scores per colour — the TS judge port must reproduce
  every value (split 157 Black / 43 White);
* full ``score_attack`` vectors (both colours, all legal moves) for
  the first 5 openings — the TS v1 attack scorer must match every
  score, not just the maxima.

Run from the repo root: ``ml/.venv/bin/python scripts/gen_swap_fixture.py``.
Re-run whenever the Python scorer, placer, or judge changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

import numpy as np  # noqa: E402

from gomoku_ml import game, research, scoring  # noqa: E402

N_OPENINGS = 200
PLACEMENT_SEED = 23
N_SCORE_VECTORS = 5
OUT = ROOT / "src" / "ai" / "swap-fixtures.ts"


def main() -> None:
    recorded = json.loads(
        (ROOT / "research" / "iterations" / "swap-judge-split-200.json").read_text()
    )
    assert recorded["n_openings"] == N_OPENINGS
    assert recorded["placement_seed"] == PLACEMENT_SEED

    rng = np.random.default_rng(PLACEMENT_SEED)
    openings = []
    tempo_black = tempo_white = 0
    for i in range(N_OPENINGS):
        board, (black1, white1, black2) = research.place_swap_opening(
            None, placement="balanced", rng=rng
        )
        rec = recorded["records"][i]
        # Regeneration must reproduce the recorded experiment exactly.
        assert [black1, white1, black2] == rec["opening"], f"opening {i} mismatch"
        vb = game._setup_value(board, scoring.score_attack, game.BLACK)
        vw = game._setup_value(board, scoring.score_attack, game.WHITE)
        assert vb == rec["tempo_value_black"], f"opening {i} tempo black mismatch"
        assert vw == rec["tempo_value_white"], f"opening {i} tempo white mismatch"
        choice = game.choose_swap_color_tempo(board, scoring.score_attack)
        assert ("black" if choice == game.BLACK else "white") == rec["tempo_choice"]
        bb = research.best_attack(board, game.BLACK)
        bw = research.best_attack(board, game.WHITE)
        assert bb == rec["attack_best_black"], f"opening {i} attack black mismatch"
        assert bw == rec["attack_best_white"], f"opening {i} attack white mismatch"
        if choice == game.BLACK:
            tempo_black += 1
        else:
            tempo_white += 1
        openings.append(
            {
                "opening": [black1, white1, black2],
                "tempoBlack": vb,
                "tempoWhite": vw,
                "tempoChoice": "black" if choice == game.BLACK else "white",
                "attackBestBlack": bb,
                "attackBestWhite": bw,
            }
        )
    assert (tempo_black, tempo_white) == (
        recorded["tempo_split"]["black"],
        recorded["tempo_split"]["white"],
    ), f"split mismatch: {(tempo_black, tempo_white)}"
    print(f"200 openings reproduced; tempo split Black {tempo_black} / White {tempo_white}")

    # Full score_attack vectors for the first few openings.
    score_vectors = []
    rng = np.random.default_rng(PLACEMENT_SEED)
    for i in range(N_SCORE_VECTORS):
        board, opening = research.place_swap_opening(
            None, placement="balanced", rng=rng
        )
        assert list(opening) == openings[i]["opening"]
        vectors = {}
        for colour, key in ((game.BLACK, "black"), (game.WHITE, "white")):
            vectors[key] = [
                [int(m), int(scoring.score_attack(board, int(m), colour))]
                for m in game.legal_moves(board)
            ]
        score_vectors.append({"opening": list(opening), **vectors})
    print(f"score vectors for {N_SCORE_VECTORS} openings")

    lines = [
        "/**",
        " * Swap-rule fixtures, GENERATED from the Python research code — do not edit",
        " * by hand. Regenerate with: ml/.venv/bin/python scripts/gen_swap_fixture.py",
        " *",
        " * - SWAP_OPENINGS_200: the 200 balanced openings (placement seed 23) from",
        " *   research/iterations/swap-judge-v2.md, with the Python tempo judge's",
        " *   setup values and choice per opening (split 157 Black / 43 White) and",
        " *   each colour's best attack score. The TS port in swap.ts must reproduce",
        " *   every value exactly.",
        " * - SWAP_SCORE_VECTORS: full v1 score_attack vectors (both colours, every",
        " *   legal move) for the first 5 openings — the TS scorer must match every",
        " *   score, not just the maxima.",
        " *",
        " * Opening triples are flat indices [black1, white1, black2] in placement",
        " * order (flat index = y * 15 + x).",
        " */",
        "",
        "export interface SwapOpeningFixture {",
        "  /** [black1, white1, black2] flat indices, in placement order. */",
        "  opening: [number, number, number];",
        "  tempoBlack: number;",
        "  tempoWhite: number;",
        "  tempoChoice: 'black' | 'white';",
        "  attackBestBlack: number;",
        "  attackBestWhite: number;",
        "}",
        "",
        f"export const SWAP_OPENINGS_200: SwapOpeningFixture[] = {json.dumps(openings)};",
        "",
        "export interface SwapScoreVectorFixture {",
        "  opening: [number, number, number];",
        "  /** [flat move, score_attack] for every legal move. */",
        "  black: Array<[number, number]>;",
        "  white: Array<[number, number]>;",
        "}",
        "",
        f"export const SWAP_SCORE_VECTORS: SwapScoreVectorFixture[] = {json.dumps(score_vectors)};",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

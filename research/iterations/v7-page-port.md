# v7 page port — TypeScript port of the v7 engine as the Master level

**Status: repo port COMPLETE and verified; Space port delegated (see below).**
Branch: `feature/v7-page-port` (from master `bf866f7`).

## What changed

v7 (batch 3, H1 — PUCT MCTS, 100 sims, c_puct 1.5, v1 priors, terminal-aware
leaf + capped ±0.2 attack-gradient leaf in quiet positions) is the first
version to pass all 4 clauses of the "better" bar and is now on master. This
ports it faithfully to TypeScript as the web app's Master level, replacing the
AlphaZero-style ONNX Master (whose bundled demo net was Below heuristic —
0W-40L vs the 1-ply heuristic — so the swap is a strict upgrade).

- `src/ai/v7.ts` (new): the port — v1 5-window scorer (max-per-direction,
  double-threat bonuses, full-value defence), candidate/winning-move scans,
  terminal + gradient leaf values, and `HeuristicMCTS` mirroring
  `ml/gomoku_ml/mcts_heuristic.py`.
- `src/kernel/scoring.ts`: added `attackAllJs` / `attackBothJs` (port of
  `kernel_numpy.attack_all` / `attack_both` — friendly part only, no centre
  term), used by the gradient leaf.
- `src/ai/engine.ts`: Master dispatches to `chooseV7Move`; `modelUrl` is
  ignored. `alphazero.ts` is kept for the cross-check match script only.
- `src/ai/levels.ts`, `src/components/AiPanel.tsx`, `src/App.tsx`: Master
  copy updated honestly (v7 engine, strongest version so far, not
  superhuman). The model picker is kept for the Trainer workflow; for
  Master it shows a note that no model is needed. No layout or feature
  changes.
- `ml/scripts/gen_v7_ts_fixtures.py` (new): generates the frozen Python
  reference fixtures (`src/ai/__fixtures__/v7-reference.json`, 84 positions:
  old + hard tactical suites, v7 regression positions, 30 random mid-game
  positions) with v1 scores, kernel attack scores, and Python v7's move.
- `scripts/v7_vs_master_match.mjs` (new, `npm run match:v7`): engine-vs-engine
  match, new Master (v7) vs old Master (MCTS fallback at its defaults).

## Verification (repo port)

All in `src/ai/v7.test.ts` (16 tests), against frozen Python values — not
against the port's own logic:

| Check | Result |
|---|---|
| v1 total scores vs Python (84 positions, >10,000 cells) | 0 mismatches |
| v1 attack scores vs Python | 0 mismatches |
| Kernel attack_both vs Python | 0 mismatches |
| Move agreement vs Python v7 (deterministic tie-break both sides) | 84/84 |
| Tactical, old suite | 20/20 |
| Tactical, hard suite | 29/29 |
| Fast-loop regression positions | 4/5 pass + game-3 xfail (matches Python) |
| Mean move time (Node, match measurement) | ~120 ms |

Full suites on the final tree: npm test 71/71, tsc clean (ml pytest:
unchanged Python code — 240 passed / 2 xfailed on master).

## Engine-vs-engine vs the old Master

`npm run match:v7` (20 games, alternating colours, seeded RNG, both sides
through the same shared win/block pre-check as the page):

<!-- MATCH_RESULT: 20 games, 2026-11-14, seeded RNG (mulberry32, seed 1000+game). -->
**Result: v7 wins 14–6 (win rate 0.70), 0 draws.** As Black 8W-2L, as White
6W-4L. Mean move time: v7 34.7 ms, old Master 468.7 ms (v7 is ~13.5x faster
per move; the old Master's 400-sim rollouts dominate its cost). v7 does not
lose the match — the port plays at least as well as the old Master, and
much faster. (Small sample: 20 games. The White split, 6-4, is better than
Python v7's White record vs v1 — different opponent, do not over-read it.)

The old Master baseline is the MCTS fallback (`chooseMctsMove`, 400 sims,
5000 ms/move — its LEVEL_META defaults), i.e. the no-model behaviour and the
stronger of the old Master's two behaviours (the ONNX demo net is Below
heuristic and does not run in Node).

## Fidelity notes

- **The Python terminal win check in `_run_one` never fires**: it calls
  `game.is_winning_move`, which returns False on an occupied cell, and the
  parent already placed the stone before recursing. Verified against the
  Python reference: 27 wins played inside a 100-sim tree, 0
  terminal-branch hits. The port reproduces this exactly (its
  `isWinningMoveV1` has the same occupied-cell behaviour). Do NOT "fix" it
  to `hasFive` without re-validating the whole engine — v7's measured
  strength (Elo 190.8, 0.7050 vs v1) comes from the code as written. Fixing
  it is a batch-4 research question, not a port decision.
- The gradient is blind to broken fours (the kernel ports v0's
  contiguous-run scoring, not v1's 5-window scoring) — same caveat as
  Python v7 (iter-7.md §1).
- Root choice is most-visited; ties break by the injected RNG (Math.random
  on the page) or to the first move when no RNG is given (tests).

## Space port

The hosted Space (slug `gomoku-ml-playground`) Master level gets the same
engine via a separate workstream (artifact tools only). Status: delegated;
results recorded here when it lands.

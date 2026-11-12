# AI workstream — engine + AI levels

## What the web app imports

```ts
import { chooseMove, analyzePosition, LEVEL_META } from './ai';
import { useAiWorker } from '../hooks/useAiWorker';
```

- **`chooseMove(req)` / `analyzePosition(req)`** (`engine.ts`) — the only
  functions the UI needs. Both are pure async functions; the worker is a thin
  wrapper around them. They never throw on a legal non-terminal position.
- **`useAiWorker()`** (`hooks/useAiWorker.ts`) — runs the AI in a Web Worker
  with main-thread fallback. `requestMove(req)` / `analyze(req)` /
  `status` / `cancel()`.
- **`LEVEL_META`** (`levels.ts`) — names, taglines, descriptions, and honest
  strength notes for the five levels. Strength copy never claims superhuman
  play; the Master note says explicitly that strength depends on the loaded
  model and the bundled demo net is not superhuman.

## Request shape

```ts
{
  board: number[],        // 225 entries, row-major, 1 Black / -1 White / 0 empty
  toMove: 1 | -1,
  level: 'beginner' | 'intermediate' | 'advanced' | 'expert' | 'master',
  options?: { sims?, timeMs?, temperature?, modelUrl? },
  history?: Point[],      // full move list; feeds the net's history planes
}
```

Pass `history` whenever you have it — the Master net's planes 2 and 4–7 are
zeros without it. Pass `options.modelUrl` (e.g. `/models/gomoku-v1.onnx`)
for the Master level, or omit it for MCTS fallback.

## Guarantees

- **Every level takes an immediate win and blocks an immediate loss.** A
  shared pre-check in `chooseMove` runs before any level: own win first, then
  block. A double threat (opponent has 2+ winning squares) is reported with
  win rate 0.05 — the position is lost and we say so.
- **Win rates are in [0, 1]** for the side to move. Search levels use a
  logistic squash of the pattern eval (`EVAL_SCALE = 3000`, uncalibrated —
  see `patterns.ts`); MCTS levels use visit counts / mean values.
- **`backend` tells the truth** about what produced the move:
  `heuristic`, `alphabeta`, `alphabeta-vct`, `mcts`, `onnx-webgpu`,
  `onnx-wasm`, `mcts-fallback`, `forced-win`, `forced-block`. Show it in the
  UI instead of claiming a model was used when it was not.

## Levels

| Level | Engine | Defaults |
| --- | --- | --- |
| Beginner | 1-ply pattern heuristic, ~15% random slip | instant |
| Intermediate | Alpha-beta, depth 3 | 500 ms |
| Advanced | Iterative-deepening alpha-beta to depth 8 + VCF forced-win search | 2000 ms |
| Expert | MCTS, heuristic priors + rollouts | 1000 sims / 2500 ms |
| Master | PUCT MCTS + ONNX policy/value net (WebGPU → WASM), else MCTS fallback | 400 sims / 5000 ms |

## Known limitations (honest)

- No quiescence search — the horizon effect applies to alpha-beta levels.
- VCF search follows fours only, not open-three threats; defender counter-threats
  that win later are not modelled.
- MCTS is single-threaded, batch size 1 (one net call per expansion).
- Search branching is capped (~12–14 moves by pattern score); quiet positional
  moves off the beaten track can be pruned.
- Win rates from search levels are a heuristic squash, not calibrated
  probabilities.
- The bundled demo net (when the ML workstream ships it) is small; Master with
  it plays at strong search level, not superhuman.

## Rules (src/engine/gomoku.ts)

Standard 15×15 Gomoku, Black first, five-or-more wins (overline counts as a
win), no captures, no Renju forbidden moves, no swap2. `GameState` is
immutable; `applyMove` returns a new state and throws on illegal moves.

/**
 * ONNX policy+value model loading and inference (Master level).
 *
 * CRITICAL: `onnxruntime-web` is NEVER imported at the top level. It is
 * loaded with a dynamic import inside `loadModel`/`predict`, so importing
 * this module in Node/vitest (no WebGPU, no DOM) never crashes. Unit tests
 * never call loadModel — they only test the Master level's MCTS fallback.
 *
 * Model I/O contract — MUST match the Python export exactly:
 * - Input: single input named `input`, float32 tensor [1, 8, 15, 15].
 *   Planes (all relative to the player to move, unless noted):
 *     0  current-player stones
 *     1  opponent stones
 *     2  last move one-hot (zeros if the history is unknown)
 *     3  side-to-move plane: all 1.0 if Black to move, else all 0.0
 *     4  stones of the player who was to move 2 plies ago, as of 2 plies ago
 *     5  stones of that player's opponent, as of 2 plies ago
 *     6  stones of the player who was to move 4 plies ago, as of 4 plies ago
 *     7  stones of that player's opponent, as of 4 plies ago
 *   Planes 4-7 are zeros when the history is too short. Note that 2 and 4
 *   are even ply counts, so the player to move at those earlier positions is
 *   the SAME player as now — plane 4 is the current player's stones 2 plies
 *   ago, plane 5 the opponent's, and likewise for planes 6/7.
 * - Outputs: `policy` [1, 225] logits, row-major (index y*15+x); `value`
 *   [1, 1] in [-1, 1] from the current player's perspective.
 *   Win rate = (value + 1) / 2, or the visit-count share in MCTS.
 *
 * Execution providers: WebGPU is tried first (best on the A20 Pro-class
 * hardware this project targets), falling back to WASM. `getBackend()`
 * reports which provider is actually in use. Sessions are cached by URL and
 * warmed up with one inference on load.
 *
 * Batching: v1 runs one inference per MCTS expansion (batch=1). Batching
 * multiple leaf evaluations per net call is future work — it would roughly
 * halve Master-level latency on GPU, but needs a batched MCTS loop.
 *
 * Deployment note: if WASM loading fails in production, copy
 * node_modules/onnxruntime-web/dist/*.wasm into public/ and set
 * `ort.env.wasm.wasmPaths` to the served path (see vite.config.ts comment).
 */

import { SIZE, type Player, type Point } from '../engine/gomoku';
import type { InferenceSession } from 'onnxruntime-web';

// Lazy ort module handle — populated on first use, never at import time.
let ortModule: typeof import('onnxruntime-web') | null = null;

async function getOrt(): Promise<typeof import('onnxruntime-web')> {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web');
  }
  return ortModule;
}

export type OnnxBackend = 'webgpu' | 'wasm' | 'none';

let currentBackend: OnnxBackend = 'none';

/** Which execution provider is actually in use ('none' if no model is loaded). */
export function getBackend(): OnnxBackend {
  return currentBackend;
}

interface CachedModel {
  session: InferenceSession;
  backend: Exclude<OnnxBackend, 'none'>;
}

const sessionCache = new Map<string, Promise<CachedModel>>();

/**
 * Load (or return the cached) ONNX session for `url`. Tries WebGPU first,
 * falls back to WASM. Warms up with one inference on an empty board.
 * Throws if neither provider can create a session — callers (alphazero.ts)
 * catch this and fall back to MCTS.
 */
export async function loadModel(url: string): Promise<CachedModel> {
  const cached = sessionCache.get(url);
  if (cached) return cached;
  const pending = (async (): Promise<CachedModel> => {
    const ort = await getOrt();
    let session: InferenceSession | null = null;
    let backend: Exclude<OnnxBackend, 'none'> = 'wasm';
    try {
      session = await ort.InferenceSession.create(url, {
        executionProviders: ['webgpu'],
      });
      backend = 'webgpu';
    } catch {
      session = await ort.InferenceSession.create(url, {
        executionProviders: ['wasm'],
      });
      backend = 'wasm';
    }
    currentBackend = backend;
    // Warm-up: one inference on an empty board so the first real move
    // doesn't pay shader/pipeline compilation cost.
    await predict(session, new Int8Array(SIZE * SIZE), 1);
    return { session, backend };
  })();
  sessionCache.set(url, pending);
  // Don't cache failures: a retry after a transient error should re-attempt.
  pending.catch(() => sessionCache.delete(url));
  return pending;
}

export interface NetOutput {
  /** Policy logits, length 225, row-major (index y*15+x). */
  policy: Float32Array;
  /** Value in [-1, 1] from the current player's perspective. */
  value: number;
}

/**
 * Build the [1, 8, 15, 15] input planes for a position. Pure function —
 * safe to call (and unit-test) anywhere, no ort dependency.
 *
 * `history` is the full move list in order; when absent or too short, the
 * last-move / history planes are zeros. The history is assumed consistent
 * with `board` and `toMove` (the last entries of `history` are the stones
 * most recently placed).
 */
export function buildInputPlanes(
  board: ArrayLike<number>,
  toMove: Player,
  history?: Point[],
): Float32Array {
  const planes = new Float32Array(8 * SIZE * SIZE);
  const plane = (p: number) => p * SIZE * SIZE;

  // Planes 0/1: current position, relative to the side to move.
  for (let i = 0; i < SIZE * SIZE; i++) {
    const v = board[i];
    if (v === toMove) planes[plane(0) + i] = 1;
    else if (v !== 0) planes[plane(1) + i] = 1;
  }

  // Plane 2: last move one-hot.
  if (history && history.length > 0) {
    const last = history[history.length - 1];
    planes[plane(2) + last.y * SIZE + last.x] = 1;
  }

  // Plane 3: side to move (all 1.0 if Black to move, else all 0.0).
  if (toMove === 1) planes.fill(1, plane(3), plane(3) + SIZE * SIZE);

  // Planes 4-7: positions 2 and 4 plies ago. The board N plies ago is the
  // current board minus the last N moves. Because 2 and 4 are even, the
  // player to move at those earlier positions is the same as now, so "own"
  // stones then are the current player's stones, and "opponent" stones then
  // are the current opponent's stones.
  if (history && history.length >= 2) {
    const removed = new Set<number>();
    for (let k = 1; k <= 2; k++) {
      const m = history[history.length - k];
      removed.add(m.y * SIZE + m.x);
    }
    for (let i = 0; i < SIZE * SIZE; i++) {
      if (removed.has(i)) continue;
      const v = board[i];
      if (v === toMove) planes[plane(4) + i] = 1;
      else if (v !== 0) planes[plane(5) + i] = 1;
    }
  }
  if (history && history.length >= 4) {
    const removed = new Set<number>();
    for (let k = 1; k <= 4; k++) {
      const m = history[history.length - k];
      removed.add(m.y * SIZE + m.x);
    }
    for (let i = 0; i < SIZE * SIZE; i++) {
      if (removed.has(i)) continue;
      const v = board[i];
      if (v === toMove) planes[plane(6) + i] = 1;
      else if (v !== 0) planes[plane(7) + i] = 1;
    }
  }

  return planes;
}

/** Run one inference (batch=1). Returns raw policy logits + value. */
export async function predict(
  session: InferenceSession,
  board: ArrayLike<number>,
  toMove: Player,
  history?: Point[],
): Promise<NetOutput> {
  const ort = await getOrt();
  const input = buildInputPlanes(board, toMove, history);
  const tensor = new ort.Tensor('float32', input, [1, 8, SIZE, SIZE]);
  try {
    const results = await session.run({ input: tensor });
    const policyTensor = results['policy'];
    const valueTensor = results['value'];
    if (!policyTensor || !valueTensor) {
      throw new Error('Model must output tensors named "policy" and "value"');
    }
    const policy = policyTensor.data as Float32Array;
    const value = (valueTensor.data as Float32Array | number[])[0];
    if (policy.length !== SIZE * SIZE) {
      throw new Error(`Expected policy of length 225, got ${policy.length}`);
    }
    return { policy, value: typeof value === 'number' ? value : 0 };
  } finally {
    tensor.dispose?.();
  }
}

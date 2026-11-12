/**
 * L5 Master (AlphaZero-style) — PUCT MCTS guided by an ONNX policy+value net.
 *
 * When a model URL is provided and loads, each MCTS expansion queries the
 * net for policy priors (softmaxed over legal candidate moves) and a leaf
 * value; the tree search is the shared `runMcts` core from mcts.ts.
 *
 * When no model URL is given, or loading / inference fails, Master falls back
 * to Expert-style heuristic MCTS and reports `backend: 'mcts-fallback'` —
 * the UI must never claim a neural net was used when it was not.
 *
 * Honesty: this level's strength depends entirely on the loaded model. The
 * bundled model is a small demo net; with it, Master plays at strong search
 * level, not superhuman. Nothing in this file claims otherwise.
 */

import { SIZE, type Player, type Point } from '../engine/gomoku';
import { chooseMctsMove, runMcts, type PolicyValue } from './mcts';
import { getBackend, loadModel, predict } from './onnxModel';
import { orderedCandidates } from './patterns';
import type { MctsResult } from './mcts';

export interface MasterOptions {
  sims: number;
  timeMs: number;
  temperature?: number;
  /** URL of the ONNX model. Omit for MCTS fallback. */
  modelUrl?: string;
  /** Full move history (for the net's history planes). */
  history?: Point[];
  /** Root Dirichlet noise epsilon (default 0 = off; used for self-play, not play). */
  dirichletEps?: number;
  rng?: () => number;
}

export type MasterResult = MctsResult & { backend: string };

export async function chooseMasterMove(
  boardIn: ArrayLike<number>,
  player: Player,
  opts: MasterOptions,
): Promise<MasterResult> {
  const board = Int8Array.from(boardIn as ArrayLike<number>);

  if (!opts.modelUrl) {
    return masterFallback(board, player, opts);
  }

  try {
    const { session } = await loadModel(opts.modelUrl);
    const backend = getBackend(); // 'webgpu' | 'wasm' — the provider actually in use.
    const rootHistory = opts.history ?? [];
    const rng = opts.rng ?? Math.random;

    const netPolicyValue = async (
      b: Int8Array,
      p: Player,
      path: Point[],
    ): Promise<PolicyValue> => {
      const history = [...rootHistory, ...path];
      const { policy, value } = await predict(session, b, p, history);
      // Softmax the policy logits over legal candidate moves only.
      const cands = orderedCandidates(b, p, 24);
      const priors = new Map<number, number>();
      let maxLogit = -Infinity;
      for (const c of cands) {
        const l = policy[c.y * SIZE + c.x];
        if (Number.isFinite(l)) maxLogit = Math.max(maxLogit, l);
      }
      let sum = 0;
      for (const c of cands) {
        const l = policy[c.y * SIZE + c.x];
        const w = Number.isFinite(l) && maxLogit !== -Infinity ? Math.exp(l - maxLogit) : 0;
        priors.set(c.y * SIZE + c.x, w);
        sum += w;
      }
      if (sum <= 0 || !Number.isFinite(sum)) {
        // Degenerate policy (all -inf / NaN): fall back to uniform priors.
        for (const c of cands) priors.set(c.y * SIZE + c.x, 1 / Math.max(1, cands.length));
      } else {
        for (const [k, w] of priors) priors.set(k, w / sum);
      }
      const v = Number.isFinite(value) ? Math.max(-1, Math.min(1, value)) : 0;
      return { priors, value: v };
    };

    const result = await runMcts(
      board,
      player,
      {
        sims: opts.sims,
        timeMs: opts.timeMs,
        cpuct: 1.5,
        branchLimit: 24,
        temperature: opts.temperature ?? 0,
        dirichletEps: opts.dirichletEps ?? 0,
        dirichletAlpha: 0.03,
        rng,
      },
      netPolicyValue,
    );
    return { ...result, backend: `onnx-${backend}` };
  } catch {
    // Model load or inference failed: degrade gracefully to heuristic MCTS.
    return masterFallback(board, player, opts);
  }
}

async function masterFallback(
  board: Int8Array,
  player: Player,
  opts: MasterOptions,
): Promise<MasterResult> {
  const result = await chooseMctsMove(board, player, {
    sims: opts.sims,
    timeMs: opts.timeMs,
    temperature: opts.temperature,
    rng: opts.rng,
  });
  return { ...result, backend: 'mcts-fallback' };
}

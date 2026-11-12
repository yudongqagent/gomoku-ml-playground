/**
 * L4 Expert — Monte Carlo tree search, and the shared PUCT core used by L5.
 *
 * One implementation serves both levels; they differ only in the
 * `PolicyValueFn` that supplies move priors and leaf values:
 * - Expert: heuristic priors from patterns.ts + heuristic rollouts.
 * - Master: ONNX policy+value net (see alphazero.ts).
 *
 * Selection uses PUCT: score = Q + cpuct * prior * sqrt(parentVisits) / (1 + visits),
 * where Q is the mean value from the perspective of the player to move at the
 * parent node. Values are backed up with a sign flip at every level (players
 * alternate), so each node's valueSum is always from its parent-player's
 * perspective — exactly what selection needs.
 *
 * Win rates come from visit counts / mean values: the best move's win rate is
 * (Q_best + 1) / 2, and top moves are the most-visited root children.
 *
 * Known limitations: batch size 1 (one net call per expansion for Master —
 * batching is future work), no transposition table in the tree, rollouts are
 * heuristic-random (not learned), and the search is single-threaded.
 */

import { SIZE, hasFive, opponent, type Player, type Point } from '../engine/gomoku';
import {
  candidateMoves,
  evaluateBoard,
  orderedCandidates,
  scoreMove,
} from './patterns';
import type { TopMove } from './types';

/** Priors over candidate moves (keys are cell indices y*15+x) + a leaf value in [-1, 1]. */
export interface PolicyValue {
  priors: Map<number, number>;
  /** Value from the perspective of the player to move at the evaluated node. */
  value: number;
}

/**
 * Supplies priors + value for a node. `path` is the sequence of moves from
 * the search root to this node (empty for the root) — the Master level uses
 * it to build the net's history planes.
 */
export type PolicyValueFn = (
  board: Int8Array,
  player: Player,
  path: Point[],
) => Promise<PolicyValue> | PolicyValue;

export interface MctsOptions {
  sims: number;
  timeMs: number;
  cpuct?: number;
  branchLimit?: number;
  /** 0 = pick the most-visited move; > 0 samples from visits^(1/temperature). */
  temperature?: number;
  /** Root Dirichlet noise epsilon (0 = off). Used for self-play exploration, not for play. */
  dirichletEps?: number;
  dirichletAlpha?: number;
  rng?: () => number;
}

export interface MctsResult {
  move: Point;
  winRate: number;
  topMoves: TopMove[];
  nodes: number;
  ms: number;
  depth: number;
}

interface MctsNode {
  /** Cell index of the move leading here, or -1 for the root. */
  moveIdx: number;
  /** Player to move at this node. */
  player: Player;
  parent: MctsNode | null;
  children: Map<number, MctsNode>;
  /** Priors for this node's candidate moves (populated on expansion). */
  priors: Map<number, number>;
  /** Candidate move indices not yet expanded, best prior first. */
  unexpanded: number[];
  visits: number;
  /** Sum of values from the perspective of the player to move at the parent. */
  valueSum: number;
  expanded: boolean;
  /** True if the move leading here won the game for its mover. */
  terminal: boolean;
  /** Value from the perspective of the player to move here (-1: the mover won). */
  terminalValue: number;
}

function createNode(moveIdx: number, player: Player, parent: MctsNode | null): MctsNode {
  return {
    moveIdx,
    player,
    parent,
    children: new Map(),
    priors: new Map(),
    unexpanded: [],
    visits: 0,
    valueSum: 0,
    expanded: false,
    terminal: false,
    terminalValue: 0,
  };
}

function expandNode(node: MctsNode, priors: Map<number, number>): void {
  node.priors = priors;
  node.unexpanded = [...priors.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([idx]) => idx);
  node.expanded = true;
}

function selectChild(node: MctsNode, cpuct: number): MctsNode {
  let best: MctsNode | null = null;
  let bestScore = -Infinity;
  const parentVisits = Math.sqrt(Math.max(1, node.visits));
  for (const child of node.children.values()) {
    const q = child.visits > 0 ? child.valueSum / child.visits : 0;
    const prior = node.priors.get(child.moveIdx) ?? 0;
    const u = cpuct * prior * parentVisits / (1 + child.visits);
    const score = q + u;
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }
  // Caller only invokes this when children exist.
  return best as MctsNode;
}

/** Marsaglia–Tsang gamma sampler, used for Dirichlet noise. */
function sampleGamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    // Boost: Gamma(a) = Gamma(a+1) * U^(1/a).
    return sampleGamma(shape + 1, rng) * Math.pow(rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0;
    let v = 0;
    do {
      // Box–Muller normal.
      const u1 = Math.max(rng(), 1e-12);
      const u2 = rng();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.max(rng(), 1e-12);
    if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function applyDirichletNoise(
  node: MctsNode,
  eps: number,
  alpha: number,
  rng: () => number,
): void {
  const indices = [...node.priors.keys()];
  if (indices.length === 0) return;
  const samples = indices.map(() => sampleGamma(alpha, rng));
  const sum = samples.reduce((a, b) => a + b, 0);
  indices.forEach((idx, i) => {
    const prior = node.priors.get(idx) ?? 0;
    node.priors.set(idx, (1 - eps) * prior + eps * (sum > 0 ? samples[i] / sum : 1 / indices.length));
  });
  node.unexpanded = [...node.priors.entries()]
    .sort((a, b) => b[1] - a[1] || a[0] - b[0])
    .map(([idx]) => idx);
}

export async function runMcts(
  boardIn: ArrayLike<number>,
  player: Player,
  opts: MctsOptions,
  policyValue: PolicyValueFn,
): Promise<MctsResult> {
  const started = performance.now();
  const deadline = started + Math.max(1, opts.timeMs);
  const cpuct = opts.cpuct ?? 1.5;
  const branchLimit = opts.branchLimit ?? 24;
  const temperature = opts.temperature ?? 0;
  const rng = opts.rng ?? Math.random;

  const board = Int8Array.from(boardIn as ArrayLike<number>);
  const root = createNode(-1, player, null);
  const rootPv = await policyValue(board, player, []);
  expandNode(root, rootPv.priors);
  if (opts.dirichletEps && opts.dirichletEps > 0) {
    applyDirichletNoise(root, opts.dirichletEps, opts.dirichletAlpha ?? 0.03, rng);
  }
  if (root.unexpanded.length === 0 && root.children.size === 0) {
    throw new Error('No legal moves: position is terminal');
  }

  let sims = 0;
  let nodesCreated = 1;
  let maxDepth = 0;

  while (sims < opts.sims) {
    if ((sims & 3) === 0 && sims > 0 && performance.now() > deadline) break;

    // Selection: descend while the node is fully expanded and has children.
    let node = root;
    const path: MctsNode[] = [root];
    while (
      node.expanded &&
      node.unexpanded.length === 0 &&
      node.children.size > 0 &&
      !node.terminal
    ) {
      node = selectChild(node, cpuct);
      board[node.moveIdx] = node.parent ? node.parent.player : player;
      path.push(node);
    }

    // Leaf handling: expand one child, or take a terminal / no-move value.
    let value: number; // From the perspective of the player to move at `node`.
    if (node.terminal) {
      value = node.terminalValue;
    } else if (node.unexpanded.length > 0) {
      const moveIdx = node.unexpanded.shift() as number;
      const mover = node.player;
      board[moveIdx] = mover;
      const child = createNode(moveIdx, opponent(mover), node);
      node.children.set(moveIdx, child);
      nodesCreated++;
      path.push(child);
      const x = moveIdx % SIZE;
      const y = Math.floor(moveIdx / SIZE);
      if (hasFive(board, x, y)) {
        child.terminal = true;
        child.terminalValue = -1; // The mover won; bad for the player to move here.
        value = -1;
      } else {
        const pathMoves = path.slice(1).map((n) => ({
          x: n.moveIdx % SIZE,
          y: Math.floor(n.moveIdx / SIZE),
        }));
        const pv = await policyValue(board, child.player, pathMoves);
        expandNode(child, pv.priors);
        value = pv.value;
      }
      node = child;
    } else {
      // Expanded node with no children: no legal moves (draw).
      value = 0;
    }

    // Backup with a sign flip at every level. Invariant: at node N, `v` is
    // the value from N.player's perspective, so -v is the value from the
    // perspective of the player who moved into N (N.valueSum's convention).
    let v = value;
    for (let i = path.length - 1; i >= 0; i--) {
      const n = path[i];
      n.visits++;
      n.valueSum += -v;
      v = -v;
    }
    if (path.length - 1 > maxDepth) maxDepth = path.length - 1;

    // Unmake all moves played this simulation.
    for (let i = path.length - 1; i >= 1; i--) board[path[i].moveIdx] = 0;
    sims++;
  }

  if (root.children.size === 0) throw new Error('No legal moves: position is terminal');

  const children = [...root.children.values()];
  children.sort((a, b) => b.visits - a.visits || b.valueSum - a.valueSum || a.moveIdx - b.moveIdx);

  let best: MctsNode;
  if (temperature > 0 && children.length > 1) {
    const weights = children.map((c) => Math.pow(c.visits, 1 / temperature));
    const total = weights.reduce((a, b) => a + b, 0);
    let r = rng() * total;
    best = children[0];
    for (let i = 0; i < children.length; i++) {
      r -= weights[i];
      if (r <= 0) {
        best = children[i];
        break;
      }
    }
  } else {
    best = children[0];
  }

  const topMoves: TopMove[] = children.slice(0, 5).map((c) => {
    const q = c.visits > 0 ? c.valueSum / c.visits : 0;
    return {
      move: { x: c.moveIdx % SIZE, y: Math.floor(c.moveIdx / SIZE) },
      winRate: Math.min(1, Math.max(0, (q + 1) / 2)),
      visits: c.visits,
    };
  });
  const bestQ = best.visits > 0 ? best.valueSum / best.visits : 0;

  return {
    move: { x: best.moveIdx % SIZE, y: Math.floor(best.moveIdx / SIZE) },
    winRate: Math.min(1, Math.max(0, (bestQ + 1) / 2)),
    topMoves,
    nodes: nodesCreated,
    ms: performance.now() - started,
    depth: maxDepth,
  };
}

/* ------------------------------------------------------------------ */
/* L4 Expert: heuristic priors + heuristic rollouts                    */
/* ------------------------------------------------------------------ */

const ROLLOUT_MAX_PLIES = 40;

/** Find an immediate winning move among the given candidates (mutates then restores `board`). */
function winningMoveAmong(board: Int8Array, mover: Player, cands: Point[]): Point | null {
  for (const c of cands) {
    const i = c.y * SIZE + c.x;
    board[i] = mover;
    const win = hasFive(board, c.x, c.y);
    board[i] = 0;
    if (win) return c;
  }
  return null;
}

/**
 * Heuristic rollout: play semi-random moves (take immediate wins, block
 * immediate losses, otherwise sample a few candidates and take the
 * best-scoring one) until someone wins or the ply cap is hit. Returns the
 * outcome from `player`'s perspective: +1 win, -1 loss, or a pattern-eval
 * squash in (-1, 1) if the cap is hit.
 */
function rollout(boardIn: Int8Array, player: Player, rng: () => number): number {
  const board = Int8Array.from(boardIn);
  let turn: Player = player;
  for (let ply = 0; ply < ROLLOUT_MAX_PLIES; ply++) {
    const cands = candidateMoves(board, 1);
    if (cands.length === 0) return 0; // Full board: draw.
    let move = winningMoveAmong(board, turn, cands);
    if (!move) move = winningMoveAmong(board, opponent(turn), cands);
    if (!move) {
      // Sample up to 5 random candidates, play the best-scoring one.
      let bestCand = cands[Math.floor(rng() * cands.length)];
      let bestScore = -Infinity;
      const samples = Math.min(5, cands.length);
      for (let s = 0; s < samples; s++) {
        const c = cands[Math.floor(rng() * cands.length)];
        const sc = scoreMove(board, c.x, c.y, turn);
        if (sc > bestScore) {
          bestScore = sc;
          bestCand = c;
        }
      }
      move = bestCand;
    }
    const i = move.y * SIZE + move.x;
    board[i] = turn;
    if (hasFive(board, move.x, move.y)) return turn === player ? 1 : -1;
    turn = opponent(turn);
  }
  return Math.tanh(evaluateBoard(board, player) / 3000);
}

/**
 * Expert policy+value: log-scaled pattern scores as priors, rollout outcome
 * as the leaf value.
 */
export function heuristicPolicyValue(
  board: Int8Array,
  player: Player,
  rng: () => number = Math.random,
): PolicyValue {
  const cands = orderedCandidates(board, player, 24);
  const priors = new Map<number, number>();
  let sum = 0;
  for (const c of cands) {
    // Log scale: pattern scores span orders of magnitude; log1p keeps the
    // prior spread usable instead of collapsing onto one move.
    const w = 1 + Math.log1p(Math.max(0, scoreMove(board, c.x, c.y, player)));
    priors.set(c.y * SIZE + c.x, w);
    sum += w;
  }
  for (const [k, w] of priors) priors.set(k, sum > 0 ? w / sum : 1 / Math.max(1, cands.length));
  return { priors, value: rollout(board, player, rng) };
}

export interface ExpertOptions {
  sims: number;
  timeMs: number;
  temperature?: number;
  rng?: () => number;
}

export async function chooseMctsMove(
  board: ArrayLike<number>,
  player: Player,
  opts: ExpertOptions,
): Promise<MctsResult & { backend: string }> {
  const rng = opts.rng ?? Math.random;
  const result = await runMcts(
    board,
    player,
    {
      sims: opts.sims,
      timeMs: opts.timeMs,
      cpuct: 1.5,
      branchLimit: 24,
      temperature: opts.temperature ?? 0,
      rng,
    },
    (b, p) => heuristicPolicyValue(b, p, rng),
  );
  return { ...result, backend: 'mcts' };
}

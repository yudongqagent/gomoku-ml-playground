/**
 * L1 Beginner — 1-ply pattern heuristic with occasional random slips.
 *
 * Picks the best-scoring move by the pattern heuristic most of the time, but
 * with a configurable probability (~15% by default) picks a random move among
 * the next-best candidates instead, like a human beginner's lapse in
 * concentration.
 *
 * Safety: immediate wins and blocks of the opponent's immediate win are
 * guaranteed by the shared pre-check in `chooseMove` (src/ai/engine.ts),
 * which runs before any level — so Beginner never misses those, even on a
 * "random slip" turn. This module itself just picks a good-looking move.
 */

import { SIZE, type Player } from '../engine/gomoku';
import { evaluateBoard, orderedCandidates, winRateFromEval } from './patterns';
import type { TopMove } from './types';

export interface BeginnerOptions {
  /** Probability of picking a random move among the top candidates (default 0.15). */
  randomRate?: number;
  /** Injectable RNG for deterministic tests. */
  rng?: () => number;
}

export interface BeginnerResult {
  move: { x: number; y: number };
  winRate: number;
  topMoves: TopMove[];
  nodes: number;
  ms: number;
  depth: number;
  backend: string;
}

export function chooseBeginnerMove(
  boardIn: ArrayLike<number>,
  player: Player,
  opts: BeginnerOptions = {},
): BeginnerResult {
  const started = performance.now();
  const board = Int8Array.from(boardIn as ArrayLike<number>);
  const randomRate = opts.randomRate ?? 0.15;
  const rng = opts.rng ?? Math.random;

  const cands = orderedCandidates(board, player, 8);
  // orderedCandidates always returns at least one move unless the board is full.
  if (cands.length === 0) throw new Error('No legal moves: position is terminal');

  let pick = cands[0];
  if (cands.length > 1 && rng() < randomRate) {
    // Random slip: pick uniformly among the 2nd–5th best candidates.
    const pool = Math.min(4, cands.length - 1);
    pick = cands[1 + Math.floor(rng() * pool)];
  }

  // Top moves with 1-ply eval win rates (best move first).
  const topMoves: TopMove[] = cands.slice(0, 5).map((c) => {
    const i = c.y * SIZE + c.x;
    board[i] = player;
    const ev = evaluateBoard(board, player);
    board[i] = 0;
    return { move: c, winRate: winRateFromEval(ev) };
  });
  // Keep the picked move first in spirit: topMoves is sorted by win rate, and
  // the picked move is one of them; the reported winRate is the picked move's.
  topMoves.sort((a, b) => b.winRate - a.winRate);
  const picked = topMoves.find((t) => t.move.x === pick.x && t.move.y === pick.y);
  const winRate = picked ? picked.winRate : winRateFromEval(0);

  return {
    move: pick,
    winRate,
    topMoves,
    nodes: cands.length,
    ms: performance.now() - started,
    depth: 1,
    backend: 'heuristic',
  };
}

/**
 * AI entry point: `chooseMove` / `analyzePosition`.
 *
 * Pure functions usable from the Web Worker AND directly (tests import them;
 * the worker is a thin wrapper). Both never throw on a legal non-terminal
 * position: they always return a legal move. (A position with no legal moves
 * at all — a full board — throws a clear Error; that is a terminal position,
 * not a legal one to move in.)
 *
 * SHARED PRE-CHECK (guarantees for ALL levels): before any level runs,
 *  1. if the side to move has an immediate winning move, it is played;
 *  2. else if the opponent has an immediate winning move, it is blocked
 *     (if the opponent has two or more winning squares — a double threat —
 *     the position is lost; we still play the best-scoring block).
 * This guarantees every level takes immediate wins and blocks immediate
 * losses, regardless of how weak its search is. The pre-check is what makes
 * the Beginner level's "random slip" safe: randomness only applies to moves
 * that are not forced.
 *
 * Level dispatch:
 * - beginner: 1-ply pattern heuristic (+ occasional random slip).
 * - intermediate: alpha-beta, iterative deepening to depth 3.
 * - advanced: alpha-beta to depth 8 + VCF forced-win search.
 * - expert: heuristic MCTS.
 * - master: PUCT MCTS with the ONNX net, or MCTS fallback without a model.
 */

import { SIZE, type Player, type Point } from '../engine/gomoku';
import { chooseMasterMove } from './alphazero';
import { chooseBeginnerMove } from './beginner';
import { LEVEL_META } from './levels';
import { chooseMctsMove } from './mcts';
import { evaluateBoard, findWinningMove, findWinningMoves, scoreMove, winRateFromEval } from './patterns';
import { searchMove } from './search';
import type { AiMoveRequest, AiMoveResult, AnalysisResult, TopMove } from './types';

interface RunResult {
  move: Point;
  winRate: number;
  topMoves: TopMove[];
  nodes: number;
  ms: number;
  depth?: number;
  backend?: string;
}

function toBoard(req: AiMoveRequest): Int8Array {
  if (!req.board || req.board.length !== SIZE * SIZE) {
    throw new Error(`Board must have ${SIZE * SIZE} entries, got ${req.board?.length ?? 0}`);
  }
  return Int8Array.from(req.board);
}

function hasLegalMove(board: Int8Array): boolean {
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 0) return true;
  }
  return false;
}

async function runAi(req: AiMoveRequest): Promise<RunResult> {
  const started = performance.now();
  const board = toBoard(req);
  const player: Player = req.toMove;
  const meta = LEVEL_META[req.level];
  const sims = req.options?.sims ?? meta.defaultSims;
  const timeMs = req.options?.timeMs ?? meta.defaultTimeMs;
  const temperature = req.options?.temperature ?? 0;

  if (!hasLegalMove(board)) {
    throw new Error('No legal moves: position is terminal');
  }

  // --- Shared pre-check 1: take an immediate win. ---
  const ownWin = findWinningMove(board, player);
  if (ownWin) {
    return {
      move: ownWin,
      winRate: 1,
      topMoves: [{ move: ownWin, winRate: 1 }],
      nodes: 1,
      ms: performance.now() - started,
      depth: 0,
      backend: 'forced-win',
    };
  }

  // --- Shared pre-check 2: block the opponent's immediate win(s). ---
  const oppWins = findWinningMoves(board, player === 1 ? -1 : 1);
  if (oppWins.length > 0) {
    // Play the best-scoring block. With 2+ winning squares the position is
    // lost (a double threat can't be blocked with one stone); we still play
    // the best block and report a near-zero win rate honestly.
    let block = oppWins[0];
    let bestScore = -Infinity;
    const topMoves: TopMove[] = [];
    for (const w of oppWins) {
      const s = scoreMove(board, w.x, w.y, player);
      if (s > bestScore) {
        bestScore = s;
        block = w;
      }
      const i = w.y * SIZE + w.x;
      board[i] = player;
      const ev = evaluateBoard(board, player);
      board[i] = 0;
      topMoves.push({ move: w, winRate: winRateFromEval(ev) });
    }
    topMoves.sort((a, b) => b.winRate - a.winRate);
    const winRate =
      oppWins.length > 1
        ? 0.05
        : topMoves.find((t) => t.move.x === block.x && t.move.y === block.y)?.winRate ??
          winRateFromEval(0);
    return {
      move: block,
      winRate,
      topMoves: topMoves.slice(0, 5),
      nodes: 1,
      ms: performance.now() - started,
      depth: 0,
      backend: 'forced-block',
    };
  }

  // --- Level dispatch. ---
  switch (req.level) {
    case 'beginner': {
      const r = chooseBeginnerMove(board, player);
      return { ...r, ms: performance.now() - started };
    }
    case 'intermediate': {
      const r = searchMove(board, player, { maxDepth: 3, timeMs, branchLimit: 12 });
      return { ...r, ms: performance.now() - started };
    }
    case 'advanced': {
      const r = searchMove(board, player, {
        maxDepth: 8,
        timeMs,
        branchLimit: 12,
        useVct: true,
        vctPlies: 9,
      });
      return { ...r, ms: performance.now() - started };
    }
    case 'expert': {
      const r = await chooseMctsMove(board, player, { sims, timeMs, temperature });
      return { ...r, ms: performance.now() - started };
    }
    case 'master': {
      const r = await chooseMasterMove(board, player, {
        sims,
        timeMs,
        temperature,
        modelUrl: req.options?.modelUrl,
        history: req.history,
      });
      return { ...r, ms: performance.now() - started };
    }
    default: {
      const _exhaustive: never = req.level;
      throw new Error(`Unknown AI level: ${_exhaustive as string}`);
    }
  }
}

/** Choose a move for the side to move at the requested level. */
export async function chooseMove(req: AiMoveRequest): Promise<AiMoveResult> {
  const r = await runAi(req);
  return {
    move: r.move,
    winRate: r.winRate,
    topMoves: r.topMoves,
    nodes: r.nodes,
    ms: r.ms,
    depth: r.depth,
    backend: r.backend,
  };
}

/**
 * Analyse a position (same search as chooseMove, surfaced as analysis).
 * `bestMove` is the recommended move for the side to move.
 */
export async function analyzePosition(req: AiMoveRequest): Promise<AnalysisResult> {
  const r = await runAi(req);
  return {
    bestMove: r.move,
    winRate: r.winRate,
    topMoves: r.topMoves,
    nodes: r.nodes,
    ms: r.ms,
    depth: r.depth,
    backend: r.backend,
  };
}

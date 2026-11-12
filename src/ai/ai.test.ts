import { describe, expect, it } from 'vitest';
import { SIZE, stateAtMove, type Point } from '../engine/gomoku';
import { chooseMove, analyzePosition } from './engine';
import {
  candidateMoves,
  evaluateBoard,
  findWinningMove,
  findWinningMoves,
  winRateFromEval,
} from './patterns';
import { findForcedWin } from './search';
import type { AiMoveRequest, LevelId } from './types';

const LEVELS: LevelId[] = ['beginner', 'intermediate', 'advanced', 'expert', 'master'];

/** Small budgets keep the suite fast; production defaults live in LEVEL_META. */
const SMALL_BUDGET = { sims: 60, timeMs: 300 };

function emptyBoard(): number[] {
  return new Array(SIZE * SIZE).fill(0);
}

function boardWith(stones: Array<{ x: number; y: number; v: 1 | -1 }>): number[] {
  const b = emptyBoard();
  for (const s of stones) b[s.y * SIZE + s.x] = s.v;
  return b;
}

function req(board: number[], toMove: 1 | -1, level: LevelId): AiMoveRequest {
  return { board, toMove, level, options: { ...SMALL_BUDGET } };
}

function isLegalMove(board: number[], move: Point): boolean {
  return (
    move.x >= 0 &&
    move.x < SIZE &&
    move.y >= 0 &&
    move.y < SIZE &&
    board[move.y * SIZE + move.x] === 0
  );
}

/** A 12-move mid-game position with no fours and no immediate win for either side. */
const MID_GAME_MOVES: Point[] = [
  { x: 7, y: 7 },
  { x: 7, y: 8 },
  { x: 8, y: 8 },
  { x: 8, y: 7 },
  { x: 9, y: 7 },
  { x: 6, y: 6 },
  { x: 9, y: 8 },
  { x: 10, y: 10 },
  { x: 6, y: 8 },
  { x: 8, y: 6 },
  { x: 10, y: 7 },
  { x: 7, y: 9 },
];

function midGameBoard(): number[] {
  return Array.from(stateAtMove(MID_GAME_MOVES, MID_GAME_MOVES.length).board);
}

describe('patterns', () => {
  it('evaluates an empty board as 0', () => {
    expect(evaluateBoard(emptyBoard(), 1)).toBe(0);
  });

  it('rates an open four above an open three', () => {
    const four = boardWith([
      { x: 6, y: 7, v: 1 },
      { x: 7, y: 7, v: 1 },
      { x: 8, y: 7, v: 1 },
      { x: 9, y: 7, v: 1 },
    ]);
    const three = boardWith([
      { x: 7, y: 7, v: 1 },
      { x: 8, y: 7, v: 1 },
      { x: 9, y: 7, v: 1 },
    ]);
    expect(evaluateBoard(four, 1)).toBeGreaterThan(evaluateBoard(three, 1));
    expect(evaluateBoard(four, 1)).toBeGreaterThan(0);
    // Evaluation is from the side-to-move perspective: the same board is bad for White.
    expect(evaluateBoard(four, -1)).toBe(-evaluateBoard(four, 1));
  });

  it('centres on an empty board and finds winning moves', () => {
    expect(candidateMoves(emptyBoard())).toEqual([{ x: 7, y: 7 }]);
    const four = boardWith([
      { x: 7, y: 7, v: 1 },
      { x: 8, y: 7, v: 1 },
      { x: 9, y: 7, v: 1 },
      { x: 10, y: 7, v: 1 },
    ]);
    const wins = findWinningMoves(four, 1);
    expect(wins).toContainEqual({ x: 6, y: 7 });
    expect(wins).toContainEqual({ x: 11, y: 7 });
    expect(findWinningMove(four, 1)).not.toBeNull();
    expect(findWinningMove(four, -1)).toBeNull();
  });

  it('winRateFromEval is a sane squash', () => {
    expect(winRateFromEval(0)).toBeCloseTo(0.5, 5);
    expect(winRateFromEval(3000)).toBeGreaterThan(0.7);
    expect(winRateFromEval(-3000)).toBeLessThan(0.3);
    expect(winRateFromEval(10_000_000)).toBe(1);
    expect(winRateFromEval(-10_000_000)).toBe(0);
  });
});

describe('shared pre-check (all levels)', () => {
  it.each(LEVELS)('%s takes an immediate win', async (level) => {
    const board = boardWith([
      { x: 7, y: 7, v: 1 },
      { x: 8, y: 7, v: 1 },
      { x: 9, y: 7, v: 1 },
      { x: 10, y: 7, v: 1 },
      { x: 3, y: 3, v: -1 },
      { x: 4, y: 4, v: -1 },
    ]);
    const r = await chooseMove(req(board, 1, level));
    expect([{ x: 6, y: 7 }, { x: 11, y: 7 }]).toContainEqual(r.move);
    expect(r.winRate).toBe(1);
    expect(r.backend).toBe('forced-win');
  });

  it.each(LEVELS)('%s blocks an opponent open four', async (level) => {
    const board = boardWith([
      { x: 7, y: 7, v: -1 },
      { x: 8, y: 7, v: -1 },
      { x: 9, y: 7, v: -1 },
      { x: 10, y: 7, v: -1 },
      { x: 3, y: 3, v: 1 },
      { x: 4, y: 4, v: 1 },
    ]);
    const r = await chooseMove(req(board, 1, level));
    expect([{ x: 6, y: 7 }, { x: 11, y: 7 }]).toContainEqual(r.move);
    expect(r.backend).toBe('forced-block');
  });

  it('plays the best block against a double threat and reports a low win rate', async () => {
    const board = boardWith([
      { x: 7, y: 7, v: -1 },
      { x: 8, y: 7, v: -1 },
      { x: 9, y: 7, v: -1 },
      { x: 10, y: 7, v: -1 },
      { x: 3, y: 3, v: -1 },
      { x: 3, y: 4, v: -1 },
      { x: 3, y: 5, v: -1 },
      { x: 3, y: 6, v: -1 },
    ]);
    const r = await chooseMove(req(board, 1, 'advanced'));
    expect([
      { x: 6, y: 7 },
      { x: 11, y: 7 },
      { x: 3, y: 2 },
      { x: 3, y: 7 },
    ]).toContainEqual(r.move);
    expect(r.winRate).toBeLessThan(0.1);
  });
});

describe('each level returns a legal move', () => {
  it.each(LEVELS)('%s on an empty board', async (level) => {
    const board = emptyBoard();
    const r = await chooseMove(req(board, 1, level));
    expect(isLegalMove(board, r.move)).toBe(true);
    expect(r.winRate).toBeGreaterThanOrEqual(0);
    expect(r.winRate).toBeLessThanOrEqual(1);
    expect(r.topMoves.length).toBeGreaterThan(0);
    expect(r.topMoves.length).toBeLessThanOrEqual(5);
    for (const t of r.topMoves) {
      expect(isLegalMove(board, t.move)).toBe(true);
      expect(t.winRate).toBeGreaterThanOrEqual(0);
      expect(t.winRate).toBeLessThanOrEqual(1);
    }
    // Top moves sorted by win rate descending.
    for (let i = 1; i < r.topMoves.length; i++) {
      expect(r.topMoves[i].winRate).toBeLessThanOrEqual(r.topMoves[i - 1].winRate + 1e-9);
    }
  });

  it.each(LEVELS)('%s on a mid-game board', async (level) => {
    const board = midGameBoard();
    const r = await chooseMove(req(board, 1, level));
    expect(isLegalMove(board, r.move)).toBe(true);
    expect(r.winRate).toBeGreaterThanOrEqual(0);
    expect(r.winRate).toBeLessThanOrEqual(1);
    expect(r.nodes).toBeGreaterThan(0);
    expect(r.ms).toBeGreaterThanOrEqual(0);
  });
});

describe('level-specific behaviour', () => {
  it('master falls back to MCTS when no model URL is provided', async () => {
    const r = await chooseMove(req(midGameBoard(), 1, 'master'));
    expect(r.backend).toBe('mcts-fallback');
    expect(isLegalMove(midGameBoard(), r.move)).toBe(true);
  });

  it('master falls back to MCTS when the model URL cannot load', async () => {
    const r = await chooseMove({
      board: midGameBoard(),
      toMove: 1,
      level: 'master',
      options: { ...SMALL_BUDGET, modelUrl: '/models/does-not-exist.onnx' },
    });
    expect(r.backend).toBe('mcts-fallback');
  });

  it('advanced finds a forced win via VCF search', () => {
    // Genuine forced win (Black to move). Row 7: black three (6,7),(7,7),(8,7)
    // with White blocking (10,7). Row 3: black three (7,3),(8,3),(9,3).
    // Winning first moves: (9,7) makes a four White must block at (5,7),
    // after which (10,3)/(6,3) makes an open four White cannot block both
    // ends of — or Black plays the open four directly ((10,3), (6,3), or
    // (5,7) on row 7, whose four (5,7)-(8,7) has two winning squares).
    const board = boardWith([
      { x: 6, y: 7, v: 1 },
      { x: 7, y: 7, v: 1 },
      { x: 8, y: 7, v: 1 },
      { x: 7, y: 3, v: 1 },
      { x: 8, y: 3, v: 1 },
      { x: 9, y: 3, v: 1 },
      { x: 10, y: 7, v: -1 },
      { x: 3, y: 3, v: -1 },
      { x: 12, y: 12, v: -1 },
    ]);
    const move = findForcedWin(board, 1, 9);
    expect(move).not.toBeNull();
    expect([
      { x: 9, y: 7 },
      { x: 10, y: 3 },
      { x: 6, y: 3 },
      { x: 5, y: 7 },
    ]).toContainEqual(move);
    // The returned move must be a genuine threat: Black has a winning move
    // on the following turn.
    if (move) {
      const b = Int8Array.from(board);
      b[move.y * SIZE + move.x] = 1;
      expect(findWinningMoves(b, 1).length).toBeGreaterThan(0);
    }
  });

  it('VCF search returns null when no forced win exists', () => {
    // Black has one three (7,7),(8,7),(9,7) with (10,7) blocked by White.
    // (6,7) or (5,7) creates a single-ended four, White blocks the one
    // winning square, and Black has no follow-up threat — no forced win.
    const board = boardWith([
      { x: 7, y: 7, v: 1 },
      { x: 8, y: 7, v: 1 },
      { x: 9, y: 7, v: 1 },
      { x: 10, y: 7, v: -1 },
      { x: 3, y: 3, v: -1 },
    ]);
    expect(findForcedWin(board, 1, 9)).toBeNull();
  });

  it('analyzePosition returns a legal best move', async () => {
    const board = midGameBoard();
    const r = await analyzePosition(req(board, 1, 'intermediate'));
    expect(isLegalMove(board, r.bestMove)).toBe(true);
    expect(r.topMoves.length).toBeGreaterThan(0);
  });

  it('throws a clear error on a full board (terminal position)', async () => {
    const board = new Array(SIZE * SIZE).fill(1);
    await expect(chooseMove(req(board, 1, 'beginner'))).rejects.toThrow('No legal moves');
  });

  it('throws on a malformed board', async () => {
    await expect(chooseMove({ board: [0, 0, 0], toMove: 1, level: 'beginner' })).rejects.toThrow(
      '225 entries',
    );
  });
});

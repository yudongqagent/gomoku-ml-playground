import { describe, expect, it } from 'vitest';
import {
  SIZE,
  applyMove,
  checkWin,
  cloneState,
  hasFive,
  isDraw,
  isLegal,
  legalMoves,
  newGame,
  stateAtMove,
  type Point,
} from './gomoku';

function play(moves: Point[]) {
  let s = newGame();
  for (const m of moves) s = applyMove(s, m.x, m.y);
  return s;
}

/** Interleave black/white points into an alternating move list (Black first). */
function interleave(black: Point[], white: Point[]): Point[] {
  const out: Point[] = [];
  for (let i = 0; i < Math.max(black.length, white.length); i++) {
    if (i < black.length) out.push(black[i]);
    if (i < white.length) out.push(white[i]);
  }
  return out;
}

describe('newGame / applyMove', () => {
  it('starts empty with Black to move', () => {
    const s = newGame();
    expect(s.board.length).toBe(SIZE * SIZE);
    expect(s.board.every((v) => v === 0)).toBe(true);
    expect(s.toMove).toBe(1);
    expect(s.winner).toBe(0);
    expect(s.winningLine).toBeNull();
    expect(s.moves).toEqual([]);
    expect(legalMoves(s)).toHaveLength(225);
  });

  it('alternates players and is immutable', () => {
    const s0 = newGame();
    const s1 = applyMove(s0, 7, 7);
    expect(s0.board[7 * SIZE + 7]).toBe(0);
    expect(s0.moves).toHaveLength(0);
    expect(s1.board[7 * SIZE + 7]).toBe(1);
    expect(s1.toMove).toBe(-1);
    expect(s1.moves).toEqual([{ x: 7, y: 7 }]);
    const s2 = applyMove(s1, 7, 8);
    expect(s2.board[8 * SIZE + 7]).toBe(-1);
    expect(s2.toMove).toBe(1);
  });

  it('rejects occupied, out-of-bounds, and post-game moves', () => {
    let s = play([
      { x: 5, y: 7 },
      { x: 0, y: 0 },
      { x: 6, y: 7 },
      { x: 0, y: 1 },
      { x: 7, y: 7 },
      { x: 0, y: 2 },
      { x: 8, y: 7 },
      { x: 0, y: 3 },
      { x: 9, y: 7 },
    ]);
    expect(s.winner).toBe(1);
    expect(() => applyMove(s, 3, 3)).toThrow('already over');

    s = newGame();
    s = applyMove(s, 7, 7);
    expect(() => applyMove(s, 7, 7)).toThrow('occupied');
    expect(() => applyMove(s, -1, 0)).toThrow('out of bounds');
    expect(() => applyMove(s, 15, 0)).toThrow('out of bounds');
    expect(() => applyMove(s, 0, 15)).toThrow('out of bounds');
    expect(isLegal(s, 7, 7)).toBe(false);
    expect(isLegal(s, 0, 0)).toBe(true);
  });

  it('cloneState is independent', () => {
    const s = play([
      { x: 7, y: 7 },
      { x: 7, y: 8 },
    ]);
    const c = cloneState(s);
    c.board[0] = 1;
    c.moves.push({ x: 0, y: 0 });
    expect(s.board[0]).toBe(0);
    expect(s.moves).toHaveLength(2);
  });
});

describe('win detection', () => {
  it('detects a horizontal win', () => {
    const s = play([
      { x: 5, y: 7 },
      { x: 0, y: 0 },
      { x: 6, y: 7 },
      { x: 0, y: 1 },
      { x: 7, y: 7 },
      { x: 0, y: 2 },
      { x: 8, y: 7 },
      { x: 0, y: 3 },
      { x: 9, y: 7 },
    ]);
    expect(s.winner).toBe(1);
    expect(s.winningLine).toHaveLength(5);
    expect(legalMoves(s)).toHaveLength(0);
  });

  it('detects a vertical win', () => {
    const s = play([
      { x: 7, y: 5 },
      { x: 0, y: 0 },
      { x: 7, y: 6 },
      { x: 1, y: 0 },
      { x: 7, y: 7 },
      { x: 2, y: 0 },
      { x: 7, y: 8 },
      { x: 3, y: 0 },
      { x: 7, y: 9 },
    ]);
    expect(s.winner).toBe(1);
    expect(s.winningLine).toHaveLength(5);
  });

  it('detects a diagonal win (down-right)', () => {
    const s = play([
      { x: 5, y: 5 },
      { x: 0, y: 0 },
      { x: 6, y: 6 },
      { x: 0, y: 1 },
      { x: 7, y: 7 },
      { x: 0, y: 2 },
      { x: 8, y: 8 },
      { x: 0, y: 3 },
      { x: 9, y: 9 },
    ]);
    expect(s.winner).toBe(1);
    expect(s.winningLine).toHaveLength(5);
  });

  it('detects a diagonal win (down-left)', () => {
    const s = play([
      { x: 9, y: 5 },
      { x: 0, y: 0 },
      { x: 8, y: 6 },
      { x: 0, y: 1 },
      { x: 7, y: 7 },
      { x: 0, y: 2 },
      { x: 6, y: 8 },
      { x: 0, y: 3 },
      { x: 5, y: 9 },
    ]);
    expect(s.winner).toBe(1);
    expect(s.winningLine).toHaveLength(5);
  });

  it('does not call four in a row a win', () => {
    const s = play([
      { x: 5, y: 7 },
      { x: 0, y: 0 },
      { x: 6, y: 7 },
      { x: 0, y: 1 },
      { x: 7, y: 7 },
      { x: 0, y: 2 },
      { x: 8, y: 7 },
      { x: 0, y: 3 },
    ]);
    expect(s.winner).toBe(0);
    expect(s.winningLine).toBeNull();
    expect(checkWin(s.board, 8, 7)).toBeNull();
    expect(hasFive(s.board, 8, 7)).toBe(false);
  });

  it('counts overline (six in a row) as a win', () => {
    // A six cannot arise through applyMove (the game ends at five), so build
    // the board directly and check the last-placed stone in the middle.
    const board = new Int8Array(SIZE * SIZE);
    for (const x of [5, 6, 7, 8, 10]) board[7 * SIZE + x] = 1;
    board[7 * SIZE + 9] = 1;
    const win = checkWin(board, 9, 7);
    expect(win).not.toBeNull();
    expect(win?.winner).toBe(1);
    expect(win?.line).toHaveLength(6);
    expect(hasFive(board, 9, 7)).toBe(true);
  });

  it('checkWin returns null for an empty point', () => {
    expect(checkWin(new Int8Array(SIZE * SIZE), 7, 7)).toBeNull();
  });
});

describe('stateAtMove', () => {
  const moves: Point[] = [
    { x: 7, y: 7 },
    { x: 7, y: 8 },
    { x: 8, y: 8 },
    { x: 8, y: 7 },
    { x: 9, y: 7 },
  ];

  it('replays the first n moves', () => {
    const s0 = stateAtMove(moves, 0);
    expect(s0.moves).toHaveLength(0);
    expect(s0.toMove).toBe(1);

    const s3 = stateAtMove(moves, 3);
    expect(s3.moves).toEqual(moves.slice(0, 3));
    expect(s3.toMove).toBe(-1);
    expect(s3.board[8 * SIZE + 8]).toBe(1);

    const sAll = stateAtMove(moves, 100); // clamped
    expect(sAll.moves).toEqual(moves);
    expect(sAll.toMove).toBe(-1);
  });

  it('matches sequential applyMove', () => {
    const a = stateAtMove(moves, moves.length);
    const b = play(moves);
    expect(a.board).toEqual(b.board);
    expect(a.moves).toEqual(b.moves);
    expect(a.toMove).toBe(b.toMove);
  });
});

describe('draw', () => {
  it('a full board with no five is a draw', () => {
    // Pattern: black iff (x + 2y) mod 4 in {0, 1}. Along every direction the
    // longest run is 2 (horizontal: BBWW; vertical: alternating; diagonals:
    // BBWW / BWWB), so no five can ever appear — including mid-fill, since
    // removing stones cannot create a five.
    const black: Point[] = [];
    const white: Point[] = [];
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const r = (x + 2 * y) % 4;
        if (r === 0 || r === 1) black.push({ x, y });
        else white.push({ x, y });
      }
    }
    expect(black).toHaveLength(113);
    expect(white).toHaveLength(112);
    const s = stateAtMove(interleave(black, white), 225);
    expect(s.winner).toBe(0);
    expect(isDraw(s)).toBe(true);
    expect(legalMoves(s)).toHaveLength(0);
  });
});

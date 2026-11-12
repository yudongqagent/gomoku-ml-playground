import { describe, expect, it } from 'vitest';
import { SIZE } from '../engine/gomoku';
import { buildInputPlanes } from './onnxModel';

/**
 * Tests for the ONNX input-plane construction only. No model is loaded here
 * (no model exists in CI, and onnxruntime-web is never imported by these
 * tests — `buildInputPlanes` is pure). The plane layout must match the Python
 * export exactly; see src/ai/onnxModel.ts for the contract.
 */

function planeSum(planes: Float32Array, p: number): number {
  let s = 0;
  for (let i = 0; i < SIZE * SIZE; i++) s += planes[p * SIZE * SIZE + i];
  return s;
}

function planeAt(planes: Float32Array, p: number, x: number, y: number): number {
  return planes[p * SIZE * SIZE + y * SIZE + x];
}

describe('buildInputPlanes', () => {
  it('empty board, Black to move: only the side-to-move plane is set', () => {
    const planes = buildInputPlanes(new Int8Array(SIZE * SIZE), 1);
    expect(planeSum(planes, 0)).toBe(0);
    expect(planeSum(planes, 1)).toBe(0);
    expect(planeSum(planes, 2)).toBe(0);
    expect(planeSum(planes, 3)).toBe(SIZE * SIZE);
    for (let p = 4; p < 8; p++) expect(planeSum(planes, p)).toBe(0);
  });

  it('empty board, White to move: side-to-move plane is zeros', () => {
    const planes = buildInputPlanes(new Int8Array(SIZE * SIZE), -1);
    expect(planeSum(planes, 3)).toBe(0);
  });

  it('builds all 8 planes from board + history (White to move)', () => {
    // Moves: (7,7)B, (7,8)W, (8,7)B, (8,8)W, (9,7)B — White to move.
    const history = [
      { x: 7, y: 7 },
      { x: 7, y: 8 },
      { x: 8, y: 7 },
      { x: 8, y: 8 },
      { x: 9, y: 7 },
    ];
    const board = new Int8Array(SIZE * SIZE);
    board[7 * SIZE + 7] = 1;
    board[8 * SIZE + 7] = -1;
    board[7 * SIZE + 8] = 1;
    board[8 * SIZE + 8] = -1;
    board[7 * SIZE + 9] = 1;

    const planes = buildInputPlanes(board, -1, history);

    // Plane 0: current-player (White) stones.
    expect(planeSum(planes, 0)).toBe(2);
    expect(planeAt(planes, 0, 7, 8)).toBe(1);
    expect(planeAt(planes, 0, 8, 8)).toBe(1);
    // Plane 1: opponent (Black) stones.
    expect(planeSum(planes, 1)).toBe(3);
    expect(planeAt(planes, 1, 7, 7)).toBe(1);
    expect(planeAt(planes, 1, 8, 7)).toBe(1);
    expect(planeAt(planes, 1, 9, 7)).toBe(1);
    // Plane 2: last move one-hot (9,7).
    expect(planeSum(planes, 2)).toBe(1);
    expect(planeAt(planes, 2, 9, 7)).toBe(1);
    // Plane 3: White to move -> all zeros.
    expect(planeSum(planes, 3)).toBe(0);
    // Planes 4/5: 2 plies ago (after 3 moves: (7,7)B, (7,8)W, (8,7)B).
    // The player to move then was White (2 is even), so plane 4 holds
    // White's stones then, plane 5 Black's.
    expect(planeSum(planes, 4)).toBe(1);
    expect(planeAt(planes, 4, 7, 8)).toBe(1);
    expect(planeSum(planes, 5)).toBe(2);
    expect(planeAt(planes, 5, 7, 7)).toBe(1);
    expect(planeAt(planes, 5, 8, 7)).toBe(1);
    // Planes 6/7: 4 plies ago (after 1 move: (7,7)B). Player to move then
    // was White; White had no stones, Black had (7,7).
    expect(planeSum(planes, 6)).toBe(0);
    expect(planeSum(planes, 7)).toBe(1);
    expect(planeAt(planes, 7, 7, 7)).toBe(1);
  });

  it('builds planes for Black to move with a 4-move history', () => {
    const history = [
      { x: 7, y: 7 },
      { x: 7, y: 8 },
      { x: 8, y: 7 },
      { x: 8, y: 8 },
    ];
    const board = new Int8Array(SIZE * SIZE);
    board[7 * SIZE + 7] = 1;
    board[8 * SIZE + 7] = -1;
    board[7 * SIZE + 8] = 1;
    board[8 * SIZE + 8] = -1;

    const planes = buildInputPlanes(board, 1, history);

    expect(planeSum(planes, 0)).toBe(2); // Black stones
    expect(planeSum(planes, 1)).toBe(2); // White stones
    expect(planeSum(planes, 2)).toBe(1);
    expect(planeAt(planes, 2, 8, 8)).toBe(1);
    expect(planeSum(planes, 3)).toBe(SIZE * SIZE); // Black to move
    // 2 plies ago (after 2 moves): Black (7,7), White (7,8); Black to move then.
    expect(planeSum(planes, 4)).toBe(1);
    expect(planeAt(planes, 4, 7, 7)).toBe(1);
    expect(planeSum(planes, 5)).toBe(1);
    expect(planeAt(planes, 5, 7, 8)).toBe(1);
    // 4 plies ago: empty board.
    expect(planeSum(planes, 6)).toBe(0);
    expect(planeSum(planes, 7)).toBe(0);
  });

  it('zeroes history planes when history is absent or too short', () => {
    const board = new Int8Array(SIZE * SIZE);
    board[7 * SIZE + 7] = 1;
    board[8 * SIZE + 7] = -1;

    const noHistory = buildInputPlanes(board, 1);
    expect(planeSum(noHistory, 0)).toBe(1);
    expect(planeSum(noHistory, 1)).toBe(1);
    for (let p = 2; p < 8; p++) {
      if (p === 3) continue; // side-to-move plane is set for Black
      expect(planeSum(noHistory, p)).toBe(0);
    }

    const oneMove = buildInputPlanes(board, 1, [{ x: 7, y: 8 }]);
    expect(planeSum(oneMove, 2)).toBe(1); // last move known
    expect(planeSum(oneMove, 4)).toBe(0); // not enough history for 2-plies-ago
    expect(planeSum(oneMove, 5)).toBe(0);
    expect(planeSum(oneMove, 6)).toBe(0);
    expect(planeSum(oneMove, 7)).toBe(0);
  });
});

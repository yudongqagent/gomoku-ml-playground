/**
 * v7 port verification: the TypeScript engine in `v7.ts` is a faithful
 * port of the Python reference (`ml/gomoku_ml/mcts_heuristic.py` +
 * `scoring.py` + `kernel_numpy.py`), and these tests prove it against
 * frozen Python values — not against the port's own logic.
 *
 * - Scorer cross-check: v1 total + attack scores match Python's
 *   `scoring._score_parts_flat` exactly, on every empty cell of 84
 *   positions (frozen by `ml/scripts/gen_v7_ts_fixtures.py`).
 * - Kernel cross-check: `attackBothJs` matches Python's
 *   `kernel_numpy.attack_both` exactly, on every empty cell.
 * - Move agreement: the TS engine's move matches Python v7's move
 *   (HeuristicMCTS, 100 sims, rng=None — deterministic first-best
 *   tie-break on both sides) on all 84 positions. Any disagreement is
 *   a port bug, not noise.
 * - Tactical suites: the old 20-position and hard 29-position suites
 *   from `ml/gomoku_ml/` (read directly — single source of truth),
 *   run through the TS engine's full dispatch. Python v7 scores
 *   20/20 + 29/29 (seed 0); the TS engine must match.
 * - Fast-loop regression positions: 4 pass; game 3 is the known xfail
 *   (forcing move starved by a low prior — see iter-7.md §2.1), marked
 *   `it.fails` exactly as in the Python suite.
 */

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { SIZE, hasFive, opponent, type Player, type Point } from '../engine/gomoku';
import { attackBothJs } from '../kernel/scoring';
import {
  HeuristicMCTS,
  candidateMovesV1,
  isWinningMoveV1,
  scoreAttackV1,
  scoreMoveV1,
  winningMovesV1,
} from './v7';

interface FixturePosition {
  id: string;
  black: Array<[number, number]>;
  white: Array<[number, number]>;
  to_move: 1 | -1;
  v1_total: Record<string, number>;
  v1_attack: Record<string, number>;
  attack_black: Record<string, number>;
  attack_white: Record<string, number>;
  v7_move: [number, number];
}

interface TacticalPosition {
  id: string;
  name: string;
  category: string;
  black: Array<[number, number]>;
  white: Array<[number, number]>;
  to_move: 1 | -1;
  expect: { type: 'win' } | { type: 'moves'; moves: Array<[number, number]> } | { type: 'vcf' };
}

interface RegressionPosition {
  id: string;
  black: Array<[number, number]>;
  white: Array<[number, number]>;
  to_move: 'black' | 'white';
  expected: Array<[number, number]>;
}

function readJson<T>(relPath: string): T {
  const url = new URL(relPath, import.meta.url);
  return JSON.parse(readFileSync(url, 'utf-8')) as T;
}

function boardFromStones(
  black: Array<[number, number]>,
  white: Array<[number, number]>,
): Int8Array {
  const board = new Int8Array(SIZE * SIZE);
  for (const [x, y] of black) board[y * SIZE + x] = 1;
  for (const [x, y] of white) board[y * SIZE + x] = -1;
  return board;
}

/** Deterministic v7 engine (no RNG — first-best tie-break, matching Python rng=None). */
function v7Engine(): HeuristicMCTS {
  return new HeuristicMCTS({ sims: 100, cPuct: 1.5, leafMode: 'gradient' });
}

/* ------------------------------------------------------------------ */
/* VCF checker (faithful port of evaluate.py forces_win /               */
/* starts_forced_win, depth 3) — test-only, mirrors the Python          */
/* tactical harness exactly.                                           */
/* ------------------------------------------------------------------ */

function nearEmptyOf(board: Int8Array, color: Player, radius: number): number[] {
  const near = new Uint8Array(SIZE * SIZE);
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] !== color) continue;
    const x = i % SIZE;
    const y = Math.floor(i / SIZE);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) continue;
        const j = ny * SIZE + nx;
        if (board[j] === 0) near[j] = 1;
      }
    }
  }
  const out: number[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (near[i]) out.push(i);
  }
  return out;
}

function fourCreatingMoves(board: Int8Array, player: Player): number[] {
  const out: number[] = [];
  for (const m of nearEmptyOf(board, player, 2)) {
    board[m] = player;
    const creates = winningMovesV1(board, player).length > 0;
    board[m] = 0;
    if (creates) out.push(m);
  }
  return out;
}

function boardFull(board: Int8Array): boolean {
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 0) return false;
  }
  return true;
}

function forcesWin(board: Int8Array, attacker: Player, movesLeft: number): boolean {
  if (winningMovesV1(board, attacker).length > 0) return true;
  if (movesLeft === 0) return false;
  for (const move of fourCreatingMoves(board, attacker)) {
    board[move] = attacker;
    if (winningMovesV1(board, opponent(attacker)).length > 0) {
      board[move] = 0;
      continue; // Defender wins at once instead of blocking.
    }
    const threats = winningMovesV1(board, attacker);
    if (threats.length >= 2) {
      board[move] = 0;
      return true; // Open four: unblockable.
    }
    if (threats.length === 1) {
      board[threats[0]] = opponent(attacker);
      const over =
        hasFive(board, threats[0] % SIZE, Math.floor(threats[0] / SIZE)) || boardFull(board);
      let ok = false;
      if (!over) ok = forcesWin(board, attacker, movesLeft - 1);
      board[threats[0]] = 0;
      board[move] = 0;
      if (ok) return true;
    } else {
      board[move] = 0;
    }
  }
  return false;
}

function startsForcedWin(board: Int8Array, player: Player, move: number, depth = 3): boolean {
  if (board[move] !== 0) return false;
  if (isWinningMoveV1(board, move, player)) return true;
  board[move] = player;
  if (winningMovesV1(board, opponent(player)).length > 0) {
    board[move] = 0;
    return false;
  }
  const threats = winningMovesV1(board, player);
  if (threats.length >= 2) {
    board[move] = 0;
    return true;
  }
  if (threats.length === 1) {
    board[threats[0]] = opponent(player);
    const over =
      hasFive(board, threats[0] % SIZE, Math.floor(threats[0] / SIZE)) || boardFull(board);
    let ok = false;
    if (!over) ok = forcesWin(board, player, depth);
    board[threats[0]] = 0;
    board[move] = 0;
    return ok;
  }
  board[move] = 0;
  return false;
}

function checkTactical(
  board: Int8Array,
  toMove: Player,
  move: Point,
  expect: TacticalPosition['expect'],
): boolean {
  const idx = move.y * SIZE + move.x;
  if (expect.type === 'win') return isWinningMoveV1(board, idx, toMove);
  if (expect.type === 'moves') {
    return expect.moves.some(([x, y]) => x === move.x && y === move.y);
  }
  return startsForcedWin(board, toMove, idx);
}

/* ------------------------------------------------------------------ */
/* Fixtures                                                            */
/* ------------------------------------------------------------------ */

const reference = readJson<{ positions: FixturePosition[] }>(
  './__fixtures__/v7-reference.json',
);
const oldSuite = readJson<{ positions: TacticalPosition[] }>(
  '../../ml/gomoku_ml/tactical_positions.json',
);
const hardSuite = readJson<{ positions: TacticalPosition[] }>(
  '../../ml/gomoku_ml/tactical_positions_hard.json',
);
const regression = readJson<{ positions: RegressionPosition[] }>(
  '../../ml/gomoku_ml/regression_positions_v7.json',
);

describe('v7 scorer cross-check vs frozen Python values', () => {
  it('v1 total scores match Python exactly on all 84 positions', () => {
    let cells = 0;
    let mismatches = 0;
    const first: string[] = [];
    for (const pos of reference.positions) {
      const board = boardFromStones(pos.black, pos.white);
      for (const [key, expected] of Object.entries(pos.v1_total)) {
        const m = Number(key);
        const got = scoreMoveV1(board, m, pos.to_move);
        cells++;
        if (got !== expected) {
          mismatches++;
          if (first.length < 5) {
            first.push(`${pos.id} cell ${m}: TS ${got} vs Python ${expected}`);
          }
        }
      }
    }
    expect(mismatches, `v1 total mismatches (${cells} cells): ${first.join('; ')}`).toBe(0);
    expect(cells).toBeGreaterThan(10000);
  });

  it('v1 attack scores match Python exactly on all 84 positions', () => {
    let cells = 0;
    let mismatches = 0;
    for (const pos of reference.positions) {
      const board = boardFromStones(pos.black, pos.white);
      for (const [key, expected] of Object.entries(pos.v1_attack)) {
        const m = Number(key);
        if (scoreAttackV1(board, m, pos.to_move) !== expected) mismatches++;
        cells++;
      }
    }
    expect(mismatches, `v1 attack mismatches over ${cells} cells`).toBe(0);
  });

  it('kernel attack-only scores match Python attack_both exactly', () => {
    let cells = 0;
    let mismatches = 0;
    for (const pos of reference.positions) {
      const board = boardFromStones(pos.black, pos.white);
      const [black, white] = attackBothJs(board);
      for (const [key, expected] of Object.entries(pos.attack_black)) {
        if (black[Number(key)] !== expected) mismatches++;
        cells++;
      }
      for (const [key, expected] of Object.entries(pos.attack_white)) {
        if (white[Number(key)] !== expected) mismatches++;
        cells++;
      }
    }
    expect(mismatches, `attack_both mismatches over ${cells} cells`).toBe(0);
  });
});

describe('v7 move agreement vs Python v7', () => {
  it(
    'TS v7 chooses the same move as Python v7 on all 84 positions',
    () => {
      const disagreements: string[] = [];
      for (const pos of reference.positions) {
        const board = boardFromStones(pos.black, pos.white);
        const r = v7Engine().chooseMove(board, pos.to_move);
        if (r.move.x !== pos.v7_move[0] || r.move.y !== pos.v7_move[1]) {
          disagreements.push(
            `${pos.id}: TS (${r.move.x},${r.move.y}) vs Python (${pos.v7_move[0]},${pos.v7_move[1]})`,
          );
        }
      }
      expect(
        disagreements,
        `move disagreements: ${disagreements.join('; ')}`,
      ).toEqual([]);
    },
    120_000,
  );
});

describe('v7 tactical suites (Python JSONs, single source of truth)', () => {
  function runSuite(suite: { positions: TacticalPosition[] }, label: string): void {
    it(
      `${label}: ${suite.positions.length}/${suite.positions.length}`,
      () => {
        const failures: string[] = [];
        for (const pos of suite.positions) {
          const board = boardFromStones(pos.black, pos.white);
          const r = v7Engine().chooseMove(board, pos.to_move);
          if (!checkTactical(board, pos.to_move, r.move, pos.expect)) {
            failures.push(
              `${pos.id} (${pos.category}): chose (${r.move.x},${r.move.y}), expected ${JSON.stringify(pos.expect)}`,
            );
          }
        }
        expect(failures, `tactical failures: ${failures.join('; ')}`).toEqual([]);
      },
      120_000,
    );
  }

  runSuite(oldSuite, 'old suite');
  runSuite(hardSuite, 'hard suite');
});

describe('v7 fast-loop regression positions', () => {
  for (const pos of regression.positions) {
    const toMove: Player = pos.to_move === 'black' ? 1 : -1;
    const testFn = pos.id === 'fl1-g3-ply23' ? it.fails : it;
    testFn(
      `${pos.id}: plays an expected move${pos.id === 'fl1-g3-ply23' ? ' (known xfail: forcing move starved by a low prior)' : ''}`,
      () => {
        const board = boardFromStones(pos.black, pos.white);
        const r = v7Engine().chooseMove(board, toMove);
        const ok = pos.expected.some(([x, y]) => x === r.move.x && y === r.move.y);
        expect(ok).toBe(true);
      },
      30_000,
    );
  }
});

describe('v7 basic behaviour and timing', () => {
  it('plays the centre on an empty board', () => {
    const r = v7Engine().chooseMove(new Int8Array(SIZE * SIZE), 1);
    expect(r.move).toEqual({ x: 7, y: 7 });
  });

  it('takes an immediate win', () => {
    const board = boardFromStones(
      [
        [7, 7],
        [8, 7],
        [9, 7],
        [10, 7],
      ],
      [[3, 3]],
    );
    const r = v7Engine().chooseMove(board, 1);
    expect(r.winRate).toBe(1);
    expect(isWinningMoveV1(board, r.move.y * SIZE + r.move.x, 1)).toBe(true);
  });

  it('blocks the opponent\'s single immediate win', () => {
    // White's four (7,7)-(10,7) has only one open end: Black already
    // occupies (11,7), so (6,7) is White's ONLY winning square.
    const board = boardFromStones(
      [
        [11, 7],
        [3, 3],
      ],
      [
        [7, 7],
        [8, 7],
        [9, 7],
        [10, 7],
      ],
    );
    const r = v7Engine().chooseMove(board, 1);
    expect([r.move.x, r.move.y]).toEqual([6, 7]);
  });

  it('candidate moves match the Python rule (Chebyshev 2 of any stone)', () => {
    const board = boardFromStones([[7, 7]], []);
    const cands = candidateMovesV1(board);
    expect(cands.length).toBe(24); // 5x5 box minus the stone itself
    expect(cands).toContain(7 * SIZE + 7 - SIZE - 1);
  });

  it(
    'move times stay reasonable (mean < 2s, max < 5s in Node)',
    () => {
      const times: number[] = [];
      for (const pos of reference.positions.slice(0, 12)) {
        const board = boardFromStones(pos.black, pos.white);
        const r = v7Engine().chooseMove(board, pos.to_move);
        times.push(r.ms);
      }
      const mean = times.reduce((a, b) => a + b, 0) / times.length;
      const max = Math.max(...times);
      expect(mean).toBeLessThan(2000);
      expect(max).toBeLessThan(5000);
    },
    120_000,
  );
});

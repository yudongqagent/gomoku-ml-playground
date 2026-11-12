/**
 * Core Gomoku rules engine — pure, dependency-free, immutable game state.
 *
 * Rules implemented (standard 15x15 Gomoku):
 * - Black (1) moves first, players alternate placing one stone on an empty point.
 * - Five-or-more in a row wins. OVERLINE COUNTS AS A WIN: six or more stones
 *   in a row is a win, not a loss. (This is the common casual/free-style rule;
 *   Renju treats overline as a loss for Black, which we do not implement.)
 * - No captures, no Renju forbidden moves (no 3-3 / 4-4 / long-connection
 *   restrictions), no swap2 / swap opening rule.
 *
 * Future options (not implemented): Renju forbidden moves, swap2, timed games.
 *
 * State is immutable: `applyMove` returns a NEW GameState and never mutates
 * its argument. The search code in src/ai works directly on raw Int8Array
 * boards with make/unmake for speed; this module is the source of truth for
 * the rules and the UI-facing state shape.
 */

export const SIZE = 15;

export type Player = 1 | -1;
export const BLACK: Player = 1;
export const WHITE: Player = -1;

export interface Point {
  x: number;
  y: number;
}

export interface GameState {
  /** Row-major board, length SIZE*SIZE (225). Values: 1 = Black, -1 = White, 0 = empty. */
  board: Int8Array;
  toMove: Player;
  /** Moves played so far, in order. */
  moves: Point[];
  /** 1 = Black won, -1 = White won, 0 = no winner yet (or draw if board full). */
  winner: Player | 0;
  /** The winning line (5+ points), or null if nobody has won. */
  winningLine: Point[] | null;
}

export function opponent(p: Player): Player {
  return p === 1 ? -1 : 1;
}

export function inBounds(x: number, y: number): boolean {
  return x >= 0 && x < SIZE && y >= 0 && y < SIZE;
}

export function idx(x: number, y: number): number {
  return y * SIZE + x;
}

export function newGame(): GameState {
  return {
    board: new Int8Array(SIZE * SIZE),
    toMove: 1,
    moves: [],
    winner: 0,
    winningLine: null,
  };
}

export function cloneState(s: GameState): GameState {
  return {
    board: new Int8Array(s.board),
    toMove: s.toMove,
    moves: s.moves.slice(),
    winner: s.winner,
    winningLine: s.winningLine ? s.winningLine.slice() : null,
  };
}

export function isLegal(s: GameState, x: number, y: number): boolean {
  return s.winner === 0 && inBounds(x, y) && s.board[y * SIZE + x] === 0;
}

/**
 * Return a NEW state with the move applied. Throws on an illegal move
 * (out of bounds, occupied point, or game already over).
 */
export function applyMove(s: GameState, x: number, y: number): GameState {
  if (s.winner !== 0) throw new Error('Game is already over');
  if (!inBounds(x, y)) throw new Error(`Move (${x}, ${y}) is out of bounds`);
  if (s.board[y * SIZE + x] !== 0) throw new Error(`Move (${x}, ${y}) is occupied`);
  const board = new Int8Array(s.board);
  board[y * SIZE + x] = s.toMove;
  const win = checkWin(board, x, y);
  return {
    board,
    toMove: opponent(s.toMove),
    moves: [...s.moves, { x, y }],
    winner: win ? win.winner : 0,
    winningLine: win ? win.line : null,
  };
}

/** All empty points (none if the game is over). */
export function legalMoves(s: GameState): Point[] {
  if (s.winner !== 0) return [];
  const out: Point[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (s.board[i] === 0) out.push({ x: i % SIZE, y: Math.floor(i / SIZE) });
  }
  return out;
}

/** True when the board is full and nobody won. */
export function isDraw(s: GameState): boolean {
  return s.winner === 0 && s.moves.length === SIZE * SIZE;
}

/**
 * Replay the first `n` moves of a move list. `n` is clamped to [0, moves.length].
 * Throws if any replayed move is illegal (e.g. a move after the game ended).
 */
export function stateAtMove(moves: Point[], n: number): GameState {
  let s = newGame();
  const count = Math.max(0, Math.min(n, moves.length));
  for (let i = 0; i < count; i++) {
    s = applyMove(s, moves[i].x, moves[i].y);
  }
  return s;
}

const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [0, 1],
  [1, 1],
  [1, -1],
];

/**
 * Fast win check for the stone just placed at (x, y): true if that stone is
 * part of a line of 5+ of its own colour. Allocation-free; used in search
 * hot paths. Returns false if the point is empty.
 */
export function hasFive(board: Int8Array, x: number, y: number): boolean {
  const color = board[y * SIZE + x];
  if (color === 0) return false;
  for (const [dx, dy] of DIRECTIONS) {
    let count = 1;
    for (let i = 1; i < SIZE; i++) {
      const nx = x + i * dx;
      const ny = y + i * dy;
      if (!inBounds(nx, ny) || board[ny * SIZE + nx] !== color) break;
      count++;
    }
    for (let i = 1; i < SIZE; i++) {
      const nx = x - i * dx;
      const ny = y - i * dy;
      if (!inBounds(nx, ny) || board[ny * SIZE + nx] !== color) break;
      count++;
    }
    if (count >= 5) return true;
  }
  return false;
}

/**
 * Win detection for the stone just placed at (x, y). Returns the winner and
 * the full contiguous winning line (5+ points; overline lines include all
 * stones), or null if the move did not win. Returns null if the point is empty.
 */
export function checkWin(
  board: Int8Array,
  x: number,
  y: number,
): { winner: Player; line: Point[] } | null {
  const color = board[y * SIZE + x];
  if (color === 0) return null;
  for (const [dx, dy] of DIRECTIONS) {
    const line: Point[] = [{ x, y }];
    for (let i = 1; i < SIZE; i++) {
      const nx = x + i * dx;
      const ny = y + i * dy;
      if (!inBounds(nx, ny) || board[ny * SIZE + nx] !== color) break;
      line.push({ x: nx, y: ny });
    }
    for (let i = 1; i < SIZE; i++) {
      const nx = x - i * dx;
      const ny = y - i * dy;
      if (!inBounds(nx, ny) || board[ny * SIZE + nx] !== color) break;
      line.unshift({ x: nx, y: ny });
    }
    if (line.length >= 5) return { winner: color as Player, line };
  }
  return null;
}

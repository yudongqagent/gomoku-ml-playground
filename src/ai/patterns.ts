/**
 * Shared pattern evaluation for all AI levels.
 *
 * Evaluation is window-based: the board is decomposed into every 5-cell
 * window (572 in total: rows, columns, both diagonals), each scored from the
 * side-to-move's perspective. A window containing both colours scores 0;
 * otherwise it scores by stone count and openness (whether the cells just
 * beyond each end are empty — board edges count as closed).
 *
 * Scores are deliberately separated by orders of magnitude so that pattern
 * classes dominate each other (a four always beats any number of threes):
 *
 *   five        10,000,000   open four    200,000   four    20,000
 *   open three       4,000   three            400   open two    80
 *   two                 10   one                1-2
 *
 * Candidate generation is bounded: only empty points within Chebyshev
 * distance 2 of an existing stone are considered (the centre on an empty
 * board). Search never scans all 225 empties with a full-board eval in its
 * inner loop; move ordering scores candidates locally (the ~20 windows
 * containing the point).
 *
 * All functions here take `ArrayLike<number>` boards so they work with both
 * Int8Array (search internals) and number[] (requests).
 */

import { SIZE, hasFive, type Player, type Point } from '../engine/gomoku';

/** Score of a completed five. Search treats any score >= WIN_SCORE - 1000 as a forced win. */
export const WIN_SCORE = 10_000_000;

/**
 * Logistic scale for converting a pattern-eval score into a win rate:
 * winRate = 1 / (1 + exp(-score / EVAL_SCALE)).
 * With EVAL_SCALE = 3000, a single open three (+4000) maps to ~0.79, a four
 * (+20000) to ~0.999, and a small edge (+400) to ~0.53. This is a rough,
 * UNCALIBRATED heuristic squash — it is not a measured probability. It is
 * only used for search/heuristic levels; MCTS levels derive win rates from
 * visit counts / mean values instead.
 */
export const EVAL_SCALE = 3000;

export function winRateFromEval(score: number): number {
  if (score >= WIN_SCORE - 1000) return 1;
  if (score <= -(WIN_SCORE - 1000)) return 0;
  return 1 / (1 + Math.exp(-score / EVAL_SCALE));
}

interface BoardWindow {
  cells: [number, number, number, number, number];
  /** Cell index just before the window along its line, or -1 if off-board. */
  before: number;
  /** Cell index just after the window along its line, or -1 if off-board. */
  after: number;
}

function buildWindows(): BoardWindow[] {
  const wins: BoardWindow[] = [];
  const dirs: ReadonlyArray<readonly [number, number]> = [
    [1, 0],
    [0, 1],
    [1, 1],
    [1, -1],
  ];
  for (const [dx, dy] of dirs) {
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const ex = x + 4 * dx;
        const ey = y + 4 * dy;
        if (ex < 0 || ex >= SIZE || ey < 0 || ey >= SIZE) continue;
        const cells = [0, 1, 2, 3, 4].map((i) => (y + i * dy) * SIZE + (x + i * dx)) as [
          number,
          number,
          number,
          number,
          number,
        ];
        const bx = x - dx;
        const by = y - dy;
        const ax = ex + dx;
        const ay = ey + dy;
        const before =
          bx >= 0 && bx < SIZE && by >= 0 && by < SIZE ? by * SIZE + bx : -1;
        const after = ax >= 0 && ax < SIZE && ay >= 0 && ay < SIZE ? ay * SIZE + ax : -1;
        wins.push({ cells, before, after });
      }
    }
  }
  return wins;
}

/** All 572 five-cell windows on the board. */
const WINDOWS: BoardWindow[] = buildWindows();

/** For each cell index, the windows containing that cell (at most 20). */
const WINDOWS_BY_CELL: number[][] = (() => {
  const byCell: number[][] = Array.from({ length: SIZE * SIZE }, () => []);
  WINDOWS.forEach((w, wi) => {
    for (const c of w.cells) byCell[c].push(wi);
  });
  return byCell;
})();

function windowScore(count: number, openEnds: number): number {
  if (count >= 5) return WIN_SCORE;
  if (count === 4) return openEnds === 2 ? 200_000 : openEnds === 1 ? 20_000 : 0;
  if (count === 3) return openEnds === 2 ? 4_000 : openEnds === 1 ? 400 : 0;
  if (count === 2) return openEnds === 2 ? 80 : openEnds === 1 ? 10 : 0;
  if (count === 1) return openEnds === 2 ? 2 : 1;
  return 0;
}

function openEndsOf(board: ArrayLike<number>, w: BoardWindow): number {
  let open = 0;
  if (w.before !== -1 && board[w.before] === 0) open++;
  if (w.after !== -1 && board[w.after] === 0) open++;
  return open;
}

/**
 * Full-board pattern evaluation from `player`'s perspective.
 * Positive = good for `player`. Cost: one pass over 572 windows.
 */
export function evaluateBoard(board: ArrayLike<number>, player: Player): number {
  let score = 0;
  for (const w of WINDOWS) {
    let own = 0;
    let opp = 0;
    for (const c of w.cells) {
      const v = board[c];
      if (v === player) own++;
      else if (v !== 0) opp++;
    }
    if ((own > 0 && opp > 0) || (own === 0 && opp === 0)) continue;
    const open = openEndsOf(board, w);
    if (own > 0) score += windowScore(own, open);
    else score -= windowScore(opp, open);
  }
  return score;
}

/**
 * Score of the windows containing cell `idx` if `color` were played there.
 * Local: only the ~20 windows through the point are evaluated.
 */
function scoreWindowsAt(board: ArrayLike<number>, idx: number, color: Player): number {
  let s = 0;
  for (const wi of WINDOWS_BY_CELL[idx]) {
    const w = WINDOWS[wi];
    let own = 0;
    let opp = 0;
    for (const c of w.cells) {
      const v = c === idx ? color : board[c];
      if (v === color) own++;
      else if (v !== 0) opp++;
    }
    if (own === 0 || opp > 0) continue;
    s += windowScore(own, openEndsOf(board, w));
  }
  return s;
}

/**
 * Heuristic value of playing at (x, y) for `player`: attack value (what the
 * move builds for us) + defence value (what it denies the opponent by
 * occupying the point). Used for move ordering, the Beginner level, and MCTS
 * priors. The point must be empty.
 */
export function scoreMove(board: ArrayLike<number>, x: number, y: number, player: Player): number {
  const idx = y * SIZE + x;
  const attack = scoreWindowsAt(board, idx, player);
  const defence = scoreWindowsAt(board, idx, player === 1 ? -1 : 1);
  return attack + defence;
}

/**
 * Candidate moves: empty points within Chebyshev distance `radius` of any
 * stone. On an empty board, returns just the centre. A winning move is always
 * adjacent to a stone, so radius 1 suffices for win detection; radius 2 is
 * the default for general move generation.
 */
export function candidateMoves(board: ArrayLike<number>, radius = 2): Point[] {
  const seen = new Uint8Array(SIZE * SIZE);
  let anyStone = false;
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 0) continue;
    anyStone = true;
    const x = i % SIZE;
    const y = Math.floor(i / SIZE);
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx < 0 || nx >= SIZE || ny < 0 || ny >= SIZE) continue;
        const j = ny * SIZE + nx;
        if (board[j] === 0) seen[j] = 1;
      }
    }
  }
  if (!anyStone) return [{ x: 7, y: 7 }];
  const out: Point[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (seen[i]) out.push({ x: i % SIZE, y: Math.floor(i / SIZE) });
  }
  return out;
}

/**
 * Candidates scored by `scoreMove` and sorted best-first (ties broken by
 * board index for determinism), truncated to `limit`.
 *
 * When there are many candidates, a cheap pre-filter keeps only the
 * `prefilterLimit` candidates with the most adjacent stones before the
 * (more expensive) pattern scoring runs. This bounds per-node cost in search.
 * A move with no adjacent stones is almost never the best move in Gomoku, so
 * the pre-filter rarely drops anything important — but it is a heuristic
 * trade-off, documented here.
 */
export function orderedCandidates(
  board: ArrayLike<number>,
  player: Player,
  limit: number,
  prefilterLimit = 24,
): Point[] {
  let cands = candidateMoves(board, 2);
  if (cands.length > prefilterLimit) {
    const scored = cands.map((c) => {
      let neighbours = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = c.x + dx;
          const ny = c.y + dy;
          if (nx >= 0 && nx < SIZE && ny >= 0 && ny < SIZE && board[ny * SIZE + nx] !== 0) {
            neighbours++;
          }
        }
      }
      return { c, neighbours };
    });
    scored.sort((a, b) => b.neighbours - a.neighbours || a.c.y * SIZE + a.c.x - (b.c.y * SIZE + b.c.x));
    cands = scored.slice(0, prefilterLimit).map((s) => s.c);
  }
  const scored = cands.map((c) => ({ c, s: scoreMove(board, c.x, c.y, player) }));
  scored.sort((a, b) => b.s - a.s || a.c.y * SIZE + a.c.x - (b.c.y * SIZE + b.c.x));
  return scored.slice(0, limit).map((s) => s.c);
}

/**
 * All moves that win immediately for `player` (usually 0 or 1; 2+ means the
 * opponent faces a double threat). Works on a copy, so the input board is
 * never mutated. Only radius-1 candidates can win, so this is cheap.
 */
export function findWinningMoves(board: ArrayLike<number>, player: Player): Point[] {
  const b = Int8Array.from(board as ArrayLike<number>);
  const out: Point[] = [];
  for (const c of candidateMoves(b, 1)) {
    const i = c.y * SIZE + c.x;
    b[i] = player;
    if (hasFive(b, c.x, c.y)) out.push(c);
    b[i] = 0;
  }
  return out;
}

/** First immediate winning move for `player`, or null. */
export function findWinningMove(board: ArrayLike<number>, player: Player): Point | null {
  const wins = findWinningMoves(board, player);
  return wins.length > 0 ? wins[0] : null;
}

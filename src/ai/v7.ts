/**
 * v7 — the research engine that passed all 4 clauses of the "better" bar
 * (batch 3, H1) and is now the Master level's engine.
 *
 * Faithful TypeScript port of:
 * - `ml/gomoku_ml/mcts_heuristic.py` — HeuristicMCTS with
 *   leaf_mode="gradient" (PUCT MCTS, 100 sims, c_puct 1.5, v1 priors,
 *   terminal-aware leaf + capped attack-gradient leaf in quiet positions).
 * - `ml/gomoku_ml/scoring.py` — the v1 5-window scorer used for priors
 *   (max-per-direction, double-threat bonuses, full-value defence). This
 *   is NOT `patterns.ts`'s scoreMove: that one sums all windows through
 *   the point (double-counting a lone four), scores a fully-closed
 *   four-window as 0, and has no double-threat bonuses. The v1 scorer is
 *   ported exactly, and cross-checked cell-by-cell against frozen Python
 *   values in `v7.test.ts`.
 * - `ml/gomoku_ml/kernel_numpy.py` attack_both — the kernel attack-only
 *   scores for the leaf gradient (via `src/kernel/scoring.ts`
 *   `attackBothJs`), also cross-checked against frozen Python values.
 *
 * Fidelity notes (do not "fix" these without re-validating the whole
 * engine against the Python reference — v7's measured strength, ladder
 * Elo 190.8 and the 0.7050 head-to-head vs v1, comes from the code as
 * written):
 * - The terminal win check in `_runOne` NEVER fires: it calls
 *   `isWinningMove`, which returns false on an occupied cell, and the
 *   parent already placed the stone before recursing. Verified against
 *   the Python reference (27 wins played inside a 100-sim tree, 0
 *   terminal-branch hits). Wins inside the tree are therefore handled
 *   by the leaf evaluation, not by an early -1 — exactly as in Python.
 *   The correct check would be `isWinningStone`; changing it would make
 *   a different engine, and is a research question (batch 4), not a port
 *   fix.
 * - The gradient is blind to broken fours: the kernel ports v0's
 *   contiguous-run scoring, not v1's 5-window scoring (see iter-7.md §1).
 * - Root choice is most-visited; ties break by the injected RNG, or to
 *   the first move when no RNG is given (Python's rng=None behaviour).
 *   The page passes Math.random; tests use the deterministic default.
 */

import { SIZE, hasFive, opponent, type Player, type Point } from '../engine/gomoku';
import { attackBothJs } from '../kernel/scoring';
import type { TopMove } from './types';

export const V7_SIMS = 100;
export const V7_C_PUCT = 1.5;
export const V7_LEAF_MODE = 'gradient' as const;

/* ------------------------------------------------------------------ */
/* v1 scorer (faithful port of ml/gomoku_ml/scoring.py)                */
/* ------------------------------------------------------------------ */

const WIN_SCORE_V1 = 10_000_000;
const OPEN_FOUR = 200_000;
const FOUR = 20_000;
const OPEN_THREE = 4_000;
const THREE = 400;
const OPEN_TWO = 80;
const TWO = 10;
const ONE_OPEN = 2;
const ONE = 1;

const DOUBLE_FOUR_BONUS = 200_000;
const DOUBLE_OPEN_THREE_BONUS = 50_000;

const KIND_NONE = 0;
const KIND_FOUR = 1;
const KIND_OPEN_THREE = 2;
const KIND_FIVE = 3;

// Directions as (dy, dx), in Python's _DIRECTIONS order.
const DIRS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 0],
  [1, 1],
  [1, -1],
];

interface V1Window {
  cells: [number, number, number, number, number];
  before: number;
  after: number;
  dir: number;
}

function buildWindows(): V1Window[] {
  const wins: V1Window[] = [];
  for (let di = 0; di < DIRS.length; di++) {
    const [dy, dx] = DIRS[di];
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const ey = y + 4 * dy;
        const ex = x + 4 * dx;
        if (ey < 0 || ey >= SIZE || ex < 0 || ex >= SIZE) continue;
        const cells = [0, 1, 2, 3, 4].map((i) => (y + i * dy) * SIZE + (x + i * dx)) as [
          number,
          number,
          number,
          number,
          number,
        ];
        const by = y - dy;
        const bx = x - dx;
        const ay = ey + dy;
        const ax = ex + dx;
        const before = by >= 0 && by < SIZE && bx >= 0 && bx < SIZE ? by * SIZE + bx : -1;
        const after = ay >= 0 && ay < SIZE && ax >= 0 && ax < SIZE ? ay * SIZE + ax : -1;
        wins.push({ cells, before, after, dir: di });
      }
    }
  }
  return wins;
}

const WINDOWS: V1Window[] = buildWindows();

/** For each cell, the windows containing it, as [window index, position in window]. */
const WINDOWS_BY_CELL: Array<Array<[number, number]>> = (() => {
  const byCell: Array<Array<[number, number]>> = Array.from({ length: SIZE * SIZE }, () => []);
  WINDOWS.forEach((w, wi) => {
    w.cells.forEach((c, i) => byCell[c].push([wi, i]));
  });
  return byCell;
})();

function scoreFromCounts(
  own: number,
  opp: number,
  emptyIdx: number,
  openEnds: number,
  board: Int8Array,
  before: number,
  after: number,
): [number, number] {
  if (opp > 0) return [0, KIND_NONE];
  if (own === 5) return [WIN_SCORE_V1, KIND_FIVE];
  if (own === 4) {
    // The empty cell inside the window is always a winning square, so
    // this is a four whatever the outside looks like. It is an OPEN
    // four (two winning squares) only when the four is contiguous and
    // the outside cell beyond its far end is empty.
    if (emptyIdx === 0) {
      if (after !== -1 && board[after] === 0) return [OPEN_FOUR, KIND_FOUR];
    } else if (emptyIdx === 4) {
      if (before !== -1 && board[before] === 0) return [OPEN_FOUR, KIND_FOUR];
    }
    return [FOUR, KIND_FOUR];
  }
  if (own === 3) {
    if (openEnds === 2) return [OPEN_THREE, KIND_OPEN_THREE];
    if (openEnds === 1) return [THREE, KIND_NONE];
    return [0, KIND_NONE];
  }
  if (own === 2) {
    if (openEnds === 2) return [OPEN_TWO, KIND_NONE];
    if (openEnds === 1) return [TWO, KIND_NONE];
    return [0, KIND_NONE];
  }
  if (own === 1) return [openEnds === 2 ? ONE_OPEN : ONE, KIND_NONE];
  return [0, KIND_NONE];
}

/**
 * Core v1 scorer (port of `scoring._score_parts_flat`): returns
 * `[attack, defence]` for `player` playing `move` on `board`. Each
 * direction contributes only its strongest window; a move creating
 * fours in >= 2 directions or open threes in >= 2 directions gets an
 * explicit bonus. Defence is discounted per direction by
 * `defenceWeight` (v1 = 1.0, full-value defence).
 */
export function scorePartsV1(
  board: Int8Array,
  move: number,
  player: Player,
  defenceWeight = 1.0,
): [number, number] {
  const opp = opponent(player);
  const bestA = [0, 0, 0, 0];
  const kindsA = [KIND_NONE, KIND_NONE, KIND_NONE, KIND_NONE];
  const bestD = [0, 0, 0, 0];
  for (const [wi, moveIdx] of WINDOWS_BY_CELL[move]) {
    const w = WINDOWS[wi];
    let p = 0;
    let o = 0;
    let emptySum = 0;
    // The move cell is empty on the board, so it counts as empty here
    // and contributes to neither side — no per-cell check needed.
    for (let i = 0; i < 5; i++) {
      const v = board[w.cells[i]];
      if (v === player) p++;
      else if (v === opp) o++;
      else emptySum += i;
    }
    if (p > 0 && o > 0) continue; // Blocked for both sides.
    let openEnds = 0;
    if (w.before !== -1 && board[w.before] === 0) openEnds++;
    if (w.after !== -1 && board[w.after] === 0) openEnds++;
    if (o === 0) {
      const [s, kind] = scoreFromCounts(
        p + 1,
        0,
        emptySum - moveIdx,
        openEnds,
        board,
        w.before,
        w.after,
      );
      if (s > bestA[w.dir]) {
        bestA[w.dir] = s;
        kindsA[w.dir] = kind;
      }
    }
    if (p === 0) {
      const [s] = scoreFromCounts(
        o + 1,
        0,
        emptySum - moveIdx,
        openEnds,
        board,
        w.before,
        w.after,
      );
      if (s > bestD[w.dir]) bestD[w.dir] = s;
    }
  }
  let attack = bestA[0] + bestA[1] + bestA[2] + bestA[3];
  const fours = kindsA.filter((k) => k === KIND_FOUR || k === KIND_FIVE).length;
  const openThrees = kindsA.filter((k) => k === KIND_OPEN_THREE).length;
  if (fours >= 2) attack += DOUBLE_FOUR_BONUS;
  if (openThrees >= 2) attack += DOUBLE_OPEN_THREE_BONUS;
  // Per-direction truncation, mirroring Python's int(defence_weight * d).
  const defence = bestD.reduce((sum, d) => sum + Math.trunc(defenceWeight * d), 0);
  return [attack, defence];
}

/** Total v1 score (attack + full-value defence) for `player` playing `move`. */
export function scoreMoveV1(board: Int8Array, move: number, player: Player): number {
  const [attack, defence] = scorePartsV1(board, move, player, 1.0);
  return attack + defence;
}

/** Attack-only v1 score (what the move builds, including double-threat bonuses). */
export function scoreAttackV1(board: Int8Array, move: number, player: Player): number {
  return scorePartsV1(board, move, player, 1.0)[0];
}

/* ------------------------------------------------------------------ */
/* Candidates and winning moves (port of scoring._candidate_moves /     */
/* scoring._winning_moves and game.is_winning_move)                    */
/* ------------------------------------------------------------------ */

/** Empty points within Chebyshev distance `radius` of any stone of `color` (null = any stone). Sorted ascending. */
function nearEmpty(board: Int8Array, color: Player | null, radius: number): number[] {
  const near = new Uint8Array(SIZE * SIZE);
  let anyStone = false;
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 0) continue;
    if (color !== null && board[i] !== color) continue;
    anyStone = true;
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
  if (!anyStone) return [];
  const out: number[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (near[i]) out.push(i);
  }
  return out;
}

/**
 * Moves worth scoring: empty points within Chebyshev distance 2 of any
 * stone (the centre on an empty board). Falls back to all legal moves if
 * the candidate set is empty (near-full board).
 */
export function candidateMovesV1(board: Int8Array): number[] {
  let stones = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] !== 0) stones++;
  }
  if (stones === 0) return [7 * SIZE + 7];
  const cands = nearEmpty(board, null, 2);
  if (cands.length > 0) return cands;
  const legal: number[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 0) legal.push(i);
  }
  return legal;
}

/**
 * True if placing `player`'s stone at `move` (flat index) creates >= 5
 * in a row. Returns false when the cell is occupied — exactly like
 * Python's `game.is_winning_move`, including the consequence noted at
 * the top of this file (the terminal check in `_runOne` never fires).
 */
export function isWinningMoveV1(board: Int8Array, move: number, player: Player): boolean {
  if (board[move] !== 0) return false;
  board[move] = player;
  const win = hasFive(board, move % SIZE, Math.floor(move / SIZE));
  board[move] = 0;
  return win;
}

/** All immediate winning moves for `player`, ascending. A winning move must be adjacent to a friendly stone. */
export function winningMovesV1(board: Int8Array, player: Player): number[] {
  const out: number[] = [];
  for (const m of nearEmpty(board, player, 1)) {
    if (isWinningMoveV1(board, m, player)) out.push(m);
  }
  return out;
}

function isFull(board: Int8Array): boolean {
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 0) return false;
  }
  return true;
}

/* ------------------------------------------------------------------ */
/* v7 leaf values (port of mcts_heuristic)                             */
/* ------------------------------------------------------------------ */

const PRIOR_TEMPERATURE = 2.0;
const MUST_BLOCK_VALUE = -0.25;

const GRADIENT_STRONG = 1000;
const GRADIENT_DEVELOPING = 100;
const GRADIENT_DEVELOPING_WEIGHT = 0.25;
const GRADIENT_SCALE = 4.0;
const GRADIENT_CAP = 0.2;

/**
 * Terminal-aware leaf value in [-1, 1] for `player` to move. +1: the
 * player has a winning move. -1: the opponent has >= 2 winning moves.
 * -0.25: the opponent has exactly 1 (must block). 0: quiet.
 */
export function terminalLeafValue(board: Int8Array, player: Player): number {
  if (winningMovesV1(board, player).length > 0) return 1.0;
  const oppWins = winningMovesV1(board, opponent(player));
  if (oppWins.length >= 2) return -1.0;
  if (oppWins.length === 1) return MUST_BLOCK_VALUE;
  return 0.0;
}

/** Attack breadth of one side: strong moves + 0.25 per developing move, over empty cells only. */
function breadth(attack: Int32Array, board: Int8Array): number {
  let strong = 0;
  let developing = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] !== 0) continue;
    const s = attack[i];
    if (s >= GRADIENT_STRONG) strong++;
    else if (s >= GRADIENT_DEVELOPING) developing++;
  }
  return strong + GRADIENT_DEVELOPING_WEIGHT * developing;
}

/**
 * Breadth-gradient leaf value in (-0.2, 0.2) for `player` to move in a
 * quiet position: `0.2 * tanh((breadth(ours) - breadth(theirs)) / 4)`
 * from the kernel's attack-only scores for both colours. The 0.2 cap
 * keeps the static gradient strictly below the smallest exact terminal
 * value (the must-block 0.25) — see iter-7.md §2.1.
 */
export function attackGradient(board: Int8Array, player: Player): number {
  const [black, white] = attackBothJs(board);
  const ours = player === 1 ? black : white;
  const theirs = player === 1 ? white : black;
  const diff = breadth(ours, board) - breadth(theirs, board);
  return GRADIENT_CAP * Math.tanh(diff / GRADIENT_SCALE);
}

/** v7 leaf value: terminal-aware values for terminal positions, the attack gradient for quiet ones. */
export function gradientLeafValue(board: Int8Array, player: Player): number {
  const value = terminalLeafValue(board, player);
  if (value !== 0.0) return value;
  return attackGradient(board, player);
}

/** Softmax over log1p(v1 scores) / temperature, over all candidate moves. Keys are flat move indices. */
function priors(board: Int8Array, player: Player): Map<number, number> {
  const cands = candidateMovesV1(board);
  const out = new Map<number, number>();
  if (cands.length === 0) return out;
  const logits = new Map<number, number>();
  let top = -Infinity;
  for (const m of cands) {
    const l = Math.log1p(Math.max(0, scoreMoveV1(board, m, player))) / PRIOR_TEMPERATURE;
    logits.set(m, l);
    if (l > top) top = l;
  }
  let total = 0;
  for (const [m, l] of logits) {
    const w = Math.exp(l - top);
    out.set(m, w);
    total += w;
  }
  for (const [m, w] of out) out.set(m, w / total);
  return out;
}

/* ------------------------------------------------------------------ */
/* PUCT MCTS (faithful port of mcts_heuristic.HeuristicMCTS)           */
/* ------------------------------------------------------------------ */

interface V7Node {
  prior: number;
  visitCount: number;
  valueSum: number;
  /** Null until expansion. Children keyed by flat move index, in candidate (ascending) order. */
  children: Map<number, V7Node> | null;
}

function createNode(prior: number): V7Node {
  return { prior, visitCount: 0, valueSum: 0, children: null };
}

function nodeQ(node: V7Node): number {
  return node.visitCount > 0 ? node.valueSum / node.visitCount : 0;
}

export interface HeuristicMctsOptions {
  sims?: number;
  cPuct?: number;
  leafMode?: 'terminal' | 'gradient';
  /** Tie-break RNG for the root choice, returning a float in [0, 1). Default: first best move (deterministic). */
  rng?: () => number;
  /** Deadline in `performance.now()` ms; checked every 8 sims. Default: no cap. */
  deadlineMs?: number;
}

export interface V7Result {
  move: Point;
  winRate: number;
  topMoves: TopMove[];
  nodes: number;
  ms: number;
  depth: number;
  sims: number;
  backend: 'v7';
}

export class HeuristicMCTS {
  private readonly sims: number;
  private readonly cPuct: number;
  private readonly rng: (() => number) | null;
  private readonly deadlineMs: number;
  private readonly leafValue: (board: Int8Array, player: Player) => number;

  constructor(opts: HeuristicMctsOptions = {}) {
    const leafMode = opts.leafMode ?? 'gradient';
    if (leafMode !== 'terminal' && leafMode !== 'gradient') {
      throw new Error(`unknown leaf_mode: ${leafMode} (known: terminal, gradient)`);
    }
    this.sims = opts.sims ?? V7_SIMS;
    this.cPuct = opts.cPuct ?? V7_C_PUCT;
    this.rng = opts.rng ?? null;
    this.deadlineMs = opts.deadlineMs ?? Infinity;
    this.leafValue = leafMode === 'gradient' ? gradientLeafValue : terminalLeafValue;
  }

  chooseMove(boardIn: ArrayLike<number>, player: Player): V7Result {
    const started = performance.now();
    const board = Int8Array.from(boardIn as ArrayLike<number>);

    // Immediate win (any winning move wins).
    const wins = winningMovesV1(board, player);
    if (wins.length > 0) {
      const move = wins[0];
      return {
        move: { x: move % SIZE, y: Math.floor(move / SIZE) },
        winRate: 1,
        topMoves: [{ move: { x: move % SIZE, y: Math.floor(move / SIZE) }, winRate: 1 }],
        nodes: 1,
        ms: performance.now() - started,
        depth: 0,
        sims: 0,
        backend: 'v7',
      };
    }
    // Block the opponent's immediate win (if exactly one; with 2+ the
    // game is lost and the search picks the best effort).
    const oppWins = winningMovesV1(board, opponent(player));
    if (oppWins.length === 1) {
      const move = oppWins[0];
      return {
        move: { x: move % SIZE, y: Math.floor(move / SIZE) },
        winRate: 0.5,
        topMoves: [{ move: { x: move % SIZE, y: Math.floor(move / SIZE) }, winRate: 0.5 }],
        nodes: 1,
        ms: performance.now() - started,
        depth: 0,
        sims: 0,
        backend: 'v7',
      };
    }
    const cands = candidateMovesV1(board);
    if (cands.length === 0) {
      throw new Error('No legal moves: position is terminal');
    }
    if (cands.length === 1) {
      const move = cands[0];
      return {
        move: { x: move % SIZE, y: Math.floor(move / SIZE) },
        winRate: 0.5,
        topMoves: [{ move: { x: move % SIZE, y: Math.floor(move / SIZE) }, winRate: 0.5 }],
        nodes: 1,
        ms: performance.now() - started,
        depth: 0,
        sims: 0,
        backend: 'v7',
      };
    }

    const root = createNode(1.0);
    const state = { nodesCreated: 1, maxDepth: 0 };
    let simsRun = 0;
    for (let s = 0; s < this.sims; s++) {
      if ((s & 7) === 0 && s > 0 && performance.now() > this.deadlineMs) break;
      this.runOne(root, board, player, null, 0, state);
      simsRun++;
    }

    const children = [...(root.children as Map<number, V7Node>).entries()];
    let bestVisits = -1;
    for (const [, c] of children) {
      if (c.visitCount > bestVisits) bestVisits = c.visitCount;
    }
    const best = children.filter(([, c]) => c.visitCount === bestVisits).map(([m]) => m);
    let chosen: number;
    if (best.length === 1) {
      chosen = best[0];
    } else if (this.rng !== null) {
      chosen = best[Math.floor(this.rng() * best.length)];
    } else {
      chosen = best[0];
    }

    const topMoves: TopMove[] = children
      .map(([m, c]) => ({
        move: { x: m % SIZE, y: Math.floor(m / SIZE) },
        // Child q is from the opponent's perspective; negate for the side to move.
        winRate: Math.min(1, Math.max(0, (-nodeQ(c) + 1) / 2)),
        visits: c.visitCount,
      }))
      .sort((a, b) => (b.visits ?? 0) - (a.visits ?? 0) || b.winRate - a.winRate)
      .slice(0, 5);
    const bestChild = (root.children as Map<number, V7Node>).get(chosen) as V7Node;

    return {
      move: { x: chosen % SIZE, y: Math.floor(chosen / SIZE) },
      winRate: Math.min(1, Math.max(0, (-nodeQ(bestChild) + 1) / 2)),
      topMoves,
      nodes: state.nodesCreated,
      ms: performance.now() - started,
      depth: state.maxDepth,
      sims: simsRun,
      backend: 'v7',
    };
  }

  /**
   * PUCT: `-child.q` (child values are from the opponent's perspective)
   * plus the exploration term. Ties keep the first child in insertion
   * order (ascending move index), matching Python's strict `>` scan.
   */
  private selectChild(node: V7Node): number {
    let bestMove = -1;
    let bestScore = -Infinity;
    const parentVisits = node.visitCount;
    for (const [move, child] of node.children as Map<number, V7Node>) {
      const score =
        -nodeQ(child) +
        (this.cPuct * child.prior * Math.sqrt(parentVisits)) / (1 + child.visitCount);
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
    }
    return bestMove;
  }

  /**
   * One simulation. Returns the value from `player`'s perspective and
   * updates every node on the path (including the newly expanded leaf).
   * Mutates `board` in place but restores it before returning. `state`
   * collects reporting counters only (nodes created, max depth reached)
   * and does not affect play.
   *
   * NOTE on the terminal check: `isWinningMoveV1` returns false on an
   * occupied cell, and the parent already placed the stone — so the win
   * branch never fires, exactly as in the Python reference (verified:
   * 27 wins played inside a 100-sim Python tree, 0 terminal-branch
   * hits). Do not "fix" this to `hasFive` without re-validating the
   * whole engine: v7's measured strength comes from the code as written.
   */
  private runOne(
    node: V7Node,
    board: Int8Array,
    player: Player,
    lastMove: number | null,
    depth: number,
    state: { nodesCreated: number; maxDepth: number },
  ): number {
    // Terminal check: the move that led here just ended the game.
    if (lastMove !== null) {
      if (isWinningMoveV1(board, lastMove, opponent(player))) {
        node.visitCount += 1;
        node.valueSum += -1.0;
        return -1.0;
      }
      if (isFull(board)) {
        node.visitCount += 1;
        return 0.0;
      }
    }
    // Expansion: first visit — compute priors, evaluate the leaf.
    if (node.children === null) {
      const ps = priors(board, player);
      node.children = new Map<number, V7Node>();
      for (const [m, p] of ps) node.children.set(m, createNode(p));
      state.nodesCreated += ps.size;
      if (depth > state.maxDepth) state.maxDepth = depth;
      if (lastMove === null) {
        // Root: no leaf value.
        node.visitCount += 1;
        return 0.0;
      }
      const value = this.leafValue(board, player);
      node.visitCount += 1;
      node.valueSum += value;
      return value;
    }
    // Selection: descend to the best child by PUCT.
    const move = this.selectChild(node);
    board[move] = player;
    const value = -this.runOne(
      (node.children as Map<number, V7Node>).get(move) as V7Node,
      board,
      opponent(player),
      move,
      depth + 1,
      state,
    );
    board[move] = 0;
    node.visitCount += 1;
    node.valueSum += value;
    return value;
  }
}

/* ------------------------------------------------------------------ */
/* Master-level entry point                                            */
/* ------------------------------------------------------------------ */

export interface V7MoveOptions {
  sims?: number;
  timeMs?: number;
  temperature?: number;
  /** Tie-break RNG for the root choice. Default: Math.random (page play). */
  rng?: () => number;
}

/**
 * Choose a move with the v7 engine (the Master level). `temperature` is
 * accepted for interface compatibility with the other MCTS levels but v7
 * always plays the most-visited move — temperature > 0 is ignored, and
 * the caller (engine.ts) documents that. `timeMs` caps the search as a
 * deadline; v7 normally finishes its 100 sims well inside it.
 */
export function chooseV7Move(
  boardIn: ArrayLike<number>,
  player: Player,
  opts: V7MoveOptions = {},
): V7Result {
  const started = performance.now();
  const timeMs = opts.timeMs ?? 5000;
  const engine = new HeuristicMCTS({
    sims: opts.sims ?? V7_SIMS,
    cPuct: V7_C_PUCT,
    leafMode: 'gradient',
    rng: opts.rng ?? Math.random,
    deadlineMs: started + Math.max(1, timeMs),
  });
  return engine.chooseMove(boardIn, player);
}

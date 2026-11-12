/**
 * L2 Intermediate + L3 Advanced — alpha-beta search over the pattern eval.
 *
 * Both levels share one code path with different parameters:
 * - Intermediate: iterative deepening capped at depth 3, no threat-space search.
 * - Advanced: iterative deepening to depth 8 (time-bounded), plus a VCF-style
 *   threat-space search that finds forced wins built from consecutive fours.
 *
 * Features: negamax alpha-beta, Zobrist transposition table (exact / lower /
 * upper bounds), killer moves + history heuristic for move ordering, and
 * time-bounded iterative deepening (on timeout, the best move from the last
 * *completed* depth is returned, so a slow position never loses the result).
 *
 * Known limitations (honest): no quiescence search, so the horizon effect
 * applies — a tactic just beyond the depth limit can be missed. The VCF
 * search only follows fours (not open-three threats) and assumes the defender
 * always blocks the single winning square; a defender counter-threat that
 * wins later is not modelled. Branching is capped (top ~10-14 moves by
 * pattern score) so quiet positional moves off the beaten track are pruned.
 */

import { SIZE, hasFive, opponent, type Player, type Point } from '../engine/gomoku';
import {
  WIN_SCORE,
  candidateMoves,
  evaluateBoard,
  findWinningMoves,
  orderedCandidates,
  scoreMove,
  winRateFromEval,
} from './patterns';
import type { TopMove } from './types';

export interface SearchOptions {
  /** Maximum iterative-deepening depth. */
  maxDepth: number;
  /** Time budget in milliseconds. */
  timeMs: number;
  /** Maximum branching factor (top N ordered moves per node). Default 12. */
  branchLimit?: number;
  /** Run the VCF threat-space search before the main search. Default false. */
  useVct?: boolean;
  /** Maximum plies for the VCF search (attacker moves + defender replies). Default 9. */
  vctPlies?: number;
}

export interface SearchResult {
  move: Point;
  winRate: number;
  topMoves: TopMove[];
  nodes: number;
  ms: number;
  depth: number;
  backend: string;
}

class SearchTimeout extends Error {
  constructor() {
    super('Search time limit reached');
    this.name = 'SearchTimeout';
  }
}

/* ------------------------------------------------------------------ */
/* Zobrist hashing (deterministic seed so runs are reproducible)        */
/* ------------------------------------------------------------------ */

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ZOBRIST: bigint[][] = (() => {
  const rand = mulberry32(0x9e3779b9);
  const table: bigint[][] = [[], []];
  for (let color = 0; color < 2; color++) {
    for (let i = 0; i < SIZE * SIZE; i++) {
      const hi = BigInt(Math.floor(rand() * 0x1_0000_0000));
      const lo = BigInt(Math.floor(rand() * 0x1_0000_0000));
      table[color][i] = (hi << 32n) | lo;
    }
  }
  return table;
})();

const ZOBRIST_SIDE: bigint = (() => {
  const rand = mulberry32(0x85ebca6b);
  const hi = BigInt(Math.floor(rand() * 0x1_0000_0000));
  const lo = BigInt(Math.floor(rand() * 0x1_0000_0000));
  return (hi << 32n) | lo;
})();

function colorIndex(p: Player): number {
  return p === 1 ? 0 : 1;
}

function computeHash(board: Int8Array, toMove: Player): bigint {
  let h = 0n;
  for (let i = 0; i < SIZE * SIZE; i++) {
    const v = board[i];
    if (v !== 0) h ^= ZOBRIST[colorIndex(v as Player)][i];
  }
  if (toMove === 1) h ^= ZOBRIST_SIDE;
  return h;
}

/* ------------------------------------------------------------------ */
/* Transposition table                                                 */
/* ------------------------------------------------------------------ */

const TT_EXACT = 0;
const TT_LOWER = 1;
const TT_UPPER = 2;

interface TTEntry {
  depth: number;
  score: number;
  flag: number;
  bestMove: number; // cell index, or -1
}

const TT_MAX_ENTRIES = 200_000;

/* ------------------------------------------------------------------ */
/* VCF threat-space search: forced wins from consecutive fours         */
/* ------------------------------------------------------------------ */

/**
 * Moves that create a four: after playing the move, `player` has a winning
 * move next turn. Immediate winning moves are excluded (they are wins, not
 * threats — the shared pre-check in chooseMove handles those first).
 * Ordered by pattern score, capped at 8.
 */
function threatMoves(board: Int8Array, player: Player): Point[] {
  const out: { c: Point; s: number }[] = [];
  for (const c of candidateMoves(board, 2)) {
    const i = c.y * SIZE + c.x;
    board[i] = player;
    const immediateWin = hasFive(board, c.x, c.y);
    const createsFour = !immediateWin && findWinningMoves(board, player).length > 0;
    board[i] = 0;
    if (createsFour) out.push({ c, s: scoreMove(board, c.x, c.y, player) });
  }
  out.sort((a, b) => b.s - a.s || a.c.y * SIZE + a.c.x - (b.c.y * SIZE + b.c.x));
  return out.slice(0, 8).map((o) => o.c);
}

/**
 * Find the first move of a forced win for `player` built from consecutive
 * fours (VCF-style threat-space search), within `maxPlies` plies.
 *
 * Logic: the attacker plays a threat (a move creating a four). The defender
 * must then either win immediately (which refutes the line) or block the
 * single winning square — any other reply loses on the spot. If the threat
 * creates two winning squares, it is unblockable and the line wins at once.
 * Returns the first move of a forced win, or null.
 *
 * Bounded approximation: only fours are followed (open-three threats are left
 * to the main search), and a defender counter-threat that wins several moves
 * later is not modelled. `deadlineMs` (a performance.now() timestamp) aborts
 * the search and returns null — used so VCF can never eat the whole time
 * budget before the main search runs.
 */
export function findForcedWin(
  boardIn: ArrayLike<number>,
  player: Player,
  maxPlies = 9,
  deadlineMs = Infinity,
): Point | null {
  const board = Int8Array.from(boardIn as ArrayLike<number>);
  const defender = opponent(player);

  // Attacker's turn with `pliesLeft` plies remaining. True if the attacker
  // can force a win from here.
  function attack(pliesLeft: number): boolean {
    if (performance.now() > deadlineMs) throw new SearchTimeout();
    if (findWinningMoves(board, player).length > 0) return true;
    if (pliesLeft <= 0) return false;
    for (const t of threatMoves(board, player)) {
      if (tryThreat(t, pliesLeft)) return true;
    }
    return false;
  }

  // Play threat move `t` (attacker) and resolve the defender's forced reply.
  function tryThreat(t: Point, pliesLeft: number): boolean {
    const ti = t.y * SIZE + t.x;
    board[ti] = player;
    // Defender wins on the spot -> this threat line is refuted.
    if (findWinningMoves(board, defender).length > 0) {
      board[ti] = 0;
      return false;
    }
    const wins = findWinningMoves(board, player);
    if (wins.length >= 2) {
      // Double threat: the defender can block at most one square.
      board[ti] = 0;
      return true;
    }
    if (wins.length === 1) {
      // Defender's only non-losing reply is the block.
      const b = wins[0];
      const bi = b.y * SIZE + b.x;
      board[bi] = defender;
      const ok = attack(pliesLeft - 2);
      board[bi] = 0;
      board[ti] = 0;
      return ok;
    }
    board[ti] = 0;
    return false;
  }

  try {
    for (const t of threatMoves(board, player)) {
      if (tryThreat(t, maxPlies)) return t;
    }
  } catch (err) {
    if (err instanceof SearchTimeout) return null;
    throw err;
  }
  return null;
}

/* ------------------------------------------------------------------ */
/* Alpha-beta search                                                   */
/* ------------------------------------------------------------------ */

export function searchMove(
  boardIn: ArrayLike<number>,
  player: Player,
  opts: SearchOptions,
): SearchResult {
  const started = performance.now();
  const deadline = started + Math.max(1, opts.timeMs);
  const branchLimit = opts.branchLimit ?? 12;
  const board = Int8Array.from(boardIn as ArrayLike<number>);

  // VCF pre-pass (Advanced only): a forced win ends the search immediately.
  // VCF gets at most 25% of the budget (capped at 500ms) so the main search
  // always gets its share.
  if (opts.useVct) {
    const vctDeadline = started + Math.min(500, opts.timeMs * 0.25);
    const forced = findForcedWin(board, player, opts.vctPlies ?? 9, vctDeadline);
    if (forced) {
      return {
        move: forced,
        winRate: 1,
        topMoves: [{ move: forced, winRate: 1 }],
        nodes: 1,
        ms: performance.now() - started,
        depth: 0,
        backend: 'alphabeta-vct',
      };
    }
    if (performance.now() > deadline) {
      // VCF ate the whole budget: fall back to the best 1-ply move.
      const cands = orderedCandidates(board, player, 5);
      if (cands.length === 0) throw new Error('No legal moves: position is terminal');
      return {
        move: cands[0],
        winRate: winRateFromEval(scoreMove(board, cands[0].x, cands[0].y, player)),
        topMoves: cands.map((c) => ({
          move: c,
          winRate: winRateFromEval(scoreMove(board, c.x, c.y, player)),
        })),
        nodes: 1,
        ms: performance.now() - started,
        depth: 0,
        backend: 'alphabeta-vct',
      };
    }
  }

  const rootMoves = orderedCandidates(board, player, Math.max(branchLimit, 16));
  if (rootMoves.length === 0) throw new Error('No legal moves: position is terminal');

  const tt = new Map<bigint, TTEntry>();
  // Killer moves: 2 per ply, stored as cell indices (-1 = empty).
  const killers: Int16Array[] = Array.from({ length: 64 }, () => new Int16Array([-1, -1]));
  // History heuristic: [color][cell].
  const history: Int32Array[] = [new Int32Array(SIZE * SIZE), new Int32Array(SIZE * SIZE)];

  let nodes = 0;
  let hash = computeHash(board, player);

  function orderMoves(
    moves: Point[],
    ply: number,
    mover: Player,
    ttMove: number,
  ): { move: Point; idx: number; key: number }[] {
    // `moves` already arrives sorted by scoreMove desc (from orderedCandidates),
    // so the list position stands in for the pattern score — no re-scoring.
    // History acts as a tiebreak within a position band, capped so it can
    // never override the pattern ordering.
    const k = killers[Math.min(ply, killers.length - 1)];
    const hist = history[colorIndex(mover)];
    const ordered = moves.map((move, pos) => {
      const i = move.y * SIZE + move.x;
      let key = (moves.length - pos) * 1e6 + Math.min(hist[i], 999_999);
      if (i === ttMove) key += 1e12;
      else if (i === k[0]) key += 1e11;
      else if (i === k[1]) key += 1e10;
      return { move, idx: i, key };
    });
    ordered.sort((a, b) => b.key - a.key || a.idx - b.idx);
    return ordered;
  }

  function negamax(
    depth: number,
    alpha: number,
    beta: number,
    ply: number,
    toMove: Player,
  ): number {
    nodes++;
    // Check the deadline often: per-node cost is high (candidate scoring),
    // so a coarse check interval overshoots the budget badly.
    if ((nodes & 255) === 0 && performance.now() > deadline) throw new SearchTimeout();
    if (depth === 0) return evaluateBoard(board, toMove);

    const alphaOrig = alpha;
    const entry = tt.get(hash);
    let ttMove = -1;
    if (entry && entry.depth >= depth) {
      ttMove = entry.bestMove;
      if (entry.flag === TT_EXACT) return entry.score;
      if (entry.flag === TT_LOWER) alpha = Math.max(alpha, entry.score);
      else if (entry.flag === TT_UPPER) beta = Math.min(beta, entry.score);
      if (alpha >= beta) return entry.score;
    } else if (entry) {
      ttMove = entry.bestMove;
    }

    const moves = orderedCandidates(board, toMove, branchLimit);
    const ordered = orderMoves(moves, ply, toMove, ttMove);
    let best = -Infinity;
    let bestMoveIdx = -1;
    for (const { move, idx: i } of ordered) {
      board[i] = toMove;
      hash ^= ZOBRIST[colorIndex(toMove)][i];
      hash ^= ZOBRIST_SIDE;
      let score: number;
      if (hasFive(board, move.x, move.y)) {
        // Win for the mover; prefer faster wins via the ply penalty.
        score = WIN_SCORE - ply;
      } else {
        score = -negamax(depth - 1, -beta, -alpha, ply + 1, opponent(toMove));
      }
      board[i] = 0;
      hash ^= ZOBRIST[colorIndex(toMove)][i];
      hash ^= ZOBRIST_SIDE;

      if (score > best) {
        best = score;
        bestMoveIdx = i;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) {
        // Beta cutoff: record killer + history.
        const k = killers[Math.min(ply, killers.length - 1)];
        if (k[0] !== i) {
          k[1] = k[0];
          k[0] = i;
        }
        history[colorIndex(toMove)][i] += depth * depth;
        break;
      }
    }

    if (best === -Infinity) return 0; // No moves: draw (full board).
    if (tt.size < TT_MAX_ENTRIES) {
      const flag = best <= alphaOrig ? TT_UPPER : best >= beta ? TT_LOWER : TT_EXACT;
      tt.set(hash, { depth, score: best, flag, bestMove: bestMoveIdx });
    }
    return best;
  }

  // Iterative deepening. On timeout, keep the last completed iteration.
  let bestMove = rootMoves[0];
  let bestScore = -Infinity;
  let rootScores = new Map<number, number>();
  let completedDepth = 0;

  for (let depth = 2; depth <= opts.maxDepth; depth++) {
    const iterationScores = new Map<number, number>();
    let iterationBest = rootMoves[0];
    let iterationBestScore = -Infinity;
    try {
      // Search the previous iteration's best move first for better ordering.
      const orderedRoot = rootMoves.slice().sort((a, b) => {
        const ai = a.y * SIZE + a.x;
        const bi = b.y * SIZE + b.x;
        if (ai === bestMove.y * SIZE + bestMove.x) return -1;
        if (bi === bestMove.y * SIZE + bestMove.x) return 1;
        return (rootScores.get(bi) ?? -Infinity) - (rootScores.get(ai) ?? -Infinity);
      });
      let alpha = -Infinity;
      for (const move of orderedRoot) {
        const i = move.y * SIZE + move.x;
        board[i] = player;
        hash ^= ZOBRIST[colorIndex(player)][i];
        hash ^= ZOBRIST_SIDE;
        let score: number;
        if (hasFive(board, move.x, move.y)) {
          score = WIN_SCORE;
        } else {
          // Full window at the root: a narrowing (null) window would return
          // only a bound for moves better than alpha, under-scoring them.
          score = -negamax(depth - 1, -Infinity, Infinity, 1, opponent(player));
        }
        board[i] = 0;
        hash ^= ZOBRIST[colorIndex(player)][i];
        hash ^= ZOBRIST_SIDE;
        iterationScores.set(i, score);
        if (score > iterationBestScore) {
          iterationBestScore = score;
          iterationBest = move;
        }
        if (score > alpha) alpha = score;
      }
    } catch (err) {
      if (err instanceof SearchTimeout) break;
      throw err;
    }
    bestMove = iterationBest;
    bestScore = iterationBestScore;
    rootScores = iterationScores;
    completedDepth = depth;
    // A forced win is found: no need to search deeper.
    if (bestScore >= WIN_SCORE - 1000) break;
    if (performance.now() > deadline) break;
  }

  const topMoves: TopMove[] = rootMoves
    .map((move) => {
      const i = move.y * SIZE + move.x;
      const score = rootScores.get(i);
      return {
        move,
        winRate: score !== undefined ? winRateFromEval(score) : winRateFromEval(0),
      };
    })
    .sort((a, b) => b.winRate - a.winRate)
    .slice(0, 5);

  return {
    move: bestMove,
    winRate: winRateFromEval(bestScore === -Infinity ? 0 : bestScore),
    topMoves,
    nodes,
    ms: performance.now() - started,
    depth: completedDepth,
    backend: 'alphabeta',
  };
}

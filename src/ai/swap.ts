/**
 * Official swap rule (standard swap / pie rule — NOT Swap2) for the web app.
 *
 * The rule (RIF Art. 2.4; see research/iterations/swap-rule.md):
 * - The opener places 3 stones in order: black, white, black.
 * - The second player then picks a side (play Black or play White).
 * - The 3 opening stones ARE moves 1–3, so the player who ends up White
 *   plays move 4. (A real batch-1 harness bug gave Black move 4 — the
 *   tests pin White-to-move after the opening explicitly.)
 * - Win condition unchanged: first to 5+ in a row (freestyle; overline
 *   counts, no Renju forbidden moves, no exactly-five).
 *
 * This module is a faithful TypeScript port of the Python research code:
 * - v1 attack-only scorer: ml/gomoku_ml/scoring.py (score_attack /
 *   _score_parts_flat). NOTE: this is the v1 5-window scorer, NOT the
 *   kernel's v0 contiguous-run scoring in src/kernel/scoring.ts (which
 *   differs — e.g. it is blind to broken fours) and NOT the TS scorer in
 *   src/ai/patterns.ts (which sums all windows per direction and scores
 *   a fully-closed four-window as 0). The port is verified score-for-score
 *   against Python fixtures in swap-fixtures.ts.
 * - Balanced opening placement: ml/gomoku_ml/research.py
 *   (place_swap_opening, placement="balanced"). The "chooser" policy is
 *   deliberately NOT ported: batch 1 proved it flawed (makes Black
 *   systematically stronger; the colour choice degenerated to 20/20 Black).
 * - Tempo-aware colour judge: ml/gomoku_ml/game.py
 *   (choose_swap_color_tempo). CRITICAL: the total score is exactly
 *   symmetric between colours (attack and defence swap roles), so the
 *   judge MUST compare attack-only scores or it always ties. Ties go to
 *   White (White moves next — the tempo breaks the tie).
 *
 * The engine needs no changes for the swap rule: the opening stones are
 * just moves 1–3 played through applyMove (Black, White, Black), and
 * after move 3 the engine already has White to move. The phases
 * (opening / choice / playing) live in the game hook (useGomokuGame).
 */

import { SIZE, type Player, type Point } from '../engine/gomoku';

export const SWAP_OPENING_STONES = 3;
export const SWAP_CENTRE: Point = { x: 7, y: 7 };

/** Weights mirror the Python v1 scorer (ml/gomoku_ml/scoring.py). */
export const SWAP_WIN_SCORE = 10_000_000;
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

/** Pattern kinds (per direction, from the strongest window in it). */
const KIND_NONE = 0;
const KIND_FOUR = 1;
const KIND_OPEN_THREE = 2;
const KIND_FIVE = 3;

const DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [1, 0],
  [0, 1],
  [1, 1],
  [1, -1],
];

interface BoardWindow {
  cells: [number, number, number, number, number];
  /** Cell index just before the window along its line, or -1 if off-board. */
  before: number;
  /** Cell index just after the window along its line, or -1 if off-board. */
  after: number;
  /** Index into DIRECTIONS. */
  dir: number;
}

function buildWindows(): BoardWindow[] {
  const wins: BoardWindow[] = [];
  DIRECTIONS.forEach(([dx, dy], dir) => {
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
        wins.push({ cells, before, after, dir });
      }
    }
  });
  return wins;
}

/** All 572 five-cell windows on the board. */
const WINDOWS: BoardWindow[] = buildWindows();

/**
 * For each cell index, the windows containing that cell (at most 20), as
 * [window index, position of the cell within the window]. The position lets
 * the scorer find the window's other empty cell for a four without a
 * per-cell "is this the move?" check (mirrors Python's _WINDOWS_BY_CELL).
 */
const WINDOWS_BY_CELL: Array<Array<readonly [number, number]>> = (() => {
  const byCell: Array<Array<readonly [number, number]>> = Array.from(
    { length: SIZE * SIZE },
    () => [],
  );
  WINDOWS.forEach((w, wi) => {
    w.cells.forEach((c, i) => byCell[c].push([wi, i] as const));
  });
  return byCell;
})();

/**
 * [score, kind] for a window holding `own` stones of the side that played
 * the candidate move (the move included) and `opp` opponent stones.
 * `emptyIdx` is the position (0–4) of the window's other empty cell (only
 * meaningful when own == 4, when there is exactly one besides the move);
 * `openEnds` counts the empty cells just outside the window. A window
 * containing an opponent stone scores 0 (blocked). Direct port of Python's
 * _score_from_counts.
 */
function scoreFromCounts(
  board: ArrayLike<number>,
  own: number,
  opp: number,
  emptyIdx: number,
  openEnds: number,
  before: number,
  after: number,
): [number, number] {
  if (opp > 0) return [0, KIND_NONE];
  if (own === 5) return [SWAP_WIN_SCORE, KIND_FIVE];
  if (own === 4) {
    // The empty cell inside the window is always a winning square, so this
    // is a four whatever the outside looks like. It is an OPEN four (two
    // winning squares) only when the four is contiguous and the outside
    // cell beyond its far end is empty: the gap of a broken four is its
    // only winning square.
    if (emptyIdx === 0) {
      // Four occupies cells 1–4; the far outside cell is `after`.
      if (after !== -1 && board[after] === 0) return [OPEN_FOUR, KIND_FOUR];
    } else if (emptyIdx === 4) {
      // Four occupies cells 0–3; the far outside cell is `before`.
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
 * V1 attack-only score for `player` playing at (x, y) on `board`: what the
 * move builds for `player`, including the double-threat bonuses, WITHOUT
 * the defence component. Direct port of Python's scoring.score_attack
 * (the attack half of _score_parts_flat):
 *
 * - 5-window scan (broken fours visible: XX_XX scores as a four).
 * - Max per direction, not sum over windows: a single contiguous four sits
 *   in two overlapping windows; summing would double-count one pattern.
 * - Double-threat bonuses: fours in >= 2 directions (+200k), open threes
 *   in >= 2 directions (+50k).
 *
 * This is the colour-ASYMMETRIC half of the v1 score — the total score
 * (attack + defence) is identical for both colours at any square, so any
 * comparison between colours (the swap colour choice) must use this, not
 * the total. The point must be empty.
 */
export function scoreAttackV1(
  board: ArrayLike<number>,
  x: number,
  y: number,
  player: Player,
): number {
  const move = y * SIZE + x;
  const opp: Player = player === 1 ? -1 : 1;
  const best: number[] = [0, 0, 0, 0];
  const kinds: number[] = [KIND_NONE, KIND_NONE, KIND_NONE, KIND_NONE];
  for (const [wi, moveIdx] of WINDOWS_BY_CELL[move]) {
    const w = WINDOWS[wi];
    let p = 0;
    let o = 0;
    let emptySum = 0;
    // The move cell is empty on the board, so it counts as empty here and
    // contributes to neither side — no per-cell check needed. For a four,
    // the other empty cell's position is emptySum - moveIdx (the empties
    // are the move cell and one other cell).
    for (let i = 0; i < 5; i++) {
      const v = board[w.cells[i]];
      if (v === player) p++;
      else if (v === opp) o++;
      else emptySum += i;
    }
    if (p > 0 && o > 0) continue; // blocked for both sides
    if (o > 0) continue; // attack needs a window with no opponent stones
    let openEnds = 0;
    if (w.before !== -1 && board[w.before] === 0) openEnds++;
    if (w.after !== -1 && board[w.after] === 0) openEnds++;
    const [s, kind] = scoreFromCounts(
      board,
      p + 1,
      0,
      emptySum - moveIdx,
      openEnds,
      w.before,
      w.after,
    );
    if (s > best[w.dir]) {
      best[w.dir] = s;
      kinds[w.dir] = kind;
    }
  }
  let attack = best[0] + best[1] + best[2] + best[3];
  const fours = kinds.filter((k) => k === KIND_FOUR || k === KIND_FIVE).length;
  const openThrees = kinds.filter((k) => k === KIND_OPEN_THREE).length;
  if (fours >= 2) attack += DOUBLE_FOUR_BONUS;
  if (openThrees >= 2) attack += DOUBLE_OPEN_THREE_BONUS;
  return attack;
}

/**
 * Empty points within Chebyshev distance `radius` of any stone, as flat
 * indices in ascending order. On an empty board, all legal moves. Direct
 * port of Python's research._attack_candidates.
 */
export function attackCandidates(board: ArrayLike<number>, radius: number): number[] {
  const near = new Uint8Array(SIZE * SIZE);
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
        if (board[j] === 0) near[j] = 1;
      }
    }
  }
  const out: number[] = [];
  if (!anyStone) {
    for (let i = 0; i < SIZE * SIZE; i++) if (board[i] === 0) out.push(i);
    return out;
  }
  for (let i = 0; i < SIZE * SIZE; i++) if (near[i]) out.push(i);
  return out;
}

/**
 * Best available attack score for `player` on `board` (max over legal
 * moves of scoreAttackV1). Scanned over the Chebyshev-4 candidates — exact
 * (see Python's research._attack_candidates: a cell further than 4 from
 * every stone shares no 5-window with any stone, so its attack score is
 * at most 8, while any board holding stones has a best attack >= 10).
 * Direct port of Python's research.best_attack.
 */
export function bestAttack(board: ArrayLike<number>, player: Player): number {
  const cands = attackCandidates(board, 4);
  let best = 0;
  for (const m of cands) {
    const s = scoreAttackV1(board, m % SIZE, Math.floor(m / SIZE), player);
    if (s > best) best = s;
  }
  return best;
}

/** [gap, bestBlack, bestWhite] for a swap-opening board. Port of research.opening_gap. */
export function openingGap(board: ArrayLike<number>): [number, number, number] {
  const bb = bestAttack(board, 1);
  const bw = bestAttack(board, -1);
  return [Math.abs(bb - bw), bb, bw];
}

/** True if `board` holds exactly a swap opening: 2 black stones, 1 white stone. */
export function isSwapOpening(board: ArrayLike<number>): boolean {
  let nBlack = 0;
  let nWhite = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] === 1) nBlack++;
    else if (board[i] === -1) nWhite++;
  }
  return nBlack === 2 && nWhite === 1;
}

/** Build the board for a swap opening from its 3 stones (placement order: black, white, black). */
export function buildSwapOpeningBoard(black1: Point, white1: Point, black2: Point): Int8Array {
  const board = new Int8Array(SIZE * SIZE);
  board[black1.y * SIZE + black1.x] = 1;
  board[white1.y * SIZE + white1.x] = -1;
  board[black2.y * SIZE + black2.x] = 1;
  return board;
}

/**
 * Best follow-up attack for `player` after its own best move. Among
 * `player`'s best immediate attack moves on `board`, take the one with the
 * best follow-up (max attack for `player` on the board after the setup
 * move) and return that follow-up value. Direct port of Python's
 * game._setup_value.
 */
export function setupValue(board: ArrayLike<number>, player: Player): number {
  const legal: number[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) if (board[i] === 0) legal.push(i);
  let top = -Infinity;
  const scored: Array<[number, number]> = legal.map((m) => {
    const s = scoreAttackV1(board, m % SIZE, Math.floor(m / SIZE), player);
    if (s > top) top = s;
    return [s, m];
  });
  let bestFollowup: number | null = null;
  for (const [score, move] of scored) {
    if (score !== top) continue;
    const after = Int8Array.from(board as ArrayLike<number>);
    after[move] = player;
    let followup = 0;
    for (let m2 = 0; m2 < SIZE * SIZE; m2++) {
      if (after[m2] !== 0) continue;
      const s = scoreAttackV1(after, m2 % SIZE, Math.floor(m2 / SIZE), player);
      if (s > followup) followup = s;
    }
    if (bestFollowup === null || followup > bestFollowup) bestFollowup = followup;
  }
  if (bestFollowup === null) throw new Error('setupValue: no legal moves');
  return bestFollowup;
}

/**
 * Tempo-aware colour choice on a swap-opening board (batch 2, H3).
 * Each colour's value is its best follow-up attack AFTER its own best
 * setup move (setupValue): the colours are compared after symmetric 1-ply
 * development, as in the real game where White plays move 4 and Black
 * move 5 before the race continues. The colour with the higher value wins;
 * ties go to WHITE, who moves next (move 4) — the tempo breaks the tie.
 * Direct port of Python's game.choose_swap_color_tempo.
 *
 * Throws unless `board` is a swap opening (2 black, 1 white).
 */
export function chooseSwapColorTempo(board: ArrayLike<number>): Player {
  if (!isSwapOpening(board)) {
    throw new Error('chooseSwapColorTempo needs a swap-opening board (2 black stones, 1 white stone)');
  }
  const valueBlack = setupValue(board, 1);
  const valueWhite = setupValue(board, -1);
  return valueWhite >= valueBlack ? -1 : 1;
}

/**
 * The plain attack-gap judge (batch-1 policy): compare each colour's best
 * attack and take the higher, ties to White. Kept for comparison/testing —
 * it picks Black on every balanced opening (see swap-rule.md), which is why
 * the app uses the tempo judge. Port of Python's game.choose_swap_color
 * with judge="attack".
 */
export function chooseSwapColorAttack(board: ArrayLike<number>): Player {
  if (!isSwapOpening(board)) {
    throw new Error('chooseSwapColorAttack needs a swap-opening board (2 black stones, 1 white stone)');
  }
  const bestBlack = bestAttack(board, 1);
  const bestWhite = bestAttack(board, -1);
  return bestWhite >= bestBlack ? -1 : 1;
}

/** Small deterministic PRNG (mulberry32) for the seeded White-stone draw. */
export function createSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Balanced swap opening (policy v2 — the harness default for research
 * batches). Direct port of Python's research._place_swap_opening_balanced:
 *
 * - Black's first stone is the centre (7,7).
 * - White's stone is drawn (seeded RNG) from the cells within Chebyshev
 *   distance 2 of the centre (24 cells) — varied openings. It is drawn
 *   rather than "White's best attack move" because with no White stones
 *   on the board every move scores ~equally, so argmax would pick an
 *   arbitrary edge cell.
 * - Black's second stone is the nearby candidate (within Chebyshev
 *   distance 2 of a stone) minimising the attack-score gap on the
 *   finished opening. Ties go to the smaller Black best-attack, then the
 *   lowest flat index.
 *
 * The compact region matters: an unconstrained gap search finds
 * degenerate scattered openings (gap 0 by isolation — stones that do
 * not interact), which are not real Gomoku positions.
 *
 * Returns the 3 stones in placement order [black1, white1, black2].
 * Pass a seeded RNG for reproducibility; otherwise Math.random is used.
 */
export function placeBalancedOpening(
  rng: () => number = Math.random,
): [Point, Point, Point] {
  const board = new Int8Array(SIZE * SIZE);
  const black1: Point = { x: 7, y: 7 };
  board[7 * SIZE + 7] = 1;
  // White's stone: draw from the cells within Chebyshev distance 2 of the
  // centre (24 cells), in ascending flat-index order.
  const region: number[] = [];
  for (let i = 0; i < SIZE * SIZE; i++) {
    if (board[i] !== 0) continue;
    const x = i % SIZE;
    const y = Math.floor(i / SIZE);
    if (Math.max(Math.abs(y - 7), Math.abs(x - 7)) <= 2) region.push(i);
  }
  const whiteIdx = region[Math.floor(rng() * region.length)];
  const white1: Point = { x: whiteIdx % SIZE, y: Math.floor(whiteIdx / SIZE) };
  board[whiteIdx] = -1;
  // Black's second stone: the nearby candidate minimising the gap.
  let best: [number, number, number] | null = null; // [gap, bestBlack, move]
  for (const cand of attackCandidates(board, 2)) {
    const trial = new Int8Array(board);
    trial[cand] = 1;
    const [gap, bb] = openingGap(trial);
    const key: [number, number, number] = [gap, bb, cand];
    if (
      best === null ||
      key[0] < best[0] ||
      (key[0] === best[0] && (key[1] < best[1] || (key[1] === best[1] && key[2] < best[2])))
    ) {
      best = key;
    }
  }
  if (best === null) throw new Error('placeBalancedOpening: no candidate for Black\'s second stone');
  const black2: Point = { x: best[2] % SIZE, y: Math.floor(best[2] / SIZE) };
  return [black1, white1, black2];
}

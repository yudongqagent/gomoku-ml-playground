// Local pattern scan for tactical threats (open fours, fours, open threes).
// This is UI-side analysis only — it lives in components/ on purpose and does
// not touch src/engine or src/ai. It scans every row, column and diagonal for
// windows that match the classic Gomoku threat shapes.
//
// Definitions used here:
//   - "four": four stones in a row with an empty square that completes five
//     (straight XXXX_ / _XXXX, or split _XXX_X_ / _XX_XX_ / _X_XXX_).
//   - "open four": four stones with BOTH ends open (_XXXX_) — unstoppable.
//   - "open three": three stones with both ends open (_XXX_, _XX_X_, _X_XX_).

import { SIZE } from '../engine/gomoku';
import type { Point } from '../engine/gomoku';

export type ThreatKind = 'open-four' | 'four' | 'open-three';

export interface Threat {
  player: 1 | -1;
  kind: ThreatKind;
  /** The stones forming the threat. */
  cells: Point[];
  /** Empty intersections that matter: play here to complete / extend it. */
  keySquares: Point[];
}

interface LineCell {
  x: number;
  y: number;
  v: number;
}

const MAX_THREATS_PER_KIND = 12;

function allLines(board: Int8Array | number[], size: number): LineCell[][] {
  const lines: LineCell[][] = [];
  const at = (x: number, y: number): number => board[y * size + x];
  // Rows and columns.
  for (let i = 0; i < size; i++) {
    const row: LineCell[] = [];
    const col: LineCell[] = [];
    for (let j = 0; j < size; j++) {
      row.push({ x: j, y: i, v: at(j, i) });
      col.push({ x: i, y: j, v: at(i, j) });
    }
    lines.push(row, col);
  }
  // Diagonals (down-right and down-left), only those of length >= 5.
  for (let start = -(size - 5); start <= size - 5; start++) {
    const diagDR: LineCell[] = [];
    const diagDL: LineCell[] = [];
    for (let k = 0; k < size; k++) {
      const x1 = start + k;
      if (x1 >= 0 && x1 < size) diagDR.push({ x: x1, y: k, v: at(x1, k) });
      const x2 = size - 1 - start - k;
      if (x2 >= 0 && x2 < size) diagDL.push({ x: x2, y: k, v: at(x2, k) });
    }
    if (diagDR.length >= 5) lines.push(diagDR);
    if (diagDL.length >= 5) lines.push(diagDL);
  }
  return lines;
}

function keyOf(cells: Point[]): string {
  return cells
    .map((c) => `${c.x},${c.y}`)
    .sort()
    .join('|');
}

function scanLine(line: LineCell[], player: 1 | -1, out: Threat[]): void {
  const n = line.length;
  const isStone = (i: number) => line[i].v === player;
  const isEmpty = (i: number) => line[i].v === 0;

  // Windows of 5: straight four (4 stones + 1 empty) and straight open
  // three (_XXX_).
  for (let i = 0; i + 5 <= n; i++) {
    let stones = 0;
    let empties = 0;
    const cells: Point[] = [];
    const emptyIdx: number[] = [];
    for (let k = 0; k < 5; k++) {
      if (isStone(i + k)) {
        stones++;
        cells.push({ x: line[i + k].x, y: line[i + k].y });
      } else if (isEmpty(i + k)) {
        empties++;
        emptyIdx.push(i + k);
      }
    }
    if (stones === 4 && empties === 1) {
      out.push({
        player,
        kind: 'four',
        cells,
        keySquares: emptyIdx.map((j) => ({ x: line[j].x, y: line[j].y })),
      });
    } else if (
      stones === 3 &&
      empties === 2 &&
      isEmpty(i) &&
      isEmpty(i + 4) &&
      isStone(i + 1) &&
      isStone(i + 2) &&
      isStone(i + 3)
    ) {
      out.push({
        player,
        kind: 'open-three',
        cells,
        keySquares: [
          { x: line[i].x, y: line[i].y },
          { x: line[i + 4].x, y: line[i + 4].y },
        ],
      });
    }
  }

  // Windows of 6: open four (_XXXX_) and split open three (_XX_X_, _X_XX_).
  for (let i = 0; i + 6 <= n; i++) {
    if (!isEmpty(i) || !isEmpty(i + 5)) continue;
    let stones = 0;
    const cells: Point[] = [];
    const emptyIdx: number[] = [i, i + 5];
    for (let k = 1; k <= 4; k++) {
      if (isStone(i + k)) {
        stones++;
        cells.push({ x: line[i + k].x, y: line[i + k].y });
      } else if (isEmpty(i + k)) {
        emptyIdx.push(i + k);
      }
    }
    if (stones === 4) {
      out.push({
        player,
        kind: 'open-four',
        cells,
        keySquares: [
          { x: line[i].x, y: line[i].y },
          { x: line[i + 5].x, y: line[i + 5].y },
        ],
      });
    } else if (stones === 3) {
      out.push({
        player,
        kind: 'open-three',
        cells,
        keySquares: emptyIdx.map((j) => ({ x: line[j].x, y: line[j].y })),
      });
    }
  }

  // Windows of 7: split fours (_XXX_X_, _XX_XX_, _X_XXX_).
  for (let i = 0; i + 7 <= n; i++) {
    if (!isEmpty(i) || !isEmpty(i + 6)) continue;
    let stones = 0;
    const cells: Point[] = [];
    let gap = -1;
    for (let k = 1; k <= 5; k++) {
      if (isStone(i + k)) {
        stones++;
        cells.push({ x: line[i + k].x, y: line[i + k].y });
      } else if (isEmpty(i + k)) {
        gap = i + k;
      }
    }
    if (stones === 4 && gap >= 0) {
      out.push({
        player,
        kind: 'four',
        cells,
        keySquares: [
          { x: line[gap].x, y: line[gap].y },
          { x: line[i].x, y: line[i].y },
          { x: line[i + 6].x, y: line[i + 6].y },
        ],
      });
    }
  }
}

/** Find all open fours, fours and open threes for both sides. */
export function findThreats(
  board: Int8Array | number[],
  size: number = SIZE,
): Threat[] {
  const found: Threat[] = [];
  for (const line of allLines(board, size)) {
    scanLine(line, 1, found);
    scanLine(line, -1, found);
  }
  // Deduplicate identical threats.
  const seen = new Set<string>();
  const deduped = found.filter((t) => {
    const k = `${t.player}|${t.kind}|${keyOf(t.cells)}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
  // A "four" fully inside an open four of the same player is the same threat
  // seen through a smaller window — drop it in favour of the open four.
  const openFourCells = new Set<string>();
  for (const t of deduped) {
    if (t.kind === 'open-four') {
      for (const c of t.cells) openFourCells.add(`${t.player}:${c.x},${c.y}`);
    }
  }
  const filtered = deduped.filter((t) => {
    if (t.kind !== 'four') return true;
    return !t.cells.every((c) => openFourCells.has(`${t.player}:${c.x},${c.y}`));
  });
  // Cap per kind so a crowded board doesn't flood the UI.
  const counts: Record<ThreatKind, number> = { 'open-four': 0, four: 0, 'open-three': 0 };
  const order: ThreatKind[] = ['open-four', 'four', 'open-three'];
  return filtered
    .sort((a, b) => order.indexOf(a.kind) - order.indexOf(b.kind))
    .filter((t) => {
      if (counts[t.kind] >= MAX_THREATS_PER_KIND) return false;
      counts[t.kind]++;
      return true;
    });
}

/**
 * Collapse threats into a board-square lookup for rendering: each key square
 * maps to the most urgent threat kind touching it.
 */
export function threatSquareMap(threats: Threat[]): Map<string, ThreatKind> {
  const rank: Record<ThreatKind, number> = { 'open-four': 0, four: 1, 'open-three': 2 };
  const map = new Map<string, ThreatKind>();
  for (const t of threats) {
    for (const sq of t.keySquares) {
      const k = `${sq.x},${sq.y}`;
      const prev = map.get(k);
      if (!prev || rank[t.kind] < rank[prev]) map.set(k, t.kind);
    }
  }
  return map;
}

export function countThreats(threats: Threat[], player: 1 | -1, kind: ThreatKind): number {
  return threats.filter((t) => t.player === player && t.kind === kind).length;
}

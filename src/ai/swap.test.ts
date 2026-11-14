import { describe, expect, it } from 'vitest';
import {
  SIZE,
  applyMove,
  checkWin,
  newGame,
  stateAtMove,
  type Point,
} from '../engine/gomoku';
import {
  attackCandidates,
  bestAttack,
  buildSwapOpeningBoard,
  chooseSwapColorAttack,
  chooseSwapColorTempo,
  createSeededRng,
  isSwapOpening,
  openingGap,
  placeBalancedOpening,
  scoreAttackV1,
  setupValue,
} from './swap';
import { SWAP_OPENINGS_200, SWAP_SCORE_VECTORS } from './swap-fixtures';

function pt(flat: number): Point {
  return { x: flat % SIZE, y: Math.floor(flat / SIZE) };
}

/** The 3 opening stones as engine moves (black, white, black), in order. */
function openingMoves(opening: [number, number, number]): Point[] {
  return opening.map(pt);
}

function openingBoard(opening: [number, number, number]): Int8Array {
  const [b1, w1, b2] = openingMoves(opening);
  return buildSwapOpeningBoard(b1, w1, b2);
}

/**
 * Scripted finish after any balanced opening: Black makes five along the
 * top row (y = 0) while White plays scattered stones far from the centre
 * (opening stones never leave Chebyshev distance 4 of the centre, so the
 * scripted points can never collide with them). Moves 4–13.
 */
const BLACK_WINS_AFTER_OPENING: Point[] = [
  { x: 14, y: 14 }, // 4  White
  { x: 0, y: 0 }, // 5  Black
  { x: 14, y: 0 }, // 6  White
  { x: 1, y: 0 }, // 7  Black
  { x: 0, y: 14 }, // 8  White
  { x: 2, y: 0 }, // 9  Black
  { x: 0, y: 7 }, // 10 White
  { x: 3, y: 0 }, // 11 Black
  { x: 14, y: 7 }, // 12 White
  { x: 4, y: 0 }, // 13 Black — five in a row, (0,0)–(4,0)
];

/** Scripted finish: White makes five along the top row. Moves 4–12. */
const WHITE_WINS_AFTER_OPENING: Point[] = [
  { x: 0, y: 0 }, // 4  White
  { x: 14, y: 14 }, // 5  Black
  { x: 1, y: 0 }, // 6  White
  { x: 14, y: 1 }, // 7  Black
  { x: 2, y: 0 }, // 8  White
  { x: 14, y: 2 }, // 9  Black
  { x: 3, y: 0 }, // 10 White
  { x: 14, y: 3 }, // 11 Black
  { x: 4, y: 0 }, // 12 White — five in a row, (0,0)–(4,0)
];

describe('v1 attack scorer (Python parity)', () => {
  it('matches Python score_attack on every legal move of 5 openings, both colours', () => {
    for (const fx of SWAP_SCORE_VECTORS) {
      const board = openingBoard(fx.opening);
      for (const [move, expected] of fx.black) {
        expect(scoreAttackV1(board, move % SIZE, Math.floor(move / SIZE), 1)).toBe(expected);
      }
      for (const [move, expected] of fx.white) {
        expect(scoreAttackV1(board, move % SIZE, Math.floor(move / SIZE), -1)).toBe(expected);
      }
    }
  });

  it('scores a broken four (XX_XX) as a four — the v0 kernel is blind to it', () => {
    // Black stones at (5,7), (6,7), (8,7), (9,7): playing (7,7) completes
    // XX_XX into a five. The horizontal direction scores WIN_SCORE and the
    // other three directions each contribute a lone open stone (2 points),
    // so the exact v1 value is 10,000,006 — the point is the five, which
    // a contiguous-run scorer (v0 kernel) cannot see at all.
    const board = new Int8Array(SIZE * SIZE);
    for (const x of [5, 6, 8, 9]) board[7 * SIZE + x] = 1;
    expect(scoreAttackV1(board, 7, 7, 1)).toBe(10_000_006);
    // Playing (4,7) extends the run to XXX_XX — a four in the window
    // (4..8) with the gap at (7,7). It must score at least four-value,
    // not the two-value a contiguous-run scorer would give.
    expect(scoreAttackV1(board, 4, 7, 1)).toBeGreaterThanOrEqual(20_000);
  });

  it('attack scores are colour-asymmetric (the total score is not)', () => {
    // On every fixture opening Black's best attack exceeds White's by
    // 77–78 — the asymmetry the colour choice turns on. A symmetric
    // attack+defence scorer would tie everywhere and make the choice
    // vacuous (see research/iterations/swap-rule.md).
    for (const fx of SWAP_OPENINGS_200) {
      const board = openingBoard(fx.opening);
      expect(bestAttack(board, 1) - bestAttack(board, -1)).toBeGreaterThan(0);
    }
  });
});

describe('tempo-aware colour judge (Python parity)', () => {
  it(
    'reproduces the Python tempo judge on all 200 openings: split 157 Black / 43 White',
    { timeout: 180_000 },
    () => {
      let black = 0;
      let white = 0;
      for (const fx of SWAP_OPENINGS_200) {
        const board = openingBoard(fx.opening);
        expect(isSwapOpening(board)).toBe(true);
        expect(setupValue(board, 1)).toBe(fx.tempoBlack);
        expect(setupValue(board, -1)).toBe(fx.tempoWhite);
        expect(bestAttack(board, 1)).toBe(fx.attackBestBlack);
        expect(bestAttack(board, -1)).toBe(fx.attackBestWhite);
        const [gap] = openingGap(board);
        expect(gap).toBe(Math.abs(fx.attackBestBlack - fx.attackBestWhite));
        const choice = chooseSwapColorTempo(board);
        expect(choice === 1 ? 'black' : 'white').toBe(fx.tempoChoice);
        if (choice === 1) black++;
        else white++;
      }
      expect([black, white]).toEqual([157, 43]);
    },
  );

  it('the plain attack-gap judge picks Black on all 200 openings (why the app uses tempo)', () => {
    let black = 0;
    for (const fx of SWAP_OPENINGS_200) {
      if (chooseSwapColorAttack(openingBoard(fx.opening)) === 1) black++;
    }
    expect(black).toBe(200);
  });

  it('ties go to White (White moves next — the tempo breaks the tie)', () => {
    // Opening 0 in the fixture is an exact tempo tie (4006 vs 4006).
    const fx = SWAP_OPENINGS_200[0];
    expect(fx.tempoBlack).toBe(fx.tempoWhite);
    expect(chooseSwapColorTempo(openingBoard(fx.opening))).toBe(-1);
  });

  it('throws on boards that are not a swap opening', () => {
    expect(() => chooseSwapColorTempo(new Int8Array(SIZE * SIZE))).toThrow(
      'swap-opening board',
    );
    const twoStones = new Int8Array(SIZE * SIZE);
    twoStones[7 * SIZE + 7] = 1;
    twoStones[7 * SIZE + 8] = -1;
    expect(() => chooseSwapColorTempo(twoStones)).toThrow('swap-opening board');
    const fourStones = openingBoard(SWAP_OPENINGS_200[0].opening);
    fourStones[0] = -1;
    expect(() => chooseSwapColorTempo(fourStones)).toThrow('swap-opening board');
  });
});

describe('balanced opening placement', () => {
  it('places centre first, White within Chebyshev 2, distinct stones', () => {
    for (let seed = 0; seed < 50; seed++) {
      const [b1, w1, b2] = placeBalancedOpening(createSeededRng(seed));
      expect(b1).toEqual({ x: 7, y: 7 });
      expect(Math.max(Math.abs(w1.y - 7), Math.abs(w1.x - 7))).toBeLessThanOrEqual(2);
      const flats = [b1, w1, b2].map((p) => p.y * SIZE + p.x);
      expect(new Set(flats).size).toBe(3);
      // Black's second stone is within Chebyshev 2 of a stone.
      const twoStone = new Int8Array(SIZE * SIZE);
      twoStone[b1.y * SIZE + b1.x] = 1;
      twoStone[w1.y * SIZE + w1.x] = -1;
      expect(attackCandidates(twoStone, 2)).toContain(b2.y * SIZE + b2.x);
    }
  });

  it('Black\'s second stone minimises (gap, bestBlack, index) over the nearby candidates', () => {
    for (let seed = 0; seed < 10; seed++) {
      const [b1, w1, b2] = placeBalancedOpening(createSeededRng(seed));
      const twoStone = new Int8Array(SIZE * SIZE);
      twoStone[b1.y * SIZE + b1.x] = 1;
      twoStone[w1.y * SIZE + w1.x] = -1;
      const chosenBoard = buildSwapOpeningBoard(b1, w1, b2);
      const [chosenGap, chosenBb] = openingGap(chosenBoard);
      const chosenKey: [number, number, number] = [chosenGap, chosenBb, b2.y * SIZE + b2.x];
      for (const cand of attackCandidates(twoStone, 2)) {
        const trial = new Int8Array(twoStone);
        trial[cand] = 1;
        const [gap, bb] = openingGap(trial);
        const key: [number, number, number] = [gap, bb, cand];
        expect(
          key[0] > chosenKey[0] ||
            (key[0] === chosenKey[0] &&
              (key[1] > chosenKey[1] ||
                (key[1] === chosenKey[1] && key[2] >= chosenKey[2]))),
        ).toBe(true);
      }
    }
  });

  it('varied openings: the White stone is drawn, not fixed', () => {
    const whites = new Set<number>();
    for (let seed = 0; seed < 50; seed++) {
      const [, w1] = placeBalancedOpening(createSeededRng(seed));
      whites.add(w1.y * SIZE + w1.x);
    }
    expect(whites.size).toBeGreaterThan(10);
  });

  it(
    'the tempo judge mixes on TS-generated openings (not always one side)',
    { timeout: 180_000 },
    () => {
      let black = 0;
      let white = 0;
      for (let seed = 0; seed < 100; seed++) {
        const [b1, w1, b2] = placeBalancedOpening(createSeededRng(1000 + seed));
        const board = buildSwapOpeningBoard(b1, w1, b2);
        if (chooseSwapColorTempo(board) === 1) black++;
        else white++;
      }
      // Python's split on its 200 openings is 157/43; the TS openings come
      // from a different RNG stream, so the exact split differs — but it
      // must mix, not degenerate to one side.
      expect(black).toBeGreaterThanOrEqual(10);
      expect(white).toBeGreaterThanOrEqual(10);
      expect(black + white).toBe(100);
    },
  );
});

describe('swap flows (engine level)', () => {
  it('human as opener: human places 3, AI (tempo judge) picks, White plays move 4, game proceeds to a win', () => {
    // Fixed human opening: centre, then White beside it, then Black diagonal.
    const opening: Point[] = [
      { x: 7, y: 7 },
      { x: 7, y: 8 },
      { x: 8, y: 8 },
    ];
    let s = newGame();
    for (const m of opening) s = applyMove(s, m.x, m.y);
    expect(s.moves).toHaveLength(3);
    expect(isSwapOpening(s.board)).toBe(true);
    // Move 4 is White's — the batch-1 bug gave Black move 4.
    expect(s.toMove).toBe(-1);
    const aiPick = chooseSwapColorTempo(s.board);
    expect(aiPick === 1 || aiPick === -1).toBe(true);
    // Play out the scripted finish: White moves 4th, Black wins.
    for (const m of BLACK_WINS_AFTER_OPENING) s = applyMove(s, m.x, m.y);
    expect(s.board[14 * SIZE + 14]).toBe(-1); // move 4 was White's stone
    expect(s.winner).toBe(1);
    expect(s.winningLine).toHaveLength(5);
    expect(s.moves).toHaveLength(13);
  });

  it('human as second player, AI opens: human picks Black — White still plays move 4', () => {
    const [b1, w1, b2] = placeBalancedOpening(createSeededRng(42));
    let s = stateAtMove([b1, w1, b2], 3);
    expect(s.toMove).toBe(-1);
    // Human picks Black (the choice is a side, not a move — the board and
    // turn order do not change).
    for (const m of BLACK_WINS_AFTER_OPENING) s = applyMove(s, m.x, m.y);
    expect(s.board[14 * SIZE + 14]).toBe(-1); // move 4 was White's stone
    expect(s.winner).toBe(1);
  });

  it('human as second player, AI opens: human picks White — White still plays move 4 and can win', () => {
    const [b1, w1, b2] = placeBalancedOpening(createSeededRng(7));
    let s = stateAtMove([b1, w1, b2], 3);
    expect(s.toMove).toBe(-1);
    for (const m of WHITE_WINS_AFTER_OPENING) s = applyMove(s, m.x, m.y);
    expect(s.board[0]).toBe(-1); // move 4 was White's stone, at (0,0)
    expect(s.winner).toBe(-1);
    expect(s.winningLine).toHaveLength(5);
  });

  it('two-player swap flow: player 1 places 3, player 2 picks, White plays move 4', () => {
    const opening: Point[] = [
      { x: 7, y: 7 },
      { x: 8, y: 7 },
      { x: 7, y: 6 },
    ];
    const s3 = stateAtMove(opening, 3);
    expect(s3.toMove).toBe(-1);
    let s = applyMove(s3, 0, 0); // player 2 (White) plays move 4
    expect(s.board[0]).toBe(-1);
    expect(s.toMove).toBe(1);
    expect(s.winner).toBe(0);
  });

  it('move 4 is White in every flow, across many generated openings', () => {
    for (let seed = 0; seed < 20; seed++) {
      const opening = placeBalancedOpening(createSeededRng(seed));
      const s = stateAtMove(opening, 3);
      expect(s.toMove).toBe(-1);
      const s4 = applyMove(s, 0, 0);
      expect(s4.board[0]).toBe(-1);
    }
  });

  it('win detection unchanged under swap: 5+ in a row wins, overline counts', () => {
    // Overline through the engine: a six cannot arise via applyMove (the
    // game ends at five), so build the board directly as in gomoku.test.ts
    // — the swap opening does not change checkWin.
    const board = new Int8Array(SIZE * SIZE);
    for (const x of [5, 6, 7, 8, 9, 10]) board[7 * SIZE + x] = 1;
    const win = checkWin(board, 9, 7);
    expect(win?.winner).toBe(1);
    expect(win?.line).toHaveLength(6);
  });

  it('freestyle smoke: Black opens, players alternate from move 1, no choice phase', () => {
    let s = newGame();
    expect(s.toMove).toBe(1);
    s = applyMove(s, 7, 7);
    expect(s.board[7 * SIZE + 7]).toBe(1);
    expect(s.toMove).toBe(-1);
    s = applyMove(s, 7, 8);
    expect(s.board[8 * SIZE + 7]).toBe(-1);
    expect(s.toMove).toBe(1);
    expect(s.winner).toBe(0);
  });
});

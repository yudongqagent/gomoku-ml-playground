import { describe, expect, it } from 'vitest';
import { stateAtMove, type Point } from '../engine/gomoku';
import { placeBalancedOpening, createSeededRng } from '../ai/swap';
import {
  gameReducer,
  getCanUndo,
  getSwapPhase,
  getUndoPlies,
  initialGameHookState,
  type GameHookAction,
  type GameHookState,
} from './useGomokuGame';

function drive(actions: GameHookAction[]): GameHookState {
  return actions.reduce(gameReducer, initialGameHookState);
}

function play(x: number, y: number): GameHookAction {
  return { type: 'play', point: { x, y } };
}

const OPENING: Point[] = [
  { x: 7, y: 7 },
  { x: 7, y: 8 },
  { x: 8, y: 8 },
];

function openingActions(opening: Point[] = OPENING): GameHookAction[] {
  return opening.map((p) => play(p.x, p.y));
}

/** Scripted finish after the opening: Black wins on move 13 (see swap.test.ts). */
const BLACK_WINS: Point[] = [
  { x: 14, y: 14 },
  { x: 0, y: 0 },
  { x: 14, y: 0 },
  { x: 1, y: 0 },
  { x: 0, y: 14 },
  { x: 2, y: 0 },
  { x: 0, y: 7 },
  { x: 3, y: 0 },
  { x: 14, y: 7 },
  { x: 4, y: 0 },
];

describe('swap phases', () => {
  it('defaults to swap rules, human opens, no choice yet', () => {
    const s = initialGameHookState;
    expect(s.rules).toBe('swap');
    expect(s.humanOpens).toBe(true);
    expect(s.swapChoice).toBeNull();
    expect(getSwapPhase(s)).toBe('opening');
    expect(getCanUndo(s)).toBe(false);
  });

  it('opening -> choice -> playing as the game progresses', () => {
    let s = drive(openingActions().slice(0, 2));
    expect(getSwapPhase(s)).toBe('opening');
    s = drive(openingActions());
    expect(getSwapPhase(s)).toBe('choice');
    s = drive([...openingActions(), { type: 'set-swap-choice', color: 1 }]);
    expect(getSwapPhase(s)).toBe('playing');
    expect(s.swapChoice).toBe(1);
  });

  it('no stones can be played during the choice phase', () => {
    const s = drive([...openingActions(), play(0, 0)]);
    expect(s.moves).toHaveLength(3);
    expect(getSwapPhase(s)).toBe('choice');
  });

  it('the choice is final: a second choice is ignored', () => {
    const s = drive([
      ...openingActions(),
      { type: 'set-swap-choice', color: 1 },
      { type: 'set-swap-choice', color: -1 },
    ]);
    expect(s.swapChoice).toBe(1);
  });

  it('the choice is ignored in freestyle mode', () => {
    const s = drive([
      { type: 'set-rules', rules: 'freestyle' },
      { type: 'set-swap-choice', color: 1 },
    ]);
    expect(s.swapChoice).toBeNull();
    expect(getSwapPhase(s)).toBeNull();
  });
});

describe('swap flows through the reducer', () => {
  it('human opener: human places 3, AI picks Black, human is White, White plays move 4, Black wins', () => {
    const s = drive([
      ...openingActions(),
      { type: 'set-swap-choice', color: 1 }, // AI (second player) picks Black
      ...BLACK_WINS.map((p) => play(p.x, p.y)),
    ]);
    expect(s.humanColor).toBe(-1); // human plays the colour the AI did not pick
    expect(s.swapChoice).toBe(1);
    const afterOpening = stateAtMove(s.moves, 3);
    expect(afterOpening.toMove).toBe(-1); // move 4 is White's
    const final = stateAtMove(s.moves, s.moves.length);
    expect(final.board[14 * 15 + 14]).toBe(-1); // move 4 was White's stone
    expect(final.winner).toBe(1);
    expect(final.winningLine).toHaveLength(5);
  });

  it('human second, picks Black: AI opens, human is Black, White still plays move 4', () => {
    const [b1, w1, b2] = placeBalancedOpening(createSeededRng(42));
    const s = drive([
      { type: 'set-human-opens', opens: false },
      play(b1.x, b1.y),
      play(w1.x, w1.y),
      play(b2.x, b2.y),
      { type: 'set-swap-choice', color: 1 }, // human picks Black
      ...BLACK_WINS.map((p) => play(p.x, p.y)),
    ]);
    expect(s.humanColor).toBe(1);
    expect(stateAtMove(s.moves, 3).toMove).toBe(-1);
    const final = stateAtMove(s.moves, s.moves.length);
    expect(final.board[14 * 15 + 14]).toBe(-1);
    expect(final.winner).toBe(1);
  });

  it('human second, picks White: AI opens, human is White and plays move 4', () => {
    const [b1, w1, b2] = placeBalancedOpening(createSeededRng(7));
    const s = drive([
      { type: 'set-human-opens', opens: false },
      play(b1.x, b1.y),
      play(w1.x, w1.y),
      play(b2.x, b2.y),
      { type: 'set-swap-choice', color: -1 }, // human picks White
      play(0, 0), // move 4, by the human as White
    ]);
    expect(s.humanColor).toBe(-1);
    const live = stateAtMove(s.moves, s.moves.length);
    expect(live.board[0]).toBe(-1);
    expect(live.toMove).toBe(1);
  });

  it('two-player: player 1 places 3, player 2 picks, White plays move 4', () => {
    const s = drive([
      { type: 'set-mode', mode: 'vs-human' },
      ...openingActions(),
      { type: 'set-swap-choice', color: -1 }, // player 2 picks White
      play(0, 0), // move 4, White
      play(14, 14), // move 5, Black
    ]);
    expect(getSwapPhase(s)).toBe('playing');
    const live = stateAtMove(s.moves, s.moves.length);
    expect(live.board[0]).toBe(-1);
    expect(live.board[14 * 15 + 14]).toBe(1);
    expect(live.toMove).toBe(-1);
  });

  it('freestyle smoke: no phases, Black opens, normal alternation', () => {
    const s = drive([
      { type: 'set-rules', rules: 'freestyle' },
      play(7, 7),
      play(7, 8),
    ]);
    expect(getSwapPhase(s)).toBeNull();
    const live = stateAtMove(s.moves, s.moves.length);
    expect(live.board[7 * 15 + 7]).toBe(1);
    expect(live.board[8 * 15 + 7]).toBe(-1);
    expect(live.toMove).toBe(1);
  });
});

describe('undo policy (swap mode)', () => {
  it('opening, human opener: undo removes 1 ply; redo replays it', () => {
    let s = drive(openingActions().slice(0, 2));
    expect(getCanUndo(s)).toBe(true);
    expect(getUndoPlies(s)).toBe(1);
    s = gameReducer(s, { type: 'undo', plies: getUndoPlies(s) });
    expect(s.moves).toHaveLength(1);
    expect(getSwapPhase(s)).toBe('opening');
    s = gameReducer(s, { type: 'redo' });
    expect(s.moves).toHaveLength(2);
    expect(s.moves[1]).toEqual({ x: 7, y: 8 });
  });

  it('opening, AI opener: undo is disabled (the human placed nothing)', () => {
    const s = drive([
      { type: 'set-human-opens', opens: false },
      ...openingActions().slice(0, 2), // AI placing, staggered
    ]);
    expect(getCanUndo(s)).toBe(false);
    expect(getUndoPlies(s)).toBe(0);
  });

  it('choice phase: undo is disabled — the opening is committed', () => {
    const s = drive(openingActions());
    expect(getSwapPhase(s)).toBe('choice');
    expect(getCanUndo(s)).toBe(false);
    expect(getUndoPlies(s)).toBe(0);
    const after = gameReducer(s, { type: 'undo', plies: 0 });
    expect(after.moves).toHaveLength(3);
  });

  it('playing, vs AI: undo removes 2 plies and is floored at 3 moves; the choice stands', () => {
    let s = drive([
      ...openingActions(),
      { type: 'set-swap-choice', color: 1 },
      play(14, 14), // move 4
      play(0, 0), // move 5
    ]);
    expect(s.moves).toHaveLength(5);
    expect(getUndoPlies(s)).toBe(2);
    s = gameReducer(s, { type: 'undo', plies: getUndoPlies(s) });
    expect(s.moves).toHaveLength(3);
    expect(s.swapChoice).toBe(1); // the choice survives the undo
    expect(getSwapPhase(s)).toBe('playing');
    expect(getCanUndo(s)).toBe(false); // floored: nothing post-opening to undo
    s = gameReducer(s, { type: 'redo' });
    expect(s.moves).toHaveLength(5);
  });

  it('playing, vs AI, at 4 moves: undo removes just 1 ply (the floor)', () => {
    let s = drive([
      ...openingActions(),
      { type: 'set-swap-choice', color: -1 },
      play(14, 14), // move 4 (AI's, if the human is Black)
    ]);
    expect(getUndoPlies(s)).toBe(1);
    s = gameReducer(s, { type: 'undo', plies: getUndoPlies(s) });
    expect(s.moves).toHaveLength(3);
    expect(s.swapChoice).toBe(-1);
    expect(getCanUndo(s)).toBe(false);
  });

  it('playing, vs human: undo removes 1 ply, floored at 3 moves', () => {
    let s = drive([
      { type: 'set-mode', mode: 'vs-human' },
      ...openingActions(),
      { type: 'set-swap-choice', color: 1 },
      play(14, 14),
      play(0, 0),
    ]);
    expect(getUndoPlies(s)).toBe(1);
    s = gameReducer(s, { type: 'undo', plies: getUndoPlies(s) });
    expect(s.moves).toHaveLength(4);
    s = gameReducer(s, { type: 'undo', plies: getUndoPlies(s) });
    expect(s.moves).toHaveLength(3);
    expect(getCanUndo(s)).toBe(false);
  });

  it('freestyle undo is unchanged: 2 plies vs AI, no floor', () => {
    let s = drive([
      { type: 'set-rules', rules: 'freestyle' },
      play(7, 7),
      play(7, 8),
      play(8, 8),
    ]);
    expect(getCanUndo(s)).toBe(true);
    expect(getUndoPlies(s)).toBe(2);
    s = gameReducer(s, { type: 'undo', plies: getUndoPlies(s) });
    expect(s.moves).toHaveLength(1);
  });
});

describe('mode / rules switches and loading', () => {
  it('set-rules, set-human-opens and set-mode start a fresh game', () => {
    const s = drive([
      ...openingActions(),
      { type: 'set-swap-choice', color: 1 },
      play(14, 14),
      { type: 'set-rules', rules: 'freestyle' },
    ]);
    expect(s.moves).toEqual([]);
    expect(s.swapChoice).toBeNull();
    expect(s.rules).toBe('freestyle');
    expect(s.gameId).toBe(1);

    const s2 = drive([
      ...openingActions(),
      { type: 'set-human-opens', opens: false },
    ]);
    expect(s2.moves).toEqual([]);
    expect(s2.humanOpens).toBe(false);
    expect(s2.gameId).toBe(1);
  });

  it('loading a recorded game switches to freestyle (recorded games have no swap opening)', () => {
    const s = drive([
      {
        type: 'load-moves',
        moves: [
          { x: 7, y: 7 },
          { x: 7, y: 8 },
          { x: 8, y: 8 },
          { x: 8, y: 7 },
          { x: 9, y: 7 },
        ],
      },
    ]);
    expect(s.rules).toBe('freestyle');
    expect(getSwapPhase(s)).toBeNull();
    expect(s.moves).toHaveLength(5);
    expect(s.swapChoice).toBeNull();
  });
});

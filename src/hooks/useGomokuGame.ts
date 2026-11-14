// Game-state management for the Play view: plain React hooks + useReducer.
// The move list is the source of truth; engine states are derived from it via
// stateAtMove, so history navigation, undo/redo and loading a recorded game
// all share one code path. No captures in Gomoku — stones are never removed.
//
// Swap rule (standard swap / pie rule — NOT Swap2; see src/ai/swap.ts):
// the opener places 3 stones (black, white, black), the second player picks
// a side, and White plays move 4. The engine needs no changes for this: the
// opening stones are just moves 1–3 through applyMove, and after move 3 the
// engine already has White to move. The phases live here:
//   opening — fewer than 3 stones placed (the opener is placing them);
//   choice  — 3 stones placed, the second player has not picked a side yet;
//   playing — the side is picked (or freestyle rules); normal play.
// In freestyle mode there are no phases (getSwapPhase returns null) and the
// game behaves exactly as before.

import { useCallback, useMemo, useReducer } from 'react';
import {
  applyMove,
  isLegal,
  newGame,
  opponent,
  stateAtMove,
  type GameState,
  type Player,
  type Point,
} from '../engine/gomoku';

/**
 * Replay a move list and keep only the longest legal prefix. Recorded games
 * sometimes include moves after the game was already decided (or a duplicate
 * point); stateAtMove throws on those, so loading must sanitize first.
 */
export function legalPrefix(moves: Point[]): Point[] {
  let state = newGame();
  const out: Point[] = [];
  for (const m of moves.slice(0, 225)) {
    try {
      state = applyMove(state, m.x, m.y);
    } catch {
      break;
    }
    out.push(m);
  }
  return out;
}

export type GameMode = 'vs-ai' | 'vs-human' | 'ai-vs-ai';

/** Swap (official rule, the default) or freestyle (Black opens, no choice). */
export type RulesMode = 'swap' | 'freestyle';

/** Swap-rule phase, or null in freestyle mode (no phases). */
export type SwapPhase = 'opening' | 'choice' | 'playing';

export interface GameHookState {
  moves: Point[];
  /** Undone moves, most-recently-undone first. */
  redoStack: Point[];
  /** Sizes of the undone batches (undo removes up to 2 plies vs AI). */
  redoBatches: number[];
  /** Ply being viewed, or null for the live position. */
  viewPly: number | null;
  mode: GameMode;
  humanColor: Player;
  flipped: boolean;
  /** Increments on every new game / loaded game; invalidates in-flight AI. */
  gameId: number;
  rules: RulesMode;
  /** Vs AI, swap mode: true when the human places the 3 opening stones. */
  humanOpens: boolean;
  /**
   * The colour the second player picked, or null until the choice is made.
   * In vs-ai mode the choice also sets humanColor (see set-swap-choice).
   */
  swapChoice: Player | null;
}

export type GameHookAction =
  | { type: 'play'; point: Point }
  | { type: 'undo'; plies: number }
  | { type: 'redo' }
  | { type: 'new-game' }
  | { type: 'load-moves'; moves: Point[] }
  | { type: 'view'; ply: number | null }
  | { type: 'set-mode'; mode: GameMode }
  | { type: 'set-human-color'; color: Player }
  | { type: 'toggle-flip' }
  | { type: 'set-rules'; rules: RulesMode }
  | { type: 'set-human-opens'; opens: boolean }
  | { type: 'set-swap-choice'; color: Player };

export const initialGameHookState: GameHookState = {
  moves: [],
  redoStack: [],
  redoBatches: [],
  viewPly: null,
  mode: 'vs-ai',
  humanColor: 1,
  flipped: false,
  gameId: 0,
  rules: 'swap',
  humanOpens: true,
  swapChoice: null,
};

/** The swap-rule phase of a state, or null in freestyle mode. */
export function getSwapPhase(state: GameHookState): SwapPhase | null {
  if (state.rules !== 'swap') return null;
  if (state.moves.length < 3) return 'opening';
  if (state.moves.length === 3 && state.swapChoice === null) return 'choice';
  return 'playing';
}

/** True when a human (not the AI) is placing the opening stones. */
export function humanIsOpener(state: GameHookState): boolean {
  return state.mode === 'vs-human' || (state.mode === 'vs-ai' && state.humanOpens);
}

// --- Undo policy (swap mode) ---------------------------------------------
// Decided 2026-11-14; see the tests in src/hooks/useGomokuGame.swap.test.ts.
// The opening + choice are final once the 3 stones are placed: undo never
// removes an opening stone or invalidates the colour choice, because the
// choice was made about those exact 3 stones.
//
// - Opening phase, human opener: undo removes 1 ply (the last placed
//   stone) — a mis-tap while placing should be cheap to fix. (When the AI
//   opens, the human never places, so there is nothing to undo.)
// - Choice phase: undo is DISABLED. The opening is committed and the
//   choice is pending; there is no move to take back, and un-placing a
//   stone would void the choice. New Game is the remedy for a bad opening.
// - Playing phase: undo works normally (2 plies vs AI, 1 ply vs human)
//   but is floored at 3 moves — it never crosses back into the opening.
//   Vs AI, undoing from 4 moves removes just 1 ply (the floor); the AI
//   then replays move 4 and the choice stands.
// Freestyle mode is unchanged: undo removes 2 plies vs AI, 1 otherwise,
// with no floor.

export function getCanUndo(state: GameHookState): boolean {
  const phase = getSwapPhase(state);
  if (phase === null) return state.moves.length > 0;
  if (phase === 'opening') return humanIsOpener(state) && state.moves.length > 0;
  if (phase === 'choice') return false;
  return state.moves.length > 3;
}

/** Plies an undo removes in this state (0 when undo is disabled). */
export function getUndoPlies(state: GameHookState): number {
  if (!getCanUndo(state)) return 0;
  const phase = getSwapPhase(state);
  if (phase === 'opening') return 1;
  if (phase === 'playing') {
    // Floored at 3 moves: never remove an opening stone or void the choice.
    return state.mode === 'vs-ai'
      ? Math.min(2, state.moves.length - 3)
      : 1;
  }
  // Freestyle: take back the pair (human move + AI reply) vs AI so it is
  // the human's turn again; otherwise a single ply.
  return state.mode === 'vs-ai' ? 2 : 1;
}

function freshGame(state: GameHookState): GameHookState {
  return {
    ...state,
    moves: [],
    redoStack: [],
    redoBatches: [],
    viewPly: null,
    swapChoice: null,
    gameId: state.gameId + 1,
  };
}

export function gameReducer(state: GameHookState, action: GameHookAction): GameHookState {
  switch (action.type) {
    case 'play':
      // During the swap choice phase the board takes no stones — the second
      // player picks a side instead (set-swap-choice). playMove guards this
      // too; the reducer enforces it so no path can play through the choice.
      if (getSwapPhase(state) === 'choice') return state;
      return {
        ...state,
        moves: [...state.moves, action.point],
        redoStack: [],
        redoBatches: [],
        viewPly: null,
      };
    case 'undo': {
      const n = Math.min(action.plies, state.moves.length);
      if (n <= 0) return state;
      const undone = state.moves.slice(state.moves.length - n).reverse();
      return {
        ...state,
        moves: state.moves.slice(0, state.moves.length - n),
        redoStack: [...undone, ...state.redoStack],
        redoBatches: [n, ...state.redoBatches],
        viewPly: null,
      };
    }
    case 'redo': {
      const [batch, ...restBatches] = state.redoBatches;
      if (batch === undefined) return state;
      const n = Math.min(batch, state.redoStack.length);
      if (n <= 0) return state;
      const replay = state.redoStack.slice(0, n).reverse();
      return {
        ...state,
        moves: [...state.moves, ...replay],
        redoStack: state.redoStack.slice(n),
        redoBatches: restBatches,
        viewPly: null,
      };
    }
    case 'new-game':
      return freshGame(state);
    case 'load-moves':
      // Recorded games (trainer / game viewer) are freestyle games: they
      // have no swap opening or colour choice, so loading one switches the
      // rules to freestyle rather than leaving the game stuck in a choice
      // phase it can never resolve.
      return {
        ...freshGame(state),
        moves: action.moves,
        rules: 'freestyle',
      };
    case 'view':
      return { ...state, viewPly: action.ply };
    case 'set-mode':
      // Switching mode starts a fresh game: mixing modes mid-game would leave
      // the AI thinking about a position it should no longer move in.
      return freshGame({ ...state, mode: action.mode });
    case 'set-human-color':
      return freshGame({ ...state, humanColor: action.color });
    case 'toggle-flip':
      return { ...state, flipped: !state.flipped };
    case 'set-rules':
      // Switching rules starts a fresh game: a swap game and a freestyle
      // game disagree about what the first 3 moves mean.
      return freshGame({ ...state, rules: action.rules });
    case 'set-human-opens':
      return freshGame({ ...state, humanOpens: action.opens });
    case 'set-swap-choice': {
      if (state.rules !== 'swap' || state.swapChoice !== null) return state;
      // In vs-ai mode the choice fixes the human's colour: the human plays
      // the colour they picked, or the colour the AI did NOT pick.
      const humanColor =
        state.mode === 'vs-ai'
          ? state.humanOpens
            ? opponent(action.color)
            : action.color
          : state.humanColor;
      return {
        ...state,
        swapChoice: action.color,
        humanColor,
        redoStack: [],
        redoBatches: [],
      };
    }
    default:
      return state;
  }
}

export interface GomokuGame {
  moves: Point[];
  liveState: GameState;
  viewState: GameState;
  /** Ply currently displayed (== moves.length when live). */
  viewPly: number;
  isLive: boolean;
  mode: GameMode;
  humanColor: Player;
  flipped: boolean;
  gameId: number;
  rules: RulesMode;
  humanOpens: boolean;
  swapChoice: Player | null;
  swapPhase: SwapPhase | null;
  canUndo: boolean;
  canRedo: boolean;
  playMove: (x: number, y: number) => boolean;
  undo: () => void;
  redo: () => void;
  newGame: () => void;
  loadMoves: (moves: Point[]) => void;
  viewPlyAt: (ply: number) => void;
  goLive: () => void;
  setMode: (mode: GameMode) => void;
  setHumanColor: (color: Player) => void;
  toggleFlip: () => void;
  setRules: (rules: RulesMode) => void;
  setHumanOpens: (opens: boolean) => void;
  chooseSwapColor: (color: Player) => void;
}

export function useGomokuGame(): GomokuGame {
  const [state, dispatch] = useReducer(gameReducer, initialGameHookState);

  const liveState = useMemo(
    () => stateAtMove(state.moves, state.moves.length),
    [state.moves],
  );
  const viewPly = state.viewPly ?? state.moves.length;
  const viewState = useMemo(
    () => (viewPly === state.moves.length ? liveState : stateAtMove(state.moves, viewPly)),
    [viewPly, state.moves, liveState],
  );
  const swapPhase = getSwapPhase(state);

  const playMove = useCallback(
    (x: number, y: number): boolean => {
      // Only the live position accepts moves; the board is non-interactive
      // while browsing history, but guard here too.
      if (state.viewPly !== null) return false;
      if (liveState.winner !== 0) return false;
      // During the swap choice phase the board takes no stones — the
      // second player picks a side instead (see chooseSwapColor).
      if (swapPhase === 'choice') return false;
      let legal = false;
      try {
        legal = isLegal(liveState, x, y);
      } catch {
        return false;
      }
      if (!legal) return false;
      dispatch({ type: 'play', point: { x, y } });
      return true;
    },
    [liveState, state.viewPly, swapPhase],
  );

  const undo = useCallback(() => {
    const plies = getUndoPlies(state);
    if (plies <= 0) return;
    dispatch({ type: 'undo', plies });
  }, [state]);

  const redo = useCallback(() => dispatch({ type: 'redo' }), []);
  const newGame = useCallback(() => dispatch({ type: 'new-game' }), []);
  const loadMoves = useCallback((moves: Point[]) => {
    dispatch({ type: 'load-moves', moves: legalPrefix(moves) });
  }, []);
  const viewPlyAt = useCallback(
    (ply: number) => {
      const clamped = Math.max(0, Math.min(ply, state.moves.length));
      dispatch({ type: 'view', ply: clamped === state.moves.length ? null : clamped });
    },
    [state.moves.length],
  );
  const goLive = useCallback(() => dispatch({ type: 'view', ply: null }), []);
  const setMode = useCallback((mode: GameMode) => dispatch({ type: 'set-mode', mode }), []);
  const setHumanColor = useCallback(
    (color: Player) => dispatch({ type: 'set-human-color', color }),
    [],
  );
  const toggleFlip = useCallback(() => dispatch({ type: 'toggle-flip' }), []);
  const setRules = useCallback(
    (rules: RulesMode) => dispatch({ type: 'set-rules', rules }),
    [],
  );
  const setHumanOpens = useCallback(
    (opens: boolean) => dispatch({ type: 'set-human-opens', opens }),
    [],
  );
  const chooseSwapColor = useCallback(
    (color: Player) => dispatch({ type: 'set-swap-choice', color }),
    [],
  );

  return {
    moves: state.moves,
    liveState,
    viewState,
    viewPly,
    isLive: state.viewPly === null,
    mode: state.mode,
    humanColor: state.humanColor,
    flipped: state.flipped,
    gameId: state.gameId,
    rules: state.rules,
    humanOpens: state.humanOpens,
    swapChoice: state.swapChoice,
    swapPhase,
    canUndo: getCanUndo(state),
    canRedo: state.redoBatches.length > 0,
    playMove,
    undo,
    redo,
    newGame,
    loadMoves,
    viewPlyAt,
    goLive,
    setMode,
    setHumanColor,
    toggleFlip,
    setRules,
    setHumanOpens,
    chooseSwapColor,
  };
}

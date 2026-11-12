// Game-state management for the Play view: plain React hooks + useReducer.
// The move list is the source of truth; engine states are derived from it via
// stateAtMove, so history navigation, undo/redo and loading a recorded game
// all share one code path. No captures in Gomoku — stones are never removed.

import { useCallback, useMemo, useReducer } from 'react';
import {
  applyMove,
  isLegal,
  newGame,
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

interface GameHookState {
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
}

type Action =
  | { type: 'play'; point: Point }
  | { type: 'undo'; plies: number }
  | { type: 'redo' }
  | { type: 'new-game' }
  | { type: 'load-moves'; moves: Point[] }
  | { type: 'view'; ply: number | null }
  | { type: 'set-mode'; mode: GameMode }
  | { type: 'set-human-color'; color: Player }
  | { type: 'toggle-flip' };

const initialState: GameHookState = {
  moves: [],
  redoStack: [],
  redoBatches: [],
  viewPly: null,
  mode: 'vs-ai',
  humanColor: 1,
  flipped: false,
  gameId: 0,
};

function reducer(state: GameHookState, action: Action): GameHookState {
  switch (action.type) {
    case 'play':
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
      return {
        ...state,
        moves: [],
        redoStack: [],
        redoBatches: [],
        viewPly: null,
        gameId: state.gameId + 1,
      };
    case 'load-moves':
      return {
        ...state,
        moves: action.moves,
        redoStack: [],
        redoBatches: [],
        viewPly: null,
        gameId: state.gameId + 1,
      };
    case 'view':
      return { ...state, viewPly: action.ply };
    case 'set-mode':
      // Switching mode starts a fresh game: mixing modes mid-game would leave
      // the AI thinking about a position it should no longer move in.
      return {
        ...state,
        mode: action.mode,
        moves: [],
        redoStack: [],
        redoBatches: [],
        viewPly: null,
        gameId: state.gameId + 1,
      };
    case 'set-human-color':
      return {
        ...state,
        humanColor: action.color,
        moves: [],
        redoStack: [],
        redoBatches: [],
        viewPly: null,
        gameId: state.gameId + 1,
      };
    case 'toggle-flip':
      return { ...state, flipped: !state.flipped };
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
}

export function useGomokuGame(): GomokuGame {
  const [state, dispatch] = useReducer(reducer, initialState);

  const liveState = useMemo(
    () => stateAtMove(state.moves, state.moves.length),
    [state.moves],
  );
  const viewPly = state.viewPly ?? state.moves.length;
  const viewState = useMemo(
    () => (viewPly === state.moves.length ? liveState : stateAtMove(state.moves, viewPly)),
    [viewPly, state.moves, liveState],
  );

  const playMove = useCallback(
    (x: number, y: number): boolean => {
      // Only the live position accepts moves; the board is non-interactive
      // while browsing history, but guard here too.
      if (state.viewPly !== null) return false;
      if (liveState.winner !== 0) return false;
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
    [liveState, state.viewPly],
  );

  const undo = useCallback(() => {
    // Vs AI: take back the pair (human move + AI reply) so it is the human's
    // turn again; otherwise a single ply.
    const plies = state.mode === 'vs-ai' ? 2 : 1;
    dispatch({ type: 'undo', plies });
  }, [state.mode]);

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
    canUndo: state.moves.length > 0,
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
  };
}

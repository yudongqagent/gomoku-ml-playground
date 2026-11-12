import type { Point } from '../engine/gomoku';

export type LevelId = 'beginner' | 'intermediate' | 'advanced' | 'expert' | 'master';

export interface AiOptions {
  /** MCTS simulation budget (Expert / Master). */
  sims?: number;
  /** Time budget in milliseconds (Advanced / Expert / Master). */
  timeMs?: number;
  /** Move-selection temperature for MCTS levels (0 = always pick the most-visited move). */
  temperature?: number;
  /** URL of the ONNX policy+value model (Master only). Omit for MCTS fallback. */
  modelUrl?: string;
}

export interface AiMoveRequest {
  /** Board as a flat array of 225 numbers (row-major, y*15+x): 1 = Black, -1 = White, 0 = empty. */
  board: number[];
  toMove: 1 | -1;
  level: LevelId;
  options?: AiOptions;
  /**
   * Full move history in order (Black first). Used to build the ONNX model's
   * history planes (last move, positions 2 and 4 plies ago). Optional: when
   * absent, those planes are zeros.
   */
  history?: Point[];
}

export interface TopMove {
  move: Point;
  /** Estimated win rate for the side to move, in [0, 1]. */
  winRate: number;
  /** MCTS visit count, when the level is MCTS-based. */
  visits?: number;
}

export interface AiMoveResult {
  move: Point;
  /** Estimated win rate for the side to move after playing `move`, in [0, 1]. */
  winRate: number;
  /** Up to 5 best moves, sorted by win rate descending. */
  topMoves: TopMove[];
  /** Search nodes / simulations performed. */
  nodes: number;
  /** Wall-clock milliseconds spent. */
  ms: number;
  /** Search depth reached (alpha-beta levels) or max MCTS tree depth. */
  depth?: number;
  /**
   * Which engine actually produced the move, e.g. 'heuristic', 'alphabeta',
   * 'alphabeta-vct', 'mcts', 'onnx-webgpu', 'onnx-wasm', 'mcts-fallback',
   * 'forced-win', 'forced-block'. The UI can show this honestly instead of
   * claiming a model was used when it was not.
   */
  backend?: string;
}

export interface AnalysisResult {
  bestMove: Point;
  /** Estimated win rate for the side to move after playing `bestMove`, in [0, 1]. */
  winRate: number;
  /** Up to 5 best moves, sorted by win rate descending. */
  topMoves: TopMove[];
  nodes: number;
  ms: number;
  depth?: number;
  backend?: string;
}

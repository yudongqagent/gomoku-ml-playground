/**
 * Public AI API surface. The web-app workstream imports from here (or from
 * the individual modules — both work).
 */

export { chooseMove, analyzePosition } from './engine';
export { LEVEL_META, LEVEL_ORDER, type LevelMeta } from './levels';
export type {
  AiMoveRequest,
  AiMoveResult,
  AiOptions,
  AnalysisResult,
  LevelId,
  TopMove,
} from './types';

// Lower-level pieces, exported for tests and for future workstreams
// (e.g. a trainer that needs the pattern eval or the MCTS core).
export {
  WIN_SCORE,
  EVAL_SCALE,
  evaluateBoard,
  scoreMove,
  candidateMoves,
  orderedCandidates,
  findWinningMove,
  findWinningMoves,
  winRateFromEval,
} from './patterns';
export { searchMove, findForcedWin, type SearchOptions, type SearchResult } from './search';
export {
  runMcts,
  chooseMctsMove,
  heuristicPolicyValue,
  type MctsOptions,
  type MctsResult,
  type PolicyValue,
  type PolicyValueFn,
} from './mcts';
export { chooseBeginnerMove, type BeginnerOptions } from './beginner';
export { chooseMasterMove, type MasterOptions, type MasterResult } from './alphazero';
export {
  buildInputPlanes,
  loadModel,
  predict,
  getBackend,
  type NetOutput,
  type OnnxBackend,
} from './onnxModel';

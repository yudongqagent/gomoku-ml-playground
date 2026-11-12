// Shared UI-side types. These mirror the contracts in the AI workstream
// (src/ai/levels.ts, src/hooks/useAiWorker.ts) structurally, so the UI keeps
// working even if that workstream's exported type names differ slightly.

export interface PointLike {
  x: number;
  y: number;
}

export interface AiOptions {
  sims: number;
  timeMs: number;
  temperature: number;
}

/** One candidate move returned by the AI / analysis. */
export interface ScoredMove {
  move: PointLike;
  winRate: number;
  visits?: number;
}

/** Normalised result of a move request or a position analysis. */
export interface AiMoveResult {
  move: PointLike;
  winRate: number;
  topMoves: ScoredMove[];
  nodes: number;
  ms: number;
  depth?: number;
  backend?: string;
}

/** Per-move performance readout shown in the AI panel. */
export interface AiPerf {
  nodes: number;
  ms: number;
  nps: number;
  backend?: string;
  depth?: number;
}

/** One point on the win-rate-over-game graph (Black's perspective). */
export interface WinRatePoint {
  ply: number;
  blackWinRate: number;
}

/** A neural model selectable in the UI, from either source. */
export interface ModelOption {
  id: string;
  name: string;
  /** URL passed to the AI worker as options.modelUrl. */
  url: string;
  source: 'bundled' | 'server';
  description?: string;
  trainedGames?: number;
  evalWinRate?: number;
  architecture?: string;
  createdAt?: string;
  formats?: string[];
  notes?: string;
}

/**
 * Normalise a raw worker result defensively: the move-request shape carries
 * `move`, the analysis shape carries `bestMove`, and either may be missing
 * fields. Returns null when there is no usable move at all.
 */
export function normalizeAiResult(raw: unknown): AiMoveResult | null {
  if (!raw || typeof raw !== 'object') return null;
  const r = raw as Record<string, unknown>;
  const moveRaw = (r.move ?? r.bestMove) as { x?: unknown; y?: unknown } | undefined;
  if (!moveRaw || typeof moveRaw.x !== 'number' || typeof moveRaw.y !== 'number') return null;
  const topMovesRaw = Array.isArray(r.topMoves) ? r.topMoves : [];
  const topMoves: ScoredMove[] = topMovesRaw
    .map((t) => {
      if (!t || typeof t !== 'object') return null;
      const tm = t as Record<string, unknown>;
      const m = tm.move as { x?: unknown; y?: unknown } | undefined;
      if (!m || typeof m.x !== 'number' || typeof m.y !== 'number') return null;
      return {
        move: { x: m.x, y: m.y },
        winRate: typeof tm.winRate === 'number' ? tm.winRate : 0.5,
        visits: typeof tm.visits === 'number' ? tm.visits : undefined,
      } as ScoredMove;
    })
    .filter((t): t is ScoredMove => t !== null);
  return {
    move: { x: moveRaw.x, y: moveRaw.y },
    winRate: typeof r.winRate === 'number' ? r.winRate : 0.5,
    topMoves,
    nodes: typeof r.nodes === 'number' ? r.nodes : 0,
    ms: typeof r.ms === 'number' ? r.ms : 0,
    depth: typeof r.depth === 'number' ? r.depth : undefined,
    backend: typeof r.backend === 'string' ? r.backend : undefined,
  };
}

export function perfFromResult(r: AiMoveResult): AiPerf {
  return {
    nodes: r.nodes,
    ms: r.ms,
    nps: r.ms > 0 ? Math.round((r.nodes / r.ms) * 1000) : 0,
    backend: r.backend,
    depth: r.depth,
  };
}

/** Format a 0..1 win rate as a percentage string. */
export function formatWinRate(wr: number): string {
  if (!Number.isFinite(wr)) return '—';
  return `${(wr * 100).toFixed(1)}%`;
}

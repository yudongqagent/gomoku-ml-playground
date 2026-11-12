// Client for the training server (FastAPI, built by the server workstream).
// All Trainer API calls go through this module. Every function is defensive
// about response shapes: the server is a parallel workstream, so unknown or
// missing fields degrade to empty lists / nulls instead of throwing.
//
// API base: VITE_API_URL, default http://localhost:8000.

import { useEffect, useRef, useState } from 'react';

export const TRAINER_API_BASE: string = (
  (import.meta.env.VITE_API_URL as string | undefined) || 'http://localhost:8000'
).replace(/\/$/, '');

/** Shown when the server is unreachable. Verified against
 * server/README.md (quickstart: venv + `.venv/bin/uvicorn`). */
export const TRAINER_START_COMMAND =
  'cd server && .venv/bin/uvicorn app.main:app --reload --port 8000';

export type JobType = 'selfplay' | 'train' | 'full';
/** 'paused' = user-paused (resumable); 'interrupted' = the server restarted
 * mid-run (resumable from the latest checkpoint). Both are new in the
 * extended contract; older servers never send them. */
export type JobStatus =
  | 'queued'
  | 'running'
  | 'paused'
  | 'interrupted'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface ArchParamSchema {
  type: 'int' | 'float' | 'str' | 'select';
  default?: unknown;
  min?: number;
  max?: number;
  options?: string[];
  description?: string;
}

/** One eval-ladder entry: wins/losses/draws vs a named scripted opponent. */
export interface EvalEntry {
  wins: number;
  losses: number;
  draws: number;
  elo_diff_vs_opponent?: number | null;
}

/** Eval results keyed by opponent name, e.g. { heuristic: { wins, ... } }. */
export type JobEval = Record<string, EvalEntry>;

/**
 * Win rate (0..1) from an eval dict for the opponent whose name matches
 * `match` (case-insensitive substring), falling back to the first entry.
 * Rate = wins / (wins + losses + draws), the plain share of games won.
 * Returns undefined when there is no entry or no games were played.
 */
export function evalWinRate(
  evalObj: JobEval | Record<string, unknown> | null | undefined,
  match = 'heuristic',
): number | undefined {
  if (!evalObj || typeof evalObj !== 'object') return undefined;
  const entries = Object.entries(evalObj) as [string, unknown][];
  const found =
    entries.find(([name]) => name.toLowerCase().includes(match.toLowerCase())) ??
    entries[0];
  if (!found) return undefined;
  const e = found[1] as Record<string, unknown>;
  if (typeof e !== 'object' || e === null) return undefined;
  const w = typeof e.wins === 'number' ? e.wins : 0;
  const l = typeof e.losses === 'number' ? e.losses : 0;
  const d = typeof e.draws === 'number' ? e.draws : 0;
  const total = w + l + d;
  return total > 0 ? w / total : undefined;
}

export interface Architecture {
  id: string;
  name: string;
  description: string;
  params: Record<string, ArchParamSchema>;
}

export interface Hyperparameters {
  games: number;
  epochs: number;
  mcts_sims: number;
  lr: number;
  batch_size: number;
  temperature: number;
}

export interface CreateJobRequest {
  type: JobType;
  name: string;
  architecture: string;
  arch_params: Record<string, unknown>;
  hyperparameters: Hyperparameters;
  base_checkpoint: string | null;
  /** Run preset id (from GET /api/presets). The server applies the preset's
   * config file; the fields above act as overrides. Omit when no preset. */
  preset?: string;
}

export interface JobSummary {
  id: string;
  name: string;
  type: JobType;
  status: JobStatus;
  progress: number;
  created_at?: string;
  architecture?: string;
  games_generated?: number;
  /** Current pipeline phase, e.g. 'selfplay' | 'train' | 'eval' | 'export'. */
  phase?: string;
}

export interface MetricPoint {
  step: number;
  loss: number;
  epoch?: number;
  /** Per-component losses, when the server reports them (it does since the
   * first server build: policy_loss / value_loss per epoch). Any may be
   * absent on a given point — parse defensively. */
  policy_loss?: number;
  value_loss?: number;
  val_loss?: number;
  games_generated?: number;
  timestamp?: string;
}

/** Winner of a self-play game, normalised. Null = unknown / not finished. */
export type GameWinner = 'Black' | 'White' | 'Draw';

export interface SelfPlayMove {
  x: number;
  y: number;
}

/** The self-play game currently being generated (job detail `current_game`). */
export interface CurrentGame {
  game: number;
  moves: SelfPlayMove[];
  to_move: 1 | -1;
  status: 'playing' | 'finished';
  winner: GameWinner | null;
}

/** One finished self-play game, for the "recent games" strip. Only present
 * when the server reports it; never invented client-side. */
export interface RecentGame {
  game: number;
  /** Number of moves played. */
  length: number;
  winner: GameWinner | null;
}

export interface MctsStats {
  sims_per_move?: number | null;
  avg_game_length?: number | null;
  avg_move_time_ms?: number | null;
}

/** One checkpoint's eval snapshot from job detail `eval_history`. The
 * summary shape is server-defined; `evalHistoryWinRates` extracts
 * per-opponent win rates from it defensively. */
export interface EvalHistoryEntry {
  checkpoint: string;
  at?: string | null;
  summary: Record<string, unknown> | null;
}

export interface JobDetail extends JobSummary {
  metrics: MetricPoint[];
  games_generated: number;
  eval?: JobEval | null;
  logs: string[];
  error?: string | null;
  model_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  // --- Extended contract (may be absent on older servers) ---
  /** Measured self-play throughput, games/hour. Null = not reported. */
  games_per_hour?: number | null;
  /** Training samples buffered. May be an estimate — see
   * buffer_size_is_estimate. Null = not reported. */
  buffer_size?: number | null;
  buffer_size_is_estimate?: boolean;
  mcts?: MctsStats | null;
  current_game?: CurrentGame | null;
  recent_games?: RecentGame[];
  eval_history?: EvalHistoryEntry[];
  can_pause?: boolean;
  can_resume?: boolean;
  /** Target self-play games for this run (from hyperparameters), used for
   * the ETA estimate. Undefined when unknown. */
  total_games?: number;
}

/** A run preset from GET /api/presets. The preset carries no config values
 * itself — the server applies its config file; the form's fields override. */
export interface TrainerPreset {
  id: string;
  name: string;
  description?: string;
  config_file?: string;
  throughput_note?: string;
}

/** One row of an eval ladder: results vs one scripted opponent. */
export interface EvalLadderRow {
  opponent: string;
  wins: number;
  losses: number;
  draws: number;
  /** Win rate 0..1 when derivable (wins / total, or as reported). */
  win_rate?: number;
  /** 95% confidence interval on the win rate, when the server reports it. */
  ci_low?: number;
  ci_high?: number;
  elo_diff_vs_opponent?: number | null;
}

/** One tactical test position the model failed. */
export interface EvalFailedPosition {
  id?: string;
  name?: string;
  description?: string;
  detail?: string;
}

export interface EvalTactical {
  pass_rate?: number;
  passed?: number;
  total?: number;
  failed?: EvalFailedPosition[];
}

/** Result of a checkpoint-vs-checkpoint match, when the report has one. */
export interface EvalCheckpointMatch {
  checkpoint_a?: string;
  checkpoint_b?: string;
  wins: number;
  losses: number;
  draws: number;
}

/** Latest eval report from GET /api/jobs/{id}/eval. Every field is optional:
 * the server may report only a subset. */
export interface EvalReport {
  ladder: EvalLadderRow[];
  /** Elo-style estimate — relative to this ladder only, never absolute. */
  elo_estimate?: number | null;
  elo_estimate_note?: string;
  tactical?: EvalTactical | null;
  verdict?: string | null;
  generated_at?: string | null;
  checkpoint_vs_checkpoint?: EvalCheckpointMatch | null;
}

/** Created eval job: POST /api/jobs/{id}/evaluate or /api/models/{id}/evaluate. */
export interface EvalJobCreated {
  id: string;
  status: string;
}

export interface TrainerModel {
  id: string;
  name: string;
  run_id?: string;
  architecture?: string;
  created_at?: string;
  onnx_url?: string;
  formats: string[];
  trained_games?: number;
  final_loss?: number;
  /** Computed from the eval dict (opponent 'heuristic'), if present. */
  eval_win_rate_vs_heuristic?: number;
  eval?: JobEval;
  metrics?: Record<string, unknown>;
  /** 'server' for trained models, 'bundled' for manifest models the server
   * also lists. The UI shows bundled models from the local manifest instead,
   * so server listings are filtered to source 'server'. */
  source?: string;
  description?: string;
  notes?: string;
}

export type JobEventType =
  | 'progress'
  | 'metric'
  | 'log'
  | 'done'
  | 'error'
  | 'game_move'
  | 'game_finished'
  | 'eval';

export interface JobEvent {
  type: JobEventType;
  progress?: number;
  metric?: MetricPoint & Record<string, unknown>;
  message?: string;
  /** Log line (server sends `{type: 'log', line, phase}`). */
  line?: string;
  log?: string;
  phase?: string;
  status?: JobStatus;
  // --- Extended contract: self-play game events ---
  /** Game number for game_move / game_finished events. */
  game?: number;
  /** Full move list for game_move events. */
  moves?: SelfPlayMove[] | Record<string, unknown>[];
  to_move?: 1 | -1;
  winner?: unknown;
  [key: string]: unknown;
}

export class TrainerApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'TrainerApiError';
    this.status = status;
  }
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${TRAINER_API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...init,
    });
  } catch (err) {
    throw new TrainerApiError(
      `Could not reach the training server at ${TRAINER_API_BASE}. Is it running?`,
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as unknown;
      if (isRecord(body) && typeof body.detail === 'string') {
        detail = body.detail;
      } else if (isRecord(body) && Array.isArray(body.detail)) {
        // FastAPI validation errors: [{loc: [...], msg: '...'}].
        const msgs = body.detail
          .map((d) => {
            if (!isRecord(d) || typeof d.msg !== 'string') return null;
            const loc = Array.isArray(d.loc) ? d.loc.slice(1).join('.') : '';
            return loc ? `${loc}: ${d.msg}` : d.msg;
          })
          .filter((m): m is string => m !== null);
        if (msgs.length > 0) detail = msgs.join('; ');
      } else if (isRecord(body) && typeof body.message === 'string') {
        detail = body.message;
      }
    } catch {
      // Keep status text.
    }
    throw new TrainerApiError(`Server error ${res.status}: ${detail}`, res.status);
  }
  if (res.status === 204) return undefined as unknown as T;
  return (await res.json()) as T;
}

/** Resolve a possibly-relative server URL (e.g. a model's onnx_url) against
 * the API base so it is fetchable from the browser. */
export function resolveApiUrl(url: string | undefined): string {
  if (!url) return '';
  if (/^https?:\/\//i.test(url)) return url;
  return `${TRAINER_API_BASE}${url.startsWith('/') ? '' : '/'}${url}`;
}

// ---------------------------------------------------------------------------
// Normalisers — accept the contracted shapes plus nearby variants.
// ---------------------------------------------------------------------------

function asArchitectures(raw: unknown): Architecture[] {
  const list = isRecord(raw) && Array.isArray(raw.architectures) ? raw.architectures : [];
  return list
    .filter(isRecord)
    .map((a) => {
      const paramsRaw = isRecord(a.params) ? a.params : {};
      const params: Record<string, ArchParamSchema> = {};
      for (const [k, v] of Object.entries(paramsRaw)) {
        if (!isRecord(v)) continue;
        // The server speaks JSON Schema style ("integer", "minimum",
        // "maximum", "enum"); the task contract used "int"/"min"/"max".
        // Accept both spellings.
        const rawType = typeof v.type === 'string' ? v.type : '';
        const hasEnum = Array.isArray(v.enum) || Array.isArray(v.options);
        const type: ArchParamSchema['type'] =
          rawType === 'int' || rawType === 'integer'
            ? 'int'
            : rawType === 'float' || rawType === 'number'
              ? 'float'
              : rawType === 'select' || rawType === 'enum' || hasEnum
                ? 'select'
                : 'str';
        const enumRaw = Array.isArray(v.enum) ? v.enum : v.options;
        params[k] = {
          type,
          default: v.default,
          min:
            typeof v.min === 'number'
              ? v.min
              : typeof v.minimum === 'number'
                ? v.minimum
                : undefined,
          max:
            typeof v.max === 'number'
              ? v.max
              : typeof v.maximum === 'number'
                ? v.maximum
                : undefined,
          options: Array.isArray(enumRaw)
            ? enumRaw.filter((o): o is string => typeof o === 'string')
            : undefined,
          description: typeof v.description === 'string' ? v.description : undefined,
        };
      }
      return {
        id: String(a.id ?? ''),
        name: String(a.name ?? a.id ?? 'Unnamed'),
        description: String(a.description ?? ''),
        params,
      };
    })
    .filter((a) => a.id !== '');
}

function asJobStatus(v: unknown): JobStatus {
  return v === 'queued' ||
    v === 'running' ||
    v === 'paused' ||
    v === 'interrupted' ||
    v === 'completed' ||
    v === 'failed' ||
    v === 'cancelled'
    ? v
    : 'queued';
}

/** Normalise a winner value from the server: 1/-1/0, 'Black'/'White'/'Draw'
 * (any case), 'B'/'W', or null. Returns null when unrecognised. */
export function normalizeWinner(v: unknown): GameWinner | null {
  if (v === 1 || v === '1') return 'Black';
  if (v === -1 || v === '-1') return 'White';
  if (v === 0 || v === '0') return 'Draw';
  if (typeof v === 'string') {
    const s = v.trim().toLowerCase();
    if (s === 'black' || s === 'b') return 'Black';
    if (s === 'white' || s === 'w') return 'White';
    if (s === 'draw' || s === 'tie' || s === 'd') return 'Draw';
  }
  return null;
}

function asSelfPlayMoves(raw: unknown): SelfPlayMove[] {
  if (!Array.isArray(raw)) return [];
  const out: SelfPlayMove[] = [];
  for (const m of raw) {
    if (!isRecord(m)) continue;
    const x = m.x;
    const y = m.y;
    if (
      typeof x === 'number' &&
      typeof y === 'number' &&
      Number.isInteger(x) &&
      Number.isInteger(y) &&
      x >= 0 &&
      x < 15 &&
      y >= 0 &&
      y < 15
    ) {
      out.push({ x, y });
    }
  }
  return out;
}

function asCurrentGame(raw: unknown): CurrentGame | null {
  if (!isRecord(raw)) return null;
  const moves = asSelfPlayMoves(raw.moves);
  const game = typeof raw.game === 'number' ? raw.game : 0;
  const toMove = raw.to_move === -1 ? -1 : 1;
  const status = raw.status === 'finished' ? 'finished' : 'playing';
  return {
    game,
    moves,
    to_move: toMove as 1 | -1,
    status,
    winner: normalizeWinner(raw.winner),
  };
}

function asRecentGames(raw: unknown): RecentGame[] {
  if (!Array.isArray(raw)) return [];
  const out: RecentGame[] = [];
  for (const g of raw) {
    if (!isRecord(g)) continue;
    const game = typeof g.game === 'number' ? g.game : typeof g.number === 'number' ? g.number : null;
    if (game === null) continue;
    // Length may arrive as a count or as the move list itself.
    const length =
      typeof g.length === 'number'
        ? g.length
        : typeof g.moves_count === 'number'
          ? g.moves_count
          : Array.isArray(g.moves)
            ? g.moves.length
            : 0;
    out.push({ game, length, winner: normalizeWinner(g.winner) });
  }
  return out;
}

function asMctsStats(raw: unknown): MctsStats | null {
  if (!isRecord(raw)) return null;
  const num = (v: unknown): number | null =>
    typeof v === 'number' && Number.isFinite(v) ? v : null;
  return {
    sims_per_move: num(raw.sims_per_move ?? raw.sims),
    avg_game_length: num(raw.avg_game_length ?? raw.avg_length),
    avg_move_time_ms: num(raw.avg_move_time_ms ?? raw.avg_move_time),
  };
}

function asEvalHistory(raw: unknown): EvalHistoryEntry[] {
  if (!Array.isArray(raw)) return [];
  const out: EvalHistoryEntry[] = [];
  for (const e of raw) {
    if (!isRecord(e)) continue;
    const checkpoint =
      typeof e.checkpoint === 'string'
        ? e.checkpoint
        : typeof e.name === 'string'
          ? e.name
          : typeof e.id === 'string'
            ? e.id
            : '';
    if (!checkpoint) continue;
    const summary = isRecord(e.summary) ? e.summary : isRecord(e.eval) ? e.eval : null;
    out.push({
      checkpoint,
      at: typeof e.at === 'string' ? e.at : typeof e.timestamp === 'string' ? e.timestamp : null,
      summary,
    });
  }
  return out;
}

/**
 * Extract per-opponent win rates (0..1) from an eval-history entry's summary.
 * Accepts: a per-opponent dict of {wins, losses, draws} or {win_rate},
 * a {ladder: {...}} wrapper, or a single {win_rate} (reported as 'overall').
 * Returns {} when nothing parseable is present.
 */
export function evalHistoryWinRates(entry: EvalHistoryEntry): Record<string, number> {
  const out: Record<string, number> = {};
  const summary = entry.summary;
  if (!summary) return out;
  const scan = (dict: Record<string, unknown>) => {
    for (const [name, value] of Object.entries(dict)) {
      if (isRecord(value)) {
        const w = value.wins;
        const l = value.losses;
        const d = value.draws;
        if (typeof w === 'number' && typeof l === 'number' && typeof d === 'number') {
          const total = w + l + d;
          if (total > 0) out[name] = w / total;
          continue;
        }
        const wr = value.win_rate ?? value.winRate;
        if (typeof wr === 'number' && Number.isFinite(wr)) out[name] = wr;
      }
    }
  };
  const ladder = summary.ladder;
  if (isRecord(ladder)) scan(ladder);
  else scan(summary);
  // A single overall win rate (no per-opponent breakdown).
  if (Object.keys(out).length === 0) {
    const wr = summary.win_rate ?? summary.winRate;
    if (typeof wr === 'number' && Number.isFinite(wr)) out.overall = wr;
  }
  return out;
}

function asJobSummary(raw: unknown): JobSummary | null {
  if (!isRecord(raw)) return null;
  const id = raw.id;
  if (typeof id !== 'string' && typeof id !== 'number') return null;
  const type = raw.type;
  return {
    id: String(id),
    name: String(raw.name ?? `Job ${id}`),
    type: type === 'selfplay' || type === 'train' || type === 'full' ? type : 'full',
    status: asJobStatus(raw.status),
    progress: typeof raw.progress === 'number' ? Math.max(0, Math.min(1, raw.progress)) : 0,
    created_at: typeof raw.created_at === 'string' ? raw.created_at : undefined,
    architecture: typeof raw.architecture === 'string' ? raw.architecture : undefined,
    games_generated: typeof raw.games_generated === 'number' ? raw.games_generated : undefined,
    phase: typeof raw.phase === 'string' ? raw.phase : undefined,
  };
}

export function normalizeMetrics(raw: unknown): MetricPoint[] {
  // Accept: [{step, loss}], [{epoch, loss}], [{value}], [number],
  // {loss: [...]}, {history: [...]}, {metrics: [...]}.
  let arr: unknown = raw;
  if (isRecord(raw)) {
    arr = raw.history ?? raw.loss ?? raw.metrics ?? raw.points ?? [];
  }
  if (!Array.isArray(arr)) return [];
  const points: MetricPoint[] = [];
  arr.forEach((item, i) => {
    if (typeof item === 'number' && Number.isFinite(item)) {
      points.push({ step: i, loss: item });
    } else if (isRecord(item)) {
      const loss = item.loss ?? item.value ?? item.train_loss;
      if (typeof loss === 'number' && Number.isFinite(loss)) {
        // Server metric entries carry `epoch` (per-epoch loss) rather than
        // a step counter; prefer epoch as the x-axis when step is absent.
        const step =
          typeof item.step === 'number'
            ? item.step
            : typeof item.epoch === 'number'
              ? item.epoch
              : i;
        const numField = (v: unknown): number | undefined =>
          typeof v === 'number' && Number.isFinite(v) ? v : undefined;
        points.push({
          step,
          loss,
          epoch: typeof item.epoch === 'number' ? item.epoch : undefined,
          // Per-component losses are optional on each point.
          policy_loss: numField(item.policy_loss),
          value_loss: numField(item.value_loss),
          val_loss: numField(item.val_loss),
          games_generated: numField(item.games_generated),
          timestamp: typeof item.timestamp === 'string' ? item.timestamp : undefined,
        });
      }
    }
  });
  return points;
}

/**
 * Build a CurrentGame from a live SSE `game_move` event. The event carries
 * the full move list; returns null when the event has no usable game number
 * or move list. For `game_finished` events (which may carry only the game
 * number and winner), prefer `finishCurrentGame`.
 */
export function currentGameFromEvent(event: JobEvent): CurrentGame | null {
  if (typeof event.game !== 'number' || !Array.isArray(event.moves)) return null;
  const moves = asSelfPlayMoves(event.moves);
  return {
    game: event.game,
    moves,
    to_move: event.to_move === -1 ? -1 : moves.length % 2 === 0 ? 1 : -1,
    status: 'playing',
    winner: null,
  };
}

/**
 * Apply a `game_finished` event to the current game: marks it finished with
 * the reported winner when the game number matches (or when there is no
 * current game, returns a finished stub with no moves — the viewer then
 * shows the result without a board).
 */
export function finishCurrentGame(
  prev: CurrentGame | null,
  event: JobEvent,
): CurrentGame | null {
  if (typeof event.game !== 'number') return prev;
  if (prev && prev.game !== event.game) return prev;
  const winner = normalizeWinner(event.winner);
  if (!prev) {
    return { game: event.game, moves: [], to_move: 1, status: 'finished', winner };
  }
  return { ...prev, status: 'finished', winner };
}

/**
 * Build a MetricPoint from a live SSE `metric` event. The server sends the
 * point either nested under `metric` or flat on the event itself; accept
 * both, and return null when the event carries no usable loss.
 */
export function metricPointFromEvent(event: JobEvent): MetricPoint | null {
  const src: Record<string, unknown> = isRecord(event.metric)
    ? (event.metric as Record<string, unknown>)
    : (event as unknown as Record<string, unknown>);
  const loss = src.loss ?? src.value ?? src.train_loss;
  if (typeof loss !== 'number' || !Number.isFinite(loss)) return null;
  const numField = (v: unknown): number | undefined =>
    typeof v === 'number' && Number.isFinite(v) ? v : undefined;
  const epoch = typeof src.epoch === 'number' ? src.epoch : undefined;
  const step = typeof src.step === 'number' ? src.step : epoch;
  return {
    step: step ?? -1, // -1: caller assigns the next index when no step is given.
    loss,
    epoch,
    policy_loss: numField(src.policy_loss),
    value_loss: numField(src.value_loss),
    val_loss: numField(src.val_loss),
    games_generated: numField(src.games_generated),
    timestamp: typeof src.timestamp === 'string' ? src.timestamp : undefined,
  };
}

function asJobDetail(raw: unknown, fallbackId: string): JobDetail {
  const base = asJobSummary(raw) ?? {
    id: fallbackId,
    name: `Job ${fallbackId}`,
    type: 'full' as JobType,
    status: 'queued' as JobStatus,
    progress: 0,
  };
  const rec = isRecord(raw) ? raw : {};
  const logsRaw = rec.logs;
  let logs: string[] = [];
  if (typeof logsRaw === 'string') logs = logsRaw.split('\n').filter(Boolean);
  else if (Array.isArray(logsRaw)) {
    logs = logsRaw
      .map((l) => {
        if (typeof l === 'string') return l;
        if (isRecord(l)) {
          const msg = l.message ?? l.text ?? l.line;
          const ts = typeof l.timestamp === 'string' ? l.timestamp : '';
          return typeof msg === 'string' ? (ts ? `${ts} ${msg}` : msg) : '';
        }
        return '';
      })
      .filter(Boolean);
  }
  const evalRaw = rec.eval;
  let evalParsed: JobEval | null = null;
  if (isRecord(evalRaw)) {
    const parsed: JobEval = {};
    for (const [name, entry] of Object.entries(evalRaw)) {
      if (!isRecord(entry)) continue;
      const wins = entry.wins;
      const losses = entry.losses;
      const draws = entry.draws;
      if (typeof wins !== 'number' || typeof losses !== 'number' || typeof draws !== 'number') {
        continue;
      }
      parsed[name] = {
        wins,
        losses,
        draws,
        elo_diff_vs_opponent:
          typeof entry.elo_diff_vs_opponent === 'number' ? entry.elo_diff_vs_opponent : null,
      };
    }
    evalParsed = parsed;
  }
  // --- Extended contract fields (absent on older servers → null/undefined) ---
  const numOrNull = (v: unknown): number | null =>
    typeof v === 'number' && Number.isFinite(v) ? v : null;
  const bufferEstimateFlag =
    rec.buffer_size_is_estimate === true ||
    rec.buffer_size_estimated === true ||
    (typeof rec.buffer_size_note === 'string' &&
      rec.buffer_size_note.toLowerCase().includes('estimat'));
  const hyper = isRecord(rec.hyperparameters) ? rec.hyperparameters : {};
  const totalGames =
    typeof hyper.games === 'number' && hyper.games > 0 ? hyper.games : undefined;
  // can_pause / can_resume: prefer the server's booleans; otherwise infer
  // from status (running → pausable, paused/interrupted → resumable).
  const canPause =
    typeof rec.can_pause === 'boolean' ? rec.can_pause : base.status === 'running';
  const canResume =
    typeof rec.can_resume === 'boolean'
      ? rec.can_resume
      : base.status === 'paused' || base.status === 'interrupted';
  return {
    ...base,
    metrics: normalizeMetrics(rec.metrics),
    games_generated: typeof rec.games_generated === 'number' ? rec.games_generated : 0,
    eval: evalParsed,
    logs,
    error: typeof rec.error === 'string' ? rec.error : null,
    model_id: typeof rec.model_id === 'string' ? rec.model_id : null,
    started_at: typeof rec.started_at === 'string' ? rec.started_at : null,
    finished_at: typeof rec.finished_at === 'string' ? rec.finished_at : null,
    games_per_hour: numOrNull(rec.games_per_hour),
    buffer_size: numOrNull(rec.buffer_size),
    buffer_size_is_estimate: bufferEstimateFlag,
    mcts: asMctsStats(rec.mcts),
    current_game: asCurrentGame(rec.current_game),
    recent_games: asRecentGames(rec.recent_games),
    eval_history: asEvalHistory(rec.eval_history),
    can_pause: canPause,
    can_resume: canResume,
    total_games: totalGames,
  };
}

function asModels(raw: unknown): TrainerModel[] {
  const list = isRecord(raw) && Array.isArray(raw.models) ? raw.models : [];
  return list
    .map((m) => {
      if (!isRecord(m)) return null;
      const id = m.id;
      if (typeof id !== 'string' && typeof id !== 'number') return null;
      const metrics = isRecord(m.metrics) ? m.metrics : undefined;
      // The server returns eval as a per-opponent W/L/D dict (no precomputed
      // win rate); derive the heuristic win rate from it. Legacy flat fields
      // are still accepted if a future server sends them.
      const evalDict = isRecord(m.eval) ? (m.eval as JobEval) : undefined;
      const evalWr =
        typeof m.eval_win_rate_vs_heuristic === 'number'
          ? m.eval_win_rate_vs_heuristic
          : evalWinRate(evalDict, 'heuristic');
      const finalLoss =
        typeof m.final_loss === 'number'
          ? m.final_loss
          : metrics && typeof metrics.final_loss === 'number'
            ? (metrics.final_loss as number)
            : undefined;
      return {
        id: String(id),
        name: String(m.name ?? m.id),
        run_id: typeof m.run_id === 'string' ? m.run_id : undefined,
        architecture: typeof m.architecture === 'string' ? m.architecture : undefined,
        created_at: typeof m.created_at === 'string' ? m.created_at : undefined,
        onnx_url: typeof m.onnx_url === 'string' ? m.onnx_url : undefined,
        formats: Array.isArray(m.formats)
          ? m.formats.filter((f): f is string => typeof f === 'string')
          : [],
        trained_games: typeof m.trained_games === 'number' ? m.trained_games : undefined,
        final_loss: finalLoss,
        eval_win_rate_vs_heuristic: evalWr,
        eval: evalDict,
        metrics,
        source: typeof m.source === 'string' ? m.source : undefined,
        description: typeof m.description === 'string' ? m.description : undefined,
        notes: typeof m.notes === 'string' ? m.notes : undefined,
      } as TrainerModel;
    })
    .filter((m): m is TrainerModel => m !== null);
}

function asPresets(raw: unknown): TrainerPreset[] {
  const list = isRecord(raw) && Array.isArray(raw.presets) ? raw.presets : [];
  return list
    .map((p) => {
      if (!isRecord(p)) return null;
      const id = p.id;
      if (typeof id !== 'string' || id === '') return null;
      return {
        id,
        name: typeof p.name === 'string' && p.name !== '' ? p.name : id,
        description: typeof p.description === 'string' ? p.description : undefined,
        config_file: typeof p.config_file === 'string' ? p.config_file : undefined,
        throughput_note:
          typeof p.throughput_note === 'string' ? p.throughput_note : undefined,
      } as TrainerPreset;
    })
    .filter((p): p is TrainerPreset => p !== null);
}

function asEvalLadder(raw: unknown): EvalLadderRow[] {
  const rows: EvalLadderRow[] = [];
  const pushRow = (opponent: string, entry: Record<string, unknown>) => {
    const wins = typeof entry.wins === 'number' ? entry.wins : 0;
    const losses = typeof entry.losses === 'number' ? entry.losses : 0;
    const draws = typeof entry.draws === 'number' ? entry.draws : 0;
    const total = wins + losses + draws;
    const reported =
      typeof entry.win_rate === 'number'
        ? entry.win_rate
        : typeof entry.winRate === 'number'
          ? entry.winRate
          : undefined;
    const ci = Array.isArray(entry.ci) ? entry.ci : entry.confidence_interval;
    const ciArr = Array.isArray(ci) ? ci : null;
    rows.push({
      opponent,
      wins,
      losses,
      draws,
      win_rate:
        reported !== undefined && Number.isFinite(reported)
          ? reported
          : total > 0
            ? wins / total
            : undefined,
      ci_low:
        ciArr && typeof ciArr[0] === 'number' ? ciArr[0] : typeof entry.ci_low === 'number'
          ? entry.ci_low
          : undefined,
      ci_high:
        ciArr && typeof ciArr[1] === 'number' ? ciArr[1] : typeof entry.ci_high === 'number'
          ? entry.ci_high
          : undefined,
      elo_diff_vs_opponent:
        typeof entry.elo_diff_vs_opponent === 'number' ? entry.elo_diff_vs_opponent : null,
    });
  };
  if (Array.isArray(raw)) {
    for (const item of raw) {
      if (!isRecord(item)) continue;
      const opponent =
        typeof item.opponent === 'string'
          ? item.opponent
          : typeof item.name === 'string'
            ? item.name
            : '';
      if (opponent) pushRow(opponent, item);
    }
  } else if (isRecord(raw)) {
    // Dict form: {opponent: {wins, losses, draws, ...}}; also accept a
    // {results: [...]} / {rows: [...]} wrapper.
    const inner = raw.results ?? raw.rows;
    if (Array.isArray(inner)) return asEvalLadder(inner);
    for (const [opponent, entry] of Object.entries(raw)) {
      if (isRecord(entry)) pushRow(opponent, entry);
    }
  }
  return rows;
}

/** Parse a raw eval-report payload defensively. Exported for testing. */
export function parseEvalReport(raw: unknown): EvalReport {
  const rec = isRecord(raw) ? raw : {};
  const ladderRaw = rec.ladder ?? rec.results ?? rec.eval;
  // elo_estimate may be a bare number or an object {elo|value|estimate, note}.
  let elo: number | null = null;
  let eloNote: string | undefined;
  const eloRaw = rec.elo_estimate ?? rec.elo;
  if (typeof eloRaw === 'number' && Number.isFinite(eloRaw)) elo = eloRaw;
  else if (isRecord(eloRaw)) {
    const v = eloRaw.elo ?? eloRaw.value ?? eloRaw.estimate;
    if (typeof v === 'number' && Number.isFinite(v)) elo = v;
    if (typeof eloRaw.note === 'string') eloNote = eloRaw.note;
  }
  // Tactical: {pass_rate, passed, total, failed: [{id, name, ...}]}.
  let tactical: EvalTactical | null = null;
  const tacRaw = rec.tactical;
  if (isRecord(tacRaw)) {
    const failedRaw = Array.isArray(tacRaw.failed) ? tacRaw.failed : [];
    const failed: EvalFailedPosition[] = failedRaw
      .map((f): EvalFailedPosition | null => {
        if (typeof f === 'string') return { name: f };
        if (!isRecord(f)) return null;
        const detail =
          typeof f.detail === 'string'
            ? f.detail
            : typeof f.expected === 'string'
              ? `Expected: ${f.expected}`
              : undefined;
        return {
          id: typeof f.id === 'string' ? f.id : undefined,
          name: typeof f.name === 'string' ? f.name : undefined,
          description: typeof f.description === 'string' ? f.description : undefined,
          detail,
        };
      })
      .filter((f): f is EvalFailedPosition => f !== null);
    tactical = {
      pass_rate: typeof tacRaw.pass_rate === 'number' ? tacRaw.pass_rate : undefined,
      passed: typeof tacRaw.passed === 'number' ? tacRaw.passed : undefined,
      total: typeof tacRaw.total === 'number' ? tacRaw.total : undefined,
      failed,
    };
  }
  // Checkpoint-vs-checkpoint: {checkpoint_a, checkpoint_b, wins, losses, draws}
  // (also accept {a, b} / {new, old} naming).
  let cvc: EvalCheckpointMatch | null = null;
  const cvcRaw = rec.checkpoint_vs_checkpoint ?? rec.checkpoint_comparison;
  if (isRecord(cvcRaw)) {
    const wins = typeof cvcRaw.wins === 'number' ? cvcRaw.wins : 0;
    const losses = typeof cvcRaw.losses === 'number' ? cvcRaw.losses : 0;
    const draws = typeof cvcRaw.draws === 'number' ? cvcRaw.draws : 0;
    if (wins + losses + draws > 0) {
      const str = (v: unknown): string | undefined =>
        typeof v === 'string' ? v : undefined;
      cvc = {
        checkpoint_a: str(cvcRaw.checkpoint_a ?? cvcRaw.a ?? cvcRaw.new ?? cvcRaw.challenger),
        checkpoint_b: str(cvcRaw.checkpoint_b ?? cvcRaw.b ?? cvcRaw.old ?? cvcRaw.incumbent),
        wins,
        losses,
        draws,
      };
    }
  }
  return {
    ladder: asEvalLadder(ladderRaw),
    elo_estimate: elo,
    elo_estimate_note: eloNote,
    tactical,
    verdict: typeof rec.verdict === 'string' ? rec.verdict : null,
    generated_at: typeof rec.generated_at === 'string' ? rec.generated_at : null,
    checkpoint_vs_checkpoint: cvc,
  };
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export async function fetchHealth(): Promise<{ status: string }> {
  const raw = await api<unknown>('/api/health');
  if (isRecord(raw) && typeof raw.status === 'string') return { status: raw.status };
  return { status: 'ok' };
}

export async function fetchArchitectures(): Promise<Architecture[]> {
  return asArchitectures(await api<unknown>('/api/architectures'));
}

export async function fetchJobs(): Promise<JobSummary[]> {
  const raw = await api<unknown>('/api/jobs');
  const list = isRecord(raw) && Array.isArray(raw.jobs) ? raw.jobs : [];
  return list.map(asJobSummary).filter((j): j is JobSummary => j !== null);
}

export async function fetchJob(id: string): Promise<JobDetail> {
  return asJobDetail(await api<unknown>(`/api/jobs/${encodeURIComponent(id)}`), id);
}

export async function createJob(req: CreateJobRequest): Promise<{ id: string; status: string }> {
  const raw = await api<unknown>('/api/jobs', {
    method: 'POST',
    body: JSON.stringify(req),
  });
  if (isRecord(raw) && (typeof raw.id === 'string' || typeof raw.id === 'number')) {
    return { id: String(raw.id), status: String(raw.status ?? 'queued') };
  }
  throw new TrainerApiError('Unexpected response from the server when creating the job.', 500);
}

export async function cancelJob(id: string): Promise<void> {
  await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

/** Pause a running job. Returns the updated job detail (the server may
 * return the full job or a partial one; callers refetch to be sure). */
export async function pauseJob(id: string): Promise<JobDetail> {
  const raw = await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/pause`, {
    method: 'POST',
  });
  return asJobDetail(raw, id);
}

/** Resume a paused or interrupted job. */
export async function resumeJob(id: string): Promise<JobDetail> {
  const raw = await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/resume`, {
    method: 'POST',
  });
  return asJobDetail(raw, id);
}

function asEvalJobCreated(raw: unknown): EvalJobCreated {
  if (isRecord(raw) && (typeof raw.id === 'string' || typeof raw.id === 'number')) {
    return { id: String(raw.id), status: String(raw.status ?? 'queued') };
  }
  throw new TrainerApiError('Unexpected response from the server when starting evaluation.', 500);
}

/** Start an eval job for a completed run's model. `games` overrides the
 * default number of eval games per opponent when given. */
export async function evaluateJob(id: string, games?: number): Promise<EvalJobCreated> {
  const body: Record<string, unknown> = {};
  if (typeof games === 'number' && games > 0) body.games = games;
  const raw = await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/evaluate`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return asEvalJobCreated(raw);
}

/** Start an eval job for a model in the library. */
export async function evaluateModel(id: string, games?: number): Promise<EvalJobCreated> {
  const body: Record<string, unknown> = {};
  if (typeof games === 'number' && games > 0) body.games = games;
  const raw = await api<unknown>(`/api/models/${encodeURIComponent(id)}/evaluate`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return asEvalJobCreated(raw);
}

/** Latest eval report for a job, or null when no eval exists yet (404).
 * Other failures throw. */
export async function fetchJobEval(id: string): Promise<EvalReport | null> {
  try {
    const raw = await api<unknown>(`/api/jobs/${encodeURIComponent(id)}/eval`);
    return parseEvalReport(raw);
  } catch (e) {
    if (e instanceof TrainerApiError && e.status === 404) return null;
    throw e;
  }
}

/** Run presets from GET /api/presets. Throws (e.g. 404 on an older server)
 * — callers fall back to the preset-less form. */
export async function fetchPresets(): Promise<TrainerPreset[]> {
  return asPresets(await api<unknown>('/api/presets'));
}

export async function fetchModels(): Promise<TrainerModel[]> {
  return asModels(await api<unknown>('/api/models'));
}

export interface ReleaseAsset {
  name: string;
  size?: number;
}

export interface ReleaseResult {
  dry_run: boolean;
  ok: boolean;
  message?: string;
  release_url?: string;
  tag?: string;
  repo?: string;
  /** 'gh' | 'api' | 'unavailable' — how a real upload would happen. */
  method?: string;
  repo_warning?: string;
  body_preview?: string;
  total_size?: number;
  assets?: ReleaseAsset[];
  [key: string]: unknown;
}

export async function releaseModel(id: string, dryRun: boolean): Promise<ReleaseResult> {
  const raw = await api<unknown>(`/api/models/${encodeURIComponent(id)}/release`, {
    method: 'POST',
    body: JSON.stringify({ dry_run: dryRun }),
  });
  if (isRecord(raw)) {
    const assetsRaw = Array.isArray(raw.assets) ? raw.assets : [];
    const assets: ReleaseAsset[] = assetsRaw
      .map((a) => {
        if (typeof a === 'string') return { name: a };
        if (!isRecord(a)) return null;
        const name =
          typeof a.name === 'string'
            ? a.name
            : typeof a.path === 'string'
              ? a.path.split('/').pop() ?? a.path
              : null;
        if (!name) return null;
        return {
          name,
          size: typeof a.size === 'number' ? a.size : undefined,
        };
      })
      .filter((a): a is ReleaseAsset => a !== null);
    return {
      dry_run: dryRun,
      ok: raw.ok !== false && raw.success !== false && !raw.error,
      message:
        typeof raw.message === 'string'
          ? raw.message
          : typeof raw.detail === 'string'
            ? raw.detail
            : undefined,
      release_url: typeof raw.release_url === 'string' ? raw.release_url : undefined,
      tag: typeof raw.tag === 'string' ? raw.tag : undefined,
      repo: typeof raw.repo === 'string' ? raw.repo : undefined,
      method: typeof raw.method === 'string' ? raw.method : undefined,
      repo_warning: typeof raw.repo_warning === 'string' ? raw.repo_warning : undefined,
      body_preview: typeof raw.body_preview === 'string' ? raw.body_preview : undefined,
      total_size: typeof raw.total_size === 'number' ? raw.total_size : undefined,
      ...raw,
      // Parsed assets win over the raw ones (raw entries are objects).
      assets,
    };
  }
  return { dry_run: dryRun, ok: true };
}

// ---------------------------------------------------------------------------
// Live updates: Server-Sent Events with polling fallback.
// ---------------------------------------------------------------------------

export interface JobEventHandlers {
  onEvent: (event: JobEvent) => void;
  onError?: () => void;
}

/**
 * Subscribe to a job's SSE stream. Returns an unsubscribe function. If the
 * EventSource errors (server without SSE support, proxy dropping it, ...),
 * onError fires once and the caller should fall back to polling.
 */
export function subscribeToJobEvents(
  jobId: string,
  { onEvent, onError }: JobEventHandlers,
): () => void {
  const url = `${TRAINER_API_BASE}/api/jobs/${encodeURIComponent(jobId)}/events`;
  let errored = false;
  const source = new EventSource(url);
  source.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data) as JobEvent;
      if (data && typeof data.type === 'string') onEvent(data);
    } catch {
      // Ignore malformed events.
    }
  };
  source.onerror = () => {
    if (!errored) {
      errored = true;
      onError?.();
    }
    // If the stream keeps failing, close it so the polling fallback owns updates.
    source.close();
  };
  return () => source.close();
}

/**
 * Hook: keep a job's detail fresh. Tries SSE first; on SSE failure, polls
 * fetchJob every 2s. Polling also runs while SSE is healthy but silent?
 * No — SSE owns updates once connected; polling only replaces a failed SSE.
 * Stops entirely once the job reaches a terminal status.
 */
export function useJobLiveUpdates(
  jobId: string | null,
  onDetail: (detail: JobDetail) => void,
  onEvent?: (event: JobEvent) => void,
): { live: boolean; fallbackPolling: boolean } {
  const onDetailRef = useRef(onDetail);
  const onEventRef = useRef(onEvent);
  const [fallbackPolling, setFallbackPolling] = useState(false);
  onDetailRef.current = onDetail;
  onEventRef.current = onEvent;

  useEffect(() => {
    setFallbackPolling(false);
    if (!jobId) return;
    let cancelled = false;
    let usePolling = false;
    let pollTimer: ReturnType<typeof setInterval> | null = null;
    let terminal = false;

    const poll = async () => {
      try {
        const detail = await fetchJob(jobId);
        if (cancelled) return;
        onDetailRef.current(detail);
        if (
          detail.status === 'completed' ||
          detail.status === 'failed' ||
          detail.status === 'cancelled'
        ) {
          terminal = true;
          if (pollTimer) clearInterval(pollTimer);
        }
      } catch {
        // Poll failures are transient; keep trying until unmounted/terminal.
      }
    };

    const startPolling = () => {
      if (usePolling || terminal) return;
      usePolling = true;
      setFallbackPolling(true);
      void poll();
      pollTimer = setInterval(() => void poll(), 2000);
    };

    // Initial fetch, then SSE.
    void poll();
    const unsubscribe = subscribeToJobEvents(jobId, {
      onEvent: (event) => {
        if (cancelled) return;
        onEventRef.current?.(event);
        // Progress/done/eval events mean state changed; refetch detail
        // cheaply. game_move/game_finished carry their own payload and are
        // applied directly by the caller — refetching every move would
        // hammer the server.
        if (
          event.type === 'progress' ||
          event.type === 'done' ||
          event.type === 'error' ||
          event.type === 'eval'
        ) {
          void poll();
        }
        if (event.type === 'done' || event.type === 'error') {
          terminal = true;
          if (pollTimer) clearInterval(pollTimer);
        }
      },
      onError: () => {
        if (!cancelled) startPolling();
      },
    });

    return () => {
      cancelled = true;
      unsubscribe();
      if (pollTimer) clearInterval(pollTimer);
    };
  }, [jobId]);

  return { live: jobId !== null, fallbackPolling };
}

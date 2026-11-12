// Live run monitor: job list with status badges; selecting a job shows its
// progress bar, live training dashboard (throughput, buffer, MCTS stats,
// elapsed + ETA estimate), live self-play game viewer, loss curves, eval
// win-rate over time, eval report, scrolling log tail, and job controls
// (Pause / Resume / Cancel).
//
// Extended-contract fields (games/hour, buffer, MCTS, current game, eval
// history) may be absent on older servers: missing stats render as "—" with
// a note, never as invented numbers.

import { useEffect, useRef, useState } from 'react';
import type { JobDetail, JobStatus, JobSummary } from '../hooks/useTrainerApi';
import { EvalWinRateOverTime, LossCurves } from './WinRateGraph';
import TrainerEvalReport from './TrainerEvalReport';
import TrainerLiveViewer from './TrainerLiveViewer';

export type JobControlAction = 'pause' | 'resume' | 'cancel';

export interface TrainerJobMonitorProps {
  jobs: JobSummary[];
  selectedJobId: string | null;
  onSelectJob: (id: string) => void;
  detail: JobDetail | null;
  detailLoading: boolean;
  fallbackPolling: boolean;
  onCancel: (id: string) => void;
  onPause: (id: string) => void;
  onResume: (id: string) => void;
  /** Which control action is in flight, if any. */
  controlBusy: JobControlAction | null;
  /** Error from the last control action, if any. */
  controlError: string | null;
  onEvaluate: (jobId: string, games: number | undefined) => void;
  evaluating: boolean;
}

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  running: 'Running',
  paused: 'Paused',
  interrupted: 'Interrupted',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const TERMINAL: JobStatus[] = ['completed', 'failed', 'cancelled'];

function formatDate(iso: string | undefined | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) return '—';
  const totalSeconds = Math.floor(ms / 1000);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export default function TrainerJobMonitor({
  jobs,
  selectedJobId,
  onSelectJob,
  detail,
  detailLoading,
  fallbackPolling,
  onCancel,
  onPause,
  onResume,
  controlBusy,
  controlError,
  onEvaluate,
  evaluating,
}: TrainerJobMonitorProps) {
  const logRef = useRef<HTMLDivElement>(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  // Keep the log tail scrolled to the bottom as lines arrive.
  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [detail?.logs.length]);

  // Reset the cancel confirmation when another job is selected.
  useEffect(() => {
    setConfirmCancel(false);
  }, [selectedJobId]);

  // Disarm the cancel confirmation after a few seconds.
  useEffect(() => {
    if (!confirmCancel) return;
    const timer = setTimeout(() => setConfirmCancel(false), 4000);
    return () => clearTimeout(timer);
  }, [confirmCancel]);

  const selected = jobs.find((j) => j.id === selectedJobId) ?? null;
  const status: JobStatus = detail?.status ?? selected?.status ?? 'queued';
  const progress = detail?.progress ?? selected?.progress ?? 0;
  const isActive = status === 'queued' || status === 'running';
  const isPausable = detail?.can_pause ?? status === 'running';
  const isResumable =
    detail?.can_resume ?? (status === 'paused' || status === 'interrupted');
  const isControllable = isActive || status === 'paused' || status === 'interrupted';

  // Tick the clock while a job is active so elapsed time stays fresh.
  useEffect(() => {
    if (!isActive || !detail?.started_at) return;
    const timer = setInterval(() => setNow(Date.now()), 5000);
    return () => clearInterval(timer);
  }, [isActive, detail?.started_at]);

  const elapsedMs = (() => {
    if (!detail?.started_at) return null;
    const start = new Date(detail.started_at).getTime();
    if (Number.isNaN(start)) return null;
    const end = detail.finished_at
      ? new Date(detail.finished_at).getTime()
      : TERMINAL.includes(status)
        ? start
        : now;
    return Number.isNaN(end) ? null : end - start;
  })();

  // ETA estimate from progress + elapsed time. Only an estimate, labelled
  // as one, and only shown once there is enough progress to extrapolate.
  const etaMs =
    isActive && elapsedMs !== null && progress > 0.02 && progress < 1
      ? elapsedMs * ((1 - progress) / progress)
      : null;

  const hasExtendedStats =
    detail !== null &&
    (detail.games_per_hour !== null && detail.games_per_hour !== undefined ||
      detail.buffer_size !== null && detail.buffer_size !== undefined ||
      detail.mcts?.sims_per_move !== null && detail.mcts?.sims_per_move !== undefined ||
      detail.mcts?.avg_game_length !== null && detail.mcts?.avg_game_length !== undefined ||
      detail.mcts?.avg_move_time_ms !== null && detail.mcts?.avg_move_time_ms !== undefined);

  const showEvalSection =
    detail !== null &&
    (status === 'completed' || (detail.eval_history?.length ?? 0) > 0);

  return (
    <div className="job-monitor">
      <div className="job-list" role="listbox" aria-label="Training jobs">
        {jobs.length === 0 && <p className="muted">No runs yet — start one from “New run”.</p>}
        {jobs.map((job) => (
          <button
            key={job.id}
            type="button"
            role="option"
            aria-selected={job.id === selectedJobId}
            className={job.id === selectedJobId ? 'job-row selected' : 'job-row'}
            onClick={() => onSelectJob(job.id)}
          >
            <span className="job-row-name">{job.name}</span>
            <span className={`status-badge ${job.status}`}>{STATUS_LABEL[job.status]}</span>
            <span className="job-row-meta">
              {job.type} · {Math.round(job.progress * 100)}%
              {job.created_at ? ` · ${formatDate(job.created_at)}` : ''}
            </span>
            <span className="job-row-bar">
              <span style={{ width: `${Math.round(job.progress * 100)}%` }} />
            </span>
          </button>
        ))}
      </div>

      <div className="job-detail">
        {!selectedJobId && <p className="muted">Select a run to watch it live.</p>}
        {selectedJobId && detailLoading && !detail && <p className="muted">Loading run…</p>}
        {selectedJobId && detail && (
          <>
            <div className="job-detail-head">
              <h3>{detail.name}</h3>
              <span className={`status-badge ${status}`}>{STATUS_LABEL[status]}</span>
            </div>
            <p className="muted small">
              {detail.type} · {detail.architecture ? `${detail.architecture} · ` : ''}
              {detail.phase ? `phase: ${detail.phase} · ` : ''}
              created {formatDate(detail.created_at)}
              {detail.started_at ? ` · started ${formatDate(detail.started_at)}` : ''}
              {detail.finished_at ? ` · finished ${formatDate(detail.finished_at)}` : ''}
              {fallbackPolling && isActive ? ' · live updates via polling (SSE unavailable)' : ''}
              {!fallbackPolling && isActive ? ' · live' : ''}
            </p>

            {status === 'interrupted' && (
              <p className="notice" role="status">
                This run was interrupted — the training server restarted while it was running.
                Resume continues from the latest checkpoint; work already done isn&apos;t lost.
              </p>
            )}

            <div className="progress-block">
              <div className="progress-bar large">
                <span style={{ width: `${Math.round(progress * 100)}%` }} />
              </div>
              <span className="progress-label">{Math.round(progress * 100)}%</span>
            </div>

            <dl className="job-stats">
              <div>
                <dt>Games generated</dt>
                <dd>
                  {detail.games_generated.toLocaleString()}
                  {detail.total_games ? ` of ${detail.total_games.toLocaleString()}` : ''}
                </dd>
              </div>
              <div>
                <dt>Games / hour</dt>
                <dd>
                  {detail.games_per_hour !== null && detail.games_per_hour !== undefined
                    ? detail.games_per_hour.toLocaleString(undefined, { maximumFractionDigits: 1 })
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>Buffer size (samples)</dt>
                <dd>
                  {detail.buffer_size !== null && detail.buffer_size !== undefined
                    ? `${detail.buffer_size_is_estimate ? '≈ ' : ''}${detail.buffer_size.toLocaleString()}${detail.buffer_size_is_estimate ? ' (estimate)' : ''}`
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>MCTS sims / move</dt>
                <dd>
                  {detail.mcts?.sims_per_move !== null && detail.mcts?.sims_per_move !== undefined
                    ? detail.mcts.sims_per_move.toLocaleString()
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>Avg game length</dt>
                <dd>
                  {detail.mcts?.avg_game_length !== null && detail.mcts?.avg_game_length !== undefined
                    ? `${detail.mcts.avg_game_length.toFixed(1)} moves`
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>Avg move time</dt>
                <dd>
                  {detail.mcts?.avg_move_time_ms !== null && detail.mcts?.avg_move_time_ms !== undefined
                    ? `${detail.mcts.avg_move_time_ms.toFixed(0)} ms`
                    : '—'}
                </dd>
              </div>
              <div>
                <dt>Elapsed</dt>
                <dd>{elapsedMs !== null ? formatDuration(elapsedMs) : '—'}</dd>
              </div>
              <div>
                <dt>ETA</dt>
                <dd>
                  {etaMs !== null ? `~${formatDuration(etaMs)} (estimate)` : '—'}
                </dd>
              </div>
              {detail.eval &&
                Object.entries(detail.eval).map(([opponent, e]) => {
                  const total = e.wins + e.losses + e.draws;
                  const wr = total > 0 ? (e.wins / total) * 100 : 0;
                  return (
                    <div key={opponent}>
                      <dt>Eval vs {opponent}</dt>
                      <dd>
                        {wr.toFixed(1)}% ({e.wins}W–{e.losses}L–{e.draws}D)
                        {e.elo_diff_vs_opponent !== null && e.elo_diff_vs_opponent !== undefined
                          ? ` · Elo ${e.elo_diff_vs_opponent >= 0 ? '+' : ''}${e.elo_diff_vs_opponent.toFixed(0)}`
                          : ''}
                      </dd>
                    </div>
                  );
                })}
              {detail.model_id && (
                <div>
                  <dt>Produced model</dt>
                  <dd>{detail.model_id}</dd>
                </div>
              )}
            </dl>
            {!hasExtendedStats && (
              <p className="muted small">
                Throughput, buffer, and MCTS stats aren&apos;t reported by this server version
                (shown as —).
              </p>
            )}

            {detail.error && (
              <p className="error-text" role="alert">
                {detail.error}
              </p>
            )}

            {isControllable && (
              <div className="job-actions">
                {isPausable && (
                  <button
                    type="button"
                    className="btn"
                    onClick={() => onPause(detail.id)}
                    disabled={controlBusy !== null}
                  >
                    {controlBusy === 'pause' ? 'Pausing…' : '⏸ Pause run'}
                  </button>
                )}
                {isResumable && (
                  <button
                    type="button"
                    className="btn primary"
                    onClick={() => onResume(detail.id)}
                    disabled={controlBusy !== null}
                  >
                    {controlBusy === 'resume' ? 'Resuming…' : '▶ Resume run'}
                  </button>
                )}
                <button
                  type="button"
                  className={confirmCancel ? 'btn danger confirm' : 'btn danger'}
                  onClick={() => {
                    if (confirmCancel) {
                      setConfirmCancel(false);
                      onCancel(detail.id);
                    } else {
                      setConfirmCancel(true);
                    }
                  }}
                  disabled={controlBusy !== null}
                >
                  {controlBusy === 'cancel'
                    ? 'Cancelling…'
                    : confirmCancel
                      ? 'Click again to confirm cancel'
                      : '■ Cancel run'}
                </button>
                {controlError && (
                  <span className="error-text small" role="alert">
                    {controlError}
                  </span>
                )}
              </div>
            )}
            {TERMINAL.includes(status) && (
              <p className="muted small">This run is finished ({STATUS_LABEL[status].toLowerCase()}).</p>
            )}

            <TrainerLiveViewer
              currentGame={detail.current_game ?? null}
              recentGames={detail.recent_games ?? []}
              isRunning={status === 'running'}
              jobType={detail.type}
            />

            <h4>Loss curves</h4>
            <LossCurves points={detail.metrics} />

            <h4>Eval win rate over time</h4>
            <EvalWinRateOverTime history={detail.eval_history ?? []} />

            {showEvalSection && (
              <TrainerEvalReport
                jobId={detail.id}
                canEvaluate={status === 'completed'}
                onEvaluate={(games) => onEvaluate(detail.id, games)}
                evaluating={evaluating}
              />
            )}

            <h4>Logs</h4>
            <div ref={logRef} className="log-tail" aria-live="polite">
              {detail.logs.length === 0 ? (
                <span className="muted">No log lines yet.</span>
              ) : (
                detail.logs.map((line, i) => <div key={i}>{line}</div>)
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

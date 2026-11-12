// Trainer view: second top-level view of the app (tabs Play / Trainer).
// Talks to the training server through useTrainerApi. If the server is
// unreachable it shows a clear "not running" state with the exact start
// command — the game itself stays fully playable regardless.

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  TRAINER_API_BASE,
  TRAINER_START_COMMAND,
  cancelJob,
  createJob,
  currentGameFromEvent,
  evaluateJob,
  evaluateModel,
  fetchArchitectures,
  fetchHealth,
  fetchJob,
  fetchJobs,
  fetchModels,
  fetchPresets,
  finishCurrentGame,
  metricPointFromEvent,
  normalizeWinner,
  pauseJob,
  resumeJob,
  resolveApiUrl,
  useJobLiveUpdates,
  type Architecture,
  type CreateJobRequest,
  type JobDetail,
  type JobEvent,
  type JobSummary,
  type TrainerModel,
  type TrainerPreset,
} from '../hooks/useTrainerApi';
import type { ModelOption } from './types';
import TrainerRunForm from './TrainerRunForm';
import TrainerJobMonitor, { type JobControlAction } from './TrainerJobMonitor';
import TrainerModelLibrary from './TrainerModelLibrary';

export interface TrainerViewProps {
  localModels: ModelOption[];
  onLoadModelIntoGame: (model: ModelOption) => void;
  /** Called whenever the server model list changes, so the Play view's model
   * picker can merge server models with the bundled manifest. */
  onServerModelsChange?: (models: ModelOption[]) => void;
}

type HealthState = 'checking' | 'ok' | 'down';
type TrainerTab = 'new' | 'runs' | 'models';

export default function TrainerView({
  localModels,
  onLoadModelIntoGame,
  onServerModelsChange,
}: TrainerViewProps) {
  const [health, setHealth] = useState<HealthState>('checking');
  const [healthError, setHealthError] = useState<string | null>(null);
  const [tab, setTab] = useState<TrainerTab>('new');
  const [architectures, setArchitectures] = useState<Architecture[]>([]);
  const [presets, setPresets] = useState<TrainerPreset[]>([]);
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [models, setModels] = useState<TrainerModel[]>([]);
  /** Bundled models the server also lists — only used as a fallback if the
   * local manifest fetch came up empty, so they never appear twice. */
  const [bundledFromServer, setBundledFromServer] = useState<ModelOption[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [jobDetail, setJobDetail] = useState<JobDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [controlBusy, setControlBusy] = useState<JobControlAction | null>(null);
  const [controlError, setControlError] = useState<string | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const onServerModelsChangeRef = useRef(onServerModelsChange);
  onServerModelsChangeRef.current = onServerModelsChange;

  const refreshModels = useCallback(async () => {
    try {
      const list = await fetchModels();
      // /api/models also lists bundled manifest models (source 'bundled',
      // onnx_url /models/... served by the web app). Those belong to the
      // local manifest, not the server list — filter them out so nothing is
      // shown twice, and never resolve their URLs against the API base.
      const serverOnly = list.filter((m) => m.source !== 'bundled');
      const bundled = list.filter((m) => m.source === 'bundled');
      setModels(serverOnly);
      setBundledFromServer(
        bundled.map((m) => ({
          id: m.id,
          name: m.name,
          url: m.onnx_url ?? '',
          source: 'bundled' as const,
          architecture: m.architecture,
          trainedGames: m.trained_games,
          evalWinRate: m.eval_win_rate_vs_heuristic,
          createdAt: m.created_at,
          formats: m.formats,
          description: m.description,
          notes: m.notes,
        })),
      );
      onServerModelsChangeRef.current?.(
        serverOnly.map((m) => ({
          id: m.id,
          name: m.name,
          // Resolve against the API base: a relative onnx_url must not be
          // fetched from the web app's own origin.
          url: resolveApiUrl(m.onnx_url),
          source: 'server' as const,
          architecture: m.architecture,
          trainedGames: m.trained_games,
          evalWinRate: m.eval_win_rate_vs_heuristic,
          createdAt: m.created_at,
          formats: m.formats,
        })),
      );
    } catch {
      // Model list failures are non-fatal; the library shows an empty state.
    }
  }, []);

  const refreshJobs = useCallback(async () => {
    try {
      setJobs(await fetchJobs());
    } catch {
      // Keep the previous list on transient failure.
    }
  }, []);

  const checkHealth = useCallback(async () => {
    setHealth('checking');
    setHealthError(null);
    try {
      await fetchHealth();
      setHealth('ok');
      const [archs, presetList] = await Promise.all([
        fetchArchitectures().catch(() => [] as Architecture[]),
        // Presets are new in the extended contract; a 404 on an older server
        // just means the form renders without a preset picker.
        fetchPresets().catch(() => [] as TrainerPreset[]),
        refreshJobs(),
        refreshModels(),
      ]);
      setArchitectures(archs);
      setPresets(presetList);
    } catch (e) {
      setHealth('down');
      setHealthError(e instanceof Error ? e.message : 'Training server unreachable.');
    }
  }, [refreshJobs, refreshModels]);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  // Refresh the job list while any job is active.
  useEffect(() => {
    if (health !== 'ok') return;
    const anyActive = jobs.some((j) => j.status === 'queued' || j.status === 'running');
    if (!anyActive) return;
    const timer = setInterval(() => void refreshJobs(), 5000);
    return () => clearInterval(timer);
  }, [health, jobs, refreshJobs]);

  // Live updates for the selected job: SSE with 2s polling fallback.
  const handleDetail = useCallback((detail: JobDetail) => {
    setJobDetail(detail);
    setDetailLoading(false);
    setJobs((prev) => {
      const i = prev.findIndex((j) => j.id === detail.id);
      if (i === -1) return prev;
      const next = [...prev];
      next[i] = { ...next[i], status: detail.status, progress: detail.progress };
      return next;
    });
  }, []);

  // SSE events stream in between detail refetches: append them directly so
  // the log tail, loss curves, and live game viewer move live. The next
  // detail refetch replaces this state with the authoritative server record.
  const handleJobEvent = useCallback((event: JobEvent) => {
    if (event.type === 'log') {
      const line = typeof event.line === 'string' ? event.line : event.message;
      if (!line) return;
      setJobDetail((prev) => {
        if (!prev) return prev;
        // The server replays recent history on (re)connect, so skip lines
        // already in the tail to avoid duplicates.
        if (prev.logs.includes(line)) return prev;
        const logs = [...prev.logs, line];
        return { ...prev, logs: logs.slice(-500) };
      });
    } else if (event.type === 'metric') {
      const point = metricPointFromEvent(event);
      if (!point) return;
      setJobDetail((prev) => {
        if (!prev) return prev;
        const step = point.step >= 0 ? point.step : prev.metrics.length;
        if (prev.metrics.some((m) => m.step === step && m.loss === point.loss)) return prev;
        return { ...prev, metrics: [...prev.metrics, { ...point, step }] };
      });
    } else if (event.type === 'game_move') {
      const game = currentGameFromEvent(event);
      if (!game) return;
      setJobDetail((prev) => {
        if (!prev) return prev;
        // Ignore stale events for an older game than the one displayed.
        if (prev.current_game && prev.current_game.game > game.game) return prev;
        return { ...prev, current_game: game };
      });
    } else if (event.type === 'game_finished') {
      setJobDetail((prev) => {
        if (!prev) return prev;
        const current = finishCurrentGame(prev.current_game ?? null, event);
        // Record the finished game in the recent-games strip (the next
        // detail refetch replaces this with the server's own list, if any).
        const gameNum = typeof event.game === 'number' ? event.game : null;
        let recent = prev.recent_games ?? [];
        if (gameNum !== null && !recent.some((g) => g.game === gameNum)) {
          const length =
            current && current.game === gameNum ? current.moves.length : 0;
          recent = [
            ...recent,
            { game: gameNum, length, winner: normalizeWinner(event.winner) },
          ].slice(-20);
        }
        return { ...prev, current_game: current, recent_games: recent };
      });
    }
    // 'eval' events trigger a detail refetch in useJobLiveUpdates.
  }, []);
  const { fallbackPolling } = useJobLiveUpdates(selectedJobId, handleDetail, handleJobEvent);

  // When the selected job completes, a new model may have been registered —
  // refresh the library once per completed job.
  const refreshedForJobRef = useRef<string | null>(null);
  useEffect(() => {
    if (jobDetail?.status === 'completed' && refreshedForJobRef.current !== jobDetail.id) {
      refreshedForJobRef.current = jobDetail.id;
      void refreshModels();
    }
  }, [jobDetail, refreshModels]);

  const selectJob = useCallback(
    (id: string) => {
      setSelectedJobId(id);
      setJobDetail(null);
      setDetailLoading(true);
      fetchJob(id)
        .then(handleDetail)
        .catch(() => setDetailLoading(false));
    },
    [handleDetail],
  );

  const handleCreateJob = useCallback(
    async (req: CreateJobRequest) => {
      setSubmitting(true);
      setFormError(null);
      setNotice(null);
      try {
        const created = await createJob(req);
        setNotice(`Run “${req.name}” started (id ${created.id}, ${created.status}).`);
        await refreshJobs();
        setSelectedJobId(created.id);
        setTab('runs');
      } catch (e) {
        setFormError(e instanceof Error ? e.message : 'Could not start the run.');
      } finally {
        setSubmitting(false);
      }
    },
    [refreshJobs],
  );

  const runControlAction = useCallback(
    async (
      action: JobControlAction,
      id: string,
      call: (id: string) => Promise<unknown>,
      successNotice: string,
    ) => {
      setControlBusy(action);
      setControlError(null);
      setNotice(null);
      try {
        await call(id);
        setNotice(successNotice);
        // Refetch: pause/resume may return a partial job, so the detail
        // refetch is the authoritative state.
        const detail = await fetchJob(id).catch(() => null);
        if (detail) handleDetail(detail);
        await refreshJobs();
      } catch (e) {
        const message =
          e instanceof Error ? e.message : `Could not ${action} the run.`;
        setControlError(message);
      } finally {
        setControlBusy(null);
      }
    },
    [handleDetail, refreshJobs],
  );

  const handlePause = useCallback(
    (id: string) => void runControlAction('pause', id, pauseJob, 'Run paused.'),
    [runControlAction],
  );

  const handleResume = useCallback(
    (id: string) => void runControlAction('resume', id, resumeJob, 'Run resumed.'),
    [runControlAction],
  );

  const handleCancel = useCallback(
    (id: string) => void runControlAction('cancel', id, cancelJob, 'Run cancelled.'),
    [runControlAction],
  );

  /** Start an eval job and switch to it so its progress is visible. */
  const handleEvaluateJob = useCallback(
    async (jobId: string, games: number | undefined) => {
      setEvaluating(true);
      setNotice(null);
      try {
        const created = await evaluateJob(jobId, games);
        setNotice(`Evaluation started (job ${created.id}, ${created.status}).`);
        await refreshJobs();
        setSelectedJobId(created.id);
        setTab('runs');
      } catch (e) {
        setNotice(e instanceof Error ? e.message : 'Could not start the evaluation.');
      } finally {
        setEvaluating(false);
      }
    },
    [refreshJobs],
  );

  const handleEvaluateModel = useCallback(
    async (modelId: string, games: number | undefined) => {
      setEvaluating(true);
      setNotice(null);
      try {
        const created = await evaluateModel(modelId, games);
        setNotice(`Evaluation started (job ${created.id}, ${created.status}).`);
        await refreshJobs();
        setSelectedJobId(created.id);
        setTab('runs');
      } catch (e) {
        setNotice(e instanceof Error ? e.message : 'Could not start the evaluation.');
      } finally {
        setEvaluating(false);
      }
    },
    [refreshJobs],
  );

  const copyCommand = async () => {
    try {
      await navigator.clipboard.writeText(TRAINER_START_COMMAND);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard unavailable; the command is visible to copy by hand.
    }
  };

  if (health === 'checking') {
    return (
      <div className="trainer-view">
        <p className="muted">Contacting the training server at {TRAINER_API_BASE}…</p>
      </div>
    );
  }

  if (health === 'down') {
    return (
      <div className="trainer-view">
        <section className="panel server-down">
          <h2 className="panel-title">Training server not running</h2>
          <p>
            The Trainer needs the training server ({TRAINER_API_BASE}), and it isn&apos;t
            reachable right now. {healthError ?? ''}
          </p>
          <p>Start it from the project root with:</p>
          <pre className="command-box">
            <code>{TRAINER_START_COMMAND}</code>
            <button type="button" className="btn small" onClick={() => void copyCommand()}>
              {copied ? '✓ Copied' : 'Copy'}
            </button>
          </pre>
          <p className="muted small">
            The API base comes from <code>VITE_API_URL</code> (default{' '}
            <code>http://localhost:8000</code>). The game itself stays fully playable — switch
            back to the Play tab to keep playing.
          </p>
          <button type="button" className="btn primary" onClick={() => void checkHealth()}>
            ↻ Retry connection
          </button>
        </section>
      </div>
    );
  }

  return (
    <div className="trainer-view">
      <div className="trainer-head">
        <p className="muted small">
          Connected to <code>{TRAINER_API_BASE}</code> · training runs on the server; the game
          keeps working either way.
        </p>
        <button type="button" className="btn small" onClick={() => void checkHealth()}>
          ↻ Refresh
        </button>
      </div>

      {notice && (
        <p className="notice" role="status">
          {notice}
        </p>
      )}
      {formError && (
        <p className="error-text" role="alert">
          {formError}
        </p>
      )}

      <div className="trainer-tabs" role="tablist" aria-label="Trainer sections">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'new'}
          className={tab === 'new' ? 'tab active' : 'tab'}
          onClick={() => setTab('new')}
        >
          ✦ New run
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'runs'}
          className={tab === 'runs' ? 'tab active' : 'tab'}
          onClick={() => setTab('runs')}
        >
          ▶ Runs ({jobs.length})
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'models'}
          className={tab === 'models' ? 'tab active' : 'tab'}
          onClick={() => setTab('models')}
        >
          🧠 Models ({models.length + Math.max(localModels.length, bundledFromServer.length)})
        </button>
      </div>

      {tab === 'new' && (
        <section className="panel" aria-label="Start a new training run">
          <h2 className="panel-title">New training run</h2>
          <TrainerRunForm
            architectures={architectures}
            models={models}
            presets={presets}
            submitting={submitting}
            disabled={false}
            onSubmit={(req) => void handleCreateJob(req)}
          />
        </section>
      )}

      {tab === 'runs' && (
        <section className="panel" aria-label="Training runs">
          <h2 className="panel-title">Runs</h2>
          <TrainerJobMonitor
            jobs={jobs}
            selectedJobId={selectedJobId}
            onSelectJob={selectJob}
            detail={jobDetail}
            detailLoading={detailLoading}
            fallbackPolling={fallbackPolling}
            onCancel={handleCancel}
            onPause={handlePause}
            onResume={handleResume}
            controlBusy={controlBusy}
            controlError={controlError}
            onEvaluate={(jobId, games) => void handleEvaluateJob(jobId, games)}
            evaluating={evaluating}
          />
        </section>
      )}

      {tab === 'models' && (
        <section className="panel" aria-label="Model library">
          <h2 className="panel-title">Model library</h2>
          <TrainerModelLibrary
            serverModels={models}
            localModels={localModels.length > 0 ? localModels : bundledFromServer}
            onLoadIntoGame={onLoadModelIntoGame}
            onEvaluateModel={(modelId, games) => void handleEvaluateModel(modelId, games)}
            evaluating={evaluating}
          />
        </section>
      )}
    </div>
  );
}

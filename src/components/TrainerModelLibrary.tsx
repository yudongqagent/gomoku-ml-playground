// Model library: server-registry models plus the bundled local-manifest
// models (marked "Bundled") in one list. Actions per server model: load into
// the game (Master level), download ONNX, publish to a GitHub release —
// dry-run first, then an explicit confirm before any real release.

import { useState } from 'react';
import {
  releaseModel,
  resolveApiUrl,
  type ReleaseResult,
  type TrainerModel,
} from '../hooks/useTrainerApi';
import type { ModelOption } from './types';

export interface TrainerModelLibraryProps {
  serverModels: TrainerModel[];
  localModels: ModelOption[];
  onLoadIntoGame: (model: ModelOption) => void;
  /** Start an eval job for a server model. `games` is the per-opponent game
   * count, or undefined for the server default. */
  onEvaluateModel?: (modelId: string, games: number | undefined) => void;
  evaluating?: boolean;
}

type ReleaseState =
  | { phase: 'idle' }
  | { phase: 'dry-running' }
  | { phase: 'dry-run-done'; result: ReleaseResult }
  | { phase: 'releasing' }
  | { phase: 'done'; result: ReleaseResult }
  | { phase: 'error'; message: string };

function formatDate(iso: string | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString();
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function TrainerModelLibrary({
  serverModels,
  localModels,
  onLoadIntoGame,
  onEvaluateModel,
  evaluating = false,
}: TrainerModelLibraryProps) {
  const [releaseById, setReleaseById] = useState<Record<string, ReleaseState>>({});
  const [evalGamesById, setEvalGamesById] = useState<Record<string, string>>({});

  const setRelease = (id: string, s: ReleaseState) =>
    setReleaseById((prev) => ({ ...prev, [id]: s }));

  const runDryRun = async (model: TrainerModel) => {
    setRelease(model.id, { phase: 'dry-running' });
    try {
      const result = await releaseModel(model.id, true);
      setRelease(model.id, { phase: 'dry-run-done', result });
    } catch (e) {
      setRelease(model.id, {
        phase: 'error',
        message: e instanceof Error ? e.message : 'Dry run failed.',
      });
    }
  };

  const runRelease = async (model: TrainerModel) => {
    setRelease(model.id, { phase: 'releasing' });
    try {
      const result = await releaseModel(model.id, false);
      setRelease(model.id, { phase: 'done', result });
    } catch (e) {
      setRelease(model.id, {
        phase: 'error',
        message: e instanceof Error ? e.message : 'Release failed.',
      });
    }
  };

  const toOption = (m: TrainerModel): ModelOption => ({
    id: m.id,
    name: m.name,
    url: resolveApiUrl(m.onnx_url),
    source: 'server',
    architecture: m.architecture,
    trainedGames: m.trained_games,
    evalWinRate: m.eval_win_rate_vs_heuristic,
    createdAt: m.created_at,
    formats: m.formats,
  });

  return (
    <div className="model-library">
      {serverModels.length === 0 && localModels.length === 0 && (
        <p className="muted">
          No models yet. Train a run to produce one, or bundle a model under{' '}
          <code>public/models/</code> with a <code>models.json</code> manifest.
        </p>
      )}

      {serverModels.length > 0 && <h3>Trained on the server</h3>}
      <div className="model-grid">
        {serverModels.map((m) => {
          const rel = releaseById[m.id] ?? { phase: 'idle' as const };
          const onnxUrl = resolveApiUrl(m.onnx_url);
          return (
            <article key={m.id} className="model-card">
              <h4>{m.name}</h4>
              {m.description && <p className="muted small">{m.description}</p>}
              {m.notes && <p className="muted small">{m.notes}</p>}
              <dl className="model-meta">
                {m.architecture && (
                  <div>
                    <dt>Architecture</dt>
                    <dd>{m.architecture}</dd>
                  </div>
                )}
                <div>
                  <dt>Trained games</dt>
                  <dd>{m.trained_games !== undefined ? m.trained_games.toLocaleString() : '—'}</dd>
                </div>
                <div>
                  <dt>Eval vs heuristic</dt>
                  <dd>
                    {m.eval_win_rate_vs_heuristic !== undefined
                      ? `${(m.eval_win_rate_vs_heuristic * 100).toFixed(1)}%`
                      : '—'}
                  </dd>
                </div>
                {m.final_loss !== undefined && (
                  <div>
                    <dt>Final loss</dt>
                    <dd>{m.final_loss.toFixed(4)}</dd>
                  </div>
                )}
                <div>
                  <dt>Created</dt>
                  <dd>{formatDate(m.created_at)}</dd>
                </div>
                <div>
                  <dt>Formats</dt>
                  <dd>{m.formats.length > 0 ? m.formats.join(', ') : '—'}</dd>
                </div>
              </dl>
              <div className="model-actions">
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => onLoadIntoGame(toOption(m))}
                  disabled={!onnxUrl}
                  title={onnxUrl ? 'Use this model for the Master level in the Play view' : 'No ONNX download URL for this model'}
                >
                  ▶ Load into game
                </button>
                {onnxUrl && (
                  <a className="btn" href={onnxUrl} download>
                    ⬇ Download ONNX
                  </a>
                )}
              </div>
              {onEvaluateModel && (
                <div className="eval-trigger model-eval">
                  <label className="field inline">
                    <span>Games / opponent</span>
                    <input
                      type="number"
                      min={1}
                      step={1}
                      value={evalGamesById[m.id] ?? '20'}
                      onChange={(e) =>
                        setEvalGamesById((prev) => ({ ...prev, [m.id]: e.target.value }))
                      }
                      placeholder="Server default"
                      disabled={evaluating}
                    />
                  </label>
                  <button
                    type="button"
                    className="btn"
                    disabled={evaluating}
                    onClick={() => {
                      const raw = evalGamesById[m.id] ?? '20';
                      const n = Number(raw);
                      onEvaluateModel(
                        m.id,
                        raw.trim() === '' || !Number.isFinite(n) || n <= 0
                          ? undefined
                          : Math.floor(n),
                      );
                    }}
                  >
                    {evaluating ? 'Starting…' : '▶ Evaluate'}
                  </button>
                  <small className="muted">
                    Runs this model against the scripted ladder; the eval job shows up in Runs.
                  </small>
                </div>
              )}
              <div className="release-box">
                <h5>Publish to GitHub release</h5>
                {rel.phase === 'idle' && (
                  <button type="button" className="btn" onClick={() => void runDryRun(m)}>
                    Dry-run release
                  </button>
                )}
                {rel.phase === 'dry-running' && <p className="muted small">Running dry run…</p>}
                {rel.phase === 'dry-run-done' && (
                  <>
                    <div className="small dry-run-summary">
                      <p>
                        Dry run {rel.result.ok ? 'looks good' : 'reported a problem'}.
                        {rel.result.repo ? ` Repo: ${rel.result.repo}.` : ''}
                        {rel.result.tag ? ` Tag: ${rel.result.tag}.` : ''}
                      </p>
                      {rel.result.assets?.length ? (
                        <ul>
                          {rel.result.assets.map((a) => (
                            <li key={a.name}>
                              {a.name}
                              {a.size !== undefined ? ` (${formatBytes(a.size)})` : ''}
                            </li>
                          ))}
                        </ul>
                      ) : null}
                      {rel.result.total_size !== undefined && (
                        <p>Total size: {formatBytes(rel.result.total_size)}.</p>
                      )}
                      {rel.result.method && (
                        <p>
                          Upload method:{' '}
                          {rel.result.method === 'unavailable'
                            ? 'none available — run `gh auth login` or set GITHUB_TOKEN first'
                            : rel.result.method}
                          .
                        </p>
                      )}
                      {rel.result.repo_warning && (
                        <p className="error-text small">{rel.result.repo_warning}</p>
                      )}
                      {rel.result.body_preview && (
                        <details>
                          <summary>Release body preview</summary>
                          <pre className="release-preview">{rel.result.body_preview}</pre>
                        </details>
                      )}
                      {rel.result.message ? <p>{rel.result.message}</p> : ''}
                    </div>
                    <button
                      type="button"
                      className="btn danger"
                      onClick={() => void runRelease(m)}
                      disabled={!rel.result.ok || rel.result.method === 'unavailable'}
                      title={
                        rel.result.method === 'unavailable'
                          ? 'No GitHub auth configured on the server'
                          : 'Publish the release to GitHub for real'
                      }
                    >
                      Confirm — publish for real
                    </button>
                    <button type="button" className="btn" onClick={() => setRelease(m.id, { phase: 'idle' })}>
                      Dismiss
                    </button>
                  </>
                )}
                {rel.phase === 'releasing' && <p className="muted small">Publishing release…</p>}
                {rel.phase === 'done' && (
                  <p className="small">
                    Release {rel.result.ok ? 'published' : 'finished with a problem'}.
                    {rel.result.release_url && (
                      <>
                        {' '}
                        <a href={rel.result.release_url} target="_blank" rel="noreferrer">
                          View release
                        </a>
                      </>
                    )}
                    {rel.result.message ? ` ${rel.result.message}` : ''}
                  </p>
                )}
                {rel.phase === 'error' && (
                  <p className="error-text small" role="alert">
                    {rel.message}{' '}
                    <button type="button" className="link-btn" onClick={() => setRelease(m.id, { phase: 'idle' })}>
                      Retry
                    </button>
                  </p>
                )}
              </div>
            </article>
          );
        })}
      </div>

      {localModels.length > 0 && <h3>Bundled with the app</h3>}
      <div className="model-grid">
        {localModels.map((m) => (
          <article key={`bundled:${m.id}`} className="model-card bundled">
            <h4>
              {m.name} <span className="source-badge">Bundled</span>
            </h4>
            {m.description && <p className="muted small">{m.description}</p>}
            <dl className="model-meta">
              <div>
                <dt>Trained games</dt>
                <dd>{m.trainedGames !== undefined ? m.trainedGames.toLocaleString() : '—'}</dd>
              </div>
              {m.notes && (
                <div>
                  <dt>Notes</dt>
                  <dd>{m.notes}</dd>
                </div>
              )}
            </dl>
            <p className="honesty-note small">
              Bundled demo net — modest strength, not superhuman.
            </p>
            <div className="model-actions">
              <button type="button" className="btn primary" onClick={() => onLoadIntoGame(m)}>
                ▶ Load into game
              </button>
              <a className="btn" href={m.url} download>
                ⬇ Download ONNX
              </a>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

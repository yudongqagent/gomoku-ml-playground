// AI panel: level cards (copy comes from LEVEL_META in the AI workstream —
// never hardcoded here), advanced search controls, the merged model picker
// (bundled manifest + training-server registry), and the per-move perf readout.
//
// Honesty: no claim here says the AI beats the best human. Master is labelled
// "AlphaZero-style" and its strength is explicitly tied to the loaded model.

import { useState } from 'react';
import { LEVEL_META, LEVEL_ORDER, type LevelId } from '../ai/levels';
import type { AiOptions, AiPerf, ModelOption } from './types';

export interface AiPanelProps {
  level: LevelId;
  onLevelChange: (level: LevelId) => void;
  options: AiOptions;
  onOptionsChange: (options: AiOptions) => void;
  models: ModelOption[];
  modelsLoading: boolean;
  modelsError: string | null;
  /** Composite key "source:id" — ids may collide across sources. */
  selectedModelId: string | null;
  onModelChange: (modelKey: string | null) => void;
  onRefreshModels: () => void;
  perf: AiPerf | null;
  status: string;
  thinking: boolean;
}

function levelMeta(level: LevelId) {
  const meta = LEVEL_META?.[level];
  return {
    name: meta?.name ?? level,
    tagline: meta?.tagline ?? '',
    description: meta?.description ?? '',
    strengthNote: meta?.strengthNote ?? '',
    defaultSims: meta?.defaultSims ?? 200,
    defaultTimeMs: meta?.defaultTimeMs ?? 1000,
  };
}

export default function AiPanel({
  level,
  onLevelChange,
  options,
  onOptionsChange,
  models,
  modelsLoading,
  modelsError,
  selectedModelId,
  onModelChange,
  onRefreshModels,
  perf,
  status,
  thinking,
}: AiPanelProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const setNum = (key: keyof AiOptions, raw: string, fallback: number) => {
    const v = Number(raw);
    onOptionsChange({ ...options, [key]: Number.isFinite(v) && v > 0 ? v : fallback });
  };

  const selectedModel =
    models.find((m) => `${m.source}:${m.id}` === selectedModelId) ?? null;
  const masterUsesModel = level === 'master';

  return (
    <section className="panel ai-panel" aria-label="AI opponent">
      <h2 className="panel-title">AI opponent</h2>

      <div className="level-grid" role="radiogroup" aria-label="AI level">
        {LEVEL_ORDER.map((id) => {
          const meta = levelMeta(id);
          const selected = id === level;
          return (
            <button
              key={id}
              type="button"
              role="radio"
              aria-checked={selected}
              className={selected ? 'level-card selected' : 'level-card'}
              onClick={() => onLevelChange(id)}
            >
              <span className="level-name">
                {meta.name}
                {id === 'master' && <span className="level-badge">AlphaZero-style</span>}
              </span>
              {meta.tagline && <span className="level-tagline">{meta.tagline}</span>}
              {meta.description && <span className="level-desc">{meta.description}</span>}
              {meta.strengthNote && <span className="level-note">{meta.strengthNote}</span>}
            </button>
          );
        })}
      </div>

      <p className="honesty-note">
        Strength depends on the loaded model — the bundled model is a small demo net, not
        superhuman. Train a stronger net in the Trainer tab to raise Master&apos;s ceiling.
      </p>

      <div className="ai-status-row">
        <span className={thinking ? 'status-pill thinking' : 'status-pill'}>
          {thinking ? '● Thinking…' : '○ Idle'}
        </span>
        {status && <span className="ai-status-text">{status}</span>}
      </div>

      {perf && (
        <dl className="perf-readout">
          <div>
            <dt>Nodes / sec</dt>
            <dd>{perf.nps.toLocaleString()}</dd>
          </div>
          <div>
            <dt>Last move</dt>
            <dd>
              {perf.ms} ms · {perf.nodes.toLocaleString()} nodes
              {perf.depth !== undefined ? ` · depth ${perf.depth}` : ''}
            </dd>
          </div>
          <div>
            <dt>Backend</dt>
            <dd>{perf.backend ?? '—'}</dd>
          </div>
        </dl>
      )}

      <button
        type="button"
        className="details-toggle"
        aria-expanded={advancedOpen}
        onClick={() => setAdvancedOpen((o) => !o)}
      >
        {advancedOpen ? '▾' : '▸'} Advanced controls
      </button>

      {advancedOpen && (
        <div className="advanced-controls">
          <label className="field">
            <span>MCTS simulations</span>
            <input
              type="number"
              min={1}
              max={100000}
              step={1}
              value={options.sims}
              onChange={(e) => setNum('sims', e.target.value, levelMeta(level).defaultSims)}
            />
          </label>
          <label className="field">
            <span>Time per move (ms)</span>
            <input
              type="number"
              min={50}
              max={60000}
              step={50}
              value={options.timeMs}
              onChange={(e) => setNum('timeMs', e.target.value, levelMeta(level).defaultTimeMs)}
            />
          </label>
          <label className="field">
            <span>Temperature</span>
            <input
              type="number"
              min={0}
              max={2}
              step={0.05}
              value={options.temperature}
              onChange={(e) => {
                const v = Number(e.target.value);
                onOptionsChange({
                  ...options,
                  temperature: Number.isFinite(v) ? Math.max(0, Math.min(2, v)) : 1,
                });
              }}
            />
            <small>0 = always the best move, higher = more variety.</small>
          </label>

          <label className="field">
            <span>
              Neural model {masterUsesModel ? '(Master)' : ''}
              <button type="button" className="link-btn" onClick={onRefreshModels} title="Refresh model list">
                ↻ Refresh
              </button>
            </span>
            <select
              value={selectedModelId ?? ''}
              onChange={(e) => onModelChange(e.target.value === '' ? null : e.target.value)}
              disabled={modelsLoading && models.length === 0}
            >
              <option value="">Search only (no neural model)</option>
              {models.map((m) => (
                <option key={`${m.source}:${m.id}`} value={`${m.source}:${m.id}`}>
                  {m.name} ({m.source === 'bundled' ? 'Bundled' : 'Server'}
                  {m.trainedGames !== undefined ? ` · ${m.trainedGames.toLocaleString()} games` : ''})
                </option>
              ))}
            </select>
            {modelsLoading && <small>Loading models…</small>}
            {!modelsLoading && models.length === 0 && (
              <small>{modelsError ?? 'No neural models found — Master falls back to search.'}</small>
            )}
            {selectedModel && (
              <small>
                {selectedModel.description ?? ''}
                {selectedModel.evalWinRate !== undefined &&
                  ` Eval win rate vs heuristic: ${(selectedModel.evalWinRate * 100).toFixed(1)}%.`}
                {selectedModel.source === 'bundled' && ' Bundled demo net — modest strength.'}
              </small>
            )}
          </label>
        </div>
      )}
    </section>
  );
}

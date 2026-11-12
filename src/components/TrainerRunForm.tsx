// "New run" form: run name, job type, architecture picker (rendered from the
// server's /api/architectures schema — int/float/str/select fields), key
// hyperparameters, base checkpoint select, and validation before submit.

import { useEffect, useMemo, useState } from 'react';
import type {
  Architecture,
  CreateJobRequest,
  Hyperparameters,
  JobType,
  TrainerModel,
  TrainerPreset,
} from '../hooks/useTrainerApi';

export interface TrainerRunFormProps {
  architectures: Architecture[];
  models: TrainerModel[];
  /** Run presets from GET /api/presets. Empty when the server doesn't offer
   * them (older server) — the form then works exactly as before, without a
   * preset picker. */
  presets: TrainerPreset[];
  submitting: boolean;
  disabled: boolean;
  onSubmit: (req: CreateJobRequest) => void;
}

/** Default preset when the server offers it. */
const DEFAULT_PRESET_ID = 'smoke';

const JOB_TYPES: { id: JobType; name: string; blurb: string }[] = [
  { id: 'selfplay', name: 'Self-play only', blurb: 'Generate games with the current net. No training.' },
  { id: 'train', name: 'Train only', blurb: 'Train on existing self-play games.' },
  { id: 'full', name: 'Full pipeline', blurb: 'Self-play, then train, then evaluate.' },
];

const DEFAULT_HYPER: Hyperparameters = {
  games: 100,
  epochs: 10,
  mcts_sims: 200,
  lr: 0.001,
  batch_size: 64,
  temperature: 1.0,
};

function defaultsFor(arch: Architecture | undefined): Record<string, unknown> {
  if (!arch) return {};
  const out: Record<string, unknown> = {};
  for (const [k, schema] of Object.entries(arch.params)) {
    if (schema.default !== undefined) out[k] = schema.default;
    else if (schema.type === 'int' || schema.type === 'float') out[k] = schema.min ?? 0;
    else if (schema.type === 'select' && schema.options?.length) out[k] = schema.options[0];
    else out[k] = '';
  }
  return out;
}

export default function TrainerRunForm({
  architectures,
  models,
  presets,
  submitting,
  disabled,
  onSubmit,
}: TrainerRunFormProps) {
  const [name, setName] = useState('');
  const [type, setType] = useState<JobType>('full');
  const [archId, setArchId] = useState('');
  const [archParams, setArchParams] = useState<Record<string, unknown>>({});
  const [hyper, setHyper] = useState<Hyperparameters>(DEFAULT_HYPER);
  const [baseCheckpoint, setBaseCheckpoint] = useState<string>('');
  const [presetId, setPresetId] = useState('');
  const [errors, setErrors] = useState<string[]>([]);

  // Default to the `smoke` preset when the server offers it. The user can
  // pick another preset or none; picking one only changes what is sent as
  // `preset` — the server applies the preset's config file, and the fields
  // below act as overrides.
  useEffect(() => {
    if (!presetId && presets.some((p) => p.id === DEFAULT_PRESET_ID)) {
      setPresetId(DEFAULT_PRESET_ID);
    }
  }, [presets, presetId]);

  const preset = useMemo(() => presets.find((p) => p.id === presetId), [presets, presetId]);

  const arch = useMemo(
    () => architectures.find((a) => a.id === archId),
    [architectures, archId],
  );

  // Default to the first architecture when the list loads.
  useEffect(() => {
    if (!archId && architectures.length > 0) {
      setArchId(architectures[0].id);
    }
  }, [architectures, archId]);

  // Reset architecture params to schema defaults when the architecture changes.
  useEffect(() => {
    setArchParams(defaultsFor(arch));
  }, [arch?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const setHyperNum = (key: keyof Hyperparameters, raw: string) => {
    const v = Number(raw);
    setHyper((h) => ({ ...h, [key]: Number.isFinite(v) ? v : h[key] }));
  };

  const validate = (): string[] => {
    const errs: string[] = [];
    if (name.trim() === '') errs.push('Give the run a name.');
    if (architectures.length > 0 && !arch) errs.push('Pick an architecture.');
    if (!Number.isInteger(hyper.games) || hyper.games <= 0) {
      errs.push('Games must be a positive whole number.');
    }
    if (!Number.isInteger(hyper.epochs) || hyper.epochs <= 0) {
      errs.push('Epochs must be a positive whole number.');
    }
    if (!Number.isInteger(hyper.mcts_sims) || hyper.mcts_sims <= 0) {
      errs.push('MCTS sims must be a positive whole number.');
    }
    if (!(hyper.lr > 0) || hyper.lr >= 1) errs.push('Learning rate must be between 0 and 1.');
    if (!Number.isInteger(hyper.batch_size) || hyper.batch_size <= 0) {
      errs.push('Batch size must be a positive whole number.');
    }
    if (!(hyper.temperature >= 0) || hyper.temperature > 10) {
      errs.push('Temperature must be between 0 and 10.');
    }
    return errs;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const errs = validate();
    setErrors(errs);
    if (errs.length > 0) return;
    onSubmit({
      type,
      name: name.trim(),
      architecture: arch?.id ?? '',
      arch_params: archParams,
      hyperparameters: hyper,
      base_checkpoint: baseCheckpoint === '' ? null : baseCheckpoint,
      ...(presetId !== '' ? { preset: presetId } : {}),
    });
  };

  return (
    <form className="trainer-form" onSubmit={handleSubmit}>
      {presets.length > 0 && (
        <fieldset className="field" disabled={disabled || submitting}>
          <legend>Preset</legend>
          <div className="preset-grid">
            {presets.map((p) => (
              <label key={p.id} className={presetId === p.id ? 'preset-card selected' : 'preset-card'}>
                <input
                  type="radio"
                  name="preset"
                  value={p.id}
                  checked={presetId === p.id}
                  onChange={() => setPresetId(p.id)}
                />
                <strong>{p.name}</strong>
                {p.description && <small>{p.description}</small>}
                {p.throughput_note && <small className="preset-throughput">{p.throughput_note}</small>}
              </label>
            ))}
            <label className={presetId === '' ? 'preset-card selected' : 'preset-card'}>
              <input
                type="radio"
                name="preset"
                value=""
                checked={presetId === ''}
                onChange={() => setPresetId('')}
              />
              <strong>No preset</strong>
              <small>Use only the settings below.</small>
            </label>
          </div>
          {preset && (
            <small className="muted">
              The preset&apos;s defaults come from the server
              {preset.config_file ? ` (${preset.config_file})` : ''}; the architecture and
              hyperparameter fields below override them.
            </small>
          )}
        </fieldset>
      )}

      <label className="field">
        <span>Run name *</span>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g. resnet-64 run 3"
          maxLength={120}
          disabled={disabled || submitting}
        />
      </label>

      <fieldset className="field" disabled={disabled || submitting}>
        <legend>Job type</legend>
        <div className="job-type-grid">
          {JOB_TYPES.map((t) => (
            <label key={t.id} className={type === t.id ? 'job-type selected' : 'job-type'}>
              <input
                type="radio"
                name="job-type"
                value={t.id}
                checked={type === t.id}
                onChange={() => setType(t.id)}
              />
              <strong>{t.name}</strong>
              <small>{t.blurb}</small>
            </label>
          ))}
        </div>
      </fieldset>

      <label className="field">
        <span>Architecture</span>
        {architectures.length === 0 ? (
          <small className="muted">
            No architectures reported by the server — the run will use the server&apos;s default.
          </small>
        ) : (
          <select
            value={archId}
            onChange={(e) => setArchId(e.target.value)}
            disabled={disabled || submitting}
          >
            {architectures.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        )}
        {arch?.description && <small>{arch.description}</small>}
      </label>

      {arch && Object.keys(arch.params).length > 0 && (
        <fieldset className="field" disabled={disabled || submitting}>
          <legend>Architecture parameters</legend>
          <div className="arch-params">
            {Object.entries(arch.params).map(([key, schema]) => {
              const value = archParams[key];
              const set = (v: unknown) =>
                setArchParams((p) => ({ ...p, [key]: v }));
              return (
                <label key={key} className="field">
                  <span>
                    {key}
                    {schema.description ? ` — ${schema.description}` : ''}
                  </span>
                  {schema.type === 'select' && schema.options ? (
                    <select
                      value={typeof value === 'string' ? value : ''}
                      onChange={(e) => set(e.target.value)}
                    >
                      {schema.options.map((o) => (
                        <option key={o} value={o}>
                          {o}
                        </option>
                      ))}
                    </select>
                  ) : schema.type === 'int' || schema.type === 'float' ? (
                    <input
                      type="number"
                      step={schema.type === 'float' ? 0.01 : 1}
                      min={schema.min}
                      max={schema.max}
                      value={typeof value === 'number' ? value : ''}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        set(Number.isFinite(v) ? v : value);
                      }}
                    />
                  ) : (
                    <input
                      type="text"
                      value={typeof value === 'string' ? value : String(value ?? '')}
                      onChange={(e) => set(e.target.value)}
                    />
                  )}
                </label>
              );
            })}
          </div>
        </fieldset>
      )}

      <fieldset className="field" disabled={disabled || submitting}>
        <legend>Hyperparameters</legend>
        <div className="hyper-grid">
          <label className="field">
            <span>Self-play games</span>
            <input
              type="number"
              min={1}
              step={1}
              value={hyper.games}
              onChange={(e) => setHyperNum('games', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Epochs</span>
            <input
              type="number"
              min={1}
              step={1}
              value={hyper.epochs}
              onChange={(e) => setHyperNum('epochs', e.target.value)}
            />
          </label>
          <label className="field">
            <span>MCTS sims / move</span>
            <input
              type="number"
              min={1}
              step={1}
              value={hyper.mcts_sims}
              onChange={(e) => setHyperNum('mcts_sims', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Learning rate</span>
            <input
              type="number"
              min={0.00001}
              max={0.999}
              step={0.0005}
              value={hyper.lr}
              onChange={(e) => setHyperNum('lr', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              step={1}
              value={hyper.batch_size}
              onChange={(e) => setHyperNum('batch_size', e.target.value)}
            />
          </label>
          <label className="field">
            <span>Temperature</span>
            <input
              type="number"
              min={0}
              max={10}
              step={0.05}
              value={hyper.temperature}
              onChange={(e) => setHyperNum('temperature', e.target.value)}
            />
          </label>
        </div>
      </fieldset>

      <label className="field">
        <span>Base checkpoint</span>
        <select
          value={baseCheckpoint}
          onChange={(e) => setBaseCheckpoint(e.target.value)}
          disabled={disabled || submitting}
        >
          <option value="">From scratch</option>
          {models.map((m) => (
            <option key={m.id} value={m.id}>
              {m.name}
              {m.architecture ? ` (${m.architecture})` : ''}
              {m.trained_games !== undefined ? ` · ${m.trained_games.toLocaleString()} games` : ''}
            </option>
          ))}
        </select>
      </label>

      {errors.length > 0 && (
        <ul className="error-text" role="alert">
          {errors.map((e) => (
            <li key={e}>{e}</li>
          ))}
        </ul>
      )}

      <button type="submit" className="btn primary large" disabled={disabled || submitting}>
        {submitting ? 'Starting run…' : '▶ Start run'}
      </button>
    </form>
  );
}

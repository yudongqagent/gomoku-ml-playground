// Eval report view: fetches GET /api/jobs/{id}/eval and renders the ladder
// table, the Elo-style estimate (labelled "relative to this ladder only"),
// tactical pass rate + failed positions, the checkpoint-vs-checkpoint
// result, and the plain-language verdict.
//
// 404 (no eval yet) is an empty state with an Evaluate button, not an error.
// Every number shown is measured; anything the server doesn't report is
// shown as "—", never invented.

import { useCallback, useEffect, useState } from 'react';
import {
  fetchJobEval,
  type EvalReport,
} from '../hooks/useTrainerApi';

export interface TrainerEvalReportProps {
  jobId: string;
  /** Whether an eval can be started for this job (e.g. a completed run). */
  canEvaluate: boolean;
  /** Start an eval job. `games` is the per-opponent game count, or
   * undefined for the server default. */
  onEvaluate: (games: number | undefined) => void;
  evaluating: boolean;
}

type LoadState =
  | { phase: 'loading' }
  | { phase: 'loaded'; report: EvalReport | null }
  | { phase: 'error'; message: string };

function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

export default function TrainerEvalReport({
  jobId,
  canEvaluate,
  onEvaluate,
  evaluating,
}: TrainerEvalReportProps) {
  const [state, setState] = useState<LoadState>({ phase: 'loading' });
  const [games, setGames] = useState<string>('20');

  const load = useCallback(async () => {
    setState({ phase: 'loading' });
    try {
      const report = await fetchJobEval(jobId);
      setState({ phase: 'loaded', report });
    } catch (e) {
      setState({
        phase: 'error',
        message: e instanceof Error ? e.message : 'Could not load the eval report.',
      });
    }
  }, [jobId]);

  useEffect(() => {
    void load();
  }, [load]);

  const startEval = () => {
    const n = Number(games);
    onEvaluate(games.trim() === '' || !Number.isFinite(n) || n <= 0 ? undefined : Math.floor(n));
  };

  if (state.phase === 'loading') {
    return (
      <div className="eval-report">
        <h4>Evaluation</h4>
        <p className="muted">Loading eval report…</p>
      </div>
    );
  }

  if (state.phase === 'error') {
    return (
      <div className="eval-report">
        <h4>Evaluation</h4>
        <p className="error-text" role="alert">{state.message}</p>
        <button type="button" className="btn small" onClick={() => void load()}>
          ↻ Retry
        </button>
      </div>
    );
  }

  const report = state.report;

  if (!report) {
    return (
      <div className="eval-report">
        <h4>Evaluation</h4>
        <p className="muted">
          No evaluation yet for this run — the server hasn&apos;t reported one.
        </p>
        {canEvaluate && (
          <div className="eval-trigger">
            <label className="field inline">
              <span>Games per opponent</span>
              <input
                type="number"
                min={1}
                step={1}
                value={games}
                onChange={(e) => setGames(e.target.value)}
                placeholder="Server default"
                disabled={evaluating}
              />
            </label>
            <button
              type="button"
              className="btn primary"
              onClick={startEval}
              disabled={evaluating}
            >
              {evaluating ? 'Starting evaluation…' : '▶ Evaluate this run'}
            </button>
            <small className="muted">
              Runs the model against the scripted ladder and tactical tests. The eval job
              appears in the runs list; its report shows up here when it finishes.
            </small>
          </div>
        )}
      </div>
    );
  }

  const cvc = report.checkpoint_vs_checkpoint;
  const cvcTotal = cvc ? cvc.wins + cvc.losses + cvc.draws : 0;

  return (
    <div className="eval-report">
      <div className="eval-report-head">
        <h4>Evaluation</h4>
        <button type="button" className="btn small" onClick={() => void load()}>
          ↻ Refresh
        </button>
      </div>
      {report.generated_at && (
        <p className="muted small">Report generated {formatDate(report.generated_at)}.</p>
      )}

      {report.verdict && (
        <blockquote className="eval-verdict">
          <strong>Verdict</strong>
          <p>{report.verdict}</p>
        </blockquote>
      )}

      {report.ladder.length > 0 && (
        <>
          <h5>Ladder results</h5>
          <table className="eval-table">
            <thead>
              <tr>
                <th>Opponent</th>
                <th>W–L–D</th>
                <th>Win rate</th>
                <th>95% CI</th>
                <th>Elo diff</th>
              </tr>
            </thead>
            <tbody>
              {report.ladder.map((row) => (
                <tr key={row.opponent}>
                  <td>{row.opponent}</td>
                  <td>
                    {row.wins}W–{row.losses}L–{row.draws}D
                  </td>
                  <td>
                    {row.win_rate !== undefined ? `${(row.win_rate * 100).toFixed(1)}%` : '—'}
                  </td>
                  <td>
                    {row.ci_low !== undefined && row.ci_high !== undefined
                      ? `${(row.ci_low * 100).toFixed(0)}–${(row.ci_high * 100).toFixed(0)}%`
                      : '—'}
                  </td>
                  <td>
                    {row.elo_diff_vs_opponent !== null && row.elo_diff_vs_opponent !== undefined
                      ? `${row.elo_diff_vs_opponent >= 0 ? '+' : ''}${row.elo_diff_vs_opponent.toFixed(0)}`
                      : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {report.elo_estimate !== null && report.elo_estimate !== undefined && (
        <p className="eval-elo">
          Elo estimate:{' '}
          <strong>
            {report.elo_estimate >= 0 ? '+' : ''}
            {report.elo_estimate.toFixed(0)}
          </strong>{' '}
          <span className="muted small">
            (relative to this ladder only — not an absolute or human-comparable rating)
          </span>
          {report.elo_estimate_note && (
            <span className="muted small"> · {report.elo_estimate_note}</span>
          )}
        </p>
      )}

      {report.tactical && (
        <div className="eval-tactical">
          <h5>Tactical tests</h5>
          <p>
            Pass rate:{' '}
            {report.tactical.pass_rate !== undefined ? (
              <strong>{(report.tactical.pass_rate * 100).toFixed(1)}%</strong>
            ) : (
              '—'
            )}
            {report.tactical.passed !== undefined && report.tactical.total !== undefined
              ? ` (${report.tactical.passed} of ${report.tactical.total} positions)`
              : ''}
          </p>
          {report.tactical.failed && report.tactical.failed.length > 0 && (
            <>
              <p className="muted small">Failed positions:</p>
              <ul className="eval-failed-list">
                {report.tactical.failed.map((f, i) => (
                  <li key={f.id ?? `${f.name ?? 'position'}-${i}`}>
                    <strong>{f.name ?? f.id ?? `Position ${i + 1}`}</strong>
                    {f.description ? ` — ${f.description}` : ''}
                    {f.detail ? ` (${f.detail})` : ''}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {cvc && cvcTotal > 0 && (
        <div className="eval-cvc">
          <h5>Checkpoint vs checkpoint</h5>
          <p>
            {cvc.checkpoint_a ?? 'New checkpoint'} vs {cvc.checkpoint_b ?? 'previous checkpoint'}:{' '}
            <strong>
              {cvc.wins}W–{cvc.losses}L–{cvc.draws}D
            </strong>{' '}
            ({((cvc.wins / cvcTotal) * 100).toFixed(1)}% win rate over {cvcTotal} games)
          </p>
        </div>
      )}

      {report.ladder.length === 0 && !report.tactical && !report.verdict && (
        <p className="muted">
          The eval report exists but carries no ladder, tactical, or verdict data this client
          understands.
        </p>
      )}

      {canEvaluate && (
        <div className="eval-trigger">
          <label className="field inline">
            <span>Games per opponent</span>
            <input
              type="number"
              min={1}
              step={1}
              value={games}
              onChange={(e) => setGames(e.target.value)}
              placeholder="Server default"
              disabled={evaluating}
            />
          </label>
          <button
            type="button"
            className="btn"
            onClick={startEval}
            disabled={evaluating}
          >
            {evaluating ? 'Starting evaluation…' : '↻ Re-evaluate'}
          </button>
        </div>
      )}
    </div>
  );
}

// Analysis panel: win rate for the side to move at the viewed position,
// top-5 suggested moves (click to preview on the board), threat summary,
// best-move / threat toggles, and the win-rate-over-game graph.

import { countThreats, type Threat } from './analysis';
import type { AiMoveResult, WinRatePoint } from './types';
import { formatWinRate } from './types';
import type { Point } from '../engine/gomoku';
import { WinRateGraph } from './WinRateGraph';

export interface AnalysisPanelProps {
  analysis: AiMoveResult | null;
  analyzing: boolean;
  onAnalyze: () => void;
  autoAnalyze: boolean;
  onAutoAnalyzeChange: (v: boolean) => void;
  showBestMove: boolean;
  onShowBestMoveChange: (v: boolean) => void;
  showThreats: boolean;
  onShowThreatsChange: (v: boolean) => void;
  threats: Threat[];
  viewPly: number;
  isLive: boolean;
  toMove: 1 | -1;
  gameOver: boolean;
  previewMove: Point | null;
  onPreviewMove: (move: Point | null) => void;
  winRateHistory: WinRatePoint[];
}

function coord(p: Point): string {
  return `${'abcdefghijklmno'[p.x]}${p.y + 1}`;
}

export default function AnalysisPanel({
  analysis,
  analyzing,
  onAnalyze,
  autoAnalyze,
  onAutoAnalyzeChange,
  showBestMove,
  onShowBestMoveChange,
  showThreats,
  onShowThreatsChange,
  threats,
  viewPly,
  isLive,
  toMove,
  gameOver,
  previewMove,
  onPreviewMove,
  winRateHistory,
}: AnalysisPanelProps) {
  const sideName = toMove === 1 ? 'Black' : 'White';
  const topMoves = (analysis?.topMoves ?? []).slice(0, 5);
  const best = analysis?.move ?? null;

  const threatSummary = (player: 1 | -1) => {
    const of = countThreats(threats, player, 'open-four');
    const f = countThreats(threats, player, 'four');
    const ot = countThreats(threats, player, 'open-three');
    return { of, f, ot, any: of + f + ot > 0 };
  };
  const blackThreats = threatSummary(1);
  const whiteThreats = threatSummary(-1);

  return (
    <section className="panel analysis-panel" aria-label="Position analysis">
      <h2 className="panel-title">Analysis</h2>
      <p className="panel-sub">
        {isLive ? `Live position · move ${viewPly}` : `Viewing move ${viewPly}`} · {sideName} to move
      </p>

      <div className="analysis-actions">
        <button type="button" className="btn primary" onClick={onAnalyze} disabled={analyzing || gameOver}>
          {analyzing ? 'Analyzing…' : '🔍 Analyze this position'}
        </button>
        <label className="check">
          <input
            type="checkbox"
            checked={autoAnalyze}
            onChange={(e) => onAutoAnalyzeChange(e.target.checked)}
          />
          Auto-analyze
        </label>
      </div>

      {analysis ? (
        <div className="analysis-result">
          <div className="winrate-big">
            <span className="winrate-value">{formatWinRate(analysis.winRate)}</span>
            <span className="winrate-label">win rate for {sideName} to move</span>
          </div>
          {(analysis.nodes > 0 || analysis.ms > 0) && (
            <p className="analysis-meta">
              {analysis.nodes.toLocaleString()} nodes · {analysis.ms} ms
              {analysis.backend ? ` · ${analysis.backend}` : ''}
            </p>
          )}
          {best && (
            <p className="best-move-line">
              Best move: <strong>{coord(best)}</strong>
              {topMoves.length > 0 && topMoves[0].move.x === best.x && topMoves[0].move.y === best.y
                ? ` (${formatWinRate(topMoves[0].winRate)})`
                : ''}
            </p>
          )}
          {topMoves.length > 0 && (
            <ol className="top-moves">
              {topMoves.map((t, i) => {
                const isPreview =
                  previewMove !== null && previewMove.x === t.move.x && previewMove.y === t.move.y;
                return (
                  <li key={`${t.move.x},${t.move.y}`}>
                    <button
                      type="button"
                      className={isPreview ? 'top-move selected' : 'top-move'}
                      onClick={() => onPreviewMove(isPreview ? null : t.move)}
                      title="Click to preview this move on the board"
                    >
                      <span className="top-move-rank">{i + 1}.</span>
                      <span className="top-move-coord">{coord(t.move)}</span>
                      <span
                        className="top-move-bar"
                        style={{ width: `${Math.round(t.winRate * 100)}%` }}
                      />
                      <span className="top-move-wr">{formatWinRate(t.winRate)}</span>
                      {t.visits !== undefined && (
                        <span className="top-move-visits">{t.visits.toLocaleString()} visits</span>
                      )}
                    </button>
                  </li>
                );
              })}
            </ol>
          )}
        </div>
      ) : (
        <p className="muted">
          {gameOver
            ? 'Game over — no analysis for a finished position.'
            : analyzing
              ? 'Analyzing the position…'
              : 'No analysis for this position yet.'}
        </p>
      )}

      <div className="toggle-row">
        <label className="check">
          <input
            type="checkbox"
            checked={showBestMove}
            onChange={(e) => onShowBestMoveChange(e.target.checked)}
          />
          Best-move marker
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={showThreats}
            onChange={(e) => onShowThreatsChange(e.target.checked)}
          />
          Threat highlights
        </label>
      </div>

      {showThreats && (
        <div className="threat-box">
          <h3>Threats on the board</h3>
          <div className="threat-grid">
            {([1, -1] as const).map((p) => {
              const t = p === 1 ? blackThreats : whiteThreats;
              return (
                <div key={p} className="threat-col">
                  <span className="threat-side">{p === 1 ? '⚫ Black' : '⚪ White'}</span>
                  {t.any ? (
                    <ul>
                      {t.of > 0 && <li className="threat-open-four">Open four × {t.of} — unstoppable</li>}
                      {t.f > 0 && <li className="threat-four">Four × {t.f} — one move from five</li>}
                      {t.ot > 0 && <li className="threat-open-three">Open three × {t.ot}</li>}
                    </ul>
                  ) : (
                    <span className="muted">No open threats</span>
                  )}
                </div>
              );
            })}
          </div>
          <p className="threat-legend">
            <span className="dot dot-red" /> open four <span className="dot dot-orange" /> four{' '}
            <span className="dot dot-yellow" /> open three
          </p>
        </div>
      )}

      <h3 className="graph-title">Win rate over the game (Black&apos;s perspective)</h3>
      <WinRateGraph history={winRateHistory} />
    </section>
  );
}

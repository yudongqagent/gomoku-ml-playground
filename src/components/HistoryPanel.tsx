// Move history: numbered move pairs; click any position to view (and
// analyze) that board state. "Back to live" returns to the current position.

import type { Point } from '../engine/gomoku';

export interface HistoryPanelProps {
  moves: Point[];
  viewPly: number;
  isLive: boolean;
  onSelectPly: (ply: number) => void;
  onGoLive: () => void;
  winner: 1 | -1 | 0;
}

function coord(p: Point): string {
  return `${'abcdefghijklmno'[p.x]}${p.y + 1}`;
}

export default function HistoryPanel({
  moves,
  viewPly,
  isLive,
  onSelectPly,
  onGoLive,
  winner,
}: HistoryPanelProps) {
  const pairs: { num: number; black?: Point; white?: Point }[] = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({ num: i / 2 + 1, black: moves[i], white: moves[i + 1] });
  }

  return (
    <section className="panel history-panel" aria-label="Move history">
      <div className="panel-head">
        <h2 className="panel-title">Moves</h2>
        {!isLive && (
          <button type="button" className="btn small" onClick={onGoLive}>
            ↩ Back to live
          </button>
        )}
      </div>

      {winner !== 0 && (
        <p className="result-banner">{winner === 1 ? '⚫ Black wins!' : '⚪ White wins!'}</p>
      )}
      {moves.length === 0 && <p className="muted">No moves yet — Black to play.</p>}

      {moves.length > 0 && (
        <ol className="move-list">
          <li>
            <button
              type="button"
              className={viewPly === 0 ? 'move-cell selected' : 'move-cell'}
              onClick={() => onSelectPly(0)}
            >
              Start
            </button>
          </li>
          {pairs.map((pair) => (
            <li key={pair.num} className="move-row">
              <span className="move-num">{pair.num}.</span>
              {pair.black && (
                <button
                  type="button"
                  className={
                    viewPly === (pair.num - 1) * 2 + 1 ? 'move-cell selected' : 'move-cell'
                  }
                  onClick={() => onSelectPly((pair.num - 1) * 2 + 1)}
                >
                  ⚫ {coord(pair.black)}
                </button>
              )}
              {pair.white && (
                <button
                  type="button"
                  className={viewPly === pair.num * 2 ? 'move-cell selected' : 'move-cell'}
                  onClick={() => onSelectPly(pair.num * 2)}
                >
                  ⚪ {coord(pair.white)}
                </button>
              )}
            </li>
          ))}
        </ol>
      )}
      <p className="muted small">Gomoku has no captures — every stone stays on the board.</p>
    </section>
  );
}

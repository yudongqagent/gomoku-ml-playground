// Self-play game viewer: load a recorded game JSON file
// ({ moves: [{x, y}, ...], meta?: {...} }) via file picker or paste, then step
// through it. Can also load the game into the main board.

import { useEffect, useMemo, useRef, useState } from 'react';
import { stateAtMove, type Point } from '../engine/gomoku';
import { legalPrefix } from '../hooks/useGomokuGame';
import Board from './Board';

export interface GameViewerProps {
  onLoadIntoGame: (moves: Point[]) => void;
}

interface ParsedGame {
  moves: Point[];
  meta: Record<string, unknown>;
}

function parseGameJson(text: string): ParsedGame {
  const data = JSON.parse(text) as unknown;
  if (!data || typeof data !== 'object') throw new Error('Expected a JSON object.');
  const rec = data as Record<string, unknown>;
  const rawMoves = rec.moves;
  if (!Array.isArray(rawMoves)) throw new Error('Expected a "moves" array of {x, y} points.');
  const moves: Point[] = rawMoves.map((m, i) => {
    if (!m || typeof m !== 'object') throw new Error(`Move ${i + 1} is not an object.`);
    const p = m as Record<string, unknown>;
    if (typeof p.x !== 'number' || typeof p.y !== 'number') {
      throw new Error(`Move ${i + 1} needs numeric x and y.`);
    }
    if (p.x < 0 || p.x > 14 || p.y < 0 || p.y > 14) {
      throw new Error(`Move ${i + 1} is off the 15×15 board.`);
    }
    return { x: Math.floor(p.x), y: Math.floor(p.y) };
  });
  const seen = new Set<string>();
  for (const m of moves) {
    const k = `${m.x},${m.y}`;
    if (seen.has(k)) throw new Error(`Duplicate move at ${'abcdefghijklmno'[m.x]}${m.y + 1}.`);
    seen.add(k);
  }
  const meta = rec.meta && typeof rec.meta === 'object' ? (rec.meta as Record<string, unknown>) : {};
  return { moves, meta };
}

function metaLine(meta: Record<string, unknown>): string | null {
  const parts: string[] = [];
  if (typeof meta.black === 'string') parts.push(`Black: ${meta.black}`);
  if (typeof meta.white === 'string') parts.push(`White: ${meta.white}`);
  if (typeof meta.result === 'string') parts.push(`Result: ${meta.result}`);
  if (typeof meta.winner === 'string') parts.push(`Winner: ${meta.winner}`);
  if (typeof meta.date === 'string') parts.push(meta.date);
  if (typeof meta.event === 'string') parts.push(meta.event);
  return parts.length > 0 ? parts.join(' · ') : null;
}

export default function GameViewer({ onLoadIntoGame }: GameViewerProps) {
  const [open, setOpen] = useState(false);
  const [paste, setPaste] = useState('');
  const [game, setGame] = useState<ParsedGame | null>(null);
  const [ply, setPly] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [truncated, setTruncated] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);
  const playTimer = useRef<ReturnType<typeof setInterval> | null>(null);

  const viewState = useMemo(
    () => (game ? stateAtMove(game.moves, ply) : null),
    [game, ply],
  );

  const loadText = (text: string) => {
    try {
      const parsed = parseGameJson(text);
      // Drop any moves after the game was already decided (stateAtMove would
      // throw on them) and say so, rather than failing the whole file.
      const legal = legalPrefix(parsed.moves);
      setTruncated(parsed.moves.length - legal.length);
      setGame({ ...parsed, moves: legal });
      setPly(0);
      setPlaying(false);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not parse that game file.');
    }
  };

  const handleFile = (file: File | undefined) => {
    if (!file) return;
    file
      .text()
      .then(loadText)
      .catch(() => setError('Could not read that file.'));
  };

  // Autoplay.
  useEffect(() => {
    if (playing && game) {
      playTimer.current = setInterval(() => {
        setPly((p) => {
          if (p >= game.moves.length) {
            setPlaying(false);
            return p;
          }
          return p + 1;
        });
      }, 900);
    }
    return () => {
      if (playTimer.current) clearInterval(playTimer.current);
    };
  }, [playing, game]);

  const lastMove = game && ply > 0 ? game.moves[ply - 1] : null;
  const info = game ? metaLine(game.meta) : null;

  return (
    <section className="panel viewer-panel" aria-label="Recorded game viewer">
      <button
        type="button"
        className="details-toggle panel-toggle"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? '▾' : '▸'} Self-play game viewer
      </button>
      {open && (
        <div className="viewer-body">
          <p className="muted small">
            Load a recorded game (<code>{'{ "moves": [{x, y}, …], "meta": {...} }'}</code>) to step
            through it.
          </p>
          <div className="viewer-load-row">
            <input
              ref={fileRef}
              type="file"
              accept=".json,application/json"
              className="sr-only"
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <button type="button" className="btn" onClick={() => fileRef.current?.click()}>
              📂 Choose game file
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => loadText(paste)}
              disabled={paste.trim() === ''}
            >
              Load pasted JSON
            </button>
          </div>
          <textarea
            className="viewer-paste"
            rows={3}
            placeholder='Paste game JSON here, e.g. {"moves": [{"x":7,"y":7},{"x":6,"y":7}], "meta": {"black": "AlphaZero", "white": "Heuristic"}}'
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
            aria-label="Paste game JSON"
          />
          {error && (
            <p className="error-text" role="alert">
              {error}
            </p>
          )}
          {game && truncated > 0 && (
            <p className="muted small">
              {truncated} move{truncated === 1 ? '' : 's'} after the game was decided{' '}
              {truncated === 1 ? 'was' : 'were'} ignored.
            </p>
          )}

          {game && viewState && (
            <>
              <div className="viewer-board">
                <Board
                  board={viewState.board}
                  toMove={viewState.toMove}
                  lastMove={lastMove}
                  winningLine={viewState.winningLine}
                  interactive={false}
                  animateKey={ply}
                  ariaLabel={`Recorded game, move ${ply} of ${game.moves.length}`}
                />
              </div>
              <p className="viewer-status">
                Move {ply} of {game.moves.length}
                {viewState.winner !== 0 &&
                  ` — ${viewState.winner === 1 ? 'Black' : 'White'} wins`}
                {info ? ` · ${info}` : ''}
              </p>
              <div className="viewer-transport">
                <button type="button" className="btn" onClick={() => setPly(0)} disabled={ply === 0}>
                  ⏮ Start
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => setPly((p) => Math.max(0, p - 1))}
                  disabled={ply === 0}
                >
                  ◀ Prev
                </button>
                <button
                  type="button"
                  className="btn primary"
                  onClick={() => {
                    if (ply >= game.moves.length) setPly(0);
                    setPlaying((v) => !v);
                  }}
                >
                  {playing ? '⏸ Pause' : '▶ Play'}
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => setPly((p) => Math.min(game.moves.length, p + 1))}
                  disabled={ply >= game.moves.length}
                >
                  Next ▶
                </button>
                <button
                  type="button"
                  className="btn"
                  onClick={() => setPly(game.moves.length)}
                  disabled={ply >= game.moves.length}
                >
                  End ⏭
                </button>
              </div>
              <button
                type="button"
                className="btn"
                onClick={() => onLoadIntoGame(game.moves)}
                title="Replace the current game with this recorded game"
              >
                ↪ Load into main board
              </button>
            </>
          )}
        </div>
      )}
    </section>
  );
}

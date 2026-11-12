// Live self-play game viewer: shows the run's current self-play game as the
// server generates it, move by move, on a read-only board.
//
// Data flow: the parent feeds `currentGame` from job detail (polling) and
// from SSE `game_move` events, which carry the full move list. The viewer
// itself only decides *which* prefix of that list to display.
//
// - "Follow live" (on by default) jumps to the newest move as it arrives.
// - Scrubbing back, stepping, or pausing turns follow-live off so the user
//   can inspect history without being yanked; "Latest" re-enables it.
// - Auto-playback replays the moves at a chosen speed.

import { useEffect, useMemo, useRef, useState } from 'react';
import { SIZE } from '../engine/gomoku';
import type { CurrentGame, GameWinner, JobType, RecentGame } from '../hooks/useTrainerApi';
import Board from './Board';

export interface TrainerLiveViewerProps {
  currentGame: CurrentGame | null;
  recentGames: RecentGame[];
  /** Whether the job is actively running (generating games). */
  isRunning: boolean;
  jobType: JobType;
}

const SPEEDS = [1, 2, 4, 8];

function winnerText(winner: GameWinner | null): string {
  if (winner === 'Black') return 'Black won';
  if (winner === 'White') return 'White won';
  if (winner === 'Draw') return 'Draw';
  return 'Winner not reported';
}

export default function TrainerLiveViewer({
  currentGame,
  recentGames,
  isRunning,
  jobType,
}: TrainerLiveViewerProps) {
  const [followLive, setFollowLive] = useState(true);
  const [displayedPly, setDisplayedPly] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const lastGameRef = useRef<number | null>(null);

  const moves = currentGame?.moves ?? [];
  const totalMoves = moves.length;

  // A new game starts: jump to its latest move and resume following.
  useEffect(() => {
    if (!currentGame) return;
    if (lastGameRef.current !== currentGame.game) {
      lastGameRef.current = currentGame.game;
      setDisplayedPly(currentGame.moves.length);
      setFollowLive(true);
      setPlaying(false);
    }
  }, [currentGame]);

  // Follow-live: track the newest move as it arrives.
  useEffect(() => {
    if (followLive) setDisplayedPly(totalMoves);
  }, [totalMoves, followLive]);

  // Auto-playback: advance one move per tick until the latest move.
  useEffect(() => {
    if (!playing) return;
    if (displayedPly >= totalMoves) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => setDisplayedPly((p) => Math.min(p + 1, totalMoves)), 1000 / speed);
    return () => clearTimeout(timer);
  }, [playing, displayedPly, totalMoves, speed]);

  const board = useMemo(() => {
    const arr = new Int8Array(SIZE * SIZE);
    for (let i = 0; i < displayedPly && i < moves.length; i++) {
      arr[moves[i].y * SIZE + moves[i].x] = i % 2 === 0 ? 1 : -1;
    }
    return arr;
  }, [moves, displayedPly]);

  const lastMove = displayedPly > 0 ? moves[displayedPly - 1] : null;
  const toMove: 1 | -1 = displayedPly % 2 === 0 ? 1 : -1;
  const atLatest = displayedPly >= totalMoves;
  const isFinished = currentGame?.status === 'finished';

  const scrubTo = (ply: number) => {
    setPlaying(false);
    setFollowLive(false);
    setDisplayedPly(Math.max(0, Math.min(ply, totalMoves)));
  };

  const jumpToLatest = () => {
    setFollowLive(true);
    setPlaying(false);
    setDisplayedPly(totalMoves);
  };

  if (!currentGame) {
    return (
      <div className="live-viewer empty">
        <h4>Live self-play game</h4>
        {jobType === 'train' ? (
          <p className="muted">
            This is a training-only run — no self-play games are generated, so there is no live
            game to watch.
          </p>
        ) : !isRunning ? (
          <p className="muted">
            No live game — this run isn&apos;t generating self-play games right now.
          </p>
        ) : (
          <p className="muted">
            Waiting for the first self-play game… moves appear here as the server generates them.
          </p>
        )}
        {recentGames.length > 0 && (
          <RecentGamesStrip recentGames={recentGames} />
        )}
      </div>
    );
  }

  return (
    <div className="live-viewer">
      <div className="live-viewer-head">
        <h4>Live self-play game</h4>
        <span className={`status-badge ${isFinished ? 'completed' : 'running'}`}>
          {isFinished ? 'Finished' : 'Playing'}
        </span>
      </div>
      <p className="muted small live-viewer-meta">
        Game {currentGame.game}
        {' · '}
        {isFinished ? winnerText(currentGame.winner) : `${toMove === 1 ? '● Black' : '○ White'} to move`}
        {' · '}
        Move {displayedPly} of {totalMoves}
        {!atLatest && ' · viewing history'}
      </p>

      <div className="live-viewer-board">
        <Board
          board={board}
          toMove={toMove}
          lastMove={lastMove}
          winningLine={null}
          interactive={false}
          animateKey={displayedPly}
          ariaLabel={`Self-play game ${currentGame.game}, move ${displayedPly} of ${totalMoves}`}
        />
      </div>

      <div className="playback-controls" role="group" aria-label="Playback controls">
        <button
          type="button"
          className="btn small"
          onClick={() => scrubTo(0)}
          disabled={displayedPly === 0}
          title="First move"
        >
          ⏮
        </button>
        <button
          type="button"
          className="btn small"
          onClick={() => scrubTo(displayedPly - 1)}
          disabled={displayedPly === 0}
          title="Previous move"
        >
          ◀
        </button>
        <button
          type="button"
          className="btn small primary"
          onClick={() => {
            if (playing) {
              setPlaying(false);
            } else {
              if (atLatest) setDisplayedPly(0);
              setFollowLive(false);
              setPlaying(true);
            }
          }}
          disabled={totalMoves === 0}
          title={playing ? 'Pause auto-playback' : 'Auto-play moves'}
        >
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>
        <button
          type="button"
          className="btn small"
          onClick={() => scrubTo(displayedPly + 1)}
          disabled={atLatest}
          title="Next move"
        >
          ▶
        </button>
        <button
          type="button"
          className="btn small"
          onClick={jumpToLatest}
          disabled={atLatest && followLive}
          title="Jump to the latest move and follow live"
        >
          ⏭ Latest
        </button>
        <label className="playback-speed">
          <span className="muted small">Speed</span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
        <label className="follow-live">
          <input
            type="checkbox"
            checked={followLive}
            onChange={(e) => {
              setFollowLive(e.target.checked);
              if (e.target.checked) {
                setPlaying(false);
                setDisplayedPly(totalMoves);
              }
            }}
          />
          Follow live
        </label>
      </div>

      <input
        type="range"
        className="playback-slider"
        min={0}
        max={Math.max(totalMoves, 1)}
        step={1}
        value={displayedPly}
        onChange={(e) => scrubTo(Number(e.target.value))}
        aria-label={`Move ${displayedPly} of ${totalMoves}`}
      />

      {recentGames.length > 0 && <RecentGamesStrip recentGames={recentGames} />}
    </div>
  );
}

function RecentGamesStrip({ recentGames }: { recentGames: RecentGame[] }) {
  // Newest first, most recent 8.
  const shown = [...recentGames].sort((a, b) => b.game - a.game).slice(0, 8);
  return (
    <div className="recent-games">
      <h5>Recent games</h5>
      <ul className="recent-games-strip">
        {shown.map((g) => (
          <li key={g.game} className="recent-game-chip">
            <strong>Game {g.game}</strong>
            <span>
              {g.length} moves · {g.winner ? winnerText(g.winner) : 'result not reported'}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

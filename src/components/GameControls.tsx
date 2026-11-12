// Game controls: mode picker, colour choice, new game, undo/redo, flip,
// and the AI-vs-AI demo transport (play/pause + speed).

import type { GameMode } from '../hooks/useGomokuGame';
import type { Player } from '../engine/gomoku';

export interface GameControlsProps {
  mode: GameMode;
  onModeChange: (mode: GameMode) => void;
  humanColor: Player;
  onHumanColorChange: (color: Player) => void;
  onNewGame: () => void;
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
  flipped: boolean;
  onFlip: () => void;
  aiThinking: boolean;
  aiVsAiPlaying: boolean;
  onToggleAiVsAi: () => void;
  aiVsAiSpeedMs: number;
  onSpeedChange: (ms: number) => void;
  gameOver: boolean;
}

export default function GameControls({
  mode,
  onModeChange,
  humanColor,
  onHumanColorChange,
  onNewGame,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
  flipped,
  onFlip,
  aiThinking,
  aiVsAiPlaying,
  onToggleAiVsAi,
  aiVsAiSpeedMs,
  onSpeedChange,
  gameOver,
}: GameControlsProps) {
  return (
    <div className="game-controls">
      <div className="control-row" role="group" aria-label="Game mode">
        <span className="control-label">Mode</span>
        <div className="segmented">
          <button
            type="button"
            className={mode === 'vs-ai' ? 'seg active' : 'seg'}
            aria-pressed={mode === 'vs-ai'}
            onClick={() => onModeChange('vs-ai')}
          >
            👤 vs 🤖
          </button>
          <button
            type="button"
            className={mode === 'vs-human' ? 'seg active' : 'seg'}
            aria-pressed={mode === 'vs-human'}
            onClick={() => onModeChange('vs-human')}
          >
            👥 2 Players
          </button>
          <button
            type="button"
            className={mode === 'ai-vs-ai' ? 'seg active' : 'seg'}
            aria-pressed={mode === 'ai-vs-ai'}
            onClick={() => onModeChange('ai-vs-ai')}
          >
            🤖 vs 🤖 Demo
          </button>
        </div>
      </div>

      {mode === 'vs-ai' && (
        <div className="control-row" role="group" aria-label="Play as">
          <span className="control-label">Play as</span>
          <div className="segmented">
            <button
              type="button"
              className={humanColor === 1 ? 'seg active' : 'seg'}
              aria-pressed={humanColor === 1}
              onClick={() => onHumanColorChange(1)}
            >
              ⚫ Black
            </button>
            <button
              type="button"
              className={humanColor === -1 ? 'seg active' : 'seg'}
              aria-pressed={humanColor === -1}
              onClick={() => onHumanColorChange(-1)}
            >
              ⚪ White
            </button>
          </div>
          <span className="control-hint">Black moves first.</span>
        </div>
      )}

      {mode === 'ai-vs-ai' && (
        <div className="control-row demo-transport" role="group" aria-label="Demo playback">
          <span className="control-label">Demo</span>
          <button
            type="button"
            className="btn primary"
            onClick={onToggleAiVsAi}
            disabled={gameOver}
          >
            {aiVsAiPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <label className="speed-label">
            Speed
            <input
              type="range"
              min={300}
              max={3000}
              step={100}
              value={aiVsAiSpeedMs}
              onChange={(e) => onSpeedChange(Number(e.target.value))}
              aria-label="Seconds between demo moves"
            />
            <span className="speed-value">{(aiVsAiSpeedMs / 1000).toFixed(1)}s / move</span>
          </label>
          {aiThinking && <span className="thinking-dot" aria-label="AI thinking">● thinking…</span>}
        </div>
      )}

      <div className="control-row actions-row">
        <button type="button" className="btn primary" onClick={onNewGame}>
          ↻ New game
        </button>
        <button type="button" className="btn" onClick={onUndo} disabled={!canUndo || aiThinking}>
          ↩ Undo
        </button>
        <button type="button" className="btn" onClick={onRedo} disabled={!canRedo || aiThinking}>
          ↪ Redo
        </button>
        <button
          type="button"
          className="btn"
          onClick={onFlip}
          aria-pressed={flipped}
          title="Flip the board"
        >
          ⇅ Flip
        </button>
      </div>
    </div>
  );
}

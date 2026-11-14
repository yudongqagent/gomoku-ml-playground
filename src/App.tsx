// Gomoku ML Playground — top-level app.
// Three views, switched by tabs (no router): Play, Trainer, Benchmark.
// Play: canvas board, game controls, AI panel, analysis, history, game viewer.
// Trainer: talks to the training server; the game works fully without it.
// Benchmark: measures the v0 scoring kernel (WebGPU shader + JS reference)
// on this device, with a self-check against frozen Python scores.
//
// Honesty: no copy here claims the AI beats the best human. Master is an
// AlphaZero-style level whose strength depends on the loaded model.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LEVEL_META, type LevelId } from './ai/levels';
import { useAiWorker } from './hooks/useAiWorker';
import { useGomokuGame } from './hooks/useGomokuGame';
import { useLocalModels } from './hooks/useLocalModels';
import { fetchModels, resolveApiUrl } from './hooks/useTrainerApi';
import Board from './components/Board';
import GameControls from './components/GameControls';
import AiPanel from './components/AiPanel';
import AnalysisPanel from './components/AnalysisPanel';
import HistoryPanel from './components/HistoryPanel';
import GameViewer from './components/GameViewer';
import TrainerView from './components/TrainerView';
import BenchmarkView from './components/BenchmarkView';
import { findThreats, threatSquareMap } from './components/analysis';
import {
  normalizeAiResult,
  perfFromResult,
  type AiMoveResult,
  type AiOptions,
  type AiPerf,
  type ModelOption,
  type WinRatePoint,
} from './components/types';
import type { Point } from './engine/gomoku';

type View = 'play' | 'trainer' | 'benchmark';

function defaultsForLevel(level: LevelId): AiOptions {
  const meta = LEVEL_META?.[level];
  return {
    sims: meta?.defaultSims ?? 200,
    timeMs: meta?.defaultTimeMs ?? 1000,
    temperature: 1,
  };
}

/** Composite key for the model picker (ids may collide across sources). */
export function modelKey(m: ModelOption): string {
  return `${m.source}:${m.id}`;
}

export default function App() {
  const [view, setView] = useState<View>('play');
  const game = useGomokuGame();
  const ai = useAiWorker();
  const { models: localModels, loading: localLoading, error: localError } = useLocalModels();

  const [level, setLevel] = useState<LevelId>('intermediate');
  const [options, setOptions] = useState<AiOptions>(() => defaultsForLevel('intermediate'));
  const [serverModels, setServerModels] = useState<ModelOption[]>([]);
  const [selectedModelKey, setSelectedModelKey] = useState<string | null>(null);

  const [aiThinking, setAiThinking] = useState(false);
  const [perf, setPerf] = useState<AiPerf | null>(null);
  const [aiError, setAiError] = useState<string | null>(null);

  const [analysis, setAnalysis] = useState<AiMoveResult | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [autoAnalyze, setAutoAnalyze] = useState(true);
  const [showBestMove, setShowBestMove] = useState(true);
  const [showThreats, setShowThreats] = useState(true);
  const [previewMove, setPreviewMove] = useState<Point | null>(null);
  const [winRateHistory, setWinRateHistory] = useState<WinRatePoint[]>([]);

  const [aiVsAiPlaying, setAiVsAiPlaying] = useState(false);
  const [aiVsAiSpeedMs, setAiVsAiSpeedMs] = useState(1200);

  const [toast, setToast] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const aiRef = useRef(ai);
  aiRef.current = ai;
  const aiRequestKeyRef = useRef<string>('');
  const aiInFlightRef = useRef(false);
  const analysisReqIdRef = useRef(0);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 5000);
  }, []);

  // ---- Models: bundled manifest + server registry, merged into one list. ----
  const mergedModels = useMemo<ModelOption[]>(
    () => [...localModels, ...serverModels],
    [localModels, serverModels],
  );
  const selectedModel = useMemo(
    () => mergedModels.find((m) => modelKey(m) === selectedModelKey) ?? null,
    [mergedModels, selectedModelKey],
  );

  const refreshServerModels = useCallback(async () => {
    try {
      const list = await fetchModels();
      // /api/models also lists bundled manifest models — those come from the
      // local manifest, so only server-trained models belong here.
      setServerModels(
        list
          .filter((m) => m.source !== 'bundled')
          .map((m) => ({
            id: m.id,
            name: m.name,
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
      // Server down is fine — the picker just shows bundled models.
    }
  }, []);

  // Best-effort server model fetch on mount so the picker is complete even
  // before the Trainer tab is opened. TrainerView refreshes the same list.
  useEffect(() => {
    void refreshServerModels();
  }, [refreshServerModels]);

  const handleServerModelsChange = useCallback((models: ModelOption[]) => {
    setServerModels(models);
  }, []);

  // ---- Derived view state. ----
  const { liveState, viewState, viewPly, isLive, moves, mode, humanColor, gameId } = game;
  const lastMove = viewPly > 0 ? moves[viewPly - 1] : null;
  const gameOver = liveState.winner !== 0;

  const threats = useMemo(
    () => (showThreats ? findThreats(viewState.board) : []),
    [viewState.board, showThreats],
  );
  const threatSquares = useMemo(
    () => (showThreats ? threatSquareMap(threats) : null),
    [threats, showThreats],
  );

  const isAiTurn = useMemo(() => {
    if (gameOver || !isLive) return false;
    if (mode === 'vs-ai') return liveState.toMove !== humanColor;
    if (mode === 'ai-vs-ai') return aiVsAiPlaying;
    return false;
  }, [gameOver, isLive, mode, liveState.toMove, humanColor, aiVsAiPlaying]);

  const aiRequestOptions = useMemo(
    () => ({
      sims: options.sims,
      timeMs: options.timeMs,
      temperature: options.temperature,
      ...(selectedModel ? { modelUrl: selectedModel.url } : {}),
    }),
    [options, selectedModel],
  );

  // ---- AI move loop. ----
  useEffect(() => {
    if (!isAiTurn) return;
    const key = `${gameId}:${moves.length}:${level}:${selectedModelKey ?? 'none'}`;
    if (aiRequestKeyRef.current === key || aiInFlightRef.current) return;
    aiInFlightRef.current = true;
    setAiThinking(true);
    setAiError(null);

    // Small delay so the previous stone's animation is visible, and so the
    // AI-vs-AI demo paces itself.
    const delay = mode === 'ai-vs-ai' ? aiVsAiSpeedMs : 250;
    let cancelled = false;
    const timer = setTimeout(() => {
      // Claim the key only when the request actually starts: the effect
      // cleanup (StrictMode double-invoke in dev, or a dep change) cancels
      // the timer, and the re-run must be able to request again.
      aiRequestKeyRef.current = key;
      const board = Array.from(liveState.board);
      const toMove = liveState.toMove;
      // History feeds the ONNX model's history planes (Master level).
      const history = moves.slice();
      aiRef.current
        .requestMove({ board, toMove, level, options: aiRequestOptions, history })
        .then((raw) => {
          if (cancelled) return;
          const result = normalizeAiResult(raw);
          if (!result) {
            setAiError('The AI returned a move I could not use. Try again.');
            return;
          }
          setPerf(perfFromResult(result));
          const ok = game.playMove(result.move.x, result.move.y);
          if (!ok) {
            setAiError(
              `The AI suggested an illegal move (${result.move.x}, ${result.move.y}). Try a new game.`,
            );
          }
        })
        .catch((e: unknown) => {
          if (cancelled) return;
          setAiError(e instanceof Error ? e.message : 'The AI move failed.');
        })
        .finally(() => {
          // Always release the in-flight flag, even for a cancelled request
          // — otherwise the next AI turn would be blocked forever.
          aiInFlightRef.current = false;
          if (!cancelled) setAiThinking(false);
        });
    }, delay);

    return () => {
      cancelled = true;
      clearTimeout(timer);
      aiInFlightRef.current = false;
      setAiThinking(false);
    };
    // liveState intentionally excluded: moves.length + gameId capture it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isAiTurn, gameId, moves.length, level, selectedModelKey, mode, aiVsAiSpeedMs, aiRequestOptions]);

  // Cancel any in-flight AI work when a new game starts or the mode changes.
  useEffect(() => {
    aiRequestKeyRef.current = '';
    return () => {
      try {
        aiRef.current.cancel();
      } catch {
        // Worker may already be gone; ignore.
      }
    };
  }, [gameId]);

  // Reset analysis state on a new game.
  useEffect(() => {
    setAnalysis(null);
    setWinRateHistory([]);
    setPreviewMove(null);
    setAiVsAiPlaying(false);
  }, [gameId]);

  // Clear the preview when the viewed position changes.
  useEffect(() => {
    setPreviewMove(null);
  }, [viewPly]);

  // ---- Analysis. ----
  const runAnalysis = useCallback(
    async (ply: number) => {
      if (viewState.winner !== 0) return;
      const reqId = ++analysisReqIdRef.current;
      setAnalyzing(true);
      try {
        const raw = await aiRef.current.analyze({
          board: Array.from(viewState.board),
          toMove: viewState.toMove,
          level,
          options: aiRequestOptions,
          history: moves.slice(0, ply),
        });
        if (analysisReqIdRef.current !== reqId) return; // superseded
        const result = normalizeAiResult(raw);
        if (!result) return;
        setAnalysis(result);
        // Record Black's win rate for the graph (analysis is side-to-move).
        const blackWr = viewState.toMove === 1 ? result.winRate : 1 - result.winRate;
        setWinRateHistory((prev) => {
          const next = prev.filter((p) => p.ply !== ply);
          next.push({ ply, blackWinRate: blackWr });
          next.sort((a, b) => a.ply - b.ply);
          return next;
        });
      } catch {
        // Analysis failure is non-fatal; the panel keeps its previous result.
      } finally {
        if (analysisReqIdRef.current === reqId) setAnalyzing(false);
      }
    },
    [viewState, level, aiRequestOptions, moves],
  );

  // Auto-analyze the viewed position (debounced), except while the AI is
  // thinking about a move — the worker is busy then.
  useEffect(() => {
    if (!autoAnalyze || gameOver) return;
    if (aiThinking) return;
    const timer = setTimeout(() => void runAnalysis(viewPly), 500);
    return () => clearTimeout(timer);
  }, [autoAnalyze, gameOver, aiThinking, viewPly, viewState, runAnalysis]);

  const handleAnalyzeClick = useCallback(() => {
    void runAnalysis(viewPly);
  }, [runAnalysis, viewPly]);

  // ---- Level / model selection. ----
  const handleLevelChange = useCallback((next: LevelId) => {
    setLevel(next);
    setOptions((prev) => ({ ...defaultsForLevel(next), temperature: prev.temperature }));
  }, []);

  const handleModelChange = useCallback((key: string | null) => {
    setSelectedModelKey(key);
  }, []);

  const handleLoadModelIntoGame = useCallback(
    (model: ModelOption) => {
      setServerModels((prev) => {
        if (model.source !== 'server') return prev;
        return prev.some((m) => m.id === model.id) ? prev : [...prev, model];
      });
      setSelectedModelKey(modelKey(model));
      if (level !== 'master') {
        setLevel('master');
        setOptions((prev) => ({ ...defaultsForLevel('master'), temperature: prev.temperature }));
      }
      setView('play');
      showToast(
        `Loaded “${model.name}” into Master level. Strength depends on the model — see its eval numbers in the Trainer.`,
      );
    },
    [level, showToast],
  );

  const handleLoadRecordedGame = useCallback(
    (recorded: Point[]) => {
      game.loadMoves(recorded);
      showToast(`Loaded a recorded game (${recorded.length} moves) onto the main board.`);
    },
    [game, showToast],
  );

  // ---- Render helpers. ----
  const turnText =
    liveState.winner !== 0
      ? liveState.winner === 1
        ? '⚫ Black wins — five in a row!'
        : '⚪ White wins — five in a row!'
      : `${liveState.toMove === 1 ? '⚫ Black' : '⚪ White'} to move · move ${moves.length + 1}`;

  const canInteract =
    isLive && !gameOver && !aiThinking && (mode === 'vs-human' || (mode === 'vs-ai' && liveState.toMove === humanColor));

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            ●
          </span>
          <div>
            <h1>Gomoku ML Playground</h1>
            <p className="brand-sub">Five in a row · search AI · AlphaZero-style training</p>
          </div>
        </div>
        <nav className="view-tabs" role="tablist" aria-label="Main views">
          <button
            type="button"
            role="tab"
            aria-selected={view === 'play'}
            className={view === 'play' ? 'tab active' : 'tab'}
            onClick={() => setView('play')}
          >
            🎮 Play
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={view === 'trainer'}
            className={view === 'trainer' ? 'tab active' : 'tab'}
            onClick={() => setView('trainer')}
          >
            🧠 Trainer
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={view === 'benchmark'}
            className={view === 'benchmark' ? 'tab active' : 'tab'}
            onClick={() => setView('benchmark')}
          >
            ⚡ Benchmark
          </button>
        </nav>
      </header>

      {view === 'play' ? (
        <main className="play-layout">
          <section className="board-column">
            <div className="status-bar" role="status">
              <span>{turnText}</span>
              {aiThinking && <span className="thinking-dot">● AI thinking…</span>}
            </div>
            {!isLive && (
              <div className="viewing-banner">
                <span>
                  Viewing move {viewPly} of {moves.length} — the board is paused here.
                </span>
                <button type="button" className="btn small" onClick={game.goLive}>
                  ↩ Back to live
                </button>
              </div>
            )}
            {aiError && (
              <p className="error-text" role="alert">
                {aiError}{' '}
                <button type="button" className="link-btn" onClick={() => setAiError(null)}>
                  Dismiss
                </button>
              </p>
            )}
            <Board
              board={viewState.board}
              toMove={viewState.toMove}
              lastMove={lastMove}
              winningLine={viewState.winningLine}
              interactive={canInteract}
              onPlay={(x, y) => {
                game.playMove(x, y);
              }}
              previewMove={previewMove}
              bestMove={showBestMove ? (analysis?.move ?? null) : null}
              showBestMove={showBestMove && analysis !== null}
              threatSquares={threatSquares}
              showThreats={showThreats}
              flipped={game.flipped}
              animateKey={viewPly}
            />
            <GameControls
              mode={mode}
              onModeChange={game.setMode}
              humanColor={humanColor}
              onHumanColorChange={game.setHumanColor}
              onNewGame={game.newGame}
              canUndo={game.canUndo}
              canRedo={game.canRedo}
              onUndo={game.undo}
              onRedo={game.redo}
              flipped={game.flipped}
              onFlip={game.toggleFlip}
              aiThinking={aiThinking}
              aiVsAiPlaying={aiVsAiPlaying}
              onToggleAiVsAi={() => setAiVsAiPlaying((v) => !v)}
              aiVsAiSpeedMs={aiVsAiSpeedMs}
              onSpeedChange={setAiVsAiSpeedMs}
              gameOver={gameOver}
            />
          </section>

          <aside className="side-column">
            <AiPanel
              level={level}
              onLevelChange={handleLevelChange}
              options={options}
              onOptionsChange={setOptions}
              models={mergedModels}
              modelsLoading={localLoading}
              modelsError={localError}
              selectedModelId={selectedModelKey}
              onModelChange={handleModelChange}
              onRefreshModels={() => void refreshServerModels()}
              perf={perf}
              status={ai.status}
              thinking={aiThinking}
            />
            <AnalysisPanel
              analysis={analysis}
              analyzing={analyzing}
              onAnalyze={handleAnalyzeClick}
              autoAnalyze={autoAnalyze}
              onAutoAnalyzeChange={setAutoAnalyze}
              showBestMove={showBestMove}
              onShowBestMoveChange={setShowBestMove}
              showThreats={showThreats}
              onShowThreatsChange={setShowThreats}
              threats={threats}
              viewPly={viewPly}
              isLive={isLive}
              toMove={viewState.toMove}
              gameOver={viewState.winner !== 0}
              previewMove={previewMove}
              onPreviewMove={setPreviewMove}
              winRateHistory={winRateHistory}
            />
            <HistoryPanel
              moves={moves}
              viewPly={viewPly}
              isLive={isLive}
              onSelectPly={game.viewPlyAt}
              onGoLive={game.goLive}
              winner={liveState.winner}
            />
            <GameViewer onLoadIntoGame={handleLoadRecordedGame} />
          </aside>
        </main>
      ) : view === 'trainer' ? (
        <main className="trainer-layout">
          <TrainerView
            localModels={localModels}
            onLoadModelIntoGame={handleLoadModelIntoGame}
            onServerModelsChange={handleServerModelsChange}
          />
        </main>
      ) : (
        <main className="benchmark-page">
          <BenchmarkView />
        </main>
      )}

      <footer className="app-footer">
        <p>
          Master is an AlphaZero-style level: its strength depends on the loaded model. The
          bundled model is a small demo net, not superhuman — train stronger nets in the
          Trainer to raise its ceiling.
        </p>
      </footer>

      {toast && (
        <div className="toast" role="status">
          {toast}
          <button type="button" className="toast-close" onClick={() => setToast(null)} aria-label="Dismiss">
            ✕
          </button>
        </div>
      )}
    </div>
  );
}

// Gomoku ML Playground — top-level app.
// Three views, switched by tabs (no router): Play, Trainer, Benchmark.
// Play: canvas board, game controls, AI panel, analysis, history, game viewer.
// The Play view supports the official swap rule (default) and freestyle:
// under swap, the opener places 3 stones (black, white, black), the second
// player picks a side, and White plays move 4 (see src/ai/swap.ts).
// Trainer: talks to the training server; the game works fully without it.
// The trainer stays freestyle. Benchmark: measures the v0 scoring kernel
// (WebGPU shader + JS reference) on this device, with a self-check against
// frozen Python scores.
//
// Honesty: no copy here claims the AI beats the best human. Master is an
// AlphaZero-style level whose strength depends on the loaded model.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { LEVEL_META, type LevelId } from './ai/levels';
import {
  chooseSwapColorTempo,
  createSeededRng,
  placeBalancedOpening,
} from './ai/swap';
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
import type { Player, Point } from './engine/gomoku';

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
  const {
    liveState,
    viewState,
    viewPly,
    isLive,
    moves,
    mode,
    humanColor,
    gameId,
    rules,
    humanOpens,
    swapPhase,
  } = game;
  const lastMove = viewPly > 0 ? moves[viewPly - 1] : null;
  const gameOver = liveState.winner !== 0;

  // ---- Swap rule: who opens, who chooses. ----
  // Vs AI: the human opens (places the 3 stones) or the AI does; the other
  // side picks a colour. Two-player: player 1 opens, player 2 chooses.
  // Demo: one AI opens, the other chooses. In freestyle mode the phases
  // are null and none of this applies.
  const aiIsOpener = mode === 'ai-vs-ai' || (mode === 'vs-ai' && !humanOpens);
  const aiIsChooser = mode === 'ai-vs-ai' || (mode === 'vs-ai' && humanOpens);
  const humanIsChooser = mode === 'vs-human' || (mode === 'vs-ai' && !humanOpens);
  // The AI is busy placing the opening or picking a colour: the board is
  // non-interactive and undo is disabled, like while it thinks. (In the
  // demo the AI only acts while playing.)
  const swapBusy =
    rules === 'swap' &&
    (mode !== 'ai-vs-ai' || aiVsAiPlaying) &&
    ((swapPhase === 'opening' && aiIsOpener) || (swapPhase === 'choice' && aiIsChooser));

  // The AI's balanced opening for this game, fixed when the game starts so
  // the staggered placement below always plays the same 3 stones. Seeded
  // per game (a per-session random base mixed with the game id), so every
  // game opens differently.
  const swapSeedBaseRef = useRef<number | null>(null);
  if (swapSeedBaseRef.current === null) {
    swapSeedBaseRef.current = Math.floor(Math.random() * 2 ** 31);
  }
  const aiOpening = useMemo(
    () =>
      rules === 'swap' && aiIsOpener
        ? placeBalancedOpening(
            createSeededRng((swapSeedBaseRef.current! + gameId * 2654435761) >>> 0),
          )
        : null,
    [rules, aiIsOpener, gameId],
  );

  // AI places the opening stones one at a time (a short pause between
  // them, so the opening is visible as it is built). The stones are moves
  // 1–3, so they go through the normal playMove path.
  useEffect(() => {
    if (rules !== 'swap' || swapPhase !== 'opening' || !aiOpening) return;
    if (mode === 'ai-vs-ai' && !aiVsAiPlaying) return;
    if (!isLive || gameOver) return;
    const next = aiOpening[moves.length];
    if (!next) return;
    const timer = setTimeout(() => {
      game.playMove(next.x, next.y);
    }, 400);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rules, swapPhase, aiOpening, mode, aiVsAiPlaying, isLive, gameOver, moves.length, gameId]);

  // AI picks a side with the tempo-aware judge (attack-only scores; ties
  // go to White, who plays move 4). Synchronous and fast; the pause is
  // just so the choice does not flash by unseen.
  useEffect(() => {
    if (rules !== 'swap' || swapPhase !== 'choice' || !aiIsChooser) return;
    if (mode === 'ai-vs-ai' && !aiVsAiPlaying) return;
    if (!isLive) return;
    const timer = setTimeout(() => {
      const pick = chooseSwapColorTempo(Array.from(liveState.board));
      game.chooseSwapColor(pick);
      if (mode === 'vs-ai') {
        // The AI picked `pick`, so the human plays the other colour.
        showToast(
          pick === 1
            ? 'The AI chose ⚫ Black — you play ⚪ White. White plays move 4.'
            : 'The AI chose ⚪ White — you play ⚫ Black. White plays move 4.',
        );
      }
    }, 600);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rules, swapPhase, aiIsChooser, mode, aiVsAiPlaying, isLive, gameId, moves.length]);

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
    // During the swap opening and choice phases the AI acts through the
    // placement / choice effects above, not through the move worker — and
    // before the choice, humanColor is not yet decided.
    if (swapPhase === 'opening' || swapPhase === 'choice') return false;
    if (mode === 'vs-ai') return liveState.toMove !== humanColor;
    if (mode === 'ai-vs-ai') return aiVsAiPlaying;
    return false;
  }, [gameOver, isLive, mode, liveState.toMove, humanColor, aiVsAiPlaying, swapPhase]);

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
  // thinking about a move — the worker is busy then — and except during
  // the swap opening / choice phases, where a "best move" marker would
  // bias the colour choice and the win-rate graph has nothing to record.
  useEffect(() => {
    if (!autoAnalyze || gameOver) return;
    if (aiThinking) return;
    if (swapPhase === 'opening' || swapPhase === 'choice') return;
    const timer = setTimeout(() => void runAnalysis(viewPly), 500);
    return () => clearTimeout(timer);
  }, [autoAnalyze, gameOver, aiThinking, viewPly, viewState, runAnalysis, swapPhase]);

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
      showToast(
        `Loaded a recorded game (${recorded.length} moves) onto the main board (freestyle — recorded games have no swap opening).`,
      );
    },
    [game, showToast],
  );

  // Human picks a side in the swap choice phase (second player in vs-ai
  // when the AI opened, or player 2 in two-player mode).
  const handleSwapChoice = useCallback(
    (color: Player) => {
      game.chooseSwapColor(color);
      if (mode === 'vs-ai') {
        showToast(
          color === 1
            ? 'You chose ⚫ Black — the AI plays ⚪ White. White plays move 4.'
            : 'You chose ⚪ White — the AI plays ⚫ Black. White plays move 4.',
        );
      }
    },
    [game, mode, showToast],
  );

  // ---- Render helpers. ----
  const turnText = (() => {
    if (liveState.winner !== 0) {
      return liveState.winner === 1
        ? '⚫ Black wins — five in a row!'
        : '⚪ White wins — five in a row!';
    }
    if (rules === 'swap' && swapPhase === 'opening') {
      if (aiIsOpener) return '🤖 AI is placing the opening stones…';
      // The opener places black, white, black — move 2 is the White stone.
      const colour = moves.length === 1 ? '⚪ White' : '⚫ Black';
      return `Place opening stone ${moves.length + 1} of 3 (${colour})`;
    }
    if (rules === 'swap' && swapPhase === 'choice') {
      if (aiIsChooser) return '🤖 AI is choosing a colour…';
      return 'Opening set — choose your colour. White plays move 4.';
    }
    return `${liveState.toMove === 1 ? '⚫ Black' : '⚪ White'} to move · move ${moves.length + 1}`;
  })();

  const canInteract = (() => {
    if (!isLive || gameOver || aiThinking || swapBusy) return false;
    if (swapPhase === 'choice') return false; // the choice is made with buttons, not the board
    if (mode === 'vs-human') return true; // player 1 places the opening, then both play
    if (mode === 'vs-ai') {
      if (swapPhase === 'opening') return humanOpens; // the opener places all 3 stones
      return liveState.toMove === humanColor;
    }
    return false;
  })();

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
            {rules === 'swap' && swapPhase === 'choice' && humanIsChooser && isLive && (
              <div className="swap-choice-banner" role="group" aria-label="Choose your colour">
                <span>
                  {mode === 'vs-human'
                    ? 'Player 2 — the opening is set. Choose your colour:'
                    : 'The opening is set. Choose your colour:'}
                </span>
                <button type="button" className="btn" onClick={() => handleSwapChoice(1)}>
                  ⚫ Play Black
                </button>
                <button type="button" className="btn" onClick={() => handleSwapChoice(-1)}>
                  ⚪ Play White
                </button>
                <span className="control-hint">White plays move 4.</span>
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
              rules={rules}
              onRulesChange={game.setRules}
              humanOpens={humanOpens}
              onHumanOpensChange={game.setHumanOpens}
              onNewGame={game.newGame}
              canUndo={game.canUndo}
              canRedo={game.canRedo}
              onUndo={game.undo}
              onRedo={game.redo}
              flipped={game.flipped}
              onFlip={game.toggleFlip}
              aiThinking={aiThinking || swapBusy}
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

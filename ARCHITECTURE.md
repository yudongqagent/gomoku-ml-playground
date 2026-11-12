# ARCHITECTURE

Gomoku ML Playground is four pieces that share one rules + model
contract: a web app, a TypeScript AI stack, a Python ML pipeline, and
a FastAPI training server.

```
┌──────────────────────────────┐   HTTP/SSE    ┌──────────────────────────────┐
│  Web app (Vite + React + TS) │◄─────────────►│  Training server (FastAPI)   │
│  Play view + Trainer view    │  :5173 → :8000│  job queue · SSE · registry  │
└──────┬───────────────┬───────┘               └──────┬───────────────┬───────┘
       │ Web Worker    │                              │ subprocess    │ model store
       ▼               │                              ▼               ▼
┌──────────────┐       │                      ┌──────────────┐  server/data/models/
│ TS AI stack  │       │                      │ ML pipeline  │  (onnx / pt / card)
│ engine+search│       │                      │ (PyTorch)    │
│ +MCTS+ONNX   │       │                      └──────────────┘
└──────────────┘       │                              │
       ▲               │                              │ run dirs
       │ model URL     │                              ▼
       │ (/models/ or  │                      ml/runs/<run-id>/  or
       │  server dl)   │                      server/data/jobs/<job-id>/
       └───────────────┘                      (config · metrics · ckpts)
```

## The shared contract

Everything hinges on two shared definitions. Change either one and
both sides must change together.

### 1. Rules — 15×15 standard Gomoku

Five-or-more in a row wins (overline counts), Black first, no
captures, no Renju forbidden moves, no Swap2. Implemented twice:

- TypeScript: `src/engine/gomoku.ts` (`GameState`, `newGame`,
  `applyMove`, `legalMoves`, `stateAtMove`, `checkWin`).
- Python: `ml/gomoku_ml/game.py` (same semantics, numpy board).

Parity is enforced by shared test vectors
(`ml/tests/parity_vectors.json`, used by both suites) including a
provably drawn full board.

### 2. Network I/O — the ONNX contract

| | |
|---|---|
| Input | `input`, float32 `[1, 8, 15, 15]` |
| Output | `policy` `[1, 225]` logits (row-major, index `y*15+x`), `value` `[1, 1]` tanh in `[-1, 1]` from the current player's perspective |
| ONNX opset | 13 (onnxruntime-web compatible) |
| Planes | 0 current-player stones, 1 opponent stones, 2 last-move one-hot, 3 side-to-move plane (all 1.0 if Black to move else 0.0), 4 own stones 2 plies ago, 5 opponent stones 2 plies ago, 6 own stones 4 plies ago, 7 opponent stones 4 plies ago (zeros if unavailable). Planes 0/1/4–7 are relative to the player to move (at that ply). |

Python side: `ml/gomoku_ml/encoding.py` builds the planes;
`ml/scripts/export_onnx.py` exports with these exact names/shapes.
TypeScript side: `src/ai/onnxModel.ts` builds the same planes from
`board + history` and reads the same outputs. A mismatch here is a
silent strength bug, so both sides have unit tests asserting the
planes by hand-computed cases.

## Web app (`src/`)

- `App.tsx` — two top-level views (no router): **Play** and **Trainer**.
- Play: `components/Board.tsx` (Canvas, DPR-crisp, animated placement,
  flip, markers), `AiPanel.tsx` (level cards from `LEVEL_META`,
  advanced controls, merged model picker), `AnalysisPanel.tsx`
  (win rate, top-5 moves, threat highlights, win-rate graph),
  `HistoryPanel.tsx`, `GameViewer.tsx` (recorded-game JSON playback).
  State: `hooks/useGomokuGame.ts` (`useReducer`, move list as source
  of truth, states derived via `stateAtMove`).
- Trainer: `TrainerView.tsx` + `TrainerRunForm.tsx` (presets,
  architecture picker, hyperparameters, base checkpoint),
  `TrainerJobMonitor.tsx` (live stats, loss curves, job controls),
  `TrainerLiveViewer.tsx` (live self-play board replay with playback
  controls), `TrainerEvalReport.tsx` (ladder table, Elo, tactics,
  verdict), `TrainerModelLibrary.tsx` (server + bundled models,
  one-click load into Master, release publish with dry-run first).
  API client: `hooks/useTrainerApi.ts` (SSE + 2 s polling fallback,
  defensive parsing, works when the server is down).
- The game is fully usable with no server: AI runs locally in a Web
  Worker and the Trainer view shows a "server not running" state.

## TypeScript AI stack (`src/ai/`, `src/engine/`, `src/workers/`)

All AI runs in `src/workers/ai.worker.ts` (ES-module Web Worker) so
the UI never blocks; `hooks/useAiWorker.ts` matches requests by id
and falls back to main-thread execution if Worker construction fails.

- `patterns.ts` — 572-window pattern eval, radius-2 candidate
  generation, win/block finders, logistic win-rate squash
  (`EVAL_SCALE = 3000`, documented as uncalibrated).
- `beginner.ts` — L1, 1-ply + ~15% slips among top candidates.
- `search.ts` — L2 (depth 3) / L3 (iterative deepening to depth 8):
  negamax alpha-beta, Zobrist transposition table, killer + history
  ordering, time-bounded; `findForcedWin` VCF threat-space search.
- `mcts.ts` — shared PUCT core + L4 Expert (heuristic priors +
  heuristic rollouts).
- `onnxModel.ts` — lazy `onnxruntime-web` import (never top-level, so
  Node/vitest never touches it), session cache by URL, WebGPU-first
  with WASM fallback, `getBackend()` reporting.
- `alphazero.ts` — L5 Master: net-guided PUCT; falls back to heuristic
  MCTS (`backend: 'mcts-fallback'`) when no model URL is given or
  loading/inference fails.
- `engine.ts` — `chooseMove` / `analyzePosition` with the shared
  pre-check (own immediate win → play it; opponent immediate win →
  best block), guaranteeing those properties for every level.
- `levels.ts` — `LEVEL_META`, the single source of truth for level
  copy, including the honest strength notes.

Known AI limits (see also LIMITS.md): no quiescence search; VCF
follows fours only; MCTS is single-threaded, batch=1; search
branching capped at ~12–14 moves; search win rates are an
uncalibrated logistic squash.

## ML pipeline (`ml/`)

Package `ml/gomoku_ml/` + scripts `ml/scripts/` + configs
`ml/configs/`. Full detail in `ml/README.md`.

- `game.py` rules (mirrors TS), `encoding.py` 8-plane tensors,
  `augment.py` dihedral symmetries (applied per batch, policy targets
  transformed identically).
- `architectures.py` — registry: `resnet` (AlphaZero-style),
  `mobilenet` (depthwise-separable, ~9× fewer params/block, fast
  on-device), `resnet-se` (ResNet + squeeze-and-excitation attention).
  Each entry exposes a builder + JSON param schema + description;
  checkpoints record arch name/params so export/eval rebuild the
  exact architecture. `list_architectures.py` prints the registry as
  JSON for the server/UI.
- `mcts.py` — PUCT MCTS (Dirichlet root noise, temperature sampling,
  resign support); `selfplay.py` — game generation → `.npz` shards +
  `current_game.json` (live feed, updated every move) +
  `data/manifest.json` (shard inventory, the resume source of truth).
- `train.py` — policy CE + value MSE + L2, AdamW, cosine/step LR,
  val split + best tracking, periodic checkpoints, graceful
  SIGTERM/SIGINT handling (finish current step, checkpoint, exit
  resumable). Resume via `resume` / `base_checkpoint` in config.
- `evaluate.py` + `baseline.py` — eval harness: ladder matches
  (random / heuristic / heuristic-MCTS, both colors, Wilson 95% CIs),
  checkpoint-vs-checkpoint, ladder-relative Elo estimate (anchors
  documented, not real Elo), tactical test suite
  (`tactical_positions.json`), JSON + Markdown eval reports with
  computed plain-language verdicts. Auto-eval runs at the END of
  training when `eval.auto` is set (mid-training eval would roughly
  double CPU run time); also runnable on demand. A Node cross-check
  driver (`scripts/play_match.mjs`, `npm run match:ts`) can play a
  checkpoint against the TypeScript search levels as an optional
  `--ts-opponent` eval mode.
- `config.py` — fail-fast YAML validation (exit 2, names the bad
  key). `runs.py` — run dirs, JSON event logging, model cards.
- Run layout (`ml/runs/<id>/` or a server job dir): `config.yaml`
  (resolved snapshot), `metrics.jsonl`, `run.log`, `eval.json`,
  `eval_report.{json,md}`, `model_card.md`, `data/*.npz`,
  `checkpoints/{checkpoint-<step>.pt, best.pt, final.pt}`.
- `compare_runs.py` — comparison table + `comparison.json` across runs.
- Exports: `export_onnx.py` (opset 13, verify vs PyTorch, `--quantize`
  fp16), `export_coreml.py` (`.mlpackage`, compute units ALL —
  code-complete, needs macOS + `coremltools`, not executed on Linux).
- Configs: `smoke.yaml` (minutes on CPU, demo only), `full.yaml`
  (real training, GPU), `longrun-cpu.yaml` (multi-hour CPU preset
  with measured throughput notes).

## Training server (`server/`)

FastAPI, one job at a time (FIFO — a deliberate v1 choice on shared
CPU hardware). Full detail in `server/README.md`.

- `app/main.py` — app factory, CORS (Vite origins), endpoints.
- `app/jobs.py` — job manager: resolves job requests into a YAML
  config in the job dir, launches the ML scripts as detached
  subprocesses under `ml/.venv/bin/python` with `--run-dir <job-dir>`
  (the job dir IS the ML run dir), parses stdout JSON event lines,
  persists `job.json` / `metrics.jsonl` / `job.log` to disk on every
  event, watches `current_game.json` for the live feed, computes
  throughput stats (games/hour, buffer size, MCTS stats).
- Job lifecycle: `queued → running → completed|failed|cancelled`,
  plus `paused` (SIGSTOP/SIGCONT) and `interrupted` (jobs left
  running across a server restart — resumable from the last
  checkpoint via `POST /api/jobs/{id}/resume`, never auto-resumed).
  Cancel terminates the subprocess (SIGTERM then SIGKILL).
- Eval jobs (`POST /api/jobs/{id}/evaluate`,
  `POST /api/models/{id}/evaluate`) run `ml/scripts/evaluate.py`
  against a checkpoint; `GET /api/jobs/{id}/eval` serves the latest
  report.
- `app/models_registry.py` — model store under
  `server/data/models/<model-id>/` (`model.json` + artifacts +
  `model_card.md`); `GET /api/models` merges server models with the
  bundled manifest (`public/models/models.json`); download endpoint
  serves ONNX/fp16/Core ML/checkpoint with CORS for browser fetch.
- `app/releases.py` + `scripts/publish_release.py` — GitHub release
  planning/upload: `gh` CLI preferred, REST API fallback; default
  repo `yudongqagent/gomoku-ml-playground` (overridable via
  `GOMOKU_MODEL_REPO` / `server/config.yaml`); tags `model-<run-id>`;
  dry-run is the default and warns when the repo doesn't exist yet.
- `scripts/jobctl.py` — CLI for `list/status/pause/resume/cancel/logs`.
- SSE (`GET /api/jobs/{id}/events`): `progress`, `metric`, `log`,
  `game_move`, `game_finished`, `paused`, `resumed`, `done`, `error`,
  15 s heartbeats; the UI falls back to polling if SSE fails.

## Data flow: a training run

1. UI (or curl) creates a job → server writes resolved `config.yaml`
   into `server/data/jobs/<job-id>/` and queues it.
2. Runner launches `selfplay.py --run-dir <job-dir>` → games stream
   to `data/*.npz` + `current_game.json`; progress events flow over
   SSE to the Trainer live viewer.
3. `train.py --run-dir <job-dir>` trains from the shards, writing
   checkpoints + metrics; auto-eval runs at checkpoints.
4. `evaluate.py` produces the eval report; `export_onnx.py` (+ fp16,
   + Core ML attempt) exports the best checkpoint.
5. The model is registered in `server/data/models/` and appears in
   the Trainer model library; one click loads it into Master.
6. Optionally, publish it as a GitHub release (dry-run first).

## What is NOT built (deliberate gaps)

- No multiplayer / accounts / cloud training — single-user, local
  server by design.
- No Renju / Swap2 rules, no quiescence search, no distributed
  self-play, no GPU scheduling — all documented as future work in
  LIMITS.md.
- No live-browser end-to-end test of the UI (verified at the
  code/contract level + served-build smoke checks).

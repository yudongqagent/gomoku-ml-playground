# Gomoku ML Playground

A web-based Gomoku game and ML playground: play against five AI levels
(from a blundering Beginner to an AlphaZero-style Master), analyse any
board position, and — the real point of the project — **train your own
neural networks from self-play, watch them learn live, evaluate them
honestly, and deploy them on-device**.

> **Honest strength note (read first).** Nothing in this repo today beats
> the best human. The strongest out-of-the-box play is the search-based
> Expert/Advanced levels (strong club/amateur level). The bundled neural
> net is a tiny demo net trained on 8 self-play games — it loses to a
> simple heuristic. True superhuman play needs AlphaZero-scale training
> (millions of self-play games, thousands of GPU-hours); this repo builds
> the full pipeline that can get there and tells you exactly what it
> takes. See [LIMITS.md](LIMITS.md).

## Quickstart

### 1. Play the game (web app only, no server needed)

```bash
npm install
npm run dev        # http://localhost:5173
```

The game is fully playable without the training server: all five AI
levels run in a Web Worker, and the Master level falls back to
search-based MCTS when no neural model is loaded. The bundled demo net
(`gomoku-smoke-v1`, 8 self-play games) is available in the model picker.

```bash
npm test           # 51 engine + AI tests
npm run build      # production build
npm run preview    # serve the build on http://localhost:4173
```

### 2. Train models (training server)

```bash
# ML environment (one-time)
python3 -m venv ml/.venv
ml/.venv/bin/pip install -r ml/requirements.txt   # CPU torch

# Training server (one-time)
python3 -m venv server/.venv
server/.venv/bin/pip install -r server/requirements.txt

# Start the server
cd server && .venv/bin/uvicorn app.main:app --port 8000
```

Then open the **Trainer** tab in the web app: pick a preset
(`smoke`, `longrun-cpu`), an architecture, hyperparameters — and watch
the run live (self-play board replay, loss curves, games/hour, eval
results). Or drive it from the CLI — see
[server/README.md](server/README.md) and the "First multi-hour run"
section below.

Verify the ML pipeline end-to-end (a few minutes on CPU):

```bash
ml/.venv/bin/python ml/scripts/run_smoke.py
ml/.venv/bin/python -m pytest ml/tests -q
```

### 3. Load a trained model into the game

Finished models appear in the Trainer **model library** — click
**Load into game** and the Master level will use that model's ONNX
file (served by the server, run in-browser via ONNX Runtime Web with
WebGPU, WASM fallback). You can also drop any compatible ONNX file
into `public/models/` and list it in `public/models/models.json`.

## What is in this repo

| Area | Path | What it is |
|---|---|---|
| Web app | `src/` | Vite + React + TypeScript. Play view (Canvas board, 5 AI levels, analysis) + Trainer view (run control, live self-play viewer, loss curves, eval reports, model library). |
| Rules + AI | `src/engine/`, `src/ai/` | 15×15 Gomoku engine (shared contract with the Python side); pattern eval, alpha-beta + VCF threat search, MCTS, AlphaZero-style PUCT + ONNX inference. All AI runs in a Web Worker. |
| ML pipeline | `ml/` | PyTorch self-play + training. Architecture registry (`resnet`, `mobilenet`, `resnet-se`), config-driven runs, run tracking, eval harness (ladder + tactical suite + ladder-relative Elo), ONNX / fp16 / Core ML export. |
| Training server | `server/` | FastAPI: job queue (FIFO), SSE progress, live self-play feed, pause/resume/cancel, restart recovery, model registry, GitHub release upload (dry-run by default). |
| iOS notes | `docs/ios-coreml.md` | Core ML export + sample Swift integration for the Neural Engine (A20 Pro). |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture, and the
per-area READMEs (`ml/README.md`, `server/README.md`,
`src/ai/README.md`) for detail.

## The AI levels

| Level | Engine | Honest strength |
|---|---|---|
| L1 Beginner | 1-ply pattern heuristic, occasional slips | Casual |
| L2 Intermediate | Alpha-beta, depth 3 | Casual–club |
| L3 Advanced | Iterative-deepening alpha-beta + VCF threat search, transposition table | Strong club |
| L4 Expert | MCTS + heuristic policy/rollouts | Strong club |
| L5 Master (AlphaZero-style) | PUCT MCTS + neural policy/value net (ONNX); falls back to L4 when no model is loaded | Depends on the loaded model. With the demo net: ≈ L4. Superhuman only after AlphaZero-scale training (see LIMITS.md). |

Every level takes an immediate win when available and blocks an
opponent open four (enforced by a shared pre-check, covered by tests).

## Training, honestly

- **Smoke run** (~2 min on CPU): 8 self-play games, tiny net. Proves the
  pipeline works. The resulting model was fully evaluated and is
  **below heuristic**: 0W–40L vs the heuristic and vs heuristic-MCTS,
  30W–10L vs random, 0/20 on the tactical suite, verdict "Below
  heuristic". It is a demo, not a player.
- **A few hours on CPU** (the `longrun-cpu` preset): a small net, a few
  hundred self-play games. Expect at best a weak club-ish player that
  sometimes beats the heuristic — not an expert, nowhere near superhuman.
  Measured on the build VM (~1.5 shared CPUs): ~150 self-play
  games/hour at 100 sims; one full round (200 games + 20 epochs +
  auto-eval) ≈ 2–2.5 hours. See `ml/configs/longrun-cpu.yaml` for the
  measured throughput notes.
- **Superhuman**: ~1M+ self-play games, ~1,000–10,000 GPU-hours on
  A100/H100-class hardware. Train on GPU servers; deploy the frozen
  model on device. See [LIMITS.md](LIMITS.md) and `ml/README.md`.

## First multi-hour run

Start the server (above), then either use the Trainer tab
(preset `longrun-cpu`) or the API:

```bash
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' -d '{
  "type": "full", "name": "longrun-1", "preset": "longrun-cpu"
}'
```

Or run the preset directly from the CLI (same pipeline, no server):

```bash
ml/.venv/bin/python ml/scripts/selfplay.py --config longrun-cpu --run-dir ml/runs/<id>
ml/.venv/bin/python ml/scripts/train.py --config longrun-cpu --run-dir ml/runs/<id>
# eval runs automatically at the end of training (eval.auto: true)
```

Expected throughput (measured on the build VM, ~1.5 shared CPUs;
Yudong's hardware will differ, probably faster): ~150 self-play
games/hour at 100 sims; one full round (200 games + 20 epochs +
auto-eval) ≈ 2–2.5 hours. A 6-hour run is ~2 rounds (~400 games,
~28k samples). Honest expectation: at best a weak club-ish player
that beats the heuristic most of the time and sometimes beats
heuristic-MCTS — NOT superhuman, NOT expert.

Monitor with the Trainer tab (live board, loss curves, eval), or:

```bash
cd server && .venv/bin/python scripts/jobctl.py status <job-id>
cd server && .venv/bin/python scripts/jobctl.py logs <job-id> --tail 50
```

Pause/resume/cancel from the Trainer tab or `jobctl.py pause|resume|
cancel`. If the server restarts mid-run, the job is marked
`interrupted` — resume it from its last checkpoint with
`POST /api/jobs/{id}/resume` (or `jobctl.py resume`). See
[server/README.md](server/README.md) for the full operations guide.

## On-device inference & the A20 Pro

- **In the browser**, the fastest path is ONNX Runtime Web with the
  WebGPU execution provider (WASM fallback). The browser cannot access
  the Neural Engine directly — WebGPU is the honest best web path.
- **Natively**, export to Core ML (`.mlpackage`, compute units ALL) and
  run on the Neural Engine. See `ml/scripts/export_coreml.py` and
  [docs/ios-coreml.md](docs/ios-coreml.md) for the export command and a
  Swift integration sketch. Core ML export needs macOS + `coremltools`
  and was **not** executed on the Linux build VM (code-complete, not
  run here).

## Publishing a model to GitHub releases

Target repo (default, overridable via `GOMOKU_MODEL_REPO` or
`server/config.yaml`): **`yudongqagent/gomoku-ml-playground`**.
That repo does not exist yet — it must be created (empty, public)
before a real upload can work; dry-runs work without it and warn
about the missing repo.

```bash
# Dry-run (default, safe — prints what WOULD be uploaded, calls nothing)
cd server && .venv/bin/python scripts/publish_release.py --model-id <id> --dry-run

# Real upload (only after the repo exists; needs `gh auth login` or GITHUB_TOKEN)
cd server && .venv/bin/python scripts/publish_release.py --model-id <id>
```

Releases are tagged `model-<run-id>` and carry the checkpoint, ONNX
(+ fp16 if present), Core ML package zipped (if present), and the
model card. The Trainer model library also offers Publish with a
dry-run preview first. **No release has been published from this repo
yet.**

## Rules

Standard 15×15 Gomoku: five-or-more in a row wins (overline counts),
Black moves first, no captures, no Renju forbidden moves, no Swap2.
The TypeScript engine and the Python rules mirror each other, with
shared parity test vectors (`ml/tests/parity_vectors.json`).
Renju/Swap2 are future options, not implemented.

## Repo status

Local git repo, no `origin` remote set yet — the target GitHub repo
`yudongqagent/gomoku-ml-playground` has not been created. Once it
exists, add the remote and push from this directory.

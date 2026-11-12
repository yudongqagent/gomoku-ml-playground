# Gomoku ML Playground — training server

FastAPI backend that runs training on the server (never in the browser),
owns a FIFO job queue, streams progress to the web UI over SSE, stores
checkpoints + exported models, and can publish models to GitHub releases.

Honesty rule (binding): model cards and API responses present honest eval
numbers only. No copy anywhere claims a model beats any human — models are
evaluated against scripted opponents (random / heuristic) unless a real
human eval has been run and recorded.

## Quickstart

```bash
cd server
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload --port 8000
```

The server listens on http://localhost:8000 by default. The web app
(Vite dev server on http://localhost:5173) calls it at `VITE_API_URL`
(default `http://localhost:8000`).

The ML pipeline itself runs under the ML venv (`ml/.venv`, owned by the
ML workstream — this server never installs packages into it, it only
invokes `ml/.venv/bin/python` as a subprocess). If `ml/.venv` is absent
the server falls back to its own interpreter, which will fail on ML
steps that need torch.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | `{"status": "ok"}` |
| GET | `/api/architectures` | Architecture registry from the ML package (`resnet`, `mobilenet`, `resnet-se`; falls back to a static ResNet-only list if the registry is unavailable — never 500s; cached 5 min) |
| POST | `/api/jobs` | Create a job → 201 `{id, status: "queued"}` |
| GET | `/api/jobs` | Job summaries, newest first |
| GET | `/api/jobs/{id}` | Full job detail: metrics, games, eval, log tail |
| POST | `/api/jobs/{id}/cancel` | Cancel a queued/running/paused job (records `cancelled_at`) |
| POST | `/api/jobs/{id}/pause` | Pause a running job (SIGSTOP: process stays alive, no CPU) → 409 unless running |
| POST | `/api/jobs/{id}/resume` | Resume a paused job (SIGCONT) or re-queue an interrupted/failed job from its last checkpoint → 409 otherwise |
| POST | `/api/jobs/{id}/evaluate` | Create an eval job for the job's latest checkpoint → 201 `{id, status}` |
| GET | `/api/jobs/{id}/eval` | Latest eval report for the job (404 if none yet) |
| GET | `/api/jobs/{id}/events` | SSE stream of progress/metric/log/done/error events (heartbeat every ~15s; closes on terminal state) |
| GET | `/api/presets` | Training presets from `ml/configs/*.yaml` |
| POST | `/api/models/{id}/evaluate` | Create an eval job for a registered model's checkpoint → 201 `{id, status}` |
| GET | `/api/models` | Registered models (server) + bundled models from `public/models/models.json` |
| GET | `/api/models/{id}` | Model detail incl. model card text |
| GET | `/api/models/{id}/download?format=onnx\|onnx-fp16\|coreml\|checkpoint` | Download an artifact (404 if the format doesn't exist) |
| POST | `/api/models/{id}/release` | `{"dry_run": true}` (default) plans the release; `{"dry_run": false}` uploads it for real |

### Creating a job

```bash
curl -X POST http://localhost:8000/api/jobs -H 'Content-Type: application/json' -d '{
  "type": "full",
  "name": "smoke run",
  "architecture": "resnet",
  "arch_params": {"channels": 16, "blocks": 2},
  "hyperparameters": {"games": 8, "epochs": 3, "mcts_sims": 32}
}'
```

Job types:

- `selfplay` — generate self-play games only (no model produced).
- `train` — train on the latest completed self-play data (fails with a
  clear error if no completed selfplay/full job has data yet), then eval
  + export + register a model. Accepts `base_checkpoint` (a model id or
  a checkpoint path) to resume from.
- `full` — self-play → train → eval → export (ONNX + fp16 + Core ML
  attempt) → register a model.

The server writes a resolved YAML config (based on
`ml/configs/smoke.yaml` plus the request's overrides — architecture goes
into the `architecture: {name, params}` section the ML scripts read)
into the job dir as `config.yaml` before running the pipeline. The job
dir IS the ML run dir: the scripts are invoked with
`--run-dir <job-dir>`, so they write `metrics.jsonl`, `eval.json`,
`model_card.md` and `checkpoints/` (including `best.pt` when a val
split is used) into it. Export prefers `checkpoints/best.pt` when
present, else `final.pt`.

`arch_params` are validated against the registry's parameter schema
(unknown keys, non-integers, out-of-range values → 400), mirroring
`gomoku_ml.architectures`.

## Presets

`GET /api/presets` lists one preset per `ml/configs/*.yaml` (currently
`smoke` and `full`; `longrun-cpu` does not exist yet — the ML
workstream is adding it). A preset block in the YAML (`preset: {name,
description, throughput_note}`) is used when present; otherwise the
entry falls back to filename-based values. Create a job with
`"preset": "<id>"` (unknown preset → 422); request overrides
(architecture / hyperparameters) still apply on top of the preset's
config.

## Job lifecycle

```
queued -> running -> completed | failed | cancelled
running <-> paused            (SIGSTOP / SIGCONT)
running/paused --server restart--> interrupted --resume--> queued
```

- One job runs at a time (FIFO, concurrency=1 — a deliberate v1 choice;
  training is CPU/GPU-bound).
- Progress is parsed from the ML scripts' stdout JSON event lines
  (`{"event": "selfplay_game"|"epoch"|"train_step"|"eval_result"|...}` —
  see `gomoku_ml/runs.py`). Plain-text lines (`game i/N`, `epoch i/N:
  loss=...`, `vs <opponent>: XW-YL-ZD`) are still parsed as a fallback
  and always kept as log events. `eval.json` in the run dir is
  authoritative for eval results when present.
- Torch threads are set per phase via `TORCH_THREADS` (self-play 1,
  train 2, eval 1, export 1): one tiny forward at a time makes intra-op
  parallelism pure overhead on this small VM. The operator can override
  with `TORCH_THREADS` in the server's environment.
- Observed speed on this VM (shared ~1.5 CPU): one self-play game at 16
  sims/move took ~10 min (578 s for a 73-move game) with the default
  2-thread setting. Even smoke jobs are slow here — this is an ML
  pipeline / hardware reality, not a server bug. Serious training needs
  a GPU machine.
- Pause / resume: `POST /api/jobs/{id}/pause` sends SIGSTOP to the job
  subprocess (alive, no CPU); `POST /api/jobs/{id}/resume` sends
  SIGCONT. Pausing a non-running job (or resuming a running/queued/
  completed/cancelled one) returns 409. Cancel works on queued,
  running, paused and interrupted jobs; a paused job is SIGCONTed
  before the kill so the SIGTERM is delivered.
- Restart recovery: on startup, jobs left `running`/`paused` are
  marked `interrupted` with a note pointing at
  `POST /api/jobs/{id}/resume`. They are NOT auto-resumed — the
  operator decides (a restart may mean the machine was repurposed, and
  a blind resume could burn hours of CPU). Queued jobs are picked back
  up. Persisted `job.json` is the source of truth: it is rewritten on
  every event (atomic temp-file + rename), `job.log` gets every stdout
  line, and `metrics.jsonl` gets every server-normalized event (the ML
  run appends its own native events to the same file; server lines
  carry `"source": "server"` and a `"type"` key instead of `"event"`).

### Resume semantics

Resuming an interrupted/failed job re-queues it; the worker re-runs it
from the phase it was interrupted in, reusing completed phases'
artifacts (tracked in `completed_phases` in `job.json`):

- `selfplay`: the ML script is resume-safe by design — numbering
  continues after existing shards and it TOPS UP to the configured
  total (a job configured for 4 games, interrupted after 3, ends with
  4 shards, not 7; `data/manifest.json` is the source of truth for
  what the run has). The server does not reset its counters on
  resume, and per-game stats dedupe by game number. A game in progress
  at interruption is lost (the script abandons it at the next move
  boundary on SIGTERM). Skipped entirely when all shards exist.
- `train`: re-run with the newest checkpoint in the job dir
  (`checkpoints/*.pt` by mtime) via a `resume-config.yaml`
  (config + `resume: <checkpoint>`) and `--epochs` set to the
  REMAINING epochs. This matters because `train.py` runs `epochs` MORE
  epochs from the resumed epoch — passing the full count would
  overshoot the configured total. Completed epochs are read from the
  job's recorded metrics. Resume restores optimiser state + step count
  (see `ml/README.md`). Skipped when train already completed.
- `eval` / `export`: re-run whenever train ran in this invocation or
  they never completed. Export is skipped if the model is already
  registered.
- `eval` jobs just re-run the eval.

### Long-run operations

Jobs are meant to run for hours unattended. Monitor via the Trainer UI
(SSE live feed + `GET /api/jobs/{id}` detail, which includes
`games_per_hour`, `buffer_size`, `mcts` stats, `current_game` and
`eval_history`) or from the shell:

```bash
python scripts/jobctl.py list
python scripts/jobctl.py status <job-id>
python scripts/jobctl.py pause <job-id>
python scripts/jobctl.py resume <job-id>
python scripts/jobctl.py cancel <job-id>
python scripts/jobctl.py logs <job-id> [--tail N]
# --base-url http://localhost:8000 is the default
```

`jobctl` is API-based and needs the server running.

### Evaluating a checkpoint or model

```bash
curl -X POST http://localhost:8000/api/jobs/<job-id>/evaluate \
  -H 'Content-Type: application/json' -d '{"games": 10}'
curl -X POST http://localhost:8000/api/models/<model-id>/evaluate \
  -H 'Content-Type: application/json' -d '{"games": 10, "opponents": ["random", "heuristic"]}'
```

Both create a standalone `eval` job (type `eval`, linked via
`parent_job_id` / `model_id`) that runs `ml/scripts/evaluate.py`
against the job's latest checkpoint (best.pt preferred) or the model's
checkpoint. The report lands in the eval job dir and, for job evals, in
the parent job dir as `eval.json` (updating the parent's `eval` +
`eval_history`). `GET /api/jobs/{id}/eval` returns the latest report
(`{ladder, elo_estimate, tactical, verdict, generated_at}` — the
current eval script writes a plain ladder dict, so the other fields are
null until the ML side produces them; 404 when no eval exists yet).
Only scripted opponents (`random`, `heuristic`) are supported here;
`checkpoint` opponents need a `--vs-checkpoint` this endpoint doesn't
take (400). Training jobs also record their built-in eval phase in
`eval_history` (`[{checkpoint, at, summary}]` in the job detail).

### Live self-play feed

The job detail includes `current_game` (the game in progress) and the
SSE stream emits `game_move` (full move list) and `game_finished`
events — fed by polling `<job-dir>/current_game.json` ~1/s while a job
runs. The ML self-play script writes that file after every move
(`{"game", "moves": [{x, y}], "to_move", "status", "winner"}`) and a
final `status: "finished"` record at game end. The watcher is
defensive: a missing or torn file (mid-write read) is skipped silently,
never a crash — so `current_game` is null for jobs whose scripts
predate the live feed.

`buffer_size` in the job detail is exact when `data/manifest.json`
exists (the ML manifest totals the samples in every shard); otherwise
it falls back to the sum of finished games' moves, or an estimate
(`games × avg game length`, flagged `buffer_size_estimate: true`).
`games_per_hour` prefers the rate the ML `selfplay_game` events carry
(running rate since generation started) and falls back to a wall-clock
computation once ≥2 games exist.

### CPU throughput (honest numbers)

From `ml/README.md` (AlphaZero-scale reference): superhuman Gomoku
needs ~1,000,000+ self-play games at ~800 sims/move, ~1,000–10,000
GPU-hours on A100/H100-class hardware — weeks to months on 1× A100.
**A few hours of CPU training will not produce a superhuman model.**
What CPU runs are good for: pipeline verification, architecture
experiments at small scale, and watching the eval ladder move against
weak baselines.

Measured on this VM (shared ~1.5 CPU, tiny 16ch/2-block net, 16
sims/move, `TORCH_THREADS=1`), solo unless noted:

- 6 self-play games: ~31 s wall (incl. a 10 s pause) — ~3.5 s/game
  active, ~78 moves/game, ~54 ms/move, ~680 games/hour including the
  pause (job detail reports the live rate per job).
- A 2-game + 2-epoch + eval full job: ~35 s end to end.
- Contention warning: the first game of one early job took 578 s while
  the ML workstream ran its own smoke run concurrently; the same job's
  later games took ~40 s each. Concurrent heavy jobs slow each other
  ~10× — hence FIFO single-job execution. Throughput depends heavily on
  hardware, net size and sims; quote these as the only measured
  numbers, not as a general rate.
- Job data lives under `server/data/jobs/<job-id>/`: `job.json` (record),
  `config.yaml` (resolved config), `job.log` (full stdout), `data/`
  (self-play shards), `metrics.jsonl`, `eval.json`, `model_card.md`
  (ML-generated), `checkpoints/` (`final.pt`, `best.pt` when a val
  split is used), `final.pt` (extra copy), `model.onnx`,
  `model.fp16.onnx` (if produced). `server/data/` is gitignored.

## Models

Completed train/full jobs register a model under
`server/data/models/model-<run-id>/` with `model.json` (metadata),
`model_card.md` (the ML run's own auto-generated card, which carries an
explicit honest-strength note; the server renders its own honest card
only if the run wrote none),
`model.pt`, `model.onnx`, `model.fp16.onnx` (if the fp16 quantisation
succeeded — needs `onnxconverter-common` in `ml/.venv`, which is
currently NOT installed, so the fp16 variant is usually absent),
`model.mlpackage.zip` (if the Core ML export succeeded — needs
`coremltools`, currently NOT installed on this machine, so this is
usually absent and recorded as such; it never fails the job),
`config.yaml`, `eval.json`.

`GET /api/models` also lists bundled models from
`public/models/models.json` (written by `ml/scripts/run_smoke.py`),
marked `source: "bundled"` with `onnx_url: /models/<file>` — those
files are served by the web app, not this server.

The download endpoint sends `Access-Control-Allow-Origin: *` so
onnxruntime-web in the browser can fetch model files cross-origin.
That is a dev-only choice; in production, serve models from the same
origin as the web app.

## Releases (GitHub)

Setup:

```bash
gh auth login            # preferred auth for uploads
export GOMOKU_MODEL_REPO=owner/repo
```

Or copy `config.example.yaml` to `config.yaml` and set `model_repo`
there (env vars override the file). `GITHUB_TOKEN` env var is the
fallback auth for the REST API path when `gh` is absent.

Dry-run first (validates config + assets, prints exactly what would be
uploaded, makes no GitHub call):

```bash
python scripts/publish_release.py --model-id model-<run-id> --dry-run
# or via the API:
curl -X POST http://localhost:8000/api/models/model-<run-id>/release \
  -H 'Content-Type: application/json' -d '{"dry_run": true}'
```

Then, when a real repo exists and auth is set up:

```bash
python scripts/publish_release.py --model-id model-<run-id> --execute
```

Tag: `model-<run-id>`. Release title: model name. Body: the model card.
Assets: `model.pt`, `model.onnx`, `model.fp16.onnx` (if present),
`model.mlpackage.zip` (if present), `model_card.md`.

**Status: the target repo `yudongqagent/gomoku-ml-playground` (public)
does NOT exist yet — Yudong creates the empty repo himself** (the Muse
GitHub App cannot create repos in a personal account). No release has
been published. `gh` is installed on this machine but not logged in
(`gh auth status` fails). The default release target is
`yudongqagent/gomoku-ml-playground` (overridable via
`GOMOKU_MODEL_REPO` or `model_repo` in `server/config.yaml`).

The dry-run validates everything it can without the repo existing
(config present, assets exist, tag naming `model-<run-id>`, release
body preview). The CLI dry-run also checks repo existence via the
public GitHub API and warns (`repo does not exist yet — create it
first`) rather than failing; the API dry-run skips that network check
so it stays fast and offline-safe. A real upload will fail until the
repo exists and `gh` auth (or `GITHUB_TOKEN`) is set up.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

Tests use a fake ML runner that emits the same stdout lines and creates
the same artifacts as the real scripts, so the full job lifecycle is
exercised without torch. No test touches real GitHub.

## Config

Precedence: env vars > `server/config.yaml` > defaults. See
`config.example.yaml`. Env vars: `GOMOKU_DATA_DIR` (default
`server/data`), `GOMOKU_MODEL_REPO`, `GOMOKU_HOST` (default
`127.0.0.1`), `GOMOKU_PORT` (default 8000), `GITHUB_TOKEN`,
`GOMOKU_ML_PYTHON` (override the ML interpreter).

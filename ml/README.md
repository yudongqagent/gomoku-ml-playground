# Gomoku ML — training pipeline, architectures & export

Python package (`gomoku_ml`) + CLI scripts for the Gomoku ML Playground:
self-play with MCTS, policy+value network training, eval ladder, and
ONNX / Core ML export for on-device inference.

The web app (see repo root) loads models from `public/models/` via
`public/models/models.json`. This directory owns the training side and
the model files in `public/models/`.

## Honest strength section — read this first

**Nothing in this repo is superhuman, and nothing here claims to be.**
The smoke model shipped in `public/models/` is a tiny demo net trained
on a handful of self-play games. It plays poorly and is labelled as a
demo net in `models.json`, in its model card, and in this README.

What real superhuman strength takes (order-of-magnitude, AlphaZero-scale):

| Item | Estimate |
|---|---|
| Self-play games | ~1,000,000+ (at ~800 MCTS sims/move) |
| Network evaluations | ~10^10 – 10^11 |
| Training compute | ~1,000–10,000 GPU-hours on A100/H100-class hardware |
| Wall-clock on 1× A100 | weeks to months of continuous running |
| Wall-clock on 8× A100 | plausibly 1–4 weeks |

Reference points: AlphaZero (2017) used ~5,000 TPUs for self-play plus
64 TPUs for training, and millions of self-play games per title. Strong
open-source Gomoku AIs train on 100k–1M+ self-play games. These numbers
are highly uncertain — they depend on net size, sims per move, and how
much strength counts as "enough" — but the order of magnitude is not
in doubt: **superhuman Gomoku is a cluster-scale project, not a
laptop-scale one.**

On-device training at this scale is not realistic: an A20 Pro-class
phone chip is a strong *inference* chip, not a training cluster. The
intended flow is: **train on GPU servers, export to ONNX/Core ML,
deploy the frozen model on device** for fast inference (Neural Engine
via a native wrapper, WebGPU via ONNX Runtime Web in the browser).
The web app's trainer screen triggers runs on the server; finished
models are uploaded to the GitHub repository as release assets.

The eval ladder in this repo (random / heuristic / prior checkpoint)
can only show relative progress against weak baselines. Beating the
heuristic player says almost nothing about absolute strength.

## Setup

```bash
python3 -m venv ml/.venv
ml/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
ml/.venv/bin/pip install -r ml/requirements.txt
```

For GPU training, install torch from the CUDA index instead (see
https://pytorch.org/get-started/locally/) and keep the rest of
`requirements.txt` unchanged. `coremltools` is optional and commented
out in `requirements.txt` — install it on macOS for Core ML export.

Run the tests:

```bash
ml/.venv/bin/python -m pytest ml/tests -q
```

## Quick start — smoke run (minutes on CPU)

```bash
ml/.venv/bin/python ml/scripts/run_smoke.py
```

Runs self-play (8 games) → train (3 epochs) → ONNX export + verify →
quick eval ladder, all in one run directory under `ml/runs/`, and
writes `public/models/gomoku-smoke-v1.onnx` + `public/models/models.json`.

## Network contract (shared with the TypeScript AI workstream)

- Input `input`: float32 `[1, 8, 15, 15]` — fixed batch 1, fixed 15×15.
  Planes (all relative to the player to move): 0 own stones, 1 opponent
  stones, 2 last move one-hot, 3 side-to-move (all 1.0 if Black to move
  else 0.0), 4 own stones 2 plies ago, 5 opponent stones 2 plies ago,
  6 own stones 4 plies ago, 7 opponent stones 4 plies ago (zeros if
  unavailable). Planes 4–7 are relative to the player who was to move
  at that earlier ply.
- Outputs: `policy` float32 `[1, 225]` logits (row-major, index y\*15+x),
  `value` float32 `[1, 1]` tanh in [-1, 1] from the current player's
  perspective.
- ONNX opset 13 (onnxruntime-web compatible).

## Architectures

Networks are selected by name from a registry
(`gomoku_ml/architectures.py`) — the project is experiment-first, not
one hardcoded net. Every architecture shares the I/O contract above.

| Name | Description |
|---|---|
| `resnet` | AlphaZero-style ResNet. Default baseline. Params: `channels` (128), `blocks` (10). |
| `mobilenet` | MobileNet-style depthwise-separable blocks: ~9× fewer params per block, faster on-device inference. Params: `channels` (128), `blocks` (10). |
| `resnet-se` | ResNet + squeeze-and-excitation (channel attention). Params: `channels` (128), `blocks` (10), `reduction` (8). |

List the registry as JSON (what the training server calls to populate
the UI):

```bash
ml/.venv/bin/python ml/scripts/list_architectures.py
```

Add a new architecture by writing the module and adding one
`ArchitectureSpec` entry in `architectures.py` — configs, training,
export and eval all go through the registry.

## Configs

Configs are YAML, validated at startup — an unknown key, unknown
architecture, or out-of-range value fails fast with an error naming
the key (exit code 2). Every section has defaults; a minimal config
only needs `run_name` and `architecture`. See `gomoku_ml/config.py`
for the full schema.

```yaml
run_name: smoke
seed: 0
architecture: {name: resnet, params: {channels: 16, blocks: 2}}
base_checkpoint: null   # load weights only (fresh optimiser)
resume: null            # resume training incl. optimiser + step
train:
  epochs: 3
  batch_size: 32
  lr: 0.001
  lr_schedule: cosine   # cosine | step
  lr_step_size: 10
  lr_gamma: 0.5
  warmup_steps: 5
  weight_decay: 0.0001
  policy_weight: 1.0
  value_weight: 1.0
  augmentation: true    # dihedral board symmetries (rotations + reflections)
  val_fraction: 0.05
  checkpoint_every_epochs: 0
selfplay:
  games: 8
  sims_per_move: 32
  c_puct: 1.5
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature: 1.0
  temperature_moves: 20
  resign_threshold: null
  max_moves: 150
  checkpoint_every_games: 10   # selfplay_state.json snapshot cadence
eval:
  games: 4                     # TOTAL per opponent, split evenly across colours
  sims: 32
  opponents: [random, heuristic]   # also: mcts, checkpoint
  mcts_sims: 100               # sims for the heuristic-MCTS opponent
  tactical: true               # run the 20-position tactical suite
  tactical_sims: 50
  auto: false                  # train.py runs this eval at end of training
```

A config may also carry an optional top-level `preset:` block
(`name`, `description`, `throughput_note`) — display metadata for
the Trainer UI's preset picker; training ignores it.

- `ml/configs/smoke.yaml` — tiny pipeline-verification run.
- `ml/configs/full.yaml` — the first serious training run, with honest
  compute estimates in the comments. Do not run it on a CPU.
- `ml/configs/longrun-cpu.yaml` — realistic multi-hour CPU-only round
  (small MobileNet, 100 sims, 200 games, 20 epochs, auto-eval). See
  "Long runs" below for measured throughput and honest expectations.

Augmentation: each training batch is transformed with random dihedral
board symmetries (4 rotations × flip). Policy targets are transformed
identically; value targets are unchanged. This is cheap and valuable
for Gomoku — it multiplies the effective data 8×.

## Run tracking

Every run writes to a run directory, `ml/runs/YYYYMMDD-HHMMSS-<name>/`
(UTC timestamp). Pass `--run-dir` to any script to use a specific
directory; otherwise a fresh one is created.

```
ml/runs/20261112-000000-smoke/
  config.yaml          exact snapshot of the resolved config
  metrics.jsonl        one JSON object per logged event
  run.log              human-readable log lines
  eval.json            eval results (ladder + Elo estimate + tactical + verdict)
  eval_report.json     full eval report (machine-readable)
  eval_report.md       full eval report (human-readable, incl. verdict)
  eval_history.jsonl   one line per eval (improvement over time)
  model_card.md        auto-generated summary (arch, config, metrics, eval)
  current_game.json    live self-play feed (updated after every move)
  selfplay_state.json  self-play snapshot (every checkpoint_every_games)
  data/
    manifest.json      shard list + sample counts (replay-buffer manifest)
    <run>_gameNNNN.npz self-play shards
  checkpoints/
    checkpoint-<step>.pt
    best.pt            lowest validation loss so far
    final.pt           end-of-training checkpoint
    interrupted.pt     written when training is interrupted (resumable)
```

Runs are resumable — see "Long runs" below for the exact semantics
(self-play tops up to the configured total; training resumes with
optimiser state and step count, then trains the requested number of
ADDITIONAL epochs).

Compare runs:

```bash
ml/.venv/bin/python ml/scripts/compare_runs.py                 # all runs under ml/runs/
ml/.venv/bin/python ml/scripts/compare_runs.py ml/runs/<a> ml/runs/<b> --out /tmp/comparison.json
```

Prints a table (arch, games, final/val loss, eval win-rates) and
writes `comparison.json` (default `ml/runs/comparison.json`).

## Training workflow

```bash
# 1. Self-play (writes shards to <run-dir>/data/)
ml/.venv/bin/python ml/scripts/selfplay.py --config smoke --run-dir ml/runs/<id>

# 2. Train (reads <run-dir>/data/ by default; writes checkpoints + model card)
ml/.venv/bin/python ml/scripts/train.py --config smoke --run-dir ml/runs/<id>

# 3. Eval ladder (writes eval.json into the run dir, updates the model card)
ml/.venv/bin/python ml/scripts/evaluate.py --checkpoint ml/runs/<id>/checkpoints/final.pt --run-dir ml/runs/<id>

# 4. Export + verify (ONNX opset 13)
ml/.venv/bin/python ml/scripts/export_onnx.py --checkpoint ml/runs/<id>/checkpoints/final.pt --out public/models/gomoku-v1.onnx

# 5. Core ML (macOS; see docs/ios-coreml.md)
python ml/scripts/export_coreml.py --checkpoint ml/runs/<id>/checkpoints/final.pt --out GomokuNet.mlpackage
```

To continue training from an earlier run, point a new config's
`base_checkpoint:` at the earlier run's `final.pt` (or `best.pt`) and
its `--data` at the accumulated shards.

For a multi-hour CPU run, use the long-run preset (same steps, shared
run dir; the eval runs automatically at the end of training):

```bash
ml/.venv/bin/python ml/scripts/selfplay.py --config longrun-cpu --run-dir ml/runs/<id>
ml/.venv/bin/python ml/scripts/train.py --config longrun-cpu --run-dir ml/runs/<id>
# eval_report.md + eval.json + model_card.md appear in ml/runs/<id> when training finishes
```

## Server compatibility

The training server (separate workstream, `server/`) invokes these
scripts as subprocesses. Every script is non-interactive, exits
non-zero with a clear stderr message on failure (exit 2 for config
errors), and respects `--run-dir`.

Logging contract:

- **stdout carries JSON event lines only** — one JSON object per line,
  each with an `"event"` key (`run_started`, `selfplay_game`,
  `train_step`, `epoch`, `checkpoint`, `eval_result`, `run_finished`,
  `error`). The server can parse every stdout line as JSON.
- Human-readable progress lines go to **stderr** (and to `run.log`).
- The same JSON events are appended to `metrics.jsonl` in the run dir.

Threading: scripts default to 2 torch threads for self-play/eval and
up to 4 for training (overridable per job with the `TORCH_THREADS` env
var). MCTS runs one tiny forward pass at a time, so more threads are
pure synchronisation overhead — on a small VM, oversubscribed OpenMP
threads spin and starve the real work (observed: a 4-thread self-play
job slowed ~10× when a second job started on the same 1.5-CPU VM).

The server command to launch a smoke training job end-to-end (from the
repo root):

```bash
ml/.venv/bin/python ml/scripts/run_smoke.py
```

For a full job, run the steps in "Training workflow" above with
`--config full` and a shared `--run-dir`, then export. The server
uploads the exported `.onnx` to the GitHub repository as a release
asset; the web app fetches models via `public/models/models.json`.

## Evaluation

`ml/scripts/evaluate.py` measures what level a checkpoint has actually
reached — ladder matches, a tactical suite, a ladder Elo estimate, and
a plain-language verdict computed from the numbers (never hand-written
per run). Reports are written to the run dir as `eval.json` (the shape
the server and `compare_runs.py` read), `eval_report.json`,
`eval_report.md`, and one line in `eval_history.jsonl`; the model card
is regenerated with the results.

```bash
ml/.venv/bin/python ml/scripts/evaluate.py --checkpoint ml/runs/<id>/checkpoints/final.pt --run-dir ml/runs/<id>
ml/.venv/bin/python ml/scripts/evaluate.py --checkpoint a.pt --run-dir ml/runs/<id> --previous
ml/.venv/bin/python ml/scripts/evaluate.py --checkpoint a.pt --opponent-checkpoint b.pt --opponents random,heuristic,mcts,checkpoint
ml/.venv/bin/python ml/scripts/evaluate.py --checkpoint a.pt --ts-opponent advanced --games 4
```

### Ladder

The checkpoint (via MCTS, temperature 0) plays every opponent in both
colours in equal numbers: `--games` / `eval.games` is the TOTAL games
per opponent (use an even number for exact balance; `full.yaml` uses
40 = 20 as Black + 20 as White). Opponents:

- `random` — uniform random legal moves.
- `heuristic` — 1-ply pattern player: takes wins, blocks wins, scores
  moves by pattern (`gomoku_ml/baseline.py`).
- `mcts` — heuristic-MCTS player: PUCT search with heuristic priors
  and a heuristic leaf value, no neural net
  (`baseline.HeuristicMCTSPlayer`, `mcts_sims` sims per move). This is
  the strongest Python ladder opponent; beating it means the net has
  learned something search alone does not already know.
- `checkpoint` — a previous checkpoint (`--opponent-checkpoint`, or
  `--previous` to auto-pick the prior checkpoint in the run dir), to
  show improvement over time.

Each matchup reports W/L/D, the win rate (draws count as half a win),
and a **Wilson-score 95% confidence interval** — bare percentages are
never reported without uncertainty. With 4 games the interval is
nearly the whole [0, 1] range; with 40 games it is roughly ±15 points
around the observed rate. Small evals cannot prove much; the report
says so via the interval.

### Ladder Elo (relative to this ladder only)

The report estimates a rating from the win rates vs the ANCHORED
opponents via the logistic formula (400·log10(p/(1−p)) + anchor):

- `random` = −800, `heuristic` = 0. These anchors are ARBITRARY
  reference points, stated in every report — they are NOT real Elo and
  the number is NOT comparable to human ratings, online ratings, or
  any other ladder. The `mcts` and `checkpoint` opponents are
  unanchored and do not contribute to the estimate.

The estimate combines the finite per-opponent estimates (games-weighted
mean); when a matchup is 0% or 100% (no finite estimate) the Wilson
interval supplies bounds instead. The report always shows the
estimate WITH its interval and the caveat. If the estimate is, say,
−400 ± 400, that means "somewhere between random and heuristic" — the
interval is the honest part.

### Tactical suite

`gomoku_ml/tactical_positions.json` holds 20 scripted positions:
take an immediate win (9 patterns: open/half-open/gap/edge/diagonal
fours), block an opponent's four (5 patterns), win when the opponent
threatens, avoid a self-trap, create a double threat, find the only
defence against a double threat, and spot a forced VCF sequence (2
positions). The eval runs the checkpoint's MCTS (small sims,
deterministic) on each and reports pass/fail per position + total —
and runs the SAME suite against the heuristic baseline for comparison.

Every position is verified sound by `ml/scripts/verify_tactics.py`
(exhaustive checks: expected moves are exactly the moves satisfying
the position's property; VCF moves really do force a win by
consecutive fours, and the tempting alternatives do not). The VCF
positions use a property-based expectation ("the move starts a forced
win") rather than an exact move list — exact lists proved brittle
(gap-fours and accidental double-threats create extra forcing moves),
and the VCF search defines the truth.

### Verdict tiers (computed from the numbers)

| Tier | Condition (vs heuristic / vs mcts / tactical) |
|---|---|
| `below heuristic` | < 40% vs heuristic |
| `around heuristic` | 40–60% vs heuristic |
| `beats heuristic` | ≥ 60% vs heuristic (no mcts run) |
| `beats heuristic, below search` | ≥ 60% vs heuristic, < 50% vs mcts |
| `challenges search` | ≥ 50% vs mcts but < 40 mcts games or tactical < 80% |
| `good level (v1)` | ≥ 50% vs mcts over ≥ 40 games AND tactical ≥ 80% |

Thresholds live in `gomoku_ml/evaluate.py`
(`VERDICT_BEATS_HEURISTIC`, `GOOD_LEVEL_*`) so they are defined in
code, not in prose.

### "Good level" — v1 working definition

A checkpoint is at a **good level (v1)** when it wins **≥ 50% against
the heuristic-MCTS ladder opponent as both colours over ≥ 40 games**
and passes **≥ 80% of the tactical suite**. This is a v1 working
definition, deliberately modest: it means "beats a competent search
opponent about half the time and handles standard tactics" — roughly
a weak club player, NOT an expert and NOT superhuman. It will be
recalibrated once stronger opponents exist (a trained-opponent ladder
and the TS cross-check are the next anchors). The verdict tiers
reference this definition directly.

### Auto-eval

When `eval.auto: true` (set in `longrun-cpu.yaml`), `train.py` runs
the eval harness at the END of training (against `best.pt` if it
exists, else `final.pt`) and appends the results to `eval.json` +
`eval_history.jsonl`. Mid-training eval is deliberately NOT done: on
CPU it would roughly double run time for little signal. A failed
auto-eval is logged, never fatal — the training result still stands.

### TypeScript cross-check (optional)

`--ts-opponent advanced|expert` plays the checkpoint against the
TypeScript game AI via `scripts/play_match.mjs`, which bundles
`src/ai` with esbuild and drives the checkpoint over a JSON-lines
protocol (`ml/scripts/play_position.py`). The result appears in the
report as a separate "TypeScript cross-check" section — it is NOT
part of the ladder Elo (different opponent, different sims/time
budgets). If the driver is unavailable or fails, the eval still
completes and records the gap ("Python ladder only"). Note the TS
levels are time-budget based, so their strength varies with machine
speed; the match records the budgets used.

## Long runs

Jobs are meant to run for hours unattended. The long-run preset is
`ml/configs/longrun-cpu.yaml` (small MobileNet, 100 sims, 200 games,
20 epochs, auto-eval); start it with the commands in "Training
workflow" or from the Trainer UI (preset "longrun-cpu").

### Measured throughput (this VM, 2026-11-12)

Hardware: ~1.5 shared CPUs, torch 2.15 CPU. Re-measure on your
hardware — Yudong's machine will differ, probably faster.

- **Self-play:** 4 games at 100 sims took 95 s (18.6–28.2 s/game,
  ~71 moves/game) → **~150 games/hour**. 200 games ≈ 80 minutes.
- **Training:** 1 epoch over 269 train samples (283 total, batch 64)
  took 1.8 s; 200 games ≈ 14,200 total samples → ~90 s/epoch,
  20 epochs ≈ 30 min.
- **Auto-eval:** 40 games × 3 opponents at 100 sims + 20 tactical
  positions took **11.8 min** on this VM (random 3.5, heuristic 4.2,
  mcts 3.8, tactical ~0.2). Caveat: the measured net was weak and
  lost fast, so its games were short — stronger nets play longer
  games, so budget ~15–25 min per round as models improve.
- **One full round ≈ 2–2.5 hours.** A 6-hour run is ~2 rounds
  (~400 games, ~28k samples).

### Honest expectation

A few hours of CPU training reaches, at best, a weak club-ish player
that beats the 1-ply heuristic baseline most of the time and sometimes
beats the heuristic-MCTS search opponent. It will take immediate wins,
block immediate losses, and find some short forced wins. It will NOT
be superhuman, NOT expert, and NOT reliably beat a decent human
amateur — that needs AlphaZero-scale training (~1M+ self-play games
at hundreds of sims per move on GPU hardware; see `full.yaml`).
The value of a CPU long run is the pipeline: proving self-play,
training, eval and export work end to end for hours, producing a
model that measurably improves round over round, and learning which
architectures and settings are worth scaling up on GPU.

### Graceful interrupt + resume

`selfplay.py` and `train.py` catch SIGINT/SIGTERM, finish the current
unit of work quickly, flush state to disk, and exit with a distinct
code and message ("interrupted — resumable from ..."):

- **Self-play:** the in-progress game is abandoned (a partial game
  has no outcome and cannot be a training sample); completed games
  are already on disk. Exit code 130 (SIGINT) or 143 (SIGTERM).
- **Training:** the current batch finishes, then `interrupted.pt` is
  written (weights + optimiser + step). Exit code 130/143.

Resume commands (printed by the scripts on interrupt):

```bash
# Self-play: re-run the SAME command — it tops up (see below).
ml/.venv/bin/python ml/scripts/selfplay.py --config longrun-cpu --run-dir ml/runs/<id> --games 200

# Training: resume from interrupted.pt with the REMAINING epochs.
ml/.venv/bin/python ml/scripts/train.py --config longrun-cpu --run-dir ml/runs/<id> \
    --resume ml/runs/<id>/checkpoints/interrupted.pt --epochs <remaining>
```

### Resume semantics (read this — they differ per script)

- **Self-play tops up.** `--games` / `selfplay.games` is the TOTAL
  number of games wanted in the output directory, not the number to
  add. A 4-game job interrupted after 3 games resumes to 4 shards,
  not 7. Game numbering continues (`game0003.npz` after `game0002`),
  game numbers stay unique and dense, and `games_total` in the
  `selfplay_game` events equals the configured target — so the
  server's dedupe-by-game-number keeps working. To generate MORE
  games than a previous run, raise the target. (The library function
  `generate_games` always appends exactly N new games; the top-up
  policy lives in `scripts/selfplay.py`.)
- **Training adds epochs.** `--epochs` / `train.epochs` is the number
  of ADDITIONAL epochs to run from the resume point, not a total.
  Resuming a run interrupted at epoch 2 of 20 with `--epochs 18`
  finishes the original plan; resuming with the config's 20 would
  train 20 more. The cosine LR schedule spans the whole run including
  the resumed epochs (via `schedule_epochs`), so the LR reaches 0 at
  the end of the resumed run, not at the interrupt point.

### Replay buffer = shards on disk

`data/manifest.json` lists every `.npz` shard with its sample count
and is updated after every game; it is the source of truth for what
data a run has. A resumed run reuses generated data instead of
regenerating it. In-memory buffer contents are NOT persisted — the
shards on disk ARE the buffer. Training reads shards via
`--data <dir>` (default: `<run-dir>/data`).

### Disk-write cadence

A long run writes something to disk at least every few minutes of
wall time, so a killed run loses at most the in-progress game:

- `current_game.json` — after EVERY move (atomic temp + rename).
- A data shard + `manifest.json` update — after EVERY game.
- `selfplay_state.json` — every `checkpoint_every_games` games.
- `metrics.jsonl` / `run.log` — every logged event (each game, each
  epoch, each checkpoint).
- `checkpoint-<step>.pt` — every `checkpoint_every_epochs` epochs
  (0 = only `best.pt` / `final.pt`).

### Live self-play feed

`current_game.json` in the run dir is updated after every move (for
the Trainer UI and the server):

```json
{"game": 12, "moves": [{"x": 7, "y": 7}, ...], "to_move": -1,
 "status": "playing", "winner": null}
```

`status` is `"playing"` or `"finished"`; `winner` is `1` (Black),
`-1` (White), `0` (draw), or `null` while playing. Each finished game
also emits a `selfplay_game` JSON event on stdout with game length,
winner, seconds, and running throughput (`games_per_hour` so far).

## Rules parity

`gomoku_ml/game.py` mirrors the TypeScript engine exactly: 15×15,
five-or-more wins (overline counts), Black first, no Renju bans, no
captures. `ml/tests/parity_vectors.json` holds hand-computed test
vectors shared with the TS engine tests — do not change a vector on
one side without changing it on the other.

## Layout

```
ml/
  gomoku_ml/
    game.py          rules (mirrors the TS engine)
    encoding.py      board+history -> 8-plane tensor
    model.py         ResNet / MobileNet / SE-ResNet nets
    architectures.py architecture registry
    augment.py       dihedral board-symmetry augmentation
    mcts.py          PUCT MCTS (batch=1 leaf eval in v1)
    selfplay.py      self-play game generation -> .npz shards
    train.py         training loop (policy CE + value MSE + L2)
    evaluate.py      eval ladder + tactical suite + verdicts + reports
    baseline.py      heuristic + random + heuristic-MCTS players
    interrupt.py     SIGINT/SIGTERM flag for graceful shutdown
    tactical_positions.json  20-position tactical suite (verified sound)
    config.py        config schema, defaults, validation
    runs.py          run dirs, JSON event logging, model cards
  scripts/
    selfplay.py  train.py  evaluate.py  export_onnx.py
    export_coreml.py  run_smoke.py  list_architectures.py
    compare_runs.py  verify_tactics.py  play_position.py  common.py
  configs/  smoke.yaml  full.yaml  longrun-cpu.yaml
  tests/    pytest suite (rules, parity vectors, encoding, nets,
            MCTS, augmentation, registry, config, runs, ONNX round-trip,
            eval harness, long-run support)
  requirements.txt
  README.md  (this file)
```

## Known limitations / future work

- MCTS evaluates leaves one at a time (batch=1). Batched leaf
  evaluation and parallel self-play (multiprocessing) would speed up
  self-play substantially and are the first things to build for a
  serious run.
- The training loader reads all shards into memory; a streaming loader
  is needed at AlphaZero-scale data volumes.
- Dynamic int8 ONNX quantisation is documented in `export_onnx.py` but
  not automated (fp16 quantisation is implemented via `--quantize fp16`).
- Core ML export is code-complete but was not executed on this Linux
  VM (coremltools not installed); run it on a Mac — see
  `docs/ios-coreml.md`.
- The ladder Elo is relative to this ladder's arbitrary anchors only
  (random −800, heuristic 0) and is NOT comparable to any human or
  online rating; the `mcts` and `checkpoint` opponents are unanchored.
  Small evals (few games) have very wide Wilson intervals — the
  interval, not the point estimate, is the honest statement.
- The heuristic-MCTS ladder opponent is search, not a trained net:
  beating it proves the net has learned something search does not
  already know, but it is not a strong opponent in absolute terms.
- The TS cross-check (`--ts-opponent`) plays time-budgeted TS levels,
  so its result varies with machine speed; it is reported separately
  and never folded into the ladder Elo.
- `verify_tactics.py` proves the tactical positions sound, but the
  suite is small (20 positions) — a high pass rate means the model
  handles standard tactics, not that it plays sound positional
  Gomoku.
- Checkpoint-vs-checkpoint eval uses deterministic MCTS (temperature
  0, no noise), so two near-identical checkpoints replay near-identical
  games: the smoke run's final.pt vs best.pt came out 20-20 with a
  perfect colour split (0-20 as Black, 20-0 as White) — effectively
  two distinct games, not 40 independent samples. Treat the
  checkpoint matchup's CI as overstating the effective sample size
  until eval adds opening randomisation or temperature.

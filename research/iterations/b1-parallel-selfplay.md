# B1 parallel self-play: multiprocessing over batched leaf eval (batch 3, B1)

Date: 2026-11-13. Branch: `research/batch-3-b1`. Status: implemented,
tested, measured on CPU. **GPU speedup is UNMEASURED** — this VM has
no GPU, and no GPU number is claimed anywhere in this document.

## The problem

Batch 2's B2 work made leaf evaluation batched (cross-game batching,
no virtual loss) and roughly doubled CPU self-play throughput
(~570 games/hour for the round-2 ResNet 32ch/4blocks at 100 sims,
concurrency 8). But the driver was still single-process: on a 4-CPU
VM, one process can only use ~1–2 cores effectively, because the
Python tree search (selection, backprop, encoding) is single-threaded
and the net forward passes are tiny. Yudong's challenge for batch 3:
self-play is still too slow, and GPU self-play should handle tens of
thousands of games in a batch. B1 is the multiprocessing step: N
worker processes, each running the batched path, with a merge step
that combines their shards into one run dir.

## Design: one process per worker, merge at the end

* **Workers are spawned, not forked** (`multiprocessing` spawn
  context). Each worker builds its own copy of the net from a
  picklable `NetSpec` (architecture name + params, checkpoint path,
  device, MCTS constants) — no shared state, no fork + torch thread
  pool hazards. Spawn costs a few seconds of interpreter + torch
  import per worker; negligible against runs of minutes to hours.
* **One torch thread per worker** (`torch.set_num_threads(1)` first
  thing in the worker). This is load-bearing, not cosmetic: with the
  default 4 torch threads, a batched forward pass on the tiny
  16ch/2block net took **579 ms/call** under contention with sibling
  jobs, vs **2.0 ms at 1 thread** (~290x — measured 2026-11-13, see
  "Correctness evidence" below). OpenMP oversubscription on a shared
  4-CPU VM is the single biggest throughput hazard in this codebase.
* **Seeds differ per worker** (`seed + worker_index`), so workers
  generate different games; the combined run is deterministic given
  (seed, workers, concurrency). Games are split as evenly as possible
  (the first workers get one extra game when the split is uneven).
* **Workers never write to the run dir directly.** Each writes its
  shards into its own shard dir under
  `<out_dir>/.parallel-<pid>-<ts>/worker<N>/` — invisible to the
  manifest, which only globs `*.npz` in `out_dir` itself. The parent
  merges the dirs one at a time after all workers finish, numbering
  from `existing_game_count(out_dir, run_name)`, so a merge into a
  non-empty dir appends after the existing shards and never
  overwrites one. That is the part the single-writer numbering
  assumed: the merge is the only writer to the run dir, and it is
  single-threaded.
* **The merge validates every shard** (all five arrays present,
  consistent lengths, planes/policy shapes, policy rows summing to 1,
  values in {-1, 0, 1}). A worker killed mid-write can leave a
  truncated `.npz` (`np.savez_compressed` writes in place, not
  atomically); corrupt shards are deleted and reported in `skipped`,
  not merged — a corrupt shard counted by `existing_game_count`
  would punch a hole in the numbering.
* **Interrupt and crash behaviour.** The parent polls the workers
  and, when its own interrupt flag fires, sets a shared stop event;
  workers see it at the next round boundary, exit gracefully, and
  whatever they completed is still merged. Stragglers are terminated
  after a 180 s grace period (their completed shards are still
  merged; a shard truncated mid-write is dropped by merge
  validation). If the parent is killed outright, the orphaned shards
  sit under `.parallel-*/` and are recovered by
  `recover_parallel_shards` on the next run (called automatically at
  the start of `generate_games_parallel` and by the self-play CLI
  before its top-up computation), so an interrupted parallel run
  resumes without losing completed games or regenerating them. If a
  worker crashes, the surviving workers' shards are still merged and
  then a `RuntimeError` is raised with the worker's traceback —
  partial data is never silently lost, and a failed worker never
  silently produces a short run.
* **Events and live feed.** Workers collect their per-game events and
  return them; the parent re-emits them with the final merged game
  numbers, in final-number order. The live feed (`current_game.json`)
  is written by worker 0 only — one representative game, as in
  batched mode. The parent writes a merged `selfplay_state.json`
  after the merge (games, samples, games/hour, workers, concurrency).
* **Two parallel calls into the same run dir at the same time are
  NOT supported** — the single-writer assumption now lives at the
  level of the merge.
* The single-process path is the default and is untouched in
  behaviour: `--workers 1` (default) runs exactly the old code. The
  parallel path is selected only by `--workers N` with N > 1.

What did *not* change: the net architecture, the training loss, the
encoding, the shard schema, the manifest/live-feed/event contracts,
the MCTS selection/backprop rules, and the batched path itself.
`baseline.py`, `scoring.py`, `mcts_heuristic.py`, `kernel_numpy.py`
and `research.py` are untouched (frozen / owned by sibling
workstreams). No change to `mcts.py` was needed — all B1 logic lives
in `selfplay.py` and the scripts.

## Files changed

* `ml/gomoku_ml/selfplay.py` — `NetSpec` (picklable net description
  for workers); `_validate_shard`; `merge_worker_shards`;
  `recover_parallel_shards`; `_EventFlag`; `_parallel_worker` (spawn
  entry point); `generate_games_parallel` (split, spawn, poll,
  merge, re-emit events, write merged state).
* `ml/scripts/selfplay.py` — `--workers N` (default 1 = the
  single-process path, unchanged); the parallel branch builds a
  `NetSpec` and calls `generate_games_parallel` (the parent never
  builds the net); orphaned-shard recovery before the top-up
  computation (for every run, not just parallel ones); `workers` in
  the `run_started` event; the interrupt resume command includes
  `--workers` / `--concurrency`.
* `ml/scripts/bench_parallel.py` — **new**: one-command benchmark
  that times one arm per worker count (default 1, 2, 3, 4) with the
  same config/checkpoint/sims/seed/game count, records load average
  before/after each arm, and prints a JSON summary.
* `ml/tests/test_parallel_selfplay.py` — **new**: 7 tests (see below).
* `ml/README.md` — the "multiprocessing is future work" limitation
  updated to point here.

## Correctness evidence

### (a) Tests

`ml/tests/test_parallel_selfplay.py` (7 tests), run with the main
worktree venv (`/home/hatch/workspace/projects/gomoku-ml-playground/ml/.venv/bin/python`)
from the worktree root:

* merged data equals the union of the workers' shards (same games,
  dense numbering from 0, manifest consistent, total samples match);
* resume-safe numbering across two merges into one dir (append, never
  overwrite; `existing_game_count` and manifest agree);
* a corrupt shard is dropped, not merged, and leaves no hole in the
  numbering;
* `recover_parallel_shards` merges orphaned `.parallel-*/` shards
  after a pre-existing shard, removes the leftover dirs, and is a
  no-op on a second call;
* a short 2-worker run completes with valid shards (planes/policy/
  value shapes, policy rows sum to 1, values in {-1, 0, 1}), a
  consistent manifest, a merged `selfplay_state.json`, distinct games
  across workers (different seeds), and no leftover worker dirs;
* two parallel calls into one dir append with dense numbering (the
  resume story for an interrupted 10k run);
* `on_event` receives one `selfplay_game` event per merged game,
  with final merged game numbers in order.

Result: **7 passed** (plus the 8 batched tests in
`test_batched_mcts.py` still passing — 15 passed together).

### (b) Torch threads: the 290x measurement

While debugging why the new tests took 7+ minutes, a direct
measurement on the tiny net (16ch/2blocks, batch of 2 positions,
sibling H1/H2 jobs running on the same 4-CPU VM):

| torch threads | `_evaluate_batch` (2 pos) | `search` (4 sims) |
|---------------|---------------------------|-------------------|
| 4 (default)   | 579.5 ms/call             | 1.206 s           |
| 2             | 4.5 ms/call               | 0.072 s           |
| 1             | 2.0 ms/call               | 0.054 s           |

Two tiny games (4 sims, 30 max moves) took 68.2 s at 4 threads and
0.8 s at 1 thread. The measurement reproduces: after a VM restart,
with only the H2 sibling job running, the same check gave 510.4 ms
(4 threads) vs 6.1 ms (2 threads) vs 2.1 ms (1 thread) — ~240x.
This is the oversubscription failure mode the README already warns
about, far worse than the ~10x observed before — because several
heavy jobs now share the VM. The test module pins itself to 1 thread
(with this measurement in its comment), and every B1 worker pins
itself to 1 thread. Any future benchmark on this VM must state its
thread count; a 4-thread number measured under contention is
meaningless.

### (c) CLI smoke test (real checkpoint, end to end)

`ml/scripts/selfplay.py --config ml/runs/b3-b1/config.yaml
--checkpoint ml/runs/b3-b1/final.pt --run-dir ml/runs/b3-b1-smoke
--run-name b3-b1-smoke --games 4 --sims 50 --workers 2
--concurrency 8` (under contention, see the measurement section):
4/4 games merged, dense numbering, manifest consistent (4 games, 186
samples), `selfplay_game` JSON events on stdout with final game
numbers 1–4, merged `selfplay_state.json`, 215.6 games/hour. The
smoke run dir is gitignored data, removed after verification.

## CPU measurement: before / after

Fixed config for every number below (all arms identical except the
worker count):

* Checkpoint: `ml/runs/b3-b1/final.pt` — a copy of the round-2 final
  net (resnet 32ch/4blocks, 192,714 params) from
  `server/data/jobs/job-20261112-122716-9069f1/final.pt` in the main
  worktree; config copied from the same job dir. The checkpoint is
  gitignored data, not committed.
* Self-play: **50 sims/move**, c_puct 1.5, Dirichlet α 0.3 / ε 0.25,
  temperature 1.0, temperature_moves 30, no resign threshold,
  max_moves 225 — from that job's own `config.yaml`. Seed 0.
  Concurrency 8 per worker, max-batch-size 32, throughout (no
  concurrency tuning was attempted — 8 is the value batch 2 used).
* 40 games per arm. Workers=1 arm: `generate_games_batched`
  in-process, torch threads 2 (as in `bench_batched.py`) — the
  B2-only "before". Workers>=2 arms: `generate_games_parallel`,
  1 torch thread per worker — the "after".
* Command: `ml/.venv/bin/python -u ml/scripts/bench_parallel.py
  --config ml/runs/b3-b1/config.yaml --checkpoint ml/runs/b3-b1/final.pt
  --games 40 --sims 50 --concurrency 8 --workers 1,2,3,4 --seed 0
  --threads 2 --work-dir /tmp/bench-b1-run1
  --out ml/runs/b3-b1/bench-b1-run1.json`.
  Summary: `ml/runs/b3-b1/bench-b1-run1.json` (gitignored data, kept
  in the worktree — `/tmp` did not survive the VM restart below).

| Workers | Wall (40 games) | Games/hour | Speedup vs workers=1 |
|---------|-----------------|------------|----------------------|
| 1 (B2-only) | 319.2 s | **451.1** | 1.00x |
| 2 | 154.3 s | **933.4** | **2.07x** |
| 3 | 134.5 s | **1070.3** | **2.37x** |
| 4 | 134.1 s | **1073.5** | **2.38x** |

**Contention caveat — read this before quoting any number.** Every
arm ran under contention: the H2 sibling workstream's 500-game
self-play job (~0.6 CPU) ran throughout the bench, and the H1
sibling's engine match resumed during the workers=3 arm (load
average jumped to 5.4). All four arms faced roughly the same
background load, so the *relative* speedups are the trustworthy
numbers; the absolute games/hour are **measured-under-contention**,
not clean. How much contention costs: the H2 job itself was running
at ~163 games/hour against batch 2's clean ~570 for the same
config — a ~3.5x slowdown — and my workers=1 arm at 50 sims
(451.1/hr) is *below* batch 2's clean 100-sim number (570/hr),
which should be impossible on a quiet VM (fewer sims = faster).
A clean-VM re-run would show higher absolute rates at every worker
count; the speedups would likely be similar or slightly better at
3–4 workers, since the plateau at 3–4 is caused by the sibling jobs
occupying ~1 of the 4 CPUs.

Two more caveats:

* **A first bench attempt was lost to a VM restart.** The bench was
  first launched at 07:27 UTC; the VM was replaced ~3 minutes later
  (uptime reset to 0, `/tmp` wiped, all processes killed). The
  numbers above are from the re-run launched at 07:27 UTC on the new
  VM. The restart also killed the sibling jobs; H2 resumed its
  500-game job (top-up from 229 games) and H1 resumed its match
  within minutes, so the re-run was contended throughout.
* **The 3→4 plateau is a contention artifact of this run, not a
  scaling limit of the design.** With ~1 CPU taken by siblings,
  only ~3 CPUs were free, so the 4th worker had no core to run on
  (1070.3 → 1073.5/hr, within noise). On a quiet 4-CPU VM, 4 workers
  should beat 3; the 10k run below uses workers=4 for that reason.

**Self-play equivalence** (same config, same net, same seed scheme):

| Arm | Black wins | White wins | Draws | Mean plies |
|-----|-----------|-----------|-------|------------|
| workers=1 | 25 | 15 | 0 | 42.8 |
| workers=2 | 24 | 16 | 0 | 43.8 |
| workers=3 | 26 | 14 | 0 | 42.5 |
| workers=4 | 23 | 17 | 0 | 42.5 |

Outcome distributions are statistically equivalent across arms
(23–26 Black wins of 40; mean game length within 1.3 plies) — the
parallel path generates the same *kind* of games as the batched
path, as expected: each worker runs the identical batched search,
differing only by RNG seed.

**GPU speedup is UNMEASURED.** No GPU exists on this VM, and no GPU
number is claimed. The parallel path is device-agnostic (each worker
loads its own copy of the net on the given device), but whether N
workers help on a GPU at all — and by how much — must be measured
on a GPU.

## Ceiling analysis

The ceiling on THIS VM is set by CPU count, not by the design: each
worker is a single-threaded Python process, and the bottleneck inside
a worker is the Python tree search (selection, backprop, encoding),
not the net evaluation (the 32ch net's batched forward pass at batch
8 is milliseconds; the 290x thread measurement above shows what
happens when the *threading* is wrong, not when the net is slow).
Measured per-worker throughput at 50 sims is ~268 games/hour
(1073.5 / 4 workers, contended). Adding workers past the CPU count
buys nothing — the 3→4 plateau above is that wall, one CPU early
because of the sibling jobs.

Projected time for staged data at the measured (contended) best rate
of **1073.5 games/hour**, 50 sims:

| Games | At 1073.5/hr (B1+B2, workers 4, contended) | At 451.1/hr (B2-only, workers 1, contended) |
|-------|--------------------------------------------|---------------------------------------------|
| 10,000 | **9.3 hours** | 22.2 hours |
| 50,000 | **46.6 hours (~1.9 days)** | 110.8 hours (~4.6 days) |
| 100,000 | **93.2 hours (~3.9 days)** | 221.7 hours (~9.2 days) |

On a quiet VM these times would be shorter by the contention factor
(unknown for this workload; the H2 job's ~3.5x slowdown is the only
in-repo bound, and it is H2's job, not this one — do not quote a
clean rate without measuring it). Even so, the shape is clear: this
4-CPU VM cannot do tens of thousands of games per batch in hours.
50k games is a ~2-day job here; 100k is a ~4-day job — and that is
at 50 sims; at 100 sims the times roughly double.

What would it take to do **10,000 games in ~1 hour** at 50 sims?
10,000 games/hour is 9.3x the measured 4-worker rate. At ~268
games/hour per worker-core, that is **~37 worker-cores** — a
~40-CPU machine (or equivalent), assuming linear scaling and no
other bottleneck. In practice the shape is: many parallel workers
(one per core) + batched inference + a faster engine. The GPU part
of that story is architectural, not measured: today the GPU sits
idle most of the time because the Python tree search on few cores is
the bottleneck, not the net eval — a GPU only helps once enough
concurrent games feed it big enough batches (high concurrency per
worker, many workers), and even then the Python selection/backprop
per simulation stays on the CPU. The durable fixes, in order, are
(1) more CPU cores with one worker each, (2) a vectorised /
torch-native encoder (per-position NumPy encoding is the next
bottleneck once the forward pass is fast), and (3) a compiled engine
for the tree search itself. None of those is claimed to reach a
particular Elo: **no Elo numbers are promised at any game count** —
the staged program (10k → train → full eval, 50k → eval, 100k+ →
eval, each stage gated on Elo improvement against the project's
"good level" bar) measures strength at each stage instead of
predicting it.

## The 10k staged dataset

**Run in progress at commit time** — this section is completed when
the run finishes (or hits the 6-hour cap). Plan: workers=4,
concurrency 8 per worker, 50 sims, round-2 checkpoint, target
10,000 games in `ml/runs/b3-b1-10k/data/` (gitignored data, not
committed). At the measured contended rate (1073.5/hr) 10k games
project to 9.3 hours, past the ~6-hour cap — so the run stops at the
cap with whatever fits (~6,400+ games at the contended rate, more
if the VM quiets when the sibling jobs finish), and the exact count
is reported here, not rounded up to 10k. The remainder stays
generatable later by re-running the same command (top-up).

## Limitations for running on a GPU later

* **Batch size needs tuning per device.** The effective batch is at
  most `concurrency` per worker (one leaf per active game per round),
  capped by `max_batch_size`. On a GPU, larger `concurrency` is
  likely better; tune by re-running `bench_parallel.py` on the target
  hardware — do not assume the CPU number transfers.
* **Workers share one GPU.** Each worker loads its own copy of the
  net on the same device; with N workers the GPU memory use is N×
  one net (tiny for this 200k-parameter net) and the forward passes
  from different workers serialise on the GPU unless MPS / streams
  are used. Whether N workers help on a GPU at all depends on whether
  the GPU is saturated by one worker's batches — measure it there;
  on CPU the win comes from parallel Python tree search, which is a
  CPU-side bottleneck, not a GPU one.
* **Encoding is still per-position NumPy** (`encoding.encode_batch`
  loops over positions). At higher throughput this becomes the
  bottleneck once the forward pass is fast; a vectorised/torch-native
  encoder would be the next optimisation.
* **The merge is serial and disk-bound at the end.** For 10k games
  the merge renames 10k files and rebuilds the manifest by opening
  every shard (a few seconds); at 100k+ games the manifest rebuild
  (one `np.load` per shard) becomes a minutes-long step — an
  incremental manifest (append, don't rebuild) would be the fix.
* **Spawn start-up cost.** Each worker re-imports the interpreter +
  torch (~seconds). Negligible for runs of minutes or more; for
  short interactive runs, prefer the single-process path.
* **No cross-worker load balancing.** Games are split statically;
  a worker whose games run long (slow games are ~2x longer than fast
  ones) finishes late while others idle. A work-stealing queue would
  close that gap; the measured effect at 40 games/arm is small (the
  last games finish within seconds of each other), but at 10k games
  the tail matters more — see the 10k run's achieved rate vs the
  bench rate.

# Batched leaf evaluation for neural MCTS self-play (batch 2, B2)

Date: 2026-11-13. Branch: `research/batch-2`. Status: implemented, tested,
measured on CPU. **GPU speedup is UNMEASURED** — this VM has no GPU, and no
GPU number is claimed anywhere in this document.

## The problem

Neural self-play was CPU-bound on single-process, batch=1 leaf
evaluation: every MCTS simulation ran the net on one position at a time.
A GPU (e.g. a Colab T4) would only have accelerated *training*, not
self-play, because a batch=1 forward pass cannot use a GPU's parallelism.
The last long run managed ~190 games/hour. This work makes leaf
evaluation batched, so a GPU can accelerate self-play too — and, as
measured below, it also speeds up CPU self-play substantially.

## Design: cross-game batching, no virtual loss

The batching collects leaf positions **across concurrent MCTS searches**
(one per game in progress), not within a single search:

* Each MCTS simulation yields **exactly one** leaf evaluation (or none,
  if it ends in a terminal position). Simulations are now generators
  (`MCTS._simulate`): selection and backprop run synchronously, and the
  single leaf evaluation is a yield point answered by the driver.
* `MCTS.search_many` drives several independent searches round by round
  — one simulation per active search per round — collects the round's
  leaf requests, evaluates them in **one batched forward pass** (chunked
  to `max_batch_size`), and resumes each simulation with its result.
* `selfplay.generate_games_batched` plays up to `concurrency` games at
  once in one process: each round, every active game runs one
  move-search through `search_many`; a finished game is saved
  immediately and a replacement starts, so the batch stays full until
  the last games.

Why cross-game batching instead of virtual loss within one search:

* **It changes no search semantics.** Simulations within one search
  still run strictly sequentially, so selection and backprop are
  identical to the single-leaf path — the tree built is the same up to
  float noise in batched vs single forward passes (~1e-7, measured
  below) and RNG interleaving order for Dirichlet noise. Virtual loss
  would perturb selection inside every search and change the trees.
* Self-play already has many independent games available to batch
  across, so there is no need to manufacture parallelism inside one
  search. Single-game latency is not the bottleneck; training-data
  throughput is.
* The single-leaf path is preserved untouched as the default:
  `MCTS.search` / `selfplay.generate_games` behave exactly as before
  (the recursive `_run_one` now drives the same `_simulate` generator
  and evaluates its one leaf immediately). The CLI flag
  `--concurrency 1` (default) selects it; `--concurrency N` selects the
  batched path. Before/after behaviour is directly comparable.

What did *not* change: the net architecture, the training loss, the
encoding, the shard schema, the manifest/live-feed/event contracts, and
the MCTS selection/backprop rules. Dirichlet noise is applied per search
after its first simulation, exactly as in `search`.

## Files changed

* `ml/gomoku_ml/mcts.py` — `LeafRequest` / `SearchRequest` types;
  `_simulate` generator (selection/backprop identical to the old
  `_run_one`); `_run_one` reimplemented as the single-leaf driver of
  `_simulate`; `_evaluate_batch` (one forward pass for many positions,
  same masking/renormalisation as `_evaluate`); `MCTS.search_many`
  (concurrent searches with batched leaf eval, `max_batch_size`
  chunking); `_apply_root_noise` shared by `search` and `search_many`.
* `ml/gomoku_ml/selfplay.py` — `generate_games_batched` (+ internal
  `_BatchedGame` state): same shard schema, manifest, progress events
  and resume-safe numbering as `generate_games`. Differences by design:
  the live feed shows one representative game (the most recent mover),
  and a stop flag discards all in-progress games, not one.
* `ml/scripts/selfplay.py` — `--concurrency` (default 1 = original
  sequential path) and `--max-batch-size` (default 32) flags; the mode
  is recorded in the `run_started` event.
* `ml/scripts/bench_batched.py` — **new**: one-command benchmark that
  runs the consistency check, the search-agreement check, and a timed
  sequential-vs-batched comparison, and prints a JSON summary. Re-run
  it on a GPU with `--device cuda` to measure GPU speedup.
* `ml/tests/test_batched_mcts.py` — **new**: 8 tests (see below).
* `ml/README.md` — the "batched leaf evaluation is future work"
  limitation updated to point here.

Not touched, per the batch-2 work split: `baseline.py`, `scoring.py`,
`research/iterations/iter-*.md`, `research/VERSIONS.md`.

## Correctness evidence

### (a) Batched vs single forward pass — same positions, same net

`ml/scripts/bench_batched.py --config
server/data/jobs/job-20261112-122716-9069f1/config.yaml --checkpoint
server/data/jobs/job-20261112-122716-9069f1/final.pt --positions 16
--agreement-sims 50 --skip-timing` (resnet 32ch/4blocks, CPU):

* 16 random positions: **max policy abs diff 9.9e-09, max value abs diff
  4.2e-07**. Pure float noise from different reduction order in batched
  kernels; the net is in eval mode so batch-norm uses running stats and
  positions in a batch do not interact.

The same check on the smoke net (resnet 16ch/2blocks, 8 positions) gave
max policy diff 3.2e-09, max value diff 8.9e-08.

### (b) Batched search vs single-leaf search — same trees

Same 32ch run, 16 random positions at 50 sims, no Dirichlet noise:

* Visit distributions: **mean TV distance 0.0000, max TV 0.0000**,
  argmax agreement **16/16**, max root-value diff 2.6e-07. The trees
  were literally identical — the ~1e-7 float noise did not flip any
  PUCT selection on these positions. (Smoke net, 8 positions at 32
  sims: same result, TV 0.0000, argmax 8/8, max root-value diff
  8.6e-08.)

A separate diagnostic (`/tmp/diag_tv.py`, not committed) re-checked the
test setup with a *random-init* net (near-tied PUCT scores, the
worst case for float-noise flips): TV 0.0000 on all 4 positions, and a
control showed plain `search` at 1 thread vs 4 threads is also
TV 0.0000 there — i.e. no divergence attributable to batching was
observed in either case. Note for the record: with a random net and
many positions, an occasional near-tie flip is possible in principle
(it would be possible between thread counts too); the test thresholds
below leave headroom for that, and the self-play equivalence numbers
in the measurement section are the stronger evidence.

### (c) Tests

`ml/tests/test_batched_mcts.py` (8 tests):

* batch vs single consistency (thresholds 1e-5; observed ~1e-8);
* empty batch returns `[]`;
* `search_many` vs `search` agreement (TV < 0.05, root-value diff <
  0.01; observed 0.0) + root/child visit-count accounting;
* `max_batch_size` chunking down to 1 does not change the search;
* terminal-root and empty-request edge cases;
* a short batched self-play run completes with valid shards
  (planes/policy/value shapes, policy rows sum to 1, values in
  {-1, 0, 1});
* resume-safe numbering across two batched calls into one directory;
* degenerate `concurrency=1` run completes.

Full suite: `ml/.venv/bin/python -m pytest ml/tests -q` →
**176 passed in 1481.12s (0:24:41)**, exit 0, 2 warnings (both
pre-existing ONNX exporter DeprecationWarnings). Run 2026-11-13
~04:22 UTC on branch `research/batch-2`, while sibling batch-2 jobs
were also running on the VM (hence the long wall time).

## CPU measurement: before / after

Fixed config for every number below (both arms identical except the
batching path):

* Checkpoint: `server/data/jobs/job-20261112-122716-9069f1/final.pt`
  (round-2 final net, resnet 32ch/4blocks, 192,714 params).
* Self-play: 100 sims/move, c_puct 1.5, Dirichlet α 0.3 / ε 0.25,
  temperature 1.0, temperature_moves 30, no resign threshold,
  max_moves 225 — from that job's own `config.yaml`. Seed 0, torch
  threads 2, CPU. Batched arm: `--concurrency 8 --max-batch-size 32`.
* 10 games per arm per run (not 20: the VM was shared with sibling
  batch-2 jobs and a 20-game sequential arm was projected at ~2 hours
  under that contention; the 10-game runs were repeated twice, so 20
  games per arm were timed in total).
* Command: `ml/.venv/bin/python -u ml/scripts/bench_batched.py
  --config server/data/jobs/job-20261112-122716-9069f1/config.yaml
  --checkpoint .../final.pt --games 10 --sims 100 --concurrency 8
  --max-batch-size 32 --seed 0 --threads 2 --skip-checks`.
  Summaries: `/tmp/bench-r2-run1.json`, `/tmp/bench-r2-run2.json`
  (bench working dirs `/tmp/bench-r2-run1/`, `/tmp/bench-r2-run2/`).

| Run | Sequential (before) | Batched (after) | Speedup |
|-----|---------------------|-----------------|---------|
| Run 1 (VM contended during sequential arm) | 10 games, 813.0s, **44.3 games/hour** | 10 games, 63.1s, **570.2 games/hour** | 12.87x — **inflated, see below** |
| Run 2 (VM quiet for both arms) | 10 games, 119.0s, **302.6 games/hour** | 10 games, 62.8s, **573.2 games/hour** | **1.89x** |

**The honest headline number is run 2: 1.89x on CPU** (302.6 →
573.2 games/hour). Run 1's 12.87x is a contention artifact, not a
speedup: its sequential arm ran while two full pytest suites and the
coordinator's v5 eval batches were hammering the VM (load average up
to 9.7 on 4 CPUs; the bench process got ~0.28 CPUs), with per-game
times of 266.0s, 132.0s, 32.4s, 130.0s, 83.5s, 31.0s, 19.6s, 80.2s,
17.7s, 20.1s — the early games ran 10–15x slower than the same games
on a quiet VM. The batched arm ran after the load eased. Run 2, with
both arms on a quiet VM, is the apples-to-apples comparison:
sequential per-game times 8.7–15.8s (tight), batched 62.8s total.

Corroborating evidence the batched number is stable: the batched arm
reproduced almost exactly across runs (63.1s vs 62.8s; 570.2 vs
573.2 games/hour). A separate smoke-net run (resnet 16ch/2blocks,
32 sims, concurrency 3, 4 games) gave 636.8 → 1677.7 games/hour,
2.63x — same direction, smaller net.

**Self-play equivalence** (same config, same net, same seed 0 — the
two runs replayed identical games, so this is 10 distinct games per
arm, each measured twice):

* Sequential: 5 Black wins / 5 White wins / 0 draws, mean 40.3 plies
  (identical in both runs — the sequential path is deterministic given
  the seed).
* Batched: 6 Black wins / 4 White wins / 0 draws, mean 39.2 plies
  (identical in both runs — the batched path is also deterministic
  given the seed and concurrency).
* Outcome distributions are statistically equivalent (5–5 vs 6–4 over
  10 games is well within noise); mean game length differs by 1.1
  plies (2.7%). Games are *not* move-for-move identical between the
  arms — the batching changes RNG interleaving order for Dirichlet
  noise and move sampling, and ~1e-7 float noise can flip a near-tied
  PUCT selection — but the search semantics are the same (see (b)
  above: identical trees on all 16 test positions), so the difference
  is sampling noise, not a behaviour change.

**GPU speedup is UNMEASURED.** No GPU exists on this VM, and no GPU
number is claimed. The batched path is GPU-ready (one forward pass
per simulation round; `--device cuda` on `bench_batched.py` /
`selfplay.py`), but the actual GPU speedup must be measured on a GPU.

## Limitations for running on a GPU later

* **Batch size needs tuning per device.** The effective batch is at
  most `concurrency` (one leaf per active game per round), capped by
  `max_batch_size`. On a GPU, larger `concurrency` (16–64) is likely
  better; on this CPU, 8 was used. Tune by re-running
  `bench_batched.py` on the target hardware — do not assume the CPU
  number transfers.
* **Encoding is still per-position NumPy** (`encoding.encode_batch`
  loops over positions). On a GPU this becomes the bottleneck once the
  forward pass is fast; a vectorised/torch-native encoder would be the
  next optimisation.
* **Memory:** each active game holds its own MCTS tree for the current
  move (discarded after the move, as in the sequential path) plus its
  game history. Memory scales with `concurrency × tree size`; with
  100 sims/move the trees are small, but at high sims and high
  concurrency on a GPU box, watch RAM, not GPU memory (the net itself
  is ~200k parameters for the 32ch/4block net).
* **The driver is single-process and synchronous.** A simulation round
  waits for the slowest selection in that round; Python-side selection
  (`_select_child` scans all children) is pure overhead that batching
  does not remove. Multiprocessing / async batching would be the next
  step after GPU batching.
* **Live feed shows one representative game** in batched mode (the most
  recent mover), not all concurrent games; the Trainer UI's
  `current_game.json` contract is otherwise unchanged.
* **No virtual loss** means a *single* game/search cannot be batched:
  if single-game latency ever matters (e.g. interactive analysis),
  virtual-loss batching inside one search would still need building.

## Coordinator verification (independent re-run, 2026-11-13)

The coordinator re-ran the bench on a fresh VM (the previous VM was
replaced mid-batch; the original `/tmp/bench-r2-*.json` summaries were
lost with it — all numbers above survive in this doc):

* Consistency checks reproduce exactly: max policy abs diff
  9.87e-09 (doc: 9.9e-09), max value abs diff 4.17e-07 (doc:
  4.2e-07); search agreement TV 0.0, argmax 16/16, max root-value
  diff 2.65e-07 (doc: 2.6e-07).
* Timed comparison, same config (10 games/arm, 100 sims,
  concurrency 8, max-batch 32, seed 0, threads 2, quiet VM):
  sequential 187.0s, **192.5 games/hour**; batched 81.0s, **444.7
  games/hour**; speedup **2.31x**. Game outcomes reproduce
  exactly (sequential 5B–5W–0D, mean 40.3 plies; batched 6B–4W–0D,
  mean 39.2 plies). Absolute games/hour differ from run 2
  (302.6 → 573.2) — VM-to-VM speed varies — but the speedup is
  the same order (~2x in both runs). Honest headline stands:
  **~2x on CPU** (1.89x–2.31x across runs); GPU speedup remains
  unmeasured.

# Iteration 25 — v25: v8 retrained on the completed 10k dataset (ML batch 1, H2)

**Status: pre-registered — design, data plan, training config, holdout
protocol and α sweep recorded BEFORE any eval (see §1). Data generation
(completion leg) LAUNCHED 2026-11-15 ~20:56 PST; training and evals
pending the dataset. Comparison target is v8 (the ML track's champion
and reference).** This is the H2 workstream of ML batch 1 (H1 is v24,
the rollout-leaf distillation; H3 is v26, the learned policy prior);
work happens on branch `research/ml-batch-1-h2` in the `gomoku-ml1-h2`
worktree. v25 does NOT change the search, the leaf structure, the
threat check, sims/priors, or the training recipe — the ONLY intended
difference from v8 is the value net's training data (the completed
10,000-shard dataset vs v8's 4,350) plus a re-swept α (see §1). The
v11 pattern (batch 4 H3): same code, different training data + α.

**Method (carried over):** the fast 10-game loop — play 10 games, read
EVERY move of every game, find THE losing move per game (verified with
the solver or a deeper search — never guessed), patch the structural
cause, record each critical position + correct move as a regression
test, repeat. Max 4 fast-loop iterations before validation, or stop
when an iteration finds no new patchable weakness class. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other — and any such change re-runs the
battery from scratch on the new form. The fast loop runs against v0
as well as v8 (the reference); each iteration plays 10 games vs v8
AND 10 games vs v0. Fast-loop games use development seeds
(1001/2001, 1002/2002, ...) distinct from the bar battery's
pre-registered seeds (100/200), so no development game overlaps an
eval game. The full 4-clause validation battery runs on the FINAL
v25 only.

## 1. Design and pre-registration

**Comparison target: v8** (primary, the ML track's champion:
ladder Elo 0.0, vs v0 0.545, vs v1 0.48, tactical 20/20 + 29/29;
see `research/TRACKS.md`). The ML bar's clause 1 is head-to-head
vs v8 (normal rules, 100 games, alternating colours, win rate >
0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is ladder
Elo strictly higher than v8's 0.0 (identical 20 games/opponent:
random, heuristic, mcts, mcts_sims=100), clause 3 is tactical
pass rate >= v8's (old suite 20/20, hard suite 29/29, run at
`--seed 100`, the battery protocol), clause 4 is all test suites
green (ml pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v25 vs v0 (100 games), v25 vs v7 (100 games),
and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v25 = v8's ValueMCTS with the value net
retrained on the completed 10,000-shard dataset.** Search, leaf
structure (terminal cases exact, net value shrunk by α in quiet
positions only), the 2-ply threat check at the root, 100 sims,
c_puct 1.5, v1 priors — all v8's, unchanged (see
`ml/gomoku_ml/mcts_value.py` and `research/iterations/iter-8.md`
§1). Any difference from v8 is attributable to the training data
(plus the re-swept α, see §1.3).

**The claim under test (from the batch plan and iter-8 §7):**
"more weak-self-play data is not the fix by itself" — v8's
4,350-game net generalised (holdout 0.7165) but played worse at
full weight than the 500-game net's smoke suggested, and v8
concluded the next batch should train on heuristic-opponent
games, not just scale weak self-play. H2 tests the scale axis
directly: 10,000 shards (2.3x v8's shard count; ~4.9x v8's
*unique* game count — see §2's duplication finding) with the
identical recipe. Holdout accuracy and played strength are
reported separately and honestly: either can move without the
other, and the battery — not the holdout — is the verdict.

### 1.1 Data budget (FIXED before generation)

Complete the B1 staged 10k dataset: 5,650 games remain of 10,000
(4,350 exist at `gomoku-b3-b1/ml/runs/b3-b1-10k/data/`). The
completion leg uses the EXACT B1 generator — round-2 checkpoint
(`ml/runs/b3-b1/final.pt`, resnet 32ch/4blocks, 192,714 params),
50 sims/move, workers=4, concurrency 8, c_puct 1.5, Dirichlet
α 0.3 / ε 0.25, temperature 1.0 for 30 moves, max_moves 225, no
resign threshold — via `ml/scripts/selfplay.py` into the SAME
run dir, so the dataset stays one staged run. The ONE deliberate
change is the seed: **seed 1, not seed 0** (see §2 — a same-seed
resume replays identical games; the existing 4,350 shards
contain only 1,449 unique games because the B1 legs all reused
seed 0). Config: `ml/configs/ml1-h2-selfplay-resume.yaml`
(seed 1, run_name b3-b1-10k, games 10000, sims 50). Expected
~5.3–8.7 h at the measured rates (1,073/hr bench, 646–724/hr
sustained under contention). Top-up semantics: `--games` is the
TOTAL wanted; a VM replacement loses at most the in-progress
games, and a further resume after an interruption of THIS leg
must use a FRESH seed again (2, 3, ...) — never re-run leg
seeds. The printed resume command from the script DROPS
`--sims` and `--checkpoint` (and the original legs' `--run-name`
is also absent from it) — re-running it verbatim would generate
random-net games at 100 sims; the full command is recorded in
the config header and in §2.

### 1.2 Hold-out protocol (pre-registered — FIXES a leak in v8's)

v8's holdout was every 10th shard by index. That leaks under
duplication: a holdout shard's exact duplicate usually sits in
the training set (e.g. game0009's copies at 654, 1498, 2910 are
all train shards), so v8's published 0.7165 partly measures
memorization of duplicated games. v25's protocol splits by
UNIQUE game:

- Deduplicate the 10,000 shards by move-sequence MD5 (verified:
  duplicates are byte-identical shards, not just identical move
  lists — see §2).
- Order unique games by first-occurrence shard index; every 10th
  unique game (positions 9, 19, 29, ... in unique order) is a
  HOLDOUT game (~710 games expected).
- Train dir: every shard EXCEPT holdout games' shards (all copies
  of a holdout game are excluded, not just the first occurrence).
- Val dir: one shard per holdout game (its first occurrence).
- Clean-comparison subset: the holdout games whose first
  occurrence is at shard index >= 4350 (the new leg) — games
  NEITHER net trained on (v8 trained only on the first 4,350
  shards). Both nets are evaluated on this subset, honestly.
- Secondary (like-for-like with v8's published number): v25 on
  v8's original holdout shards (every 10th shard of the first
  4,350), labelled LEAKY — both nets' numbers there are inflated
  by the same leak, so the comparison is fair but the absolute
  levels are not generalization measures.

The split is computed by `ml/scripts/split_holdout_unique.py`
(deterministic; writes train/val/clean-subset dirs + a JSON
split manifest) AFTER the dataset completes — the unique-game
set is not final until all 10,000 shards exist. The rule above
is fixed now; only the resulting counts are TBD.

**Checkpoint selection (pre-registered rule, same as v8):**
after training, compute the holdout value sign-accuracy for
BOTH `best.pt` (best internal val loss) and `final.pt` on the
HONEST holdout; use the higher-accuracy one (no-history measure,
the one that predicts leaf quality) as the leaf net, copied to
`ml/runs/ml1-h2/value-net.pt` (the path the registry entry
points at); record BOTH accuracies here. Tie → best.pt. No other
selection criterion is used.

### 1.3 Training config (pre-registered — v8's recipe, unchanged)

Config `ml/configs/ml1-h2-train-10k.yaml` (copy of v8's
`b3-h2-train-4350.yaml` with only the run name, data size, and
comments changed): smoke ResNet 16ch/2 blocks, trained from
scratch with `ml/scripts/train.py` (policy+value loss; only the
value head is used as the leaf): **5 epochs**, batch 64, lr
0.002 cosine, warmup 20 steps, weight decay 1e-4, augmentation
on, internal val_fraction 0.05 (for best.pt selection only —
the game-level holdout above is the reported accuracy), torch
threads 1 (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, per B1's
finding). Run dir `ml/runs/ml1-h2-value-10k`. 5 epochs on
~9,000 train games (~383k samples) is ~1.9M sample-passes,
~2.3x v8's optimization — the same scaling logic v8 used
(epochs fixed, data grows). No architecture change, no
hyperparameter tuning: the variable under test is data scale.

### 1.4 α sweep (pre-registered — v11's protocol, both opponents)

Grid α ∈ {1.0, 0.5, 0.25, 0.125}, via
`ml/scripts/sweep_value_weight.py` (ported from batch 4/v11;
overrides `player._mcts.value_weight` directly, so the registry
records the winner only AFTER the sweep). Two sweep runs, same
seeds (1001/2001), paired per α: 20 games vs v0 AND 20 games vs
v8. Selection rule: highest COMBINED win rate over the 40
games; ties broken toward LARGER α (v11's tie-break: prefer
using the net more when equal). The sweep winner is recorded
in the registry entry (`ENGINE_VERSIONS[25]["value_weight"]`)
before the battery. Outputs:
`research/iterations/sweep-v25-vs-v0.json`,
`research/iterations/sweep-v25-vs-v8.json`.

### 1.5 Integration (pre-registered)

- `ml/gomoku_ml/mcts_value.py`: port v11's per-version α —
  `ValueMCTS.__init__` takes `value_weight` (default
  `VALUE_WEIGHT` = 0.25, so v8 is byte-identical in behaviour),
  stored as `self.value_weight`, used in `_value_leaf_value`
  instead of the module constant. Terminal cases ignore α.
- `ml/gomoku_ml/scoring.py`: ONLY the `ENGINE_VERSIONS[25]`
  registry entry (`{"name": "scoring-v25", ...,
  "value_checkpoint": "ml/runs/ml1-h2/value-net.pt",
  "value_checkpoint_env": "GOMOKU_V25_CHECKPOINT",
  "value_weight": <sweep winner>}` — path relative to the repo
  root, resolved in `ScoringPlayer.__init__`) plus the v11-style
  dispatch update (per-version env var + `value_weight`
  passthrough). `baseline.py` is frozen and untouched;
  `mcts_heuristic.py`, `mcts.py`, `selfplay.py` untouched.
- Per-leaf net eval cost and per-move cost at 100 sims are
  measured and recorded before the matches
  (`ml/scripts/bench_value_leaf.py`).

### 1.6 Eval seeds (pre-registered)

h2h vs v8: v25=100 / v8=200; h2h vs v0: v25=100 / v0=200; h2h vs
v7: v25=100 / v7=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge; tactical `--seed 100`.

### 1.7 Eval order (pre-registered)

(1) Dataset completion + verification (§2). (2) Holdout split
(§1.2). (3) Training (§1.3) + holdout accuracy (§1.2) +
checkpoint selection. (4) Leaf-cost measurement. (5) Tactical
old+hard (full dispatch, `--seed 100`). (6) α sweep (§1.4) —
registry updated to the winner. (7) FAST LOOP (§4): up to 4
iterations of 10 games vs v8 AND 10 games vs v0. (8) FULL
VALIDATION BATTERY on the final v25: tactical re-run, h2h vs
v8 100 games (`ml/scripts/fast_loop_match.py`, seeds 100/200)
-> `research/iterations/batch-v25-vs-v8.json`, 100 games vs v0
-> `batch-v25-vs-v0.json`, 100 games vs v7 ->
`batch-v25-vs-v7.json`, 100-game swap match vs v7
(`ml/scripts/selfplay_batch.py --swap-rule`, balanced,
placement_seed 11, tempo judge) ->
`batch-v25-vs-v7-swap.json`, ladder 20 games/opponent via
`ml/scripts/eval_version.py`. Any battery numbers measured
before the fast loop completes are INTERMEDIATE, labelled as
such; only the final battery counts for the bar.

## 2. Data generation and training

### 2.1 The duplication finding (measured BEFORE the completion leg)

The existing "4,350-game dataset" contains **only 1,449 unique
games**. Verified 2026-11-15 by MD5 of the move sequence over
all 4,350 shards (and full-shard equality for a duplicate pair
— duplicates are byte-identical `.npz` files, not just
identical move lists):

| Copies | Unique games | Shards |
|---|---|---|
| 4x | 645 | 2,580 |
| 3x | 199 | 597 |
| 2x | 568 | 1,136 |
| 1x | 37 | 37 |
| **Total** | **1,449** | **4,350** |

Leg boundaries confirm the mechanism: game0000 == game0645 ==
game1489 == game2901 (the first game of every leg is the same
game). The parallel path seeds worker w with `seed + w`
(`ml/gomoku_ml/selfplay.py:989`), and every B1 leg reused seed
0 — so each leg regenerated the same deterministic stream
prefix. Marginal unique contribution per leg: leg 1: 645, leg
2: +199, leg 3: +568, leg 4: +37 (the longest leg added only 37
new games — its workers barely extended past leg 3's prefixes).

Consequences, all recorded here because they change what v8's
numbers mean and what v25's mean:

- **v8 trained on 1,449 unique games, not 4,350** (each seen ~3x
  over 5 epochs — effectively ~15 epochs on the unique set).
  The "8.7x data fixed generalization" story in iter-8 §2.1 is
  really "3.2x unique games (450 → 1,449) fixed generalization".
- **v8's holdout (0.7165) is leaky** (see §1.2): most holdout
  shards have exact duplicates in the training set. The honest
  holdout number for v8's net is unknown; §1.2's clean-subset
  comparison measures it.
- **A seed-0 completion leg would have added only ~4,200 unique
  games** (its first ~1,449 regenerate the existing prefixes).
  The seed-1 leg adds 5,650 fresh ones (smoke-verified: 4/4
  seed-1 games are new — see §2.2). Final dataset: 10,000
  shards / ~7,099 unique games (1,449 at 3x + 5,650 at 1x).
- The same-seed resume trap was already found by v8's author on
  the batched path (iter-8 §2: game0239 duplicated game0001;
  fresh seeds 1/2/3 used per resume). The B1 10k legs did not
  follow that rule. RULE (restated): a fresh seed every resume,
  forever.

### 2.2 Completion leg (seed 1)

**Smoke test (2026-11-15 ~20:52 PST):** 4 games in /tmp with the
exact completion command shape (seed-1 config, round-2
checkpoint by absolute path, 50 sims, workers 4, concurrency 8):
4/4 merged, manifest consistent, shard schema verified
(planes [T,8,15,15], policy [T,225] rows summing to 1, values in
{-1,1}), and all 4 games confirmed fresh (0 duplicates of the
existing 1,449 unique games). 43 s total. Smoke dir removed.

**Launch (2026-11-15 ~20:56 PST, parent PID 13807):**

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nohup setsid \
  /home/hatch/workspace/projects/gomoku-ml-playground/ml/.venv/bin/python -u \
  ml/scripts/selfplay.py --config ml/configs/ml1-h2-selfplay-resume.yaml \
  --checkpoint /home/hatch/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1/final.pt \
  --run-dir /home/hatch/workspace/projects/gomoku-b3-b1/ml/runs/b3-b1-10k \
  --run-name b3-b1-10k --games 10000 --sims 50 --workers 4 --concurrency 8 \
  > ml/runs/ml1-h2/10k-completion-launch.log 2>&1 &
```

Run log confirms: "resuming: 4350 game(s) already in ...;
generating 5650 more to reach 10000", 50 sims, checkpoint
loaded, parallel mode workers=4/concurrency=8. Launch log:
`ml/runs/ml1-h2/10k-completion-launch.log` (gitignored data).
A waiter (`ml/runs/ml1-h2/wait-10k.sh`, 5-min polls) watches
for completion; it exits when the parent exits. Early rate:
79 games in the first ~5 min (~935/hr); B1's legs declined
under contention (1033 -> 646/hr), so ETA is ~6-8.7 h
(completion ~02:45-05:30 PST Mon 2026-11-16).

**VM replacement (~05:20 UTC 2026-11-16, leg 1 killed).** The VM
was replaced 42 min into leg 1 (uptime reset; parent PID 13807
and all workers killed outright, no graceful exit). 564 games
were orphaned in `.parallel-13807-*/worker*/`; 0 lost. The
recovery path worked as designed: relaunching the same command
shape with the **seed-2 config**
(`ml/configs/ml1-h2-selfplay-resume2.yaml` — fresh seed per the
§1.1 rule, since seed 1's stream prefix is now on disk)
recovered all 564 orphans ("recovered 564 orphaned shard(s)",
numbered game4350–game4913) and resumed at 4,914/10,000,
generating 5,086 more. Note: the recovery's manifest rebuild
(opening all 4,914 shards) took ~2 min of wall time under
sibling contention before the first log line appeared — the
parent is not stuck, just throttled (~11% CPU). Seed split of
the final dataset: shards 0–4349 seed 0 (1,449 unique), shards
4350–4913 seed 1 (564 games), shards 4914–9999 seed 2 (5,086
games). Launch log: `ml/runs/ml1-h2/10k-completion-launch2.log`;
parent PID 1838; waiter relaunched against the new PID.

**Second VM replacement (~07:25 UTC 2026-11-16, leg 2 killed).**
The VM was replaced again ~2 h into leg 2 (parent PID 1838
killed outright). 1,021 games orphaned in
`.parallel-1838-*/worker*/`; 0 lost. Relaunched with the
**seed-3 config** (`ml/configs/ml1-h2-selfplay-resume3.yaml`):
recovered all 1,021 orphans (numbered game4914–game5934) and
resumed at 5,935/10,000, generating 4,065 more. Leg-2 rate:
1,021 games in ~117 min ≈ 523/hr (heavier sibling contention
than leg 1's ~935/hr start); at ~520/hr the remaining 4,065
take ~7.8 h (ETA ~15:30 UTC ≈ 07:30 PST, plus the 08:37 PST
cross-track cron contention). Seed split: shards 0–4349 seed 0
(1,449 unique), 4350–4913 seed 1 (564), 4914–5934 seed 2
(1,021), 5935–9999 seed 3 (4,065). Launch log:
`ml/runs/ml1-h2/10k-completion-launch3.log`; parent PID 1749
(1746 was the setsid wrapper, which exits after forking — the
waiter must watch the python parent, found via ps, not $! or
pgrep); waiter relaunched against the new PID.

**Third VM replacement (~09:25 UTC 2026-11-16, leg 3 killed).**
The VM was replaced again ~2 h into leg 3 (parent PID 1749
killed outright). 1,138 games orphaned in
`.parallel-1749-*/worker*/`; 0 lost. Relaunched with the
**seed-4 config** (`ml/configs/ml1-h2-selfplay-resume4.yaml`):
recovered all 1,138 orphans (numbered game5935–game7072) and
resumed at 7,073/10,000, generating 2,927 more. Leg-3 rate:
1,138 games in ~117 min ≈ 584/hr; at ~580/hr the remaining
2,927 take ~5 h (ETA ~14:30 UTC ≈ 06:30 PST). Seed split:
shards 0–4349 seed 0 (1,449 unique), 4350–4913 seed 1 (564),
4914–5934 seed 2 (1,021), 5935–7072 seed 3 (1,138), 7073–9999
seed 4 (2,927). Launch log:
`ml/runs/ml1-h2/10k-completion-launch4.log`; parent PID 1740;
waiter relaunched against the new PID.

**Fourth VM replacement (~11:25 UTC 2026-11-16, leg 4 killed).**
The VM was replaced again ~2 h into leg 4 (parent PID 1740
killed outright). 1,342 games orphaned in
`.parallel-1740-*/worker*/`; 0 lost. Relaunched with the
**seed-5 config** (`ml/configs/ml1-h2-selfplay-resume5.yaml`):
recovered all 1,342 orphans (numbered game7073–game8414) and
resumed at 8,415/10,000, generating 1,585 more. Leg-4 rate:
1,342 games in ~112 min ≈ 690/hr (contention easing — the
siblings' data-generation jobs appear to have finished); at
~690/hr the remaining 1,585 take ~2.3 h (ETA ~13:45 UTC ≈
05:45 PST). Seed split: shards 0–4349 seed 0 (1,449 unique),
4350–4913 seed 1 (564), 4914–5934 seed 2 (1,021), 5935–7072
seed 3 (1,138), 7073–8414 seed 4 (1,342), 8415–9999 seed 5
(1,585). Launch log:
`ml/runs/ml1-h2/10k-completion-launch5.log`; parent PID 1702;
waiter relaunched against the new PID.

**Fifth VM replacement (~13:26 UTC 2026-11-16, leg 5 killed).**
The VM was replaced again ~2 h into leg 5 (parent PID 1702
killed outright). 1,368 games orphaned in
`.parallel-1702-*/worker*/`; 0 lost. Relaunched with the
**seed-6 config** (`ml/configs/ml1-h2-selfplay-resume6.yaml`):
recovered all 1,368 orphans (numbered game8415–game9782) and
resumed at 9,783/10,000, generating the final 217 games.
Leg-5 rate: 1,368 games in ~119 min ≈ 690/hr; the final 217
take ~19 min (ETA ~13:50 UTC ≈ 05:50 PST). Seed split: shards
0–4349 seed 0 (1,449 unique), 4350–4913 seed 1 (564),
4914–5934 seed 2 (1,021), 5935–7072 seed 3 (1,138), 7073–8414
seed 4 (1,342), 8415–9782 seed 5 (1,368), 9783–9999 seed 6
(217). Launch log:
`ml/runs/ml1-h2/10k-completion-launch6.log`; parent PID 1730;
waiter relaunched against the new PID.

**Clause-4 interim (before the battery):** npm test 55/55,
tsc clean (fresh `npm ci`); targeted ml pytest green
(`test_mcts_value` + `test_scoring` 52 passed,
`test_split_holdout` 3 passed); full ml pytest **246 passed, 1
skipped, 2 xfailed, 0 failures** in 13:24 (run under self-play
contention; the 2 xfails are the known v7 regression xfails).
Clause 4 is re-verified on the final tree after the battery.
Code path verified identical to the B1 legs: `ml/scripts/
selfplay.py`, `ml/gomoku_ml/selfplay.py`, `mcts.py`,
`encoding.py`, `architectures.py`, `game.py` are byte-identical
between this worktree and `gomoku-b3-b1` (the only differing
library files — `kernel_numpy.py`, `mcts_heuristic.py`,
`scoring.py` — are not imported by the neural self-play path:
`mcts.py` imports only `encoding` and `game`).

(TBD — completion time, final games/hour, shard verification:
10,000 shards, manifest total_samples, dedup count (expect
~7,099 unique), policy/value sanity, winner split, game-length
stats.)

**Dataset COMPLETE 2026-11-16 13:54 UTC** ("done. 10000 shards
in .../data"). Final leg: 217 games in 1,437.5 s (543/hr
including the merge + manifest rebuild). Total wall clock for
the completion: ~17 h across 6 legs and 5 VM replacements
(04:38–13:54 UTC); 0 games lost to any replacement (every
orphaned shard recovered).

**Verification (all 10,000 shards read, 2026-11-16):**

* Shard count: 10,000. Total samples: **427,255** (manifest
  agrees exactly).
* Shapes: 0 bad (every shard planes `[T, 8, 15, 15]`, policy
  `[T, 225]`, value `[T]`, moves `[T]`, T = game length).
* Policy rows: 0 bad — every row sums to 1 (atol 1e-4).
* Value targets: 0 bad — all in {-1, 1}; +1 216,280 (50.6%),
  -1 210,975 (49.4%), 0 draws.
* Game length: mean 42.7 plies, min 11, max 105, median 41.
* Winners: Black 5,305 (53.1%), White 4,695 (46.9%), draws 0.

**The unique-game count is 3,244, NOT ~7,099 — the fresh-seed
fix was insufficient, and the mechanism is now proven.** The
completion added only 1,795 genuinely new unique games
(3,244 − 1,449), not 5,650. Multiplicity: {1x: 493, 2x: 965,
3x: 555, 4x: 707, 5x: 203, 6x: 178, 7x: 143}. The cause is the
worker-seed rule in `ml/gomoku_ml/selfplay.py:989`: worker w
seeds its RNG with `seed + w`, so config seeds 1 apart share 3
of 4 worker RNG streams. Consecutive legs (seeds 1, 2, 3, 4,
5, 6) replayed each other's worker streams:

* Every new-leg segment is internally 100% unique (S1 564/564,
  S2 1021/1021, S3 1138/1138, S4 1342/1342, S5 1368/1368, S6
  217/217) — the generator is fine within a leg; only S0 (the
  four seed-0 B1 legs) has internal duplicates.
* Cross-segment overlap follows the shared-worker-seed pattern
  EXACTLY: every pair with NO shared worker seed has ZERO
  shared games (S0×S4, S0×S5, S0×S6, S1×S5, S1×S6, S2×S6 —
  all 0), and every pair WITH shared worker seeds overlaps
  heavily (S0×S1 417, S2×S3 775, S3×S4 885, S4×S5 992), with
  magnitude scaling in the number of shared worker seeds.
* The 4-game smoke test did NOT catch this: with 1 game per
  worker the batching pattern differs from the 8-concurrent
  production pattern, so the RNG interleaving differs and the
  smoke games were fresh. A smoke is only valid at production
  concurrency and game counts.

**Consequences.** (1) The "more data" test is now 3,244 unique
games (2.24x v8's 1,449), not 4.9x — still a real scale-up,
and the shard count is 10,000 as assigned. The retrain
proceeds on the completed dataset as pre-registered; the
unique count is what the generator produced, and regenerating
5,650 games with spaced seeds (~9+ h) is not justified for
this version. (2) The dedup-aware holdout (§1.2) is now
essential, not just prudent: with multiplicity up to 7x, a
shard-index holdout would leak catastrophically. (3) RULE for
future legs (recorded for the program): space resume config
seeds **≥ workers apart** (e.g. 0, 4, 8, 12, … for 4 workers),
or seed workers with a hash of (seed, w) that does not collide
across nearby seeds. (4) A dedup-train variant (one copy per
unique game) is a legitimate follow-up experiment — NOT this
version, which trains v8-style on all non-holdout shards
(§1.2/§1.3 as pre-registered).

### 2.3 Holdout split (COMPLETE 2026-11-16 ~14:05 UTC)

`ml/scripts/split_holdout_unique.py` (deterministic, §1.2 rule):
10,000 shards / 3,244 unique games → **train 9,004 shards
(2,920 unique games)**, **val 324 shards (324 unique games,
honest holdout)**, **val-clean 180 shards** (holdout games
first occurring at index ≥4350 — neither net trained on
them), **val-v8-leaky 435 shards** (v8's original holdout).
996 shards excluded (all copies of holdout games).
Manifest: `ml/runs/ml1-h2-10k/split-manifest.json`.
Train samples ≈ 384k (9,004 × ~42.7); val ≈ 13.8k positions;
val-clean ≈ 7.7k positions.

### 2.4 Training (attempt 2 RUNNING since 2026-11-16 14:31 UTC)

Config `ml/configs/ml1-h2-train-10k.yaml` (v8's recipe, §1.3),
run dir `ml/runs/ml1-h2-value-10k/`, 385,982 samples, 5 epochs,
batch 64, lr 0.002 cosine, warmup 20, wd 1e-4, augmentation on,
internal val_fraction 0.05. **Attempt 1 OOM-killed** (2026-11-16
14:19 UTC, ~30 s after `run_started`, no traceback, no
checkpoint): the in-RAM load held 3.2 GB while host
MemAvailable swung to 1.26 GB (another tenant's job); the
process was the largest on the host and the OOM killer took
it. (Host memory swings wildly — MemAvailable 326 MB → 6.6 GB
→ 1.26 GB → 6.8 GB within ~30 min — so this was not a
one-off.) **Fix (committed):** (1) `load_shards` pre-allocates
outputs (peak ~1x data, not ~2x — the old concatenate peak
was ~6.2 GB); (2) a memmap bundle
(`ml/scripts/build_mmap_bundle.py` → `ml/runs/ml1-h2-10k/
data-mmap/`, 3.1 GB on disk) loaded with `mmap_mode='r'`;
(3) `train()` indexes batches lazily via `train_idx` instead
of materialising 95% copies. Training math is UNCHANGED
(test: RAM vs memmap loss history bit-identical except
wall-clock seconds; bundle spot-verified row-exact against
shards #0 and #5000). Attempt 2 RSS ~0.5 GB (vs 3.2 GB) and
trains. Rate ~2.4 steps/s under sibling contention (load
avg ~4.3; iowait ~3% — the bottleneck is CPU starvation,
not memmap I/O); ~28,650 steps total, ETA ~3 h if the rate
holds, sooner as sibling jobs finish. Failed attempt 1 dir
preserved at `ml/runs/ml1-h2-value-10k-attempt1-oom/`.

**Attempt 2 killed by VM replacement** (2026-11-16 ~15:27 UTC,
uptime reset to 0) at step 17,170 / epoch 3 (of 5). Completed
epochs: **epoch 1** loss 5.7011 / policy 4.7508 / value 0.9502
/ internal-val 5.5526 (1,739 s — slow start, heavy contention
+ cold cache); **epoch 2** loss 5.4806 / policy 4.5995 /
value 0.8811 / internal-val 5.4291 (596 s — rate improved as
siblings finished). `best.pt` = epoch 2 (step 11,460, val
5.4291 < 5.5526), with optimizer state. **Resume:** same run
dir, `--resume .../checkpoints/best.pt --epochs 3` (the
documented resume semantics: 3 ADDITIONAL epochs, cosine
schedule spans the full 5 epochs via schedule_epochs=5).
Note the resume re-seeds the batch-order RNG, so resumed
epochs 3–5 use different batch orders than an uninterrupted
run would — same data, same schedule, valid 5-epoch run.
The epoch-2 best.pt was copied aside to
`ml/runs/ml1-h2/best-epoch2.pt` BEFORE resuming, because the
resumed run resets best_val_loss and would overwrite best.pt
at epoch 3 regardless; the pre-registered selection stays
between the final run's best.pt and final.pt.

(TBD — final epoch losses, val losses, best.pt epoch,
holdout accuracies (honest holdout: best.pt + final.pt, full +
no-history; clean subset: v25 net + v8; leaky v8 holdout,
labelled), checkpoint selection, the winner copied to
`ml/runs/ml1-h2/value-net.pt`.)

**Training COMPLETE 2026-11-16 16:07 UTC** (resumed run:
epochs 3–5, ~38 min). Final losses — epoch 3: loss 5.3471 /
policy 4.5347 / value 0.8124 / val 5.3398 (1,192 s); epoch 4:
5.2527 / 4.4937 / 0.7590 / val 5.3074 (565 s); epoch 5:
5.2771 / 4.5199 / 0.7572 / val 5.2980 (527 s). Internal val
loss fell monotonically across all 5 epochs
(5.5526 → 5.4291 → 5.3398 → 5.3074 → 5.2980), so best.pt =
epoch 5 = final.pt (verified: identical weights, step 28,650).

**Holdout (evaluated 2026-11-16, `ml/runs/ml1-h2/
holdout-{honest,clean,leaky}.json`):**

| set | net | full-planes sign acc | no-history sign acc |
|---|---|---|---|
| honest (324 games, 13,485 pos) | v25 best.pt | 0.5720 | 0.5607 |
| honest | v25 final.pt | 0.5720 (identical weights) | 0.5607 |
| clean, neither trained on (180 games, 7,468 pos) | v25 best.pt | 0.5775 | 0.5599 |
| clean | v8 | 0.5577 | 0.5707 |
| v8's leaky holdout (435 games, 18,446 pos) | v25 best.pt | 0.6891 (LEAKY for v25 too, ~90%) | 0.5731 |
| v8's leaky holdout | v8 (published) | 0.7165 | 0.6278 |

**Reading.** The clean-subset comparison (the only honest
v8-vs-v25 one) is MIXED: v25 wins full-planes (+0.0198) but
LOSES no-history (−0.0108, MSE 1.133 vs v8's 0.987). The
leaf sees search-path history at depth ≥ 2 and zeroed history
at depth < 2, so both measures matter — and on balance 2.24x
unique data did NOT clearly improve the value signal. v8's
published 0.7165/0.6278 was inflated by leakage (its holdout
duplicates sat in its training set); v25's honest 0.5720 /
0.5607 is the number to compare against future honest
holdouts, not against v8's leaky one. **Checkpoint selection
(§1.2): best.pt and final.pt are identical weights → tie →
best.pt**, copied to `ml/runs/ml1-h2/value-net.pt`. The played
strength (battery) is the verdict — holdout is diagnostic.

## 3. Leaf cost measurement (2026-11-16, `ml/runs/ml1-h2/leaf-bench.json`)

`bench_value_leaf.py --checkpoint ml/runs/ml1-h2/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-leaf net eval
mean 7.17 ms / median 2.01 ms / p95 65.9 ms (range 1.24–94.6 ms);
per-move at 100 sims mean 2,233 ms / median 2,187 ms / p95
2,893 ms; net leaf evals per move mean 80.7 (18–99). Same
architecture as v8 (16ch/2 blocks), so the leaf cost is the same
by construction — v8's per-move mean was 5.8 s under heavier
contention; the difference is contention, not the net.

### 3.1 Pre-sweep tactical (α=0.25 placeholder, `--seed 100`)

`eval_tactical_full.py --version 25 --suite both --seed 100`:
**old 20/20, hard 29/29 — perfect.** The retrained net does not
break tactics. (winning-quiet-setup-2 passed with the listed
(10,10); the batch-8 suite-list gap about (9,7) does not bite
here.) The battery re-runs tactical on the FINAL α (§5).

### 3.2 α sweep (RUNNING since 2026-11-16 16:14 UTC)

Grid {1.0, 0.5, 0.25, 0.125} × 20 games vs v0 and vs v8
(seeds 1001/2001), both sweeps in parallel (PIDs 5632/5682).
Selection (§1.4): highest COMBINED win rate over the 40 games;
ties toward LARGER α. (TBD — sweep results, selected α.)

## 4. Fast loop (TBD)

(TBD — up to 4 iterations × (10 games vs v8 + 10 games vs v0),
losing-move workups, patches, regression positions.)

## 5. v25 evaluation (FULL VALIDATION BATTERY — final v25 only) (TBD)

(TBD — tactical, h2h vs v8 / v0 / v7, swap vs v7, ladder.)

## 6. Bar check (v25 vs v8) (TBD)

| Clause | Requirement | v25 | Verdict |
|---|---|---|---|
| 1. h2h vs v8 (100 games) | win rate > 0.50, Wilson 95% CI lower > 0.50 | TBD | TBD |
| 2. Ladder Elo | strictly > v8's 0.0 (identical counts/opponents) | TBD | TBD |
| 3. Tactical | old 20/20 AND hard 29/29 | TBD | TBD |
| 4. Test suites | ml pytest, npm test, tsc green | TBD | TBD |

## 7. What this means (TBD)

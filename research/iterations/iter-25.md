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

### 2.3 Holdout split (TBD)

(TBD — split manifest: train/val game counts and sample counts,
clean-subset size, excluded-duplicate count.)

### 2.4 Training (TBD)

(TBD — training completion time, final/val loss, holdout
accuracy for best.pt / final.pt on the honest holdout (full +
no-history), on the clean subset (v25 net AND v8 checkpoint),
on v8's original leaky holdout (labelled), checkpoint selection,
comparison with v8's net.)

## 3. Leaf cost measurement (TBD)

(TBD — per-leaf net eval cost, per-move cost at 100 sims,
`bench_value_leaf.py` numbers.)

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

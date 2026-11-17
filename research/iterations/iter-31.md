# Iteration 31 — v31: bigger value net on distilled rollout-leaf labels (ML batch 2, H2)

**Status: IN PROGRESS — pre-registered; data labeling running.**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H2 workstream of ML batch 2 (H1 is v30,
the dedup + colour-balanced retrain; H3 is v32, VCF-at-root for
the value engine); work happens on branch
`research/ml-batch-2-h2` in the `gomoku-ml2-h2` worktree. v31
differs from v8 ONLY in the value net's weights, architecture,
and the swept α — same PUCT search, same v1 priors, same sims
(100), same c_puct (1.5), same threat check, same terminal cases
— so the battery is a clean A/B of the bigger distilled net
against v8's weak-self-play net (and, less cleanly, against
v24's small distilled net: v31 also changes the data source).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v8. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v31 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Ports and cost measurements (before pre-registration)

**Ports (ML batch 1 was never merged to master, so its code was
ported into this worktree — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: ML batch 1 H2's memmap version
  (pre-allocated `load_shards` + memmap bundle support + lazy-index
  `train()`), copied VERBATIM from `gomoku-ml1-h2` (sha256
  `7e8cc28d…` matches the source).
- `ml/scripts/generate_distill_data.py`: v24's games+label
  pipeline, copied VERBATIM from `gomoku-ml1-h1` (sha256
  `1ce8e6d3…` matches). The rollout-leaf teacher (v18 code) is on
  master since the batch-9 merge — no cherry-picking needed, and
  no re-verification of the leaf port is needed either (v24's
  401/401 bit-for-bit check stands; the teacher code is the same
  master code v24 verified against).
- `ml/scripts/build_mmap_bundle.py`, `ml/scripts/
  split_holdout_unique.py`, `ml/tests/test_load_shards.py`:
  copied VERBATIM from `gomoku-ml1-h2` (sha256 `92664381…`,
  `2b7930bc…`, `1844f241…` all match).
- `ml/scripts/sweep_alpha.py`: v24's sweep driver, adapted ONLY
  to parameterise the engine version (`--version`, default 31)
  for the player lookup, engine name, and output filenames; the
  match loop, auto-resume, and selection logic are unchanged.
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port (ML batch 1 H1's change, applied by hand to this tree's
  file — same 3-hunk shape); default None → v8's 0.25, so v8 is
  byte-identical in behaviour. `ml/gomoku_ml/scoring.py`:
  dispatch passes `ENGINE_VERSIONS[version].get("value_weight")`
  through to `ValueMCTS`, plus the v31 registry entry
  (`value_checkpoint: ml/runs/ml2-h2/value-net.pt`,
  `value_weight: 1.0` placeholder until the §1.3 sweep sets it).
- `ml/scripts/extract_dedup_games.py`: NEW for this workstream —
  deduplicates the B1 10k dataset by MD5 of the move sequence
  (reusing `split_holdout_unique.split_games`) and writes a
  `generate_distill_data`-format games JSON in unique-order.
- `ml/baseline.py` is FROZEN and untouched.

**Port smoke checks:** ResNet 32ch/4 blocks builds from the
registry (192,714 params — the round-1/2 architecture, matching
iter-25's figure for the B1 round-2 checkpoint); v8 loads and
plays in this worktree (first move 112, centre; `value_weight`
0.25 default preserved); v31 is registered.

**Cost measurements** (run under sibling contention, so means are
inflated — medians are the planning numbers):
- Value-net forward (1 torch thread): 16ch/2 blocks batch=1
  3.8 ms, batch=64 119.7 ms (1.87 ms/sample); **32ch/4 blocks
  batch=1 13.6 ms (3.6x), batch=64 320.2 ms (5.00 ms/sample,
  2.7x)**. Battery games will run ~1.5–2x slower than v24's.
- Rollout-leaf labeling on B1 positions (first 6 deduped games):
  **245 quiet / 275 positions (89% quiet)**; per-call median
  8.77 ms, mean 24.46 ms, p90 73.48 ms. Projected labeling for
  ~120k quiet positions: ~17 min (median-based) to ~48 min
  (mean-based) uncontended — NOT the bottleneck.
- Training estimate: ~111k train samples × 10 epochs / batch 64
  ≈ 17,300 steps; at v25's observed 0.1–0.3 s/step (16ch) scaled
  ~2.7x for the bigger net, ~2.5–5 h under contention.
  Per-epoch checkpoints bound a crash loss to one epoch.

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, tactical 20/20 + 29/29 on the CORRECTED suite — re-verified
by the coordinator at seed 100; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical ≥ v8's (old suite 20/20,
hard suite 29/29, run at `--seed 100` on the corrected suite),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v31 vs v7 (100
games), v31 vs v0 (100 games), and a 100-game swap match vs v7
under the tempo judge (the harness default). Non-transitivity is
recorded prominently.

**The hypothesis (plan H2):** v24 proved distillation works as ML
— the small net learned the rollout leaf (pooled in-distribution
holdout sign accuracy 0.7766 vs v8's 0.504 on the same labels)
— but a faithful copy of the rollout leaf plays only at v8's
level (0.515 h2h, ladder Elo 52.5) AT V8'S SIZE. The value head,
not the prior, is the binding constraint (v26: the prior ranks
the setup move 1st on all 3 tactical failures; the value head
overrides it). v31 tests whether SCALE breaks the ceiling: a
larger net (32ch/4 blocks, 1.52x v8's params) trained on ~3x
v24's labelled positions from the deduped dataset. If the bigger
net fits the leaf function better (higher holdout sign accuracy)
AND that fit converts to played strength, v31 beats v8; if the
fit improves but strength does not, the ceiling is the leaf
function itself (or the 100-sim search horizon, v24's finding),
not the net's capacity — and that is itself a result.

**The improvement: v31 = v8's search + a ResNet 32ch/4 blocks
value net trained to predict v18's rollout-leaf value.**
Everything else is v8 unchanged (see the header). The net is
consulted ONLY in quiet positions (terminal cases stay exact, as
in v8), so training data is quiet positions only.

### 1.1 Dataset (pre-registered)

- **Source: the DEDUPED B1 10k dataset — 3,244 unique games**
  (10,000 shards, 6,756 duplicates; dedup by MD5 of the move
  sequence via `ml/scripts/extract_dedup_games.py`). VERIFIED at
  extraction: unique count is exactly 3,244, and the implied
  holdout indices match v25's split manifest exactly (see §2).
  Games file: `ml/runs/ml2-h2/games-b1-dedup.json` (gitignored
  data). No new game generation is needed — the games exist.
- **Sampling:** ALL quiet positions of every game (labeling is
  ~9 ms median per position; game generation is already paid
  for). A position is quiet iff `_terminal_leaf_value == 0.0`.
  Positions are encoded with the real game history (boards 2/4
  plies back, last move) exactly as in training
  (`encoding.encode`). Expected yield: ~123k quiet positions
  (3,244 games × ~42.7 plies × ~89% quiet), ~111k train /
  ~12k holdout.
- **Labels:** the v18 rollout-leaf value (`_rollout_leaf_value`,
  blend=False) — a single leaf evaluation per position, from the
  side-to-move perspective, in [-1, 1]. Rollout-found terminals
  (±1) are kept as-is: they are the leaf's real signal.
- **Holdout (game-level, dedup-aware, v25's rule):** every 10th
  unique game in first-occurrence order (positions 9, 19, 29,
  ... — 324 games) is routed to `ml/runs/ml2-h2/data/b1-holdout/`
  by the label phase (`--holdout-every 10`) and NEVER trained on.
  Because the games file is already deduped, no holdout game can
  leak into training through a duplicate copy. The holdout set is
  IDENTICAL to v25's (verified against the manifest at
  extraction), so v31's holdout numbers are comparable to v25's
  by game — though the labels differ (rollout vs outcome).
- **NOT used for training:** v24's engine-game distill data
  (37,227 positions from v7/v18 self-play + random). It is used
  ONLY as a cross-distribution diagnostic (§1.5c): v31 trains on
  B1 positions, so its accuracy on v24's engine-game holdout
  measures distribution transfer in the opposite direction from
  v24's B1-diag (0.6257). No decision hinges on it.
- **Variety check:** not needed — the B1 games are 3,244 distinct
  move sequences by construction (dedup verified).

### 1.2 Architecture and training (pre-registered)

- **Architecture:** ResNet **32ch/4 blocks** (192,714 params),
  trained from scratch (random init — NOT from v8's or v24's
  checkpoint; the labels are a different target and the shape
  differs anyway). Confirmed in the registry (`resnet` with
  `channels: 32, blocks: 4`).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **10 epochs** (~1.06M
  sample-passes, ~2x v24's 525k — enough optimization for the
  larger net on ~3x the data), augmentation on (dihedral),
  val_fraction 0.05 (random split within the train set, for
  best-checkpoint selection only), seed 0.
- **Config:** `ml/configs/ml2-h2.yaml`; run dir
  `ml/runs/ml2-h2-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, B1's finding).
- **Memmap path (mandatory):** the train shards are bundled with
  `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml2-h2/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as v24):**
  best-by-internal-val-loss (`best.pt`), copied to
  `ml/runs/ml2-h2/value-net.pt` (gitignored data, the path v31's
  registry entry points at). `checkpoint_every_epochs: 1` for
  crash recovery (v24's lesson: a restart drain killed training
  mid-run with no per-epoch checkpoint).
- **Resume rule (pre-registered, v24's lesson):** if training is
  killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans the
  full 10 via `schedule_epochs`); back up the pre-kill `best.pt`
  first, because the resumed run resets `best_val_loss` and
  would overwrite it at the next epoch regardless — the final
  best is chosen by comparing the resumed run's best val loss
  against the pre-kill best.

### 1.3 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.5, 1.0, 2.0}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 31`. The grid brackets
v24's winner (1.0) on both sides: distilled values are small
(mean|v| ~0.06 for v24), so 2.0 is a live arm — it would make
the net's effective contribution (~0.12) comparable to v8's
(α=0.25 × mean|v| 0.53 ≈ 0.13) — while 0.5 covers the case
where the bigger net's better-calibrated values want less
weight. Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure). **No-signal
rule:** if all three arms are within one win of each other,
pick α=1.0 (v24's winner) and say so — the sweep found no
signal. The sweep SELECTS α; the 100-game battery is the
verdict (standing lesson: small-sample smokes mislead — v24's
0.65 smoke at α=1.0 did not reproduce at 100 games). The
registry entry's `value_weight` is updated to the winner after
the sweep.

### 1.4 What would change the design

- If the distilled net's holdout sign-accuracy on its OWN
  (B1-dedup) holdout is at chance (~0.50), the distillation
  failed to learn the leaf function — stop and report, do not
  battery a chance-level net.
- If the α sweep's best is within noise of the worst (all three
  within one win of each other), pick α=1.0 per the §1.3
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain (e.g. adding
  v24's engine-game data, or more epochs) re-runs the α sweep
  and the battery from scratch on the new form.

### 1.5 Holdout diagnostics (pre-registered)

All via `ml/scripts/eval_value_holdout.py` (sign accuracy +
MSE, full-planes and no-history views) on rollout-leaf labels:
- (a) **v31 net on the B1-dedup holdout** (324 games, ~12k
  positions) — in-distribution; the §1.4 stop rule applies
  here.
- (b) **v24's net** (`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`,
  by absolute path) **on the same B1-dedup holdout** — same
  labels, cross-net comparison: does the bigger net fit the
  leaf better than the small net on B1 positions?
- (c) **v31 net on v24's engine-game holdout**
  (`gomoku-ml1-h1/ml/runs/ml1-h1/data/{v7,v18,random}-holdout/`,
  3,378 positions) — out-of-distribution transfer; compare
  with v24's pooled 0.7766 on the same positions.
- (d) **v8's net on the B1-dedup holdout** — baseline (expect
  ~0.50, as on v24's labels).
No decision hinges on (b)–(d); they diagnose whether scale
improved the fit and whether the fit transfers across
distributions.

### 1.6 Eval seeds (pre-registered)

h2h vs v8: v31=100 / v8=200; h2h vs v0: v31=100 / v0=200; h2h vs
v7: v31=100 / v7=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge; tactical `--seed 100`.

### 1.7 Eval order (pre-registered)

(1) Dataset dedup + labeling (§2). (2) Memmap bundle + training
(§3) + holdout diagnostics (§4). (3) Leaf-cost measurement.
(4) α sweep (§5) — registry updated to the winner. (5) FAST
LOOP (§5): up to 4 iterations of 10 games vs v8 AND 10 games
vs v0. (6) FULL VALIDATION BATTERY on the final v31 (§6):
tactical re-run, h2h vs v8 100 games, 100 games vs v0, 100
games vs v7, 100-game swap match vs v7, ladder 20
games/opponent, clause-4 suite re-run. Any battery numbers
measured before the fast loop completes are INTERMEDIATE,
labelled as such; only the final battery counts for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games
fast_loop_match.py --version-a 31 --opponent v8 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v31-vs-v8.json
# Clause 2: ladder (20 games/opponent, mcts_sims 100)
eval_version.py --version 31 --ladder-games 20
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 31 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, swap vs v7 (defaults: balanced/tempo, seeds 100/200)
fast_loop_match.py --version-a 31 --opponent v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v31-vs-v7.json
fast_loop_match.py --version-a 31 --opponent v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v31-vs-v0.json
selfplay_batch.py --engine-a v31 --engine-b v7 --games 100 --swap-rule --out research/iterations/batch-v31-vs-v7-swap.json
```

## 2. Data generation (actuals)

**Dedup (done 2026-11-17 ~00:51 PST):**
`ml/scripts/extract_dedup_games.py` over the B1 10k shards:
10,000 shards → **3,244 unique games** (6,756 duplicate
shards), written to `ml/runs/ml2-h2/games-b1-dedup.json`.
Verification PASSED: unique count is exactly 3,244 AND the
implied holdout first-shard indices match v25's split manifest
exactly (324 games) — v31's holdout set is identical to v25's.

**Labeling (COMPLETE 2026-11-17 ~01:42 PST):**
`generate_distill_data.py label` over all 3,244 unique games
(`--holdout-every 10`). **VM replacement killed the first run
at game 2949/3244 (91%)**; the resume-safe label phase skipped
the 2,959 existing shards and finished the remaining 295 games
in 217 s. Final totals (summed from the shards — the resumed
run's LABEL_JSON covers only its own 295 games):

| Split | Games | Quiet positions | ±1 labels | Gradient labels | Mean \|v\| |
|---|---|---|---|---|---|
| Train (`data/b1/`) | 2,920 | 110,905 | 10,751 (9.7%) | 100,154 | 0.1965 |
| Holdout (`data/b1-holdout/`) | 324 | 12,009 | 1,070 (8.9%) | 10,939 | 0.1904 |
| **Total** | **3,244** | **122,914** | **11,821 (9.6%)** | **111,093** | — |

138,695 total positions, 88.6% quiet — close to the §0
projection (~123k). Note the label magnitudes: mean|v| ~0.19
is ~3x v24's ~0.06 — B1 weak-net games are tactically wilder
(9.7% rollout-terminal ±1 vs 3.8–5.1% in v24's engine sources),
so the ±1 mass pulls the mean up. The §1.3 α grid {0.5, 1.0,
2.0} still brackets the right region (α=0.5 on these values
contributes about what α=1.0 did on v24's); the sweep decides,
no pre-registration change needed. Memmap bundle built by the
chain script: `ml/runs/ml2-h2/data-mmap/` (110,905 samples).

## 3. Training (actuals)

**Launched 2026-11-17 09:44 UTC** by the chain script
(`ml/runs/ml2-h2/chain-label-train.sh`, which verified the
2920/324 shard counts and built the bundle first): config
`ml/configs/ml2-h2.yaml`, 10 epochs, batch 64, memmap bundle.
Run dir `ml/runs/ml2-h2-train/`. RSS ~335 MB — the memmap path
works (vs the 3.2 GB in-RAM load that OOM-killed v25's attempt
1). Steps: 105,360 train samples (95% of 110,905) / 64 ≈ 1,646
steps/epoch, **16,460 total**. Early pace ~2.2 s/step (epoch 1,
step 350 at 09:57 UTC) — ETA ~10 h at that pace, sooner if
sibling contention eases (v25's rate improved 3x as siblings
finished). Per-epoch checkpoints on (`checkpoint_every_epochs:
1`); the §1.2 resume rule applies if a restart drain kills it.

**Interrupt and resume (2026-11-17 11:36 UTC):** a VM
replacement killed training at step 3,600 / epoch 3. Completed
epochs: **epoch 1** train 0.0691 / val 0.0677 (3,144 s),
**epoch 2** train 0.0326 / val 0.0218 (3,054 s) — val loss
already 3x below v24's final 0.0617 after 2 epochs on 3x the
data. `best.pt` verified before resuming (step 3294, epoch 2,
ResNet 32ch/4 blocks, 192,714 params, optimiser state present,
all 72 tensors finite) and backed up to
`ml/runs/ml2-h2/best-epoch2.pt` (the resumed run resets
`best_val_loss`, so the final best is chosen by comparing the
resumed run's best against 0.0218). Resumed 11:37 UTC into the
same run dir with `--resume .../checkpoints/best.pt --epochs 8`
(8 ADDITIONAL epochs; cosine schedule continues exactly from
step 3294 — LR 0.00180 at step 3410 vs 0.00178 at the kill).
Post-resume pace ~0.9 s/step (contention easing); ~13,030 steps
remain.

**Second interrupt and resume (2026-11-17 13:37 UTC):** another
VM replacement killed training at step 12,580 / epoch 8.
Completed epochs since the resume: **epoch 3** train 0.0206 /
val 0.0176 (1,019 s), **epoch 4** 0.0158 / 0.0163 (839 s),
**epoch 5** 0.0147 / 0.0131 (1,005 s), **epoch 6** 0.0126 /
0.0123 (1,452 s), **epoch 7** 0.0117 / 0.0120 (1,719 s) — val
loss still decreasing monotonically, flattening after epoch 5.
`best.pt` verified (step 11529, epoch 7, all 72 tensors finite,
optimiser state present) and backed up to
`ml/runs/ml2-h2/best-epoch7.pt` (val 0.0120 — the standing best
to beat). Resumed 13:38 UTC with `--resume .../best.pt
--epochs 3` (3 ADDITIONAL epochs, completing the 10-epoch plan;
cosine schedule continues from step 11529, LR 0.00040 at step
11600). ~4,941 steps remain, ETA ~45–60 min at the recent
~15–25 min/epoch pace.

**Training COMPLETE 2026-11-17 14:38 UTC** (10 epochs total:
2 pre-interrupt + 5 resumed + 3 re-resumed; 16,460 steps;
best.pt = final.pt = epoch 10, step 16470, verified identical
weights). Final epochs: **epoch 8** train 0.0108 / val 0.0113
(1,433 s), **epoch 9** 0.0103 / 0.0114 (559 s), **epoch 10**
0.0102 / 0.0112 (1,513 s). Val loss decreased monotonically
across all 10 epochs (0.0677 → 0.0218 → 0.0176 → 0.0163 →
0.0131 → 0.0123 → 0.0120 → 0.0113 → 0.0114 → 0.0112) — no
overfitting by epoch 10, but the pre-registered plan is 10
epochs and the battery is the verdict, so no extension. **Best
checkpoint: epoch 10 (`best.pt`, step 16470, val 0.0112)** —
the resumed runs' best beats the epoch-7 backup's 0.0120
outright, so the resume best-reset flaw never bit and no
restore from `best-epoch7.pt` was needed (same as v24's
resume). Final checkpoint copied to the registry path:
**`ml/runs/ml2-h2/value-net.pt`** (gitignored data; the
coordinator needs this path to verify on the merged tree).
Total wall clock for training: ~4h54m across 3 runs and 2 VM
replacements (09:44–14:38 UTC); 0 epochs lost to either
replacement (per-epoch checkpoints + the §1.2 resume rule).

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th unique game, never trained on),
targets = rollout-leaf labels. v31 net =
`ml/runs/ml2-h2/value-net.pt`; v24 net =
`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml2-h2/holdout-{b1,v7,v18,random}.txt`. Sanity check:
this run reproduces v24's published numbers EXACTLY on its own
holdout (v7 0.7713, v18 0.8384, random 0.7055, pooled 0.7766)
— the eval setup is sound.

| Holdout | n pos | v31 sign (full) | v31 sign (no-hist) | v31 MSE | v31 mean\|v\| | v24 sign (full) | v8 sign (full) |
|---|---|---|---|---|---|---|---|
| B1-dedup (in-dist) | 12,009 | **0.8675** | 0.6292 | 0.0129 | 0.163 | 0.6252 | 0.4516 |
| v7 (engine) | 2,050 | 0.5718 | 0.3144 | 0.3202 | 0.354 | 0.7713 | — |
| v18 (engine) | 790 | 0.7148 | 0.2733 | 0.1550 | 0.188 | 0.8384 | — |
| random | 538 | 0.8069 | 0.6157 | 0.0932 | 0.322 | 0.7055 | — |
| **engine pooled** | **3,378** | **0.6458** | **0.3290** | — | — | **0.7766** | — |

**Verdict: the distillation learned the leaf function ON ITS
OWN DISTRIBUTION — the §1.4 stop condition (chance-level
holdout) does NOT trigger.** In-distribution sign agreement
0.8675 vs 0.50 chance, far above v24's 0.6252 on the same B1
positions: the bigger net + 3x data fits the leaf much better
where it trained. v8's net is below chance on rollout labels
(0.4516), as expected — the two nets encode genuinely
different value functions.

**But the fit does NOT transfer to engine positions, and the
net is catastrophically history-dependent out-of-distribution.**
On v24's engine-game holdout v31 scores only 0.6458 pooled
(v24: 0.7766), and the no-history view is BELOW chance on
engine positions (v7 0.3144, v18 0.2733 — systematically the
wrong sign, with mean|v| ballooning to 0.46–0.57). Even
in-distribution the net leans hard on history planes (B1 0.8675
→ 0.6292 without them), while v24's net is history-robust on
engine positions (0.7713 both views on v7). At inference the
leaves get search-path history (neither the real-game history
of the full-planes view nor the zeroed history of the
no-history view), so the operating point lies between the two
views — but on engine positions both bounds are worse than
v24's. The inference distribution is engine games (v31 plays
v8/v7/v0), so v31's effective leaf quality at inference is
closer to the engine numbers than the B1 numbers. **This
foreshadows that scale alone may not beat v8** — v24 tied v8
with a BETTER engine-position net — but holdout is diagnostic,
not the verdict: the α sweep and the battery measure played
strength, and they proceed as pre-registered. No retrain is
authorised by §1.4 on holdout grounds (the stop rule is
chance-level own-holdout, which did not trigger); a
data-mixing change would need a fast-loop loss to motivate it,
and none exists yet.

## 5. α sweep and fast loops

**Leaf cost (2026-11-17, `ml/runs/ml2-h2/leaf-bench.json`):**
`bench_value_leaf.py --checkpoint ml/runs/ml2-h2/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-leaf net eval
mean 27.52 ms / median 6.05 ms / p95 91.75 ms (range 2.16–303.9
ms); per-move at 100 sims mean 6,583 ms / median 3,657 ms / p95
13,204 ms; net leaf evals per move mean 79.9 (11–99). v31 is
~3x slower per move than v24 (mean 2,233 ms) — the 32ch net's
forward is 3.6x v8's (13.6 ms vs 3.8 ms batch=1). Battery games
will run ~3–5 min each under contention; the four 100-game
matches are the long pole of this workstream.

TODO — sweep results, registry α update, fast loops.

**α sweep — COMPLETE 2026-11-17 ~09:38 PST** (20 games each vs
v8, seeds 1001/2001; per-alpha JSONs
`research/iterations/alpha-v31-<α>-vs-v8.json`). **VM
replacement killed the first run** after the α=0.5 arm
completed (11W-9L-0D); the sweep's auto-resume skipped the
complete arm and restarted α=1.0 fresh — no games lost.

| α | W-L-D | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| **0.5** | **11-9-0** | **0.550** | **[0.3421, 0.7418]** | 7-3-0 | 4-6-0 |
| 1.0 | 4-16-0 | 0.200 | [0.0807, 0.4160] | 2-8-0 | 2-8-0 |
| 2.0 | 10-10-0 | 0.500 | [0.2993, 0.7007] | 10-0-0 | 0-10-0 |

**Winner: α = 0.5** — registry `value_weight` set to 0.5. The
result is non-monotone in α again (v24's sweep was too), and
v24's winner (1.0) is a VALLEY for v31 (0.20) — the mechanism
is the one §2 predicted: v31's values are ~3x larger
(mean|v| ~0.19 vs v24's ~0.06), so α=1.0 here contributes ~3x
what it did for v24, and the noisier engine-position values
(see §4) hurt at full weight. α=2.0 is the strangest arm:
**Black 10-0, White 0-10** — at full amplification the net's
values dominate the search completely, and the bias aligns
with Black's attacking edge while destroying White's defence.
The §1.3 no-signal rule does not apply (0.5 vs 1.0 differ by 7
wins). All three CIs overlap, so the sweep selects α but
proves nothing — the 100-game battery is the verdict, as
pre-registered.

**Fast loop iteration 1** (TODO).

## 6. Validation battery

TODO — clauses 1–4 + secondaries.

### Battery verdicts

TODO.

## 7. Verdict

TODO.

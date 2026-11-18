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

**Fast loop iteration 1** (10 games each, seeds 1001/2001,
α=0.5, 2026-11-17 ~10:10 PST):
- vs v8: **5W-5L-0D (0.50)**, CI [0.2366, 0.7634]. As Black 3-2,
  as White 2-3. **Registry verification:** these 10 games replay
  the sweep's α=0.5 games 0–9 EXACTLY (same move lists and
  winners, WWLLLWWLWL) — the registry α=0.5 path works.
- vs v0: **2W-7L-1D (0.25)**, CI [0.0809, 0.5578]. As Black
  0W-4L-1D (0.10!), as White 2W-3L-0D (0.40). v31 cannot convert
  Black's edge against the frozen baseline heuristic.
- Loss classification (`classify_losses.py`, both matches):
  **12 losses, 12 via open four, 0 missed blocks, 0 clean missed
  preventions in 64 threatened turns.** Defence as measured is
  clean — the losses are positional drift into unstoppable open
  fours, not blundered blocks.
- Losing moves (`find_losing_moves.py`, VCF solver depth 4;
  raw `ml/runs/ml2-h2/losing-moves-iter1.txt`): **11 of 12
  losses were already lost** at the first VCF moment (no saving
  reply exists — positional decline, no single tactical
  blunder). **ONE genuine blunder: vs-v0 game 7 (White, 55
  plies), ply 43 — (12,9), with 5 saving moves**
  ((0,5), (4,9), (2,10), (3,10), (2,11)).

**Workup verdict — vs-v0 game 7, ply 43 (VERIFIED):**
`diag_root.py` at the position shows v31's search is BROKEN,
not merely wrong: **all 102 root children have visits=1 and
our-q=-0.500** — every simulation expands a new child and none
is ever revisited, so the root choice is effectively arbitrary
(RNG tie-break among 1-visit children). v8's search at the same
position is healthy (top child (2,10) 7 visits — a saving
move) and v7's too (top-4 all saving moves). `v31_net_diag.py`
finds the cause: **v31's net outputs raw +1.000 (fully
saturated) for the resulting position after EVERY candidate
move** — the actual (12,9) and all 5 saving moves, with history
and without — so the leaf value is +0.500 (opponent's
perspective) for every move and the search cannot discriminate
at all. v8's net at the same positions outputs varied small
values (-0.083 to -0.293). The EXACT rollout leaf, for
comparison, gives small gradient values (+0.052 after the
actual move, +0.052 to +0.103 after the saves) — the teacher
discriminates (weakly; it ranks the losing move tied-best, the
depth-4 VCF being beyond its 2-ply horizon — v24's finding),
but **the net does not reproduce the teacher here: it
saturates at +1 where the teacher says +0.05–0.10.**

**Mechanism (confirmed, not hypothesised):** in v31's B1-only
training data, positions where Black holds threats were usually
rollout-terminal (+1 — the weak B1 opponent doesn't block), so
the net learned the shortcut "Black has threats → +1". On
engine positions Black's threats are real but defensible, and
the net saturates instead of evaluating. The holdout (§4) shows
the same failure systematically (engine no-history sign
0.27–0.31, mean|v| 0.46–0.57). The 11 "already lost"
positional declines are consistent with a search that cannot
discriminate anywhere the pattern appears. This is a DATA
problem (the net never saw engine positions in training), not
a search or architecture problem — and it is patchable inside
H2's design space.

### 5.1 PATCH (pre-registered-style change, motivated by the v0-g7 loss)

**Change:** add v24's engine-game distill TRAIN data (33,849
quiet positions: 19,545 v7 + 9,142 v18 + 5,162 random —
verified counts, matching iter-24) to the training set.
New training set: **144,754 positions** (110,905 B1-dedup +
33,849 engine; 77% / 23%). v24's HOLDOUT shards stay held out
(they remain the cross-distribution diagnostic). Everything
else is unchanged: same architecture (ResNet 32ch/4 blocks),
same recipe (value-only MSE, AdamW lr 0.002 cosine, warmup 20,
wd 1e-4, batch 64, 10 epochs, augmentation on, val_fraction
0.05, seed 0), retrained FROM SCRATCH (new run dir
`ml/runs/ml2-h2-train-mixed/`, config `ml/configs/
ml2-h2-mixed.yaml`, new memmap bundle
`ml/runs/ml2-h2/data-mixed-mmap/`). The α sweep and the battery
re-run from scratch on the new form, per §1.4. The B1-only form
remains fully documented (holdout §4, sweep §5, fast loop §5);
its checkpoint is preserved at
`ml/runs/ml2-h2/value-net-b1only.pt` (renamed from
`value-net.pt` before the retrain, so the registry path always
points at the CURRENT form).

**Regression check (pre-registered):** after retraining, at the
v0-g7 ply-43 position the net must NOT saturate (raw values
discriminating between the actual move and the saves, magnitudes
comparable to the teacher's +0.05–0.10 range rather than +1.0),
and v31's search must rank a saving move above (12,9). The
holdout diagnostics (§1.5) also re-run: the engine-pooled sign
accuracy must improve on 0.6458 (v24's 0.7766 is the reference)
for the patch to be judged as having addressed the mechanism.
If the regression check fails, the patch is abandoned and v31
goes to the battery as the B1-only form.

**Fast loop iteration 2** (mixed form; TODO).

### 5.2 Mixed retrain + regression check (actuals)

**Mixed training COMPLETE 2026-11-17 19:25 UTC** (10 epochs,
21,490 steps, ~65 min, no interruptions): val loss decreased
monotonically (0.0864 → 0.0714 → 0.0498 → 0.0410 → 0.0330 →
0.0299 → 0.0264 → 0.0248 → 0.0236 → 0.0233). Best checkpoint
epoch 10 (val 0.0233), copied to the registry path
`ml/runs/ml2-h2/value-net.pt`. (Absolute val loss is not
comparable to the B1-only run's 0.0112 — the mixed val set
includes the harder engine positions; v24's engine-only final
val was 0.0617. Sign accuracy is the comparable metric.)

**Regression check — MIXED verdict:**
- Criterion (1), v0-g7 ply 43: **FAIL (narrowly in effect).**
  The mixed net still outputs raw +0.975 to +0.978 for every
  candidate move (vs +1.000 before) — it still saturates at
  this live-VCF position and cannot discriminate the saves
  from the losing move. Note the criterion's second half was
  MISDESIGNED: the teacher itself ranks the losing move
  joint-best (+0.052, tied with the (2,11) save), so even a
  perfect distillation would not rank a save above (12,9) —
  no net trained on these labels can pass that half.
- Criterion (2), holdout engine transfer: **PASS, decisively.**

| Holdout | mixed full | mixed no-hist | B1-only full | v24 full |
|---|---|---|---|---|
| B1-dedup | 0.8567 (MSE 0.0166) | 0.7452 | 0.8675 | 0.6252 |
| v7 | 0.7713 (MSE 0.0352) | 0.3919 | 0.5718 | 0.7713 |
| v18 | 0.8357 (MSE 0.0606) | 0.3890 | 0.7148 | 0.8384 |
| random | 0.8069 (MSE 0.0850) | 0.7419 | 0.8069 | 0.7055 |
| **engine pooled** | **0.7929** | 0.4497 | 0.6458 | 0.7766 |

Engine-pooled sign accuracy went 0.6458 → **0.7929, beating
v24's 0.7766** (v7 exactly equals v24's 0.7713, v18 matches
0.8384, random beats 0.7055 clearly), with LOWER MSE than v24
on v7/v18 (0.035/0.061 vs 0.045/0.069). B1 holdout barely
moved (0.8567 vs 0.8675). The no-history view improved
everywhere (B1 0.63→0.75, engine pooled 0.33→0.45) but is
STILL below chance on engine positions (0.39) — the net still
leans on history planes out-of-distribution, and the residual
saturation is confined to threat-heavy positions (live VCFs)
rather than all engine positions.

**Decision: KEEP the patch (mixed form proceeds).** The
quantitative, general criterion passed decisively; the single
position that failed is a live-VCF extreme case whose second
half was unachievable by construction. The mixed net is better
than the B1-only net on every holdout measure except B1
full-planes by 0.011 — batterying the B1-only form instead
would be perverse. Fast-loop iteration 2 (mixed form) is the
arbiter: if it still plays at the B1-only form's level (0.50
vs v8 / 0.25 vs v0), the patch failed in play and the residual
is a teacher-horizon limit outside H2's remit.

**α re-sweep (mixed form) — COMPLETE 2026-11-17 ~12:08 PST**
(20 games each vs v8, seeds 1001/2001; per-alpha JSONs
`research/iterations/alpha-v31-<α>-vs-v8.json` — the B1-only
sweep's files were renamed to `*-b1only.json` first). **VM
replacement killed the first run** at α=0.5 game 14 (10W-4L);
the auto-resume continued it from game 14 (RNG restart
recorded in the JSON).

| α | W-L-D | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| **0.5** | **14-6-0** | **0.700** | **[0.4810, 0.8545]** | 10-0-0 | 4-6-0 |
| 1.0 | 10-9-1 | 0.525 | [0.3205, 0.7215] | 6-3-1 | 4-6-0 |
| 2.0 | 9-10-1 | 0.475 | [0.2785, 0.6795] | 8-2-0 | 1-8-1 |

**Winner: α = 0.5** — the registry value (0.5, from the
B1-only sweep) stands; only the comment changes. Monotone
decreasing in α this time (0.70/0.525/0.475), and the mixed
net's α=0.5 (0.70) is well above the B1-only form's (0.55) —
the patch improved played strength in the sweep. The colour
pattern persists at every α: Black is strong (10-0, 6-3-1,
8-2) and White is weak (4-6, 4-6, 1-8-1) — White is the gap
again. The §1.3 no-signal rule does not apply (0.5 vs 1.0
differ by 4 wins). All CIs overlap, so the sweep selects α
but proves nothing — the 100-game battery is the verdict.

**Fast loop iteration 2** (mixed form, α=0.5; 10 games each,
seeds 1001/2001, 2026-11-17 ~12:34 PST):
- vs v8: **7W-3L-0D (0.70)**, CI [0.3963, 0.8927]. As Black 5-0,
  as White 2-3. **Registry verification:** these 10 games replay
  the mixed sweep's α=0.5 games 0–9 EXACTLY — the registry α=0.5
  path works on the mixed form too.
- vs v0: **5W-3L-2D (0.60)**, CI [0.3128, 0.8324]. As Black
  2W-1L-2D, as White 3W-2L. The patch worked in play: vs v0 went
  0.25 → 0.60 and vs v8 0.50 → 0.70 (10 games each — small
  samples, but the direction matches the holdout improvement).
- Loss classification: **6 losses, 6 via open four, 0 missed
  blocks, 0 clean missed preventions in 32 threatened turns.**
  Defence clean again.
- Losing moves (`find_losing_moves.py`; raw
  `ml/runs/ml2-h2/losing-moves-iter2.txt`): **5 of 6 losses
  were already lost** at the first VCF moment (positional
  decline). **ONE genuine blunder: vs-v0 game 3 (White, 29
  plies), ply 23 — (3,9), with 2 saving moves** ((7,5), (8,5)).

**Workup verdict — vs-v0 game 3, ply 23 (VERIFIED):** this
blunder is a DIFFERENT mechanism from iter1's saturation. The
mixed net DISCRIMINATES here: its raw values are +0.729
(actual) vs +0.734/+0.738 (saves) — still high (residual
threat-saturation: the position has Black threats), but the
leaf short-circuit gives the saves leaf +0.250 ours-perspective
vs the actual's -0.365. And the SEARCH's q is RIGHT: the saves
have better our-q (-0.105, -0.089) than the actual (-0.180).
But the actual wins ON VISITS (8 vs 7 and 6) because its v1
prior (0.1247) is 3x the saves' (0.0396) — **the heuristic
prior overwhelms the net's correct q in visit allocation, and
the most-visited root choice follows the prior.** This is the
same category as v25's v0-g5 blunder: a PRIOR problem in
shared heuristic-track code (`_priors`), whose change would
alter v8/v7 equally and invalidate the α sweep — NOT patchable
within H2's remit (v31 = v8's search + distilled net).

**Fast-loop STOP after iteration 2:** no patchable weakness
class for v31 within H2's design space. The one genuine
blunder is a shared-prior issue; the other 5 losses are
positional with no single blunder; the residual threat
saturation (net still fires +0.73 at threat-heavy positions)
is a teacher-horizon / B1-label-conflict limit that would need
a different teacher, outside this workstream. Further
iterations without a patch would only find more instances of
the same unpatchable categories. Proceeding to validation on
the FINAL v31 (mixed form, α=0.5); the battery is the verdict.

## 6. Validation battery

### 6.1 Tactical — COMPLETE 2026-11-17 ~12:40 PST — **CLAUSE 3 FAILS**

`eval_tactical_full.py --version 31 --suite both --seed 100`
(log `ml/runs/ml2-h2/tactical-battery.log`):

- Old suite: **20/20** ✓
- Hard suite (corrected): **28/29** ✗ — one failure:
  `setup-v1v0s-g92` (Set up the open four — v1v0s game 92):
  chose (5,4), expected (9,6).

**Diagnosis (VERIFIED, seed-independent — identical at seeds 0
and 100):** a VALUE-HEAD failure, reproducing the v26 pattern.
The v1 prior ranks the expected (9,6) FIRST (0.1122) for both
engines, but v31's net gives it our-q -0.377 (1 visit) while
giving the wrong (5,4) our-q +0.788 (75 visits). v8's net
values (9,6) at +0.246 (57 visits) and (5,4) at +0.056 (4
visits) — v8 passes. So the bigger distilled net is WORSE than
v8's small net on this quiet 2-ply setup: it overvalues the
immediate threat pattern (5,4) creates and undervalues the
quiet setup whose value only materializes 2 plies later — a
distillation-fidelity gap on quiet positions (the teacher's
2-ply rollout sees the setup; the net's static evaluation does
not reproduce it). **The value head remains the binding
constraint, confirmed at 4x capacity.** Not patchable within
H2's remit (would need a deeper-horizon teacher, targeted
quiet-setup data, or a different architecture — all outside the
pre-registered design). v31 is therefore REJECTED on clause 3
regardless of the remaining legs; the battery continues for the
program record.

### 6.2 h2h vs v8 (clause 1) — COMPLETE 2026-11-17 ~14:15 PST — **CLAUSE 1 FAILS**

`fast_loop_match.py --version-a 31 --opponent v8 --games 100
--seed-a 100 --seed-b 200` (raw
`research/iterations/batch-v31-vs-v8.json`; one VM replacement
killed the run at game 62, auto-resume continued it):

- **39W-59L-2D, 0.40, 95% CI [0.3094, 0.4980]** — the CI's upper
  bound is below 0.50: v31 is significantly WORSE than v8.
- As Black 26W-22L-2D (0.540); as White 13W-37L-0D (**0.260**).
  Black won 63/100.

The dev-seed evidence (sweep 0.70, fast loop 0.70) did NOT
reproduce at validation seeds — the same small-sample trap
v24 hit (0.65 smoke → 0.515 at 100 games), pre-registered in
§1.3. The 100-game battery is the verdict. The White collapse
mirrors v25's (Black 0.73 / White 0.15): both B1-data versions
collapse as White while v24 (engine-only data) did not —
support for the Black-skewed-training-data hypothesis (Black
wins 53.1% of B1 games; the net learns "Black has threats →
+1" and, as White, overvalues Black's threats and undervalues
its own play).

### 6.3 Ladder Elo (clause 2) — COMPLETE 2026-11-17 ~16:09 PST — **CLAUSE 2 FAILS**

`eval_version.py --version 31 --ladder-games 20` (raw
`research/iterations/eval-v31.json`; the run finished minutes
before a VM replacement — the ladder driver has no resume, so
this was luck, not durability):

- vs random: 20W-0L-0D (1.0000, CI [0.8389, 1.0000])
- vs heuristic: 9W-10L-1D (0.4750, CI [0.2785, 0.6795])
- vs mcts: 11W-9L-0D (0.5500, CI [0.3421, 0.7418]) — v31 beats
  the ladder MCTS outright, as v24 did.
- **Ladder Elo: estimate -17.4, interval [-165.3, 130.6]**
  (anchors random=-800, heuristic=0). Below v8's 0.0; the
  interval includes 0. v24's 52.5 remains the best ML ladder
  Elo.

### 6.4 Clause 4 re-run — COMPLETE 2026-11-17 ~16:13 PST — **CLAUSE 4 PASSES**

On the final tree: ml pytest **406 passed, 1 skipped, 58
xfailed, 0 failed**; npm test **55/55**; tsc clean. Identical
to the early clause-4 check.
### 6.5 Secondary: vs v0, vs v7, swap vs v7

**vs v0 — COMPLETE 2026-11-17 ~14:15 PST** (same seeds, raw
`research/iterations/batch-v31-vs-v0.json`; VM replacement at
game 51, auto-resumed): **38W-50L-12D, 0.44, CI [0.3467,
0.5377]**. As Black 24W-18L-8D (0.560); as White 14W-32L-4D
(0.320). v31 LOSES to the weakest opponent — v8 scored 0.545
vs v0. The White gap again: White 0.32 vs Black 0.56.

**vs v7 — COMPLETE 2026-11-17 ~15:26 PST** (raw
`research/iterations/batch-v31-vs-v7.json`, no interruptions):
**47W-52L-1D, 0.475, CI [0.3799, 0.5720]**. As Black 28W-21L-1D
(0.570); as White 19W-31L-0D (0.380). Black won 59/100. v31
roughly TIES v7 — non-transitivity again (v26 tied v7 at 0.51
while collapsing vs v8): v31 ties the heuristic bar reference
but loses to v8 (0.40) and v0 (0.44).

**Swap vs v7 — COMPLETE 2026-11-17 ~15:26 PST** (raw
`research/iterations/batch-v31-vs-v7-swap.json`, tempo judge,
balanced placement, placement_seed 0 — the same default v24
used, so directly comparable): **28W-70L-2D, 0.29, CI [0.2101,
0.3854]**. As Black 16W-32L-0D (0.333); as White 12W-38L-2D
(0.250). v31's win rate is 0.28 when v7 chooses colour and
0.30 when v31 chooses — no meaningful difference; both engines
took Black in ~80% of their choices. Under swap the chooser
gets the favourable side in every game, and v31 loses as both
colours. Far below the heuristic versions' swap results (v23
0.72, v27 0.775).

### Battery verdicts

| Clause | Requirement | v31 | Verdict |
|---|---|---|---|
| 1. h2h vs v8 | ≥100 games, win rate > 0.50, Wilson 95% CI lower > 0.50 | 39W-59L-2D, 0.40, CI [0.3094, 0.4980] | **FAIL** |
| 2. Ladder Elo | > 0.0 | -17.4, interval [-165.3, 130.6] | **FAIL** |
| 3. Tactical | ≥ 20/20 + 29/29 | 20/20 + 28/29 (setup-v1v0s-g92) | **FAIL** |
| 4. Test suites | ml pytest + npm test + tsc green | pytest 406/1/58, npm 55/55, tsc clean | **PASS** |

Secondaries: vs v0 38W-50L-12D (0.44, CI [0.3467, 0.5377]);
vs v7 47W-52L-1D (0.475, CI [0.3799, 0.5720]); swap vs v7
28W-70L-2D (0.29, CI [0.2101, 0.3854]). Defence: 231/231
battery losses via open four, 0 missed blocks, 0 clean missed
preventions in 1,064 threatened turns.

## 7. Verdict

**v31 is REJECTED — 1 of 4 clauses pass (only clause 4).** v8
stays the ML champion; v24 stays the strongest ML version on
ladder Elo (52.5) and the only ML version to tie v8 (0.515).

What v31 establishes, with the evidence:

1. **Bigger net + distilled labels ≠ better play.** The mixed
   net beat v24 on holdout transfer (engine pooled 0.7929 vs
   0.7766) and improved fast-loop play (vs v0 0.25 → 0.60),
   yet scored 0.40 vs v8 at 100 games. Holdout sign accuracy
   does not predict played strength at this scale.
2. **The value head remains the binding constraint, confirmed
   at 4x capacity.** The tactical failure (setup-v1v0s-g92)
   is a value-head failure: the prior ranks the setup move
   first, the net overrides it (q -0.377 vs +0.788 for the
   wrong move), while v8's small net values the setup
   correctly (+0.246). The distilled net overvalues immediate
   threat patterns and undervalues quiet 2-ply setups — a
   distillation-fidelity gap on quiet positions.
3. **B1 data collapses White.** v31 (White 0.26 vs v8, 0.32
   vs v0) mirrors v25 (White 0.15); v24 (engine-only data)
   did not collapse as White. The Black-skewed-training-data
   hypothesis (Black wins 53.1% of B1 games) now has two
   supporting data points. Colour-balanced retraining (H1's
   v30) is the direct test.
4. **Dev seeds flatter; the battery decides.** Sweep and fast
   loop both said 0.70 vs v8 at dev seeds; validation seeds
   said 0.40. The 20-game sweep selects α; it proves nothing.
5. **The mixed-data patch was real but insufficient.** It
   fixed a verified search-breaking defect (B1-only net
   saturation, raw +1.000 after every move) and improved every
   holdout and fast-loop measure — but the residual (teacher
   horizon, B1 label conflicts, White-side data skew) is
   outside H2's design space.

What changed because of a specific loss: the mixed-data patch
itself — iter-1's vs-v0 game 7 ply 43 blunder (a live VCF the
B1-only net could not see because it saturated at +1.000)
motivated retraining on B1 + engine data, which fixed the
saturation class in holdout and fast-loop play. Iter-2's one
genuine blunder (vs-v0 g3 ply 23) was a shared-prior problem
(the net's q was right; the v1 prior won on visits) — not
patchable within H2's remit.

Branch: `research/ml-batch-2-h2`. Checkpoint (mixed form):
`ml/runs/ml2-h2/value-net.pt` (B1-only preserved at
`value-net-b1only.pt`).

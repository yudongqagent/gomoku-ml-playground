# Iteration 42 — v42: history augmentation / history-dropout on v24's recipe (ML batch 4, H1)

**Status: IN PROGRESS — pre-registered; no eval has run.**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H1 workstream of ML batch 4 (H2 is v43,
history dropout on the v37 net; H3 is v44, 3-ply labels); work
happens on branch `research/ml-batch-4-h1` in the `gomoku-ml4-h1`
worktree. v42 differs from v8 ONLY in the value net's weights
(and the swept α) — same PUCT search, same v1 priors, same sims
(100), same c_puct (1.5), same threat check, same terminal cases
— so the battery is a clean A/B of the history-robust net
against v8's weak-self-play net. Against v24 the comparison is
even cleaner: v42 differs from v24 in ONE thing — the training
augmentation (history-plane dropout, §1.1). Same positions, same
1-ply rollout-leaf labels, same holdout split, same architecture,
same training recipe — so the v42-vs-v24 comparison isolates the
augmentation change.

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v8. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v42 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Ports (before pre-registration)

**Ports (ML batches 1–3 were never merged to master, so their
code was ported into this worktree — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: ML batch 1 H2's memmap version
  (pre-allocated `load_shards` + memmap bundle support + lazy-index
  `train()`), copied VERBATIM from `gomoku-ml1-h2` (sha256
  `7e8cc28d…` matches the source and iter-31's/iter-37's record)
  — then the `history_dropout` parameter added as a documented
  edit (§1.1; the diff from the verbatim port is the parameter,
  one call site, and the docstring).
- `ml/gomoku_ml/mcts_value.py`: copied VERBATIM from
  `gomoku-ml3-h2` — the per-version `value_weight` port ONLY
  (ML batch 1 H1's 3-hunk shape: docstring, `value_weight: float
  | None = None` parameter on `ValueMCTS.__init__`,
  `self.value_weight` assignment, leaf shrink via
  `self.value_weight`); default None → v8's 0.25, so v8 is
  byte-identical in behaviour. This does NOT include v36's
  threshold tie-break (a CLOSED lever — not ported).
  **NOTE for the coordinator: this DOES touch
  `ValueMCTS.__init__`'s signature** (it must — the α sweep
  needs a per-instance weight); the hunk is identical to
  v24's/v31's/v37's, so a merge against a sibling applying the
  same hunk is a clean same-change merge — but CHECK for a
  duplicate parameter after merging (the silent auto-merge
  breakage has recurred twice on this exact signature).
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS` (v24's exact hunk), plus the v42 registry entry
  (`value_checkpoint: ml/runs/ml4-h1/value-net.pt`,
  `value_weight: 1.0` placeholder until the §1.5 sweep sets it).
- `ml/scripts/generate_distill_data.py`: copied VERBATIM from
  `gomoku-ml3` (supports `--label-mode rollout` and `rollout2`)
  — used ONLY for the §1.2 verification re-run (value labels
  are reused from v24's shards, not re-generated).
- `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/scripts/sweep_alpha.py`,
  `ml/scripts/find_losing_moves.py`: copied VERBATIM from
  `gomoku-ml3-h2`.
- `ml/tests/test_load_shards.py`: copied VERBATIM from
  `gomoku-ml3-h2` (tests the memmap loader).
- `ml/gomoku_ml/augment.py`: NEW `drop_history_planes_batch`
  (this workstream's mechanism, §1.1).
- `ml/gomoku_ml/config.py`: `history_dropout` added to
  `DEFAULTS["train"]` (default 0.0) + range validation [0, 1].
- `ml/scripts/train.py`: passes `history_dropout` from the
  config through to `train()`.
- `ml/gomoku_ml/baseline.py` is FROZEN and untouched.
- `ml/scripts/eval_value_holdout.py` is already on master — no
  port needed. It defines the no-history view (planes 2, 4–7
  zeroed) that the §1.4 gate measures.

**Port smoke checks (2026-11-19):** v8 loads and plays in this
worktree (first move 112, centre; `value_weight` 0.25 default
preserved — the port does not change v8's behaviour); v42 is
registered (registry entry + `test_v42_registered_and_dispatches_to_
value_mcts`); `ValueMCTS.__init__` has exactly ONE `value_weight`
parameter (checked by introspection — no silent auto-merge
duplication). Targeted tests green: augment (11, incl. 5 new
history-dropout tests), config (13, incl. 2 new), mcts_value
(17, incl. 2 new), load_shards — **46 passed** (see §6.4 for
the full-suite run).

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, tactical 20/20 + 29/29 on the CORRECTED suite; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (**40 games/opponent now** — batch 11's re-basing; same
anchors, same formula, so point estimates remain comparable to
historical versions), clause 3 is tactical ≥ v8's (old suite
20/20, hard suite 29/29, run at `--seed 100` on the corrected
suite), clause 4 is all test suites green (ml pytest, npm test,
tsc). **Secondary head-to-heads, also pre-registered:** v42 vs v7
(100 games), v42 vs v0 (100 games), a 100-game swap match vs v7
under the tempo judge (the harness default), and a 100-game
match vs **v37** (the strongest ML version by head-to-head,
0.75 vs v8 — the progress measure for the ML track; v37 is
registered in this worktree as a harness-only entry, checkpoint
copied from `gomoku-ml3-h2`, α=0.25 as swept). Non-transitivity
is recorded prominently.

**The hypothesis (plan H1):** history-dependence is the top
ML-track risk (ML batch 3). Both new nets collapse on engine
positions without history planes — v37: v7 holdout 0.7395 →
0.5163 (MSE 0.078 → 0.436, mean|v| 0.090 → 0.436); v38:
no-history sign 0.5035 on v7 — while v24's net is
history-robust on the same positions (v7 0.7713 both views;
v18 0.8384 → 0.8344). At inference the leaves get SEARCH-PATH
history — neither the real-game history of the full-planes view
nor the zeroed history of the no-history view — so a net that
routes its value judgment through history features plays
between its two holdout views, both worse than v24's on engine
positions. v38's battery (loses to v0 0.415, tactical 27/29)
is what that pathology costs in games. The fix: train v24's
recipe with history-plane dropout, so the trunk cannot route
through history features and the net predicts the same label
from the board alone. If the dropout net is history-robust
(the §1.4 gate) AND keeps v24's full-planes fit, it should
play at least at v24's level with the history risk removed —
and the battery measures whether removing the risk moves
played strength. If the gate fails, the version does not go
to battery (the plan's hard gate).

**The improvement: v42 = v8's search + a value net trained on
v24's positions and labels with history-plane dropout during
training.** Everything else is v8 unchanged (see the header).
The net is consulted ONLY in quiet positions (terminal cases
stay exact, as in v8), so training data is quiet positions
only.

### 1.1 The history-dropout scheme (EXACT — pre-registered)

- **History planes** (the same set `eval_value_holdout.py`
  zeroes in its no-history view — see `gomoku_ml/encoding.py`):
  plane 2 (last-move one-hot), planes 4–5 (own/opponent stones
  2 plies ago), planes 6–7 (own/opponent stones 4 plies ago).
  **Plane 3 (side-to-move) is NOT history and is NEVER
  dropped**; planes 0–1 (current stones) are never dropped.
- **Scheme: per-sample whole-history dropout.** For each
  training sample in each batch, independently with probability
  **p = 0.5**, zero ALL five history planes of that sample;
  otherwise leave the sample intact. The two views (full and
  no-history) of the same position carry the SAME label — the
  net is trained to predict the rollout-leaf value from the
  board alone half the time, so the loss cannot be reduced by
  routing through history features that are absent half the
  time. Over 15 epochs each position is seen ~15 times with
  independent draws, so it is trained in both views many times.
- **p = 0.5 is FIXED** (no sweep — the gate measures the
  outcome; sweeping p against the gate would be post-hoc
  tuning on the gate). 0.5 is the symmetric choice: half the
  gradient signal comes from no-history views (forcing
  board-only features) and half from full views (preserving
  the full-planes fit).
- **Applied when:** in `train()`, after the dihedral
  augmentation, on TRAIN batches only. Never on the internal
  val split (val loss stays the full-planes measure, same as
  v24's, so checkpoint selection is comparable) and never on
  holdout eval (the gate measures the trained net on both
  views separately).
- **Implementation:** `augment.drop_history_planes_batch(planes,
  rng, p)` — a pure function, unit-tested (drop rate ≈ p over
  many samples; dropped samples have planes 2/4–7 exactly zero
  and planes 0/1/3 untouched; p=0.0 is the identity; p=1.0
  drops everything) — called from `train()` when
  `history_dropout > 0`. Config key `train.history_dropout`
  (default 0.0 — v24's recipe is exactly the p=0.0 case, so
  the augmentation is the ONLY change from v24).
- **NOT used:** per-plane independent dropout (it would create
  mixed views — e.g. last-move present but 2-ply history
  absent — that never occur at inference); duplicating each
  sample in both views (equivalent in expectation to p=0.5
  dropout over many epochs, at twice the dataset size); a
  history-free architecture (the fallback if dropout costs
  full-planes sign — see §1.6).

### 1.2 Dataset (pre-registered)

- **Source: v24's shards, REUSED VERBATIM** (v38's precedent —
  NOT re-generated, so there is zero label drift vs v24):
  `gomoku-ml1-h1/ml/runs/ml1-h1/data/` copied to
  `ml/runs/ml4-h1/data/` — 33,849 train / 3,378 holdout quiet
  positions from the SAME 817 games (488 v7 self-play at 50
  sims, 229 v18 self-play at 50 sims, 100 random playouts),
  with v24's 1-ply rollout-leaf labels (`_rollout_leaf_value`,
  blend=False). The ONLY change from v24 is the §1.1
  augmentation, so the v42-vs-v24 comparison isolates it.
- **Holdout (game-level, v24's protocol, IDENTICAL split):**
  every 10th game of each source, routed to `*-holdout` dirs
  and NEVER trained on — the same 80 games / 3,378 positions
  v24 held out, so v42's holdout numbers are comparable to
  v24's BY POSITION (only the training differs).
- **NOT used for training:** the B1 10k games (B1 data collapses
  White — v25, v31 — and v24's engine-only data did not; this
  workstream does not re-open that lever).
- **Verification (before training):**
  (a) shard counts match iter-24 §2 exactly (train 440 v7 /
  207 v18 / 90 random games; holdout 48 / 22 / 10; 33,849 /
  3,378 positions);
  (b) spot re-labelling — re-run the ported label phase on 20
  games (all three sources) and confirm the regenerated value
  labels AND planes match v24's shards exactly (validates that
  master's rollout-leaf teacher is the same function v24
  distilled; v38's verification (b) passed on the same check).

### 1.3 Architecture and training (pre-registered)

- **Architecture:** v24's ResNet **16ch/2 blocks** (126,586
  params), trained from scratch (random init — NOT from v8's or
  v24's checkpoint; the augmentation changes the training
  distribution, so a fresh run is the clean comparison). Do NOT
  go larger: v31 proved 4x capacity plays worse on distilled
  labels (the scale lever is CLOSED).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes — v24's recipe exactly), augmentation on
  (dihedral), val_fraction 0.05 (random split within the train
  set, for best-checkpoint selection only), seed 0.
- **Plus:** `history_dropout: 0.5` (§1.1) — the ONLY recipe
  change from v24.
- **Config:** `ml/configs/ml4-h1.yaml` (v24's `ml1-h1.yaml`
  with run name/dir changed + `history_dropout: 0.5`); run dir
  `ml/runs/ml4-h1-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, B1's finding).
  `checkpoint_every_epochs: 1` from the start (v24's resume
  lesson).
- **Memmap path (mandatory):** the train shards are bundled with
  `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml4-h1/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as v24):**
  best-by-internal-val-loss (`best.pt`), copied to
  `ml/runs/ml4-h1/value-net.pt` (gitignored data, the path v42's
  registry entry points at).
- **Resume rule (pre-registered, v24's lesson):** if training is
  killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans the
  full 15 via `schedule_epochs`); back up the pre-kill `best.pt`
  first, because the resumed run resets `best_val_loss` and
  would overwrite it at the next epoch regardless — the final
  best is chosen by comparing the resumed run's best val loss
  against the pre-kill best.

### 1.4 THE GATE (hard, pre-registered — from the plan)

**The no-history holdout must match full-planes BEFORE any games
are played.** Measured with `ml/scripts/eval_value_holdout.py`
on v24's game-level holdout (3,378 positions, 1-ply labels),
v42 net. ALL three clauses must hold:

1. **Engine pooled (v7+v18 holdouts, 2,840 positions):
   |no-history sign − full-planes sign| ≤ 0.02.** The match
   clause. v24's engine gap is ≈0.001 (v7 0.7713 both views;
   v18 0.8384 → 0.8344); v37's is 0.223 on v7 alone.
2. **Engine pooled full-planes sign ≥ 0.74.** The
   no-degenerate-match clause: the gate must not be passable
   by collapsing both views to chance (v24's engine full is
   0.790 — this allows ~0.05 regression for the robustness
   gain, no more).
3. **All-source pooled no-history sign ≥ 0.70** (v24's is
   0.7491). A floor on the no-history view including the
   random source, so the net is genuinely usable without
   history anywhere.

**Why engine-pooled for the match clause:** v24 itself has a
pooled (all-source) gap of 0.0275, driven by random positions
(0.7055 → 0.5468) where history genuinely carries signal
(random games are tactically wild; the last move matters).
At inference the leaves look like engine positions, not
random playouts — the engine sources are the operating
distribution, and the random source is still reported and
bounded by clause 3.

**If ANY gate clause fails: STOP.** Report the holdout numbers
and stop; do not burn battery time on a net that fails its own
gate. The version is recorded as gate-failed in VERSIONS.md
and no games are played. (The §1.6 stop rule — chance-level
own holdout — is subsumed by gate clause 2.)

### 1.5 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.5, 1.0, 2.0}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 42`. The grid brackets
v24's winner (1.0) on both sides: the labels are identical to
v24's, so 1.0 is the prior favourite — but dropout training may
change the net's value magnitudes (hedging toward the mean
without history would shrink them; sharper board-only features
could grow them), and 0.5/2.0 cover both directions. Pick the
best win rate; tie-break: the smaller α (conservative — less
optimizer's-curse exposure). **No-signal rule:** if all three
arms are within one win of each other, pick α=1.0 (v24's
winner) and say so — the sweep found no signal. The sweep
SELECTS α; the 100-game battery is the verdict (standing
lesson: small-sample smokes mislead). The registry entry's
`value_weight` is updated to the winner after the sweep.

### 1.6 What would change the design

- If the gate (§1.4) fails, the version stops — no battery,
  no retrain within this version. A different p or scheme is
  a NEW version (v43's remit covers history dropout on the
  v37 net; a v42-variant would need the coordinator's
  re-assignment).
- If the dropout net's full-planes sign collapses (gate
  clause 2 fails), the history planes carry real signal the
  board alone cannot replace — the fallback is a history-free
  architecture (train on planes 0/1/3 only), NOT more dropout.
- If the α sweep's best is within noise of the worst (all three
  within one win of each other), pick α=1.0 per the §1.5
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain re-runs the α
  sweep and the battery from scratch on the new form.

### 1.7 Holdout diagnostics (pre-registered)

All via `ml/scripts/eval_value_holdout.py` (sign accuracy +
MSE, full-planes and no-history views) on 1-ply labels:
- (a) **v42 net on its own holdout** (80 games, 3,378
  positions) — in-distribution; the §1.4 gate applies here.
  ALSO report accuracy BY SIDE TO MOVE (Black/White) — the
  v30 method: the B1-White-collapse lesson says the split is
  always measured.
- (b) **v24's net** (`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`,
  by absolute path) **on the same holdout** — reproduces
  v24's published numbers (sanity check on the eval setup)
  and gives the exact per-source gaps the dropout must match
  or beat.
- (c) **v8's net on the same holdout** — baseline (expect
  ~0.50, as on v24's labels).
- (d) **v37's net** (`gomoku-ml3-h2/ml/runs/ml3-h2/value-net.pt`)
  **on the same 1-ply holdout** — diagnostic only: does the
  2-ply net's history collapse show on 1-ply labels too?
No decision hinges on (b)–(d); they diagnose whether the
dropout worked and how v42's fit compares to v24's and v37's.

### 1.8 Eval seeds (pre-registered)

h2h vs v8: v42=100 / v8=200, WITH `--openings
research/openings-ml4.json` (the pre-registered 10-opening set
— ML batch 4's battery change; fixes the mirror-match collapse:
v36-vs-v8 was 21 unique in 100); h2h vs v0: v42=100 / v0=200
(no openings — different engines, no mirror collapse); h2h vs
v7: v42=100 / v7=200; h2h vs v37: v42=100 / v37=200, WITH
`--openings` (same mirror-match reason as vs v8); ladder seed
100, **40 games/opponent**, mcts_sims=100; swap 100 games vs
v7, balanced placement, **placement_seed 11** (the
pre-registered seed — v31/v32 used 0 by mistake, do not repeat
that), tempo judge; tactical `--seed 100`.

### 1.9 Eval order (pre-registered)

(1) Data copy + verification (§2). (2) Memmap bundle +
training (§3) + holdout diagnostics + **THE GATE** (§4) — if
the gate fails, STOP here. (3) α sweep (§5) — registry updated
to the winner. (4) FAST LOOP (§5): up to 4 iterations of 10
games vs v8 AND 10 games vs v0. (5) FULL VALIDATION BATTERY
on the final v42 (§6): tactical re-run, h2h vs v8 100 games
(with openings), 100 games vs v0, 100 games vs v7, 100-game
swap match vs v7, 100 games vs v37, ladder 40 games/opponent,
clause-4 suite re-run. Any battery numbers measured before the
fast loop completes are INTERMEDIATE, labelled as such; only
the final battery counts for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games, pre-registered openings
selfplay_batch.py --engine-a v42 --engine-b v8 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v42-vs-v8.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 42 --ladder-games 40 --out research/iterations/ladder-v42.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 42 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, swap vs v7 (balanced/tempo, placement_seed 11, seeds 100/200), vs v37 (openings)
fast_loop_match.py --version-a 42 --opponent v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v42-vs-v7.json
fast_loop_match.py --version-a 42 --opponent v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v42-vs-v0.json
selfplay_batch.py --engine-a v42 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v42-vs-v7-swap.json
selfplay_batch.py --engine-a v42 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v42-vs-v37.json
```

## 2. Data (actuals)

**Data copy COMPLETE 2026-11-19** — v24's shards copied VERBATIM
from `gomoku-ml1-h1/ml/runs/ml1-h1/data/` into
`ml/runs/ml4-h1/data/` (gitignored data), plus the games files
`games-{v7,v18,random}.json` (for the §1.2 verification re-run).
Checkpoints copied: v8's (`ml/runs/b3-h2/value-net.pt`, md5
`5deb26f3df67883f24f04800d5356da6` — matches iter-37's record)
and v24's (`ml/runs/ml1-h1/value-net.pt`, md5
`08c8ec91510a58841333887d51f6c7bf`).

**Verification (a) — shard counts: PASS.** Counts match iter-24
§2 in every cell:

| Source | Games | Quiet positions | ±1 labels |
|---|---|---|---|
| v7 self-play (train) | 440 | 19,545 | 746 (3.8%) |
| v18 self-play (train) | 207 | 9,142 | 453 (5.0%) |
| Random playouts (train) | 90 | 5,162 | 1,470 (28.5%) |
| **Train total** | **737** | **33,849** | **2,669** |
| v7-holdout | 48 | 2,050 | 78 |
| v18-holdout | 22 | 790 | 51 |
| random-holdout | 10 | 538 | 156 |
| **Holdout total** | **80** | **3,378** | **285** |

**Verification (b) — spot re-labelling: PASS.** The ported label
phase re-ran on 20 games (7 v7 + 7 v18 + 6 random, 888 quiet
positions, `--label-mode rollout`); the regenerated planes,
policy, AND value labels match v24's shards EXACTLY (**0
mismatches** in 20 shards). Master's rollout-leaf teacher is the
same function v24 distilled — the reused shards carry zero label
drift, and the v42-vs-v24 comparison isolates the augmentation.

**Memmap bundle:** `ml/runs/ml4-h1/data-mmap/` built from
`ml/runs/ml4-h1/data-train/` (symlinks to the three train dirs
only — holdouts excluded): 737 shards, 33,849 samples, planes
(33849, 8, 15, 15), mean|v| — see §3.

## 3. Training (actuals)

**Interrupt and recovery (2026-11-19):** the original training run
(started 19:38 UTC, config `ml/configs/ml4-h1.yaml`, 15 epochs,
33,849 samples) was killed by a runtime restart drain at **epoch
10, step 4590 (19:46 UTC)** — no `interrupted.pt` was written
(the SIGTERM handler never ran). Epoch val losses before death:
e1 0.0877, e2 0.0875, e3 0.0870, e4 0.0844, e5 0.0923, e6 0.0843,
e7 0.0834, e8 0.0814, e9 **0.0797** (best); train loss fell
0.0799 → 0.0640 over epochs 1–9.

**Resume decision: RESUME from `checkpoint-4527.pt`, do not
restart** (the pre-registered §1.3 rule). Verified before
resuming: the checkpoint loads cleanly (step 4527, epoch 9,
ResNet 16ch/2 blocks, 126,586 params, optimiser state present,
all tensors finite) and `best.pt` was backed up to
`checkpoints/best-pre-kill.pt` first (the resumed run resets
`best_val_loss` and would overwrite it at the next epoch
regardless). Resumed with `--epochs 6` (15 − 9 completed); the
cosine schedule spans the full 15 via `schedule_epochs`. The
resumed run's best (epoch 15, val 0.0761) beats the pre-kill
best (epoch 9, 0.0797) outright, so the resume best-reset flaw
never bit and no restore from `best-pre-kill.pt` was needed.
Original run log backed up at `ml/runs/ml4-h1/train-pre-kill.log`.

**Final (completed 2026-11-19 19:54 UTC):** 15 epochs total
(9 pre-interrupt + 6 resumed), 7,545 steps, 33,849 samples
(32,157 train / 1,692 val, same seed-0 split both runs). Train
loss 0.0799 → 0.0575; val loss 0.0877 → **0.0761**, decreasing
monotonically from epoch 6 on (e6 0.0843, e8 0.0814, e10 0.0794,
e12 0.0768, e14 0.0762, e15 0.0761 — one bump at epoch 5,
0.0923) — no overfitting by epoch 15, but the pre-registered
plan is 15 epochs and the battery is the verdict, so no
extension. **Best checkpoint: epoch 15 (`best.pt`, step 7545,
val 0.0761) = `final.pt`** (same step/epoch). Val loss is
higher than v24's 0.0617 — expected: dropout training is a
harder task (half the samples lack history), so the full-planes
val MSE is worse in absolute terms; the gate measures sign
accuracy, not val loss, and §4 is the verdict on the fit.
Per-epoch checkpoints `checkpoint-<step>.pt` (steps 503–7545)
also saved. Final checkpoint copied to the registry path:
**`ml/runs/ml4-h1/value-net.pt`** (md5
`88a69de35216b7965682b8d77418df31`; verified bit-identical in
weights to `best.pt`, max |Δ| 0.0; ResNet 16ch/2 blocks,
126,586 params, 48 tensors, all finite).

## 4. Holdout diagnostics + THE GATE (actuals)

Game-level holdout (every 10th game, never trained on — the SAME
80 games / 3,378 positions v24 held out), targets = 1-ply
rollout-leaf labels. v42 net = `ml/runs/ml4-h1/value-net.pt`;
v24 net = `ml/runs/ml1-h1/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`; v37 net =
`ml/runs/ml3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml4-h1/holdout-{v7,v18,random}.txt`. Sanity check: this
run reproduces v24's published 1-ply numbers EXACTLY on its own
holdout (v7 0.7713, v18 0.8384, random 0.7055, pooled 0.7766),
v37's §4(c) numbers exactly (v7 0.7550 full / 0.5409 no-history),
and v8's (0.523 / 0.501 / 0.463) — the eval setup is sound.

### (a) v42 net on its own holdout (in-distribution) — THE GATE

| Holdout | n pos (non-draw) | v42 sign (full) | v42 sign (no-hist) | v42 MSE full / no-hist | v42 mean\|v\| full / no-hist | v24 sign (full) | v24 sign (no-hist) |
|---|---|---|---|---|---|---|---|
| v7 | 2,050 (1,845) | 0.7550 | **0.7604** | 0.0441 / 0.0441 | 0.0633 / 0.0649 | 0.7713 | 0.7713 |
| v18 | 790 (761) | 0.8502 | 0.8449 | 0.0679 / 0.0672 | 0.0720 / 0.0717 | 0.8384 | 0.8344 |
| random | 538 (523) | 0.6979 | **0.7094** | 0.2558 / 0.2651 | 0.2567 / 0.2473 | 0.7055 | 0.5468 |
| **pooled** | **3,378 (3,129)** | **0.7686** | **0.7725** | — | — | **0.7766** | **0.7491** |
| **engine pooled (v7+v18)** | **2,840 (2,606)** | **0.7828** | **0.7851** | — | — | **0.7909** | **0.7897** |

**THE GATE PASSES — all three clauses:**

1. Engine pooled |no-history − full| = |0.7851 − 0.7828| =
   **0.0023** ≤ 0.02 ✓ (v24's gap: 0.0012; v37's on v7 alone:
   0.223).
2. Engine pooled full-planes sign = **0.7828** ≥ 0.74 ✓ (v24's:
   0.7909 — v42 is 0.008 below v24, well inside the allowance).
3. All-source pooled no-history sign = **0.7725** ≥ 0.70 ✓
   (v24's: 0.7491 — v42 is BETTER).

**The dropout worked — beyond matching, the no-history view is
slightly BETTER than the full-planes view** (pooled 0.7725 vs
0.7686; v7 0.7604 vs 0.7550; random 0.7094 vs 0.6979). The net
was trained with half its samples history-free, so the
board-only mapping is well-trained and the full-planes view's
history features are now slightly noisy rather than
informative. The fix lands exactly where v24 was weakest: on
random positions v42's no-history sign is **0.7094 vs v24's
0.5468 (+0.163)** — the collapse is gone. MSE and mean|v| are
stable across views too (v7 MSE 0.0441 both views, mean|v|
0.0633 → 0.0649 — contrast v37's MSE 0.078 → 0.436 and mean|v|
0.090 → 0.436 ballooning). The full-planes cost of the dropout
is small: pooled full 0.7686 vs v24's 0.7766 (−0.008), engine
full 0.7828 vs 0.7909 (−0.008) — the history planes carry
almost no signal the board alone cannot replace at this scale.
**v42 proceeds to battery.**

**By side to move** (v42 net, full planes; `/tmp/by_side_v42.py`):
v7 Black 0.7223 / White 0.7864; v18 Black 0.8952 / White 0.8072;
random Black 0.5781 / White 0.8127. **No White collapse** —
White is BETTER than Black on v7 and random. The weak cell is
random-Black (0.578, same as v38's finding — random positions
with Black to move are the hardest for both heads). No-history
≈ full in EVERY side cell (largest gap: v18 Black 0.8952 →
0.8844) — the dropout worked uniformly across sides.

### (d) v37's net on the 1-ply holdout (diagnostic)

| Holdout | v37 sign (full) | v37 sign (no-hist) | v37 MSE full / no-hist |
|---|---|---|---|
| v7 | 0.7550 | 0.5409 | 0.0447 / 0.4010 |
| v18 | 0.8449 | 0.5848 | 0.0624 / 0.2919 |
| random | 0.6482 | 0.6750 | 0.2071 / 0.2664 |
| **pooled** | **0.7590** | — | — |

Reproduces iter-37 §4(c) exactly. The history collapse is a
property of the v37 NET, not of the 2-ply labels: it collapses
on 1-ply labels too (0.5409 no-history on v7). v42's dropout
net beats v37's net on the no-history view by 0.22 on v7 while
matching it on the full-planes view (0.7550 vs 0.7550 on v7;
0.8502 vs 0.8449 on v18). Whether that holdout robustness
translates to played strength is the battery's question — v37
plays at 0.75 vs v8 DESPITE its collapse, so robustness is not
sufficient for strength; it is necessary for the operating
point to be predictable.

## 5. α sweep and fast loops

**α sweep — COMPLETE 2026-11-19 ~22:05 UTC** (20 games each vs
v8, seeds 1001/2001; per-alpha JSONs
`research/iterations/alpha-v42-<α>-vs-v8.json`). Interrupted
twice by runtime restart drains (α=0.5 at game 9, α=2.0 at game
11); both resumed cleanly via the sweep driver's auto-resume
(completed alphas skipped, partial alpha continued from its
saved game count with fresh players; RNG restarts recorded in
the JSONs). Three sibling sweeps (v43, v44) shared the VM for
much of the run, so games ran ~2–3x slower than solo (two
225-ply draws took 521s and 705s).

| α | W-L-D | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| **0.5** | **9-11-0** | **0.450** | **[0.2582, 0.6579]** | 7-3-0 | 2-8-0 |
| 1.0 | 7-11-2 | 0.400 | [0.2188, 0.6134] | 5-4-1 | 2-7-1 |
| 2.0 | 7-12-1 | 0.375 | [0.1998, 0.5905] | 5-4-1 | 2-8-0 |

**Winner: α = 0.5** (pre-registered rule: best win rate; the
no-signal rule does not apply — 9 vs 7 wins differ by 2).
Registry `value_weight` updated 1.0 → 0.5. The pattern is
monotone DECREASING in α — the more the dropout net's value is
trusted, the worse v42 plays, and all three arms are BELOW
0.50 on dev seeds (v24's winner was α=1.0 at 0.65; v38's was
α=0.5 at 0.60). The α effect is almost entirely a WHITE
effect in reverse of v37's: Black is 0.55–0.70 at every α
while White is 0.20–0.25 at every α — v42's White loses to v8
no matter the weight. The sweep selects α but proves nothing
(20 games, heavily overlapping CIs) — the 100-game battery is
the verdict. But the sweep's message is discouraging: the
history-robust net plays WORSE than v24's net on dev seeds
despite passing the gate — holdout robustness again fails to
predict played strength (v31/v38's lesson, repeated).

TODO — fast loops.

**Fast loop iteration 1** (10 games each, seeds 1001/2001, α=0.5):
- vs v8: **2W-7L-1D (0.25)**, CI [0.0809, 0.5578]. As Black 1-3-1
  (0.30), as White 1-4-0 (0.20). All 10 games unique.
- vs v0: **6W-4L-0D (0.60)**, CI [0.3127, 0.8318]. As Black 4-1-0
  (0.80), as White 2-3-0 (0.40). All 10 games unique.
- Loss classification (`classify_losses.py`, both matches):
  **11 losses, 11 via open four, 0 missed blocks, 0 clean missed
  preventions in 48 threatened turns.** Defence as measured is
  clean — the losses are positional drift into unstoppable open
  fours, not blundered blocks.
- Losing moves (`find_losing_moves.py`, VCF solver depth 4; raw
  `ml/runs/ml4-h1/losing-moves-{v8,v0}-k1.txt`): **8 of 11 losses
  were already lost** at the first VCF moment (no saving reply
  exists — positional decline, no single tactical blunder).
  **THREE genuine blunders: vs v8 game 4 (Black, 34 plies), ply
  24 — (8,9), with 3 saving moves ((8,4), (8,5), (12,5)); vs v0
  game 1 (White, 25 plies), ply 13 — (10,5), with 1 saving move
  ((6,9)); vs v0 game 9 (White, 21 plies), ply 13 — (9,10),
  with 1 saving move ((9,6)).**

**Workup verdict — vs-v8 game 4, ply 24 (VERIFIED):** `diag_root.py`
(raw `ml/runs/ml4-h1/diag-v8-g4-ply24.txt`) shows the search's q
is RIGHT and the prior is wrong: the save (8,5) has the BEST
our-q (+0.056) but only 15 visits vs the actual's 23 — its v1
prior (0.0462) is 3x the actual's (0.1345), and the most-visited
root choice follows the prior. The 1-ply exact leaves AGREE the
saves are better ((8,4)/(8,5) terminal +0.250, the must-block
value, vs the actual's 0.000). This is the SAME category as v37
iter-1's g2 blunder and v31 iter-2's: a PRIOR problem in shared
heuristic-track code (`_priors`), whose change would alter v8/v7
equally and invalidate the α sweep — NOT patchable within H1's
remit (v42 = v8's search + distilled net). The net is not the
problem here: its q already ranks the save first.

**Workup verdict — vs-v0 game 1, ply 13 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml4-h1/diag-v0-g1-ply13.txt`) shows
a NEAR-TIE: actual (10,5) 18 visits q −0.062 prior 0.1444 vs
save (6,9) 14 visits q −0.106 prior 0.1430 — the priors differ
by 0.0014 and the q gap is 0.044. The 1-ply exact leaves are
MIXED (v22/v14 prefer the save, v20 prefers the actual). The
VCF that decides it (depth-4, up to 8 plies) is beyond the
100-sim search's horizon AND beyond the 2-ply label's horizon —
the shared search-horizon limit (v24's d9 class), NOT a
value-head failure.

**Workup verdict — vs-v0 game 9, ply 13 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml4-h1/diag-v0-g9-ply13.txt`) shows
the net AGREES with the exact leaves: actual (9,10) 23 visits
q −0.018 prior 0.1299 vs save (9,6) 12 visits q −0.066 prior
0.1286 — and ALL THREE exact heuristic leaves (v22/v20/v14)
rank the loser first too (+0.168/+0.124/−0.070 vs
+0.131/+0.085/−0.129). Same shared-horizon class as game 1.

**Fast-loop STOP after iteration 1:** no patchable weakness class
for v42 within H1's design space. The one q-vs-prior blunder is
a shared-prior issue; the other two are shared-horizon issues;
the other 8 losses are positional with no single blunder. The §4
history-dependence is NOT implicated in any blunder (g4: the
net's q was already right; g1/g9: the net agrees with the exact
leaves) — so no loss-motivated retrain is authorised by §1.6,
and a history-augmentation retrain on holdout grounds alone
would repeat v31's error of chasing a diagnostic. The battery
is the verdict; it proceeds on the final v42 (α=0.5).

## 6. Validation battery

TODO.

## 7. Verdict

TODO.

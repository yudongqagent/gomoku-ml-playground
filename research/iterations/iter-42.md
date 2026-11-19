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

**Port smoke checks:** TODO — v8 loads and plays in this
worktree (checkpoint copied from the main repo, md5
`5deb26f3df67883f24f04800d5356da6`); v42 is registered;
targeted tests green (see §6.4).

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

TODO — fill in when the data copy + verification complete:
shard counts vs iter-24 §2, spot re-labelling verification.

## 3. Training (actuals)

TODO — fill in when training completes: final train/val loss,
epochs run, best checkpoint (step/epoch), checkpoint path, md5.

## 4. Holdout diagnostics + THE GATE (actuals)

TODO — fill in when measured: the gate numbers (per-source and
pooled, full vs no-history), the by-side split, the v24/v8/v37
comparisons, and the gate verdict.

## 5. α sweep and fast loops

TODO.

## 6. Validation battery

TODO.

## 7. Verdict

TODO.

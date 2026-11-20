# Iteration 43 — v43: history dropout on the v37 net (ML batch 4, H2)

**Status: COMPLETE 2026-11-20 — REJECTED (2/4 clauses pass).**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H2 workstream of ML batch 4 (H1 is v42,
history robustness on v24's recipe; H3 is v44, 3-ply labels);
work happens on branch `research/ml-batch-4-h2` in the
`gomoku-ml4-h2` worktree. v43 differs from v8 ONLY in the value
net's weights (and the swept α) — same PUCT search, same v1
priors, same sims (100), same c_puct (1.5), same threat check,
same terminal cases — so the battery is a clean A/B of the
history-dropout net against v8's weak-self-play net (and, less
cleanly, against v37's net: v43 changes ONLY the training
procedure — history dropout — on v37's identical positions and
labels — see §1.2 — so the v43-vs-v37 comparison isolates the
dropout change).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v8. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v43 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Ports and verification (before pre-registration)

**Ports (ML batches 1–3 were never merged to master, so their
code was ported into this worktree — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: ML batch 1 H2's memmap version
  (pre-allocated `load_shards` + memmap bundle support + lazy-index
  `train()`), copied VERBATIM from `gomoku-ml3-h2` (sha256
  `7e8cc28d…` matches iter-37's record). History dropout (§1.1)
  is added AFTER the verbatim copy as a separate, documented edit.
- `ml/scripts/generate_distill_data.py`: v24's games+label
  pipeline WITH the `--label-mode rollout2` extension, copied
  VERBATIM from `gomoku-ml3-h2` (sha256 `2057f0a5…`) — used for
  verification re-labelling (§2), not for new labels (v37's
  labelled shards are reused verbatim, §1.2).
- `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/tests/test_load_shards.py`: copied VERBATIM from
  `gomoku-ml3-h2` (sha256 `92664381…`, `2b7930bc…`,
  `1844f241…` all match iter-37's record).
- `ml/scripts/sweep_alpha.py`: v31's parameterised version as
  adapted for ML batch 3 (the `resolve_opponent` 3-value
  unpack for the per-game-seeding signature), copied VERBATIM
  from `gomoku-ml3-h2` (sha256 `9b90201e…`).
- `ml/scripts/find_losing_moves.py`: v31's parameterised version
  (takes match files as arguments), copied VERBATIM from
  `gomoku-ml3-h2` (sha256 `1f1ca6cc…` matches iter-37's record).
- `ml/gomoku_ml/mcts_heuristic.py`: `_rollout2_leaf_value`
  (v37's 2-ply label function — the ONLY diff from master in
  this file), ported VERBATIM from `gomoku-ml3-h2`. Found
  missing when the §2 spot re-labelling first ran
  (`AttributeError: ... has no attribute '_rollout2_leaf_value'`
  — the task's port list did not include it, because it lives
  in a master file rather than a script); this tree's file is
  now BYTE-IDENTICAL to `gomoku-ml3-h2`'s.
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port (ML batch 1 H1's change), applied by hand to this tree's
  file — the SAME 4-hunk shape as v37's (docstring,
  `value_weight: float | None = None` parameter on
  `ValueMCTS.__init__`, `self.value_weight` assignment, leaf
  shrink via `self.value_weight`); default None → v8's 0.25, so
  v8 is byte-identical in behaviour. Verified: this tree's file
  is now BYTE-IDENTICAL to `gomoku-ml3-h2`'s. **NOTE for the
  coordinator: this DOES touch `ValueMCTS.__init__`'s signature**
  (it must — the α sweep needs a per-instance weight); the hunk
  is identical to v24's/v31's/v37's, so a merge against a sibling
  applying the same hunk is a clean same-change merge — but
  CHECK for a duplicate parameter after merging (the silent
  auto-merge breakage has recurred twice on this exact
  signature).
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS` (v24's exact hunk), plus the v43 registry entry
  (`value_checkpoint: ml/runs/ml4-h2/value-net.pt`,
  `value_weight: 1.0` placeholder until the §1.5 sweep sets it)
  and a v37 registry entry (`value_checkpoint:
  ml/runs/ml3-h2/value-net.pt`, `value_weight: 0.25` — its
  swept α) SOLELY as the comparison engine for v43's
  secondary head-to-head (v37 was never merged to master;
  its checkpoint is copied into this worktree, md5
  `2fde492aae6d44cbe945e15bd5f1f959` — matches the task).
- `ml/gomoku_ml/config.py`: ONE new train key,
  `history_dropout` (default 0.0, validated to [0, 1]) —
  default 0.0 reproduces pre-dropout training exactly.
- `ml/scripts/train.py`: ONE new passthrough,
  `history_dropout=float(tr["history_dropout"])`.
- `ml/scripts/eval_value_holdout.py` is already on master
  (sha256 matches `gomoku-ml3-h2`'s) — no port needed. It
  defines the gate's no-history view (planes 2, 4-7 zeroed).
- The rollout-leaf teacher (v18 code) is on master since the
  batch-9 merge — no cherry-picking needed.
- `ml/baseline.py` is FROZEN and untouched.

**Data verification (before pre-registration):** v37's labelled
shards copied verbatim from `gomoku-ml3-h2/ml/runs/ml3-h2/data/`
into `ml/runs/ml4-h2/data/` (gitignored data). Position counts
and label stats match iter-37 §2 EXACTLY in every cell:
v7 train 19,545 (mean|v| 0.1570, ±1 6.2%), v18 train 9,142
(0.1775, 7.4%), random train 5,162 (0.4022, 32.6%); holdouts
2,050 / 790 / 538 = 3,378 positions; total 37,227. Shard counts
440 / 207 / 90 train + 48 / 22 / 10 holdout = 737 + 80. The
games files (`games-{v7,v18,random}.json`, 817 games) are copied
alongside for re-labelling verification (§2). v37's checkpoint
(md5 `2fde492aae6d44cbe945e15bd5f1f959`) is referenced by
absolute path for holdout comparison, not copied.

**Port smoke checks:** v8 loads and plays in this worktree
(checkpoint copied from the main repo, md5
`5deb26f3df67883f24f04800d5356da6` — matches iter-37's record;
first move 112, centre; `value_weight` 0.25 default preserved
— the port does not change v8's behaviour); v43 is registered;
config `ml/configs/ml4-h2.yaml` resolves with
`history_dropout: 0.5`; targeted tests green: history-dropout
(7), config (11), mcts_value (16), load_shards (8) — **42
passed**.

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, tactical 20/20 + 29/29 on the CORRECTED suite — re-verified
by the ML batch 2 coordinator at seed 100; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (identical opponents; legs are now **40 games/opponent**
per batch 11's re-basing — the point estimates remain comparable
to historical versions on the same scale), clause 3 is tactical
≥ v8's (old suite 20/20, hard suite 29/29, run at `--seed 100`
on the corrected suite), clause 4 is all test suites green (ml
pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v43 vs v7 (100 games), v43 vs v0 (100 games),
v43 vs v37 (100 games, with the openings set — the direct
dropout comparison), and a 100-game swap match vs v7 under the
tempo judge (the harness default). Non-transitivity is recorded
prominently.

**The hypothesis (plan H2):** v37 is the strongest ML version
ever by head-to-head (0.75 vs v8, CI clear of 0.50 — the first
ML version ever to beat v8), but its net is catastrophically
history-dependent on engine positions (v7 holdout 0.7395 →
0.5163 without history planes; v18 0.8436 → 0.5846), while
v24's net is history-robust (v7 0.7713 both views). The
mechanism (iter-37 §4, consistent with the numbers): the 2-ply
label is a harder function of the CURRENT board than the 1-ply
label, so with 33,849 samples the optimizer grabbed the history
planes as a shortcut (game phase / last-move character
correlates with the label) instead of learning the true
board→label mapping. At inference the leaves get SEARCH-PATH
history — neither the real-game history of the full-planes view
nor the zeroed history of the no-history view — so v37 plays
between its two holdout views, both worse than v24's on engine
positions. If randomly dropping the history planes during
training forces the trunk to read the value from the current
board alone, v43 keeps v37's played strength AND is robust to
missing history. If the dropout costs full-planes fit or played
strength, the history planes carry real signal the 2-ply labels
need — and that is itself a result (v38's §7 fallback: a
history-free architecture instead).

**The improvement: v43 = v37's recipe + history-plane dropout
during training.** Everything else is v8/v37 unchanged (see the
header). The net is consulted ONLY in quiet positions (terminal
cases stay exact, as in v8), so training data is quiet positions
only.

### 1.1 The dropout scheme (pre-registered, exact)

- **Rate: p = 0.5.** Per training sample, independently, with
  probability 0.5 the history planes are zeroed before the
  forward pass. Over 15 epochs each sample is seen ~7.5 times
  with history and ~7.5 times without — the net must learn the
  board→label mapping from the current board alone while still
  seeing real history half the time.
- **Which planes: (2, 4, 5, 6, 7)** — plane 2 (last-move
  one-hot), planes 4-5 (board 2 plies ago), planes 6-7 (board 4
  plies ago). EXACTLY the planes `eval_value_holdout.py`
  zeroes in its no-history view (enforced by test:
  `test_history_planes_match_holdout_view`); planes 0, 1
  (current stones) and 3 (side to move) are NEVER touched.
- **Joint, not per-plane:** when a sample is drawn, ALL five
  history planes are zeroed together — matching the gate's
  intervention (the gate zeroes all of them at once).
- **Fresh every epoch:** the draw uses the training loop's RNG
  at batch-build time, after augmentation (zeroing commutes
  with the dihedral transform), so each epoch sees a different
  random subset — true dropout augmentation, not a fixed
  split of the dataset.
- **Train only:** validation is always full-planes (checkpoint
  selection uses the standard view, unchanged from v37); the
  no-history gate is measured separately on the holdout (§1.4).
- **Why 0.5, not 1.0:** p=1.0 would make the gate trivial (the
  net never sees history, so it cannot depend on it) — but at
  inference the leaves DO carry search-path history, so a p=1.0
  net would see out-of-distribution inputs at every leaf. p=0.5
  keeps both views in-distribution: the net sees real history
  and zeroed history during training, and search-path history
  at inference lies between two views it has trained on.
- **Implementation:** `gomoku_ml.train.drop_history_planes`
  (new helper, unit-tested) + `history_dropout` parameter on
  `train()` + config key `train.history_dropout` (default 0.0
  = pre-dropout behaviour exactly). Config:
  `ml/configs/ml4-h2.yaml` (v37's `ml3-h2.yaml` with run
  name/dir changed and `history_dropout: 0.5` added).

### 1.2 Dataset (pre-registered)

- **Source: v37's labelled shards, REUSED VERBATIM** — no new
  game generation and NO re-labelling: the shards already exist
  (v37's §2) and re-labelling would only re-derive identical
  labels from the same teacher. The positions AND labels are
  EXACTLY v37's: the 817 games in
  `gomoku-ml1-h1/ml/runs/ml1-h1/games-{v7,v18,random}.json`
  (488 v7 self-play at 50 sims, 229 v18 self-play at 50 sims,
  100 random playouts), 37,227 quiet positions, 2-ply
  rollout-leaf labels. The ONLY change from v37 is the
  training procedure (dropout), so the v43-vs-v37 comparison
  isolates the dropout change.
- **Holdout (game-level, v24's protocol, IDENTICAL split):**
  every 10th game of each source, routed to `*-holdout` dirs
  and NEVER trained on — the same games v24/v37 held out (48
  v7 / 22 v18 / 10 random = 80 games, 3,378 positions), so
  v43's holdout numbers are comparable to v37's BY POSITION
  (only the training procedure differs). Expected train set:
  33,849 positions.
- **Verification (§2):** (a) shard counts + label stats match
  iter-37 §2 exactly (already done, §0); (b) spot re-labelling
  — re-run the ported label phase on 20 games (all three
  sources) and confirm the regenerated labels match v37's
  shards exactly (validates that master's rollout-leaf teacher
  is the same function v37 distilled; if it mismatches, the
  shards are still v37's labels and training proceeds — the
  mismatch would be reported as a teacher-drift finding, not
  a blocker).
- **NOT used for training:** the B1 10k games (B1 data collapses
  White — v25, v31 — and v24's engine-only data did not; this
  workstream does not re-open that lever).

### 1.3 Architecture and training (pre-registered — v37's exactly, plus dropout)

- **Architecture:** v24's/v37's ResNet **16ch/2 blocks**
  (126,759 params), trained from scratch (random init — NOT
  from v8's, v24's, or v37's checkpoint; the training
  procedure is the change, and starting from v37's weights
  would confound it). Do NOT go larger: v31 proved 4x capacity
  plays worse on distilled labels (the scale lever is CLOSED).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes — v37's recipe exactly), augmentation on
  (dihedral), **history_dropout: 0.5** (§1.1), val_fraction
  0.05 (random split within the train set, for
  best-checkpoint selection only), seed 0.
- **Config:** `ml/configs/ml4-h2.yaml`; run dir
  `ml/runs/ml4-h2-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, B1's finding).
  `checkpoint_every_epochs: 1` from the start (v24's resume
  lesson).
- **Memmap path (mandatory):** the train shards are bundled
  with `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml4-h2/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as v37):**
  best-by-internal-val-loss (`best.pt`), copied to
  `ml/runs/ml4-h2/value-net.pt` (gitignored data, the path
  v43's registry entry points at).
- **Resume rule (pre-registered, v24's lesson):** if training
  is killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans
  the full 15 via `schedule_epochs`); back up the pre-kill
  `best.pt` first, because the resumed run resets
  `best_val_loss` and would overwrite it at the next epoch
  regardless — the final best is chosen by comparing the
  resumed run's best val loss against the pre-kill best.

### 1.4 THE GATE (hard — before any games are played)

**The no-history holdout must match full-planes BEFORE any
games are played.** Measured with `eval_value_holdout.py` on
the game-level holdout (80 games, 3,378 positions, 2-ply
labels), pooled across sources. Two clauses, BOTH must pass:

- **G1 (match):** pooled no-history sign accuracy within **0.02**
  of pooled full-planes sign accuracy:
  `|sign(no-history) − sign(full)| ≤ 0.02`.
- **G2 (fit floor):** pooled full-planes sign accuracy **≥ 0.70**
  (v37's was 0.7519; v24's 0.7766 on 1-ply labels). Without G2
  a collapsed net (e.g. 0.55 full / 0.54 no-history) would
  "pass" G1 while being useless.

**If either clause fails, v43 does NOT go to battery — report
the holdout numbers and stop; do not burn battery time on a net
that fails its own gate.** No fallback retrain is pre-registered
(a different dropout rate would be a new version, not a v43
patch). The gate is also reported per source (v7 / v18 /
random, both views) and by side to move (§1.7) — a pooled pass
with a per-source collapse (e.g. v7 no-history still at chance)
is reported prominently even though the gate is defined on the
pooled numbers.

### 1.5 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.125, 0.25, 0.5}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 43`. The grid brackets
v37's winner (0.25) on both sides: the dropout net's values
should be calibrated like v37's (same labels, same recipe), so
0.25 is the prior favourite and 0.125/0.5 cover both
directions. Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure). **No-signal
rule:** if all three arms are within one win of each other,
pick α=0.25 (v37's winner) and say so — the sweep found no
signal. The sweep SELECTS α; the 100-game battery is the
verdict (standing lesson: small-sample smokes mislead — v24's
0.65 smoke at α=1.0 did not reproduce at 100 games; v31's 0.70
did not either). The registry entry's `value_weight` is updated
to the winner after the sweep. (Note: α is an inference-time
weight, so holdout sign accuracy cannot select it — the sweep
is by played win rate, as in v24/v31/v37.)

### 1.6 What would change the design

- If the gate (§1.4) fails, v43 stops — report the holdout
  numbers, do not battery.
- If the distilled net's holdout sign-accuracy on its OWN
  (2-ply-label) holdout is at chance (~0.50), the distillation
  failed to learn the deeper label function — stop and report,
  do not battery a chance-level net (subsumed by G2, stated
  separately for clarity).
- If the α sweep's best is within noise of the worst (all three
  within one win of each other), pick α=0.25 per the §1.5
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain re-runs the α
  sweep and the battery from scratch on the new form.

### 1.7 Holdout diagnostics (pre-registered)

All via `ml/scripts/eval_value_holdout.py` (sign accuracy +
MSE, full-planes and no-history views) on 2-ply labels:
- (a) **v43 net on its own holdout** (80 games, 3,378
  positions) — in-distribution; THE GATE (§1.4). ALSO report
  accuracy BY SIDE TO MOVE (Black/White) — the B1-White-collapse
  lesson: v37's engine-only data did not collapse White, and
  this checks the dropout didn't reintroduce a side skew.
- (b) **v37's net** (`gomoku-ml3-h2/ml/runs/ml3-h2/value-net.pt`,
  by absolute path) **on the same holdout** — reproduces v37's
  published numbers (pooled 0.7519 full / 0.5568 no-history;
  v7 0.7395 / 0.5163) as a sanity check on the eval setup AND
  the direct dropout comparison: same positions, same labels,
  only the training procedure differs.
- (c) **v43 net on v24's 1-ply labels** (v24's holdout shards,
  `gomoku-ml1-h1/ml/runs/ml1-h1/data/{v7,v18,random}-holdout/`)
  — reverse transfer: does the dropout net still reproduce the
  shallower function, and is IT history-robust too?
- (d) **v8's net on the 2-ply holdout** — baseline (expect
  ~0.50, as on v24's labels).
No decision hinges on (c)–(d); they diagnose whether the
dropout changed the fit and whether the fit transfers across
label depths.

### 1.8 Eval seeds (pre-registered)

h2h vs v8: v43=100 / v8=200, WITH `--openings
research/openings-ml4.json` (the pre-registered 10-opening set
— new this batch; fixes the mirror-match collapse, v36-vs-v8
was 21 unique in 100); h2h vs v37: same, with openings; h2h
vs v0: v43=100 / v0=200, NO openings; h2h vs v7: v43=100 /
v7=200, NO openings; ladder seed 100, **40 games/opponent**,
mcts_sims=100; swap 100 games vs v7, balanced placement,
**placement_seed 11** (the pre-registered seed — v31/v32 used 0
by mistake, do not repeat that), tempo judge; tactical `--seed
100`. Every leg reports its uniqueness audit prominently.

### 1.9 Eval order (pre-registered)

(1) Data verification (§2). (2) Memmap bundle + training (§3)
+ holdout diagnostics and THE GATE (§4) — STOP if the gate
fails. (3) α sweep (§5) — registry updated to the winner.
(4) FAST LOOP (§5): up to 4 iterations of 10 games vs v8 AND
10 games vs v0 (dev seeds, no openings — the fast loop follows
v37's protocol). (5) FULL VALIDATION BATTERY on the final v43
(§6): tactical re-run, h2h vs v8 100 games (openings), 100
games vs v0, 100 games vs v7, 100 games vs v37 (openings),
100-game swap match vs v7, ladder 40 games/opponent, clause-4
suite re-run. Any battery numbers measured before the fast loop
completes are INTERMEDIATE, labelled as such; only the final
battery counts for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games, openings
selfplay_batch.py --engine-a v43 --engine-b v8 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v43-vs-v8.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 43 --ladder-games 40 --out research/iterations/ladder-v43.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 43 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, vs v37 (openings), swap vs v7 (balanced/tempo, placement_seed 11, seeds 100/200)
selfplay_batch.py --engine-a v43 --engine-b v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v43-vs-v7.json
selfplay_batch.py --engine-a v43 --engine-b v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v43-vs-v0.json
selfplay_batch.py --engine-a v43 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v43-vs-v37.json
selfplay_batch.py --engine-a v43 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v43-vs-v7-swap.json
```

## 2. Data verification (actuals)

**Shard counts + label stats: PASS (§0).** v37's shards copied
verbatim; every cell matches iter-37 §2 exactly (37,227
positions, 33,849 train / 3,378 holdout, mean|v| 0.1570 /
0.1775 / 0.4022 by source).

**Spot re-labelling: PASS — 0 mismatches.** The ported label
phase (`generate_distill_data.py label --label-mode rollout2`)
re-ran on 20 games (7 v7 + 7 v18 + 6 random, the first games of
each source — all train shards, no holdout routing) and the
regenerated shards match v37's shards EXACTLY (888 quiet
positions; planes, policy, value, moves, winner all
bit-identical). Master's rollout-leaf teacher (with the ported
`_rollout2_leaf_value`) is the same function v37 distilled —
no teacher drift. (First attempt failed with `AttributeError`
on the missing `_rollout2_leaf_value` — see the §0 port note;
after the verbatim port, labelling took 21 s for the 20 games.)

## 3. Training (actuals)

**Training COMPLETE 2026-11-19 19:59 UTC** (started 19:41 UTC;
~19 min wall including one interruption): config
`ml/configs/ml4-h2.yaml`, 15 epochs, batch 64, memmap bundle
(33,849 samples; 32,157 train / 1,692 val, seed-0 split).
Run dir `ml/runs/ml4-h2-train/`; log `ml/runs/ml4-h2/train.log`.
RSS stayed low (memmap path). 7,545 steps, ~47–103 s/epoch.

**Interrupt and recovery (2026-11-19 ~19:46 UTC):** a VM
replacement killed the run at **epoch 4, step ~1810**. Per the
pre-registered §1.3 resume rule: the pre-kill `best.pt`
(epoch 3, val 0.1201) was backed up at
`checkpoints/best-epoch3.pt`, and training resumed from the
latest per-epoch checkpoint (`checkpoint-1509.pt`, epoch 3,
step 1509 — verified to load with optimiser state) with the
REMAINING 12 epochs (`--resume ... --epochs 12`; cosine
schedule spans the full 15 via `schedule_epochs`). Resume
caveat (same as v24's): the RNG restarts from seed 0 on
resume, so resumed epochs 4–15 reuse the permutation /
augmentation / dropout RNG sequence of epochs 1–12 — the
dropout is still Bernoulli(0.5) per sample per epoch, just not
a fresh stream. The resumed run's best (epoch 14, val 0.1115)
beats the pre-kill best (0.1201) outright, so the resume
best-reset flaw never bit and no restore from `best-epoch3.pt`
was needed.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.1080 | 0.1228 |
| 2 | 0.1019 | 0.1227 |
| 3 | 0.1000 | 0.1201 |
| 4 | 0.0980 | 0.1222 |
| 5 | 0.0962 | 0.1200 |
| 6 | 0.0946 | 0.1216 |
| 7 | 0.0954 | 0.1178 |
| 8 | 0.0929 | 0.1184 |
| 9 | 0.0922 | 0.1171 |
| 10 | 0.0907 | 0.1148 |
| 11 | 0.0888 | 0.1136 |
| 12 | 0.0875 | 0.1136 |
| 13 | 0.0867 | 0.1120 |
| 14 | 0.0859 | **0.1115** |
| 15 | 0.0853 | 0.1115 |

Val loss decreased overall (noisy in epochs 4–6, then
monotone from epoch 7) — no overfitting by epoch 15, but the
pre-registered plan is 15 epochs and the battery is the
verdict, so no extension. **Best checkpoint: epoch 14
(`best.pt`, step 7042, val 0.11153)** — epoch 15's 0.11154 is
a hair worse, so `best.pt` ≠ `final.pt` (unlike v37, where
epoch 15 was best). **The `--out` copy was `final.pt`, NOT
`best.pt`** (verified: max|out−final| = 0.0, max|out−best| =
503 — the 503 is `num_batches_tracked` only, 503 more batches
in epoch 15; median weight diff 0.0009) — so `best.pt` was
copied manually to the registry path
**`ml/runs/ml4-h2/value-net.pt`** (md5
`e9c8befbbd1e3de1ed2c309e9cb050e9`, verified identical to
`best.pt`), per the pre-registered best-by-val-loss rule.
Val loss is worse than v37's (0.1115 vs 0.0986) — expected:
dropout makes the value function harder to fit (half the
samples lack history). Sign accuracy is the comparable metric
(§4), and there the dropout costs nothing. Architecture
verified: ResNet 16ch/2 blocks, 126,759 params, 48 tensors —
identical counts to v37's and v8's checkpoints.

## 4. Holdout diagnostics and the gate (actuals)

Game-level holdout (every 10th game, never trained on — the SAME
80 games / 3,378 positions v24/v37 held out), targets = 2-ply
labels unless noted. v43 net = `ml/runs/ml4-h2/value-net.pt`;
v37 net = `ml/runs/ml3-h2/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml4-h2/holdout-2ply-{v7,v18,random}.txt` (2-ply) and
`holdout-1ply-{v7,v18,random}.txt` (1-ply). Sanity check: this
run reproduces v37's published 2-ply numbers EXACTLY on its own
holdout (v7 0.7395/0.5163, v18 0.8436/0.5846, random 0.6623/
0.6604, pooled 0.7519) and v24's published 1-ply numbers
EXACTLY (pooled 0.7766) — the eval setup is sound. (One
rounding note: iter-37 printed v37's pooled no-history as
0.5568; recomputed from the JSON counts it is 1791/3217 =
0.5567 — the counts agree, the last digit is rounding.)

### (a) v43 net on its own 2-ply holdout (in-distribution) — THE GATE

| Holdout | n pos | v43 sign (full) | v43 sign (no-hist) | v43 MSE (full) | v37 sign (full) | v37 sign (no-hist) | v8 sign (full) |
|---|---|---|---|---|---|---|---|
| v7 | 2,050 | 0.7458 | 0.7589 | 0.0801 | 0.7395 | 0.5163 | 0.5572 |
| v18 | 790 | 0.8423 | 0.8410 | 0.0917 | 0.8436 | 0.5846 | 0.4897 |
| random | 538 | 0.6942 | 0.6942 | 0.3110 | 0.6623 | 0.6604 | 0.5009 |
| **pooled** | **3,378** | **0.7606** (2447/3217) | **0.7681** (2471/3217) | — | **0.7519** | **0.5567** | **0.5316** |

**THE GATE PASSES — both clauses:**
- **G1 (match):** |0.7681 − 0.7606| = **0.0075 ≤ 0.02** ✓
- **G2 (fit floor):** 0.7606 **≥ 0.70** ✓

**The dropout fixed the history dependence completely.** The
no-history view is now EQUAL OR BETTER than full-planes on
every source (v7: 0.5163 → 0.7589, +0.243; v18: 0.5846 →
0.8410, +0.256; random unchanged at 0.6942), and the
mean|v| ballooning is gone (v7 no-history mean|v| 0.0679 vs
v37's 0.4356; MSE 0.0797 vs 0.4361). And the dropout cost NO
full-planes fit: v43's pooled full-planes sign (0.7606)
slightly BEATS v37's (0.7519, +0.009) — the val-MSE cost (§3:
0.1115 vs 0.0986) is magnitude calibration, not sign. The
§1.6 chance-level stop does not trigger either. **v43 proceeds
to the α sweep and the battery.**

**By side to move** (v43 net, 2-ply holdout, full planes;
`ml/runs/ml4-h2/by_side.py`, raw `by-side.txt`): Black-to-move
sign 0.7411 (1626 non-draw), MSE 0.1440; White-to-move sign
**0.7806** (1591 non-draw), MSE 0.0948. **No White collapse**
— White is BETTER than Black in holdout, as in v37 (0.7269 /
0.7775). No-history by side is equally robust: Black 0.7485,
White 0.7882 (v37's no-history by side: Black 0.5012 at
chance, White 0.6135). The dropout did not reintroduce a side
skew.

### (c) Reverse transfer: v43 net on v24's 1-ply labels

| Holdout | v43 sign (full) | v43 sign (no-hist) | v37 sign (full) | v24 sign (full) |
|---|---|---|---|---|
| v7 | 0.7534 | 0.7669 | 0.7550 | 0.7713 |
| v18 | 0.8357 | 0.8344 | 0.8449 | 0.8384 |
| random | 0.7113 | 0.6922 | 0.6482 | 0.7055 |
| **pooled** | **0.7664** | **0.7708** | **0.7590** | **0.7766** |

The dropout net reproduces the shallower function at 0.7664 vs
v37's 0.7590 and v24's 0.7766 — good reverse transfer, and
history-robust on the 1-ply labels too (gap 0.0044; v37's
1-ply no-history pooled is 0.5740). Note v43 even beats v24's
net on random no-history (0.6922 vs 0.5468) — the dropout fixed
v24's random-position history dependence as well (iter-24 §4:
v24 leans on history planes for random ±1 labels).

### Test-suite note (clause 4, early check)

Full ml pytest on the implementation tree: **434 passed, 1
skipped, 58 xfailed, 1 FAILED** (14:51; log
`ml/runs/ml4-h2/pytest-full.log`) — the failure is
`test_per_game_seeding.py::test_resume_reproduces_uninterrupted`.
**Pre-existing, NOT caused by this workstream:** it reproduces
on a clean worktree at the base commit 4457fe7 (before any v43
change) and in isolation. Root cause (verified by reading the
driver): the openings tooling commit (4457fe7) added a
`"opening"` key to EVERY non-swap record in
`selfplay_batch.py` (`rec["opening"] = []` when no openings
are given) but did not update the test's `_play_prefix`
duplicate, which builds records without the key — so the
resumed-vs-ref record comparison fails at index 0 on the
missing key even though the MOVES are identical. The driver's
resume behaviour itself is correct (per-game seeding
reproduces games; the swap variant of the test, whose
`_play_prefix` path DOES set `"opening"`, passes). Fix: the
test's non-swap path now sets `rec["opening"] = []` (one line
+ comment); re-run: **7/7 in test_per_game_seeding.py pass**.
The full suite is re-run at the end (§6) on the final tree.
npm test / tsc are checked in §6.

## 5. α sweep and fast loops

**α sweep — COMPLETE 2026-11-19 ~23:00 UTC** (20 games each vs
v8, seeds 1001/2001; per-alpha JSONs
`research/iterations/alpha-v43-<α>-vs-v8.json`). Interrupted
once by a VM replacement at α=0.25 game 12; resumed cleanly
(α=0.125 already complete, α=0.25 resumed from game 13 with
per-game seeding). All 60 games unique (20/20 per leg).

| α | W-L-D | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| 0.125 | 10-9-1 | 0.525 | [0.3205, 0.7215] | 6-4-0 | 4-5-1 |
| **0.25** | **10-7-3** | **0.575** | **[0.3641, 0.7617]** | 9-1-0 | 1-6-3 |
| 0.5 | 11-6-3 | 0.625 | [0.4095, 0.8002] | 7-1-2 | 4-5-1 |

**Winner by the pre-registered rule: α = 0.25** — registry
`value_weight` set to 0.25. The §1.5 **no-signal rule
triggers**: the three arms' win counts are 10 / 10 / 11 —
within one win of each other — so the rule picks v37's winner
(0.25) and the sweep is recorded as finding no signal. This is
the first time the no-signal rule has triggered in the program
(v24: spread 6; v37: spread 5; v38: spread 7). The win-rate
spread (0.525 → 0.575 → 0.625, monotone in α) is noted but
does NOT override the rule: the entire spread is one flipped
game (0.25 → 0.5) plus draw-vs-loss differences, all three CIs
overlap heavily, and the rule exists precisely because 20-game
sweeps cannot resolve differences this small. The sweep script's
own nominal winner (0.5, by win rate) is recorded, not followed
— following it would be post-hoc selection on noise. The
battery is the verdict either way.

**Colour-split concern (recorded, not acted on):** at α=0.25
the split is extreme — Black 9-1 (0.90), White 1-6-3 (0.25) —
while α=0.5 is balanced (Black 0.80, White 0.45). v37's α=0.25
sweep had the OPPOSITE pattern (White 8-2). Ten games per
colour cannot resolve whether v43-at-0.25 is genuinely
White-weak or this is a cold draw; the fast loop (§5.1) and the
battery (50 games per colour) are the arbiters. The α grid is
NOT extended post-hoc.

### 5.1 Fast loop, iteration 1

**vs v8 (dev seeds 1001/2001): 5W-2L-3D in the first 10 games**
of the α=0.25 sweep leg — the sweep leg IS the fast-loop leg
(same seeds, same α; re-playing would reproduce the games
exactly under per-game seeding, so the sweep JSON was analysed
directly; first 10 copied to
`research/iterations/fast-v43-vs-v8-k1.json`).
`classify_losses`: 2 losses, **both via open four, 0 missed
blocks**, 0 clean missed preventions in 8 threatened turns.
`find_losing_moves`: both losses already-lost (game 4, ply 38;
game 9, ply 17 — 0 saving moves in 99 / 65 replies checked).
The full 20-game leg agrees: 7 losses, all 7 via open four, 0
missed blocks in 35 threatened turns.

**vs v0 (dev seeds 1001/2001): 2W-7L-1D (0.25, CI [0.0809,
0.5578])** — v43 LOSES to the weakest engine, a stark
non-transitivity against its 0.575 vs v8. `classify_losses`:
7 losses, **all 7 via open four, 0 missed blocks**, 0 clean
missed preventions in 41 threatened turns. `find_losing_moves`:
5/7 already-lost (games 1, 2, 3, 4, 7 — gradual positional
drift, no saving move); 2 real blunders with saving moves:

- **Game 8 (Black, 42 plies), ply 36:** played (2,7); saves
  (9,8), (9,11), (10,11). `diag_root`: 89 root children over
  100 sims — the search is visit-starved. Actual (2,7) and save
  (9,11) are TIED at 5 visits with q −0.049 vs −0.044; the
  tie-break picked the loser despite the save's higher prior
  (0.0555 vs 0.0510). The 1-ply leaves cannot distinguish them
  either (all ≈ −0.200). The position was already bad for Black
  (opponent breadth: 4 fours, 7 true-threes). Mechanism:
  visit-allocation / tie-break noise in a flat, bad position —
  the v40 forensics mechanism ("blunder dilution, visit
  allocation"), NOT a leaf mis-ranking.
- **Game 9 (White, 23 plies), ply 13:** played (8,11); save
  (8,7). `diag_root`: the search ACTIVELY preferred the loser —
  (8,11) 16 visits, q −0.064 vs (8,7) 12 visits, q −0.099 —
  and the 1-ply heuristic leaves agree (+0.192 vs +0.186 under
  v22). The loser is the greedy attacking move (4 own fours);
  the save is the defensive move (opponent development 11 vs
  21 after it). The leaf says White is winning (+0.19) after a
  move from which Black forces a win in 4 plies; the backed-up
  q (−0.064) moved the right way from the leaf but not far
  enough to flip the ranking. Mechanism: leaf mis-ranking on
  own-threat breadth vs opponent counter-threat, partially
  corrected by search but not enough at 100 sims.

**No patchable weakness class found — fast loop stops after
iteration 1.** The two blunders are (a) visit-tie noise and (b)
a leaf mis-ranking that search partially corrects; every lever
that could address either is CLOSED: more sims ("more sims is
NOT monotonically better"), VCF checks at root/leaf (v32, v33,
v35, v40 — the leaf-blind direction is closed), q-based root
decisions / tie-break gates (v34, root-decision lever
exhausted). The dominant signature — 9/9 losses via open four,
0 missed blocks across both legs — is gradual positional drift
against opponents who build double threats, not a one-move
defect. The vs-v0 collapse is a measurement question for the
battery (100 games), not a fast-loop patch target.

**Control (v37 vs v0, same seeds 1001/2001): 3W-6L-1D
(0.35, CI [0.1369, 0.6463])** — reproduces iter-37's recorded
fast loop vs v0 EXACTLY (same score, same CI), confirming the
setup is correct and the engines are deterministic. Correction
to an early working note: v37's fast loop vs v0 was 3-6-1, not
9-1 (the 9-1 figure was a mis-recollection; iter-37 §5 says
3W-6L-1D). v43's 2-7-1 is one game from v37's 3-6-1 — the same
cold-start pattern, and v37's 100-game battery vs v0 recovered
to 0.585. The two fast loops are near-identical in signature:
9 losses each, 9/9 via open four, 0 missed blocks, 7/9
already-lost, 2 genuine blunders both vs v0. v43 plays like
v37 in the fast loop; the battery measures whether it plays
like v37 at full sample.

## 6. Validation battery

### 6.1 Tactical — COMPLETE 2026-11-19 ~23:50 UTC — **CLAUSE 3 PASSES**

`eval_tactical_full.py --version 43 --suite both --seed 100`
(log `ml/runs/ml4-h2/tactical-v43.log`):

- Old suite: **20/20** ✓
- Hard suite (corrected): **29/29** ✓ — including
  **setup-v1v0s-g92: chose (9,6)**, the quiet setup v31 failed
  and v37 fixed. The dropout retrain keeps v37's quiet-setup
  fix at the engine level.

### 6.2 h2h vs v8 (clause 1) — COMPLETE 2026-11-20 03:09 UTC — **CLAUSE 1 FAILS**

`selfplay_batch.py --engine-a v43 --engine-b v8 --games 100
--openings research/openings-ml4.json --seed-a 100 --seed-b 200`
(raw `research/iterations/batch-v43-vs-v8.json`; recomputed
from the per-game records by
`ml/runs/ml4-h2/recompute-aggregates.py` — W-L-D, win rate,
Wilson CI, colour splits, uniqueness, and de-dup rate all
reproduce the driver's stored aggregates exactly):

- **47W-46L-7D, 0.5050, 95% CI [0.4086, 0.6010].** The win rate
  is above 0.50 by half a game, but the CI lower bound (0.4086)
  is far below 0.50 — clause 1 fails. The de-dup cut agrees:
  97/100 unique games, de-dup 47W-43L-7D, 0.5206, CI [0.4223,
  0.6173] — also fails. The opening variety worked (97 unique
  vs v36-vs-v8's 21 without openings).
- Colour split INVERTED vs the usual ML pattern: as Black
  18W-27L-5D (0.4100); as White 29W-19L-2D (0.6000). v43 is
  White-strong against v8 — the opposite of v37's sweep
  pattern and of the §5 α=0.25 leg (Black 9-1, White 1-6-3),
  which confirms that leg's split was a cold draw, not a real
  White weakness at this α.
- **v37 scored 0.7500 (CI [0.6570, 0.8245]) on this exact
  pairing.** The dropout gave up nearly all of v37's edge over
  v8 (−0.245): the gate-passing net plays like v8, not like
  v37.
- Battery history: the leg was interrupted twice by VM
  replacements (original worker killed at 5 games; pickup
  worker killed at 70); auto-resume reproduced the
  uninterrupted run exactly both times (per-game seeding).

### 6.3 h2h vs v0 — COMPLETE 2026-11-20 04:26 UTC

`selfplay_batch.py --engine-a v43 --engine-b v0 --games 100
--seed-a 100 --seed-b 200` (raw
`research/iterations/batch-v43-vs-v0.json`; recomputed —
exact match):

- **46W-49L-5D, 0.4850, CI [0.3894, 0.5817].** v43 LOSES to the
  frozen baseline outright — the fast loop's 0.25 did NOT
  recover at full sample, unlike v37's (fast loop 0.35 →
  battery 0.5850). 96/100 unique; de-dup 44W-47L-5D, 0.4844,
  CI [0.3869, 0.5830].
- Colour split is extreme: as Black 36W-10L-4D (0.7600); as
  White **10W-39L-1D (0.2100)**. The White collapse vs the
  weakest engine is far worse than v37's White 0.44 against
  the same opponent. The dropout net defends as White far worse than
  v37's net — consistent with the §5.1 game-9 mechanism (the
  leaf overvalues own-threat breadth and undervalues the
  opponent's counter-threat), now priced at 100 games.
- One VM replacement during the leg (killed at 60 games);
  auto-resumed from game 68.

### 6.4 h2h vs v7 — COMPLETE 2026-11-20 06:18 UTC

`selfplay_batch.py --engine-a v43 --engine-b v7 --games 100
--seed-a 100 --seed-b 200` (raw
`research/iterations/batch-v43-vs-v7.json`; recomputed —
exact match):

- **40W-57L-3D, 0.4150, CI [0.3233, 0.5130].** Loses to the
  heuristic bar reference, slightly below v37's 0.4500 (within
  noise). 97/100 unique; de-dup 37W-57L-3D, 0.3969, CI
  [0.3053, 0.4964].
- Colour split near-even: as Black 21W-28L-1D (0.4300); as
  White 19W-29L-2D (0.4000) — v43's White vs v7 (0.40) is
  actually BETTER than v37's (0.31); the v7 gap is a Black
  gap for v43 (0.43 vs v37's 0.59).
- One VM replacement during the leg (killed at 80 games);
  auto-resumed from game 89.

### 6.5 h2h vs v37 — COMPLETE 2026-11-20 09:05 UTC

`selfplay_batch.py --engine-a v43 --engine-b v37 --games 100
--openings research/openings-ml4.json --seed-a 100 --seed-b 200`
(raw `research/iterations/batch-v43-vs-v37.json`; recomputed
— exact match). The direct dropout comparison: same positions,
same labels, same search — only the training procedure differs.

- **34W-60L-6D, 0.3700, 95% CI [0.2818, 0.4678].** The CI lies
  ENTIRELY below 0.50: the dropout net is decisively weaker
  than v37's net head-to-head. 94/100 unique; de-dup
  31W-57L-6D, 0.3617, CI [0.2718, 0.4625] — the same verdict on
  the de-dup cut.
- As Black 21W-25L-4D (0.4600); as White 13W-35L-2D (0.2800)
  — the White gap again (v37's net beats the dropout net
  hardest as Black).
- **This is the experiment's answer.** The §1 hypothesis had
  two branches: if dropout kept v37's played strength AND
  fixed the history dependence, the history planes were a
  shortcut; if it cost played strength, the history planes
  carry real signal the 2-ply labels need. The second branch
  is the result: the gate passed (history dependence fixed,
  §4) AND played strength fell by 0.13 against v37's own net
  on identical data. The 2-ply label function genuinely
  depends on game-phase / last-move information that the
  current-board planes alone do not carry at this sample
  scale — forcing the trunk to read the value from the current
  board alone makes it a worse value function where it counts
  (played games), even though holdout sign accuracy is
  unchanged (0.7606 vs v37's 0.7519). Holdout sign accuracy
  does not predict played strength here — the same lesson as
  v31 (holdout transfer does not predict played strength).
- One VM replacement during the leg (killed at 50 games);
  auto-resumed from game 51.

### 6.6 Swap vs v7 — COMPLETE 2026-11-20 11:06 UTC

`selfplay_batch.py --engine-a v43 --engine-b v7 --games 100
--swap-rule --placement balanced --placement-seed 11 --judge
tempo --seed-a 100 --seed-b 200` (raw
`research/iterations/batch-v43-vs-v7-swap.json`; recomputed —
exact match):

- **35W-58L-7D, 0.3850, CI [0.2956, 0.4829].** Under the
  official swap rule v43 loses to v7 by more than under normal
  rules (0.4150) — the swap does not help this pairing (v37's
  swap was 0.4400, also below its normal-rules 0.4500).
- Placement seed 11 confirmed in the JSON (the pre-registered
  seed); tempo judge; v43 played Black in 52 games and White
  in 48. As Black 20W-27L-5D (0.4327); as White 15W-31L-2D
  (0.3333).
- Uniqueness audit: **100 unique games of 100** — perfect
  uniqueness (the balanced placements diversify every game);
  de-dup = nominal.
- One VM replacement during the leg (killed at 30 games);
  auto-resumed from game 38.

### 6.7 Ladder (clause 2) — COMPLETE 2026-11-20 11:41 UTC — **CLAUSE 2 FAILS**

`eval_version.py --version 43 --ladder-games 40 --seed 100`
(raw `research/iterations/ladder-v43.json` — committed
artifact; Elo recomputed from the heuristic leg's score by
`recompute-aggregates.py`: −88.7, interval [−198.2, 20.7] —
the 0.1 difference from the stored [−198.1, 20.6] is the
stored score CI rounded to 4dp):

- vs random: **40W-0L-0D** (1.0000, CI [0.9124, 1.0000]).
- vs heuristic: **14W-24L-2D** (0.3750, CI [0.2422, 0.5297]);
  as Black 8W-11L-1D, as White 6W-13L-1D.
- vs mcts: **16W-23L-1D** (0.4125, CI [0.2742, 0.5661]); as
  Black 13W-7L-0D, as White **3W-16L-1D** — the White collapse
  again, worst against the search opponent. v37 BEAT the
  ladder MCTS (0.575); v43 loses to it.
- **Ladder Elo estimate −88.7, interval [−198.1, 20.6]** —
  below v8's 0.0, so **clause 2 FAILS**. The interval includes
  0.0, but the bar is the point estimate strictly above 0.0
  and −88.7 is not. v37's Elo was −34.9 (20 games/opponent);
  v43 is worse on the point estimate at 40 games/opponent —
  the 40-game re-basing narrowed the interval as designed
  (±~110 vs v37's ±~150), and the verdict does not turn on
  noise: v43's heuristic leg (0.375) is below v37's (0.45 at
  20 games) and below v8's anchor level outright.
- The ladder's tactical leg re-ran the old suite at 20/20 (the
  hard suite was re-run separately in §6.8).

### 6.8 Test suites (clause 4) — COMPLETE 2026-11-20 ~11:52 UTC — **CLAUSE 4 PASSES**

Full suites on the final tree (logs `ml/runs/ml4-h2/`):

- ml pytest: **435 passed, 1 skipped, 58 xfailed, 0 failures**
  (238.85 s; log `pytest-final.log`). The 435th pass vs the
  §4 early check's 434 is the §4 seeding-test fix now in the
  suite. One pytest run was SIGTERM'd by a VM replacement
  mid-run with no output; the re-run completed cleanly.
- npm test: **55/55 passed** (log `npm-test-final.log`).
- tsc: **clean, exit 0** (log `tsc-final.log`).
- `ValueMCTS.__init__` checked for the duplicate-parameter
  trap before the battery ran: a single `value_weight: float |
  None = None` parameter — no duplicate.
- Clause 3 re-confirmed on the final tree (§6.1 re-run):
  tactical **20/20 + 29/29** at `--seed 100` (log
  `tactical-v43-final.log`), including setup-v1v0s-g92 choosing
  (9,6).

## 7. Verdict

**v43 is REJECTED — 2 of 4 clauses pass.**

| Clause | Bar | v43 | Verdict |
|---|---|---|---|
| 1 | h2h vs v8 > 0.50, Wilson 95% CI lower > 0.50 | 0.5050, CI [0.4086, 0.6010]; de-dup 0.5206, CI [0.4223, 0.6173] | **FAIL** |
| 2 | ladder Elo > 0.0 | −88.7, interval [−198.1, 20.6] (40 games/opponent) | **FAIL** |
| 3 | tactical ≥ 20/20 + 29/29 | 20/20 + 29/29 (re-verified on the final tree) | **PASS** |
| 4 | all test suites green | pytest 435/1/58, npm 55/55, tsc clean | **PASS** |

**v8 stays the ML champion; v37 stays the strongest ML version
ever by head-to-head (0.75 vs v8).** v43 is the first version
in the program to pass its own training gate decisively (gap
0.0075, full-planes 0.7606 — the history dependence is GONE)
and still play worse than the net it was distilled from: it
ties v8 (0.505), loses to v0 (0.485), loses to v7 (0.415),
loses to v37 head-to-head (0.370, CI entirely below 0.50),
and posts a ladder Elo (−88.7) below v37's (−34.9) and below
v8's 0.0.

**The result (pre-registered §1, second branch): the history
planes carry real signal the 2-ply labels need.** At 33,849
samples the 2-ply label function is not learnable from the
current board alone at played-strength quality — the dropout
forces the trunk to drop information (game phase, last-move
character) that the value function genuinely uses, and the
played cost is concentrated as White (vs v0 White 0.21, vs
the ladder MCTS White 3-16-1, vs v37 White 0.28), where
defence depends on reading the opponent's developing threats
through the recent-move history. The holdout sign accuracy
(0.7606, slightly ABOVE v37's 0.7519) did not see it coming —
sign accuracy on quiet holdout positions is not a proxy for
played strength, and the gate as designed (sign-accuracy
match + fit floor) cannot detect a played-strength regression.
**History dropout in this form is CLOSED as a lever** — do not
retry p=0.5 joint dropout on the 2-ply recipe. The §1 fallback
(a history-free architecture, v38's §7) is also suspect now:
if the labels need history, removing history from the
architecture faces the same wall, and the fix — if any — is on
the label side (labels that are a function of the current board
alone, e.g. deeper search that washes out phase information)
or the data side (enough samples that the board→label mapping
is learnable without the shortcut), not the dropout side.

**Non-transitivity, recorded prominently:** v43 ≈ v8 (0.505)
> v37 loses to v43? No — v37 beats v43 (0.63 from v37's side)
while v37 beats v8 (0.75) and v43 ties v8: the ordering v37 >
v43 ≈ v8 is transitive on this trio. But v43 loses to v0
(0.485) while v8 beats v0 (0.545) and v37 beats v0 (0.585):
v0 > v43 ≈ v8 > v0 is a genuine cycle — v43 loses to the
frozen baseline outright while tying the version that beats
it (v8: 0.545 vs v0), driven entirely by v43's White game
(0.21 vs v0 as White).

**Battery hygiene note:** the battery survived SIX VM
replacements (one before pickup at 5 games, five during the
pickup — legs 1–5 each killed once); every leg auto-resumed
from its JSON with per-game seeding and no game was lost or
replayed. Uniqueness was 94–100/100 on every leg — the
openings set and per-game seeding have ended the
mirror-match collapse (v36-vs-v8 was 21 unique in 100).

**What v43 leaves behind:** the ported `_rollout2_leaf_value`
+ spot re-labelling verification (0/888 mismatches), the
history-dropout training machinery (`drop_history_planes`,
`history_dropout` config key, unit tests), the seeding-test
fix (`rec["opening"] = []`), and the v37 registry entry +
checkpoint as a comparison engine — all reusable by H1/H3 and
future batches. The checkpoint `ml/runs/ml4-h2/value-net.pt`
(md5 `e9c8befbbd1e3de1ed2c309e9cb050e9`) is gitignored data;
the registry entry points at it for any future reproduction.

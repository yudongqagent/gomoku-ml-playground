# Iteration 43 — v43: history dropout on the v37 net (ML batch 4, H2)

**Status: IN PROGRESS — pre-registered; training starting.**
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

TODO — fill in when §2 runs: spot re-labelling result.

## 3. Training (actuals)

TODO — fill in when training completes.

## 4. Holdout diagnostics and the gate (actuals)

TODO — fill in when the gate is measured.

## 5. α sweep and fast loops

TODO — fill in when the sweep runs (gate must pass first).

## 6. Validation battery

TODO — fill in when the battery runs.

## 7. Verdict

TODO — fill in at the end.

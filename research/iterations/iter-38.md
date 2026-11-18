# Iteration 38 — v38: multi-task value net (value + threat classification) (ML batch 3, H3)

**Status: IN PROGRESS — pre-registered; implementation starting.**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H3 workstream of ML batch 3 (H1 is v36,
v27's threshold ported to the value engine; H2 is v37,
deeper-teacher distillation); work happens on branch
`research/ml-batch-3-h3` in the `gomoku-ml3-h3` worktree. v38
differs from v8 ONLY in the value net's weights, architecture
(an auxiliary threat-classification head on the shared trunk),
and the swept α — same PUCT search, same v1 priors, same sims
(100), same c_puct (1.5), same threat check, same terminal cases
— so the battery is a clean A/B of the multi-task net against
v8's weak-self-play net (and, less cleanly, against v24's
single-task distilled net: v38 changes ONLY the training signal,
not the data or the value labels).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v8. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v38 only. Fast-loop games use
development seeds (1001/2001, 1002/2002, ...), never the validation
battery's seeds (100/200).

## 0. Ports and cost measurements (before pre-registration)

**Ports (ML batches 1–2 were never merged to master, so their
code was ported into this worktree — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: ML batch 1 H2's memmap version
  (pre-allocated `load_shards` + memmap bundle support + lazy-index
  `train()`), copied VERBATIM from `gomoku-ml1-h2` (sha256
  `7e8cc28d…` matches the source; identical file in `gomoku-ml2-h2`).
  Multi-task support (§1.2) is added AFTER the verbatim copy as a
  separate, documented edit.
- `ml/scripts/build_mmap_bundle.py`: copied VERBATIM from
  `gomoku-ml1-h2` (sha256 `92664381…` matches; identical in
  `gomoku-ml2-h2`). Extended afterwards to carry `threat` labels
  into the bundle when the shards have them (auto-detected).
- `ml/scripts/generate_distill_data.py`: v24's games+label
  pipeline, copied VERBATIM from `gomoku-ml1-h1` (sha256
  `1ce8e6d3…` matches; identical in `gomoku-ml2-h2`). Used ONLY
  for the §1.4 verification re-run (value labels are reused from
  v24's shards, not re-generated).
- `ml/scripts/sweep_alpha.py`: copied from `gomoku-ml2-h2`
  (sha256 `a2b2c0c7…`), adapted ONLY in the docstring and the
  `--version` default (31 → 38); the match loop, auto-resume,
  and selection logic are unchanged. It sets
  `player._mcts.value_weight` directly per run, so the registry
  entry is touched only once (to the winner) afterwards.
- `ml/tests/test_load_shards.py`: copied VERBATIM from
  `gomoku-ml2-h2` (tests the memmap loader).
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port (ML batch 1 H1's change, carried into ML batch 2),
  applied by hand to this tree's file — the SAME 3-hunk shape
  as `gomoku-ml2-h2`'s (docstring + `value_weight: float | None
  = None` parameter + `self.value_weight` in the leaf);
  default None → v8's 0.25, so v8 is byte-identical in
  behaviour. **NOTE FOR THE COORDINATOR: this workstream DOES
  touch `ValueMCTS.__init__`'s signature (adds the
  `value_weight` keyword, exactly as v24/v31 did) — the known
  silent-auto-merge-breakage site. Check for duplicate
  parameters after merging.**
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS`, plus the v38 registry entry
  (`value_checkpoint: ml/runs/ml3-h3/value-net.pt`,
  `value_weight: 1.0` placeholder until the §1.6 sweep sets it).
- `ml/gomoku_ml/baseline.py` is FROZEN and untouched.
- `ml/scripts/eval_value_holdout.py` is already on master
  (sha256 `d109131d…` matches `gomoku-ml1-h1`'s) — no port
  needed. A threat-accuracy companion script
  (`eval_threat_holdout.py`) is NEW for this workstream.

**Port smoke checks (2026-11-18):** v8 loads and plays in this
worktree (first move 112, centre; `value_weight` 0.25 default
preserved — the port does not change v8's behaviour); v38 is
registered (registry entry + `test_v38_registry_entry`); the
multi-task net builds from the registry (ResNet 16ch/2 blocks +
threat head, 141,263 params — v8's 126,586 + 14,677 for the
threat head). Targeted tests green: threat (8), multitask (8),
load_shards, architectures, config, mcts_value — 60 passed.
npm test 55/55, tsc clean. Full pytest on the implementation
tree: **434 passed, 1 skipped, 58 xfailed, 0 failed** (959s;
log `ml/runs/ml3-h3/pytest-full-1.log`).

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, tactical 20/20 + 29/29 on the CORRECTED suite; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical ≥ v8's (old suite 20/20,
hard suite 29/29, run at `--seed 100` on the corrected suite),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v38 vs v7 (100
games), v38 vs v0 (100 games), and a 100-game swap match vs v7
under the tempo judge (balanced placement, placement_seed 11 —
the pre-registered seed; v31/v32 used 0 by mistake, do not
repeat that). Non-transitivity is recorded prominently.

**The hypothesis (plan H3):** the value head overrides a correct
prior on quiet setups because the net cannot distinguish threat
patterns from quiet setups — it overvalues immediate threats
(v26 §2.4 Group A: the prior ranks the setup move 1st, the value
head gives the non-setup move q +0.555; v31 §6.1: on
setup-v1v0s-g92 the prior ranks (9,6) first but the net values it
−0.377 vs (5,4)'s +0.788, while v8's small net values (9,6) at
+0.246). An auxiliary threat-classification task — trained
jointly with the value head on the SAME trunk — should teach the
shared trunk the threat-vs-quiet distinction, so the value head
stops overriding correct priors on quiet positions. The threat
head is TRAINING-ONLY: at inference, ValueMCTS uses only the
value head (the net's forward returns (policy, value) by
default; threat logits are returned only when training/eval code
explicitly requests them).

**The improvement: v38 = v8's search + a multi-task value net
trained on v24's positions with v24's rollout-leaf value labels
PLUS exact threat-class labels.** Everything else is v8
unchanged (see the header). The net is consulted ONLY in quiet
positions (terminal cases stay exact, as in v8), so training
data is quiet positions only — and the threat classes below are
defined on quiet positions.

### 1.1 The auxiliary task (EXACT — pre-registered)

- **Input:** a quiet position + side to move, as the same
  8-plane encoding the value head sees.
- **Label function** (`gomoku_ml.threat.threat_class(board,
  player) -> int`), computable from the board alone via the
  existing, tested pattern functions — NO teacher, NO rollout:
  let M = max over candidate moves m of the number of winning
  moves the side to move holds after playing m (candidates from
  `scoring._candidate_moves`, winning-move counts from
  `scoring._winning_moves` after `game.apply_move`).
  - **Class 2 "forcing": M ≥ 2** — the side to move can create
    an open four (two winning moves next ply: unstoppable).
  - **Class 1 "threatening": M == 1** — the side to move can
    create a single four (forces a block).
  - **Class 0 "quiet": M == 0** — no move creates a four.
- **Pre-registered limitation:** a move creating two open
  threes creates no four in one move, so it is class 0/1 by
  this definition — the auxiliary task teaches FOUR-based
  threat structure; double-threat setups must still be learned
  from the value labels. This is a deliberate scope choice
  (exact labels from existing functions only, no new pattern
  code to validate); the v31 failure it targets IS a
  four-pattern overvaluation ((5,4) creates an immediate
  four-threat).
- **Pre-registered contingency (decided on the class
  distribution alone, before any training):** if any class is
  <2% of train positions, merge classes 1+2 into a binary
  {threatening (M≥1), quiet (M=0)} scheme and report the merge.
  Otherwise the 3-class scheme stands.
- **The threat head is TRAINING-ONLY** (see the header): it
  never influences a move choice. This is NOT a prior
  replacement (v26's closed lever) — the v1 prior is untouched
  and the threat head has no inference path into the search.

### 1.2 Loss and checkpoint selection (pre-registered)

- **Loss:** `1.0 × value MSE + λ × threat CE`, with **λ = 0.5
  FIXED** (no schedule). Policy weight 0.0 (as v24 — the policy
  head is unused at inference). Rationale for λ=0.5: the threat
  task must be strong enough to shape the shared trunk (that is
  the whole point of the auxiliary task) while staying
  subordinate to the value task in the loss; 0.5 puts the two
  terms in the same order of magnitude (threat CE ~0.3–1.1,
  value MSE ~0.06).
- **Checkpoint selection: best by VALUE val loss** (the
  inference-relevant metric). Threat accuracy is a reported
  diagnostic, NOT a selection criterion — selecting on the
  total loss would let the larger-magnitude threat term
  dominate the choice of the value net.

### 1.3 Architecture (pre-registered)

- **Trunk:** v8's ResNet 16ch/2 blocks (126,586 params before
  the extra head) — the SAME size as v8's and v24's nets, so
  inference cost is identical to v8's and the comparison
  isolates the training signal, not the capacity. (v31: 4x
  capacity played worse — scale is a CLOSED lever; it is not
  re-opened here.)
- **Value head: UNCHANGED in form** (`_PolicyValueHeads`'
  value branch, verbatim structure). Policy head likewise
  unchanged (and unused, policy_weight 0.0).
- **Threat head (NEW, on the shared trunk):** 1×1 conv
  channels→1 + BatchNorm + ReLU, flatten 225, FC 225→64 +
  ReLU, FC 64→3 logits — mirroring the value head's shape.
- **Module:** `GomokuMultiTaskNet` in `model.py`, registered
  as architecture `"resnet-multitask"` (params: channels 16,
  blocks 2, threat_classes 3). `forward(x)` returns
  `(policy, value)` (the existing I/O contract — ValueMCTS
  unpacks a 2-tuple unchanged); `forward(x,
  return_threat=True)` returns `(policy, value,
  threat_logits)` for training/eval.
- **Trained from scratch** (random init — NOT from v8's or
  v24's checkpoint; the labels are the same target as v24's
  but the head set differs).

### 1.4 Dataset (pre-registered)

- **Source: v24's engine-game distill data, positions
  IDENTICAL to v24's** — this isolates the multi-task change
  against v24 (same positions, same value labels; the ONLY
  difference is the added threat labels + the threat head):
  - v7 self-play (50 sims): 488 games → 19,545 train / 2,050
    holdout quiet positions.
  - v18 self-play (50 sims): 229 games → 9,142 train / 790
    holdout quiet positions.
  - Random playouts: 100 games → 5,162 train / 538 holdout
    quiet positions.
  - **Total: 33,849 train / 3,378 holdout quiet positions**
    (817 games). Counts verified from the shards at copy time
    (§2).
- **Value labels:** v24's rollout-leaf labels, REUSED VERBATIM
  from v24's shards (`gomoku-ml1-h1/ml/runs/ml1-h1/data/`,
  copied to `ml/runs/ml3-h3/data/`) — NOT re-generated, so
  there is zero label drift vs v24.
- **Threat labels:** computed from the SAME positions by the
  §1.1 function, from boards reconstructed from the stored
  planes (planes 0/1 = own/opponent stones, plane 3 = side to
  move). Stored as a `threat` array in the copied shards.
- **Holdout (game-level, v24's rule):** every 10th game of
  each source, routed to `*-holdout` dirs and NEVER trained on
  — the SAME holdout games as v24, so v38's holdout numbers
  are directly comparable to v24's (pooled 0.7766) and v31's
  (mixed 0.7929).
- **NOT used:** B1 data (B1 collapses White — v25, v31; the
  Black-skewed-data hypothesis has two supporting data points
  and v24's engine-only data did not collapse White). NO new
  game generation is needed — the games exist.
- **Verification (before training):**
  (a) quiet-filter consistency check — every copied position
  must satisfy `_terminal_leaf_value == 0.0` after board
  reconstruction from planes (validates the reconstruction;
  a wrong reconstruction would fail this on many positions);
  (b) spot re-labelling — re-run the ported label phase on 20
  games (all three sources) and confirm the regenerated value
  labels match v24's shards exactly (validates that master's
  rollout-leaf teacher is the same function v24 distilled).

### 1.5 Training recipe (pre-registered — v24's exactly, plus λ)

- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes, as v24), augmentation on (dihedral — threat
  labels are symmetry-invariant, no augmentation change),
  val_fraction 0.05 (random split within the train set, for
  best-checkpoint selection only), seed 0.
- **Memmap path (mandatory):** the train shards are bundled
  with `ml/scripts/build_mmap_bundle.py` (threat-aware
  extension) into `ml/runs/ml3-h3/data-mmap/` and training
  reads the bundle (`load_shards` memmap branch + lazy-index
  `train()`). The in-RAM load OOM-killed at this scale
  (iter-25 §2.4); do not train from the raw shards.
- **Config:** `ml/configs/ml3-h3.yaml`; run dir
  `ml/runs/ml3-h3-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, B1's finding).
- **Checkpoint selection (pre-registered rule, §1.2):**
  best-by-VALUE-val-loss (`best.pt`), copied to
  `ml/runs/ml3-h3/value-net.pt` (gitignored data, the path
  v38's registry entry points at). `checkpoint_every_epochs:
  1` for crash recovery.
- **Resume rule (pre-registered, v24's lesson):** if training
  is killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans
  the full 15 via `schedule_epochs`); back up the pre-kill
  `best.pt` first, because the resumed run resets
  `best_val_loss` and would overwrite it at the next epoch
  regardless — the final best is chosen by comparing the
  resumed run's best VALUE val loss against the pre-kill best.

### 1.6 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.5, 1.0, 2.0}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 38`. The grid brackets
v24's winner (1.0) on both sides: the multi-task net's values
should be calibrated like v24's (same labels, same recipe), so
1.0 is the prior favourite and 0.5/2.0 cover both directions.
Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure). **No-signal
rule:** if all three arms are within one win of each other,
pick α=1.0 (v24's winner) and say so — the sweep found no
signal. The sweep SELECTS α; the 100-game battery is the
verdict (standing lesson: small-sample smokes mislead — v24's
0.65 smoke at α=1.0 did not reproduce at 100 games; v31's 0.70
dev-seed fast loop became 0.40 at validation seeds).

### 1.7 What would change the design

- If the multi-task net's holdout VALUE sign-accuracy on its
  OWN (engine) holdout is at chance (~0.50), the distillation
  failed to learn the leaf function — stop and report, do not
  battery a chance-level net.
- If the threat head's holdout accuracy is at the
  majority-class baseline, the auxiliary task failed to learn —
  report it, but PROCEED (the value head is the verdict; the
  auxiliary task is a means, not the end).
- If the α sweep's best is within noise of the worst (all three
  within one win of each other), pick α=1.0 per the §1.6
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain re-runs the α
  sweep and the battery from scratch on the new form.

### 1.8 Holdout diagnostics (pre-registered)

All on v24's engine-game holdout (3,378 positions, game-level,
never trained on), targets = rollout-leaf labels:
- (a) **v38 net value sign accuracy + MSE** (full planes and
  no-history views) via `eval_value_holdout.py` — compare
  with v24's pooled 0.7766 and v31-mixed's 0.7929 on the same
  positions. Also **by side to move** (plane 3 splits Black /
  White to move) — B1-data versions collapse White; v38 uses
  engine-only data, but the split is measured anyway.
- (b) **v38 threat-class accuracy** (overall, per class, and
  by side to move) via the new `eval_threat_holdout.py` —
  full planes and no-history views. Majority-class baseline
  reported alongside.
- (c) **v24's net** (`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`,
  by absolute path) **on the same holdout** — reproduces
  v24's published numbers (sanity check on the eval setup).
- (d) **v8's net on the same holdout** — baseline (expect
  ~0.50, as on v24's labels).
No decision hinges on (c)–(d); they diagnose whether the
multi-task training changed the value fit and whether the
threat head learned anything.

### 1.9 The g92 check (pre-registered)

The task's named diagnostic: on tactical position
`setup-v1v0s-g92` (v26/v31's failure), report the net's value
for the prior's top move (9,6) vs the chosen move, and whether
the search picks (9,6). v31's numbers for comparison: prior
ranks (9,6) 1st (0.1122); v31's net values it −0.377 (1 visit)
vs (5,4)'s +0.788 (75 visits); v8's net values (9,6) +0.246
(57 visits) and passes. The multi-task hypothesis predicts
v38's net values (9,6) ABOVE (5,4) (or at least above v31's
−0.377) — the check is reported with the numbers either way.

### 1.10 Eval seeds (pre-registered)

h2h vs v8: v38=100 / v8=200; h2h vs v0: v38=100 / v0=200; h2h vs
v7: v38=100 / v7=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge; tactical `--seed 100`.

### 1.11 Eval order (pre-registered)

(1) Data copy + threat labelling (§2) + verification (§1.4).
(2) Memmap bundle + training (§3) + holdout diagnostics (§4).
(3) α sweep (§5) — registry updated to the winner. (4) FAST
LOOP (§5): up to 4 iterations of 10 games vs v8 AND 10 games
vs v0. (5) FULL VALIDATION BATTERY on the final v38 (§6):
tactical re-run, h2h vs v8 100 games, 100 games vs v0, 100
games vs v7, 100-game swap match vs v7, ladder 20
games/opponent, clause-4 suite re-run. Any battery numbers
measured before the fast loop completes are INTERMEDIATE,
labelled as such; only the final battery counts for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games
fast_loop_match.py --version-a 38 --opponent v8 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v38-vs-v8.json
# Clause 2: ladder (20 games/opponent, mcts_sims 100)
eval_version.py --version 38 --ladder-games 20 --out research/iterations/ladder-v38.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 38 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, swap vs v7 (balanced/tempo, placement_seed 11, seeds 100/200)
fast_loop_match.py --version-a 38 --opponent v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v38-vs-v7.json
fast_loop_match.py --version-a 38 --opponent v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v38-vs-v0.json
selfplay_batch.py --engine-a v38 --engine-b v7 --games 100 --swap-rule --placement-seed 11 --out research/iterations/batch-v38-vs-v7-swap.json
```

## 2. Data (actuals)

**Threat labelling COMPLETE 2026-11-18** (`ml/scripts/
label_threat.py`, logs `ml/runs/ml3-h3/label-threat*.log`).
Two interruptions, both recovered by the script's resume-safe
skip: the run was killed once deliberately (to deploy the
threat-labelling speedup below) and once by a VM replacement at
random 46/90 — no shard was lost or double-written (output
shards are skipped on re-run; the per-run LABEL_JSONs cover
only the newly written shards, so the authoritative totals are
computed from the shards below).

**Speedup (exact, verified):** `threat.max_winning_moves_after`
was restricted to candidates within Chebyshev distance 2 of a
player stone when the player holds no winning move (all quiet
positions — the whole training set; proof in the docstring).
Verified 0 mismatches on 1,553 already-labelled positions
against the unoptimised labels, and the 8 unit tests pass;
interleaved A/B on 160 positions: 30–37 ms/pos vs 37–68 ms/pos
(1.2–1.8x; candidates 61 → 43 per position).

**Counts (from the shards — identical to v24's published
numbers in every cell):**

| Source | Games | Quiet positions | ±1 labels | Threat classes [quiet, threatening, forcing] |
|---|---|---|---|---|
| v7 self-play (train) | 440 | 19,545 | 746 (3.8%) | [16,027, 3,095, 423] |
| v18 self-play (train) | 207 | 9,142 | 453 (5.0%) | [7,333, 1,605, 204] |
| Random playouts (train) | 90 | 5,162 | 1,470 (28.5%) | [2,910, 967, 1,285] |
| **Train total** | **737** | **33,849** | **2,669** | **[26,270, 5,667, 1,912] = 77.6% / 16.7% / 5.6%** |
| v7-holdout | 48 | 2,050 | 78 | [1,711, 293, 46] |
| v18-holdout | 22 | 790 | 51 | [628, 140, 22] |
| random-holdout | 10 | 538 | 156 | [308, 85, 145] |
| **Holdout total** | **80** | **3,378** | **285** | **[2,647, 518, 213] = 78.4% / 15.3% / 6.3%** |

**Merge contingency (§1.1): NOT triggered.** Every class is
above 2% of train positions (forcing is 5.6% overall — 2.2%
on engine positions, 24.9% on random playouts, which are
tactically wild). The 3-class scheme stands. Note the
distribution skew the threat head must handle: random
positions are 25% forcing vs 2.2% for engine positions, so
the head sees most of its class-2 examples from the random
source (1,285 of 1,912).

**Verification (a) — quiet-filter consistency: PASS.** 0
non-quiet positions across all 37,227 positions in all three
runs: the board reconstruction from planes is correct.

**Verification (b) — spot re-labelling: PASS.** The ported
label phase re-ran on 20 games (7 v7 + 7 v18 + 6 random, 888
quiet positions); the regenerated value labels AND planes
match v24's shards EXACTLY (0 mismatched shards). Master's
rollout-leaf teacher is the same function v24 distilled.

**Memmap bundle:** `ml/runs/ml3-h3/data-mmap/` (737 shards,
33,849 samples; planes/policy/values/threats). Verified on
load: threat distribution [26,270, 5,667, 1,912] exact, and
the first sample's planes/value/threat match its source shard.
Holdout shards are NOT in the bundle (never trained on).

## 3. Training (actuals)

**Training COMPLETE 2026-11-18 ~15:06 UTC** (15 epochs, 7,545
steps, ~35 min wall, NO interruptions): config
`ml/configs/ml3-h3.yaml`, run dir `ml/runs/ml3-h3-train/`,
memmap bundle. Log `ml/runs/ml3-h3/train.log`. RSS stayed low
(the memmap path works). Per-epoch checkpoints on; no resume
was needed.

| Epoch | Train value loss | Val value loss | Train threat loss / acc | Val threat loss / acc |
|---|---|---|---|---|
| 1 | 0.0776 | 0.0876 | 0.5345 / 0.7796 | 0.5114 / 0.7884 |
| 4 | 0.0689 | 0.0858 | 0.4350 / 0.8107 | — / 0.8032 |
| 8 | 0.0625 | 0.0790 | 0.3793 / 0.8325 | — / 0.8327 |
| 12 | 0.0573 | 0.0721 | 0.3479 / 0.8480 | — / 0.8316 |
| 14 | 0.0555 | **0.0720** | 0.3386 / 0.8514 | — / 0.8381 |
| 15 | 0.0561 | 0.0722 | 0.3398 / 0.8506 | — / 0.8410 |

**Best checkpoint: epoch 14** (`best.pt`, step 7042, val VALUE
loss 0.0720 — the pre-registered selection metric, §1.2; epoch
15's 0.0722 is slightly worse, so `best.pt` ≠ `final.pt`),
copied to the registry path **`ml/runs/ml3-h3/value-net.pt`**
(md5 `7c3fade60d87a85037cf81cd4bdebd10`; all tensors finite).
Val value loss decreased monotonically until epoch 14 — no
overfitting by the selection metric, but the pre-registered
plan is 15 epochs and the battery is the verdict, so no
extension.

**Comparison with v24 (same data, same val split — seed 0 over
the same 33,849 samples, so the numbers are directly
comparable):** v38's train value loss (0.0555) matches v24's
(0.0567), but its VAL value loss is worse (0.0720 vs v24's
0.0617). The multi-task training costs ~17% relative val MSE
on the value task — the threat gradient trades against value
generalisation. Whether that cost shows up in sign accuracy
(the played-strength-relevant measure) is §4's question.

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th game, never trained on),
targets = rollout-leaf labels. v38 net =
`ml/runs/ml3-h3/value-net.pt`; v24 net =
`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml3-h3/holdout-{v7,v18,random}.txt` (value) and the
threat script's JSON. Sanity check: this run reproduces v24's
published numbers EXACTLY on its own holdout (v7 0.7713, v18
0.8384, random 0.7055, pooled 0.7766) and v8's (0.523 / 0.501
/ 0.463) — the eval setup is sound.

### 4.1 Value head — sign accuracy (full planes)

| Holdout | n pos (non-draw) | v38 sign | v38 MSE | v38 mean\|v\| | v24 sign | v8 sign |
|---|---|---|---|---|---|---|
| v7 | 2,050 (1,845) | 0.7669 | 0.0397 | 0.083 | 0.7713 | 0.523 |
| v18 | 790 (761) | 0.8344 | 0.0616 | 0.092 | 0.8384 | 0.501 |
| random | 538 (523) | 0.7132 | 0.1713 | 0.254 | 0.7055 | 0.463 |
| **pooled** | **3,378 (3,129)** | **0.7744** | — | — | **0.7766** | — |

**Verdict: the distillation learned the leaf function — the
§1.7 stop condition (chance-level holdout) does NOT trigger.**
v38's pooled full-planes sign (0.7744) is statistically
identical to v24's (0.7766) — the multi-task training did NOT
hurt the value fit in-distribution, despite the worse val MSE
(§3: the MSE gap is magnitude calibration, not sign). v38's
MSE is LOWER than v24's on every source (0.040 vs 0.045, 0.062
vs 0.069, 0.171 vs 0.238) — the multi-task net is
better-calibrated in magnitude where it trained.

### 4.2 Value head — the history-dependence problem (no-history view)

| Holdout | v38 sign (no-hist) | v38 mean\|v\| (no-hist) | v24 sign (no-hist) |
|---|---|---|---|
| v7 | **0.5035** (chance) | 0.468 | 0.7713 |
| v18 | 0.5742 | 0.324 | 0.8344 |
| random | 0.6998 | 0.237 | 0.5468 |
| **pooled** | **0.5536** | — | **0.7491** |

v38 is catastrophically history-dependent on ENGINE positions:
without history planes its value sign falls to chance on v7
(0.5035) with mean|v| ballooning to 0.47 — the same pathology
v31's B1-only net showed (engine no-history 0.27–0.31). v24 is
history-robust (0.7713/0.8344 both views). On random positions
the pattern INVERTS (v38 0.6998 vs v24 0.5468). At inference
the leaves get search-path history (neither the real-game
history of the full-planes view nor the zeroed history of the
no-history view), so the operating point lies between the two
views — but on engine positions both bounds are worse than
v24's. **This foreshadows that v38 may play worse than v24
despite equal full-planes holdout** — the battery is the
verdict, and §4.3 explains the mechanism.

### 4.3 Threat head — accuracy (full planes)

| Holdout | Threat acc | Majority baseline | Per-class acc (quiet / threatening / forcing) |
|---|---|---|---|
| v7 | 0.859 | 0.835 | 0.960 / 0.403 / **0.000** (0/46) |
| v18 | 0.846 | 0.795 | 0.959 / 0.471 / **0.000** (0/22) |
| random | 0.723 | 0.573 | 0.916 / 0.082 / 0.690 |
| **pooled** | **0.834** | **0.784** | — |

**The auxiliary task PARTIALLY learned — the §1.7
majority-baseline stop does NOT trigger (0.834 vs 0.784), but
the forcing class is the story:** on engine holdouts the head
gets 0% of forcing positions right — it never predicts
forcing there. On random positions it gets 69% of forcing
right. Mechanism: 1,285 of 1,912 forcing training examples
(67%) come from random playouts, which look different
(tactically wild, long games); the head learned "forcing =
looks like a random position", not the general four-creator
concept. On engine positions (2.2% forcing) it calls them
quiet. The threatening class is middling on engine (0.40–0.47)
and near-zero on random (0.08 — random threatening positions
are mostly called forcing, since random positions are 25%
forcing). The threat signal the trunk learned is
distribution-specific, not the general threat concept the
hypothesis needs.

**No-history view:** threat acc collapses (v7 0.558 — BELOW
the 0.835 majority baseline; v18 0.70 vs 0.795; random 0.617
vs 0.573). The threat head leans on the last-move/history
planes as a shortcut (the last move often IS the
threat-creating move) — and the shared trunk routes threat
information through history features, which is exactly the
mechanism behind the value head's history dependence (§4.2).
The auxiliary task taught the trunk to use history, not to
read threats from the board.

### 4.4 By side to move (full planes)

| Holdout | Threat acc Black / White | Value sign Black / White |
|---|---|---|
| v7 | 0.862 / 0.856 | 0.739 / 0.794 |
| v18 | 0.841 / 0.850 | 0.866 / 0.805 |
| random | 0.639 / 0.807 | 0.606 / 0.817 |

No uniform White collapse (unlike the B1-data versions v25/v31)
— the engine-only data does not collapse White, as expected.
The weak cells are random-Black (threat 0.639, value sign
0.606) — random positions with Black to move are the hardest
for both heads. Value sign by side is roughly balanced on
engine positions.

## 5. α sweep and fast loops

### 5.0 The g92 check (§1.9) — MULTI-TASK DID NOT FIX THE OVERRIDE

`ml/scripts/diag_g92.py` (100 sims, seed 0; the diagnostic
reproduces iter-31's v8 numbers exactly — (9,6) +0.246 at 57
visits, (5,4) +0.056 at 4 visits — so the instrument is
validated):

| Engine | (9,6) expected: visits / our-q | (5,4) wrong: visits / our-q | Raw net after (9,6) (White to move) | Picks |
|---|---|---|---|---|
| **v38** (α=1.0) | 2 / **−0.062** | 93 / **+0.729** | +0.060 (wrong sign) | (5,4) ✗ |
| v31 (iter-31) | 1 / −0.377 | 75 / +0.788 | — | (5,4) ✗ |
| v8 (α=0.25) | 57 / +0.246 | 4 / +0.056 | −0.545 | (9,6) ✓ |
| v24 (α=1.0) | 56 / +0.461 | 5 / +0.031 | −0.121 | (9,6) ✓ |

The prior ranks (9,6) 1st (0.1122) for all engines. v38
reproduces the v26/v31 failure mode UNMITIGATED: the net
values the quiet setup at −0.062 ours (wrong-signed, near
zero) while the (5,4) line gets +0.729 ours, and the search
picks (5,4) 93–2. v38 is less wrong than v31 on (9,6)
(−0.062 vs −0.377) but equally wrong on (5,4) (+0.729 vs
+0.788). Note v38 is WORSE than v24's single-task net on BOTH
moves (v24: (9,6) +0.461, (5,4) +0.031) — the threat head
appears to have AMPLIFIED the threat-overvaluation rather
than correcting it: the trunk now represents threats more
strongly, and the value head leans on that representation.
(The (5,4) line's immediate leaf is the +0.250 must-block
short-circuit for all engines — the +0.729 comes from deeper
threat-heavy leaves the net overvalues.) The hypothesis's
core prediction — v38's net values (9,6) ABOVE (5,4) — is
REFUTED at the named diagnostic position. The battery
proceeds as pre-registered (the g92 check is diagnostic, not
a stop rule); clause 3 is now expected to fail on this
position, and the fast loop will look for whether any
patchable mechanism remains.

### 5.1 α sweep — winner α = 0.5 (2026-11-18)

Grid {0.5, 1.0, 2.0} × 20 games vs v8, seeds 1001/2001,
per-game seeding. Interrupted once by a VM replacement at
α=2.0 game 2; resumed cleanly (α=0.5 and 1.0 already
complete, α=2.0 resumed from game 2 with per-game seeding
reproducing the uninterrupted run). JSONs:
`research/iterations/alpha-v38-{0.5,1.0,2.0}-vs-v8.json`.

| α | Record | Win rate | Wilson 95% CI |
|---|---|---|---|
| **0.5** | 12W-8L-0D | 0.600 | [0.387, 0.781] |
| 1.0 | 9W-10L-1D | 0.475 | [0.279, 0.680] |
| 2.0 | 5W-15L-0D | 0.250 | [0.112, 0.469] |

**Winner: α = 0.5** (pre-registered rule: best win rate;
the no-signal rule does not apply — 12 vs 9 vs 5 wins is a
clear spread). Registry `value_weight` updated 1.0 → 0.5.
The pattern is monotone DECREASING in α — the more the
multi-task net's value is trusted, the worse v38 plays, and
at α=2.0 it is significantly BELOW 0.50 (CI upper 0.469).
Consistent with §5.0: the multi-task net's value is actively
harmful at full weight (it overvalues threats); halving the
weight lets the v1 prior carry more of the decision. For
comparison v24's sweep winner was α=1.0 at 0.65, and v31's
was α=0.5 at 0.70 — v38's 0.60 at its best α is the weakest
of the three distilled versions on dev seeds, and dev seeds
flatter (v31: 0.70 dev → 0.40 validation).

### 5.2 Fast loop, iteration 1 (seeds 1001/2001) — NO PATCHABLE CLASS FOUND; LOOP STOPS

vs v8: the α=0.5 sweep leg (20 games, same seeds, same α)
IS the fast-loop leg — re-playing it would reproduce the
first 10 games exactly under per-game seeding, so the sweep
JSON was analysed directly. vs v0: 10 fresh games via
`fast_loop_match.py` (`fast-v38-vs-v0-k1.json`).

| Leg | Record | Win rate | As Black | As White |
|---|---|---|---|---|
| vs v8 (α=0.5) | 12W-8L-0D | 0.60 | 8W-2L | 4W-6L |
| vs v0 | 4W-6L-0D | 0.40 | 2W-3L | 2W-3L |

**v38 LOSES to v0 (0.40) on dev seeds** — the no-search
heuristic anchor. No White collapse this time (2-3 as Black,
2-3 as White; the v8 leg is White-heavy instead: 6 of the 8
losses as White). All 10 v0 games are unique.

Loss classification (`classify_losses.py`), both legs:

| Leg | Losses | Via open four | Missed blocks | Clean missed preventions | Threatened turns |
|---|---|---|---|---|---|
| vs v8 | 8 | 8 | 0 | 0 | 33 |
| vs v0 | 6 | 6 | 0 | 0 | 38 |

Every loss in both legs is a positional open-four loss with
ZERO missed blocks and ZERO clean missed preventions — v38
never blunders a block and never misses a prevention that a
single candidate move would have supplied. Two losses read
in full (v8 leg game 7, 19 plies; game 3, 31 plies): White
lets Black build a column/row into an open four over several
moves, playing its own threats (7,9; 5,4) instead of
preventing — the same "overvalues own threats, undervalues
quiet/preventive moves" failure the auxiliary task was
supposed to fix. The classifier's zero on clean missed
preventions means there is no single-move prevention patch:
at every threatened turn, either v38's move was forcing/safe
or no safe alternative existed among the candidates. The
failure is the net's value judgment, which only a retrain
could change — and a retrain with a redesigned auxiliary
task (history dropout, class-balanced forcing examples) is a
NEW version, not a v38 patch. The α grid is NOT extended
post-hoc (the 0.5-vs-1.0 gap is 3 wins in 20 games with
heavily overlapping CIs — only the α=2.0 harm is
significant). **Per the stop rule, the fast loop ends after
iteration 1: no new patchable weakness class.** The battery
proceeds on the final v38 (α=0.5).

## 6. Validation battery

TODO — clauses 1–4 + secondaries.

## 7. Verdict

TODO.

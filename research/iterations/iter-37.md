# Iteration 37 — v37: deeper-teacher distillation (ML batch 3, H2)

**Status: IN PROGRESS — pre-registered; labelling running.**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H2 workstream of ML batch 3 (H1 is v36,
v27's threshold ported to the value engine; H3 is v38, the
multi-task value net); work happens on branch
`research/ml-batch-3-h2` in the `gomoku-ml3-h2` worktree. v37
differs from v8 ONLY in the value net's weights (and the swept
α) — same PUCT search, same v1 priors, same sims (100), same
c_puct (1.5), same threat check, same terminal cases — so the
battery is a clean A/B of the deeper-teacher net against v8's
weak-self-play net (and, less cleanly, against v24's small
distilled net: v37 changes ONLY the labels, on identical
positions — see §1.1 — so the v24 comparison isolates the
labelling change).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v8. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v37 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Ports and cost measurements (before pre-registration)

**Ports (ML batches 1–2 were never merged to master, so their
code was ported into this worktree — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: ML batch 1 H2's memmap version
  (pre-allocated `load_shards` + memmap bundle support + lazy-index
  `train()`), copied VERBATIM from `gomoku-ml1-h2` (sha256
  `7e8cc28d…` matches the source and iter-31's record).
- `ml/scripts/generate_distill_data.py`: v24's games+label
  pipeline, copied VERBATIM from `gomoku-ml1-h1` (sha256
  `1ce8e6d3…` matches) — then extended with `--label-mode
  rollout2` (see §1.1; the diff from the verbatim port is 5 small
  hunks: docstring, label-fn selection, one call site, LABEL_JSON
  field, CLI flag; the default `rollout` reproduces v24's labels
  exactly).
- `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/tests/test_load_shards.py`: copied VERBATIM from
  `gomoku-ml1-h2` (sha256 `92664381…`, `2b7930bc…`, `1844f241…`
  all match iter-31's record).
- `ml/scripts/sweep_alpha.py`: v31's parameterised version,
  copied VERBATIM from `gomoku-ml2-h2` (sha256 `a2b2c0c7…`).
- `ml/scripts/find_losing_moves.py`: copied VERBATIM from
  `gomoku-ml1` (sha256 `24814e64…`).
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port (ML batch 1 H1's change, applied by hand to this tree's
  file — the SAME 3-hunk shape as v24's, byte-identical hunks:
  docstring, `value_weight: float | None = None` parameter on
  `ValueMCTS.__init__`, `self.value_weight` assignment, leaf
  shrink via `self.value_weight`); default None → v8's 0.25, so
  v8 is byte-identical in behaviour. **NOTE for the coordinator:
  this DOES touch `ValueMCTS.__init__`'s signature** (it must —
  the α sweep needs a per-instance weight); the hunk is identical
  to v24's/v31's, so a merge against a sibling applying the same
  hunk is a clean same-change merge — but CHECK for a duplicate
  parameter after merging (the silent auto-merge breakage has
  recurred twice on this exact signature).
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS` (v24's exact hunk), plus the v37 registry entry
  (`value_checkpoint: ml/runs/ml3-h2/value-net.pt`,
  `value_weight: 1.0` placeholder until the §1.3 sweep sets it).
- The rollout-leaf teacher (v18 code) is on master since the
  batch-9 merge — no cherry-picking needed, and no re-verification
  of the leaf port is needed either (v24's 401/401 bit-for-bit
  check stands; the teacher code is the same master code v24
  verified against).
- `ml/baseline.py` is FROZEN and untouched.

**Port smoke checks:** v8 loads and plays in this worktree
(checkpoint copied from the main repo, md5
`5deb26f3df67883f24f04800d5356da6`); v37 is registered;
targeted tests green (see §6.4).

**Cost measurements** (run under sibling contention, so means
are inflated — medians are the planning numbers):
- 2-ply label (`_rollout2_leaf_value`) on 298 quiet positions
  from 10 v7 games: **median 11.15 ms, mean 28.17 ms**
  (p90 72.61 ms) vs the 1-ply label's mean 17.50 ms — only
  **1.61x** the 1-ply cost. Projected labelling for the full
  v24 dataset (37,227 positions): **~17.5 min** (mean-based;
  ~7 min median-based) — NOT the bottleneck. No dataset-size
  compromise is needed: v37 labels v24's FULL position set.
- The 2-ply labels are a genuinely different target from the
  1-ply labels on the same 298 positions: mean |Δ| 0.1039,
  median 0.0462, p90 0.1586, max 1.1771; sign agreement only
  0.7550; Pearson r 0.7188. Mean|v| 0.2073 vs 0.1564; ±1
  fraction 0.1107 vs 0.0604 — the 2-ply labels are "hotter"
  (more terminal mass), which the α sweep must absorb (§1.3).
  Resolution: ply-1 terminal 0/298, ply-2 terminal 10/298
  (3.4%), leaf eval 288/298 — nearly all labels come from the
  leaf at the 2-ply-ahead position.
- Minimax-reply 2-ply label (opponent picks the reply
  minimising our leaf value over all candidates): **~1.3–2.0 s
  per position** (~50x the rollout-policy version) — full
  dataset ≈ 15 h, infeasible under contention; REJECTED (see
  the g92 diagnostic below — it buys no better discrimination
  at the motivating position either).

**The setup-v1v0s-g92 diagnostic (the motivating position,
measured BEFORE pre-registration):** v31's net valued the quiet
setup (9,6) at our-q −0.377 and the wrong immediate-threat move
(5,4) at +0.788, while v8's net valued (9,6) at +0.246. What do
the TEACHERS say?
- At the root (Black to move): the 1-ply label is **+0.1290**
  (a small gradient — the payoff is 4 plies out, beyond the
  2-ply rollout); the 2-ply label is **+1.0000** (the 4-ply
  horizon reaches the forced win). The teacher plays the setup
  (9,6) on ply 1. THIS is the mechanism in miniature: the 1-ply
  label of a pre-setup position is a faint gradient hint, the
  2-ply label is the win itself.
- At the children: the 1-ply teacher ranks (9,6) +1.000 above
  (5,4) +0.250 (the must-block value — (5,4) creates one
  immediate threat); the 2-ply teacher ranks them EQUAL, both
  +1.000 — because (5,4) does NOT lose the win within the
  teacher's horizon, it only wins slower (after (5,4), White's
  unique refuting reply (4,3) — prior rank 1, the rollout
  policy's move — Black still plays (9,6) and the win
  materialises). The minimax-reply variant gives the SAME
  +1.000/+1.000 split: the equality is not a move-selection
  failure, it is the position's content.
- Honest consequence, pre-registered: the 2-ply labels do NOT
  improve child-level discrimination at g92 — they teach the
  win at the PRE-setup position instead. For the tactical suite,
  v37's net must NOT invert the order (as v31's did); if it
  values (9,6) ≈ (5,4), the v1 prior (which ranks (9,6) FIRST,
  0.1122) breaks the tie correctly. The clause-3 run of
  setup-v1v0s-g92 is the pre-registered mechanism check (§6.1).

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, tactical 20/20 + 29/29 on the CORRECTED suite — re-verified
by the ML batch 2 coordinator at seed 100; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical ≥ v8's (old suite 20/20,
hard suite 29/29, run at `--seed 100` on the corrected suite),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v37 vs v7 (100
games), v37 vs v0 (100 games), and a 100-game swap match vs v7
under the tempo judge (the harness default). Non-transitivity is
recorded prominently.

**The hypothesis (plan H2):** v24 proved distillation works as ML
— the small net learned the rollout leaf (pooled in-distribution
holdout sign accuracy 0.7766 vs v8's 0.504 on the same labels)
— but the distilled net overvalues immediate threat patterns and
undervalues quiet 2-ply setups (v31's verified failure at
setup-v1v0s-g92: the prior ranks the setup first, the value head
overrides it, −0.377 vs +0.788). The mechanism: a quiet setup's
payoff materialises 4 plies from the pre-setup position (setup
move, opponent reply, open-four creation, forced win) — beyond
the 1-ply label's 2-ply rollout horizon — so the 1-ply label of a
pre-setup position is only a faint gradient (+0.1290 at g92's
root) and the net never learns "quiet + setup available =
winning" as a direct mapping. Labelling with 2-ply values — the
teacher plays 2 plies before evaluating, a 4-ply horizon — puts
the payoff IN the label (+1.0000 at g92's root) and should teach
the quiet setups directly. If the deeper labels fix the
quiet-setup blindness without breaking anything else, v37 beats
v8; if the fit improves but strength does not (v31's lesson:
holdout transfer does not predict played strength), the ceiling
is the leaf function / search horizon, not the label depth — and
that is itself a result.

**The improvement: v37 = v8's search + a value net trained to
predict v18's rollout-leaf value evaluated 2 plies ahead.**
Everything else is v8 unchanged (see the header). The net is
consulted ONLY in quiet positions (terminal cases stay exact, as
in v8), so training data is quiet positions only.

### 1.1 The 2-ply labelling procedure (pre-registered, exact)

The teacher is **v18's rollout leaf** (`_rollout_leaf_value`,
blend=False) — the SAME teacher as v24, for comparability. The
2-ply label for a quiet position (`board`, `player` to move) is
`_rollout2_leaf_value(board, player)`:

1. **Ply 1 — the teacher's move:** `m1, term1 =
   _rollout_move(board, player)` — the teacher's OWN move
   policy: win if possible; block a single opponent winning
   move (forced); else the argmax over empty cells of the
   player's v1 total score (own v1 attack + opponent's v1
   attack), with open-four creations taking priority (the
   iter-16 §2.2 patch); ties by lowest flat index —
   deterministic. If `term1` is not None, the label is `term1`
   (from `player`'s perspective: +1 the teacher wins, −1 the
   opponent holds ≥2 winning moves, 0.0 board full).
2. **Ply 2 — the opponent's reply:** `m2, term2 =
   _rollout_move(board+m1, -player)` — the SAME policy for the
   opponent. If `term2` is not None, the label is `−term2`
   (`term2` is from the opponent's perspective; negated to
   `player`'s — the same sign convention as
   `_rollout_leaf_value` at odd plies).
3. **Otherwise** the label is `_rollout_leaf_value(board+m1+m2,
   player)` — `player` to move again at the resulting position,
   blend=False (v24's setting).

**Perspective:** the side to move throughout, in [−1, 1],
exactly like v24's labels. **Horizon:** 2 plies of
rollout-policy play + the 2 plies inside `_rollout_leaf_value`
= 4 plies, vs v24's 2. **Why the rollout policy, not a
leaf-value argmax or a minimax reply:** (a) it IS the teacher's
move policy — the same policy `_rollout_leaf_value` uses for
its internal 2 plies — so the v37 label is exactly the 4-ply
extension of v24's 2-ply label, with no new move-selection
machinery; (b) maximising the noisy leaf over ~40 candidates
would inject optimizer's-curse bias into the LABELS (the label
would be the max of 40 noisy values, systematically optimistic);
(c) the minimax reply was measured (§0): ~1.5 s/position, 50x
the cost, with NO better discrimination at setup-v1v0s-g92.
**Known limitation (pre-registered):** the rollout-policy
reply is v1-greedy, not minimax — it can miss refutations, so
2-ply labels can overvalue moves that win slowly against weak
defence (at g92, (5,4) goes +0.250 → +1.000). The battery is
the arbiter.

### 1.2 Dataset (pre-registered)

- **Source: v24's engine-game positions, re-labelled.** NO new
  game generation: the games already exist (v24's §2) and game
  generation was the expensive step. The positions are EXACTLY
  v24's: the 817 games in `gomoku-ml1-h1/ml/runs/ml1-h1/
  games-{v7,v18,random}.json` (488 v7 self-play at 50 sims,
  seeds 50000+; 229 v18 self-play at 50 sims, seeds 60000+;
  100 random playouts, seeds 70000+), copied into this
  worktree's `ml/runs/ml3-h2/` (gitignored data). Expected
  yield: **37,227 quiet positions** (21,595 v7 / 9,932 v18 /
  5,700 random), the same positions v24 trained on — the ONLY
  change from v24 is the label function, so the v37-vs-v24
  comparison isolates the labelling change.
- **Sampling:** ALL quiet positions of every game (labelling is
  ~11 ms median per position; §0). A position is quiet iff
  `_terminal_leaf_value == 0.0`. Positions are encoded with the
  real game history (boards 2/4 plies back, last move) exactly
  as in training (`encoding.encode`). Policy targets are one-hot
  on the move actually played (unused at policy_weight=0).
- **Labels:** `_rollout2_leaf_value` (§1.1) via
  `generate_distill_data.py label --label-mode rollout2`.
  Rollout-found terminals (±1) are kept as-is: they are the
  label's real signal.
- **Holdout (game-level, v24's protocol, IDENTICAL split):**
  every 10th game of each source is routed to a `*-holdout`
  dir and NEVER trained on — the same games v24 held out (48
  v7 / 22 v18 / 10 random = 80 games, 3,378 positions), so
  v37's holdout numbers are comparable to v24's BY POSITION
  (only the labels differ). Expected train set ≈ 33,849
  positions.
- **NOT used for training:** the B1 10k games (B1 data collapses
  White — v25, v31 — and v24's engine-only data did not; this
  workstream does not re-open that lever). They are NOT used
  even as a diagnostic here (v24's B1-diag already established
  the transfer numbers; no decision hinges on re-measuring it).
- **Variety check:** not needed — the games are v24's, whose
  variety check passed (14 distinct first-5-ply prefixes in 15
  v7 games).
- **Stopping rule:** labelling is ~18 min for the full set (§0);
  no cutoff is needed. If a VM replacement kills the labelling
  run, the label phase is resume-safe (existing shards skipped)
  — re-run the same command.

### 1.3 Architecture and training (pre-registered)

- **Architecture:** v24's ResNet **16ch/2 blocks** (126,586
  params), trained from scratch (random init — NOT from v8's or
  v24's checkpoint; the labels are a different target). Do NOT
  go larger: v31 proved 4x capacity plays worse on distilled
  labels (the scale lever is CLOSED).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes — v24's recipe exactly), augmentation on
  (dihedral), val_fraction 0.05 (random split within the train
  set, for best-checkpoint selection only), seed 0.
- **Config:** `ml/configs/ml3-h2.yaml` (v24's `ml1-h1.yaml`
  with run name/dir changed); run dir `ml/runs/ml3-h2-train/`;
  threads pinned to 1 (`OMP_NUM_THREADS=1 TORCH_THREADS=1`,
  B1's finding). `checkpoint_every_epochs: 1` from the start
  (v24's resume lesson).
- **Memmap path (mandatory):** the train shards are bundled with
  `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml3-h2/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as v24):**
  best-by-internal-val-loss (`best.pt`), copied to
  `ml/runs/ml3-h2/value-net.pt` (gitignored data, the path v37's
  registry entry points at).
- **Resume rule (pre-registered, v24's lesson):** if training is
  killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans the
  full 15 via `schedule_epochs`); back up the pre-kill `best.pt`
  first, because the resumed run resets `best_val_loss` and
  would overwrite it at the next epoch regardless — the final
  best is chosen by comparing the resumed run's best val loss
  against the pre-kill best.

### 1.4 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.25, 0.5, 1.0}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 37`. The grid brackets
v24's winner (1.0) below: the 2-ply labels are hotter than
v24's (mean|v| ~0.21 vs ~0.06 on engine positions — §0), so
the net's values will be larger in magnitude and a smaller α
may win (v31's lesson: 3x larger values → α=0.5 beat 1.0).
Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure). **No-signal
rule:** if all three arms are within one win of each other,
pick α=1.0 (v24's winner) and say so — the sweep found no
signal. The sweep SELECTS α; the 100-game battery is the
verdict (standing lesson: small-sample smokes mislead — v24's
0.65 smoke at α=1.0 did not reproduce at 100 games; v31's 0.70
did not either). The registry entry's `value_weight` is updated
to the winner after the sweep. (Note: α is an inference-time
weight, so holdout sign accuracy cannot select it — the sweep
is by played win rate, as in v24/v31.)

### 1.5 What would change the design

- If the distilled net's holdout sign-accuracy on its OWN
  (2-ply-label) holdout is at chance (~0.50), the distillation
  failed to learn the deeper label function — stop and report,
  do not battery a chance-level net.
- If the α sweep's best is within noise of the worst (all three
  within one win of each other), pick α=1.0 per the §1.4
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain re-runs the α
  sweep and the battery from scratch on the new form.

### 1.6 Holdout diagnostics (pre-registered)

All via `ml/scripts/eval_value_holdout.py` (sign accuracy +
MSE, full-planes and no-history views) on 2-ply labels:
- (a) **v37 net on its own holdout** (80 games, 3,378
  positions) — in-distribution; the §1.5 stop rule applies
  here. ALSO report accuracy BY SIDE TO MOVE (Black/White) —
  the B1-White-collapse lesson: v24's engine-only data did not
  collapse White, and this checks the 2-ply labels didn't
  reintroduce a side skew.
- (b) **v24's net** (`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`,
  by absolute path) **on the same 2-ply labels** — same
  positions, cross-net comparison: how much of the 2-ply
  function does the 1-ply-trained net already capture?
- (c) **v37 net on v24's 1-ply labels** (v24's holdout shards,
  `gomoku-ml1-h1/ml/runs/ml1-h1/data/{v7,v18,random}-holdout/`)
  — the reverse transfer: does the deeper net still reproduce
  the shallower function?
- (d) **v8's net on the 2-ply holdout** — baseline (expect
  ~0.50, as on v24's labels).
No decision hinges on (b)–(d); they diagnose whether the deeper
labels were learned and whether the fit transfers across label
depths.

### 1.7 Eval seeds (pre-registered)

h2h vs v8: v37=100 / v8=200; h2h vs v0: v37=100 / v0=200; h2h vs
v7: v37=100 / v7=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
**placement_seed 11** (the pre-registered seed — v31/v32 used 0
by mistake, do not repeat that), tempo judge; tactical `--seed
100`.

### 1.8 Eval order (pre-registered)

(1) Labelling (§2). (2) Memmap bundle + training (§3) +
holdout diagnostics (§4). (3) Leaf-cost measurement. (4) α
sweep (§5) — registry updated to the winner. (5) FAST LOOP
(§5): up to 4 iterations of 10 games vs v8 AND 10 games vs v0.
(6) FULL VALIDATION BATTERY on the final v37 (§6): tactical
re-run, h2h vs v8 100 games, 100 games vs v0, 100 games vs v7,
100-game swap match vs v7, ladder 20 games/opponent, clause-4
suite re-run. Any battery numbers measured before the fast loop
completes are INTERMEDIATE, labelled as such; only the final
battery counts for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games
fast_loop_match.py --version-a 37 --opponent v8 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v37-vs-v8.json
# Clause 2: ladder (20 games/opponent, mcts_sims 100)
eval_version.py --version 37 --ladder-games 20 --out research/iterations/ladder-v37.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 37 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, swap vs v7 (balanced/tempo, seeds 100/200, placement_seed 11)
fast_loop_match.py --version-a 37 --opponent v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v37-vs-v7.json
fast_loop_match.py --version-a 37 --opponent v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v37-vs-v0.json
selfplay_batch.py --engine-a v37 --engine-b v7 --games 100 --swap-rule --placement-seed 11 --out research/iterations/batch-v37-vs-v7-swap.json
```

## 2. Data generation (actuals)

TODO — labelling results.

## 3. Training (actuals)

TODO — training log.

## 4. Holdout diagnostics (actuals)

TODO — holdout results.

## 5. α sweep and fast loops

TODO — sweep results, registry α update, fast loops.

## 6. Validation battery

TODO — battery results.

## 7. Verdict

TODO — verdict.

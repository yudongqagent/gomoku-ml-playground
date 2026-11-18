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
  copied VERBATIM from `gomoku-ml2-h2` (sha256 `a2b2c0c7…`) —
  then ONE adaptation: the `resolve_opponent` unpack changed
  from 2 values to 3 (`player, choose_move, name`) for the
  batch-3 per-game-seeding signature (the sweep itself keeps
  v24/v31's fixed-seed protocol — see the comment at the call
  site; it only SELECTS α).
- `ml/scripts/find_losing_moves.py`: v31's parameterised version
  (takes match files as arguments), copied VERBATIM from
  `gomoku-ml2-h2` (sha256 `1f1ca6cc…`).
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

**Labelling COMPLETE 2026-11-18 14:10:45 UTC** (started 13:45:49
UTC; 24.9 min total, no interruptions): `generate_distill_data.py
label --label-mode rollout2` over v24's exact games files
(copied verbatim into `ml/runs/ml3-h2/games-{v7,v18,random}.json`).
Full log `ml/runs/ml3-h2/label.log` (LABEL_JSON lines).

| Source | Games | Plies | Quiet positions | ±1 labels | Gradient labels | Held-out games | Label time |
|---|---|---|---|---|---|---|---|
| v7 self-play (50 sims, seeds 50000+) | 488 | 25,277 | 21,595 (19,545 train / 2,050 holdout) | 1,347 (6.2%) | 20,248 | 48 | 887.2 s |
| v18 self-play (50 sims, seeds 60000+) | 229 | 11,511 | 9,932 (9,142 train / 790 holdout) | 736 (7.4%) | 9,196 | 22 | 351.1 s |
| Random playouts (seeds 70000+) | 100 | 11,009 | 5,700 (5,162 train / 538 holdout) | 1,856 (32.6%) | 3,844 | 10 | 207.3 s |
| **Total** | **817** | **47,797** | **37,227 (33,849 train / 3,378 holdout)** | **3,939 (10.6%)** | **33,288** | **80** | **1,445.6 s** |

Position counts match v24's shards EXACTLY (verified
per-source: same shard counts, same per-shard position counts
in the train dirs) — the same 37,227 positions, only the labels
differ, as pre-registered. The 2-ply labels are hotter than
v24's 1-ply labels on every source (±1: 6.2% vs 3.8% v7, 7.4%
vs 5.1% v18, 32.6% vs 28.5% random; train mean|v|: v7 0.1570,
v18 0.1775, random 0.4022; pooled train mean|v| 0.1999) — the
4-ply horizon finds more forced wins, as the §0 measurement
predicted (10.6% ±1 pooled vs v24's 7.9%).

Memmap bundle `ml/runs/ml3-h2/data-mmap/` built from
`ml/runs/ml3-h2/data-train/` (symlinks to the three train dirs
only — holdouts excluded; recursive glob follows the symlinked
dirs, 737 shards found): 33,849 samples, planes (33849, 8, 15,
15), mean|v| 0.1999.

## 3. Training (actuals)

**Training COMPLETE 2026-11-18 14:29 UTC** (started 14:13:25 UTC,
~16 min, no interruptions — a VM replacement happened minutes
AFTER completion, killing only the wait loop): config
`ml/configs/ml3-h2.yaml`, 15 epochs, batch 64, memmap bundle
(33,849 samples; 32,157 train / 1,692 val, seed-0 split).
Run dir `ml/runs/ml3-h2-train/`; log `ml/runs/ml3-h2/train.log`.
RSS stayed low (memmap path). 7,545 steps, ~46–78 s/epoch.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.1079 | 0.1237 |
| 2 | 0.1017 | 0.1226 |
| 3 | 0.1004 | 0.1180 |
| 4 | 0.0981 | 0.1165 |
| 5 | 0.0967 | 0.1148 |
| 6 | 0.0944 | 0.1152 |
| 7 | 0.0926 | 0.1109 |
| 8 | 0.0899 | 0.1102 |
| 9 | 0.0872 | 0.1093 |
| 10 | 0.0855 | 0.1066 |
| 11 | 0.0834 | 0.1035 |
| 12 | 0.0814 | 0.1005 |
| 13 | 0.0799 | 0.0988 |
| 14 | 0.0794 | 0.0988 |
| 15 | 0.0788 | **0.0986** |

Val loss decreased monotonically (one tiny bump at epoch 6,
0.1148 → 0.1152) — no overfitting by epoch 15, but the
pre-registered plan is 15 epochs and the battery is the verdict,
so no extension. **Best checkpoint: epoch 15 (`best.pt`, step
7545, val 0.0986) = final.pt** (same step/epoch; the `--out`
copy at `ml/runs/ml3-h2/value-net.pt` is verified
bit-identical in weights to `best.pt`, max |Δ| 0.0), so the
registry path holds the best-by-val-loss checkpoint as
pre-registered. (Absolute val loss is not comparable to v24's
0.0617 — the 2-ply labels are hotter, mean|v| 0.20 vs 0.06, so
the target variance is higher; sign accuracy is the comparable
metric.) Architecture verified: ResNet 16ch/2 blocks, 126,759
params, 48 tensors — identical counts to v24's and v8's
checkpoints (iter-24's "126,586" was a miscount).

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th game, never trained on — the SAME
80 games / 3,378 positions v24 held out), targets = 2-ply labels
unless noted. v37 net = `ml/runs/ml3-h2/value-net.pt`; v24 net =
`gomoku-ml1-h1/ml/runs/ml1-h1/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml3-h2/holdout-{2ply,1ply}-{v7,v18,random}.txt`.
Sanity check: this run reproduces v24's published 1-ply numbers
EXACTLY on its own holdout (v7 0.7713, v18 0.8384, random 0.7055,
pooled 0.7766) — the eval setup is sound.

### (a) v37 net on its own 2-ply holdout (in-distribution)

| Holdout | n pos | v37 sign (full) | v37 sign (no-hist) | v37 MSE | v37 mean\|v\| | v24 sign (full) | v8 sign (full) |
|---|---|---|---|---|---|---|---|
| v7 | 2,050 | 0.7395 | 0.5163 | 0.0781 | 0.090 | 0.7348 | 0.5572 |
| v18 | 790 | 0.8436 | 0.5846 | 0.0819 | 0.104 | 0.8115 | 0.4897 |
| random | 538 | 0.6623 | 0.6604 | 0.2675 | 0.290 | 0.6529 | 0.5009 |
| **pooled** | **3,378** | **0.7519** | **0.5568** | — | — | **0.7398** | **0.5316** |

**Verdict: the distillation learned the deeper label function —
the §1.5 stop condition (chance-level holdout) does NOT
trigger.** Pooled in-distribution sign agreement 0.7519 vs 0.50
chance. But it is LOWER than v24's 0.7766 on 1-ply labels — the
2-ply function is harder to learn, as expected (a 4-ply lookahead
compressed into a static evaluation). And v24's 1-ply-trained
net already captures the 2-ply function at 0.7398 — the marginal
gain from training on 2-ply labels is only +0.012 in holdout
terms (the two label functions correlate at r=0.72, §0).

**The net is catastrophically history-dependent on engine
positions — the v31 pattern, from a different cause.** The
no-history view collapses to near chance on v7 (0.7395 → 0.5163,
MSE 0.078 → 0.436, mean|v| ballooning 0.090 → 0.436) and badly
on v18 (0.8436 → 0.5846), while v24's net is history-robust on
the same positions (v7 0.7713 both views; v18 0.8344
no-history). Random positions are unaffected (0.6623 → 0.6604)
— their values are visible in the current board. Mechanism
(hypothesis, consistent with the numbers): the 2-ply label is a
harder function of the CURRENT board than the 1-ply label, so
with 33,849 samples the optimizer grabbed the history planes as
a shortcut (game phase / last-move character correlates with the
label) instead of learning the true board→label mapping. At
inference the leaves get SEARCH-PATH history — neither the
real-game history of the full-planes view nor the zeroed history
of the no-history view — so the operating point lies between the
two views, but on engine positions both bounds are worse than
v24's. **This foreshadows that v37 may play worse than v24**
(the same warning sign v31 showed before scoring 0.40 vs v8) —
but holdout is diagnostic, not the verdict: the α sweep and the
battery measure played strength, and they proceed as
pre-registered. No retrain is authorised by §1.5 on holdout
grounds (the stop rule is chance-level own-holdout, which did
not trigger); a training change (e.g. history augmentation)
would need a fast-loop loss to motivate it, and none exists yet.

**By side to move** (v37 net, 2-ply holdout, full planes;
`/tmp/by_side.py`): Black-to-move sign 0.7269 (1182/1626), MSE
0.1242; White-to-move sign **0.7775** (1237/1591), MSE 0.0939.
**No White collapse** — White is BETTER than Black in holdout.
The engine-only data did not reintroduce a side skew (contrast
v25/v31's B1-data White collapses).

### (c) Reverse transfer: v37 net on v24's 1-ply labels

| Holdout | v37 sign (full) | v37 sign (no-hist) | v24 sign (full) |
|---|---|---|---|
| v7 | 0.7550 | 0.5409 | 0.7713 |
| v18 | 0.8449 | 0.5848 | 0.8384 |
| random | 0.6482 | 0.6750 | 0.7055 |
| **pooled** | **0.7590** | — | **0.7766** |

The deeper net reproduces the shallower function at 0.7590 vs
v24's 0.7766 — decent reverse transfer (it beats v24 on v18,
0.8449 vs 0.8384, with lower MSE 0.0624 vs 0.0691) but not full,
and the no-history collapse persists (0.54 on v7). The two nets
encode overlapping but distinct value functions, as expected
from their different label depths.

## 5. α sweep and fast loops

**Leaf cost (2026-11-18, `ml/runs/ml3-h2/leaf-bench.txt`):**
`bench_value_leaf.py --checkpoint ml/runs/ml3-h2/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-move at 100
sims mean 2,155 ms / median 2,137 ms / p95 2,924 ms; net leaf
evals per move mean 80.2 (15–99). v37 runs at v24's speed (mean
2,233 ms) — the same 16ch net — so battery games run ~2–4 min
each under contention; the four 100-game matches are the long
pole of this workstream.

**α sweep — COMPLETE 2026-11-18 ~07:58 PST** (20 games each vs
v8, seeds 1001/2001; per-alpha JSONs
`research/iterations/alpha-v37-<α>-vs-v8.json`). No
interruptions. Clause-4 early check ran concurrently: ml pytest
**428 passed, 1 skipped, 58 xfailed, 0 failed** (19:50 wall;
log `ml/runs/ml3-h2/pytest-full.log`); npm test 55/55 and tsc
clean were verified earlier (§0).

| α | W-L-D | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| **0.25** | **15-5-0** | **0.750** | **[0.5313, 0.8881]** | 7-3-0 | 8-2-0 |
| 0.5 | 11-7-2 | 0.600 | [0.3866, 0.7812] | 7-2-1 | 4-5-1 |
| 1.0 | 10-10-0 | 0.500 | [0.2993, 0.7007] | 8-2-0 | 2-8-0 |

**Winner: α = 0.25** — registry `value_weight` set to 0.25.
Monotone decreasing in α, as the §1.4 prediction (hotter labels
→ smaller α) said. The α effect is almost entirely a WHITE
effect: Black is 0.70–0.80 at every α, while White collapses
0.80 → 0.45 → 0.20 as α rises — at full weight the net's hot
values destroy White's defence (the optimizer's curse hits the
defending side hardest). At α=0.25 White is 8-2, the strongest
White sweep split in the program — but 10 games per colour is a
small sample, and v24 (0.65) and v31 (0.70) both flattered in
sweeps before the battery said otherwise. The §1.4 no-signal
rule does not apply (0.25 vs 0.5 differ by 4 wins). The sweep
selects α but proves nothing — the 100-game battery is the
verdict.

TODO — fast loops.

**Fast loop iteration 1** (10 games each, seeds 1001/2001, α=0.25):
- vs v8: **7W-3L-0D (0.70)**, CI [0.3968, 0.8922]. As Black 5-0
  (all short wins, 19–25 plies), as White 2-3. All 10 games unique
  (per-game seeding; dedup rate 1.00).
- vs v0: **3W-6L-1D (0.35)**, CI [0.1369, 0.6463]. As Black 2-3,
  as White 1-3-1. All 10 games unique.
- Loss classification (`classify_losses.py`, both matches):
  **9 losses, 9 via open four, 0 missed blocks, 0 clean missed
  preventions in 46 threatened turns.** Defence as measured is
  clean — the losses are positional drift into unstoppable open
  fours, not blundered blocks.
- Losing moves (`find_losing_moves.py`, VCF solver depth 4; raw
  `ml/runs/ml3-h2/losing-moves-iter1.txt`): **7 of 9 losses were
  already lost** at the first VCF moment (no saving reply exists
  — positional decline, no single tactical blunder). **TWO
  genuine blunders, both vs v0: game 2 (Black, 48 plies), ply 36
  — (6,3), with 3 saving moves ((10,3), (13,3), (12,4)); game 9
  (White, 53 plies), ply 47 — (4,11), with 4 saving moves ((9,8),
  (7,10), (5,12), (7,14)).**

**Workup verdict — vs-v0 game 2, ply 36 (VERIFIED):** `diag_root.py`
(raw `ml/runs/ml3-h2/diag-v0-g2-ply36.txt`) shows the search's q
is RIGHT and the prior is wrong: the saves (13,3)/(12,4) have
BETTER our-q (−0.063/−0.062) than the actual (6,3) (−0.120),
but only 7 visits each vs 11 — their v1 priors (0.040/0.044)
are 3x the actual's (0.1266), and the most-visited root choice
follows the prior. Worse, the actual TIED the saving (10,3) at
11 visits with an identical prior (0.1266) and won the RNG
tie-break. This is the SAME category as v31 iter-2's blunder:
a PRIOR problem in shared heuristic-track code (`_priors`),
whose change would alter v8/v7 equally and invalidate the α
sweep — NOT patchable within H2's remit (v37 = v8's search +
distilled net).

**Workup verdict — vs-v0 game 9, ply 47 (VERIFIED):** `diag_root.py`
(raw `ml/runs/ml3-h2/diag-v0-g9-ply47.txt`) shows a FLAT root:
the actual (4,11) has the best our-q (−0.159), the most visits
(4), and the highest prior (0.0545); every save sits at our-q
≈ −0.250 (must-block leaves). The 1-ply heuristic leaves AGREE
(actual +0.087..+0.166, best; saves +0.033..+0.113). The VCF
(depth-4 attacker moves, up to 8 plies) is beyond the 100-sim
search's horizon AND beyond the 2-ply label's 4-ply horizon —
the net agrees with the exact heuristic leaf, and both rank the
loser first. This is the shared search-horizon limit (v24's d9
class), NOT a value-head failure — no retraining on these labels
fixes it (a perfect 2-ply-label net ranks the loser first too).

**Fast-loop STOP after iteration 1:** no patchable weakness class
for v37 within H2's design space. The one q-vs-prior blunder is
a shared-prior issue; the other blunder is a shared-horizon
issue; the other 7 losses are positional with no single blunder.
The §4 history-dependence is NOT implicated in either blunder
(g2: the net's q was already right; g9: the net agrees with the
exact leaf) — so no loss-motivated retrain is authorised by
§1.5, and a history-augmentation retrain on holdout grounds
alone would repeat v31's error of chasing a diagnostic. The
battery is the verdict; if v37 plays badly, the battery losses
are the motivation the next batch needs.

## 6. Validation battery

### 6.1 Tactical — COMPLETE 2026-11-18 ~09:15 PST — **CLAUSE 3 PASSES**

`eval_tactical_full.py --version 37 --suite both --seed 100`
(log `ml/runs/ml3-h2/tactical-v37.log`):

- Old suite: **20/20** ✓
- Hard suite (corrected): **29/29** ✓ — including
  **setup-v1v0s-g92: chose (9,6)**, the quiet setup v31 failed.

**Mechanism check — the 2-ply labels FIXED the quiet-setup
failure at the engine level** (`/tmp/diag_g92_v37.py`): v37's
search at g92 ranks (9,6) FIRST with 69 visits and our-q +0.244,
vs (5,4) 5 visits and our-q +0.077 — the v31 inversion
((9,6) 1 visit, our-q −0.377; (5,4) 75 visits, our-q +0.788) is
gone and reversed. The fix works through the SUBTREE, not the
single-leaf ranking: v37's net raw values (no-history view)
still slightly prefer (5,4) (+0.134) over (9,6) (+0.108) — both
far below the teacher's +1.000 labels (distillation compression
toward the mean) — but the gap closed from v31's catastrophic
inversion to near-parity (0.026), and the search propagates the
deeper values from the post-setup subtree back to the root,
where the setup dominates. Note v8's raw net ALSO prefers (5,4)
(+0.318 vs +0.264) yet passes in search — the single-leaf
ranking was never the deciding level; the search-level behaviour
is what the battery measures, and it is fixed.

TODO — clause 1, ladder, secondaries, clause 4 re-run.

## 7. Verdict

TODO — verdict.

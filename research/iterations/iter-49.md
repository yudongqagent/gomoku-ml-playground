# Iteration 49 — v49: 4-ply labels (ML batch 5, H2)

**Status: IN PROGRESS — pre-registered; labelling not yet started.**
Comparison target is v44 (the ML track's champion and bar
reference). This is the H2 workstream of ML batch 5 (H1 is v48,
the White-collapse diagnosis; H3 is v50, the α ablation); work
happens on branch `research/ml-batch-5-h2` in the `gomoku-ml5-h2`
worktree. v49 differs from v44 ONLY in the value net's weights
(and the swept α) — same PUCT search, same v1 priors, same sims
(100), same c_puct (1.5), same threat check, same terminal cases
— so the battery is a clean A/B of the 4-ply-label net against
v44's 3-ply-label net (and, less cleanly, against v37's 2-ply-label
net: v49 changes ONLY the labels, on identical positions — see
§1.1 — so the v49-vs-v37 comparison isolates the labelling
change, two plies deeper than v37-vs-v24 did).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v44. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v49 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Ports and cost measurements (before pre-registration)

**Ports (v44 was merged to master, so this worktree already
carries the ML batch 4 code — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: the memmap version (pre-allocated
  `load_shards` + memmap bundle support + lazy-index `train()`),
  on master since the v44 merge — no port needed.
- `ml/scripts/generate_distill_data.py`: v44's version with
  `--label-mode rollout3`, on master — extended with
  `--label-mode rollout4` (see §1.1; the diff is 3 small hunks:
  docstring, label-fn selection, CLI flag; the defaults
  `rollout`/`rollout2`/`rollout3` reproduce v24's/v37's/v44's
  labels exactly).
- `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/scripts/sweep_alpha.py`, `ml/scripts/find_losing_moves.py`,
  `ml/scripts/eval_value_holdout.py` (v30's by-side-to-move
  version): on master since the v44 merge — no port needed.
- `ml/gomoku_ml/mcts_heuristic.py`: `_rollout2_leaf_value` and
  `_rollout3_leaf_value` on master, plus the new
  `_rollout4_leaf_value` (§1.1). The rollout-leaf teacher (v18
  code) is on master since the batch-9 merge — no
  cherry-picking needed, and no re-verification of the leaf port
  is needed either (v24's 401/401 bit-for-bit check stands; the
  teacher code is the same master code v24 verified against).
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port on master. **Checked: `ValueMCTS.__init__` has no
  duplicate parameters** (the silent auto-merge trap — verified
  by inspection of the signature: a single `value_weight`
  parameter at line 114, a single assignment at line 139).
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS` (on master), plus the v49 registry entry
  (`value_checkpoint: ml/runs/ml5-h2/value-net.pt`,
  `value_weight: 0.25` placeholder until the §1.4 sweep sets
  it). The v44 and v37 registry entries are on master.
- `ml/configs/ml5-h2.yaml`: v44's `ml4-h3.yaml` with run
  name/dir changed (and the iter-44 reference updated to
  iter-49).
- `ml/baseline.py` is FROZEN and untouched.

**Checkpoint copies (gitignored data, md5-verified):** v44
`ml/runs/ml4-h3/value-net.pt` (md5
`d22af606f7d2ee0254f372da895ae7ad`, matches iter-44); v37
`ml/runs/ml3-h2/value-net.pt` (md5
`2fde492aae6d44cbe945e15bd5f1f959`, matches iter-37); v8
`ml/runs/b3-h2/value-net.pt` (md5
`5deb26f3df67883f24f04800d5356da6`, matches iter-37/44). The
v24 games files were copied verbatim from `gomoku-ml4-h3`
(`games-{v7,v18,random}.json`, md5-verified against the source).

**Port smoke checks:** v44 loads and plays in this worktree
(checkpoint copied from the main repo, md5 above); v37 loads
and plays (checkpoint copied from `gomoku-ml3`, md5 above);
v49 is registered (no checkpoint yet — training produces it);
`_rollout4_leaf_value` runs (empty board: −0.0265).

**Cost measurements** (500 quiet positions from v7 games,
`/tmp/bench_label4.py`, sibling batch-5 workers running at
measure time, so absolute ms are inflated vs iter-44's
unloaded measures — ratios are the comparable numbers):
- 4-ply label (`_rollout4_leaf_value`): **median 15.11 ms,
  mean 23.44 ms** (p90 54.54 ms) vs the 3-ply label's mean
  17.13 ms, the 2-ply label's mean 14.32 ms and the 1-ply
  label's mean 9.76 ms on the same machine at the same time
  — **2.40x** the 1-ply cost (mean), **1.37x** the 3-ply
  cost. The plan's "~4–6x 1-ply" estimate was pessimistic:
  most of the cost is fixed overhead (kernel calls,
  winning-move scans), not the extra rollout plies.
  Projected labelling for the full v24 dataset (37,227
  positions): **~14.5 min** — NOT the bottleneck, far under
  the 2-hour bar. **No dataset-size compromise is needed:
  v49 labels v24's FULL position set** (no sampled subset;
  the pre-registered subset rule below is therefore moot but
  recorded anyway).
- **Subset rule (pre-registered, moot):** had the full run
  projected over ~2 hours, labelling would have used every
  2nd quiet position of every game (deterministic,
  stratified by game source — the same fraction of each
  source), recorded in the LABEL_JSON. It is not needed.

**The 4-ply labels are a genuinely different target from the
2-ply labels on the same 500 positions — but much closer than
the 3-ply labels were:** mean |Δ| vs 2-ply 0.0843 (vs 3-ply's
0.1619 vs 2-ply), median 0.0388, p90 0.1826, max 1.1740;
Pearson r 0.8709 (vs 3-ply-vs-2-ply's 0.8865 — adjacent
depths correlate more, and 4-ply-vs-3-ply is 0.9072);
mean|v| 0.2134 vs 0.1881; ±1 fraction 0.1100 vs 0.0860 — the
4-ply labels are slightly hotter (the 6-ply horizon finds
more forced wins). Resolution: ply-1 terminal 0/500, ply-2
terminal 14/500 (2.8%), ply-3 terminal 10/500 (2.0%), ply-4
terminal 13/500 (2.6%), leaf eval 463/500 — nearly all labels
come from the leaf at the 4-ply-ahead position.
**Sign agreement with 2-ply is 0.8280** (vs 3-ply-vs-2-ply's
0.4920) — see the parity finding below. This is NOT a bug:
the implementation was verified by manual re-derivation on
all 500 positions (**0/500 mismatches** — `/tmp/bench_label4.py`
recomputes the §1.1 recursion step by step and compares).

**The parity finding (measured BEFORE pre-registration — it
shapes §1.4 and §1.6):** the 4-ply label has the SAME horizon
parity as the 1-ply and 2-ply labels (EVEN), and the labels
reverse the 3-ply shift exactly as predicted:
- Label means on the same 500 positions: 1-ply **−0.0627**,
  2-ply **−0.0560**, 3-ply **+0.0999**, 4-ply **−0.0630**.
  Fraction positive: 0.212 / 0.234 / **0.734** / **0.216**.
- Pure-gradient pairs (|v| < 0.2 both, n=381): mean(v4−v2) =
  **−0.0012**, median +0.0029, and v4 > v2 on only **51.4%**
  of pairs — NO systematic shift between 4-ply and 2-ply.
  Between 4-ply and 3-ply (n=337): mean(v4−v3) = **−0.1234**,
  median −0.1161, and v4 > v3 on **0.6%** of pairs — the full
  +0.12 parity shift, reversed.
- Mechanism (iter-44 §0's, confirmed in the other
  direction): the 1-ply, 2-ply and 4-ply labels evaluate at a
  position reached after an EVEN number of rollout plies —
  the opponent just moved, so the leaf is pessimistic-tilted.
  The 3-ply label evaluates after an ODD number — WE just
  moved, so the leaf is optimistic-tilted by roughly one
  greedy move's worth of attack-breadth gain. The 4-ply
  label's mean (−0.0630) is statistically indistinguishable
  from the 1-ply (−0.0627) and 2-ply (−0.0560) means on the
  same 500 positions: the parity shift is a function of
  horizon parity, not of depth.
- Consequences, pre-registered: (1) the v49 net's raw values
  will centre near −0.06 on quiet positions, like v37's —
  the α sweep (§1.4) brackets v37's winner (0.25) and v44's
  winner (0.125) plus one step cooler (0.0625), since the
  labels are slightly hotter than v37's (mean|v| 0.21 vs
  0.19) but NOT shifted; (2) cross-depth holdout sign
  agreement IS a fit measure again for 4-ply-vs-2-ply (same
  parity — no shift to confound it), but NOT for
  4-ply-vs-3-ply (§1.6 uses Pearson r there, as iter-44
  did); (3) the battery is the arbiter of whether the deeper
  horizon's extra information (6 plies, more forced wins
  found: ±1 fraction 0.110 vs 0.086) outweighs the
  distillation cost of a harder target — a 4-ply net that
  plays no better than v37 despite a deeper teacher would
  show the label-depth lever saturating at 2 plies even at
  even parity, which is itself a result.

**The setup-v1v0s-g92 diagnostic (the motivating position,
measured BEFORE pre-registration — `/tmp/bench_label4.py`):**
reproduces v44's numbers exactly at the root (1-ply +0.1290,
2-ply +1.0000, 3-ply +1.0000) and extends them:
- At the root (Black to move): the 4-ply label is **+1.0000**
  (same as 2-ply and 3-ply — the forced win is inside the
  6-ply horizon too). The teacher plays the setup (9,6) on
  ply 1.
- At the children (from Black's perspective): (9,6) is +1.0000
  at all depths 2–4; (5,4) is +0.2500 at 1-ply but **+1.0000
  at 2-ply, 3-ply and 4-ply** — the slow win after (5,4) is
  still inside the 6-ply horizon, so the 4-ply labels do NOT
  improve child-level discrimination at g92 over the 2-ply
  labels.
- Honest consequence, pre-registered (same as v37's and
  v44's): the 4-ply labels teach the win at the PRE-setup
  position, not the child ranking. For the tactical suite,
  v49's net must NOT invert the order (as v31's did); if it
  values (9,6) ≈ (5,4), the v1 prior (which ranks (9,6)
  FIRST, 0.1122) breaks the tie correctly. The clause-3 run
  of setup-v1v0s-g92 is the pre-registered mechanism check
  (§6.1).

## 1. Design and pre-registration

**Comparison target: v44** (ML bar reference: ladder Elo 98.1
[−12.1, 208.2], vs v8 0.67, vs v0 0.58, tactical 20/20 + 29/29
on the CORRECTED suite — re-verified by the ML batch 4 H3
worker at seed 100; checkpoint
`ml/runs/ml4-h3/value-net.pt`, ResNet 16ch/2 blocks, α=0.125).
The ML bar's clause 1 is head-to-head vs v44 (normal rules,
100 games, alternating colours, win rate > 0.50 with Wilson
95% CI lower bound > 0.50), clause 2 is ladder Elo strictly
higher than v44's 98.1 (identical 40 games/opponent per leg
after batch 11's re-basing: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical ≥ v44's (old suite
20/20, hard suite 29/29, run at `--seed 100` on the corrected
suite — v44's tactical is re-verified on it first, per the
batch plan), clause 4 is all test suites green (ml pytest,
npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v49 vs v7
(100 games), v49 vs v0 (100 games), a 100-game swap match vs
v7 under the tempo judge (the harness default), and v49 vs
v37 (100 games — the direct 4-ply-vs-2-ply comparison on
identical positions; v37 is the strongest ML version by
head-to-head at 0.69 vs v44). Non-transitivity is recorded
prominently.

**The hypothesis (plan H2):** v37 proved the label depth is
causal — 2-ply labels moved played strength +0.235 over v24
(0.75 vs 0.515 vs v8) on identical positions with only the
labels differing. v44 then showed the 3-ply extension does
NOT help — the odd-horizon parity shift (+0.12 optimistic)
cost more than the extra ply gained (0.67 vs v8, 0.31 vs
v37). The 4-ply label extends the teacher's horizon one ply
further (6 plies vs 5) at EVEN parity, where §0 shows the
shift reverses exactly (mean −0.0630, sign agreement with
2-ply 0.828). So it should teach the value function more
accurately still — OR the marginal label information shrinks
(2-ply already captures the quiet setups, §0's g92 diagnostic
shows 4-ply adds no child-level discrimination there) and the
deeper target is simply harder to distil, in which case v49
plays no better than v37 or worse. Either outcome is
informative: it locates the distillation ceiling in label
depth vs label parity, at the parity the 2-ply recipe already
won at. The battery is the arbiter, as always (v31's lesson:
holdout transfer does not predict played strength).

**The improvement: v49 = v8's search + a value net trained to
predict v18's rollout-leaf value evaluated 4 plies ahead.**
Everything else is v8 unchanged (see the header). The net is
consulted ONLY in quiet positions (terminal cases stay exact,
as in v8), so training data is quiet positions only.

### 1.1 The 4-ply labelling procedure (pre-registered, exact)

The teacher is **v18's rollout leaf** (`_rollout_leaf_value`,
blend=False) — the SAME teacher as v24, v37 and v44, for
comparability. The 4-ply label for a quiet position (`board`,
`player` to move) is `_rollout4_leaf_value(board, player)`:

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
3. **Ply 3 — the teacher's move:** `m3, term3 =
   _rollout_move(board+m1+m2, player)`. If `term3` is not
   None, the label is `term3` (from `player`'s perspective,
   as at ply 1).
4. **Ply 4 — the opponent's reply:** `m4, term4 =
   _rollout_move(board+m1+m2+m3, -player)`. If `term4` is not
   None, the label is `−term4` (`term4` is from the
   opponent's perspective; negated to `player`'s, as at ply
   2).
5. **Otherwise** the label is `_rollout_leaf_value(
   board+m1+m2+m3+m4, player)` — `player` to move again at
   the resulting position, so NO negation (the even-horizon
   form, like the 2-ply form — unlike the 3-ply form, which
   ends with the opponent to move and negates).
   `blend=False` (v24's/v37's/v44's setting).

**Perspective:** the side to move throughout, in [−1, 1],
exactly like v24's, v37's and v44's labels. **Horizon:** 4
plies of rollout-policy play + the 2 plies inside
`_rollout_leaf_value` = 6 plies, vs v44's 5, v37's 4 and
v24's 2. **Why the rollout policy, not a leaf-value argmax or
a minimax reply:** the same reasons as v37's (§1.1 there):
(a) it IS the teacher's move policy — the same policy
`_rollout_leaf_value` uses for its internal 2 plies — so the
v49 label is exactly the 6-ply extension of v24's 2-ply label,
with no new move-selection machinery; (b) maximising the noisy
leaf over ~40 candidates would inject optimizer's-curse bias
into the LABELS; (c) the minimax reply was measured and
rejected at 2-ply (~1.5 s/position, 50x the cost, with NO
better discrimination at setup-v1v0s-g92). **Known limitation
(pre-registered):** the rollout-policy reply is v1-greedy, not
minimax — it can miss refutations, so 4-ply labels can
overvalue moves that win slowly against weak defence (at g92,
(5,4) goes +0.250 → +1.000 at 2-ply and stays +1.000 at
3-ply and 4-ply). **Known limitation 2 (pre-registered, §0):**
the 4-ply target is harder to distil than the 2-ply target
(mean|v| 0.21 vs 0.19, ±1 fraction 0.110 vs 0.086) — the
battery is the arbiter of whether the extra horizon
information survives distillation.

### 1.2 Dataset (pre-registered)

- **Source: v24's engine-game positions, re-labelled.** NO new
  game generation: the games already exist (v24's §2) and game
  generation was the expensive step. The positions are EXACTLY
  v24's, v37's and v44's: the 817 games in
  `gomoku-ml4-h3/ml/runs/ml4-h3/games-{v7,v18,random}.json`
  (488 v7 self-play at 50 sims, seeds 50000+; 229 v18
  self-play at 50 sims, seeds 60000+; 100 random playouts,
  seeds 70000+), copied into this worktree's `ml/runs/ml5-h2/`
  (gitignored data, md5-verified). Expected yield: **37,227
  quiet positions** (21,595 v7 / 9,932 v18 / 5,700 random),
  the same positions v24, v37 and v44 trained on — the ONLY
  change from v44 is the label function, so the v49-vs-v44
  comparison isolates the labelling change.
- **Sampling:** ALL quiet positions of every game (labelling
  is ~15 ms median per position; §0 — the full run is ~14.5
  min, so no subset is needed). A position is quiet iff
  `_terminal_leaf_value == 0.0`. Positions are encoded with the
  real game history (boards 2/4 plies back, last move) exactly
  as in training (`encoding.encode`). Policy targets are one-hot
  on the move actually played (unused at policy_weight=0).
- **Labels:** `_rollout4_leaf_value` (§1.1) via
  `generate_distill_data.py label --label-mode rollout4`.
  Rollout-found terminals (±1) are kept as-is: they are the
  label's real signal.
- **Holdout (game-level, v24's protocol, IDENTICAL split):**
  every 10th game of each source is routed to a `*-holdout`
  dir and NEVER trained on — the same games v24, v37 and v44
  held out (48 v7 / 22 v18 / 10 random = 80 games, 3,378
  positions), so v49's holdout numbers are comparable to
  v24's, v37's and v44's BY POSITION (only the labels differ).
  Expected train set ≈ 33,849 positions.
- **NOT used for training:** the B1 10k games (B1 data collapses
  White — v25, v31 — and v24's engine-only data did not; this
  workstream does not re-open that lever). They are NOT used
  even as a diagnostic here (v24's B1-diag already established
  the transfer numbers; no decision hinges on re-measuring it).
- **Variety check:** not needed — the games are v24's, whose
  variety check passed (14 distinct first-5-ply prefixes in 15
  v7 games).
- **Stopping rule:** labelling is ~14.5 min for the full set
  (§0); no cutoff is needed. If a VM replacement kills the
  labelling run, the label phase is resume-safe (existing shards
  skipped) — re-run the same command.

### 1.3 Architecture and training (pre-registered)

- **Architecture:** v24's/v37's/v44's ResNet **16ch/2 blocks**
  (126,759 params), trained from scratch (random init — NOT
  from v8's, v24's, v37's, or v44's checkpoint; the labels are
  a different target). Do NOT go larger: v31 proved 4x capacity
  plays worse on distilled labels (the scale lever is CLOSED).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes — v24's/v37's/v44's recipe exactly),
  augmentation on (dihedral), val_fraction 0.05 (random split
  within the train set, for best-checkpoint selection only),
  seed 0.
- **Config:** `ml/configs/ml5-h2.yaml` (v44's `ml4-h3.yaml`
  with run name/dir changed); run dir `ml/runs/ml5-h2-train/`;
  threads pinned to 1 (`OMP_NUM_THREADS=1 TORCH_THREADS=1`,
  B1's finding). `checkpoint_every_epochs: 1` from the start
  (v24's resume lesson).
- **Memmap path (mandatory):** the train shards are bundled with
  `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml5-h2/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as
  v24/v37/v44):** best-by-internal-val-loss (`best.pt`), copied
  to `ml/runs/ml5-h2/value-net.pt` (gitignored data, the path
  v49's registry entry points at).
- **Resume rule (pre-registered, v24's lesson):** if training is
  killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans
  the full 15 via `schedule_epochs`); back up the pre-kill
  `best.pt` first, because the resumed run resets
  `best_val_loss` and would overwrite it at the next epoch
  regardless — the final best is chosen by comparing the
  resumed run's best val loss against the pre-kill best.

### 1.4 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.0625, 0.125, 0.25}**, **20 games each vs
v44**, seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 49`. The grid brackets
v37's winner (0.25) and v44's winner (0.125) and goes one step
cooler: the 4-ply labels are slightly hotter than v37's
(mean|v| ~0.21 vs ~0.19 on engine positions — §0) but NOT
shifted (mean −0.06, same as v37's — §0), so the net's values
will be larger in magnitude without the optimistic centre —
the sweep must cover both the v37 winner (same parity) and
the cooler direction (hotter labels). 0.5 is NOT on the grid:
v44's sweep already showed 0.5 losing (0.475) on hotter
labels than v37's, and re-testing it on hotter-still labels
spends 20 games on a dominated arm.
Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure). **No-signal
rule:** if all three arms are within one win of each other,
pick α=0.25 (v37's winner — the same-parity winner) and say
so — the sweep found no signal. The sweep SELECTS α; the
100-game battery is the verdict (standing lesson: small-sample
smokes mislead — v24's 0.65 smoke at α=1.0 did not reproduce
at 100 games; v31's 0.70 did not either). The registry entry's
`value_weight` is updated to the winner after the sweep. (Note:
α is an inference-time weight, so holdout sign accuracy cannot
select it — the sweep is by played win rate, as in
v24/v31/v37/v44.)

### 1.5 What would change the design

- If the distilled net's holdout sign-accuracy on its OWN
  (4-ply-label) holdout is at chance (~0.50), the distillation
  failed to learn the deeper label function — stop and report,
  do not battery a chance-level net.
- If the α sweep's best is within noise of the worst (all three
  within one win of each other), pick α=0.25 per the §1.4
  no-signal rule.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain re-runs the α
  sweep and the battery from scratch on the new form.

### 1.6 Holdout diagnostics (pre-registered)

All via `ml/scripts/eval_value_holdout.py` (sign accuracy +
MSE, full-planes and no-history views) on 4-ply labels unless
noted. **Parity caveat (§0): cross-depth sign agreement IS a
fit measure for 4-ply-vs-2-ply (same parity) but NOT for
4-ply-vs-3-ply** — the +0.12 parity shift alone drives the
latter to ~0.47. For the 3-ply comparison (c), Pearson r
(shift-invariant) is the primary measure; sign agreement is
reported with the caveat.
- (a) **v49 net on its own holdout** (80 games, 3,378
  positions) — in-distribution; the §1.5 stop rule applies
  here (sign accuracy vs chance). ALSO report accuracy BY SIDE
  TO MOVE (Black/White) on both the full-planes and the
  no-history views — the B1-White-collapse lesson (v30's
  method): v24's/v37's/v44's engine-only data did not collapse
  White, and this checks the 4-ply labels didn't reintroduce a
  side skew. ALSO report the no-history view (v37's net
  collapsed without history planes on engine positions, 0.7395
  → 0.5163; v44's did not, 0.7442 → 0.7368 — check which
  pattern the 4-ply net follows).
- (b) **v37's net** (`ml/runs/ml3-h2/value-net.pt`) **on the
  same 4-ply labels** — same positions, cross-net comparison:
  how much of the 4-ply function does the 2-ply-trained net
  already capture? (Same parity — sign agreement IS meaningful
  here, plus Pearson r.)
- (c) **v49 net on v44's 3-ply labels** (v44's holdout shards,
  `gomoku-ml4-h3/ml/runs/ml4-h3/data/{v7,v18,random}-holdout/`)
  — the reverse transfer: does the deeper net still reproduce
  the shallower function? (Pearson r primary — the parity
  shift confounds sign agreement.)
- (d) **v44's net on the 4-ply holdout** — the odd-parity
  net on even-parity targets (expect sign below chance on
  engine positions, as v37's net showed on v44's 3-ply
  targets; Pearson r primary).
- (e) **v8's net on the 4-ply holdout** — baseline (expect
  ~0.50 sign, ~0 r, as on v24's/v37's/v44's labels).
No decision hinges on (b)–(e); they diagnose whether the
deeper labels were learned and whether the fit transfers
across label depths.

### 1.7 Eval seeds (pre-registered)

h2h vs v44: v49=100 / v44=200; h2h vs v0: v49=100 / v0=200;
h2h vs v7: v49=100 / v7=200; h2h vs v37: v49=100 / v37=200;
ladder seed 100, **40 games/opponent** (batch 11's re-basing),
mcts_sims=100; swap 100 games vs v7, balanced placement,
**placement_seed 11** (the pre-registered seed — v31/v32 used 0
by mistake, do not repeat that), tempo judge; tactical `--seed
100`. The vs-v44 leg uses the pre-registered openings file
`research/openings-ml4.json` (ML batch 4 tooling — the
mirror-match fix); the vs-v37 leg uses it too (same
mirror-match rationale — two value-net engines); the vs-v0 and
vs-v7 legs do not (different engine families; v37's legs were
95–99 unique without openings).

### 1.8 Eval order (pre-registered)

(1) Labelling (§2). (2) Memmap bundle + training (§3) +
holdout diagnostics (§4). (3) Leaf-cost measurement. (4) α
sweep (§5) — registry updated to the winner. (5) FAST LOOP
(§5): up to 4 iterations of 10 games vs v44 AND 10 games vs
v0. (6) FULL VALIDATION BATTERY on the final v49 (§6): v44
tactical re-verification, then tactical re-run, h2h vs v44
100 games (with --openings), 100 games vs v0, 100 games vs
v7, 100 games vs v37 (with --openings), 100-game swap match
vs v7, ladder 40 games/opponent, clause-4 suite re-run. Any
battery numbers measured before the fast loop completes are
INTERMEDIATE, labelled as such; only the final battery counts
for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v44, 100 games, pre-registered openings
selfplay_batch.py --engine-a v49 --engine-b v44 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v49-vs-v44.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 49 --ladder-games 40 --out research/iterations/ladder-v49.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 49 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, vs v37 (openings), swap vs v7 (balanced/tempo, seeds 100/200, placement_seed 11)
selfplay_batch.py --engine-a v49 --engine-b v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v49-vs-v7.json
selfplay_batch.py --engine-a v49 --engine-b v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v49-vs-v0.json
selfplay_batch.py --engine-a v49 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v49-vs-v37.json
selfplay_batch.py --engine-a v49 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v49-vs-v7-swap.json
```

## 2. Data generation (actuals)

**Labelling COMPLETE 2026-11-21 ~09:32 UTC** (started 09:04 UTC;
no interruptions): `generate_distill_data.py label
--label-mode rollout4` over v24's exact games files (copied
verbatim into `ml/runs/ml5-h2/games-{v7,v18,random}.json`).
Full log `ml/runs/ml5-h2/label.log` (LABEL_JSON lines).

| Source | Games | Plies | Quiet positions | ±1 labels | Gradient labels | Held-out games | Label time |
|---|---|---|---|---|---|---|---|
| v7 self-play (50 sims, seeds 50000+) | 488 | 25,277 | 21,595 (19,545 train / 2,050 holdout) | 1,788 (8.3%) | 19,807 | 48 | 937.5 s |
| v18 self-play (50 sims, seeds 60000+) | 229 | 11,511 | 9,932 (9,142 train / 790 holdout) | 943 (9.5%) | 8,989 | 22 | 405.5 s |
| Random playouts (seeds 70000+) | 100 | 11,009 | 5,700 (5,162 train / 538 holdout) | 2,020 (35.4%) | 3,680 | 10 | 234.0 s |
| **Total** | **817** | **47,797** | **37,227 (33,849 train / 3,378 holdout)** | **4,751 (12.8%)** | **32,476** | **80** | **~1,577 s** |

Position counts match v24's, v37's and v44's shards EXACTLY
(verified per-source: same shard counts, same per-shard
position counts in the train dirs) — the same 37,227
positions, only the labels differ, as pre-registered. The
4-ply labels are hotter than v44's 3-ply labels on every
source (±1: 8.3% vs 7.3% v7, 9.5% vs 8.5% v18, 35.4% vs 34.2%
random; train mean|v|: v7 0.1760, v18 0.1903, random 0.4290;
pooled train mean|v| 0.2185 vs v44's 0.2099 and v37's 0.1999)
— the 6-ply horizon finds more forced wins, as the §0
measurement predicted (12.8% ±1 pooled vs v44's 11.7% and
v37's 10.6%). **And the parity reversal is in the data:**
train label means are NEGATIVE on both engine sources (v7
−0.0511, v18 −0.0708; pooled train mean −0.0221, pulled up
only by the random source's +0.1737) — the net will learn a
value function centred near −0.02 to −0.06, like v37's, not
near +0.11 like v44's. The α sweep (§5) brackets v37's winner
accordingly.

Memmap bundle `ml/runs/ml5-h2/data-mmap/` built from
`ml/runs/ml5-h2/data-train/` (symlinks to the three train dirs
only — holdouts excluded; recursive glob follows the symlinked
dirs, 737 shards found): 33,849 samples, planes (33849, 8, 15,
15), mean|v| 0.2185. Bundle build took ~10 s.

## 3. Training (actuals)

**Training COMPLETE 2026-11-21 09:48 UTC** (started ~09:33 UTC,
~15 min, no interruptions): config `ml/configs/ml5-h2.yaml`,
15 epochs, batch 64, memmap bundle (33,849 samples; 32,157
train / 1,692 val, seed-0 split). Run dir
`ml/runs/ml5-h2-train/`; log `ml/runs/ml5-h2/train.log`. RSS
stayed low (memmap path). 7,545 steps, ~53–64 s/epoch (epoch 1
103 s, warmup).

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.1287 | 0.1434 |
| 2 | 0.1239 | 0.1423 |
| 3 | 0.1229 | 0.1405 |
| 4 | 0.1207 | 0.1433 |
| 5 | 0.1188 | 0.1405 |
| 6 | 0.1172 | 0.1399 |
| 7 | 0.1142 | 0.1356 |
| 8 | 0.1115 | 0.1338 |
| 9 | 0.1096 | 0.1286 |
| 10 | 0.1060 | 0.1279 |
| 11 | 0.1035 | 0.1254 |
| 12 | 0.1018 | 0.1234 |
| 13 | 0.1002 | 0.1212 |
| 14 | 0.0994 | 0.1210 |
| 15 | 0.0987 | **0.1207** |

Val loss decreased overall (one bump at epoch 4) — no
overfitting by epoch 15, but the pre-registered plan is 15
epochs and the battery is the verdict, so no extension. **Best
checkpoint: epoch 15 (`best.pt`, step 7545, val 0.1207)** —
the `--out` copy at `ml/runs/ml5-h2/value-net.pt` is verified
bit-identical to `best.pt` (md5 `ecea9623cd7d3f4cd45b21b122b6ebfc`
for both files), so the registry path holds the
best-by-val-loss checkpoint as pre-registered. (Absolute val
loss is not comparable to v44's 0.1136, v37's 0.0986 or v24's
0.0617 — the 4-ply labels are hotter and shifted, mean|v|
0.22 vs 0.21 vs 0.20 vs 0.06, so the target variance is
higher; sign accuracy is the comparable metric.)
Architecture verified via the project's own checkpoint loader
(`load_model_from_checkpoint`, used by the holdout script):
ResNet 16ch/2 blocks, 126,759 params, 48 tensors — identical
counts to v24's, v37's, v44's and v8's checkpoints.
Checkpoint md5: `ecea9623cd7d3f4cd45b21b122b6ebfc`.

**Runtime note (security-notice recurrence):** while verifying
the checkpoint, a direct `torch.load` probe of the freshly
trained `value-net.pt` failed with a `weights_only` error, and
the runtime delivered an execution-risk security notice on
that tool result (the traceback text). Per the task brief the
flagged item was skipped (no further processing of that
specific result) and the work continued via the project's own
sanctioned checkpoint loader, which loads the file normally —
the holdout diagnostics in §4 use it and confirm the
checkpoint is valid. This is the second such notice on this
workstream (the first, on the previous worker, stopped the
task; this one did not stop anything).

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th game, never trained on — the SAME
80 games / 3,378 positions v24, v37 and v44 held out), targets =
4-ply labels unless noted. v49 net =
`ml/runs/ml5-h2/value-net.pt`; v37 net =
`ml/runs/ml3-h2/value-net.pt`; v44 net =
`ml/runs/ml4-h3/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml5-h2/holdout-4ply-{v7,v18,random}.txt`,
`holdout-3ply-{v7,v18,random}.txt` and
`holdout-2ply-{v7,v18,random}.txt`; pooled Pearson r computed
by concatenating all three sources (not by averaging
per-source r).
Sanity check: this run reproduces v37's published 2-ply numbers
EXACTLY on its own holdout (v7 0.7395, v18 0.8436, random 0.6623
— full planes; no-history 0.5163/0.5846/0.6604; MSEs
0.0781/0.0819/0.2675) — the eval setup is sound.

### (a) v49 net on its own 4-ply holdout (in-distribution)

| Holdout | n pos | v49 sign (full) | v49 sign (no-hist) | v49 MSE | v49 mean\|v\| | v37 sign (full) | v44 sign (full) | v8 sign (full) |
|---|---|---|---|---|---|---|---|---|
| v7 | 2,050 | 0.7383 | 0.5340 | 0.0925 | 0.0897 | 0.7483 | 0.2865 | 0.5192 |
| v18 | 790 | 0.7500 | 0.5242 | 0.0982 | 0.1022 | 0.7717 | 0.2844 | 0.5204 |
| random | 538 | 0.6786 | 0.6598 | 0.2443 | 0.2611 | 0.6429 | 0.4624 | 0.4868 |
| **pooled** | **3,378** | **0.7313** | **0.5524** | — | — | **0.7365** | **0.3151** | **0.5142** |

**Verdict: the distillation learned the deeper label function —
the §1.5 stop condition (chance-level holdout) does NOT
trigger.** Pooled in-distribution sign agreement 0.7313 vs 0.50
chance. It is LOWER than v44's 0.7442 on 3-ply labels, v37's
0.7519 on 2-ply labels and v24's 0.7766 on 1-ply labels — the
deeper the label, the harder the function to learn, monotone
across all four depths (0.7766 → 0.7519 → 0.7442 → 0.7313).
Per-source, v49 is worse than v37 on v7 (0.7383 vs 0.7395,
a wash) and v18 (0.7500 vs 0.8436 — the strongest source is
again where the deeper net's fit is weakest, as for v44) but
better on random (0.6786 vs 0.6623).

**The history collapse is BACK.** The no-history view drops
0.179 pooled (0.7313 → 0.5524): v7 0.7383 → 0.5340, v18
0.7500 → 0.5242, random 0.6786 → 0.6598 (no collapse on
random, as for v37). Compare v37: v7 0.7395 → 0.5163, v18
0.8436 → 0.5846 — a 0.195 pooled collapse; v44: 0.007 pooled
drop. The no-history MSE also blows up (v7 0.5487 vs full
0.0925; mean|v| 0.5172 vs 0.0897 — without history planes the
net outputs huge values). The pattern now tracks label parity
across the 2/3/4-ply versions: the even-parity nets (v37, v49)
both grabbed the history shortcut while the odd-parity net
(v44) did not. The
mechanism hypothesis from iter-44 §4 (the 3-ply label is more
determined by the current board, so the optimizer had less
reason to grab the history shortcut) now has its mirror image:
the even-parity labels leave the history shortcut open, and
the optimizer takes it. History-dependence is the top ML-track
risk again for this version — but it is a holdout diagnostic,
not a played-strength verdict (v37 collapsed on holdout and
is still the strongest ML version head-to-head).

**By side to move** (v49 net, 4-ply holdout; v30's method via
the ported `eval_value_holdout.py`):

| Holdout | Black (full) | White (full) | Black (no-hist) | White (no-hist) |
|---|---|---|---|---|
| v7 | 0.7281 (707/971) | 0.7489 (695/928) | 0.5026 (488/971) | 0.5668 (526/928) |
| v18 | 0.7143 (285/399) | 0.7870 (303/385) | 0.4612 (184/399) | 0.5896 (227/385) |
| random | 0.6045 (162/268) | 0.7538 (199/264) | 0.6604 (177/268) | 0.6591 (174/264) |
| **pooled** | **0.7045 (1154/1638)** | **0.7590 (1197/1577)** | **0.5183 (849/1638)** | **0.5878 (927/1577)** |

**No White collapse — White is BETTER than Black** (+0.055
pooled on full planes, +0.070 on no-history). The engine-only
data did not reintroduce a side skew (contrast v25/v31's
B1-data White collapses); the split matches v37's
White-better pattern (0.7775 vs 0.7269). The random source's
Black weakness (0.6045) is the only soft spot, and it is a
holdout curiosity, not a played-strength measure.

### (b) v37's net on the 4-ply labels (cross-depth, same parity)

Sign agreement 0.7483 / 0.7717 / 0.6429 (pooled **0.7365**) —
HIGHER than v49's own 0.7313 on its own labels. The 2-ply-trained
net captures the 4-ply function in sign terms as well as (or
better than) the 4-ply-trained net does, on every engine source.
**Pearson r (pooled): v37 0.3585 vs v49's own 0.4459** — in
correlation terms the 4-ply net IS better (per-source too:
0.2782/0.3863/0.5183 vs 0.2104/0.3149/0.3760). The same
pattern as v44's §4(b): the marginal holdout gain from deeper
labels is nil-to-negative in sign terms, positive in
correlation terms — but holdout is diagnostic, not the
verdict (v37's holdout marginal gain over v24 was also tiny,
+0.012 sign, while its played-strength gain was +0.235).

### (c) Reverse transfer: v49 net on v44's 3-ply labels

Sign agreement 0.3393 / 0.3954 / 0.4972 (pooled 0.3791) —
below chance on engine positions, the parity shift again
(v49's net centres near −0.06, the 3-ply targets near +0.10).
**Pearson r: v7 0.2476, v18 0.3447, random 0.4818, pooled
0.4076** — vs v44's own pooled r of 0.3149 (v7 0.2027, v18
0.1480, random 0.3508). The 4-ply net captures the 3-ply
function BETTER in correlation terms than the 3-ply-trained
net itself does, on every source. The deeper net is the best
correlator of the shallower functions (see also the 2-ply
transfer below) — depth buys correlation, not sign agreement.

**v49 net on v37's 2-ply labels** (extra, from the §4 sanity
run): sign 0.7337 / 0.8141 / 0.7036 (pooled 0.7482) vs v37's
own pooled 0.7519 — the 4-ply net reproduces the 2-ply
function at 99.5% of v37's own sign agreement. **Pearson r:
v7 0.2622, v18 0.4092, random 0.5372, pooled 0.4723** — vs
v37's own pooled r of 0.3979 (v7 0.1987, v18 0.3455, random
0.4177): again higher on every source. The 4-ply net encodes
the 2-ply function better in correlation terms than the
2-ply-trained net does.

### (d) v44's net on the 4-ply holdout

Sign agreement 0.2865 / 0.2844 / 0.4624 (pooled 0.3151) —
BELOW chance on engine positions, exactly as §1.6 predicted:
v44's net outputs values centred near +0.10 (its optimistic
3-ply target) while the 4-ply targets centre near −0.06, so
the signs systematically disagree. **Pearson r (pooled):
0.3224** (v7 0.1272, v18 0.1130, random 0.3729) — vs v49's
own pooled r of 0.4459. The odd-parity net correlates with
the even-parity function at ~72% of the in-distribution
level, but its signs are unusable — the parity shift is a
sign-level distortion, not a correlation-level one.

### (e) v8's net on the 4-ply holdout

Sign agreement 0.5192 / 0.5204 / 0.4868 (pooled 0.5142),
Pearson r pooled 0.0548 (v7 −0.0134, v18 −0.0354, random
0.1060) — at/below chance, as on v24's/v37's/v44's labels.
v8's net encodes a genuinely different value function, as
expected from its different target (weak-self-play game
outcomes).

**A note on the low Pearson r values (0.28–0.52 even
in-distribution):** r is dragged down by distillation
compression — the nets predict small values (mean|v| ~0.09 on
engine positions) while the targets carry ±1 spikes (12.8%
of positions), so magnitudes barely track even when signs
agree 73–75% of the time. r is used here ONLY for cross-depth
comparison (where it is shift-invariant); the in-distribution
fit measure is sign agreement, as in v24/v37/v44.

## 5. α sweep and fast loops

**Leaf cost (2026-11-21, `ml/runs/ml5-h2/leaf-bench.txt`):**
`bench_value_leaf.py --checkpoint ml/runs/ml5-h2/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-move at 100
sims mean 3,364 ms / median 3,051 ms / p95 4,895 ms; net leaf
evals per move mean 81.2 (21–99); per-leaf net eval median
2.33 ms (mean 9.48 ms — a heavy tail under load: load average
7.22 at measure time, sibling batch-5 workers running).
Battery games run ~2–5 min each under contention; the 100-game
matches are the long pole of this workstream.

### 5.1 α sweep (2026-11-21, ~10:00–13:35 UTC)

`sweep_alpha.py --version 49 --alphas 0.0625 0.125 0.25 --games
20 --opponent v44 --seed-a 1001 --seed-b 2001 --out-dir
research/iterations` (dev seeds, no openings). One VM
replacement killed the run at α=0.125 game 7; the driver
resumed from game 8 (4W-3L-1D) with per-game seeding —
nothing lost. Log: `ml/runs/ml5-h2/sweep-alpha.log`;
per-alpha JSONs in `research/iterations/alpha-v49-<α>-vs-v44.json`.
All numbers below recomputed from the raw per-game records.

| α | W-L-D | win rate | 95% CI | as Black | as White | unique |
|---|-------|----------|--------|----------|----------|--------|
| **0.0625** | **10-8-2** | **0.5500** | [0.3421, 0.7418] | 5-3-2 (0.600) | 5-5-0 (0.500) | 20/20 |
| 0.125 | 9-9-2 | 0.5000 | [0.2993, 0.7007] | 5-4-1 (0.550) | 4-5-1 (0.450) | 20/20 |
| 0.25 | 8-11-1 | 0.4250 | [0.2383, 0.6359] | 5-4-1 (0.550) | 3-7-0 (0.300) | 20/20 |

Win rate is MONOTONE DECREASING in α — the third consecutive
distilled net where the coolest swept α wins (v37: 0.25 > 0.5 >
1.0; v44: 0.125 > 0.25 > 0.5; v49: 0.0625 > 0.125 > 0.25).
The no-signal rule (§1.4) does NOT trigger: the arms span 10
to 8 wins, a 2-win spread, not within one win of each other.
**Selection (§1.4 rule, best win rate): α=0.0625.** Registry
updated (`scoring.py`, `value_weight: 0.0625`); targeted
tests 16/16 pass. The White split is where α bites: White
falls 0.500 → 0.450 → 0.300 as α rises, while Black is flat
(0.60/0.55/0.55) — the hotter 4-ply values hurt most when
playing from behind, the same shape as v44's parity-shift
story but milder (the 4-ply labels are NOT shifted; they are
only hotter). The 20-game CIs overlap, so the α ranking is a
point-estimate call, not a proven ordering — but the monotone
shape across three grid points plus the mechanism (hotter
labels → cooler α) makes 0.0625 the defensible pick, and the
100-game clause-1 leg is the real verdict.

Note the contrast with v37: its sweep winner was α=0.25 on
the same-parity 2-ply labels. v49's winner sits two steps
cooler, even though the labels share v37's parity — the
hotness (mean|v| 0.22 vs 0.20, ±1 fraction 0.128 vs 0.106)
alone pushes α down. If the 100-game battery confirms the
sweep, the α lesson generalises further: cooler α tracks
label hotness, not just parity.

### 5.2 Fast loop iteration 1 (α=0.0625, seeds 1001/2001)

- vs v44: the α=0.0625 sweep leg itself (20 games, same dev
  seeds) — **10W-8L-2D (0.5500)**. As Black 5-3-2, as White
  5-5-0. All 20 games unique.
- vs v0: **3W-7L-0D (0.30)**, CI [0.1078, 0.6032], 442 s
  (`research/iterations/fast-v49-vs-v0-iter1.json`). Worse
  than v44's 0.55 and v37's 0.35 in the same loop, though all
  three CIs are wide (10 games each).
- Loss classification (both matches): **15 losses, 15 via open
  four, 0 missed blocks, 0 clean missed preventions in 72
  threatened turns.** Defence as measured is clean — the
  losses are positional drift into unstoppable open fours,
  not blundered blocks. Same pattern as v37/v44.
- Losing moves (`find_losing_moves.py`, VCF solver depth 4;
  raw `ml/runs/ml5-h2/losing-moves-iter1.txt`): **14 of 15
  losses were already lost** at the first VCF moment. **ONE
  genuine blunder:** vs v44 game 15 (White, 57 plies), ply
  43 — (12,5), 1 save (8,9).

**Workup — vs-v44 game 15, ply 43 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml5-h2/diag-v44-g15-ply43.txt`)
shows a q-vs-LEAF inversion: the actual (12,5) has our-q
−0.114 vs the save's −0.142 (a 0.028 gap, within noise), but
the exact 1-ply leaf ranks the save clearly first (v22 leaf
+0.139 vs +0.014; v14 leaf +0.039 vs −0.098). The priors are
essentially tied (0.1110 actual vs 0.1112 save — the prior
marginally favours the save). The leaf is RIGHT; the
backed-up root q inverts it through subtree averaging on a
very flat root (top move 10 visits over 108 children). This
is v40's "blunder dilution / visit allocation" class — a
search problem, not a leaf problem (the leaf-blind direction
is CLOSED, and batch 12 rejected both blunder-dilution fixes:
v46's leaf-level pure-min and v47's pure minimax backup).
Not patchable within H2's remit: changing visit allocation is
shared heuristic-track code, and "more sims" is a closed
lever.

**Fast-loop STOP after iteration 1:** no patchable weakness
class for v49 within H2's design space. The single blunder is
a visit-allocation inversion in shared search code, not a
value-head failure. The §4 history-dependence is NOT
implicated in the blunder (the leaf is right, the averaging
is wrong) — so no loss-motivated retrain is authorised by
§1.5. The battery is the verdict.

## 6. Validation battery

### 6.1 Tactical — COMPLETE 2026-11-21 — **CLAUSE 3 PASSES**

`eval_tactical_full.py --version 49 --suite both --seed 100`
(log `ml/runs/ml5-h2/tactical-v49.log`):

- Old suite: **20/20** ✓
- Hard suite (corrected): **29/29** ✓ — including
  **setup-v1v0s-g92: chose (9,6)**, the quiet setup v31
  failed. (As §0 predicted: 4-ply labels both (9,6) and
  (5,4) at +1.0000, so the choice between them comes from
  the search, not the label — and the search picks (9,6).)
- v1bl-g2 (the v40 regression position): passed, chose
  (7,5) — no leaf-value tie-break regression from the
  4-ply labels.
- v44's tactical was re-verified on the corrected suite
  first (log `ml/runs/ml5-h2/tactical-v44-reverify.log`):
  **20/20 + 29/29** — the bar reference's numbers stand.

### 6.2 vs v44 — COMPLETE 2026-11-21 ~16:32 UTC — **CLAUSE 1 FAILS**

`selfplay_batch.py --engine-a v49 --engine-b v44 --games 100
--openings research/openings-ml4.json --seed-a 100 --seed-b 200`
(log `ml/runs/ml5-h2/batt-v44.log`; JSON
`research/iterations/batch-v49-vs-v44.json`). 13,194 s wall,
no interruptions. All numbers below recomputed from the raw
per-game records.

- **43W-48L-9D, win rate 0.4750, Wilson 95% CI [0.3799,
  0.5720]** — win rate < 0.50 and CI lower < 0.50 ✗. v49
  LOSES to v44 head-to-head. The even-parity 4-ply labels do
  NOT beat the odd-parity 3-ply labels: the parity reversal
  (§0) did not translate into played strength.
- Uniqueness audit: **100 unique games in 100** (the
  --openings set + per-game seeding; the best uniqueness of
  any ML-vs-ML leg — v44's was 93). De-dup cut identical.
- Colour splits: as Black **25W-19L-6D (0.5600)**, as White
  **18W-29L-3D (0.3900)** — a White gap (0.17), though not
  the v44-vs-v37-style collapse (0.16). Black won 54, White
  won 37.
- Comparison: the 20-game sweep leg at the same α was 0.55
  (10W-8L-2D) — the 100-game leg reverses it to 0.475. The
  sweep's point-estimate ranking did not reproduce at 100
  games (the CIs overlap: sweep [0.3421, 0.7418] vs battery
  [0.3799, 0.5720]); the battery is the verdict, as always.
  The α sweep selected the best of three noisy arms, and the
  winner's-regression is visible: all three arms' 20-game
  CIs contain the 100-game result.

### 6.3 vs v0 (secondary) — COMPLETE 2026-11-21 ~19:04 UTC

`selfplay_batch.py --engine-a v49 --engine-b v0 --games 100
--seed-a 100 --seed-b 200` (no --openings; log
`ml/runs/ml5-h2/batt-v0.log`; JSON
`research/iterations/batch-v49-vs-v0.json`). Two VM
replacements interrupted the leg (at 12 and 84 games); both
resumes were clean.

- **37W-61L-2D, win rate 0.3800, Wilson 95% CI [0.2910,
  0.4779]** — v49 LOSES to the no-search anchor outright
  (the CI is entirely below 0.50). The weakest vs-v0 number
  of any recent ML version (v44 0.58, v37 0.585, v24 0.60,
  v42 0.44, v43 0.485). Uniqueness:
  **97/100**; de-dup cut 36W-59L-2D on 97 unique, 0.3814,
  CI [0.2910, 0.4809].
- Colour splits: as Black 26W-22L-2D (0.5400), as White
  **11W-39L-0D (0.2200)** — a severe White collapse against
  the heuristic anchor. Black won 65, White won 33. The
  whole deficit is White: v49's Black (0.54) is close to
  v44's Black (0.57) vs v0, while its White (0.22) is far
  below v44's White (0.59). The White-collapse problem
  (this batch's H1 topic) is v49's sharpest weakness — and
  the §4 holdout by-side numbers (White BETTER than Black,
  0.7590 vs 0.7045) predicted nothing about it, the fifth
  instance of holdout not predicting played strength.
- Note the sample spread across v49-vs-v0 measurements:
  fast loop 0.30 (10 games), this leg 0.38 (100 games).
  Both CIs overlap; the 100-game leg is the honest
  headline.

### 6.4 vs v7 (secondary) — COMPLETE 2026-11-21 ~20:58 UTC

`selfplay_batch.py --engine-a v49 --engine-b v7 --games 100
--seed-a 100 --seed-b 200` (no --openings; log
`ml/runs/ml5-h2/batt-v7.log`; JSON
`research/iterations/batch-v49-vs-v7.json`). Two VM
replacements interrupted the leg (at 90 and 95 games); both
resumes were clean.

- **46W-47L-7D, win rate 0.4950, Wilson 95% CI [0.3990,
  0.5914]** — v49 holds the heuristic bar reference
  essentially even, the best vs-v7 number of any ML version
  (v44 0.45, v37 0.45, v24 0.445). Uniqueness: **100/100**;
  de-dup cut identical.
- Colour splits: as Black 24W-22L-4D (0.5200), as White
  22W-25L-3D (0.4700) — balanced; no White gap against v7
  (contrast v37's White 0.31 and v44's White 0.40 vs v7).
  Black won 49, White won 44.
- Non-transitivity, stark: v49 ≈ v7 (0.495) and v7 beats v0
  (0.665), yet v49 loses to v0 (0.38) — the cycle v49 ≈ v7 >
  v0 > v49. v49's White collapse is opponent-specific: White
  0.22 vs v0, 0.39 vs v44, but 0.47 vs v7. The distilled net
  plays the strong heuristic opponent evenly and falls apart
  against the crude greedy one — the same inversion v7 itself
  showed vs v0 (0.665 vs v0, below its 0.705 vs v1), in the
  opposite direction.

### 6.5 vs v37 (secondary) — COMPLETE 2026-11-22 ~00:43 UTC

`selfplay_batch.py --engine-a v49 --engine-b v37 --games 100
--openings research/openings-ml4.json --seed-a 100 --seed-b 200`
(log `ml/runs/ml5-h2/batt-v37.log`; JSON
`research/iterations/batch-v49-vs-v37.json`). Two VM
replacements interrupted the leg (at 40 and 49 games); both
resumes were clean.

- **45W-46L-9D, win rate 0.4950, Wilson 95% CI [0.3990,
  0.5914]** — v49 holds v37 essentially even. The 4-ply net
  is NOT stronger than the 2-ply net head-to-head (the second
  branch of the §1 hypothesis: the label-depth lever
  saturates at 2 plies even at even parity), but it is also
  not weaker the way v44 was (v44 lost 0.31 to v37).
  Uniqueness: **99/100**; de-dup cut 44W-46L-9D on 99
  unique, 0.4899, CI [0.3937, 0.5869].
- Colour splits: as Black **31W-13L-6D (0.6800)**, as White
  **14W-33L-3D (0.3100)** — v49's Black beats v37's White
  decisively while its White loses. Black won 64, White won
  27. In both colours v49 is better against v37 than v44 was
  (v44: Black 0.46, White 0.16) — yet v49 loses to v44
  head-to-head (0.475). The ML-track cycle is now: v37 >
  v44 (0.69), v44 > v49 (0.525), v49 ≈ v37 (0.495).
  No single ranking of the three distilled nets is honest.

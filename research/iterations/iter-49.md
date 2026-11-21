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

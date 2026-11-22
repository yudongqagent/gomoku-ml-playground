# Iteration 56 — v56: parity-corrected 3-ply labels (ML batch 6, H3)

**v56 = v8's search + a value net trained on parity-corrected 3-ply
labels.** The ONLY difference from v44 is the label function: v44's
3-ply labels are replaced by their colour-swap-symmetrised form
(§1.1), which cancels the odd-horizon parity shift per position.
Everything else is v44's recipe exactly (same positions, same
architecture, same training recipe, same search). The battery is a
clean A/B of the label correction — and, against v37's numbers, of
whether the parity shift was what cost v44 played strength vs v37.

**Comparison targets:** v44 (ML champion, the bar reference: vs v8
0.67, ladder Elo 98.1, vs v0 0.58, vs v7 0.45, vs v37 0.31, swap vs
v7 0.39; tactical 20/20 + 29/29) and v37 (strongest ML version by
head-to-head: vs v8 0.75, Elo −34.9, vs v0 0.585, vs v7 0.45, vs
v44 0.69). Non-transitivity is the norm in the ML track — v56 is
measured against BOTH.

**The ML bar (reference v44, Elo 98.1):** (1) h2h vs v44 ≥100
games, win rate > 0.50 with Wilson 95% CI lower bound > 0.50;
(2) ladder Elo > 98.1; (3) tactical ≥ 20/20 + 29/29 on the
corrected suite; (4) all test suites green (ml pytest, npm test,
tsc).

**Pre-registration note:** §1 below was written and committed
BEFORE any eval, any label measurement, or any training run. The
correction choice (§1.1) is justified on principle, a priori — the
§0 verification measurements (run after the §1 lock) check the
IMPLEMENTATION of the pre-registered design; a failed check is a
bug to fix, not a reason to change the design.

**Task-premise correction (verified 2026-11-22, before
pre-registration):** the tasking stated the standard distill
positions were gone ("all distill game records and shards have
been cleaned; only checkpoints survive") and ordered
regeneration of ~817 fresh games. That premise is FALSE. The
games files exist, intact, in four worktrees
(`gomoku-ml1-h1`, `gomoku-ml3-h2`, `gomoku-ml4-h3`,
`gomoku-ml5-h2`), byte-identical across all four (md5
`0e71ec51…` / `b383a700…` / `5a5eaee8…` for v7/v18/random) and
matching iter-24 §2's published fingerprint EXACTLY (v7 488
games / 25,277 plies, seeds 50000–50487; v18 229 / 11,511,
seeds 60000–60228; random 100 / 11,009, seeds 70000–70099).
v44's labelled shards (`gomoku-ml4-h3/ml/runs/ml4-h3/data/`,
817 shards: 440+48 / 207+22 / 90+10) and v37's
(`gomoku-ml3-h2/ml/runs/ml3-h2/data/`, same counts) are intact
too. Regenerating with v24's seeds would reproduce the same
games exactly (the engines are deterministic given seeds — a
3-game replay check is recorded in §2 as proof). So this
version trains on v24's ORIGINAL games, re-labelled — the
comparison to v44 is on IDENTICAL positions (only the label
function differs), exactly like v37-vs-v24 and v44-vs-v37, and
the "comparison across different game sets" caveat the tasking
asked for does NOT apply. This is strictly better science than
the ordered regeneration (which would have spent ~7h of VM time
to produce byte-identical files), and it is flagged here and in
the final report so the coordinator can correct the record.
(H2's premise may be equally stale: v48's patch data also
exists at `gomoku-ml5-h1/ml/runs/ml5-h1/` — `data-base`,
`data-new`, `data-train`, `games-vs-*.json`.)

## 0. Ports and verification measurements (after the §1 lock)

**Ports (v44 was merged to master, so this worktree already
carries the ML batch 4 code — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: the memmap version, on master since
  the v44 merge — no port needed.
- `ml/scripts/generate_distill_data.py`: v44's version with
  `--label-mode rollout3`, on master — extended with
  `--label-mode rollout3sym` (see §1.1; the diff is 3 small
  hunks: docstring, label-fn selection, CLI flag; the defaults
  `rollout`/`rollout2`/`rollout3` reproduce v24's/v37's/v44's
  labels exactly).
- `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/scripts/sweep_alpha.py`, `ml/scripts/find_losing_moves.py`,
  `ml/scripts/eval_value_holdout.py`: on master since the v44
  merge — no port needed. `eval_value_holdout.py` carries the
  coordinator's side-to-move fix (commit `038232c`: read plane
  3, not index parity — the ML batch 5 finding; the fixed
  numbers should reproduce the known White gap on v44's
  holdout).
- `ml/gomoku_ml/mcts_heuristic.py`: `_rollout2_leaf_value` and
  `_rollout3_leaf_value` on master, plus the new
  `_rollout3sym_leaf_value` (§1.1). The rollout-leaf teacher
  (v18 code) is on master since the batch-9 merge — no
  cherry-picking needed, and no re-verification of the leaf
  port is needed either (v24's 401/401 bit-for-bit check
  stands; the teacher code is the same master code v24
  verified against).
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port on master. **Checked: `ValueMCTS.__init__` has no
  duplicate parameters** (the silent auto-merge trap —
  verified by inspection of the signature after the edit).
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS` (on master), plus the v56 registry entry
  (`value_checkpoint: ml/runs/ml6-h3/value-net.pt`,
  `value_weight: 0.25` placeholder until the §1.4 sweep sets
  it). The v44, v37 and v8 registry entries are on master.
- `ml/configs/ml6-h3.yaml`: v44's `ml4-h3.yaml` with run
  name/dir changed (and the iter-44 reference updated to
  iter-56).
- `ml/baseline.py` is FROZEN and untouched.

**Checkpoint copies (gitignored data, md5-verified):** v44
`ml/runs/ml4-h3/value-net.pt` (md5
`d22af606f7d2ee0254f372da895ae7ad`, matches the task and
iter-44); v37 `ml/runs/ml3-h2/value-net.pt` (md5
`2fde492aae6d44cbe945e15bd5f1f959`, matches the task and
iter-37); v8 `ml/runs/b3-h2/value-net.pt` (md5
`5deb26f3df67883f24f04800d5356da6`, matches iter-37/44).
Games files copied verbatim from `gomoku-ml1-h1`
(`games-{v7,v18,random}.json`, md5-verified against the source
— see the task-premise correction above).

**Port smoke checks:** v44, v37 and v8 all load and play in this
worktree (empty-board move 112 each — the centre, as expected);
v56 is registered (no checkpoint yet — training produces it);
`_rollout3sym_leaf_value` runs (empty board: 0.0, as the
colour-symmetry argument requires).

**Verification measurements (implementation checks of the
pre-registered §1.1 design — `/tmp/diag_v56_s0.py`, 500 quiet
positions from the v7 games, the SAME sample as iter-44 §0 —
confirmed by the 1/2/3-ply numbers reproducing iter-44 §0
EXACTLY: means −0.0627 / −0.0560 / +0.0999, fraction positive
0.212 / 0.234 / 0.734, pure-gradient mean(v3−v2) = +0.1236 with
v3 > v2 on 99.4%, sign agreement 3-ply-vs-2-ply 0.4920):**

### §0.1 Mirror identity — 0/500 mismatches

`L3(−B, p) == L3(B, −p)` on all 500 sampled positions (tolerance
1e-9). The labeller IS mirror-symmetric as derived in §1.1, so
the symmetric form cancels the shift as designed. **The §1.5
stop condition (mirror-identity failure) does NOT trigger.**

### §0.2 Terminal handling — wins are halved, losses preserved

| L3 value | n | preserved (±1) | near-half (≈±0.5) | mean\|L3sym\| |
|---|---|---|---|---|
| +1 | 32 | 0.000 | 0.812 | 0.351 |
| −1 | 18 | 0.833 | 0.167 | 0.932 |

The asymmetry is structural and was NOT predicted in §1.1 (only
the halving mechanism was): rollout WINS almost always depend on
moving first — the side to move sets the win up with their next
move, and the opponent moving first disrupts it — while rollout
LOSSES are robust, because they come from the opponent's
EXISTING threats, which do not depend on move order. So the
symmetric label halves essentially every +1 (to ~+0.35) while
preserving most −1s. The halved wins still top the scale
(+0.35 vs gradients ≤ 0.2), so the ranking "winning >
threatening > neutral" is preserved — but the win signal is
compressed, exactly as the §1.1 known limitation warned (the
fraction halved is now measured: 100% of +1s, 17% of −1s).
The battery is the arbiter of whether the compression costs
play. The ±0.25 must-block values are gone entirely (fraction
0.000 vs 3-ply's 0.130) — halved into gradient-range values,
as §1.1 predicted.

### §0.3 Shift cancellation — 70% of the gradient gap removed; the mean prediction is REFUTED

| Label | mean | mean\|v\| | frac > 0 | ±1 frac |
|---|---|---|---|---|
| 1-ply | −0.0627 | 0.1469 | 0.212 | 0.048 |
| 2-ply | −0.0560 | 0.1881 | 0.234 | 0.086 |
| 3-ply | +0.0999 | 0.2056 | 0.734 | 0.100 |
| **3-ply-sym** | **−0.1292** | 0.1979 | **0.242** | 0.030 |

Pure-gradient pairs (|v| < 0.2 both): mean(vsym−v2) = **+0.0369**
(median +0.0335, vsym > v2 on 71.2%), vs mean(v3−v2) = +0.1236
(99.4%). **Symmetrisation removes 70% of the 3-ply-vs-2-ply
gap on gradients.** Sign agreement with 2-ply is 0.7360 (vs
0.4920 for raw 3-ply). Pearson r: sym-vs-2-ply 0.6893,
sym-vs-3-ply 0.7509, 3-ply-vs-2-ply 0.8865 — the symmetric
function is a genuinely different target from both, closer to
2-ply in distribution (mean −0.13 vs −0.06, frac>0 0.24 vs
0.23, mean|v| 0.20 vs 0.19) than to 3-ply.

**The §1.1 prediction "mean ≈ 0, fraction positive ≈ 0.5" is
REFUTED.** The symmetric label centres at −0.1292, not 0. The
reason is §0.2's asymmetry: the tempo-neutral value
`(V(B,p) − V(B,−p))/2` is NOT centred at 0 on this position
distribution, because wins are first-move-dependent (halved)
while losses are not (preserved) — the terminal contribution
drags the mean down by ~0.11 relative to the gradients. On
gradients alone the symmetric label sits +0.037 ABOVE 2-ply,
which is the tempo-neutral-vs-pessimistic convention gap (the
2-ply label includes the opponent-just-moved pessimism; the
symmetric label does not), not uncancelled shift. Net
assessment: the correction does what §1.1 designed it to do
on gradients (the harmful non-uniform part of the shift is
largely cancelled) and introduces a known, measured
pessimistic tilt on terminals. Whether the trade helps play is
what the battery measures — the design stands (no §1.5 stop
condition triggers).

### §0.4 Cost — median 88.95 ms, mean 95.72 ms per label

Measured under contention (sibling ML-batch-6 workers running
— iter-49 noted the same inflation vs iter-44's unloaded
measures, so absolute ms are not comparable across batches;
the RATIO is what matters: ~2× the 3-ply label, as designed).
Projected full set (37,227 positions): ~59 min — far under the
2-hour bar. **No dataset-size compromise is needed: v56 labels
v24's FULL position set** (the pre-registered subset rule is
therefore moot but recorded anyway).

**The setup-v1v0s-g92 diagnostic (the motivating position):**
reproduces v44's numbers exactly at the root (1-ply +0.1290,
2-ply +1.0000, 3-ply +1.0000) and extends them: the symmetric
root label is **+0.3750** — the setup win IS first-move-
dependent (White moving first disrupts it), so it is halved,
as §1.1's limitation predicted; it still reads as a clear win
(well above the 0.2 gradient cap). At the children (from
Black's perspective): (9,6) is +1.0000 and (5,4) is +1.0000 —
once the setup stone is placed, the win is first-player-
robust, so both children preserve exactly. The symmetric
labels do NOT improve child-level discrimination at g92 over
the 2-ply/3-ply labels (both children +1.0, as at every depth
≥ 2) — the same honest consequence as v37's and v44's: the
labels teach the win at the PRE-setup position, not the child
ranking. For the tactical suite, v56's net must NOT invert
the order (as v31's did); if it values (9,6) ≈ (5,4), the v1
prior (which ranks (9,6) FIRST, 0.1122) breaks the tie
correctly. The clause-3 run of setup-v1v0s-g92 is the
pre-registered mechanism check (§6.1).

## 1. Design and pre-registration

**The hypothesis (plan H3):** v44 proved a 3-ply teacher beats
the ladder (Elo 98.1, first ML version ever to beat the ladder
MCTS outright) but plays WORSE than v37 head-to-head (0.31 vs
v37, vs v37's 0.75 vs v8). The 3-ply labels carry a systematic
odd-horizon parity shift (iter-44 §0: mean +0.0999 on 500
positions, +0.1094 on the full set, vs 2-ply's −0.0560;
fraction positive 0.734 vs 0.234; pure-gradient pairs
mean(v3−v2) = +0.1236, v3 > v2 on 99.4%). The 4-ply reversal
(iter-49 §0: mean −0.0630, fraction positive 0.216, sign
agreement with 2-ply 0.828) proves the shift is a function of
horizon PARITY, not depth — and v49's rejection (0.475 vs v44,
Elo −137.0) proves that merely flipping the parity back does
not recover v37-level play either. This version tests whether
REMOVING the shift from the 3-ply labels — keeping the 5-ply
horizon but cancelling the parity distortion — recovers
v37-level played strength while keeping v44's Elo. If it does,
the parity shift was the cost; if it plays at v44's level, the
shift was not the binding constraint (and the 3-ply horizon
itself, or the distillation of a 5-ply function into a static
net, is the cost).

**Why the shift should hurt play (the mechanism, stated before
any measurement):** if the shift were a pure constant it would
barely matter — MCTS decisions are driven by value
DIFFERENCES, and a uniform shift cancels in the backup. The
harm is the shift's NON-UNIFORM part: the shift is "one greedy
move's worth of attack-breadth gain", so it is large exactly
where our greedy move creates threats and small in quiet
positions. The 3-ply label therefore systematically overvalues
positions where WE have a strong greedy move — double-counting
our attacking potential (the greedy move's gain is counted
once in the rollout, and the leaf then evaluates the resulting
position's inflated breadth) — while the rollout's greedy
(rather than minimax) replies miss refutations. The net learns
to prefer lines where we keep making greedy attacking moves,
even when they are refuted. A correction must cancel the shift
PER POSITION, not just on average — which is what decides the
choice below.

### 1.1 The parity-corrected labelling procedure (pre-registered, exact)

**Correction chosen: (b) the symmetric label.** (a) centering
was rejected on principle BEFORE any measurement: subtracting
a constant removes only the shift's constant part — the part
that barely matters for play — and leaves the harmful
non-uniform part untouched; it also needs a fitted constant
and corrupts terminals (+1 → +0.89, −1 → −1.11, out of range)
unless terminals are special-cased. The symmetric label
cancels the shift per position, is parameter-free, and (as
shown below) preserves terminals exactly whenever the forced
win is first-player-robust.

**Definition.** The teacher is **v18's rollout leaf**
(`_rollout_leaf_value`, blend=False) — the SAME teacher as
v24, v37 and v44, for comparability. Let `L3(B, p)` denote
`_rollout3_leaf_value(B, p)` (iter-44 §1.1: 3 plies of
rollout-policy play + the 2-ply leaf = 5-ply horizon, from the
side-to-move perspective). The symmetric 3-ply label for a
quiet position (`board`, `player` to move) is:

```
L3sym(B, p) = ( L3(B, p) − L3(−B, p) ) / 2
```

where `−B` is the board with colours swapped (BLACK↔WHITE,
EMPTY unchanged — board negation, since BLACK=+1, WHITE=−1),
with the SAME side to move `p`. In words: the average of the
3-ply label and its negation under colour swap, exactly as the
plan states it. Implemented as `_rollout3sym_leaf_value` in
`ml/gomoku_ml/mcts_heuristic.py`, wired as `--label-mode
rollout3sym` in `generate_distill_data.py`.

**Why this cancels the shift.** `L3(B, p)` ≈ `V(B, p) + s(B,
p)`, where `V` is the position's value and `s` ≈ +0.12 is the
parity shift — optimism for the side to move, because the
leaf is evaluated after OUR greedy move. The colour-swapped
board with `p` still to move is the mirror image of the
position `(B, −p)` (same board, opponent to move), so by the
mirror symmetry of the game and the labeller,
`L3(−B, p) = L3(B, −p)` (this identity is CHECKED in §0.1 —
it must hold on every sampled position). And `L3(B, −p)` ≈
`V(B, −p) + s(B, −p)` — the SAME shift `s`, now favouring the
opponent. So `L3sym(B, p)` ≈ `(V(B, p) − V(B, −p)) / 2 +
(s(B, p) − s(B, −p)) / 2`: the shift cancels to first order
AT EACH POSITION (the two shift terms are the same greedy-gain
mechanism applied to opposite sides), leaving the
tempo-neutral value — the average of the position's value with
each side to move. The mean of the corrected labels should be
≈ 0 (vs +0.10 for 3-ply, −0.06 for 2-ply) and the fraction
positive ≈ 0.5 (vs 0.734) — CHECKED in §0.3 on the full label
set.

**The degenerate reading (excluded explicitly).** Swapping
BOTH the board colours AND the side to move would give
`L3(−B, −p)`, which equals `L3(B, p)` exactly by mirror
symmetry — so `(L3(B, p) − L3(−B, −p)) / 2` is identically
zero for every position. That reading is NOT what is meant
and is NOT implemented. The side to move is KEPT; only the
board colours swap.

**Terminal preservation (derived, checked in §0.2).** For a
position where the 3-ply rollout finds a forced win for the
side to move that is ROBUST to who moves first (the opponent's
first move cannot disrupt it — e.g. an open four at the root,
or a double threat the rollout reaches regardless),
`L3(B, −p) = −1` (from the opponent's perspective the
position is lost even moving first), so `L3sym = (+1 −
(−1))/2 = +1` — preserved exactly. The same holds with signs
reversed for forced losses. **Known limitation
(pre-registered):** a forced win that DEPENDS on moving first
(the opponent's first move disrupts it — e.g. a single open
three the opponent can block) is HALVED: `L3(B, p) = +1` but
`L3(B, −p)` is a gradient value, so `L3sym ≈ +0.5`. The
ranking is preserved (a halved win at ~+0.5 still tops the
gradient scale, |v| ≤ 0.2, and the ±0.25 must-block values),
so the net can still learn "winning > threatening > neutral" —
but the win signal is compressed, and the fraction of ±1
labels that are halved is MEASURED in §0.2 (not gated — it is
a property of the label, and the battery is the arbiter of
whether the compression costs play). The ±0.25 must-block
values are likewise roughly halved (`(0.25 − g)/2 ≈ +0.1`) —
note the 2-ply and 3-ply ±0.25s NEVER coincide (iter-44 §0),
so neither depth's ±0.25 is "correct" in an absolute sense
and the symmetric middle is a principled value.

**Perspective:** the side to move throughout, in [−1, 1],
exactly like v24's, v37's and v44's labels. **Horizon:** 5
plies, exactly like v44's — the ONLY change from v44 is the
symmetrisation. **Cost:** 2× the 3-ply label (~24 ms/position
expected — CHECKED in §0.4; the full 37,227-position set
should take ~15–25 min per source set, ~50 min total, far
under the 2-hour bar).

### 1.2 Dataset (pre-registered)

- **Source: v24's engine-game positions, re-labelled.** NO new
  game generation: the games exist intact (see the
  task-premise correction above) and game generation was the
  expensive step. The positions are EXACTLY v24's, v37's and
  v44's: the 817 games in `ml/runs/ml6-h3/games-{v7,v18,
  random}.json` (488 v7 self-play at 50 sims, seeds 50000+;
  229 v18 self-play at 50 sims, seeds 60000+; 100 random
  playouts, seeds 70000+), copied md5-verified from
  `gomoku-ml1-h1`. Expected yield: **37,227 quiet positions**
  (21,595 v7 / 9,932 v18 / 5,700 random), the same positions
  v24, v37 and v44 trained on — the ONLY change from v44 is
  the label function, so the v56-vs-v44 comparison isolates
  the labelling change. (The tasking's ordered regeneration
  would have reproduced these same games — see §2's replay
  check.)
- **Sampling:** ALL quiet positions of every game (labelling
  is ~24 ms/position expected; §0.4 — the full run is ~50
  min, so no subset is needed). A position is quiet iff
  `_terminal_leaf_value == 0.0`. Positions are encoded with
  the real game history (boards 2/4 plies back, last move)
  exactly as in training (`encoding.encode`). Policy targets
  are one-hot on the move actually played (unused at
  policy_weight=0).
- **Labels:** `_rollout3sym_leaf_value` (§1.1) via
  `generate_distill_data.py label --label-mode rollout3sym`.
  Rollout-found terminals are kept as the symmetric form
  computes them (§1.1: preserved when first-player-robust,
  halved when first-move-dependent — no special-casing).
- **Holdout (game-level, v24's protocol, IDENTICAL split):**
  every 10th game of each source is routed to a `*-holdout`
  dir and NEVER trained on — the same games v24, v37 and v44
  held out (48 v7 / 22 v18 / 10 random = 80 games, 3,378
  positions), so v56's holdout numbers are comparable to
  v24's, v37's and v44's BY POSITION (only the labels
  differ). Expected train set ≈ 33,849 positions.
- **NOT used for training:** the B1 10k games (B1 data
  collapses White — v25, v31 — and v24's engine-only data did
  not; this workstream does not re-open that lever). They are
  NOT used even as a diagnostic here.
- **Variety check:** not needed — the games are v24's, whose
  variety check passed (14 distinct first-5-ply prefixes in 15
  v7 games).
- **Stopping rule:** labelling is ~50 min for the full set
  (§0.4); no cutoff is needed. If a VM replacement kills the
  labelling run, the label phase is resume-safe (existing
  shards skipped) — re-run the same command.
- **Subset rule (pre-registered, expected moot):** had the
  full run projected over ~2 hours, labelling would have used
  every 2nd quiet position of every game (deterministic,
  stratified by game source — the same fraction of each
  source), recorded in the LABEL_JSON. It is not expected to
  be needed.

### 1.3 Architecture and training (pre-registered)

- **Architecture:** v24's/v37's/v44's ResNet **16ch/2 blocks**
  (126,759 params), trained from scratch (random init — NOT
  from v8's, v24's, v37's or v44's checkpoint; the labels are
  a different target). Do NOT go larger: v31 proved 4x
  capacity plays worse on distilled labels (the scale lever is
  CLOSED).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes — v24's/v37's/v44's recipe exactly),
  augmentation on (dihedral), val_fraction 0.05 (random split
  within the train set, for best-checkpoint selection only),
  seed 0.
- **Config:** `ml/configs/ml6-h3.yaml` (v44's `ml4-h3.yaml`
  with run name/dir changed); run dir
  `ml/runs/ml6-h3-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, B1's finding).
  `checkpoint_every_epochs: 1` from the start (v24's resume
  lesson).
- **Memmap path (mandatory):** the train shards are bundled
  with `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml6-h3/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as
  v24/v37/v44):** best-by-internal-val-loss (`best.pt`),
  copied to `ml/runs/ml6-h3/value-net.pt` (gitignored data,
  the path v56's registry entry points at).
- **Resume rule (pre-registered, v24's lesson):** if training
  is killed, resume from the latest per-epoch checkpoint with
  `--resume` and the REMAINING epochs (cosine schedule spans
  the full 15 via `schedule_epochs`); back up the pre-kill
  `best.pt` first, because the resumed run resets
  `best_val_loss` and would overwrite it at the next epoch
  regardless — the final best is chosen by comparing the
  resumed run's best val loss against the pre-kill best.

### 1.4 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position
value) is swept over **{0.0625, 0.125, 0.25}**, **40 games each
vs v44** (the bar reference), seeds 1001/2001 (fast-loop
seeds), via `ml/scripts/sweep_alpha.py --version 56 --opponent
v44 --games 40`. The grid brackets v44's winner (0.125) BELOW
and v37's winner (0.25) above: the symmetric labels are
cooler than v44's 3-ply labels (tempo removed, first-move-
dependent wins halved — mean|v| expected slightly below
3-ply's ~0.21, closer to 2-ply's ~0.19), so the winner may sit
at either end. 0.5 is NOT on the grid: v44's sweep already
showed 0.5 losing (0.475) on HOTTER labels, and re-testing it
on cooler labels spends 40 games on a dominated arm. **40
games per arm** (not 20): ML batch 5 showed 20-game sweeps
rank noisily (winner's regression 0.55 → 0.475 at 100 games).
Pick the best win rate; tie-break: the smaller α
(conservative — less optimizer's-curse exposure). **No-signal
rule:** if all three arms are within one win of each other,
pick α=0.125 (v44's winner — the closest prior, same 5-ply
horizon) and say so — the sweep found no signal. The sweep
SELECTS α; the 100-game battery is the verdict (standing
lesson: small-sample smokes mislead). The registry entry's
`value_weight` is updated to the winner after the sweep.
(Note: α is an inference-time weight, so holdout sign accuracy
cannot select it — the sweep is by played win rate, as in
v24/v31/v37/v44/v49.)

### 1.5 What would change the design

- If the distilled net's holdout sign-accuracy on its OWN
  (symmetric-label) holdout is at chance (~0.50), the
  distillation failed to learn the corrected label function —
  stop and report, do not battery a chance-level net.
- If the α sweep's best is within noise of the worst (all
  three within one win of each other), pick α=0.125 per the
  §1.4 no-signal rule.
- If the §0.1 mirror-identity check FAILS (L3(−B, p) ≠
  L3(B, −p) on sampled positions), the labeller is not mirror-
  symmetric as derived — stop and diagnose before labelling
  (the symmetric form's justification depends on it). A
  float-tolerance mismatch (≤1e-6) is not a failure.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them. A retrain re-runs the α
  sweep and the battery from scratch on the new form.

### 1.6 Holdout diagnostics (pre-registered)

All via `ml/scripts/eval_value_holdout.py` (sign accuracy +
MSE, full-planes and no-history views) on symmetric labels
unless noted, using the FIXED side-to-move inference
(coordinator commit `038232c` — read plane 3, not index
parity; iter-37/iter-44 by-side numbers are unreliable for
this reason, so v56's by-side numbers are the first
trustworthy ones on this protocol). **Parity caveat (§0):
cross-label sign agreement is NOT a fit measure where the
labels' centres differ** — the 3-ply labels centre near +0.10
and the symmetric labels near 0, so sign agreement between
them is confounded. For cross-label comparisons (b)–(d),
Pearson r (shift-invariant) is the primary measure; sign
agreement is reported with the caveat. For (e) (v37's 2-ply
net, centre −0.06) the same caveat applies, milder.
- (a) **v56 net on its own holdout** (80 games, 3,378
  positions) — in-distribution; the §1.5 stop rule applies
  here (sign accuracy vs chance). ALSO report accuracy BY SIDE
  TO MOVE (Black/White) on both the full-planes and the
  no-history views — the White-collapse check (v30's method,
  now with the fixed inference): the symmetric labels are
  tempo-neutral BY CONSTRUCTION, so a White gap here would be
  a net-side defect, not a label-side one. ALSO report the
  no-history view (v37's net collapsed without history planes
  on engine positions, 0.7395 → 0.5163; v44's did not, 0.007
  drop — check whether the symmetric net inherits v44's
  robustness).
- (b) **v44's net** (`ml/runs/ml4-h3/value-net.pt`) **on the
  same symmetric labels** — same positions, cross-net
  comparison: how much of the corrected function does the
  3-ply-trained net already capture? (Pearson r primary.)
- (c) **v56 net on v44's 3-ply labels** (v44's holdout shards,
  `gomoku-ml4-h3/ml/runs/ml4-h3/data/{v7,v18,random}-holdout/`)
  — the reverse transfer: does the corrected net still
  reproduce the uncorrected function? (Pearson r primary.)
- (d) **v37's net** (`ml/runs/ml3-h2/value-net.pt`) **on the
  symmetric labels** — the 2-ply-trained net on the corrected
  3-ply function: if the symmetric labels are close to the
  2-ply function (same centre, deeper horizon), v37's net
  should transfer better here than on raw 3-ply labels (where
  its pooled sign was 0.3666, below chance). (Pearson r
  primary.)
- (e) **v8's net on the symmetric holdout** — baseline (expect
  ~0.50 sign, ~0 r, as on v24's/v37's/v44's labels).
No decision hinges on (b)–(e); they diagnose whether the
corrected labels were learned and whether the fit transfers
across label functions. **The battery is the verdict, as
always** (holdout does not predict played strength — now six
instances: v31, v38, v42, v43, v48, v49).

### 1.7 Eval seeds (pre-registered)

h2h vs v44: v56=100 / v44=200; h2h vs v0: v56=100 / v0=200;
h2h vs v7: v56=100 / v7=200; h2h vs v37: v56=100 / v37=200;
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

Tactical (clause 3) → vs v44 (clause 1) → ladder (clause 2)
→ vs v0 → vs v7 → vs v37 → swap vs v7 → test suites (clause
4). Fast loops (10 games vs v0 AND vs v44, seeds 1001/2001,
at the swept α) run BEFORE the battery, after the α sweep.
Every leg reports its uniqueness audit (distinct games /
games played) and its de-dup rate (win rate recomputed on
distinct games only), prominently. Clause 1 is judged on the
pre-registered nominal rule (Wilson CI lower > 0.50 on all 100
games); the de-dup cut is reported alongside (v44/v52
precedent: a nominal pass with a de-dup miss is recorded with
the caveat, not hidden).

## 2. Data generation (actuals)

**No regeneration was needed** (see the task-premise correction in
the header): v24's original games exist intact. The games files
were copied md5-verified from `gomoku-ml1-h1` (v7
`0e71ec51…`, v18 `b383a700…`, random `5a5eaee8…`) and
re-labelled with `--label-mode rollout3sym` (chain:
`ml/runs/ml6-h3/chain-label.sh`, 2026-11-22 15:20–16:22 UTC,
~62 min total — no VM interruptions).

**Replay check (proves regeneration would have reproduced the
same games):** `generate_distill_data.py games --source v7
--games 3 --sims 50 --seed 50000` replayed games 0–2 of
`games-v7.json` with IDENTICAL move lists and winners (23/26/30
plies). The engines are deterministic given seeds, so the
ordered regeneration would have produced byte-identical files
after ~7h of VM time. The copy is equivalent to regeneration,
verified.

| Source | Games | Plies | Quiet positions | ±1 labels | Gradient labels | Held-out games |
|---|---|---|---|---|---|---|
| v7 self-play (50 sims, seeds 50000+) | 488 | 25,277 | 21,595 (19,545 train / 2,050 holdout) | 586 (2.7%) | 21,009 | 48 |
| v18 self-play (50 sims, seeds 60000+) | 229 | 11,511 | 9,932 (9,142 train / 790 holdout) | 501 (5.0%) | 9,431 | 22 |
| Random playouts (seeds 70000+) | 100 | 11,009 | 5,700 (5,162 train / 538 holdout) | 585 (10.3%) | 5,115 | 10 |
| **Total** | **817** | **47,797** | **37,227 (33,849 train / 3,378 holdout)** | **1,672 (4.5%)** | **35,555** | **80** |

Game counts, ply totals and quiet-position counts match iter-24
§2 EXACTLY (same games, same quiet definition). The ±1 fraction
fell from v44's 7.9% (2,954) to 4.5% (1,672) — the §0.2
halving: most rollout wins became gradient-range values
(~+0.35). The drop is largest on random positions (28.5% →
10.3%), where first-move-dependent wins dominate.

**Label distribution (full set, pooled):** mean −0.1039 (vs
3-ply's +0.1094, 2-ply's ≈ −0.06), mean|v| ≈ 0.19, fraction
positive ≈ 0.24 (vs 3-ply's 0.734). Per split: v7 train mean
−0.1136 / mean|v| 0.1729 / frac>0 0.2337; v18 train −0.1282 /
0.1978 / 0.2276; random train −0.0269 / 0.2369 / 0.3512;
holdouts within 0.005 of their train splits. The symmetric
labels are pessimistic overall (§0.3's refuted mean prediction,
confirmed on the full set) and closest in distribution to the
2-ply labels.

**Shard-equivalence check (`/tmp/check_shards_v56.py`):** every
shard's position count matches v44's shards EXACTLY (37,227
total, 0 bad per-shard counts across all 6 splits), and
planes/policy/moves/winner are bit-identical on 20 sampled
shards — the positions are v44's positions; ONLY the values
differ (mean |diff| 0.3295 on the sampled shard). The
v56-vs-v44 comparison isolates the label function, as designed.

## 3. Training (actuals)

(to be filled in.)

## 4. Holdout diagnostics (actuals)

(to be filled in.)

## 5. α sweep and fast loops

(to be filled in.)

## 6. Validation battery

(to be filled in.)

## 7. Verdict

(to be filled in.)

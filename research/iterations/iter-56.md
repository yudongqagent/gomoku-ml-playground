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
`data-new`, `data-train`, `games-vs-*.json`.) Corroborated
independently 2026-11-23 by the coordinator: H2 reconstructed
the base games VERBATIM from v37's shards at
`gomoku-ml3-h2/ml/runs/ml3-h2/data/` (same counts as above)
with `ml/scripts/reconstruct_distill_games.py` on
`research/ml-batch-6-h2`, and its verification gate —
re-labelling with rollout2 and requiring 0 mismatches against
the stored values — passed 0/37,227. The coordinator's
data-source correction therefore changes nothing for v56: this
version already trained on the verbatim original positions.

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

**Bundle:** 737 shards, 33,849 samples → `ml/runs/ml6-h3/
data-mmap/` (memmap path, as pre-registered). **Training:**
15 epochs on the bundle, 2026-11-22 16:28–17:06 UTC (~38 min),
no interruptions, no resume needed. Config `ml6-h3.yaml`
(v44's recipe exactly).

| Epoch | Train loss | Val loss | Epoch | Train loss | Val loss |
|---|---|---|---|---|---|
| 1 | 0.0812 | 0.0926 | 9 | 0.0644 | 0.0719 |
| 2 | 0.0797 | 0.0924 | 10 | 0.0612 | 0.0706 |
| 3 | 0.0790 | 0.0954 | 11 | 0.0596 | 0.0698 |
| 4 | 0.0785 | 0.0929 | 12 | 0.0580 | 0.0691 |
| 5 | 0.0766 | 0.0944 | 13 | 0.0568 | 0.0678 |
| 6 | 0.0736 | 0.0827 | 14 | 0.0557 | 0.0677 |
| 7 | 0.0695 | 0.0785 | 15 | 0.0555 | **0.0674** |
| 8 | 0.0668 | 0.0748 | | | |

Val loss falls monotonically from epoch 6; the best is the
FINAL epoch (15, val_loss 0.06737 — still improving at the
end, so 15 epochs did not overfit). Final train loss 0.0555.
**Checkpoint selection (pre-registered rule):**
best-by-val-loss = epoch 15. `ml/runs/ml6-h3/value-net.pt`
(md5 `78e3e71f6098cb2fe3269d178b6cff64`) was written by
train.py's `--out`; its WEIGHTS are bit-identical to
`best.pt` and `final.pt` (max abs diff 0.0 across all 48
tensors; all epoch 15, step 7545) — the md5s differ only in
payload metadata (`--out` adds `config`/`history` keys). The
pre-registered rule is satisfied: the shipped checkpoint IS
the best-by-val-loss net.

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th game, never trained on — the SAME
80 games / 3,378 positions v24, v37 and v44 held out), targets =
symmetric labels unless noted. v56 net =
`ml/runs/ml6-h3/value-net.pt`; v44 net =
`ml/runs/ml4-h3/value-net.pt`; v37 net =
`ml/runs/ml3-h2/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml6-h3/holdout.log`; Pearson r via
`/tmp/holdout_r_v56.py`. Sanity check: this run reproduces
v37's published 2-ply numbers EXACTLY on its own holdout (v7
0.7395, v18 0.8436, random 0.6623 — full planes; no-history
0.5163/0.5846/0.6604; MSEs 0.0781/0.0819/0.2675) — the eval
setup is sound. By-side numbers use the FIXED side-to-move
inference (coordinator commit `038232c` — plane 3, not index
parity); these are the first trustworthy by-side numbers on
this protocol (iter-37/iter-44's are unreliable for this
reason).

### (a) v56 net on its own symmetric holdout (in-distribution)

| Holdout | n pos | v56 sign (full) | v56 sign (no-hist) | v56 MSE | v56 mean\|v\| | v44 sign (full) | v37 sign (full) | v8 sign (full) |
|---|---|---|---|---|---|---|---|---|
| v7 | 2,050 | 0.7392 | 0.3748 | 0.0485 | 0.1306 | 0.3137 | 0.7185 | 0.5428 |
| v18 | 790 | 0.7239 | 0.3700 | 0.0700 | 0.1396 | 0.3378 | 0.7319 | 0.5054 |
| random | 538 | 0.5889 | 0.5843 | 0.1593 | 0.0924 | 0.4042 | 0.5173 | 0.5312 |
| **pooled** | **3,378** | **0.7138** | **0.4037** | — | — | **0.3327** | **0.6929** | **0.5319** |

**Verdict: the distillation learned the corrected label
function — the §1.5 stop condition (chance-level holdout) does
NOT trigger.** Pooled in-distribution sign agreement 0.7138 vs
0.50 chance. It is slightly LOWER than v44's 0.7442 on 3-ply
labels and v37's 0.7519 on 2-ply labels — the symmetric
function is harder to learn (the halved wins compress the
signal the net must fit), as expected. Per-source, v56 is
about even with v44 on v7 (0.7392 vs 0.7460) and v18 (0.7239
vs 0.7163) but much worse on random (0.5889 vs 0.7786) — the
inversion of v44's pattern (random was v44's BEST source):
random positions carry the most halved wins (28.5% → 10.3%
±1), and the net underfits that noisier target. No decision
hinges on it.

**The v56 net is HISTORY-DEPENDENT — catastrophically so on
engine positions.** The no-history view drops 0.310 pooled
(0.7138 → 0.4037): v7 0.7392 → 0.3748, v18 0.7239 → 0.3700
— BELOW chance on both engine sources — while random is
robust (0.5889 → 0.5843). This is WORSE than v37's collapse
(0.195 pooled, to 0.5163/0.5846 — still above chance) and the
reverse of v44's robustness (0.007 drop). Mechanism
(hypothesis): the tempo-neutral label removes the tempo
information from the target, so the net routes its value
judgment through the history planes (last move, boards 2/4
plies back) to reconstruct the game context the current board
alone no longer determines — the optimizer grabbed the
history shortcut harder because the symmetrised target needs
it more. Or it is luck of the optimisation (different target,
different init). Either way it is a measured fact, and it
re-opens the §4 warning sign that v44 had closed: at
inference the net ALWAYS has history planes (real games), so
the collapse costs nothing directly — but it means the net's
value function is not a function of the current position
alone, and any deployment that ever lacks history (or any
opponent that plays off-distribution openings) leans on a
brittle feature. The battery is the arbiter (v37 played the
best despite its collapse; holdout does not predict played
strength — six instances).

**By side to move** (v56 net, symmetric holdout; FIXED
inference):

| Holdout | Black (full) | White (full) | Black (no-hist) | White (no-hist) |
|---|---|---|---|---|
| v7 | 0.7001 (628/897) | 0.7767 (727/936) | 0.3289 (295/897) | 0.4188 (392/936) |
| v18 | 0.6878 (249/362) | 0.7578 (291/384) | 0.2403 (87/362) | 0.4922 (189/384) |
| random | 0.5070 (109/215) | 0.6697 (146/218) | 0.3953 (85/215) | 0.7706 (168/218) |
| **pooled** | **0.6689 (986/1474)** | **0.7568 (1164/1538)** | **0.3168 (467/1474)** | **0.4870 (749/1538)** |

**No White collapse — White is BETTER by 0.088 pooled**
(0.7568 vs 0.6689), on all three sources. The symmetric
labels did not introduce a White-side holdout defect (contrast
v25/v31's B1-data White collapses); the split resembles v37's
White-better split (0.7775 vs 0.7269). The no-history view
collapses for BOTH sides (Black 0.317, White 0.487 — White
less so, a curiosity, not a decision input).

### (b) v44's net on the symmetric labels (cross-label)

Sign agreement 0.3137 / 0.3378 / 0.4042 (pooled 0.3327) —
BELOW chance on engine positions, exactly as §1.6 predicted:
v44's net outputs values centred near +0.10 (its optimistic
3-ply target) while the symmetric targets centre near −0.10,
so the signs systematically disagree. **Pearson r (the
pre-registered primary measure): v7 0.0617, v18 0.0437,
random 0.0213, pooled 0.0819** — nearly ZERO. The 3-ply-
trained net captures almost NONE of the symmetric function in
correlation terms (contrast iter-44 §4(b), where v37's net
captured the 3-ply function at r 0.3285). The symmetrisation
changed the target MORE than the depth change did: the mirror
term is not recoverable from the 3-ply function alone.

### (c) Reverse transfer: v56 net on v44's 3-ply labels

Sign agreement 0.4009 / 0.3967 / 0.5366 — below chance on
engine positions, the parity shift again (v56's net centres
near −0.10, the 3-ply targets near +0.10). **Pearson r: v7
0.3128, v18 0.2743, random −0.0196, pooled 0.2675** — vs
v44's own pooled r of 0.3149 on its own labels. Decent reverse
transfer (85% of v44's own correlation): the symmetric net
DOES encode most of the 3-ply function (shift aside), while
the 3-ply net encodes almost none of the symmetric function
(r 0.082) — the transfer is asymmetric, as expected, since the
symmetric function is derived FROM the 3-ply function plus the
mirror term. Curiosity: the no-history view scores 0.68 pooled
on 3-ply labels (vs 0.42 full) — the no-history outputs are
positive-shifted (mean|v| 0.2979 vs 0.1306 on v7), accidentally
aligning with the 3-ply targets' +0.10 centre. Diagnostic
only.

### (d) v37's net on the symmetric labels

Sign agreement 0.7185 / 0.7319 / 0.5173 (pooled 0.6929) —
nearly v56's own 0.7138, as §1.6 predicted (the symmetric
labels are close to the 2-ply function in DISTRIBUTION).
**But Pearson r is only 0.1342 pooled** (v7 0.1804, v18
0.2325, random 0.0271). The sign agreement is inflated by the
shared pessimism (both nets predict negative most of the time,
and the targets are negative most of the time) — the
magnitudes do NOT track. Distribution similarity ≠ function
similarity: the symmetric function is genuinely different
from the 2-ply function (its own net reaches r 0.4312), and
v37's net does not capture it beyond the shared centre.

### (e) v8's net on the symmetric holdout

Sign agreement 0.5428 / 0.5054 / 0.5312 (pooled 0.5319),
Pearson r pooled 0.0792 — at/below chance, as on v24's/v37's/
v44's labels. v8's net encodes a genuinely different value
function, as expected from its different target (weak-self-
play game outcomes).

**A note on the Pearson r values:** v56's own pooled r of
0.4312 (v7 0.5342, v18 0.5211, random −0.0146) EXCEEDS
v44's own 0.3149 and v37's own 0.3979 — the symmetric
function is learned with better correlation than either
predecessor's function was (the halved wins reduce the ±1
spike mass that drags r down: 4.5% ±1 vs v44's 7.9%). r is
used here ONLY for cross-label comparison (where it is
shift-invariant); the in-distribution fit measure is sign
agreement, as in v24/v37/v44. Random-holdout r is ≈ 0 for
every distilled net (v56 −0.0146, v44 0.0213, v37 0.0271) —
random positions are mostly terminal/noisy; v8's 0.1679 there
is a curiosity (B1-trained nets track random positions
better). No decision hinges on any of this: the battery is
the verdict.

## 5. α sweep and fast loops

### 5.0 Threading note (applies to all game-playing work from here on)

The sweep started at the engine default (2 torch threads) and was
prohibitively slow under the day's contention (5 game-playing
processes: this sweep, H1's fast loop, H2's sweep, 2× cross-track
v52-vs-v44 matches; load 5–6.5): ~6–8s per ply, with 225-ply draws
costing 1,360–1,888s each. `ValueMCTS.__init__` hard-codes
`torch.set_num_threads(2)`, and `common.torch_threads` documents that
batch=1 forward passes gain nothing from intra-op parallelism while
oversubscribed threads spin and starve the work. A/B measured
2026-11-22 with `ml/runs/ml6-h3/bench_threads.py` (same 6 fixed
positions, 100 sims): 2 threads median 5.96s/move then 16.48s/move
across two runs (thrashing spikes); 1 thread 4.19s then 3.20s — a
1.4–5x speedup, stable. Move choices verified IDENTICAL on 6/6
positions between 1 and 2 threads, so the pin changes only tensor-op
parallelisation, not the math (up to ~1e-7 reduction-order noise,
the same tolerance the program accepts for batched-vs-single eval).
All game-playing run scripts from here on go through
`ml/runs/ml6-h3/run_1thread.py`, a process-local launcher that pins
1 thread and neutralises the engine's `set_num_threads(2)` — no
shared code is touched. Consequence recorded: sweep arm 1 games 0–6
were played at 2 threads, games 7–39 at 1 thread; the engines are
the same up to the float noise above, so no game is discarded and
no arm is restarted.

### 5.2 Fast loops (α = 0.25, seeds 1001/2001)

| Opponent | W–L–D | Win rate | Black | White | Mean plies |
|---|---|---|---|---|---|
| v0 (1-ply heuristic) | 5–3–2 | 0.60 | 3–1–1 | 2–2–1 | 78.5 |
| v44 (champion) | 4–4–2 | 0.50 | 1–4–0 | 3–0–2 | 81.5 |

10 games each — noise dominates (±0.3 CI), so these are a
smoke test, not a verdict: v56 plays coherent Gomoku, beats
the 1-ply heuristic, and holds even with v44 on this tiny sample
(the sweep's 0.4625 at α=0.25 is the better read). No
systematic SEARCH flaw appeared (no blowout, no repeated
tactical blunder pattern), so no fast-loop patch is made —
and there is none to make for the pattern that DID appear.

**The colour split vs v44 is the pattern to watch in the
battery: as Black v56 went 1–4–0 (mean 47.4 plies); as White
3–0–2 (mean 115.6 plies).** v56-as-White is unbeaten and its
games drag long; v56-as-Black loses quickly. That is exactly
what the §0 mechanism predicts for a loss-averse net (wins
halved to ~0.35, losses kept at ~1.0): as White, the
defensive style holds and punishes v44's optimistic
overextension; as Black, it cannot convert the first-move
edge because it will not attack. n=10 cannot carry this
claim — the 100-game battery leg vs v44 (§6) will test the
split at n=100. If it holds, it is a property of the LABELS,
not a patchable search bug: fixing it means a new version
(a different correction), not a fast-loop patch.

### 5.1 α sweep results

40 games per arm vs v44, seeds 1001/2001 (same seeds all arms
— paired), 1-thread pin from game 7 of arm 1 on (§5.0). Two
VM replacements during the sweep; both resumes verified
(`resumed_from_game` matches the saved count; per-game seeding
reproduces the uninterrupted run).

| α | W–L–D | Win rate | Wilson 95% CI |
|---|---|---|---|
| 0.0625 | 11–26–3 | 0.3125 | [0.1907, 0.4671] |
| 0.125 | 15–22–3 | 0.4125 | [0.2742, 0.5661] |
| **0.25** | **15–18–7** | **0.4625** | [0.3182, 0.6134] |

**Winner: α = 0.25** (the sweep script's pick; the registry
placeholder already was 0.25, so no value change — only the
comment above was updated). The trend is monotonic in α, and
the winner sits at the grid's top end — which §1.4 explicitly
anticipated ("the winner may sit at either end") and declined
to extend (0.5 was excluded because v44's sweep already
showed it losing on HOTTER labels). The no-signal rule does
not apply (arms span 0.3125–0.4625, far more than one win
apart). Paired check on the first 22 games (same seeds):
α=0.0625 0.3636 vs α=0.125 ~0.31 — the arm-1/arm-2 gap
reverses on the paired subset, so the α=0.125 > α=0.0625
ranking rests on the full 40 games, not the paired prefix;
the α=0.25 win is by 4.5 points over α=0.125 (18.5 vs 16.5
in win-equivalents) and is the sweep's call under the
pre-registered pick-the-best rule.

**The headline is not the winner but the level: even at its
best α, v56 LOSES to v44 in the sweep (0.4625, CI includes
0.50 but centred below it).** Compare the sweeps that
preceded it: v37's α sweep picked 0.25 at 0.65 vs v8; v44's
picked 0.125 at 0.60 vs v8. v56's best arm against the
CHAMPION is 0.4625 — the parity correction did not produce a
net that beats v44 at any α on this grid. The battery
(clause 1, 100 games, different seeds + openings) is the
verdict, and the sweep is not the battery — but the sweep
says the correction, in this form, did not recover
v37-level played strength (v37 beat v44 0.69–0.31). The
mechanism hypothesis for §7: the symmetric label's
win-halving (L3==+1 preserved 0.000, §0) with loss
preservation (L3==−1 preserved 0.833) makes the net's value
scale asymmetric — wins worth ~0.35, losses costing ~1.0 —
so the engine plays loss-averse, defensive Gomoku and gets
out-attacked by v44's optimistic net. The fast loops (§5.2)
will show whether that reads as passive play in the games.

## 6. Validation battery

Run in the pre-registered order (§1.8): tactical → vs v44 →
ladder → vs v0 → vs v7 → vs v37 → swap vs v7, via
`ml/runs/ml6-h3/chain-battery.sh` with the 1-thread pin (§5.0)
throughout. Validation seeds 100/200; ladder seed 100, 40
games/opponent, mcts_sims=100; the vs-v44 and vs-v37 legs use
`research/openings-ml4.json`; swap uses balanced placement,
placement_seed 11, tempo judge. The battery survived several
VM replacements (logs `battery.nohup` through
`battery6.nohup`); every driver auto-resumed or skipped
completed outputs, verified each time (tactical guard via the
JSON check, selfplay_batch "already complete",
`eval_version` leg resume with rng_state) — zero games lost.
Artifacts committed per leg (tactical `18b9bb8`, v44
`70619f2`, ladder `edaefcf`, v0 `60db39f`, v7 `707e0de`, v37
`fa0d7b8`, swap `7016c5d`).

**Opponent identities (verified from code 2026-11-23 — read
this before comparing legs):** battery `v0` is
`baseline.HeuristicPlayer` — the 1-ply heuristic (the ladder's
"heuristic" opponent, anchor 0), NOT the ladder MCTS. There is
no v0 entry in the scoring registry;
`selfplay_batch.resolve_engine` maps "v0" to the baseline
player. Ladder `mcts` is `baseline.HeuristicMCTSPlayer`
(sims=100) — a different, stronger opponent with NO
battery-leg equivalent. Ladder `heuristic` is the same frozen
baseline player, not registry v7. Ladder Elo uses ONLY the
anchored opponents (random −800, heuristic 0); the mcts leg
does not enter the estimate (`evaluate.ladder_elo`:
"Unanchored opponents (mcts, checkpoint) do not enter the
estimate") — which matters for the clause-2 caveat below.

### 6.1 Tactical — CLAUSE 3 PASSES

**20/20 (old suite) + 29/29 (hard suite)**, seed 100.
Artifact `research/iterations/tactical-v56.json`; the ladder
JSON independently embeds the old-suite 20/20. The symmetric
labels cost nothing tactically: the net plays the forcing
moves and the blocks exactly as v44's did.

### 6.2 Battery legs (100 games each)

Win rate = (W + D/2)/n; CI = Wilson 95%. De-dup cut: drop
duplicate full move sequences, keep the first occurrence
(program convention).

| Leg | W–L–D | Rate | Wilson 95% CI | Unique | De-dup rate | De-dup CI | Black | White | Mean plies |
|---|---|---|---|---|---|---|---|---|---|
| vs v44 (openings) | 46–42–12 | 0.52 | [0.4232, 0.6154] | 100/100 | 0.5200 | [0.4232, 0.6154] | 0.59 | 0.45 | 72.3 |
| vs v0 | 53–40–7 | 0.565 | [0.4672, 0.658] | 88/100 | 0.5170 | [0.4141, 0.6185] | 0.63 | 0.50 | 59.9 |
| vs v7 | 43–48–9 | 0.475 | [0.3799, 0.572] | 95/100 | 0.4684 | [0.3713, 0.5680] | 0.60 | 0.35 | 63.8 |
| vs v37 (openings) | 32–48–20 | 0.42 | [0.328, 0.5179] | 97/100 | 0.4227 | [0.3292, 0.5221] | 0.55 | 0.29 | 88.4 |
| swap vs v7 | 43–48–9 | 0.475 | [0.3799, 0.572] | 99/100 | 0.4798 | [0.3840, 0.5771] | 0.519 (n=52) | 0.427 (n=48) | 62.9 |

**vs v44 — CLAUSE 1 FAILS.** Win rate 0.52 > 0.50, but the
Wilson lower bound is 0.4232 < 0.50; the bar needs ≈0.60 at
n=100. The de-dup cut is identical (100/100 unique), so the
clause fails on both cuts. Progression was never near the
bar: 0.548 at 21 games, 0.5253 at 79, 0.5361 at 97. The
sweep's read (0.4625 at the best α) was, if anything,
pessimistic — the battery says v56 is roughly v44's equal,
not its better.

**The fast-loop colour-split prediction is REFUTED.** §5.2
predicted, from n=10, that v56 would be Black-weak (1–4–0)
and White-strong (3–0–2) against v44 — the loss-aversion
signature. The 100-game leg shows the OPPOSITE split: Black
0.59 (26W-17L-7D), White 0.45 (20W-25L-5D). The fast-loop
split was noise, and the loss-aversion hypothesis's specific
prediction (Black-weak play) is refuted by the battery:
v56-as-Black is 0.55–0.63 in every leg, consistently its
stronger colour.

**vs v0:** nominal 0.565 overstates the leg — the de-dup cut
is 0.5170 (42W-39L-7D on 88 unique games): 11 of the 12
dropped duplicates were v56 wins, the deterministic-ish
heuristic repeating losing lines. On the nominal rate v56 sits
in the same band as v44 (0.58) and v37 (0.585); on the de-dup
rate it is slightly below both. The 88/100 uniqueness is the
lowest of any v56 leg and is an opponent property (v0's
greedy determinism), not a v56 property — the other legs are
95–100/100.

**vs v7:** 0.475 nominal, 0.4684 de-dup — the same band as
v44 (0.45) and v37 (0.45). The colour split is the story:
Black 0.60 (28W-18L-4D), White 0.35 (15W-30L-5D).

**vs v37:** 0.42 nominal, 0.4227 de-dup — v56 does better
than v44 did against v37 (0.31), but v37 still beats v56
clearly (CI upper 0.5179 barely touches 0.50). Colour split
Black 0.55 (22W-17L-11D), White 0.29 (10W-31L-9D) — the
widest gap of any leg, and the longest games (mean 88.4
plies both colours).

**Swap vs v7:** 0.475 nominal, 0.4798 de-dup (the single
duplicate was a loss). Better than v44's swap 0.39. Colour
split 52 Black / 48 White games (swap placement decides
colour, not alternation): Black 0.519, White 0.427.

**The White-side weakness is the battery's clearest
pattern.** v56-as-White: 0.45 / 0.50 / 0.35 / 0.29 vs v44 /
v0 / v7 / v37, against Black 0.59 / 0.63 / 0.60 / 0.55 — a
0.13–0.26 gap, wider than the normal first-move edge
(~0.07–0.10), and worsening against stronger opponents. The
ladder confirms it independently (§6.3): mcts-leg White
3–16–1 (0.175) vs Black 11–9–0 (0.55); heuristic-leg White
10–10 (0.50) vs Black 17–2–1 (0.875). The holdout said the
opposite — White BETTER than Black (0.757 vs 0.669, §4) —
the seventh instance of holdout not predicting played
strength (v31, v38, v42, v43, v48, v49, v56). One
mitigation: v56's White 0.29 vs v37 is better than v44's
0.16 vs v37 — the correction did not fix White, but it did
not repeat v44's White collapse either.

### 6.3 Ladder — CLAUSE 2 PASSES (with a caveat)

**Ladder Elo 137.0, interval [22.9, 251.1] — PASSES clause 2**
(137.0 > 98.1, point-estimate rule). 40 games/opponent, seed
100, mcts_sims=100.

| Opponent | W–L–D | Rate | CI | Black | White |
|---|---|---|---|---|---|
| random | 40–0–0 | 1.000 | [0.9124, 1.0] | 20–0–0 | 20–0–0 |
| heuristic | 27–12–1 | 0.6875 | [0.5329, 0.8093] | 17–2–1 | 10–10–0 |
| mcts (unanchored) | 14–25–1 | 0.3625 | [0.2317, 0.5174] | 11–9–0 | 3–16–1 |

v44's ladder for comparison: random 40–0, heuristic
24W-13L-3D (0.6375, Elo 98.1 [−12.1, 208.2]), mcts 25W-15L-0D
(0.625).

**Caveat — the Elo pass and the mcts leg disagree.** The Elo
comes from the heuristic leg alone (the random leg is
saturated at 40–0 for both versions, the mcts leg is
unanchored and does not enter the estimate): v56's heuristic
leg is 27.5 win-equivalents vs v44's 25.5 — a 2-game gap,
well within noise, and the intervals overlap heavily
([22.9, 251.1] vs [−12.1, 208.2]). Meanwhile the unanchored
mcts leg shows a REAL deficit: v56 0.3625 vs v44 0.625, a
~10.5-game gap, ~2.4 sigma. v44 was the first ML version
ever to beat the ladder MCTS outright; v56 loses to it
clearly, driven by White (3–16–1). So the clause-2 pass is
real under the pre-registered rule (point estimate, the same
rule v44's own pass used — and v44's interval included 0),
but it is a thin pass: it rests on the weakest anchored leg
while the strongest ladder opponent says v56 is worse than
v44. The honest summary is that v56's ladder strength is
indistinguishable from v44's, with the mcts leg pointing the
other way.

## 7. Verdict

(to be filled in.)

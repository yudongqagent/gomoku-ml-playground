# Iteration 44 — v44: 3-ply labels (ML batch 4, H3)

**Status: IN PROGRESS — pre-registered; labelling not yet started.**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H3 workstream of ML batch 4 (H1 is v42,
history augmentation on v24's recipe; H2 is v43, history
dropout on the v37 net); work happens on branch
`research/ml-batch-4-h3` in the `gomoku-ml4-h3` worktree. v44
differs from v8 ONLY in the value net's weights (and the swept
α) — same PUCT search, same v1 priors, same sims (100), same
c_puct (1.5), same threat check, same terminal cases — so the
battery is a clean A/B of the 3-ply-label net against v8's
weak-self-play net (and, less cleanly, against v37's 2-ply-label
net: v44 changes ONLY the labels, on identical positions — see
§1.1 — so the v44-vs-v37 comparison isolates the labelling
change, one ply deeper than v37-vs-v24 did).

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v8. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v44 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Ports and cost measurements (before pre-registration)

**Ports (ML batches 1–3 were never merged to master, so their
code was ported into this worktree — reuse, don't re-derive):**
- `ml/gomoku_ml/train.py`: ML batch 1 H2's memmap version
  (pre-allocated `load_shards` + memmap bundle support + lazy-index
  `train()`), copied VERBATIM from `gomoku-ml1-h2` (sha256
  `7e8cc28d…` matches the source and iter-37's record). NOT the
  ml-batch-3 copy: that one carries v38's multi-task threat code
  (a CLOSED lever), which this workstream does not need.
- `ml/scripts/generate_distill_data.py`: v37's version, copied
  VERBATIM from `gomoku-ml3` (sha256 `2057f0a5…`) — then extended
  with `--label-mode rollout3` (see §1.1; the diff from the
  verbatim port is 3 small hunks: docstring, label-fn selection,
  CLI flag; the defaults `rollout`/`rollout2` reproduce v24's/v37's
  labels exactly).
- `ml/scripts/build_mmap_bundle.py`,
  `ml/scripts/split_holdout_unique.py`,
  `ml/scripts/sweep_alpha.py`, `ml/scripts/find_losing_moves.py`,
  `ml/tests/test_load_shards.py`: copied VERBATIM from
  `gomoku-ml3` (sha256 `004ec68e…`, `2b7930bc…`, `26c0d827…`,
  `1f1ca6cc…`, `1844f241…` — the last three match iter-37's
  record; `sweep_alpha.py` is the batch-3 adapted version whose
  `resolve_opponent` unpack matches this tree's 3-tuple
  signature).
- `ml/scripts/eval_value_holdout.py`: v30's by-side-to-move
  version, copied VERBATIM from `gomoku-ml2` (sha256
  `d47d262d…`) — the ONLY difference from this tree's master
  version is the by-side reporting (verified by diff), which
  §1.6 needs for the White-collapse check.
- `ml/gomoku_ml/mcts_heuristic.py`: `_rollout2_leaf_value`
  copied VERBATIM from `gomoku-ml3` (verified by string
  comparison of the extracted function), plus the new
  `_rollout3_leaf_value` (§1.1). The rollout-leaf teacher (v18
  code) is on master since the batch-9 merge — no
  cherry-picking needed, and no re-verification of the leaf port
  is needed either (v24's 401/401 bit-for-bit check stands; the
  teacher code is the same master code v24 verified against).
- `ml/gomoku_ml/mcts_value.py`: the per-version `value_weight`
  port (ML batch 1 H1's change, applied by hand to this tree's
  file — the SAME 3-hunk shape as v24's/v37's, byte-identical
  hunks: docstring, `value_weight: float | None = None`
  parameter on `ValueMCTS.__init__`, `self.value_weight`
  assignment, leaf shrink via `self.value_weight`); default None
  → v8's 0.25, so v8 is byte-identical in behaviour. v36's
  tie-break code was NOT ported (CLOSED lever). **Checked:
  `ValueMCTS.__init__` has no duplicate parameters** (the
  silent auto-merge trap — verified by inspection of the
  signature after the edit).
- `ml/gomoku_ml/scoring.py`: dispatch passes
  `ENGINE_VERSIONS[version].get("value_weight")` through to
  `ValueMCTS` (v24's exact hunk), plus the v44 registry entry
  (`value_checkpoint: ml/runs/ml4-h3/value-net.pt`,
  `value_weight: 0.25` placeholder until the §1.4 sweep sets
  it), plus the v37 registry entry ported from ml-batch-3
  (`value_checkpoint: ml/runs/ml3-h2/value-net.pt`,
  `value_weight: 0.25`) so the secondary v44-vs-v37 leg can run
  in this worktree.
- `ml/configs/ml4-h3.yaml`: v37's `ml3-h2.yaml` with run name/dir
  changed (and the iter-37 reference updated to iter-44).
- `ml/baseline.py` is FROZEN and untouched.

**Port smoke checks:** v8 loads and plays in this worktree
(checkpoint copied from the main repo, md5
`5deb26f3df67883f24f04800d5356da6`, matching iter-37's record);
v37 loads and plays (checkpoint copied from `gomoku-ml3`,
md5 `2fde492aae6d44cbe945e15bd5f1f959`, matching the task);
v44 is registered (no checkpoint yet — training produces it);
`_rollout3_leaf_value` runs (empty board: +0.1365).

**Cost measurements** (500 quiet positions from v7 games,
`/tmp/bench_label3.py`, no sibling contention at measure time):
- 3-ply label (`_rollout3_leaf_value`): **median 12.03 ms, mean
  12.21 ms** (p90 15.14 ms) vs the 2-ply label's median 10.03 ms
  and the 1-ply label's median 6.54 ms — only **1.84x** the
  1-ply cost (median), **1.20x** the 2-ply cost. The plan's
  "~3–4x 1-ply" estimate was pessimistic: most of the cost is
  fixed overhead (kernel calls, winning-move scans), not the
  extra rollout ply. Projected labelling for the full v24
  dataset (37,227 positions): **~7.5 min** — NOT the
  bottleneck, far under the 2-hour bar. **No dataset-size
  compromise is needed: v44 labels v24's FULL position set**
  (no sampled subset; the pre-registered subset rule below is
  therefore moot but recorded anyway).
- **Subset rule (pre-registered, moot):** had the full run
  projected over ~2 hours, labelling would have used every 2nd
  quiet position of every game (deterministic, stratified by
  game source — the same fraction of each source), recorded in
  the LABEL_JSON. It is not needed.

**The 3-ply labels are a genuinely different target from the
2-ply labels on the same 500 positions:** mean |Δ| 0.1619,
median 0.1295, p90 0.3621, max 1.1740; Pearson r 0.8865 (vs
2-ply-vs-1-ply's 0.7188 — adjacent depths correlate more);
mean|v| 0.2056 vs 0.1881; ±1 fraction 0.1000 vs 0.0860 — the
3-ply labels are slightly hotter. Resolution: ply-1 terminal
0/500, ply-2 terminal 14/500 (2.8%), ply-3 terminal 10/500
(2.0%), leaf eval 476/500 — nearly all labels come from the
leaf at the 3-ply-ahead position.
**BUT sign agreement is only 0.4920** (vs 2-ply-vs-1-ply's
0.7550) — see the parity finding below. This is NOT a bug:
the implementation was verified by manual re-derivation on all
500 positions (**0/500 mismatches** — `/tmp/diag_sign.py`
recomputes the §1.1 recursion step by step and compares).

**The parity finding (measured BEFORE pre-registration — it
shapes §1.4 and §1.6):** the 3-ply label has the OPPOSITE
horizon parity to the 1-ply and 2-ply labels, and the labels
are systematically shifted by it:
- Label means on the same 500 positions: 1-ply **−0.0627**,
  2-ply **−0.0560**, 3-ply **+0.0999**. Fraction positive:
  0.212 / 0.234 / **0.734**.
- Pure-gradient pairs (|v| < 0.2 both, n=354): mean(v3−v2) =
  **+0.1236**, median +0.1200, and v3 > v2 on **99.4%** of
  pairs. Between 1-ply and 2-ply there is NO such shift
  (mean(v2−v1) = +0.0035, frac(v2>v1) = 0.493).
- Mechanism: the 1-ply and 2-ply labels evaluate at a position
  reached after an EVEN number of rollout plies — the opponent
  just moved, so the leaf is pessimistic-tilted. The 3-ply
  label evaluates after an ODD number — WE just moved (m5 is
  our greedy best move at b4), so the leaf is optimistic-tilted
  by roughly one greedy move's worth of attack-breadth gain.
  The gradient itself is antisymmetric under colour swap
  (`_attack_gradient_v1`'s docstring), so the shift is the
  extra ply of OUR play, not a sign error — and the
  must-block values flip with it (3-ply has 65 ±0.25 labels vs
  2-ply's 33, and they NEVER coincide: a 3-ply ±0.25 is +0.25
  when WE hold a single winning move at b3, a 2-ply ±0.25 is
  −0.25 when the OPPONENT holds one at b2).
- Consequences, pre-registered: (1) the v44 net's raw values
  will centre near +0.10 on quiet positions, not near −0.06
  like v37's — the α sweep (§1.4) brackets BELOW v37's 0.25 to
  absorb the hotter, shifted values; (2) cross-depth holdout
  sign agreement is NOT a fit measure (the parity shift alone
  drives it to ~0.49) — §1.6 uses Pearson r (shift-invariant)
  as the primary cross-depth measure and reserves sign
  agreement for the in-distribution check; (3) the battery is
  the arbiter of whether the deeper horizon's extra information
  outweighs the parity shift's distortion — a 3-ply net that
  plays worse than v37 despite a deeper teacher would show the
  shift dominating, which is itself a result.

**The setup-v1v0s-g92 diagnostic (the motivating position,
measured BEFORE pre-registration — `/tmp/diag_g92b.py`):**
reproduces v37's numbers exactly at the root (1-ply +0.1290,
2-ply +1.0000 — the setup is sound) and extends them:
- At the root (Black to move): the 3-ply label is **+1.0000**
  (same as 2-ply — the forced win is inside the 5-ply horizon
  too). The teacher plays the setup (9,6) on ply 1.
- At the children (from Black's perspective): (9,6) is +1.0000
  at all three depths; (5,4) is +0.2500 at 1-ply but **+1.0000
  at both 2-ply and 3-ply** — the slow win after (5,4) is still
  inside the 5-ply horizon, so the 3-ply labels do NOT improve
  child-level discrimination at g92 over the 2-ply labels.
- Honest consequence, pre-registered (same as v37's): the 3-ply
  labels teach the win at the PRE-setup position, not the
  child ranking. For the tactical suite, v44's net must NOT
  invert the order (as v31's did); if it values (9,6) ≈ (5,4),
  the v1 prior (which ranks (9,6) FIRST, 0.1122) breaks the tie
  correctly. The clause-3 run of setup-v1v0s-g92 is the
  pre-registered mechanism check (§6.1).

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, tactical 20/20 + 29/29 on the CORRECTED suite —
re-verified by the ML batch 2 coordinator at seed 100; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (identical 40 games/opponent per leg after batch 11's
re-basing: random, heuristic, mcts, mcts_sims=100), clause 3 is
tactical ≥ v8's (old suite 20/20, hard suite 29/29, run at
`--seed 100` on the corrected suite), clause 4 is all test suites
green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v44 vs v7 (100
games), v44 vs v0 (100 games), a 100-game swap match vs v7 under
the tempo judge (the harness default), and v44 vs v37 (100 games
— the direct 3-ply-vs-2-ply comparison on identical positions;
v37 is the strongest ML version by head-to-head at 0.75 vs v8).
Non-transitivity is recorded prominently.

**The hypothesis (plan H3):** v37 proved the label depth is
causal — 2-ply labels moved played strength +0.235 over v24
(0.75 vs 0.515 vs v8) on identical positions with only the
labels differing. The 3-ply label extends the teacher's horizon
one ply further (5 plies vs 4), so it should teach the value
function more accurately still — OR the marginal label
information shrinks (2-ply already captures the quiet setups,
§0's g92 diagnostic shows 3-ply adds no child-level
discrimination there) and the parity shift (§0: +0.12
optimistic, 99.4% of gradient pairs) dominates, in which case
v44 plays no better than v37 or worse. Either outcome is
informative: it locates the distillation ceiling in label depth
vs label parity. The battery is the arbiter, as always (v31's
lesson: holdout transfer does not predict played strength).

**The improvement: v44 = v8's search + a value net trained to
predict v18's rollout-leaf value evaluated 3 plies ahead.**
Everything else is v8 unchanged (see the header). The net is
consulted ONLY in quiet positions (terminal cases stay exact, as
in v8), so training data is quiet positions only.

### 1.1 The 3-ply labelling procedure (pre-registered, exact)

The teacher is **v18's rollout leaf** (`_rollout_leaf_value`,
blend=False) — the SAME teacher as v24 and v37, for
comparability. The 3-ply label for a quiet position (`board`,
`player` to move) is `_rollout3_leaf_value(board, player)`:

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
4. **Otherwise** the label is `−_rollout_leaf_value(
   board+m1+m2+m3, -player)` — the OPPONENT to move at the
   resulting position, so the leaf value (from the opponent's
   perspective) is negated to `player`'s. This is the one
   place the 3-ply form differs structurally from the 2-ply
   form, which ends with `player` to move again and needs no
   negation. `blend=False` (v24's/v37's setting).

**Perspective:** the side to move throughout, in [−1, 1],
exactly like v24's and v37's labels. **Horizon:** 3 plies of
rollout-policy play + the 2 plies inside `_rollout_leaf_value`
= 5 plies, vs v37's 4 and v24's 2. **Why the rollout policy,
not a leaf-value argmax or a minimax reply:** the same reasons
as v37's (§1.1 there): (a) it IS the teacher's move policy —
the same policy `_rollout_leaf_value` uses for its internal 2
plies — so the v44 label is exactly the 5-ply extension of
v24's 2-ply label, with no new move-selection machinery; (b)
maximising the noisy leaf over ~40 candidates would inject
optimizer's-curse bias into the LABELS; (c) the minimax reply
was measured and rejected at 2-ply (~1.5 s/position, 50x the
cost, with NO better discrimination at setup-v1v0s-g92).
**Known limitation (pre-registered):** the rollout-policy
reply is v1-greedy, not minimax — it can miss refutations, so
3-ply labels can overvalue moves that win slowly against weak
defence (at g92, (5,4) goes +0.250 → +1.000 at 2-ply and stays
+1.000 at 3-ply). **Known limitation 2 (pre-registered, §0):**
the odd-horizon parity shift — 3-ply labels are systematically
optimistic (+0.12 mean shift vs 2-ply on gradient positions).
The battery is the arbiter of both.

### 1.2 Dataset (pre-registered)

- **Source: v24's engine-game positions, re-labelled.** NO new
  game generation: the games already exist (v24's §2) and game
  generation was the expensive step. The positions are EXACTLY
  v24's and v37's: the 817 games in `gomoku-ml1-h1/ml/runs/ml1-h1/
  games-{v7,v18,random}.json` (488 v7 self-play at 50 sims,
  seeds 50000+; 229 v18 self-play at 50 sims, seeds 60000+;
  100 random playouts, seeds 70000+), copied into this
  worktree's `ml/runs/ml4-h3/` (gitignored data). Expected
  yield: **37,227 quiet positions** (21,595 v7 / 9,932 v18 /
  5,700 random), the same positions v24 and v37 trained on —
  the ONLY change from v37 is the label function, so the
  v44-vs-v37 comparison isolates the labelling change.
- **Sampling:** ALL quiet positions of every game (labelling is
  ~12 ms median per position; §0 — the full run is ~7.5 min,
  so no subset is needed). A position is quiet iff
  `_terminal_leaf_value == 0.0`. Positions are encoded with the
  real game history (boards 2/4 plies back, last move) exactly
  as in training (`encoding.encode`). Policy targets are one-hot
  on the move actually played (unused at policy_weight=0).
- **Labels:** `_rollout3_leaf_value` (§1.1) via
  `generate_distill_data.py label --label-mode rollout3`.
  Rollout-found terminals (±1) are kept as-is: they are the
  label's real signal.
- **Holdout (game-level, v24's protocol, IDENTICAL split):**
  every 10th game of each source is routed to a `*-holdout`
  dir and NEVER trained on — the same games v24 and v37 held
  out (48 v7 / 22 v18 / 10 random = 80 games, 3,378 positions),
  so v44's holdout numbers are comparable to v24's and v37's BY
  POSITION (only the labels differ). Expected train set ≈ 33,849
  positions.
- **NOT used for training:** the B1 10k games (B1 data collapses
  White — v25, v31 — and v24's engine-only data did not; this
  workstream does not re-open that lever). They are NOT used
  even as a diagnostic here (v24's B1-diag already established
  the transfer numbers; no decision hinges on re-measuring it).
- **Variety check:** not needed — the games are v24's, whose
  variety check passed (14 distinct first-5-ply prefixes in 15
  v7 games).
- **Stopping rule:** labelling is ~7.5 min for the full set (§0);
  no cutoff is needed. If a VM replacement kills the labelling
  run, the label phase is resume-safe (existing shards skipped)
  — re-run the same command.

### 1.3 Architecture and training (pre-registered)

- **Architecture:** v24's/v37's ResNet **16ch/2 blocks** (126,759
  params), trained from scratch (random init — NOT from v8's,
  v24's, or v37's checkpoint; the labels are a different
  target). Do NOT go larger: v31 proved 4x capacity plays worse
  on distilled labels (the scale lever is CLOSED).
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes — v24's/v37's recipe exactly), augmentation on
  (dihedral), val_fraction 0.05 (random split within the train
  set, for best-checkpoint selection only), seed 0.
- **Config:** `ml/configs/ml4-h3.yaml` (v37's `ml3-h2.yaml`
  with run name/dir changed); run dir `ml/runs/ml4-h3-train/`;
  threads pinned to 1 (`OMP_NUM_THREADS=1 TORCH_THREADS=1`,
  B1's finding). `checkpoint_every_epochs: 1` from the start
  (v24's resume lesson).
- **Memmap path (mandatory):** the train shards are bundled with
  `ml/scripts/build_mmap_bundle.py` into
  `ml/runs/ml4-h3/data-mmap/` and training reads the bundle
  (`load_shards` memmap branch + lazy-index `train()`). The
  in-RAM load OOM-killed at this scale (iter-25 §2.4); do not
  train from the raw shards.
- **Checkpoint selection (pre-registered rule, same as v24/v37):**
  best-by-internal-val-loss (`best.pt`), copied to
  `ml/runs/ml4-h3/value-net.pt` (gitignored data, the path v44's
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
is swept over **{0.125, 0.25, 0.5}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds), via
`ml/scripts/sweep_alpha.py --version 44`. The grid brackets
v37's winner (0.25) BELOW: the 3-ply labels are hotter than
v37's (mean|v| ~0.21 vs ~0.19 on engine positions — §0) AND
systematically optimistic (+0.10 mean vs −0.06 — §0), so the
net's values will be larger in magnitude and shifted positive,
and a smaller α may win (v37's lesson: hotter labels → smaller
α; v37's sweep was monotone decreasing in α, 0.25 > 0.5 >
1.0). 1.0 is NOT on the grid: v37's sweep already showed 1.0
losing (0.50, White 0.20) on cooler labels, and re-testing it
on hotter labels spends 20 games on a dominated arm.
Pick the best win rate; tie-break: the smaller α
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

### 1.5 What would change the design

- If the distilled net's holdout sign-accuracy on its OWN
  (3-ply-label) holdout is at chance (~0.50), the distillation
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
MSE, full-planes and no-history views) on 3-ply labels unless
noted. **Parity caveat (§0): cross-depth sign agreement is NOT
a fit measure** — the +0.12 parity shift alone drives it to
~0.49. For cross-depth comparisons (b)–(d), Pearson r
(shift-invariant) is the primary measure; sign agreement is
reported with the caveat.
- (a) **v44 net on its own holdout** (80 games, 3,378
  positions) — in-distribution; the §1.5 stop rule applies
  here (sign accuracy vs chance). ALSO report accuracy BY SIDE
  TO MOVE (Black/White) on both the full-planes and the
  no-history views — the B1-White-collapse lesson (v30's
  method): v24's/v37's engine-only data did not collapse White,
  and this checks the 3-ply labels didn't reintroduce a side
  skew. ALSO report the no-history view (v37's net collapsed
  without history planes on engine positions, 0.7395 → 0.5163
  on v7 — check whether the 3-ply net inherits that).
- (b) **v37's net** (`ml/runs/ml3-h2/value-net.pt`) **on the
  same 3-ply labels** — same positions, cross-net comparison:
  how much of the 3-ply function does the 2-ply-trained net
  already capture? (Pearson r primary.)
- (c) **v44 net on v37's 2-ply labels** (v37's holdout shards,
  `gomoku-ml3/ml/runs/ml3-h2/data/{v7,v18,random}-holdout/`)
  — the reverse transfer: does the deeper net still reproduce
  the shallower function? (Pearson r primary.)
- (d) **v8's net on the 3-ply holdout** — baseline (expect
  ~0.50 sign, ~0 r, as on v24's/v37's labels).
No decision hinges on (b)–(d); they diagnose whether the deeper
labels were learned and whether the fit transfers across label
depths.

### 1.7 Eval seeds (pre-registered)

h2h vs v8: v44=100 / v8=200; h2h vs v0: v44=100 / v0=200; h2h vs
v7: v44=100 / v7=200; h2h vs v37: v44=100 / v37=200; ladder seed
100, **40 games/opponent** (batch 11's re-basing),
mcts_sims=100; swap 100 games vs v7, balanced placement,
**placement_seed 11** (the pre-registered seed — v31/v32 used 0
by mistake, do not repeat that), tempo judge; tactical `--seed
100`. The vs-v8 leg uses the pre-registered openings file
`research/openings-ml4.json` (ML batch 4 tooling — the
mirror-match fix); the vs-v37 leg uses it too (same
mirror-match rationale — two value-net engines); the vs-v0 and
vs-v7 legs do not (different engine families; v37's legs were
95–99 unique without openings).

### 1.8 Eval order (pre-registered)

(1) Labelling (§2). (2) Memmap bundle + training (§3) +
holdout diagnostics (§4). (3) Leaf-cost measurement. (4) α
sweep (§5) — registry updated to the winner. (5) FAST LOOP
(§5): up to 4 iterations of 10 games vs v8 AND 10 games vs v0.
(6) FULL VALIDATION BATTERY on the final v44 (§6): tactical
re-run, h2h vs v8 100 games (with --openings), 100 games vs v0,
100 games vs v7, 100 games vs v37 (with --openings), 100-game
swap match vs v7, ladder 40 games/opponent, clause-4 suite
re-run. Any battery numbers measured before the fast loop
completes are INTERMEDIATE, labelled as such; only the final
battery counts for the bar.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games, pre-registered openings
selfplay_batch.py --engine-a v44 --engine-b v8 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v44-vs-v8.json
# Clause 2: ladder (40 games/opponent, mcts_sims 100)
eval_version.py --version 44 --ladder-games 40 --out research/iterations/ladder-v44.json
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 44 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, vs v37 (openings), swap vs v7 (balanced/tempo, seeds 100/200, placement_seed 11)
selfplay_batch.py --engine-a v44 --engine-b v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v44-vs-v7.json
selfplay_batch.py --engine-a v44 --engine-b v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v44-vs-v0.json
selfplay_batch.py --engine-a v44 --engine-b v37 --games 100 --openings research/openings-ml4.json --seed-a 100 --seed-b 200 --out research/iterations/batch-v44-vs-v37.json
selfplay_batch.py --engine-a v44 --engine-b v7 --games 100 --swap-rule --placement balanced --placement-seed 11 --judge tempo --seed-a 100 --seed-b 200 --out research/iterations/batch-v44-vs-v7-swap.json
```

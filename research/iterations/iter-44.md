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

## 2. Data generation (actuals)

**Labelling COMPLETE 2026-11-19 ~20:11 UTC** (started 19:43 UTC;
one VM replacement at ~19:46 UTC killed the run after 91 v7
games — the label phase is resume-safe, so the relaunch skipped
the existing 91 shards and lost nothing; the v7 LABEL_JSON below
counts only the resumed segment): `generate_distill_data.py
label --label-mode rollout3` over v24's exact games files
(copied verbatim into `ml/runs/ml4-h3/games-{v7,v18,random}.json`).
Full log `ml/runs/ml4-h3/label.log` (LABEL_JSON lines).

| Source | Games | Plies | Quiet positions | ±1 labels | Gradient labels | Held-out games | Label time |
|---|---|---|---|---|---|---|---|
| v7 self-play (50 sims, seeds 50000+) | 488 | 25,277 | 21,595 (19,545 train / 2,050 holdout) | 1,567 (7.3%) | 20,028 | 48 | ~891 s (711.1 s resumed segment + ~180 s pre-kill) |
| v18 self-play (50 sims, seeds 60000+) | 229 | 11,511 | 9,932 (9,142 train / 790 holdout) | 842 (8.5%) | 9,090 | 22 | 482.7 s |
| Random playouts (seeds 70000+) | 100 | 11,009 | 5,700 (5,162 train / 538 holdout) | 1,949 (34.2%) | 3,751 | 10 | 263.5 s |
| **Total** | **817** | **47,797** | **37,227 (33,849 train / 3,378 holdout)** | **4,358 (11.7%)** | **32,869** | **80** | **~1,637 s** |

Position counts match v24's and v37's shards EXACTLY (verified
per-source by counting shard contents: same shard counts, same
per-shard position counts in the train dirs) — the same 37,227
positions, only the labels differ, as pre-registered. The 3-ply
labels are hotter than v37's 2-ply labels on every source (±1:
7.3% vs 6.2% v7, 8.5% vs 7.4% v18, 34.2% vs 32.6% random; train
mean|v|: v7 0.1681, v18 0.1775, random 0.4255; pooled train
mean|v| 0.2099 vs v37's 0.1999) — the 5-ply horizon finds more
forced wins, as the §0 measurement predicted (11.7% ±1 pooled
vs v37's 10.6%). **And the parity shift is in the data:** train
label means are positive on every source (v7 +0.0881, v18
+0.0586, random +0.2798; pooled train mean **+0.1094**) — the
net will learn a value function centred near +0.11, not near
−0.06 like v37's. The α sweep (§5) is where that shift gets
absorbed.

Memmap bundle `ml/runs/ml4-h3/data-mmap/` built from
`ml/runs/ml4-h3/data-train/` (symlinks to the three train dirs
only — holdouts excluded; recursive glob follows the symlinked
dirs, 737 shards found): 33,849 samples, planes (33849, 8, 15,
15), mean|v| 0.2099. (One service restart killed the first
bundle-build attempt before it wrote anything; the detached
re-run completed in ~30 s.)

## 3. Training (actuals)

**Training COMPLETE 2026-11-19 20:37 UTC** (started ~20:14 UTC,
~23 min, no interruptions): config `ml/configs/ml4-h3.yaml`,
15 epochs, batch 64, memmap bundle (33,849 samples; 32,157
train / 1,692 val, seed-0 split). Run dir
`ml/runs/ml4-h3-train/`; log `ml/runs/ml4-h3/train.log`. RSS
stayed low (memmap path). 7,545 steps, ~60–151 s/epoch.

| Epoch | Train loss | Val loss |
|---|---|---|
| 1 | 0.1115 | 0.1299 |
| 2 | 0.1065 | 0.1282 |
| 3 | 0.1052 | 0.1251 |
| 4 | 0.1037 | 0.1261 |
| 5 | 0.1023 | 0.1232 |
| 6 | 0.1013 | 0.1254 |
| 7 | 0.1002 | 0.1241 |
| 8 | 0.0986 | 0.1208 |
| 9 | 0.0971 | 0.1194 |
| 10 | 0.0953 | 0.1196 |
| 11 | 0.0924 | 0.1169 |
| 12 | 0.0909 | 0.1169 |
| 13 | 0.0909 | 0.1147 |
| 14 | 0.0893 | 0.1138 |
| 15 | 0.0897 | **0.1136** |

Val loss decreased overall (small bumps at epochs 4, 6, 10) —
no overfitting by epoch 15, but the pre-registered plan is 15
epochs and the battery is the verdict, so no extension. **Best
checkpoint: epoch 15 (`best.pt`, step 7545, val 0.1136)** — the
`--out` copy at `ml/runs/ml4-h3/value-net.pt` is verified
bit-identical in weights to `best.pt` (max |Δ| 0.0), so the
registry path holds the best-by-val-loss checkpoint as
pre-registered. (Absolute val loss is not comparable to v37's
0.0986 or v24's 0.0617 — the 3-ply labels are hotter and
shifted, mean|v| 0.21 vs 0.20 vs 0.06, so the target variance
is higher; sign accuracy is the comparable metric.)
Architecture verified: ResNet 16ch/2 blocks, 126,759 params,
48 tensors — identical counts to v24's, v37's, and v8's
checkpoints. Checkpoint md5: `d22af606f7d2ee0254f372da895ae7ad`.

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th game, never trained on — the SAME
80 games / 3,378 positions v24 and v37 held out), targets =
3-ply labels unless noted. v44 net =
`ml/runs/ml4-h3/value-net.pt`; v37 net =
`ml/runs/ml3-h2/value-net.pt`; v8 net =
`ml/runs/b3-h2/value-net.pt`. Raw outputs in
`ml/runs/ml4-h3/holdout-3ply-{v7,v18,random}.txt` and
`ml/runs/ml4-h3/holdout-2ply-{v7,v18,random}.txt`; Pearson r via
`/tmp/holdout_r.py`.
Sanity check: this run reproduces v37's published 2-ply numbers
EXACTLY on its own holdout (v7 0.7395, v18 0.8436, random 0.6623
— full planes; no-history 0.5163/0.5846/0.6604; MSEs
0.0781/0.0819/0.2675) — the eval setup is sound. (Note: v37's
holdout shards live in the H2 worktree,
`gomoku-ml3-h2/ml/runs/ml3-h2/data/` — the merged ml-batch-3
worktree holds only the checkpoint; the first (c) run failed on
the wrong path and was re-run.)

### (a) v44 net on its own 3-ply holdout (in-distribution)

| Holdout | n pos | v44 sign (full) | v44 sign (no-hist) | v44 MSE | v44 mean\|v\| | v37 sign (full) | v8 sign (full) |
|---|---|---|---|---|---|---|---|
| v7 | 2,050 | 0.7460 | 0.7360 | 0.0847 | 0.0835 | 0.3145 | 0.4468 |
| v18 | 790 | 0.7163 | 0.7035 | 0.1057 | 0.0808 | 0.3607 | 0.4339 |
| random | 538 | 0.7786 | 0.7880 | 0.2624 | 0.3523 | 0.5403 | 0.5347 |
| **pooled** | **3,378** | **0.7442** | **0.7368** | — | — | **0.3666** | **0.4583** |

**Verdict: the distillation learned the deeper label function —
the §1.5 stop condition (chance-level holdout) does NOT
trigger.** Pooled in-distribution sign agreement 0.7442 vs 0.50
chance. It is slightly LOWER than v37's 0.7519 on 2-ply labels
and v24's 0.7766 on 1-ply labels — the 3-ply function is harder
to learn, as expected (a 5-ply lookahead compressed into a
static evaluation). Per-source, v44 is better than v37 on v7
(0.7460 vs 0.7395) and random (0.7786 vs 0.6623) but worse on
v18 (0.7163 vs 0.8436) — the strongest source is where v44's
fit is weakest, a diagnostic curiosity the battery will speak
to (no decision hinges on it).

**The v44 net is HISTORY-ROBUST — the v37 collapse is GONE.**
The no-history view drops only 0.007 pooled (0.7442 → 0.7368):
v7 0.7460 → 0.7360, v18 0.7163 → 0.7035, random 0.7786 →
0.7880 (no-history is BETTER on random). Compare v37: v7
0.7395 → 0.5163, v18 0.8436 → 0.5846 — a 0.195 pooled
collapse. History-dependence was the top ML-track risk, and
the 3-ply net does not show it. Mechanism (hypothesis): the
3-ply label is more determined by the current board than the
2-ply label — the extra ply of greedy rollout play plus the
parity shift make the target more of a "current position + one
greedy move" function and less of a game-phase function, so
the optimizer had less reason to grab the history shortcut.
Or it is luck of the optimisation (different target, different
init). Either way it is a measured fact, and it removes the
§4 warning sign that hung over v37.

**By side to move** (v44 net, 3-ply holdout; v30's method via
the ported `eval_value_holdout.py`):

| Holdout | Black (full) | White (full) | Black (no-hist) | White (no-hist) |
|---|---|---|---|---|
| v7 | 0.7667 (736/960) | 0.7249 (680/938) | 0.7625 (732/960) | 0.7090 (665/938) |
| v18 | 0.6880 (269/391) | 0.7448 (289/388) | 0.6931 (271/391) | 0.7139 (277/388) |
| random | 0.7584 (204/269) | 0.7992 (211/264) | 0.7472 (201/269) | 0.8295 (219/264) |
| **pooled** | **0.7463 (1209/1620)** | **0.7421 (1180/1590)** | **0.7432 (1204/1620)** | **0.7302 (1161/1590)** |

**No White collapse** — White is within 0.004 of Black pooled
on full planes (0.7421 vs 0.7463) and within 0.013 on
no-history planes. The engine-only data did not reintroduce a
side skew (contrast v25/v31's B1-data White collapses, and
v37's White-better split of 0.7775 vs 0.7269). The small
per-source wiggles (White −0.042 on v7, +0.057 on v18) cancel
in the pooled number and are within noise at n≈400/side.

### (b) v37's net on the 3-ply labels (cross-depth)

Sign agreement 0.3145 / 0.3607 / 0.5403 (pooled 0.3666) —
BELOW chance on engine positions, exactly as §1.6 predicted:
v37's net outputs values centred near −0.06 (its pessimistic
2-ply target) while the 3-ply targets centre near +0.10, so
the signs systematically disagree. **Pearson r (the
pre-registered primary measure): v7 0.1825, v18 0.2926, random
0.3483, pooled 0.3285** — vs v44's own pooled r of 0.3149
(v7 0.2027, v18 0.1480, random 0.3508). The 2-ply-trained net
captures the 3-ply function about as well in correlation terms
as the 3-ply-trained net does (and better on v18, 0.2926 vs
0.1480) — the same pattern as v37's §4, where v24's net
already captured the 2-ply function at 0.7398 sign vs v37's
own 0.7519. The marginal holdout gain from 3-ply labels is
nil-to-negative — but holdout is diagnostic, not the verdict
(v37's holdout marginal gain was also tiny, +0.012 sign, while
its played-strength gain was +0.235).

### (c) Reverse transfer: v44 net on v37's 2-ply labels

Sign agreement 0.2862 / 0.2128 / 0.4597 — below chance on
engine positions, the parity shift again (v44's net centres
near +0.10, the 2-ply targets near −0.06). **Pearson r: v7
0.1356, v18 0.1074, random 0.3981, pooled 0.3571** — vs v37's
own pooled r of 0.3979 (v7 0.1987, v18 0.3455, random 0.4177).
Decent reverse transfer (90% of v37's own correlation), and
the MSEs are close to v37's own (0.0966/0.1195/0.3015 vs
0.0781/0.0819/0.2675) — the v44 net's predictions are close in
VALUE to the 2-ply targets, just shifted. The two nets encode
overlapping but distinct value functions, as expected from
their different label depths (and parities).

### (d) v8's net on the 3-ply holdout

Sign agreement 0.4468 / 0.4339 / 0.5347 (pooled 0.4583),
Pearson r pooled 0.0955 (v7 0.0706, v18 −0.0352, random
0.1145) — at/below chance, as on v24's/v37's labels. v8's net
encodes a genuinely different value function, as expected from
its different target (weak-self-play game outcomes).

**A note on the low Pearson r values (0.15–0.42 even
in-distribution):** r is dragged down by distillation
compression — the nets predict small values (mean|v| ~0.08 on
engine positions) while the targets carry ±1 spikes (10–12%
of positions), so magnitudes barely track even when signs
agree 74–75% of the time. r is used here ONLY for cross-depth
comparison (where it is shift-invariant); the in-distribution
fit measure is sign agreement, as in v24/v37.

## 5. α sweep and fast loops

**Leaf cost (2026-11-19, `ml/runs/ml4-h3/leaf-bench.txt`):**
`bench_value_leaf.py --checkpoint ml/runs/ml4-h3/value-net.pt
--positions 20 --leaf-samples 200 --sims 100`: per-move at 100
sims mean 5,190 ms / median 3,716 ms / p95 9,809 ms; net leaf
evals per move mean 81.8 (17–99); per-leaf net eval median
7.49 ms. Slower than v37's bench (mean 2,155 ms) — the machine
was heavily loaded at measure time (load average 7.45, sibling
batch-4 workers running), not a net difference (same 16ch
architecture; per-leaf median 7.49 ms vs v24's 5.1 ms batch=1).
Battery games run ~2–5 min each under contention; the 100-game
matches are the long pole of this workstream.

**Clause-4 early check (run during the sweep):** ml pytest
**437 passed, 1 skipped, 58 xfailed, 1 failed** (25:29 wall;
log `ml/runs/ml4-h3/pytest-full.log`). The one failure,
`test_per_game_seeding.py::test_resume_reproduces_uninterrupted`,
was PRE-EXISTING on the base branch (verified: it fails on a
clean worktree of 4457fe7 too) — the ML batch 4 openings
tooling (4457fe7) made the driver write `rec["opening"]` on
every normal-rules record, but the test's in-process prefix
helper never set it, so the resumed records differed from the
reference on record SHAPE (the moves were identical — verified
field by field). Test-only fix (commit `32f555e`):
`rec["opening"] = []` in the non-swap branch;
`test_per_game_seeding.py` now **7 passed**. The full suite
will be re-run at the end (clause 4 requires green on the
final tree). npm test + tsc still to run (see §6.7).

### 5.1 α sweep (2026-11-19, ~20:49–22:38 UTC)

`sweep_alpha.py --version 44 --alphas 0.125 0.25 0.5 --games 20
--opponent v8 --seed-a 1001 --seed-b 2001` (dev seeds, no
openings). One VM replacement ~22:00 UTC killed the run at
α=0.25 game 8; the driver resumed from game 8 (5W-3L-0D)
with per-game seeding — nothing lost. Log:
`ml/runs/ml4-h3/sweep-alpha.log`; per-alpha JSONs in
`research/iterations/alpha-v44-<α>-vs-v8.json`.

| α | W-L-D | win rate | 95% CI | as Black | as White | unique |
|---|-------|----------|--------|----------|----------|--------|
| **0.125** | **14-5-1** | **0.7250** | [0.5059, 0.8716] | 7-3-0 (0.700) | 7-2-1 (0.750) | 20/20 |
| 0.25 | 12-8-0 | 0.6000 | [0.3866, 0.7812] | 6-4-0 (0.600) | 6-4-0 (0.600) | 20/20 |
| 0.5 | 9-10-1 | 0.4750 | [0.2785, 0.6795] | 5-5-0 (0.500) | 4-5-1 (0.450) | 20/20 |

Win rate is MONOTONE DECREASING in α — the §0 parity
prediction made manifest: the optimistic 3-ply labels want
the coolest value weight, and at α=0.5 the net's optimism
actively hurts (below 0.50). **Selection (§1.4 rule, best
win rate): α=0.125.** Registry updated (`scoring.py`,
`value_weight: 0.125`); targeted tests 14/14 pass.

Note the contrast with v37: its sweep winner was α=0.25
(0.75 vs v8, CI [0.657, 0.8245] on the full 100-game leg),
and 1.0 lost on the cooler 2-ply labels. v44's winner sits
one step cooler, exactly as the +0.10 parity shift predicts.
The 20-game CIs overlap, so the α ranking is a point-estimate
call, not a proven ordering — but the monotone shape across
three grid points plus the mechanism (parity) makes 0.125
the defensible pick, and the 100-game clause-1 leg is the
real verdict.

### 5.2 Fast loop iteration 1 (α=0.125, seeds 1001/2001)

- vs v8: the α=0.125 sweep leg itself (20 games, same dev
  seeds) — **14W-5L-1D (0.7250)**. As Black 7-3-0, as White
  7-2-1. All 20 games unique.
- vs v0: **5W-4L-1D (0.55)**, CI [0.2737, 0.7986], 963 s
  (`research/iterations/fast-v44-vs-v0-iter1.json`). Better
  than v37's 0.35 in the same loop, though both CIs are wide.
- Loss classification (both matches): **9 losses, 9 via open
  four, 0 missed blocks, 0 clean missed preventions in 46
  threatened turns.** Defence as measured is clean — the
  losses are positional drift into unstoppable open fours,
  not blundered blocks. Same pattern as v37.
- Losing moves (`find_losing_moves.py`, VCF solver depth 4;
  raw `ml/runs/ml4-h3/losing-moves-sweep.txt` and
  `losing-moves-v0.txt`): **6 of 9 losses were already lost**
  at the first VCF moment. **THREE genuine blunders:**
  vs v8 game 12 (Black, 84 plies), ply 48 — (3,6), 5 saves;
  vs v0 game 5 (White, 33 plies), ply 17 — (8,4), 4 saves;
  vs v0 game 9 (White, 23 plies), ply 15 — (10,4), 2 saves.

**Workup — vs-v8 game 12, ply 48 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml4-h3/diag-v8-g12-ply48.txt`)
shows a FLAT root: the actual (3,6) has the best our-q
(+0.007), the most visits (6), and the 1-ply leaf agrees
(actual −0.161 v22, saves −0.163/−0.161/−0.135/−0.168/−0.175;
only (12,6) is better on the leaf, and its our-q is −0.098
like the other saves). The prior even FAVOURS the saves
(0.0535 vs 0.0242) — the q overrides it. The VCF (depth-4
attacker moves, up to 8 plies) is beyond the 100-sim
search's horizon AND beyond the 3-ply label's 5-ply horizon
— the net agrees with the exact heuristic leaf, and both
rank the loser first. Shared search-horizon limit (v37's
g9 class), NOT a value-head failure: a perfect 3-ply-label
net ranks the loser first too.

**Workup — vs-v0 game 5, ply 17 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml4-h3/diag-v0-g5-ply17.txt`)
shows a q-vs-LEAF inversion: the actual (8,4) has our-q
−0.047 vs the saves' −0.127/−0.129, but the exact 1-ply
leaf ranks the saves clearly first (+0.161/+0.118/+0.075/
+0.123 vs the actual's −0.014). The leaf is RIGHT; the
backed-up root q inverts it through subtree averaging on a
very flat root (top move 6 visits over 79 children). This
is v40's "blunder dilution / visit allocation" class — a
search problem, not a leaf problem (the leaf-blind
direction is CLOSED). Not patchable within H3's remit:
changing visit allocation is shared heuristic-track code,
and "more sims" is a closed lever.

**Workup — vs-v0 game 9, ply 15 (VERIFIED):**
`diag_root.py` (raw `ml/runs/ml4-h3/diag-v0-g9-ply15.txt`)
shows the q RIGHT and the prior wrong: the saves (10,9)/
(12,9) have BETTER our-q (−0.044/−0.046) than the actual
(−0.119) and the exact leaf agrees (saves at the +0.250
must-block cap vs the actual's +0.187), but only 6 visits
each vs the actual's 11 — their priors (0.0392) are 3x the
actual's (0.1238). EXACTLY v37's game-2 blunder class
(and v31 iter-2's): a PRIOR problem in shared
heuristic-track code (`_priors`), whose change would alter
v8/v7 equally and invalidate the α sweep — NOT patchable
within H3's remit (v44 = v8's search + distilled net).

**Fast-loop STOP after iteration 1:** no patchable weakness
class for v44 within H3's design space. The three blunders
are one shared-horizon, one visit-allocation, one shared-prior
— all in shared search code, none a value-head failure.
The §4 history-robustness is NOT implicated in any blunder
(g12: net agrees with exact leaf; g5: the leaf is right,
the averaging is wrong; g9: the q is already right) — so
no loss-motivated retrain is authorised by §1.5. The
battery is the verdict.

## 6. Validation battery

### 6.1 Tactical — COMPLETE 2026-11-19 ~23:40 UTC — **CLAUSE 3 PASSES**

`eval_tactical_full.py --version 44 --suite both --seed 100`
(log `ml/runs/ml4-h3/tactical-v44.log`):

- Old suite: **20/20** ✓
- Hard suite (corrected): **29/29** ✓ — including
  **setup-v1v0s-g92: chose (9,6)**, the quiet setup v31
  failed. (As §0 predicted: 3-ply labels both (9,6) and
  (5,4) at +1.0000, so the choice between them comes from
  the search, not the label — and the search picks (9,6).)
- v1bl-g2 (the v40 regression position): passed, chose
  (7,5) — no leaf-value tie-break regression from the
  3-ply labels.

### 6.2 vs v8 — COMPLETE 2026-11-20 ~02:13 UTC — **CLAUSE 1 PASSES**

`selfplay_batch.py --engine-a v44 --engine-b v8 --games 100
--openings research/openings-ml4.json --seed-a 100 --seed-b 200`
(log `ml/runs/ml4-h3/batt-v8.log`; JSON
`research/iterations/batch-v44-vs-v8.json`). Two VM
replacements interrupted the leg (at 9 and 95 games); both
times the driver resumed from the saved game with per-game
seeding — nothing lost, and the resumed legs reproduce the
uninterrupted run (verified by the §5 test fix).

- **66W-32L-2D, win rate 0.6700, Wilson 95% CI [0.5731,
  0.7544]** — win rate > 0.50 and CI lower > 0.50 ✓
- Uniqueness audit: **93 unique games in 100** (the
  --openings set + per-game seeding; the 7 duplicates are
  opening repeats that played out identically).
- De-dup cut (93 unique): **62W-29L-2D, 0.6774, CI [0.5770,
  0.7638]** — the verdict stands on independent games alone.
- Colour splits: as Black **33W-16L-1D (0.6700)**, as White
  **33W-16L-1D (0.6700)** — exactly symmetric; no White
  weakness in played games (consistent with the §4
  by-side holdout: Black 0.7463 vs White 0.7421).
- Black won 49, White won 49, 2 draws — the match was
  perfectly colour-balanced overall.

Comparison: v37 scored 0.75 vs v8 (CI [0.657, 0.8245]);
v44's 0.67 sits BELOW v37's point estimate (the CIs
overlap: v44's upper 0.7544 vs v37's lower 0.657). The
3-ply extension did NOT beat the 2-ply labels on
head-to-head — diminishing returns, the second branch of
the §1 hypothesis. The parity shift (§0) is the likely
mechanism: the optimistic labels cost strength even at
the coolest swept α, and the sweep could only mitigate,
not remove, the shift. Clause 1 passes on its own terms
(vs v8), but v44 is not stronger than v37.

### 6.3 Ladder — COMPLETE 2026-11-20 ~03:40 UTC — **CLAUSE 2 PASSES**

`eval_version.py --version 44 --ladder-games 40 --out
research/iterations/ladder-v44.json` (log
`ml/runs/ml4-h3/ladder.log`; JSON committed). 40 games per
leg, seed 100, no interruptions (the ladder ran straight
through — the only leg of the battery with no VM kill).

| Opponent | W-L-D | win rate | 95% CI | as Black | as White |
|----------|-------|----------|--------|----------|----------|
| random | 40-0-0 | 1.0000 | [0.9124, 1.0000] | 20-0-0 | 20-0-0 |
| heuristic (v0) | 24-13-3 | 0.6375 | [0.4826, 0.7683] | 13-6-1 | 11-7-2 |
| mcts (100 sims) | 25-15-0 | 0.6250 | [0.4703, 0.7578] | 14-6-0 | 11-9-0 |

- **Ladder Elo: 98.1, interval [−12.1, 208.2]** (anchors:
  random −800, heuristic 0) — estimate > 0.0 ✓. The
  interval is wide and includes 0, so the pass is on the
  point estimate, as for every prior version (v24 passed
  at 52.5, v37 failed at −34.9). Stated plainly: the
  evidence favours v44 above the bar, but a 40-game leg
  cannot prove it.
- **v44 beats the ladder MCTS outright (0.625)** — the
  first ML version ever to do so (v37's Elo was −34.9;
  v18 was the first on the heuristic track). The MCTS
  leg is what killed v37's clause 2; v44 clears it.
- **Elo 98.1 is the highest ML-track ladder Elo ever**
  (v8 0.0, v24 52.5, v37 −34.9). The paradox of this
  version: BELOW v37 head-to-head (0.67 vs 0.75 vs v8)
  yet ABOVE it on the ladder (98.1 vs −34.9). The
  ladder's cold 40-game legs reward v44's consistency
  (no leg below 0.62) while v37's strength was
  concentrated in the v8 matchup. Both numbers are real;
  they measure different things.

### 6.4 vs v0 (secondary) — COMPLETE 2026-11-20 ~06:15 UTC

`selfplay_batch.py --engine-a v44 --engine-b v0 --games 100
--seed-a 100 --seed-b 200` (no --openings; log
`ml/runs/ml4-h3/batt-v0.log`; JSON
`research/iterations/batch-v44-vs-v0.json`). 5,260 s wall.

- **55W-39L-6D, win rate 0.5800, Wilson 95% CI [0.4821,
  0.6720]** — the CI includes 0.50, so the edge over v0
  is not proven at 100 games (a secondary metric, not a
  clause). Uniqueness: **99/100**; de-dup cut 55W-38L-6D
  on 99 unique, 0.5859, CI [0.4874, 0.6779].
- Colour splits: as Black 27W-20L-3D (0.5700), as White
  28W-19L-3D (0.5900) — balanced. Black won 46, White
  won 48.
- Note the sample spread across v44-vs-v0 measurements:
  fast loop 0.55 (10 games), ladder heuristic leg 0.6375
  (40 games, different seed sequence), this leg 0.58
  (100 games). All three CIs overlap; the 100-game leg
  is the biggest sample and the honest headline.


### 6.7 Test suites — **CLAUSE 4 PASSES**

- ml pytest (final tree, 2026-11-20 ~04:20 UTC):
  **438 passed, 1 skipped, 58 xfailed, 0 failed** (12:32;
  log `ml/runs/ml4-h3/pytest-final.log`). The first
  attempt was killed by a VM replacement at 57% (no
  failures to that point); the re-run completed clean.
  The one pre-existing failure from the early run
  (`test_resume_reproduces_uninterrupted`, a base-branch
  record-shape drift from the 4457fe7 openings tooling)
  was fixed in commit `32f555e` — see §5 preamble.
- npm test: **55/55 passed** (vitest, 4 files).
- `tsc --noEmit`: **clean** (exit 0).











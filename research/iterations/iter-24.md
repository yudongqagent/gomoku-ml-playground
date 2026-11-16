# Iteration 24 — v24: distill the v18 rollout leaf into a value net (ML batch 1, H1)

**Status: IN PROGRESS — pre-registered; data generation running.**
Comparison target is v8 (the ML track's champion and bar
reference). This is the H1 workstream of ML batch 1 (H2 is
completing the 10k dataset, H3 is the learned policy prior); work
happens on branch `research/ml-batch-1-h1` in the `gomoku-ml1-h1`
worktree. v24 differs from v8 ONLY in the value net's weights (and
the swept α) — same PUCT search, same v1 priors, same sims (100),
same c_puct (1.5), same threat check, same terminal cases — so the
battery is a clean A/B of the distilled net against v8's
weak-self-play net.

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move, find THE losing move per game (solver/deeper-search
verified, never guessed), patch, regression-test, repeat. For an ML
version the "patch" may be a data, architecture, or training change,
pre-registered like any other. Fast loops run against v0 as well as
v7. Max 4 fast-loop iterations before validation, or stop when an
iteration finds no new patchable weakness class. The full 4-clause
validation battery runs on the FINAL v24 only. Fast-loop games use
development seeds (1001/2001, ...), never the validation battery's
seeds (100/200).

## 0. Port verification and cost measurements (before pre-registration)

**The v18 rollout leaf lives on `research/batch-7`, not master.**
Ported for this workstream (see §1): `kernel_v1.py` checked out
verbatim from `research/batch-7`; `_breadth_v1`,
`_attack_gradient_v1`, `_rollout_move`, `_blend_rollout_terminal`,
`_rollout_leaf_value` ported verbatim (logic-identical, with
attribution comments) into `ml/gomoku_ml/mcts_heuristic.py`, plus
`leaf_mode="rollout2_v1_both"` in `_LEAF_FUNCTIONS` and a v18 entry
in `ENGINE_VERSIONS` (for the standard harness: v18 self-play data
generation and port verification only — v18 itself lives on
`research/batch-7`, versions 9–17 are NOT registered here).

**Port verification — 0 mismatches.** `ml/scripts/
verify_rollout_port.py`: a deterministic 401-position set (3 v7
self-play games at 30 sims, every ply; 4 random-playout games,
every 2nd ply; both tactical suites) evaluated by this branch's
`_rollout_leaf_value` and by the `research/batch-7` worktree's
(`~/workspace/projects/gomoku-b7/`): **401/401 agree bit-for-bit.**
The port is exact; no re-derivation was done.

**Cost measurements** (`ml/scripts/bench_distill.py`,
`ml/runs/ml1-h1/bench.log`; run under sibling contention, so means
are inflated — medians are the planning numbers):
- Rollout-leaf cost per call (quiet positions): **median 6.35 ms**
  (mean 18.4 ms, p90 69.2 ms, n=672). Labeling 40k positions ≈
  4–12 min — NOT the bottleneck.
- v7 self-play game at 50 sims: 16.0s (21 plies), 39.0s (61 plies)
  → ~0.7 s/move, ~27 s per ~40-ply game (~130 games/hr
  uncontended).
- v18 self-play game at 50 sims: 34.9s (25 plies) → ~1.4 s/move,
  ~56 s per ~40-ply game (~64 games/hr uncontended).
- Value-net forward (v8's 16ch/2-block net): **5.1 ms batch=1**,
  80 ms batch=64 (1.25 ms/sample) — v24's inference cost equals
  v8's, so battery games run at roughly v7 speed; training at
  batch 64 is ~0.3 s/step (forward+backward ≈ 3–4x forward).

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, vs v1 0.48, tactical 20/20 + 29/29; checkpoint
`ml/runs/b3-h2/value-net.pt`, ResNet 16ch/2 blocks, α=0.25). The
ML bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower
bound > 0.50), clause 2 is ladder Elo strictly higher than v8's
0.0 (identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical ≥ v8's (old suite 20/20, hard
suite 29/29, run at `--seed 100`), clause 4 is all test suites
green (ml pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v24 vs v7 (100 games), v24 vs v0 (100 games),
and a 100-game swap match vs v7 under the tempo judge (the harness
default). Non-transitivity is recorded prominently.

**The hypothesis (plan H1):** the rollout leaf (v16/v18) is the
strongest value signal in the program and the value signal is the
binding constraint (v15: the starved-save problem is partly a
value problem; v13: White's losses are beyond-horizon value
errors). A net that learns the rollout leaf's values could match
its value quality at much lower inference cost — and might
generalise beyond the rollout's 2-ply horizon. v8's net was
trained on WEAK self-play outcomes (random-init net games); its
labels are noisy in exactly the way rollout labels are not. The
distillation target is a deterministic function of the board, so
it should be learnable from far fewer samples than game outcomes.

**The improvement: v24 = v8's search + a value net trained to
predict v18's rollout-leaf value.** Everything else is v8
unchanged (see the header). The net is consulted ONLY in quiet
positions (terminal cases stay exact, as in v8), so training data
is quiet positions only.

### 1.1 Dataset (pre-registered)

- **Sources and game counts (50 sims throughout):**
  - v7 self-play: **600 games** (seed base 50000) → ~23k quiet
    positions expected (~38 quiet/game).
  - v18 self-play: **250 games** (seed base 60000) → ~9.5k quiet
    positions expected.
  - Random playouts: **100 games** (seed base 70000) → ~8k quiet
    positions expected (random games run long, ~80+ plies).
  - **Total target ≈ 40k quiet positions** (mix ≈ 60% v7 / 24%
    v18 / 16% random by position count). The 100k target in the
    plan does NOT fit the ~6h generation budget at the measured
    rates (v7 ~27 s/game, v18 ~56 s/game); 40k is the
    budget-sized target, and the labels — not the count — carry
    the quality (standing lesson: data quality > quantity).
- **Sampling:** ALL quiet positions of every game (game generation
  is the expensive step; labeling is ~6 ms/position, so every
  quiet position is kept). A position is quiet iff
  `_terminal_leaf_value == 0.0`. Positions are encoded with the
  real game history (boards 2/4 plies back, last move) exactly as
  in training (`encoding.encode`).
- **Labels:** the v18 rollout-leaf value (`_rollout_leaf_value`,
  blend=False) — a single leaf evaluation per position, from the
  side-to-move perspective, in [-1, 1]. Rollout-found terminals
  (±1) are kept as-is: they are the leaf's real signal.
- **Holdout (game-level, v8's protocol):** every 10th game of
  each source is routed to a `*-holdout` dir and NEVER trained
  on, so the holdout measures generalisation to unseen games
  (a random sample-level split would leak near-duplicate plies
  of a seen game). Expected train set ≈ 36k positions.
- **Stopping rule:** generation runs as two parallel background
  processes (v7 batch, v18 batch) + the fast random batch; if
  generation is still running at 6h wall, it stops and training
  uses what exists — the counts above are aims, the battery is
  the verdict either way. Actual counts are recorded in §2.
- **NOT used for training:** the B1 10k games (random-init net
  self-play — too weak/noisy as a primary source). They are used
  ONLY as a distribution-generalisation diagnostic (§4): 2,000
  of their positions, labelled with the rollout leaf, scored for
  value MSE / sign accuracy. No decision hinges on it.
- **Variety check:** after the first ~20 v7 games, count distinct
  opening prefixes in the games file; if degenerate (<5 distinct
  first-5-ply prefixes in 20 games), opening randomisation is
  added as a documented change BEFORE training (no eval has run
  by then, so it stays pre-registration).

### 1.2 Architecture and training (pre-registered)

- **Architecture:** v8's ResNet 16ch/2 blocks (126,586 params),
  trained from scratch (random init — NOT from v8's checkpoint;
  the labels are a different target). Go larger ONLY if the data
  supports it — it does not at ~36k samples, so 16ch/2 blocks
  stands.
- **Loss:** value MSE only — `policy_weight: 0.0`,
  `value_weight: 1.0`. The policy head is unused at inference.
- **Optimiser:** AdamW, lr 0.002 cosine, warmup 20 steps,
  weight_decay 1e-4, batch 64, **15 epochs** (~525k
  sample-passes, comparable to v8's 834k), augmentation on
  (dihedral), val_fraction 0.05 (random split within the train
  set, for best-checkpoint selection only), seed 0.
- **Config:** `ml/configs/ml1-h1.yaml`; run dir
  `ml/runs/ml1-h1-train/`; threads pinned to 1
  (`OMP_NUM_THREADS=1 TORCH_THREADS=1`, B1's finding).
- **Checkpoint:** best-by-val-loss (`best.pt`) copied to
  `ml/runs/ml1-h1/value-net.pt` (gitignored data, the path v24's
  registry entry points at).

### 1.3 α sweep (pre-registered)

α (the `value_weight` shrink on the net's quiet-position value)
is swept over **{0.125, 0.25, 0.5, 1.0}**, **20 games each vs v8**,
seeds 1001/2001 (fast-loop seeds). Pick the best win rate;
tie-break: the smaller α (conservative — less optimizer's-curse
exposure). The sweep SELECTS α; the 100-game battery is the
verdict (standing lesson: small-sample smokes mislead). The
registry entry's `value_weight` is updated to the winner after
the sweep. v8's α was 0.25, v11's best 0.125 — the sweep covers
both and the full-weight case.

### 1.4 What would change the design

- If the distilled net's holdout sign-accuracy is at chance
  (~0.50), the distillation failed to learn the leaf function —
  stop and report, do not battery a chance-level net.
- If the α sweep's best is within noise of the worst (all four
  within one win of each other), pick α=0.25 (v8's value) and say
  so — the sweep found no signal.
- Fast-loop patches (data/arch/training changes) are allowed
  only as pre-registered-style changes recorded in §5 with the
  specific loss that motivated them.

## 2. Data generation (actuals)

**Final (cutoff 11:00 UTC = 03:00 PST, per the §1.1 stopping rule;
three VM reboots during the run, jobs relaunched each time —
resumable output lost only the in-progress game each time):**

| Source | Games | Plies | Avg plies/game | Quiet positions | ±1 labels | Gradient labels | Held-out games |
|---|---|---|---|---|---|---|---|
| v7 self-play (50 sims, seeds 50000+) | 488 / 600 | 25,277 | 51.8 | 21,595 (19,545 train / 2,050 holdout) | 824 (3.8%) | 20,771 | 48 |
| v18 self-play (50 sims, seeds 60000+) | 229 / 250 | 11,511 | 50.3 | 9,932 (9,142 train / 790 holdout) | 504 (5.1%) | 9,428 | 22 |
| Random playouts (seeds 70000+) | 100 / 100 | 11,009 | 110.1 | 5,700 (5,162 train / 538 holdout) | 1,626 (28.5%) | 4,074 | 10 |
| **Total** | **817** | **47,797** | | **37,227 (33,849 train / 3,378 holdout)** | **2,954 (7.9%)** | **34,273** | **80** |

Train mix 58/27/15 (v7/v18/random) — almost exactly the
pre-registered 60/25/15. Labeling took 576 s (v7) + 238 s (v18)
+ 372 s (random, earlier, under generation contention). Note the
label skew: engine games are only ~4–5% rollout-terminal ±1
(most quiet positions are gradient-valued in ±0.2), while random
positions are 28.5% ±1 (tactically wild — many forced wins
within 2 plies). The net must learn fine distinctions inside
±0.2 for the bulk of positions; the ±1 subset is small but
high-signal.

Games files (full move lists, H3-reusable):
`ml/runs/ml1-h1/games-v7.json`, `ml/runs/ml1-h1/games-v18.json`,
`ml/runs/ml1-h1/games-random.json`. Shards:
`ml/runs/ml1-h1/data/{v7,v18,random}/` (train) and
`ml/runs/ml1-h1/data/{v7,v18,random}-holdout/` (every 10th game).

**Variety check (done 2026-11-15 ~21:05 PST, first 15 v7 games):
14 distinct first-5-ply prefixes in 15 games** (most common prefix
shared by 2 games) — RNG tie-break variety is ample; NO opening
randomisation added.

**Pace under contention** (siblings H2/H3 running; load ~3.5–5.7):
v7 ~42 s/game (smoke: 23 s/game), v18 ~150–200 s/game (smoke:
89.5 s/game). At the 6h cutoff (~03:00 PST) the expected yield is
~500 v7 games + ~120 v18 games + 100 random games ≈ 26k quiet
positions — below the 40k target, covered by the §1.1 stopping
rule (train on what exists; the battery is the verdict).

**Rebalancing rule (pre-committed 2026-11-15 ~21:45 PST, then
CANCELLED ~21:50 PST — see below):** post-restart, v18 ran at
~339 s/game (2-game sample, CPU-starved at ~20%) while v7 ran at
~42 s/game. v18 positions are the stronger distribution (closest
to inference) and the scarcest; v7 positions past ~400 games have
diminishing value. So: **when the v7 batch reaches 400 games, stop
it and let the v18 batch run solo for the remaining generation
time** (it should roughly double its CPU share). Game counts are
aims under the stopping rule, so this rebalances the mix toward
the stronger source without changing any pre-registered design
parameter. If v7 has not reached 400 by the cutoff, both simply
stop at the cutoff.

**Cancellation (~21:50 PST, before the rule triggers, before any
training/eval):** the 339 s/game figure was a noisy 2-game sample
— v18's post-restart pace over 7 games is ~163 s/game, and v7's
effective pace is ~35 s/game. Projected to the cutoff WITHOUT
intervention: v7 ~600 games (~18k positions), v18 ~141 games
(~5.9k), random 5.6k → ~29.5k total at mix 61/20/19, very close
to the pre-registered 60/25/15. WITH the rule: ~24.5k total at
49/28/23 — the rule would trade ~5k total positions for ~1k v18
positions (v7 yields ~3.3x more positions per CPU-second). The
no-intervention outcome matches the pre-registration better on
both total and mix, so the rule is cancelled and both batches run
to the cutoff untouched.

**Battery runbook** (validation seeds 100/200 throughout; run
after the α sweep sets the registry `value_weight`):
```
# Clause 1: vs v8, 100 games
fast_loop_match.py --version-a 24 --opponent v8 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v24-vs-v8.json
# Clause 2: ladder (20 games/opponent, mcts_sims 100)
eval_version.py --version 24 --ladder-games 20
# Clause 3: tactical, both suites, seed 100
eval_tactical_full.py --version 24 --suite both --seed 100
# Clause 4: ml pytest + npm test + tsc
# Secondary: vs v7, vs v0, swap vs v7 (defaults: balanced/tempo, seeds 100/200)
fast_loop_match.py --version-a 24 --opponent v7 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v24-vs-v7.json
fast_loop_match.py --version-a 24 --opponent v0 --games 100 --seed-a 100 --seed-b 200 --out research/iterations/batch-v24-vs-v0.json
selfplay_batch.py --engine-a v24 --engine-b v7 --games 100 --swap-rule --out research/iterations/batch-v24-vs-v7-swap.json
```

## 3. Training (actuals)

**Interrupt and recovery (2026-11-16):** the original training run
(started 11:19:22 UTC, config `ml/configs/ml1-h1.yaml`, 15 epochs,
33,849 samples) was killed by a runtime restart drain at **epoch 6,
step 2810 (11:27:06 UTC)** — no `interrupted.pt` was written (the
SIGTERM handler never ran). Epoch val losses before death: e1
0.07863, e2 0.07912, e3 0.08157, e4 **0.07251** (best), e5 0.07403;
train loss fell 0.0797 → 0.0705 over epochs 1–5.

**Resume decision: RESUME from `best.pt`, do not restart.** Verified
before resuming: `best.pt` loads cleanly (step 2012, epoch 4,
ResNet 16ch/2 blocks, 126,586 params, optimiser state present, all
48 tensors finite) and `ml/scripts/train.py --resume` restores
weights + optimiser + step with the cosine schedule continuing
exactly (`schedule_epochs = start_epoch + epochs`). A restart would
cost only ~20 min of compute, but resume loses nothing and keeps
the pre-registered 15-epoch plan exact: resume with `--epochs 11`
(15 − 4 completed). Two resume flaws handled explicitly:
(1) `train()` resets `best_val_loss` to None on resume, so the
first resumed epoch would overwrite `best.pt` even if worse than
epoch 4's 0.07251 — the epoch-4 checkpoint is backed up at
`checkpoints/best-epoch4.pt` and the final best is chosen by
comparing the resumed run's best val loss against 0.07251;
(2) per-epoch checkpoints were off (`checkpoint_every_epochs: 0`),
so a second crash would lose up to ~6 epochs — resumed with a
config copy `ml/configs/ml1-h1-resume.yaml` that differs ONLY in
`checkpoint_every_epochs: 1` (crash-recovery only; no training-math
change). Original run-dir config snapshotted at
`ml/runs/ml1-h1-train/config-original.yaml`; pre-resume metrics at
`metrics-pre-resume.jsonl`. Resumed 2026-11-16 ~11:30 UTC into the
same run dir (metrics append).

TODO — fill in when training completes: final train/val loss,
epochs run, best checkpoint (step/epoch), checkpoint path.

**Final (completed 2026-11-16 11:45:26 UTC):** 15 epochs total
(4 pre-interrupt + 11 resumed), 7,545 steps, 33,849 samples
(32,157 train / 1,692 val, same seed-0 split both runs). Train
loss 0.0797 → 0.0567; val loss 0.0786 → **0.0617**, improving
monotonically from epoch 6 on (e6 0.0717, e8 0.0679, e10 0.0656,
e12 0.0623, e14 0.0618, e15 0.0617) — no overfitting by epoch 15,
but the pre-registered plan is 15 epochs and the battery is the
verdict, so no extension. **Best checkpoint: epoch 15
(`best.pt`, step 7545, val 0.06169)** — the resumed run's best
beats epoch 4's 0.07251 outright, so the resume best-reset flaw
never bit and no restore from `best-epoch4.pt` was needed.
Per-epoch checkpoints `checkpoint-<step>.pt` (steps 2515–7545)
also saved. Final checkpoint copied to the registry path:
**`ml/runs/ml1-h1/value-net.pt`** (gitignored data; the
coordinator needs this path to verify on the merged tree).

## 4. Holdout diagnostics (actuals)

Game-level holdout (every 10th game, never trained on), targets =
rollout-leaf labels. v24 net = `ml/runs/ml1-h1/value-net.pt`;
v8 net = `ml/runs/b3-h2/value-net.pt` scored on the SAME targets
for a direct comparison (v8 was trained on game outcomes, so its
numbers here measure outcome-vs-rollout agreement, not its own
holdout quality — compare with care). Full planes unless noted.
Raw outputs in `ml/runs/ml1-h1/holdout-{v7,v18,random,b1diag}.txt`.

| Holdout | n pos | v24 sign | v24 MSE | v24 mean\|v\| | v8 sign | v8 MSE |
|---|---|---|---|---|---|---|
| v7 | 2,050 | 0.7713 | 0.0447 | 0.062 | 0.523 | 0.412 |
| v18 | 790 | 0.8384 | 0.0691 | 0.074 | 0.501 | 0.323 |
| random | 538 | 0.7055 | 0.2378 | 0.308 | 0.463 | 0.615 |
| **pooled in-dist** | **3,378** | **0.7766** | — | — | **0.504** | — |
| B1-diag (off-dist) | 1,948 | 0.6257 | 0.1173 | 0.117 | 0.422 | 0.568 |

**Verdict: the distillation learned the leaf function — the §1.4
stop condition (chance-level holdout) does NOT trigger.**
Pooled in-distribution sign agreement 0.777 vs 0.50 chance; even
off-distribution (B1 random-init-net games, a distribution the net
never saw) it is 0.626. v8's net is at/below chance on rollout
labels everywhere (0.50 pooled) — the two nets encode genuinely
different value functions, as expected from their different
targets (v8's own outcome-holdout 0.7165 is a different target
and not directly comparable). Notes: (a) random positions are
the hardest (MSE 0.238 vs 0.045–0.069 on engine sources) — 28.5%
are ±1 rollout-terminal labels, and the net leans on history
planes for those (no-history sign falls to 0.547, mean|v| 0.31 →
0.04); at inference leaves get search-path history, so the
no-history view is a lower bound, not the operating point.
(b) On v7 holdout the sign is history-robust (0.7713 both views)
but no-history MSE nearly doubles (0.045 → 0.083) — magnitudes
shift without history, signs do not. (c) B1-diag generalisation
(0.626) is well above chance but clearly below in-distribution —
distribution mismatch is reduced, not eliminated.

## 5. α sweep and fast loops

**VM replacement during the sweep (2026-11-16 ~13:27 UTC):** the
first sweep run was killed by SIGTERM after ~86 min with α=0.125
at 12/20 games (5W-5L-2D). The per-game flush saved all 12 games;
no completed alpha was lost (none had finished). The driver now
auto-resumes: completed alphas are skipped, a partial alpha
continues from its saved game count with fresh players. The
saved games stand (played under the pre-registered protocol);
the only deviation is that the MCTS tie-break RNG restarts for
the remaining games of the interrupted alpha instead of
continuing the pre-kill stream — the 20 games remain 20 fair
games vs v8 at that α with alternating colours, and the restart
is recorded in the JSON (`resumed_from_game`). Note the sweep's
pace is dominated by 225-ply draws (games 8 and 11 took ~1000s
each — full-board draws at ~5s/ply under contention).

TODO — sweep results, registry α update, fast loops.

**α sweep — COMPLETE 2026-11-16 ~14:47 UTC** (20 games each vs
v8, seeds 1001/2001; per-alpha JSONs
`research/iterations/alpha-v24-<α>-vs-v8.json`):

| α | W-L-D | Win rate | 95% CI | As Black | As White |
|---|---|---|---|---|---|
| 0.125 | 9-9-2 | 0.500 | [0.299, 0.701] | 7-2-1 | 2-7-1 |
| 0.25 | 10-7-3 | 0.575 | [0.364, 0.762] | 5-4-1 | 5-3-2 |
| 0.5 | 7-12-1 | 0.375 | [0.200, 0.591] | 3-6-1 | 4-6-0 |
| **1.0** | **13-7-0** | **0.650** | **[0.433, 0.819]** | 8-2-0 | 5-5-0 |

**Winner: α = 1.0 (full weight)** — registry `value_weight` set
to 1.0. The result is non-monotone in α (0.5 is the worst), so
the §1.4 no-signal rule does not apply (1.0 vs 0.5 differ by 6
wins). Full weight winning is the OPPOSITE of v8/v11's lesson —
the likely mechanism is calibration: the distilled net's values
are small in magnitude (mean|v| ~0.06 on engine positions, most
labels in ±0.2) and well-calibrated (holdout MSE 0.045–0.069,
sign 0.78), so α=1.0 here contributes about as much as α≈0.12
did on v8's net (mean|v| 0.53) — the optimizer's curse has
little magnitude to exploit. Colour note: at α=1.0 Black is
strong (8-2) and White is even (5-5); at α=0.125 the split was
extreme (Black 7-2-1, White 2-7-1). All four CIs overlap, so the
sweep selects α but proves nothing — the 100-game battery is
the verdict, as pre-registered.

**Fast loop iteration 1** (10 games each, seeds 1001/2001, α=1.0):
- vs v0: **3W-7L (0.30)**, CI [0.108, 0.603]. As Black 2-3, as
  White 1-4. v24 won the three long games (89, 94, 151 plies) and
  lost all seven short ones (25–58 plies).
- vs v7: **2W-7L-1D (0.25)**, CI [0.081, 0.558].
- Loss classification (both loops): all 14 losses via open four;
  **0 missed blocks, 0 clean missed preventions** (48 threatened
  turns vs v0, 18 vs v7). Defence as measured is clean — the
  losses are positional drift into unstoppable open fours, not
  blundered blocks.
- Games 1 and 9 vs v0 are near-identical (first 23 plies the
  same): v24 as White walks into the same loss twice. Board
  reading of game 1 (HYPOTHESIS, pending workup verification):
  Black's 21.e8 creates a four (e8-f8-g8-h8, winning move d8)
  AND a diagonal three (e8-f7-g6) in one move; White must block
  d8, then Black's 23.d9 is an open four and 25.h5 wins
  (d9-e8-f7-g6-h5). The 1-ply threat check cannot see it (e8
  leaves only ONE winning move, so it is not a "creator"), and
  the exact 1-ply enumeration confirms White's moves at plies 18
  (f9) and 20 (i10) each left Black with 0 winning moves and 0
  threat moves — the failure is 3–4 plies deep, where the
  100-sim search with small noisy net values does not propagate
  the loss back to the root. workup_loss (same-engine playout
  oracle) running to verify THE losing move.

**Workup verdict — game 1 vs v0 (VERIFIED):** workup_loss
(v24-vs-v0 playouts, seeds 0/1/2, last 4 White turns) finds all
4 turns qualify (actual move loses 3/3 playouts) and exactly ONE
turn with a saving move: **ply 19 — the actual i10 loses 3/3,
d9 saves 2/3 ([-1,+1,+1])**. THE losing move is White's 20th
move, i10 instead of d9. My hand hypothesis (e8 saves) is
REFUTED: e8 loses 3/3 at both ply 17 and ply 19. At ply 17 no
tested alternative saves. At ply 21 (forced d8) and ply 23
(open four live) nothing saves, as expected. Raw:
`ml/runs/ml1-h1/workup-v24-v0-g1.json`.

**Root-cause diagnostic (ply 19, 100 sims):** v24's root ranks
i10 (14 visits, q +0.077) above d9 (6 visits, q +0.033) — the
search does not see d9's preventive value. **v7 makes the SAME
miss** (i10 16 visits q +0.078; d9 7 visits q +0.064): the
net's ranking AGREES with the exact gradient leaf's ranking.
v8 plays d9 (11 visits) but only via visit dynamics — its q
ranks i10 (+0.535) far above d9 (+0.154). So the d9 miss is a
shared search-horizon limitation (100-sim PUCT + visit-based
root choice cannot propagate a 3–4 ply preventive value), NOT
a distillation failure: the net faithfully reproduces the
leaf's judgment, and the leaf itself cannot see the tactic.
More training would move the net TOWARD the exact leaf — which
also misses d9 — so retraining cannot fix this class. The
vs-v7 short loss (game 5, 15 plies) is the same class by a
different route: at ply 11 Black already holds FOUR threat
moves and the best White reply leaves TWO live — distributed
drift, no single losing move (exact enumeration).

**Fast-loop STOP after iteration 1:** no patchable weakness
class for v24 within H1's design space. The found class
(missed 3–4 ply preventive moves) is shared with v7 and would
need search-side changes (more sims, deeper threat check, VCF
at root) that alter v8/v7 equally and lie outside "v8's search
+ distilled net". α is NOT re-opened: the sweep selected 1.0
vs v8 (the bar reference) and 10 fast-loop games do not
override a pre-registered selection. Proceeding to validation
at α=1.0; the battery measures the vs-v0/v7 weakness at full
sample.

## 6. Validation battery

**Clause 4 (test suites) — run early, while the sweep runs:**
- ml pytest: **239 passed, 1 skipped, 2 xfailed, 0 failed**
  (19:04 wall under contention; log
  `ml/runs/ml1-h1/pytest-full.log`). Targeted subset
  (scoring/mcts_value/mcts_heuristic) also green: 69 passed,
  1 xfailed.
- npm test: **55 passed (55)**; tsc `--noEmit`: clean.
- Clause 4 is green; it will be re-run at the end if any code
  changes after this point (fast-loop patches would trigger a
  re-run).

**Clause 3 (tactical) — PASS:** `eval_tactical_full.py --version
24 --suite both --seed 100`: **old suite 20/20, hard suite 29/29**
(log `ml/runs/ml1-h1/tactical-v24.log`). Includes
winning-quiet-setup-2 (chose (10,10), in the expected list) —
the suite-list gap that sank v23's clause 3 is not an issue for
v24. Expected: v24 shares v8's search, threat check, and exact
terminal handling; tactical positions are 1–2 ply tactics the
search handles regardless of the leaf net.

**Clause 1 (vs v8, 100 games) — RUNNING** (seeds 100/200,
`research/iterations/batch-v24-vs-v8.json`). **VM replacement
#2 (2026-11-16 ~15:27 UTC):** the first run was killed after 12
games (5W-6L-1D, wr 0.4583); the per-game flush saved all 12.
`fast_loop_match.py` now auto-resumes (same pattern as the sweep
driver: saved games stand, remaining games continue with fresh
players, RNG restart recorded as `resumed_from_game`); the
match was relaunched with the same command and resumes from
game 12. The 100 games remain 100 fair games vs v8 at seeds
100/200 with alternating colours. **VM replacement #3
(2026-11-16 ~17:28 UTC):** the resumed run was killed after 94
games (46W-40L-8D, wr 0.5319); relaunched again with the same
command, resuming from game 94 (second RNG restart — the final
JSON records only the latest `resumed_from_game`, so both
restarts, at games 12 and 94, are recorded here).

**Clause 1 (vs v8, 100 games) — FAIL:** **47W-44L-9D, win rate
0.5150, Wilson 95% CI [0.4183, 0.6106].** The win rate clears
0.50 but the CI lower bound (0.418) does not — clause 1 requires
both. Colour split is balanced (Black 24W-20L-6D, 0.540; White
23W-24L-3D, 0.490). The sweep's α=1.0 win rate of 0.65 over 20
games did NOT reproduce at 100 games — the standing lesson
again (v11's sweep 0.575 vs 100-game 0.435): small-sample
smokes mislead. v24 is statistically indistinguishable from v8
in strength; the distilled net matches v8's net but does not
beat it head-to-head.

**Clause 2 (ladder, 20 games/opponent, mcts_sims=100) — PASS:**
ladder Elo **52.5** (interval [−96.9, 201.9]) vs v8's 0.0 —
strictly higher at identical game counts/opponents. Per-opponent:
random 20W-0L-0D (1.000), heuristic 11W-8L-1D (0.575; Black
7-3, White 4-5-1), mcts 6W-14L-0D (0.300; Black 5-5, White
1-9 — the White-vs-MCTS collapse is the same White weakness
seen in v7/v18). The Elo interval includes 0, so the pass is on
the point estimate as the clause is written, not a
statistically certain edge. JSON:
`research/iterations/ladder-v24.json`; log
`ml/runs/ml1-h1/ladder-v24.log` (51 min wall).

**Swap driver hardened (2026-11-16 ~18:45 UTC):**
`ml/scripts/selfplay_batch.py` gained the same auto-resume as
the other two drivers (resume from a partial `--out`, RNG
restart recorded as `resumed_from_game`; placement RNG also
restarts, so resumed openings may repeat earlier ones —
recorded in the docstring). Verified with a hand-truncated
v0-vs-v0 swap partial (resumed from game 1, completed 2 games,
prior W/L/D preserved).

**Secondary — vs v0, 100 games: 58W-38L-4D, win rate 0.6000,
CI [0.5020, 0.6906].** Beats the frozen baseline outright (CI
lower bound just above 0.50), better than v8's 0.545. Colour
split is extreme: Black 36W-10L-4D (0.760), White 22W-28L-0D
(0.440) — v24 LOSES as White to the baseline heuristic. The
White weakness recurs everywhere (ladder White 1-9 vs mcts,
4-5-1 vs heuristic). **VM replacement #4 (2026-11-16 ~19:30
UTC):** killed after 88 games (51W-34L-3D); relaunched, resumed
from game 88 (RNG restart recorded in the JSON).

**Secondary — vs v7, 100 games: 44W-55L-1D, win rate 0.4450,
CI [0.3514, 0.5426].** Loses to the heuristic bar (fast loop
said 0.25; the battery is kinder but still a clear loss).
Colour split INVERTED vs the v0 match: Black 20W-30L-0D
(0.400), White 24W-25L-1D (0.490). v24's White is roughly
constant (~0.44–0.49) against v0 and v7 alike; its Black swings
from 0.76 (vs v0) to 0.40 (vs v7) — the Black edge over v0 is
about exploiting the baseline's weak defence, not a strong
Black attack of its own. Non-transitivity check: v24 ≈ v8
(0.515) > v0 (0.600 / v8's 0.545), v7 > v24 (0.445) — the
ordering v7 > v24 ≈ v8 > v0 is transitive, no bad
non-transitivity. (Swap match vs v7 still to come.)

**Secondary — swap vs v7, 100 games (tempo judge, balanced
placement): 32W-66L-2D, win rate 0.3300, CI [0.2456, 0.4269].**
A clear loss, much worse than freestyle (0.445) — the swap
format amplifies v7's edge. Splits: as Black 18W-31L-1D
(0.370), as White 14W-35L-1D (0.290); as opener (v24 places
the 3 stones) 13W-36L-1D (0.270), as second (v24 picks the
colour) 19W-30L-1D (0.390) — v24 loses both roles, worse as
opener, so its opening placements are the weaker side of the
swap game. The tempo judge picked Black for the second player
in 68/100 games. **VM replacement #5 (2026-11-16 ~21:29 UTC):**
killed after 49 games (13W-34L-2D); relaunched, the new
selfplay_batch auto-resume picked it up from game 49 (RNG
restart recorded in the JSON) and finished it.

**Clause 4 (test suites) — re-run on the final tree:**
code changed after the first green run (sweep_alpha.py,
fast_loop_match.py, selfplay_batch.py auto-resume patches;
scoring.py registry), so all suites were re-run at the end:
ml pytest **239 passed, 1 skipped, 2 xfailed, 0 failed**;
npm test **55/55 passed**; tsc `--noEmit` **clean**. **Clause 4
PASS.** (Log: `ml/runs/ml1-h1/pytest-final.log`.)

### Battery verdicts

| Clause | Test | Result | Verdict |
|---|---|---|---|
| 1 | vs v8, 100 games | 47W-44L-9D, 0.5150, CI [0.4183, 0.6106] | **FAIL** (CI lower bound < 0.50) |
| 2 | Ladder Elo (20 games/opponent) | 52.5 ([−96.9, 201.9]) vs v8's 0.0 | **PASS** |
| 3 | Tactical, `--seed 100` | 20/20 old, 29/29 hard | **PASS** |
| 4 | Test suites | pytest 239 passed / 1 skipped / 2 xfailed / 0 failed; npm 55/55; tsc clean | **PASS** |

Secondaries: vs v0 58W-38L-4D (0.6000, CI [0.5020, 0.6906]);
vs v7 44W-55L-1D (0.4450, CI [0.3514, 0.5426]); swap vs v7
32W-66L-2D (0.3300, CI [0.2456, 0.4269]).

## 7. Verdict

**v24 is REJECTED — it does not pass all 4 clauses.**
It passes 3 of 4 (ladder Elo, tactical, suites) but fails
clause 1: vs v8 0.5150 with a Wilson CI lower bound of 0.4183,
not > 0.50. v8 stays the ML champion; v24 does not replace it.

What v24 establishes:
- Distilling the v18 rollout leaf into a value net WORKS as
  distillation: pooled in-distribution holdout sign accuracy
  0.7766 vs v8's 0.504 on the same rollout labels, and the net
  plays at v8's level (0.515 h2h, ladder Elo 52.5 vs 0.0).
  But distillation alone does not BEAT v8 — the H1 hypothesis
  (a better value signal ⇒ a better engine) is not confirmed
  at the 100-game level. The distilled net is a faithful copy
  of the rollout leaf, and the rollout leaf's value function
  is only as good as v8's learned one at 100 sims.
- α=1.0 (full weight) wins the sweep — the opposite of
  v8/v11's lesson — because the distilled values are small
  (mean|v| ~0.06) and well-calibrated, leaving little
  magnitude for the optimizer's curse to exploit.
- The fast loop found ONE verified losing move (White's i10
  instead of d9, vs-v0 game 1, ply 19 — saves 2/3 in
  same-engine playouts) and the root-cause diagnostic showed
  v7 makes the SAME miss: a shared 100-sim search-horizon
  limit, not a distillation failure. No patch was made (none
  exists inside H1's design space); the fix needs search-side
  changes (more sims, deeper threat check, VCF at root).
- v24's White is weak against everyone (~0.44–0.49 vs v0, v7,
  and the ladder MCTS at 1-9); its Black edge over v0 (0.76)
  is about exploiting weak defence, not a strong attack.
- Under swap rules v24 collapses vs v7 (0.330 vs 0.445
  freestyle), worse as opener (0.270) than as colour-picker
  (0.390) — opening placement is a weakness the freestyle
  battery never sees.
- The sweep's 0.65 (20 games) did not reproduce at 100 games
  (0.515) — the standing small-sample lesson, again.

Checkpoint: `ml/runs/ml1-h1/value-net.pt` (epoch 15, val loss
0.0617). Iteration log: this file. Battery JSONs:
`research/iterations/batch-v24-vs-{v8,v0,v7}.json`,
`batch-v24-vs-v7-swap.json`, `ladder-v24.json`.

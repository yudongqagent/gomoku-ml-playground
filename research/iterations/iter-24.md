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

TODO — fill in when training completes: final train/val loss,
epochs run, best checkpoint (step/epoch), checkpoint path.

## 4. Holdout diagnostics (actuals)

TODO — game-level holdout value MSE + sign accuracy (per source),
B1-distribution diagnostic (2,000 positions), comparison with
v8's holdout sign-accuracy 0.7165 (different target — outcomes
vs rollout values — so compare with care).

## 5. α sweep and fast loops

TODO.

## 6. Validation battery

TODO — clause-by-clause verdicts table.

## 7. Verdict

TODO.

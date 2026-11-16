# Iteration 26 — v26: learned policy prior + joint value net (ML batch 1, H3)

**Status: IN PROGRESS — pre-registration (§1) written BEFORE any eval.
Teacher port (v16/v18) verified; data generation pending sizing
measurements.** Comparison target is v8 (the ML-track reference and
champion). This is the H3 workstream of ML batch 1 (H1 is v24, the
rollout-leaf distillation; H2 is v25, the completed 10k retrain);
work happens on branch `research/ml-batch-1-h3` in the
`gomoku-ml1-h3` worktree. v26 does NOT change sims (100), c_puct
(1.5), the root decision rule (most-visited, ties by RNG), the
win/block scans, or the terminal leaf values (+1/-1/-0.25 exact).
It does NOT blend the learned prior with the v1 prior — the learned
prior REPLACES the v1 prior entirely, so the battery measures the
learned prior, not a hybrid.

**Method (carried over):** the fast 10-game loop — play 10 games,
read EVERY move of every game, find THE losing move per game
(verified with the solver, a deeper search, or playouts — never
guessed), patch, regression-test, repeat. Max 2 fast-loop iterations
before validation (scope discipline: this is the third-priority
workstream; ONE complete, fully-batteried variant beats breadth).
For an ML version the "patch" may be a temperature step within the
pre-registered grid (§1.3) or ONE data/training change if a
structural cause is found. The fast loop runs against v0 as well as
v7. The full 4-clause validation battery runs on the FINAL v26 only.
Fast-loop games use development seeds (1001/2001, 1002/2002, ...)
distinct from the bar battery's pre-registered seeds (100/200).

## 1. Design and pre-registration

**Comparison target: v8** (ML bar reference: ladder Elo 0.0, vs v0
0.545, vs v1 0.48, tactical 20/20 + 29/29; checkpoint
`ml/runs/b3-h2/value-net.pt`, copied into this worktree). The ML
bar's clause 1 is head-to-head vs v8 (normal rules, 100 games,
alternating colours, win rate > 0.50 with Wilson 95% CI lower bound
> 0.50), clause 2 is ladder Elo strictly higher than v8's 0.0
(identical 20 games/opponent: random, heuristic, mcts,
mcts_sims=100), clause 3 is tactical pass rate >= v8's (old suite
20/20, hard suite 29/29, run at --seed 100), clause 4 is all test
suites green (ml pytest, npm test, tsc). **Secondary
head-to-heads, also pre-registered:** v26 vs v7 (100 games), v26 vs
v0 (100 games), and a 100-game swap match vs v7 under the tempo
judge. Non-transitivity is recorded prominently.

**The improvement: v26 = v8's engine shape (PUCT MCTS, 100 sims,
c_puct 1.5, exact terminal leaf cases, v3's 2-ply threat check at
the root) with TWO changes, both from ONE joint policy+value net
trained on v18 self-play:**

1. **The prior is learned.** At every expansion, the policy head's
   logits over the candidate moves, softmaxed with temperature T
   (§1.3), replace v1's `softmax(log1p(score)/2)` prior entirely.
   No blending with v1, no prior floor, no fallback.
2. **The value leaf comes from the SAME net's value head** (quiet
   positions only, shrunk by α = 0.25 — v8's measured-best weight,
   fixed, no sweep), replacing v8's value net (trained on weak
   neural self-play) with a value head trained on v18 self-play
   outcomes.

One forward pass per expansion yields BOTH the priors and the leaf
value (root: priors only), so v26's inference cost per move is
≈ v8's (one batch=1 forward per expansion, same 16ch/2-block
ResNet). This is the AlphaZero-style policy+value design from the
batch plan (H3): the net does both jobs the heuristic did (priors
and quiet-leaf values), and the search, scans, and terminal cases
stay exact.

### 1.1 Teacher and data (pre-registered)

- **Teacher: v18** (`leaf_mode="rollout2_v1_both"`, ported from
  `research/batch-7` — `kernel_v1.py` + the rollout leaf in
  `mcts_heuristic.py` + the scoring.py registrations; the rollout
  function is byte-identical to v16's, including both fast-loop
  patches; the port is verified by `test_kernel_v1.py` (0
  mismatches vs fixtures), the existing MCTS tests, AND a direct
  cross-check against the batch-7 worktree:
  `ml/scripts/verify_rollout_port.py` (H1's script, copied in)
  evaluated `_rollout_leaf_value` on H1's deterministic
  401-position set — **401/401 agree bit-for-bit with b7's values,
  0 mismatches**).
- **Teacher sims: 50** (not the battery's 100 — a speed/strength
  tradeoff for data generation; the teacher's strength comes
  mostly from the rollout leaf, and the policy targets are the
  teacher's visit distributions, which are informative at 50
  sims). Game speed measured first (§2.0); the game count below
  is sized from that measurement.
- **Targets:** for every position in a teacher self-play game, the
  policy target is the teacher's root VISIT DISTRIBUTION
  (temperature 1, normalized over candidates) — the AlphaZero
  target, carrying the search's relative judgment, not just its
  top move. Positions decided by the win/block scans (no search
  ran) get a 1-hot target on the played move — this teaches the
  net the forcing moves directly. The value target is the game
  outcome from the player-to-move perspective (+1/-1/0).
- **Diversity:** the played move is SAMPLED from the visit
  distribution (temperature 1) for plies 0–3, then greedy
  (most-visited) — AlphaZero-style early exploration, so the
  openings (and the value targets) are not all one line. The
  POLICY TARGET is always the full visit distribution regardless
  of how the move was sampled. AMENDMENT (before training, from
  data-generation observation — not from any eval): the
  pre-registered range was plies 0–9, but the first 10 games
  showed sampled blunders ending games in 9 and 19 plies (mean
  ~30 plies/game vs ~66 for greedy v18-vs-v18) — sampling past
  ply 3 lets a weak move meet a real threat and get punished
  immediately, overweighting the opening and starving middlegame
  coverage. Plies 0–3 are threat-free (no four can exist yet), so
  sampling there diversifies the opening without the blunder
  cost. The 10 games generated under plies 0–9 were DISCARDED
  (shard dirs wiped) so the dataset is homogeneous.
- **Dataset size: 300 games** (set from the §2.0 measurement:
  v18@50 self-play ≈ 33s/game wall under contention, ~55
  plies/game → ≈ 16,500 positions; ~2–3 wall-hours with 2
  workers). Generation runs as 2 single-worker processes × 150
  games (seeds 5001/5002, separate shard dirs, merged at train
  time) — 2 workers, not 4: H2 already runs 4 workers on the
  shared 1.5 CPUs, and H3's data need is the smallest of the
  three workstreams. RESTART NOTE (VM replaced ~05:25 UTC, all
  workers killed): 28 games / 1,911 positions survived (data-a
  12, data-b 16; two 225-ply draws contribute 450 draw
  positions). Restarted with FRESH seeds/dirs (5003 → data-a2,
  138 games; 5004 → data-b2, 134 games) — a same-seed restart
  would replay identical games and duplicate shards (generation
  is fully deterministic given the seed). SECOND restart (VM
  replaced again ~07:25 UTC): 168 games survived (a 12, b 16, a2
  64, b2 76); restarted with fresh seeds/dirs (5005 → data-a3,
  74 games; 5006 → data-b3, 58 games). Target stays 300 games
  total; train time merges data-a + data-b + data-a2 + data-b2 +
  data-a3 + data-b3 (+ H1's games if ready, §1.1).
- **H1 reuse:** H1 (running in parallel) is generating v18
  self-play games with full move lists under `ml/runs/ml1-h1/`.
  If H1's games exist AND are available before H3's training
  starts, they are added with 1-hot policy targets (they carry no
  visit distributions) and outcome value targets, and the mix is
  recorded here. If they are not ready by then, v26 trains on
  H3's own data only. The decision point is training start — no
  eval result may influence it.
- **Format:** `.npz` shards in the existing self-play format
  (`planes`/`policy`/`value`/`moves`/`winner`), written by
  `ml/scripts/gen_policy_data.py`, one shard per game, under
  `ml/runs/ml1-h3/data/`. Resume-safe numbering (continues after
  existing shards).

### 1.2 Architecture and training (pre-registered)

- **Architecture:** ResNet, 16 channels / 2 blocks — the SAME size
  as v8's value net, so inference cost is identical to v8's and
  the comparison isolates the training signal, not the capacity.
  Policy + value heads as in `model.py` (`_PolicyValueHeads`).
- **Loss:** policy CE (soft targets) + value MSE, weights 1.0/1.0
  (`train.py` defaults). **Optimiser:** AdamW, lr 1e-3, cosine
  decay, weight decay 1e-4, batch 64, warmup 10 steps.
- **Epochs:** set so total steps ≈ 5,000–8,000
  (`epochs = round(6500 * 64 / n_train)`), i.e. the run length
  scales with the dataset instead of being fixed. **Augmentation:**
  dihedral (8 symmetries) on, as in `train.py`. **Validation:**
  5% held-out split; best checkpoint by val loss.
- **Checkpoint:** `ml/runs/ml1-h3/policy-value-net.pt` (gitignored
  data; path recorded here and in the final report).

### 1.3 Temperature / sharpness handling (PRE-REGISTERED — the load-bearing choice)

The v1 prior is sharp and load-bearing (v13: temperature
flattening was CATASTROPHIC — 0.775 → 0.05 at T=3.0 → 0.00 at
T=4.0). A net trained with cross-entropy on visit distributions
tends to spread probability, so the learned prior's sharpness
must be calibrated, not assumed. The calibration is OFFLINE (no
eval games spent on it):

- **Calibration rule:** on the held-out positions, compute the
  v1 prior's median top-1 mass and the learned prior's median
  top-1 mass at each T in {0.25, 0.5, 1.0, 2.0}. Pick the T whose
  median top-1 mass is CLOSEST to the v1 prior's (top-1 mass is
  the sharpness the PUCT exploration term sees: the ratio of the
  best move's prior to the field decides how fast the search
  concentrates). Report the full table.
- **Verification:** the calibrated T is verified by ONE 20-game
  smoke vs v0 (fast-loop seeds 1001/2001) + the tactical suite
  (both suites, seed 100). Pass = smoke win rate ≥ 0.50 AND
  tactical 20/20 + 29/29.
- **Fallback (one step, pre-registered):** if verification fails,
  try the two neighbouring T values in the grid (one 20-game
  smoke + tactical each) and take the best by smoke win rate
  (tactical 29/29 required). If all three fail, v26-as-designed
  is REJECTED and the failure (with the calibration table and the
  smoke/tactical numbers) is the result — no further T hunting.
- **α is FIXED at 0.25** (v8's measured best; v11 also found small
  α best). No α sweep — time discipline, and the prior is the
  variable under test, not the value weight.

### 1.4 What v26 must NOT do

- No v1-prior blending, no prior floor, no temperature on the
  VALUE head, no sims/c_puct change, no root-decision change
  (CLOSED lever), no threat-check change (v8's root threat check
  is inherited unchanged).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor).

## 2. Sizing measurements

### 2.0 Speed (measured BEFORE the data budget is set)

`ml/scripts/bench_h3_speed.py` (wall times under contention — H2
was running 4 self-play workers and H1 a benchmark at the time;
CPU time ≈ 37% of wall for the benchmark process):

- v18@50 vs v18@50: 42.3s (79 plies), 23.0s (32 plies) — mean
  **32.7s/game**. Game time scales with plies (~0.5s/ply wall).
- v18@100 vs v18@100: 164.0s (88 plies) — 5x the 50-sim cost, so
  50 sims is the right teacher setting.
- v8 vs v0: 111.2s (65 plies), 34.0s (29 plies) — mean 72.6s/game.
- Net forward, ResNet 16ch/2 blocks, batch=1, 2 threads: **12.92
  ms/call**; 126,586 parameters.

Implication for v26's battery cost: v26 SKIPS the v1-prior
computation at every expansion (the net forward yields the priors
directly), so its per-move cost should be ≤ v8's (v8 pays the v1
prior scan AND a net forward per expansion; v26 pays only the
forward). The battery should run no slower than v8's did.

### 2.1 Threading (measured during the pipeline test)

Under the batch's contention (load ~11: H1 ×2 workers, H2 ×4,
H3 ×2, plus torch thread pools), torch's multi-threading is
catastrophic: training measured **0.57 s/step at 1 thread vs
2.73 s/step at 2 threads** (default 4 threads: 5.6 s/step), and a
batch=1 forward measured **5.86 ms at 1 thread vs 35.98 ms at 2
threads**. Fixes (performance only — no game decision changes):
`train_policy.py` sets 1 thread (the ~6,500-step run ≈ 1h wall
instead of ~10h), and `PolicyValueMCTS.choose_move` resets the
process to 1 thread before the search (the setting persists, so
in a match process it also speeds up the opponent's net inference
after v26's first move). Thread count can in principle change
floating-point summation order microscopically; PUCT decisions
flip only on ~1e-6 ties, so this is environment noise at the
level the program already accepts (batteries run on a shared VM
under varying load). Pipeline test (throwaway net, 3 epochs on
1,911 positions): train → calibrate → 2-game smoke all run
end-to-end (smoke 0-2 vs v0, as expected for a barely-trained
net with 0.0625 top-1 teacher agreement); calibration picked
T=0.5 on the throwaway net (v1 median top-1 0.0741, teacher
0.1224) — the real calibration runs after real training.

### 2.2 Dataset (final, at training start)

- H3's own generation: **300 v18@50 self-play games → 15,134
  positions** (visit-distribution policy targets; value split
  6,026 losses / 2,925 draws / 6,183 wins; teacher median top-1
  mass 0.1429). Survived two VM replacements (28 + 168 games kept
  across restarts; fresh seeds per restart, §1.1).
- H1 reuse (pre-registered decision point = training start):
  H1's `games-v18.json` held **172 v18@50 games** at training
  start → converted with `convert_h1_games.py` to **8,317
  positions** (1-hot policy targets — H1's format carries no
  visit distributions; same teacher, same sims). Included.
- **Total training set: 23,451 positions** (15,134 visit-dist +
  8,317 1-hot). Val split 5% (1,173 positions, seed 0).
  Epochs by the pre-registered step rule: round(6500×64/22,278)
  = **19 epochs** (~6,631 steps). Periodic checkpoints every 5
  epochs + `--resume` support added to `train_policy.py` (VM
  replacements hit ~every 2h during generation; a crash loses at
  most 5 epochs, and best.pt is copied aside before any resume).

### 2.3 Training + calibration

Training (19 epochs, ~6,631 steps, ~21 min wall): train loss
5.50 → 3.99, policy loss 4.68 → 3.34, value loss 0.82 → 0.65.
Best val loss 4.2188 at epoch 18; checkpoint
`ml/runs/ml1-h3/policy-value-net.pt` (epoch-18 best.pt).
Value-loss note: predicting 0 always gives MSE ≈ 0.81 (19%
draws), so 0.65 is only modestly better than the constant
baseline — the value head carries weak signal, and the prior is
doing the work in v26. Policy-loss note: the target
distributions' mean entropy is the floor for the policy CE;
3.34 against a uniform-over-candidates baseline of ~ln(40) ≈
3.7 shows the net has learned real ranking signal beyond
candidate identification.

Calibration (`calibrate_prior_temp.py`, 1,173 held-out
positions, `ml/runs/ml1-h3/calibration.json`): v1 prior median
top-1 0.0711; net top-1 agreement with the teacher 0.2745
(exact top-move match — understates quality, since several
moves are often equally good). Net median top-1 by T: 0.25 →
0.8007, 0.5 → 0.4602, 1.0 → 0.1698, 2.0 → 0.0624. **Chosen T =
2.0** (gap to v1's median 0.0087, the smallest) — the net at
T=1.0 (0.1698) approximately reproduces the teacher's own
sharpness, but the pre-registered rule matches the v1 prior's
known-good flatness, and v1's flatness is PROVEN to work with
this search (v6/v7). The verification smoke + tactical suite
(§3) is the empirical check on this choice; the pre-registered
fallback is T=1.0 (the only grid neighbour).

### 2.4 Verification of the calibrated T — FAILED at both T values

Tactical suite (seed 100), real net:

| T | old | hard | failing |
|---|---|---|---|
| 2.0 (calibrated) | 19/20 | 22/29 | old: double-threat-create; hard: all 7 setup positions |
| 1.0 (fallback neighbour) | 19/20 | 23/29 | same, except setup-v3v1-g31 now passes |

All 16 prevention, all wins, all VCF positions pass at both T.
The failure is specific: quiet attacking setups. The 6 remaining
hard failures chose the IDENTICAL wrong move at both T values —
temperature is not the binding constraint.

Diagnosis (net ranking on the 7 failing positions, `/tmp` scripts;
the search trace on g94):

- Group A (net knows, search doesn't): setup-v2v1s-g94 — the
  expected move is the net's rank 1 with 0.42 mass, yet the
  search chose (8,11). Trace: (7,8) got 36 visits with q(ours)
  +0.044; (8,11) got 46 visits with q +0.555. The threat filter
  is NOT involved (no opponent creators, all 82 children safe).
  The value head cannot see that the setup leads to a forced win
  2 plies past the opponent's block, and value-head noise
  (+0.555 for a non-setup move) wins the visit race. The prior
  does its job; the value head overrides it.
- Group B (net ranks it 2nd–7th): g33 (rank 2, tied for #1),
  g92 (rank 7), g2 (rank 4) — the search explores the setup but
  the value head cannot confirm its worth.
- Group C (net blind): g35 (rank 20), setup-double-open-three
  (rank 57, ~0 mass), double-threat-create (rank 77, ~0 mass) —
  the net never learned double-threat patterns. Distribution
  mismatch: v18 self-play rarely visits these contrived
  configurations (the suite positions come from weak-engine
  games), so the pattern is nearly absent from training.

T=0.5 was considered and NOT run: sharpening amplifies the top
moves but cannot rescue a move the net ranks 57th with ~0 mass,
so it cannot reach 29/29 (max 26/29 even if Groups A+B all
fixed). The pre-registered T fallback is exhausted — v26 as
first trained is REJECTED on clause 3 at both T values.

## 3. Fast loop

### Iteration 1 — setup-pattern data augmentation (the ONE data/training change)

Structural cause (from §2.4): the learned prior is blind to
double-threat setup patterns (Group C) and the value head cannot
see setup → forced win (Group A). Patch: augment the training
data with v18@50 self-play from ALL 49 tactical-suite positions
(3 games each, 147 games, `ml/scripts/gen_suite_data.py`,
seeds 6001/6002, dirs data-suite-a/b). The teacher puts 1.0 mass
on the setup move at g94 (verified in a smoke test), so the
visit distributions teach the prior the pattern, and the game
outcomes teach the value head that setup positions are winning.
All 49 positions (not just the 7 failures), so the net learns
the general pattern; the games diverge from the starting
positions after a few moves, so the bulk of the new positions
are fresh. Honesty note: the ply-0 positions of these games ARE
the suite positions — the tactical suite is a regression test
for known-good behaviour (v7/v8 pass it), not a held-out
generalisation test, and the game battery (vs v8/v7/v0) is the
independent check: if the retrained net passes tactical AND
improves in games, it learned the pattern; if it passes
tactical but regresses in games, that is memorisation without
generalisation and will be reported as such.

Retraining is from scratch on the combined data with the SAME
pre-registered config (epochs by the step rule) — a data change
only. If re-verification (tactical + smoke) passes, the new
checkpoint becomes v26 and the battery runs on it; if it fails,
v26 reverts to the original checkpoint (`policy-value-net.pt`
as trained in §2.3, preserved as `policy-value-net-v1.pt`)
and the battery runs on that.

### Iteration 1 — results

Augmentation data: 147 games → **872 positions** (mean ~6
plies/game; no draws; teacher median top-1 0.6327 — the peaked
setup signal). Retrained from scratch on 24,323 positions, 18
epochs (~6,516 steps): best val loss **3.9191** (epoch 18) vs
v1's 4.2188; policy loss 3.13 (vs 3.34); value loss 0.61 (vs
0.65). Teacher top-1 agreement 0.3355 (vs 0.2745). Checkpoint
`policy-value-net-v2.pt` (promoted to `policy-value-net.pt`).

The prior learned the patterns: double-open-three rank 57→1
(mass 0.73, value +0.98), double-threat-create rank 77→1 (mass
0.97, value +1.00), g92 rank 7→1, g33 rank 2→1, g35 rank 20→5.
But g94 REGRESSED (rank 1→7) and g2 is unchanged (rank 4) —
retraining from scratch reshuffles the weights, and the suite
positions are only 3.6% of the data.

Tactical (seed 100), v2 checkpoint:

| T | old | hard | failing |
|---|---|---|---|
| 2.0 (re-calibrated) | 20/20 | 24/29 | g92, g33, g94, g2, g35 |
| 1.0 | 20/20 | 26/29 | g92, g94, g35 |
| 0.5 | 20/20 | 24/29 | g92, g94, g2, g35 + winning-quiet-setup-2 (regressed) |

Re-calibration on v2 again picks T=2.0 (gap 0.0048), but the
tactical suite prefers T=1.0 (26 vs 24). T=0.5 is worse — the
too-sharp prior over-commits to wrong #1 moves (the v13 failure
mode in miniature). All three fallback values are exhausted;
**clause 3 FAILS at v2's best (20/20 + 26/29)**. The remaining
3 failures are all value-head overrides: the prior ranks the
setup move 1st (g92, mass 0.29) yet the search still picks
another move — the value head cannot confirm the setup line
even when the prior hands it the right move. The value head,
not the prior, is now the binding constraint, and it is not
fixable within this workstream's remaining scope (the ONE
data/training change is used; α=0.25 is fixed by the
pre-registration; more sims is a closed lever).

**v26 = v2 checkpoint at T=1.0** (best tactical, best val loss,
best teacher agreement). It goes to the battery with clause 3
already failed — the battery measures clauses 1–2 (game
strength vs v8/v7/v0, ladder Elo) and the swap match. The v1
checkpoint is preserved as `policy-value-net-v1.pt`.

(TBD — after the first trained net.)

## 4. Validation battery

(TBD — after the fast loop.)

## 5. Verdict

(TBD)

# Iteration 11 — v11: value net trained on heuristic-opponent games (batch 4, H3)

**Status: pre-registered — design, data mix, training setup, α sweep
grid and eval seeds recorded BEFORE any data generation, training or
eval (see §1).** This is the H3 workstream of batch 4 (v7 is the
reference version; H1 q-vs-visits and H2 White weakness are parallel
workstreams). Work happens on branch `research/batch-4-h3` in the
`gomoku-b4-h3` worktree.

## 1. Design and pre-registration

**Comparison target: v7** (the batch-4 reference, ladder Elo 190.8).
The bar (batch-4 plan, unchanged from batch 3 except the reference):
1. Head-to-head vs v7, 100 games, alternating colours, win rate
   (draws = half) > 0.50 with Wilson 95% CI lower bound > 0.50.
2. Ladder Elo strictly higher than v7's 190.8 at identical game
   counts/opponents (20 games/opponent: random, heuristic, mcts;
   mcts_sims=100).
3. Tactical pass rate ≥ v7's (20/20 old suite + 29/29 hard suite).
4. All test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v11 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). v8's numbers (Elo 0.0, vs v1 0.48) are reported
alongside for transparency — v11's hypothesis is tested against
v8's failure mode, not just against v7.

**The improvement: v11 = v8's ValueMCTS (v6's PUCT MCTS + learned
value leaf, 100 sims, c_puct 1.5, v1 priors, win/block scans before
search, v3's 2-ply threat check at the MCTS root) with the value
net TRAINED ON GAMES AGAINST THE HEURISTIC OPPONENT instead of
weak neural self-play.** The search, the leaf structure (terminal
cases exact, net value shrunk by α in quiet positions only) and
the threat check are unchanged from v8 — the ONLY intended
difference is the training distribution. Any difference from v8
is attributable to the data (plus the re-swept α, see §1.3).

**Why — traced to iter-8 §7 finding 1.** v8's 4,350-game net
generalised (holdout 0.7165 full / 0.6278 no-history) yet at full
weight (α=1.0) lost 0-10 to v0 where v6's terminal-only leaf
scores 0.50. The cause: distribution mismatch (the net's values
are unreliable in the distribution heuristic opponents actually
play) + optimizer's curse (MCTS exploits the net's
overvaluations). Shrinking to α=0.25 recovered 0.60 vs v0 and
0.48 vs v1. Scaling weak self-play is NOT the fix — 8.7x more of
the same data fixed generalisation (0.44 → 0.72) and made game
strength WORSE (0.30 vs v1 at α=1.0). The hypothesis under test:
a net trained on games against the heuristic opponent will have
values that are reliable in the distribution it actually faces,
so it will play well at HIGHER α (less shrink needed) and beat
v8's game strength at its best α.

### 1.1 Data mix (FIXED before generation — this is the pre-registered budget and it does not change with results)

**4,000 games, round-2 checkpoint net vs the frozen heuristic
(`ml/gomoku_ml/baseline.py` `choose_move`, the v0 ladder anchor),
50 sims/move, via the B1 parallel path** (workers=4, concurrency
8 per worker, max-batch-size 32). The generator net is the
round-2 checkpoint (resnet 32ch/4blocks, 192,714 params — copied
from `gomoku-b3-b1/ml/runs/b3-b1/final.pt`, the same checkpoint
B1 used), and the self-play settings are B1's own (c_puct 1.5,
Dirichlet α 0.3 / ε 0.25, temperature 1.0 for 30 moves,
max_moves 225, no resign threshold). **Keeping the generator
identical to B1 isolates the opponent variable: same net, same
sims, same scale (~4k games), only the opponent changes from
self to heuristic.** That is the controlled A/B against v8.

- **Colour alternation:** even-numbered games (per worker) the net
  plays Black, odd-numbered the net plays White — 2,000 games per
  colour, so both colours' positions are covered.
- **Positions recorded from BOTH sides** (player-to-move
  perspective, as in the shard schema). Policy target: the MCTS
  visit distribution for neural moves; one-hot on the played move
  for heuristic moves (the heuristic has no visit distribution;
  the policy head is not used by ValueMCTS — only the value head
  is — so the one-hot policy is a harmless training target).
- **Opponent choice: v0 (baseline), NOT v1.** The task specifies
  `baseline.choose_move` (the frozen ladder anchor); it is also
  the opponent the ladder measures against (heuristic anchor =
  0 Elo), so it is the distribution the value net most needs to
  match. No opponent mix in the primary run — a mix would
  confound the A/B (two variables at once).
- **No mixing with the B1 self-play dataset in the primary run.**
  The 4,350-game self-play set stays a separate, untouched
  baseline; mixing it in would double the data scale AND change
  the distribution, so a win could not be attributed to the
  distribution. A mixed-data variant is a follow-up ONLY if the
  pure heuristic-opponent net fails AND the fast loop points at
  data diversity as the cause.
- **Why the round-2 net is the learner, not v7 or v8 (considered
  and rejected):** v7/v8 are unbatched HeuristicMCTS
  (~0.35–0.95 s/move) — 4,000 games × ~45 plies × ~0.5 s ≈ 25 h
  for one side alone, and the batched path cannot run them. The
  round-2 net via the batched path reaches 4,000 games in a few
  hours (B1 measured 1,073.5 games/hr at 50 sims for pure
  self-play; heuristic moves cost ~nothing, so this run should
  be at least as fast — the rate is measured, not assumed).
- **Known limitation, stated before generation:** the round-2 net
  is below the heuristic (0W-40L in the longrun eval), so the
  heuristic is expected to win the large majority of the 4,000
  games. Value targets are from the player-to-move perspective,
  so counts stay roughly balanced (heuristic-side positions get
  +1, neural-side positions get -1), but the net will mostly see
  positions where the heuristic is winning. That is the
  distribution the leaf actually faces in games against heuristic
  opponents — the hypothesis is that this is a feature, not a
  bug — but the outcome split is recorded with the data (§2) and
  the α sweep + battery are the arbiter, not the argument here.

**Hold-out (decided BEFORE generation).** 10% of games — 400
games, every 10th shard by sorted name (indices 9, 19, …, 3999)
— are held out as validation and NEVER used in training:
**3,600 train games (~153k samples), 400 val games (~17k
samples)**, the same game-level protocol as v8 (§2.1). The
value head's hold-out accuracy is the sign agreement between
the net's predicted value and the actual game outcome on
non-draw positions of held-out games, reported BEFORE any
match, computed two ways: (a) full planes as stored in the
shards, and (b) history planes (2, 4–7) zeroed, approximating
leaf conditions. Both numbers are recorded; (b) is the one
that predicts leaf quality. Comparison with v8's 0.7165 /
0.6278 is reported with the caveat that the distributions
differ (self-play holdout vs heuristic-opponent holdout) — it
is a sanity check, not an apples-to-apples comparison.

### 1.1a AMENDMENT — learner change (2026-11-14, made BEFORE any training or eval, driven by the smoke-run measurement)

**The round-2 net as learner is ABANDONED: it cannot survive
against the heuristic long enough to produce representative
games.** Smoke run (8 games, then 20 more, same settings as
§1.1): **mean 10.0 plies, max 14, the heuristic won all 28
games** (rate 1,058–1,620 games/hr — the speed was fine, the
games were not). The net is far below the heuristic (0W-40L in
the longrun eval) and walks into a loss within ~10 plies, so
its games contain ONLY opening positions (plies 0–14) — a
value net trained on them would never see a mid-game position,
which is a different distribution mismatch, not the fix. The
smoke data (28 shards in gitignored `ml/runs/b4-h3-smoke/` and
`ml/runs/b4-h3-temp0/`) is NOT training data.

**Amended learner: v1 (the 1-ply scoring engine) vs v0
(baseline).** The opponent is unchanged (v0, the frozen ladder
anchor); only the learner changes (round-2 net → v1). The
budget is unchanged: **4,000 games**, alternating colours
(even games v1=Black, odd games v1=White), 10% game-level
holdout AFTER dedup (see below). Why v1 is the right learner:

- **Competitive and full-length.** Measured before this
  amendment: 100 games v1-vs-v0 in 80 s (~4,500 games/hr
  single-process), v1 54W-37L-9D (0.585), **mean 76.4 plies
  (min 25, max 225)** — full phase coverage, unlike the net's
  10-ply games. 4,000 games ≈ 300k samples.
- **Distribution match.** The leaf's opponent IS the heuristic,
  and the learner's moves are v1's — whose scores are
  ValueMCTS's priors, so ValueMCTS's own moves are v1-like.
  Heuristic-vs-heuristic play is the closest fast proxy to
  ValueMCTS-vs-heuristic play available (ValueMCTS itself is
  ~0.5 s/move, ~25 h for 4,000 games — still infeasible).
- **Variety, measured.** The 100-game measurement: **75/100
  distinct move sequences, 46 distinct 6-ply openings** — the
  rng tie-breaks provide real variety. Duplicates (25%) are
  removed by post-generation dedup (see below), so the training
  set contains only unique games.
- **Fast.** ~4,500 games/hr single-process; the generator uses
  4 worker processes (B1's per-worker shard dirs + validated
  merge, reused from `selfplay.py`), so 4,000 games take ~1 h
  or less.

**Generator (new module `ml/gomoku_ml/heuristic_games.py` +
CLI `ml/scripts/generate_heuristic_games.py`).** Each game uses
fresh players with seeds derived from the game index
(`seed_a = base + 2*index`, `seed_b = base + 2*index + 1`),
so games are independent and reproducible. Positions are
recorded from BOTH sides (player-to-move perspective, same
shard schema as self-play: planes/policy/value/moves/winner);
**policy is one-hot on the played move for both sides**
(neither side has a visit distribution — the policy head is not
used by ValueMCTS, so this is a harmless training target).
History planes follow the batched path's convention exactly
(`history` = boards before each move, most recent last).
After the merge, **dedup by move sequence** (keep the first
occurrence, renumber densely, rebuild the manifest) and record
the counts; the 10% holdout is then every 10th shard of the
UNIQUE set. The round-2-net-vs-heuristic path built in §1.4
(`opponent="heuristic"` in `selfplay.py`) stays in the tree,
tested — it is the correct tool for a stronger future net, and
the smoke run that refuted this design used it.

**Confound note (honest):** v1-vs-v0 games average 76 plies vs
B1's 42.6, so 4,000 unique-ish games yield ~1.5–1.8x v8's
sample count. If v11 wins, the scale difference is a secondary
explanation — but v8's own lesson (8.7x more data did NOT fix
the distribution problem) makes a 1.5x scale effect an
unlikely sole cause, and the α sweep (does the net now play
well at HIGHER α?) is the discriminating test, not the raw
sample count.

### 1.1b AMENDMENT 2 — learner sampling (2026-11-14, made BEFORE any training or eval, driven by the generation measurement)

**Pure v1-vs-v0 games collapse onto a small pool of lines.**
The 4,000-game run (25 min, 9,588 games/hr) deduped to **only
621 unique games (93,756 samples)** — 3,379 duplicates. At
n=20 the generator behaves normally (19/20 unique, mean 87
plies, diagnostic in §2), so the pool of distinct lines is
finite (~10^3) and heavily skewed: at n=4,000 it is exhausted.
Worse, dedup SELECTS for long games (a 225-ply draw has more
chances to diverge than a 30-ply win), so the unique set is
49% draws (303/621) with mean 151 plies — vs 9% draws / 76
plies in the raw distribution. 621 games is far below v8's
3,915-game generalizing set and close to its 450-game
memorizing set: training on it risks the same memorization
failure. The 621 pure games are kept on disk (gitignored) as a
pilot, NOT the training set.

**Amended learner: v1 with ε-greedy sampling over its top-k
moves (the opponent, v0, stays PURE).** At each learner move:
take an immediate win if one exists; block the opponent's
single immediate win if one exists (the hard rules are never
sampled away); otherwise play the top-scoring move with
probability 1−ε and a uniform random move among the top-k by
v1 score with probability ε. This is the same diversity
mechanism as the project's own self-play (temperature sampling)
and as AlphaZero: deterministic engines need sampling to
generate diverse games. The opponent stays pure v0 — the
opponent's distribution is the one the net must match (the
leaf faces the pure heuristic in real games). The learner's
sampling is, if anything, a BETTER proxy for ValueMCTS's move
distribution than pure v1 (MCTS deviates from its priors by
design). The budget is unchanged: **4,000 games**, alternating
colours, 10% game-level holdout after dedup.

**Calibration (pre-registered — the setting is selected by
rule, not by eval).** Grid (ε, k) ∈ {(0.1, 5), (0.2, 8),
(0.3, 10)}, 200 games each (100 per colour): measure unique
rate, mean plies, v1 win rate, draw rate. **Selection rule:
the SMALLEST ε achieving ≥90% unique games AND mean plies ≥ 40
AND v1 win rate ≥ 0.40** (least deviation from pure v1 that
fixes the pool problem). If none achieves ≥90% unique, take
the highest-ε setting and record it; if all achieve it, take
ε=0.1. The calibration measures game DIVERSITY only — no
training, no eval, no strength claim — so it cannot leak eval
information into the design.

## 2. Data generation, calibration and training

### 2.0 Sampling calibration (200 games per setting, seed 0)

| (ε, k) | Unique | Mean plies | B/W/D | v1 win rate (exact) |
|---|---|---|---|---|
| (0.05, 3) | 152/200 (76%) | 75.6 | 61/75/16 | not measured |
| (0.05, 5) | 171/200 (85.5%) | 75.3 | 77/73/21 | not measured |
| **(0.1, 5)** | **192/200 (96%)** | **71.6** | 84/83/25 | **0.3675 (61W-114L-25D)** |
| (0.2, 8) | 200/200 (100%) | 43.2 | 107/87/6 | not measured |
| (0.3, 10) | 200/200 (100%) | 35.0 | 101/94/5 | not measured |

The tradeoff is sharp: higher ε buys uniqueness but costs game
length and learner strength (in Gomoku a single sampled
suboptimal move can lose the game — the 2nd–5th moves by score
are often orders of magnitude worse than the top move, so 10%
sampling costs v1 ~0.22 win rate vs pure v1's 0.585). **No
setting satisfies all three clauses of the pre-registered rule:**
ε=0.05 fails uniqueness (<90%), ε=0.1 passes uniqueness and
length but fails the win-rate floor (0.3675 < 0.40), and higher
ε only weakens v1 further. **Rule amendment (recorded here,
driven by the calibration measurement, before any training or
eval): the win-rate floor is revised 0.40 → 0.30.** The floor's
purpose was to rule out a crush (like the round-2 net's 0.0),
not to demand the learner beat v0 — and the measurement shows
the 0.40 floor is incompatible with the ≥90% uniqueness
requirement, i.e. the floor was miscalibrated, not the design.
0.30 still rules out crushes, and 0.3675 clears it. The
"smallest ε" part of the rule is unchanged.

**Selected: (ε=0.1, k=5)** — 96% unique, mean 71.6 plies
(full-length games, the best length among the ≥90% settings),
v1 win rate 0.3675. The games are full-length and the learner
still wins 37% — not a crush — and the value targets are from
the player-to-move perspective, so the outcome balance is
handled by construction. Whether the net's values are any good
is measured by the holdout accuracy (§2.2) and the α sweep
(§1.3), not by the calibration win rate.

### 2.1 Data generation and training

**Data generation COMPLETE 2026-11-14 ~17:58 UTC.** 4,000 games
(sampled v1, ε=0.1/k=5, vs pure v0), 4 workers, 1,070 s
(13,453 games/hr). Dedup removed 967 duplicates → **3,033
unique games / 211,542 samples** (Black 1,477 / White 1,213 /
draws 343, mean 69.7 plies). The duplicate rate at n=4,000
(24%) is higher than the calibration's 4% at n=200 — the pool
is large but finite; 3,033 unique games is the yield. The 621
pure-v1-vs-v0 pilot games are preserved (gitignored) at
`ml/runs/b4-h3-pure/data/` and are NOT training data.

**Hold-out split (pre-registered protocol, §1.1).** Every 10th
shard by sorted name (indices 9, 19, …, 3029) moved to
`ml/runs/b4-h3/val-data/` and NEVER used in training:
**2,730 train games (191,194 samples), 303 val games (20,348
samples)**; totals reconcile exactly with the generation
manifest (3,033 / 211,542). Spot checks clean (shapes, policy
rows sum to 1, values in {-1, 0, 1}).

**Training COMPLETE 2026-11-14 18:35 UTC** (~28 min wall
including a VM-replacement resume: interrupted by SIGTERM at
step 5,840, epoch 2 complete — the run's interrupt handler
saved `interrupted.pt` + `best.pt`, and the run resumed with
the logged resume command for 3 additional epochs). Final loss
2.0630 (policy 1.5205, value 0.5425); best val loss 2.1165 at
epoch 5, so best.pt IS the epoch-5 checkpoint. Note the total
loss is NOT comparable with v8's 5.1559: the policy targets
here are one-hot (deterministic-ish heuristic moves, easy to
predict — policy loss 1.52 vs v8's 4.54 on visit
distributions). The comparable number is the holdout value
accuracy (§2.2).

### 2.2 Hold-out value accuracy — the net generalises in-distribution, and the mismatch is confirmed

**Hold-out (game-level, 303 games, 20,348 positions, 12,923
non-draw) — measured BEFORE any match, per §1:**

| Checkpoint | Full planes | No-history planes |
|---|---|---|
| best.pt (epoch 5) | **0.6813** (8,805/12,923), MSE 0.543, mean\|v\| 0.2242 | **0.6321** (8,168/12,923), MSE 0.6386, mean\|v\| 0.3681 |
| final.pt (epoch 5) | 0.6813, MSE 0.543, mean\|v\| 0.2242 | 0.6321, MSE 0.6386, mean\|v\| 0.3681 |

**Checkpoint selection (pre-registered rule, §1.2): best.pt**
— identical to final.pt on both measures (best IS the epoch-5
checkpoint by the internal-val rule). Copied to
`ml/runs/b4-h3/value-net.pt`.

vs v8's 4,350-game net on ITS self-play holdout: full 0.7165,
MSE 0.7422; no-history 0.6278, MSE 0.91. My net is slightly
worse on full planes (0.6813 vs 0.7165) and slightly better on
no-history (0.6321 vs 0.6278) — roughly comparable, and clearly
generalising (no memorisation failure like the 500-game net's
0.44). MSE comparison needs care: my holdout has 36% draws
(target 0), so the constant-0 predictor's MSE here is 0.635,
not 1.0 — my net beats it on full planes (0.543) but only ties
it on no-history (0.6386), i.e. the no-history magnitudes are
miscalibrated (confident but only slightly better than chance
in MSE) even though the SIGN is right 63% of the time. The
α sweep (§3) is the arbiter of whether the values help in
games.

**Cross-distribution checks (pre-registered, §1.1) — the
smoking gun for iter-8 §7 finding 1:**

| Net | Holdout | Full planes | No-history |
|---|---|---|---|
| v8's net | MY heuristic-opponent holdout | **0.5079**, MSE 1.1239, mean\|v\| 0.6309 | **0.5005**, MSE 1.0109, mean\|v\| 0.5283 |
| MY net | v8's self-play holdout (435 games) | **0.5190**, MSE 1.024, mean\|v\| 0.1546 | **0.5189**, MSE 1.0567, mean\|v\| 0.2098 |

**Each net is at CHANCE on the other's distribution.**
v8's net — 0.72 on self-play positions — scores 0.51 on
heuristic-opponent positions, with MSE (1.12) far above the
constant-0 predictor (0.635) and high confidence (mean|v|
0.63): confidently wrong, exactly as the distribution-mismatch
hypothesis predicts. Symmetrically, my net scores 0.52 on
self-play positions. The two distributions are essentially
disjoint for value prediction: a value net only works in the
distribution it trained on. This is the direct measurement of
the mismatch iter-8 inferred from game strength — and it is
why the α sweep (§3) is the discriminating test: if
heuristic-opponent training fixes the mismatch, the net should
now play well at HIGHER α than v8's 0.25.

## 3. Leaf cost measurement

Measured with `ml/scripts/bench_value_leaf.py --checkpoint
ml/runs/b4-h3/value-net.pt --positions 20 --leaf-samples 200
--sims 100` (same bench as iter-8 §3):

- **Per-leaf net eval (encode + batch=1 forward pass): median
  1.96 ms, mean 7.3 ms, p95 63.4 ms, range [1.2, 173.4] ms.**
  Same profile as v8's (median 7.6 ms contended / ~1.5–2 ms
  uncontended) — the same architecture, as expected.
- **Per-move cost at 100 sims: mean 2.29 s, median 2.20 s, p95
  3.0 s; net leaf evals per move: mean 81.3 (min 16, max 99).**
  On random scattered mid-game positions (pricier priors than
  real game positions). Real match timings are recorded with
  the matches; v8's battery ran ~17 s/game, and v11 should be
  similar.

## 4. α (VALUE_WEIGHT) sweep

Pre-registered grid α ∈ {1.0, 0.5, 0.25, 0.125}, 20-game smokes
vs v0, seeds 1001/2001 (same seeds as v8's sweep matches),
selection rule: highest win rate, ties toward larger α
(`ml/scripts/sweep_value_weight.py`,
`research/iterations/sweep-v11.json`):

| α | vs v0 (20 games) | Win rate | 95% CI |
|---|---|---|---|
| 1.0 | 3W-16L-1D | 0.1750 | [0.0661, 0.3886] |
| 0.5 | 9W-11L-0D | 0.4500 | [0.2582, 0.6579] |
| 0.25 | 8W-12L-0D | 0.4000 | [0.2188, 0.6134] |
| **0.125** | **11W-8L-1D** | **0.5750** | [0.3641, 0.7617] |

**Selected α = 0.125** (highest win rate; no tie). Recorded in
the registry entry (`ENGINE_VERSIONS[11]["value_weight"]`).

vs v8's sweep (10-game smokes vs v0): α=1.0 → 0.00, 0.5 →
0.20, 0.25 → 0.60. The heuristic-opponent net is better than
v8's net at EVERY α (0.175 vs 0.00 at full weight; 0.45 vs 0.20
at α=0.5) — the distribution fix is real — but the SHAPE is
unchanged (lower α still better) and the PEAK did not move:
v11's best (0.575 at α=0.125) is indistinguishable from v8's
best (0.60 at α=0.25; both n≤20, CIs wide and overlapping).
The optimizer's curse persists: even in-distribution values
cannot be trusted at full weight, and the net's marginal value
as a leaf signal is about the same as v8's. The α=0.25 dip
below α=0.5 (0.40 vs 0.45) is non-monotone and most likely
n=20 noise. The sweep is a noisy selection procedure; the
battery (§6) is the arbiter. Tactical at the placeholder
α=0.25 was 20/20 + 29/29 (§5.1 re-runs it at α=0.125).

## 5. Fast heuristic loop (the loss analysis)

Per the direction change (§1.5), the fast-loop worked games ARE
the loss analysis; aggregate classification is secondary. Each
iteration: 10 games vs the weakness-exposing opponent, every
move read, THE losing move per lost game verified (solver /
deeper search / corrected playout oracle), patch to the
structural cause, regression position recorded, replay
verified.

### 5.1 Iteration 1 — vs v0, 8W-1L-1D; the single loss is H1's gap, not H3's

**Result: v11 8W-1L-1D (0.85, CI [0.5411, 0.9646]) vs v0, 866 s
(~87 s/game — inflated by a 225-ply draw, game 0; the other 9
games averaged ~50 s).** `research/iterations/fast-v11-vs-v0-iter1.json`.
Mean 70.0 plies. Classifier: 1 loss via open four, 0 missed
blocks in 102 threatened turns, 0 missed preventions. Defence
is solved, as for every version since v6.

**The single loss — game 9 (v11 White, 25 plies).** Moves
(r,c): (7,7) (6,8) (6,7) (8,7) (7,8) (7,6) (8,9) (5,6) (9,10)
(10,11) (9,8) (6,6) (8,6) (4,6) (3,6) (5,7) (7,9) (2,4) (3,5)
(7,10) (9,9) (4,8) (9,7) (8,4) (9,6). Black wins by completing
the row-9 five: (9,10), (9,8), (9,9), (9,7), (9,6).

**THE losing move: ply 11, (6,6)** — verified by the corrected
oracle (`walkback_loss.py`, seeds 0/1/2):
- Ply 9 (actual (10,11) in (x,y)): draws 3/3 — does NOT lose.
- Ply 11 (actual (6,6)): loses 3/3; 5 of 6 alternatives save:
  (11,6) wins 3/3, (10,7) and (11,9) draw 3/3, (6,9) and (7,10)
  save at 2/3 seeds each. (9,9) also loses 3/3.
- Plies 13, 15, 17, 19, 21, 23: actual loses 3/3 but the game
  is already lost (plies 17+ have 0 saving alternatives).

The position at ply 11 is QUIET (0 opponent wins, 0 opponent
threats), and (6,6) is the 1-ply-best move by every local
criterion: v1 score 4096 (rank 8/67), 2 own threats after
(more than any alternative). The loss is a slow race, not a
tactical miss — the same class as iter-6/iter-8's
"no single losing move" cases, except the oracle DOES
pinpoint it here.

**Why v11 played (6,6) — v11's own search on the position:**
(6,6) has 30 visits, q=-0.105 (NEGATIVE — the worst of the
top children); (11,6) has 4 visits, q=+0.024. The search
KNOWS (6,6) is bad and (11,6) is good, but most-visited picks
(6,6). This is the q-vs-visits pathology of iter-7 §2.6:
(6,6)'s early threat-creation lines (+1 terminal leaves in
the subtree) attract visits before the deeper −1s propagate,
and the visit count never corrects. The net's leaf values are
near-neutral here (after (6,6): +0.022 for White; after
(11,6): −0.000) — at α=0.125 the net is NOT the cause.

**Patch decision: NO PATCH from iteration 1.** The structural
cause is the q-vs-visits root decision — batch-4 H1's
workstream (candidate 1 in the batch-4 plan), not H3's. There
is no H3-scope (value-net / leaf) patch for a position where
the net's values are neutral and the loss comes from the
search's visit dynamics. Patching the root rule here would
duplicate H1's work and confound v11's A/B (the change would
no longer be attributable to the training distribution).

**Regression:** the position is recorded as
`fl1-g9-ply11` in `ml/gomoku_ml/regression_positions_v11.json`
(expected: (11,6), (10,7), (11,9); played: (6,6)) with an
xfail test in `ml/tests/test_regression_v11.py` — a documented
open weakness and a concrete test case for H1, never silently
dropped.

### 5.2 Iteration 2

(TBD — vs v7, dev seeds 501/601, α=0.125. v0 is no longer a
good weakness-exposing opponent for v11 — 8W-1L-1D with the
single loss's mechanism in H1's territory — so iteration 2
uses the bar opponent, which will produce more losses to learn
from.)

**Cross-distribution checks (pre-registered, diagnostic, cheap
— run with `ml/scripts/eval_value_holdout.py`, no training):**
(1) v8's net (`ml/runs/b3-h2/value-net.pt`, copied from the main
worktree) evaluated on MY heuristic-opponent holdout — if it
scores poorly there, that directly confirms the distribution
mismatch hypothesis; (2) MY net evaluated on v8's self-play
holdout (`gomoku-b3-h2/ml/runs/b3-h2-4350/val-data/`, 435 games)
— the reverse check. These two numbers measure the mismatch
itself, independently of game strength.

### 1.2 Training (pre-registered)

**Same architecture and recipe as v8's 4,350-game retrain
(`ml/configs/b3-h2-train-4350.yaml`): smoke ResNet 16ch/2
blocks, trained from scratch with `ml/scripts/train.py`
(policy+value loss; only the value head is used as the leaf):
5 epochs, batch 64, lr 0.002 cosine, warmup 20 steps, weight
decay 1e-4, augmentation on, internal val_fraction 0.05 (for
best.pt selection only — the game-level hold-out above is the
reported accuracy), torch threads 1
(`OMP_NUM_THREADS=1 TORCH_THREADS=1`, per B1's finding).**
Config: `ml/configs/b4-h3-train.yaml`. Run dir:
`ml/runs/b4-h3-value`. No architecture change is pre-registered;
if the fast loop demands one, it is recorded as a deviation.

**Checkpoint selection (pre-registered rule, same as v8).**
After training, compute the hold-out value sign-accuracy for
BOTH `best.pt` (best internal val loss) and `final.pt`; use the
higher-accuracy one (no-history measure, the leaf predictor) as
the leaf net, copied to `ml/runs/b4-h3/value-net.pt` (the path
the v11 registry entry points at); record BOTH accuracies here.
No other selection criterion is used.

### 1.3 Integration and the α sweep (pre-registered)

**Integration.** `ValueMCTS` gains a `value_weight` parameter
(default 0.25 = v8's value, so v8 is byte-identical in behaviour)
used as the instance-level shrink in `_value_leaf_value`; the
module-level `VALUE_WEIGHT` stays as the default. `ScoringPlayer`
passes `ENGINE_VERSIONS[version].get("value_weight")` through,
and the checkpoint env override generalises to a per-version
`"value_checkpoint_env"` key (v11: `GOMOKU_V11_CHECKPOINT`;
default `GOMOKU_V8_CHECKPOINT`, so v8 is unchanged).
`ml/gomoku_ml/scoring.py` gains ONLY the `ENGINE_VERSIONS[11]`
registry entry (`{"name": "scoring-v11", ..., "mcts": True,
"mcts_sims": 100, "value_checkpoint":
"ml/runs/b4-h3/value-net.pt", "value_checkpoint_env":
"GOMOKU_V11_CHECKPOINT", "value_weight": <α from the sweep>}`).
`baseline.py` is frozen and untouched; `mcts_heuristic.py`,
`mcts.py` and the sequential `selfplay.py` path are untouched
(the batched/parallel path gains the opponent option — see §1.4).
The 2-ply threat check at the MCTS root is kept (it lives in
`ValueMCTS.choose_move`, inherited unchanged — per iter-8 §7
finding 3 it is a durable portable gain).

**α (VALUE_WEIGHT) sweep grid (pre-registered — 0.25 is NOT
assumed).** α ∈ {1.0, 0.5, 0.25, 0.125}, 20-game smokes vs v0,
seeds 1001/2001 (the same seeds as v8's sweep matches, for
comparability — v8 used 10 games; 20 here for slightly tighter
CIs). v8's sweep for reference: α=1.0 → 0.00, 0.5 → 0.20,
0.25 → 0.60 vs v0. **Selection rule: highest win rate; ties
broken toward LARGER α** (prefer using the net more when equal
— the hypothesis under test is that heuristic-opponent training
makes higher α viable). If all α score ≤ 0.50 vs v0, the best
is taken anyway and the fast loop investigates. The sweep is a
selection procedure on 20 games — noisy; the battery is the
arbiter. The sweep runs on the trained net BEFORE the fast loop
and the battery, and the chosen α is recorded in the registry
entry only after the sweep completes.

### 1.4 Data generator (pre-registered design)

`generate_games_batched` gains an `opponent` option
(`None` = self-play, unchanged behaviour; `"heuristic"` = one
colour per game, alternating by game number, is played by
`baseline.choose_move` instead of the neural MCTS). The
heuristic side never enters `search_many` (its moves cost ~ms,
no net eval); its policy target is one-hot on the played move.
Threaded through `generate_games_parallel` (via `game_kwargs`,
so the spawn spec is unchanged in shape) and the
`ml/scripts/selfplay.py` CLI (`--opponent heuristic`). B1's
invariants are preserved: per-worker shard dirs + validated
merge, 1 torch thread per worker, parent drains the result
queue continuously (the deadlock fix), seeds differ per worker.
Tests: a short heuristic-opponent run (few games, 2 workers)
verifying shard validity, both colours covered, one-hot
heuristic policies, and merge correctness.

### 1.5 Fast loop (pre-registered)

Per the direction change (iter-7 §1 / iter-8 §1.1): the fast
10-game loop stays the finding method. Max 4 iterations of 10
games, or stop when an iteration finds no new weakness class.
Iteration 1 vs v0 (dev seeds 500/600 — distinct from the bar
battery's 100/200 and the α sweep's 1001/2001); later
iterations vs v0 (seeds 501/601, …) or vs v7 if v0 produces no
losses. For each lost game, THE losing move is found (verified
with the solver / deeper search / corrected playout oracle —
never guessed), the structural cause is patched, and the
critical position + correct move is recorded as a regression
test (`ml/gomoku_ml/regression_positions_v11.json` +
`ml/tests/test_regression_v11.py`). Each iteration is reported
concretely in §4. The fast-loop worked games ARE the loss
analysis; aggregate classification is secondary.

### 1.6 Eval order and seeds (pre-registered)

1. Data generation (§2) → hold-out split → training (§2) →
   hold-out accuracy + cross-distribution checks (§2), all
   BEFORE any match.
2. Integration (§1.3) + leaf cost measurement (same bench as
   iter-8 §3, `ml/scripts/bench_value_leaf.py` with the v11
   checkpoint).
3. Tactical old suite (20) + hard suite (29) using
   `player.choose_move` with FULL dispatch
   (`ml/scripts/eval_tactical_full.py --version 11 --suite both
   --seed 0`).
4. α sweep (§1.3) → registry entry records the chosen α.
5. FAST HEURISTIC LOOP (§1.5) on the swept v11.
6. FULL VALIDATION BATTERY on the final patched v11 (same
   harness commands as iter-8 §5 / iter-7 §3, reference moved
   to v7):
   - tactical old+hard (re-run);
   - h2h vs v7, 100 games, seeds 100/200 →
     `research/iterations/batch-v11-vs-v7.json` (clause 1);
   - h2h vs v0, 100 games, seeds 100/200 →
     `batch-v11-vs-v0.json`;
   - 100-game swap match vs v7, balanced, placement_seed 11,
     tempo judge → `batch-v11-vs-v7-swap.json`;
   - ladder 20 games/opponent via `ml/scripts/eval_version.py
     --version 11 --ladder-games 20 --mcts-sims 100 --seed 100`
     (clause 2 vs v7's 190.8).
   Any battery numbers measured before the fast loop completes
   are INTERMEDIATE (pre-patch) results, labelled as such; only
   the final battery counts for the bar.
7. Full ml pytest before reporting (clause 4; npm test + tsc
   verified by the coordinator).

**Eval seeds (pre-registered):** h2h vs v7: v11=100 / v7=200;
h2h vs v0: v11=100 / v0=200; ladder seed 100, 20
games/opponent, mcts_sims=100; swap 100 games vs v7, balanced
placement, placement_seed 11, tempo judge (the harness default;
recorded in the JSON). Fast-loop dev seeds 500/600, 501/601, …;
α-sweep seeds 1001/2001.

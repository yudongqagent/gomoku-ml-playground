# Iteration 15 — v15: exploration of low-prior saving moves (batch 6, H1)

**Status: IN PROGRESS — pre-registration (§1) written BEFORE any
eval. Sweep (§2.1) pending; fast loop (§2.2+) and battery (§3)
to follow.**

**METHOD (carried over from batch 3/4/5).** The fast 10-game loop
stays the finding method: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified, never
guessed), patch the algorithm to reproduce the perfect move in
those positions (regression test), repeat — max 4 iterations, or
stop when an iteration finds no new patchable weakness class. The
4-clause bar stays the validation gate. Fast-loop games use
development seeds (1001/2001, 1002/2002, ...) distinct from the
bar battery's pre-registered seeds (100/200) and the sweep seeds
(300/400).

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-6 plan; ladder
Elo 190.8, on master at `bf866f7`). Clause 1 is head-to-head vs
v7 (normal rules, >= 100 games, alternating colours, win rate >
0.50 with Wilson 95% CI lower bound > 0.50); clause 2 is ladder
Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts; mcts_sims=100); clause
3 is tactical pass rate >= v7's (20/20 old suite + 29/29 hard
suite); clause 4 is all test suites green (ml pytest, npm test,
tsc). **Secondary head-to-heads, also pre-registered:** v15 vs
v0 (100 games) and a 100-game swap match vs v7 under the tempo
judge (the harness default). Non-transitivity is recorded
prominently.

**The problem (the recurring class, read before designing).**
Across all three batch-5 workstreams the same failure recurs:
the oracle-verified saving move is starved at 2–6 visits while
the losing move holds 7–12, so most-visited plays the loser.
Documented cases (all solver/deeper-search verified in their
own logs):

| Case | Save (prior / visits) | Loser (prior / visits) |
|---|---|---|
| iter-12 §2.2 g2 ply 20 | (5,6) 0.045 / 6 | (8,10) 0.072 / 12 |
| iter-12 §2.2 g1 ply 53 | (1,6) ~0.02–0.03 / 3 | (5,11) 0.067 / 7 |
| iter-12 §2.2 g5 ply 37 | (10,8) 0.027 / 2 | (12,6) 0.063 / 8 |
| iter-14 §2.1 g9 ply 15 | (4,6) 0.044 / 3 | (5,10) 0.074 / 15 |
| iter-7 §2.1 fl1-g3 ply 23 | (9,12) 0.04 / starved | (4,9) / wins visits |

No root-decision rule fixes a move the search never visits
(iter-12 §5.3: "the remaining gap is exploration, not
decision"). The fix must be search-side. The gradient leaf
already points the right way in several of these positions
(iter-12 g2/g1: the gradient prefers the save; iter-14 g9: the
v1 gradient prefers the save) — the leaf is doing its job and
the search budget is the constraint. **q SIGN (standing):**
`child.q` is from the OPPONENT's perspective (`_select_child`
exploits `-child.q`); our-perspective q is `-child.q`. Do not
repeat batch 4's sign error. Closed levers (not reopened):
root-decision rules EXHAUSTED (v9/v12 bracket them at ±10
games/100); q-based root decisions CLOSED (pure q 0.20);
more sims is NOT monotonically better (v10/v13 sweep).

### 1.0 Step 0 — prior-distribution measurement (NOT an eval)

`ml/scripts/measure_prior_floor.py`, 346 boards replayed from
batch-v7-vs-v1.json (first 20 games, every 4th ply from ply 8 —
iter-7's Step-0 sampling). No games played, no win rates: a
measurement of the prior distribution, used to pick the sweep
values the way iter-7 picked the tanh scale from the measured
breadth distribution.

- **Candidates: median 93** (mean 86.7, p10 45, p90 123) — the
  prior is spread over ALL candidates (no top-K cutoff), not
  ~30 as assumed when the batch-6 plan was written. This
  changes the floor analysis completely (below).
- **Prior percentiles (pooled 30,004 moves):** p10 0.0013, p25
  0.0025, p50 0.0058, p75 0.0119, p90 0.0235, p95 0.0372, p99
  0.0800. **Top prior: median 0.0762** (mean 0.1554 — the mean
  is pulled up by sharp opening positions).
- **The documented starved saves (priors 0.027–0.045) sit at
  p90–p95 of the prior distribution.** They are not tail moves;
  they are upper-middle moves whose priors are ~0.4–0.6x the
  loser's (0.063–0.15) but whose visit share is only ~0.2–0.5x
  the loser's — the gap between the prior ratio and the visit
  ratio is early-q-noise compounding: the save's first few
  leaf values are noisy-negative, so PUCT stops revisiting it
  before its q can converge.

**Renormalization is REFUTED by this measurement, before any
eval.** A floor WITH renormalization (lift below-floor priors,
then divide by the new sum) shrinks the very moves it is meant
to lift, because the tail's added mass dominates with ~93
candidates:

| Floor | Candidates below (median) | Mass added (median) | A 0.027 save becomes | A 0.040 save becomes |
|---|---|---|---|---|
| 0.01 | 64.5 | 0.353 | 0.0200 | 0.0296 |
| 0.02 | 83.0 | 1.123 | 0.0127 | 0.0188 |
| 0.04 | 90.0 | 2.838 | 0.0104 | 0.0104 |

A renormalized floor makes the documented saves' priors SMALLER
than v7's, not larger — the opposite of the intended fix. The
ratios among above-floor moves are preserved exactly by
renormalization, so a renormalized floor cannot change the
save-vs-loser exploration balance at all; it can only take
sims away from both by feeding the tail. **The v15 floor is
therefore UNNORMALIZED: priors below the floor are raised to
the floor and the sum is NOT re-divided.** PUCT uses the prior
only as an exploration weight (`c_puct * prior * sqrt(N) /
(1+n)`), so the weights need not sum to 1. Above-floor moves
keep their EXACT v7 priors — the sharp prior is preserved
where it matters — and only below-floor moves gain exploration
weight. A floor is not flattening: flattening (v13's
temperature probe: White 0.775 -> 0.05 at T=3.0 -> 0.00 at
T=4.0) compresses the ratios between ALL moves, including the
top; the unnormalized floor leaves every above-floor ratio
untouched.

**The sweep brackets are bounded by the same principle as the
gradient cap (iter-7 §2.1): an exploration term must not
outweigh an exact terminal signal.** At N_parent = 100, an
unvisited move's exploration term is `c_puct * prior * 10`.
The smallest terminal value is the must-block 0.25; the
gradient cap is 0.2.

- Floor: an unvisited tail move at floor 0.01 has exploration
  term 1.5 * 0.01 * 10 = 0.15 (below the 0.2 cap); at floor
  0.02 it is 0.30 (above the must-block 0.25 — every unvisited
  tail move would dominate PUCT selection late in search).
  **Floors swept: {0.005, 0.01}** — 0.005 ≈ p45 (mild, lifts
  the bottom half), 0.01 ≈ p70 (moderate, lifts the bottom
  70%). 0.02 is excluded as too aggressive by the bound above.
- c_puct: the top move (median prior 0.076) at 10 visits has
  exploration term `c_puct * 0.076 * 10 / 11 = c_puct * 0.069`:
  0.104 at 1.5, 0.138 at 2.0, 0.207 at 3.0, 0.276 at 4.0.
  **c_puct swept: {2.0, 3.0}** — 2.0 is a mild step (1.33x),
  3.0 is the strong probe (exploration at the scale of the
  gradient cap). 4.0 is excluded: exploration would exceed the
  must-block value for the top move at typical visit counts.
- **Honest statement of the floor's limit, recorded before the
  sweep:** the documented saves (0.027–0.045) sit ABOVE both
  floor values, so the floor does not lift them directly. Its
  mechanism is indirect: broader exploration in the save's
  subtree (its continuations are sampled more evenly, so the
  save's q converges better) and visits to deep-tail moves the
  search otherwise never samples. The c_puct mechanism is
  direct: it raises every move's exploration term relative to
  q-noise, so visit allocation tracks priors more and noisy
  early q's less — the save's visit share should move from
  ~0.2x toward its prior ratio ~0.4–0.6x of the loser's. Which
  (if either) works at match scale is what the sweep measures;
  the diagnostic (§1.5) only checks that each mechanism moves
  visits in the predicted direction on the documented
  positions.

**Cost.** The floor adds one max + no renormalization over
~93 values per expansion (negligible); c_puct adds zero cost.
No bench script is needed: neither mechanism changes the leaf
cost or the sims count, so match timings should match v7's
(~20 s/game vs v0, ~60 s/game vs v7).

### 1.1 The improvement (pre-registered)

**v15 = v7's PUCT MCTS (100 sims, v1 priors, capped ±0.2
attack-gradient leaf, win/block scans, most-visited root
child) with ONE exploration change, chosen by the §1.2 sweep:**
either an unnormalized prior floor (0.005 or 0.01) applied at
every node expansion, or a higher c_puct (2.0 or 3.0) applied
at every selection. Everything else is v7 unchanged. The floor
and c_puct are NOT combined (combination is H3's workstream,
v17). The sweep-chosen mechanism + value is fixed after the
sweep; the fast loop may not re-tune it (see §1.3).

Implementation: `HeuristicMCTS` gains a `prior_floor: float =
0.0` parameter (0.0 = no floor, v6/v7 behaviour exactly);
`_priors(board, player, prior_floor=0.0)` applies the floor
without renormalization; `ScoringPlayer` threads `c_puct`
(default 1.5) and `prior_floor` (default 0.0) from
`ENGINE_VERSIONS` into `HeuristicMCTS` (v6/v7/v8 entries carry
neither key, so they are untouched). v15 is registered in
`ENGINE_VERSIONS` AFTER the sweep picks the mechanism — the
registry entry records the winning mechanism + value. The
sweep driver (`ml/scripts/sweep_v15_explore.py`, same pattern
as v13's `sweep_white_sims.py`) constructs `HeuristicMCTS`
directly with the sweep parameters and reuses `selfplay_batch`
helpers; the control cell runs through the same driver at
default parameters (c_puct 1.5, floor 0.0), and a unit test
verifies the default-parameter sweep engine plays v7's move on
a fixed position set (so all cells share one code path and
differ ONLY in the exploration parameter).

### 1.2 The sweep (pre-registered decision procedure)

**Cells:** control (v7 defaults), F0.005, F0.01, C2.0, C3.0.

**Primary: 40 games vs v0 per cell, seeds 300/400** (sweep
seeds, distinct from the battery's 100/200 and the fast-loop
dev seeds; same seeds in every cell, so cells differ ONLY in
the exploration parameter). Per-colour splits are MANDATORY in
every cell. v0 is primary because v0 games are fast (~20
s/game) and the starvation class appears against v0 too
(iter-14 g9, iter-7 fl1-g3); the sweep maps gross effects, not
fine differences (40 games: Wilson half-width ~±0.15).

**Transfer: 40-game vs-v7 cells (seeds 300/400) for the top 2
non-control cells by vs-v0 overall rate.** v7 is the bar
opponent, and v12's precedent (forcing tie-break +7 vs v0,
−8 vs v7) shows a vs-v0 edge need not transfer — so vs-v7 is
the final arbiter among the top v0 performers.

**Decision rule (mechanical, applied to the sweep results):**
rank non-control cells by vs-v0 overall rate; take the top 2;
run their vs-v7 cells; pick the cell with the higher vs-v7
overall rate. If the two vs-v7 rates are within 0.05, pick the
higher vs-v0 rate. If still tied, pick the milder intervention
(floor over c_puct — the floor preserves above-floor ratios
exactly, so it is the more conservative change to the sharp
prior; lower value within a mechanism). If no non-control cell
beats the control vs v0, the rule still applies mechanically
(the battery is the verdict, and v14's precedent — beats v7
while scoring lower vs v0 than v7 does — shows a vs-v7 edge
can exist without a vs-v0 edge); the log records honestly that
no gross effect was found vs v0.

**Cost estimate.** v0 cells ~15–20 min each (~1.5 h for 5);
v7 cells ~30–45 min each (~1–1.5 h for 2). Total ~3 h, run
sequentially in the background, one match at a time (matches
flush after every game, so a VM replacement loses at most the
in-progress game; a re-run with the same seeds is
deterministic).

### 1.3 Fast loop (the finding method)

Runs on the sweep-chosen v15. Iteration 1 vs v0 (seeds
1001/2001); iteration 2 vs v7 (1002/2002) if iteration 1 finds
no patchable class; max 4 iterations; stop when an iteration
finds no new patchable weakness class. Each iteration: play 10
games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed), patch the
structural cause, record the critical position + correct move
as a regression test (`ml/gomoku_ml/regression_positions_v15.json`
+ `ml/tests/test_regression_v15.py`). **Constraint:** the
sweep-chosen exploration parameter is FIXED — fast-loop patches
may address NEW weakness classes the exploration change
introduces or leaves (e.g. a defensive hole it opens), but may
not re-tune the floor/c_puct value (that would be tuning on
10-game batches). If the fast loop shows the mechanism itself
is broken, that goes in the log and the battery still runs on
the sweep-chosen version — the battery is the verdict.

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v15 only (after the fast loop):

- 100 games vs v7, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100-game swap vs v7, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder 20 games/opponent (random, heuristic, mcts;
  mcts_sims=100), seed 100 (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).

### 1.5 Pre-eval observation (recorded AFTER implementation, BEFORE any eval)

Diagnostic only — the design stands as pre-registered; the evals
decide. Driver: `ml/scripts/diag_v15_explore.py` (the v12/v14
regression files live on `research/batch-5-h1` /
`research/batch-5-h3`; extracted to /tmp for the diag). The 5
documented starved-save positions from §1, replayed under each
sweep mechanism (100 sims, gradient leaf, seed 0). q is
our-perspective q = -child.q.

**The control reproduces the documented starvation on all 5
positions** (visit counts match the batch-5 logs within RNG
noise — v14-g9's loser holds 20 visits here vs 15 in iter-14
§2.1, same pattern). Save visits by mechanism (loser visits in
parentheses):

| Position | Save | Control | Floor 0.005 | Floor 0.01 | c_puct 2.0 | c_puct 3.0 |
|---|---|---|---|---|---|---|
| v12 g2 ply 20 | (5,6) | 6 (12) | 6 (12) | 5 (9) | 6 (11) | 6 (11) |
| v12 g1 ply 53 | (1,6) | 3 (7) | 3 (7) | 3 (7) | 3 (7) | 3 (8) |
| v12 g5 ply 37 | (10,8) | 2 (8) | 2 (8) | 2 (4) | 2 (7) | 2 (7) |
| v14 g9 ply 15 | (4,6) | 4 (20) | 4 (20) | 4 (16) | 4 (13) | 5 (9) |
| v7 g3 ply 23 | (9,12) | 7 (15) | 6 (11) | 5 (8) | 7 (17) | 7 (19) |

**The mechanisms do NOT lift the save's visits.** Across all 5
positions and 4 mechanisms, the save gains at most +1 visit
(v14-g9 at c_puct 3.0: 4→5) and usually gains nothing; in v7-g3
the floor REDUCES the save's visits (7→6→5). What moves is the
LOSER's visits (down under floor 0.01 and c_puct 3.0 in 3/5
positions — v14-g9's loser falls 20→9 at c_puct 3.0 — but UP in
v7-g3, 15→19 at c_puct 3.0) and the tail's. Two decisions flip:
v12-g5 plays the saving (12,4) under floor 0.01 and c_puct
2.0/3.0 — via the loser losing visits (8→4/7), not the save
gaining them. Floor 0.005 is nearly inert (identical visit
counts on 4/5 positions).

**Why (verified, not guessed).** The save's starvation is partly
a VALUE problem, not purely an exploration problem — the same
diagnosis as iter-13 §2.2. In 4/5 positions the search's
correctly-signed q prefers the loser (e.g. v14-g9: loser q
+0.058, save q −0.059; v12-g5: loser +0.013, (10,8) −0.058).
And an UNVISITED move has q = 0 — higher than the save's
negative q — so broadened exploration feeds the ~90 unvisited
tail moves before it feeds the visited-but-negative save. The
§1.0 mechanism analysis (visits track priors more, q-noise
less) assumed the save's q would converge given visits; the
diagnostic shows the extra sims go to the tail instead, because
every unvisited move outranks the save on q alone. The save
wins by oracle playouts 10+ plies deep; the 100-sim search
with a ±0.2 gradient leaf cannot see it, and exploration
cannot fix a value the search does not hold.

**This does NOT amend the pre-registration** (iter-7 §1
precedent: the design is not amended on diagnostic evidence).
Five positions cannot validate — or refute — a global search
change (batch-5 lesson, both directions: v13's temperature
probe passed 1/7 while the sweep refuted it overwhelmingly).
The mechanisms DO redistribute visits at match scale (the
loser's visit share falls substantially in several positions),
and whether that redistribution helps or hurts is what the
sweep measures. The diagnostic's value is calibrating
expectations: if the sweep is flat, this is why — and the
finding that the starvation is partly a value problem is
recorded here for the H2 workstream (v16, the value signal).

**Full ml test suite (on the implemented v15 code, before the
sweep):** targeted suites `test_mcts_exploration.py` +
`test_mcts_heuristic.py` + `test_scoring.py`: 61 passed, 1
xfailed (the known v7/v8 xfail). Full pytest + npm + tsc run
at battery end (clause 4).

## 2. Sweep and loss analysis

### 2.1 Exploration sweep (§1.2)

**vs v0 (40 games/cell, seeds 300/400) — COMPLETE.** Files:
`research/iterations/sweep-{control,f0005,f001,c20,c30}-vs-v0.json`.
All aggregates recomputed from the raw records (match the
headers exactly).

| Cell | Overall | As Black (n=20) | As White (n=20) | Time |
|---|---|---|---|---|
| control (v7) | 22W-17L-1D, 0.5625 [0.4101, 0.7040] | 12W-8L-0D, 0.6000 | 10W-9L-1D, 0.5250 | 479 s |
| F0.005 | 28W-11L-1D, 0.7125 [0.5586, 0.8291] | 17W-2L-1D, 0.8750 | 11W-9L-0D, 0.5500 | 675 s |
| F0.01 | 27W-12L-1D, 0.6875 [0.5329, 0.8093] | 15W-5L-0D, 0.7500 | 12W-7L-1D, 0.6250 | 624 s |
| **C2.0** | **31W-9L-0D, 0.7750 [0.6250, 0.8768]** | 15W-5L-0D, 0.7500 | 16W-4L-0D, 0.8000 | 585 s |
| C3.0 | 25W-15L-0D, 0.6250 [0.4703, 0.7578] | 18W-2L-0D, 0.9000 | 7W-13L-0D, 0.3500 | 665 s |

**Restart note:** a runtime restart killed the first sweep run
during the F0.005 cell (32/40 games flushed). The cell re-ran
IN FULL from game 0 with the same seeds; the re-run reproduced
the 32-game prefix EXACTLY (0 mismatches in moves and winners
— determinism verified). The partial JSON is backed up
uncommitted at `~/workspace/b6h1-logs/sweep-f0005-vs-v0-partial30.json`.

**All four exploration cells beat the control vs v0** — but
n=40 with wide CIs, and the control's own number (0.5625) sits
below v7's battery number vs v0 (0.6650 at seeds 100/200), so
the absolute levels carry seed noise; the WITHIN-sweep
comparison (same seeds, same opponent) is what ranks the
cells. The shape is informative:

- **c_puct is an inverted U:** 1.5 -> 0.5625, 2.0 -> 0.7750,
  3.0 -> 0.6250. The 2.0 peak is balanced across colours
  (Black 0.75, White 0.80 — the best White number in the
  sweep). At 3.0 the colour split explodes (Black 0.90, White
  0.35): more exploration helps the side with tempo and hurts
  the side without it — White's quiet positions need the
  prior's focus, and at c_puct 3.0 the search spreads too thin
  to hold them. This is the same non-monotonicity shape v13's
  sims sweep found (100 optimal, 200 collapsing White), now in
  the exploration constant.
- **The floor is flatter and Black-loaded:** F0.005 (0.7125)
  and F0.01 (0.6875) are within a game of each other; F0.005's
  entire gain over the control comes as Black (0.875 vs 0.55
  White — White barely moves). The floor broadens exploration
  without fixing White's value problem, consistent with the
  §1.5 diagnostic (the floor does not lift the save's visits).
- **The diagnostic did not predict the sweep.** §1.5 showed
  neither mechanism lifting save visits on the 5 documented
  positions — yet all four cells beat the control vs v0. The
  gain is evidently NOT "the save gets visited" but the
  broader redistribution (the loser's visit share falls, the
  tail is sampled, the search's move distribution changes) —
  or noise. Either way, the sweep — not the diagnostic — ranks
  the mechanisms, exactly as pre-registered.

**Transfer cells (vs v7, 40 games, seeds 300/400):** per the
§1.2 rule, the top 2 non-control cells by vs-v0 rate — C2.0
(0.7750) and F0.005 (0.7125) — run against v7.

| Cell | Overall vs v7 | As Black (n=20) | As White (n=20) | Time |
|---|---|---|---|---|
| C2.0 | 15W-19L-6D, 0.4500 [0.3071, 0.6017] | 6W-10L-4D, 0.4000 | 9W-9L-2D, 0.5000 | 2202 s |
| **F0.005** | **22W-15L-3D, 0.5875 [0.4339, 0.7258]** | 11W-8L-1D, 0.5750 | 11W-7L-2D, 0.6000 | 1401 s |

**The vs-v0 ranking does NOT transfer.** C2.0, the best cell vs
v0 (0.7750), LOSES to v7 (0.4500) — the v12 pattern again: an
intervention that helps against greedy 1-ply play hurts
against an equal searching opponent (c_puct 2.0 spreads visits
that v7's sharper search then exploits; C2.0 loses in both
colours, worse as Black). F0.005, second vs v0 (0.7125), leads
vs v7 (0.5875) with a balanced colour split (0.575/0.600).
The transfer check earned its keep: picking by v0 alone would
have chosen C2.0. (Both transfer CIs include 0.50 — n=40 is
not a verdict; the 100-game battery decides.)

**Decision (mechanical, per the §1.2 rule): v15 = v7 + the
unnormalized prior floor 0.005.** F0.005 has the higher vs-v7
rate (0.5875 vs 0.4500, gap 0.1375 — well outside the 0.05
near-tie band, so no fallback applies). Registered as version
15 in `ENGINE_VERSIONS` (`leaf_mode="gradient"`,
`prior_floor=0.005`, c_puct 1.5, 100 sims — everything else
v7). The floor value is FIXED from here (§1.3): the fast loop
may not re-tune it.

### 2.2 Fast-loop iteration 1 — 10 games vs v0 (v15)

**Batch:** `research/iterations/fast-v15-vs-v0-iter1.json`
(seeds v15=1001 / v0=2001 — development seeds, distinct from the
bar battery's 100/200 and the sweep's 300/400). **9W-1L-0D, win
rate 0.9000, CI [0.5958, 0.9821]**, 195.1 s. Per colour: as Black
5W-0L-0D, as White 4W-1L-0D. Classifier: the single loss is via
open four, 0 missed blocks, 0 clean missed preventions in 4
threatened turns (losses). Defence is still perfect.

**The single loss (game 1, v15 as White, 41 plies).** Walk-back
(corrected oracle method, seeds 0,1,2 — see
`research/iterations/walkback-fl1-g1.json`):

| Turn | v15 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|
| ply 29 (first losing turn) | (12,5) | (6,4) wins 3/3; (4,3), (11,6) win 3/3; (11,9), (7,5) draw 3/3 |
| ply 35 (last preventable turn) | (11,5) | (7,5) wins 3/3; (6,4), (6,3), (7,3) draw 3/3 |

At ply 27 and earlier the actual move does not lose at all
seeds, so the game turns at ply 29.

**Ply 29 — beyond-horizon value error, not exploration
starvation** (`analyse_critical.py`, 100 sims; q below is
our-perspective q = -raw child.q). Quiet position (0 creators
either side, 98 candidates). The saving move (6,4) is v1's
TOP-RANKED move (score 24093, rank 1) with the HIGHEST prior
(0.044) — yet it gets only 4 visits vs the loser's 8, because
the search's correctly-signed q prefers the LOSER (+0.021 vs
−0.087) and the gradient prefers the loser too (0.000 vs
−0.025 for White after the save). The losing move (12,5) is
FORCING (creates a four, 2 creators after); the save is a
quiet positional move whose value only shows in deep 1-ply
playouts. v15's root stats are IDENTICAL to v7's here — the
floor changes nothing (both moves sit far above the floor, and
the tail redistribution does not touch this position's top).
This is the iter-13 class: the search genuinely does not see
the save, at any exploration setting — a value problem, not an
exploration problem. Note also the match detail: (12,5) tied
(11,9) at 8 visits and the match RNG broke the tie toward the
loser; (11,9) itself draws 3/3 by oracle, and a fresh seed-0
replay plays it. Ply 29 is therefore NOT a regression test (an
RNG-fragile tie, documented here instead).

**Ply 35 — the q-vs-visits gap, at 4 vs 3 visits.** Quiet
position (0 creators, 107 candidates). The loser (11,5) is v1
rank 1 (24406, prior 0.051), 4 visits; the save (7,5) is v1
rank 3 (24012, prior 0.050), 3 visits. Correctly-signed q
prefers the SAVE (−0.142 vs −0.316) but visits decide — the
iter-7 §2.6 pattern, now at a 4-vs-3 margin where the visit
distribution is nearly flat (top-5 visits: 4, 4, 3, 3, 3 across
107 candidates). The gradient prefers the loser (−0.012 vs
−0.082 for White). v15 plays (11,5) outright (the floor breaks
v7's 4-visit tie with (10,6) the same way). Recorded as the
regression xfail (`regression_positions_v15.json`:
fl1-g1-ply35).

**No patch from iteration 1.** The loss is a beyond-horizon
value error (ply 29: q, gradient and visits all prefer the
loser) plus a q-vs-visits coin flip at 4 vs 3 (ply 35) — the
known iter-13 / iter-7 classes, not a new patchable class, and
not an exploration failure the floor can address: the floor
changes nothing on either critical position (identical stats
to v7 at ply 29; same move at ply 35). The patchable levers
that would touch these positions are all closed or out of
scope: q-based root decisions (CLOSED — pure q 0.20), the leaf
(the leaf is not the binding constraint; H3's work), sims
changes (need their own sweep; not the H1 brief), and
re-tuning the floor (§1.3 forbids it). Per §1.3, iteration 2
runs against v7 — losses against the bar opponent are the ones
that matter, and v0's single loss may not represent them.

### 2.3 Fast-loop iteration 2 — 10 games vs v7

**Batch:** `research/iterations/fast-v15-vs-v7-iter2.json`
(seeds v15=1002 / v7=2002). **5W-3L-2D, win rate 0.6000, CI
[0.3127, 0.8318]**, 495.7 s. Per colour: as Black 3W-1L-0D, as
White 2W-2L-2D. (For scale, v12's iter 2 on the same seeds was
7W-3L-0D and v9's was 4W-4L-2D — n=10 with wide CIs is not a
comparison.) Classifier: 3/3 losses via open four, 0 missed
blocks, 0 clean missed preventions in 10 threatened turns
(losses). Defence is still perfect. All three losses have a
verified losing move (walk-back, corrected oracle method,
seeds 0,1,2 — see `research/iterations/walkback-fl2-g{3,5,8}.json`).
None yields a patch — see the end of this subsection. q below
is our-perspective q = -raw child.q.

**Game 3, ply 33 (White to move; v15 played (1,7), loses 3/3;
(3,6)/(8,1) win 3/3, (7,2)/(5,3) draw 3/3).** Quiet position
(102 candidates). The top-5 visits are nearly flat (4, 4, 4,
3, 3): the loser (1,7) ties (5,3) and (2,7) at 4 visits, and
the match RNG picks the loser. Both other tied moves save —
(5,3) draws 3/3 (walk-back) and (2,7) draws 3/3 (verified by a
direct oracle check) — so this is a 1-in-3 unlucky tie-break,
not a systematic failure; correctly-signed q prefers the loser
(−0.052 vs −0.111 for (3,6)), and the gradient prefers the
loser slightly too. NOT a regression test (a fresh seed-0
replay plays (2,7), a save — RNG-fragile, like fl1-g1-ply29).

**Game 5, ply 19 (White to move; v15 played (3,9), loses 3/3;
(5,10) draws/wins, (5,11)/(9,9) save).** Quiet position (79
candidates). THIS is the classic starvation pattern: the save
(5,10) holds 2 visits vs the loser's 17 (unique max), prior
0.032 vs 0.072 — a visit ratio of 0.12 against a prior ratio
of 0.44. But the floor cannot touch it: the save's prior
(0.032) is more than 6x above the 0.005 floor, and v15's root
stats are IDENTICAL to v7's (17 vs 2 in both). Worse, the
search's correctly-signed q prefers the loser (+0.032 vs
−0.089) and the gradient prefers the loser slightly (+0.082
vs +0.072 for White) — even a floor high enough to reach this
save would fight the value signal, not just the prior. The
loser is v1's rank-1 move (24008); the save is rank 3 (4821).
Recorded as a regression xfail (fl2-g5-ply19).

**Game 8, ply 18 (Black to move; v15 played (6,6), loses 3/3;
(10,5) draws 3/3, (5,6)/(11,6) save).** Quiet position (80
candidates). The save (10,5) holds 6 visits vs the loser's 11
— roughly prior-proportional (prior ratio 0.046/0.072 = 0.64,
visit ratio 0.55), so this is NOT starvation: the loser is a
FORCING move (creates a four) with the higher prior, and the
search visits it more for both reasons. Correctly-signed q
prefers the loser slightly (−0.018 vs −0.030); the gradient
prefers the save slightly (−0.012 vs −0.025 for Black). A
close call with no structural cause — every signal is within
noise of the other. Recorded as a regression xfail
(fl2-g8-ply18).

**No patch from iteration 2; loop stops here (stopping rule).**
The three worked losses share one fact: **v15's root stats are
IDENTICAL to v7's on all three critical positions** — the 0.005
floor changes nothing where the losses are decided, because
every save and loser in these positions holds a prior of
0.032–0.072, far above the floor. The floor's only effect is on
the deep tail (prior < 0.005), and no loss in this loop turns
on a deep-tail move. The loss classes are all known: an RNG
tie-break (g3, 1-in-3 unlucky), the starvation class at priors
the floor cannot reach (g5 — and q is against the save there),
and a close call (g8). No new patchable weakness class, so the
fast loop ends after 2 iterations (20 games) per the stopping
rule. The battery runs on the as-registered v15 (floor 0.005,
no fast-loop patches). The honest implication, recorded here:
if v15 beats v7 in the battery, the gain comes from the floor's
tail redistribution across many positions (the sweep's
mechanism), not from fixing any single worked loss — and if it
does not, the floor's reach (prior < 0.005) is simply too
narrow for the starvation class as documented (priors
0.027–0.045).

### 2.4 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.5 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.6 Aggregate counts (secondary)

[To be filled in.]

## 3. v15 evaluation (FULL VALIDATION BATTERY — final v15)

### 3.1 Head-to-head vs v0 (100 games, seeds 100/200)

[To be filled in.]

### 3.2 Tactical suites

[To be filled in.]

### 3.3 Head-to-head vs v7 (100 games, seeds 100/200)

[To be filled in.]

### 3.4 Ladder (20 games/opponent, mcts_sims=100, seed 100)

[To be filled in.]

### 3.5 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge)

[To be filled in.]

## 4. Bar check (v15 vs v7)

[To be filled in.]

## 5. What this means

[To be filled in.]

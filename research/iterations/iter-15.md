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

[To be filled in.]

### 2.2 Fast-loop iteration 1 — 10 games vs v0 (sweep-chosen v15)

[To be filled in.]

### 2.3 Fast-loop iteration 2 — 10 games vs v7

[To be filled in.]

### 2.4 Fast-loop iteration 3 — NOT RUN unless iterations 1–2 find a patchable class

### 2.5 Fast-loop iteration 4 — NOT RUN unless iterations 1–3 find a patchable class

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

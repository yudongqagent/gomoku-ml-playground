# Iteration 17 — v17: the combination (batch 6, H3)

**Status: IN PROGRESS — fast loop complete (2 iterations, no
patchable class, loop stopped per the stopping rule); the battery
runs on the as-registered v17. Clause 4 pre-check green (ml pytest
299 passed / 7 skipped / 20 xfailed / 0 failures; npm test 55/55;
tsc clean). Pre-registration (§1) was written BEFORE any eval.**

**METHOD (carried over from batch 3/4/5).** The fast 10-game loop
stays the finding method: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified, never
guessed), patch the algorithm to reproduce the perfect move in
those positions (regression test), repeat — max 4 iterations, or
stop when an iteration finds no new patchable weakness class.
The 4-clause bar stays the validation gate. Fast-loop games use
development seeds (1001/2001, 1002/2002, ...) distinct from the
bar battery's pre-registered seeds (100/200). The full battery
runs on the FINAL version only. The 100-game battery is the
verdict — small-sample smokes mislead.

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-6 plan: ladder
Elo 190.8, on master at `bf866f7`). Clause 1 is head-to-head vs
v7 (normal rules, >= 100 games, alternating colours, win rate >
0.50 with Wilson 95% CI lower bound > 0.50); clause 2 is ladder
Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100); clause
3 is tactical pass rate >= v7's (20/20 old suite + 29/29 hard
suite); clause 4 is all test suites green (ml pytest, npm test,
tsc). **Secondary head-to-heads, also pre-registered:** v17 vs
v0 (100 games) and a 100-game swap match vs v7 under the tempo
judge (the harness default). Non-transitivity is recorded
prominently.

**The improvement: v17 = v7 + tie-prior-forcing root rule +
v1-kernel gradient leaf + the 2-ply threat check.** Each
component passed 3/4 in a different way in batch 5, and each
failed a different clause:

| Component | Batch-5 version | Clauses passed | Clause failed |
|---|---|---|---|
| tie-prior-forcing root rule | v12 (iter-12) | 2, 3, 4 | 1 (0.4950 vs v7) |
| threat check at the root | v13 (iter-13) | 2, 3, 4 | 1 (0.5250 vs v7) |
| v1-kernel gradient leaf | v14 (iter-14) | 1, 3, 4 | 2 (ladder Elo -34.9) |

The hypothesis is that the components' strengths cover each
other's failures: v14's gradient supplies the head-to-head edge
v12/v13 lacked (clause 1: 0.6000, CI lower 0.5020), while v12's
root rule (ladder interval [286.6, None] — heuristic 20-0, mcts
10-10) and v13's threat check (ladder 240.8) supply the ladder
strength v14 lacked (clause 2: Elo -34.9, heuristic 9-11).
**CAUTION, stated in advance:** combination is not guaranteed.
v12 was 8 games WORSE than v9 vs v7 on identical seeds
(0.4950 vs 0.5500) while 7 games better vs v0 — a component
that helps against weak opposition can hurt against an equal
one. That is why v17 gets its own full battery, not an
assumption, and why the fast loop runs on the combined engine
before the battery.

### 1.0 The exact combination (pre-registered)

v17 is `HeuristicMCTS` with the following, and nothing else:

- **Base (v7, unchanged):** PUCT MCTS, **100 sims in both
  colours**, c_puct 1.5, v1 priors over all candidates
  (temperature 2.0 — the sharp prior is load-bearing at 100
  sims; flattening is catastrophic, iter-13 §2.4), win/block
  scans before the search.
- **Leaf (v14, ported exactly):** `leaf_mode="gradient_v1"` —
  the v1-kernel breadth gradient in quiet positions, thresholds
  strong >= 4000 / developing [80, 4000), tanh scale 7.5, cap
  ±0.2 (strictly below the smallest terminal value 0.25).
  Terminal values identical to v7's by construction. The v1
  kernel (`ml/gomoku_ml/kernel_v1.py`) is used directly, not
  re-ported (0 mismatches vs frozen + live references).
- **Root decision (v12, ported exactly):**
  `root_decision="tie-prior-forcing"` — most-visited root
  child; ties broken by forcing-ness first (moves creating a
  four, via `scoring._is_forcing`), then the v1 prior, then the
  seeded RNG / first move. **q is NOT used anywhere in this
  rule** — q-based root decisions are CLOSED (pure q
  catastrophic: 0.20). q SIGN, restated so batch 4's error does
  not recur: `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`); our-perspective q is
  `-child.q`.
- **Threat check (v13, ported exactly): INCLUDED.** v3's 2-ply
  threat check at the root, as in `WhiteMCTS.choose_move`:
  when the opponent has creators (moves leaving them >= 2
  winning moves), the root pool is restricted to safe moves
  (forcing, or no creator remains live); if no root child is
  safe, the root rule's choice on the full pool is played
  (best effort, as in v3/v8/v13). With no creators the choice
  is exactly the root rule's. See §1.1 for why it is included.

**Composition order (pre-registered):** win scan → block scan
→ 100-sim search → threat filter restricts the pool → the
tie-prior-forcing rule decides within the pool. The threat
check filters BEFORE the root rule, never after: it removes
unsafe moves from consideration, and the root rule then picks
among the safe ones. (If the filter applied after the root
rule, it could only veto the rule's pick and fall back to an
arbitrary safe move — the pre-filter composition is the one
v13's `WhiteMCTS` implements, and it is the one ported here.)

**Implementation (pre-registered).** A new module
`ml/gomoku_ml/mcts_v17.py` with `V17MCTS`, subclassing
`HeuristicMCTS` with `leaf_mode="gradient_v1"` and
`root_decision="tie-prior-forcing"`. The subclass exists
because `WhiteMCTS` cannot carry this combination: it
hard-codes `leaf_mode="gradient"` and replicates the root
decision with plain most-visited + RNG tie-break, bypassing
`_choose_root_child` — so the v12 root rule and the v14 leaf
cannot be expressed through it. `V17MCTS.choose_move`
replicates `WhiteMCTS.choose_move`'s structure verbatim (win
scan, block scan, candidate checks, search loop, creator
scan, safe-pool filter) with exactly two changes: the sims
count is 100 in both colours (no colour-asymmetric sims —
v10's 200-White-sims is refuted, iter-13 §2.1), and the
final decision line applies `_choose_root_child` to the
(filtered) root instead of plain most-visited. The filter
itself (creators via `scoring._opponent_creators`, safety
via `scoring._move_is_safe`, the `safe and len(safe) <
len(root.children)` condition, the `threat_filter_used`
counter) is ported verbatim from `WhiteMCTS`. Registered as
version 17 in `ENGINE_VERSIONS` with a `"v17"` dispatch key
in `ScoringPlayer`, following the v13 dispatch pattern.

**NOT in v17 (pre-registered exclusions):** White prior
temperature (refuted — catastrophic, iter-13 §2.4); the (6,8)
opening book (refuted out-of-sample, iter-13 §2.5);
colour-asymmetric sims (refuted — 200 White sims collapse
White to 0.00 vs v0, iter-13 §2.1); any q-based root
decision (CLOSED); any prior flattening; any leaf change
beyond the ported v14 gradient; any change to
`baseline.py` (FROZEN). No kernel re-sync (NumPy/WebGPU/
Metal) unless v17 passes the bar.

### 1.1 The threat check: included, and why

The threat check is the ONLY v13 component that survives
batch 5: the temperature was refuted (White 0.775 → 0.05 →
0.00) and the book was refuted (0.55 vs the 0.775 control),
so v13 = v7 + threat check is the component being combined.
It is included for four reasons:

1. **It is the component that passed.** v13 passed clauses 2,
   3 and 4 and failed clause 1 at 0.5250 — the same 3/4 shape
   as v12, failing only the head-to-head. The combination's
   hypothesis is that v14's gradient supplies the missing
   head-to-head edge; the threat check is not the part that
   failed.
2. **It is attack-neutral safety infrastructure, not a
   strength claim.** It never creates a move, it only removes
   unsafe ones when the opponent holds creators. Across
   v13's battery it produced 0 missed preventions in 520
   threatened turns, and it fixed v8's hard-tactical gap
   (24/29 → 29/29, iter-8 §7 finding 3). It costs nothing
   tactically (v10 and v13 both 20/20 + 29/29).
3. **Its ladder contribution is positive — and the ladder is
   the combination's main risk.** v13's ladder Elo is 240.8
   vs v7's 190.8, gained entirely from the heuristic matchup
   (0.80 vs 0.75 — two of v7's losses become draws under the
   threat check). v14's clause-2 failure was driven entirely
   by the same matchup (9-11 vs the heuristic anchor). If
   the threat check's defensive solidity transfers to the
   combination, it directly shores up v14's failure mode.
4. **It composes cleanly.** It restricts the pool before the
   root rule and never touches the leaf, so it cannot
   interact with the gradient's ordering or the root rule's
   tie-breaks except by removing moves both would agree are
   unsafe.

**Risk, acknowledged in advance:** v13 vs v0 was 0.6400,
slightly BELOW v7's 0.6650 on both colour splits — the
check's safe moves are sometimes passive blocks, and
against greedy 1-ply play passivity costs. If v17's vs-v0
or ladder numbers come in below v12's, the threat check is
the first suspect — but it is not removed mid-battery: the
pre-registered engine is the engine that is evaluated, and
a threat-check-free variant would be a new version, not a
patch.

### 1.2 Component verification (identity checks, NOT evals)

Each ported component is verified to behave identically to
its batch-5 version BEFORE the combination is evaluated.
These checks compare against batch-5 records; they are not
new evals of v17 and they do not select anything:

- **v1 kernel:** `ml/tests/test_kernel_v1.py` — 0 mismatches
  vs the frozen fixtures AND the live `scoring.score_attack`
  reference, both players, all cells (the batch-5 bar).
- **v12 root rule:** the iter-9 §2.6 diagnostic position
  (swap game 41, ply 21, White to move) — v12 must play
  (3,11), the higher-prior forcing move, as in iter-12 §1.5.
  Driver: `ml/scripts/diag_v12_forcing.py` (ported from
  batch-5-h1). Plus the ported `test_mcts_heuristic.py`
  root-decision tests.
- **v13 threat check / v14 leaf:** the ported regression
  suites (`test_regression_v12/v13/v14.py`, `test_mcts_v13.py`,
  `test_mcts_white.py`) pass with the recorded xfail counts.
- **Tactical suites (full-dispatch driver):** v12, v13 and
  v14 each 20/20 old + 29/29 hard, as recorded in iter-12
  §3.5, iter-13 §3.1, iter-14 §3.5.
- **Deterministic prefix replays:** 10-game matches at the
  batch-5 battery seeds (100/200) for v12-vs-v7, v13-vs-v7
  and v14-vs-v7, compared game-by-game (move lists and
  winners) against the first 10 games of the batch-5 match
  JSONs on `research/batch-5`. The engines are
  deterministic, so an identical port reproduces the records
  exactly; any mismatch is a port error, not noise. (10
  games each — enough to catch a port error, too few to be
  an eval; the 100-game battery is the verdict.)

### 1.3 Eval seeds (pre-registered, battery)

h2h vs v7: v17=100 / v7=200; h2h vs v0: v17=100 / v0=200;
ladder seed 100, 20 games/opponent, mcts_sims=100; swap 100
games vs v7, balanced placement, placement_seed 11, tempo
judge (the harness default; recorded in the JSON). Harness
commands follow iter-7 §3 exactly (`selfplay_batch.py`,
`eval_version.py` for the ladder, `eval_tactical_full.py`
for tactical). Fast-loop dev seeds: iter 1 vs v0 1001/2001,
iter 2 vs v7 1002/2002, iter 3 1003/2003, iter 4 1004/2004
— all distinct from the battery's 100/200.

### 1.4 Component verification — RESULTS (identity checks, not evals)

All checks from §1.2 pass. The ported components behave
identically to their batch-5 versions:

- **v1 kernel:** `test_kernel_v1.py` green — 0 mismatches vs
  the frozen fixtures AND the live `scoring.score_attack`
  reference, both players, all cells, all 60 fixtures, plus
  the 6 targeted broken-four/gap-three boards.
- **v12 root rule:** `diag_v12_forcing.py` reproduces iter-12
  §1.5 EXACTLY on the §2.6 position (swap game 41, ply 21):
  the three-way 5-visit tie, our-q -0.204355 (forcing moves)
  vs -0.421490 ((7,8)), priors 0.039113/0.038961/0.124404,
  v1 scores 20167/20010/204024; tie-prior-forcing plays
  (3,11); v9 regressions 1/3 for all three rules.
- **Regression suites:** the ported `test_regression_v9/v12/
  v13/v14.py`, `test_mcts_v13.py`, `test_mcts_white.py` all
  pass with the recorded xfail counts (full ml suite: 299
  passed, 7 skipped, 15 xfailed, 0 failures — batch-5's 290
  + the 9 new v17 tests).
- **Tactical suites (full-dispatch driver, seed 0):** v12
  20/20 + 29/29, v13 20/20 + 29/29, v14 20/20 + 29/29 — all
  matching their batch-5 records. **v17 (the combination):
  20/20 + 29/29** — clause 3 holds on the as-registered
  engine; re-run on the final engine in §3.
- **Deterministic prefix replays** (10 games each, battery
  seeds 100/200, move lists + winners vs the batch-5 match
  JSONs on `research/batch-5`): v12-vs-v7 **0 mismatches**
  (5W-3L-2D in the prefix, 754s), v13-vs-v7 **0 mismatches**
  (4W-6L-0D, 257s), v14-vs-v7 **0 mismatches** (4W-4L-2D,
  624s). The ports are exact; the combination is built on
  verified components. (A runtime restart wiped /tmp
  mid-verification, including the v12 replay file — the
  v12 0-mismatch result was recorded before the restart and
  the reference JSONs were re-extracted from git into
  `~/workspace/b6h3-logs/`; the v13/v14 replays and all
  later work write there or into the repo, not /tmp.)

## 2. Loss analysis for iteration 17

[Fast-loop worked games go here — see the method note at the top.]

### 2.1 Fast-loop iteration 1 — 10 games vs v0

**Batch:** `research/iterations/fastloop1-v17-vs-v0.json`
(seeds v17=1001 / v0=2001 — development seeds, distinct from
the bar battery's 100/200). **0W-10L-0D, win rate 0.0000, CI
[0.0000, 0.2775]**, 85.1 s. **This number is a transposition
artefact, not a measurement — see the isolation experiment
below.** The 10 games are just 2 distinct lines repeated 5
times each (as Black: 28 plies, White wins; as White: 25
plies, Black wins). Classifier: 10/10 losses via open four,
0 missed blocks, 0 clean missed preventions in 55 threatened
turns. Defence is perfect; the losses are lost races.

**The losing move in each distinct game** (walk-back,
`ml/scripts/walkback_loss.py`, corrected oracle method,
seeds 0,1,2):

| Game | Last preventable turn | v17 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 0 (28 plies, Black) | none | — (no alternative saves at any turn from ply 18 on; at ply 16 and earlier the actual does not lose at all seeds) | — |
| 1 (25 plies, White) | ply 13 | (6,9) | (6,5) — draws 3/3 |

**Game 1 ply 13 is not a v17 defect.** At the critical
position Black holds exactly 2 creators, (6,9) and (6,5) —
and v7, v12, v13, v14, v17 and the unfiltered leaf+root
engine ALL play (6,9). Both moves kill all Black creators,
both are safe, and v17's own search prefers (6,9) (31 visits
vs 14, our-q +0.047 vs -0.035, prior 0.1294 vs 0.1282).
This is iter-14 games 1/3 exactly: two creator-blocks
indistinguishable to every signal in the engine. No patch.

**Game 0 has no losing move** (distributed drift, the iter-6
result). A 12-turn walk-back flags plies 4 and 6 ((8,5) and
(9,6), both "lose 3/3 with 5/5 alternatives saving") — but
v14 played the IDENTICAL moves at plies 4 and 6 in its game
2 on these seeds (the two engines' games share plies 0-8)
and WON that game in 39 plies. The oracle verdict is
contradicted by actual engine play, so per iter-13's
g9-ply-5 precedent it is not a patch target — patching on
it would be overfitting to the oracle proxy.

**The 0-10, isolated (the iteration's real finding).** A
0-10 against v0 — worse than any version in the program —
demanded isolation before any patch was considered. Three
controls, all 10 games vs v0 at the same seeds (1001/2001):

| Engine | Result | Note |
|---|---|---|
| v17 (leaf + root + threat check) | 0W-10L-0D | the fast-loop batch |
| v97 (leaf + root, NO threat check; temporary diagnostic version, removed before the battery) | 0W-10L-0D | identical game shapes (28/25 plies); move-identical to v17 except game 0's final, already-lost ply |
| v14 (leaf only, plain visits root) | 6W-4L-0D | reproduces iter-14 §2.1 exactly |

The threat check is EXONERATED (v97 without it goes 0-10
too). The collapse sits in the leaf+root combination — but
**it does not reproduce at fresh seeds**: v97 vs v0 at
1002/2002 goes **6W-4L-0D** (v14 at the same seeds: 5W-5L).
So the 0-10 is seed luck amplified by deterministic
transposition, not a systematic defect:

- The tie-prior-forcing rule is nearly deterministic: ties
  are broken by forcing-ness, then prior — both
  deterministic signals — and the RNG is consulted only
  when forcing-ness AND prior both tie (rare). v17's play
  therefore barely depends on RNG state, so the 10-game
  batch collapses into 2 repeated lines instead of sampling
  10 different ones. v14's visits rule consults the RNG on
  every visit tie, so its 10 games are 10 different lines.
- At seeds 1001/2001 both of v17's lines happen to lose to
  v0; at 1002/2002 its lines win 6/10. n=10 that is really
  n=2 cannot distinguish "the rule is bad with this leaf"
  from "these two lines are bad" — and the fresh-seed
  result says it is the latter.
- Corroboration: with IDENTICAL RNG state (fresh seed 0),
  all three root rules pick the SAME move at both analysed
  diverging positions (v14 game 4 ply 2: all pick (8,7);
  v14 game 2 ply 10: all pick (9,7), the 14-visit tie with
  (5,6) broken by the forcing move's higher prior). The
  game-list divergences between v14 and v97 are RNG-state
  artefacts, not rule effects. (And the (9,7)-vs-(5,6)
  choice is exonerated by the walk-back: (9,7) does not
  lose at all seeds at ply 10.)

**Methodological lesson, recorded for the program:** for a
near-deterministic engine, a 10-game fast loop against a
deterministic opponent is nearly useless as a STRENGTH
signal — it measures 2 lines, not 10 games. The loop still
works as a FINDING method (it produced the ply-13
position), but its W/L record must not be read as an eval,
and a shocking fast-loop number (0-10) must be checked at
fresh seeds before any conclusion is drawn. The 100-game
battery does not have this problem against v7: v7's RNG
tie-breaks vary its play per game, so v17 faces different
boards even though its own replies are near-deterministic.

**No patch from iteration 1; the loop continues to
iteration 2 vs v7.** Iteration 1 found no new patchable
weakness class (game 0: drift; game 1: an all-versions
mistake; plies 4/6: oracle contradicted by engine play),
but the vs-v7 losses are the ones that matter for the bar
and iteration 1 played only v0 — so iteration 2 runs
against v7 before the stopping rule is applied. The v17
regression file records the game-1 ply-13 position as an
xfail (`regression_positions_v17.json`, `fl1-g1-ply13`).

### 2.2 Fast-loop iteration 2 — 10 games vs v7

**Batch:** `research/iterations/fastloop2-v17-vs-v7.json`
(seeds v17=1002 / v7=2002 — development seeds, distinct from
the bar battery's 100/200). **3W-5L-2D, win rate 0.4000, CI
[0.1682, 0.6873]**, 650.7 s. Per colour: as Black 1W-3L-1D,
as White 2W-2L-1D. (For scale: v12's iter 2 on the same
seeds was 7W-3L, v9's 4W-4L-2D — n=10 with wide CIs, not a
comparison; the loop's job is finding weaknesses, and the
100-game battery decides.) Classifier: 5/5 losses via open
four, 0 missed blocks, 0 clean missed preventions in 25
threatened turns (losses). Defence is still perfect.

**The losing move in each game** (walk-back, corrected
oracle method, seeds 0,1,2; JSONs:
`~/workspace/b6h3-logs/walkback-fl2-g{1,2,3,4,8}.json`):

| Game | Last preventable turn | v17 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 1 (67 plies, White) | ply 51 | (6,5) | (9,4)/(7,5)/(8,3) — draw 3/3 |
| 2 (104 plies, Black) | none | — (no alternative saves; oracle draws at every turn from ply 88 on) | — |
| 3 (23 plies, White) | ply 9 | (8,6) | (9,6) — wins 3/3; (4,6) — draws 3/3 |
| 4 (48 plies, Black) | ply 42 | (6,3) | (13,8) — draws 3/3 |
| 8 (36 plies, Black) | ply 30 | (10,7) | (5,10)/(4,11) — draw 3/3 |

**Critical-position analysis** (`analyse_v17_critical.py`,
100 sims, gradient_v1 leaf; q is OUR-perspective q =
-child.q; gradients are the mover's perspective after the
move):

| Position | Actual (v1 rank / prior) | Actual visits / our-q | Save visits / our-q | Creators? | Forcing? |
|---|---|---|---|---|---|
| g1 ply 51 | (6,5) rank 1, p=0.0731 | 4 / -0.207 | (9,4): 4 / -0.197 | 0 | neither |
| g3 ply 9 | (8,6) rank 5, p=0.0576 | 6 / -0.013 | (9,6): 5 / -0.021 | 0 | neither |
| g4 ply 42 | (6,3) rank 1, p=0.0363 | 2 / -0.177 | (13,8): 2 / -0.195 | 0 | neither |
| g8 ply 30 | (10,7) rank 4, p=0.0286 | 5 / -0.052 | (5,10): 4 / -0.084 | 0 | neither |

**The combination's new components are not implicated in
any loss.** All four critical positions are QUIET (0
opponent creators), so the threat check is disengaged in
every one — it cannot have caused any of these moves. And
no forcing move is involved in any actual/save pair (all
forcing=False), so the forcing-first tie-break never fires
either. What the four losses share is the KNOWN classes:

1. **Visit-tie knife-edges (g1, g4).** g1 ties 4-4 visits
   and the prior picks the loser by 0.0004 (0.0731 vs
   0.0727); g4 ties 2-2 and the prior picks the loser by
   0.0002 (0.0363 vs 0.0361). In g1 the q and both gradients
   marginally prefer the save (-0.197 vs -0.207; v1 -0.172
   vs -0.176) — noise-level gaps no tie-break signal in the
   pre-registered set separates. This is iter-12 game 5's
   1-point knife-edge again. The root-decision lever is
   exhausted; no rule change fixes a 0.0002 prior gap.
2. **Beyond-horizon value error (g3, g8).** The search's
   correctly-signed q prefers the LOSING move (g3: -0.013
   vs -0.021; g8: -0.052 vs -0.084) and the saving move is
   starved of visits (5 vs 6, 4 vs 5) — the iter-13
   diagnosis. In g3 the v1 gradient DOES prefer the save
   (+0.087 vs +0.020, where the v0 gradient prefers the
   loser +0.037 vs +0.025) but the search does not follow
   it; in g8 the v1 gradient points the WRONG way (-0.103
   for the save vs -0.052 for the loser, where the v0
   gradient points right, 0.000 vs -0.012) — and v7/v12/v13
   (v0 leaf) play the save (5,10) while v14/v17 (v1 leaf)
   miss it. One position each way: the v1 gradient is right
   in g3 and wrong in g8, so there is no leaf patch here —
   changing the gradient on one position would be
   overfitting, and the leaf is v14's ported component.
   (g3's save (9,6) WINS 3/3 by oracle — the biggest single
   miss of the loop — yet no engine signal prefers it:
   visits 5 vs 6, q -0.021 vs -0.013, prior tied at 0.0576,
   v1 score 4092 vs 4091, a 1-point gap.)
3. **Game 2 (104 plies) has no losing move** — the oracle
   draws at every turn from ply 88 on. Distributed drift in
   a long game, the iter-6 result.

**No patch from iteration 2; loop stops here (stopping
rule).** Iteration 2 found no new patchable weakness class:
the losses are the known exploration/value-error classes
(and two knife-edge ties), the threat check and the
forcing tie-break are disengaged in every worked loss, and
the one leaf-level observation (g8) cuts both ways across
the positions. All four critical positions are recorded as
xfail regression tests (`regression_positions_v17.json`:
fl2-g1-ply51, fl2-g3-ply9, fl2-g4-ply42, fl2-g8-ply30) —
documented open weaknesses for the next batch, not silently
dropped. The battery runs on the as-registered v17 (no
fast-loop patches). The temporary diagnostic version 97
(used only for the §2.1 isolation) is removed from
`ENGINE_VERSIONS` before the battery.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

[PENDING — filled after the battery, from the classifier on the
three 100-game matches.]

## 3. v17 evaluation

**Engine note:** the fast loop made NO patch, so the battery runs
on the pre-registered engine exactly as implemented in §1 (v7's
MCTS + the v1-kernel gradient leaf, scale 7.5, cap ±0.2, +
tie-prior-forcing root rule + the 2-ply threat check at the
root). Tactical was measured on this engine in §1.4 (20/20 +
29/29) and is re-run in §3.5 for the record.

### 3.1 Smoke — NOT RUN (no smoke; the fast loop is the smoke)

Per the batch-5 method the 10-game fast loop (§2.1-2.2) is the
only small-sample evidence, and small-sample smokes mislead
(v8, v11, and v17's own 0-10 transposition artefact in §2.1)
— the 100-game battery is the verdict.

### 3.2 Tactical suites — pre-battery (re-run on the final engine in §3.5)

Old 20-position suite, full-dispatch driver
(`ml/scripts/eval_tactical_full.py`): **v17 20/20**. Hard suite
(29 positions): **v17 29/29** — a perfect score, matching v7's
20/20 + 29/29. The combination costs nothing tactically.

### 3.3 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

[PENDING — running.]

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

[PENDING]

### 3.5 Tactical suites — FINAL BATTERY (re-run on the final engine)

[PENDING]

### 3.6 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

[PENDING]

### 3.7 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

[PENDING]

## 4. Bar check (v17 vs v7)

[PENDING]

## 5. What this means

[PENDING]

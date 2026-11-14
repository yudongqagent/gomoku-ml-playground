# Iteration 10 — v10: White weakness / colour-asymmetric search (batch 4, H2)

**Status: PRE-REGISTERED — design, eval seeds and White-measurement
plan recorded BEFORE any eval (see §1).** Comparison target is v7
(the batch-4 reference). This is the H2 workstream of batch 4
(H1 is the q-vs-visits root decision, a separate workstream on
`research/batch-4-h1` — v10 does NOT change the root decision
rule); work happens on branch `research/batch-4-h2` in the
`gomoku-b4-h2` worktree.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-4 reference:
ladder Elo 190.8, vs v1 0.7050). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating
colours, win rate > 0.50 with Wilson 95% CI lower bound >
0.50), clause 2 is ladder Elo strictly higher than v7's 190.8
(identical 20 games/opponent, mcts_sims=100), clause 3 is
tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green. **Secondary head-to-heads,
also pre-registered:** v10 vs v0 (100 games) and a 100-game
swap match vs v7 under the tempo judge (the harness default).

**The improvement: v10 = v7 + the 2-ply threat check at the
root + colour-asymmetric search depth.** In full:

- Base is v7 EXACTLY: `HeuristicMCTS` with
  `leaf_mode="gradient"` (capped ±0.2 attack-gradient leaf),
  c_puct 1.5, v1 priors over all candidates, win/block scans
  before search, most-visited root child with RNG tie-break.
- Addition 1 (colour-SYMMETRIC, directed by the brief): v3's
  2-ply threat check at the MCTS root, ported exactly as in
  v8 (`ValueMCTS.choose_move`, iter-8 §4.1): if the opponent
  has creators, the root choice is restricted to safe moves
  (forcing, or no creator remains live); if no root child is
  safe, the most-visited move is played (best effort). With
  no creators the choice is unchanged. v7 does NOT have the
  threat check; iter-8 §7 finding 3 calls it a durable
  portable gain (fixed v8's hard tactical 24/29 → 29/29;
  defence airtight).
- Addition 2 (colour-CONDITIONAL — the H2 hypothesis): the
  root search runs **200 sims when the engine plays White
  and 100 sims when it plays Black**. NOTHING else varies by
  colour: same priors, same leaf, same c_puct, same threat
  check, same decision rule. The colour is known at
  `choose_move` time (the `player` argument is the engine's
  own colour for the game).

**Why more search for White, and why NOT a colour-conditional
value net.** The brief offers three H2 designs (colour-
conditional value training, more search for White, colour-
aware leaf/priors); this workstream chooses more search for
White, for evidence-based reasons:

1. v8's lesson (iter-8 §7 finding 1) is that a value net
   trained on weak self-play is WORSE than no value function
   unless shrunk hard — the 4,350-game net generalises
   (holdout 0.72) yet at full weight loses 0-10 to v0, and
   at α=0.25 the net leaf still lands at ladder Elo 0.0
   against v7's gradient leaf at 190.8. The net leaf is a
   step BACK from the gradient leaf; grafting a
   (colour-conditional) net onto v7's leaf would contaminate
   the best leaf in the program with the weakest. v10 uses
   NO net value, so no α sweep is needed (the sweep rule in
   the brief applies only to designs that use a net value).
2. The White gap is better explained as an EXPLORATION /
   tempo problem than a value-estimation problem. Iter-7
   §2.1's fast loop found exactly one White weakness class
   the cap patch did NOT fix: game 3, ply 23 — the saving
   move (9,12) is a forcing move ranked 8th by v1 (prior
   0.04), starved of visits in 100 sims (recorded as the
   `fl1-g3-ply23` regression xfail in test_regression_v7.py).
   Doubling White sims directly targets visit starvation of
   low-prior forcing moves, with no new tuning knob and no
   distribution-mismatch risk.
3. The threat check (addition 1) complements the sims: when
   Black holds creators, the root choice is restricted to
   safe moves — which INCLUDES forcing counterplay — so
   White is biased toward forcing moves exactly in the
   positions where it is behind in the race.

**What varies by colour (exact, the pre-registered
contract):** the sims count ONLY — 200 for White, 100 for
Black. Verified by a `last_sims_used` diagnostic on the MCTS
(set in `choose_move`) and asserted in tests
(`test_mcts_white.py`): a White `choose_move` on a
search-reaching position runs exactly 200 simulations, a
Black one exactly 100.

**Built-in mechanism check (pre-registered, NOT an eval):**
v10, as White with 200 sims, should play the saving move
(9,12) on the `fl1-g3-ply23` regression position (v7's open
xfail: forcing move starved by a low prior in 100 sims). If
it does, the visit-starvation mechanism is confirmed on a
real position; if it does not, the sims increase does not
fix the known exploration weakness and the design's
mechanism is in doubt before any match is played. This check
runs as a unit test, not a match.

**Pre-eval observation (recorded BEFORE any eval, during
implementation testing — the design stands as pre-registered;
the evals decide).** The built-in mechanism check FAILS: v10
as White with 200 sims still plays the losing block (4,9) on
`fl1-g3-ply23`, not the saving forcing move (9,12). Root
statistics (same position, gradient leaf, seed 0):

| sims | (4,9) visits / q | (9,12) visits / q |
|---|---|---|
| 100 (v7) | 15 / 0.052 | 7 / 0.009 |
| 200 (v10 White) | 26 / 0.045 | 22 / 0.016 |
| 400 (probe) | 113 / 0.012 | 29 / 0.021 |

200 sims close the visit gap (7→22 vs 15→26) but do not flip
the move, and at 400 sims the search converges ON the block
(113 vs 29). The q-values do not distinguish the moves
(0.01–0.05 throughout), so this position's real problem is a
VALUE problem — the search does not see that (9,12) wins —
not pure visit starvation, and more sims alone does not fix
it. (The threat check engages on this position but both
moves are safe, so it cannot decide between them.) The
position is recorded as an xfail test
(`test_v10_white_plays_forcing_save_on_fl1_g3_ply23`, the v7
pattern) so the behaviour stays visible. Following iter-7's
precedent (§1 pre-eval observation), the pre-registered
design is NOT amended on the basis of one diagnostic
position — amending after seeing v10 behave badly on a
known-hard position would be moving the goalposts, and the
evals are the pre-registered decision procedure. The fast
loop (§2) is where a patchable White weakness class would
show up; if the matches show no White gain, the sims
hypothesis is refuted and the next batch should look at the
leaf's value signal for White, not at more search.

**How the White improvement is measured (per-colour splits
are MANDATORY in every match reported here).** White is the
target; the overall match score alone cannot show whether
the colour-conditional change did its job:

- PRIMARY: v10-as-White win rate vs v7-as-Black in the
  100-game h2h (n=50). Reference points: v7-as-White vs v1
  was 0.52 (n=50); v8-as-White vs v1 was 0.29 (n=50).
  A v10-as-White rate > 0.50 against v7-as-Black — the
  reference engine holding Black's first-move edge — is a
  clear White gain. (Caveat: no v7-vs-v7 self-play baseline
  exists, so the White split is judged against the 0.50
  line and the v7/v8 reference splits, not against a
  same-engine baseline.)
- SECONDARY: per-colour splits in every other match — vs v0
  (v7-as-White 0.62, 30W-18L-2D), swap vs v7 (v7's swap
  splits vs v1: Black 0.52, White 0.62), ladder (v7 White
  splits: vs heuristic 7W-3L = 0.70, vs mcts 3W-7L = 0.30).
- v10-as-Black vs v7-as-White is also reported: it should
  track v7's Black game (modulo the threat check), so a
  Black-side regression would show the threat check costing
  attack, not the White change helping.

**Eval seeds (pre-registered, same as iter-7/iter-8):**
h2h vs v7: v10=100 / v7=200; h2h vs v0: v10=100 / v0=200;
ladder seed 100, 20 games/opponent, mcts_sims=100; swap 100
games vs v7, balanced placement, placement_seed 11, tempo
judge (the harness default; recorded in the JSON).

**Fast loop (the finding method, per the batch-4 plan).**
Opponent is v7 (the reference — losses against it are the
ones that matter for the bar); development seeds 500/600,
501/601, ... distinct from the bar battery's 100/200. If an
iteration produces no losses against v7, switch the probe to
v0 (fast; v0's crude play exposed v7's as-Black inversion
and produces White losses). Max 4 iterations (40 games),
stop when an iteration finds no new patchable weakness
class. Each iteration: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified,
never guessed), patch the structural cause, record the
critical position + correct move as a regression test
(`ml/gomoku_ml/regression_positions_v10.json` +
`ml/tests/test_regression_v10.py`).

**Cost estimate (pre-registered, from v7's measured
timings).** v7's 100-game matches took ~3600 s (vs v0/v1)
and ~6444 s (swap); the ladder took 1469 s. v10's White
moves run 2x sims, so matches cost ~1.5x: ~5400 s per
100-game normal match, ~9700 s for the swap, ~2200 s for
the ladder. Total battery ~6 h, run sequentially in the
background, one match at a time. Matches flush after every
game (`selfplay_batch.py`), so a VM replacement loses at
most the in-progress game; a re-run with the same seeds is
deterministic.

**Integration.** New module `ml/gomoku_ml/mcts_white.py`
subclassing `HeuristicMCTS` (the v8 pattern — `mcts_value.py`
is the model); `mcts_heuristic.py` is NOT edited (the
parallel H1 workstream owns the root-decision change there,
and the batch-3 merge lesson says to keep workstreams in
separate modules). `ml/gomoku_ml/scoring.py` gains ONLY the
`ENGINE_VERSIONS[10]` registry entry and a minimal dispatch
hook in `ScoringPlayer.__init__` (a version dict with a
`"white_sims"` key instantiates `WhiteMCTS`).
`baseline.py` is frozen and untouched.

## 2. Loss analysis (fast loop)

The fast-loop worked games (§2.1) ARE the loss analysis — see
the method note in §1. Aggregate counts (§2.5) are secondary.

### 2.1 Fast-loop iteration 1 — 10 games vs v7

**Batch:** `research/iterations/fastloop1-v10-vs-v7.json`
(seeds v10=500 / v7=600 — development seeds, distinct from the
bar battery's 100/200). **3W-7L-0D, win rate 0.3000, CI
[0.1078, 0.6032]**, 919.2 s (~92 s/game under heavy VM
contention with the parallel H1/H3 workstreams). The split is
almost purely colour: v10 wins 3/5 as Black (games 2, 4, 6)
and loses ALL 5 as White (games 1, 3, 5, 7, 9) — 200 White
sims did NOT fix the White weakness at 10-game scale.
Classifier (`ml/scripts/classify_batch_losses.py`): 7/7
losses via open four, 0 missed blocks, 0 clean missed
preventions in 26 threatened turns (losses). Defence is still
solved; the losses are lost races, as in every batch before.
All 10 games are distinct lines (no deterministic duplicates).

**The losing move in each game** (walk-back,
`ml/scripts/walkback_loss.py`, corrected oracle method —
the actual move's v1-vs-v1 playout must lose at all 3 seeds
before an alternative counts; seeds 0,1,2; alternatives are
the top-6 by v1 score plus forcing moves):

| Game | Last preventable turn | v10 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 1 (White, 79 plies) | ply 67 | (0,10) | (8,6)/(8,9)/(9,8) — draw 3/3 |
| 3 (White, 77 plies) | ply 67 | (14,10) | (14,12) — draws 3/3 |
| 5 (White, 77 plies) | ply 71 | (4,1) | (5,3) — wins 3/3 |
| 7 (White, 47 plies) | ply 31 | (8,1) | (10,12) — wins 3/3; (5,7)/(9,9)/(7,11) draw |
| 9 (White, 31 plies) | ply 19 | (8,7) | (4,9) — wins 3/3 |
| 0 (Black, 56 plies) | ply 40 | (11,11) | (11,5) — wins/draws; (12,7)/(12,12) draw |
| 8 (Black, 34 plies) | — | no single losing move (ply 22 loses 3/3, 0 saves; ply 20 draws) | distributed drift |

**The recurring structural weakness — ONE class, 4 of 6
analysed losses: the q-vs-visits root-decision gap
(iter-7 §2.6).** Critical-position analysis
(`ml/scripts/analyse_critical.py`, 200 sims for White):

| Position | Actual (v1 rank / prior) | Actual visits / q | Save visits / q |
|---|---|---|---|
| g1 ply 67 | (0,10) rank 1, p=0.114 | 27 / 0.172 | (8,9): 4 / 0.207 |
| g3 ply 67 | (14,10) rank 1, p=0.255 | 60 / 0.161 | (14,12): 11 / 0.271 |
| g9 ply 19 | (8,7) rank 3, p=0.049 | 82 / -0.131 | (4,9): 8 / 0.064 |
| g0 ply 40 | (11,11) rank 1, p=0.062 | 6 / 0.109 | (11,5): 2 / 0.149 |

In all four, the saving move has the HIGHER q but far fewer
visits, and most-visited decides. Game 9 is the extreme case:
the losing move (8,7) has 82 visits at q -0.131 (the search
KNOWS it is bad) while the forcing save (4,9) has 8 visits
at q +0.064 — and (8,7) leaves Black 2 open-four creators
while (4,9) leaves 0. The search knows the right move and
the decision rule discards that knowledge. This is EXACTLY
iter-7 §2.6's finding, now confirmed as the dominant White
weakness class at 200 sims: doubling White sims does not fix
a decision-rule problem — visits follow the prior, and the
prior is what it is.

**The other two losses are beyond-horizon value errors, not
the same class.** In game 5 (ply 71) the save (5,3) has LOWER
q than the actual (0.103 vs 0.163) and in game 7 (ply 31)
slightly lower (0.097 vs 0.108): the search genuinely does
not see the save at 200 sims, and the oracle's verdict is
beyond the search horizon. No leaf tweak is verified to fix
these — the gradient has no consistent story across the six
positions (it prefers the loser in g1/g3/g5, the save in g7,
and is tied in g9).

**What is NOT the cause (checked, not assumed).**
- The threat check: 5 of the 6 critical positions are
  QUIET (0 creators for either side), so the check never
  engages. In game 3 (1 Black creator) both the actual and
  the save are safe, so the filter cannot distinguish them.
  The threat check neither helps nor hurts these losses.
- "Creates creators": only game 9's losing move leaves the
  opponent creators (2) while the save leaves 0 — 1 of 6
  positions is not a class, and a "don't create creators"
  filter would be an unverified new mechanism.
- The gradient leaf: no consistent direction (above).
- The prior: the losing moves have higher priors than the
  saves in all 6 positions — but the prior is v1's shared
  scoring, and downweighting four-makers for White would be
  an unverified guess, not a verified fix.

**NO PATCH from iteration 1.** The dominant, recurring class
is the q-vs-visits gap — which belongs to the parallel H1
workstream (v9, the q-vs-visits root decision on
`research/batch-4-h1`), NOT to this workstream. Implementing
a q-based root decision here would duplicate H1's work and
conflict at merge. There is no OTHER verified patchable
class in these games that is v10's to fix. Patching on an
unverified hypothesis would violate the loop's "never
guessed" rule (iter-7 §2.2 precedent).

**Regression tests:** `ml/gomoku_ml/regression_positions_v10.json`
(6 positions, stone lists + expected moves, generated from
the game records and verified: v10 seed 0 reproduces the
losing move on all 6) +
`ml/tests/test_regression_v10.py`. All 6 are documented
xfails — the executable White-weakness suite a future
root-decision change (H1/v9) should fix. Targeted suites
after the (non-)patch: test_regression_v10 1 passed,
6 xfailed; test_mcts_white + test_mcts_heuristic +
test_scoring green (see §4).

### 2.2 Fast-loop iteration 2 — NOT RUN (loop stopped after iteration 1)

**Loop stops here (stopping rule).** Iteration 1 found no
new patchable weakness class for THIS workstream: the
dominant class (q-vs-visits) is H1's, and the two
beyond-horizon losses have no verified fix. A second
iteration vs v7 would only re-confirm the same class (and
the v0 fallback does not trigger — iteration 1 DID produce
losses against v7). The loop ends after 1 iteration
(10 games) per the direction change's stopping rule. v10
goes to the full battery as pre-registered (§1): v7 +
threat check + 200 White / 100 Black sims, with the honest
expectation from the fast loop (3W-7L, White 0W-5L) that the
sims hypothesis is likely refuted at 100-game scale too —
the battery measures it precisely.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 1)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 1)

### 2.5 Aggregate counts (secondary)

Across the final battery's matches (classifier:
`ml/scripts/classify_batch_losses.py`):

| match | losses | via open four | missed blocks | threatened turns (losses) | clean missed preventions (losses) |
|---|---|---|---|---|---|
| vs v7 | 59 | 59 | 0 | 251 | 0 (0.0%) |
| vs v0 | 61 | (TBD — classifier run after the ladder) | | | |
| vs v7 swap | (TBD) | | | | |

Defence is solved: 59/59 vs-v7 losses via open four, 0 missed
blocks, 0 clean missed preventions in 251 threatened turns.
Every loss is the opponent winning the open-four race, not a
defensive error — the same shape as every version since v6.

## 3. v10 evaluation (FULL VALIDATION BATTERY — final v10)

No fast-loop patch was made (§2.1–2.2), so the battery runs
on the pre-registered engine (§1) unchanged. The tactical
suites were measured on this exact engine before the fast
loop (§3.1) and are NOT re-run — there is no patch to
re-verify, and the engine is byte-identical.

### 3.1 Tactical suites

Old 20-position suite, full-dispatch driver
(`ml/scripts/eval_tactical_full.py`, `player.choose_move`):
**v10 20/20** (= v7's 20/20). Hard suite (29 positions):
**v10 29/29** (= v7's 29/29). Raw JSON:
`research/iterations/batch-v10-tactical-pre.json` (seed 0,
149 s for 49 positions). **Clause 3 PASSES** (old 20/20 =
v7's 20/20; hard 29/29 = v7's 29/29). The threat check and
the White sims increase cost nothing tactically.

### 3.2 Head-to-head vs v7 (100 games, seeds 100/200)

**v10 loses to v7: 37W-59L-4D, win rate 0.3900, 95% CI
[0.3002, 0.4880]**, 6176.1 s. File:
`research/iterations/batch-v10-vs-v7.json` (seeds 100/200).
Per colour: as Black 20W-26L-4D (0.4400), as White
17W-33L-0D (0.3400) — v10 loses in BOTH colours. The White
split (0.34) is far below the pre-registered target (> 0.50
as White against v7-as-Black) and below v7-as-White vs v1
(0.52). **Clause 1 FAILS decisively** (0.3900 < 0.50, and
the CI upper bound 0.4880 is itself below 0.50 — v10 loses
to v7 outright, not within noise). Mean game length 56.53
plies (min 15, max 225). Loss classifier: 59/59 losses via open
four, **0 missed blocks**, 251 threatened turns in losses,
**0 clean missed preventions** — defence solved; v10 loses
the open-four race to v7 in both colours.

(First run of this match was killed by a VM/runtime restart
at 42 games (15W-27L-0D, partial JSON flushed); the re-run
with the same seeds reproduced the 42-game prefix EXACTLY
(all 42 records byte-identical — determinism verified), so
the completed match is the pre-registered one, not a
different sample.)

### 3.3 Head-to-head vs v0 (100 games, seeds 100/200)

**v10 loses to v0: 39W-61L-0D, win rate 0.3900, 95% CI
[0.3002, 0.4880]**, 2042.5 s. File:
`research/iterations/batch-v10-vs-v0.json` (seeds 100/200).
Per colour: as Black 39W-11L-0D (0.7800), as White
**0W-50L-0D (0.0000)** — v10-as-White lost EVERY game to the
frozen baseline heuristic. This is the most extreme colour
split in the program's history (v4-as-White vs v0 was
1W-49L, 0.02). v7-as-White vs v0 was 0.62 (30W-18L-2D): the
White-specific change took White from 0.62 to 0.00 against
v0, while Black IMPROVED (0.78 vs v7's 0.71 as Black).
All 50 White losses run 27–29 plies (mean 28.4) — v0's
crude greedy attack converts against v10-White in ~28
plies from any opening (11 distinct 10-ply openings across
the 50 games, 12 distinct lines — not a single-line
artefact). Mean game length 37.23 plies overall (min 22,
max 115).

(First run killed by a VM/runtime restart at 27 games
(11W-16L-0D, split Black 11W-3L / White 0W-13L); the re-run
with the same seeds reproduced the 27-game prefix exactly.)

**Why White collapses — diagnostic probe (NOT an eval).**
v10-White differs from v7-White in TWO ways (200 sims, the
threat check), so the collapse could be either. A three-way
probe on White turns from the v0 losses separates them:
v7 (100 sims, no check) vs WhiteMCTS with white_sims=100
(check only) vs v10 (200 sims + check), same seed:

| Position | Black creators | v7 (100, no chk) | chk-only (100) | v10 (200, chk) | v10's actual move |
|---|---|---|---|---|---|
| g1 ply 15 | 2 | (10,5) | (10,5) [filter=1] | (10,5) [filter=1] | (10,5) |
| g1 ply 21 | 0 | (5,10) | (5,10) [filter=0] | (5,8) [filter=0] | (5,8) |
| g3 ply 15 | 2 | (4,5) | (4,5) [filter=1] | (4,5) [filter=1] | (4,5) |
| g3 ply 21 | 0 | (9,10) | (9,10) [filter=0] | (9,8) [filter=0] | (9,8) |
| g5 ply 17 | 2 | (7,4) | (7,4) [filter=1] | (7,4) [filter=1] | (7,4) |

When Black has creators, all three play the SAME move — the
threat check engages but does not change the choice (v7
already plays the safe move). When the position is QUIET
(0 creators, check disengaged), v7 and the check-only
version agree, but v10 (200 sims) plays a DIFFERENT move.
**The sims increase, not the threat check, changes White's
play.** The threat check is exonerated as the cause of the
collapse (it is also colour-symmetric, and Black is fine).

A walk-back of game 1 vs v0 refines the mechanism: the
losing move is ply 21, (5,8) — loses 3/3, with saves (7,5)
[wins 3/3] and (7,6) [draws 3/3]. (5,8) is exactly the
sims-flipped move (100 sims plays (5,10)). BUT (5,10) also
loses 3/3 by the oracle — so the collapse is a SYSTEMATIC
line divergence across the game (200-sim White takes
different, worse lines from the opening), not a single
flipped move. The saves ((7,5), (7,6)) are found by NEITHER
sims count. Mechanism (hypothesis, supported by the probe
and the mechanism check in §1): with a misleading prior and
a weak leaf signal (capped gradient ±0.2), deeper search
concentrates visits on the prior's favourite — more sims
means more consistent wrongness in quiet positions, not
better play. The §1 mechanism check showed the same
non-monotonicity: the visit gap on fl1-g3-ply23 closes at
200 sims (26 vs 22) then EXPLODES at 400 (113 vs 29).

### 3.4 Ladder (20 games/opponent, mcts_sims=100, seed 100)

(TBD — running.)

### 3.5 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge)

(TBD — not started.)

## 4. Bar check (v10 vs v7)

(TBD.)

## 5. What this means

(TBD.)

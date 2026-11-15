# Iteration 18 — v18: rollout leaf for both colours (batch 7, H1)

**Status: IN PROGRESS — pre-registration (§1) written before any eval;
implementation not yet started.** Comparison target is v7 (the
batch-7 reference). This is the H1 workstream of batch 7 (H2 is v19,
the blended rollout leaf; H3 is v20, the v1 leaf vs greedy play);
work happens on branch `research/batch-7-h1` in the `gomoku-b7-h1`
worktree. v18 does NOT change the root decision rule
(CLOSED/EXHAUSTED), does NOT change sims/priors/book (CLOSED), does
NOT change exploration constants (CLOSED without a new mechanism),
and does NOT include the threat check (minimal, isolable change —
see §1).

**Method (carried over from batch 3/4/5/6, with the iter-17 §5
correction):** the fast 10-game loop — play 10 games, read EVERY
move of every game, find THE losing move per game (verified with
the solver or a deeper search — never guessed), patch the
algorithm so that, replayed to that critical position, it plays
the perfect move, and record each critical position + correct move
as a regression test. Then play 10 more games and repeat. Max 4
fast-loop iterations before validation, or stop when an iteration
finds no new patchable weakness class. Patches must fix the
STRUCTURAL cause, not the single position. **In this batch the
fast loop runs against v0 as well as v7** (iter-17 §5: v17's
weakness only showed at battery scale against v0, where the fast
loop never looked) — each iteration plays 10 games vs v7 AND 10
games vs v0, and losses from both opponents are walked back. The
full 4-clause validation battery (100-game matches + ladder) runs
on the FINAL patched v18 only. Fast-loop games use development
seeds (1001/2001, 1002/2002, ...) distinct from the bar battery's
pre-registered seeds (100/200), so no development game overlaps an
eval game.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-7 reference: ladder
Elo 190.8, on master at `bf866f7`). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100), clause 3 is
tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v18 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v18 = v16's 2-ply rollout leaf applied as
Black too.** Everything else is v16/v7 unchanged: 100 sims,
c_puct 1.5, v1 priors over all candidates, win/block scans before
search, most-visited root child, terminal leaf values identical.
The ONLY change vs v16 is the quiet-leaf value WHEN THE ENGINE
PLAYS BLACK: instead of v7's static v0-kernel gradient, the leaf
rolls out exactly 2 plies (opponent reply + our reply) under the
v1 rollout policy and evaluates the SYMMETRIC v1-kernel gradient
at the resulting position — the same `_rollout_leaf_value` v16
uses as White, with both of v16's fast-loop patches kept
unchanged. When the engine plays White, v18 IS v16 exactly (same
leaf function, same priors, same search) — the White games are a
built-in control in reverse: any overall edge over v16 must come
from the Black split.

**The rollout, exactly (ported unchanged from v16 — see
`mcts_heuristic._rollout_leaf_value` and `_rollout_move`, and
`research/iterations/iter-16.md` §1):**

- The leaf first applies v7's terminal check: +1 (player to move
  has a winning move), -1 (opponent has >= 2), -0.25 (opponent has
  exactly 1). Terminal positions return the exact terminal value;
  the rollout runs ONLY from quiet positions.
- Rollout policy per ply (player P to move): a winning move wins
  (terminal +1 for P); >= 2 opponent winning moves loses (terminal
  -1 for P); exactly 1 opponent winning move is blocked (forced);
  else the argmax of P's v1 total score (own v1 attack +
  opponent's v1 attack at the square, from one `attack_v1_both`
  call), with open-four priority: if P's best own v1 attack is >=
  OPEN_FOUR (200,000), play the argmax of own attack alone. A
  terminal reached inside the rollout returns the exact terminal
  value, sign-adjusted to the leaf player's perspective.
- After exactly 2 plies, the value is the symmetric v1 gradient at
  the resulting position: `0.2 * tanh((breadth_v1(ours) -
  breadth_v1(theirs)) / 7.5)` (v14's form: v1 thresholds strong >=
  4000, developing [80, 4000), scale 7.5, cap +/-0.2), with a
  terminal check first — but at the rollout's END only
  wins/losses (+/-1) are terminal; a must-block is gradient-
  evaluated (a rollout-found must-block is a prediction under the
  1-ply rollout policy, not a fact about the position).
- 2 plies, not 1 and not 4 (v16's measurements, iter-16 §1.1: the
  save's value comes from OUR next move, so the rollout must
  include it; 4 plies returns false terminal losses for 3 of 10
  saving moves).

**What changes vs v16 (exactly one thing).** The leaf dispatch:
v16's `_rollout_v1_leaf_value` returns v7's gradient leaf when the
root player is Black and the rollout leaf when White. v18 adds
leaf mode `rollout2_v1_both`, whose dispatch returns
`_rollout_leaf_value` for BOTH colours. The rollout function
itself, the policy, the gradient, the cap, and both fast-loop
patches are byte-identical to v16's — the port is checked out
unchanged from `research/batch-6-h2` (commit `34950d2`), and the
v1 kernel (`kernel_v1.py`) is v14's exact port, reused directly,
NOT re-ported. The dispatch uses the conditional form (the naive
`self._leaf_value = _LEAF_FUNCTIONS[leaf_mode]` raises KeyError
on rollout modes — batch 6 caught this as a silent merge
breakage).

**Why this, and what it is meant to fix.** v16's battery
(iter-16 §3): vs v7 52W-37L-11D, 0.5750, CI [0.4771, 0.6673] —
Black 0.51 / White 0.64. The White gain is real (0.64 vs v7, 50W-0L
vs v0, ladder Elo 381.7 — highest in the program), but the Black
mirror games — v16-as-Black IS v7 — ran 0.51 and capped the
combined h2h 2.5 points short of an established margin (the CI
lower bound needs ~0.60 to clear 0.50). The question this
iteration answers is whether the rollout leaf's value is a White
phenomenon (the diagnosed beyond-horizon counterplay problem,
iter-13 §2.2, is a White problem) or a general leaf improvement
that also helps the attacker. If Black gains anything like White's
+12 points, the combined h2h clears 0.60 and clause 1 passes; if
Black is flat, v18 reproduces v16's 0.575 and fails the same way.

**Expected effect — honestly two-sided.** The upside case: Black
loses games the same way White does — lost open-four races where
the value of a quiet counterplay move materialises past the
forcing exchange — and the rollout's 2-ply horizon sees it. The
downside case, and it is real: (a) Black is already strong
(v7-as-Black vs v1: 0.89), so there is less headroom and more
room for the leaf to mislead; (b) iter-16 §3.1's q-compression —
in creator-rich positions the rollout returns -1.0 from nearly
every quiet leaf, flattening the q signal the search needs to
discriminate among defences — bites hardest exactly where Black
lives, since Black attacks first and Black's positions are
creator-rich by nature. The rollout leaf could therefore make
Black WORSE even as it made White better. There is no design
measurement that settles this: the v16 critical-position
diagnostics are all White-to-move (the v7/v13/v14 regression
positions), so no pre-registration diagnostic exists for Black.
The fast loop (§2) is the finding method, and the 100-game
battery is the verdict — as always, small-sample smokes mislead
(v15: two false positives at n=40 in one iteration).

**Tactical expectation (pre-registered).** v16 failed one hard
position, `prev-v2v1s-g12` (28/29) — a White-to-move forced loss
decided by a 5-5 visit-tie RNG coin-flip under q-compression.
v18-as-White IS v16, so the failure MECHANISM is unchanged; but
the tactical driver runs one player instance across the whole
suite (seed 100), so the RNG stream at g12 diverges from v16's
run (v18's Black leaf changes RNG consumption at earlier
Black-to-move positions), and the tie-break could go either way
by luck. Expected: 28/29 most likely, 29/29 possible only by
tie-break luck, not by a fix. A genuine fix for q-compression is
H2's (v19, the blend) brief, not H1's — UNLESS the fast loop
finds a leaf-level patch (not a root rule — CLOSED) that restores
discrimination among losing lines, in which case it is recorded
in §2 and the battery runs on the patched version. The tactical
suites are run after EVERY fast-loop patch, not just at battery
time (iter-16 §3.1's lesson: the mechanism checks did not cover
g12, so the regression was caught only at the battery).

**Threat check: NOT included.** Same reasoning as v16: the ONLY
difference from v16 is the Black leaf value. Including the threat
check would add a confound without an expected gain (iter-13
measured it as worth ~nothing against v7).

**Cost (from v16's measurements — the leaf function is
unchanged, so the cost is identical by construction).**
`_rollout_leaf_value` costs ~14 ms/call after the v1-policy
patch (vs v7's 3.64 ms). v16's White moves ran ~1.9 s; v18's
Black moves now cost the same, so a 100-game match vs v7 is
~2-2.5 h (v16's took 105 min with Black at v7 speed), vs v0
~1 h, swap ~2 h. Affordable.

**Eval seeds (pre-registered):** h2h vs v7: v18=100 / v7=200; h2h
vs v0: v18=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Fast-loop dev seeds 1001/2001, 1002/2002, ...
(iteration k uses 1000+k / 2000+k, per iter-8 §4's policy).

### 1.2 Mechanism check (post-implementation, pre-fast-loop)

After implementation and before any fast-loop game, v18 was run
once on the 5 v16 regression positions (seed 0, full 100-sim
search). This is an implementation check, not pre-registration
evidence and not an eval — it verifies the dispatch does what §1
says it does, on the only Black diagnostic positions available
(the two v16 fast-loop Black xfails):

| Position | To move | v18 played | Expected | Result |
|---|---|---|---|---|
| mech-v13-g9-ply15 | White | (7,6) | (7,6) | PASS |
| mech-v14-g3-ply13 | White | (6,5) | (6,5) | PASS |
| fl1-g7-ply89 | White | (12,0) | (12,0)/(12,1)/(11,12)/(1,9) | PASS |
| fl1-g0-ply12 | Black | (7,10) | (9,7)/(5,5)/(7,8) | FAIL (xfail) |
| fl1-g6-ply78 | Black | (5,7) | (5,7)/(9,13)/(1,9)/(6,6) | PASS |

The three White positions pass, as they must — v18-as-White IS
v16 (the dispatch preserves White behaviour exactly). On the two
Black positions v16 could not fix, the Black rollout leaf flips
ONE: fl1-g6-ply78, where v18 plays the oracle save (5,7) that
v16/v7 missed (played (9,14)). On fl1-g0-ply12 v18 changes the
move ((11,9) → (7,10)) but still does not find a save — whether
(7,10) also loses 3/3 under the oracle was not tested here, so
no claim is made about it; the position stays an xfail. The
signal is thin (n=2 Black positions) and diagnostics do not
predict match strength (iter-13's lesson) — but it is positive:
the rollout leaf is not White-specific at the mechanism level,
and the fast loop now has a Black diagnostic that passes. These
5 positions are v18's regression file
(`regression_positions_v18.json`, copied unchanged from v16's),
with fl1-g0-ply12 as the single xfail.

### 1.3 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v18 only (after the fast loop):

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

**Integration.** `ml/gomoku_ml/mcts_heuristic.py` gains the
`rollout2_v1_both` leaf mode (dispatch only — the rollout
function is v16's, unchanged) and `ml/gomoku_ml/scoring.py`
gains ONLY the `ENGINE_VERSIONS[18]` registry entry.
`kernel_v1.py` (+ its tests and fixtures) is checked out
unchanged from `research/batch-6-h2`. `baseline.py` is frozen
and untouched.

## 2. Loss analysis for iteration 18

(TODO — the fast-loop worked games ARE the loss analysis.)

### 2.1 Fast-loop iteration 1 — 10 games vs v7 + 10 games vs v0 (pre-patch engine)

**Batches:** `research/iterations/fastloop1-v18-vs-v7.json`
(seeds v18=1001 / v7=2001) and
`research/iterations/fastloop1-v18-vs-v0.json` (seeds v18=1001 /
v0=2001) — development seeds, distinct from the bar battery's
100/200. **vs v7: 8W-1L-1D, win rate 0.8500, CI [0.5411, 0.9646]**,
690.5 s. Split: as Black **5W-0L (1.00)**, as White 3W-1L-1D
(0.70). **vs v0: 10W-0L-0D, win rate 1.0000**, 408.7 s (as Black
5W-0L, as White 5W-0L). Classifier (both batches): 1/1 loss via
open four, 0 missed blocks, 0 clean missed preventions in 3
threatened turns (losses). Defence is still perfect; the Black
leaf produced no loss in 10 Black games. (n=10 per opponent —
development signal, not the bar; the fast loop's job is finding
weaknesses, not measuring strength.)

**The only loss — game 1 vs v7 (White, 21 plies) — has NO
patchable losing move.** Walk-back
(`research/iterations/walkback-fl1-g1.json`, corrected oracle
method, seeds 0,1,2) nominates ply 17: actual (6,4) loses 3/3,
(9,4) "saves" (draws 3/3); ply 13 nominates two "saves" ((7,4),
(9,8)). **Both nominations are v1-oracle artifacts, refuted by
exact computation and a stronger oracle** (`/tmp/diag_fl1_g1.py`,
`/tmp/verify_fl1_g1.py`, `/tmp/trace_fl1_g1_ply13.py`):

- Ply 17: after White (9,4), Black (6,5) leaves Black TWO winning
  moves — (4,3) and (9,8) — with White having none. (9,4) loses
  by force; the rollout leaf's -1.000 for (9,4), (6,4) and (6,5)
  alike is CORRECT, not q-compression error. The v1 oracle calls
  (9,4) a draw because v1-as-Black plays (9,6) instead of (6,5)
  — even though (6,5)'s v1 attack score is 200,091, an open-four
  creation. v7-as-Black plays (6,5) after (9,4); the real
  opponent finds the killer. After the actual (6,4), (6,5)
  leaves the same two winning moves: at ply 17 EVERY White move
  loses by force, and v18's root (6,4)/(6,5) tied at 6 visits,
  (9,4) at 5, all our-q ≈ -1.0) is a tie among losing moves.
  (Fresh-seed v18, v16 and v7 all play (6,5) here; the game
  played (6,4) on RNG state. The choice does not matter.)
- Ply 13: after White (9,8) — the oracle's "wins 3/3" save —
  Black plays (7,8) and has TWO winning moves, (7,4) and (7,9),
  with White having none: a forced loss, exact. v7-vs-v7
  confirms (Black wins in 3 plies at all 3 seeds). The other
  "save", (7,4), loses 3/3 under v7-vs-v7 (17/23/21 plies). The
  actual (7,8) is the BEST tested move at ply 13 under the
  stronger oracle (loses 2/3, wins 1/3 in 72 plies).
- Ply 11: after the actual (9,7), v7-vs-v7 loses 2/3 (both in 9
  plies) — the game was already going bad under the strong
  oracle while the v1 oracle called it a draw.

So the loss is distributed drift (the iter-6 result): no single
verified losing move, at every White turn from ply 11 on the
tested alternatives fare no better than the actual under the
stronger oracle. **No patch.** Game 9 vs v7 (White, 225 plies)
was a draw — no losing move by definition.

**Method finding — the v1 oracle shares the argmax-of-sum flaw.**
v1's 1-ply argmax picks a defensive block over a forced-win
open-four creation in this game twice ((9,6) over (6,5) after
(9,4); missing (7,8) after (9,8)) — the same flaw iter-16 §2.2
found in the rollout policy and patched there with open-four
priority, now found in the walk-back oracle itself. The rollout
leaf, which has the patch, sees these forced wins exactly; the
v1 oracle does not. **Rule for the rest of this iteration (and
future rollout-leaf work): a walk-back save must be re-verified
with v7-vs-v7 playouts — or exact computation — whenever the
rollout leaf disagrees with the v1 oracle. The v1 oracle alone is
no longer sufficient verification for a rollout-leaf engine.**
This iteration's walk-backs apply that rule throughout.

**Iteration 1 verdict: no new patchable weakness class — but the
loop does NOT stop here.** The loop has patched nothing yet, and
the Black leaf — the whole point of v18 — produced no loss in
10 Black games, so it has had no chance to show a weakness.
Iteration 2 runs on fresh seeds (1002/2002); if it also finds no
patchable class, the loop stops (two consecutive barren
iterations, 40 games). No regression positions added (a forced
loss has no "expected save" to record).

### 2.2 Fast-loop iteration 2

(TODO)

### 2.3 Fast-loop iteration 3

(TODO)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration N)

(TODO)

### 2.5 Aggregate counts (secondary)

(TODO)

## 3. v18 evaluation

(TODO — battery on the FINAL version only.)

### 3.1 Tactical suites

(TODO)

### 3.2 Head-to-head vs v7

(TODO)

### 3.3 Head-to-head vs v0

(TODO)

### 3.4 Swap-rule head-to-head vs v7

(TODO)

### 3.5 Ladder

(TODO)

### 3.6 Test suites (clause 4)

(TODO)

## 4. Bar check (v18 vs v7)

(TODO)

## 5. What this means

(TODO)

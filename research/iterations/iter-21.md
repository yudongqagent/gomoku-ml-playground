# Iteration 21 — v21: v18 + blend (rollout leaf both colours + blended terminals) (batch 8, H1)

**Status: IN PROGRESS — pre-registration (§1) written before any eval;
implementation not yet run.** Comparison target is v7 (the batch-8
reference). This is the H1 workstream of batch 8 (H2 is v22, the
conditional gap-three demotion; H3 is v23, the q-aware visit-tie
break); work happens on branch `research/batch-8-h1` in the
`gomoku-b8-h1` worktree. v21 does NOT change the root decision rule
(CLOSED/EXHAUSTED), does NOT change sims/priors/book (CLOSED), does
NOT change exploration constants (CLOSED without a new mechanism),
does NOT change the rollout policy or horizon, and does NOT include
the threat check (minimal, isolable change — see §1).

**Method (carried over from batch 3/4/5/6/7, with the iter-17 §5
correction):** the fast 10-game loop — play 10 games, read EVERY
move of every game, find THE losing move per game (verified with
the solver or a deeper search — never guessed), patch the
algorithm so that, replayed to that critical position, it plays
the perfect move, and record each critical position + correct move
as a regression test. Then play 10 more games and repeat. Max 4
fast-loop iterations before validation, or stop when an iteration
finds no new patchable weakness class. Patches must fix the
STRUCTURAL cause, not the single position. **The fast loop runs
against v0 as well as v7** (iter-17 §5: v17's weakness only showed
at battery scale against v0, where the fast loop never looked) —
each iteration plays 10 games vs v7 AND 10 games vs v0, and losses
from both opponents are walked back. The full 4-clause validation
battery (100-game matches + ladder) runs on the FINAL patched v21
only. Fast-loop games use development seeds (1001/2001, 1002/2002,
...) distinct from the bar battery's pre-registered seeds (100/200),
so no development game overlaps an eval game.

**Methodology (iter-18 §2.1, binding for rollout-leaf engines):**
the v1 walk-back oracle shares the argmax-of-sum flaw and can no
longer verify saves alone — every nominated save needs v7-vs-v7
playouts or exact computation.

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-8 reference: ladder
Elo 190.8, on master at `bf866f7`). The bar's clause 1 is
head-to-head vs v7 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100), clause 3 is
tactical pass rate >= v7's (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v21 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v21 = v18's 2-ply rollout leaf applied in BOTH
colours PLUS v19's blended rollout terminals.** Everything else is
v18/v19 unchanged: 100 sims, c_puct 1.5, v1 priors over all
candidates, win/block scans before search, most-visited root child,
terminal leaf values identical. The ONLY change vs v18 is what a
rollout-found terminal is worth; the ONLY change vs v19 is that the
blended rollout leaf applies as Black too (v19-as-Black IS v7;
v21-as-Black is the blended rollout leaf). v21-as-White IS v19
exactly (same leaf function, same priors, same search) — the White
games are a built-in control in reverse: any overall edge over v19
must come from the Black split, and any tactical change on
White-to-move positions must come from RNG-stream divergence, not
the leaf.

**The rollout, exactly (ported unchanged from v16/v18 — see
`mcts_heuristic._rollout_leaf_value` and `_rollout_move`, and
`research/iterations/iter-16.md` §1 / `iter-18.md` §1):**

- The leaf first applies v7's terminal check: +1 (player to move
  has a winning move), -1 (opponent has >= 2), -0.25 (opponent has
  exactly 1). Terminal positions return the exact terminal value;
  the rollout runs ONLY from quiet positions. **Leaf-start
  terminals keep their exact values under the blend** — those are
  FACTS about the real position, not predictions.
- Rollout policy per ply (player P to move): a winning move wins
  (terminal +1 for P); >= 2 opponent winning moves loses (terminal
  -1 for P); exactly 1 opponent winning move is blocked (forced);
  else the argmax of P's v1 total score (own v1 attack +
  opponent's v1 attack at the square, from one `attack_v1_both`
  call), with open-four priority: if P's best own v1 attack is >=
  OPEN_FOUR (200,000), play the argmax of own attack alone. A
  terminal reached inside the rollout returns the (blended)
  terminal value, sign-adjusted to the leaf player's perspective.
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

**The blend, exactly (ported unchanged from v19 — see
`mcts_heuristic._blend_rollout_terminal` and
`research/iterations/iter-19.md` §1).** When the rollout reaches
a terminal (±1 — a win/loss found inside the rollout, or at the
rollout's end), v18 returns the raw terminal value. v21 returns
the terminal BLENDED with the symmetric v1 gradient evaluated at
the terminal position:

```
blended(term, board_T, player_T) = clamp(term + g, -1.0, 1.0)
    where g = _attack_gradient_v1(board_T, player_T)
```

`board_T` is the position where the rollout reached the terminal
(the position BEFORE the winning move is played — the terminal is
a prediction that the player to move there wins, not a finished
game), `player_T` is the player to move there, and the blend is
formed in `player_T`'s perspective and sign-adjusted to the leaf
player's perspective exactly as v16 sign-adjusts the raw terminal
(negated on odd plies). The gradient is v14's symmetric v1 form
(cap ±0.2, scale 7.5, v1 thresholds) — the SAME gradient v16
already evaluates at quiet rollout ends, so **zero new tunable
constants**. A draw terminal (term 0.0, board full) is returned
unchanged — a full board has no empty cells, so its gradient is 0
anyway. The blend applies ONLY to rollout-found ±1 terminals.

**What changes vs v18 (exactly one thing).** The leaf dispatch:
v18's `rollout2_v1_both` returns `_rollout_leaf_value` (raw
terminals) for both colours. v21 adds leaf mode
`rollout2_v1_both_blend`, whose dispatch returns
`_rollout_leaf_value(board, player, blend=True)` for BOTH colours.
The rollout function itself, the policy, the gradient, the cap,
and both fast-loop patches are byte-identical to v16's/v18's —
the port is checked out unchanged from `research/batch-7-h2`
(commit `d73e7ca`, whose `_rollout_leaf_value` already carries the
`blend` parameter with `blend=False` reproducing v18 exactly),
and the v1 kernel (`kernel_v1.py`) is v14's exact port, reused
directly, NOT re-ported. The dispatch uses the conditional form
(the naive `self._leaf_value = _LEAF_FUNCTIONS[leaf_mode]` raises
KeyError on rollout modes — batch 6 caught this as a silent merge
breakage). v16, v18 and v19 are all registered on this tree as
controls (see §1.2).

**Why this, and what it is meant to fix.** v18 is the strongest
version in the program on every game-strength measure — h2h vs v7
0.7750 (CI [0.6839, 0.8458]), ladder Elo 636.4 [240.3, 1032.5],
first version ever to beat the ladder MCTS outright (14-6), vs v0
0.97, swap vs v7 0.63 — and fails the bar on tactical alone
(25/29 hard). Three of its four hard failures are Black-to-move
q-compression visit-ties (3-3, 3-3, 8-8) where the leaf KNOWS the
answer (chosen −1.000 vs expected +0.250) but cannot convert it
into visits against the losers' ~3x priors; the fourth is a
suite-list gap (the chosen move wins 3/3 under v7-vs-v7). v19's
blend was designed to fix exactly this root cause — a continuous
terminal+gradient value spreading q away from the ±1 caps — but
was REFUTED as a general mechanism: on White it was inert on g12
(the gradient at terminal positions aligns with the terminal and
the clamp pins everything) and it REGRESSED g45 (softening the
losing move (10,4) from q −1.000 to −0.932, its 3× prior advantage
earning a second visit, tying 2-2-2, tie-break picks the loser).
The v18+blend combination was NEVER tested — v19 is White-only,
v18 is unblended. This iteration tests it, with its own fast-loop
validation and full battery, not an assumption.

**Expected effect — honestly two-sided.** The upside case: in
Black's creator-rich positions the blend spreads q among losing
lines (losing lines with counterplay breadth rise above −1.0,
hopeless lines stay pinned), so PUCT visits follow q instead of
tying on priors — converting some or all of v18's three Black
tie failures — while v19's White gains (White split vs v7 0.74,
best in the program; swap 0.73) carry over unchanged, since
v21-as-White IS v19. The downside case, and it is real: (a) the
g45 mechanism — the blend softens a losing attacker's q, and with
a ~3x prior advantage the softened loser earns the tying visit —
can fire on Black's side too, regressing positions v18 passes;
(b) the blend was inert on g12 because terminal-position
gradients align with the terminal, and the same alignment may
hold on v18's three tie positions (their expected moves hold
+0.250 leaf values, but the blend acts on the terminal positions
inside the rollout, not on the leaf-start values); (c) game
strength could move either way — v19's White split improved
+10 points over v16 while its Black mirror games ran −10 on seed
noise, so the combined h2h is fragile to RNG-stream divergence.
There is no design measurement that settles this: the blend's
effect on a position is a search-level outcome, not a leaf-level
one. The fast loop (§2) is the finding method, and the 100-game
battery is the verdict — as always, small-sample smokes mislead
(v15: two false positives at n=40 in one iteration).

**Tactical expectation (pre-registered).** v18: 25/29 (fails
g26, g79, v1bl-g2, winning-quiet-setup-2 — all Black-to-move).
v19: 27/29 (fails g12 inert, g45 regressed — both White-to-move).
v21-as-White IS v19, so the White positions behave as v19's
except for RNG-stream divergence (the tactical driver runs one
player instance across the suite, so v21's Black leaf changes RNG
consumption at earlier Black-to-move positions and the g12/g45
tie-breaks could fall either way by luck). The Black positions
are the experiment: the blend spreads q on the three tie
failures, but the g45 mechanism shows spreading can also CREATE
ties. Expected: 26–28/29 most likely; 29/29 requires the blend
to fix all three Black ties without regressing g45, g12, or any
position v18 passes — possible, not predicted. A genuine fix for
q-compression at the root is H3's (v23, the q-aware tie-break)
brief, not H1's — UNLESS the fast loop finds a leaf-level patch
(not a root rule — CLOSED) that restores discrimination among
losing lines, in which case it is recorded in §2 and the battery
runs on the patched version. The tactical suites are run after
EVERY fast-loop patch, not just at battery time (iter-16 §3.1's
lesson: the mechanism checks did not cover g12, so the regression
was caught only at the battery).

**Threat check: NOT included.** Same reasoning as v16/v18/v19:
the ONLY difference from v18 is the blended terminal. Including
the threat check would add a confound without an expected gain
(iter-13 measured it as worth ~nothing against v7).

**Cost (from v18's measurements + v19's blend cost).**
`_rollout_leaf_value` costs ~14 ms/call (v16's measurement);
the blend adds one `attack_v1_both` call per rollout terminal
(~24% of leaves, ~4 ms each — roughly +1 ms per leaf call).
v18's 100-game match vs v7 took 6301 s (~1.75 h); v21's should
take ~2–2.5 h. vs v0 ~1–1.5 h, swap ~2 h, ladder ~1 h.
Affordable.

**Eval seeds (pre-registered):** h2h vs v7: v21=100 / v7=200; h2h
vs v0: v21=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Fast-loop dev seeds 1001/2001, 1002/2002, ...
(iteration k uses 1000+k / 2000+k, per iter-8 §4's policy).

### 1.1 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v21 only (after the fast loop):

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
`rollout2_v1_both_blend` leaf mode (dispatch only — the rollout
function and the blend are v16's/v19's, unchanged) plus v18's
`rollout2_v1_both` dispatch (needed for the v18 control), and
`ml/gomoku_ml/scoring.py` gains the `ENGINE_VERSIONS[21]`
registry entry (plus [16], [18], [19] as controls).
`kernel_v1.py` (+ its tests and fixtures) is checked out
unchanged from `research/batch-7-h2`. `baseline.py` is frozen
and untouched.

### 1.2 Port controls (post-implementation, pre-fast-loop) — BOTH PASS

Before any v21 eval, the port was verified by reproducing the two
parent versions' tactical numbers on this tree, under the
identical protocol (`eval_tactical_full.py --seed 100`):

- v18: **20/20 old + 25/29 hard**, failing exactly g26 (chose
  (12,3)), g79 (chose (5,1)), v1bl-g2 (chose (7,9)) and
  winning-quiet-setup-2 (chose (9,7)) — identical to iter-18 §3.1,
  including passing g12 (chose (12,2)) by RNG tie-break luck.
- v19: **20/20 old + 27/29 hard**, failing exactly g12 (chose
  (7,3)) and g45 (chose (10,4)) — identical to iter-19 §3.1.

Both controls reproduce exactly, so the port is trusted: v18's
rollout leaf and v19's blend behave on this tree exactly as they
did on their own branches. These are implementation checks, not
pre-registration evidence and not eval.

### 1.3 Mechanism check (post-implementation, pre-fast-loop)

After implementation and before any fast-loop game, v21 was run
once on v18's 6 regression positions (seed 0, full 100-sim
search) and root dumps were taken on v18's four tactical failures
and v19's g45/g12 (`/tmp/mech_v21.py`). This is an implementation
check, not pre-registration evidence and not an eval — it verifies
the dispatch does what §1 says it does, and records whether the
blend moves any of the documented tie-breaks at seed 0. (Tactical
results are seed-dependent for rollout-leaf engines; the
mechanism check at seed 0 does not predict the battery's seed-100
outcome.)

**Regression positions (seed 0):** v21 matches v18 on all 6 —
passes the 4 v18 passes (mech-v13-g9-ply15 (7,6), mech-v14-g3-ply13
(6,5), fl1-g7-ply89 (12,0), fl1-g6-ply78 (5,7)) and fails the same
2 xfails (fl1-g0-ply12 (7,10), fl2-g7-ply9 (6,4)). v21-as-White
matches v19 on all three White positions, as designed. These 6
positions are v21's regression file
(`regression_positions_v21.json`, copied unchanged from v18's),
with the same 2 xfails.

**Root dumps (fresh seed 0):**

| Position | To move | v21 | v18 | What the blend did |
|---|---|---|---|---|
| prev-v2v1s-g12 | White | (7,3) FAIL | (7,3) FAIL | INERT — root dumps byte-identical (5-5-5-5-5 tie, good moves our-q −0.75 vs blockers −1.00); the gradient at terminal positions aligns with the terminal and the clamp pins everything, exactly as iter-19 §3.1 found for v19. |
| prev-v1v0s-g26 | Black | (9,4) PASS | (9,4) PASS | Spread q (expected −0.583→−0.48/−0.518, losers −1.000→−0.94) but the 3-3-3-3 visit tie PERSISTS; both pass at seed 0 by RNG luck. The blend widens the q gap (0.42→0.46) without converting it into visits. |
| prev-v1v1s-g79 | Black | (7,9) PASS | (7,8) PASS | BREAKS the tie: (7,9) takes 4 visits outright (our-q −0.577) vs 3s — a genuine visit-allocation change, not RNG luck (v18 ties seven moves at 3 and passes by luck). |
| prev-v1bl-g2 | Black | (7,5) PASS | (7,9) FAIL | BREAKS the tie: (7,5) 8 visits vs (7,9) 7 (v18 ties 8-8 and RNG picks the loser). The expected move's q rose more (−0.594→−0.513) than the loser's (−0.724→−0.663). A genuine blend fix at the mechanism level. |
| prev-v1v1s-g45 | White | (12,7) PASS | (12,7) PASS | The g45 regression mechanism is PRESENT: the loser (10,4) is softened to −0.932 and earns a second visit, tying 2-2-2-2 with the good moves; v21 passes at seed 0 by RNG luck only (iter-19: fresh-player v19 also passes here; the battery RNG picked (10,4)). |
| winning-quiet-setup-2 | Black | (9,7) FAIL | (9,7) FAIL | INERT — root dumps byte-identical (79 visits on (9,7), our-q +0.959, vs 16 on (8,7) at +1.000). The suite-list gap: (9,7) wins 3/3 under v7-vs-v7 (iter-18 §3.1). |

**Pre-fast-loop tactical diagnostic (seed 100, the battery
protocol — NOT the verdict, which runs on the final patched
version): v21 scores 20/20 old + 26/29 hard**, failing g12
(chose (7,3)), g45 (chose (10,4)) and winning-quiet-setup-2
(chose (9,7)). All three of v18's Black q-compression ties are
fixed at seed 100 (g26 chose (8,4), g79 chose (7,9), v1bl-g2
chose (7,5)) — the blend does what it was designed to do on
Black's side — but v21 inherits v19's two White failures (g12
inert, g45 regressed) plus the suite-list gap. 26/29 sits between
v18's 25/29 and v19's 27/29, inside the pre-registered 26–28/29
range. Clause 3 needs 29/29; the fast loop (§2) is the only
remaining route to a leaf-level patch — the three failures are
all tie/RNG or suite-list cases, and a root rule is CLOSED.

## 2. Loss analysis for iteration 21

(TODO — the fast-loop worked games ARE the loss analysis.)

## 3. v21 evaluation

(TODO — battery on the FINAL version, pre-registered seeds
100/200 throughout §1.1.)

## 4. Bar check (v21 vs v7)

(TODO)

## 5. What this means

(TODO)

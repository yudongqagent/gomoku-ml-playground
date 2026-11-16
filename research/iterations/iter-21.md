# Iteration 21 — v21: v18 + blend (rollout leaf both colours + blended terminals) (batch 8, H1)

**Status: COMPLETE — battery done, 3/4 clauses; REJECTED on clause 3
(tactical 26/29) alone.** Comparison target is v7 (the batch-8
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

(The fast-loop worked games in §2.1–§2.2 ARE the loss analysis;
§2.5 aggregates. No patches were made — see §2.2.)

### 2.1 Fast-loop iteration 1 — 10 games vs v7 + 10 games vs v0 (pre-patch engine)

**Batches:** `research/iterations/fastloop1-v21-vs-v7.json`
(seeds v21=1001 / v7=2001) and
`research/iterations/fastloop1-v21-vs-v0.json` (seeds v21=1001 /
v0=2001) — development seeds, distinct from the bar battery's
100/200. **vs v7: 6W-2L-2D, win rate 0.7000, CI [0.3968,
0.8922]**, 1042.8 s. Split: as Black **5W-0L (1.00)**, as White
1W-2L-2D (0.40). **vs v0: 10W-0L-0D, win rate 1.0000**, 253.1 s
(as Black 5W-0L, as White 5W-0L). Classifier (both batches): 2/2
losses via open four, 0 missed blocks, 0 clean missed preventions
in 5 threatened turns (losses). Defence is still perfect; the
blended Black leaf produced no loss in 10 Black games. (n=10 per
opponent — development signal, not the bar; the fast loop's job
is finding weaknesses, not measuring strength.) Two of the ten
vs-v7 games were 225-ply full-board draws (games 3 and 5, both
v21-as-White): v21's defence holds v7-as-Black to a full board
without a win.

**Game 1 vs v7 (White, 21 plies) — byte-identical to v18's
fast-loop-1 game 1, and its nominations are refuted the same
way.** The move list is identical to v18's fl1-g1 (verified:
`research/batch-7-h1:research/iterations/fastloop1-v18-vs-v7.json`
game 1) — the blend changed nothing in this game; v21-as-White
plays v18-as-White's moves throughout. Walk-back
(`research/iterations/walkback-fl1-g1.json`, corrected oracle
method, seeds 0,1,2) nominates ply 17: actual (6,4) loses 3/3,
(9,4) "saves" (draws 3/3); ply 13 nominates two "saves" ((7,4),
(9,8)). **Both nominations are v1-oracle artifacts, refuted by
exact computation on v21's own game record**
(`/tmp/verify_fl1_g1.py` — the iter-18 §2.1 refutation,
re-verified, not merely cited):

- Ply 17: after White (9,4), Black (6,5) leaves Black TWO winning
  moves — (4,3) and (9,8) — with White having none. (9,4) loses
  by force; the rollout leaf's −1.000 for (9,4), (6,4) and (6,5)
  alike is CORRECT, not q-compression error. The v1 oracle calls
  (9,4) a draw because v1-as-Black plays (9,6) instead of (6,5)
  — even though (6,5)'s v1 attack score is 200,091, an open-four
  creation. v7-as-Black plays (6,5) after (9,4); the real
  opponent finds the killer. After the actual (6,4), (6,5)
  leaves the same two winning moves: at ply 17 EVERY White move
  loses by force.
- Ply 13: after White (9,8) — the oracle's "wins 3/3" save —
  Black plays (7,8) and has TWO winning moves, (7,4) and (7,9),
  with White having none: a forced loss, exact. The other "save",
  (7,4), leaves only one winning move after (7,8) — but iter-18
  verified it loses 3/3 under v7-vs-v7, and the actual (7,8) is
  the BEST tested move at ply 13 under the stronger oracle.

So the loss is distributed drift (the iter-6 result): no single
verified losing move. **No patch.**

**Game 9 vs v7 (White, 29 plies) — no nominated losing move at
all.** Walk-back (`research/iterations/walkback-fl1-g9.json`):
at every White turn from ply 13 on, the actual move's oracle
playout does not lose at all 3 seeds (draws or wins at some
seeds), and the last two turns (ply 25, 27) have no saving
alternative. The loss is distributed drift with no single losing
move. **No patch.**

**Iteration 1 verdict: no new patchable weakness class — but the
loop does NOT stop here.** The loop has patched nothing yet, and
the blended Black leaf — the whole point of v21 — produced no
loss in 10 Black games, so it has had no chance to show a
weakness; the specific risk (a loss where v21 plays a loser v18
does NOT play — a blend-caused regression) has not been tested by
a loss at all. Iteration 2 runs on fresh seeds (1002/2002); if it
also finds no patchable class, the loop stops (two consecutive
barren iterations, 40 games). No regression positions added (a
forced loss / distributed drift has no "expected save" to
record).

### 2.2 Fast-loop iteration 2 — 10 games vs v7 + 10 games vs v0 (LOOP STOPS)

**Batches:** `research/iterations/fastloop2-v21-vs-v7.json`
(seeds v21=1002 / v7=2002) and
`research/iterations/fastloop2-v21-vs-v0.json` (seeds v21=1002 /
v0=2002). **vs v7: 7W-1L-2D, win rate 0.8000, CI [0.4902,
0.9433]**, 940.9 s. Split: as Black 4W-0L-1D (0.90), as White
3W-1L-1D (0.70). **vs v0: 10W-0L-0D, win rate 1.0000**, 298.1 s.
Classifier (vs v7): 1/1 loss via open four, 0 missed blocks, 0
clean missed preventions in 2 threatened turns (losses). Two of
the ten vs-v7 games were 225-ply full-board draws (games 1 and
4 — the fifth marathon draw of the loop): v21's defence holds
v7-as-Black to a full board without a win, and v21-as-Black once
failed to beat v7-as-White the same way. Draws are not losses;
they count as half and they say the defence is airtight on both
sides.

**The only loss — game 7 vs v7 (White, 21 plies) — is
byte-identical to v18's fast-loop-2 game 7, and has a nominated
losing move but NO patchable mechanism.** Walk-back
(`research/iterations/walkback-fl2-g7.json`): last preventable
turn ply 9 — actual (6,4) loses 3/3 under the v1 oracle, (10,8)
"wins" 3/3 (ply 7 nominates (10,8) too, among 4 saves; plies
11–19 have no saves). **Re-verification (per the §2.1 rule) plus
a blend check** (`/tmp/diag_fl2_g7.py`):

- The two moves are near-twins by every 1-ply measure: (6,4) v1
  score 200,167 rank 1, (10,8) 200,091 rank 2; priors both
  0.204; both reduce Black's 2 creators to 0 (iter-18 §2.2's
  measurements, on the identical position).
- **The blend changes NOTHING here:** the blended and raw rollout
  leaf values are identical after both moves (+0.107 for White
  after (6,4), +0.039 after (10,8)) — both lines end in quiet
  gradient evaluations, with no rollout terminal to blend. v21's
  root prefers the actual (6,4) 46 visits to 35 (our-q +0.024 vs
  +0.009). This loss is v18's weakness inherited unchanged, not
  a blend regression — the blend is exonerated as its cause.
- Stronger oracle (v7-vs-v7, 3 seeds): after (6,4) White loses
  3/3; after (10,8) White wins 2/3. (10,8) is probably better —
  but it is a noisy edge, not a verified save, and the
  distinction plays out past the 2-ply horizon (Black's win came
  at ply 20 via (8,9), a slow squeeze in the 7–9 columns, not a
  tactic). The rollout traces show why the leaf cannot separate
  them: after (6,4) Black's rollout reply is the twin (10,8)
  and White replies (11,9); after (10,8) Black replies (6,4)
  and White replies (5,3). Both lines end quiet; the gradient
  prefers the (11,9) ending — the move White actually played in
  the losing game.

**No patch.** A patch needs a structural cause, and there is
none here to fix: the rollout policy is faithful for 2 plies
(it plays the twin, as the real Black did), the leaf values the
lines as the gradient sees them, and the difference is a
positional judgement beyond any 1-ply breadth measure. A deeper
rollout (4-ply) was already refuted (false terminals, iter-16
§1.1). Patching on this position would be fitting noise. The
position IS recorded as a regression xfail (`fl2-g7-ply9` —
already in v21's regression file, copied from v18's), with the
noisy verification noted in the test, so the twin-choice weakness
class is documented for future work rather than dropped.

**No White loss in this iteration produced a patchable class,
and no Black loss occurred at all — the fast loop STOPS here
(2 iterations) per the pre-commitment in §2.1** (two
consecutive barren iterations, 40 games). The blended Black leaf
— the whole point of v21 — went 19W-0L-1D across the loop (10 vs
v7, 10 vs v0) without ever showing a weakness to patch. The
battery is the verdict.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast-loop totals (2 iterations, 40 games): vs v7 13W-3L-4D
(0.7250); vs v0 20W-0L-0D (1.0000). Colour splits vs v7: as
Black 9W-0L-1D (0.95), as White 4W-3L-3D (0.55). Fast-loop
defence: 3/3 losses via open four, 0 missed blocks, 0 clean
missed preventions (7 threatened turns in losses). Defence
remains solved; all three losses were White games with no
patchable move (one forced-loss/distributed drift with oracle
artifacts, one distributed drift with no nominated move, one
noisy beyond-horizon twin choice). No patches were made in
this iteration — v21-as-batteried is v21-as-pre-registered.

## 3. v21 evaluation

Battery on the FINAL v21 (no patches were made in the fast loop, so
v21-as-batteried IS v21-as-pre-registered — §2.5), pre-registered
seeds 100/200 throughout (§1.1). The battery survived two runtime
restarts: the first worker's vs-v7 run was killed at 81/100 games
(55W-14L-12D, 7139.9 s) and the replacement worker's vs-v0 run at
40/100 (38W-2L-0D, 1223.0 s); both matches were re-run from scratch
with the identical pre-registered seeds and the flushed prefixes
verified game-for-game before proceeding (vs-v7: 81/81 identical,
0 mismatches; vs-v0: 40/40 identical, 0 mismatches — the batch-7
restart practice). Verification logs:
`research/iterations/battery-v21-chain.log` (stage 1) and
`battery-v21-chain2.log` (stage 2).

### 3.1 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

**v21 beats v7: 67W-20L-13D, win rate 0.7350, 95% CI
[0.6410, 0.8116]**, 8918.9 s. File:
`research/iterations/batch-v21-vs-v7.json` (seeds 100/200).
Per colour: as Black 40W-6L-4D (0.8400), as White 27W-14L-9D
(0.6300). Mean game length 67.63 plies (min 19, max 225).
Winners Black 54, White 33, draws 13. Loss classifier
(`ml/scripts/classify_batch_losses.py`): 20/20 losses via open
four, **0 missed blocks**, 107 threatened turns in
losses, **1 clean missed prevention (0.9%)** — defence is
solved; v21's losses are v7 out-attacking it in the open-four
race, not defensive blunders. **Clause 1 PASSES** (0.7350 >
0.50, CI lower bound 0.6410 > 0.50) — the second-best h2h vs v7
in the program (v18: 0.7750, CI [0.6839, 0.8458]; the two CIs
overlap substantially, so v21 and v18 are statistically
indistinguishable head-to-head). The colour splits tell the
story: as Black v21 is 0.84, IDENTICAL to v18's Black split —
the blend costs Black nothing — while as White v21 is 0.63,
essentially v16's/v18's White 0.64 and 11 points BELOW v19's
0.74. v21-as-White IS v19's leaf, so the White gain did NOT
reproduce at battery scale: the v19 White 0.74 looks like the
high end of its own noise band (n=50 per colour, ±~13 points at
95%), with RNG-stream divergence (stateful players across
alternating-colour games; v21's Black leaf changes the stream)
supplying the rest. The fast loop's 0.7250 (40 games) predicted
the battery's 0.7350 well.

### 3.2 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

**v21 beats v0: 95W-2L-3D, win rate 0.9650, 95% CI
[0.9085, 0.9871]**, 1988.4 s. File:
`research/iterations/batch-v21-vs-v0.json` (seeds 100/200).
Per colour: as Black 45W-2L-3D (0.9300), as White 50W-0L-0D
(1.0000) — a perfect White split. Mean game length 48.96 plies
(min 22, max 225). Winners Black 45, White 52, draws 3. Loss
classifier: 2/2 losses via open four, **0 missed blocks**, 7
threatened turns in losses, **1 clean missed prevention** —
14.3% of a 7-turn denominator, i.e. one turn in a two-loss
match; all-games count is 4 clean missed preventions in 2
games. No colour-split inversion of the v7 kind (v7 scored
LOWER vs v0 than vs v1): v21 scores higher against the weaker
v0 (0.9650) than against v7 (0.7350), as expected. Essentially
identical to v18's 0.9700 (96W-2L-2D) — one draw instead of one
win — and second only to v12's perfect 100-0 in the program.

### 3.3 Tactical suites — FINAL BATTERY

Full-dispatch driver (`ml/scripts/eval_tactical_full.py`,
`player.choose_move`, `--seed 100`, the battery protocol).
Old 20-position suite: **v21 20/20**. Hard suite (29
positions): **v21 26/29** (prevention 14/16, setup 7/7,
winning 1/2, vcf 4/4), failing exactly g12 (chose (7,3)),
g45 (chose (10,4)) and winning-quiet-setup-2 (chose (9,7)).
File: `research/iterations/batch-v21-tactical.json`. This is
byte-identical to the §1.3 pre-fast-loop diagnostic — no
patches were made, so the battery re-run reproduces it exactly,
which is also a determinism check on the protocol. All three
of v18's Black q-compression ties are fixed at seed 100 (g26
chose (8,4), g79 chose (7,9), v1bl-g2 chose (7,5)) — the blend
does what it was designed to do on Black's side — but v21
inherits v19's two White failures (g12 inert, g45 regressed)
plus the suite-list gap, for 26/29: one better than v18's
25/29, one worse than v19's 27/29, inside the pre-registered
26–28/29 range. **Clause 3 FAILS** (26/29 < v7's 29/29) — the
only failing clause.

### 3.4 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

vs random 20W-0L-0D (1.0000, CI [0.8389, 1.0000]; as Black
10W-0L-0D, as White 10W-0L-0D); vs heuristic 19W-0L-1D (0.9750,
CI [0.7995, 0.9974]; as Black 9W-0L-1D, as White 10W-0L-0D); vs
mcts 12W-8L-0D (0.6000, CI [0.3866, 0.7812]; as Black 8W-2L-0D,
as White 4W-6L-0D). **Elo 636.4, interval [240.3, 1032.5]** —
IDENTICAL to v18's estimate and interval, because the Elo is
driven by the vs-heuristic matchup and both versions score
0.975 there (19W-0L-1D). **Clause 2 PASSES** (636.4 > v7's
190.8 at identical game counts and opponents; even the lower
bound 240.3 clears v7's 190.8). Run time ~895 s; EVAL_JSON
saved to `research/iterations/ladder-v21.json` (full log:
`research/iterations/ladder-v21.log`). The script's tactical
probe line (20/20 old) was disregarded per the brief (it
bypasses the MCTS dispatch); tactical is §3.3's full-dispatch
20/20 + 26/29. v21 is the second version ever to beat the
ladder MCTS outright (v18: 14W-6L; v21: 12W-8L), with the same
familiar colour pattern (8-2 as Black, 4-6 as White): against
a searching opponent, v21-as-White still cannot create the
attack — the White gap that v19's battery split briefly seemed
to close.

### 3.5 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

**67W-27L-6D, win rate 0.7000, CI [0.6041, 0.7811]**, 3293.6 s
(secondary metric, not the bar). File:
`research/iterations/batch-v21-vs-v7-swap.json` (seeds 100/200).
As Black 37W-10L-3D (0.7700); as White 30W-17L-3D (0.6300).
second_colour Black 76 / White 24 — the same deterministic
split as every tempo-judge swap on these placements; v21's win
rate is 0.6974 when the second player chose Black and 0.7083
when White, so the choice itself does not drive the result.
Opener A (v21 opens) 0.6300 vs opener B (v7 opens) 0.7700 —
this mirrors the colour split, since when v21 opens, v7
chooses Black 38/50 times and v21 plays White. Winners Black
54, White 40, draws 6. Plies mean 58.40 (min 19, max 225).
Loss classifier: 27/27 losses via open four, **0 missed
blocks**, 128 threatened turns in losses, **1 clean missed
prevention (0.8%)**. v21's swap edge over v7 IS established
(CI lower 0.6041) — unlike v7's 0.57 over v1, whose CI
included 0.50 — and at 0.7000 it is the third-strongest swap
result in the program (v17 0.74, v19 0.73) and 7 points above
v18's 0.6300 (CIs overlap, so the ordering vs v18 is not
established, but the direction favours the blend under swap
rules: from a balanced 3-stone start, the blended Black leaf
holds its edge where v18's gave some back).

### 3.6 Pooled defence and transitivity

Pooled across the three 100-game battery matches: 49 losses,
**49/49 via open four, 0 missed blocks**, 3 clean missed
preventions in 242 threatened turns in losses (1.2%). Defence
remains solved. **Ordering is transitive — no
non-transitivity:** v21 beats v7 (0.7350 normal, 0.7000 swap)
and v0 (0.9650), and v7 beats v0 (0.6650). v21 also scores
higher against the weaker opponent, so there is no v7-style
colour-split inversion either.

## 4. Bar check (v21 vs v7)

| Clause | Requirement | v21 result | Verdict |
|---|---|---|---|
| 1. Head-to-head vs v7 | >= 100 games, win rate > 0.50, Wilson 95% CI lower bound > 0.50 | 100 games: 67W-20L-13D, 0.7350, CI [0.6410, 0.8116] | **PASS** |
| 2. Ladder Elo | strictly higher than v7's 190.8, identical games/opponents | 636.4 (interval [240.3, 1032.5]), vs v7's 190.8; lower bound 240.3 also clears 190.8 | **PASS** |
| 3. Tactical | pass rate >= v7's (20/20 old + 29/29 hard) | old 20/20 = v7's 20/20; hard 26/29 vs v7's 29/29 (fails g12, g45, winning-quiet-setup-2) | **FAIL** |
| 4. Test suites | ml pytest + npm test + tsc green | ml pytest 290 passed, 7 skipped, 15 xfailed, 0 failures; npm test 55/55; tsc clean; baseline.py unchanged vs master | **PASS** |

**Verdict: v21 is REJECTED — 3/4 clauses.** It passes clauses
1, 2 and 4 decisively and fails clause 3 alone (26/29 hard),
exactly like v18 (25/29) before it. Nothing is pushed to
master from this workstream; the branch
`research/batch-8-h1` holds the full record.

## 5. What this means

**The combination works as designed — on Black.** The blend
fixed all three of v18's Black q-compression visit-ties at the
battery protocol (g26, g79, v1bl-g2 all pass at seed 100, two
of them by genuine visit-allocation changes confirmed in the
§1.3 mechanism check, not RNG luck), and game strength holds
at v18's level: h2h vs v7 0.7350 vs v18's 0.7750 (CIs overlap
heavily), ladder Elo IDENTICAL at 636.4, vs v0 0.9650 vs
0.9700, swap 0.7000 vs 0.6300 (directionally better). v21-as-
Black is 0.84 vs v7, identical to v18's Black split — the
blend costs Black nothing and buys the three tie fixes.

**But the blend's White-side failures come along for the
ride, and they are the same mechanism in reverse.** v21
inherits v19's g12 (blend inert — terminal-position gradients
align with the terminal and the clamp pins everything) and
g45 (blend REGRESSES — softening the losing move (10,4) from q
−1.000 to −0.932 lets its 3× prior advantage earn the tying
visit) exactly, plus the winning-quiet-setup-2 suite-list gap
(the chosen (9,7) wins 3/3 under v7-vs-v7; the suite, not the
engine, is arguably wrong there — but the bar is the bar, and
changing the suite retroactively is a coordinator decision,
not a unilateral edit). 26/29 sits between v18's 25/29 and
v19's 27/29, inside the pre-registered 26–28/29 range. The
blend spreads q among losing lines, and spreading cuts both
ways: on Black it converts ties into wins; on White it creates
the tie that loses g45. A leaf-level value change cannot tell
"counterplay that saves the game" from "counterplay that loses
more slowly" — that was iter-19's verdict on the blend as a
general mechanism, and v21 confirms it from the other side:
the blend is now doubly bounded. **The blend as a mechanism
is CLOSED** — v21 was the last untested combination, and it
lands exactly where the two parents' failures predicted.

**The White split is a noise lesson.** v21-as-White IS v19's
leaf, yet its White split vs v7 came in at 0.63 against v19's
battery 0.74. Either v19's 0.74 was the high end of its own
noise band, or RNG-stream divergence moved it, or both — at
n=50 per colour (±~13 points at 95%) the two numbers are not
distinguishable. Do not over-read n=50 colour splits in either
direction; the headline 100-game numbers are the verdict, and
there v21 (0.7350) and v18 (0.7750) are also not
distinguishable. The honest summary of v21 vs v18: a lateral
move on the bar (both fail clause 3 alone), one tactical
position better (26 vs 25), game strength statistically
identical, swap directionally better.

**What changed because of a specific loss: nothing was
patched.** The fast loop (§2) found no patchable weakness
class in 40 games across 2 iterations: the three losses were
all White games with no verified losing move (one forced
loss/distributed drift with refuted oracle nominations, one
distributed drift with no nominated move at all, one noisy
beyond-horizon twin choice where the blend changed nothing).
v21-as-batteried IS v21-as-pre-registered. The twin-choice
weakness class (fl2-g7-ply9) is recorded as a regression xfail
for future work, not dropped.

**What remains open.** The q-compression visit-tie problem is
now H3's (v23, the q-aware tie-break) to solve — it is the
only open route to 29/29 for this lineage, since the leaf
cannot discriminate among losing lines by value alone and the
root rule was CLOSED only for q-based decisions, not for a
tie-break that uses q as a secondary signal. Separately, the
winning-quiet-setup-2 suite-list gap deserves a coordinator
decision: if (9,7) genuinely wins, the suite should list it,
and v21/v18/v19 would each gain a point — but that decision
must be made once, for the program, not per version. Finally,
the ladder MCTS remains the honest measuring stick for White:
v21-as-White loses 4-6 to it, the same gap every version since
v6 has shown, and no leaf change in batches 5–8 has closed it.

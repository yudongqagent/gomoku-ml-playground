# Iteration 53 — v53: 3-ply rollout leaf on v18/v27 (batch 13, H3)

**Status: IN PROGRESS — re-scoped to the 3-ply rollout leaf.**
v53 was originally specified as a 2-ply rollout leaf; §§0–5
record that work and its finding (COMPLETE: the specified 2-ply
leaf IS v27's leaf — NOT A NEW VERSION, battery vacated per the
pre-registered §1.3 rule). Per §5 option 1 the coordinator
re-scoped H3 to the only unpriced rung of the depth ladder: the
3-ply rollout leaf (3 rollout plies + gradient, ending with the
opponent to move, gradient negated). §6 pre-registers the
3-ply spec, the parity handling, the cost budget, and the
mechanism check BEFORE any eval of the 3-ply leaf; §§7+
record the mechanism check, the fast loop, the battery, and
the verdict. §§0–5 stay as written — the 2-ply identity
finding is part of this version's history, and the 2-ply
differential tests stay as regression tests pinning it. This is
the H3 workstream of batch 13 (H1 is v51, gradient-preserving
backup; H2 is v52, threat-space search); work happens on branch
`research/batch-13-h3` in the `gomoku-b13-h3` worktree,
branched from `research/batch-13` @ `ca0a005` (which includes
the batch-13 plan). Comparison target is v27 (the batch-13 bar
reference: ladder Elo 636.4, on master).

**§0 records a premise failure found by code-reading BEFORE any
eval: v18/v27's leaf is ALREADY a 2-ply rollout leaf.** The
batch-13 plan (§H3) and this workstream's tasking both describe
v18's leaf as "1-ply (one random move, then evaluate)". That
description is wrong on both counts — the leaf plays exactly 2
rollout plies under a deterministic win/block/v1-argmax policy,
not 1 random move. §1 therefore pre-registers v53 exactly as
specified (a fresh 2-ply rollout-leaf implementation on the v27
base) together with the DECISION RULE the verdict turns on: a
differential mechanism check against v27's leaf. If the
fresh-from-spec implementation agrees with v27's leaf everywhere
it is sampled, v53 IS v27 behaviourally, the H3 premise is
refuted at the mechanism level, and the battery is vacated — a
v27-vs-v27 battery would measure 0.50 by construction and prove
nothing. If it differs anywhere, the difference is characterised
and the full battery runs as tasked.

## 0. Premise check (code read, before any eval)

### 0.1 v18/v27's leaf is a 2-ply rollout leaf

Four independent sources agree; all were read before any eval:

1. **The code.** `ml/gomoku_ml/mcts_heuristic.py`:
   `_ROLLOUT_PLIES = 2`, and `_rollout_leaf_value(board, player,
   blend=False)` "roll[s] out exactly 2 plies (opponent reply +
   our reply, `_rollout_move`) and evaluate[s] the symmetric v1
   gradient at the resulting position". `_rollout_move` is a
   DETERMINISTIC policy: win if possible; block a single
   opponent winning move (forced); open-four priority; else the
   argmax over empty cells of the player's v1 total score (ties
   by lowest flat index). There is no random move anywhere in
   the leaf.
2. **The dispatch.** `HeuristicMCTS._rollout_v1_both_leaf_value`
   (v18's leaf, both colours) returns `_rollout_leaf_value`
   directly; v27 uses `leaf_mode="rollout2_v1_both"` — the mode
   name itself says 2-ply rollout.
3. **The registry.** `ml/gomoku_ml/scoring.py`:
   "v18 (iteration 18, batch 7): H1 — v16's 2-PLY ROLLOUT leaf
   applied as Black too", and v27's entry: "Everything else is
   v23 unchanged: 100 sims, c_puct 1.5, v1 priors, rollout leaf
   both colours."
4. **The iteration logs.** `research/iterations/iter-18.md`:
   "2 plies, not 1 and not 4 (v16's measurements, iter-16 §1.1:
   the save's value comes from OUR next move, so the rollout
   must include it; 4 plies returns false terminal losses for 3
   of 10 saving moves)." `research/VERSIONS.md`'s v29 row:
   "v18's 2-ply rollout leaf in both colours".

The "1-ply" in the plan/tasking conflates two different depth
countings. The ML track's label-depth naming counts the plies
BEFORE the teacher leaf: `_rollout2_leaf_value` (v37's "2-ply"
label) plays 2 rollout plies and then calls `_rollout_leaf_value`
— which itself plays 2 more plies — for an effective horizon of
4 plies (its docstring says so: "Effective horizon: 2 plies of
rollout-policy play + the 2 plies inside `_rollout_leaf_value` =
4 plies"). Likewise `_rollout3_leaf_value` (v44) is 5 plies
effective. The ML finding "2-ply labels are the strongest
labelling" is a finding about a 4-ply-effective-horizon teacher,
not about a 2-ply search leaf — and the 2-ply search leaf is
already v18/v27.

### 0.2 The neighbouring depths are already priced

* **1 ply:** measured and rejected by v16 (iter-16 §1.1, quoted
  in iter-18.md): a 1-ply (opponent-reply-only) leaf measures
  robustness to the opponent's best attack, which favours the
  block — the save's value comes from OUR next move, so the
  rollout must include it.
* **4 plies:** measured and rejected TWICE. v16 (iter-16 §1.1):
  the crude attack-argmax policy blunders at that depth,
  returning false terminal losses (-1.000) for 3 of the 10
  saving moves. v40 (iter-40 §1.1e, `test_v40_rollout4.py`,
  monkeypatched `_ROLLOUT_PLIES`): fixes 0/4 diagnosed
  instances, moves two of them the WRONG way, and HALLUCINATES
  — g61's safe (4,5) gets a +1.0000 ("win") under 4 plies
  although the move loses 3/3 in playouts — "the weak argmax
  policy drifts further from true play over more plies". Cost
  ~2x/leaf.
* **3 plies (pure rollout leaf, 3 plies + gradient):** genuinely
  untested as a search leaf — the only unpriced rollout depth.
  It is NOT what this workstream was tasked to build, so it is
  recorded here as a re-scope candidate for the batch
  coordinator (§5), not built here. Note the ML track's
  parity-shift finding (ML batch 5: odd-horizon labels are
  systematically optimistic, 3-ply mean +0.1094 vs 2-ply ~-0.06)
  as a prior against it.

### 0.3 What "2-ply rollout leaf on v18/v27" can mean

* **Reading 1 (literal):** a search leaf that plays 2 rollout
  moves then evaluates — that is `_rollout_leaf_value`, i.e.
  v27 itself. This is the reading this workstream implements
  and tests (§1).
* **Reading 2 (ML naming):** `_rollout2_leaf_value` as a search
  leaf — effective horizon 4 plies — the rollout depth refuted
  twice (§0.2). The tasking itself rules this out ("2-ply is
  between 1-ply (works) and 4-ply (hallucinates)") and states
  `_rollout2_leaf_value` "is a LABELLING function for
  distillation, not a search leaf".
* **Reading 3 (random policy):** the tasking's "one random move,
  then evaluate" could be read as prescribing a RANDOM rollout
  policy. There is no hypothesis in the plan for random
  rollouts; the program's evidence runs the other way (v16's
  fast-loop patch replaced a WEAKER rollout policy with v1's
  because the weak policy let quiet moves borrow forcing
  values; the 4-ply hallucination is the weak policy drifting).
  A random policy is strictly weaker than the argmax policy
  and would hallucinate more, not less. Not built; recorded
  for the coordinator.

## 1. Design and pre-registration

**Comparison target: v27** (primary, the batch-13 reference:
ladder Elo 636.4). The bar's clause 1 is head-to-head vs v27
(normal rules, 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50), clause 2 is ladder Elo
strictly higher than v27's 636.4 (40 games per opponent per leg,
batch 11's re-basing), clause 3 is tactical pass rate >= v27's
on the CORRECTED suite (old suite 20/20, hard suite 29/29),
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v53 vs v0 (100
games) and a 100-game swap match vs v27 under the tempo judge.
**Strength champion:** v18 (h2h vs v7 0.775, ladder Elo 636.4).
Under champion rule v2 a bar-passer plays a 100-game direct
challenge against v18; **only run the challenge match for a
bar-passer.**

**The improvement, as specified: v53 = v27 + a 2-ply rollout
leaf, implemented fresh from the spec below.** Everything else
is v27 unchanged: 100 sims, c_puct 1.5, v1 priors over all
candidates, win/block scans before search, most-visited root
child, threshold τ=0.05 on the q gap. The leaf is the ONLY
change from v27 — and §0's finding is that the specified leaf
is v27's leaf, so the implementation is expected to coincide
with it. The implementation is written FRESH from the spec in
§1.1 (not by aliasing `_rollout_leaf_value`), so that the
differential check in §1.3 tests the SPEC against v27's leaf
rather than testing an alias against itself.

### 1.1 The leaf, exactly (pre-registered)

At a leaf, with `player` to move on `board`:

1. **Terminal check at the leaf start** (v7's convention, as in
   `_rollout_leaf_value`): if `_terminal_leaf_value(board,
   player)` is non-zero (a win/loss, or a must-block at -0.25),
   return it. The must-block at the START is a fact about the
   real position and keeps its exact -0.25.
2. **Rollout ply 1** (`player` to move): `m1, term1 =
   _rollout_move(board, player)`. If `term1` is not None, the
   game ends on this ply: return `term1` (from `player`'s
   perspective).
3. **Rollout ply 2** (the opponent to move): `m2, term2 =
   _rollout_move(board+m1, -player)`. If `term2` is not None,
   return `-term2` (`term2` is from the opponent's perspective;
   negated to `player`'s).
4. **Rollout end** (`player` to move again on `board+m1+m2`):
   only wins/losses are terminal — if `player` has a winning
   move, return +1.0; if the opponent has >= 2 winning moves,
   return -1.0. A must-block here is NOT an exact -0.25: it is
   gradient-evaluated (iter-16 §2.1: a rollout-end must-block
   is a prediction under the rollout policy, not a fact).
   Otherwise return the symmetric v1 gradient at the resulting
   position for `player`: `0.2 * tanh((breadth_v1(ours) -
   breadth_v1(theirs)) / 7.5)`, cap ±0.2. The board is not
   mutated (the rollout plays on copies).

**Rollout policy** (held fixed — the depth is the variable, the
policy is not): `_rollout_move` exactly as it stands — win if
possible; block a single opponent winning move (forced); an own
open-four creation (own v1 attack >= OPEN_FOUR) wins the argmax;
else the argmax over empty cells of the player's v1 total score
(own v1 attack + opponent's v1 attack at the square), ties by
lowest flat index, deterministic. This is the teacher's own
policy — the same policy the ML label-depth finding used — and
reusing it isolates depth as the only change.

**What is NOT changed:** no blend (v19's blend was rejected and
is not part of v27); no VCF check at the leaf or root (v35
CLOSED VCF-prevention for the v27 family); no tie-break, prior,
sims, or root-rule change (all CLOSED); `baseline.py` stays
FROZEN.

### 1.2 Cost budget (pre-registered)

The tasking budgets "roughly double the leaf cost" for a 2-ply
rollout — but that budget was written against the (false)
1-ply premise. Against the TRUE base (v27's leaf is already
2-ply), the specified leaf performs the same work as v27's, so
the expected cost ratio is 1.0x. **Pre-registered ceiling:
mean leaf cost ≤ 1.5x v27's on the bench** (the v40 precedent:
~1.45x was accepted as battery-feasible). The cost is measured
FIRST on a bench of positions before any eval, as tasked: the
bench is 200 quiet midgame positions (sampled from committed
v27 battery games + random playouts), each leaf timed
individually, v53 vs v27 on the same positions, reported as
mean/median/p90 per leaf and the ratio. If the ratio exceeds
1.5x, the implementation is re-examined before any eval — a
ratio above 1.0x on identical work would itself be evidence of
an implementation difference.

### 1.3 Decision rule (pre-registered — the verdict turns on this)

The mechanism check is a DIFFERENTIAL test of v53 against v27,
run after the cost bench and before any game eval:

* **(a) Leaf-value identity.** v53's leaf value vs
  `_rollout_leaf_value` on >= 2000 sampled positions: every
  position of 10 committed v27 battery games (both colours to
  move), all tactical-suite positions (old + hard), and random
  playout positions to reach the count — including terminal,
  must-block, and creator-rich positions, not just quiet ones.
  Pass = exact equality (float-identical) on ALL sampled
  positions.
* **(b) Move-choice identity.** Full 100-sim searches, v53 vs
  v27 engines at identical seeds, on >= 30 positions (tactical
  suite + sampled midgames, both colours). Pass = the same
  chosen move on ALL sampled searches. (Same leaf + same
  search + same RNG consumption ⇒ same move; any difference
  here is a difference somewhere in the chain.)
* **(c) Cost identity.** The §1.2 bench ratio within noise of
  1.0x (and below the 1.5x ceiling).

**If (a), (b), and (c) all pass, v53 is behaviourally identical
to v27.** The H3 premise is then refuted at the mechanism level
and the battery is VACATED: head-to-head vs v27, vs v0, swap,
and the ladder would all re-measure v27 against itself, and
clause 1 (win rate > 0.50 with CI lower > 0.50 against v27)
would be decided by colour assignment and RNG, not by any
property of v53. Running 300+ games to confirm a proven
identity would burn the VM budget for a number that means
nothing. The verdict is then recorded as **NOT A NEW VERSION**
(not a bar rejection — v53 never reaches the bar, because
there is no version to evaluate), with the re-scope options in
§5 for the coordinator.

**If any of (a), (b), or (c) fails, the difference is
characterised** (which positions, which direction, what
mechanism) and the full battery runs as tasked — the fast
10-game loop first (vs v27 AND v0), then the 100-game legs,
swap, ladder, and tactical — with the difference as the
version's actual content.

**Tactical as a wiring check:** v53's tactical run (20/20 +
29/29 expected under identity) doubles as the end-to-end wiring
check that the registered engine runs through the eval harness.
It is run regardless of the differential outcome, before any
game eval, and v27 is re-verified first on the corrected suite.

**Tactical as a wiring check:** v53's tactical run (20/20 +
29/29 expected under identity) doubles as the end-to-end wiring
check that the registered engine runs through the eval harness.
It is run regardless of the differential outcome, before any
game eval, and v27 is re-verified first on the corrected suite.

## 2. The mechanism check (results)

All three pre-registered checks pass. v53 is behaviourally
identical to v27.

### 2.1 (a) Leaf-value identity — 2,477/2,477 exact

`tests/test_regression_v53.py::test_v53_leaf_value_identical_to_v27_leaf`:
`_rollout2ply_search_leaf_value` (fresh from the §1.1 spec) vs
`_rollout_leaf_value` (v27's leaf), exact float equality, on
2,477 sampled positions:

| Source | Positions |
|---|---|
| Every ply of the first 10 committed v27-vs-v7 battery games (`batch-v27-vs-v7.json`) | 779 |
| All tactical-suite positions (old 20 + hard 29) | 49 |
| Seeded random playouts (30 games, seed 53, every 2nd ply) | 1,649 |
| **Total** | **2,477** |

Value-class coverage of the sample (v27 leaf): 934 exact +1,
88 exact −1, 187 must-block −0.25, 1,268 gradient values.
2,079 Black-to-move, 398 White-to-move. **Mismatches: 0.**
The spec's natural implementation coincides with v27's leaf on
every sampled position, including terminal, must-block, and
creator-rich positions — not just quiet ones.

### 2.2 (b) Move-choice identity — 30/30 searches

`test_v53_move_choice_identical_to_v27`: full 100-sim searches,
v53 vs v27 engines at identical seeds (100–129), on 30
positions — 10 old tactical, 10 hard tactical, 10 sampled
midgames (≥10 stones; 21 Black-to-move, 9 White-to-move).
**Same chosen move on all 30.** Same leaf + same search + same
RNG consumption ⇒ same move; the chain is identical end to end.

### 2.3 (c) Cost — ratio ~1.0x, ceiling 1.5x

`test_v53_leaf_cost_within_ceiling` (200 quiet midgame
positions, min-of-3 per position): v53 1.4264 s vs v27 1.4218 s
total, **ratio 1.003**. A separate bench run with per-leaf
stats: v53 mean 7.582 ms / median 7.098 ms / p90 9.556 ms vs
v27 mean 8.397 ms / median 7.655 ms / p90 10.960 ms, ratio of
means 0.903. Both runs are within VM noise of 1.0x — the
specified leaf performs the same work as v27's, as §1.2
predicted — and far below the pre-registered 1.5x ceiling.

### 2.4 Tactical wiring check — identical moves on all 49

v27 re-verified first on the corrected suite at `--seed 100`:
**20/20 + 29/29** (`research/iterations/tactical-v27-b13.json`).
v53 at `--seed 100`: **20/20 + 29/29**
(`research/iterations/tactical-v53-b13.json`). v53 chooses v27's move on **all 49
positions** — the registered engine runs through the eval
harness and behaves as v27 throughout, including on
prev-v1bl-g2 (the v40 visit-allocation failure) and
prev-v1v0s-g13 (the v47 q-collapse position).

## 3. The battery is vacated (per the pre-registered rule)

§1.3's rule fires: (a), (b), and (c) all pass, so v53 is
behaviourally identical to v27 and the battery is vacated. No
100-game legs, no swap leg, and no ladder were run — they would
re-measure v27 against itself:

* vs v27, 100 games: both engines play the same move in every
  position at the same seed, so the leg's outcome would be
  decided by colour assignment and the per-game seeding, not by
  any property of v53. A win rate of ~0.50 with a CI around it
  would be a measurement of the harness's colour balance, not
  of a version.
* vs v0 and swap vs v27: the same — v53's numbers would be
  v27's numbers (0.9650 vs v0; 0.7750 swap vs v7 is v27's, not
  vs-v27), re-measured with different seeds.
* Ladder: v53's Elo would be v27's 636.4 up to seed noise and
  draw-count artefacts (batch 12's finding: a single draw swings
  the top Elo by 123 points). A "pass" or "fail" on clause 2
  would be draw-count luck.

Running 300+ games (~3–4 hours of VM time at v27's pace) to
confirm a proven identity would burn the batch's budget for
numbers that mean nothing. The fast 10-game loop was likewise
not run: there is no version to loop on, and no losing move of
v53's could differ from v27's.

## 4. Verdict: NOT A NEW VERSION

v53 as specified is not a new version — it is v27 under a
different name. This is **not a bar rejection**: the 4-clause
bar evaluates a version against v27, and there is no version
here to evaluate. Recording it as REJECTED would misstate the
finding (a rejection says "this change was tried and lost";
here there was no change). Per-clause status:

1. **Head-to-head vs v27:** VACATED — v53 ≡ v27, so the leg
   would measure colour balance, not strength.
2. **Ladder Elo > 636.4:** VACATED — v53's Elo is v27's 636.4
   up to seed noise and draw-count artefacts.
3. **Tactical ≥ 20/20 + 29/29:** HOLDS TRIVIALLY — 20/20 +
   29/29 at seed 100, identical moves to v27 on all 49
   positions (§2.4).
4. **All test suites green:** HOLDS — ml pytest 461 passed, 1
   skipped, 58 xfailed, 0 failures (12m07s); npm test 55/55;
   tsc clean. Includes the 4 new v53 differential tests, all
   passing.

**What changed:** nothing in behaviour. v53's leaf
(`_rollout2ply_search_leaf_value`, `leaf_mode="rollout2ply_v53"`,
`ENGINE_VERSIONS[53]`) is a fresh, independent implementation
of the pre-registered 2-ply rollout spec that coincides with
v27's leaf everywhere it was sampled (2,477 leaf positions,
30 searches, 49 tactical positions). It is kept on this branch
as the durable record of the finding — the differential tests
pin the identity — and must NOT be merged to master as a new
version.

**Champions:** unchanged. v27 stays the bar reference (Elo
636.4); v18 stays the strength champion. No direct-challenge
match (only for bar-passers). No change to the daily
cross-track pairing (v18 vs v44).

## 5. What the coordinator should do with H3

The H3 slot's premise — "v18's leaf is 1-ply; 2-ply is
untested" — is refuted. The depth ladder on the heuristic
track is now fully priced except one rung:

| Rollout depth (search leaf) | Status |
|---|---|
| 1 ply | Rejected (v16: favours the block; the save's value needs our reply) |
| 2 plies | **v18/v27 — the current best** |
| 3 plies | **Untested as a search leaf** |
| 4 plies | Rejected twice (v16: false −1.000 on 3/10 saves; v40: hallucinates a win on a losing safe move) |

Options for the H3 slot, in the order this workstream would
rank them:

1. **Re-scope H3 to the 3-ply rollout leaf** (3 rollout plies +
   gradient, ending with the opponent to move, gradient
   negated). It is the only unpriced rung, and it is the true
   "between 2 (works) and 4 (hallucinates)" candidate the plan
   meant to test. Prior against it: the ML track's parity-shift
   finding (odd-horizon labels systematically optimistic, 3-ply
   mean +0.1094) and the 4-ply lesson that the weak argmax
   policy drifts with depth — 3 plies may already drift. Needs
   its own pre-registration (the parity handling at the rollout
   end is a design decision, not a detail).
2. **Close H3 and run batch 13 as two versions** (v51, v52).
   The H3 hypothesis was empty; there is no obligation to fill
   the slot.
3. **Do NOT re-scope H3 to `_rollout2_leaf_value` as a search
   leaf.** That is the 4-ply effective horizon, refuted twice
   (§0.2). The ML "2-ply" naming is what caused this
   workstream's premise failure; reusing it as a search leaf
   would repeat the confusion with a known-bad depth.

**Durable lesson (for the plan and future taskings):** the ML
track's label-depth naming counts plies BEFORE the teacher
leaf, so an "N-ply label" is an (N+2)-ply effective horizon
(v37's "2-ply" = 4 plies; v44's "3-ply" = 5 plies). Any future
tasking that ports a label-depth finding to the search leaf
must state the depth in ONE counting — total rollout plies —
and check it against `_ROLLOUT_PLIES` and the priced depth
ladder above before the plan is written. The code read that
catches this takes minutes; the battery it saves takes hours.


## 6. Re-scope: v53 is the 3-ply rollout leaf (pre-registration)

**Committed BEFORE any eval of the 3-ply leaf.** §6 was
written and committed before the 3-ply implementation existed,
so the spec, the parity handling, the cost budget, and the
mechanism check's pass/fail criteria are fixed before any
number from the 3-ply leaf is seen.

### 6.0 Why 3 plies

The depth ladder on the heuristic track (§0.2, §5) is fully
priced except one rung, in TOTAL ROLLOUT PLIES (the only
counting used in this section — the ML track's label-depth
naming counts plies BEFORE the teacher leaf, so an "N-ply
label" is an (N+2)-ply effective horizon; v37's "2-ply" = 4
plies, v44's "3-ply" = 5 plies — neither is this leaf):

| Rollout depth (search leaf) | Status |
|---|---|
| 1 ply | Rejected (v16: favours the block; the save's value needs our reply) |
| 2 plies | **v18/v27 — the current best** |
| 3 plies | **Untested as a search leaf — this version** |
| 4 plies | Rejected twice (v16: false −1.000 on 3/10 saves; v40: hallucinates a win on a losing safe move) |

v53 (re-scoped) = v27 + a 3-ply rollout leaf: the true
"between 2 (works) and 4 (hallucinates)" candidate the plan
meant to test. **Prior against it, recorded honestly:** the ML
track's parity-shift finding (ML batch 5: odd-horizon labels
are systematically optimistic — 3-ply labels mean +0.1094 vs
2-ply ~−0.06) and the 4-ply lesson that the weak argmax policy
drifts with depth — 3 plies may already drift. The mechanism
check (§6.3a) tests directly whether the parity-shift optimism
appears in the search leaf.

**Not built:** `_rollout2_leaf_value` as a search leaf — that
is the refuted 4-ply effective horizon (§0.3 reading 2, §5
option 3).

### 6.1 The leaf, exactly (pre-registered)

At a leaf, with `player` to move on `board`:

1. **Terminal check at the leaf start** (v7's convention, as in
   `_rollout_leaf_value` and §1.1 step 1): if
   `_terminal_leaf_value(board, player)` is non-zero (a
   win/loss, or a must-block at −0.25), return it. The
   must-block at the START is a fact about the real position
   and keeps its exact −0.25.
2. **Rollout ply 1** (`player` to move): `m1, term1 =
   _rollout_move(board, player)`. If `term1` is not None, the
   game ends on this ply: return `term1` (from `player`'s
   perspective).
3. **Rollout ply 2** (the opponent to move): `m2, term2 =
   _rollout_move(board+m1, -player)`. If `term2` is not None,
   return `−term2` (`term2` is from the opponent's perspective;
   negated to `player`'s).
4. **Rollout ply 3** (`player` to move again): `m3, term3 =
   _rollout_move(board+m1+m2, player)`. If `term3` is not None,
   return `term3` (from `player`'s perspective — ply 3 is
   `player`'s ply). This is `_rollout_leaf_value`'s in-rollout
   sign convention extended by one ply: `term` if the ply index
   is even (0, 2), `−term` if odd (1).
5. **Rollout end** (the OPPONENT to move on `board+m1+m2+m3`):
   only wins/losses are terminal — if the opponent (side to
   move) has a winning move, return −1.0; if `player` has >= 2
   winning moves, return +1.0. A must-block here (`player`
   holds exactly one winning move, which the opponent to move
   must block) is NOT an exact −0.25: it is gradient-evaluated
   (iter-16 §2.1: a rollout-end must-block is a prediction
   under the rollout policy, not a fact — the same convention
   as the 2-ply leaf's rollout end). Otherwise return the
   symmetric v1 gradient at the resulting position, computed
   from the side-to-move's (opponent's) perspective and
   NEGATED to `player`'s perspective:
   `−_attack_gradient_v1(b3, −player)`. The board is not
   mutated (the rollout plays on copies).

**Parity handling at the rollout end (the design decision,
stated explicitly, not a detail):** after an ODD number of
rollout plies the side to move at the rollout end is the
opponent, so the rollout-end value must be computed from the
opponent's perspective and negated to the leaf player's
perspective — for BOTH the terminal checks and the gradient.
This is the mirror image of the 2-ply leaf, where the rollout
end has the leaf player to move and no negation is needed. For
the gradient the negation is transparent by construction:
`_attack_gradient_v1` is antisymmetric under colour swap, so
`−_attack_gradient_v1(b3, −player)` EQUALS
`_attack_gradient_v1(b3, player)`; the implementation uses the
explicit negated form so the parity handling is visible in the
code, and a test asserts the two forms agree on the sampled
positions. For the terminal checks the negation is NOT
transparent — it flips which side's winning move is +1 — so
the checks in step 5 are written out in full (opponent winning
move → −1.0; our double threat → +1.0) and tested with
positions that exercise each branch.

**Rollout policy** (held fixed — the depth is the variable, the
policy is not): `_rollout_move` exactly as it stands — win if
possible; block a single opponent winning move (forced); an own
open-four creation (own v1 attack >= OPEN_FOUR) wins the argmax;
else the argmax over empty cells of the player's v1 total score
(own v1 attack + opponent's v1 attack at the square), ties by
lowest flat index, deterministic. This is the teacher's own
policy — the same policy the ML label-depth finding used — and
reusing it isolates depth as the only change.

**What is NOT changed:** no blend (v19's blend was rejected and
is not part of v27); no VCF check at the leaf or root (v35
CLOSED VCF-prevention for the v27 family); no tie-break, prior,
sims, or root-rule change (all CLOSED); `baseline.py` stays
FROZEN. Everything else is v27 unchanged: 100 sims, c_puct 1.5,
v1 priors over all candidates, win/block scans before search,
most-visited root child, threshold τ=0.05 on the q gap.

### 6.2 Cost budget (pre-registered)

The 3-ply leaf does 3 rollout plies instead of 2 (each ply a
`_winning_moves` scan pair plus a possible `attack_v1_both`
call) plus the rollout-end terminal checks, so the expected
cost ratio vs v27's 2-ply leaf is **~1.5x**. **Pre-registered
ceiling: mean leaf cost ≤ 2.0x v27's on the bench** (200 quiet
midgame positions — the same bench as §1.2/§2.3 — each leaf
timed individually, min-of-3 per position to cut VM noise).
The ceiling sits above the expected 1.5x with headroom for VM
noise, and at the v40 precedent's level (4-ply ~2x/leaf was
battery-feasible, though the version was rejected on strength).
The cost is measured FIRST on the bench, before any eval: if
the ratio exceeds 2.0x, the implementation is re-examined
before any eval.

### 6.3 Mechanism check (pre-registered — criteria stated BEFORE running)

Same >= 2000-position sample as §1.3 (every ply of the first
10 committed v27-vs-v7 battery games, all 49 tactical-suite
positions, seeded random playouts to reach the count — seed 53,
both colours to move, terminal/must-block/creator-rich
positions included, not just quiet ones).

* **(a) Leaf-value distribution vs the 2-ply leaf.** Report for
  EACH leaf (3-ply and 2-ply, same positions): mean, fraction
  positive, fraction exact +1, fraction exact −1, fraction
  must-block −0.25, fraction gradient. The parity-shift
  question: does the ML track's odd-horizon optimism appear in
  the search leaf? **Pre-registered interpretation:** if
  mean(3-ply) − mean(2-ply) > +0.05 (half the ML track's
  +0.1094 shift, same direction), the optimism is PRESENT in
  the search leaf — recorded as a finding against the
  version's prior, NOT a battery-vacating condition (the
  battery is the verdict). If the difference is ≤ +0.05, the
  parity shift does not carry over to the search leaf at this
  depth.
  * **Sanity gate (BLOCKS eval):** every 3-ply leaf value
    finite and in [−1, 1]. Any NaN, inf, or out-of-range value
    is an implementation bug — fix before any eval.
  * **Difference gate (BLOCKS eval):** the 3-ply leaf must
    DIFFER from the 2-ply leaf on > 5% of sampled positions. A
    3-ply rollout cannot coincide with a 2-ply rollout on
    forcing positions (the 3rd ply can win) or on quiet
    positions (the gradient is evaluated at a different board).
    Agreement everywhere would mean the implementation is not
    actually rolling out 3 plies — re-examine before any eval.
    (This is the anti-repeat of the §2 identity finding:
    v53-3ply must be a genuinely different version.)
* **(b) Move-choice agreement vs v27 on the tactical suite
  positions.** Full 100-sim searches, v53 (3-ply) vs v27 at
  identical seeds (100–148), on all 49 tactical positions (old
  20 + hard 29). Report the agreement count. **Pre-registered
  interpretation:** ≥45/49 = tactical behaviour preserved (the
  leaf change does not disturb forcing positions — a good sign
  for clause 3); <45/49 = tactical behaviour changed — clause
  3 at risk, watch the suite run. This is characterisation,
  not a gate — the clause-3 tactical run is the verdict on
  tactical behaviour.
* **(c) Cost** per §6.2 (ceiling 2.0x, expected ~1.5x).

**Tactical as a wiring check:** as in §1.3 — v27 re-verified
first on the corrected suite at `--seed 100`, then v53 (3-ply)
at `--seed 100`, before any game eval. The v53 tactical run
doubles as clause 3.

### 6.4 Eval seeds and the bar (pre-registered)

* Battery h2h vs v27: v53=100 / v27=200; vs v0: v53=100 /
  v0=200. Swap vs v27: 100 games, balanced placement,
  placement_seed 11, tempo judge, seeds 100/200.
* Ladder: seed 100, 40 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 53 --ladder-games 40
  --out research/iterations/ladder-v53.json` — the auto-resume
  driver; every ladder JSON is a committed artifact).
* Tactical: `eval_tactical_full.py --version 53 --suite both
  --seed 100`. v27's tactical is re-verified first on the
  corrected suite.
* Challenge match (ONLY if v53 passes all 4 clauses): 100
  games vs v18, seeds 100/200, normal rules.
* Fast-loop dev seeds: 1053/2053 (vs v0), 1054/2054 (vs v27)
  — distinct from the battery's 100/200 and from batch-11/12
  loop seeds.
* All match drivers auto-resume from partial `--out` files —
  if a run is killed, relaunch the SAME command. Per-game
  seeding (`seed + game_index`) with a uniqueness audit on
  every leg; report distinct-game counts and de-dup rates for
  every leg. Recompute every aggregate from the raw per-game
  records (A win = winner_is_a is True; draws = winner ==
  'draw') and recompute the Wilson 95% CIs; report the
  recomputed numbers, not just the script summaries.

**The bar (reference v27, Elo 636.4):** (1) h2h vs v27 ≥100
games, win rate > 0.50 with Wilson 95% CI lower bound > 0.50;
(2) ladder Elo > 636.4 — read with the leg composition
alongside (batch 12 finding: Elo is draw-count sensitive at
the top; a 123-point swing from a single draw; do NOT claim a
"highest Elo ever" from a single draw; if a clause-2 pass
rests on a draw-count difference, say so explicitly); (3)
tactical ≥ 20/20 + 29/29; (4) all test suites green. Fail any
clause = REJECTED.

## 7. The mechanism check (results)

All three pre-registered checks (§6.3) ran; both gates pass.
v53-3ply is a genuinely different version from v27, the
parity-shift optimism IS present in the search leaf (just above
the pre-registered threshold), and tactical behaviour is
preserved on 48/49 positions.

### 7.1 (a) Leaf-value distribution — parity-shift optimism present

`tests/test_regression_v53.py::test_v53_3ply_leaf_distribution_vs_2ply`,
same 2,477-position sample as §2.1 (779 battery + 49 tactical +
1,649 random playouts):

| | 3-ply leaf | 2-ply leaf (v27) |
|---|---|---|
| Mean | **0.3710** | 0.3150 |
| Fraction positive | 0.7247 | 0.5632 |
| Fraction exact +1 | 0.3912 (969) | 0.3771 (934) |
| Fraction exact −1 | 0.0355 (88) | 0.0355 (88) |
| Fraction must-block −0.25 | 0.0755 (187) | 0.0755 (187) |
| Fraction gradient | 0.4437 | 0.4449 |
| Positions differing | **1,159/2,477 (46.79%)** | — |

**Mean difference (3-ply minus 2-ply): +0.0561** — above the
pre-registered +0.05 threshold, so per §6.3a the parity-shift
optimism IS PRESENT in the search leaf, in the same direction
as the ML track's label finding (3-ply labels mean +0.1094),
at about half the magnitude. The mechanism is visible in the
table: the 3rd ply is OUR move (the argmax reply), so the
gradient at b3 sees the position after our best reply rather
than after the opponent's — 16 points more of the sample is
positive, and the extra ply finds 35 more exact wins (969 vs
934). The −1 and −0.25 fractions are IDENTICAL — the extra ply
adds wins and optimism, never new losses or must-blocks at the
leaf start (both are leaf-start facts, unchanged by depth).

**Sanity gate: PASS** — every 3-ply value finite and in
[−1, 1]. **Difference gate: PASS** — the 3-ply leaf differs
from the 2-ply leaf on 46.79% of positions, far above the 5%
floor. The §2 identity finding does not repeat: v53-3ply is a
real version. The optimism is recorded as a finding against
the version's prior (per §6.3a it does not vacate the
battery — the battery is the verdict), and §8's fast loop
watches specifically for optimism-driven blunders (moves the
3-ply leaf overvalues because the rollout's last move is ours).

Rollout outcome counts over the 2,477 positions
(`test_v53_3ply_leaf_matches_spec_replay`, 0 mismatches
against the independent spec transcription): end-gradient
1,229; ply3 terminal 37; ply2 terminal 188; start-terminal
988; end-win 35; end-loss 0. Every reachable branch is
exercised; the terminal branches are grounded in the win rule
(`is_winning_stone`); the gradient antisymmetry the spec
relies on holds on all 1,229 rollout-end gradient positions.

**Dead-code finding (both leaves).** The 3-ply leaf's
rollout-end "opponent winning move → −1.0" branch is
UNREACHABLE under the rollout policy: at b2 `_rollout_move`
ends the rollout if the opponent has >= 2 winning moves and
blocks the single winning move if the opponent has exactly 1,
and our ply-3 stone cannot CREATE an opponent winning move
(it adds no opponent stones and occupies one square). The test
asserts the property (opponent has no winning move at b3) on
every sampled rollout-end position instead of requiring
instances. The same argument — verified empirically on the
same sample — shows the 2-ply leaf's rollout-end "+1.0"
check is dead too: 0/1,301 rollout-end positions have the leaf
player holding a winning move (the opponent's ply-2 move
cannot create OUR winning moves, and our single threat is
blocked by ply 2), while the −1.0 branch fires 35 times. So
each leaf has exactly one LIVE rollout-end terminal branch
(the side that just moved created a double threat) and one
dead branch (the side to move holding a winning move). The
dead checks stay in the code (they are in the pre-registered
specs, mirror each other, and guard against future policy
changes).

### 7.2 (b) Move-choice agreement vs v27 — 48/49, behaviour preserved

`test_v53_3ply_move_choice_agreement_vs_v27`: full 100-sim
searches at identical seeds (100–148) on all 49 tactical
positions. **Agreement 48/49** — above the pre-registered
≥45/49 line, so tactical behaviour is preserved. The single
disagreement is hard-suite position 14, prev-v1v1s-g39
("Prevent the open four — v1v1s game 39", White to move):
v53 plays (6,9), v27 plays (4,9); the expected preventing
moves are (9,4) and (9,6). Note the agreement test seeds each
position separately (100+i), while the tactical run uses one
RNG stream from seed 100 for all positions — so this
disagreement does NOT directly predict a tactical failure; the
clause-3 run (§9) is the verdict on this position.

### 7.3 (c) Cost — ratio 1.283, ceiling 2.0x

`test_v53_3ply_leaf_cost_within_ceiling` (200 quiet midgame
positions, min-of-3): v53 1.9359 s vs v27 1.5090 s total,
**ratio 1.283** — below the pre-registered 2.0x ceiling and
below the ~1.5x expectation (the third ply often terminates
early on forcing positions, and the rollout-end checks are
cheap scans). Battery-feasible.

### 7.4 Mechanism-check verdict

v53-3ply is a genuinely different version (46.79% of leaf
values differ, 48/49 tactical moves agree), it costs 1.28x
v27's leaf, and its known risk is the confirmed parity-shift
optimism (+0.0561 mean shift). Proceed to the tactical wiring
check, the fast loop, and the battery.

## 8. Fast loop

Pre-registered dev seeds (§6.4): 1053/2053 vs v0, 1054/2054 vs
v27 — distinct from the battery's 100/200. v53's design is
frozen (§6.1: the rollout policy is held fixed, depth is the
only variable) — the fast loop is defect shake-out, not
redesign; no refinement is made unless a genuine v53-specific
defect is found. All three batch-13 workstreams ran
concurrently on the ~1.5-CPU VM, so wall times are ~2x the
uncontended pace.

### 8.1 vs v0 (seeds 1053/2053)

**10W-0L-0D** (1.0000, CI [0.7225, 1.0000]) in 213.5s.
Uniqueness: 8/10 distinct games (the known vs-v0 determinism
collapse); de-dup cut 8W-0L-0D. No losses to read. Artefact:
`research/iterations/fast-v53-vs-v0.json`.

### 8.2 vs v27 (seeds 1054/2054)

**8W-1L-1D** (0.8500, CI [0.5411, 0.9646]) in 1294.3s.
Colour splits: as Black 4W-1L-0D (0.8000); as White 4W-0L-1D
(0.9000). Uniqueness: 10/10 distinct games. Games: g0 W (37
plies), g1 W (78), g2 W (43), g3 draw (225 plies — the
maximal-length draw, ~10 min of the leg), g4 W (45), g5 W
(76), g6 L (Black, 56), g7 W (28), g8 W (29), g9 W (40).
Artefact: `research/iterations/fast-v53-vs-v27.json`.

Loss classification (`classify_losses.py`, artefact
`research/iterations/losses-fast-v53-vs-v27.json`): **1 loss,
via open four, 0 missed blocks, 0 clean missed preventions in
5 threatened turns** — defence stays solved.

### 8.3 The one loss (g6): a SHARED v27/v53 defect, no patch

`find_losing_moves.py` (VCF solver, depth 4): the losing move
is ply 50, Black (4,6); the ONLY saving move is (9,5)
(1 saving move among 100 checked); the position was not
already lost before ply 50.

What happens after (4,6): White plays (9,5) — creating the
row-9 open three (9,5),(9,6),(9,7) alongside the diagonal
threat (6,6),(7,5),(8,4) — then (9,3), which is a DOUBLE
threat: the diagonal five at (10,2) and the horizontal five
at (9,4) (row 9: (9,3),(9,4),(9,5),(9,6),(9,7)). Black blocks
(9,4) at ply 54 — correctly, one of White's two winning moves
— and White completes the diagonal at (10,2). The loss is via
open four, exactly as classified; the ply-54 "missed block"
reading is wrong (White held TWO winning moves there, so the
game was lost anyway).

Why both engines miss the save — traced, not guessed: after
Black (4,6), BOTH leaves return −1.0000 from White's
perspective, i.e. a hallucinated BLACK WIN, although the VCF
solver says White forces a win from that position. The rollout
trace: ply 1 White plays the argmax (5,5) — missing the
forcing (9,5) — and ply 2 Black plays (4,7), leaving Black
with TWO winning moves ((4,3) and (4,8)). The weak argmax
policy drifts: rollout-White fails to find the forcing win
while rollout-Black finds a double threat. This is the 4-ply
hallucination's failure mode (v40) already present at 2-ply
depth in this position — and it is SHARED: v27, run on the
ply-50 position at its own per-game seed (2060), ALSO chooses
(4,6); the 2-ply and 3-ply leaves agree on the hallucinated
+1.0. The 3-ply leaf does rate the save (9,5) more pessimistically
for Black than the 2-ply leaf (−0.0521 vs +0.1026), but the
+1.0 hallucination on (4,6) dominates the move choice in BOTH
engines, so the difference is not decision-relevant here.

**No patch.** The defect is in the shared rollout policy,
which §6.1 holds fixed (depth is the variable, the policy is
not), and it costs v53 nothing against the bar reference —
v27 makes the identical mistake. Patching the policy would be
a different version, not this one. The fast loop's verdict:
no v53-specific defect; defence solved; proceed to the
battery. (The 8-1-1 fast-loop score is a small-sample signal
only — n=10, CI [0.54, 0.96]; the 100-game battery is the
verdict.)

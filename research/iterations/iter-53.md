# Iteration 53 — v53: 2-ply rollout leaf on v18/v27 (batch 13, H3)

**Status: PRE-REGISTRATION — §0 and §1 committed BEFORE any eval of
v53.** This is the H3 workstream of batch 13 (H1 is v51,
gradient-preserving backup; H2 is v52, threat-space search); work
happens on branch `research/batch-13-h3` in the `gomoku-b13-h3`
worktree, branched from `research/batch-13` @ `ca0a005` (which
includes the batch-13 plan). Comparison target is v27 (the
batch-13 bar reference: ladder Elo 636.4, on master).

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

*(§§2–5 to follow after the mechanism check.)*

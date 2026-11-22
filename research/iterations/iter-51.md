# Iteration 51 — v51: soft-min backup at opponent nodes (batch 13, H1)

**Status: IN PROGRESS — §1 pre-registered BEFORE any eval of v51.**
§1 was written and committed before any v51 eval ran; the mechanism
verification in §1.1 uses v27/v47 results only (cited from
iter-47.md — diagnosis, not a v51 eval); §§2–5 will record the
mechanism check, the fast loop, and the battery. Comparison target
is v27 (the batch-13 bar reference: ladder Elo 636.4, on master).
This is the H1 workstream of batch 13 (H2 is v52, threat-space
search at the root; H3 is v53, 2-ply rollout leaf); work happens on
branch `research/batch-13-h1` in the `gomoku-b13-h1` worktree,
branched from `research/batch-13` @ `ca0a005` (the batch-13 plan
commit). v51 does NOT change the rollout function, policy, gradient,
or either fast-loop patch (v16's, byte-identical), does NOT change
sims/priors/book (CLOSED), does NOT change exploration constants
(CLOSED), does NOT change the root rule or the tie-break (CLOSED —
the tie-break lever is doubly bounded), and does NOT include the
threat check or the VCF-at-root check (v35 CLOSED VCF-prevention for
the v27 family). The ONLY change from v27 is the backup rule in
§1.2 — a search change, not a leaf change. It is NOT pure minimax
(v47, CLOSED) and NOT pure min in any form (v46 leaf-level pure-min,
CLOSED): it is a smooth minimum that weights the worst lines heavily
but keeps the others' contribution, which is the narrow exception
the batch-13 plan pre-registers as a NEW mechanism.

**Method (carried over from batch 3/4/5/6/7/8/9/10/11/12):** the
fast 10-game loop — play 10 games, read EVERY move of every game,
find THE losing move per game (verified with the solver or a deeper
search — never guessed; for rollout-leaf engines, playouts or exact
computation, NOT the v1 walk-back oracle alone — iter-18 §2.1),
patch, regression-test, repeat. Max 2 fast-loop iterations before
validation (the two-consecutive-barren-iterations stop rule, iter-27
§2.1), or stop when two consecutive iterations find no new patchable
weakness class. The fast loop is for DEFECT SHAKE-OUT ONLY: the
pre-registered backup rule in §1.2 is fixed — no design retuning,
no β tuning (β=2.0 is fixed in the pre-registration), no root-rule /
tie-break / prior / sims change (all CLOSED). A fast-loop change is
allowed only to fix an implementation BUG in the backup rule itself
(e.g. the soft-min value not updating), verified by a failing test.
**The fast loop runs against v0 as well as v27** — each iteration
plays 10 games vs v27 AND 10 games vs v0, and losses from both
opponents are walked back. The full 4-clause validation battery
(100-game matches + ladder) runs on the FINAL v51 only. Fast-loop
games use development seeds (1051/2051, 1052/2052) distinct from the
bar battery's pre-registered seeds (100/200), so no development game
overlaps an eval game.

## 1. Design and pre-registration

**Comparison target: v27** (primary, the batch-13 reference:
ladder Elo 636.4). The bar's clause 1 is head-to-head vs v27
(normal rules, 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50), clause 2 is ladder Elo
strictly higher than v27's 636.4 (**40 games per opponent per
leg**, batch 11's re-basing; point estimates stay comparable to
historical versions, only the intervals narrow; leg-result reading
for saturation: a 40-0 leg strictly dominates a 39-0-1 leg; batch
12's finding: the Elo scale is draw-count sensitive at the top — a
123-point swing from a single draw — so point estimates must be
read with the leg composition alongside, and a clause-2 pass that
rests on a draw-count difference must say so explicitly), clause 3
is tactical pass rate >= v27's on the CORRECTED suite (old suite
20/20, hard suite 29/29), clause 4 is all test suites green (ml
pytest, npm test, tsc). **Secondary head-to-heads, also
pre-registered:** v51 vs v0 (100 games) and a 100-game swap match
vs v27 under the tempo judge (the harness default).
Non-transitivity is recorded prominently. **Strength champion:**
v18 (h2h vs v7 0.775, ladder Elo 636.4). Under champion rule v2
(direct challenge, TRACKS.md), a bar-passer plays a 100-game match
against v18 and takes the title if it wins the match (win rate >
0.50 with Wilson 95% CI lower bound > 0.50) AND its ladder Elo
beats 636.4 — a bar-passer already clears the Elo clause by
passing, so the challenge match is the deciding test. **Only run
the challenge match for a bar-passer.**

**The improvement: v51 = v27 + soft-min backup at opponent
nodes.** Everything else is v27 unchanged: 100 sims, c_puct 1.5,
v1 priors over all candidates, win/block scans before search,
most-visited root child, rollout leaf both colours (leaf mode
`rollout2_v1_both` — v18's rollout leaf, byte-identical), and the
threshold τ=0.05 on the q gap. The backup rule is the ONLY change
from v27.

### 1.1 Diagnosis: q-collapse, and what the mean backup preserves

The diagnosis is batch 12's v47 result (iter-47.md §1.4, §3.1,
§3.3 — cited here, not re-run; the numbers below are iter-47's):

- **Blunder dilution is real** (iter-40 §1.1a, re-verified by
  iter-47 §1.1 on g61/g43 with v27 only): the mean backup mixes
  the opponent's best reply with their blunders, so the unsafe
  move ranks above the safe moves even though its 1-ply leaf is
  −1.0 (g61: unsafe 11v, our-q −0.6364 vs best safe 3v,
  −0.6755; g43: unsafe 9v, −0.5556 vs best safe 8v, −0.6356).
  The leaf does NOT mis-rank; the mis-ranking is MCTS averaging
  over the opponent's moves instead of pricing their best move.
- **Pure minimax fixes the dilution by destroying the
  gradient** (iter-47 §1.4, §3.1): under v47 the unsafe child's
  our-q becomes exactly −1.0 on both instances — but the safes
  ALSO go to −1.0, so all four moves tie. In creator-rich
  positions nearly every subtree contains a visited −1.0 line,
  so minimax sets q = −1.0 for almost every root child (g13:
  ALL 74 visited children at −1.0; g2: 76/77). The mean backup
  preserved the prevention signal as an average — the
  prevention's subtree mixes −1.0 leaves with gradient leaves
  (g13: preventions at −0.75, 5v each, top-visit tie; g2:
  preventions at −0.6875/−0.5938, (7,5) leading 8v) while
  non-preventions average −1.0 — a 0.25 gap the search can rank
  on. Minimax erases it; with all q gaps exactly 0, v27's
  τ=0.05 tie-break can never fire, and UCB becomes purely
  prior-driven — and the v1 prior systematically prefers
  attacking moves (~0.10–0.11) over quiet preventions (~0.032),
  so v47 picks the losing move on g13 and g2 alike (tactical
  21/29, all 8 failures prevention positions; battery 0.245 vs
  v27).

The requirement on any successor backup rule follows directly:
it must price the opponent's best reply heavily enough to undo
the dilution (the unsafe move must rank worst on g61/g43) while
preserving a gradient across moves whose subtrees all contain a
−1.0 line (the preventions must keep a strictly-better-than-−1.0
value on g13/g2, separated from the non-preventions by more than
the τ=0.05 tie-break threshold). Pure minimax satisfies the
first and fails the second, by construction. A rule that
satisfies both must be a SMOOTH extremum: the worst line
dominates the value without silencing the rest.

### 1.2 The backup rule (pre-registered, exact)

`HeuristicMCTS` gains a `backup_rule` parameter, `"mean"`
(default — every other version's behaviour is byte-identical) or
`"softmax-opp"` (v51), plus a `softmax_beta` parameter (default
2.0; v51 sets it explicitly). The rule, exactly:

1. **Node types.** The root is an OUR node (mean backup). Every
   node at ODD depth from the root (opponent to move) is a
   SOFT-MIN node; every node at EVEN depth (us to move,
   including the root) is a MEAN node. The type is fixed at
   node creation: children of a node whose player-to-move
   equals the root player are soft-min nodes; all other
   children are mean nodes. (Same flag-propagation structure as
   v47's minimax flag.)
2. **Mean nodes.** q = value_sum / visit_count, UNCHANGED in
   form. Each simulation contributes the value returned from
   the child (negated): for a soft-min child that is the
   child's current soft-min q, for a mean child the leaf value.
   Our nodes therefore report the MEAN of the (soft-min)
   values of the lines the search explored.
3. **Soft-min nodes.** q is the softmax-weighted mean of the
   opponent-perspective values of the node's VISITED children:
   q = Σᵢ wᵢ·vᵢ, where vᵢ = −childᵢ.q (the value of child i
   from the opponent's perspective — the opponent maximises
   vᵢ) and wᵢ = exp(β·vᵢ) / Σⱼ exp(β·vⱼ), with β =
   `softmax_beta` = 2.0. Unvisited children are EXCLUDED: an
   unvisited child's q is undefined (0.0 by convention), not a
   signal — the same principle as the root-decision visited
   filter and as v47's minimax. (Including them would pin every
   soft-min node toward 0 until all children are visited.) A
   soft-min node with NO visited children yet (just expanded)
   reports its mean (= its leaf value). The weights are NOT
   visit-weighted: a child's contribution depends on its VALUE
   only, so a heavily-visited best reply and a once-visited
   best reply price the same — this is what undoes the
   dilution (the creator reply prices by its value, not by its
   visit share), and it avoids the visit-feedback loop in which
   PUCT concentrates visits on the worst reply and
   visit-weighting would then collapse the node toward minimax.
   Numerically the weights use the max-subtracted form
   (exp(β·(vᵢ − v_max))) so no overflow is possible; q stays
   in [−1, 1] as a convex combination of child values.
4. **Maintenance.** The soft-min value is maintained
   incrementally: after each descent through a soft-min node,
   it is recomputed from its visited children. Every simulation
   that changes a child's q passes through the parent, so the
   cached value is always fresh. Visit counts are UNCHANGED:
   every simulation increments visit_count on every node on
   its path, as before.
5. **Return values.** `_run_one` returns the node's updated q
   from the node's player's perspective: a soft-min node
   returns its soft-min q; a mean node returns the
   single-simulation value (as before). The parent negates it,
   as before.
6. **UCB selection.** `_select_child` is UNCHANGED in form:
   it reads `child.q`, which is now the soft-min value for
   opponent-node children and the mean for our-node children.
   The backup rule therefore changes visit allocation THROUGH
   the UCB term — that is the intended effect, not a side
   effect.
7. **Root decision.** `_choose_root_child` (most-visited +
   v27's τ=0.05 q-gap tie-break) is UNCHANGED in form; the
   tie-break reads root-child q, which is now soft-min. The
   threshold τ=0.05 is UNCHANGED.
8. **Scope.** The rule applies at ALL opponent nodes in the
   tree, not only at depth 1: the dilution is a general
   averaging artefact, and g61/g43 show it at depth 1 only
   because that is where it was diagnosed. (Same scope
   reasoning as v47 §1.2 item 8.)

**Why soft-min, not a second-moment backup.** The batch-13 plan
offers two candidate forms; the choice is soft-min. The dilution
is a weighting problem — the mean weights the opponent's replies
by visit share, so blunder leaves drag the value toward the
opponent's average reply instead of their best reply. Soft-min
fixes exactly that: it reweights replies by VALUE, so the best
reply dominates while the other replies still contribute, and
the gradient survives as a smooth function of the whole reply
distribution. A second-moment backup (rank by mean − λ·std, or
back up mean and variance) penalises VARIANCE, but variance
conflates two things — the opponent having a much better reply
than their average (the dilution signal) and genuine positional
uncertainty in quiet lines — and in creator-rich positions
nearly every subtree has high variance, so a variance penalty
would push values toward uniform pessimism rather than ranking
by "how it loses". It also introduces λ, which would need
calibration against the same collapse risk with no cleaner
theory. Soft-min has a single interpretable parameter (β, the
worst-line weight) with a direct worst-case analysis against the
τ=0.05 tie-break threshold (below). A min-mean blend was already
rejected by iter-47 as a confound (a failed blend would not tell
us whether minimax is wrong or the weight was too small), and
iter-47 predicted a blend would inherit the collapse in
proportion to its min weight — soft-min is not a blend: it is a
smooth extremum, and the analysis below shows it preserves the
gradient at the chosen β rather than inheriting the collapse.

**Why β = 2.0 (fixed, not tuned).** Values span [−1, 1]. At
β=2 a 1.0 value gap gets weight ratio e² ≈ 7.4 — the worst line
dominates without silencing the rest — and a 0.25 gap (the
mean's prevention signal on g13) gets ratio e^0.5 ≈ 1.65.
Worst-case analysis for the tie-break: a prevention whose
subtree has one +1.0 child (opponent perspective) and one +0.5
child scores (e²·1 + e¹·0.5)/(e² + e¹) ≈ 0.866, a 0.134 gap vs
a pure −1.0 (opponent +1.0) non-prevention — above τ=0.05, so
the tie-break can fire. At β=4 the same case gives 0.940, a
0.060 gap — too close to the threshold; at β=8 it gives 0.958,
a 0.042 gap — below it, i.e. collapse. And a single +1.0 child
among five +0.5 children scores 0.676 at β=2 — the node is NOT
pinned at +1.0 by one bad line, which is exactly the property
minimax lacked. β=2 is therefore the largest temperature that
keeps the worst-case gradient comfortably above the tie-break
threshold while still pricing the worst line ~7x a line 1.0
worse. β is fixed here; the fast loop may not retune it.

**Predicted mechanism check (on g13/g2/g61/g43, after
implementation).** The check is the q RANKING and the gradient,
not the root pick: the root is most-visited, and the backup
change moves visit allocation through UCB, so the pick may or
may not change; the tactical battery (clause 3) is the pick-level
verdict. Predicted, on 100-sim search dumps (seed 0) on the
reconstructed positions:

- **g13 / g2 (gradient preservation):** the best prevention's
  our-q is strictly greater than −1.0 (the prevention keeps a
  gradient value, unlike v47 where preventions sat at exactly
  −1.0), AND it exceeds the best non-prevention's our-q by
  more than 0.05 (the tie-break threshold — the gap the search
  can rank on; v27's gap is 0.25 on g13).
- **g61 / g43 (dilution fix):** the unsafe move ranks strictly
  worst by our-q among {unsafe + safes} (the dilution is
  undone — v47 achieved the −1.0 pricing but tied the safes),
  AND the best safe's our-q is strictly greater than −1.0 (the
  gradient survives — v47 failed this, all safes at −1.0).

**Pass/fail criterion for the mechanism check (stated BEFORE
running it):** PASS requires all four sub-criteria on all four
positions (g13/g2: prevention > −1.0 AND prevention gap >
0.05; g61/g43: unsafe strictly worst AND best safe > −1.0).
If the check fails, the failure is recorded honestly in §1.4
with the measured numbers, and the battery still runs — the
battery is the verdict on game strength, and the check is a
mechanism diagnostic, not a gate (same treatment as iter-47,
whose mechanism check partially failed and whose battery ran
anyway). The root picks on the four positions are reported as
secondary observations, not part of the pass criterion.

**Tactical interaction risk (pre-registered).** v40's lesson
and v47's §3.1: a backup-rule change interacts with the
tie-break through root-child q. The soft-min compresses q
gaps relative to the mean (the worst line pulls every value
toward −1.0), so some v27 visit ties may now have gaps below
τ=0.05 and fall back to RNG — the mechanism check measures
exactly this on g13/g2. If tactical regresses, diagnose
whether it is a visit-allocation / tie-break interaction and
record it honestly. Do NOT patch the suite.

**Cost risk (pre-registered).** The soft-min recompute is
O(visited children) per descent with an exp() per child —
the same order as v47's max recompute, which cost ~3x per
move (fast loop ~70–110s/game vs v27's ~22s/game pace). v51's
battery legs are expected to run at a similar pace; per-move
time is measured in the mechanism check and reported.

### 1.3 Battery (pre-registered)

All 100 games, validation seeds 100/200 unless noted; every
leg recorded as a committed JSON artifact; drivers seed per
game with a uniqueness audit (report distinct-game counts and
de-dup rates for every leg):

- vs v27, normal rules — clause 1.
  `ml/scripts/selfplay_batch.py --engine-a v51 --engine-b v27
  --games 100 --seed-a 100 --seed-b 200
  --out research/iterations/batch-v51-vs-v27.json`
- vs v0, normal rules — secondary.
  Same with `--engine-b v0` and
  `--out research/iterations/batch-v51-vs-v0.json`.
- vs v27, SWAP rules (`--swap-rule --placement balanced
  --placement-seed 11 --judge tempo`) — secondary.
  `--out research/iterations/batch-v51-vs-v27-swap.json`.
- Ladder: `ml/scripts/eval_version.py --version 51
  --ladder-games 40 --out research/iterations/ladder-v51.json`
  (40 games/opponent, seed 100, auto-resume driver) — clause 2.
- Tactical: `eval_tactical_full.py --version 51 --suite both
  --seed 100`. Re-verify v27's tactical first on the corrected
  suite — clause 3.
- All test suites green (ml pytest, npm test, tsc) — clause 4.
- Report the clean-missed-prevention rate (v27-family versions
  run ~0.4–1.3%) and confirm defence stays solved (all losses
  via open four, 0 missed blocks) via `classify_batch_losses.py`.
- Recompute every aggregate from the raw per-game records
  (A win = winner_is_a is True; draws = winner == 'draw') and
  recompute the Wilson 95% CIs; report the recomputed numbers,
  not just the script summaries.

### 1.4 Mechanism-check outcome (after implementation — NOT a redesign)

**FAIL — the soft-min collapses exactly like v47's minimax on
three of the four positions.** Run after the §1.2 implementation,
before any game eval (script `/tmp/diag_v51_check.py`; the check
itself is a test:
`ml/tests/test_mcts_v51.py::test_v51_mechanism_outcome`).
The v27 control reproduces iter-47's numbers EXACTLY on all four
positions (g13: preventions −0.75, 5v each, pick (11,7); g2:
(7,5) −0.5938, 8v, pick (7,5); g61: unsafe 11v −0.6364 vs best
safe −0.6755; g43: unsafe 9v −0.5556 vs best safe −0.6356), so
the fixtures are right and the failure is v51's, not the setup's.
Backup invariants hold on all four trees (0 violations: soft-min
nodes report the softmax over visited children of −child.q, mean
nodes report value_sum / visit_count); flag propagation is correct
(root children soft-min, depth-2 mean). Cost: v51's 100-sim search
takes 0.47–0.93s on these positions vs v27's 0.59–1.04s — the
soft-min is NOT slower (v47's ~3x cost does not reproduce), so
cost is not the issue.

| Position | Criterion (pre-registered) | v51 measured | Result |
|---|---|---|---|
| g13 | best prevention our-q > −1.0 AND gap vs best non-prevention > 0.05 | preventions −1.0000 (2v each); ALL 74 visited children at −1.0; gap 0.0; pick (5,4) — the move that lost the original game, same as v47 | **FAIL** |
| g2 | same | preventions −1.0000 (2v each); 75/77 at −1.0; best non-prevention −0.8008, gap −0.199 (prevention WORSE); pick (7,9) | **FAIL** |
| g61 | unsafe strictly worst AND best safe > −1.0 | unsafe −0.9293 (7v) — ranks BEST of the four; safes all −1.0000; pick (4,5) (a safe, via 9v vs 7v) | **FAIL** (ranking inverted) |
| g43 | same | unsafe −0.9293 (4v) strictly worst; safes −0.7071 (11v) / −0.8752 (9v); pick (8,8) ✓ | **PASS** |

**Verified mechanism of the collapse** (script
`/tmp/diag_v51_collapse.py`, g13 prevention subtrees):

1. **The mean backup's prevention gradient comes from the
   EXPANSION LEAF, not from mixing children.** Under v27 the
   prevention (2,3) has ONE visited child — the opponent's
   reply (3,3), 4v, opponent-perspective q = +1.0 (its
   grandchildren are all at our-q −1.0). The prevention's
   −0.75 is the average of its +0.25 expansion leaf with four
   −1.0 descents (−3.75/5). The "gradient" is the expansion
   visit counted in the mean.
2. **The soft-min discards the expansion leaf.** A soft-min
   node recomputes q from its VISITED CHILDREN ONLY. With one
   visited child, the softmax IS that child's value — no
   smoothing is possible with one child, whatever β is. Under
   v51 the prevention (2,3) has 2 visits and one visited child
   at +1.0 (opponent), so q = −1.0 exactly — while its
   discarded mean is −0.375 (the same +0.25 expansion leaf
   averaged with one −1.0 descent). The formula's smoothing
   never engages: the analysis in §1.2 assumed several visited
   children with mixed values, but at 100 sims over ~74 root
   children a root child gets 1–5 visits, hence 0–1 visited
   children.
3. **Visit starvation compounds it.** Once the soft-min
   reports −1.0 after the first descent, UCB sees the
   prevention as tied with the −1.0 mass and stops visiting it
   (2v vs v27's 5v) — so it never accumulates the multiple
   visited children the softmax would need. A vicious circle:
   the rule needs visits to show a gradient, and the gradient
   is what earns visits.
4. **The gradient preservation is visit-count-dependent,
   which is why g43 passes and g61 inverts.** The soft-min
   mixes only where visits are plentiful. On g43 the safes
   have high priors (0.1342) → 9–11 visits → multiple visited
   children → the softmax mixes and the gradient survives. On
   g61 the unsafe (prior 0.0757, 7v) mixes its creator child
   (+1.0) with its blunder children (−1.0) and lands at
   −0.9293 — the dilution IS partially undone (0.636 → 0.929)
   — but the safes (2–3v) collapse to −1.0, inverting the
   ranking. The preventions on g13/g2 have LOW priors (~0.032)
   → starved → collapsed. The soft-min cannot rescue
   low-prior preventions: it preserves a gradient only where
   the prior already points, which is exactly where the
   gradient is least needed.

**What this establishes.** The failure is structural, not a
β problem and not an implementation bug (the invariants hold;
the formula is verified by unit test against the analytic
value — the g61 unsafe's −0.9293263119 is exactly the β=2
softmax of {+1.0, −1.0, −1.0}, i.e. its subtree is the creator
child plus two blunder children, as diagnosed).
ANY child-level backup aggregation
(min, max, softmax) degenerates at 100 sims: a node with one
visited child reports that child's value and discards its own
expansion leaf, and the mean backup is the only rule that
counts the expansion visit. v51's collapse is v47's collapse
with a different formula — the pre-registration's worst-case
analysis was correct about the formula but wrong about the
tree: it assumed the soft-min node would HAVE several visited
children to smooth over. A rule that aggregated the expansion
leaf WITH the children might escape this, but that is a
different rule requiring its own pre-registration — it is NOT
attempted here. The pre-registered v51 stands as implemented,
and per §1.2 the battery still runs: the battery is the
verdict on game strength, and the check is a mechanism
diagnostic, not a gate. (Prediction for the battery, recorded
here before it runs: tactical should fail on the prevention
positions much as v47's did — 8 prevention failures is the
shape to expect — and the head-to-head should land well below
0.50, though probably above v47's 0.245, since the soft-min
keeps a gradient on well-visited nodes where minimax kept
none.)

## 2. Fast loop

### 2.1 Iteration 1 (seeds 1051/2051)

- vs v27: **3W-7L-0D (0.3000, CI [0.1078, 0.6032])**,
  713.2s (~71s/game — v51's search is NOT slower than v27's
  per move; the game pace is the two-search cost, as for v27
  vs v27).
- vs v0: **5W-5L-0D (0.5000, CI [0.2366, 0.7634])**, 393.6s.
  v51 TIES the baseline v0 that v27 beats 0.965 — the same
  shape as v47's iteration 1 (5-5 vs v0).
- Loss classification (`classify_batch_losses.py` on both
  legs): all 12 losses via open four; **0 missed blocks** (all
  games); **2 clean missed preventions** in 30 threatened turns
  (6.7%) on the vs-v27 leg — well above the v27-family range
  (0.4–1.3%), the tactical prevention failure (§1.4) showing
  up in real games already at 10 games — and 0 in 32 threatened
  turns on the vs-v0 leg, where the losses are distributed
  drift (v51's prior-driven picks in sharp positions are often
  forcing moves, which count as "preventing" to the classifier,
  but weak attacks that build no lasting threat — the same
  manifestation iter-47 §2.1 recorded for v47).
- No implementation bug found (the §1.4 invariants hold; the
  behaviour follows from the rule). No patchable weakness class
  WITHIN the remit: the weakness IS the pre-registered rule
  (child-level aggregation collapse → prior-driven play in
  sharp positions, §1.4), and the rule is fixed — the fast loop
  is defect shake-out only (§1). Iteration 1 is therefore
  barren for patching purposes; a second iteration runs to
  confirm the pattern per the two-consecutive-barren-iterations
  stop rule.

### 2.2 Iteration 2 (seeds 1052/2052)

- vs v27: **3W-7L-0D (0.3000, CI [0.1078, 0.6032])**,
  364.7s (~36s/game) — identical to iteration 1.
- vs v0: **4W-6L-0D (0.4000, CI [0.1682, 0.6873])**, 266.2s.
- Loss classification: all 13 losses via open four; 0 missed
  blocks; 3 clean missed preventions in 38 threatened turns
  (7.9%) on the vs-v27 leg and 0 in 33 on the vs-v0 leg. Same
  pattern as iteration 1.
- **Fast-loop totals: vs v27 6W-14L-0D (0.3000); vs v0
  9W-11L-0D (0.4500).** Two consecutive iterations with no
  patchable weakness class — the loop STOPS per the stop rule.
  No fast-loop change was made (none is allowed: the rule is
  pre-registered and fixed; no bug was found). Pooled
  clean-missed-prevention rate vs v27: 5 in 68 threatened turns
  (7.4%) — roughly 6x the v27-family range, the §1.4 mechanism
  visible in games.

## 3. Battery

### 3.1 Tactical — clause 3 FAILS (21/29 hard)

v27 re-verified first on the corrected suite: 20/20 + 29/29
(`research/iterations/tactical-v27-reverify.json`). v51:
**20/20 old, 21/29 hard** (`research/iterations/tactical-v51.json`,
seed 100). All 8 failures are PREVENTION positions — the EXACT
same 8 as v47 (iter-47 §3.1): prev-v1v0s-g13, prev-v2v0-g0,
prev-v2v0-g2, prev-v2v1s-g12, prev-v1v0s-g26, prev-v1v0s-g48,
prev-v1v1s-g79, prev-v1bl-g2. All setup / winning / VCF positions
pass. The §1.4 prediction (tactical fails on the prevention
positions much as v47's did) is confirmed exactly — the soft-min
and the minimax are indistinguishable on the tactical suite,
because at 100 sims both degenerate to a single child's value on
the positions that matter.

### 3.2 Head-to-head legs — clause 1 FAILS (0.3150 vs v27)

All aggregates below were recomputed from the raw per-game
records and match the stored JSON values exactly. All legs use
validation seeds 100/200, per-game seeding, 100 games.

**Leg 1 — vs v27, normal rules (clause 1): 28W-65L-7D,
win rate 0.3150, Wilson 95% CI [0.2322, 0.4115]**
(`research/iterations/batch-v51-vs-v27.json`; the leg survived
two runtime restarts / VM replacements — resumed from game 14,
then from game 49 — and auto-resume + per-game seeding lost
zero games; the resumed run's elapsed is 5337.1s). As Black
19W-25L-6D (0.4400); as White 9W-40L-1D (0.1900) — a large
White deficit, the same shape as v47's (Black 0.31 / White
0.18). Uniqueness 95/100; de-dup 25W-63L-7D, 0.3000, CI
[0.2171, 0.3985] — the de-dup cut agrees, so the failure is
not a determinism artefact. Clause 1 fails decisively: the CI
upper bound (0.4115) is far below 0.50. v51 beats v47's 0.245
(+0.07 — the soft-min keeps a gradient on well-visited nodes
where minimax kept none, §1.4) but loses to its own parent by
a margin no other v27-family version except v47 has posted
(v39 0.495, v40 0.525, v41 0.52 — all near-ties). All seven
draws are 225-ply full-board games — the two engines grind to
a full board when neither finds a win (which is also why the
leg's wall time is dominated by a handful of very long games).

**Leg 2 — vs v0, normal rules (secondary): 47W-53L-0D,
win rate 0.4700, CI [0.3751, 0.5671]**
(`research/iterations/batch-v51-vs-v0.json`; resumed once from
game 35 after a runtime restart; resumed run 2190.6s). v51
LOSES to the frozen baseline on the raw count — far below
v27's 0.965 and below v47's 0.78 — but the leg is dominated by
a determinism collapse and a colour split that must be read
together. As Black 47W-3L-0D (0.9400); as White **0W-50L-0D
(0.0000)** — v51-as-White loses every game to v0. Uniqueness
is poor: **39/100 unique** — the worst determinism collapse in
the program (v36's 21/100 was a different mechanism: v36's
games repeated across BOTH colours; here the collapse is
entirely White: as White only **4 unique games in 50**, the
same 4 losses repeated 12–13 times each, while Black has 35
unique in 50). De-dup 32W-7L-0D, 0.8205, CI [0.6733, 0.9102]
— the de-dup cut clears 0.50, but it is driven by Black
(de-dup as Black 32W-3L, 0.9143; as White 0W-4L, 0.0000, CI
[0.0000, 0.4899]). The honest read: v51-as-Black genuinely
beats v0, and v51-as-White genuinely loses to v0 — but the
White evidence is 4 independent losses, not 50. The effective
N is ~39, not 100. The mechanism is the mirror-match one
(iter-47 §3.2): per-game seeding does not help when the two
engines' decisions are near-deterministic functions of the
position — v0 plays greedily, and v51's collapsed q's make its
search near-deterministic too, so the same White loss replays
game after game. The White collapse vs the greedy baseline is
v51's worst single result: even v47-as-White beat v0 0.86.

**Leg 3 — vs v27, swap rules (secondary): 37W-57L-6D,
win rate 0.4000, CI [0.3094, 0.4980]**
(`research/iterations/batch-v51-vs-v27-swap.json`; resumed
twice, from games 37 and 89, after two runtime restarts;
resumed run 996.6s). Better than the normal-rules leg
(0.3150 → 0.4000) — swap removes the colour edge v51-as-White
suffered — but still decisively below 0.50: the CI upper bound
is 0.4980. Uniqueness is clean here: 100/100 unique (the
balanced placements give variety the mirror match lacked), so
the de-dup cut is identical to the raw count. As Black
24W-25L-3D (0.4904); as White 13W-32L-3D (0.3021) — the White
gap persists under swap (muted vs normal rules: White 0.19 →
0.30), consistent with the fast-loop finding that v51's
prevention failures are concentrated in White's defence. The
swap leg's five 225-ply draws (and the normal leg's seven) are
all full-board grinds; they inflate wall time but not the win
rate.

## 4. Verdict

[To be filled.]

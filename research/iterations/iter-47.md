# Iteration 47 — v47: minimax backup at opponent nodes (batch 12, H3)

**Status: IN PROGRESS — pre-registered, implementation pending.**
§1 was pre-registered BEFORE any eval of v47 (the mechanism
verification in §1.1 uses v27 only — it is diagnosis, not a v47
eval); §§2–5 will record the fast loop and battery. Comparison
target is v27 (the batch-12 bar reference: ladder Elo 636.4, on
master at `128c327`). This is the H3 workstream of batch 12 (H1
is v45, the v35 VCF-hole fix; H2 is v46, White-side statistics);
work happens on branch `research/batch-12-h3` in the
`gomoku-b12-h3` worktree, branched from `research/batch-12` @
`7e8072c` (the batch-12 plan commit). v47 does NOT change the
rollout function, policy, gradient, or either fast-loop patch
(v16's, byte-identical), does NOT change sims/priors/book
(CLOSED), does NOT change exploration constants (CLOSED), does
NOT change the root rule or the tie-break (CLOSED — the
tie-break lever is doubly bounded), and does NOT include the
threat check or the VCF-at-root check (v35 CLOSED VCF-prevention
for the v27 family). The ONLY change from v27 is the backup
rule in §1.2 — a search change, not a leaf change.

**Method (carried over from batch 3/4/5/6/7/8/9/10/11):** the
fast 10-game loop — play 10 games, read EVERY move of every
game, find THE losing move per game (verified with the solver or
a deeper search — never guessed; for rollout-leaf engines,
playouts or exact computation, NOT the v1 walk-back oracle
alone — iter-18 §2.1), patch, regression-test, repeat. Max 2
fast-loop iterations before validation (the two-consecutive-
barren-iterations stop rule, iter-27 §2.1), or stop when two
consecutive iterations find no new patchable weakness class.
The fast loop is for DEFECT SHAKE-OUT ONLY: the pre-registered
backup rule in §1.2 is fixed — no design retuning, no α to tune
(there is none: the rule is pure minimax, not a blend), no
root-rule / tie-break / prior / sims change (all CLOSED). A
fast-loop change is allowed only to fix an implementation BUG
in the backup rule itself (e.g. the minimax value not updating),
verified by a failing test. **The fast loop runs against v0 as
well as v27** (the bar reference is v27 now, not v7) — each
iteration plays 10 games vs v27 AND 10 games vs v0, and losses
from both opponents are walked back. The full 4-clause
validation battery (100-game matches + ladder) runs on the
FINAL v47 only. Fast-loop games use development seeds
(1047/2047, 1048/2048) distinct from the bar battery's
pre-registered seeds (100/200), so no development game overlaps
an eval game.

## 1. Design and pre-registration

**Comparison target: v27** (primary, the batch-12 reference:
ladder Elo 636.4). The bar's clause 1 is head-to-head vs v27
(normal rules, 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50), clause 2 is ladder Elo
strictly higher than v27's 636.4 (**40 games per opponent per
leg**, batch 11's re-basing; point estimates stay comparable
to historical versions, only the intervals narrow; leg-result
reading for saturation: a 40-0 leg strictly dominates a
39-0-1 leg), clause 3 is tactical pass rate >= v27's on the
CORRECTED suite (old suite 20/20, hard suite 29/29), clause 4
is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v47 vs v0
(100 games) and a 100-game swap match vs v27 under the tempo
judge (the harness default). Non-transitivity is recorded
prominently. **Strength champion:** v18 (h2h vs v7 0.775,
ladder Elo 636.4). Under champion rule v2 (direct challenge,
TRACKS.md), a bar-passer plays a 100-game match against v18
and takes the title if it wins the match (win rate > 0.50 with
Wilson 95% CI lower bound > 0.50) AND its ladder Elo beats
636.4 — a bar-passer already clears the Elo clause by passing,
so the challenge match is the deciding test. **Only run the
challenge match for a bar-passer.**

**The improvement: v47 = v27 + minimax backup at opponent
nodes.** Everything else is v27 unchanged: 100 sims, c_puct
1.5, v1 priors over all candidates, win/block scans before
search, most-visited root child, rollout leaf both colours
(leaf mode `rollout2_v1_both` — v18's rollout leaf,
byte-identical), and the threshold τ=0.05 on the q gap. The
backup rule is the ONLY change from v27.

### 1.1 Diagnosis: blunder dilution reproduces exactly on g61 and g43

The mechanism (iter-40 §1.1a) was re-verified on this branch
BEFORE pre-registration, using v27 only: the boards were
reconstructed from the v34 battery JSONs on
`research/batch-10-h2` (`batch-v34-vs-v27.json` game 61, ply
21; `batch-v34-vs-v27-swap.json` game 43, ply 17 — coordinates
in iter-40 are (col, row); the swap opening is (black1,
white1, black2) with White to move next), v27's search was
run (100 sims, seed 0), and the root dump, 1-ply leaf values,
and the subtree under the unsafe move were compared to
iter-40's numbers. Script `/tmp/diag_v47_mech.py` (the
batch-11 diag scripts lived in /tmp and are gone after VM
replacements; this one is re-derived from the committed
battery JSONs).

| Instance | Root (v27, seed 0): unsafe vs best safe | 1-ply leaf (ours): unsafe vs safes | Subtree under unsafe |
|---|---|---|---|
| vs-v27 g61 ply 21, played (8,1); safes (3,5),(4,5),(8,5) | 11v, −0.6364 vs 3v, −0.6755 (8,5) | −1.0000 vs −0.0133/−0.0521/−0.0265 | creator (4,5) 8v at −1.0 (our persp); blunders (9,1),(11,4) 1v each at +1.0 |
| swap g43 ply 17, played (7,8); safes (8,8),(12,8) | 9v, −0.5556 vs 8v, −0.6356 (8,8) | −1.0000 vs +0.0924/−0.0976 | creator (12,8) 6v at −1.0 (our persp); blunders (9,6),(10,11) 1v each at +1.0 |

Every number reproduces iter-40 EXACTLY (visits, q to 4
decimals, leaf values, subtree shape). The root pick on both
is the unsafe move. The arithmetic is the dilution: g61's
unsafe child mixes the expansion visit (+1 for Black, the
rollout plays the creator) with 8 creator descents (+1) and 2
blunder descents (−1) for 7/11 = 0.6364 — the two blunder
leaves, each CORRECT (the unsafe move creates our own
open-four race threats, so a non-creator reply loses at once),
drag the average 0.039 above the best safe's. g43 is the same
shape (margin 0.080). The leaf does NOT mis-rank; the
mis-ranking is MCTS averaging over the opponent's moves
instead of taking the opponent's best move. **Fixing the
average is a backup-rule (search) change, not a leaf change**
— iter-40 §1.1e recorded exactly this fix ("Minimax backup /
max-backup for the opponent") as not attempted. That is this
version.

**Outcome-free caveat (iter-40 §1.1d):** g61/g43 are
outcome-free — every safe move loses 3/3 in playouts, as does
the played move. Changing the ranking on these instances is a
MECHANISM CHECK, not a strength claim. The 100-game battery
is the verdict.

**Correction to the plan's wording (recorded here, per the
task):** the batch-12 plan says "at the leaf, aggregate the
opponent's reply values by min instead of mean." The rollout
leaf (`_rollout_leaf_value`) returns a SINGLE value — it plays
exactly 2 plies and evaluates the gradient; there is no mean
to replace at the leaf. The mean is in the MCTS BACKUP (q =
mean of subtree leaf values). This version changes the
backup, not the leaf. (The leaf-level variant — evaluating the
opponent's top-K replies at the leaf and taking the min — is
reserved as H2's fallback; it is NOT implemented here.)

### 1.2 The backup rule (pre-registered, exact)

`HeuristicMCTS` gains a `backup_rule` parameter, `"mean"`
(default — every other version's behaviour is byte-identical)
or `"minimax-opp"` (v47). The rule, exactly:

1. **Node types.** The root is an OUR node (mean backup).
   Every node at ODD depth from the root (opponent to move)
   is a MINIMAX node; every node at EVEN depth (us to move,
   including the root) is a MEAN node. The type is fixed at
   node creation: children of a node whose player-to-move
   equals the root player are minimax nodes; all other
   children are mean nodes.
2. **Mean nodes.** q = value_sum / visit_count, UNCHANGED in
   form. Each simulation contributes the value returned from
   the child (negated): for a minimax child that is the
   child's current minimax q, for a mean child the leaf
   value. Our nodes therefore report the MEAN of the
   (worst-case) values of the lines the search explored —
   a "mean of maxes".
3. **Minimax nodes.** q = max over VISITED children of
   (−child.q) — the opponent's best reply, from the
   opponent's perspective; equivalently the MIN over visited
   children of child.q from our perspective. Unvisited
   children are EXCLUDED: an unvisited child's q is undefined
   (0.0 by convention), not a signal — the same principle as
   the root-decision visited filter. (Including them would
   pin every minimax node at ≥ 0 for the opponent until all
   children are visited.) A minimax node with NO visited
   children yet (just expanded) reports its mean (= its leaf
   value).
4. **Maintenance.** The minimax value is maintained
   incrementally: after each descent through a minimax node,
   it is recomputed as the max over its visited children of
   −child.q. Every simulation that changes a child's q
   passes through the parent, so the cached value is always
   fresh. Visit counts are UNCHANGED: every simulation
   increments visit_count on every node on its path, as
   before.
5. **Return values.** `_run_one` returns the node's updated q
   from the node's player's perspective: a minimax node
   returns its minimax value; a mean node returns the
   single-simulation value (as before). The parent negates
   it, as before.
6. **UCB selection.** `_select_child` is UNCHANGED in form:
   it reads `child.q`, which is now the minimax value for
   opponent-node children and the mean for our-node
   children. The backup rule therefore changes visit
   allocation THROUGH the UCB term — that is the intended
   effect, not a side effect.
7. **Root decision.** `_choose_root_child` (most-visited +
   v27's τ=0.05 q-gap tie-break) is UNCHANGED in form; the
   tie-break reads root-child q, which is now minimax. The
   threshold τ=0.05 is UNCHANGED.
8. **Scope.** The rule applies at ALL opponent nodes in the
   tree, not only at depth 1: the dilution is a general
   averaging artefact, and g61/g43 show it at depth 1 only
   because that is where it was diagnosed.

**Why pure minimax, not a min-mean blend.** The hypothesis is
"averaging over the opponent's moves is the bug"; the blend
(α·min + (1−α)·mean) only half-fixes it and makes α the
confound if it fails — a failed blend would not tell us
whether minimax is wrong or α was too small. Pure minimax is
the clean test: it prices the opponent's best reply exactly,
and the battery (vs v0 AND vs v27) judges the
over-pessimism risk. The known risk is stated plainly: pure
minimax assumes the opponent always finds their best reply,
so it gives up exploiting blunders — vs v0, which blunders
constantly, that may cost games (v3's over-defensiveness
lesson, though v3's was a root-level move preference, not a
backup change). The battery decides.

**Predicted mechanism check (on g61/g43, after
implementation):** the unsafe root child's our-q becomes
−1.0 (the opponent's best reply — the creator — is priced
exactly) and ranks WORST among the unsafe + safe moves on
both instances. The check is the q RANKING, not the root
pick: the root is most-visited, and the backup change moves
visit allocation through UCB, so the pick may or may not
change. The exact visit counts will differ from v27's; only
the ranking is predicted.

**Tactical interaction risk (pre-registered).** v40's lesson:
exact leaf values broke the v1bl-g2 visit tie that v27's
tie-break relies on. A backup-rule change interacts with the
tie-break the same way (the tie-break reads root-child q,
now minimax, with coarser gaps — many children may sit at
exactly −1.0). If tactical regresses, diagnose whether it is
a visit-allocation / tie-break interaction and record it
honestly. Do NOT patch the suite.

### 1.3 Battery (pre-registered)

All 100 games, validation seeds 100/200 unless noted; every
leg recorded as a committed JSON artifact; drivers seed per
game with a uniqueness audit (report distinct-game counts and
de-dup rates for every leg):

- vs v27, normal rules — clause 1.
- vs v0, normal rules — secondary.
- vs v27, SWAP rules (`--swap-rule --placement balanced
  --placement-seed 11 --judge tempo`) — secondary.
- Ladder: `ml/scripts/eval_version.py --out
  research/iterations/ladder-v47.json` (40 games/opponent,
  auto-resume driver) — clause 2.
- Tactical: 20/20 + 29/29 on the corrected suite — clause 3.
  Re-verify v27's tactical first.
- All test suites green (ml pytest, npm test, tsc) —
  clause 4.
- Report the clean-missed-prevention rate (v27-family
  versions run ~0.4–1.3%) and confirm defence stays solved
  (all losses via open four, 0 missed blocks) via
  `classify_batch_losses.py`.

### 1.4 Mechanism-check outcome (after implementation — NOT a redesign)

Run after the §1.2 implementation, before any game eval
(script `/tmp/diag_v47_check.py`; the check itself is a test:
`ml/tests/test_mcts_v47.py::test_v47_prices_opponent_best_reply`):

- **Predicted, holds:** the unsafe root child's our-q becomes
  exactly −1.0 on both instances — the opponent's best reply
  (the creator) is priced, not diluted.
- **Predicted, FAILS:** the unsafe does NOT rank strictly
  worst by q — the safes ALSO go to −1.0, so all four moves tie.
  Verified explanation (`/tmp/diag_v47_safe_subtree.py`): under
  v27, every safe's subtree already contains a visited −1.0
  line (Black's (6,4) in g61 — exactly iter-40's "Black wins
  slowly via (6,4)" — and (10,7) in g43). The positions are
  lost; minimax correctly values EVERY move at −1.0. The mean
  backup distinguished the moves by HOW they lose (slowly vs
  at once); minimax destroys that "loss-speed" gradient.
- **Unpredicted, observed:** the root pick nevertheless flips
  to a SAFE move on both instances (g61: (4,5), 9v vs the
  unsafe's 5v; g43: (12,8), 9v vs 5v) — via visit reallocation,
  not via the q ranking. With all q's tied at −1.0, UCB is
  prior-driven, and the highest-prior safe moves win visits
  (safes' priors 0.1404 / 0.1342 vs the unsafe's 0.0757 /
  0.0838). The flip is real but its channel is the prior, not
  the backup's q signal — noted here because §3.1 shows the
  same channel failing systematically on prevention positions,
  where the preventions have LOW priors.
- Backup invariants hold on both trees (0 violations: minimax
  nodes report max over visited children of −child.q, mean
  nodes report value_sum / visit_count); flag propagation is
  correct (root children minimax, depth-2 mean).

## 2. Fast loop

(TODO)

## 3. Battery

### 3.1 Tactical — clause 3 FAILS (21/29 hard)

v27 re-verified first on the corrected suite: 20/20 + 29/29
(`research/iterations/tactical-v27.json`). v47:
**20/20 old, 21/29 hard** (`research/iterations/tactical-v47.json`,
seed 0). All 8 failures are PREVENTION positions:
prev-v1v0s-g13, prev-v2v0-g0, prev-v2v0-g2, prev-v2v1s-g12,
prev-v1v0s-g26, prev-v1v0s-g48, prev-v1v1s-g79, prev-v1bl-g2
(the v40 regression position, again). All setup / winning /
VCF positions pass.

**Diagnosis — q-collapse, verified on two positions**
(script `/tmp/diag_v47_tactical.py`, raw 100-sim search dump):

| Position | v27 root | v47 root |
|---|---|---|
| prev-v1v0s-g13 (4 expected preventions) | 74 visited children, 70 at our-q −1.0; preventions at −0.75 (5v each, top-visit tie) → picks (11,7) ✓ | ALL 74 visited children at our-q −1.0; preventions 2v each → picks (5,4), the move that lost the original game |
| prev-v1bl-g2 (2 expected preventions) | 77 visited, 73 at −1.0; preventions at −0.6875/−0.5938, (7,5) leads 8v → picks (7,5) ✓ | 76/77 at −1.0; preventions at −1.0, 2v each → picks (11,5) |

The mechanism is exactly iter-16 §3.1's observation, now
weaponised against the search: in creator-rich positions
nearly every quiet leaf returns −1.0, so nearly every subtree
contains a visited −1.0 line. The MEAN backup preserves the
prevention signal as an average — the prevention's subtree
mixes −1.0 leaves with gradient leaves (−0.75), while
non-preventions average −1.0 — a 0.25 gap the search can
rank on. The MINIMAX backup destroys it: ANY −1.0 leaf in a
subtree sets q = −1.0, so preventions and non-preventions tie
exactly, the v27 tie-break can never fire (all q gaps are 0),
and UCB becomes purely prior-driven. The prior (v1 score)
systematically prefers attacking moves over quiet preventions
(preventions ~0.032 vs attacks ~0.10–0.11), so the pick is
the highest-prior attacking move — the worst possible
fallback for a prevention position. v47 picks the losing move
on g13 and g2 alike.

This is a property of the pre-registered rule, not an
implementation bug (the invariants in §1.4 hold; the
collapse follows from the rule + the leaf distribution). No
redesign is attempted: the rule was pre-registered, the
battery is the verdict on its game strength, and the clause-3
failure stands as recorded. The suite is NOT patched.

## 4. Verdict

(TODO)

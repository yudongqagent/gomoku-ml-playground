# Iteration 40 — v40: must-block VCF upgrade at the leaf start (batch 11, H2)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v40. Fast loop (§2) and battery (§3) follow.** Comparison target is
v27 (the batch-11 bar reference: ladder Elo 636.4, on master at
`128c327`). This is the H2 workstream of batch 11 (H1 is v39,
setup-signature mining; H3 is v41, White-side forensics); work
happens on branch `research/batch-11-h2` in the `gomoku-b11-h2`
worktree, branched from `research/batch-11` @ `2188b53` (which
includes the batch-11 plan). v40 does NOT change the rollout
function, policy, gradient, or either fast-loop patch (v16's,
byte-identical), does NOT change sims/priors/book (CLOSED), does
NOT change exploration constants (CLOSED), does NOT change the
root rule or the tie-break (CLOSED — the tie-break lever is doubly
bounded), and does NOT include the threat check or the VCF-at-root
check (v35 CLOSED VCF-prevention for the v27 family). The ONLY
change from v27 is the must-block VCF upgrade at the leaf start in
§1 — a narrow, exact leaf change targeting the ONE genuine leaf
gap the diagnosis found (§1.1).

**Method (carried over from batch 3/4/5/6/7/8/9/10):** the fast
10-game loop — play 10 games, read EVERY move of every game, find
THE losing move per game (verified with the solver or a deeper
search — never guessed; for rollout-leaf engines, playouts or exact
computation, NOT the v1 walk-back oracle alone — iter-18 §2.1),
patch the algorithm so that, replayed to that critical position,
it plays the perfect move, and record each critical position +
correct move as a regression test. Then play 10 more games and
repeat. Max 2 fast-loop iterations before validation (the
two-consecutive-barren-iterations stop rule, iter-27 §2.1), or
stop when two consecutive iterations find no new patchable
weakness class. Max 1 fast-loop refinement of the pre-registered
design (v35's rule): a refinement must be a LEAF change
addressing a verified leaf-value-error class — it must NOT touch
the root rule, the tie-break, the priors, or the sims (all
CLOSED), and it must NOT be a fitted constant. **The fast loop
runs against v0 as well as v27** (the bar reference is v27 now,
not v7) — each iteration plays 10 games vs v27 AND 10 games vs
v0, and losses from both opponents are walked back. The full
4-clause validation battery (100-game matches + ladder) runs on
the FINAL patched v40 only. Fast-loop games use development seeds
(1040/2040, 1041/2041) distinct from the bar battery's
pre-registered seeds (100/200), so no development game overlaps an
eval game.

## 1. Design and pre-registration

**Comparison target: v27** (primary, the batch-11 reference:
ladder Elo 636.4). The bar's clause 1 is head-to-head vs v27
(normal rules, 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50), clause 2 is ladder Elo
strictly higher than v27's 636.4 (**RE-BASED this batch,
pre-registered in the batch-11 plan: 40 games per opponent per
leg**, up from 20 — the point estimates stay comparable to
historical versions, only the intervals narrow; leg-result
reading for saturation: a 40-0 leg strictly dominates a
39-0-1 leg), clause 3 is tactical pass rate >= v27's on the
CORRECTED suite (old suite 20/20, hard suite 29/29), clause 4
is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v40 vs v0 (100
games) and a 100-game swap match vs v27 under the tempo judge
(the harness default). Non-transitivity is recorded prominently.
**Strength champion:** v18 (h2h vs v7 0.775, ladder Elo 636.4).
Under champion rule v2 (direct challenge, TRACKS.md), a
bar-passer plays a 100-game match against v18 and takes the
title if it wins the match (win rate > 0.50 with Wilson 95% CI
lower bound > 0.50) AND its ladder Elo beats 636.4 — a
bar-passer already clears the Elo clause by passing, so the
challenge match is the deciding test. **Only run the challenge
match for a bar-passer.**

**The improvement: v40 = v27 + the must-block VCF upgrade at the
leaf start.** Everything else is v27 unchanged: 100 sims,
c_puct 1.5, v1 priors over all candidates, win/block scans
before search, most-visited root child, rollout leaf both
colours (leaf mode `rollout2_v1_both_mbvcf` — v18's rollout
leaf, byte-identical, plus the §1.2 upgrade), and the threshold
τ=0.05 on the q gap. The upgrade is the ONLY change from v27.

### 1.1 Diagnosis: the leaf-blind hypothesis is REFUTED at the 1-ply level

The H2 hypothesis (batch-11 plan, from iter-34 §3.6 / iter-35
§2.4): batch 10 H2's flavour — the leaf ranks the unsafe move
best — is a LEAF problem. The diagnosis reproduces all 4
distinct missed-prevention instances from v34's battery
(classifier re-run on this branch reproduces iter-34 §3.6
exactly: 6 clean missed preventions in 454 threatened turns
across the vs-v27 and swap legs, 4 distinct positions, all
v34-as-White) and measures the 1-ply rollout-leaf value of the
unsafe move vs each safe move, the root dump (visits / our-q /
prior), the subtree under each root child, and the rollout trace
step by step. Scripts in `/tmp` (not committed): `diag_v40_leaf.py`,
`diag_v40_subtree.py`, `diag_v40_g61_blunder.py`,
`test_v40_rollout4.py`, `bench_v40_vcf.py`, `bench_v40_mb.py`.
The playout verifications use `ml/scripts/targeted_playout.py`
(v27-vs-v27, seeds 0/1/2).

**The 1-ply leaf does NOT mis-rank — on any instance.** In all
4, the rollout leaf ranks the unsafe move WORST (by a wide
margin) and the safe moves better. The "leaf-blind" label in
iter-34/iter-35 conflated the root-child q with the leaf value
(the quoted numbers, e.g. −0.6364 vs −0.6755…−0.8420, are
root-child our-q values, reproduced exactly here). The
mis-ranking lives in the backed-up root q and the visit
allocation, not in the leaf:

| Instance | Root (v27, seed 0): unsafe vs best safe | 1-ply leaf (ours): unsafe vs safes | Mechanism |
|---|---|---|---|
| vs-v27 g61 ply 21, played (8,1); safes (3,5),(4,5),(8,5) | 11v, −0.6364 vs 3v, −0.6755 (8,5) | −1.0000 vs −0.0133/−0.0521/−0.0265 | **Blunder dilution** (§1.1a) |
| vs-v27 g31 ply 21, played (9,6); safes (8,10),(9,11) | 5v, −0.8117 vs 4v, −0.6875 | −1.0000 vs +0.2500/+0.2500 | **Visit allocation** (§1.1b) |
| swap g73 ply 35, played (11,9); safes (11,5),(10,6) | 5v, −0.3501 vs 3v, −0.5833 | −1.0000 vs +0.2500/+0.2500 | **Unvisited creator + VCF undervaluation** (§1.1c) |
| swap g43 ply 17, played (7,8); safes (8,8),(12,8) | 9v, −0.5556 vs 8v, −0.6356 (8,8) | −1.0000 vs +0.0924/−0.0976 | **Blunder dilution** (§1.1a) |

#### 1.1a Blunder dilution (g61, g43) — an averaging artefact, no leaf patch

g61's subtree under the unsafe (8,1) (11 visits): the expansion
visit returns +1 for Black (the rollout plays the creator
(4,5)); 8 descents go to the creator (q +1.0 for Black); but 2
descents go to Black blunders (9,1) and (11,4), each returning
+1.0 for WHITE. The blunder leaves are CORRECT: (8,1) creates
White's own open-four creators (the (11,4) diagonal and the
(5,1) row), so if Black plays anything but the creator, White
plays her creator and wins at once (rollout-traced: after Black
(9,1), White's rollout ply-0 plays (11,4), an open four; after
Black (11,4), White plays (5,1), an open four). The position is
a mutual open-four race where the player to move (Black) wins by
playing the creator. The backed-up average (−0.6364) mixes 9
creator wins with 2 blunder losses; the safe moves' subtrees
have no such +1 leaves (their 1-ply leaves are ~0 and their
subtrees return −0.68…−0.84 — Black wins slowly via (6,4),
playout-confirmed), so the average flips the ranking by 0.039.
g43 is the same shape (expansion +1, 6 creator visits at +1, 2
blunder visits at −1 for Black; margin 0.080). The blunders get
explored because their priors tie the creator's (~0.100 vs
0.099 — the v1 score cannot distinguish "create an open four"
from "block one of two open-four threats": both score ~200k).
**No leaf patch addresses this:** every leaf value in the tree
is correct; the mis-ranking is MCTS averaging over the
opponent's moves instead of taking the opponent's best move.
Fixing the average is a backup-rule (search) change, not a leaf
change, and is out of scope for this workstream.

#### 1.1b Visit allocation (g31) — q already prefers the safes

In g31 the root q ALREADY ranks the safe moves better (−0.6875
vs −0.8117) — the "leaf-knows" flavour — but the unsafe (9,6)
leads on visits (5 vs 4) because its prior is 3x the safes'
(0.1309 vs 0.0410), and the root rule is most-visited. The
1-ply leaves are decisive (+0.25 safes vs −1.0 unsafe). Fixing
this needs the CLOSED q-based root-decisions lever (promoting a
lower-visit move on q) or a prior change (CLOSED). No leaf
patch addresses it.

#### 1.1c Unvisited creator + VCF undervaluation (g73) — the ONE genuine leaf gap

g73's subtree under the unsafe (11,9) (5 visits): the creator
(12,12) is NEVER visited (its prior does not dominate the VCF
start's, and the VCF start (8,12)'s mildly-positive q (+0.19)
then outcompetes the unvisited creator's pure-exploration UCB
within the 5-visit budget). The visited line, (8,12) (3 visits,
q +0.1884 for Black), is UNDERVALUED: (8,12) is a VCF start
(verified: `starts_forced_win` True), and after White's forced
block (12,12) — which also occupies Black's creator square, so
Black holds 0 open-four creators — Black still holds 3 VCF
starts [(9,13), (11,13), (11,14)] and wins by force. The leaf
after Black's (8,12) (White to move, must-block) is −0.25 under
v27; the truth is −1.0. **This is the genuine leaf gap: the
2-ply rollout cannot see a depth-3 VCF resolve, so a forcing
position is valued as a must-block (−0.25) instead of a forced
loss (−1.0).** The creator-based check would MISS this instance
(0 creators after the block) — only the VCF check catches it,
which is also what distinguishes this patch from v3's
creator-based 2-ply threat check (REJECTED).

#### 1.1d The safe moves do not save the games — the class is outcome-free

Playout verification (v27-vs-v27, seeds 0/1/2) on all 4
instances: the actual move loses 3/3 AND every safe move loses
3/3. g61: actual (8,1) [−1,−1,−1]; (4,5), (8,5), (3,5) all
[−1,−1,−1]. g73: actual (11,9) [−1,−1,−1]; (11,5), (10,6)
[−1,−1,−1]. g31: actual (9,6) [−1,−1,−1]; (8,10), (9,11)
[−1,−1,−1]. g43: actual (7,8) [−1,−1,−1]; (8,8), (12,8)
[−1,−1,−1]. The positions are lost regardless of the
prevention — the safe moves delay the loss (Black wins via a
second threat: g61's (6,4), g73's (10,6) block-then-creator).
This is consistent with v35's null result (fixing ALL missed
preventions gained ~0 games: 0.53 vs v27, CI straddling 0.50).
**The patch's game value, if any, must come from the GENERAL
forcing-sequence pricing (every must-block leaf in every game),
not from fixing these 4 preventions.** The battery decides.

#### 1.1e Refuted candidates (measured before pre-registration)

* **Full VCF enumeration at every leaf — INFEASIBLE on cost.**
  `starts_forced_win` over a position's four-creators costs
  ~1609 ms/position (measured, contended) — 83x the leaf cost.
  At ~100 leaves/move that is ~160 s/move. Dead.
* **4-ply rollout (extend the horizon to see the VCF resolve) —
  REFUTED by prototype** (`test_v40_rollout4.py`,
  monkeypatched `_ROLLOUT_PLIES`): fixes 0/4 instances. g61:
  unsafe still top (10v, −0.60). g31: unchanged. g73: the
  unsafe move's q RISES from −0.35 to −0.1262 (wrong
  direction — the longer rollout lets the weak-policy defender
  find more escapes). g43: unsafe still top, margin widens the
  wrong way. Worse, the 4-ply rollout HALLUCINATES: g61's safe
  (4,5) gets a 1-ply leaf of +1.0000 (a "win") under 4 plies,
  although the move loses 3/3 in playouts — the weak argmax
  policy drifts further from true play over more plies. A
  longer rollout is not the fix (and it costs ~2x/leaf).
* **Creator proxy for the must-block upgrade (upgrade to −1
  when the opponent holds a four-creator after the forced
  block) — REFUTED.** It fails the diagnosed instance (0
  creators after White's block in g73) and it is unsound (a
  four-creator is not a forced win — v3's over-defensiveness
  lesson). Only the exact VCF test is sound.
* **Minimax backup / max-backup for the opponent** (fix the
  blunder dilution by taking the opponent's best move instead
  of the average) — NOT a leaf change; a search/backup change
  out of scope for this workstream. Recorded, not attempted.

### 1.2 The patch (pre-registered, exact)

**v40 = v27 + the must-block VCF upgrade at the leaf start.**
The rule, exactly:

1. At a leaf, if `_terminal_leaf_value` returns the must-block
   value (−0.25) — the opponent holds exactly one winning move
   and we hold none — the block is forced (any other move loses
   at once).
2. Play the forced block (the opponent's single winning square).
3. If the opponent then forces a win by consecutive fours
   (`evaluate.forces_win`, depth 3 — the tactical suite's own
   VCF definition, the same check v29/v35 use at the root),
   the leaf value is −1.0, not −0.25.
4. Otherwise (no forced win after the block), the leaf is
   exactly v27's (`_rollout_leaf_value` — the must-block case
   returns −0.25 there).

The upgrade is EXACT, not heuristic: `forces_win` is True only
when the attacker wins against any defence (defender immediate
wins refute the line inside the search, and a block that wins
for the defender refutes it too). The depth-3 limit makes it
conservative (a VCF longer than 3 attacker moves is missed and
the leaf stays −0.25), never wrong. It applies ONLY at the
leaf start (the real tree position). The rollout END keeps
v16's rule (a must-block there is gradient-evaluated —
iter-16 §2.1: the rollout-end position is a weak-policy
prediction, not a fact, and valuing it −0.25 lets a quiet move
borrow a forcing move's value). The upgrade is a pure leaf
change: it does not touch the search, the priors, the root
rule, or the tie-break.

**Cost (measured, uncontended).** The VCF check runs only on
must-block leaves: 5.5% of leaves in real 100-sim searches
(54/990 over 10 searches), ~104 ms per check
(`forces_win` on 14 sampled must-block positions), upgrading
3/14 (21%) to −1. Expected slowdown ~1.45x per move (100-sim
search ~1.27 s → ~1.84 s on this VM). The battery takes ~1.45x
a check-free version's pace — affordable with auto-resume.

**What would refute the hypothesis.** If v40's battery vs v27
is a statistical tie (win rate ≈ 0.50, CI including 0.50) with
vs-v0 and ladder numbers indistinguishable from v27's, the
must-block upgrade is a no-op in real games: forcing sequences
priced exactly at must-block leaves do not change enough moves
to matter (the upgrade fires on ~5.5% of leaves and upgrades
~21% of those — roughly 1% of leaf evaluations). If v40 is
WORSE than v27 (h2h CI upper < 0.50, or vs-v0/ladder below
v27's), the upgrade's interaction with the tie-break (§1.3)
costs more than the exact pricing gains.

### 1.3 Mechanism check (post-implementation, pre-fast-loop)

After implementation and before any fast-loop game, v40 was run
on the 4 diagnosed instances (fresh-seed-0 root dumps,
`/tmp/mech_v40.py`) and on the full tactical suites at
`--seed 100`. This is an implementation check, not
pre-registration evidence and not an eval — it verifies the
dispatch does what §1.2 says it does.

**The upgrade fires exactly where it should, and nowhere else:**

| Check | v27 | v40 |
|---|---|---|
| g73 (8,12) line leaf (White to move, must-block; Black holds 3 VCF starts after the forced block) | −0.250 | **−1.000** |
| g73 safe (11,5) line leaf (Black to move, must-block; no White VCF after the block) | −0.250 | −0.250 (no fire) |
| 1-ply leaves on all 4 instances (quiet roots) | — | identical to v27 |

**Root dumps on the 4 instances (fresh seed 0):**

| Instance | v27 top move | v40 top move |
|---|---|---|
| g61 | (8,1) unsafe, 11v, −0.6364 | (8,1) unsafe, 11v, −0.6364 — UNCHANGED (by design: blunder dilution) |
| g31 | (9,6) unsafe, 5v, −0.8117 | (9,6) unsafe, 5v, −0.8117 — UNCHANGED (by design: visit allocation) |
| g73 | (11,9) unsafe, 5v, −0.3501 | **(10,6) SAFE, 4v, −0.6875** — FIXED (unsafe drops to 2v, our-q −1.0000) |
| g43 | (7,8) unsafe, 9v, −0.5556 | (7,8) unsafe, 9v, −0.5556 — UNCHANGED (by design: blunder dilution) |

g61/g31/g43 are byte-identical to v27 (same visits, same q,
same pick) — the upgrade does not touch their mechanisms, as
designed. g73 flips: the unsafe move is now valued at −1.0 and
a safe move leads.

**REGRESSION — v1bl-g2 (clause 3 at risk).** Full suites at
`--seed 100`: old 20/20, hard **28/29** — v40 FAILS
prev-v1bl-g2 (chose (7,9), expected (6,5)/(7,5)). Root dumps
(`/tmp/diag_v40_v1bl.py`, fresh seed 0) show the mechanism:
under v27, (7,5) and (7,9) TIE at 8 visits with q gap 0.1306 >
τ=0.05, so the tie-break fires → (7,5) (expected). Under v40,
the upgrade shifts q values in the (7,5) subtree (must-block
leaves there upgrade to −1 for Black — correct, those lines
really are lost), (7,5)'s our-q drops −0.5938 → −0.6429 while
(7,9)'s drops only −0.7244 → −0.7550, and PUCT gives (7,9) the
extra visit (prior 0.1058 vs 0.0335): (7,9) is the UNIQUE
leader at 9 visits vs (7,5)'s 7. No tie exists, so the
tie-break cannot fire, and most-visited picks (7,9) — the
documented loser (v1 played it and lost; the 1-ply leaf is
−1.0). v40's VALUES still prefer (7,5) (our-q −0.6429 vs
−0.7550); the failure is visit allocation, and the tie-break
needs an exact tie to exist. The other two tie-fix positions
(g26, g79) still pass, as does winning-quiet-setup-2 ((9,7)).
**Clause 3 is therefore expected to FAIL at 28/29** (the
battery's official re-run confirms). The battery still runs:
it measures whether exact must-block pricing gains STRENGTH
(clauses 1/2) despite the tactical regression — the v18
precedent (batteried at 25/29, became strength champion).
No fix is attempted: retuning τ cannot help (there is no tie
to fire on), and any root-rule change would reopen the CLOSED
tie-break lever and conflate two mechanisms. The regression is
recorded as an xfail in `test_regression_v40.py`.

**Unit tests (new):** 7 v40 tests in
`ml/tests/test_mcts_heuristic.py` (registry + dispatch, upgrade
fires on the g73 VCF line, no fire without a VCF after the
block, identical to v27 on quiet and terminal positions, no
board mutation, unknown leaf mode rejected) — all pass.
`ml/tests/test_regression_v40.py` (5 positions): 1 passed
(g73 fix), 4 xfailed (g61/g31/g43 out-of-scope mechanisms,
v1bl-g2 regression), plus the well-formedness test.

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v40 only (after the fast loop):

- 100 games vs v27, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100-game swap vs v27, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder **40 games/opponent** (random, heuristic, mcts;
  mcts_sims=100), seed 100, via `ml/scripts/eval_version.py
  --out` (auto-resume; the ladder JSON is a committed
  artifact) (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).
- If — and only if — v40 passes all 4 clauses: a 100-game
  challenge match vs v18 (seeds 100/200, normal rules); the
  coordinator makes the champion call under rule v2.
- Every leg reports its distinct-game count and de-dup rate
  (per-game seeding + uniqueness audit are standard battery
  hygiene from ML batch 3), the clean-missed-preventions rate,
  and a defence audit (all pooled losses via open four, 0
  missed blocks — defence must stay solved).

**Integration.** `ml/gomoku_ml/mcts_heuristic.py` gains
`_rollout_leaf_value_mbvcf` (the §1.2 wrapper — the v18
rollout function itself is untouched), the
`rollout2_v1_both_mbvcf` leaf mode (bound method
`_rollout_v1_both_mbvcf_leaf_value`, validation + dispatch in
`HeuristicMCTS.__init__` in the conditional form — the batch-6
trap: rollout modes are NOT in `_LEAF_FUNCTIONS`), and
docstring entries. `ml/gomoku_ml/scoring.py` gains ONLY the
`ENGINE_VERSIONS[40]` registry entry (v27's entry + the new
leaf mode). The v1 kernel (`kernel_v1.py`), the rollout
function, and v27's dispatch are unchanged. `baseline.py` is
frozen and untouched.

**Game-strength expectation (pre-registered, honest).** The
upgrade fires on ~1% of leaf evaluations (5.5% must-block ×
21% upgrade) and fixes 1 of the 4 diagnosed instances (the
others are outcome-free anyway). Its game value, if any, comes
from pricing forcing sequences exactly at must-block leaves
throughout the tree — a small, diffuse effect, not a
mechanism that wins specific games. **Expected: v40 ≈ v27**
on clause 1 (a tie, failing clause 1 unless the exact pricing
moves enough games), vs v0 ≈ v27's 0.9650, ladder ≈ v27's
636.4 (the 40-game legs narrow the interval; a 40-0 heuristic
leg is unlikely at v27's strength). Clause 3 is expected to
FAIL at 28/29 (§1.3). A tie everywhere is a clean result: the
upgrade is exact but too narrow to matter at 100 sims, and the
leaf-blind direction is exhausted (the remaining mis-rankings
are averaging and visit-allocation, not leaf values). A win vs
v27 with CI excluding 0.50 would be a genuine surprise in the
upgrade's favour. The battery decides.

**Cost (measured).** Fast loop: ~1.5 h for 2 iterations
(1.45x slowdown). Battery: 100 games vs v27 ~3 h, vs v0 ~1.5 h,
swap ~2 h, ladder (40/leg) ~1.5 h, tactical ~1 min.
Affordable with auto-resume.

**Eval seeds (pre-registered):** h2h vs v27: v40=100 / v27=200;
h2h vs v0: v40=100 / v0=200; ladder seed 100, 40
games/opponent, mcts_sims=100; swap 100 games vs v27, balanced
placement, placement_seed 11, tempo judge (the harness
default), seeds 100/200; tactical `--seed 100`. Fast-loop dev
seeds 1040/2040, 1041/2041 (iteration k uses 1040+k-1 /
2040+k-1).

## 2. Loss analysis for iteration 40

### 2.1 Fast-loop iteration 1 (seeds 1040/2040, 10 games vs v27 + 10 vs v0)

**Results:** vs v27 **6W-3L-1D** (win rate 0.6500, 95% CI
[0.3537, 0.8631], 2132s); vs v0 **10W-0L-0D** (1.0000, CI
[0.7225, 1.0000], 1311s). Small samples — the CI on the v27 leg
includes 0.50 — but no regression vs v27's own level, and v0
is swept. Classifier (`classify_losses.py` on both legs): 3
losses, all via open four, **0 missed blocks, 0 clean missed
preventions in 8 threatened turns** — defence stays solved, and
the upgrade introduces no new prevention failures.

**The 3 losses, worked up with `workup_loss.py` (playout oracle,
seeds 0/1/2, last 6-8 turns, top-3/5 1-ply alternatives + the
400-sim MCTS choice):**

| Game | v40 colour | Losing move (earliest saving turn) | Saving move(s) | Class |
|---|---|---|---|---|
| g2 (34 plies) | Black | ply 18 (7,4): actual loses 3/3 | (10,7) 3/3, (12,11) 2/3 (1-ply topk); (12,8) 3/3 (400 sims) | search horizon |
| g5 (27 plies) | White | ply 15 (6,3): actual loses 3/3 | (9,6) 3/3 (400 sims only) | search horizon |
| g9 (51 plies) | White | ply 41 (12,4): actual loses 3/3 | (11,3) [0,1,1] (400 sims only) | search horizon (+ tie-break non-fire) |

In all 3, every later tested turn also qualifies with NO saving
move — each game was lost at the earliest saving turn and never
recoverable after. In all 3, the ONLY saving move (g2 excepted,
where two 1-ply topk moves also save) is the 400-sim MCTS
choice: deeper search finds what 100 sims misses.

**Root-dump diagnosis at the 3 losing positions
(`~/workspace/b11h2-logs/diag_fl1_g2.py`, `diag_fl1_g59.py`):**
in ALL 3, v40's root dump is BYTE-IDENTICAL to v27's (same
visits, same q, same pick) — the upgrade fires nowhere in these
searches, and the losses are not caused by the upgrade. The
positions are all quiet (no winning move either side), and the
1-ply leaf differences are small gradient values, not
categorical errors:

- g2 ply 18: leaf (ours) (7,4) +0.0816 vs (10,7) −0.0521,
  (12,8) +0.0458, (12,11) −0.1166. Root q all near zero
  ((7,4) +0.0094, (12,8) −0.0039). The loss is a slow
  positional squeeze beyond the 2-ply horizon.
- g5 ply 15: leaf (6,3) +0.2500 vs (9,6) −0.0000 — the leaf
  prefers the losing move, but by a rollout-end gradient in a
  quiet position, with no must-block/VCF/terminal signature
  to patch. Root: (6,3) 11v −0.0961 vs (9,6) 3v −0.1007 (q
  essentially tied; the visit gap is prior-driven: 0.0566 vs
  0.0329).
- g9 ply 41: leaf (12,4) −0.1723 vs (11,3) −0.1771 — the leaf
  does not distinguish them (0.005 apart). Root: (10,4) and
  (12,4) TIE at 4 visits with q gap 0.0128 < τ=0.05, so the
  tie-break does NOT fire and the RNG picks the loser (12,4).
  The saving (11,3) is not in the top 8. This is the CLOSED
  tie-break lever plus search horizon.

**Verdict: iteration 1 is BARREN for patchable leaf classes.**
All 3 losses are the search-horizon class (v35's forensics:
10 of 22 v27 losses) — the leaf has no categorical error at
any losing position, and the only fixes are more sims
(CLOSED: not monotonic) or root-rule changes (CLOSED). No
refinement is made. The fast loop continues to iteration 2
(fresh seeds 1041/2041) per the two-consecutive-barren-
iterations stop rule.

(Note: the first workup run was killed by a VM replacement at
~53 min, mid-game-2; the game-2 finding above is from the
surviving log `~/workspace/b11h2-logs/workup-fl1.log`, and
games 5/9 were worked up in a second run with
`--last-turns 6 --topk 3`, logs `workup-fl1-g59.log`,
JSONs `workup-fl1-g5.json` / `workup-fl1-g9.json`.)

### 2.2 Fast-loop iteration 2 (seeds 1041/2041)

**Results:** vs v27 **4W-4L-2D** (win rate 0.5000, 95% CI
[0.2366, 0.7634], 1933s); vs v0 **10W-0L-0D** (1.0000, CI
[0.7225, 1.0000], 1163s). Combined fast loop: vs v27 10W-7L-3D
(0.575), vs v0 20W-0L-0D. Classifier: 4 losses, all via open
four, **0 missed blocks, 0 clean missed preventions in 21
threatened turns** — defence stays solved across both
iterations (29 threatened turns total, 0 preventions missed).

**The 4 losses, worked up with `workup_loss.py` (same settings
as iteration 1):**

| Game | v40 colour | Earliest saving turn | Saving move(s) | Class |
|---|---|---|---|---|
| g0 (44 plies) | Black | none in window (lost before ply 32) | none (6/6 qualify, 0 save) | search horizon (already lost) |
| g1 (33 plies) | White | none in window (lost before ply 21) | none (6/6 qualify, 0 save) | search horizon (already lost) |
| g5 (27 plies) | White | ply 17 | (5,7) 3/3 (1-ply topk) | search horizon (leaf is RIGHT) |
| g9 (31 plies) | White | ply 21 | (4,8) 3/3 (400 sims only) | search horizon |

In g0 and g1 every tested turn qualifies (actual move loses
3/3) and NO alternative saves — not even the 400-sim choice
(g0 ply 38: (5,4) loses 3/3). The games were already lost
before the earliest tested turn; the losing move lies earlier
than the 6-turn window, i.e. a slow positional loss with no
single identifiable blunder in the window. (A VM replacement
killed the first g5/g9 workup run mid-game-5; the rerun
completed both, logs `workup-fl2-g59.log`, JSONs
`workup-fl2-g5.json` / `workup-fl2-g9.json`.)

**Root-dump diagnosis at g5 ply 17 and g9 ply 21
(`~/workspace/b11h2-logs/diag_fl2_g59.py`):**

- g5 ply 17: the leaf is RIGHT — 1-ply leaf (ours) (9,4)
  −0.0133 vs saving (5,7) **+0.2500**. But the backed-up root
  q for (5,7) is −0.2314 (5 visits) vs (9,4) −0.0083 (29
  visits): the deeper tree's weak-policy rollouts destroy the
  advantage the leaf sees. A rollout-policy/search failure,
  not a leaf-value error — no leaf patch addresses it. The
  root also shows a 29-29 tie between the two losing moves
  (9,4) and (5,8) with q gap 0.0030 < τ (tie-break cannot
  fire; the game seed's RNG picked (9,4)). **The upgrade
  fired in a real game subtree here:** (4,8)'s our-q is
  −0.1735 under v40 vs −0.0485 under v27 — the only
  root-dump difference between the versions in all 5
  diagnosed fast-loop positions. It did not change the pick.
- g9 ply 21: v40 and v27 byte-identical. The search is
  confidently wrong: (7,5) leads 37 visits at our-q +0.1962
  while the saving (4,8) is not in the top 8. The leaf does
  not distinguish them ((7,5) +0.0067 vs (4,8) −0.0067) —
  a horizon failure, not a leaf error.

**Verdict: iteration 2 is BARREN.** Two consecutive barren
iterations → the fast loop STOPS (per the §1 stop rule).
v40 goes to the battery as designed, with NO refinement. The
fast loop's combined record: vs v27 10W-7L-3D (0.575), vs v0
20W-0L-0D; 0 missed blocks and 0 clean missed preventions in
29 threatened turns across both iterations.

## 3. v40 evaluation

(TODO — the battery.)

## 4. Bar check (v40 vs v27)

(TODO — the battery.)

## 5. What this means

(TODO — the battery.)

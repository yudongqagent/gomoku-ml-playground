# Iteration 35 — v35: v27 loss forensics → targeted patch (batch 10, H3)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v35. Forensics (§2), patch or fallback (§3), fast loop (§4), and
battery (§5) follow.**

**METHOD (carried over).** The fast 10-game loop stays the finding
method: play 10 games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed — for rollout-leaf
engines, playouts or exact computation, NOT the v1 walk-back oracle
alone — iter-18 §2.1), patch the algorithm to reproduce the perfect
move in those positions (regression test), repeat — max 4
iterations, or stop when an iteration finds no new patchable
weakness class. The 4-clause bar stays the validation gate.
Fast-loop games use development seeds distinct from the bar
battery's pre-registered seeds (100/200). **Fast loops run against
v0 as well as v27** (the bar reference is v27 now, not v7). The
full battery runs on the FINAL version only. The 100-game battery
is the verdict — small-sample smokes mislead; single-position
diagnostics cannot validate a global search change; the v1
walk-back oracle shares the argmax-of-sum flaw (iter-18) — every
nominated save needs playouts or exact computation.

## 0. The problem

v27 (batch 9, H1: v23 + the τ=0.05 q-gap threshold on the
visit-tie break) is the heuristic track's bar reference (ladder
Elo 636.4, vs v7 0.7900, vs v0 0.9650, swap vs v7 0.7750 — all
program bests or ties). It is also the strongest version ever
built that still loses games: **19 losses vs v7 and 3 vs v0**
over its 200-game normal-rules battery, plus 21 swap losses vs
v7. This workstream reads the 22 normal-rules losses (full
per-game move records in `research/iterations/batch-v27-vs-v7.json`
and `batch-v27-vs-v0.json`, on master via the batch-9 merge),
finds each game's losing move, and looks for a SHARED mechanism
across losses. If one exists, the patch IS v35.

**The live possibility that nothing is patchable.** v27's own
fast loop (iter-27 §2, 40 games, 4 losses) found NO patchable
weakness class: all 4 losses were distributed drift, the one
threshold fire in the loop separated two verified losers, and
the only move with any saving signal sat one visit below the
tie — promotable only by the CLOSED q-based root-decisions
lever. The battery corpus (22 losses) is 5.5x larger, so a
shared mechanism may emerge that the fast loop could not see —
but "distributed drift, no shared mechanism" is a live outcome,
and if the forensics concludes that, the pre-registered fallback
(§1.2) triggers. A negative forensics result is a program
finding, not a failed iteration.

**Base engine (forensics patch).** v27 exactly as on master
(100 sims, c_puct 1.5, sharp v1 priors, rollout leaf both
colours, τ=0.05 threshold tie-break, most-visited root). A
forensics patch builds on v27, not on v18 — v27 is the bar
reference, and the patch must beat IT.

## 1. Design and pre-registration

**Comparison target: v27** (the bar reference, batch-10 plan:
ladder Elo 636.4). Clause 1 is head-to-head vs v27 (normal
rules, 100 games, alternating colours, win rate > 0.50 with
Wilson 95% CI lower bound > 0.50); clause 2 is ladder Elo
strictly higher than v27's 636.4 (identical 20 games/opponent:
random, heuristic, mcts, mcts_sims=100) — note this effectively
requires a 20-0 heuristic leg, since 636.4 is the estimate from
a 19W-0L-1D leg; clause 3 is tactical pass rate >= v27's
(20/20 old suite + 29/29 hard suite, on the CORRECTED suite);
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v35 vs v0
(100 games) and a 100-game swap match vs v27 under the tempo
judge (the harness default). **Strength champion:** v18
(ladder Elo 636.4). Champion rule v2 (TRACKS.md, direct
challenge): IF v35 passes all 4 clauses, it plays a 100-game
challenge match vs v18 (seeds 100/200, normal rules) and takes
the title only if it wins that match (win rate > 0.50 with
Wilson 95% CI lower bound > 0.50) AND its ladder Elo beats
636.4 (automatic for a bar-passer). The coordinator makes the
champion call; this log reports the challenge match if it runs.

### 1.1 Phase 1 — forensics (pre-registered protocol)

**Corpus:** v27's 22 normal-rules battery losses — 19 vs v7
(8 as Black, 11 as White; 21–65 plies) and 3 vs v0 (all as
Black; 36–60 plies). The 21 swap losses vs v7 are NOT in the
corpus (different rules, different opening phase; the swap
match is a secondary metric, and mixing corpora would confound
the mechanism search). Defence classification is already known
(iter-27 §3.6, re-verified with `classify_losses.py` on this
branch before §2: 22/22 via open four, 0 missed blocks, 0 clean
missed preventions in 83 threatened turns) — defence is solved,
so the mechanism (if any) is NOT a missed block or a missed
prevention in the classifier's sense.

**Workup instrument:** `ml/scripts/workup_loss.py` — same-engine
playout oracle (v27 vs v7, resp. v27 vs v0), seeds 0/1/2, the
loser's last 4 turns, top-5 1-ply alternatives + the 400-sim
deeper-search proposal + any VCF alternative. A turn
QUALIFIES only when the actual move's playout loses at ≥2/3
seeds; an alternative SAVES only when it draws or wins at ≥2/3
of the SAME seeds (the corrected method, iter-6 §2.2(a)). The
v1 walk-back oracle (`walkback_loss.py`) is NEVER used alone
to verify a save — it shares the argmax-of-sum flaw (iter-18)
and iter-27 §2.1 shows it inventing false saves in v27's own
games (ply 13 of fast-loop game 1: the v1 oracle's saves lose
3/3 under v7-vs-v7). Every nominated losing move is verified
with same-engine playouts or exact computation. Workup JSONs
go to `~/workspace/b10h3-logs/workup-v27-*.json` (outside the
repo; the log records the findings).

**Classification scheme** (iter-29 §1.1's, carried over — a
loss with a verified losing move is classified by the ROOT
CAUSE of that move, not by the loss's surface):

1. **Missed block** — a defensive error (classifier flag or
   exact enumeration).
2. **Missed forced win** — a VCF/forced win existed at a v27
   turn (verified by `evaluate.starts_forced_win` or exact
   enumeration) that v27 did not play, and the game was lost
   afterwards.
3. **Horizon loss** — no single losing move: the workup finds
   no saving move at any of the last 4 v27 turns (distributed
   drift into an unstoppable open four).
4. **Prior starvation** — the saving move is verified by
   playouts but its root prior is low and it received 0–1
   visits at the root (diagnosed with a root dump at the
   losing position).
5. **Leaf value error** — the saving move is verified by
   playouts, its prior is adequate, but the root q/visits rank
   the losing move above the save (diagnosed with a root dump
   at the losing position).
6. **Root-decision (visits-over-q; scheme gap)** — the saving
   move is verified, q prefers it, but visits rank the loser
   above it (iter-29 §2.5's addition).
7. **Search-horizon value error** — the saving move is
   verified, but q/visits/leaf rank the loser above it or are
   neutral, with no exact threat signature distinguishing the
   save (iter-29 §2.5's addition).

**Root dumps** (`ml/scripts/diag_root.py` or a root-dump
script) are run at every verified losing position to diagnose
which of classes 4–7 applies: visits, q (correctly signed,
our-q = `−child.q`), priors, and the 1-ply rollout-leaf values
of loser vs save. **Exact threat analysis**
(`ml/scripts/targeted_playout.py` — opponent VCF starts /
open-four creators / setup moves on the resulting board) is
run on every verified save to test whether an exact signature
(VCF-prevention, creator-minimisation, setup-minimisation)
distinguishes it from the loser.

**Patch rule.** If a class has ≥3 verified instances sharing
ONE mechanism AND a patch reproduces the perfect move at those
positions (regression test), the patch IS v35, on the v27 base.
The patched engine is fast-looped (vs v0 AND v27, 10 games
each, dev seeds 1035/2035 and 1036/2036) and then runs the
full battery (§1.3). Max 2 patch iterations: if the first patch
fails to reproduce the saves at the verified positions, ONE
refinement is allowed; if that fails too, the fallback (§1.2)
triggers. A patch that needs a free parameter fitted to the
forensics positions (e.g. a threshold tuned on 2–3 positions)
is NOT allowed — the mechanism must be principled (a feature,
a rule, or a search change), not a fitted constant. A patch
that reopens a CLOSED lever (q-based root decisions as a
general mechanism, root-decision for strength, White
search/prior/book, exploration tuning, gap-three demotion in
any combination, the blend, combination-stacking, more sims)
is NOT allowed — the mechanism must be genuinely new.

**Extension rule.** If after all 22 losses no single class has
≥3 verified instances, the forensics concludes "no patchable
mechanism found" and the fallback (§1.2) triggers. The corpus
is the complete battery loss set — there is no second corpus
to extend into (the swap losses are a different ruleset and
were excluded above by design, not by oversight). The
classification table is still committed in full in §2.

**Negative-result rule.** If the forensics finds nothing
patchable, the classification table is still committed in full
in §2 — a negative forensics result is a program finding, not
a failed iteration.

### 1.2 Phase 2 — fallback: cheaper VCF check for v29

Triggers ONLY if Phase 1 finds nothing patchable under the
rules of §1.1. It does NOT trigger merely because the
forensics is slow or the patch is hard — the patch rule's
two-iteration limit is the trigger.

**Base: v29** (v18's 2-ply rollout leaf in both colours +
the VCF-at-root check, 100 sims, c_puct 1.5, sharp v1 priors,
most-visited root) — exactly as on master. The fallback
changes ONLY the VCF check's cost, not its design: the
offensive VCF step and the defensive prevention step keep
their exact semantics (depth-3 VCF definition, highest-prior
selection), and the engine is otherwise exactly v29.

**The cheaper check (exact, pre-registered).** Two changes to
`HeuristicMCTS._vcf_root_move`, both in the defensive step
(the expensive part — the offensive step tests only our few
four-creators; the defensive step scans all candidates, and
the no-prevention full scan is the 36s worst case, iter-29
§3.2):

1. **Defensive candidate limit (K=8).** Test only the top-8
   candidates by (prior desc, flat index asc) as preventions.
   If none of the top 8 prevents, fall through to the search
   instead of scanning the remaining candidates. Rationale:
   the selected prevention is the highest-prior one, so when
   a prevention exists in the top 8 the move is IDENTICAL to
   v29's; a prevention ranked below 8th by prior is a move
   the search would rarely pick anyway, and falling through
   to the search there is the safe default. This bounds the
   defensive scan to 8 candidates (from ~30+), cutting the
   worst case roughly 4x.
2. **VCF-starts-only re-check.** On each resulting board,
   test only the opponent's ORIGINAL VCF starts (`opp_vcf`),
   not all four-creators (`ordered = opp_vcf + others`).
   Rationale (monotonicity): our move adds our stone, which
   can only REMOVE the opponent's forcing options — a
   four-creator that was not a VCF start on the original
   board cannot become one after our move (the opponent's
   subsequent four-creators are a subset, and the defender's
   block options for a genuine four are unchanged — a block
   square our stone occupies means the four is already
   blocked). So testing the non-VCF four-creators is
   redundant. This is a pure speedup IF the monotonicity
   argument holds — verified empirically before any eval
   (mechanism check, §3.0): the cheaper check must produce
   IDENTICAL moves to v29's check on all 7 smoke-test
   positions (iter-29 §3.1) and all 49 tactical-suite
   positions, plus a 100-position sample of quiet v7
   self-play positions (the §3.2 bench set), wherever the
   candidate limit does not bind. Any disagreement is
   recorded and the monotonicity claim is withdrawn (the
   candidate limit alone is the fallback then).

**Why NOT depth-limiting.** The batch-10 plan allows
"depth/candidate-limited". Depth-limiting (depth 2 instead of
3) changes the exact VCF definition: a prevention that removes
all depth-2 VCF starts but leaves a depth-3 VCF start live
would be wrongly played as a prevention, and the three
tactical fixes (g26, g79, v1bl-g2) may need depth 3. The
candidate limit + monotonicity re-check keep the depth-3
definition exact, so the tactical fixes are preserved by
construction (verified by the mechanism check and the battery
clause 3).

**What the fallback is and is not.** It is a COST reduction
for v29 (target: mean root-check time well below v29's 1.65s
on the bench set, worst case bounded), keeping v29's tactical
29/29. It is NOT expected to beat v27 on strength — v29
scores 0.715 vs v7 and Elo 240.8, well below v27's 0.7900 /
636.4, and a cheaper check does not add strength. The battery
measures it honestly against the v27 bar; the cost numbers
are the finding. If the fallback triggers, v35 = v29 +
cheaper check, registered as version 35 with a `vcf_root`
variant flag (e.g. `vcf_root_cheap`), and the fast loop runs
vs v0 AND v27 as usual.

**Cost guard.** Median/p90/max root-check time measured on
the 100-position bench set (`ml/scripts/bench_vcf_root.py`)
before the battery and reported in §3; no design change is
made after any eval on the basis of cost alone.

### 1.3 Eval seeds (pre-registered)

- Battery h2h vs v27: v35=100 / v27=200; vs v0: v35=100 /
  v0=200.
- Ladder: seed 100, 20 games/opponent, mcts_sims=100
  (`ml/scripts/eval_version.py --version 35 --ladder-games 20
  --out research/iterations/ladder-v35.json` — the auto-resume
  driver, new on `research/batch-10`).
- Swap: 100 games vs v27, balanced placement, placement_seed
  11, tempo judge (the harness default), seeds 100/200.
- Tactical: `eval_tactical_full.py --version 35 --suite both
  --seed 100`.
- Challenge match (ONLY if v35 passes all 4 clauses): 100
  games vs v18, seeds 100/200, normal rules.
- Fast-loop dev seeds (patched engine): 1035/2035 (vs v0),
  1036/2036 (vs v27) — distinct from the battery's 100/200
  and from batch-9's loop seeds.
- Harness commands follow iter-27 §1.4 / iter-29 §1.3 exactly
  (`selfplay_batch.py`, `eval_version.py` for the ladder,
  `eval_tactical_full.py` for tactical). All match drivers
  AUTO-RESUME from partial `--out` files — if a run is
  killed, relaunch the SAME command.

## 2. Forensics (Phase 1)

### 2.0 Defence classification (all 22 losses)

`classify_losses.py` on both battery batches (re-run on this
branch before §2.1): **22/22 losses via open four, 0 missed
blocks, 0 clean missed preventions in 83 threatened turns**
(vs v7: 19/19, 69 threatened turns; vs v0: 3/3, 14 threatened
turns) — reproducing iter-27 §3.6 exactly. Defence is solved;
the mechanism (if any) is not a missed block or a missed
prevention in the classifier's sense.

### 2.1 Workups (running — 10/22 complete at this writing)

Same-engine playout oracle (`workup_loss.py`, v27 vs v7 / v27
vs v0, seeds 0/1/2, last 4 turns, top-5 + 400-sim + VCF
alternatives). Workup JSONs in `~/workspace/b10h3-logs/`.
Each completed workup is diagnosed with a root dump
(`diag_root.py`, 100 sims, seed 0), the rollout-leaf 1-ply
values (`/tmp/diag_rollout.py`), and exact threat analysis
(`targeted_playout.py --no-playout`).

| Game | v27 | Plies | Losing move (verified) | Save (playout) | Exact class of the save |
|---|---|---|---|---|---|
| 4 (vs v7) | B | 50 | ply 44 (3,5) | (13,7) 3/3, (14,7) 2D+1W | **clean VCF-prevention (VCF 3→0)** |
| 7 (vs v7) | W | 31 | none in window | — | horizon loss |
| 10 (vs v7) | B | 52 | ply 44 (7,10) | (3,6) 1W+1D+1L | **clean VCF-prevention (VCF 1→0, creators 0, setups 0)** |
| 15 (vs v7) | W | 23 | none in window | — | horizon loss |
| 21 (vs v7) | W | 41 | ply 33 (11,6) | (5,7) 1W+1D+1L, (5,6) 1W+1D+1L | **clean VCF-prevention (VCF 3→0)** |
| 30 (vs v7) | B | 24 | none in window | — | horizon loss |
| 32 (vs v7) | B | 32 | ply 24 (10,2) | (6,6) 2W+1D | VCF-prevention — but the check picks the LOSING prevention (10,2) by prior noise (see below) |
| 35 (vs v7) | W | 55 | none in window | — | horizon loss |
| 44 (vs v7) | B | 48 | ply 40 (12,10) | (10,13) 1W+2D, (9,12) 2W+1L, (11,10) 3W | **clean VCF-prevention (VCF 1→0)** |
| 77 (vs v7) | W | 21 | ply 13 (7,6) | (7,4) 3/3, (8,7) 3/3, (7,5) 2/3 | none (saves leave MORE setups, or identical counts) |

**Game 4, ply 44 (Black to move) — missed VCF prevention.**
The actual (3,5) loses 3/3; (13,7) WINS 3/3 and (14,7) draws
2/3 + wins 1/3. Exact analysis: after (3,5) White holds **3
VCF starts** [(10,5), (11,6), (13,7)], 0 creators, 4 setups;
after (13,7) White holds **0 VCF starts, 0 creators, 0
setups**; after (14,7) 0 VCF starts, 0 creators, 1 setup.
Both saves are clean VCF-preventions. The root is extremely
flat (94 children, top move 4 visits): (3,5), (13,7), (14,7)
tie at 4 visits; our-q prefers the LOSER (−0.032 vs −0.071
for (13,7)), gap 0.039 < τ=0.05, so the threshold correctly
does not fire and the choice is RNG over the tied triple —
and even if it fired, it would pick the loser (higher q).
The rollout leaf ranks a DIFFERENT loser highest: (10,5)
+0.1928 (loses 3/3), (14,7) +0.1433, (13,7) +0.1328, actual
(3,5) +0.0924. No root or leaf measure separates the saves
from the losers — but the exact VCF signature does: the
v29 defensive check would play (13,7) here (opponent holds
VCF starts; (13,7) is the highest-prior prevention, prior
0.0536 vs (14,7)'s 0.0496; (10,5), prior 0.0504, is tested
second but is not a prevention — it leaves (13,7) live).
**This is the VCF-prevention mechanism: 1 verified instance.**

**Game 7 (White, 31 plies) — horizon loss.** The workup finds
no saving move at any of the last 4 White turns (4
qualifying turns, 0 saves). Class 3.

**Game 10, ply 44 (Black to move) — missed VCF prevention
(2nd instance).** The actual (7,10) loses 3/3; (3,6) SAVES
[1, 0, −1] (2/3 non-losses). Exact analysis: after (7,10)
White holds **1 VCF start** [(4,6)], 0 creators, 1 setup
[(3,6)]; after (3,6) White holds **0 VCF starts, 0 creators,
0 setups** — the ONLY clean prevention among the tested
alternatives ((1,9), (0,9), (7,4) remove the VCF start but
leave a creator + 7–9 setups, and all lose 3/3; (4,6)
occupies White's VCF square but creates TWO new VCF starts
and loses 3/3). The root here is the OPPOSITE of g4's flat
tie: the search is CONFIDENT in the loser — (7,10) holds 96
visits, our-q +0.595 (breadth ours four=10 after the move),
while the save (3,6) holds 1 visit, our-q +0.195. The search
sees Black's attack and rates it winning; it cannot see that
White's VCF (3+ plies deep) lands first. The rollout leaf
cannot distinguish them either ((7,10) +0.1933 vs (3,6)
+0.1951), and ranks the losers (1,9)/(0,9) highest (+0.2500,
the must-block value). The v29 defensive check DOES
reproduce the save here: `_vcf_root_move` returns (3,6)
(the highest-prior prevention — the partial preventions rank
below it by prior). **VCF-prevention mechanism: 2 verified
instances, both reproduced by the VCF check.**

**Game 15 (White, 23 plies) — horizon loss.** No saving move
at any of the last 4 White turns (3 qualifying turns, 0
saves). Class 3 (2nd horizon loss).

**Game 21, ply 33 (White to move) — missed VCF prevention
(3rd instance — the patch rule triggers).** The actual
(11,6) loses 3/3; (5,7) and (5,6) both SAVE [1, 0, −1].
Exact analysis: after (11,6) Black holds **3 VCF starts**
[(3,5), (4,5), (5,7)], 0 creators, 2 setups; after (5,7)
or (5,6) Black holds **0 VCF starts, 0 creators** (3
setups) — both are clean VCF-preventions. The root is flat
(103 children, top 6 visits): (11,6) and (5,7) tie at 6
visits; our-q prefers the loser (+0.005 vs −0.026), gap
0.031 < τ=0.05, so the threshold does not fire and RNG over
the tied pair picked the loser (and firing would pick the
loser anyway — higher q). The rollout leaf DOES prefer both
saves ((5,7) +0.1328, (5,6) +0.1026 vs actual +0.0702) but
the backed-up tree q does not — the tree below the saves
looks bad within the 2-ply horizon. The v29 defensive check
reproduces the save: `_vcf_root_move` returns (5,7).
**VCF-prevention mechanism: 3 verified instances, all
reproduced by the VCF check. The patch rule (§1.1) is
satisfied: the patch IS v35 (see §3). The remaining
workups still run to complete the classification table and
to check for other classes (in particular missed forced
wins, which would bear on whether the patch includes the
offensive VCF step).**

**Game 30 (Black, 24 plies) — horizon loss.** No saving move
at any of the last 4 Black turns (4 qualifying turns, 0
saves). Class 3 (3rd horizon loss).

**Game 32, ply 24 (Black to move) — VCF-prevention where the
check picks the WRONG prevention.** The actual (10,2) loses
3/3; (6,6) SAVES [1, 0, 1]. White holds 2 VCF starts on the
current board, [(10,2), (6,6)] — and BOTH (10,2) and (6,6)
are preventions (each leaves White with 0 VCF starts, 0
creators, identical setup sets). The check picks (10,2):
its prior (0.1381) exceeds (6,6)'s (0.1380) by **1e-4** —
prior noise, the same flaw as iter-29 §3.4's 3e-6 case
(there the check's move lost 3/3 while the search's won
3/3; here the check's move loses 3/3 while the other
prevention saves 3/3). The search ALSO prefers (10,2) (7
visits vs 5, our-q −0.321 vs −0.459), so VCF-filtered search
(H1's direction) would likely pick (10,2) here too. This is
the 4th VCF-prevention instance, and the check does NOT
reproduce the save at it. Recorded honestly: the check
fixes 3 of the 4 VCF-prevention instances in the corpus.
The patch rule (§1.1) requires the patch to reproduce the
perfect move at the ≥3 instances that establish the
mechanism (g4, g10, g21 — it does); g32 is a documented
limitation of the prior-based prevention selection, not a
refutation of the mechanism.

**Game 35 (White, 55 plies) — horizon loss.** No saving move
at any of the last 4 White turns (3 qualifying turns, 0
saves). Class 3 (4th horizon loss).

**Game 44, ply 40 (Black to move) — missed VCF prevention
(4th fixed by the check).** The actual (12,10) loses 3/3;
(10,13) SAVES [0, 1, 0]. White holds 1 VCF start [(10,13)]
on the current board; (10,13) occupies it (a clean
prevention — 0 VCF starts, 0 creators after). (9,12) and
(11,10) also save at ply 40 ((11,10) wins 3/3 — it is a
four-creator like the loser, and the difference between
the two forcing moves is beyond the search's horizon);
(9,12) also saves at ply 42. The actual
(12,10) ALSO removes White's VCF start (it is a prevention
by the check's definition too) but loses: it is a forcing
move (creates a four, rollout leaf +0.2500, the must-block
value) that White answers while keeping a stronger
counter-threat. The search prefers the forcing move (8
visits, our-q −0.044, best q among the top) over the quiet
prevention (5 visits, our-q −0.088, rollout leaf −0.0871).
Here the prior DOES distinguish the save: (10,13) holds
the highest prior (0.0551), so the check returns it — the
opposite of g32, where the priors were noise-tied. The
check reproduces the save at 4 of the 5 VCF-prevention
instances so far.

**Game 77, ply 13 (White to move) — search-horizon value
error, no exact signature.** The actual (7,6) loses 2/3;
(7,4) and (8,7) win 3/3, (7,5) wins 2/3. The root is flat
(83 children, top move 6 visits): actual 6 visits, our-q
−0.007; (8,7) 5 visits, our-q +0.010 (the BEST q — but one
visit short of the tie, promotable only by the CLOSED q-based
root-decisions lever); (7,4) 4 visits, our-q −0.034; (7,5)
2 visits, our-q −0.073. The saves are heterogeneous in q.
The rollout leaf prefers two saves over the actual ((7,4)
+0.1026, (8,7) +0.0871 vs actual +0.0643) but ranks (7,5)
lowest (−0.0133). Exact analysis: NO signature — the loser
leaves 0 VCF starts, 0 creators, 2 setups; the saves (7,4)
and (8,7) leave MORE setups (3), and (7,5) is identical to
the loser on every count. The 400-sim deeper-search proposal
(7,9) loses 3/3 — deeper search does not find the save.
Class 7 (search-horizon value error), no patchable signature.

**Running tally: VCF-prevention 5 (check reproduces the save
at 4, picks the losing prevention at 1), horizon 4,
search-horizon (no signature) 1.** The patch rule (§1.1)
is satisfied at 3 VCF-prevention instances with the patch
reproducing all three saves (g4, g10, g21). The remaining
workups complete the classification table.

## 3. Patch / fallback (v35)

TODO.

## 4. Fast loop

TODO.

## 5. Battery

TODO.

## 6. Bar check

TODO.

## 7. Verdict

TODO.

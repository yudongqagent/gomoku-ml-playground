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

### 2.1 Workups (running — 21/22 complete at this writing)

**Runtime interruption (recorded, no workups lost).** The VM
was replaced mid-chain (uptime reset; the chain died during
game 62's workup, after its ply-16 alternatives had printed
but before the JSON was written). The chain was relaunched
with the same script, which skips completed JSONs and
resumed at g62. The resumed workup reproduced ply 16
exactly ((3,5) saves [1,1,-1]) and found a second save
there, (4,4) [1,1,-1], which the first run had not reached
before the kill.

Same-engine playout oracle (`workup_loss.py`, v27 vs v7 / v27
vs v0, seeds 0/1/2, last 4 turns, top-5 + 400-sim + VCF
alternatives). Workup JSONs in `~/workspace/b10h3-logs/`.
Each completed workup is diagnosed with a root dump
(`diag_root.py`, 100 sims, seed 0), the rollout-leaf 1-ply
values (`~/workspace/b10h3-logs/scripts/diag_rollout.py`), and exact threat analysis
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
| 45 (vs v7) | W | 23 | none in window | — | horizon loss |
| 56 (vs v7) | B | 62 | none in window | — | horizon loss |
| 62 (vs v7) | B | 24 | ply 16 (4,5) | (3,5) 2W+1L, (4,4) 2W+1L | none (loser, saves, and other loser all 0/0/0) |
| 65 (vs v7) | W | 41 | none in window | — | horizon loss |
| 67 (vs v7) | W | 65 | none in window | — | horizon loss (by ply 57 Black holds 5 VCF starts; no prevention exists) |
| 73 (vs v7) | W | 31 | ply 23 (3,8) | (8,5) 2W+1L | **clean VCF-prevention (VCF 1→0, setups 3→1)** |
| 91 (vs v7) | W | 29 | ply 23 (5,5) | (8,5) 2W+1L | **clean VCF-prevention — SAME position as g73 ply 23 (transposition; see note)** |
| 93 (vs v7) | W | 57 | ply 49 (8,5) | (5,4) 3W (400-sim proposal) | none (save leaves MORE setups, 6 vs 4) |
| 96 (vs v7) | B | 30 | none in window | — | horizon loss |
| 4 (vs v0) | B | 60 | ply 52 (11,3) | (9,4) 3W, (5,8) 2W+1D (400-sim) | setup signature (2→0) — see note |
| 32 (vs v0) | B | 60 | ply 52 (3,11) | (4,9) 3W | setup signature (2→0) — TRANSPOSE of v0-g4 ply 52 |
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

**Game 45 (White, 23 plies) — horizon loss.** No saving move
at any of the last 4 White turns (3 qualifying turns, 0
saves). Class 3 (5th horizon loss).

**Game 56 (Black, 62 plies) — horizon loss.** No saving move
at any of the last 4 Black turns (4 qualifying turns, 0
saves). Class 3 (6th horizon loss).

**Game 62, ply 16 (Black to move) — search-horizon value
error, no exact signature (2nd).** The actual (4,5) loses
3/3; (3,5) SAVES [1, 1, −1]. The position is completely
quiet: no four-creators and no VCF starts for either side,
and the loser, the save, and the other loser (6,5) are
IDENTICAL on every exact count (0 VCF starts, 0 creators,
0 setups after each). The root is flat (75 children, top
6 visits, 4-way tie): the tie-break does not fire (q gap
0.005 < τ=0.05) and RNG over the tied four picked (4,5);
the save sits one visit below the tie (5 visits, our-q
−0.016) — promotable only by the CLOSED q-based
root-decisions lever. The rollout leaf ranks the loser
(6,5) highest (+0.0583), the save second (+0.0458), the
actual lowest (+0.0199). No root, leaf, or exact measure
distinguishes the save. Class 7. The VCF check would not
fire here (no VCF starts anywhere).

**Game 65 (White, 41 plies) — horizon loss.** No saving move
at any of the last 4 White turns (4 qualifying turns, 0
saves). Class 3 (7th horizon loss).

**Game 67 (White, 65 plies) — horizon loss.** No saving move
at any of the last 4 White turns (4 qualifying turns, 0
saves). By ply 57 Black holds **5 VCF starts** (all five of
its four-creators) and no single White move removes them —
v35's check returns None there and falls through to the
search, playing (6,10) like v27. The loss was decided
earlier or is distributed drift. Class 3 (8th horizon loss).

**Game 73, ply 23 (White to move) — missed VCF prevention
(5th fixed by the check).** The actual (3,8) loses 3/3;
(8,5) SAVES [1, 1, −1]. Black holds 1 VCF start [(8,5)]
on the current board. Both (3,8) and (8,5) are preventions
(0 VCF starts after either), but (3,8) leaves 3 setups and
loses while (8,5) leaves 1 setup and saves. The root ties
(5,5) and (3,8) at 9 visits; our-q prefers (3,8) (+0.029),
gap 0.012 < τ=0.05, so the threshold does not fire and RNG
picked (3,8). The rollout leaf prefers the forcing moves
((3,8) and (4,7) both +0.2500) over the quiet prevention
(8,5) (+0.0871) — the same motif as g44. The check picks
(8,5): it holds the highest prior (0.0538) among
preventions. (8,5) also saves at ply 25 ([1,−1,1]), as does
(5,8) ([1,1,1]) — the earliest saving turn is ply 23. The
check reproduces the save at 5 of the 6 VCF-prevention
instances so far.

**Game 91, ply 23 (White to move) — missed VCF prevention
(6th fixed by the check; SAME position as g73 ply 23).**
The actual (5,5) loses 3/3; (8,5) SAVES [1, 1, −1] — and
(8,5) is exactly the check's move. Black holds 1 VCF start
[(8,5)]; (5,5) leaves it live while (8,5) occupies it.
**Transposition note:** g91's ply-23 board is IDENTICAL to
g73's ply-23 board (same 23 moves — the two games share
their opening line and diverge only after ply 23, when
v27's different RNG streams picked different moves: (3,8)
in g73, (5,5) in g91). So g73 and g91 are ONE distinct
position, not two independent instances. The
VCF-prevention class has 7 game-instances across 6 distinct
positions (5 fixed by the check, 1 — g32 — not). The patch
rule is satisfied at the position level too (5 distinct
positions ≥ 3). No other VCF-prevention positions collide
(they sit at different plies, hence different stone counts).

**Game 93, ply 49 (White to move) — search-horizon value
error, no exact signature (3rd).** The actual (8,5) loses
3/3; (5,4) SAVES [1, 1, 1] — and (5,4) is the 400-sim
deeper-search proposal (the only save in the corpus found
by deeper search rather than the 1-ply topk). The save
leaves MORE setups than the loser (6 vs 4) and the loser
(6,12) leaves 6 too — no exact signature. The root is
flat (115 children, top 6 visits); the save holds 3
visits with worse q (−0.033 vs −0.002), and the rollout
leaf prefers the loser (+0.0199 vs −0.0133). The VCF check
does not fire (Black holds 0 VCF starts) and v35 plays
(8,5) like v27. The miss is depth, not evaluation — but
more sims is CLOSED (v10) and this is one instance. Class 7.

**Game 96 (Black, 30 plies) — horizon loss.** No saving move
at any of the last 4 Black turns (4 qualifying turns, 0
saves). Class 3 (9th horizon loss). All 19 vs-v7 losses are
now worked up; the 3 vs-v0 losses remain.

**Games 4 and 32 vs v0, ply 52 — search-horizon with a
setup signature (2 game-instances, 1 distinct position).**
Both are Black, 60-ply losses vs v0. In g4 the actual
(11,3) loses 3/3; (9,4) SAVES [1,1,1] and (5,8) (the
400-sim proposal) SAVES [1,0,1]. In g32 the actual (3,11)
loses 3/3; (4,9) SAVES [1,1,1]. **Transpose note:** g32's
ply-52 board is the TRANSPOSE (x↔y) of g4's ply-52 board —
the same position up to board symmetry (verified: the
transposed move set is identical; the root dumps match
move-for-move under the transpose, including visit counts,
q, and priors). So the two games are ONE distinct
position. In both, neither side holds a VCF start (the
check does not fire; v35 plays like v27), but BOTH saves
remove ALL opponent setups (2→0) while the losers leave 2
— the only positions in the corpus with a setup signature.
(5,8) is additionally the best-q move (−0.033 vs the
actual's −0.109) and the best-leaf move (+0.1209), but
visits rank it 5th (5 vs 11) — promoting it would need the
CLOSED q-based root-decisions lever. 'Minimise setups' is
a single-distinct-position observation, not a mechanism
(needs ≥3; and a 2-ply setup-minimisation rule is v3's
territory, REJECTED). Recorded for future work, not
patched. Class 7 for both games.

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

**Running tally: VCF-prevention 7 game-instances across 6
distinct positions (check reproduces the save at 6
game-instances / 5 distinct positions, picks the losing
prevention at 1), horizon 9, search-horizon (no signature)
3, setup-signature 2 game-instances / 1 distinct position
(see note).** The patch rule (§1.1)
is satisfied at 3 VCF-prevention instances with the patch
reproducing all three saves (g4, g10, g21). The remaining
workups complete the classification table.

## 3. Patch (v35)

### 3.0 Design

**v35 = v27 + the VCF-at-root check** (v29's `_vcf_root_move`,
unchanged — the full check, offensive step first, then
defensive). Everything else is v27 unchanged: 100 sims,
c_puct 1.5, sharp v1 priors, rollout leaf both colours,
τ=0.05 threshold tie-break, most-visited root. The registry
entry (`scoring.py`) is v27's entry + `"vcf_root": True`;
no code in `mcts_heuristic.py` changed for v35 (the
`_vcf_root_move` implementation is v29's, byte-identical).

**Why this is the patch for the verified mechanism.** The
forensics (§2) found 5 missed VCF preventions: positions
where v27's 100-sim search (2-ply rollout leaf) cannot see
the opponent's VCF 3+ plies out, so it plays a move leaving
VCF starts live — and the exact depth-3 VCF check sees the
threat and plays a prevention. The check reproduces the
verified save at 4 of the 5 (g4 ply 44 → (13,7), g10 ply 44
→ (3,6), g21 ply 33 → (5,7), g44 ply 40 → (10,13)); at g32
ply 24 it picks the losing prevention by prior noise (a
documented limitation, xfail in `test_regression_v35.py`).
The mechanism spans flat roots (g4, g21: visit ties the
threshold does not fire on) and concentrated roots (g10:
96 visits on the loser) — the common thread is the
search's horizon, not the root rule, so the fix is a
search-external exact check, not a root-rule change (the
root-decision lever is CLOSED and exhausted — iter-27 §5).

**Why NOT defensive-only (refuted by the mechanism check).**
The first design was defensive-only (the forensics verified
missed preventions, not missed forced wins — 0 in the
corpus): a `vcf_root_defensive_only` flag skipping the
offensive step. The tactical mechanism check REFUTED it:
defensive-only v35 scored **27/29** on the hard suite,
failing both race positions (`vcf-race-block-or-win` and
`-white`): in a race both sides hold a VCF start, and the
defensive-only check blocked (8,11) instead of taking our
forced win (8,3). The offensive step's priority is
load-bearing — when we hold a VCF start we win by force,
and the defensive step must not override it. The flag was
removed (no version uses it) and v35 uses v29's full check.
With the full check, v35 scores **20/20 old + 29/29 hard**
at `--seed 100` (mechanism check, §3.1). The offensive step
cannot hurt game strength (a VCF start wins by force — there
is no better outcome), and it costs little (our
four-creators are few in quiet positions).

**On combination-stacking.** v35 combines v27's threshold
tie-break with v29's VCF check — two passed mechanisms. This
is NOT the CLOSED combination-stacking lever (v17: the
combination does not add): v17 stacked mechanisms without a
forensics finding that the combination addresses a specific
verified mechanism. Here the forensics VERIFIED that the
VCF check fixes v27's specific losses (4 of 5
VCF-prevention instances, each with playout-verified saves
and the check reproducing the save). The combination is
mechanism-driven, and the battery measures whether it adds
strength against v27 itself.

**Cost.** The VCF check is v29's, so v35 inherits its cost
(~1.65s/move mean on quiet positions, ~7s/move in
creator-rich games; iter-29 §3.2). The battery will take
roughly 2x a check-free version's pace. Cost is reported,
not acted on — the battery measures strength.

### 3.1 Mechanism checks (post-implementation, pre-eval)

- **Regression tests** (`ml/tests/test_regression_v35.py`):
  **6 passed, 1 xfailed** — v35 plays a saving prevention
  at all 4 VCF-prevention positions through the full
  `choose_move` dispatch (g4 → (13,7), g10 → (3,6), g21 →
  (5,7), g44 → (10,13)); g32 is the documented xfail (the
  check picks (10,2) by prior noise); the race test pins
  the offensive step's priority (v35 plays (8,3) on
  `vcf-race-block-or-win`).
- **Tactical** (`eval_tactical_full.py --version 35 --suite
  both --seed 100`): **20/20 old + 29/29 hard** on the
  corrected suite. Clause 3 is expected to PASS (the battery
  re-runs this as the official clause-3 measurement).
- **v29 unaffected:** `test_mcts_heuristic.py` 98 passed, 1
  xfailed (run after the flag was added; the flag has since
  been removed, restoring `mcts_heuristic.py` to v29's
  code — the full suite re-runs at §6).

## 4. Fast loop

TODO.

## 5. Battery

TODO.

## 6. Bar check

TODO.

## 7. Verdict

TODO.

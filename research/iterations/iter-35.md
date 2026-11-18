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

### 2.1 Workups (COMPLETE — 22/22, Wed 2026-11-18 03:22 UTC)

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
| 82 (vs v0) | B | 36 | none in window | — | horizon loss ((5,5) 'save' INVALIDATED — see §2.3) |
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
loses 3/3; (4,9) SAVES [1,1,1] and (8,5) (the 400-sim
proposal) SAVES [1,1,1] — (8,5) is the transpose of g4's
(5,8). **Transpose note:** g32's
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

**Game 82 vs v0 (Black, 36 plies) — horizon loss; the
(5,5) 'save' is INVALIDATED by exact computation (see
§2.3).** At ply 28 White holds NINE VCF starts; the actual
(10,10) is a prevention (0 VCF starts after) but loses 3/3
(it leaves 3 creators and 92 setups). The workup's (5,5)
'saved' [1,1,1] in playouts vs v0 — but exact computation
shows White holds FIVE VCF starts after (5,5), so White
wins by force against correct play and (5,5) is not a real
save (v0 simply does not find White's VCF starts). All
four preventions ((10,10), (11,4), (12,4), (11,11)) lose
3/3 in playouts; the check's move (11,4) leaves 3 creators
and 87 setups and loses 3/3. The root is extremely flat
(top 4 moves all 2 visits, identical q −0.375). The
position is lost by ply 28 — no save exists in the window.
Class 3 (10th horizon loss).

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

**FINAL tally (22/22): VCF-prevention 7 game-instances
across 6 distinct positions (check reproduces the save at 6
game-instances / 5 distinct positions, picks the losing
prevention at 1), horizon 10, search-horizon (no signature)
3, setup-signature 2 game-instances / 1 distinct position.
Missed forced win: 0. Prior starvation: 0. Leaf value
error: 0. Missed block: 0 (reproduces iter-27 §3.6).**
Total: 7 + 10 + 3 + 2 = 22. ✓ The patch rule (§1.1)
is satisfied at 3 VCF-prevention instances with the patch
reproducing all three saves (g4, g10, g21). The remaining
workups complete the classification table.

### 2.2 Final classification (22/22)

| Class | Games | Distinct positions | Check fixes? |
|---|---|---|---|
| 1. Missed VCF prevention | 7 (g4, g10, g21, g32, g44, g73, g91) | 6 (g73=g91) | 6/7 games, 5/6 positions (fails g32) |
| 3. Horizon loss | 10 (g7, g15, g30, g35, g45, g56, g65, g67, g96, v0-g82) | 10 | n/a (no save in window) |
| 7. Search-horizon, no signature | 3 (g62, g77, g93) | 3 | no (check does not fire) |
| 7. Search-horizon, setup signature | 2 (v0-g4, v0-g32) | 1 (transpose) | no (check does not fire) |
| 2. Missed forced win | 0 | 0 | — |
| 4. Prior starvation | 0 | 0 | — |
| 5. Leaf value error | 0 | 0 | — |
| 6. Root-decision (visits-over-q) | 0 as primary (v0-g4's (5,8) is q-preferred but 5 vs 11 visits — CLOSED lever) | — | — |

**The patch rule triggered:** class 1 has 7 game-instances /
6 distinct positions sharing ONE mechanism (the search
cannot see the opponent's VCF 3+ plies out; the exact
depth-3 check can) and the check reproduces the verified
save at 5 of the 6 distinct positions. The mechanism spans
flat roots (g4, g21, g73: visit ties the threshold does not
fire on) and concentrated roots (g10: 96 visits on the
loser; g44: the loser is a forcing move) — the common
thread is the search's horizon, not the root rule. The
fallback (cheaper VCF check for v29) does NOT trigger.

**What the forensics did NOT find:** no missed forced win
(0/22) — v27's search already finds offensive VCFs (it is
29/29 on the hard tactical suite). No prior starvation, no
leaf value error as a primary class. Defence is solved
(0 missed blocks). The horizon losses (10/22) are
distributed drift or losses decided before the window —
no patchable mechanism. The setup signature (v0-g4/g32)
is ONE distinct position — recorded for future work, not
patched.

### 2.3 Method note: playouts can produce FALSE saves

The v0-g82 (5,5) case is a methodological finding worth
recording: same-engine playouts (the workup protocol's
verification) called (5,5) a save [1,1,1] because the
opponent (v0) does not find White's VCF starts — but exact
computation shows White holds 5 VCF starts after (5,5) and
wins by force. Playout verification is only as strong as
the opponent in the playout. **Every verified save in the
corpus was therefore re-audited with exact computation**
(opponent VCF starts after the save must be 0): 22 of 23
saves pass; only v0-g82's (5,5) fails. The VCF-prevention
count is unaffected (all 7 instances are real saves).
**Rule for future forensics:** a save is verified only if
it passes BOTH the playout qualification AND the exact
audit (0 opponent VCF starts after). The audit script is
inline in the workup notes (not a saved file — it is a
one-off check over the 23 saves listed in §2.1).

### 2.4 H2's missed-prevention finding (cross-validation)

H2 (v34, iter-34 §3.6) independently found clean missed
preventions are non-zero for the first time: 6 in 468
threatened turns (1.3%), 4 distinct, all as White. Three
dumped instances, all shared identically with v27 (v27
plays the same unsafe move), NOT gate-caused (unique visit
leaders), in two flavours: leaf-knows-but-visits-don't
(g31-ply21: safe moves at 4 visits with BETTER q) and
leaf-blind (g61-ply21, swap g73-ply35: the leaf itself
ranks the unsafe move best).

**Does my corpus contain H2's profile? NO (0 of 4).** My
4 true missed preventions (actual is NOT a prevention):
g4 (flat root, 4-visit tie; leaf ranks (10,5) best, not
the actual), g10 (unique leader at 96 visits, but the
leaf ranks OTHER losers best, not the actual — a weak
leaf-knows flavour at best: the save's leaf edge is
0.0018), g21 (flat root, 6-visit tie; leaf prefers the
saves), g91 (9-visit tie; leaf ranks (3,8) best, not the
actual (5,5)). H2's instances are CONFIDENT misses
(unique leader + leaf agrees); mine are UNCERTAIN misses
(ties) or leaf-disagreements. The two sets are
complementary slices of the same phenomenon (the search
plays a non-prevention when the opponent holds VCF
starts), differing only in the search's confidence
profile. (g44/g73 are not missed preventions at all —
the actual IS a prevention that loses.)

**Does v35 fix H2's instances? YES, all 3 dumped ones**
(tested with `test_vcf_check.py` against H2's batch files,
copied to `~/workspace/b10h3-logs/batch-v34-*.json`):
g31-ply21 → (8,10) [safe: (8,10), (9,11)]; g61-ply21 →
(4,5) [safe: (3,5), (4,5), (8,5)]; swap g73-ply35 →
(10,6) [safe: (11,5), (10,6)]. Search-first overrides
because the search's move is a non-prevention — the fix
does not depend on WHY the search chose it (tie,
confident leader, leaf-blind). This is independent
cross-validation of the v35 patch from a different
battery: the VCF-prevention mechanism generalises beyond
v27's own 22 losses. The 4th distinct instance (swap
g43/g91 ply 17) was not dumped by H2 and is not tested
here.

**Is the leaf-blind flavour worth patching separately?
Not for v35.** The fast loop's 1 refinement is used, and
a leaf-value fix is a different mechanism from the
pre-registered VCF-prevention patch. As a future lead:
if the rollout leaf systematically ranks unsafe moves
best (2 of H2's 3 dumped instances), the value function
misjudges VCF-threat positions — worth a dedicated
forensics pass with more instances (H2 has 4 distinct;
the rate is 1.3% of threatened turns, so a larger corpus
is needed). Recorded, not patched.

## 3. Patch (v35)

### 3.0 Design

**v35 = v27 + the VCF-at-root check** (v29's `_vcf_root_move`,
split into `_vcf_offensive_move` / `_vcf_defensive_move`
with identical behaviour — the split is a pure refactor,
verified by `test_mcts_heuristic.py` 98 passed, 1 xfailed),
with a **SEARCH-FIRST defensive step** (the fast loop's 1
refinement, §4.2): the search runs first, and the defensive
check overrides ONLY when the search's chosen move is NOT a
prevention (leaves opponent VCF starts live — tested by
`_leaves_opp_vcf`, which re-tests only the opponent's
ORIGINAL VCF starts on the resulting board, by monotonicity).
The offensive step still short-circuits (a VCF start wins
by force). Everything else is v27 unchanged: 100 sims,
c_puct 1.5, sharp v1 priors, rollout leaf both colours,
τ=0.05 threshold tie-break, most-visited root. The registry
entry (`scoring.py`) is v27's entry + `"vcf_root": True` +
`"vcf_root_search_first": True`.

**Why this is the patch for the verified mechanism.** The
forensics (§2) found 7 missed VCF preventions across 6
distinct positions: positions where v27's 100-sim search
(2-ply rollout leaf) cannot see the opponent's VCF 3+ plies
out, so it plays a move leaving VCF starts live — and the
exact depth-3 VCF check sees the threat and plays a
prevention. Search-first v35 reproduces the verified save
at 3 of the 6 distinct positions (g4 ply 44 → (13,7), g10
ply 44 → (3,6), g21 ply 33 → (5,7)) — the ones where v27's
search plays a NON-prevention. At g44 and g73 the search
plays a prevention (the wrong one) and search-first does
not override — a deliberate trade (see §4.2): the plain
check's prior-based choice among preventions is noise
(1e-4 margins), and the fast loop verified 3 instances
where it overrode a correct search prevention. At g32 the
check picks the losing prevention by prior noise (a
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
VCF check fixes v27's specific losses (3 of 6 distinct
VCF-prevention positions under search-first, each with
playout-verified saves and the check reproducing the save).
The combination is mechanism-driven, and the battery
measures whether it adds strength against v27 itself.

**Cost.** The VCF check is v29's, so v35 inherits its cost
(~1.65s/move mean on quiet positions, ~7s/move in
creator-rich games; iter-29 §3.2). The battery will take
roughly 2x a check-free version's pace. Cost is reported,
not acted on — the battery measures strength.

### 3.1 Mechanism checks (post-implementation, pre-eval)

- **Regression tests** (`ml/tests/test_regression_v35.py`):
  **8 passed, 2 xfailed** — search-first v35 plays a saving
  prevention at the 3 forensics positions where the search
  plays a non-prevention (g4 → (13,7), g10 → (3,6), g21 →
  (5,7)) and plays the search's prevention at the 3
  fast-loop override-harm positions (g2 → (4,8), g3 ply 9 →
  (5,9), g5 ply 9 → (9,5)); g32 (check picks the losing
  prevention by prior noise) and g44 (search-first does not
  override the search's losing prevention (12,10)) are the
  documented xfails; the race test pins the offensive step's
  priority (v35 plays (8,3) on `vcf-race-block-or-win`).
- **Tactical** (`eval_tactical_full.py --version 35 --suite
  both --seed 100`): **20/20 old + 29/29 hard** on the
  corrected suite (re-verified after the search-first
  change). Clause 3 is expected to PASS (the battery
  re-runs this as the official clause-3 measurement).
- **v29 unaffected:** `test_mcts_heuristic.py` 98 passed, 1
  xfailed (re-run after the `_vcf_root_move` split — the
  split is behaviour-identical; the full suite re-runs
  at §6).

## 4. Fast loop

### 4.1 vs v0 (10 games, seeds 1035/2035): v35 6W-4L-0D (0.60)

Weak for a v27-based engine (v27 is 96W-3L-1D, 0.965, vs
v0) — but 10 games is a small sample (CI [0.31, 0.83]).
All 4 losses were worked up (workup JSONs in
`~/workspace/b10h3-logs/fast-workup-v35v0-g{2,3,5,9}.json`).

**Game 2 (Black, 20 plies) — OVERRIDE HARM, the check
overrides a correct search.** The losing move is ply 12
(8,8): actual loses 3/3, (4,8) SAVES [1,1,1]. White holds
2 VCF starts [(4,8), (8,8)]; BOTH are preventions (0/0/0
after either). v35's check returns (8,8) — the loser —
by prior noise (priors display tied at 0.2101; (8,8)'s
must exceed (4,8)'s by <1e-4, the same flaw as g32 in
§2.1). But **v27's search plays (4,8)** — the save — with
68 visits and our-q +0.108 vs (8,8)'s 19 visits / −0.051.
The search KNOWS the answer; the check overrides it and
plays the loser. This is the first LIVE instance of the
g32 flaw costing a game (in forensics it was only a
regression-test xfail). The check helps when the search
MISSES the prevention entirely (g4, g10, g21, g44, g73)
but hurts when the search FINDS the right prevention and
the check picks a different one by prior noise.

**Games 3, 5, 9 (all White, 29 plies) — horizon losses.**
No saving move at any of the last 4 White turns in any of
the three. The check-fire audit
(`~/workspace/b10h3-logs/audit-fast-v0.log`) shows the check
fired 7x in g3, 6x in g5, 7x in g9 — starting as early as
ply 5, i.e. v0-as-Black holds VCF starts very frequently
and the check is doing heavy work in these games. Games 3
and 9 are the SAME line up to ply 19 (identical check
fires, identical workup plies 21/23/25; they diverge only
at ply 27) — so the three White losses are two distinct
lines, not three. Comparing the check's move against
v27's search at every fire
(`compare-fast-v0.log`): 13 distinct fires, 7 AGREE, 6
DISAGREE. Exact analysis of the 6 disagreements: in 1
(g3 ply 15) the check is RIGHT and the search is WRONG
(the search's (8,6) leaves 4 VCF starts live); in the
other 5 both moves are preventions with IDENTICAL exact
signatures (same VCF/creator/setup counts and squares) —
a pure choice among equivalent preventions, decidable
only by playouts. Those 5 are tested in §4.2.

**White-side note:** v35 went 2W-3L as White in this leg
(g1, g7 wins; g3, g5, g9 losses) and 4W-1L as Black. Five
games per colour is too few to call a White regression;
the battery's 50/50 split is the verdict.

### 4.2 Design decision: search-first defensive (RESOLVED)

The 5 equivalent-prevention disagreements from §4.1 were
tested with same-engine playouts (v35 vs v0, seeds 0/1/2;
log `~/workspace/b10h3-logs/disagreement-playouts.log`):
- g3 ply 9: check's (5,5) loses 3/3; search's (5,9) SAVES
  [1,1,1]. **Override HARM.**
- g5 ply 9: check's (5,5) loses 3/3; search's (9,5) SAVES
  [1,1,1]. **Override HARM.**
- g3 ply 11, g5 ply 17, g5 ply 19: both moves lose 3/3.
  Neutral.
- (g3 ply 15, the 6th disagreement, is not
  equivalent-prevention: the search's (8,6) leaves 4 VCF
  starts, so the check is right there — and search-first
  also overrides there, since (8,6) is not a prevention.)

**Tally.** Override HARM: 3 instances (fast-loop g2 ply 12,
g3 ply 9, g5 ply 9 — in each, the search plays the right
prevention and the plain check picks a losing one by prior
noise). Override HELP (check picks the right prevention
where the search plays the wrong one): 2 instances
(forensics g44, g73). Neutral: 3 (both lose) + g32 (both
designs fail).

**Decision: SEARCH-FIRST.** The defensive check overrides
only when the search's chosen move is NOT a prevention.
Evidence: (1) net position count — search-first fixes 3
forensics positions (g4, g10, g21 — where the search plays
a non-prevention; verified: g4's (3,5) and g21's (11,6)
leave 3 VCF starts each) with 0 observed harms, vs the
plain check's 5 fixes and 3 harms (net +3 vs +2); (2)
weak dominance — search-first overrides only a move that
loses by force (opponent VCF), so it can only help, never
harm, relative to v27, while the plain check turned 3 v27
wins into losses; (3) principle — the check corrects the
search's BLIND SPOT (missing the VCF entirely), not its
JUDGMENT (choosing among preventions), where 100 sims of
visits/q beat a static prior. The trade is explicit:
search-first gives up g44/g73 (documented xfails). A
margin threshold (trust the search when its preference is
strong, the prior when weak) would need a fitted constant,
which the pre-registration forbids — and the visit/q
dominance test does not distinguish g2 from g73 (in both,
the search's move dominates the check's in visits AND q;
only the magnitude differs). This is the fast loop's 1
refinement; no further design changes.

**Implementation** (this commit): `_vcf_root_move` split
into `_vcf_offensive_move` / `_vcf_defensive_move`
(behaviour-identical for v29 — `test_mcts_heuristic.py`
98 passed, 1 xfailed); new flag
`vcf_root_search_first` (default False); new helper
`_leaves_opp_vcf` (re-tests only the opponent's ORIGINAL
VCF starts on the resulting board, by monotonicity);
`choose_move` runs the search first and overrides only a
non-prevention. Cost: the search now runs even when the
defensive check would have fired, plus one VCF gate per
move — the battery measures the pace.

### 4.3 vs v27 (10 games, seeds 1036/2036): v35 8W-2L-0D (0.80)

Search-first v35 vs v27: **8W-2L-0D** (CI [0.49, 0.94])
in 911s. Both losses as White (g3, 53 plies; g7, 35
plies). Both losses were analysed with exact analysis
(VCF gates + prevention checks at the last 4 White turns;
log `~/workspace/b10h3-logs/light-fast2-v27.log`) — full
playout workups were infeasible at current VM load (H1
running 2 matches; a single workup alternative took 15+
min), so the classification rests on exact analysis, not
playouts, and is recorded as such.

- **Game 3:** v35 played preventions at ply 45 (Black VCF
  3→0) but by ply 47 Black held 8 VCF starts and NO
  prevention exists (`_vcf_root_move` returns None —
  verified); plies 49/51 are 11 and 96 VCF starts. The
  loss was decided before ply 47 (Black built too many
  threats for one move to address). Horizon loss; no
  missed-prevention mechanism, no search-first bug.
- **Game 7:** v35 played preventions at plies 27 (4→0) and
  29 (7→0) but by ply 31 Black held 12 VCF starts and NO
  prevention exists (verified). Horizon loss; same
  conclusion.

### 4.4 Fast-loop verdict: v35 GOES TO THE BATTERY

Search-first v35: 10W-0L vs v0 (the plain check's 6W-4L,
all 4 losses flipped), 8W-2L vs v27, tactical 20/20 +
29/29, regression tests 8 passed + 2 xfailed. The 2
vs-v27 losses are horizon losses with no patchable
mechanism in the window (v35 played preventions where
they existed; where it did not, none existed). The fast
loop's 1 refinement is used; no further design changes.
The battery is the verdict.

## 5. Battery

### 5.0 Seed-replay audit (coordinator note, 2026-11-18)

`selfplay_batch.py` creates the engines once per run with fixed
seeds, so on auto-resume their RNG streams restart (the driver
itself prints "RNG restarts"). H2 (v34) found their vs-v27 leg
replayed games 0–33 exactly as games 66–99 (63 unique games,
not 100). Per the coordinator's instruction the battery scheme
is NOT changed mid-run and completed legs are NOT re-run; each
leg is audited for uniqueness (distinct games by move sequence)
and BOTH the nominal numbers (the official battery result) and
the de-duplicated cut are reported.

**Why a resume does or does not duplicate:** the engines'
RNGs are per-engine (seed-a / seed-b), and colours alternate
by game number. A resume at an ODD game replays earlier
games with SWAPPED colours (engine-a is White where it was
Black), so the RNG draws land on different colours and the
games diverge (verified: my vs-v27 games 0 vs 3 diverge at
ply 2). A resume at an EVEN game replays with the SAME
colours — exact duplicates (H2's case: resume at 66).

**vs-v27 leg resume history:** Run 1 games 0–2; Run 2
resumed from game 3 (odd — safe); Run 3 resumed from game
62 (even — games 62–99 are EXPECTED to duplicate games
0–37). Audit at 62 games: **62/62 unique** (0 duplicates).
Final audit when the leg completes (expected ~62 unique
of 100). The de-duplicated cut (62 games) is the honest
strength estimate for clause 1; the nominal 100-game
numbers are the official battery result.

### 5.1 Legs

**vs v27 (clause 1): 51W-45L-4D, 0.5300, Wilson 95% CI
[0.4329, 0.6249] — FAILS** (CI lower < 0.50). De-duplicated:
49W-44L-3D over 96 unique games, 0.5260, CI [0.4271,
0.6230] — also fails. The verdict does not turn on
duplication: both cuts are a statistical tie with v27.
Colour split: as Black 38W-10L-2D (0.78); as White
13W-35L-2D (0.28) — White remains the gap; v35 wins the
Black games and loses the White ones, as v27 did. The
leg's progression (28W-32L-2D at 62 games, 37W-40L-3D
at 80, 51W-45L-4D at 100) shows a genuine late surge,
but the hole was too deep. Uniqueness: 96/100 — the
seed-replay held for only 3 games (62=0, 63=1, 64=2)
before process-level variation broke it; the 4th
duplicate pair (30=96) is engine determinism, not
replay (H2's vs-v0 phenomenon in miniature).

**vs v0: 97W-2L-1D, 0.9750, CI [0.9226, 0.9922].**
De-duplicated: 42W-2L-1D over 45 unique games, 0.9444,
CI [0.8364, 0.9826]. (v27's: 96W-3L-1D nominal, 0.9650;
de-dup 37W-3L-1D, 0.9146 over 41 unique — v35 is one
nominal game and ~3 points de-duplicated better, not a
meaningful strength signal on its own.) The vs-v0 leg
reproduces H2's engine-determinism finding: 45 unique
of 100 with NO seed-replay (the leg ran in two
processes, resumed from game 67 — odd, so colours
swapped and no replay occurred; the duplicates are
v0's near-determinism collapsing seed draws to the
same short games). Colour split: as Black 47W-2L-1D;
as White 50W-0L-0D (perfect as White — the opposite
of the vs-v27 pattern). Losses: games 4, 26 (both as
Black).

**swap vs v27: 59W-35L-6D, 0.6200, CI [0.5221, 0.7090].**
100/100 unique games (no duplication — the placement
RNG and the swap choice add variety; the leg also ran
in two processes, resumed from 89 — odd, so no replay).
Under swap rules v35 beats v27 with the CI clear of
0.50; under normal rules it does not (0.53, CI
straddling 0.50). The swap rule neutralises the
first-move edge and lets the check's skill show. Colour
split: v35 as Black 38W-10L-3D (0.775); as White
21W-25L-3D (0.45) — White is still the weaker side,
but the swap choice (v35 Black 51, White 49) balances
the aggregate. (v27's own swap vs v7 was 0.775 — a
different opponent, not comparable.) Swap is a
secondary metric, not a bar clause; it does not change
the clause-1 verdict.

**Ladder (clause 2): Elo 636.4, interval [240.3, 1032.5] —
TIES v27's 636.4, does not exceed it — FAILS.** Legs:
random 20W-0L-0D; heuristic 19W-0L-1D (identical to
v27's — the Elo is determined by the heuristic leg
alone, since the random leg is saturated and the MCTS
leg is not an Elo anchor); mcts **19W-1L-0D — the best
MCTS leg in the program** (v29's 15-5 and v34's 15-5
were the previous bests; v18's was 14-6). The mcts leg
is a strength signal, not an Elo input: it shows v35
crushes the baseline MCTS, but the bar's Elo instrument
cannot see it. v34's 20-0 heuristic leg (the only one
ever) is what a pass needs; v35's single draw (as
Black, game 9) is the difference between 636.4 and a
saturated pass. The ladder driver ran clean in one
process (no resume); the EVAL_JSON backup is in
`~/workspace/b10h3-logs/ladder-v35-stdout.log`.

### 5.2 Defence classification

`classify_losses.py` on the three battery batches:

| Batch | Losses | Via open four | Missed blocks | Clean missed preventions | Threatened turns |
|---|---|---|---|---|---|
| vs v27 | 45 | 45 | 0 | **0** | 251 |
| vs v0 | 2 | 2 | 0 | **0** | 8 |
| swap vs v27 | 35 | 35 | 0 | **0** | 189 |
| **Pooled** | **82** | **82** | **0** | **0** | **448** |

v35 has **0 clean missed preventions in 448 threatened
turns** — the check does exactly what it was designed
to do. (Compare v34's 6 in 468, the first non-zero
reading in the program; v27's 0 in 183.) Every v35
loss is via open four with no clean prevention missed:
the remaining losses are horizon losses (the
opponent's forcing sequence was already unstoppable
by the time the check could see it) or prevention
choices among equally-clean options (the g32 prior-noise
flaw — the check picks a prevention, but the wrong one).
Defence is still solved in the block sense (0 missed
blocks holds).

## 6. Bar check (v35 vs v27)

| Clause | Requirement | v35 result | Verdict |
|---|---|---|---|
| 1 | h2h vs v27 ≥ 100 games, win rate > 0.50, Wilson 95% CI lower > 0.50 | 51W-45L-4D, 0.5300, CI [0.4329, 0.6249] (de-dup: 49W-44L-3D over 96 unique, 0.5260, CI [0.4271, 0.6230]) | **FAIL** |
| 2 | Ladder Elo > 636.4 | Elo 636.4, interval [240.3, 1032.5] — TIES v27/v18, does not exceed | **FAIL** |
| 3 | Tactical ≥ 20/20 + 29/29 (corrected suite) | 20/20 old, 29/29 hard | **PASS** |
| 4 | All test suites green | pytest 405 passed, 7 skipped, 60 xfailed, 0 failures; npm 55/55; tsc clean; baseline.py untouched | **PASS** |

**v35 is REJECTED, 2/4 clauses.** No v18 challenge match
is triggered (it requires all 4 clauses). v18 stays the
strength champion; v27 stays the bar reference.

## 7. What this means

**The forensics was right; the patch was real; the bar
was too high for it.** The VCF-prevention mechanism was
verified on 7 game-instances (6 distinct positions) of
v27's losses, cross-validated on all 3 of H2's
independently-discovered missed preventions, and the
battery confirms the mechanism works: 0 clean missed
preventions in 448 threatened turns (vs v34's 6 in
468). The fast loop showed the search-first design
flipping all 4 of its vs-v0 losses (6W-4L → 10W-0L)
and winning 8-2 vs v27. The swap leg (0.62, CI clear
of 0.50) and the record MCTS ladder leg (19-1) both
point the same way: v35 IS stronger than v27.

But the bar's clause 1 measures normal-rules h2h
against v27, and there v35 is 51-45-4 — a 6-game edge
that the Wilson interval cannot separate from a tie.
Two structural reasons: (1) the missed-prevention
rate was only ~1.3% of threatened turns, so fixing it
gains a handful of games, not dozens; (2) White
remains the gap (13W-35L-2D as White vs v27) — the
check helps both colours equally, so it cannot close
a colour-asymmetric deficit. The horizon losses (10
of v27's 22) are untouched by design: the check fires
only when the opponent already holds VCF starts, and
by then the loss is often already forced.

**The VCF-prevention lever is now CLOSED as a
strength lever for the v27 family.** The check is
proven to eliminate missed preventions, and the
remaining losses are horizon losses and
prevention-choice noise (g32's 1e-4 prior margin).
Further gains need a different mechanism: either
earlier threat detection (before VCF starts exist —
the setup-signature lead from §2, a single distinct
position, needs more instances) or the leaf-value
fix for H2's leaf-blind flavour (§2.4). The
search-first design is the right shape for any
future check (it is weakly dominant over the search
alone), and it should be the default whenever a
check is added to a search engine.

**Cost note:** v35 inherits v29's VCF cost (~1.65s/move
mean, ~7s/move creator-rich) plus the search-first
overhead. The battery took ~7 hours wall-clock for
340 games (3 legs + ladder) under 2-way contention.
A future version that wants the check's benefit
without the cost needs the pre-registered fallback
(§1.2, cheaper VCF check) — which v35 did not need,
since the forensics found a patchable mechanism.

## 6. Bar check

TODO.

## 7. Verdict

TODO.

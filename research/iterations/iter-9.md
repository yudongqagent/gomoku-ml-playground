# Iteration 9 — v9: q-vs-visits root decision (batch 4, H1)

**Status: IN PROGRESS — pre-registration (§1) written BEFORE any eval.
Smoke (§3.1) and fast loop (§2) to follow; full battery (§3.5–3.8) runs
on the smoke winner only. Comparison target and eval seeds were
recorded BEFORE the first eval (see §1).**

**METHOD (carried over from batch 3).** The fast 10-game loop stays
the finding method: play 10 games, read every move, find each game's
losing move (solver/deeper-search verified, never guessed), patch the
algorithm to reproduce the perfect move in those positions (regression
test), repeat — max 4 iterations, or stop when an iteration finds no
new patchable weakness class. The 4-clause bar stays the validation
gate. Fast-loop games use development seeds (1001/2001, 1002/2002,
...) distinct from the bar battery's pre-registered seeds (100/200).
Per the §2.6 warning, fast loops include as-Black games vs v0, and a
10-game batch alone never concludes "no weakness".

## 1. Design and pre-registration

**Comparison target: v7** (the new reference, batch-4 plan). Clause 1
is head-to-head vs v7 (normal rules, >= 100 games, alternating
colours, win rate > 0.50 with Wilson 95% CI lower bound > 0.50);
clause 2 is ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts; mcts_sims=100); clause 3 is
tactical pass rate >= v7's (20/20 old suite + 29/29 hard suite);
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v9 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v9 = v7 with ONLY the root move-decision rule
changed.** Everything else is v7 unchanged: v6's PUCT MCTS (100 sims,
c_puct 1.5, v1 priors over all candidates, win/block scans before
search) + the capped (+/-0.2) attack-gradient leaf in quiet
positions. The ONLY change is how the root child is picked after the
100 sims.

### 1.0 SIGN CORRECTION — read this before the variants

**Iter-7 §2.6's (and §2.1's, and iter-6 §2.2c's) "q prefers the
saving move" claim is SIGN-INVERTED, and the batch-4 plan's H1
rationale inherits the error.** Verified three independent ways
BEFORE any eval, by exact reproduction:

1. **Exact reproduction.** Replaying batch-v7-vs-v0.json game 16 to
   ply 14 and re-running v7's search reproduces iter-7's numbers
   exactly: (7,4) 10 visits, q 0.042; (8,4) 10 visits, q 0.032;
   (9,8) 7 visits, q 0.071; (6,9) 7 visits, q 0.065.
2. **Code.** The backup is negamax: each node's `value_sum` is from
   the perspective of the player to move AT THAT NODE, so a root
   child's raw `q` is the OPPONENT's value. `_select_child`
   confirms it: PUCT exploits `-child.q`. `analyse_critical.py`
   (the script behind the §2.6 numbers) reports raw `c.q`.
3. **Unit check.** A root child whose subtree is a forced win for
   the opponent has raw q = +1.0 — good for the opponent, bad for
   us.

So in §2.6, correctly signed (our-perspective value = `-child.q`),
the saving moves rank LOWEST: (8,4) -0.032 beats (7,4) -0.042, and
(9,8) -0.071 / (6,9) -0.065 are worse still. **The search's q does
NOT know the right move in this position — it prefers the loser.**
The same inversion applies to §2.1 ("q preferred the saving move in
4/5 positions": game 1 saving -0.07 vs losing -0.257, game 5 saving
+0.106 vs losing -0.381 — correctly signed, q prefers the LOSING
move in all of them) and to iter-6 §2.2c's "pathology" ((5,6) at
raw q 0.333 is a move the search correctly AVOIDS, not a good move
starved of visits). The engines are unaffected — only the analysis
text is wrong: the engines use `-child.q` correctly in PUCT and
most-visited for the decision.

**What §2.6 actually shows, correctly read:** at 100 sims, in this
position, the search (visits AND q) cannot see the 5-ply-deep loss
— iter-7 itself says the -1 is beyond what 100 sims reliably reach,
and the gradient masks the -0.25 penalties. The ONLY signal that
prefers the saves is v1's 1-ply score ranking ((9,8) rank 2, (7,4)
rank 3, (6,9) rank 5, (8,4) rank 6). The q hypothesis therefore
loses its motivating evidence — but the general question "does
q-based root choice beat visit-based root choice?" is still a
legitimate empirical question (one position does not settle a
match-level question), so the q variants are tested as tasked,
with correctly-signed q. Variant D (below) is the variant the
CORRECTED evidence actually supports, and is included in the smoke
on that basis.

**"q" in every variant below means our-perspective value,
`-child.q`.** Using raw child q would play the move the search
thinks is WORST for us — a bug, not a hypothesis. All variants
consider only root children with visit_count >= 1: an unvisited
child's q is undefined (0.0 by convention), not a signal — a
0-visit child must never win an argmax over q.

### 1.1 The variants (pre-registered)

- **A — pure q.** Argmax our-q over visited root children; ties
  broken by visits, then RNG. The naive hypothesis: trust the
  searched values entirely. Known risk, stated in advance: q of a
  1-visit child is a single leaf value — noisy — so A may pick
  low-visit children on lucky leaves. The smoke is the test of
  that risk.
- **B — tie-q.** Most-visited; ties broken by our-q, then RNG. The
  minimal change: identical to v7 except on exact visit ties.
- **C — blend.** Score = our-q + 0.1 * (visits / max_visits);
  argmax score; ties broken by visits, then RNG. The 0.1
  coefficient is fixed here, NOT tuned on any eval: q gaps below
  0.1 are within one gradient step (the gradient's full range is
  +/-0.2, so 0.1 is half the maximum gradient swing), so visits
  override only sub-gradient-scale q differences, while a real
  searched q gap (> 0.1) always wins.
- **D — tie-prior.** Most-visited; ties broken by the v1 prior
  (a monotone transform of the v1 1-ply score, so this is
  tie-breaking by v1 score), then RNG. The variant the CORRECTED
  §2.6 evidence supports: v1's ranking prefers the saves. The
  minimal change, like B, but with a deterministic, prior-tested
  tie-break signal instead of noisy q.

Implementation: `HeuristicMCTS(root_decision=...)` with values
`"visits"` (v6/v7 default, unchanged behaviour), `"q"`, `"tie-q"`,
`"blend"`, `"tie-prior"`; wired through `ENGINE_VERSIONS` via a
`"root_decision"` key (default `"visits"` when absent, so v6/v7
are untouched). v9 is registered in `scoring.py` AFTER the smoke,
with the winner's `root_decision`.

### 1.2 Winner selection (pre-registered)

**Smoke: 20 games vs v7 per variant** (seeds 300/400 — fresh,
NOT the battery's 100/200, so selection is independent of the
verdict battery; alternating colours). Driver:
`ml/scripts/smoke_root_decision.py`; one JSON per variant under
`research/iterations/` recording the variant. Winner = highest
smoke win rate vs v7. Tie-breaks, in order: (1) higher hard-suite
tactical score (29 positions, full-dispatch driver); (2) plays a
saving move on the §2.6 diagnostic position (see §1.3) and keeps
all 5 v7 regression positions passing; (3) simplicity order
D > B > C > A (D and B are minimal single-rule changes; D's
tie-break signal is deterministic and prior-tested, B's is
search-noisy; C blends; A discards visits entirely).

**The full 100-game battery is the verdict, never the smoke.**
The battery runs on the smoke winner. Only the winner gets the
battery. If the winner fails the battery, v9 is REJECTED and the
other variants' smoke numbers stay recorded as evidence (a
runner-up battery would be a different version, for the
coordinator to decide).

**Honest caveat, stated in advance:** B and D differ from v7 only
on exact visit ties (rare events), so their 20-game smokes will
likely sit near 0.50 with wide CIs and the smoke "winner" between
them may be decided by noise. The smoke picks which variant gets
the battery; the 100-game battery is the real verdict.

### 1.3 The §2.6 diagnostic position (pre-registered, not the verdict)

batch-v7-vs-v0.json game 16, replayed to ply 14 (Black to move;
v7 played the losing (8,4), flat 68). Oracle-verified saving
moves (iter-7 §2.6, walk-back seeds 0,1,2): (7,4) flat 67, (9,8)
flat 129, (6,9) flat 141. Each variant's move here is recorded as
a pre-eval observation after implementation (like iter-7 §1's
pre-eval observation). **Expected from the reproduced root stats:
B picks the LOSER (8,4)** — correctly-signed q prefers it
(-0.032 vs -0.042) — **and C likely does too** (blend scores
(8,4) 0.068 vs (7,4) 0.058 among the top children); A picks
argmax our-q, which among the top-10 visited children is (7,8)
-0.030 (oracle-untested); D picks by prior among the tied pair
(full-precision priors to be checked). The design stands as
pre-registered regardless: the evals decide. If the smoke winner
plays a saving move here, the position becomes v9's first
regression test (`regression_positions_v9.json`); if it plays
(8,4), that is recorded honestly as a known miss, not hidden.

### 1.4 Eval seeds (pre-registered, battery)

h2h vs v7: v9=100 / v7=200; h2h vs v0: v9=100 / v0=200; ladder
seed 100, 20 games/opponent, mcts_sims=100; swap 100 games vs v7,
balanced placement, placement_seed 11, tempo judge (the harness
default; recorded in the JSON). Harness commands follow iter-7
§3 exactly (`selfplay_batch.py`, `eval_version.py` for the
ladder, `eval_tactical_full.py` for tactical).

### 1.5 Pre-eval observation (recorded AFTER implementation, BEFORE any eval)

Diagnostic only — the design stands as pre-registered; the evals
decide. Driver: `ml/scripts/diag_root_decision.py`.

**§2.6 position** (full-precision root stats): (7,4) v=10,
our-q -0.041882, prior 0.077230; (8,4) v=10, our-q -0.032228,
prior 0.076957; (9,8) v=7, our-q -0.071229; (6,9) v=7, our-q
-0.064911. The prior DOES separate the tied pair ((7,4) >
(8,4)), as v1's ranking (3rd vs 6th) implied.

| rule | plays at ply 14 | verdict |
|---|---|---|
| visits (v7, no RNG) | (7,4) | SAVING — the no-RNG first-tie-break picks (7,4); the match RNG picked (8,4). The §2.6 loss was literally a coin flip. |
| q (A) | (11,3) | other (oracle-untested) — a **1-visit** child, our-q +0.0249 from a single lucky leaf. The pre-registered noise risk, manifesting: argmax our-q over 33 visited children is dominated by 1-visit children (the only well-visited child above (8,4)'s our-q is (7,8), 9 visits, -0.0295). |
| tie-q (B) | (8,4) | LOSING — as predicted in §1.3: correctly-signed q prefers the loser. |
| blend (C) | (8,4) | LOSING — as predicted (blend 0.068 vs 0.058). |
| tie-prior (D) | (7,4) | SAVING — as predicted; the prior separates the tie toward v1's higher-ranked move. |

**v7 regression positions (5):** visits 4/5 (known xfail
fl1-g3-ply23 only); q 3/5 (also breaks fl1-g1-ply13); tie-q 4/5;
blend 3/5 (also breaks fl1-g1-ply13); tie-prior 4/5.

**Tactical suites (full-dispatch driver, same driver/seeds/suites
as the battery — so these ARE the clause-3 numbers for each
variant):**

| rule | old | hard | hard failures |
|---|---|---|---|
| visits (v7) | 20/20 | 29/29 | — |
| q (A) | 19/20 | **22/29** | 7 prevention positions |
| tie-q (B) | 20/20 | 29/29 | — |
| blend (C) | 20/20 | **26/29** | prev-v1v0s-g13, prev-v1v0s-g26, prev-v1v0s-g48 |
| tie-prior (D) | 20/20 | 29/29 | — |

**Consequence, recorded before the smoke: A and C FAIL clause 3
(tactical < v7's 29/29) and are disqualified as battery
candidates.** The battery cannot pass them — their outcome is
already decided (fail) — so the smoke chooses between the viable
candidates B and D. The A/C smokes still run as pre-registered,
to give the q hypothesis its match-level numbers for the record;
if A or C somehow wins the smoke, that number is reported
prominently but does not earn it the battery, because clause 3
is a hard gate of the bar and it is already failed. Pure q's
failure mode is now visible twice over: it picks 1-visit
children on lucky leaves (§2.6: (11,3)) and it fails prevention
positions (22/29) — the q-values at 100 sims are too noisy to
decide the root move, which is exactly why most-visited is the
standard rule. The q hypothesis, in its pure and blended forms,
is refuted on tactical evidence before a single match game; the
smoke tests whether the tie-break forms (B, D) — which keep the
visits rule and only change tie-breaking — differ from v7 at all.

## 2. Loss analysis for iteration 9

[Fast-loop worked games go here — see the method note at the top.]

### 2.1 Fast-loop iteration 1 — 10 games vs v0

[To be filled.]

### 2.2 Fast-loop iteration 2 — 10 games vs v7

[To be filled.]

### 2.3 Fast-loop iteration 3 — NOT RUN unless iterations 1–2 find a new patchable class

### 2.4 Fast-loop iteration 4 — NOT RUN unless iterations 1–3 find a new patchable class

### 2.5 Aggregate counts (secondary)

[To be filled from the final battery.]

### 2.6 Worked game(s) from the final battery

[To be filled — verified walk-backs like iter-7 §2.6.]

## 3. v9 evaluation

### 3.1 Smoke (20 games vs v7 per variant, seeds 300/400)

Driver: `ml/scripts/smoke_root_decision.py`. One JSON per variant
under `research/iterations/smoke-v9{a,b,c,d}-vs-v7.json`.

| variant | result | win rate | 95% CI | as Black | as White |
|---|---|---|---|---|---|
| A — pure q | 4W-16L-0D | 0.2000 | [0.0807, 0.4160] | 4W-6L-0D | **0W-10L-0D** |
| B — tie-q | 9W-11L-0D | 0.4500 | [0.2582, 0.6579] | 3W-7L-0D | 6W-4L-0D |
| C — blend | 6W-14L-0D | 0.3000 | [0.1455, 0.5190] | **0W-10L-0D** | 6W-4L-0D |
| **D — tie-prior** | **12W-5L-3D** | **0.6750** | [0.4567, 0.8369] | 5W-4L-1D (0.55) | 7W-1L-2D (0.80) |

**Winner: D (tie-prior), outright** — highest smoke win rate,
tactical 29/29 (§1.5), plays the saving move at §2.6. No
tie-break needed. D is registered as v9
(`scoring.py`: `leaf_mode="gradient"`, `root_decision="tie-prior"`).

What the smoke says about the q hypothesis: every q-involving
variant loses to v7, and two of them collapse entirely in one
colour (A as White 0-10, C as Black 0-10). The q-values at 100
sims are not just noisy — used as a decision signal they are
actively misleading, exactly as the sign correction (§1.0)
predicts: correctly-signed q prefers the losing move in the
analysed positions, so tie-breaking by q (B, 0.45) is slightly
worse than v7's RNG coin flip, and deciding by q (A, 0.20) or
blending it in (C, 0.30) is much worse. The ONLY root-decision
change that helps is the one that does not use q at all: break
visit-ties by the v1 prior, the deterministic signal that
preferred the saving moves in §2.6. D differs from v7 ONLY on
exact visit ties — yet wins 12-5-3: a single tie-break
difference diverges the whole game, and D wins those
divergences systematically (notably as White, 7-1-2 — the colour
v7 is weakest in). The smoke CI still includes 0.50
([0.4567, 0.8369]) — n=20 is weak evidence, and the 100-game
battery is the verdict.

### 3.2 Tactical suites (per variant, diagnostic)

Measured in §1.5 (full-dispatch driver, same as the battery):
visits (v7) 20/20 + 29/29; A 19/20 + 22/29; B 20/20 + 29/29; C
20/20 + 26/29; D 20/20 + 29/29. A and C fail clause 3; B and D
keep v7's perfect scores. The battery re-runs tactical for the
registered v9 (D) in §3.5.

### 3.3 Fast-loop summary (final engine)

[To be filled.]

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

[To be filled.]

### 3.5 Tactical suites — FINAL BATTERY

[To be filled.]

### 3.6 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

[To be filled.]

### 3.7 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

[To be filled.]

### 3.8 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

[To be filled.]

## 4. Bar check (v9 vs v7)

[To be filled.]

## 5. What this means

[To be filled.]

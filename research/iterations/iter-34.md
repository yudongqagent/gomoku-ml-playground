# Iteration 34 — v34: position-dependent threshold for v27's q-gap tie-break (batch 10, H2)

**Status: IN PROGRESS — pre-registration (§1).** Comparison target is
v27 (the batch-10 bar reference: ladder Elo 636.4, on master at
`d0cf07d`). This is the H2 workstream of batch 10 (H1 is v33,
VCF-filtered search on v29; H3 is v35, v27 loss forensics);
work happens on branch `research/batch-10-h2` in the
`gomoku-b10-h2` worktree, branched from `research/batch-10` @
`a4ed543` (NOT from master — the coordinator branch already
contains the batch-10 plan and the auto-resume ladder driver).
v34 does NOT change the leaf (v18's rollout leaf, byte-identical),
does NOT change sims/priors/book (CLOSED), does NOT change
exploration constants (CLOSED without a new mechanism), and does
NOT include the threat check or the VCF check. The ONLY change
from v27 is the tactical gate on the threshold-gated tie-break in
§1 — a narrow refinement of v27's mechanism, targeting the
murky-midgame fires batch 9 recorded as a caution (iter-27 §2.2).
It is NOT a reopening of the closed q-based root-decisions lever
(see §1.1).

**Method (carried over from batch 3/4/5/6/7/8/9):** the fast 10-game
loop — play 10 games, read EVERY move of every game, find THE
losing move per game (verified with the solver or a deeper search —
never guessed; for rollout-leaf engines, playouts or exact
computation, NOT the v1 walk-back oracle alone — iter-18 §2.1),
patch the algorithm so that, replayed to that critical position,
it plays the perfect move, and record each critical position +
correct move as a regression test. Then play 10 more games and
repeat. Max 4 fast-loop iterations before validation, or stop when
two consecutive iterations find no new patchable weakness class.
Patches must fix the STRUCTURAL cause, not the single position.
**The fast loop runs against v0 as well as v27** (the bar reference
is v27 now, not v7) — each iteration plays 10 games vs v27 AND 10
games vs v0, and losses from both opponents are walked back. The
full 4-clause validation battery (100-game matches + ladder) runs
on the FINAL patched v34 only. Fast-loop games use development
seeds (1001/2001, 1002/2002, ...) distinct from the bar battery's
pre-registered seeds (100/200), so no development game overlaps an
eval game.

## 1. Design and pre-registration

**Comparison target: v27** (primary, the batch-10 reference: ladder
Elo 636.4, on master at `d0cf07d`). The bar's clause 1 is
head-to-head vs v27 (normal rules, 100 games, alternating colours,
win rate > 0.50 with Wilson 95% CI lower bound > 0.50), clause 2 is
ladder Elo strictly higher than v27's 636.4 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100 — note
clause 2 is now demanding: 636.4 is the estimate from a
19W-0L-1D heuristic leg, so beating it effectively requires a 20-0
leg), clause 3 is tactical pass rate >= v27's on the CORRECTED
suite (old suite 20/20, hard suite 29/29), clause 4 is all test
suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v34 vs v0 (100
games) and a 100-game swap match vs v27 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.
**Strength champion:** v18 (h2h vs v7 0.775, ladder Elo 636.4).
Under champion rule v2 (direct challenge, TRACKS.md), a bar-passer
plays a 100-game match against v18 and takes the title if it wins
the match (win rate > 0.50 with Wilson 95% CI lower bound > 0.50)
AND its ladder Elo beats 636.4 — a bar-passer already clears the
Elo clause by passing, so the challenge match is the deciding test.

**The improvement: v34 = v27 + a tactical gate on the tie-break.**
Everything else is v27 unchanged: 100 sims, c_puct 1.5,
v1 priors over all candidates, win/block scans before search,
most-visited root child, rollout leaf both colours (leaf mode
`rollout2_v1_both`, v16's function byte-identical including both
fast-loop patches), and the threshold τ=0.05 on the q gap. The
tactical gate is the ONLY change from v27.

### 1.1 The tactical gate (pre-registered, exact)

v27's rule (iter-27 §1.1): at the root, on an exact top-visit
tie, break the tie by correctly-signed q (our-q = `−child.q`)
ONLY if the gap between the highest and second-highest DISTINCT
our-q among the tied children exceeds τ=0.05; below the
threshold, v18's RNG-over-all-tied behaviour. v34 adds ONE
further condition to the fire:

1. **When it applies:** only at the root, only when v27's rule
   would otherwise fire — i.e. an exact top-visit tie exists
   AND the q gap exceeds τ=0.05. If there is no tie, or the gap
   is at/below the threshold, v34 behaves EXACTLY like v27 (no
   extra work is done at all: the gate is evaluated only when
   the tie-break would otherwise fire, so the check costs
   nothing on the ~91% of searched moves with no tie).
2. **The tactical condition:** the root board position is
   TACTICAL iff either player (Black or White) holds an open
   three or better on the board BEFORE our move, where "open
   three or better" is defined in the scorer's own terms: some
   5-window contains ≥3 stones of that player and 0 stones of
   the opponent, and — when exactly 3 — both cells just outside
   the window are empty (open_ends == 2). A window with 4 stones
   of one player and 0 of the opponent counts regardless of the
   outside cells (the empty cell inside the window is a winning
   square); a window with 5 counts as a win. Equivalently:
   `scoring._score_from_counts` called with the window's ACTUAL
   stone counts (no +1 for a move being placed) returns kind
   `_KIND_OPEN_THREE`, `_KIND_FOUR`, or `_KIND_FIVE` for either
   colour. The check is evaluated on the root board, for both
   colours, before our move is played.
3. **Fire condition:** the tie-break fires ONLY if the position
   is tactical (rule 2) AND the q gap exceeds τ=0.05 (v27's
   rule). When it fires, v34 behaves EXACTLY like v27: play the
   child with the highest our-q; if two or more share q_max
   exactly (float equality), fall back to RNG choice among that
   subset when an RNG is set, else the first in dict order.
4. **Below the gate:** when the position is NOT tactical, the
   tie-break does NOT fire at all — v34 behaves EXACTLY like
   v18: RNG choice among ALL visit-tied children when an RNG is
   set, else the first in dict order. (Not restricted to the
   q-tied subset: in a quiet position q carries no reliable
   signal, so no q information is used — the same reasoning as
   v27's below-threshold behaviour.)

**Interpolation check (by construction).** If every position
were tactical, v34 would reproduce v27 exactly. If no position
were tactical, v34 would reproduce v18 exactly (the tie-break
never fires). v34 sits strictly between them, and the gate's
only job is to separate positions where q gaps carry terminal
signal (forcing threats within the 2-ply rollout horizon) from
positions where they are gradient noise.

**Why this retest is justified (and why it is not a reopening
of the closed lever).** The τ=0.05/0.10 sweep tied (39W-0L-1D
vs v0 over 40 games at both values — iter-27 §1.5), so a flat
threshold change is NOT the lever — the CONDITION is. Batch 9
recorded the caution (iter-27 §2.2, fast-loop ply 22): the
threshold fired on a signal-sized gap (0.1481) between two
verified losers in a murky midgame — a large gap there is not
the same evidence as in a tactical position, because the
rollout leaf's q values are estimates, not exact values,
outside forced lines. The mechanism claim: in a tactical
position (a forcing threat on the board), the 2-ply rollout
sees the threat's resolution within its horizon, so the leaf
values — and hence the root q gaps — reflect near-terminal
outcomes; in a quiet position, the same gap size is gradient
noise from the leaf's capped ±0.2 attack gradient. This is a
refinement of v27's narrow mechanism, not a reopening of
q-based root decisions as a general mechanism: pure q is still
catastrophic (0.20), and v34 still never consults q except on
an exact visit tie with a large gap in a tactical position.

**What would refute the hypothesis.** If v34's battery vs v27
is a statistical tie (win rate ≈ 0.50, CI including 0.50) with
vs-v0 and ladder numbers indistinguishable from v27's, the
gate is a no-op in real games: quiet-position above-threshold
fires are too rare (or too outcome-free) to matter, and the
root-decision lever is exhausted — confirming iter-27 §5's
"last such refinement available" reading. If v34 is WORSE than
v27 (h2h CI upper < 0.50, or vs-v0/ladder below v27's), the
gate blocks signal, not noise: quiet-position fires were
carrying real information (e.g. a tied move creating a threat
the position-based condition cannot see), and the condition —
not just the threshold — is the wrong separator.

### 1.2 Pre-registration diagnostics (v27 behaviour, measured before v34 exists)

These measurements are on v27 (the existing, batteried engine),
not on v34 — they test whether the pre-registered condition is
well-formed (preserves the known fixes, has any bite at all)
before v34 is built. Scripts in `/tmp` (not committed):
`diag_v34_threats.py`, `dump_v27_nothreat.py`,
`measure_v34_gate.py`.

**Diagnostic 1 — the condition preserves all three tie fixes.**
Every hard-suite position was scanned for open-three-or-better
for either side. The three tie-fix positions are ALL tactical:
prev-v1v0s-g26 (White holds a threat), prev-v1v1s-g79 (both),
prev-v1bl-g2 (White holds a threat). Fresh-seed-0 root dumps
reproduce iter-27 §1.3 exactly: 4-way tie @3, gap 0.4167,
FIRES → (9,4) EXPECTED; 7-way tie @3, gap 0.4167, FIRES →
(8,7) EXPECTED; 2-way tie @8, gap 0.1306, FIRES → (7,5)
EXPECTED. The gate does not touch any of them.

**Diagnostic 2 — the gate blocks nothing load-bearing on the
suites.** The 11 no-threat hard-suite positions (all 7 setup
positions, winning-quiet-setup-2, both vcf-two-four-sequence
positions) and the 1 no-threat old-suite position
(double-threat-create) ALL have a UNIQUE visit leader at fresh
seed 0 — the tie-break never fires there, so gating it off
changes nothing; v27 and v18 pick identically (and correctly)
at all 12. **Expected tactical: 20/20 old + 29/29 hard.**

**Diagnostic 3 — the gate's bite in real games is small.**
10 games v27 vs v0 (dev seeds 5001/6001, 10W-0L): 185 searched
root decisions, 61 tactical (33.0%), 124 quiet (67.0%);
16 top-visit ties (8.6% of searched) — ALL 16 in QUIET
positions, ZERO in tactical positions; and ZERO tie-break
fires (gap > 0.05) in 10 games. Two readings: (a) ties cluster
in quiet positions, where visits spread thin — exactly where
the gate bites; (b) above-threshold fires are rare in real
games vs v0 (0/16 ties, 0/185 decisions), so the gate's
expected effect vs v0 is ~zero. The battery vs v27 (longer,
more contested games) is where any quiet fires would show.

**Diagnostic 4 — the motivating murky fire is NOT blocked.**
The iter-27 §2.2 position (fastloop2-v27-vs-v7 game 2, ply 22)
reproduces exactly (2-way tie @8, gap 0.1481, FIRES → (5,0)),
but the position IS tactical by the pre-registered definition
(a threat is on the board — White holds it; Black is losing).
v34 fires there too. This is CORRECT, not a defect: the fire
separated two verified losers and was outcome-free; the gate's
job is to block noise fires in quiet positions, not to rescue
lost tactical positions. The position stays a documented
illustration of the gap-size caution, not a v34 regression
position.

**Game-strength expectation (pre-registered, honest).** The
diagnostics say the gate is near-neutral by construction: it
blocks only quiet-position above-threshold fires, of which 0
occurred in 10 diagnostic games vs v0. **Expected: v34 ≈ v27**
— h2h vs v27 near 0.50 (a tie, failing clause 1 unless quiet
fires are materially more common vs v27 than vs v0), vs v0 ≈
v27's 0.9650, ladder ≈ v27's 636.4. The hypothesis test is
narrow: whether ANY of v27's residual cost (3 losses/100 vs
v0, the 1 heuristic-leg draw) traces to a quiet-position
above-threshold fire. A tie everywhere is a clean refutation
of the gate as a strength lever (and closes the root-decision
lever); a win vs v27 with CI excluding 0.50 would be a genuine
surprise in the gate's favour. The battery decides — the
diagnostics predict the shape, not the verdict.

**Cost (from v27's measurements — the leaf function is
unchanged, so the cost is identical by construction, and the
gate itself runs only when the tie-break would otherwise fire,
a few times per game).** Fast loop: ~1 h for 2 iterations.
Battery: 100 games vs v27 ~2 h, vs v0 ~1 h, swap ~1.5 h,
ladder ~30 min, tactical ~1 min. Affordable.

**Eval seeds (pre-registered):** h2h vs v27: v34=100 / v27=200;
h2h vs v0: v34=100 / v0=200; ladder seed 100, 20
games/opponent, mcts_sims=100; swap 100 games vs v27, balanced
placement, placement_seed 11, tempo judge (the harness
default), seeds 100/200; tactical `--seed 100`. Fast-loop dev
seeds 1001/2001, 1002/2002, ... (iteration k uses 1000+k /
2000+k, per iter-8 §4's policy).

### 1.3 Mechanism check (post-implementation, pre-fast-loop)

(TODO — after implementation, before any fast-loop game: root
dumps on the 3 tie-fix positions + the no-threat suite
positions + the fl2-g2 ply-22 position, and the full tactical
suites at `--seed 100`. Verifies the dispatch does what §1
says it does.)

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v34 only (after the fast loop):

- 100 games vs v27, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100-game swap vs v27, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder 20 games/opponent (random, heuristic, mcts;
  mcts_sims=100), seed 100 (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).
- If — and only if — v34 passes all 4 clauses: a 100-game
  challenge match vs v18 (seeds 100/200, normal rules); the
  coordinator makes the champion call under rule v2.

**Integration.** `ml/gomoku_ml/scoring.py` gains a
`_has_forcing_threat(board)` helper (the §1.1 rule-2 condition,
in the scorer's own window terms) plus ONLY the
`ENGINE_VERSIONS[34]` registry entry (v27's entry + the
`"tie_break_q_tactical_only"` flag) and passing the flag
through to `HeuristicMCTS`. `ml/gomoku_ml/mcts_heuristic.py`
gains a `tie_break_q_tactical_only` parameter on
`HeuristicMCTS` (default False — v27 and all earlier versions
are unaffected) and the gate in BOTH root-selection methods:
the LIVE `_choose_root_child` (the second definition, line
1322 — verified at runtime via `inspect.getsourcelines`; the
first definition at line 1207 is shadowed dead code and is NOT
touched) and the test-only `_pick_root_move` (kept in sync —
the batch-8/9 dead-code trap). Without a board (synthetic-root
tests), the position cannot be shown tactical, so the gate
blocks the fire — documented in both methods. The v1 kernel
(`kernel_v1.py`), the rollout function, and v27's dispatch are
unchanged. `baseline.py` is frozen and untouched.

## 2. Loss analysis for iteration 34

(TODO — the fast-loop worked games ARE the loss analysis.)

### 2.1 Fast-loop iteration 1 — 10 games vs v27 + 10 games vs v0

(TODO)

### 2.2 Fast-loop iteration 2

(TODO)

### 2.3 Fast-loop iteration 3 — NOT RUN unless iterations 1–2 find a patchable class

### 2.4 Fast-loop iteration 4 — NOT RUN unless iterations 1–3 find a patchable class

### 2.5 Aggregate counts (secondary)

(TODO)

## 3. v34 evaluation

(TODO — battery on the FINAL version only.)

### 3.1 Tactical (clause 3)

(TODO)

### 3.2 Head-to-head vs v27 (clause 1)

(TODO)

### 3.3 Head-to-head vs v0

(TODO)

### 3.4 Swap match vs v27

(TODO)

### 3.5 Ladder (clause 2)

(TODO)

### 3.6 Defence classification

(TODO)

### 3.7 Test suites (clause 4)

(TODO)

## 4. Bar check (v34 vs v27)

(TODO)

## 5. What this means

(TODO)

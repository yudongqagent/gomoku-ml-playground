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

After implementation and before any fast-loop game, v34 was run
on the 4 tactical positions (fresh-seed-0 root dumps,
`/tmp/dump_v34_mechanism.py`) and on the full tactical suites at
`--seed 100`, plus the fl2-g2 ply-22 murky-fire position. This
is an implementation check, not pre-registration evidence and
not an eval — it verifies the dispatch does what §1 says it
does.

**Root dumps reproduce iter-27 §1.3 exactly on the three tie
fixes, and v34 == v27 on every dumped position:**

| Position | Tactical? | Root state (fresh seed 0) | v34 | v27 |
|---|---|---|---|---|
| prev-v1v0s-g26 | YES (White threat) | 4-way tie at 3 visits; gap 0.4167 | FIRES → (9,4) EXPECTED | (9,4) |
| prev-v1v1s-g79 | YES (both) | 7-way tie at 3 visits; gap 0.4167 | FIRES → (8,7) EXPECTED | (8,7) |
| prev-v1bl-g2 | YES (White threat) | 2-way tie at 8 visits; gap 0.1306 | FIRES → (7,5) EXPECTED | (7,5) |
| winning-quiet-setup-2 | NO | NOT a tie: 79 visits on (9,7) | (9,7), as v27 | (9,7) |
| fl2-g2 ply 22 | YES (White threat) | 2-way tie at 8 visits; gap 0.1481 | FIRES → (5,0), as v27 | (5,0) |

All 11 other no-threat suite positions (7 setup positions, both
vcf-two-four-sequence positions, old-suite double-threat-create)
have a UNIQUE visit leader (99 visits) at fresh seed 0 — the
gate changes nothing there, and v34 picks identically to v27
(and correctly) at all of them. The fl2-g2 ply-22 fire is
reproduced exactly, as pre-registered in §1.2 diagnostic 4:
the position IS tactical, so v34 fires there too.

**Full suites at `--seed 100`: old 20/20, hard 29/29**, with
the three tie fixes choosing (8,4), (6,7), (7,5) — identical
choices to v27's battery run. Clause 3 is on track (the final
verdict re-runs the suites on the final tree, §3.1).

**Unit tests (new):** 6 v34 tests in
`ml/tests/test_mcts_heuristic.py` (gate fires on a tactical
board, blocks on a quiet board with RNG over ALL tied children,
blocks without a board, still needs the threshold on a tactical
board, registry entry, `_has_forcing_threat` cases) — all pass,
on BOTH root paths (`_choose_root_child` and `_pick_root_move`).
`ml/tests/test_regression_v34.py` (12 positions copied from
v27's file, same 2 xfails): 11 passed, 2 xfailed.

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

## 2. Loss analysis for iteration 34

### 2.1 Fast-loop iteration 1 — 10 games vs v27 + 10 games vs v0 (pre-patch engine)

**Batches:** `research/iterations/fastloop1-v34-vs-v27.json`
(seeds v34=1001 / v27=2001) and
`research/iterations/fastloop1-v34-vs-v0.json` (seeds v34=1001 /
v0=2001) — development seeds, distinct from the bar battery's
100/200. **vs v27: 5W-4L-1D, win rate 0.5500, CI [0.2737,
0.7986]**, 1186.8 s. Split: as Black 3W-2L (0.60), as White
2W-2L-1D (0.50). **vs v0: 10W-0L-0D, win rate 1.0000, CI
[0.7225, 1.0000]**, 351.0 s. Aggregates recomputed from the raw
records, matching the drivers' headers exactly. Classifier (both
batches): 4/4 losses via open four, 0 missed blocks, 0 clean
missed preventions in 22 threatened turns (losses). Defence is
still perfect. (n=10 per opponent — development signal, not the
bar; the fast loop's job is finding weaknesses, not measuring
strength.)

**The gate changed NO move in any lost game.** A gate-difference
scan (`/tmp/scan_v34_gate_diff.py`) replayed every v34 turn of
all 4 losses and ran BOTH v34 and v27 (fresh seed 0) from each
position: 87 turns compared, picks differ 0 times, gate-caused
differences 0. v34 and v27 are move-identical on every turn of
every lost game — so no loss in this iteration is attributable
to the gate, in either direction.

**Games 1 and 3 (White, 27 and 29 plies) have NO patchable
losing move.** Their walk-backs
(`research/iterations/walkback-fl1-v34-g1.json`,
`walkback-fl1-v34-g3.json`, v1 oracle, seeds 0,1,2) nominate
nothing: at every White turn the actual either does not lose at
all seeds or loses 3/3 with 0 saving alternatives. Distributed
drift.

**Game 8 (Black, 56 plies) nominates ply 40 — and the stronger
oracle refutes it.** The v1 oracle says actual (8,6) loses 3/3
with save (13,6) winning 3/3. Under v34-vs-v27 playouts
(`targeted_playout.py`, seeds 0,1,2): actual (8,6) [-1,-1,-1]
(qualifies), but (13,6) is [-1,-1,-1] too — it loses 3/3 under
strong play. The exact analysis shows (13,6) is the better move
by the setups metric (White setups 5 → 1), but it does not save
the game: the position at ply 40 is lost for Black with no
verified save. (Ply 42's nomination is weaker — (13,6) saves
only 1/3 under the v1 oracle, below the ≥2/3 bar — and the
same move fails at the earlier ply 40, so it was not pursued.)
Distributed drift.

**Game 2 (Black, 64 plies) has ONE verified losing move — and
it is unpatchable.** The v1 oracle nominates plies 48, 50, 54,
56. The stronger oracle (v34-vs-v27 playouts, seeds 0,1,2)
refutes ply 48 outright: actual (1,4) [-1,-1,-1] (qualifies),
but the "save" (0,5) is [-1,-1,-1] too — the exact analysis
shows (0,5) removes all 3 of White's VCF starts, yet the
playout still loses (White's 3 setups remain). But ply 50
CONFIRMS: actual (2,7) [-1,-1,-1] (qualifies) and (0,5) is
[+1,+1,+1] — the save WINS 3/3. (Ply 48's (0,5) does not save
but ply 50's does: White's ply-49 move changed the position.)
The root dump at ply 50 (`/tmp/dump_v34_g2_ply50.py`, fresh
seed 0) exonerates the gate twice over: the position is QUIET
(no threat for either side), (2,7) is the UNIQUE visit leader
(9 visits, our-q +0.0265) and (0,5) is second at 8 visits with
WORSE q (+0.0004). This is a NEAR-TIE (9 vs 8), not an exact
tie — the tie-break never applies — and (0,5) loses on visits
AND on q, so no root rule can promote it: promoting it would
require the CLOSED q-based root-decisions lever, and even q
ranks the loser above the saver. The leaf itself ranks the
loser above the saver; the miss is a search/leaf-horizon
limitation, identical for v27 (v27's root state is
byte-identical: same visits, same q, same pick). No patch —
and no regression position: the save is a quiet edge move
whose value is invisible to every instrument in the program
(the exact analysis even scores it worse: White setups 2 → 3
after (0,5)).

**Iteration 1 verdict: no new patchable weakness class — but
the loop does NOT stop here** (same pre-commitment as iter-23
§2.1 / iter-27 §2.1: two consecutive barren iterations, 40
games). Iteration 2 runs on fresh seeds (1002/2002); if it
also finds no patchable class, the loop stops. No regression
positions added: the one verified losing move has a verified
save no root rule can promote.

### 2.2 Fast-loop iteration 2 — 10 games vs v27 + 10 games vs v0 (LOOP STOPS)

**Batches:** `research/iterations/fastloop2-v34-vs-v27.json`
(seeds v34=1002 / v27=2002) and
`research/iterations/fastloop2-v34-vs-v0.json` (seeds v34=1002 /
v0=2002). **vs v27: 5W-4L-1D, win rate 0.5500, CI [0.2737,
0.7986]**, 945.7 s — identical to iteration 1. Split: as Black
4W-0L-1D (0.90), as White 1W-4L (0.20) — all 4 losses as White.
**vs v0: 10W-0L-0D, win rate 1.0000, CI [0.7225, 1.0000]**,
289.9 s. Classifier (both batches): 4/4 losses via open four, 0
missed blocks, 0 clean missed preventions in 21 threatened turns
(losses). Defence still solved. (One runtime restart interrupted
the walk-back chain after games 1 and 3; games 7 and 9 were
re-run — no work was lost, and the /tmp diagnostic scripts were
recreated from the committed code afterwards.)

**The gate changed moves in this iteration — and cost one game.**
The gate-difference scan (`/tmp/scan_v34_gate_diff.py`, recreated
after the restart) compared 93 v34 turns across the 4 losses and
found THREE gate-caused differences (after zero in iteration 1's
87 turns):

- **Game 7, ply 37 — the gate COST the game.** The position is
  QUIET (no threat for either side). The root (fresh seed 0)
  has a 3-way tie at 6 visits: (6,2) our-q +0.0007, (2,6)
  +0.0680, (7,11) +0.0001. Gap 0.0673 > τ=0.05, so v27's
  tie-break FIRES and picks (2,6); v34's gate BLOCKS the fire
  and the RNG coin flip over the tied triple lands on (7,11) —
  the actual losing move. Under v34-vs-v27 playouts
  (`targeted_playout.py`, seeds 0,1,2): (7,11) [-1,-1,-1],
  (2,6) [+1,+1,+1] (SAVES), (6,2) [-1,+1,-1] (split). The
  gate converted a deterministic save into a 1/3 coin flip and
  the flip landed wrong. This is direct evidence AGAINST the
  hypothesis: a quiet-position above-threshold fire carried
  real signal — the 0.0673 gap separated a saver from a loser
  and a split. (The walkback's nominated save (11,7) also wins
  3/3, but it sits at 3 visits with our-q −0.0629, rank 5/96 —
  outside the tie, unreachable by any root rule. The loss is
  recorded as regression xfail fl2-g7-ply37, expected (11,7)
  and (2,6).)
- **Game 1, ply 31 — outcome-free.** v34 played (4,9), v27's
  fired pick was (9,3) (gap 0.0533, quiet). Under v34-vs-v27
  playouts BOTH lose 3/3 — the gate changed the move but not
  the outcome. (The exact analysis even scores (9,3) worse: it
  leaves Black 3 VCF starts vs 0 after (4,9). The v1 oracle
  had said (4,9) wins 2/3 here — wrong again under strong
  play.)
- **Game 3, ply 15 — no match effect.** Fresh-seed-0 v34 picks
  (3,8) vs v27's (3,7) (gap 0.0806, quiet), but the MATCH move
  was (3,7) — v34's coin flip landed on v27's fired pick, so
  the gate changed nothing in the actual game.

**The other nominations were all refuted by the stronger
oracle.** Game 1 ply 23: (9,7) loses 3/3 (the v1 oracle said
win/win/win). Game 9 ply 13: (10,5) wins only 1/3 (below the
≥2/3 bar — a split, not a save). Game 3 ply 69: (10,7) loses
3/3 (only the first-listed of five v1-oracle saves was verified
under the strong oracle; the others were not pursued — same
class, same oracle, and the pattern across both iterations is
5 of 7 v1-oracle saves failing under strong play).

**No patchable weakness class in this iteration either.** The
one gate-caused loss (game 7) is "fixed" only by removing the
gate — i.e. by being v27 — which is abandoning the hypothesis,
not a patch. The other confirmed losing move (game 7's (11,7)
save) is unreachable by any root rule. **The fast loop STOPS
here (2 iterations)** per the pre-commitment in §2.1 (two
consecutive barren iterations, 40 games). v34-as-batteried is
v34-as-pre-registered. The battery is the verdict on whether
the gate costs or saves games at scale — the fast loop's
evidence cuts against it (1 game cost, 0 saved, in 40 games),
but n=40 cannot measure a small effect.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast-loop totals (2 iterations, 40 games): vs v27 10W-8L-2D
(0.5500); vs v0 20W-0L-0D (1.0000). Colour splits vs v27: as
Black 7W-2L-1D (0.75), as White 3W-6L-1D (0.35) — the White
gap is iteration 2's 1W-4L (n=10 per colour per iteration;
noise, not a finding). Fast-loop defence: 8/8 losses via open
four, 0 missed blocks, 0 clean missed preventions (43
threatened turns in losses). Defence remains solved.
Gate-difference totals: 180 turns compared in lost games, 3
gate-caused differences (all in iteration 2), 1 game cost
(g7-ply37), 0 games saved, 1 outcome-free (g1-ply31), 1 with no
match effect (g3-ply15). No patches were made in this
iteration — v34-as-batteried is v34-as-pre-registered (the
tactical gate on τ=0.05). One regression xfail added:
fl2-g7-ply37 (the gate's documented cost).

## 3. v34 evaluation

Battery run Wed 2026-11-18 on the FINAL version (v34-as-batteried =
v34-as-pre-registered; no patches in the fast loop). All numbers
below were recomputed from the raw per-game records
(`/tmp/verify_v34.py`) and cross-checked against the driver headers
(all MATCH). Artefacts: `batch-v34-vs-v27.json`,
`batch-v34-vs-v0.json`, `batch-v34-vs-v27-swap.json`,
`ladder-v34.json`, `ladder-v34.log`, `tactical-v34.json` — all
`complete=true`, 100 games per match leg.

The battery chain ran across THREE process lifetimes (two runtime
restarts SIGTERM'd it: first at 66/100 games of the vs-v27 leg,
second at 25/100 of the swap leg). Auto-resume saved all progress
(`resumed_from_game` 66 vs v27, 25 swap; the vs-v0 leg and the
ladder ran clean in a single process each). Total chain time
~5.3h. **The resume has a seed-replay side effect — see §3.8.**
It does not change any verdict below (de-duplicated numbers are
reported alongside nominal ones throughout).

### 3.1 Tactical (clause 3)

`eval_tactical_full.py --version 34 --suite both --seed 100`:
**old 20/20, hard 29/29** on the corrected suite (the three
tie-fix positions choose (8,4), (6,7), (7,5) — identical to v27).
Clause 3 PASSES.

### 3.2 Head-to-head vs v27 (clause 1)

`batch-v34-vs-v27.json` (seeds 100/200, normal rules, 2253.6s
for the resumed leg):

- **45W-51L-4D over 100 games, win rate 0.4700, Wilson 95% CI
  [0.3751, 0.5671].**
- Colour split: as Black 31W-17L-2D (0.6400); as White
  14W-34L-2D (0.3000). Black won 65/100 games overall. The
  6-game margin is split evenly by colour: v27-as-Black beat
  v34-as-White 34-14-2, while v34-as-Black beat v27-as-White
  31-17-2 — v27 outplayed v34 by 3 games in each colour. There
  is no colour-specific gate effect; v27 is just slightly
  better (or equal within noise — the CI includes 0.50, so
  "v27 is better" is not proven either).
- De-duplicated (see §3.8): **29W-31L-3D over 63 unique games,
  0.4841, CI [0.3652, 0.6049].** Pre-resume prefix (games 0-65,
  all unique): 30W-33L-3D, 0.4773, CI [0.3614, 0.5957]. Same
  conclusion on every cut: v34 does not beat v27.

**Clause 1 FAILS** (win rate < 0.50 and CI lower bound 0.3751 <
0.50, on both the nominal and the de-duplicated numbers).

### 3.3 Head-to-head vs v0

`batch-v34-vs-v0.json` (seeds 100/200, normal rules, 3015.7s):

- **97W-3L-0D, 0.9700, CI [0.9155, 0.9897].** As Black
  47W-3L-0D (0.9400, all 3 losses as Black, as in v27's
  battery); as White 50W-0L-0D (1.0000).
- Nominally this equals v18's 0.97 and is one win better than
  v27's 96W-3L-1D (0.9650) — but the nominal comparison is
  mostly duplication noise (§3.8): de-duplicated, v34 is
  **38W-3L-0D over 41 unique games, 0.9268, CI [0.8057,
  0.9748]**, vs v27's de-duplicated 37W-3L-1D over 41 unique,
  0.9146. Same 3 losses, one fewer draw — v34 is one game
  better than v27 vs v0 on unique games. The two engines play
  nearly identically against v0: 26 of the 41 unique games are
  shared move-for-move, all with the same result — the
  pre-registered expectation (gate bite vs v0 ≈ zero) holds.

### 3.4 Swap match vs v27

`batch-v34-vs-v27-swap.json` (seeds 100/200, balanced placement,
placement_seed 11, tempo judge, 4866.3s):

- **56W-42L-2D, 0.5700, CI [0.4722, 0.6627].** As Black
  36W-12L-1D (0.7449, 49 games); as White 20W-30L-1D (0.4020,
  51 games).
- De-duplicated: 55W-41L-2D over 98 unique, 0.5714, CI
  [0.4726, 0.6649] — essentially unchanged (only 2 duplicate
  games in the swap leg).
- The swap edge is positive but not significant (CI includes
  0.50), and it points the OTHER way from the normal-rules leg
  (0.47). With both CIs straddling 0.50 and overlapping
  heavily, no firm conclusion — v34 is statistically
  indistinguishable from v27 under swap too.

### 3.5 Ladder (clause 2)

`eval_version.py --version 34 --ladder-games 20 --mcts-sims 100
--seed 100` (`ladder-v34.json`, `resumed=false` — clean single
run):

| Leg | v34 | v27 (bar reference) |
|---|---|---|
| vs random | 20W-0L-0D | 20W-0L-0D |
| vs heuristic | **20W-0L-0D** | 19W-0L-1D |
| vs mcts | **15W-5L-0D** (Black 8-2, White 7-3) | 13W-7L-0D |
| Ladder Elo | estimate **None**, interval **[286.6, None]** | 636.4, [240.3, 1032.5] |

v34 beats v27 on EVERY ladder leg. The 20-0 heuristic leg is
the first perfect heuristic leg in the program (v18 and v27
both stopped at 19-0-1), and the 15-5 mcts leg ties v29's
program best.

The Elo point estimate is `None` because the instrument
saturates at a perfect score: `_elo_from_score`
(`evaluate.py:126`) returns None for score 0 or 1 (the logistic
formula diverges), and with both anchored legs at 1.0 there is
no finite per-opponent estimate and no upper bound — only the
Wilson-lower-bound-derived floor of 286.6. Clause 2 as written
("ladder Elo > 636.4") therefore has three readings:

1. **Leg result (the plan's own operationalization):** the
   batch-10 plan states "beating [636.4] effectively requires
   a 20-0 leg" — v34 has exactly that, beating v27's 19-0-1
   leg head-on. **PASS.**
2. **Point estimate:** undefined (None) — the clause is not
   evaluable as a number comparison. **UNDECIDED.**
3. **Interval lower bound:** 286.6 < 636.4. **FAIL.**

Reading 1 is the faithful one — the plan defines what beating
636.4 means in practice, and v34 meets it — so clause 2 is
recorded as PASS (leg result) with the estimator saturation
stated here in full. The ambiguity is flagged to the
coordinator; it does not affect the overall verdict (clause 1
fails regardless). Program note: the ladder cannot distinguish
a 20-0 performance from infinite strength — at the top of the
ladder the instrument is saturated, and a stronger anchor (or
more games per leg) is needed to rank future versions.

### 3.6 Defence classification

`classify_losses.py` on the three battery batches:

| Batch | Losses | Via open four | Missed blocks | Clean missed preventions | Threatened turns |
|---|---|---|---|---|---|
| vs v27 | 51 | 51 | 0 | 3 (3 games) | 228 |
| vs v0 | 3 | 3 | 0 | 0 | 14 |
| swap vs v27 | 42 | 42 | 0 | 3 (3 games) | 226 |
| **Pooled** | **96** | **96** | **0** | **6 (6 games)** | **468** |

0 missed blocks holds — defence is still solved in the block
sense. But the clean missed preventions are **non-zero for the
first time in the program** (v27: 0 in 183 threatened turns;
v29: 0 in 463). The 6 are really 4 distinct turns: games 31
and 97 (vs v27) are the same game replayed (§3.8), as are swap
games 43 and 91. All 6 are v34 as White.

Are they gate-caused? NO — verified on three of the four
distinct instances (vs-v27 g31-ply21, g61-ply21; swap
g73-ply35) with fresh 100-sim root dumps for v34 AND v27:

- g31-ply21: played (9,6), unsafe; safe (8,10), (9,11) exist.
  (9,6) is the UNIQUE visit leader (5 visits) for BOTH
  engines — no tie, so the tie-break (gated or not) never
  fires. The safe moves sit at 4 visits with BETTER q
  (−0.6875 vs −0.8117): the leaf knows, the visit count
  doesn't. Identical dumps for v27.
- g61-ply21: played (8,1), unsafe; safe (3,5), (4,5), (8,5)
  exist. (8,1) unique leader at 11 visits, both engines
  identical — and here even the leaf ranks the unsafe move
  best (−0.6364 vs −0.6755…−0.8420 for the safe moves).
- swap g73-ply35: played (11,9), unsafe; safe (11,5), (10,6)
  exist. (11,9) unique leader (5 visits), both engines
  identical; safe moves at 3 visits with worse q.

So the prevention misses are **search-level misses shared
identically with v27, not a v34 regression and not
gate-caused**. The rate is low (6/468 = 1.3% of threatened
turns) and the opponent is stronger than in any previous
battery (v27, not v7) — more creator situations arise. Flag
for H3's v35 forensics: v27 itself misses preventions the
exact analysis catches, in two flavours (leaf-knows-but-visits-don't,
g31; leaf-blind, g61/g73). The fourth distinct instance (swap
g43/g91 ply 17) was not individually dumped.

### 3.7 Test suites (clause 4)

- ml pytest (full suite, run on the final code): **414 passed,
  7 skipped, 61 xfailed, 0 failures** (814.39s, contended).
  (Counts differ from ML batch 2's 423/1/58 because that
  branch is not merged to master; 0 failures is what matters.)
- npm test **55/55**; `npx tsc --noEmit` clean.
- `ml/gomoku_ml/baseline.py`: **0 lines changed** vs `a4ed543`
  (frozen file untouched).
- Clause 4 PASSES.

### 3.8 Battery hygiene finding: duplicated games (program-level)

Counting unique move-lists per batch:

| Batch | Games | Unique | Cause |
|---|---|---|---|
| vs v27 | 100 | **63** | resume seed-replay (below) + determinism |
| vs v0 | 100 | **41** | engine determinism (no resume involved) |
| swap vs v27 | 100 | 98 | — |

**Resume seed-replay.** `selfplay_batch.py` creates each engine
ONCE with the fixed seeds (`--seed-a 100`, `--seed-b 200`), so
each engine's RNG is a single stream across the run. On
auto-resume the engines are re-created with the same seeds and
their streams restart from the beginning. In the vs-v27 leg
(resumed from game 66) this replayed games 0-33 EXACTLY as
games 66-99 — all 34 pairs duplicated move-for-move (the colour
schedule aligns: game i and game i+66 have the same
`a_colour`). The driver's docstring warns only that "the
placement RNG restarts" — the engine-RNG restart is the
bigger effect. (The replay is not always exact: v27's vs-v7
leg, resumed from 53, has 94 unique games — the more the
engines diverge, the more process-level variation breaks the
replay. v34-vs-v27 games are short and forcing enough that
the replay was perfect.)

**Engine determinism.** The vs-v0 leg ran in ONE process with
no resume, yet has only 41 unique games (one move-list
repeated 16 times). v0 is near-deterministic and v34's RNG
matters only at root ties, so most seed draws collapse to the
same short game. v27's vs-v0 batch shows the SAME 41-unique
phenomenon (de-duplicated 37W-3L-1D, 0.9146, vs nominal
96W-3L-1D, 0.9650) — so the nominal vs-v0 win rates in the
program are inflated by duplication, on both sides equally.

**No verdict is overturned by de-duplication:**
v27's bar pass survives (vs v7 de-duplicated: 71W-19L-4D over
94 unique, 0.7766, CI [0.6825, 0.8490]; swap de-duplicated
0.7892 over 83 unique). v34's clause-1 failure survives
(0.4841 de-duplicated). But every past battery's nominal CI
overstates the effective sample size. Recommendations for the
coordinator: (1) per-game seeding (`seed + game_index`) so
resumed runs cannot replay; (2) a uniqueness audit
(unique move-lists + de-duplicated aggregates) as standard
battery hygiene in every iteration log.

## 4. Bar check (v34 vs v27)

| Clause | Requirement | v34 result | Verdict |
|---|---|---|---|
| 1 | h2h vs v27 ≥ 100 games, win rate > 0.50, Wilson 95% CI lower > 0.50 | 45W-51L-4D, 0.4700, CI [0.3751, 0.5671] (de-dup: 29W-31L-3D over 63 unique, 0.4841) | **FAIL** |
| 2 | Ladder Elo > 636.4 | Heuristic leg 20W-0L-0D (first ever; beats v27's 19-0-1 — the plan's stated requirement); mcts leg 15-5 (ties program best); Elo point estimate None (instrument saturated at a perfect score), interval [286.6, None] | **PASS (leg result)** — see §3.5 for the three readings |
| 3 | Tactical ≥ 20/20 + 29/29 (corrected suite) | 20/20 old, 29/29 hard | **PASS** |
| 4 | All test suites green | pytest 414/7/61, 0 failures; npm 55/55; tsc clean; baseline.py untouched | **PASS** |

**v34 is REJECTED, 3/4 clauses.** No v18 challenge match is
triggered (it requires all 4 clauses). v18 stays the strength
champion; v27 stays the bar reference.

## 5. What this means

**The hypothesis is refuted.** The tactical gate does not
improve on v27: v34 loses the clause-1 leg on points (0.47
nominal, 0.4841 de-duplicated) and is statistically
indistinguishable from v27 on every other leg (swap 0.57 with
CI straddling 0.50; vs v0 one unique game better; ladder legs
better but the Elo instrument saturated). The pre-registered
expectation — v34 ≈ v27 — was right, and the fast loop
supplied the mechanism for the small deficit: quiet-position
fires DO sometimes carry real signal (fast-loop game 7 ply 37,
where v27's blocked fire was the only saving move), so
blocking them costs games at least as often as it saves any.
The batch-9 caution that motivated the gate (a large gap
between two verified losers in a murky midgame) turned out to
be a tactical position anyway (§1.2) — the gate fires there
too, by design. There is no evidence that quiet-position fires
are systematically worse than tactical ones.

**The gate is a near-no-op.** Against v0 the two engines play
nearly identically (26/41 unique games shared move-for-move);
in the fast loop, 180 turns compared across lost games yielded
only 3 gate-caused differences. v34's main observable
difference from v27 is one lost fast-loop game and a 6-game
nominal deficit over 100 battery games — both consistent with
"the gate changes almost nothing, and what it changes is
slightly bad."

**The tie-break lever is now doubly bounded — CLOSED.** v23's
unconditional q-aware tie-break hurt vs v0 (0.91); v27's
τ=0.05 threshold fixed it (0.9650) and passed the bar; v34's
tactical conditioning of the same threshold adds nothing.
Further conditions on the tie-break (position-dependent
thresholds, phase-dependent thresholds, different τ per
colour) should not be tried without a new mechanism — the
space has been explored from three directions and the
threshold version is the local optimum.

**Two program-level findings ride along with this rejection:**
(1) the resume seed-replay bug (§3.8) — resumed batteries
replay earlier games, overstating effective sample size
(63 unique, not 100, in the clause-1 leg); per-game seeding
and a uniqueness audit are the fixes; (2) the ladder is
saturated at the top — a 20-0 heuristic leg, the best in the
program, yields no finite Elo estimate, so the ladder can no
longer rank versions that beat the heuristic anchor outright.
Both are coordinator-level tooling decisions, flagged here
rather than fixed in this branch (changing the driver or the
anchors mid-batch would break comparability with v27's bar).

**Defence:** 0 missed blocks holds, but clean missed
preventions are non-zero for the first time (6 in 468
threatened turns, 1.3%) — verified shared with v27, not a
v34 regression. The prevention misses are a v27-family
search-level property (unique visit leaders; in 2 of 3 dumped
instances the leaf itself ranks the unsafe move best) and
belong to H3's v35 forensics, not to this workstream.

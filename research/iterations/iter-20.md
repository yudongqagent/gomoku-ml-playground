# Iteration 20 — v20: gap-three demotion in the v1-kernel leaf (batch 7, H3)

**Status: IN PROGRESS — fast loop complete (2 iterations); mechanism
found and causally verified; v20 pre-registered below (§1) BEFORE any
eval of v20. Battery pending.**

**METHOD (carried over from batch 3/4/5/6).** The fast 10-game loop
stays the finding method: play 10 games, read every move, find each
game's losing move (solver/deeper-search verified, never guessed),
patch the algorithm to reproduce the perfect move in those positions
(regression test), repeat — max 4 iterations, or stop when an
iteration finds no new patchable weakness class. The 4-clause bar
stays the validation gate. Fast-loop games use development seeds
distinct from the bar battery's pre-registered seeds (100/200).
**Batch-7 correction (iter-17 §5): the fast loop runs against v0 as
well as v7** — v17's weakness only showed at battery scale against
v0, where the fast loop never looked. The full battery runs on the
FINAL version only. The 100-game battery is the verdict —
small-sample smokes mislead (v8, v11, v15's two n=40 false positives
in one iteration); single-position diagnostics cannot validate a
global search change.

**BASE ENGINE.** v20 is built on v14 (batch 5, H3: v7's PUCT MCTS
with the v1-kernel gradient leaf), ported exactly from
`research/batch-6` (which carries batch 5's v14 implementation
verbatim, including `kernel_v1.py`, 0 mismatches). v17's root rule
(tie-prior-forcing) and threat check are NOT in v20: the root-decision
lever is EXHAUSTED, and the batch-7 plan tasks H3 with fixing the
leaf itself, not re-combining (v17: the combination does not add).
The fast loop therefore runs on v14, the clean v1-leaf engine with a
plain most-visited root.

## 0. The problem (from batches 5–6)

The v1-kernel leaf (v14: v7's PUCT MCTS with the gradient leaf's
input scores ported from the v0 kernel to a vectorized v1 5-window
attack scorer) has a specific, measured weakness against greedy
play:

- v14: vs v7 0.6000 (clause 1 passes by 0.002), vs v0 0.6000,
  ladder Elo -34.9 (clause 2 FAILS; heuristic anchor 9-11).
- v17 (v14's leaf + v12's root rule + v13's threat check): vs v7
  0.6650 (best h2h vs v7 in the program), vs v0 0.5150 with Black
  0.3700, ladder Elo -17.4 (clause 2 FAILS; heuristic 9-10-1),
  swap vs v7 0.7400 (strongest swap result of any version).
- iter-14 §5's hypothesis (unverified): "the v1 gradient's denser
  threat signal seems to help against a searching opponent and hurt,
  or at least not help, against greedy 1-ply play — the gradient
  rewards building threats, and against an opponent that takes every
  immediate threat at face value, the extra threat-building does not
  convert." The three worked Black losses to v0 (§2.6) showed
  distributed drift, no blunder, so there was no single mechanism to
  point at — until this iteration (§2.0).

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-7 plan: ladder Elo
190.8, on master at `bf866f7`). Clause 1 is head-to-head vs v7
(normal rules, >= 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50); clause 2 is ladder Elo
strictly higher than v7's 190.8 (identical 20 games/opponent:
random, heuristic, mcts, mcts_sims=100); clause 3 is tactical pass
rate >= v7's (20/20 old suite + 29/29 hard suite); clause 4 is all
test suites green (ml pytest, npm test, tsc). **Secondary
head-to-heads, also pre-registered:** v20 vs v0 (100 games) and a
100-game swap match vs v7 under the tempo judge (the harness
default). Non-transitivity is recorded prominently.

**The improvement: v20 = v14 with gap-three makers demoted from
the strong tier to the developing tier in the v1 breadth.** Everything
else is v14 unchanged: 100 sims, c_puct 1.5, v1 priors over all
candidates (temperature 2.0 — the sharp prior is load-bearing at
100 sims; flattening is catastrophic, iter-13 §2.4), win/block
scans before search, most-visited root child with RNG tie-breaks,
terminal leaf values identical. The ONLY change is the quiet-leaf
gradient's three-tier weighting: a three-tier move (v1 attack in
[4000, 20000)) counts as STRONG (weight 1.0) only if it is a TRUE
open-three maker, and as DEVELOPING (weight 0.25) if it is a
gap-three maker.

**The defect being fixed, measured (§2.0).** The v1 scorer counts
a gap three (XX_X) as an "open three" (4006 ≈ 4000), but a gap
three is killable by a single opponent move occupying the gap,
while a true open three cannot be killed in one move — and greedy
v0 finds the gap block automatically. Gap-three breadth is phantom
against greedy play. Verified causally: 6/6 engine-swap
continuations flip a v14 Black loss to v0 into a win when v7's
move replaces v14's; the ply-10 decomposition shows the v1
gradient's +4.0 preference for the losing cluster four comes
entirely from the three tier, 4 of whose 6 makers are gap-three
makers; fast-loop game 6 ply 28 reproduces the flip at a fresh
seed (v14 plays the gap-heavy loser, v7 plays the save).

**The filter (exact, no new machinery).** The v1 score cannot
distinguish a gap-three maker (4006) from an open-three maker
(4000+). The v0 kernel can: a true open three scores >= 1000 in
v0's contiguous-run scoring (the (3,2) weight), while a gap-three
maker scores ~40–200 (v0 cannot see the gap pattern — the longest
contiguous run through the move is 2). So: three-tier AND v0
attack >= 1000 → true open-three maker (strong); three-tier AND
v0 attack < 1000 → gap-three maker (developing). Both kernels are
0-mismatch verified (`kernel_v1.py` ported unchanged from batch 5;
`kernel_numpy.py` from batch 1). Edge cases, checked: a move
creating a true open three in one direction and a gap three in
another has v0 >= 1000 (from the true open three) and counts as
strong — correct, it does create a true open three. A move
creating fours in any direction scores v1 >= 20000 and is in the
four tier, not the three tier — unaffected by the filter.

**What does NOT change (pre-registered exclusions).** Broken-four
makers (20,006) stay STRONG: a broken four has exactly one winning
square, the same as a straight four — both force a block
identically; the defect is specific to gap THREE makers
(non-forcing, killable), not broken fours (forcing). The developing
tier [80, 4000) is unchanged. The cap stays ±0.2 — a static
heuristic must never outweigh an exact terminal signal. No root-rule
change (EXHAUSTED — v20 keeps v14's plain most-visited root; the
fix is in the leaf itself, as tasked). No exploration change
(CLOSED). No sims change (100 both colours — more search is NOT
monotonically better). No prior change (sharp prior load-bearing).
No threat check, no opening book, no q-based root decision
(CLOSED). No change to `baseline.py` (FROZEN). No kernel re-sync
(NumPy/WebGPU/Metal) unless v20 passes the bar. The v1 kernel is
used directly, not re-ported.

**The scale question (pre-registered rule, Step-0 value).** The
demotion narrows the breadth-difference distribution, so the tanh
scale is re-derived from the DEMOTED distribution by v7's rule
(scale ~= 1.8 x std) — the same rule v7 (4.0 from std 2.26) and
v14 (7.5 from std 4.09) used — measured in Step 0 (§1.1) on the
same 304-position sample as iter-14, BEFORE any v20 eval. The
scale is fixed here from the measured distribution, NOT tuned on
any v20 eval result. Value: **5.5** (1.8 x std 3.02 = 5.44,
rounded to the nearest 0.5).

**What this change is meant to fix.** v14's gradient overvalues
cluster positions rich in gap-three makers and undervalues spread /
forcing positions, steering the search into fast tactical races
that greedy 1-ply play wins (v14's Black losses to v0 are mostly
26–30 plies; the swap test shows v7's spread/forcing moves flip
them into slow grinds v14 wins). Demoting the phantom breadth
should restore the search's discrimination between forcing moves
(four-makers, true open-three makers) and non-forcing breadth
(gap-three makers) — the distinction the race turns on. Expected
effect, stated in advance: v20 should recover v14's Black deficit
against v0 (20 Black losses → toward v7's 13) WITHOUT losing
v14's edge over v7 (0.6000) — because the demotion removes
phantom breadth both sides share against a searching opponent
(v7's positions hold gap-three makers too), while against greedy
v0 only OUR phantom breadth was being counted (v0 never builds
gap-three setups deliberately). If v20 still loses the race to
v0, the gap-three correction is not the binding constraint either
— and the battery decides.

**Eval seeds (pre-registered):** h2h vs v7: v20=100 / v7=200; h2h
vs v0: v20=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Harness commands follow iter-7 §3 exactly
(`selfplay_batch.py`, `eval_version.py` for the ladder,
`eval_tactical_full.py` for tactical). Fast-loop dev seeds
1011/2011 (iter 1 vs v0) and 1012/2012 (iter 2 vs v7) are distinct
from the battery's 100/200.

### 1.1 Step 0 — cost and distribution measurement

`ml/scripts/bench_v20_leaf.py` on 346 boards replayed from
batch-v7-vs-v1.json (first 20 games, every 4th ply from ply 8;
290 quiet), median of 3 rounds x 50 calls. (Quiet count 290 vs
iter-14's 304: a sampling-convention difference — this bench
follows `bench_v16_leaf.py`'s sampling, 1-ply shifted from
iter-14's; the v14 breadth distribution is re-measured on THIS
sample for the comparison, so the derivation is apples-to-apples.
Re-measured v14 std 4.91 vs iter-14's reported 4.09 reflects the
same sample difference; the v20 scale derives from the v20 std on
this sample.)

| Call | Median ms/call |
|---|---|
| v20 `_attack_gradient_v1_gap` (full) | 7.40 (min 5.94, max 8.12) |
| v20 `_gradient_v1_gap_leaf_value` (full) | 8.01 (min 7.80, max 8.07) |
| v14 `_attack_gradient_v1` (full) | 3.40 (min 2.74, max 4.23) |
| v7 `_attack_gradient` (full) | 2.16 (min 2.15, max 2.36) |
| v6 `_terminal_leaf_value` | 1.78 (min 1.50, max 1.95) |

The v20 gradient costs ~7.4 ms per quiet leaf vs ~3.4 ms for
v14's on this run (absolute numbers are contention-noisy; iter-14
measured v14 at 5.76 ms — the ordering is what matters: v20 ≈
2x v14, one extra kernel call). At up to ~100 leaf calls per move
that is at most ~+0.4 s/move over v14, ~+15 min per 100-game
match. Affordable.

Demoted breadth-difference distribution on the 290 quiet
positions: mean -2.00, std 3.02, p10 -6.00, p50 -1.75, p90 +1.23;
|diff| >= 2 in 55.5%, >= 4 in 27.2%, >= 6 in 11.7%. (v14 on the
same sample: mean -2.98, std 4.91; |diff| >= 2 in 61.7%, >= 4 in
40.7%, >= 6 in 25.5%.) Scale: 1.8 x 3.02 = 5.44 -> **5.5**
(pre-registered above). Mean gap-three makers per position: ours
1.50, theirs 2.80 (the side to move holds fewer — the opponent
just moved and just created them; the demotion therefore hits the
opponent's breadth harder and shifts the mean diff from -2.98 to
-2.00 — less pessimistic for the player to move, as it should be:
part of the deficit was phantom). The demotion changes the breadth
difference in 58.3% of quiet positions (mean |change| 1.71) — a
substantial revision of the signal, not a tweak at the margin.

### 1.2 Implementation (pre-registered)

- `ml/gomoku_ml/mcts_heuristic.py`: `_breadth_v1_gap(attack_v1,
  attack_v0, empty_mask)` (strong = four-makers v1 >= 20000 +
  true open-three makers v1 in [4000, 20000) with v0 >= 1000;
  developing at 0.25 = [80, 4000) + gap-three makers v1 in [4000,
  20000) with v0 < 1000), `_attack_gradient_v1_gap` (one
  `attack_v1_both` + one `attack_both` call, tanh scale 5.5, cap
  ±0.2), `_gradient_v1_gap_leaf_value` (terminal values identical
  by construction), registered as `leaf_mode="gradient_v1_gap"` in
  `_LEAF_FUNCTIONS`. The v1 kernel (`kernel_v1.py`) is used
  directly, unchanged; the v0 kernel (`kernel_numpy.py`) is used
  as the filter, unchanged.
- `ml/gomoku_ml/scoring.py`: version 20 in `ENGINE_VERSIONS`
  (`{"name": "scoring-v20", "defence_weight": 1.0, "threat_check":
  False, "mcts": True, "mcts_sims": 100, "leaf_mode":
  "gradient_v1_gap"}`), dispatching through the existing generic
  `HeuristicMCTS` path (no new dispatch code).
- Tests: `ml/tests/test_mcts_heuristic.py` gains v20 leaf tests
  (gap-three demotion unit cases, terminal identity with v14,
  antisymmetry); `ml/tests/test_regression_v20.py` +
  `ml/gomoku_ml/regression_positions_v20.json` record the §2
  critical positions.

### 1.3 Direction checks (NOT evals — identity/direction only)

After implementation and before the battery, v20's search (100
sims, seed 0) is run on the §2 critical positions to check the
patch has the intended DIRECTIONAL effect — whether it plays the
oracle-verified save where v14 played the loser. These are
single-position diagnostics: they CANNOT validate the version
(standing lesson), they only check the patch does what it was
designed to do at the positions that motivated it. The battery
(§3) is the verdict.

**Results (measured, seed 0).** v20 plays the save at **3 of 9**
positions; v14 plays it at **0 of 9**:

| Position | v20 plays | v14 plays | Expected (save) | Verdict |
|---|---|---|---|---|
| bl-ply8 | (7,10) [x,y] | (7,10) | (8,5) | miss (both) |
| bl-ply10 | (8,5) | (7,9) | (5,12) | miss (v20 avoids the loser, not the save) |
| bl-ply12 | (5,12) | (6,11) | (5,12) | **SAVE** |
| bl-ply14 | (6,9) | (4,9) | (6,9) | **SAVE** |
| bl-ply16 | (9,4) | (9,4) | (6,9) | miss (both — prior gap) |
| fl1-g6-ply28 | (0,9) | (7,5) | (4,9)/(0,9) | **SAVE** |
| fl1-g6-ply34 | (6,11) | (6,11) | (4,2) | miss (both — starvation) |
| fl1-g5-ply27 | (12,10) | (5,11) | (7,6)/... | miss (both — beyond-horizon) |
| fl2-g5-ply65 | (11,9) | (11,9) | (13,6)/... | miss (both — beyond-horizon) |

The three saves v20 finds are exactly the positions where the
gap-three mechanism was the driver (bl-ply12: the distant
four-maker; bl-ply14: the forcing open three over the flashy
double open three; fl1-g6-ply28: the fresh-seed confirmation).
The six misses are the positions where §2 predicted the fix
would not reach: starvation (fl1-g6-ply34, prior 0.0214),
beyond-horizon value error (fl1-g5-ply27, fl2-g5-ply65),
prior-dominated (bl-ply16, 200091 vs 20252), and the deeper-tree
effect (bl-ply8). At bl-ply10 v20 avoids the losing move (9,7)
but plays an unverified move (5,8) rather than the save — a miss
on the regression, recorded honestly; whether (5,8) also wins was
not tested and is not claimed. The patch does what it was
designed to do, at the positions that motivated it, and nothing
it was not designed to do. The 6 misses are xfails in
`test_regression_v20.py`; the 3 saves are passing tests.

## 2. Loss analysis for iteration 20

### 2.0 Battery-line forensics (before the fast loop)

Before running fresh fast-loop games, this iteration mined the two
committed 100-game batteries (v14-vs-v0 on `research/batch-5`,
v17-vs-v0 on `research/batch-6`) for the shared losing lines. The
forensics below use only committed records plus fresh engine runs at
seed 0 — no new evals, no battery seeds consumed.

**Transposition first.** v17's Black losses are nearly all the SAME
line: 27 of 31 are 28 plies, in 12 distinct lines (top: x5, x4, x4,
x3); all 17 White losses are 25 plies in 4 distinct lines. v17's
near-deterministic tie-prior-forcing root rule collapses its 100
games into 31 distinct lines — so v17's 0.37 as Black is largely
ONE bad line repeated, a transposition artefact at battery scale
(iter-17 §2.1's fast-loop finding, now confirmed at battery scale).
v14 (plain visits root, RNG tie-breaks) plays 88 distinct lines, so
its 0.60 vs v0 (20 Black losses, 14 of them 26–30 plies) is a real
leaf-level effect, not transposition: v14 loses ~7 more Black games
to v0 than v7 does (20 vs 13).

**The shared line.** One 28-ply Black loss line is IDENTICAL in
both batteries (v14 x2, v17 x4):

(7,7),(7,8),(8,7),(6,7),(8,9),(8,8),(9,8),(7,6),(10,7),(7,10),(9,7),
(11,7),(11,6),(12,5),(9,4),(5,8),(4,9),(6,8),(4,8),(6,9),(9,6),(9,5),
(9,10),(9,9),(10,6),(4,7),(3,6),(8,11) — White (v0) wins at ply 28.

Both v1-leaf engines play the same moves as Black and lose to v0 in
28 plies. The leaf is shared; the root rule differs. This is a
leaf-level loss.

**Divergence scan.** At each Black ply of the shared line, v14 and
v7 were run fresh (seed 0). v14 reproduces the actual line at 11/14
Black plies (plies 2 and 24 differ by RNG state); v7 diverges at
SEVEN plies:

| Ply | Actual (v14) | v7 | Character |
|---|---|---|---|
| 2 | (8,7) adjacent | (6,8) diagonal | spread vs cluster |
| 8 | (10,7) cluster extension | (5,8) tenuki | away vs contact |
| 10 | (9,7) cluster four | (12,5) distant four-maker | distant vs cluster |
| 12 | (11,6) cluster fill | (12,5) distant | distant vs cluster |
| 14 | (9,4) double open three | (9,6) single open three | forcing vs flashy |
| 16 | (4,9) direct block | (9,6) forcing four-maker | counter-threat vs block |
| 20 | (9,6) | (9,5) | (too late — see below) |

v7's moves BUILD NEW THREATS or KEEP THE INITIATIVE (spread,
tenuki, forcing counter-threats); v14's moves CONSOLIDATE THE
CLUSTER or PLAY DIRECTLY (adjacent, fills, direct blocks).

**Engine-swap continuations (the causal test).** At each divergence
ply, v7's move was substituted for the actual move and the game
continued with fresh v14-as-Black vs fresh v0-as-White to the end;
the control (actual move + fresh continuation) ran for every ply
where the substitution flipped the outcome. The walk-back oracle
(v1-vs-v1 playouts) had said these short Black losses have NO losing
move (iter-14 §2.6) — the engine-swap test, using the REAL engines,
says otherwise:

| Ply | Substitute v7's move | Control (actual move) |
|---|---|---|
| 2 | (6,8) → BLACK WINS (+40 plies) | (8,7) → WHITE wins (+25) |
| 8 | (5,8) → BLACK WINS (+128 plies) | (10,7) → WHITE wins (+19) |
| 10 | (12,5) → BLACK WINS (+138 plies) | (9,7) → WHITE wins (+17) |
| 12 | (12,5) → BLACK WINS (+136 plies) | (11,6) → WHITE wins (+15) |
| 14 | (9,6) → BLACK WINS (+88 plies) | (9,4) → WHITE wins (+13) |
| 16 | (9,6) → BLACK WINS (+14 plies) | (4,9) → WHITE wins (+11) |
| 20 | (9,5) → WHITE wins (+5 plies) | — (too late) |

**6 of 6 substitutions flip the loss to a win; 6 of 6 controls
reproduce the loss.** Every divergence ply from 2 to 16 is
individually causal. (Ply 2 is the weakest — the earliest
substitution, longest continuation, single deterministic line; and
(8,7) is v14's most common ply-2 move overall, so it does not lose
in general. The plies 8–16 results are the strong ones: later
substitutions, shorter continuations, controls reproducing the
recorded loss within a few plies.) The pattern in the continuations
is itself informative: substitutions win SLOWLY (+88 to +138 plies
— long positional grinds where MCTS out-plans 1-ply play) while
controls lose FAST (+11 to +25 plies — the tactical race). v14's
moves steer into a fast race v0 wins; v7's moves steer into a slow
game v14 wins.

**The ply-16 case, dissected.** (4,9) is a direct block: White's
attack there is 200006 (White would create an open four on the
anti-diagonal (4,9),(5,8),(6,7),(7,6) — both ends (3,10),(8,5)
empty, two winning squares). (9,6) is a forcing four-maker: after
it, Black has a winning move at (9,5), so White must block (9,5)
instead of creating the open four — the counter-threat defence.
v14's search gives (4,9) 37 visits (q +0.028) vs (9,6) 7 visits
(q -0.009); v7's search gives (9,6) 24 visits (q +0.032) vs (4,9)
23 visits (q -0.012). The v1 leaf rates the forcing counter-threat
lines LOWER than the direct-block lines; the v0 leaf rates them
HIGHER. The substitution test proves the counter-threat wins and
the block loses.

**The ply-10 case, decomposed (the mechanism, precisely).** After
Black's (9,7) [cluster four] vs (12,5) [distant four-maker]:

| After | Black v1: open4/four/three/dev | Black v0: strong/dev | v1 diff | v0 diff |
|---|---|---|---|---|
| (9,7) | 1 / 3 / 6 / 33 | 4 / 7 | +4.50 | +1.25 |
| (12,5) | 1 / 3 / 0 / 46 | 2 / 17 | +0.50 | +1.50 |

The four/open-four counts are IDENTICAL (1 open4, 3 four). The v1
gradient's entire +4.0 preference for the cluster four comes from
the THREE tier: 6 three-makers after (9,7), 0 after (12,5). And of
those 6 three-makers, **4 are GAP-three makers** (v0 attack 40 —
the v0 kernel sees them as ~nothing) and only 2 are true open-three
makers (v0 attack 1111–1120):

| Move | v1 attack | v0 attack | Kind |
|---|---|---|---|
| (9,4) | 4005 | 40 | gap-three |
| (9,5) | 4084 | 40 | gap-three |
| (9,6) | 4092 | 1120 | true open three |
| (9,9) | 4083 | 1111 | true open three |
| (9,10) | 4014 | 40 | gap-three |
| (9,11) | 4006 | 40 | gap-three |

**THE MECHANISM (verified).** The v1 scorer counts a gap three
(XX_X) as an "open three" (4000): `_score_from_counts` awards
OPEN_THREE whenever a window holds 3 friendly stones with both
OUTSIDE cells empty, whether the 3 stones are contiguous or gapped
(`scoring.py`). But tactically a gap three is much weaker than a
true open three: a true open three (_XXX_) has two four-making
squares that cannot both be blocked, so it forces a reply and
survives; a gap three (XX_X) is KILLED outright by a single
opponent move occupying the gap (XXOX is dead — neither end makes
a four). And greedy v0 finds the gap block automatically: its
1-ply defence term scores the gap-fill as blocking a 4-run
(0.9 × 1000–10000). So gap-three breadth is PHANTOM against greedy
play: it evaporates on contact, one block per gap three, while v0
builds its own race. The v1 breadth counts gap-three makers as
strong (weight 1.0, same tier as four-makers and true open-three
makers), so the v1 gradient systematically overvalues cluster
positions rich in gap-three makers (our breadth) and over-penalizes
positions where the opponent holds them (their breadth) — steering
the search into cluster play and away from forcing moves, exactly
the pattern the divergence scan and the swap test show. The v0
breadth does not have this defect: a gap-three maker scores ~130
in the v0 kernel (developing tier), because v0's contiguous-run
scoring cannot see the gap pattern at all.

Two scope notes. (1) Broken-four makers (20,006) stay strong: a
broken four (XX_XX) has exactly one winning square (the gap), the
same as a straight four (XXXX_) — both force a block, identically.
The defect is specific to gap THREE makers (non-forcing, killable),
not to broken fours (forcing). (2) The v1-vs-v0 breadth difference
is EXACTLY the gap-three makers (three tier) plus the broken-four
makers (four tier) — everything else shares a tier. The fix targets
only the gap-three makers: the surgical correction of the one
v1-only tier content that is phantom against greedy play.

### 2.1 Fast-loop iteration 1 — 10 games vs v0

**Batch:** `research/iterations/fastloop1-v14-vs-v0.json` (seeds
v14=1011 / v0=2011 — development seeds, distinct from the bar
battery's 100/200 AND from iter-14's loop seeds 1001/2001, so this
is a fresh 10 games, not a replay of iter-14 §2.1).
**5W-5L-0D, win rate 0.5000, CI [0.2366, 0.7634]**, 236.5 s. Per
colour: as Black 2W-3L, as White 3W-2L. Classifier: 5/5 losses via
open four, 0 missed blocks, 0 clean missed preventions in 30
threatened turns (losses). Defence is still perfect; the losses are
lost races, as in every batch before.

**The losing move in each game** (walk-back, corrected oracle
method, seeds 0,1,2):

| Game | Last preventable turn | v14 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 0 (26 plies, Black) | none | — (no alternative saves; actual draws at earlier turns) | — |
| 2 (94 plies, Black) | none | — (actual draws at all seeds, last 8 turns) | — |
| 3 (31 plies, White) | none | — (loses 3/3 from ply 15, 0 saving alternatives) | — |
| 5 (35 plies, White) | ply 27 | (5,5) | (7,6) [x,y] — wins 3/3; (6,5)/(6,3)/(5,6) — draw 3/3 |
| 6 (42 plies, Black) | ply 34 | (6,11) [x,y] | (4,2) [x,y] — wins 3/3 |
| 6 (42 plies, Black) | ply 28 | (7,5) [x,y] | (4,9) [x,y] — wins 3/3; (0,9) — draws 3/3 |

(Walk-back JSONs report moves as (x, y) = (column, row); the
critical-position analysis below converts to (row, column).)

**Game 6 ply 28 — the clean confirmation.** (Row, column) from here.
Actual (5,7): v1score 20185, grad_v0 +0.170, grad_v1 +0.193;
v14-root 19 visits, q +0.079; v7-root 6 visits, q -0.038. Save
(9,4): v1score 20493, grad_v0 +0.157, grad_v1 +0.171; v14-root 15
visits, q +0.066; v7-root 17 visits, q +0.066. **v7-fresh plays the
save (9,4); v14-fresh plays the loser (5,7).** Breadth after the
actual: ours four=0, true3=3, gap3=8, dev=61. Breadth after the
save: ours four=0, true3=1, gap3=2, dev=67. The v1 gradient's
preference for the loser is driven by the gap-three phantom (8 vs
2 — six extra gap-three makers ≈ +6 breadth points); the v0
gradient prefers the loser by less (+0.170 vs +0.157), and v7's
search rates the loser's lines NEGATIVELY (q -0.038) while v14's
search rates them positively (q +0.079). The gap-three phantom
flips the search's choice from the oracle-verified save to the
losing move. This is the mechanism of §2.0 reproduced at a fresh
seed in a fresh game.

**Game 6 ply 34 — consistent, but both engines miss it.** Actual
(11,6): v1score 200172, grad_v0 +0.183, grad_v1 +0.197; v14-root
39 visits, q +0.169; v7-root 38 visits, q +0.142 — BOTH engines
play it. Save (2,4): v1score 4409, grad_v0 +0.170, grad_v1 +0.174;
both roots 3 visits, q ≈ -0.24. Breadth after the actual: ours
four=0, true3=4, gap3=15, dev=62. Breadth after the save: ours
four=5, true3=1, gap3=2, dev=78. The actual leaves Black with
FIFTEEN gap-three makers and ZERO four-makers — a position that
looks rich in v1 breadth (15 phantom points) but holds no forcing
move; the save leaves FIVE four-makers (forcing) and 2 gap-three
makers. The v1 gradient prefers the actual (+0.197 vs +0.174) on
the phantom; demoting gap-three makers would cut the actual's
breadth by 15 × 0.75 = 11.25 points against the save's 1.5 — the
gradient would flip. But the save is ALSO starved (prior 0.0214 vs
0.1441) and its q is very negative (-0.24, near the must-block
-0.25: the (2,4) lines run through a White four Black must block
before Black's five four-makers win the race) — a beyond-horizon
value error compounded by exploration starvation, a KNOWN class,
not a clean v1-specific patch target. Whether the demotion rescues
it is an empirical question for the v20 check (§1.3), not a claim
made here. Recorded as a regression position either way.

**Game 5 ply 27 — consistent, both engines miss it.** Actual
(5,5): grad_v1 +0.133; v14-root 6 visits, q -0.085; v7-root 6
visits, q -0.094. Save (6,7): grad_v1 -0.013; v14-root 3 visits,
q -0.120. Breadth after the actual: ours four=2, true3=3, gap3=8;
after the save: ours four=4, true3=0, gap3=0. The pattern matches
(gap-heavy loser, forcing-rich save) but BOTH leaves prefer the
loser and both searches miss the save — beyond-horizon value error,
not a v1-specific defect. Recorded as a regression position.

**Games 0, 2, 3 — no losing move (distributed drift).** The oracle
finds no single losing move in three of the five losses — the
position worsens gradually, no turn where a different move saves
the game. This is consistent with the §2.0 mechanism: the
gap-three phantom steers EVERY move slightly toward cluster play,
so the loss accumulates over many moves rather than turning on one
blunder. (The engine-swap test of §2.0 is the stronger instrument
here: it found causal losing moves in a game the oracle called
drift — the v1-vs-v1 oracle is too weak to see MCTS-vs-greedy
race dynamics.)

**No new weakness class beyond §2.0's mechanism.** Iteration 1
confirms the gap-three mechanism at fresh seeds (game 6 ply 28 is
the clean case) and adds no second patchable class: the other two
verified losing moves are known classes (beyond-horizon value
error + starvation), and three losses are drift. The loop continues
to iteration 2 vs v7 (per the batch-7 method: loops run against v0
AND v7) before the stopping rule is applied.

### 2.2 Fast-loop iteration 2 — 10 games vs v7

**Batch:** `research/iterations/fastloop2-v14-vs-v7.json` (seeds
v14=1012 / v7=2012 — development seeds, distinct from the bar
battery's 100/200). **6W-2L-2D, win rate 0.7000, CI [0.3968,
0.8922]**, 828.8 s (two 225-ply draws). Per colour: as Black 3W-1L-1D,
as White 3W-1L-1D. Classifier: 2/2 losses via open four, 0 missed
blocks, 0 clean missed preventions in 7 threatened turns (losses).
Defence is still perfect.

**The losing move in each game** (walk-back, corrected oracle
method, seeds 0,1,2):

| Game | Last preventable turn | v14 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 5 (81 plies, White) | ply 65 | (11,9) [x,y] | (13,6)/(12,6)/(3,6) [x,y] — draw or win |
| 6 (26 plies, Black) | none | — (actual draws at all seeds, last 8 turns) | — |

**Game 5 ply 65 — beyond-horizon value error, NOT the gap-three
mechanism.** (Row, column) from here. Actual (9,11): v1score 4481,
grad_v0 -0.049, grad_v1 +0.039; v14-root 8 visits, q +0.007;
v7-root 2 visits, q -0.088. Save (6,13): v1score 20084 (a
four-maker), grad_v0 -0.152, grad_v1 -0.121; both roots 4 visits,
q ≈ -0.13. The saves are four-makers with HIGHER priors (0.0604 vs
0.0285), yet both engines' searches rate them lower and both
gradients prefer the actual. The actual's advantage is four=4 vs
the save's four=2 — the gap-three counts are small (3 vs 0) and
are not the driver. Both engines miss the save identically
(v7-fresh plays (11,12), neither plays a save). A late-game
(ply 65 of 81) tactical melee whose value materializes beyond the
search horizon — the known beyond-horizon class, not a
v1-specific defect, and not a patch target for this iteration.
Recorded as a regression position.

**Game 6 — no losing move (distributed drift).**

**No patch from iteration 2; loop stops here (stopping rule).**
Iteration 2 found no new patchable weakness class: game 5's losing
move is the known beyond-horizon class (both leaves miss it, the
gap-three counts are not the driver), and game 6 has no losing
move. The fast loop ends after 2 iterations with ONE patchable
class — the gap-three phantom breadth of §2.0, confirmed at fresh
seeds in iteration 1 (game 6 ply 28) — and the fix is designed
around it (§1). The five verified losing positions (fl1-g5-ply27,
fl1-g6-ply28, fl1-g6-ply34, fl2-g5-ply65, plus the battery-line
plies 10 and 16) are recorded as regression tests
(`regression_positions_v20.json`); whether v20 plays the save at
each is measured in §1.3, honestly, as a direction check — not as
validation.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast loop pooled (20 games, 7 v14 losses):

- Losses via open four: **7/7** (5 vs v0 + 2 vs v7)
- Missed blocks: **0** in all games
- Clean missed preventions: **0** in 37 threatened turns (losses)
- Losses with a verified losing move (oracle): 3 of 7 (fl1-g5,
  fl1-g6, fl2-g5); 4 of 7 are distributed drift (fl1-g0, fl1-g2,
  fl1-g3, fl2-g6)
- Losses where the gap-three mechanism is the driver: 1 clean case
  (fl1-g6-ply28) + 2 consistent (fl1-g6-ply34, fl1-g5-ply27) +
  the 6 battery-line swap cases of §2.0

Defence is solved, as in every batch: every loss is an open-four
race lost, never a defensive error.

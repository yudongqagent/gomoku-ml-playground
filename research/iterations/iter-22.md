# Iteration 22 — v22: colour-conditional gap-three demotion (batch 8, H2)

**Status: PRE-REGISTERED — design fixed below (§1) BEFORE any eval of
v22. Step 0 (§1.1), implementation (§1.2), direction checks (§1.3),
fast loop (§2), and battery (§3) follow.**

**METHOD (carried over from batch 3/4/5/6/7).** The fast 10-game
loop stays the finding method: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified, never
guessed), patch the algorithm to reproduce the perfect move in those
positions (regression test), repeat — max 4 iterations, or stop when
an iteration finds no new patchable weakness class. The 4-clause bar
stays the validation gate. Fast-loop games use development seeds
distinct from the bar battery's pre-registered seeds (100/200).
**Fast loops run against v0 as well as v7** — v20's whole mechanism
came from the v0 loop (iter-20 §2.0). The full battery runs on the
FINAL version only. The 100-game battery is the verdict —
small-sample smokes mislead (v8, v11, v15's two n=40 false positives
in one iteration); single-position diagnostics cannot validate a
global search change.

**BASE ENGINE.** v22 is built on v14 (batch 5, H3: v7's PUCT MCTS
with the v1-kernel gradient leaf), ported exactly from
`research/batch-7-h3` (which carries batch 5's v14 implementation
verbatim, including `kernel_v1.py`, 0 mismatches) together with
v20's demoted-breadth function. v17's root rule (tie-prior-forcing)
and threat check are NOT in v22: the root-decision lever is
EXHAUSTED, and the batch-8 plan tasks H2 with fixing the leaf's
condition, not re-combining (v17: the combination does not add).
The fast loop therefore runs on v22 itself, the clean v1-leaf
engine with a plain most-visited root and the conditional
demotion.

## 0. The problem (from batch 7, H3 — v20)

v20 = v14 + GLOBAL gap-three demotion (a three-tier move counts as
STRONG only if it is a true open-three maker, v0 attack >= 1000;
gap-three makers demoted to the developing tier, weight 0.25).
Results (iter-20 §3-§4, REJECTED 2/4):

- vs v7: 51W-37L-12D, 0.5700, CI [0.4722, 0.6627] — clause 1 FAIL.
  Black 0.61 (v14: 0.72), White 0.53 (v14: 0.48).
- vs v0: 53W-36L-11D, 0.5850. Black **0.72** (losses 20→9 vs v14),
  White **0.45** (losses 20→27 vs v14).
- Ladder Elo 88.7 [-63.6, 241.1] — clause 2 FAIL (v14: -34.9,
  v7: 190.8). Tactical 20/20 + 29/29 PASS. Swap vs v7 0.49, no
  edge (v14: 0.64, v17: 0.74).

The colour split is the result: the fix worked exactly as designed
for Black — Black's losses to v0 fall 20→9, nearly back to v7's
level (0.74, 13 losses) — but broke White (losses 20→27). v14 was
symmetric (0.60/0.60); net vs v0 is a wash vs v14's 0.60. The
lesson (iter-20 §5): **gap-three breadth is colour- and
opponent-dependent, not purely phantom.** It is phantom against
greedy play as the attacker (the opponent kills gap threes for
free), but signal for White's counterplay and against a searching
opponent. A global demotion helps the attacker and blinds the
defender — right at three verified positions, wrong in aggregate.
GLOBAL GAP-THREE DEMOTION IS CLOSED. The open follow-up (iter-20
§5, batch-8 plan H2): a COLOUR- OR INITIATIVE-CONDITIONAL
demotion — demote only for the player holding the initiative, or
only as Black.

Note the second-order split, recorded here because it constrains
the design: against v7 (searching), the demotion's effect by
colour REVERSES vs the v0 split — Black 0.72→0.61 (cost), White
0.48→0.53 (gain). Against a searching opponent the demotion helps
White (the defender does not over-fear the attacker's phantom
breadth — the "theirs" term) and costs Black (the attacker's own
breadth is part of its edge). Any conditional rule must be judged
against both opponents, not just v0.

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-8 plan: ladder Elo
190.8, on master at `bf866f7`). Clause 1 is head-to-head vs v7
(normal rules, >= 100 games, alternating colours, win rate > 0.50
with Wilson 95% CI lower bound > 0.50); clause 2 is ladder Elo
strictly higher than v7's 190.8 (identical 20 games/opponent:
random, heuristic, mcts, mcts_sims=100); clause 3 is tactical pass
rate >= v7's (20/20 old suite + 29/29 hard suite); clause 4 is all
test suites green (ml pytest, npm test, tsc). **Secondary
head-to-heads, also pre-registered:** v22 vs v0 (100 games) and a
100-game swap match vs v7 under the tempo judge (the harness
default). Non-transitivity is recorded prominently.

**The improvement: v22 = v14 with gap-three makers demoted ONLY
in BLACK's breadth.** Everything else is v14 unchanged: 100 sims,
c_puct 1.5, v1 priors over all candidates (temperature 2.0 — the
sharp prior is load-bearing at 100 sims; flattening is
catastrophic, iter-13 §2.4), win/block scans before search,
most-visited root child with RNG tie-breaks, terminal leaf values
identical. The ONLY change is the quiet-leaf gradient's breadth:
Black's breadth is computed with v20's demoted breadth
(`_breadth_v1_gap`: four-makers + true open-three makers strong,
gap-three makers at 0.25); White's breadth is computed with v14's
undemoted breadth (`_breadth_v1`: all three-tier moves strong).
The diff is `breadth(ours) - breadth(theirs)` as usual, with each
side's breadth computed by ITS OWN colour's rule:

- Leaf with Black to move: diff = breadth_gap(black) -
  breadth_v1(white).
- Leaf with White to move: diff = breadth_v1(white) -
  breadth_gap(black).

**Why this condition (the mechanism, from v20's battery).** Black
moves first and holds the tempo — Black IS the player with the
initiative in Gomoku (established by the tempo-judge work, batch
2/5). The phantom is the initiative-holder's gap-three breadth:
Black builds clusters with a free hand, and the defender kills
the gap threes for free (a defensive block often occupies a gap
square incidentally). White's gap-three breadth is counterplay
signal: White creates threats in reply, and Black — who must
answer them — cannot spend the move killing the gap. Demoting
only Black's breadth keeps the Black recovery in BOTH leaf
parities: at Black-to-move leaves, Black's OWN breadth is demoted
(the ply-10 fix — the v1 gradient's +4.0 preference for the
losing cluster move came entirely from Black's gap-three makers);
at White-to-move leaves, Black's THEIRS breadth is demoted (White
does not over-fear Black's phantom threats — the term that made
v20's White BETTER against v7, 0.48→0.53). White's breadth is
never demoted, so White's counterplay vocabulary is preserved at
every leaf. The leaf remains antisymmetric under PLAYER swap
(value(board, WHITE) = -value(board, BLACK)) — the property MCTS
needs — but is deliberately NOT antisymmetric under colour swap:
Black and White are treated differently, which is the whole
point.

**Conditions considered and REJECTED (pre-registered, so the
choice is auditable):**

1. Position-based initiative (demote the gap-three makers of
   whichever side holds more strong moves / the higher best
   attack): no clean board-only definition of initiative exists;
   the v20 evidence splits by COLOUR, not by position type; a
   data-dependent switch adds a discontinuity the battery cannot
   attribute. Colour (Black moves first, holds the tempo) IS the
   initiative proxy in Gomoku.
2. Per-game switch (v20's leaf when the engine plays Black,
   v14's leaf when it plays White): cruder — when the engine
   plays White, Black's phantom breadth would count fully at
   every leaf, so White would over-fear Black's clusters. The
   per-colour-breadth rule demotes exactly the phantom component
   everywhere and preserves exactly the signal component
   everywhere. It also needs no plumbing: the leaf already
   computes both colours' breadth arrays.
3. Demote only in the OURS term (the side to move's own breadth):
   at White-to-move leaves this demotes White's own breadth —
   the v20 White problem, unchanged for the engine's own colour.
4. Demote only in the THEIRS term (the opponent's breadth):
   does not fix the diagnosed case — the ply-10 decomposition
   shows the bad choice was driven by OUR breadth (+4.0 from
   our gap-three makers), not by fear of the opponent's.

**What does NOT change (pre-registered exclusions).** Broken-four
makers (20,006) stay STRONG for both colours: a broken four has
exactly one winning square, the same as a straight four — both
force a block identically; the defect is specific to gap THREE
makers (non-forcing, killable), not broken fours (forcing). The
developing tier [80, 4000) is unchanged. The cap stays ±0.2 — a
static heuristic must never outweigh an exact terminal signal. No
root-rule change (EXHAUSTED — v22 keeps v14's plain most-visited
root). No exploration change (CLOSED). No sims change (100 both
colours — more search is NOT monotonically better). No prior
change (sharp prior load-bearing). No threat check, no opening
book, no q-based root decision (CLOSED). No change to
`baseline.py` (FROZEN). No kernel re-sync (NumPy/WebGPU/Metal)
unless v22 passes the bar. The v1 kernel is used directly, not
re-ported; the v0 kernel is used as the filter, unchanged.

**The scale question (pre-registered rule, Step-0 value).** The
conditional demotion narrows the breadth-difference distribution,
so the tanh scale is re-derived from the CONDITIONAL distribution
by v7's rule (scale ~= 1.8 x std) — the same rule v7 (4.0 from
std 2.26), v14 (7.5 from std 4.09), and v20 (5.5 from std 3.02)
used — measured in Step 0 (§1.1) on the same 304-position sample
as iter-14/iter-20, BEFORE any v22 eval. The scale is fixed here
from the measured distribution, NOT tuned on any v22 eval result.
Value: **TBD in §1.1** (filled from the Step-0 measurement; the
rule above is the pre-registration).

**What this change is meant to fix, and the expected effect
(stated in advance).** v20's global demotion traded Black for
White and v7-margin for v0-margin. v22 should keep Black's
recovery against greedy play (v20-as-Black vs v0: 0.72, 9
losses) while restoring White's counterplay (v14-as-White vs v0:
0.60, 20 losses) — expected vs v0 ≈ 0.66 if the two effects add.
Against v7 the expectation is genuinely unclear, and is stated
as such: the demotion's vs-v7 cost was concentrated in Black
(0.72→0.61), which v22 keeps for Black's breadth, so clause 1 is
the risk; but v22's White should be ≥ v20's 0.53 (White's breadth
full AND Black's theirs-breadth demoted). The ladder should
improve over v20's 88.7 if the vs-v0 (heuristic anchor) colour
split becomes ≈0.72/0.60. If v22 still loses the race to v0 as
White, the White weakness is not a gap-three-vocabulary problem
— and the battery decides.

**Eval seeds (pre-registered):** h2h vs v7: v22=100 / v7=200; h2h
vs v0: v22=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v7, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON). Harness commands follow iter-7 §3 exactly
(`selfplay_batch.py`, `eval_version.py` for the ladder,
`eval_tactical_full.py` for tactical). Fast-loop dev seeds
1013/2013 (iter 1 vs v0) and 1014/2014 (iter 2 vs v7) are distinct
from the battery's 100/200 AND from iter-20's loop seeds
1011/2011 and 1012/2012, so these are fresh games.

### 1.1 Step 0 — cost and distribution measurement

(TBD — run `ml/scripts/bench_v22_leaf.py` on the iter-20 §1.1
sample BEFORE any v22 eval; fill in cost table, conditional
breadth-difference distribution, and the scale value.)

### 1.2 Implementation (pre-registered)

- `ml/gomoku_ml/mcts_heuristic.py`: `_attack_gradient_v1_gap_black`
  (one `attack_v1_both` + one `attack_both` call; Black's breadth
  via `_breadth_v1_gap`, White's via `_breadth_v1`; tanh scale
  from §1.1, cap ±0.2), `_gradient_v1_gap_black_leaf_value`
  (terminal values identical by construction), registered as
  `leaf_mode="gradient_v1_gap_black"` in `_LEAF_FUNCTIONS`. Both
  breadth functions are reused unchanged; the v1 kernel
  (`kernel_v1.py`) and the v0 kernel (`kernel_numpy.py`) are used
  directly, unchanged.
- `ml/gomoku_ml/scoring.py`: version 22 in `ENGINE_VERSIONS`
  (`{"name": "scoring-v22", "defence_weight": 1.0, "threat_check":
  False, "mcts": True, "mcts_sims": 100, "leaf_mode":
  "gradient_v1_gap_black"}`), dispatching through the existing
  generic `HeuristicMCTS` path (no new dispatch code).
- Tests: `ml/tests/test_mcts_heuristic.py` gains v22 leaf tests
  (Black-only demotion unit cases on the ply-10 board, terminal
  identity with v14/v20, player-swap antisymmetry + boundedness,
  scale test); `ml/tests/test_regression_v22.py` +
  `ml/gomoku_ml/regression_positions_v22.json` record the §2
  critical positions (plus the v20 positions re-checked in §1.3).

### 1.3 Direction checks (NOT evals — identity/direction only)

(TBD — after implementation and before the battery, v22's search
(100 sims, seed 0) is run on the §2 critical positions and the 9
v20 regression positions, to check the patch has the intended
DIRECTIONAL effect. Single-position diagnostics: they CANNOT
validate the version, they only check the patch does what it was
designed to do at the positions that motivated it. The battery
(§3) is the verdict.)

## 2. Loss analysis for iteration 22

(TBD — fast-loop iterations vs v0 and v7 on v22.)

### 2.1 Fast-loop iteration 1 — 10 games vs v0

(TBD.)

### 2.2 Fast-loop iteration 2 — 10 games vs v7

(TBD.)

### 2.3 Fast-loop iteration 3 — NOT RUN unless a new patchable class appears

### 2.4 Fast-loop iteration 4 — NOT RUN unless a new patchable class appears

### 2.5 Aggregate counts (secondary)

(TBD.)

## 3. v22 evaluation

(TBD.)

### 3.1 Smoke — NOT RUN (no smoke; the fast loop is the smoke)

Per the batch-5 method the 10-game fast loop (§2.1-2.2) is the
only small-sample evidence, and small-sample smokes mislead (v8,
v11, v15's two n=40 false positives) — the 100-game battery is
the verdict.

### 3.2 Tactical suites — pre-battery (re-run on the final engine in §3.5)

(TBD.)

### 3.3 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

(TBD.)

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

(TBD.)

### 3.5 Tactical suites — FINAL BATTERY (re-run on the final engine)

(TBD.)

### 3.6 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

(TBD.)

### 3.7 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

(TBD.)

## 4. Bar check (v22 vs v7)

(TBD.)

## 5. What this means

(TBD.)

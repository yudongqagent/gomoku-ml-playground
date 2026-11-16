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
Value: **7.0** (1.8 x std 3.75 = 6.76, rounded to the nearest 0.5).

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

`ml/scripts/bench_v22_leaf.py` on 346 boards replayed from
batch-v7-vs-v1.json (first 20 games, every 4th ply from ply 8;
290 quiet) — the SAME sample as iter-20 §1.1, and the v14/v20
distributions reproduce iter-20's numbers exactly (v14 mean
-2.98, std 4.91; v20 mean -2.00, std 3.02), so the comparison is
apples-to-apples. Median of 3 rounds x 50 calls.

| Call | Median ms/call |
|---|---:|
| v22 `_attack_gradient_v1_gap_black` (full) | 8.15 (min 7.52, max 8.26) |
| v22 `_gradient_v1_gap_black_leaf_value` (full) | 9.70 (min 8.46, max 9.88) |
| v20 `_attack_gradient_v1_gap` (full) | 7.50 (min 6.27, max 7.85) |
| v14 `_attack_gradient_v1` (full) | 3.85 (min 2.66, max 4.05) |
| v7 `_attack_gradient` (full) | 2.19 (min 2.17, max 3.39) |
| v6 `_terminal_leaf_value` | 1.92 (min 1.82, max 2.38) |

The v22 gradient costs ~8.2 ms per quiet leaf vs ~7.5 ms for
v20's on this run (absolute numbers are contention-noisy; the
ordering is what matters: v22 ≈ v20, one extra kernel call over
v14). At up to ~100 leaf calls per move that is at most ~+0.4
s/move over v14, ~+15 min per 100-game match. Affordable — the
same cost class as v20.

Conditional breadth-difference distribution on the 290 quiet
positions: mean -0.88, std 3.75, p10 -5.50, p50 -0.50, p90 +3.75;
|diff| >= 2 in 58.6%, >= 4 in 29.7%, >= 6 in 12.1%. (v14 on the
same sample: mean -2.98, std 4.91; v20: mean -2.00, std 3.02.)
Scale: 1.8 x 3.75 = 6.76 -> **7.0** (pre-registered above). Mean
gap-three makers per position: Black 2.80, White 1.50 — the
sample's boards all have White to move (every 4th ply from ply
8), so this reproduces iter-20's ours 1.50 / theirs 2.80 exactly:
the side to move holds fewer gap-three makers, the opponent just
moved and just created them. The conditional demotion changes the
breadth difference in 55.5% of quiet positions vs v14 (mean
|change| 2.10) and in 37.2% vs v20 (mean |change| 1.12) — a
substantial revision of the signal in both directions, not a
tweak at the margin. The mean shift (-2.98 → -0.88) is the
intended direction: part of the side-to-move's deficit was
Black's phantom breadth, now removed from the theirs term.

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

After implementation and before the battery, v22's search (100
sims, seed 0) is run on the 9 v20 regression positions
(`regression_positions_v20.json`) to check the patch has the
intended DIRECTIONAL effect — whether it plays the oracle-verified
save where v14 played the loser, and whether it keeps v20's saves.
These are single-position diagnostics: they CANNOT validate the
version (standing lesson), they only check the patch does what it
was designed to do at the positions that motivated it. The battery
(§3) is the verdict.

**Results (measured, seed 0).** v22 plays the save at **2 of 9**
positions; v20 at **3 of 9**; v14 at **0 of 9**:

| Position | v22 plays | v20 plays | v14 plays | Expected (save) | Verdict |
|---|---|---|---|---|---|
| bl-ply8 | (7,10) | (7,10) | (7,10) | (8,5) | miss (all) |
| bl-ply10 | (8,5) | (8,5) | (7,9) | (5,12) | miss (v22 avoids the loser, not the save) |
| bl-ply12 | (6,11) | (5,12) | (6,11) | (5,12) | **miss (v22 REGRESSES vs v20)** |
| bl-ply14 | (6,9) | (6,9) | (4,9) | (6,9) | **SAVE** |
| bl-ply16 | (9,4) | (9,4) | (9,4) | (6,9) | miss (all — prior gap) |
| fl1-g6-ply28 | (4,9) | (0,9) | (7,5) | (4,9)/(0,9) | **SAVE (winning save, not v20's drawing save)** |
| fl1-g6-ply34 | (6,11) | (6,11) | (6,11) | (4,2) | miss (all — starvation) |
| fl1-g5-ply27 | (11,9) | (6,10) | (8,9) | (7,6)/... | miss (all — beyond-horizon; different misses) |
| fl2-g5-ply65 | (12,11) | (11,9) | (11,9) | (13,6)/... | miss (all — beyond-horizon; different misses) |

(Coordinates are (x, y) throughout, as in the walk-back JSONs
and the iter-20 §1.3 table — the v20/v14 columns reproduce
iter-20 §1.3 exactly, which cross-checks the driver. (5,12) is
the distant four-maker.)

Two of v20's three saves survive: bl-ply14 (the forcing open
three over the flashy double open three) and fl1-g6-ply28 — where
v22 plays (4,9), the WINNING save, while v20 played (0,9), the
drawing save. The third, bl-ply12, REGRESSES: v22 plays (6,11)
like v14 instead of the distant four-maker (5,12). The regression
is NOT a 1-ply gradient effect: the gradients after the save and
after (6,11) are IDENTICAL for all three leaves (v14 -0.039, v20
-0.053, v22 +0.042, White to move — v22 rates both positions
better for White than v20 does, because White's 4 gap-three
makers count fully: White breadth 17.5 vs demoted 14.5). The
difference is a deeper-tree effect: below the save, White's
gap-three breadth counts fully at every White-to-move leaf, so
White's replies look better for White and the save's lines look
worse for Black on backup. That is the theirs-term working as
designed — and costing Black at this position, where White's
gap-three breadth after the save turns out to be phantom in the
engine-swap sense (the save wins). The condition is a trade, not
a pure win: the same term that restores White's counterplay
vocabulary can steer Black away from a forcing move when White's
replies are gap-three-rich. Whether that cost matters in
aggregate is what the battery measures; no single-position
diagnostic can say. The two White positions (fl1-g5-ply27,
fl2-g5-ply65) are missed by all three engines, but v22 plays
different moves than v20/v14 at both — whether they are better is
unknown (they are not the oracle-verified saves) and is not
claimed. The 2 saves are passing tests in
`test_regression_v22.py`; the 7 misses are xfails.

## 2. Loss analysis for iteration 22

The fast loop runs on v22 itself (the pre-registered engine),
against v0 AND v7. Its purpose is finding patchable weakness
classes the conditional demotion introduces or leaves open — not
validating v22 (the battery does that).

### 2.1 Fast-loop iteration 1 — 10 games vs v0

**Batch:** `research/iterations/fastloop1-v22-vs-v0.json` (seeds
v22=1013 / v0=2013 — development seeds, distinct from the bar
battery's 100/200 AND from iter-20's loop seeds 1011/2011 and
1012/2012, so these are fresh games). **10W-0L-0D, win rate
1.0000, CI [0.7225, 1.0000]**, 433.3 s. Per colour: as Black 5W-0L
(51, 63, 79, 69, 69 plies), as White 5W-0L (68, 66, 138, 162, 84
plies). Mean game length 84.9 plies — much longer than v14's
loop-1 games: v22 wins slow grinds in both colours, including
two very long White wins (138 and 162 plies), the games v20-as-
White lost outright. No losses, so there is no losing move to
find in this iteration — the loop's finding method has nothing to
work on against v0 at these seeds. That is itself the directional
signal the design predicted (v20-as-Black's recovery kept, White
restored), but 10 games is a smoke, not a verdict: v14's loop 1
was 5W-5L at its seeds, and the 100-game battery (§3.4) decides
whether the sweep holds. No patch from iteration 1; the loop
continues to iteration 2 vs v7 (per the method: loops run
against v0 AND v7) before the stopping rule is applied.

### 2.2 Fast-loop iteration 2 — 10 games vs v7

**Batch:** `research/iterations/fastloop2-v22-vs-v7.json` (seeds
v22=1014 / v7=2014 — development seeds, distinct from the bar
battery's 100/200). **4W-6L-0D, win rate 0.4000, CI [0.1682,
0.6873]**, 465.9 s. Per colour: as Black 2W-3L (games 0, 2, 6
lost; 4, 8 won), as White 2W-3L (games 1, 7, 9 lost; 3, 5 won).
Classifier: 6/6 losses via open four, 0 missed blocks, 0 clean
missed preventions in 25 threatened turns (losses). Defence is
still perfect; the losses are lost races, as in every batch.

**The losing move in each game** (walk-back, corrected oracle
method, seeds 0,1,2 — then ENGINE-SWAP VERIFIED with fresh v22
vs fresh v7, the iter-20 §2.0 instrument; the oracle alone is
not trusted for MCTS-vs-MCTS games, and this iteration shows
why):

| Game | Last preventable turn | v22 played (loses 3/3) | Perfect move (oracle) | Engine-swap verdict |
|---|---|---|---|---|
| 0 (56 plies, Black) | none | — (actual draws at plies 40–48) | — | — (drift) |
| 1 (39 plies, White) | ply 29 | (6,6) | (4,8) wins 3/3; (9,3) draws 3/3 | **WITHDRAWN** — save loses +19, control WINS +46 |
| 2 (78 plies, Black) | ply 70 | (3,2) | (3,5)/(8,11) draw 3/3 | **WITHDRAWN** — save loses +7, control loses +7 |
| 6 (24 plies, Black) | ply 14 | (11,8) | (7,8) wins 1 / draws 2 | **WITHDRAWN** — save loses +7, control loses +9 |
| 7 (31 plies, White) | ply 17 | (6,10) | (5,9)/(3,11)/(4,5) win 3/3 | **CONFIRMED** — save wins +18, control loses +29 |
| 9 (29 plies, White) | ply 19 | (8,5) | (8,9) wins 3/3 | **CONFIRMED** — save wins +64, control loses +11 |

**Three of the five oracle verdicts are WITHDRAWN.** Game 1 is
the sharpest case: the oracle's save (4,8) LOSES in +19 plies
with the real engines, while the control (the actual move (6,6))
WINS in +46 — the oracle's verdict is exactly reversed, and the
position was not lost at ply 29 at all under v22-vs-v7 dynamics.
Games 2 and 6: the oracle's saves lose as fast as the controls
(+7 plies both) — the positions were already lost by the
walk-back window, and the v1-vs-v1 oracle's "draw" reflects
1-ply dynamics, not MCTS-vs-MCTS play. This is the iter-18
caveat in both directions: the v1 walk-back oracle diverges from
real-engine dynamics, and for MCTS-vs-MCTS games every nominated
save needs engine-swap confirmation. (iter-20 §2.0 found the
oracle missing real losing moves; this iteration finds it
inventing false ones.) Only games 7 and 9 survive as verified
losing moves.

**Game 7 ply 17 — the clean v22-specific case.** (x, y) from
here. Actual (6,10): v22-fresh plays it; v20-fresh and v14-fresh
play the save (5,9); v7-fresh plays (4,5), also a save. v22 is
the ONLY engine that misses it. Breadth after the actual: White
(ours) four=0, true3=5, gap3=**18**, dev=47; after the save:
four=0, true3=6, gap3=14, dev=52. The actual leaves White with
EIGHTEEN gap-three makers — the maximum of the Step-0 sample
distribution (mean 2.64, p90 7, p99 14, max 18; P(>=18) = 0.003).
v22 counts White's gap-three breadth fully, so its gradient
prefers the actual (-0.198 vs -0.195 for the next player);
v20's gradient (White's gap threes demoted) prefers the save
(-0.135 vs -0.144). v22-root: actual 24 visits, our-q +0.151;
save 4 visits, our-q -0.027 — priors nearly equal (0.0559 vs
0.0558), so this is a VALUE error, not starvation: the tree
strongly prefers the gap-saturated actual. The mechanism is the
gap-three phantom, now for White: at a saturated position,
White's gap-three breadth does not convert (Black wins the race),
but v22's leaf counts all 18 makers at full weight at every
White-to-move leaf below the actual.

**Game 9 ply 19 — the same mechanism, closer.** Actual (8,5):
v22-fresh and v14-fresh play it; v20-fresh and v7-fresh play the
save (8,9). Breadth after the actual: White (ours) four=1,
true3=5, gap3=**13**, dev=51; after the save: four=0, true3=4,
gap3=10, dev=64. Thirteen gap-three makers is p98.5 of the
distribution (P(>=13) = 0.015). v22's gradient prefers the
actual (-0.193 vs -0.189); v20's prefers the save (-0.130 vs
-0.144). v22-root: actual 35 visits, our-q +0.089; save 32
visits, our-q +0.073 — nearly tied (priors 0.1496 vs 0.1495);
the search just barely prefers the gap-rich loser. The save's
advantage is developing breadth (64 vs 51), which the gap-three
count outweighs in v22's leaf.

**Games 1, 2, 6 — known classes, all engines miss.** Game 1 ply
29 (oracle verdict withdrawn, but the position is still
instructive): all four engines play the actual; the save (4,8)
leaves SIX four-makers yet has prior 0.049 vs the actual's 0.152
and equal our-q (+0.025 vs +0.024) — prior-dominated. Game 2 ply
70: all four play the actual; the save's our-q is HIGHER (-0.161
vs -0.338) but its prior is 0.046 vs 0.145 — prior-dominated +
beyond-horizon (ply 70 of 78, both q's very negative). Game 6
ply 14: a FLAT root (top-3 all 4 visits; actual 4 visits, save 3
visits, priors 0.0376 vs 0.0383) — the choice among ~equal early
moves is a coin flip, and v7 playing the save is luck as much as
skill. None is a v22-specific patch target.

**No patch from iteration 2; loop stops here (stopping rule).**
Games 7 and 9 document a REAL cost of the pre-registered design:
White's gap-three breadth is NOT purely signal — at saturated
positions (13–18 makers, the top 1.5% of the distribution) it is
phantom, and v22 falls for it where v20 does not. But the class
is the KNOWN gap-three phantom, now documented for White; the
design's premise (White's gap-three breadth is signal ON
AVERAGE) comes from v20's 100-game battery, and 2 positions
cannot overturn an aggregate claim. Any patch distinguishing
"phantom White gap threes" from "signal White gap threes" needs
a position-level feature — a saturation threshold (a free
parameter fitted to 2 positions) or kill-square dedup (count
distinct gap squares instead of makers — principled IF the
makers share kill-squares, but unverified here, and a substantial
new leaf feature requiring its own Step-0, tests, and
re-validation). Patching now would decide the aggregate question
on 2 positions — exactly the overfitting the method forbids
(v15's two n=40 false positives; v8's first-20 fluke). The
battery (§3) measures the aggregate cost directly: if v22's
White holds up vs v0 (iteration 1: 10-0) while the vs-v7 cost
stays small, the condition is right; if not, the saturation
hypothesis (with kill-square dedup as the candidate mechanism)
is the batch-9 follow-up. The two confirmed positions are
recorded as regression tests (`regression_positions_v22.json`,
both xfails — v22 misses both saves).

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

Fast loop pooled (20 games, 6 v22 losses):

- Losses via open four: **6/6** (0 vs v0 + 6 vs v7)
- Missed blocks: **0** in all games
- Clean missed preventions: **0** in 25 threatened turns (losses)
- Losses with a verified losing move (oracle + engine-swap):
  2 of 6 (fl2-g7, fl2-g9); 3 of 6 oracle verdicts WITHDRAWN by
  engine-swap (fl2-g1, fl2-g2, fl2-g6); 1 of 6 is distributed
  drift (fl2-g0)
- Losses where White gap-three saturation is the driver: 2
  (fl2-g7, fl2-g9) — the conditional demotion's cost side,
  documented, not patched (see §2.2)

Defence is solved, as in every batch: every loss is an open-four
race lost, never a defensive error.

## 3. v22 evaluation

**Engine note:** the fast loop produced NO patch (§2.2), so the
battery runs on the pre-registered v22 exactly as implemented in
§1.2 (v14's MCTS + the Black-only gap-demoted v1 gradient leaf,
scale 7.0, cap ±0.2). Tactical was measured on this engine before
the battery (§3.2) and is re-run in §3.5 for the record.

### 3.1 Smoke — NOT RUN (no smoke; the fast loop is the smoke)

Per the batch-5 method the 10-game fast loop (§2.1-2.2) is the
only small-sample evidence, and small-sample smokes mislead (v8,
v11, v15's two n=40 false positives) — the 100-game battery is
the verdict.

### 3.2 Tactical suites — pre-battery (re-run on the final engine in §3.5)

Old 20-position suite, full-dispatch driver
(`ml/scripts/eval_tactical_full.py --version 22 --suite both
--seed 100`): **v22 20/20**. Hard suite (29 positions): **v22
29/29** — a perfect score, matching v7's 20/20 + 29/29. The
conditional demotion costs nothing tactically.

### 3.3 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

[NOTE: the first run was killed by a runtime restart at 64/100
games (36W-23L-5D, partial JSON backed up to
`~/workspace/b8h2-logs/batch-v22-vs-v7-partial.json`, not
committed). The re-run started IN FULL from game 0 with the same
seeds — matches are deterministic, so the re-run reproduces the
64-game prefix identically (verified: 0 mismatches in moves and
winners across all 64 games, including identical W/L/D at the 30-
and 44-game marks) and completes 64-99 as pre-registered.]

**v22 beats v7: 54W-38L-8D, win rate 0.5800, 95% CI [0.4821,
0.6720]**, 5270.4 s. File:
`research/iterations/batch-v22-vs-v7.json` (seeds 100/200). Per
colour: as Black 25W-17L-8D (0.5800), as White 29W-21L-0D
(0.5800) — a PERFECTLY SYMMETRIC split, the first of any v1-leaf
version (v14: 0.72/0.48; v20: 0.61/0.53). The conditional
demotion balanced the colours exactly as designed: Black gave
back part of v14's opening edge (0.72 → 0.58) while White gained
(0.48 → 0.58), and White's gain exceeds v20's (0.53) — the
theirs-term demotion (Black's phantom breadth not over-feared)
plus White's full counterplay vocabulary. Mean game length 60.42
plies (min 21, max 225); wins mean 44.6 plies, losses mean 48.3
— v22 wins slightly FASTER than it loses, reversing the familiar
pattern (v20: wins 50.0, losses 36.6). 96 distinct lines in 100
games (no transposition collapse). Loss classifier: 38/38 losses
via open four, **0 missed blocks**, 177 threatened turns in
losses, **0 clean missed preventions** — defence is solved.
**Clause 1 FAILS** (0.5800 > 0.50, but the CI lower bound 0.4821
< 0.50 — the edge over v7 is not established). v22 scores above
v20 against v7 (0.5800 vs 0.5700) but below v14 (0.6000, CI lower
0.5020): the conditional demotion recovers 1 game per 100 over
the global demotion, not the ~3 the design hoped for — the
bl-ply12 regression (§1.3) and the White gap-saturation misses
(§2.2) are the visible costs, and Black's vs-v7 edge was partly
in the gap-three breadth v22 still demotes for Black.

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

[NOTE: the first run was killed by a runtime restart at 44/100
games (27W-13L-4D, partial JSON backed up to
`~/workspace/b8h2-logs/batch-v22-vs-v0-partial.json`, not
committed). The re-run started IN FULL from game 0 with the same
seeds and reproduces the 44-game prefix identically (verified: 0
mismatches in moves and winners across all 44 games, including
identical W/L/D at the 6- and 34-game marks).]

**v22 beats v0: 61W-30L-9D, win rate 0.6550, 95% CI [0.5577,
0.7409]**, 4235.8 s. File:
`research/iterations/batch-v22-vs-v0.json` (seeds 100/200). Per
colour: as Black **43W-0L-7D (0.9300)** — ZERO losses as Black;
as White 18W-30L-2D (0.3800). The aggregate (0.6550) matches the
pre-registered prediction (≈0.66) but through the OPPOSITE colour
mechanism: Black far exceeded the prediction (0.93 vs the
predicted ≈0.72 — v20's level) while White fell below it (0.38
vs the predicted ≈0.60 — v14's level). The conditional demotion
did NOT restore White against greedy play — White is WORSE than
v20's (0.45, 27 losses) — and it made Black nearly unbeatable.
Mean game length 74.42 plies (min 21, max 225); wins mean 75.9
plies, losses mean 26.3 — v22 wins very slow grinds and loses
fast races, the familiar pattern in its extreme form. 90 distinct
lines. Loss classifier: 30/30 losses via open four (all as
White, since Black lost 0), **0 missed blocks**, 166 threatened
turns in losses, **0 clean missed preventions** — defence is
still solved.

**Decomposition (clean: each comparison changes exactly one
thing).** v14 (no demotion): Black 0.60, White 0.60. v20 (demote
both): Black 0.72, White 0.45. v22 (demote Black only): Black
0.93, White 0.38. So demoting BLACK's breadth (v14 → v22) is
worth **+0.33 for Black and −0.22 for White**; demoting WHITE's
breadth on top (v22 → v20) is worth **−0.21 for Black and +0.07
for White**. v22's condition is the best of the three tested
combinations vs v0 (0.655 > v14's 0.60 > v20's 0.585) — but the
colour asymmetry is now EXTREME (0.93/0.38, a 0.55 gap, wider
than v20's 0.27). The mechanism for White's −0.22 is the
theirs-term demotion: at White-to-move leaves, Black's gap-three
breadth is demoted, so White UNDER-FEARS Black's threats — the
same mechanism as fast-loop games 7/9 (§2.2), now measured at
battery scale against greedy play, where Black's gap threes
convert most directly. And White's own full breadth (the
"restored counterplay vocabulary") does not save it: White's
gap-three breadth is phantom at the positions that matter (games
7/9), so counting it fully makes White overvalue its own
gap-rich positions while under-fearing Black's. The design's
premise — "White's gap-three breadth is signal" — is WRONG in
aggregate against greedy play: it is signal only as the
OPPONENT's breadth (Black fearing White's counterplay is worth
+0.21 for Black), not as White's own. The open follow-up is
demoting Black's gap threes ONLY in the ours term (when Black is
to move), never in the theirs term — White would fear Black's
gap threes fully. Untested; batch 9.

The non-transitivity sharpens by colour: v22 beats v0 as Black
(0.93) and loses to v0 as White (0.38), while v7 beats v0 in both
colours (0.74/0.59) and v22 beats v7 in both colours (0.58/0.58).
The ladder (§3.6) measures the consequence directly, since its
heuristic anchor IS v0.

### 3.5 Tactical suites — FINAL BATTERY (re-run on the final engine)

**v22: 20/20 old + 29/29 hard** (full-dispatch driver,
`ml/scripts/eval_tactical_full.py`, `--seed 100`, run on the
final engine after the battery). Identical to the pre-battery run
(§3.2) and to v7's scores. The conditional demotion costs no
tactical accuracy: every forced win, block, prevention, setup,
and VCF sequence is still found. **Clause 3 PASSES.**

### 3.6 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

**Ladder Elo: 147.2, interval [-13.2, 307.6]** (anchors:
random=-800, heuristic=0; relative to this ladder only).
Per-opponent (20 games each, seed 100):
- vs random: 20W-0L-0D (1.0000; as Black 10W-0L-0D, as White
  10W-0L-0D).
- vs heuristic (= v0): 14W-6L-0D (0.7000, CI [0.4810, 0.8545];
  as Black 10W-0L-0D, as White 4W-6L-0D).
- vs mcts: 5W-15L-0D (0.2500, CI [0.1119, 0.4687]; as Black
  3W-7L-0D, as White 2W-8L-0D).
Log: `~/workspace/b8h2-logs/ladder-v22.log` (not committed).
**Clause 2 FAILS** (147.2 < v7's 190.8; the interval includes
190.8, so the shortfall is not statistically established, but the
clause requires the estimate to be strictly higher and it is
not). v22's Elo is above v20's (88.7) and v14's (-34.9) — the
vs-heuristic result (0.70, Black 10-0) reflects §3.4's Black
strength — but the vs-mcts result (0.25) is the binding
constraint, and it is the WORST vs-mcts of any v1-leaf version
(v14: 0.55, v20: 0.65, v17: ~0.50; even v7 itself scores 0.40
against the ladder MCTS). v22 loses to the terminal-leaf MCTS
in BOTH colours (Black 0.30, White 0.20), while v20-as-Black
beat it 9-1. The mechanism is NOT diagnosed: the leading
hypothesis is the full theirs-term — v22 counts the opponent's
gap-three breadth at full weight, so against the terminal-leaf
MCTS's forcing play (which creates gap-three-rich positions
without respecting them, since its leaf has no gradient) v22
over-fears and plays passively — but this is a hypothesis,
stated as such, not a verified mechanism. It needs its own
forensics (a batch-9 question): the ladder MCTS is the one
opponent v22 is clearly worse than v20 against, and the ladder
Elo is where that cost is priced.

### 3.7 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

**v22 vs v7 under the swap rule: 51W-44L-5D, win rate 0.5350,
95% CI [0.4378, 0.6296]**, 4531.1 s. File:
`research/iterations/batch-v22-vs-v7-swap.json` (balanced
placement, placement seed 11, tempo judge, seeds 100/200). Per
colour: as Black 28W-18L-4D (0.6000), as White 23W-26L-1D
(0.4700) — White under swap (0.47) is above v20's (0.39) but
still below 0.50, the same White weakness as §3.4 in milder
form. The tempo judge picked Black for the second player in
76/100 games (White 24/100), identical to v20's split; winners
overall: Black 54, White 41, draws 5. v22 as the opener: 23W-24L-
3D; as the second player (choosing colour): 28W-20L-2D — v22
does better choosing than opening. Mean game length 51.29 plies
(min 17, max 225); wins mean 46.5, losses mean 37.1. 98 distinct
lines. Loss classifier: 44/44 losses via open four, **0 missed
blocks**, 169 threatened turns in losses, **0 clean missed
preventions** — defence is solved. **No swap edge** (0.5350, CI
includes 0.50). v22's swap result sits between v20's (0.49) and
v14's (0.64): the conditional demotion recovers about half of
the swap edge the global demotion cost, consistent with §3.3 —
under swap, where the empty-board opening advantage is removed
and the midgame leaf is measured directly, the Black-only
demotion is worth ~4.5 games per 100 over the global demotion
but still ~10.5 games short of the undemoted v1 leaf.

## 4. Bar check (v22 vs v7)

| Clause | Requirement | v22 | Verdict |
|---|---|---|---|
| 1. h2h vs v7 | ≥100 games, win rate > 0.50 with Wilson 95% CI lower > 0.50 | 54W-38L-8D, 0.5800, CI [0.4821, 0.6720] | **FAIL** |
| 2. Ladder Elo | strictly higher than v7's 190.8 | 147.2, interval [-13.2, 307.6] | **FAIL** |
| 3. Tactical | ≥ v7's 20/20 old + 29/29 hard | 20/20 + 29/29 | PASS |
| 4. Test suites | ml pytest + npm test + tsc green | 340 passed / 7 skipped / 40 xfailed; npm 55/55; tsc clean | PASS |

**v22 is REJECTED (2/4).** v7 remains the reference; master is
untouched. v22's vs-v0 record (61W-30L-9D, 0.6550) is the best of
any v1-leaf version (v14: 0.60, v20: 0.585) but with the most
extreme colour split in the program: Black 0.93 (0 losses) and
White 0.38 (30 losses). Swap vs v7: 0.5350, no edge (v14: 0.64,
v20: 0.49).

## 5. What this means

**The condition worked in aggregate and failed in its premise.**
v22's aggregate vs v0 (0.6550) matches the pre-registered
prediction (≈0.66) — but the colour mechanism is the opposite of
the prediction. Black did not merely recover to v20's 0.72; it
became nearly unbeatable (0.93, ZERO losses in 50 games). White
did not recover to v14's 0.60; it fell to 0.38, worse than v20's
0.45. The decomposition (§3.4) is the iteration's sharpest
result: demoting Black's breadth is worth +0.33 for Black and
−0.22 for White; demoting White's breadth (on top) is worth
−0.21 for Black and +0.07 for White. So the design's premise —
"White's gap-three breadth is signal" — is WRONG in aggregate
against greedy play. White's gap-three breadth is signal only
as the OPPONENT's breadth: Black fearing White's counterplay
fully is worth +0.21 for Black. As White's OWN breadth, it is
phantom: counting it fully makes White overvalue its own
gap-rich positions (fast-loop games 7/9: 18 and 13 gap-three
makers, the top 1.5% of the distribution) while the theirs-term
demotion makes White under-fear Black's threats. Both errors
point the same way — White over-optimistic — and the battery
prices them at 30 White losses.

**Against the searching opponent, the condition balanced the
colours but bought nothing net.** v22 vs v7 is 0.58/0.58 — the
first perfectly symmetric split of any v1-leaf version (v14:
0.72/0.48; v20: 0.61/0.53) — but the aggregate (0.58) is below
v14's 0.60 and only 1 game per 100 above v20's 0.57. The swap
result (0.535, between v20's 0.49 and v14's 0.64) says the same
thing: under swap, where the midgame leaf is measured directly,
the Black-only demotion recovers about half of what the global
demotion cost, and no more. And the ladder exposes a new cost
nobody predicted: v22 collapses against the terminal-leaf MCTS
(0.25, the worst of any v1-leaf version), which caps the ladder
Elo at 147.2 despite the strong vs-heuristic result (0.70,
Black 10-0).

**What changed because of a specific loss:** the shared 28-ply
Black loss line (batch-v14-vs-v0 game 2) produced v20's
mechanism, and v20's battery produced v22's condition — the
colour split (Black 0.72 / White 0.45) is the result v22 was
designed around. The fast loop's two engine-swap-verified White
losses (fl2-g7 ply 17, fl2-g9 ply 19) showed the condition's
cost side before the battery did: White's gap-three breadth is
phantom at saturated positions, and v22 falls for it where v20
does not. The battery confirmed the cost at scale (White 0.38)
and revealed the benefit's true size (Black 0.93). The
regression tests pin all of it: v22 plays the winning save at
fl1-g6-ply28 (where v20 played the drawing save), misses the
save at bl-ply12 (where v20 plays it — the theirs-term working
as designed and costing Black), and misses both fast-loop saves
(xfails).

**What is closed / what is open.** The colour-conditional
demotion as tested (Black's breadth demoted in BOTH terms) is now
measured: it is the best v1-leaf combination vs v0 and a wash vs
v7, and it is REJECTED on clauses 1–2. The condition's two terms
are now separately priced, which opens the precise follow-up:
demote Black's gap threes ONLY in the ours term (when Black is
to move), never in the theirs term — White would fear Black's
gap threes fully, which should repair White's −0.22 without
touching Black's +0.33. Untested; batch 9. Also open: White
gap-saturation (saturation-conditional demotion, or kill-square
dedup — count distinct gap squares instead of makers, IF the
makers share kill-squares, unverified); and the vs-mcts collapse
forensics (why the full theirs-term loses to the terminal-leaf
MCTS in both colours). The v1 kernel itself remains exact and
reusable; the leaf's defect was never in the kernel's arithmetic
but in what the breadth formula counts as a threat — and now,
in whose breadth it counts it.

**Non-transitivity (prominent, as required):** v22 > v7 in both
colours (0.58/0.58) but v22 < v0 as White (0.38), while v7 > v0
in both colours (0.74/0.59); v22 > v0 as Black (0.93) but v22 <
ladder-MCTS in both colours (0.30/0.20), while v20 > ladder-MCTS
(0.65) and v7 < ladder-MCTS (0.40). Ladder Elo ordering: v7
(190.8) > v22 (147.2) > v20 (88.7) > v17 (-17.4) > v14 (-34.9).
The ladder and the h2h orderings disagree (v22 has the best
vs-v0 record of any v1-leaf version but a middling ladder Elo,
because the ladder's MCTS anchor is the opponent v22 is worst
against), which is the program's standing non-transitivity, now
extended to v22.

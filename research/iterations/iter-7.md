# Iteration 7 — v7: MCTS leaf with an attack gradient (batch 3, H1)

**Status: pre-registered — implementation in progress, no evals run
yet.** Comparison target and eval seeds recorded BEFORE the first
eval (see §1). This is the first version of batch 3 (max 3
versions: v7, v8, v9).

**METHOD CHANGE (directed mid-batch by Yudong, ~23:00 PST
2026-11-13; see `research/batch-3-direction-change.md` in the
main repo — cited, not copied here).** The H1 direction and the
4-clause "better" bar are UNCHANGED. The method for finding
weaknesses changes to a FAST HEURISTIC LOOP: play 10 games
against an opponent that produces losses to learn from (v0 is
the default — v6 lost to it 0.41), read EVERY move of every
game, find THE losing move per game (verified with the solver or
a deeper search — never guessed), patch the algorithm so that,
replayed to that critical position, it plays the perfect move,
and record each critical position + correct move as a regression
test. Then play 10 more games and repeat. Max 4 fast-loop
iterations (40 games) before validation, or stop when an
iteration finds no new weakness class. Patches must fix the
STRUCTURAL cause (a recurring weak point / weak strategy), not
the single position — the regression test guards the position;
the patch must generalise. Regression files:
`ml/gomoku_ml/regression_positions_v7.json` +
`ml/tests/test_regression_v7.py`. The full 4-clause validation
battery (100-game matches + ladder) runs on the FINAL patched
v7 only. Numbers already run on unpatched v7 (the smoke and the
100-game v7-vs-v1 match in flight when the direction changed)
are kept as INTERMEDIATE (pre-patch) results in §3; the bar
verdict in §4 uses the final patched engine's battery only. The
aggregate loss analysis (counts) becomes secondary — the
fast-loop worked games ARE the loss analysis. Fast-loop games
use development seeds (500/600, 501/601, ...) distinct from the
bar battery's pre-registered seeds (100/200), so no development
game overlaps an eval game.

## 1. Design and pre-registration

**Comparison target: v1** (primary, same as v4/v5/v6 for
comparability). The bar's clause 1 is head-to-head vs v1 (normal
rules, >= 40 games, alternating colours), clause 2 is ladder Elo
vs v1's 70.4 (identical 20 games/opponent, mcts_sims=100),
clause 3 is tactical pass rate vs v1's on the hard suite (12/29)
and the old suite (20/20). **Secondary head-to-heads, also
pre-registered:** v7 vs v0 (100 games) and a 100-game swap match
vs v1 under the tempo judge (the harness default). v3's numbers
(Elo 107.5, hard suite 26/29) are reported alongside for
transparency.

**The improvement: v7 = v6's PUCT MCTS with an attack gradient
in quiet leaf positions.** Everything else is v6 unchanged: 100
sims, c_puct 1.5, v1 priors over all candidates, win/block scans
before search, most-visited root child. The ONLY change is the
leaf value in quiet positions (no winning move for either side,
opponent with < 2 winning moves): v6 values those at 0; v7
values them by the BREADTH of each side's attack, computed from
the portable kernel's per-cell attack scores:

- `breadth(side) = #strong + 0.25 * #developing`, where strong =
  empty cells with v0 attack score >= 1000 (a move creating a
  four or an open three — a (4,1)/(3,2) pattern in at least one
  direction) and developing = empty cells with attack score in
  [100, 1000) (a move creating a half-open three or an open
  two). Counts are over empty cells only.
- Leaf value (quiet positions) =
  `tanh((breadth(ours) - breadth(theirs)) / 4)`.
- Terminal positions keep v6's values exactly: player to move
  with a winning move +1; opponent with >= 2 winning moves -1;
  opponent with exactly 1 winning move -0.25.

**Why BREADTH, not best-attack difference.** Iter-5 §2.2 showed
the static leaf's failure mode precisely: it values the BEST
attack, not the breadth of threats — after the losing (4,9) Black
had six single-four creators (a forced blocking sequence White
eventually loses) while after the winning (7,11) Black had one
open-four threat (answerable next move), yet the best-attack leaf
ranked (4,9) far above (7,11). A best-attack gradient would
repeat that error inside the MCTS leaf. A count of high-attack
moves measures how MANY threats a side holds: six creators count
six times, one creator counts once. The developing tier (weight
0.25) keeps the gradient informative in early quiet positions
where neither side has a strong move yet, without letting
low-grade two-makers dominate the count.

**Why the kernel, and what it does NOT see.** The gradient uses
the portable kernel's attack-only scores (`kernel_numpy.attack_both`,
added in this iteration — friendly part only, no centre term,
cross-checked against a Python reference built from baseline
internals with 0 mismatches on all fixtures, both players).
**The kernel ports v0 (contiguous-run scoring), NOT v1 — stated
plainly:** it is blind to broken fours (a broken-four creation
scores as two separate runs, ~100-200, not as a four), so the
gradient undervalues broken-four setups relative to v1's
5-window scoring. That is the known cost of the fast path; a
v1-kernel gradient is a documented follow-up if v7 shows promise
but the gradient looks too weak. The gradient is also the reason
for the attack-only accessor: the full kernel total is (up to the
0.9 defence discount) colour-symmetric, so it cannot distinguish
"our attack" from "their attack" — only the friendly part can.

**Why the squash scale is 4.** Measured on 357 quiet positions
replayed from batch-2 v6-vs-v1 games (see Step 0 below): the
breadth difference has std 2.26, p10 -3.25, p90 2.50, and
|diff| >= 4 in only 7.6% of positions. Scale 4 (~1.8 sigma) keeps
the squash mostly in its near-linear region — a real attack edge
(diff +2) maps to 0.46, a decisive edge (diff +4, top ~8%) to
0.76 — graded signal, not a near-binary one, and always below the
terminal values. Scale 2 would push a diff of 4 to 0.96, nearly a
terminal win for a mere breadth edge. The scale is fixed here
from the measured distribution, NOT tuned on any v7 eval result.

**Step 0 — cost measurement (re-measured here, NOT quoted from
batch-2 notes).** `ml/scripts/bench_v7_leaf.py` on 417 boards
replayed from batch-v6-vs-v1.json (20 games, every 4th ply from
ply 8; median 57 stones), median of 3 rounds x 50 calls, on this
VM while the parallel H2 workstream shares it (so absolute
numbers are contention-noisy; the ordering is what matters):

| Call | Median ms/call |
|---|---|
| (a) `scoring._leaf_value` (Python static leaf) | 9.72 (min 9.04, max 13.83) |
| (b) `kernel_numpy.score_all` (single position) | 2.65 (min 1.65, max 3.29) |
| (c1) `kernel_numpy.attack_all` (one colour) | 1.65 (min 0.56, max 1.79) |
| (c2) `kernel_numpy.attack_both` | 1.94 (min 1.85, max 2.08) |
| (c3) full breadth gradient (`attack_both` + counts) | 2.06 (min 1.69, max 2.28) |
| (d) v6 `_terminal_leaf_value` | 0.56 (min 0.55, max 1.27) |

The gradient costs ~2.1 ms per quiet leaf vs 0.56 ms for v6's
terminal-only leaf — ~4x the leaf cost, but the leaf is only
part of a move's cost (the v1 prior computation at each expansion
dominates v6's ~0.2 s/move). Expected v7 move cost ~0.35-0.4 s,
a 100-game match ~25-30 min. Affordable. `attack_both` (1.94 ms)
is cheaper than two `attack_all` calls and about the same as one
`score_all` (2.65 ms) — the two-sided gradient costs roughly one
kernel call, not two. The Python `_leaf_value` (9.72 ms) would
have made a 100-sim move cost ~1.3 s and is NOT used (same
conclusion as iter-6, re-measured).

**What this change is meant to fix — traced to the analysed gaps
(iter-6 §2).** v6 defends perfectly (166/166 losses via open
four, 0 missed blocks, 1 missed prevention in 1,061 threatened
turns) and loses the open-four race because its terminal-only
leaf values every quiet position at 0: with a sparse leaf, root
visit counts track priors more than values (iter-6 §2.2c — (6,7)
26 visits at q 0.038 beat (5,6) 3 visits at q 0.333), so v6
plays its priors (v1's scoring) plus noise. The direct A/B
confirmed the mechanism: v6 lost to the ladder's baseline MCTS
(9W-11L at 20 games, 21W-29L at 50) because the baseline's crude
local-attack leaf provides a gradient and v6's does not. The
gradient gives the q values something to separate on in exactly
the quiet positions where the race is decided. If v7 still loses
the race, the next step is a learned value function (H2) or a
v1-kernel gradient — the gradient's information content, not the
search, is then the binding constraint.

**Eval seeds (pre-registered):** h2h vs v1: v7=100 / v1=200;
h2h vs v0: v7=100 / v0=200; ladder seed 100, 20 games/opponent,
mcts_sims=100; swap 100 games vs v1, balanced placement,
placement_seed 11, tempo judge (the harness default; recorded in
the JSON).

**Pre-eval observation (recorded BEFORE any eval, during
implementation testing — the design stands as pre-registered;
the evals decide).** v7 FAILS the iter-5 game-1 regression that
v6 passes: on the move-30 position (White to move, Black holding
exactly one creator, (4,9)), v7 plays the losing block (4,9)
while v6 finds the oracle-verified saving moves (7,11)/(6,12).
Mechanism, verified by direct computation on the position:

- (7,11)/(6,12) are FORCING (White makes a four: White winning
  moves after are [(6,12)] / [(7,11)]), but the gradient at the
  resulting position is -0.555 / -0.462 for White — Black still
  holds its creator and more breadth, so the two-sided gradient
  scores the counterattack as bad for White.
- (4,9) is NOT forcing but kills Black's creator (0 creators
  after) and scores a gradient of -0.062 for White — nearly
  neutral, far above the forcing moves. The gradient rewards
  immediate threat-reduction (blocking) over forcing counterplay:
  it reintroduces the 1-ply myopia v6's terminal-only leaf
  avoided. v6's sparse leaf lets the -0.25 penalties from Black's
  forcing sequence ((3,9) -> four -> block -> ...) differentiate
  the moves over the search; v7's gradient swamps them at
  shallow depth (100 sims cannot reach the 26-ply loss).
- Second mechanism: after (4,9)+(3,9) Black holds SIX creators
  behind a single four — but the position is not quiet (Black
  has a winning move, (2,10)), so the leaf is -0.25 flat and the
  gradient is NOT applied. The breadth information is discarded
  exactly where it matters most.

This is ONE diagnostic position, and the program has repeatedly
shown diagnostics do not predict match results (v6: tactically
perfect, loses to v0 0.41). The pre-registered design is NOT
amended on its basis — amending after seeing v7 behave badly on
a known-hard position would be moving the goalposts, and the
evals are the pre-registered decision procedure. If v7 fails the
matches, the one-sided gradient (reward only OUR breadth; let
the search handle THEIR threats through terminal values, so
blocking earns no direct gradient reward) is the concrete v8
candidate — see §5. The regression is recorded as an xfail test
(`test_v7_iter5_game1_regression_xfail`) so the behaviour is
visible in the suite without breaking clause 4.

## 2. Loss analysis for iteration 7

The fast-loop worked games (§2.1-2.4, one subsection per
iteration) ARE the loss analysis — see the method-change note at
the top of this log. Aggregate counts (§2.5) are secondary.

### 2.1 Fast-loop iteration 1 — 10 games vs v0

**Batch:** `research/iterations/fastloop1-v7-vs-v0.json`
(seeds v7=500 / v0=600 — development seeds, distinct from the bar
battery's 100/200). **5W-5L-0D, win rate 0.5000, CI [0.2366,
0.7634]**, 112.1 s. The split is PURELY colour: v7 wins all 5 as
Black and loses all 5 as White, all fast (21, 31, 29, 21, 21
plies). v7 as White cannot overcome Black's first-move tempo
against v0's direct attacking play — the same White weakness v6
showed vs v0 (0.26 as White). Classifier
(`ml/scripts/classify_batch_losses.py`): 5/5 losses via open
four, 0 missed blocks, 0 clean missed preventions in 21
threatened turns (in losses). Defence is still perfect; the
losses are lost races, as in every batch before.

**The losing move in each game** (walk-back,
`ml/scripts/walkback_loss.py`, corrected oracle method — the
actual move's v1-vs-v1 playout must lose at all 3 seeds before an
alternative counts; seeds 0,1,2):

| Game | Last preventable turn | v7 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 1 (21 plies) | ply 13 | (4,6) | (8,6) — wins 3/3 |
| 3 (31 plies) | ply 23 | (4,9) | (9,12) — wins 3/3 |
| 5 (29 plies) | ply 21 | (9,11) | (5,7) or (8,12) — win 3/3 |
| 7 (21 plies) | ply 15 | (7,6) | (7,10) — draws/wins |
| 9 (21 plies) | ply 15 | (8,7) | (4,7) — wins/draws |

**The recurring structural weakness — ONE class, all 5 games.**
In every critical position Black holds exactly 2 creators and
White holds 0: White is defending against Black's open-four
threat. In every case the losing move is a PURE BLOCK (v1 rank
1-2, ~200k) and the saving move is either a DUAL-PURPOSE
block+counterattack (games 1, 7, 9 — v1 rank 1) or a FORCING
move that creates a four (games 3, 5 — v1 rank 8 and 4). And in
every case the uncapped gradient PREFERS the losing move:
game 1: +0.510 (actual) vs +0.245 (saving); game 3: +0.358 vs
0.000; game 5: +0.762 vs +0.303; games 7/9: +0.671 vs +0.635
(leaf values for White, from `ml/scripts/analyse_critical.py`).
The gradient rewards immediate threat-reduction (blocking kills
Black's breadth NOW) over counterplay (building White's attack
for the NEXT move) — the 1-ply myopia of iter-5, rebuilt inside
the leaf. Worse, the gradient's dense positive values on the
blocking move's lines MASK the sparse -0.25 penalties from
Black's forcing sequences: in 4/5 positions v7's own q-values
ALREADY prefer the saving move (game 1: -0.07 vs -0.257;
game 5: +0.106 vs -0.381; games 7/9: -0.253 vs -0.335), but the
visit counts follow the gradient-inflated early values (48/68/67
vs 15/2/18) and most-visited decides — iter-6 §2.2c's pathology,
amplified by the gradient. v6's terminal-only leaf finds a
saving move in 4/5 positions (games 1, 5, 7, 9 — game 5 via
(8,12)); the gradient is what breaks it. The one-sided gradient
(reward only our breadth) was measured and REFUTED as a fix: it
also prefers the losing move in all 5 cases (the block raises
White's breadth too), and it saturates (breadths 7-11.5 all map
to 0.94-0.99) — see the analysis in §2.1's working notes.

**The patch (structural, generalises): cap the gradient at
+/-0.2.** A static heuristic must never outweigh an exact
terminal signal: the cap is strictly below the smallest terminal
value (the must-block 0.25), so forcing moves always beat quiet
gradient differences in leaf value and the gradient acts as a
tie-breaker between quiet lines — all a static heuristic can
honestly claim. (`_GRADIENT_CAP = 0.2` in `mcts_heuristic.py`.)
Patch candidates were tested by replaying all 5 critical
positions (`ml/scripts/test_patch_candidates.py`): uncapped
base 0/5, cap 0.2 **4/5**, suppress-on-creator 3/5,
cap+suppress 4/5 (no better than the cap alone — the cap is the
single patch). Game 3 is fixed by NONE of them: its saving move
(9,12) is a forcing move ranked 8th (prior 0.04), starved of
visits in 100 sims — an EXPLORATION weakness, not a leaf-value
weakness, and the target for iteration 2 if it recurs.

**Regression tests:** `ml/gomoku_ml/regression_positions_v7.json`
(5 positions, stone lists + expected moves, generated from the
game records) + `ml/tests/test_regression_v7.py`. After the
patch: 4/5 pass (g1, g5, g7, g9); g3 is a documented xfail (the
open forcing-move exploration weakness). The pre-eval iter-5
regression (test_mcts_heuristic.py xfail) is NOT fixed by the
cap either — its mechanism is the gradient's ORDERING (the block
beats the counterattack in breadth terms), not its magnitude —
and stays xfail. Targeted suites after the patch:
test_regression_v7 + test_mcts_heuristic 26 passed, 2 xfailed.

### 2.2 Fast-loop iteration 2 — 10 games vs v0 (patched engine)

**Batch:** `research/iterations/fastloop2-v7-vs-v0.json`
(seeds v7=501 / v0=601). **9W-1L-0D, win rate 0.9000, CI
[0.5958, 0.9821]**, 120.8 s — up from 5W-5L in iteration 1.
v7 now wins 4/5 as White (was 0/5). The cap patch holds:
White's fast losses to v0 are largely gone. Classifier: the
single loss is via open four, 0 missed blocks, 0 clean missed
preventions (4 threatened turns in the loss).

**The single loss (game 5, v7 as White, 21 plies) has NO single
losing move.** Walk-back (corrected oracle method, seeds 0,1,2):
from ply 13 on, the actual move loses at all 3 seeds but NO
tested alternative saves — not in the top-6+forcing set, and not
in a widened top-15+forcing set re-tested at ply 13 and ply 15
(14 alternatives each, 0 saves). At ply 11 the actual move
(5,4) does NOT lose ([draw, win, draw]), so by the corrected
method it is not the losing move either. This is the iter-6
result: distributed drift, not a blunder.

**What happened (verified, not guessed).** White played
CORRECTLY by prevention logic at every turn: at ply 11 Black
had 0 creators; at ply 13 Black had 1 creator (10,4) and White
occupied it (v1 rank 1); at ply 15 Black had 2 creators
(10,3),(6,7), White played (6,7) — verified preventing (it
kills both) — and White had NO forcing move available. Then
Black's (9,6) at ply 16 created a DOUBLE OPEN THREE from a
zero-creator position: row 6 (7,6),(8,6),(9,6) plus the diagonal
(7,4),(8,5),(9,6), giving Black FOUR creators at ply 17
[(6,3),(6,6),(10,6),(10,7)] — verified. No single White move
can stop four creators; Black plays (6,3) (open four on the
diagonal), White blocks (5,2), Black plays (10,7) and wins.
The structural note: a double-open-three CREATION is invisible
to prevention logic, which tracks EXISTING creators only — the
creators did not exist before (9,6). But there is no verified
losing move to patch against (no alternative saves at any turn),
so there is NO PATCH from this iteration: patching on an
unverified hypothesis would violate the loop's "never guessed"
rule. The double-three shape is recorded here as an observation
for the next batch, not a patch target.

**Loop stops here (stopping rule).** Iteration 2 found no new
patchable weakness class, so the fast loop ends after 2
iterations (20 games) per the direction change's stopping rule.
Open items carried forward (not patched): the iteration-1 game-3
forcing-move exploration weakness (regression xfail) and the
iter-5 regression xfail — both single-position cases without a
recurring class behind them in the loop's games.

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

(TODO — filled in after the final battery: losses via open four,
missed blocks, clean missed preventions with threatened-turn
counts, across the three 100-game matches of the final battery,
via `ml/scripts/classify_batch_losses.py`.)

## 3. v7 evaluation

### 3.0 Intermediate (pre-patch) results

The smoke (§3.1) and the 100-game v7-vs-v1 match (§3.3) ran on
UNPATCHED v7, before the fast loop (the match was in flight when
the direction changed, ~23:00 PST). They are intermediate
results, NOT the bar verdict — the bar battery (§3.5-3.8) runs
on the final patched engine. Tactical (§3.2) was also measured
on unpatched v7 and is re-run on the final engine in §3.5.

(TODO — filled in after the evals.)

### 3.1 Smoke (20 games vs v1, seeds 100/200)

**14W-2L-4D, win rate 0.8000, CI [0.5840, 0.9193]**, 790.8 s
(~39.5 s/game; plies mean 83.2, min 22, max 225); as Black
9W-0L-1D (0.95), as White 5W-2L-3D (0.65). Win plies mean 49.1,
loss plies mean 39.0 (v7 wins and loses FAST — unlike v6, which
ground out 85-ply games and lost slower than it won). The 4
draws are full-board 225-ply games. Far above v6's smoke
(6W-10L-4D, 0.40) — but n=20 with a wide CI, and the 2 losses
are the SAME game played twice (games 9 and 13, both v7 as
White, both 39 plies, identical move lists: deterministic
engines transpose). Not conclusive; the pre-registered 100-game
match decides. Cost note: 39.5 s/game is ~4x v6's 9.3 s/game —
the gradient roughly doubles v7's per-move cost (~0.95 s/move
measured here, vs v6's ~0.21 s/move), and this run shared the VM
with the parallel H2 workstream plus a full test-suite run for
its first 12 minutes. The 100-game matches will each take
~35-60 min under contention.

### 3.2 Tactical suites

Old 20-position suite, full-dispatch driver
(`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
NOT `eval_version.py`'s probe, which bypasses the MCTS dispatch):
**v7 20/20** (= v1's 20/20, = v6's 20/20).
Hard suite (29 positions): **v7 26/29** (prevention 15/16, setup
6/7, winning 1/2, vcf 4/4) vs v1's 12/29 and v6's 29/29. The
gradient COST v7 three positions vs v6's perfect score:

- `prev-v1v1s-g45` (prevention): chose (6,2), expected (11,7)/(12,7).
- `setup-v1v1s-g33` (setup): chose (9,4), expected (6,7).
- `winning-quiet-setup-2` (winning): chose (9,7), expected
  (7,7)/(8,7)/(8,9)/(10,10) — one of the 3 positions v3 failed
  that v6 passed. The trap is "the 58k double three is a decoy"
  (gap-threes whose extensions give only single-ended fours) —
  and a breadth gradient is exactly the wrong tool there: it
  rewards the move that creates the most threats, decoys included.

Clause 3 still passes (26/29 >= v1's 12/29), but the direction
is the wrong one: the gradient makes single-move tactical
decisions WORSE than v6's terminal-only leaf, consistent with
the pre-eval iter-5 regression failure (§1). If v7 nevertheless
wins matches, the gradient's match value comes from multi-move
attack-building, not from tactical accuracy.

### 3.3 Head-to-head vs v1 (100 games, seeds 100/200) — INTERMEDIATE (pre-patch)

**NOT COMPLETED — interrupted by a runtime restart, not re-run.**
The match ran 54 min (~97/100 games by CPU-time estimate) and
was killed by a VM/runtime restart at ~07:25 PST 2026-11-14
(process gone, no output JSON — `selfplay_batch.py` writes only
at completion, and the progress output was piped through `tail`,
so no partial record survives). Decision, recorded here: NOT
re-run. The direction change frames intermediate numbers as
conditional ("if you have already run 100-game matches on
unpatched v7") — the completed 20-game smoke (§3.1, 0.80) is the
pre-patch evidence vs v1, and an hour of compute for a number
the bar does not use is not worth delaying the fast loop and the
final battery. The final battery (§3.5-3.8) re-runs the v1 match
on the patched engine with the same pre-registered seeds. Cost
of the loss: no precise unpatched-v7-vs-v1 baseline, so the
patch's isolated effect vs v1 can only be gauged against the
smoke (n=20, wide CI), not a 100-game number.

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

(TODO — runs on the final patched engine.)

### 3.5 Tactical suites — FINAL BATTERY (re-run on the final patched engine)

(TODO — old suite + hard suite, full-dispatch driver.)

### 3.6 Head-to-head vs v1 (100 games, seeds 100/200) — FINAL BATTERY

(TODO — runs on the final patched engine; the §3.3 numbers are
the pre-patch intermediate.)

### 3.7 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

(TODO — runs on the final patched engine.)

### 3.8 Swap match vs v1 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

(TODO — runs on the final patched engine.)

## 4. Bar check (v7 vs v1)

(TODO — filled in after the evals.)

## 5. What this means

(TODO — filled in after the evals.)

# Iteration 7 — v7: MCTS leaf with an attack gradient (batch 3, H1)

**Status: COMPLETE — final battery done on the final patched engine
(§3.4–3.8). Clauses 1–3 PASS; clause 4 is green on every suite
runnable in this worktree (full ml pytest green), with npm test
pending coordinator verification at batch end — see §4.** This is
the first version of batch 3 (max 3 versions: v7, v8, v9).
Comparison target and eval seeds were recorded BEFORE the first
eval (see §1).

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

Across the final battery's three 100-game matches (classifier:
`ml/scripts/classify_batch_losses.py`, validated in batch 2 to
reproduce iter-6's numbers exactly):

| match | losses | via open four | missed blocks | threatened turns (losses) | clean missed preventions (losses) |
|---|---|---|---|---|---|
| vs v1 | 21 | 21 | 0 | 93 | 1 (1.1%) |
| vs v0 | 31 | 31 | 0 | 141 | 0 (0.0%) |
| vs v1 swap | 37 | 37 | 0 | 150 | 0 (0.0%) |

Defence is solved: 89 losses, 89 via open four, 0 missed
blocks, 1 clean missed prevention in 384 threatened turns.
Every loss is the opponent winning the open-four race, not a
defensive error.

### 2.6 Worked game from the final battery — v7-as-Black loses to v0 (game 16)

The most informative final-battery loss: v7 as Black loses to
v0 in 22 plies (games 16 and 38 are the identical deterministic
replay). Walk-back (corrected oracle, seeds 0,1,2): the losing
move is **ply 14, (8,4)** — loses 3/3, with saving moves (6,9)
[win,win,win], (9,8) [draw,win,draw], (7,4) [win,win,loss];
at ply 12 the actual move (11,4) draws/wins, so the game turns
at ply 14. Mechanism (verified): (8,4) creates a half-open four
(8,4)-(11,4), open only at (7,4) — White's block there is
FORCED (verified: White playing (5,5), (9,9) or (3,3) instead
leaves Black winning at (7,4) immediately). But (7,4) is the key
square of White's own diagonal attack (7,4),(8,5),(9,6),(10,7),
(11,8): Black's forcing move compels White to occupy it, and
White then plays (8,5) (attack score 58001), (10,7) (open
four), (11,8) (win). After the forced (7,4), White has 0
creators and 0 winning moves — the danger is invisible to the
terminal leaf and worth only the capped −0.2 gradient — and the
−1 is 5 plies deep, beyond what 100 sims reliably reach.

The deeper finding is in the root statistics at ply 14: (7,4)
and (8,4) TIE at 10 visits each (the max), and the match RNG
broke the tie toward the losing (8,4). The saving move (7,4)
had the HIGHER q (0.042 vs 0.032), and the other saving moves
rank higher still on q ((9,8) 0.071, (6,9) 0.065, both at 7
visits). v1's own 1-ply ranking also prefers the saves ((9,8)
rank 2, (7,4) rank 3, (6,9) rank 5, (8,4) rank 6). So the
search's q-values know the right move, but the **most-visited
decision rule discards q and the RNG tie-break loses the
coin-flip** — the same pattern as fast-loop iteration 1 (§2.1),
where q preferred the saving move in 4/5 positions but visits
decided. Recommendation for the next batch (v8): decide the
root move by q (or a q/visit blend), or at minimum break
visit-ties by q — that single change saves this game and
addresses the recurring q-vs-visits gap. No patch is made here:
this finding comes from the final battery, whose numbers are the
verdict; it is recorded for the next batch.

A limitation of the fast loop, noted honestly: the 10-game
batches never surfaced this as-Black weakness (v7 won 5/5 as
Black in both iterations); the 13/50 as-Black loss rate to v0
only appears at 100-game scale. Ten-game batches find the
dominant weakness fast but miss sub-dominant patterns.

## 3. v7 evaluation

### 3.0 Intermediate (pre-patch) results

The smoke (§3.1) and the 100-game v7-vs-v1 match (§3.3) ran on
UNPATCHED v7, before the fast loop (the match was in flight when
the direction changed, ~23:00 PST). They are intermediate
results, NOT the bar verdict — the bar battery (§3.5-3.8) runs
on the final patched engine. Tactical (§3.2) was also measured
on unpatched v7 and is re-run on the final engine in §3.5.

(Intermediate results in §3.1–3.3; final battery on the patched
engine in §3.4–3.8.)

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

**v7 beats v0: 64W-31L-5D, win rate 0.6650, 95% CI
[0.5679, 0.7499]**, 3598.5 s. File:
`research/iterations/batch-v7-vs-v0.json` (seeds 100/200).
Per colour: as Black 34W-13L-3D (0.7100), as White 30W-18L-2D
(0.6200). Note the anomaly: v7 scores LOWER against the weaker
v0 (0.6650) than against v1 (0.7050), and loses 13 games as
Black to v0 versus only 3 as Black to v1 — v0's crude greedy
play exploits something in v7's Black game that v1's stronger
play does not (see §2.5 for the classifier breakdown). This is
NOT full non-transitivity (v7 still beats v0 outright), but the
colour-split inversion is recorded here prominently per the
brief. (First attempt at this match was killed by a runtime
restart at ~09:25 PST with no JSON written; the match is
deterministic, so the re-run reproduces the same games.)

### 3.5 Tactical suites — FINAL BATTERY (re-run on the final patched engine)

Old 20-position suite, full-dispatch driver: **v7 20/20** (=
v1's 20/20, = v6's 20/20). Hard suite (29 positions): **v7
29/29 — a perfect score** (prevention 16/16, setup 7/7, winning
2/2, vcf 4/4), matching v6's 29/29 and up from the unpatched
v7's 26/29. The cap patch restored all three positions the
uncapped gradient had cost (`prev-v1v1s-g45`,
`setup-v1v1s-g33`, `winning-quiet-setup-2`) — independent
confirmation, on positions the fast loop never touched, that the
uncapped gradient's magnitude was the problem: capping it
restores v6's tactical perfection while keeping the gradient's
match-play value (see §3.6-3.8). **Clause 3 PASSES** (old 20/20
= v1's 20/20; hard 29/29 vs v1's 12/29).

### 3.6 Head-to-head vs v1 (100 games, seeds 100/200) — FINAL BATTERY

**v7 beats v1: 62W-21L-17D, win rate 0.7050, 95% CI
[0.6094, 0.7855]**, 3598.1 s. File:
`research/iterations/batch-v7-vs-v1.json` (seeds 100/200).
Per colour: as Black 42W-3L-5D (0.8900), as White 20W-18L-12D
(0.5200) — a strong colour asymmetry; 18 of v7's 21 losses
come as White. Mean game length 71.55 plies (min 20, max 225).
Loss classifier (`ml/scripts/classify_batch_losses.py`):
21/21 losses via open four, **0 missed blocks**, 93 threatened
turns in losses, **1 clean missed prevention (1.1%)** — defence
is essentially solved; v7's losses as White are v1 out-attacking
it in the open-four race, not defensive blunders. **Clause 1
PASSES** (0.7050 > 0.50, CI lower bound 0.6094 > 0.50) — the
first MCTS version in this program to beat v1 head-to-head
(v6's h2h vs v1 was 0.535, CI lower 0.4378).

### 3.7 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

vs random 20W-0L-0D (1.0000, CI [0.8389, 1.0000]; as Black
10W-0L-0D, as White 10W-0L-0D); vs heuristic 15W-5L-0D (0.7500,
CI [0.5313, 0.8881]; as Black 8W-2L-0D, as White 7W-3L-0D); vs
mcts 8W-12L-0D (0.4000, CI [0.2188, 0.6134]; as Black 5W-5L-0D,
as White 3W-7L-0D). **Elo 190.8, interval [21.8, 359.9]** —
strictly above v1's 70.4 (interval [-80.2, 221.1]) at identical
game counts and opponents. **Clause 2 PASSES.** (The intervals
overlap — the bar requires the point estimate strictly higher,
which holds; the overlap is recorded honestly, and the Elo
estimate is driven by the vs-heuristic matchup, elo_diff
+190.8.) Run time 1469.1 s; EVAL_JSON saved to
`research/iterations/ladder-v7.json` (full log:
`research/iterations/ladder-v7.log`). The script's tactical
probe line was disregarded per the brief (it bypasses the MCTS
dispatch); tactical is §3.5's full-dispatch 20/20 + 29/29.

Two things stand out. First, the vs-heuristic 15W-5L is the
first time an MCTS version in this program beats the heuristic
anchor outright (v6: 7W-13L; v5: 0.28 h2h vs v0) — the capped
gradient turns the search from a liability against 1-ply play
into a real edge, in both colours (8-2 as Black, 7-3 as White).
Second, the vs-mcts 8W-12L is slightly WORSE than v6's 9W-11L:
against the baseline MCTS (same 100 sims / c_puct 1.5, crude
local-attack leaf), v7's capped gradient still does not win the
direct A/B — the gradient's match value shows against the
1-ply heuristic, not against a search opponent whose own leaf
already carries an (uncapped, cruder) attack signal. The
colour pattern vs mcts is the familiar one (5-5 as Black, 3-7
as White): v7 as White still cannot create the attack against
a searching opponent.

### 3.8 Swap match vs v1 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

**51W-37L-12D, win rate 0.5700, CI [0.4722, 0.6627]**, 6444.5 s
(secondary metric, not the bar). File:
`research/iterations/batch-v7-vs-v1-swap.json` (seeds 100/200).
As Black 25W-23L-2D (0.5200); as White 26W-14L-10D (0.6200).
second_colour Black 76 / White 24 — both engines chose
identically (38 Black / 12 White each), since the tempo judge is
deterministic on the same balanced placements; v7's win rate is
0.570 whether v7 or v1 chose the colour, so the choice itself
does not drive the result. Winners Black 39, White 49, draws 12
(White-favouring by 10 — the reverse of v6's swap, where Black
won 43 vs White 37). Plies mean 73.0 (min 19, max 225). Loss
classifier: 37/37 losses via open four, **0 missed blocks**,
150 threatened turns in losses, **0 clean missed preventions**.

Note the contrast with §3.6: under swap rules v7's edge over v1
is NOT statistically established (CI [0.4722, 0.6627] includes
0.50), while the normal-rules match is decisive (0.7050, CI
lower 0.6094). The swap rule removes the empty-board opening
phase where v7's Black edge is largest (normal rules as Black
0.89; swap as Black 0.52) — v7's advantage over v1 lives
substantially in the opening, and from a balanced 3-stone start
the two engines are much closer. The bar's clause 1 is the
normal-rules match and it passes; the swap number is recorded
here as the honest tempering caveat.

## 4. Bar check (v7 vs v1)

| Clause | Requirement | v7 result | Verdict |
|---|---|---|---|
| 1. Head-to-head vs v1 | >= 40 games, win rate > 0.50, Wilson 95% CI lower bound > 0.50 | 100 games: 62W-21L-17D, 0.7050, CI [0.6094, 0.7855] | **PASS** |
| 2. Ladder Elo | strictly higher than v1's 70.4, identical games/opponents | 190.8 (interval [21.8, 359.9]), vs v1's 70.4 | **PASS** |
| 3. Tactical | pass rate >= v1's | old 20/20 = v1's 20/20; hard 29/29 vs v1's 12/29 | **PASS** |
| 4. Test suites | ml pytest + npm test green | ml pytest 208 passed, 1 skipped, 2 xfailed, 0 failures (targeted suites: test_regression_v7 + test_mcts_heuristic 26 passed, 2 xfailed); npm test NOT run here — no node_modules in this worktree; verified by the coordinator at batch end | **PENDING** coordinator verification (npm test) |

**Verdict: v7 PASSES clauses 1–3 — the first version in this
program to pass all three.** Clause 4 is green on every suite
runnable in this worktree (full ml pytest green, 0 failures),
but npm test could not run here and is verified by the
coordinator at batch end. The overall verdict is therefore
**PASSES SUBJECT TO the coordinator's npm-test verification**;
only the coordinator may decide a master push, and only if all
four clauses pass. Nothing has been pushed to master from this
workstream.

## 5. What this means

**The attack-gradient hypothesis is CONFIRMED — with the cap
as part of the result, not a footnote.** v6 proved the
terminal-only leaf defends perfectly and attacks nothing (loses
to v0 0.41, ties v1 0.535, Elo -107.5). v7 adds exactly one
thing — a breadth-of-attack gradient in quiet leaves — and
becomes the first MCTS version in the program to beat v1
head-to-head (0.7050, CI lower 0.6094), the first to beat the
heuristic ladder anchor (15W-5L), and the highest-Elo version
yet (190.8, above v3's 107.5 and v1's 70.4). But the UNCAPPED
gradient failed: tactical 26/29 (three positions lost vs v6),
fast-loop 5W-5L vs v0 with all five White losses from the same
1-ply myopia (the gradient rewards blocking over counterplay),
and the iter-5 regression failure. The cap at ±0.2 — strictly
below the smallest terminal value (0.25) — is what makes the
gradient safe: a static heuristic must never outweigh an exact
terminal signal, and as a tie-breaker between quiet lines the
gradient is worth ~300 Elo points of ladder improvement (v6's
-107.5 to v7's 190.8) with zero tactical cost. The cap patch
restored tactical 26/29 -> 29/29 AND the match wins (fast loop
5W-5L -> 9W-1L; final battery 0.7050 vs v1), confirmed on
positions the fast loop never touched. The lesson generalises:
any future leaf signal (learned value included) should be
capped below the terminal floor, or the search will trade exact
forcing sequences for heuristic comfort.

**The remaining weaknesses, in order of importance.**

1. **White play.** v7 as White vs v1 is 20W-18L-12D (0.52);
   18 of its 21 losses come as White. The fast loop fixed
   White's fast losses to v0 (0/5 -> 4/5), but against v1
   White still cannot CREATE the attack to overcome Black's
   first-move tempo — the same weakness every version has
   shown. The gradient helps White defend-and-counterattack,
   but building an attack from nothing, a move ahead of
   Black's, is still missing. This is the single biggest
   remaining gap and the main reason the v1 match is 0.7050
   rather than higher.
2. **The q-vs-visits root-decision gap (§2.6).** In the worked
   game the saving moves had the HIGHER q ((7,4) 0.042 vs
   (8,4) 0.032; (9,8) 0.071, (6,9) 0.065) but tied or trailed
   on visits, and most-visited plus an RNG tie-break picked
   the loser. The same pattern appeared in fast-loop
   iteration 1 (q preferred the saving move in 4/5 positions;
   visits decided). The search KNOWS the right move and the
   decision rule discards that knowledge. **Recommendation
   for the next batch (v8): decide the root move by q (or a
   q/visit blend), or at minimum break visit-ties by q.**
   That single change saves the §2.6 game and addresses the
   recurring gap. It is cheap, it is verified against real
   losses, and it should be the first v8 candidate.
3. **The colour-split inversion vs v0 — recorded prominently.**
   v7 scores LOWER against the weaker v0 (0.6650) than
   against v1 (0.7050), and loses 13 games as Black to v0
   versus only 3 as Black to v1. This is NOT full
   non-transitivity (v7 beats v0 outright, 64W-31L-5D), but
   the inversion is real and unexplained: v0's crude greedy
   play exploits something in v7's Black game that v1's
   stronger play does not. The §2.6 worked game is one such
   loss (a forcing move that compels White onto the key
   square of its own attack — a danger invisible to the
   terminal leaf and worth only the capped −0.2 gradient,
   5 plies deep). The next batch should work at least one
   more as-Black loss to v0 before assuming the inversion is
   noise: if v7's Black game has a systematic hole against
   greedy attack, it will show against stronger opponents
   too.
4. **The direct A/B vs the baseline MCTS still goes the wrong
   way.** Ladder vs mcts is 8W-12L, slightly worse than v6's
   9W-11L. v7's capped gradient beats the 1-ply heuristic
   decisively but does not beat a searching opponent whose
   own leaf already carries an (uncapped, cruder) attack
   signal. The gradient's edge is opponent-dependent: it
   punishes engines that cannot search, and roughly ties
   engines that can.
5. **Open single-position items (no recurring class behind
   them in the loop's games):** the iteration-1 game-3
   forcing-move exploration weakness (regression xfail — the
   saving move ranked 8th, prior 0.04, starved of visits in
   100 sims) and the iter-5 regression xfail (the gradient's
   ORDERING, not its magnitude, prefers the block over the
   counterattack). Plus the §2.2 observation: a
   double-open-three CREATION is invisible to prevention
   logic, which tracks existing creators only.

**A tempering note from the swap match (§3.8).** Under swap
rules v7 beats v1 only 0.5700 with a CI that includes 0.50 —
the edge is not established there, even though the normal-rules
match is decisive (0.7050, CI lower 0.6094). The swap rule
removes the empty-board opening where v7's Black edge is
largest (0.89 as Black in normal rules, 0.52 under swap), so
much of v7's advantage over v1 lives in the opening phase. The
bar's clause 1 is the normal-rules match and it passes; but a
version that only wins from the empty board has not yet shown
it outplays v1 from a balanced start. The next batch should
treat the swap number as a second target, not a curiosity.

**What the next version/batch should try.** In priority order:
(a) the §2.6 root-decision change (q or q/visit blend, or
visit-ties broken by q) — the cheapest verified fix in the log;
(b) a worked game on the as-Black inversion vs v0, to find
whether there is a second structural weakness behind the 13
Black losses; (c) a v1-kernel gradient — the current gradient
ports v0's contiguous-run scoring and is blind to broken fours,
so it undervalues broken-four setups relative to v1's 5-window
scoring (a documented follow-up from §1); (d) gradient signal
in the -0.25 leaf: §1's pre-eval observation showed the breadth
information is discarded exactly where it matters most (opponent
holds a winning move -> flat -0.25, even with six creators
behind a single four) — a capped breadth term there is a
candidate, kept below the terminal floor per this iteration's
lesson; (e) White-specific attack creation — the persistent
weakness across all seven versions, and the likeliest place H2's
learned value function earns its keep. NOT to retry: the
one-sided gradient, refuted in §2.1 (it also prefers the losing
move in all 5 critical positions, and it saturates).

**A note on the fast loop itself.** Ten-game batches found the
dominant weakness fast (iteration 1: all five White losses,
one structural class, one patch) and the stopping rule worked
(iteration 2: no new patchable class, loop stops). But the
10-game batches never surfaced the as-Black weakness against
v0 (5/5 as Black in both iterations; 13/50 as-Black losses only
appear at 100-game scale). Fast loops find the dominant
weakness; the full battery still finds the sub-dominant ones.
Both are needed, in that order.

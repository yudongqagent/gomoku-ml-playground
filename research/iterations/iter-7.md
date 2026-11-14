# Iteration 7 — v7: MCTS leaf with an attack gradient (batch 3, H1)

**Status: pre-registered — implementation in progress, no evals run
yet.** Comparison target and eval seeds recorded BEFORE the first
eval (see §1). This is the first version of batch 3 (max 3
versions: v7, v8, v9).

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

## 2. Loss analysis for iteration 7

(TODO — filled in after the evals.)

## 3. v7 evaluation

(TODO — filled in after the evals.)

### 3.1 Smoke (20 games vs v1, seeds 100/200)

### 3.2 Tactical suites

### 3.3 Head-to-head vs v1 (100 games, seeds 100/200)

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200)

### 3.5 Ladder (20 games/opponent, mcts_sims=100, seed 100)

### 3.6 Swap match vs v1 (100 games, balanced, placement_seed 11, tempo judge)

## 4. Bar check (v7 vs v1)

(TODO — filled in after the evals.)

## 5. What this means

(TODO — filled in after the evals.)

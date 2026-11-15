# Iteration 13 — v13: White focus (batch 5, H2)

**Status: PRE-REGISTERED — sweep grid, opponents, game counts, seeds
and the decision rule recorded BEFORE any eval (see §1).** Comparison
target is v7 (the batch-5 reference). This is the H2 workstream of
batch 5 (H1 is v12, the tie-prior default + forcing-first tie-break,
a separate workstream on `research/batch-5-h1`; H3 is v14, the
v1-kernel attack gradient, on `research/batch-5-h3`); work happens
on branch `research/batch-5-h2` in the `gomoku-b5-h2` worktree.
v13 does NOT change the root decision rule (H1's work) and does NOT
touch the leaf (the leaf is not the binding constraint — iter-11;
H3 owns the gradient).

## 1. Design and pre-registration

**Comparison target: v7** (primary, the batch-5 reference: ladder
Elo 190.8, vs v1 0.7050, on master at `bf866f7`). The bar's clause
1 is head-to-head vs v7 (normal rules, 100 games, alternating
colours, win rate > 0.50 with Wilson 95% CI lower bound > 0.50),
clause 2 is ladder Elo strictly higher than v7's 190.8 (identical
20 games/opponent, mcts_sims=100), clause 3 is tactical pass rate
>= v7's (old suite 20/20, hard suite 29/29), clause 4 is all test
suites green. **Secondary head-to-heads, also pre-registered:**
v13 vs v0 (100 games) and a 100-game swap match vs v7 under the
tempo judge (the harness default).

**The work has two steps, in order (per the batch-5 plan and the
H2 brief):**

1. **A White sims sweep (50/100/150/200)** to map the
   non-monotonicity v10 found — 200 sims as White made White the
   worst in program history (0W-50L vs v0) while Black was fine.
   Sims changes need a sweep, never an assumption.
2. **White-specific priors or a White opening book** at the best
   sims count from the sweep. The v13 design is pre-registered in
   §1b AFTER the sweep results are in and BEFORE any v13 code is
   written — the sweep informs the design, so the design cannot be
   honestly pre-registered before it. Nothing in step 2 is decided
   here.

### 1.1 The sweep engine (step 1)

The sweep engine is `WhiteMCTS` ported EXACTLY from
`research/batch-4-h2` (`ml/gomoku_ml/mcts_white.py`, the v10
module — v7's PUCT MCTS with the capped attack-gradient leaf,
plus v3's 2-ply threat check at the root, plus colour-asymmetric
sims). The port is byte-identical to the batch-4-h2 file; the
only thing that varies across sweep cells is the `white_sims`
constructor argument. Black sims are FIXED at 100 (v7's count)
in every cell; the threat check is ON in every cell.

**Adopting the threat check as part of v13's base is
pre-registered here.** Iter-10 §3.3's three-way probe
exonerated it as the cause of v10's White collapse (when Black
has creators all three variants play the same move; the
sims increase, not the check, changes White's play), and
iter-8 §7 finding 3 calls it a durable portable gain (fixed
v8's hard tactical 24/29 -> 29/29; defence airtight). It is
colour-symmetric, so it cannot by itself fix a White-specific
gap — the core hypothesis remains White-specific (step 2).

The `white_sims=100` cell is the control: with symmetric 100
sims it is v7 + the threat check, isolating the check's effect
against both opponents.

### 1.2 Sweep grid, opponents, game counts, seeds

| Cell | White sims | Black sims | Opponent | Games |
|---|---|---|---|---|
| S50-v0 | 50 | 100 | v0 | 40 |
| S100-v0 | 100 | 100 | v0 | 40 |
| S150-v0 | 150 | 100 | v0 | 40 |
| S200-v0 | 200 | 100 | v0 | 40 |
| S50-v7 | 50 | 100 | v7 | 40 |
| S100-v7 | 100 | 100 | v7 | 40 |
| S150-v7 | 150 | 100 | v7 | 40 |
| S200-v7 | 200 | 100 | v7 | 40 |

- **Opponent v0 is PRIMARY.** v0 exposed v10's collapse most
  starkly (v10-White 0W-50L at 200 sims; v7-White 30W-18L-2D,
  0.62, at 100 sims), v0 games are fast (the 1-ply opponent is
  cheap), and the v0 sweep maps the shape of the
  non-monotonicity at the lowest cost.
- **Opponent v7 is the transfer check.** The bar is vs v7, so
  the sims count must also be sane against the reference — but
  20 White games per cell is noisy (Wilson half-width ~±0.20),
  so the v7 cells confirm the shape, they do not re-rank close
  counts on their own.
- **40 games per cell** (20 as White, 20 as Black, alternating
  colours, empty-board starts, normal rules). The v10 effect
  was enormous (0.62 -> 0.00 vs v0); the sweep maps gross
  non-monotonicity, not fine differences, and 40 games/cell is
  the right scale for that. The 100-game battery on v13 — not
  the sweep — is the verdict.
- **Sweep seeds: 300/400** (sweep engine / opponent),
  documented here and distinct from the bar battery's 100/200
  and the fast-loop dev seeds (1001/2001, ...). Same seeds in
  every cell, so cells differ ONLY in the White sims count.
- **Per-colour splits are MANDATORY in every sweep cell.**
  White is the target; the overall match score alone cannot
  show whether a sims count fixed White (v10's aggregate 0.39
  vs v0 hid a 0.78/0.00 split).

**Decision rule (pre-registered, applied mechanically to the
sweep results):** choose the White sims count with the highest
as-White win rate vs v0. Ties or near-ties (within 0.10 on the
White-vs-v0 rate) are broken by the as-White win rate vs v7;
a remaining tie is broken by the lower sims count (cheaper,
and v10 showed more search is not monotonically better). If
100 is the best or tied-best count, v13 keeps 100 White sims
and the White gain must come from step 2 alone. The chosen
count is recorded in §1b with the sweep table; no count may be
chosen by any other reasoning.

**Cost estimate (pre-registered, from v10's measured
timings).** v10's 100-game vs-v0 match took 2042 s (~20
s/game); the v0 sweep cells should take ~15 min each, ~1 h
for all four. v10's 100-game vs-v7 match took 6176 s (~62
s/game); the v7 cells should take ~30-45 min each under
three-way VM contention, ~2-3 h for all four. Total sweep
~3-4 h, run sequentially in the background (v0 cells first —
fast and most informative), one match at a time. Matches
flush after every game, so a VM replacement loses at most the
in-progress game; a re-run with the same seeds is
deterministic.

### 1.3 Fast loop (the finding method, per the batch-5 plan)

Runs on v13 AFTER step 2's design is pre-registered (§1b) and
built. Opponent is v7 (the reference — losses against it are
the ones that matter for the bar); development seeds
1001/2001, 1002/2002, ... distinct from the bar battery's
100/200 and the sweep's 300/400. If an iteration produces no
losses against v7, switch the probe to v0 (fast; v0's crude
play exposed v7's as-Black inversion and produces White
losses). Max 4 iterations (40 games), stop when an iteration
finds no new patchable weakness class. Each iteration: play 10
games, read every move, find each game's losing move
(solver/deeper-search verified, never guessed), patch the
structural cause, record the critical position + correct move
as a regression test (`ml/gomoku_ml/regression_positions_v13.json`
+ `ml/tests/test_regression_v13.py`).

**Standing constraints on patches (batch-5 plan):** q-based
root decisions are CLOSED (pure q: 0.20 vs v7, tactical
22/29) — do not revisit. Any q-based analysis must use the
CORRECTLY SIGNED q: `child.q` is the OPPONENT's value
(`_select_child` exploits `-child.q`); our-perspective q is
`-child.q` (iter-7 §2.6's "q prefers the saving move" was
sign-inverted; correctly signed, q prefers the LOSING move).
The leaf is not the binding constraint — no leaf work. Sims
changes need a sweep, never an assumption (this sweep is that
sweep; a further sims change after v13 is built needs its own
justification, not a quiet edit).

### 1.4 Battery (the verdict — pre-registered seeds and commands)

On the FINAL v13 only (after the fast loop):

- 100 games vs v7, seeds 100/200 (clause 1).
- 100 games vs v0, seeds 100/200.
- 100-game swap vs v7, balanced placement, placement_seed 11,
  tempo judge (the harness default), seeds 100/200.
- Ladder 20 games/opponent (random, heuristic, mcts;
  mcts_sims=100), seed 100 (clause 2).
- Tactical old 20/20 + hard 29/29, full-dispatch driver
  (`ml/scripts/eval_tactical_full.py`, `player.choose_move` —
  NOT `eval_version.py`'s probe, which bypasses the MCTS
  dispatch) (clause 3).
- ml pytest + npm test + tsc green (clause 4).

**Integration.** New module `ml/gomoku_ml/mcts_white.py`
(ported from batch-4-h2, see §1.1) plus the v13 module (named
in §1b when the design is pre-registered); `mcts_heuristic.py`
is NOT edited (parallel H1/H3 workstreams own the root
decision and the leaf). `ml/gomoku_ml/scoring.py` gains ONLY
the `ENGINE_VERSIONS[13]` registry entry and a minimal
dispatch hook. `baseline.py` is frozen and untouched.

## 1b. v13 design (pre-registered AFTER the sweep — WRITTEN 2026-11-14)

**Base (decided by the §1.2 rule, §2.1):** v7's PUCT MCTS
(capped attack-gradient leaf) + v3's 2-ply threat check at the
root, 100 sims in both colours. The sweep proved 100 is the
White optimum (0.775 vs v0; 50 -> 0.525, 150 -> 0.55,
200 -> 0.00), so the sims lever is exhausted and v13's White
gain must come from the priors.

**The White-specific change: a White-only prior temperature.**
`_priors` softmaxes `log1p(v1 score) / 2.0` over all
candidates; the /2.0 temperature gives a ~7x prior ratio
between successive pattern classes, so a 200k move (an
open-four maker or its block) takes a huge prior share. The
fast loop (§2.2) diagnosed the mechanism on White's losses:
in g9 ply 5 the losing block (6,7) holds prior 0.34 (38 of
100 visits) while the oracle-verified saves (9,7)/(4,7) hold
0.108 each; in g7 ply 25 the two 20k four-makers take 0.087
each (10 visits each) while the save (8,9) takes 0.04 (4
visits). White's saving moves are quiet counterplay the v1
scorer ranks below the big single-pattern moves, so the
prior starves them of visits. v13 flattens the prior ONLY
when the engine plays White: temperature `T_w` (Black keeps
2.0), applied to every node of the search when the engine is
White. One knob, White-specific, falsifiable.

**Decision procedure (pre-registered, mechanical):**

1. **Position probe (NOT an eval).** Run the temperature
   variants on the 7 White-to-move diagnostic positions (the
   2 White critical positions from fast-loop iteration 1,
   §2.2 — g7-ply25, g9-ply15 — + the 5 White v10 regression
   positions) at T_w in {2.0, 3.0, 4.0} and count
   oracle-verified saves played. Proceed to step 2 ONLY if
   some T_w > 2.0 plays strictly more saves than the
   T_w = 2.0 control. If no temperature beats the control on
   the probe, the temperature hypothesis is refuted on
   arrival — no match sweep, no temperature in v13.
2. **Temperature sweep (if the probe passes).** T_w in
   {2.0, 3.0, 4.0} vs v0, 40 games each (20 as White),
   sweep seeds 300/400, same harness as §1.2. Pick the
   T_w with the highest as-White win rate vs v0; ties or
   near-ties (within 0.10) broken by a 40-game match vs v7
   at the tied temperatures; a remaining tie broken by the
   lower temperature. If no T_w beats the T_w = 2.0 control,
   v13 ships with T_w = 2.0 (i.e. the base).
3. **Fast loop iteration 2** on the chosen v13, then the
   §1.4 battery.

**Contingency (not pre-registered as a design — it would
need its own §1b entry if pursued):** if the temperature
hypothesis is refuted at the probe, the remaining
White-specific direction is a White opening book derived
from probe matches (play N games from each candidate White
reply to Black's centre opening, book the best) — NOT a
book hand-written from one fast-loop game (the g9 ply-5
oracle verdict disagrees with actual engine play from the
same position: g3 won from the identical opening, so a
(6,7) ban on that evidence alone would be overfitting to
the oracle proxy).

**What this design does NOT do:** no q-based root decision
(CLOSED), no leaf change (H3's work; the leaf is not the
binding constraint), no sims change beyond the sweep's
verdict, no net value.

**Mechanism check (pre-registered, NOT an eval):** at the
chosen T_w, v13-as-White should play the saving move on
the g9-ply-5 opening position (9,7) or (4,7) instead of the
losing block (6,7). If it does not, the temperature does
not fix the known opening pathology and the design's
mechanism is in doubt before any match is played — recorded
as an xfail, following iter-10's precedent of not amending
a pre-registered design on one diagnostic position (the
sweep + battery decide).

## 2. Sweep and loss analysis

### 2.1 White sims sweep

**vs v0 (40 games/cell, seeds 300/400) — COMPLETE.** Files:
`research/iterations/sweep-s{50,100,150,200}-vs-v0.json`.

| White sims | Overall | As Black (n=20) | As White (n=20) | Time |
|---|---|---|---|---|
| 50 | 24W-15L-1D, 0.6125 | 14W-6L-0D, 0.7000 | 10W-9L-1D, 0.5250 | 761 s |
| **100** | 27W-11L-2D, 0.7000 | 12W-7L-1D, 0.6250 | **15W-4L-1D, 0.7750** | 1302 s |
| 150 | 27W-12L-1D, 0.6875 | 16W-3L-1D, 0.8250 | 11W-9L-0D, 0.5500 | 1443 s |
| 200 | 14W-26L-0D, 0.3500 | 14W-6L-0D, 0.7000 | **0W-20L-0D, 0.0000** | 942 s |

**The non-monotonicity reproduces at new seeds.** White vs v0
peaks at 100 sims (0.775) and collapses to 0.00 at 200 —
v10's 0W-50L (seeds 100/200, n=50) is not a seed artefact.
The shape is an inverted U with a cliff: 0.525 -> 0.775 ->
0.55 -> 0.00. Black is fine at every count (0.625-0.825;
Black is fixed at 100 sims in all cells, so its variation is
RNG-state noise across cells, not a sims effect) — the damage
is exactly the White sims count, as in v10. The 200-sim
White losses are fast (v10's ran 27-29 plies); the 50-sim
White losses are faster still (four at 17 plies) — at 50
sims White sometimes plays a bad opening reply and is
overrun immediately. The 100-sim White losses run 35-41
plies. Classifier on the s100 cell: 11/11 losses via open
four, 0 missed blocks, 0 clean missed preventions in 52
threatened turns — defence solved at every sims count; the
sims count moves the ATTACK (whether White creates first),
never the defence.

**Decision (mechanical, per the §1.2 rule): 100 White sims.**
Highest as-White rate vs v0 is 0.775 at 100 sims; the gap to
the next-best count (0.55 at 150) is 0.225, well outside the
0.10 near-tie band, so no v7 tie-break is needed. v13 keeps
v7's 100 sims in both colours — the sims lever is exhausted
(100 was already the optimum; the sweep's value is proving
that, and mapping the cliff). The White gain must come from
step 2 (priors/book) alone.

**vs v7 (40 games/cell, seeds 300/400).** Files:
`research/iterations/sweep-s{50,100,150,200}-vs-v7.json`.

| White sims | Overall | As Black (n=20) | As White (n=20) | Time |
|---|---|---|---|---|
| 50 | 18W-19L-3D, 0.4875 | 13W-5L-2D, 0.7000 | 5W-14L-1D, 0.2750 | 1907 s |
| **100** | 23W-14L-3D, 0.6125 | 11W-8L-1D, 0.5750 | **12W-6L-2D, 0.6500** | 1739 s |
| 150 | — running — | | | |
| 200 | — pending — | | | |

The inverted-U shape transfers to v7: White 0.275 at 50
sims, 0.65 at 100. (The s100-vs-v7 cell was interrupted by
a runtime restart at 7 games; the re-run with the same seeds
reproduced the 7-game prefix exactly — determinism
verified.) Note the s100 cell (v7 + threat check, symmetric
100 sims) beats v7 0.6125 overall in 40 games — the threat
check alone may be worth something against v7, or it is
noise (CI [0.4581, 0.7472]); the 100-game battery decides.
The sims decision (§2.1, v0 primary) does not depend on the
remaining cells.

### 2.2 Fast-loop iteration 1 — 10 games vs v7 (v13 BASE)

**Batch:** `research/iterations/fastloop1-v13-vs-v7.json`
(seeds v13=1001 / v7=2001 — development seeds, distinct from
the bar battery's 100/200 and the sweep's 300/400).
**6W-4L-0D, win rate 0.6000, CI [0.3127, 0.8318]**, 364.7 s.
Split: as Black 3W-2L (games 0, 2 lost; 4, 6, 8 won), as
White 3W-2L (games 1, 3, 5 won; 7, 9 lost). Classifier:
4/4 losses via open four, 0 missed blocks, 0 clean missed
preventions in 10 threatened turns (losses). Defence is
still solved; the losses are lost races. All 10 games are
distinct lines. Note: the (7,5) White-reply opening line —
(7,7)(7,5)(5,7)(3,7)(8,7)(6,7) — appears in 4 of the 10
games (g0, g2, g3, g9); v13 goes 1W-3L from it, and White
wins 3 of the 4 regardless of engine.

**The losing move in each game** (walk-back,
`ml/scripts/walkback_loss.py`, corrected oracle method —
the actual move's v1-vs-v1 playout must lose at all 3 seeds
before an alternative counts; seeds 0,1,2; alternatives are
the top-6 by v1 score plus forcing moves; JSONs:
`research/iterations/walkback-fl1-g{0,2,7,9}.json`):

| Game | Last preventable turn | v13 played (loses 3/3) | Perfect move (oracle) |
|---|---|---|---|
| 0 (Black, 18 plies) | ply 12 | (11,9) | (9,7) / (7,8) — win 3/3 |
| 2 (Black, 36 plies) | ply 30 | (12,7) | (4,5) — draws/wins; (3,4)/(6,4) partial |
| 7 (White, 33 plies) | ply 25 | (5,10) | (8,9) — wins 2/3; (6,11)/(4,11) draw 3/3 |
| 9 (White, 21 plies) | ply 15 | (7,10) | (7,6) — wins 2/3 |

First losing turns (earliest turn where the actual loses
3/3 with a save): g0 ply 8 (7,9), saves incl. (3,5) wins
3/3; g2 ply 20 (7,6), saves (10,3)/(10,8) draw 3/3; g7 ply
19 (7,9), saves incl. (2,10) draws 3/3; g9 ply 5 (6,7) —
the OPENING — saves (9,7) draws 3/3, (4,7) wins 2/3.

**Critical-position analysis** (`ml/scripts/analyse_critical.py`,
100 sims, gradient leaf; q reported here is OUR-perspective
q = -raw child.q, per iter-9 §1.0 — the script reports raw
q, which is the opponent's value):

| Position | Actual (v1 rank / prior) | Actual visits / our-q | Save visits / our-q |
|---|---|---|---|
| g0 ply 12 | (11,9) rank 8, p=0.037 | 5 / -0.013 | (9,7): 4 / -0.053 |
| g2 ply 30 | (12,7) rank 11, p=0.030 | 5 / -0.040 | (4,5): 2 / -0.113 |
| g7 ply 25 | (5,10) rank 2, p=0.087 | 10 / -0.045 | (8,9): 4 / -0.104 |
| g9 ply 15 | (7,10) rank 1, p=0.148 | 24 / -0.005 | (7,6): 16 / +0.032 |
| g9 ply 5 (opening) | (6,7) rank 1, p=0.340 | 38 / -0.021 | (9,7): 10 / -0.044 |

**The dominant class is beyond-horizon value error, NOT a
decision-rule or visit-starvation problem.** In 4 of the 5
positions the search's own correctly-signed q prefers the
LOSING move (g0, g2, g7, g9-ply-5); only in g9 ply 15 does
q prefer the save (+0.032 vs -0.005) while visits override
it (24 vs 16) — and that save leaves a Black creator live,
so the threat check would filter it out anyway. The
gradient leaf gives no consistent signal (it prefers the
loser in g0/g7/g9-15, the save in g9-5, tied in g2). The
positions are quiet (0 creators) except g9 ply 15 and g9
ply 5 (1 Black creator each), so the threat check is mostly
disengaged. The White losses share one visible pattern:
**the losing move is the top-v1-score move** — a 200k
open-four maker/block (g9 ply 15: (7,10), 200090; g9 ply 5:
(6,7), 200012, prior 0.34) or a 20k four-maker (g7 ply 25:
(5,10), 20008) — while the saves are quiet counterplay
(rank 2-5, scores 4k-24k). The v1 prior concentrates the
search on the big single-pattern move; the leaf cannot see
far enough to correct it. That concentration is the
mechanism the §1b White-prior-temperature design targets.

**Caveat on the opening position (g9 ply 5).** The oracle
says (6,7) loses 3/3 — but game 3 won from the IDENTICAL
opening (same first 8 plies, diverging only at ply 8) under
actual engine play. The v1-vs-v1 oracle is a proxy, and
here it disagrees with engine play; a book entry banning
(6,7) on this evidence alone would be overfitting to the
oracle (see §1b's contingency note).

**NO PATCH from iteration 1 yet — the loop continues to
the §1b temperature probe.** Iteration 1 produced no
verified patchable class with a certain fix (the dominant
class is value error, whose fix is not established), but
it DID produce a testable White-specific mechanism (prior
concentration on big-score moves) with a one-knob
intervention (White prior temperature). Per the §1b
decision procedure, the probe decides whether the
temperature is worth a match sweep; the loop does not stop
while a pre-registered probe is pending.

**Regression tests:** `ml/gomoku_ml/regression_positions_v13.json`
(4 positions from this iteration: g0-ply12, g2-ply30,
g7-ply25, g9-ply15 — stone lists + expected saves, generated
from the game records) +
`ml/tests/test_regression_v13.py`. All 4 are documented
xfails until a patch plays the save. (g9-ply-5 is the §1b
mechanism check, tracked separately — its oracle verdict
disagrees with engine play in g3, so it is not a regression
test.) Reproduction note (verified): the unpatched base with
a fresh seed-0 player reproduces the exact losing move on
g0 (11,9) and g9-ply15 (7,10); on g2 it plays (9,10) and on
g7 (5,13) — both tied alternatives at the same visit count
in the critical analysis, still non-saves — so the base
plays a non-save on all 4 positions.

### 2.3 White prior-temperature probe (iter-13 §1b step 1)

**Probe: T=2.0: 0/7 saves; T=3.0: 1/7; T=4.0: 0/7.**
Script: `ml/scripts/probe_white_temp.py` (log:
`research/iterations/probe-white-temp.log`, gitignored).
The 7 White-to-move diagnostic positions are the 2 White
fast-loop positions (g7-ply25, g9-ply15) + the 5 White v10
regression positions. The single flip is at T=3.0 on
fl1-g9-ply19, where the engine plays the forcing save (4,9)
instead of the loser (3,10) — the temperature gives the
low-prior forcing move enough visits to win the root. At
T=4.0 the flip disappears (back to (3,10)): more flattening
is not monotonically better either. **Per the pre-registered
§1b rule the probe PASSES (1 > 0)** — thinly — so the match
sweep (step 2) runs. Honestly recorded: 1/7 is a weak pass,
and the mechanism check FAILS at all three temperatures
(g9-ply-5: (6,7) at T=2.0, 3.0 and 4.0 — the temperature
does not fix the opening pathology). Per §1b the design is
NOT amended on one diagnostic position; the sweep + battery
decide. If the sweep shows no temperature beating the
control, v13 ships with T=2.0 (the base).

### 2.4 White prior-temperature sweep (§1b step 2) — RUNNING

### 2.5 Fast-loop iteration 2 — TO BE RUN

### 2.6 Fast-loop iteration 3 — TO BE RUN

### 2.7 Fast-loop iteration 4 — TO BE RUN

### 2.8 Aggregate counts (secondary)

## 3. v13 evaluation (FULL VALIDATION BATTERY — final v13)

### 3.1 Tactical suites — TO BE RUN

### 3.2 Head-to-head vs v7 (100 games, seeds 100/200) — TO BE RUN

### 3.3 Head-to-head vs v0 (100 games, seeds 100/200) — TO BE RUN

### 3.4 Ladder (20 games/opponent, mcts_sims=100, seed 100) — TO BE RUN

### 3.5 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — TO BE RUN

## 4. Bar check (v13 vs v7) — TO BE RUN

## 5. What this means — TO BE WRITTEN

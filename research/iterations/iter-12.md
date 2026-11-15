# Iteration 12 — v12: tie-prior default + forcing-first tie-break (batch 5, H1)

**Status: IN PROGRESS — pre-registration (§1) written BEFORE any eval.
Pre-eval observation (§1.5), fast loop (§2), and full battery (§3)
to follow. Comparison target and eval seeds were recorded BEFORE
the first eval (see §1).**

**METHOD (carried over from batch 3/4).** The fast 10-game loop stays
the finding method: play 10 games, read every move, find each game's
losing move (solver/deeper-search verified, never guessed), patch the
algorithm to reproduce the perfect move in those positions (regression
test), repeat — max 4 iterations, or stop when an iteration finds no
new patchable weakness class. The 4-clause bar stays the validation
gate. Fast-loop games use development seeds (1001/2001, 1002/2002,
...) distinct from the bar battery's pre-registered seeds (100/200).
Per the §2.6 warning in iter-9, fast loops include as-Black games vs
v0, and a 10-game batch alone never concludes "no weakness".

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-5 plan). Clause 1
is head-to-head vs v7 (normal rules, >= 100 games, alternating
colours, win rate > 0.50 with Wilson 95% CI lower bound > 0.50);
clause 2 is ladder Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts; mcts_sims=100); clause 3 is
tactical pass rate >= v7's (20/20 old suite + 29/29 hard suite);
clause 4 is all test suites green (ml pytest, npm test, tsc).
**Secondary head-to-heads, also pre-registered:** v12 vs v0 (100
games) and a 100-game swap match vs v7 under the tempo judge (the
harness default). Non-transitivity is recorded prominently.

**The improvement: v12 = v7 with ONLY the root move-decision rule
changed.** Everything else is v7 unchanged: v6's PUCT MCTS (100 sims,
c_puct 1.5, v1 priors over all candidates, win/block scans before
search) + the capped (+/-0.2) attack-gradient leaf in quiet
positions. The ONLY change is how the root child is picked after the
100 sims — and it is v9's rule (tie-prior) plus one more tie-break
level.

### 1.0 The failure mode (iter-9 §2.6, read before the rule)

v9's tie-prior rule has a known failure mode, worked in iter-9
§2.6 (swap game 41, ply 21, v9 as White): a **three-way tie at 5
visits** between (2,12), (3,11) and (7,8). The preventing moves
(3,11) and (2,12) are FORCING — they create a White four, so Black
must answer instead of playing the creator — but v1's 1-ply score
undervalues forcing counter-threats (~20k: (3,11) 20167, (2,12)
20010) against occupying a creator (~204k: (7,8) 204024, rank 1).
So the priors are (7,8) 0.1244 vs (3,11) 0.0391 / (2,12) 0.0390,
and **the tie-prior rule picks the loser (7,8)**. With two
diagonal-adjacent creators ((6,7) and (7,8)), blocking one is
useless — Black plays the other and creates the open four.

**The honest limit, stated in advance (iter-9 §2.6): ply 21 does
NOT verify as a single losing move.** Match-engine playouts from
ply 21 (v9-as-White vs v7-as-Black, seeds 0,1,2) show the
preventing moves do NOT save the game either: actual (7,8) loses
3/3, (3,11) loses 3/3, (2,12) loses 3/3. The 1-ply prevention
only defuses the creator for one move; the position was already
losing. **The forcing-first rule is therefore a hypothesis to
test at match scale, NOT a verified fix — and the position is NOT
added as a regression test asserting a save** (there is no
verified saving move to assert). The same structural class
accounts for all 4 flagged swap preventions in v9's battery
(games 41, 43/83 identical, 55 — three distinct positions, one
class), so if the rule helps anywhere, it helps in this class;
but §2.6's own evidence says even the right tie-break doesn't
save game 41, so the expected gain is small. The battery decides.

### 1.1 The rule (pre-registered, exact)

**v12 root decision — `"tie-prior-forcing"`:**

1. Most-visited root child (as v6/v7/v9). Only visited children
   (visit_count >= 1) are considered.
2. Among children tied at max visits: prefer FORCING moves —
   moves for which `scoring._is_forcing(board, move, player)` is
   true, i.e. the move wins immediately or leaves the mover with
   >= 1 winning move (creates a four). Immediate wins are
   handled by the pre-search win scan, so in practice this means
   "creates a four". If exactly one tied child is forcing, play
   it. If several tied children are forcing, go to step 3 among
   the forcing subset. If NONE of the tied children is forcing,
   go to step 3 among all tied children.
3. Among the remaining tied children, break by the v1 prior
   (highest prior wins) — v9's tie-prior rule.
4. Final tie: the seeded RNG's choice, or the first move when
   there is no RNG (v6/v7 behaviour).

**q is NOT used anywhere in this rule.** q-based root decisions
are CLOSED at this sim count (iter-9 §5: pure q 0.20 vs v7,
tactical 22/29; tie-q 0.45). The rule uses only visits, a
board-derived forcing predicate, and the v1 prior — all
deterministic signals. (Correctly-signed our-q DID prefer the
preventing moves in the §2.6 position, −0.2044 vs −0.4215 — but
q's match-level record as a decision signal is catastrophic, so
the forcing predicate, not q, is the tie-break that captures
that signal.)

**Considered and NOT pre-registered — the conditional variant:**
break ties by forcing-ness ONLY when the opponent has >= 1 live
open-four creator on the current board (i.e. only when a
prevention is actually at stake), else plain tie-prior. Rejected
as the v12 rule for three reasons: (a) a forcing move is strong
in general — it compels a reply and threatens a win next move —
not only as prevention, so the unconditional rule is the
simpler hypothesis; (b) v12 must be a single minimal change from
v9 so the battery isolates the forcing tie-break's effect; (c)
the conditional variant adds a creator scan and a second branch
to test. If the fast loop shows forcing-first hurting in quiet
positions (no opponent threat), the conditional variant is the
documented fallback for a future version — it is NOT smuggled
into v12 after the fact.

**No variant smoke.** Unlike v9 (four pre-registered variants,
smoked 20 games each), v12 is a SINGLE pre-registered hypothesis:
tie-prior + forcing-first. There is nothing to select between,
and a 20-game smoke between v12 and v9 would be noise (batch-4
lesson: small-sample smokes mislead; v11's sweep 0.575 and fast
loop 0.85 vs the 100-game 0.435). The fast loop is the FINDING
method (loss analysis, patches), not a selection method. The
100-game battery is the verdict.

Implementation: `HeuristicMCTS(root_decision=...)` gains the
value `"tie-prior-forcing"`; `_choose_root_child` takes the
board and player (optional — when absent, e.g. synthetic-root
unit tests, the forcing filter is skipped and the rule degrades
to plain tie-prior); `choose_move` passes them. Wired through
`ENGINE_VERSIONS` via the existing `"root_decision"` key
(default `"visits"` when absent, so v6/v7 are untouched). v12 is
registered in `scoring.py` with `leaf_mode="gradient"` and
`root_decision="tie-prior-forcing"`.

### 1.2 v9 code provenance (not re-derived)

v9's implementation lives on branch `research/batch-4-h1`, NOT
on master or on `research/batch-5`. It was ported, not
re-derived: commit `97cbc26` ("implement v9 root-decision rules
in HeuristicMCTS") cherry-picked cleanly onto
`research/batch-5-h1`, plus the v9 regression files and the v9
`ENGINE_VERSIONS` entry checked out from `research/batch-4-h1`
(`ml/tests/test_regression_v9.py`,
`ml/gomoku_ml/regression_positions_v9.json`,
`ml/gomoku_ml/scoring.py`). The cherry-picked code passes the
targeted suites before any v12 work is built on it
(`test_mcts_heuristic.py` + `test_regression_v9.py`: 35 passed,
3 xfailed).

### 1.3 The §2.6 diagnostic position (pre-registered, not the verdict)

batch-v9-vs-v7-swap.json game 41, replayed to ply 21 (White to
move; v9 played the losing (7,8)). Root stats at ply 21 (100
sims, gradient leaf): three-way tie at 5 visits between (2,12),
(3,11), (7,8); (3,11) and (2,12) forcing, (7,8) not forcing.
**Expected: v12 plays a forcing move** — the rule is designed
for exactly this tie. Between the two forcing moves the prior
decides: (3,11) 20167 > (2,12) 20010, so (3,11) is the expected
play if the priors separate them. v12's move here is recorded
as a pre-eval observation (§1.5) after implementation. **It is
NOT a regression test**: §1.0's verification says neither
preventing move saves the game, so asserting a save would be
asserting a falsehood. The observation tests only that the rule
does what it claims (prefers forcing among ties), not that the
move wins.

### 1.4 Eval seeds (pre-registered, battery)

h2h vs v7: v12=100 / v7=200; h2h vs v0: v12=100 / v0=200; ladder
seed 100, 20 games/opponent, mcts_sims=100; swap 100 games vs v7,
balanced placement, placement_seed 11, tempo judge (the harness
default; recorded in the JSON). Harness commands follow iter-7
§3 exactly (`selfplay_batch.py`, `eval_version.py` for the
ladder, `eval_tactical_full.py` for tactical). Fast-loop dev
seeds: iter 1 vs v0 1001/2001, iter 2 vs v7 1002/2002, iter 3
1003/2003, iter 4 1004/2004 — all distinct from the battery's
100/200.

### 1.5 Pre-eval observation (recorded AFTER implementation, BEFORE any eval)

Diagnostic only — the design stands as pre-registered; the evals
decide. Driver: `ml/scripts/diag_v12_forcing.py` (the v9 swap JSON
lives on `research/batch-4-h1`; extracted to /tmp for the diag).

**§2.6 position** (swap game 41, ply 21, White to move) — the
root stats reproduce iter-9 §2.6's numbers exactly: (3,11) v=5,
our-q −0.204355, prior 0.039113, v1 score 20167, forcing=True;
(2,12) v=5, our-q −0.204355, prior 0.038961, v1 score 20010,
forcing=True; (7,8) v=5, our-q −0.421490, prior 0.124404, v1
score 204024, forcing=False. The three-way tie at 5 visits is
confirmed, and the forcing predicate separates exactly the two
preventing moves.

| rule | plays at ply 21 | verdict |
|---|---|---|
| visits (v7, no RNG) | (7,8) | PLAYED — the move v9/v7 played in the match |
| tie-prior (v9, no RNG) | (7,8) | PLAYED — the §2.6 failure mode, reproduced |
| tie-prior-forcing (v12, no RNG) | (3,11) | FORCING — the rule does what it claims: among the tied triple, the forcing subset {(3,11), (2,12)} wins, and the prior picks (3,11) (20167 > 20010) |

**v9 regression positions (3):** visits 1/3, tie-prior 1/3,
tie-prior-forcing 1/3 — the SAME pattern for all three rules (the
2 failures are the known xfails fl2-g2-ply22, fl2-g6-ply20 from
iter-9 §2.2). The forcing tie-break changes nothing on v9's
regression positions.

**Tactical suites (full-dispatch driver, same driver/seeds/suites
as the battery — so these ARE the clause-3 numbers for v12 unless
the fast loop patches the engine):** old **20/20**, hard **29/29**
(prevention 16/16, setup 7/7, winning 2/2, vcf 4/4). The
forcing-first tie-break costs nothing tactically: no tactical
position is decided by an exact visit tie in a way the forcing
filter changes. **Clause 3 holds at pre-eval.**

**Full ml test suite (on the implemented v12): 254 passed, 7
skipped, 4 xfailed, 0 failures** (417.6 s). The 7 skips are the
known missing-gitignored-checkpoint skips in a fresh worktree
(v8/v11 nets live only in the batch-3/4 worktrees).

## 2. Loss analysis for iteration 12

[Fast-loop worked games go here — see the method note at the top.]

### 2.1 Fast-loop iteration 1 — 10 games vs v0

**Batch:** `research/iterations/fast-v12-vs-v0-iter1.json`
(seeds v12=1001 / v0=2001 — development seeds, distinct from the bar
battery's 100/200). **10W-0L-0D, win rate 1.0000, CI [0.7225,
1.0000]**, 282.4 s. Per colour: as Black 5W-0L-0D, as White
5W-0L-0D. Mean game length 45.1 plies. **No losses — nothing to
patch from this iteration.**

Comparison on identical seeds: v9's iter 1 (same seeds 1001/2001)
was 8W-2L, both losses as Black in 124-ply games (iter-9 §2.1,
distributed drift, no verified single losing move). v12 wins both
of those games — game 0 in 89 plies, game 2 in 33 plies. The
forcing tie-break changes v9's two long as-Black losses into wins
at this scale. n=10 is weak evidence and the 100-game battery vs
v0 is the real measure, but the direction is the one the rule was
designed for: the as-Black long-game weakness class that v9's
loop could not patch is where the forcing-first tie-break shows
up first.

### 2.2 Fast-loop iteration 2 — 10 games vs v7

**Batch:** `research/iterations/fast-v12-vs-v7-iter2.json`
(seeds v12=1002 / v7=2002). **7W-3L-0D, win rate 0.7000, CI
[0.3968, 0.8922]**, 461.5 s. Per colour: as Black 4W-1L-0D, as
White 3W-2L-0D. (v9's iter 2 on the same seeds: 4W-4L-2D, 0.50.)
Classifier: 3/3 losses via open four, 0 missed blocks, 0 clean
missed preventions in 13 threatened turns (losses). Defence is
still perfect. All three losses have a verified losing move
(v1-vs-v1 oracle walk-back, seeds 0,1,2 — see
`~/workspace/b5h1-logs/diag_v12_positions.py` for the
full-precision root stats behind this section). None yields a
patch — see the end of this subsection.

**Game 2, ply 20 (Black to move; v12 played (8,10), loses 3/3;
(5,6)/(3,6) draw 3/3, (7,9) saves at 1 seed).** Quiet position
(no creators either side). `analyse_critical.py`: (8,10) is v1
rank 2 (20098), FORCING, gradient 0.000, **12 visits (the unique
max)**, our-q +0.020. (5,6) is v1 rank 4 (8102), not forcing,
creates 1 creator, gradient +0.049 for Black, 6 visits, our-q
−0.001. (7,9) is v1 rank 1 (20105), forcing, 9 visits, our-q
−0.012. **There is no tie — the actual move wins visits
outright (12 vs 9/6)** — so no root-decision rule in the
pre-registered set fixes it. The saving moves are starved of
visits by their lower priors (0.045 vs 0.072) despite the
gradient preferring (5,6). This is the SAME exploration class as
iter-9 §2.2 game 2 and iter-7's fl1-g3 xfail — a known open
weakness class, not a new one. Note the inversion vs §1.0: here
the LOSING move is forcing and the saving moves are quiet, so
the forcing-first tie-break is not even the wrong signal — it
simply does not apply (no tie).

**Game 1, ply 53 (White to move; v12 played (5,11), loses 3/3;
(1,6) draws 3/3, (1,10) saves at 2 seeds).** Quiet position.
(5,11) is v1 rank 1 (20163), FORCING, gradient +0.025, 7 visits
(tied max with (6,12)), our-q +0.004. (1,6) is v1 rank 6 (4015),
not forcing, gradient +0.072 for White, **3 visits**, our-q
−0.016. The tie at 7 visits is between (5,11) and (6,12) —
BOTH forcing, BOTH losing (walk-back: (6,12) loses 3/3) — so the
forcing filter cannot discriminate and the prior picks (5,11)
(0.0672 vs 0.0670). The saving moves sit at 3 visits vs 7: the
same exploration starvation as game 2. Again the gradient
prefers the save (+0.072 vs +0.025) but visits decide.

**Game 5, ply 37 (White to move; v12 played (12,6), loses 3/3;
all 5 alternatives save).** Quiet position. (12,6) is v1 rank 1
(24016), not forcing, gradient +0.127, **8 visits (tied max
with (12,4))**, our-q +0.013. (12,4) is v1 rank 2 (24015), not
forcing, gradient untested, 8 visits, our-q +0.015 — **a
saving move (draws 3/3) tied at max visits with the loser, and
the two are separated by ONE v1 point (24016 vs 24015)**. The
priors are equal to 4 decimals (0.0633); at full precision the
prior is a monotone transform of log1p(score), so (12,6)'s
prior is microscopically higher and tie-prior (and
tie-prior-forcing — neither move is forcing) picks the loser.
`visits` with no RNG plays (12,4) by dict order — a coin flip
that happened to land right. Correctly-signed our-q prefers the
saver (+0.015 vs +0.013) but the 0.002 gap is noise-level, and
re-introducing q on one position contradicts batch 4's
match-level verdict (tie-q smoke 0.45; q-based root decisions
are closed). Under the TERMINAL leaf, (12,4) wins outright (9
visits vs 5) — v6 plays the saving move — so this is also a
gradient-ordering knife-edge, the same mechanism as iter-9 §2.2
game 6. The strongest save, (10,8) (W/D/W), sits at 2 visits
(prior 0.027 vs 0.063) — exploration starvation again.

**No patch from iteration 2; loop stops here (stopping rule).**
The three worked losses share one mechanism — the saving move is
starved of visits by a low prior (game 2: 6 vs 12; game 1: 3 vs
7; game 5: 2 vs 8 for the strongest save) — which is the KNOWN
exploration class from iter-7/iter-9, not a new patchable class,
and not a root-decision fix: a fix would be exploration-side
(higher c_puct, a prior floor), a search change beyond the H1
root-decision brief needing its own pre-registration as a new
version. Game 5's tie is the only root-decision-relevant loss,
and it is a 1-point knife-edge no tie-break signal in the
pre-registered set separates. **The forcing-first rule itself is
not implicated in any loss**: in games 2 and 1 the losing moves
are forcing and win on visits (the tie-break never applies); in
game 5 neither tied move is forcing. All three critical
positions are recorded as xfail regression tests
(`regression_positions_v12.json`: fl2-g2-ply20, fl2-g1-ply53,
fl2-g5-ply37) — documented open weaknesses for the next batch,
not silently dropped. The battery runs on the as-registered v12
(tie-prior-forcing, no fast-loop patches).

### 2.3 Fast-loop iteration 3 — NOT RUN (loop stopped after iteration 2)

### 2.4 Fast-loop iteration 4 — NOT RUN (loop stopped after iteration 2)

### 2.5 Aggregate counts (secondary)

[To be filled after the battery.]

### 2.6 Worked game

[To be filled if a worked game earns it.]

## 3. v12 evaluation

### 3.1 Smoke

No variant smoke (single pre-registered hypothesis — see §1.1).
The fast loop (§2) is the finding method; the battery is the
verdict.

### 3.2 Tactical suites (diagnostic)

Measured in §1.5 (full-dispatch driver, same as the battery).

### 3.3 Fast-loop summary (final engine)

Two iterations, no patches — the battery runs on the
as-registered v12 (tie-prior-forcing). Iteration 1 vs v0:
10W-0L-0D, no losses (v9's iter 1 on the same seeds was 8W-2L;
v12 wins both games v9 lost as Black). Iteration 2 vs v7:
7W-3L-0D (v9's iter 2 on the same seeds: 4W-4L-2D); all three
losses worked (verified losing moves at g2 ply 20, g1 ply 53,
g5 ply 37) — all share the known exploration-starvation class
(saving moves starved by low priors), none a root-decision fix,
all three recorded as xfail regressions. Loop stopped per the
stopping rule (no new patchable weakness class). Details in §2.
The fast-loop numbers (10-0, 7-3) are encouraging but n=10 with
wide CIs — the 100-game battery is the verdict.

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

**v12 beats v0 in all 100 games: 100W-0L-0D, win rate 1.0000,
95% CI [0.9630, 1.0000]**, 1957.1 s. File:
`research/iterations/batch-v12-vs-v0.json` (seeds 100/200).
Per colour: as Black **50W-0L-0D (1.0000)**, as White
**50W-0L-0D (1.0000)**. Mean game length 41.64 plies. Loss
classifier: no losses to classify — defence is perfect by
construction (0 losses).

Compare on the SAME seeds: v9 93W-7L-0D (0.9300; as Black
43W-7L-0D, as White 50W-0L-0D), v7 64W-31L-5D (0.6650; as Black
34W-13L-3D, as White 30W-18L-2D). v12 gains 7 games over v9 and
36 over v7 against v0 — and the gain is exactly where v9's
weakness was: v9's 7 losses were ALL as Black (long endgames,
iter-9 §2.1); v12 as Black is 50-0. The as-Black long-game
weakness class that v9's fast loop could not patch (distributed
drift, no verified single losing move) is closed at 100-game
scale by the forcing-first tie-break — the same signal the
fast-loop iter 1 gave on 10 games (v12 10-0 where v9 was 8-2).
This is NOT the bar (clause 1 is vs v7), but it is the
strongest evidence that the forcing tie-break is a real
improvement, not noise: a perfect 100-0 on identical seeds,
against the opponent that exposed v7's and v9's Black-game
weakness, cannot be a coin flip. The colour-split inversion is
gone: v12 scores the same (1.0000) in both colours against v0.

### 3.5 Tactical suites — FINAL BATTERY

[To be filled.]

### 3.6 Head-to-head vs v7 (100 games, seeds 100/200) — FINAL BATTERY

[To be filled.]

### 3.7 Ladder (20 games/opponent, mcts_sims=100, seed 100) — FINAL BATTERY

[To be filled.]

### 3.8 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — FINAL BATTERY

[To be filled.]

## 4. Bar check (v12 vs v7)

[To be filled.]

## 5. What this means

[To be filled.]

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

[To be filled after implementation: diagnostic on the §2.6
position (what v12 plays at swap game 41 ply 21) + tactical
suites for v12 (full-dispatch driver, same driver/seeds/suites
as the battery — so these ARE the clause-3 numbers for v12
unless the fast loop patches the engine, in which case tactical
is re-run in §3.5).]

## 2. Loss analysis for iteration 12

[Fast-loop worked games go here — see the method note at the top.]

### 2.1 Fast-loop iteration 1 — 10 games vs v0

[To be filled.]

### 2.2 Fast-loop iteration 2 — 10 games vs v7

[To be filled.]

### 2.3 Fast-loop iteration 3 — NOT RUN (unless iterations 1–2 find a patchable class)

### 2.4 Fast-loop iteration 4 — NOT RUN (unless iterations 1–3 find a patchable class)

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

[To be filled.]

### 3.4 Head-to-head vs v0 (100 games, seeds 100/200) — FINAL BATTERY

[To be filled.]

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

# Iteration 17 — v17: the combination (batch 6, H3)

**Status: IN PROGRESS — pre-registration (§1) written BEFORE any eval.
Components ported from `research/batch-5`, not re-derived.**

**METHOD (carried over from batch 3/4/5).** The fast 10-game loop
stays the finding method: play 10 games, read every move, find
each game's losing move (solver/deeper-search verified, never
guessed), patch the algorithm to reproduce the perfect move in
those positions (regression test), repeat — max 4 iterations, or
stop when an iteration finds no new patchable weakness class.
The 4-clause bar stays the validation gate. Fast-loop games use
development seeds (1001/2001, 1002/2002, ...) distinct from the
bar battery's pre-registered seeds (100/200). The full battery
runs on the FINAL version only. The 100-game battery is the
verdict — small-sample smokes mislead.

## 1. Design and pre-registration

**Comparison target: v7** (the reference, batch-6 plan: ladder
Elo 190.8, on master at `bf866f7`). Clause 1 is head-to-head vs
v7 (normal rules, >= 100 games, alternating colours, win rate >
0.50 with Wilson 95% CI lower bound > 0.50); clause 2 is ladder
Elo strictly higher than v7's 190.8 (identical 20
games/opponent: random, heuristic, mcts, mcts_sims=100); clause
3 is tactical pass rate >= v7's (20/20 old suite + 29/29 hard
suite); clause 4 is all test suites green (ml pytest, npm test,
tsc). **Secondary head-to-heads, also pre-registered:** v17 vs
v0 (100 games) and a 100-game swap match vs v7 under the tempo
judge (the harness default). Non-transitivity is recorded
prominently.

**The improvement: v17 = v7 + tie-prior-forcing root rule +
v1-kernel gradient leaf + the 2-ply threat check.** Each
component passed 3/4 in a different way in batch 5, and each
failed a different clause:

| Component | Batch-5 version | Clauses passed | Clause failed |
|---|---|---|---|
| tie-prior-forcing root rule | v12 (iter-12) | 2, 3, 4 | 1 (0.4950 vs v7) |
| threat check at the root | v13 (iter-13) | 2, 3, 4 | 1 (0.5250 vs v7) |
| v1-kernel gradient leaf | v14 (iter-14) | 1, 3, 4 | 2 (ladder Elo -34.9) |

The hypothesis is that the components' strengths cover each
other's failures: v14's gradient supplies the head-to-head edge
v12/v13 lacked (clause 1: 0.6000, CI lower 0.5020), while v12's
root rule (ladder interval [286.6, None] — heuristic 20-0, mcts
10-10) and v13's threat check (ladder 240.8) supply the ladder
strength v14 lacked (clause 2: Elo -34.9, heuristic 9-11).
**CAUTION, stated in advance:** combination is not guaranteed.
v12 was 8 games WORSE than v9 vs v7 on identical seeds
(0.4950 vs 0.5500) while 7 games better vs v0 — a component
that helps against weak opposition can hurt against an equal
one. That is why v17 gets its own full battery, not an
assumption, and why the fast loop runs on the combined engine
before the battery.

### 1.0 The exact combination (pre-registered)

v17 is `HeuristicMCTS` with the following, and nothing else:

- **Base (v7, unchanged):** PUCT MCTS, **100 sims in both
  colours**, c_puct 1.5, v1 priors over all candidates
  (temperature 2.0 — the sharp prior is load-bearing at 100
  sims; flattening is catastrophic, iter-13 §2.4), win/block
  scans before the search.
- **Leaf (v14, ported exactly):** `leaf_mode="gradient_v1"` —
  the v1-kernel breadth gradient in quiet positions, thresholds
  strong >= 4000 / developing [80, 4000), tanh scale 7.5, cap
  ±0.2 (strictly below the smallest terminal value 0.25).
  Terminal values identical to v7's by construction. The v1
  kernel (`ml/gomoku_ml/kernel_v1.py`) is used directly, not
  re-ported (0 mismatches vs frozen + live references).
- **Root decision (v12, ported exactly):**
  `root_decision="tie-prior-forcing"` — most-visited root
  child; ties broken by forcing-ness first (moves creating a
  four, via `scoring._is_forcing`), then the v1 prior, then the
  seeded RNG / first move. **q is NOT used anywhere in this
  rule** — q-based root decisions are CLOSED (pure q
  catastrophic: 0.20). q SIGN, restated so batch 4's error does
  not recur: `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`); our-perspective q is
  `-child.q`.
- **Threat check (v13, ported exactly): INCLUDED.** v3's 2-ply
  threat check at the root, as in `WhiteMCTS.choose_move`:
  when the opponent has creators (moves leaving them >= 2
  winning moves), the root pool is restricted to safe moves
  (forcing, or no creator remains live); if no root child is
  safe, the root rule's choice on the full pool is played
  (best effort, as in v3/v8/v13). With no creators the choice
  is exactly the root rule's. See §1.1 for why it is included.

**Composition order (pre-registered):** win scan → block scan
→ 100-sim search → threat filter restricts the pool → the
tie-prior-forcing rule decides within the pool. The threat
check filters BEFORE the root rule, never after: it removes
unsafe moves from consideration, and the root rule then picks
among the safe ones. (If the filter applied after the root
rule, it could only veto the rule's pick and fall back to an
arbitrary safe move — the pre-filter composition is the one
v13's `WhiteMCTS` implements, and it is the one ported here.)

**Implementation (pre-registered).** A new module
`ml/gomoku_ml/mcts_v17.py` with `V17MCTS`, subclassing
`HeuristicMCTS` with `leaf_mode="gradient_v1"` and
`root_decision="tie-prior-forcing"`. The subclass exists
because `WhiteMCTS` cannot carry this combination: it
hard-codes `leaf_mode="gradient"` and replicates the root
decision with plain most-visited + RNG tie-break, bypassing
`_choose_root_child` — so the v12 root rule and the v14 leaf
cannot be expressed through it. `V17MCTS.choose_move`
replicates `WhiteMCTS.choose_move`'s structure verbatim (win
scan, block scan, candidate checks, search loop, creator
scan, safe-pool filter) with exactly two changes: the sims
count is 100 in both colours (no colour-asymmetric sims —
v10's 200-White-sims is refuted, iter-13 §2.1), and the
final decision line applies `_choose_root_child` to the
(filtered) root instead of plain most-visited. The filter
itself (creators via `scoring._opponent_creators`, safety
via `scoring._move_is_safe`, the `safe and len(safe) <
len(root.children)` condition, the `threat_filter_used`
counter) is ported verbatim from `WhiteMCTS`. Registered as
version 17 in `ENGINE_VERSIONS` with a `"v17"` dispatch key
in `ScoringPlayer`, following the v13 dispatch pattern.

**NOT in v17 (pre-registered exclusions):** White prior
temperature (refuted — catastrophic, iter-13 §2.4); the (6,8)
opening book (refuted out-of-sample, iter-13 §2.5);
colour-asymmetric sims (refuted — 200 White sims collapse
White to 0.00 vs v0, iter-13 §2.1); any q-based root
decision (CLOSED); any prior flattening; any leaf change
beyond the ported v14 gradient; any change to
`baseline.py` (FROZEN). No kernel re-sync (NumPy/WebGPU/
Metal) unless v17 passes the bar.

### 1.1 The threat check: included, and why

The threat check is the ONLY v13 component that survives
batch 5: the temperature was refuted (White 0.775 → 0.05 →
0.00) and the book was refuted (0.55 vs the 0.775 control),
so v13 = v7 + threat check is the component being combined.
It is included for four reasons:

1. **It is the component that passed.** v13 passed clauses 2,
   3 and 4 and failed clause 1 at 0.5250 — the same 3/4 shape
   as v12, failing only the head-to-head. The combination's
   hypothesis is that v14's gradient supplies the missing
   head-to-head edge; the threat check is not the part that
   failed.
2. **It is attack-neutral safety infrastructure, not a
   strength claim.** It never creates a move, it only removes
   unsafe ones when the opponent holds creators. Across
   v13's battery it produced 0 missed preventions in 520
   threatened turns, and it fixed v8's hard-tactical gap
   (24/29 → 29/29, iter-8 §7 finding 3). It costs nothing
   tactically (v10 and v13 both 20/20 + 29/29).
3. **Its ladder contribution is positive — and the ladder is
   the combination's main risk.** v13's ladder Elo is 240.8
   vs v7's 190.8, gained entirely from the heuristic matchup
   (0.80 vs 0.75 — two of v7's losses become draws under the
   threat check). v14's clause-2 failure was driven entirely
   by the same matchup (9-11 vs the heuristic anchor). If
   the threat check's defensive solidity transfers to the
   combination, it directly shores up v14's failure mode.
4. **It composes cleanly.** It restricts the pool before the
   root rule and never touches the leaf, so it cannot
   interact with the gradient's ordering or the root rule's
   tie-breaks except by removing moves both would agree are
   unsafe.

**Risk, acknowledged in advance:** v13 vs v0 was 0.6400,
slightly BELOW v7's 0.6650 on both colour splits — the
check's safe moves are sometimes passive blocks, and
against greedy 1-ply play passivity costs. If v17's vs-v0
or ladder numbers come in below v12's, the threat check is
the first suspect — but it is not removed mid-battery: the
pre-registered engine is the engine that is evaluated, and
a threat-check-free variant would be a new version, not a
patch.

### 1.2 Component verification (identity checks, NOT evals)

Each ported component is verified to behave identically to
its batch-5 version BEFORE the combination is evaluated.
These checks compare against batch-5 records; they are not
new evals of v17 and they do not select anything:

- **v1 kernel:** `ml/tests/test_kernel_v1.py` — 0 mismatches
  vs the frozen fixtures AND the live `scoring.score_attack`
  reference, both players, all cells (the batch-5 bar).
- **v12 root rule:** the iter-9 §2.6 diagnostic position
  (swap game 41, ply 21, White to move) — v12 must play
  (3,11), the higher-prior forcing move, as in iter-12 §1.5.
  Driver: `ml/scripts/diag_v12_forcing.py` (ported from
  batch-5-h1). Plus the ported `test_mcts_heuristic.py`
  root-decision tests.
- **v13 threat check / v14 leaf:** the ported regression
  suites (`test_regression_v12/v13/v14.py`, `test_mcts_v13.py`,
  `test_mcts_white.py`) pass with the recorded xfail counts.
- **Tactical suites (full-dispatch driver):** v12, v13 and
  v14 each 20/20 old + 29/29 hard, as recorded in iter-12
  §3.5, iter-13 §3.1, iter-14 §3.5.
- **Deterministic prefix replays:** 10-game matches at the
  batch-5 battery seeds (100/200) for v12-vs-v7, v13-vs-v7
  and v14-vs-v7, compared game-by-game (move lists and
  winners) against the first 10 games of the batch-5 match
  JSONs on `research/batch-5`. The engines are
  deterministic, so an identical port reproduces the records
  exactly; any mismatch is a port error, not noise. (10
  games each — enough to catch a port error, too few to be
  an eval; the 100-game battery is the verdict.)

### 1.3 Eval seeds (pre-registered, battery)

h2h vs v7: v17=100 / v7=200; h2h vs v0: v17=100 / v0=200;
ladder seed 100, 20 games/opponent, mcts_sims=100; swap 100
games vs v7, balanced placement, placement_seed 11, tempo
judge (the harness default; recorded in the JSON). Harness
commands follow iter-7 §3 exactly (`selfplay_batch.py`,
`eval_version.py` for the ladder, `eval_tactical_full.py`
for tactical). Fast-loop dev seeds: iter 1 vs v0 1001/2001,
iter 2 vs v7 1002/2002, iter 3 1003/2003, iter 4 1004/2004
— all distinct from the battery's 100/200.

## 2. Loss analysis for iteration 17

[Fast-loop worked games go here — see the method note at the top.]

### 2.1 Fast-loop iteration 1 — 10 games vs v0

[PENDING]

### 2.2 Fast-loop iteration 2 — NOT RUN

### 2.3 Fast-loop iteration 3 — NOT RUN

### 2.4 Fast-loop iteration 4 — NOT RUN

### 2.5 Aggregate counts (secondary)

[PENDING — filled after the battery.]

## 3. v17 evaluation

[PENDING]

## 4. Bar check (v17 vs v7)

[PENDING]

## 5. What this means

[PENDING]

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

## 1b. v13 design (pre-registered AFTER the sweep — TO BE WRITTEN)

[To be written once the sweep (§2.1) is complete: the chosen
White sims count per the §1.2 decision rule, the White-priors
or White-opening-book design, and the mechanism check. No v13
code is written before this section exists.]

## 2. Sweep and loss analysis

### 2.1 White sims sweep — TO BE RUN

### 2.2 Fast-loop iteration 1 — TO BE RUN

### 2.3 Fast-loop iteration 2 — TO BE RUN

### 2.4 Fast-loop iteration 3 — TO BE RUN

### 2.5 Fast-loop iteration 4 — TO BE RUN

### 2.6 Aggregate counts (secondary)

## 3. v13 evaluation (FULL VALIDATION BATTERY — final v13)

### 3.1 Tactical suites — TO BE RUN

### 3.2 Head-to-head vs v7 (100 games, seeds 100/200) — TO BE RUN

### 3.3 Head-to-head vs v0 (100 games, seeds 100/200) — TO BE RUN

### 3.4 Ladder (20 games/opponent, mcts_sims=100, seed 100) — TO BE RUN

### 3.5 Swap match vs v7 (100 games, balanced, placement_seed 11, tempo judge) — TO BE RUN

## 4. Bar check (v13 vs v7) — TO BE RUN

## 5. What this means — TO BE WRITTEN

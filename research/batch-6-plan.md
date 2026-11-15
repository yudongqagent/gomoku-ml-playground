# Batch 6 plan — exploration, White's value signal, and the combination

## Starting point
- Batch 5: 0 of 3 passed, but all three passed 3 of 4 clauses — each failed
  a different one. v7 (Elo 190.8) remains the reference; master is
  untouched (`bf866f7`).
- **v7 stays the reference version.** The bar is unchanged:
  1. Head-to-head vs v7, ≥100 games, alternating colours, win rate (draws =
     half) > 0.50 with Wilson 95% CI lower bound > 0.50.
  2. Ladder Elo strictly higher than v7's 190.8 at identical game
     counts/opponents (20 games/opponent: random, heuristic, mcts).
  3. Tactical pass rate ≥ v7's (20/20 old suite + 29/29 hard suite).
  4. All test suites green (ml pytest, npm test, tsc).
- Every version also plays 100 games vs v0 and a 100-game swap match vs v7
  (tempo judge); record non-transitivity prominently.
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel re-sync
  (NumPy/WebGPU/Metal) only when a version passes the bar.

## Candidates, in priority order
1. **H1 (v15) — exploration of low-prior saving moves.** The recurring
   class across all three batch-5 workstreams: saving moves starved at
   2–6 visits while the loser holds 7–12. No root-decision rule fixes a
   move the search never visits. Search-side fix: a prior floor or a
   higher c_puct — pre-register the exact mechanism (and the sweep, if
   both are tried) before any eval. Constraint: the sharp prior is
   load-bearing at 100 sims (v13's temperature probe: flattening is
   catastrophic), so the fix must preserve the prior's sharpness while
   letting low-prior moves get visited. A floor is not flattening.
2. **H2 (v16) — White attack creation / the value signal.** White's losses
   are beyond-horizon value errors where correctly-signed q prefers the
   losing move (v13 fast loop: 4/5 critical positions). Use v14's
   v1-kernel gradient as a value signal — e.g. a White-specific
   attack-creation bonus in the leaf, or the v1 gradient feeding the
   value estimate. Pre-register the exact mechanism before any eval.
   The v1 kernel (`kernel_v1.py`) is exact and reusable (0 mismatches
   vs frozen + live references) — use it directly, do not re-port it.
3. **H3 (v17) — the combination.** v7 + tie-prior + forcing-first +
   v1-kernel gradient (+ the threat check). Each component passed 3/4 in
   a different way; test whether they combine to 4/4. Caution: v12 was
   8 games worse than v9 vs v7 on identical seeds, so combination is not
   guaranteed — that is why it gets its own full battery, not an
   assumption.

Max 3 versions, each pre-registered in its iteration log BEFORE any eval.

## Standing lessons — levers now closed
- **Root-decision lever EXHAUSTED** (v9/v12 bracket it at ±10 games/100).
  Do not spend a workstream on another root-decision rule.
- **White search/prior/book levers CLOSED.** Sims optimum mapped at 100
  (50/100/150/200: vs v0 0.525/0.775/0.55/0.00); temperature flattening
  catastrophic (0.775 → 0.05 at T=3.0 → 0.00 at T=4.0); the (6,8)
  opening book failed out-of-sample (0.55 vs the 0.775 control — the
  mining signal was post-hoc noise). White work must be value-side.
- **q SIGN:** `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`). Our-perspective q is
  `-child.q`. Batch 4's sign error must not recur.
- **q-based root decisions are CLOSED** (pure q catastrophic: 0.20).
- **More search is NOT monotonically better.** Sims changes need a sweep,
  never an assumption.
- **Small-sample smokes mislead.** The 100-game battery is the verdict.
  Single-position diagnostics cannot validate a global search change
  (v13's temperature probe: 1/7 saves did not predict match strength).
- **The leaf is not the binding constraint** (v11, v14 — v14's exact
  v1-kernel fix bought a marginal h2h edge and nothing on the ladder).

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read every
  move, find each game's losing move (solver/deeper-search verified, never
  guessed), patch, regression-test, repeat. Think like a human master
  about weak points and weak strategies.
- The 4-clause bar stays the validation gate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.

## Constraints
- Branch `research/batch-6` from master (`bf866f7`). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b6/`).
- v12/v13/v14 implementations live on `research/batch-5`, not on master —
  cherry-pick or port; do not re-derive.
- No cloud GPU spend, no money, no wallets, no releases, no Colab driving,
  no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes all
  4 clauses, with the same plain-merge approach as batch 3.

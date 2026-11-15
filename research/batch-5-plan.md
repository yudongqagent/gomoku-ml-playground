# Batch 5 plan — build on v9's tie-prior, attack White, fix the gradient's blind spot

## Starting point
- Batch 4: 0 of 3 passed. v7 (Elo 190.8) remains the reference; master is
  untouched (`bf866f7`). v9 came closest (3/4): its tie-prior root rule is
  a real, cheap, durable gain.
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
1. **H1 (v12) — tie-prior default + forcing-first tie-break.** v9 (tie-prior:
   most-visited, ties broken by the v1 prior) beat v0 93-7 and went 20-0
   vs the heuristic ladder anchor, but only 0.55 vs v7 (CI includes 0.50).
   Adopt the tie-prior rule as the default root decision and add the
   forcing-first tie-break for v9's §2.6 failure mode: the v1 prior
   undervalues forcing counter-threats (~20k) vs occupying a creator
   (~204k). NOTE: v9's implementation lives on `research/batch-4`
   (merge `c01a24e`), not on master — cherry-pick or port it; do not
   re-derive it from scratch.
2. **H2 (v13) — White focus.** White is the gap in every version (v7 0.52
   vs v1, v8 0.29, v10 0.00 vs v0, v11 0.27 vs v7). First run a White sims
   sweep (50/100/150/200) to map the non-monotonicity v10 found, then
   build White-specific priors or a White opening book at the best count.
3. **H3 (v14) — v1-kernel attack gradient.** The gradient leaf uses the
   portable kernel's attack scores, which port v0's contiguous-run
   scoring and are BLIND to broken fours (XX_XX). Port the v1 5-window
   attack scorer into the kernel path so the gradient sees broken fours.

Max 3 versions, each pre-registered in its iteration log BEFORE any eval.

## Standing lessons — do not repeat batch 4's errors
- **q SIGN (batch 4's biggest finding):** in this codebase `child.q` is
  from the OPPONENT's perspective (`_select_child` exploits `-child.q`;
  `analyse_critical.py` reports raw `c.q`). Our-perspective q is
  `-child.q`. Iter-7 §2.6's "q prefers the saving move" was
  sign-inverted; correctly signed, q prefers the LOSING move. Any
  q-based analysis must use the correctly signed value, verified
  against the backup code and by exact reproduction.
- **q-based root decisions are CLOSED at this sim count.** H1 tested them
  directly: pure q is catastrophic (0.20 vs v7, tactical 22/29). Do not
  revisit.
- **More search is NOT monotonically better (v10).** 200 sims as White made
  White the worst in program history (0-50 vs v0) while Black was fine.
  Sims changes need a sweep, never an assumption.
- **Small-sample smokes mislead.** v11: sweep 0.575 and fast loop 0.85 vs
  the 100-game 0.435. v8's 0.60 smoke was a first-20 fluke. The 100-game
  battery is the verdict.
- **The leaf is not the binding constraint (v11).** Fixing the value net's
  training distribution did not fix game strength; the optimizer's curse
  persists in-distribution. Do not spend a workstream on another leaf.

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read every
  move, find each game's losing move (solver/deeper-search verified, never
  guessed), patch, regression-test, repeat. Think like a human master
  about weak points and weak strategies.
- The 4-clause bar stays the validation gate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
  Batch 4 did this and caught no silent breakage — keep it up.

## Constraints
- Branch `research/batch-5` from master (`bf866f7`). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b5/`).
- No cloud GPU spend, no money, no wallets, no releases, no Colab driving,
  no Space edits. Yudong runs nothing.
- The v11 net checkpoint (`ml/runs/b4-h3/value-net.pt`) is gitignored data
  in the batch-4 H3 worktree (copy in `~/workspace/projects/gomoku-b4/`
  for verification). A fresh clone cannot run v8/v11.
- Push the branch when done. Push to master ONLY if a version passes all
  4 clauses, with the same plain-merge approach as batch 3.

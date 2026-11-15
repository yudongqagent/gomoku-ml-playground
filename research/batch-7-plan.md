# Batch 7 plan — the rollout leaf, both colours and blended, plus the v1 leaf vs greedy play

## Starting point
- Batch 6: 0 of 3 passed. v7 (Elo 190.8) remains the reference; master is
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
1. **H1 (v18) — rollout leaf for both colours.** v16's 2-ply rollout leaf
   is the strongest single White-side component available (White 0.64 vs
   v7, 50W-0L vs v0, ladder Elo 381.7 — highest in the program). But as
   Black, v16 IS v7 exactly, and the Black mirror games ran 0.51,
   capping the combined h2h 2.5 points short of an established margin.
   Apply the rollout leaf as Black too and test whether it helps there.
   Pre-register the exact mechanism before any eval.
2. **H2 (v19) — blend the rollout terminal with the end gradient.**
   iter-16 §5's suggested direction: v16's rollout leaf compresses all
   q values toward −1.0 in creator-rich positions, and its one tactical
   failure (`prev-v2v1s-g12`, 28/29) is a 5–5 visit tie broken by RNG
   on a forced-loss position. Blend the rollout terminal value with the
   end gradient to restore discrimination. Needs its own fast-loop
   validation; pre-register the blend form before any eval.
3. **H3 (v20) — fix the v1-kernel leaf against greedy play.** The binding
   constraint for any v1-leaf version: v17 loses to the 1-ply baseline
   as Black (0.37) and v14/v17 both fail clause 2 on it. Find the
   mechanism with the fast loop, then patch and pre-register. **Method
   correction (iter-17 §5): the fast loop must run against v0 as well
   as v7** — v17's weakness only showed at battery scale against v0,
   where the fast loop never looked. This applies to all three
   workstreams' loops in this batch, not just H3.

Max 3 versions, each pre-registered in its iteration log BEFORE any eval.

## Standing lessons — levers now closed
- **Exploration tuning CLOSED without a new mechanism.** v15 measured
  it: floor 0.005 inert, floor 0.01 flat, c_puct 2.0 wins vs v0 but
  loses to v7, c_puct 3.0 inverts by colour (Black 0.90 / White 0.35).
  The starved-save problem is partly a VALUE problem, not a visit
  problem: the starved saves hold priors 0.027–0.045 (p90–p95), far
  above any floor, and an unvisited move has q=0 — higher than the
  save's negative q — so broadened exploration feeds the ~90 unvisited
  tail moves before the visited-but-negative save. Do not reopen
  exploration tuning without a new mechanism, not a new constant.
- **Root-decision lever EXHAUSTED** (v9/v12 bracket it at ±10 games/100).
- **White search/prior/book levers CLOSED.** Sims optimum mapped at 100;
  temperature flattening catastrophic; opening book refuted
  out-of-sample. White work must be value-side.
- **The combination does not add** (v17): components that each passed
  3/4 under the v0 leaf contributed nothing against greedy play under
  the v1 leaf. Do not re-combine without a mechanism that addresses the
  leaf itself.
- **q SIGN:** `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`). Our-perspective q is
  `-child.q`. Batch 4's sign error must not recur.
- **q-based root decisions are CLOSED** (pure q catastrophic: 0.20).
- **More search is NOT monotonically better.** Sims changes need a sweep,
  never an assumption.
- **Small-sample smokes mislead.** v15 had two false positives in one
  iteration (F0.005 beat the control vs v0 0.7125 vs 0.5625 and beat v7
  in the transfer cell 0.5875, both at n=40; both regressed at 100
  games). The 100-game battery is the verdict. Single-position
  diagnostics cannot validate a global search change.
- **The leaf is not the binding constraint** (v11, v14) — BUT the
  v1-kernel leaf's behaviour against greedy play IS the binding
  constraint for any v1-leaf version (v17). That distinction is H3's
  whole premise.

## Method (carried over, with the iter-17 §5 correction)
- The fast 10-game loop stays the finding method: play 10 games, read every
  move, find each game's losing move (solver/deeper-search verified, never
  guessed), patch, regression-test, repeat. Think like a human master
  about weak points and weak strategies.
- **Fast loops run against v0 as well as v7 in this batch.**
- The 4-clause bar stays the validation gate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
  Batch 6 caught two real silent merge breakages this way (a KeyError on
  `"rollout2_v1"` and a dead-code `root_decision` assignment) — the
  discipline works; keep it.

## Constraints
- Branch `research/batch-7` from master (`bf866f7`). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b7/`).
- v15/v16/v17 implementations live on `research/batch-6`, not on master —
  cherry-pick or port; do not re-derive. The v1 kernel (`kernel_v1.py`)
  is exact and reusable; v16's rollout leaf is the strongest single
  White-side component available.
- No cloud GPU spend, no money, no wallets, no releases, no Colab driving,
  no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes all
  4 clauses, with the same plain-merge approach as batch 3.

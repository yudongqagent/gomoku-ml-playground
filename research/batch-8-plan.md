# Batch 8 plan — v18 + blend, conditional gap-three demotion, q-aware tie-break

## Starting point
- Batch 7: 0 of 3 passed. v7 (Elo 190.8) remains the reference; master is
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
1. **H1 (v21) — v18 + blend.** v18 (rollout leaf, both colours) is the
   strongest version in the program on every game-strength measure —
   h2h vs v7 0.775, ladder Elo 636.4 with even its lower bound (240.3)
   clearing v7's 190.8, first version ever to beat the ladder MCTS
   outright (14-6) — and fails the bar on tactical alone (25/29). The
   v19 blend was designed to fix exactly its root cause (q-compression)
   and was refuted as a general mechanism — but the v18+blend
   combination was never tested (v19 is White-only, v18 is unblended).
   Test it: rollout leaf both colours + blended terminals. Risk: the
   blend regressed g45 for v19, so this needs its own fast-loop
   validation and full battery, not an assumption. Pre-register the
   exact blend form before any eval.
2. **H2 (v22) — colour- or initiative-conditional gap-three demotion.**
   The open follow-up from v20: global demotion helped Black (losses
   to v0 20→9) but broke White (losses 20→27) — gap-three breadth is
   phantom against greedy play as the attacker but signal for White's
   counterplay and against a searching opponent. Demote only for the
   player holding the initiative (or only as Black — pre-register the
   exact condition before any eval).
3. **H3 (v23) — q-aware visit-tie break for rollout-leaf engines.**
   v18's 3 tactical failures are all q-compression visit-ties where the
   leaf KNOWS the answer (chosen −1.000 vs expected +0.250) but can't
   convert it into visits against losers' ~3x priors. Narrow,
   mechanism-specific test: when root visits tie at the top, break the
   tie by correctly-signed q (our-q = −child.q). This is NOT a
   reopening of the closed q-based root decisions lever — pure q
   (replacing visits entirely) was catastrophic (0.20), and v9's
   tie-by-q variant smoked 0.45 on the v0/v1 leaf — but on the rollout
   leaf, q-compression makes ties far more frequent and the failure
   mode is documented. Frame the test as clause-3 recovery for
   rollout-leaf engines, pre-register the exact rule, and run the full
   battery.

Max 3 versions, each pre-registered in its iteration log BEFORE any eval.

## Standing lessons — levers now closed
- **The blend is REFUTED as a general mechanism** (v19: it cannot
  distinguish "counterplay that saves the game" from "counterplay that
  loses more slowly"). H1 tests the one combination never tried
  (v18+blend), with its own validation — it does not revive the blend
  as a general mechanism.
- **Global gap-three demotion CLOSED** (v20: right at three verified
  positions, wrong in aggregate — helps the attacker, blinds the
  defender).
- **The combination does not add** (v17) — do not re-combine without a
  mechanism that addresses the leaf itself. H1 is a combination of
  v18 + blend, but the blend is a leaf mechanism addressing v18's
  documented root cause (q-compression), not a stacking of unrelated
  components.
- **q SIGN:** `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`). Our-perspective q is
  `-child.q`. Batch 4's sign error must not recur.
- **q-based root decisions are CLOSED as a general mechanism** (pure q
  catastrophic: 0.20; v9 tie-by-q 0.45 on the v0/v1 leaf). H3 is a
  narrow, pre-registered retest on the rollout leaf only, targeting
  the documented q-compression tie failure mode; it does not reopen
  the general lever.
- **Root-decision lever EXHAUSTED for strength** (v9/v12 bracket it at
  ±10 games/100). H3 is a tactical-recovery test, not a strength lever.
- **Exploration tuning CLOSED without a new mechanism** (v15). The
  starved-save problem is partly a VALUE problem, not a visit problem.
- **White search/prior/book levers CLOSED.** White work must be value-side.
  (Note: White is no longer the weakness — v19's White split vs v7 is
  0.74, the best in the program; the remaining gap is Black, and Black
  is v7 itself. Further combined-h2h gains must come from the Black
  seat — which is exactly what v18 delivered.)
- **More search is NOT monotonically better.** Sims changes need a sweep,
  never an assumption.
- **Small-sample smokes mislead.** The 100-game battery is the verdict.
  Single-position diagnostics cannot validate a global search change.
- **The leaf is not the binding constraint** (v11, v14) — BUT the
  v1-kernel leaf's behaviour against greedy play IS the binding
  constraint for any v1-leaf version (v17, v20).
- **The v1 walk-back oracle shares the argmax-of-sum flaw** and can no
  longer verify saves alone for rollout-leaf engines — every nominated
  save needs playouts or exact computation (iter-18). This is
  methodological and applies to all three workstreams.

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read every
  move, find each game's losing move (solver/deeper-search verified, never
  guessed — for rollout-leaf engines, playouts or exact computation, not
  the v1 oracle alone), patch, regression-test, repeat. Think like a
  human master about weak points and weak strategies.
- **Fast loops run against v0 as well as v7** (established practice —
  H3's v20 mechanism came from the v0 loop).
- The 4-clause bar stays the validation gate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
  Batch 6 caught two real silent merge breakages this way; batch 7's
  merges carried the same discipline and changed no version's behaviour.

## Constraints
- Branch `research/batch-8` from master (`bf866f7`). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b8/`).
- v18/v19/v20 implementations live on `research/batch-7`, not on master —
  cherry-pick or port; do not re-derive. The v1 kernel (`kernel_v1.py`)
  is exact and reusable; v18's rollout leaf is the strongest engine in
  the program.
- No cloud GPU spend, no money, no wallets, no releases, no Colab driving,
  no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes all
  4 clauses, with the same plain-merge approach as batch 3.

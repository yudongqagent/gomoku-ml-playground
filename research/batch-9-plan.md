# Batch 9 plan — q-gap threshold, ours-only demotion, v22 forensics

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the heuristic track's batch after ML batch 1.
- **Bar reference stays v7** (Elo 190.8); **strength champion v18**
  (Elo 636.4, h2h vs v7 0.775). A new strength champion needs to
  beat v18 on BOTH h2h-vs-v7 rate AND ladder Elo on the same battery.
  A bar pass (all 4 clauses vs v7) makes the version the new bar
  reference and it goes to master.
- **Versions v27–v29** (ML batch 1 owns v24–v26; the "v24" label in
  batch 8's handoff for the q-gap threshold is stale — confirm the
  next numbers from `research/VERSIONS.md` at start).
- v18/v23 code lives on `research/batch-7` and `research/batch-8` —
  cherry-pick/port, don't re-derive.

## Pre-registered program correction (apply FIRST, before any version's battery)
- Suite fix for `winning-quiet-setup-2`: add (9,7) to the expected
  list in `ml/gomoku_ml/tactical_positions_hard.json`. Rationale:
  v23's chosen (9,7) wins 3/3 under v7-vs-v7 (iter-23 §3) — the suite
  list is incomplete, not the engine wrong. This is a program-wide
  correction, NEVER a rescue of a version's clause-3 score: v23's
  verdict stands (REJECTED 3/4, 28/29 on the old suite). Adding an
  expected move can only raise pass rates, so re-verify v7 still
  passes 29/29 on the corrected suite (it must — document the
  re-run). All batch-9 versions are assessed against the corrected
  suite.

## Candidates, in priority order
1. **H1 (v27) — q-gap threshold tie-break.** v23's q-aware tie-break
   recovered all 3 q-compression ties and set the best h2h vs v7 ever
   (0.795) but lost Elo (511.5 vs 636.4) and vs-v0 (0.91 vs 0.97):
   it helps where q gaps carry terminal signal and hurts where they
   are gradient noise. Break visit ties by correctly-signed q (our-q
   = `-child.q`) ONLY when the q gap exceeds a threshold.
   Pre-register the threshold sweep (e.g. 0.05, 0.10) and the
   selection rule before any eval; the 100-game battery is the
   verdict. This is a narrow refinement of v23's mechanism, NOT a
   reopening of the closed q-based root-decisions lever.
2. **H2 (v28) — ours-only Black demotion.** v22's finding: demoting
   Black's gap-three breadth helped Black (+0.33) but broke White
   (−0.22) because White's gap-three breadth is signal only as the
   OPPONENT's breadth (Black fearing White's counterplay). Demote
   Black's gap threes only in the ours term (when Black is to move),
   never in the theirs term. Pre-register the exact condition before
   any eval.
3. **H3 (v29) — v22 vs-mcts forensics → targeted patch.** v22
   collapses vs the ladder MCTS (0.25, worst of any v1-leaf version),
   which caps its Elo at 147.2. Play v22 vs the ladder MCTS, read
   the losses, find the mechanism of the collapse, patch it. If the
   forensics finds nothing patchable, fall back to the VCF-at-root
   lever for the rollout-leaf engine (v24's finding: the 100-sim
   search horizon is a shared limit — v7 makes the same miss as v24;
   a VCF/deeper threat check at the root could catch forced wins the
   search misses).

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — heuristic track
- **q SIGN:** `child.q` is from the OPPONENT's perspective
  (`_select_child` exploits `-child.q`). Our-perspective q is
  `-child.q`.
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: q-based root decisions as a general mechanism (pure q
  0.20); root-decision for strength (±10/100); White search/prior/
  book (100 sims optimum; prior-temperature flattening catastrophic);
  exploration tuning (prior floor inert); global gap-three demotion;
  the blend as a general mechanism (doubly bounded: v19, v21);
  combination-stacking (v17); more sims is NOT monotonically better
  (v10 White 0W-50L at 200 sims).
- The leaf is not the binding constraint (v11, v14) — EXCEPT the
  v1-kernel leaf's behaviour against greedy play IS the binding
  constraint for v1-leaf versions (v17, v20).
- The sharp prior is load-bearing at 100 sims. Defence is solved
  (all pooled losses via open four, 0 missed blocks) — new versions
  must keep it solved.
- Small-sample smokes mislead (the 100-game battery is the verdict);
  single-position diagnostics cannot validate a global search change;
  the v1 walk-back oracle shares the argmax-of-sum flaw (iter-18) —
  every nominated save needs playouts or exact computation.
- Fast loops run against v0 as well as v7 (iter-17 §5).

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed — for rollout-leaf engines, playouts or
  exact computation, not the v1 oracle alone), patch, regression-test,
  repeat.
- The 4-clause bar stays the validation gate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.

## Constraints
- Branch `research/batch-9` from master (`bf866f7`). Commit this
  plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b9/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3.

# Batch 10 plan — VCF-filtered search, position-dependent threshold, v27 forensics

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the heuristic track's batch after ML batch 2.
- **Bar reference is v27** (Elo 636.4, batch 9 — v7 was the reference
  until batch 9). The bar is unchanged in shape, re-anchored on v27:
  (1) h2h vs v27 ≥100 games, win rate > 0.50 with Wilson 95% CI lower
  bound > 0.50; (2) ladder Elo > 636.4; (3) tactical ≥ 20/20 + 29/29
  on the corrected suite; (4) all test suites green. Note clause 2
  is now demanding: 636.4 is the estimate from a 19W-0L-1D heuristic
  leg, so beating it effectively requires a 20-0 leg.
- **Strength champion is v18** (Elo 636.4). **Champion rule v2
  (direct challenge, adopted 2026-11-17):** a version that passes
  the bar plays a 100-game match against the current champion
  (v18); it takes the title if it wins the match (win rate > 0.50
  with Wilson 95% CI lower bound > 0.50) AND its ladder Elo beats
  the champion's (636.4). A bar-passer already clears the Elo
  clause by passing, so the challenge match is the deciding test.
  (The old rule compared h2h-vs-v7 rates, which no longer appear
  on a battery now that the bar reference is v27; the direct
  challenge measures the same thing without the stale reference.
  v18 keeps the title under both rules — v27 ties it on Elo.)
- **Versions v33–v35** (ML batch 2 owns v30–v32; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7) to
  winning-quiet-setup-2's expected list). All batch-10 versions are
  assessed against it.
- Master contains the v18/v23/v27/v28/v29 code (batch-9 merge) — no
  cherry-picking needed for the base engines.

## Tooling (do FIRST, alongside the version work)
- **Add auto-resume to the ladder driver (`eval_version.py`).** ML
  batch 2 H2's note: the ladder run finished minutes before a VM
  replacement by luck — the ladder driver is the only match driver
  without resume support. Follow the existing pattern in the other
  drivers (flush after every game, resume from the partial `--out`
  file); verify with a kill-and-resume test. This is a tooling
  change, not a version — it goes on the batch branch either way.

## Candidates, in priority order
1. **H1 (v33) — VCF-filtered search on v29.** v29's verified flaw:
   its defensive check picks preventions by prior noise (3e-6
   differences between preventions with opposite outcomes — the
   check's move lost 3/3 while the search's move won 3/3, iter-29
   §3.4). Choose among preventions by SEARCH (the rollout-leaf
   search), not prior. Pre-register the selection rule before any
   eval. Caveat from ML batch 2 (v32): if the prevention pool
   contains no good prevention, no selection rule fixes it (v0-g4's
   pool both lose 3/3) — the fast loop must verify the pool
   contains a winner before claiming the fix.
2. **H2 (v34) — position-dependent threshold for v27's tie-break.**
   The τ=0.05/0.10 sweep tied (39W-0L-1D vs v0 over 40 games), so a
   flat threshold change is not the lever — the CONDITION is.
   Batch 9 recorded a caution: at one fast-loop ply the threshold
   fired on a signal-sized gap (0.1481) between two verified losers
   — a large gap in a murky midgame is not the same evidence as in
   a tactical position. Fire the q-gap tie-break only when the
   position is tactical. Pre-register the condition (e.g. when
   either side holds a forcing threat / open three or better)
   before any eval; the 100-game battery is the verdict.
3. **H3 (v35) — v27 loss forensics → targeted patch.** Read v27's
   battery losses (19 vs v7, 3 vs v0), find any shared mechanism,
   patch it. If the forensics finds nothing patchable, fall back
   to a cheaper VCF check for v29 (depth/candidate-limited) that
   keeps the tactical fixes while reducing the ~1.65s/move cost
   (~7s/move in creator-rich games).

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
  exploration tuning (prior floor inert); gap-three demotion in ALL
  FOUR combinations (v14 0.60, v20 0.585, v22 0.655, v28 0.545 vs
  v0 — priced and closed); the blend as a general mechanism (doubly
  bounded: v19, v21); combination-stacking (v17); more sims is NOT
  monotonically better (v10 White 0W-50L at 200 sims).
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
- **Fast loops run against v0 as well as v27** (the bar reference is
  v27 now, not v7).

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed — for rollout-leaf engines, playouts or
  exact computation, not the v1 oracle alone), patch, regression-test,
  repeat.
- The 4-clause bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) — use that durability; batteries must
  survive VM replacements.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the dead-code root-selection trap
  (batch 8, batch 9 H1 — `_pick_root_move` vs `_choose_root_child`),
  the shared-tail trap (batch 8, batch 9 H1/H3), and the silent
  auto-merge breakage on `ValueMCTS.__init__` (ML batch 1, ML batch
  2 — duplicate parameters with NO conflict flag).

## Constraints
- Branch `research/batch-10` from master (`4cc31af`). Commit this
  plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b10/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is unchanged
  (v18 vs v8, swap rules).

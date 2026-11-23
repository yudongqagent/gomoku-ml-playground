# Batch 14 plan — TSS depth/budget sweep, v52's swap/colour interaction, TSS speed

## Starting point
- The program now runs two tracks (see `research/TRACKS.md`). This is
  the heuristic track's batch after ML batch 6.
- **Heuristic bar reference is v52** (threat-space search at the
  root; ladder Elo unbounded — saturated, interval [407.0, ∞)) —
  the strongest version that passed all 4 clauses in batch 13
  (2026-11-22). **Strength champion is also v52** (it beat v18 in
  the 100-game direct challenge, 0.625, CI lower 0.5271, passing
  on both cuts). v18 held the title from batch 7 (2026-11-15)
  until batch 13; v27 (Elo 636.4) was the bar reference until
  batch 13.
- **Champion rule v2** (TRACKS.md): a version that passes the bar
  plays a 100-game DIRECT CHALLENGE match against the current
  champion (v52); it takes the title if it wins the match (win
  rate > 0.50 with Wilson 95% CI lower bound > 0.50) AND its
  ladder Elo beats the champion's (by the leg-result reading —
  see below). Only run the challenge match for a bar-passer.
- **The 4-clause bar** (`research/README.md`), re-anchored on v52:
  (1) h2h vs v52 ≥100 games, win rate > 0.50 with Wilson 95% CI
  lower bound > 0.50; (2) ladder Elo > v52's (see the leg-result
  rule below); (3) tactical ≥ 20/20 + 29/29 on the corrected
  suite; (4) all test suites green.
- **Clause 2 for the v52 reference (pre-registered in TRACKS.md,
  commit 8e1f87f):** v52's Elo is unbounded (its heuristic leg is
  40-0-0, the first version ever to saturate the 40-game ladder),
  so clause 2 is judged by the leg-result reading — a future
  version must strictly dominate v52's legs (random 40-0,
  heuristic 40-0, mcts 31-9). In practice that means matching the
  two perfect legs and beating 31-9 on the mcts leg. (Or the
  ladder is re-based with a stronger anchor or more games per leg
  — the saturation problem batch 10 flagged has now arrived in
  full.) Do NOT claim a "highest Elo ever" from a single draw
  (batch 12 finding: v45's Elo 759.1 vs v27's 636.4 came from one
  fewer draw in the heuristic leg with identical play).
- **Versions v57–v59** (ML batch 6 owns v54–v56; confirm the next
  numbers from `research/VERSIONS.md` at start).
- The tactical suite is the CORRECTED suite (batch 9 added (9,7)
  to winning-quiet-setup-2's expected list). Re-verify v52's
  tactical on it first.
- **The ladder is 40 games per opponent per leg** (batch 11's
  re-basing — it worked: no saturation, intervals narrowed).
  Point estimates remain comparable to historical versions (same
  anchors, same formula). The ladder driver has auto-resume
  (`eval_version.py --out`, on master) — use it; every ladder
  JSON must be a committed artifact.
- The match drivers now seed per game (`seed + game_index`) with a
  uniqueness audit on every leg (ML batch 3 tooling, on master).

## Candidates, in priority order
1. **H1 (v57) — TSS depth/budget sweep.** Depth 4 + node budget
   3000 was v52's pre-registered design. Test whether deeper TSS
   (depth 5–6) or a larger budget (6000 nodes) improves strength,
   or whether the budget binds first. Pre-register the sweep grid
   (depth × budget combinations) and the mechanism check (does
   deeper TSS find forcing wins the depth-4 search misses, on a
   set of test positions?) before any eval. Measure the cost
   first (deeper TSS is more expensive per move; the budget
   binds ~1% at 3000 — a larger budget may help only in
   creator-rich positions).
2. **H2 (v58) — diagnose v52's swap/colour interaction, then patch
   the mechanism found.** v52 wins as White under swap (0.54 vs
   v27) but loses as White under normal rules (0.44 vs v27, 0.48
   vs v18). Diagnose why — is it the swap choice itself (the
   tempo judge picks the stronger colour for the second player),
   or does v52's TSS behave differently as White under swap
   (e.g. the opening positions differ)? Then patch the mechanism
   found. Pre-register the diagnosis (measurements + the decision
   rule for which mechanism it is) and the patch before any eval.
   If the diagnosis finds no patchable mechanism, say so. Note:
   the White gap as a single-flaw framing is CLOSED (v41, v46),
   so this needs a genuinely new mechanism, not another White
   patch — the swap/colour interaction is the new angle.
3. **H3 (v59) — TSS speed optimisation.** The p99 < 1s criterion
   was NOT met (~5s uncontended; a budget low enough for it
   (~500 nodes) would abort a quarter of all searches). Use the
   batch-10 `four_creating_moves` optimisation (still on the
   batch-10 branch, commit `efb71a3` — 5.1× faster VCF checks,
   proof-checked at merge and independently A/B'd with 0
   mismatches; changes only speed, never which move is selected)
   and/or node-budget tuning to bring the p99 under 1s WITHOUT
   losing strength. Pre-register the speed target (p99 < 1s
   uncontended) and the strength regression check (must not lose
   vs v52 on a 100-game battery; a tie is acceptable for a speed
   version, reported as such — a pure speed version cannot pass
   the bar on strength, and the report must say so honestly).
   This version matters for deployment: v52 at ~1.6s/move mean
   in creator-rich positions is too slow for a good web
   experience; a speed-optimised v52 at the same strength would
   be the version to deploy to the page.

Max 3 versions, each pre-registered in its iteration log BEFORE any
eval.

## Standing lessons — heuristic track
- **CLOSED levers** — do not reopen without a genuinely new
  mechanism: q-based root decisions as a general mechanism (pure
  q 0.20; v9 tie-by-q 0.45 on v0/v1 leaf) — EXCEPT v23's q-aware
  visit-tie break and v27's threshold refinement, which are the
  narrow exceptions (and batch 10 closed the tie-break lever
  entirely); root-decision for strength (v9/v12 bracket ±10/100);
  White search/prior/book; exploration tuning (v15); prior-
  temperature flattening (catastrophic: 0.775 → 0.05 at T=3.0);
  gap-three demotion in ALL FOUR combinations (batch 9: v14 0.60,
  v20 0.585, v22 0.655, v28 0.545 vs v0 — CLOSED as a strength
  lever); the blend as a general mechanism (v19, v21 — doubly
  bounded); combination-stacking (v17); assuming more sims helps
  (v10 collapsed at 200 sims); the leaf as binding constraint
  (v11, v14) — EXCEPT the v1-kernel leaf's behaviour against
  greedy play IS the binding constraint for v1-leaf versions
  (v17, v20); VCF-prevention for the v27 family (v35 — defensive
  check priced in all three forms: prior-pick v29, search-pick
  v33, search-first v35; re-confirmed v45, batch 12 — the hole
  fix found 0 hole instances in 6,258 turns); the tie-break
  lever (v34 — doubly bounded); setup-prevention in this form
  (v39 — the check fires ~6% of moves while the mechanism
  matters in ~0.3% of turns); the leaf-blind direction (v40 —
  the remaining mis-rankings are averaging/visit-allocation,
  i.e. search problems); the White gap as a single-flaw framing
  (v41 — the gap is matchup-dependent, not a systematic
  v27-White defect; population-statistics confirmation v46,
  batch 12); both blunder-dilution fixes (v46 leaf-level
  pure-min, v47 pure minimax backup — the mean backup's
  averaging is load-bearing, not a bug); child-level backup
  aggregation in any form (v47 pure minimax, v51 soft-min — the
  mean backup's prevention gradient comes from the expansion
  leaf, which any child-only aggregation discards; with one
  visited child the softmax IS that child's value, so preventions
  collapse to q=−1.0); rollout-depth changes (v53 — the ladder
  is fully priced: 1 rejected, 2 = v18/v27, 3 beats v27 h2h but
  fails the bar, 4 rejected twice).
- **Defence is solved** (all pooled losses via open four, 0
  missed blocks) and new versions must keep it solved.
- **q sign:** our-q = −child.q (`_select_child` exploits
  −child.q); q prefers the *losing* move in §2.6's corrected
  reading. Do not re-introduce the sign error.
- **The sharp prior is load-bearing at 100 sims.**
- Small-sample smokes mislead; the 100-game battery is the
  verdict. Single-position diagnostics cannot validate a global
  search change.
- **Fast loops run against v0 as well as v52.**
- The v1 walk-back oracle shares the argmax-of-sum flaw and can
  no longer verify saves alone for rollout-leaf engines — every
  nominated save needs playouts or exact computation.
- **VCF starts are NOT monotone under our move** (batch 11) —
  occupying a setup square can CREATE opponent VCF starts by
  removing one of our own defensive replies. Any future check
  must full-re-check VCF after the candidate move. An immediate
  winning move IS a VCF start but is NOT necessarily a
  four-creator — the fast VCF test must check winning moves
  first.
- Exact leaf values interact with the tie-break (v40) — future
  leaf changes must expect visit-allocation regressions; v40's
  v1bl-g2 failure is the concrete instance.
- **Kernel blindness (batch 13):** v0 is blind to broken-four
  creators (22 mismatches on the first v0-only TSS attempt), v1
  is blind to some open threes (v1 405 vs v0 1021) — each kernel
  enumerates one class. Any TSS work must use the right kernel
  for each move class (or both).
- **Depth counting (batch 13 lesson):** the ML track's label-depth
  naming counts plies BEFORE the teacher leaf, so an "N-ply
  label" is an (N+2)-ply effective horizon. State depth in total
  rollout plies and check against `_ROLLOUT_PLIES` before the
  plan is written.

## Method (carried over)
- The fast 10-game loop stays the finding method: play 10 games, read
  every move, find each game's losing move (solver/deeper-search
  verified, never guessed), patch, regression-test, repeat.
- The 4-clause bar stays the validation gate.
- All match drivers auto-resume from partial `--out` files (ML batch
  1 H1's patch, on master) and now seed per game (`seed +
  game_index`, ML batch 3 tooling, on master) with a uniqueness
  audit on every leg — use that durability and hygiene; batteries
  must survive VM replacements, and every leg reports its
  distinct-game count and de-dup rate.

## Merge discipline (batch-3 lesson)
- Run the full suite PLUS targeted per-version tests after every merge.
- Watch for the recurring traps: the silent auto-merge breakage on
  `ValueMCTS.__init__` (ML batch 1, ML batch 2 — duplicate
  parameters with NO conflict flag; check for duplicate parameters
  before running anything), the dead-code root-selection trap
  (batch 8, batch 9), and the shared-tail trap (batch 8, batch 9,
  ML batch 5).

## Constraints
- Branch `research/batch-14` from master (current HEAD — verify
  with git rev-parse before branching). Commit this plan first.
- Use your own git worktree (e.g. `~/workspace/projects/gomoku-b14/`).
- `ml/gomoku_ml/baseline.py` stays FROZEN (ladder anchor). Kernel
  re-sync (NumPy/WebGPU/Metal) only when a version passes the bar.
- No cloud GPU spend, no money, no wallets, no releases, no Colab
  driving, no Space edits. Yudong runs nothing.
- Push the branch when done. Push to master ONLY if a version passes
  all 4 clauses of the bar, with the same plain-merge approach as
  batch 3. A bar pass changes the heuristic champion (via the
  direct-challenge rule) and the daily cross-track match — say so
  explicitly in the report.
- The daily cross-track cron runs each morning ~08:37 PST; do not
  touch it or its output files. The daily pairing is v52 vs v44
  (swap rules), run in the main worktree (see TRACKS.md).
